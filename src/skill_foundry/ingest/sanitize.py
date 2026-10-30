"""Untrusted-input handling: limits, structured secret exclusion, redaction, injection screening.

Order matters and is fixed:

1. hard size / shape limits (reject),
2. structured exclusion of secret-named and hidden-reasoning fields (drop),
3. regex + entropy redaction of every remaining string (replace),
4. prompt-injection heuristics (quarantine the record for human review).

Regex redaction is a safety net, not a guarantee; see docs/security.md.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

SANITIZER_VERSION = "sanitizer.v1"


@dataclass(frozen=True)
class Limits:
    max_bundle_bytes: int = 20 * 1024 * 1024
    max_episodes: int = 5000
    max_string_chars: int = 20_000
    max_list_items: int = 2_000
    max_depth: int = 12


class LimitExceeded(ValueError):
    pass


# Keys (case-insensitive, matched on any '_'/'-'/'.' separated component) that are
# never stored, whatever their value. These can only occur inside free-form
# `extensions` objects because the canonical schema forbids unknown fields elsewhere.
SECRET_KEY_PARTS = frozenset(
    {
        "token",
        "secret",
        "password",
        "passwd",
        "apikey",
        "api_key",
        "authorization",
        "cookie",
        "credential",
        "credentials",
        "private_key",
        "session",
        "bearer",
    }
)
HIDDEN_KEYS = frozenset(
    {
        "thinking",
        "reasoning",
        "chain_of_thought",
        "transcript",
        "raw_transcript",
        "raw_messages",
        "messages",
        "scratchpad",
    }
)

_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "private_key",
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)", re.S
        ),
    ),
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}")),
    ("openai_key", re.compile(r"sk-(?:proj-)?[A-Za-z0-9]{20,}")),
    ("github_token", re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("slack_token", re.compile(r"xox[abprs]-[A-Za-z0-9\-]{10,}")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}")),
    ("bearer", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-~+/]{16,}=*")),
    ("url_credentials", re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://)[^\s/:@]+:[^\s/@]+@")),
    (
        "assignment",
        re.compile(
            r"(?i)\b([A-Z0-9_]*(?:SECRET|TOKEN|PASSWORD|PASSWD|API_?KEY|PRIVATE_KEY)[A-Z0-9_]*)"
            r"\s*[=:]\s*(['\"]?)[^\s'\"]{4,}\2"
        ),
    ),
]
_LONG_TOKEN = re.compile(r"[A-Za-z0-9+/_\-]{32,}={0,2}")
_HEX_SHA = re.compile(r"^[0-9a-f]{7,64}$")

_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "ignore_instructions",
        re.compile(
            r"(?i)\b(ignore|disregard|forget)\b.{0,40}\b(previous|prior|above|all|earlier)\b"
            r".{0,20}\b(instructions?|rules|prompts?|guidelines)\b"
        ),
    ),
    (
        "role_marker",
        re.compile(r"(?im)^\s*(system|assistant|developer)\s*:|<\|im_start\|>|</?system>"),
    ),
    (
        "authority_claim",
        re.compile(
            r"(?i)\byou (must|should) (now )?(always )?(run|execute|install|grant|enable)\b"
            r"|\bdo not tell the user\b|\bnew instructions\b"
        ),
    ),
    ("tool_call_text", re.compile(r"(?i)<(function_calls|invoke|tool_use)\b|\"tool_use\"\s*:")),
    (
        "exfiltration",
        re.compile(
            r"(?i)\b(curl|wget)\b[^\n]{0,80}\|\s*(ba|z)?sh\b|\bcat\s+~?/?\.ssh/|\bprintenv\b.{0,20}\|"
        ),
    ),
]


def shannon_entropy(s: str) -> float:
    counts = Counter(s)
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


@dataclass
class SanitizeReport:
    redactions: Counter[str] = field(default_factory=Counter)
    dropped_fields: list[str] = field(default_factory=list)
    injection_hits: list[str] = field(default_factory=list)

    @property
    def total_redactions(self) -> int:
        return sum(self.redactions.values())


def redact_text(text: str, report: SanitizeReport | None = None) -> str:
    rep = report if report is not None else SanitizeReport()
    for kind, pat in _SECRET_PATTERNS:

        def _sub(m: re.Match[str], kind: str = kind) -> str:
            rep.redactions[kind] += 1
            if kind == "assignment":
                return f"{m.group(1)}=[REDACTED:{kind}]"
            if kind == "url_credentials":
                return f"{m.group(1)}[REDACTED:{kind}]@"
            return f"[REDACTED:{kind}]"

        text = pat.sub(_sub, text)

    def _entropy(m: re.Match[str]) -> str:
        tok = m.group(0)
        if _HEX_SHA.match(tok) or "REDACTED" in tok:
            return tok  # commit SHAs / content hashes are provenance, not secrets
        if (
            shannon_entropy(tok) >= 4.0
            and any(c.isdigit() for c in tok)
            and any(c.isupper() for c in tok)
        ):
            rep.redactions["high_entropy"] += 1
            return "[REDACTED:high_entropy]"
        return tok

    return _LONG_TOKEN.sub(_entropy, text)


def scan_injection(text: str) -> list[str]:
    return [kind for kind, pat in _INJECTION_PATTERNS if pat.search(text)]


def _key_is_secret(key: str) -> bool:
    k = key.lower()
    if k in HIDDEN_KEYS:
        return True
    parts = set(re.split(r"[_\-.]", k)) | {k.replace("-", "_")}
    return bool(parts & SECRET_KEY_PARTS)


def check_limits(obj: Any, limits: Limits, path: str = "$", depth: int = 0) -> None:
    if depth > limits.max_depth:
        raise LimitExceeded(f"{path}: nesting deeper than {limits.max_depth}")
    if isinstance(obj, str) and len(obj) > limits.max_string_chars:
        raise LimitExceeded(f"{path}: string longer than {limits.max_string_chars} chars")
    if isinstance(obj, list):
        if len(obj) > limits.max_list_items:
            raise LimitExceeded(f"{path}: list longer than {limits.max_list_items} items")
        for i, v in enumerate(obj):
            check_limits(v, limits, f"{path}[{i}]", depth + 1)
    if isinstance(obj, dict):
        for k, v in obj.items():
            check_limits(v, limits, f"{path}.{k}", depth + 1)


# Fields whose values are identifiers, never free text. They are not redacted so
# that provenance links survive (but they are length-limited by the schema/limits).
_ID_FIELDS = frozenset(
    {
        "episode_id",
        "observation_id",
        "attempt_id",
        "evidence_id",
        "lesson_id",
        "export_id",
        "revision",
        "repo_id",
        "source_event_ids",
        "event_ids",
        "verification_evidence_ids",
        "source_episode_ids",
        "observed_at",
        "timestamp",
        "status",
        "kind",
        "schema_version",
    }
)


def sanitize(obj: Any, report: SanitizeReport, path: str = "$", key: str | None = None) -> Any:
    """Return a sanitised deep copy of a contract episode."""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if _key_is_secret(k):
                report.dropped_fields.append(f"{path}.{k}")
                continue
            out[k] = sanitize(v, report, f"{path}.{k}", k)
        return out
    if isinstance(obj, list):
        return [sanitize(v, report, f"{path}[{i}]", key) for i, v in enumerate(obj)]
    if isinstance(obj, str):
        if key in _ID_FIELDS:
            return obj
        hits = scan_injection(obj)
        if hits:
            report.injection_hits += [f"{path}: {h}" for h in hits]
        return redact_text(obj, report)
    return obj
