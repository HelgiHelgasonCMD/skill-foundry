"""Deterministic feature extraction from experiences: commands, step signatures, tools.

The canonical contract has no structured command field, so commands are taken only
from backtick spans in attempt/evidence summaries. Specific arguments (paths, node
ids, SHAs, ``<placeholders>``) are parameterised to ``<arg>`` so that procedures
generalise instead of copying a training case's exact files.
"""

from __future__ import annotations

import re
from collections import Counter

from skill_foundry.domain.models import Attempt, Experience
from skill_foundry.util import tokens

_BACKTICK = re.compile(r"`([^`\n]{1,200})`")
KNOWN_TOOLS = frozenset(
    {
        "python",
        "python3",
        "pytest",
        "git",
        "actionlint",
        "make",
        "npm",
        "pnpm",
        "yarn",
        "pip",
        "uv",
        "tox",
        "nox",
        "act",
        "docker",
        "cargo",
        "go",
        "node",
        "ruff",
        "mypy",
        "nosetests",
        "bash",
        "sh",
        "gh",
        "jq",
        "yamllint",
    }
)
_HEX = re.compile(r"^[0-9a-f]{12,64}$")
_FILEISH = re.compile(r"[/\\]|\.(py|ya?ml|json|toml|cfg|ini|md|txt|js|ts)$")


def extract_commands(text: str) -> list[str]:
    out = []
    for span in _BACKTICK.findall(text):
        parts = span.split()
        if not parts:
            continue
        head = parts[0]
        if head in KNOWN_TOOLS or (
            len(parts) >= 2
            and re.fullmatch(r"[a-z][a-z0-9_.\-]*", head)
            and not parts[1].endswith(":")
        ):
            out.append(span.strip())
    return out


def normalize_command(cmd: str, *, keep_placeholders: bool = False) -> str:
    """Parameterise specific arguments. ``keep_placeholders`` keeps ``<named>`` slots for
    display; matching always uses the fully normalised form."""
    out = []
    for i, tok in enumerate(cmd.split()):
        if (
            (keep_placeholders and tok.startswith("<") and tok.endswith(">"))
            or i == 0
            or tok.startswith("-")
            or tok.isdigit()
            or (tok.isalpha() and tok.islower() and len(tok) < 20)
        ):
            out.append(tok)
        elif (tok.startswith("<") and tok.endswith(">")) or _HEX.match(tok) or _FILEISH.search(tok):
            out.append("<arg>")
        else:
            out.append(tok if re.fullmatch(r"[a-z][a-z0-9_.\-]*", tok) else "<arg>")
    # collapse repeated <arg>
    collapsed: list[str] = []
    for tok in out:
        if tok.startswith("<") and collapsed and collapsed[-1] == tok:
            continue
        collapsed.append(tok)
    return " ".join(collapsed)


def tool_of(cmd: str) -> str:
    parts = cmd.split()
    if len(parts) >= 3 and parts[0].startswith("python") and parts[1] == "-m":
        return parts[2]
    return parts[0] if parts else ""


def step_signature(action: str) -> str:
    """Stable identity of a step across episodes.

    Prefer the normalised command(s); otherwise use the generic clause before a
    colon (episode-specific detail conventionally follows the colon).
    """
    cmds = extract_commands(action)
    if cmds:
        return " && ".join(normalize_command(c) for c in cmds)
    head = action.split(":", 1)[0]
    head = _BACKTICK.sub(lambda m: m.group(1), head)
    return " ".join(tokens(head)[:10])


def step_signatures(attempts: list[Attempt], status: str = "completed") -> list[str]:
    """Ordered signatures of attempts with ``status``; repeats get an ordinal suffix."""
    seen: Counter[str] = Counter()
    out = []
    for a in attempts:
        if a.status != status:
            continue
        sig = step_signature(a.action_summary)
        seen[sig] += 1
        out.append(sig if seen[sig] == 1 else f"{sig} #{seen[sig]}")
    return out


def tools(x: Experience) -> set[str]:
    cmds = [c for a in x.attempts for c in extract_commands(a.action_summary)]
    return {tool_of(c) for c in cmds if c}


def task_terms(x: Experience) -> set[str]:
    terms = set(tokens(x.task_summary.split(":", 1)[0]))
    terms |= {o.kind for o in x.observations}
    terms |= {o.error_code.lower() for o in x.observations if o.error_code}
    return terms
