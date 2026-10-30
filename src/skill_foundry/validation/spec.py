"""Agent Skills specification validator (https://agentskills.io/specification).

Implements the published constraints directly (stricter where the spec says
"must"), then also runs the reference validator ``skills_ref`` when importable
and merges its findings.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from skill_foundry.domain.models import ValidationIssue, ValidationReport

ALLOWED_FIELDS = frozenset(
    {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}
)
NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
MAX_BODY_LINES = 500
APPROX_TOKEN_BUDGET = 5000


def split_frontmatter(text: str) -> tuple[dict[str, Any] | None, str, str | None]:
    if not text.startswith("---\n"):
        return None, text, "SKILL.md must start with YAML frontmatter ('---' on line 1)"
    end = text.find("\n---", 4)
    if end == -1:
        return None, text, "frontmatter is not closed with '---'"
    raw = text[4:end]
    body = text[end + 4 :].lstrip("\n")
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        return None, body, f"frontmatter is not valid YAML: {exc}"
    if not isinstance(data, dict):
        return None, body, "frontmatter must be a YAML mapping"
    return data, body, None


def _reference(skill_dir: Path) -> tuple[str | None, list[str]]:
    try:
        from skills_ref import validator
    except ImportError:  # pragma: no cover - dependency is declared
        return None, []
    try:
        import importlib.metadata as md

        version = md.version("skills-ref")
    except Exception:  # pragma: no cover
        version = "unknown"
    return f"skills-ref {version}", list(validator.validate(skill_dir))


def validate_skill_dir(skill_dir: Path) -> ValidationReport:
    issues: list[ValidationIssue] = []

    def err(code: str, msg: str, path: str | None = "SKILL.md") -> None:
        issues.append(ValidationIssue(code=code, severity="error", message=msg, path=path))

    def warn(code: str, msg: str, path: str | None = "SKILL.md") -> None:
        issues.append(ValidationIssue(code=code, severity="warning", message=msg, path=path))

    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        err("missing_skill_md", "SKILL.md is required")
        return ValidationReport(kind="spec", passed=False, issues=issues)
    text = skill_md.read_text(encoding="utf-8")
    fm, body, problem = split_frontmatter(text)
    if problem or fm is None:
        err("frontmatter", problem or "invalid frontmatter")
        return ValidationReport(kind="spec", passed=False, issues=issues)

    for key in sorted(set(fm) - ALLOWED_FIELDS):
        err("unknown_field", f"frontmatter field '{key}' is not defined by the specification")
    name = fm.get("name")
    if not isinstance(name, str) or not name:
        err("name_missing", "'name' is required and must be a string")
    else:
        if len(name) > 64:
            err("name_length", f"'name' is {len(name)} chars (max 64)")
        if not NAME_RE.match(name):
            err(
                "name_format",
                "'name' must be lowercase a-z/0-9 and single hyphens, "
                "not starting or ending with a hyphen",
            )
        if name != skill_dir.name:
            err("name_dir_mismatch", f"'name' ({name}) must match directory ({skill_dir.name})")
    desc = fm.get("description")
    if not isinstance(desc, str) or not desc.strip():
        err("description_missing", "'description' is required and must be non-empty")
    elif len(desc) > 1024:
        err("description_length", f"'description' is {len(desc)} chars (max 1024)")
    elif len(desc) < 40 or not re.search(r"(?i)\buse (it )?when\b", desc):
        warn("description_quality", "description should say what the skill does and when to use it")
    if "compatibility" in fm:
        comp = fm["compatibility"]
        if not isinstance(comp, str) or not 1 <= len(comp) <= 500:
            err("compatibility", "'compatibility' must be a 1-500 character string")
    if "license" in fm and not isinstance(fm["license"], str):
        err("license", "'license' must be a string")
    if "metadata" in fm:
        meta = fm["metadata"]
        if not isinstance(meta, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in meta.items()
        ):
            err("metadata", "'metadata' must map string keys to string values")
    if "allowed-tools" in fm and not isinstance(fm["allowed-tools"], str):
        err("allowed_tools", "'allowed-tools' must be a space-separated string")

    if not body.strip():
        err("empty_body", "SKILL.md body is empty")
    n_lines = body.count("\n") + 1
    if n_lines > MAX_BODY_LINES:
        warn("body_length", f"body has {n_lines} lines; keep SKILL.md under {MAX_BODY_LINES}")
    if len(body) / 4 > APPROX_TOKEN_BUDGET:
        warn("body_tokens", f"body is roughly {len(body) // 4} tokens (> {APPROX_TOKEN_BUDGET})")

    root = skill_dir.resolve()
    for target in _LINK.findall(body):
        if (
            re.match(r"^[a-z]+://", target)
            or target.startswith("#")
            or target.startswith("mailto:")
        ):
            continue
        rel = target.split("#", 1)[0]
        if rel.startswith("/") or re.match(r"^[A-Za-z]:\\", rel):
            err("absolute_link", f"link '{target}' must be relative to the skill root")
            continue
        resolved = (skill_dir / rel).resolve()
        if not resolved.is_relative_to(root):
            err("link_escapes", f"link '{target}' points outside the skill directory")
        elif not resolved.exists():
            err("broken_link", f"link '{target}' does not exist")
        elif len(Path(rel).parts) > 2:
            warn("deep_reference", f"link '{target}' is more than one level deep")

    for p in sorted(skill_dir.rglob("*")):
        if p.is_symlink():
            err(
                "symlink",
                "symlinks are not allowed in a skill package",
                p.relative_to(skill_dir).as_posix(),
            )

    ref_name, ref_errors = _reference(skill_dir)
    for e in ref_errors:
        err("reference_validator", e)
    passed = not any(i.severity == "error" for i in issues)
    return ValidationReport(kind="spec", passed=passed, issues=issues, reference_validator=ref_name)
