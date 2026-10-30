"""Content and security lints for generated skills (beyond the format spec)."""

from __future__ import annotations

import re
from pathlib import Path

from skill_foundry.domain.models import ProcedureSpec, ValidationIssue, ValidationReport
from skill_foundry.ingest.sanitize import redact_text, scan_injection
from skill_foundry.mining.features import extract_commands, normalize_command
from skill_foundry.synthesis.leakage import LeakageGuard
from skill_foundry.validation.spec import split_frontmatter

REQUIRED_SECTIONS = (
    "When to use",
    "When NOT to use",
    "Assumptions",
    "Procedure",
    "Verification",
    "Failure recovery",
    "References",
)
UNSAFE = [
    ("pipe_to_shell", re.compile(r"(?i)\b(curl|wget)\b[^\n`]*\|\s*(sudo\s+)?(ba|z)?sh\b")),
    ("sudo", re.compile(r"(?<![\w-])sudo\s")),
    ("rm_rf_root", re.compile(r"rm\s+-[a-z]*r[a-z]*f?\s+(/|~|\$HOME)(\s|$)")),
    ("chmod_777", re.compile(r"chmod\s+(-R\s+)?777")),
    ("skip_hooks", re.compile(r"--no-verify\b")),
    ("force_push", re.compile(r"git\s+push\s+[^\n]*(--force\b|-f\b)")),
    (
        "disable_tests",
        re.compile(r"(?i)(pytest\.mark\.skip|@unittest\.skip|--deselect|-k\s+'?not\s)"),
    ),
    ("credential_read", re.compile(r"(?i)(~/\.ssh|~/\.aws|\.netrc|id_rsa|printenv|env\s*\|)")),
    ("skip_permissions", re.compile(r"--dangerously-skip-permissions|bypassPermissions")),
]
_ABS_PATH = re.compile(r"(?<![\w.<])(/home/|/Users/|/root/|[A-Z]:\\\\Users)")
_CODE = re.compile(r"`([^`\n]+)`")


def lint_skill_dir(
    skill_dir: Path,
    spec: ProcedureSpec | None = None,
    *,
    scripts_approved: bool = False,
    leakage: LeakageGuard | None = None,
    training_texts: list[str] | None = None,
) -> ValidationReport:
    issues: list[ValidationIssue] = []

    def add(code: str, sev: str, msg: str, path: str | None = "SKILL.md") -> None:
        issues.append(ValidationIssue(code=code, severity=sev, message=msg, path=path))

    text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    fm, body, _ = split_frontmatter(text)
    headings = {h.strip() for h in re.findall(r"(?m)^##\s+(.+)$", body)}
    for section in REQUIRED_SECTIONS:
        if section not in headings:
            add("missing_section", "error", f"required section '## {section}' is missing")
    if fm and "allowed-tools" in fm:
        add(
            "allowed_tools",
            "error",
            "generated skills may not pre-approve tools "
            "(verified source evidence never grants extra authority)",
        )
    scripts = skill_dir / "scripts"
    if scripts.exists() and any(scripts.rglob("*")) and not scripts_approved:
        add("unapproved_scripts", "error", "scripts/ present without explicit approval", "scripts/")

    for p in sorted(skill_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(skill_dir).as_posix()
        content = p.read_text(encoding="utf-8", errors="replace")
        if redact_text(content) != content:
            add("secret", "error", "content matches a secret pattern", rel)
        if _ABS_PATH.search(content):
            add("absolute_path", "error", "machine-specific absolute path found", rel)
        for kind, pat in UNSAFE:
            if pat.search(content):
                add(f"unsafe_{kind}", "error", f"unsafe instruction pattern: {kind}", rel)
        hits = scan_injection(content)
        if hits:
            add("injection_text", "error", f"instruction-like injected text: {hits}", rel)
        if leakage is not None:
            for finding in leakage.findings(content):
                add("heldout_leakage", "error", finding, rel)

    if spec is not None:
        allowed = {normalize_command(c) for s in spec.steps for c in s.commands}
        allowed |= {
            normalize_command(c) for b in spec.branches for s in b.steps for c in s.commands
        }
        allowed |= {normalize_command(c) for v in spec.verification for c in v.commands}
        allowed |= {normalize_command(c) for r in spec.recovery for c in r.commands}
        # Commands quoted from observed failures (what *not* to do) are evidence too.
        evidenced_text = [r.when for r in spec.recovery]
        evidenced_text += [f.text for f in spec.failed_approaches]
        evidenced_text += [c.text for c in spec.contraindications]
        allowed |= {normalize_command(c) for t in evidenced_text for c in extract_commands(t)}
        for span in _CODE.findall(body):
            for cmd in extract_commands(f"`{span}`"):
                if normalize_command(cmd) not in allowed:
                    add(
                        "invented_command",
                        "error",
                        f"command `{cmd}` is not supported by any evidence in the spec",
                    )
    if training_texts:
        body_grams = _grams(body, 12)
        for i, t in enumerate(training_texts):
            if body_grams & _grams(t, 12):
                add(
                    "verbatim_training_text",
                    "warning",
                    f"SKILL.md shares a 12-word span with training experience #{i}",
                )
                break
    passed = not any(i.severity == "error" for i in issues)
    return ValidationReport(kind="lint", passed=passed, issues=issues)


def _grams(text: str, n: int) -> set[str]:
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}
