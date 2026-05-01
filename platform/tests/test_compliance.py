"""Compliance tests.

These tests fail the build if any forbidden capability is introduced into
the codebase. The list is the canonical set from docs/COMPLIANCE.md.

A line that mentions a forbidden token in clearly-negated context (e.g. the
boundary banner that says "passive · no jamming · no engagement", or a
docstring saying "this API does not intercept") is permitted, since that is
a disclaimer, not an implementation. The check looks for the token NOT
appearing alongside a negation indicator on the same line.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

FORBIDDEN_TOKENS = [
    r"\bjam(?:mer|ming)?\b",
    r"\bneutrali[sz]e",
    r"\bneutrali[sz]ation",
    r"\binterceptor\b",
    r"\bintercept_target\b",
    r"\bkinetic\b",
    r"\bweapon(?:ize|ise)?\b",
    r"\bengage_target\b",
    r"\bspoof_gps\b",
    r"\bgps_spoof\b",
    r"\bhijack\b",
    r"\btakeover_drone\b",
    r"\bkill_switch\b",
    r"\bfire_command\b",
]

# Files that are allowed to mention these tokens (because they enumerate
# what's forbidden).
ALLOWLIST = {
    "docs/COMPLIANCE.md",
    "docs/SCOPE.md",
    "docs/ARCHITECTURE.md",
    "docs/RUNBOOK.md",
    "docs/PLAN.md",
    "docs/TEST_PLAN.md",
    "tests/test_compliance.py",
    "README.md",
}

SKIP_DIRS = {".git", "node_modules", "dist", "build", "__pycache__", ".pytest_cache"}

# Words that mark a line as a disclaimer / negation context.
NEGATION = re.compile(
    r"\b(no|not|never|without|cannot|excludes?|forbidden|prohibited|"
    r"do(?:es)? not|will not|won['']?t|n['']?t|disallow(?:ed|s)?)\b",
    re.IGNORECASE,
)


def _iter_source_files():
    for path in REPO.rglob("*"):
        if path.is_dir():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        rel = str(path.relative_to(REPO)).replace("\\", "/")
        if rel in ALLOWLIST:
            continue
        if path.suffix in {".py", ".ts", ".tsx", ".js", ".jsx",
                           ".html", ".md", ".json", ".yml", ".yaml",
                           ".cfg", ".ini", ".toml", ".sh"}:
            yield path, rel


def test_no_forbidden_symbols():
    """A forbidden token outside a disclaimer line is a violation."""
    pattern = re.compile("|".join(FORBIDDEN_TOKENS), re.IGNORECASE)
    hits: list[str] = []
    for path, rel in _iter_source_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not pattern.search(line):
                continue
            # Allow lines whose only purpose is to disclaim the token.
            if NEGATION.search(line):
                continue
            hits.append(f"{rel}:{line_no}: {line.strip()[:120]}")
    assert hits == [], (
        "Forbidden capability tokens found in source. "
        "Maple Shield is passive-only by design.\n"
        + "\n".join(hits)
    )


def test_no_delete_endpoint_anywhere():
    """No router file should expose @app.delete or .router.delete."""
    hits: list[str] = []
    for path, rel in _iter_source_files():
        if not rel.endswith(".py"):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for m in re.finditer(r"@\w+\.delete\s*\(", text):
            line_no = text.count("\n", 0, m.start()) + 1
            hits.append(f"{rel}:{line_no}")
    assert hits == [], (
        "DELETE endpoints are not allowed; the platform is append-only by design. "
        f"Found: {hits}"
    )


def test_negation_check_actually_catches_implementations():
    """Smoke test: the test would catch a real violation."""
    sample = "def engage_target(track_id):\n    return False\n"
    pat = re.compile("|".join(FORBIDDEN_TOKENS), re.IGNORECASE)
    for line in sample.splitlines():
        if pat.search(line):
            assert not NEGATION.search(line), (
                "Negation regex too permissive — would let through implementations"
            )
            return
    raise AssertionError("Forbidden token regex didn't match the smoke sample")
