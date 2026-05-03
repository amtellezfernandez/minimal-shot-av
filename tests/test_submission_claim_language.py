from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]

CLAIM_DOCS = [
    ROOT / "README.md",
    ROOT / "models" / "DECLARATION.md",
    ROOT / "docs",
    ROOT / "notebooks",
]

RISKY_CLAIMS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bleaderboard winner\b",
        r"\bproduction autonomy\b",
        r"\bproduction AV\b",
        r"\bstrict zero-shot\b",
        r"\bsolved WOD(?:-E2E)?\b",
        r"\bcompleted leaderboard\b",
        r"\bcomplete leaderboard\b",
        r"\bhidden-test (?:leaderboard )?result\b",
        r"\bhidden test (?:leaderboard )?result\b",
    ]
]

CAVEAT_MARKERS = [
    "not",
    "no ",
    "does not",
    "do not",
    "cannot",
    "blocked",
    "missing",
    "needs",
    "future",
    "without",
    "rather than",
    "should not",
]


class SubmissionClaimLanguageTests(unittest.TestCase):
    def test_risky_competition_claims_are_caveated(self) -> None:
        violations: list[str] = []
        for path in _claim_files():
            lines = path.read_text(encoding="utf-8").splitlines()
            for index, line in enumerate(lines):
                if not any(pattern.search(line) for pattern in RISKY_CLAIMS):
                    continue
                window = "\n".join(lines[max(0, index - 1) : min(len(lines), index + 2)]).lower()
                if not any(marker in window for marker in CAVEAT_MARKERS):
                    violations.append(f"{path.relative_to(ROOT)}:{index + 1}: {line.strip()}")

        self.assertEqual([], violations)


def _claim_files() -> list[Path]:
    files: list[Path] = []
    for entry in CLAIM_DOCS:
        if entry.is_file():
            files.append(entry)
        else:
            files.extend(path for path in entry.rglob("*") if path.suffix in {".md", ".ipynb"})
    return sorted(files)


if __name__ == "__main__":
    unittest.main()
