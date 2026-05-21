from __future__ import annotations

import argparse
import py_compile
from dataclasses import dataclass
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[4]
PYPROJECT = ROOT / "pyproject.toml"
DEFAULT_PATHS = (
    ROOT / "src" / "minimal_shot_av" / "model",
    ROOT / "src" / "minimal_shot_av" / "neutral",
    ROOT / "scripts",
    ROOT / "tests",
)
QUALITY_SECTION = "tool.minimal_shot_av.quality"
_SECTION_RE = re.compile(r"^\[(?P<section>.+)]$")
_INT_ASSIGNMENT_RE = re.compile(r"^(?P<key>[A-Za-z0-9_.-]+)\s*=\s*(?P<value>\d+)\s*$")
_ARRAY_START_RE = re.compile(r"^(?P<key>[A-Za-z0-9_.-]+)\s*=\s*\[\s*$")
_STRING_ITEM_RE = re.compile(r'^"(?P<value>.*)",?\s*$')


@dataclass(frozen=True)
class QualityConfig:
    max_line_length: int
    forbidden_python310_tokens: tuple[str, ...]


@dataclass(frozen=True)
class QualityIssue:
    path: Path
    line_number: int
    message: str

    def format(self) -> str:
        relative_path = self.path.relative_to(ROOT)
        return f"{relative_path}:{self.line_number}: {self.message}"


def discover_python_files(paths: tuple[Path, ...] = DEFAULT_PATHS) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.py")))
    return sorted(set(files))


def load_quality_config(path: str | Path = PYPROJECT) -> QualityConfig:
    section = _load_pyproject_section(Path(path), QUALITY_SECTION)
    max_line_length = section.get("max_line_length")
    forbidden_tokens = section.get("forbidden_python310_tokens")
    if not isinstance(max_line_length, int):
        raise ValueError(f"{QUALITY_SECTION}.max_line_length must be configured as an integer")
    if not isinstance(forbidden_tokens, tuple):
        raise ValueError(f"{QUALITY_SECTION}.forbidden_python310_tokens must be configured as a string array")
    return QualityConfig(max_line_length=max_line_length, forbidden_python310_tokens=forbidden_tokens)


def check_files(
    paths: tuple[Path, ...] = DEFAULT_PATHS,
    *,
    config: QualityConfig | None = None,
) -> list[QualityIssue]:
    config = config or load_quality_config()
    issues: list[QualityIssue] = []
    for path in discover_python_files(paths):
        issues.extend(_compile_issues(path))
        issues.extend(_line_quality_issues(path, config=config))
    return issues


def _compile_issues(path: Path) -> list[QualityIssue]:
    try:
        py_compile.compile(str(path), doraise=True)
    except py_compile.PyCompileError as exc:
        return [QualityIssue(path, 1, f"compile failed: {exc.msg}")]
    return []


def _line_quality_issues(path: Path, *, config: QualityConfig) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if len(line) > config.max_line_length:
            issues.append(QualityIssue(path, line_number, f"line too long: {len(line)} > {config.max_line_length}"))
        for token in config.forbidden_python310_tokens:
            if token in line:
                issues.append(QualityIssue(path, line_number, f"Python 3.10-incompatible token: {token}"))
    return issues


def _load_pyproject_section(path: Path, section_name: str) -> dict[str, int | tuple[str, ...]]:
    values: dict[str, int | tuple[str, ...]] = {}
    active_section: str | None = None
    active_array_key: str | None = None
    active_array_values: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if active_array_key is not None:
            if line == "]":
                values[active_array_key] = tuple(active_array_values)
                active_array_key = None
                active_array_values = []
                continue
            item_match = _STRING_ITEM_RE.match(line)
            if item_match:
                active_array_values.append(item_match.group("value"))
            continue
        section_match = _SECTION_RE.match(line)
        if section_match:
            active_section = section_match.group("section")
            continue
        if active_section != section_name:
            continue
        int_match = _INT_ASSIGNMENT_RE.match(line)
        if int_match:
            values[int_match.group("key")] = int(int_match.group("value"))
            continue
        array_match = _ARRAY_START_RE.match(line)
        if array_match:
            active_array_key = array_match.group("key")
            active_array_values = []
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="Run lightweight code-quality checks without external linters.")
    parser.add_argument("paths", nargs="*", type=Path, help="Optional files or directories to check.")
    parser.add_argument("--config", type=Path, default=PYPROJECT, help="pyproject.toml containing quality config.")
    args = parser.parse_args()

    paths = tuple(path if path.is_absolute() else ROOT / path for path in args.paths) or DEFAULT_PATHS
    issues = check_files(paths, config=load_quality_config(args.config))
    for issue in issues:
        print(issue.format())
    if issues:
        return 1
    print(f"checked {len(discover_python_files(paths))} Python file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
