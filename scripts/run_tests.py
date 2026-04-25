from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TESTS = ROOT / "tests"


@dataclass(frozen=True)
class TestResult:
    module: str
    returncode: int
    elapsed_s: float
    output: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unittest modules with laptop-friendly parallelism.")
    parser.add_argument("tests", nargs="*", help="Optional unittest modules or test file paths.")
    parser.add_argument(
        "-w",
        "--workers",
        default="auto",
        help="Worker count, 'auto', 'max', or 1 for serial. Auto is laptop-friendly and caps at 4.",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Stop scheduling output after the first failed module.")
    args = parser.parse_args()

    modules = _selected_modules(args.tests)
    workers = _worker_count(args.workers, len(modules))
    print(f"Running {len(modules)} test module(s) with {workers} worker(s)")

    start = time.perf_counter()
    results = _run_parallel(modules, workers)
    elapsed = time.perf_counter() - start
    failed = [result for result in results if result.returncode != 0]

    for result in sorted(results, key=lambda item: item.elapsed_s, reverse=True):
        status = "ok" if result.returncode == 0 else "FAIL"
        print(f"{status:4s} {result.elapsed_s:6.2f}s {result.module}")
        if result.returncode != 0:
            print(result.output.rstrip())
            if args.fail_fast:
                break

    print(f"Finished in {elapsed:.2f}s")
    if failed:
        raise SystemExit(1)


def _selected_modules(items: list[str]) -> list[str]:
    if not items:
        return [f"tests.{path.stem}" for path in sorted(TESTS.glob("test_*.py"))]
    modules: list[str] = []
    for item in items:
        path = Path(item)
        if path.suffix == ".py":
            modules.append(f"tests.{path.stem}")
        else:
            modules.append(item)
    return modules


def _worker_count(value: str, module_count: int) -> int:
    if module_count <= 0:
        return 1
    if value == "auto":
        cpu_count = os.cpu_count() or 1
        return max(1, min(module_count, max(1, cpu_count - 1), 4))
    if value == "max":
        return module_count
    workers = int(value)
    if workers < 1:
        raise ValueError("workers must be >= 1")
    return min(workers, module_count)


def _run_parallel(modules: list[str], workers: int) -> list[TestResult]:
    if workers == 1:
        return [_run_module(module) for module in modules]
    results: list[TestResult] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_module, module): module for module in modules}
        for future in as_completed(futures):
            results.append(future.result())
    return results


def _run_module(module: str) -> TestResult:
    env = os.environ.copy()
    existing_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC) if not existing_path else f"{SRC}{os.pathsep}{existing_path}"
    start = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", module],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return TestResult(
        module=module,
        returncode=proc.returncode,
        elapsed_s=time.perf_counter() - start,
        output=proc.stdout,
    )


if __name__ == "__main__":
    main()
