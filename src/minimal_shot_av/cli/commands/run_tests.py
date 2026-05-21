from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[4]
TESTS = ROOT / "tests"
SLOW_MODULES = frozenset({"tests.test_certification", "tests.test_compass"})


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
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip slow benchmark/evidence modules for a tight local edit loop.",
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Run only slow benchmark/evidence modules.",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failed module.")
    args = parser.parse_args()
    if args.quick and args.slow:
        parser.error("--quick and --slow cannot be used together")

    modules = _selected_modules(args.tests, quick=args.quick, slow=args.slow)
    workers = _worker_count(args.workers, len(modules))
    mode = "quick" if args.quick else "slow" if args.slow else "full"
    print(f"Running {len(modules)} {mode} test module(s) with {workers} worker(s)")

    start = time.perf_counter()
    results = _run_parallel(modules, workers, fail_fast=args.fail_fast)
    elapsed = time.perf_counter() - start
    failed = [result for result in results if result.returncode != 0]

    for result in sorted(results, key=lambda item: item.elapsed_s, reverse=True):
        status = "ok" if result.returncode == 0 else "FAIL"
        print(f"{status:4s} {result.elapsed_s:6.2f}s {result.module}")
        if result.returncode != 0:
            print(result.output.rstrip())

    print(f"Finished in {elapsed:.2f}s")
    if failed:
        raise SystemExit(1)


def _selected_modules(items: list[str], *, quick: bool = False, slow: bool = False) -> list[str]:
    if not items:
        modules = [f"tests.{path.stem}" for path in sorted(TESTS.glob("test_*.py"))]
        if quick:
            return [module for module in modules if module not in SLOW_MODULES]
        if slow:
            return [module for module in modules if module in SLOW_MODULES]
        return modules
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


def _run_parallel(modules: list[str], workers: int, *, fail_fast: bool = False) -> list[TestResult]:
    if workers == 1:
        results: list[TestResult] = []
        for module in modules:
            result = _run_module(module)
            results.append(result)
            if fail_fast and result.returncode != 0:
                break
        return results
    results: list[TestResult] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_module, module): module for module in modules}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if fail_fast and result.returncode != 0:
                for pending in futures:
                    pending.cancel()
                break
    return results


def _run_module(module: str) -> TestResult:
    env = os.environ.copy()
    existing_path = env.get("PYTHONPATH")
    src_path = ROOT / "src"
    env["PYTHONPATH"] = str(src_path) if not existing_path else f"{src_path}{os.pathsep}{existing_path}"
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
