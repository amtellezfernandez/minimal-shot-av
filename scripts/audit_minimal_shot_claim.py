#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIM_EVAL = ROOT / "artifacts" / "sota_submission_bundles" / "minor_eval" / "scenario_eval.json"
DEFAULT_MODEL_DECLARATION = ROOT / "models" / "DECLARATION.md"
DEFAULT_GRAND_SUBMISSION = ROOT / "docs" / "grand-submission.md"
DEFAULT_SUBMISSION_FORM = ROOT / "docs" / "sota-grand-submission-form.md"
DEFAULT_OUTPUT = ROOT / "artifacts" / "minimal_shot_claim_audit.json"
DEFAULT_POLICY_SOURCES = (
    ROOT / "src" / "minimal_shot_av" / "simulator" / "spotlight_reflex.py",
    ROOT / "src" / "minimal_shot_av" / "simulator" / "policy.py",
    ROOT / "src" / "minimal_shot_av" / "simulator" / "perception.py",
    ROOT / "src" / "minimal_shot_av" / "simulator" / "world_model.py",
    ROOT / "src" / "minimal_shot_av" / "simulator" / "safety.py",
    ROOT / "src" / "minimal_shot_av" / "simulator" / "trajectory_selector.py",
    ROOT / "src" / "minimal_shot_av" / "simulator" / "planner.py",
)

EPISODE_DEPENDENCY_PATTERNS = (
    r"\bload_preference_frames\b",
    r"\bwod_preference\b",
    r"\bpreference_trajectories\b",
    r"\bframe-cache\b",
    r"\bnearest_train\b",
    r"\bnearest_preferences\b",
    r"\bmemory_neighbor\b",
    r"\branker_score\b",
    r"\bfit_[a-z0-9_]*selector\b",
    r"\btrain_margin\b",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit whether the Grand claim is substantively minimal-shot.")
    parser.add_argument("--sim-eval", type=Path, default=DEFAULT_SIM_EVAL)
    parser.add_argument("--model-declaration", type=Path, default=DEFAULT_MODEL_DECLARATION)
    parser.add_argument("--grand-submission", type=Path, default=DEFAULT_GRAND_SUBMISSION)
    parser.add_argument("--submission-form", type=Path, default=DEFAULT_SUBMISSION_FORM)
    parser.add_argument("--policy-source", type=Path, action="append", default=list(DEFAULT_POLICY_SOURCES))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = build_report(
        sim_eval=args.sim_eval,
        model_declaration=args.model_declaration,
        grand_submission=args.grand_submission,
        submission_form=args.submission_form,
        policy_sources=tuple(args.policy_source),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


def build_report(
    *,
    sim_eval: Path = DEFAULT_SIM_EVAL,
    model_declaration: Path = DEFAULT_MODEL_DECLARATION,
    grand_submission: Path = DEFAULT_GRAND_SUBMISSION,
    submission_form: Path = DEFAULT_SUBMISSION_FORM,
    policy_sources: tuple[Path, ...] = DEFAULT_POLICY_SOURCES,
) -> dict[str, Any]:
    sim = _load_json(sim_eval)
    model_text = _read_text(model_declaration)
    grand_text = _read_text(grand_submission)
    form_text = _read_text(submission_form)

    policy_scan = _scan_policy_sources(policy_sources)
    sim_metrics = _sim_metrics(sim)
    boundary = _claim_boundary(model_text=model_text, grand_text=grand_text, form_text=form_text)

    checks = {
        "active_policy_has_no_episode_lookup": not policy_scan["matches"],
        "randomized_long_tail_coverage": sim_metrics["cluster_count"] >= 11 and sim_metrics["run_count"] >= 550,
        "closed_loop_zero_collision": sim_metrics["max_collision_rate"] == 0.0,
        "closed_loop_no_near_miss": sim_metrics["max_near_miss_rate"] == 0.0,
        "wod_evidence_declared_auxiliary": boundary["wod_auxiliary"],
        "validation_preferences_not_primary_claim": boundary["validation_preferences_not_primary"],
        "no_strict_zero_shot_wod_claim": boundary["no_strict_zero_shot_wod_claim"],
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "schema": "minimal_shot_claim_audit_v1",
        "valid": not failures,
        "failures": failures,
        "checks": checks,
        "inputs": {
            "sim_eval": str(sim_eval),
            "model_declaration": str(model_declaration),
            "grand_submission": str(grand_submission),
            "submission_form": str(submission_form),
            "policy_sources": [str(path) for path in policy_sources],
        },
        "metrics": {
            "simulation": sim_metrics,
            "policy_static_scan": policy_scan,
            "claim_boundary": boundary,
        },
        "interpretation": (
            "Pass means the primary Grand claim is supported by the closed-loop Spotlight Reflex "
            "runtime path and randomized scenario evidence, while WOD preference-calibrated results "
            "remain auxiliary benchmark analysis."
        ),
    }


def _scan_policy_sources(paths: tuple[Path, ...]) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    missing: list[str] = []
    for path in paths:
        if not path.is_file():
            missing.append(str(path))
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in EPISODE_DEPENDENCY_PATTERNS:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                line = text.count("\n", 0, match.start()) + 1
                matches.append({"path": str(path), "line": line, "pattern": pattern})
    return {
        "source_count": len(paths),
        "missing": missing,
        "matches": matches,
        "forbidden_patterns": list(EPISODE_DEPENDENCY_PATTERNS),
    }


def _sim_metrics(sim: dict[str, Any]) -> dict[str, Any]:
    summary = sim.get("summary", [])
    if not isinstance(summary, list) or not summary:
        raise ValueError("simulation eval must contain a non-empty summary list")
    return {
        "cluster_count": len(summary),
        "run_count": sum(int(row.get("runs", 0)) for row in summary),
        "min_success_rate": min(float(row.get("success_rate", 0.0)) for row in summary),
        "min_benchmark_pass_rate": min(float(row.get("benchmark_pass_rate", 0.0)) for row in summary),
        "max_collision_rate": max(float(row.get("collision_rate", 1.0)) for row in summary),
        "max_near_miss_rate": max(float(row.get("near_miss_rate", 1.0)) for row in summary),
    }


def _claim_boundary(*, model_text: str, grand_text: str, form_text: str) -> dict[str, bool]:
    all_text = "\n".join([model_text, grand_text, form_text]).lower()
    return {
        "wod_auxiliary": (
            "wod validation-cv evidence is declared as auxiliary" in all_text
            or "wod-e2e parsing/submission harness is included, but it is benchmark infrastructure" in all_text
            or "not the centerpiece of the minimal-shot claim" in all_text
        ),
        "validation_preferences_not_primary": (
            "validation preference labels" in all_text
            and ("not strict zero-shot" in all_text or "not the core zero-shot claim" in all_text)
        ),
        "no_strict_zero_shot_wod_claim": (
            "strict zero-shot wod-e2e" in all_text
            and ("should not be described" in all_text or "not a completed leaderboard entry" in all_text)
        ),
    }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


if __name__ == "__main__":
    raise SystemExit(main())
