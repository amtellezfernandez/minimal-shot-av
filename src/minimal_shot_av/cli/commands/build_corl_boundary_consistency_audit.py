#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from statistics import pstdev
from typing import Any
from typing import Mapping


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT_JSON = ROOT / "artifacts" / "corl2027" / "replay_value_repeated_splits_summary.json"
DEFAULT_OUTPUT_JSON = ROOT / "artifacts" / "corl2027" / "boundary_consistency_audit.json"
DEFAULT_OUTPUT_MD = ROOT / "artifacts" / "corl2027" / "boundary_consistency_audit.md"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a CoRL-facing boundary-consistency audit from repeated DB-split replay-value results."
    )
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--tolerance", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = json.loads(args.input_json.read_text(encoding="utf-8"))
    audit = build_audit(report, tolerance=float(args.tolerance), input_json=str(args.input_json))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown_audit(audit) + "\n", encoding="utf-8")
    print(markdown_audit(audit))
    return 0


def build_audit(report: Mapping[str, Any], *, tolerance: float, input_json: str) -> dict[str, Any]:
    splits = []
    scene_token_improved = 0
    safe_within = 0
    safe_exact = 0
    balanced_within = 0
    balanced_exact = 0
    balanced_progress_gain_positive = 0
    for split in report.get("splits", []):
        selectors = split["selectors"]
        proxy = selectors["proxy_top1"]
        scene = selectors["scene_token_student"]
        safe = selectors["replay_value_safe"]
        balanced = selectors["replay_value_balanced"]
        oracle = selectors["replay_value_oracle"]
        scene_gain = float(proxy["replay_fail_rate"]) - float(scene["replay_fail_rate"])
        safe_gap = float(safe["replay_fail_rate"]) - float(oracle["replay_fail_rate"])
        balanced_gap = float(balanced["replay_fail_rate"]) - float(oracle["replay_fail_rate"])
        balanced_progress_gain = float(balanced["progress_m"]) - float(safe["progress_m"])
        scene_token_improved += int(scene_gain > 0.0)
        safe_within += int(safe_gap <= tolerance)
        safe_exact += int(abs(safe_gap) <= 1.0e-9)
        balanced_within += int(balanced_gap <= tolerance)
        balanced_exact += int(abs(balanced_gap) <= 1.0e-9)
        balanced_progress_gain_positive += int(balanced_progress_gain > 0.0)
        splits.append(
            {
                "seed": int(split["seed"]),
                "scene_token_fail_improvement": round(scene_gain, 6),
                "safe_oracle_gap": round(safe_gap, 6),
                "balanced_oracle_gap": round(balanced_gap, 6),
                "balanced_progress_gain_vs_safe_m": round(balanced_progress_gain, 6),
            }
        )
    safe_gaps = [row["safe_oracle_gap"] for row in splits]
    balanced_gaps = [row["balanced_oracle_gap"] for row in splits]
    progress_gains = [row["balanced_progress_gain_vs_safe_m"] for row in splits]
    scene_improvements = [row["scene_token_fail_improvement"] for row in splits]
    split_count = max(1, len(splits))
    return {
        "schema": "corl_boundary_consistency_audit_v1",
        "input_json": input_json,
        "split_count": len(splits),
        "tolerance": float(tolerance),
        "summary": {
            "scene_token_improved_over_proxy_count": scene_token_improved,
            "scene_token_improved_over_proxy_rate": round(scene_token_improved / split_count, 6),
            "safe_within_tolerance_count": safe_within,
            "safe_within_tolerance_rate": round(safe_within / split_count, 6),
            "safe_exact_match_count": safe_exact,
            "safe_exact_match_rate": round(safe_exact / split_count, 6),
            "balanced_within_tolerance_count": balanced_within,
            "balanced_within_tolerance_rate": round(balanced_within / split_count, 6),
            "balanced_exact_match_count": balanced_exact,
            "balanced_exact_match_rate": round(balanced_exact / split_count, 6),
            "balanced_progress_gain_positive_count": balanced_progress_gain_positive,
            "balanced_progress_gain_positive_rate": round(balanced_progress_gain_positive / split_count, 6),
            "safe_oracle_gap_mean": round(mean(safe_gaps), 6) if safe_gaps else 0.0,
            "safe_oracle_gap_std": round(pstdev(safe_gaps), 6) if len(safe_gaps) > 1 else 0.0,
            "balanced_oracle_gap_mean": round(mean(balanced_gaps), 6) if balanced_gaps else 0.0,
            "balanced_oracle_gap_std": round(pstdev(balanced_gaps), 6) if len(balanced_gaps) > 1 else 0.0,
            "balanced_progress_gain_mean_m": round(mean(progress_gains), 6) if progress_gains else 0.0,
            "balanced_progress_gain_std_m": round(pstdev(progress_gains), 6) if len(progress_gains) > 1 else 0.0,
            "scene_token_fail_improvement_mean": round(mean(scene_improvements), 6) if scene_improvements else 0.0,
            "scene_token_fail_improvement_std": round(pstdev(scene_improvements), 6) if len(scene_improvements) > 1 else 0.0,
        },
        "splits": splits,
    }


def markdown_audit(audit: Mapping[str, Any]) -> str:
    summary = audit["summary"]
    lines = [
        "# Boundary-Consistency Audit",
        "",
        f"- Input JSON: `{audit['input_json']}`",
        f"- Split count: `{audit['split_count']}`",
        f"- Oracle-gap tolerance: `{audit['tolerance']:.3f}`",
        "",
        "## Summary",
        "",
        f"- Scene-token improved over proxy in `{summary['scene_token_improved_over_proxy_count']}/{audit['split_count']}` splits.",
        f"- Safety-heavy replay value stayed within oracle tolerance in `{summary['safe_within_tolerance_count']}/{audit['split_count']}` splits and matched exactly in `{summary['safe_exact_match_count']}/{audit['split_count']}`.",
        f"- Balanced replay value stayed within oracle tolerance in `{summary['balanced_within_tolerance_count']}/{audit['split_count']}` splits and matched exactly in `{summary['balanced_exact_match_count']}/{audit['split_count']}`.",
        f"- Balanced replay value improved progress over the safety-heavy point in `{summary['balanced_progress_gain_positive_count']}/{audit['split_count']}` splits.",
        f"- Mean oracle gap: safety-heavy `{summary['safe_oracle_gap_mean']:.3f} ± {summary['safe_oracle_gap_std']:.3f}`, balanced `{summary['balanced_oracle_gap_mean']:.3f} ± {summary['balanced_oracle_gap_std']:.3f}`.",
        f"- Mean balanced progress gain over safety-heavy: `{summary['balanced_progress_gain_mean_m']:.3f} ± {summary['balanced_progress_gain_std_m']:.3f} m`.",
        "",
        "| Seed | Scene-token fail improvement | Safety-heavy oracle gap | Balanced oracle gap | Balanced progress gain vs safety-heavy (m) |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in audit["splits"]:
        lines.append(
            f"| {split['seed']} | {split['scene_token_fail_improvement']:.3f} | "
            f"{split['safe_oracle_gap']:.3f} | {split['balanced_oracle_gap']:.3f} | "
            f"{split['balanced_progress_gain_vs_safe_m']:.3f} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
