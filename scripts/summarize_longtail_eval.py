#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_SLICE_KEYS = (
    "unseen:yes",
    "unseen:no",
    "rare_behavior:yes",
    "rare_behavior:no",
    "uncertainty:high",
    "uncertainty:medium",
    "uncertainty:low",
    "failure_recovery:missed",
    "failure_recovery:successful",
    "ambiguity:high",
    "ambiguity:medium",
    "ambiguity:low",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a compact long-tail comparison summary from two WOD CV reports.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    summary = {
        "baseline_report": str(args.baseline),
        "candidate_report": str(args.candidate),
        "overall": _overall_summary(baseline, candidate),
        "long_tail_slices": _slice_summary(baseline, candidate),
        "failure_case_buckets": _failure_case_summary(baseline, candidate),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(_markdown_summary(summary), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    return 0


def _overall_summary(baseline: dict[str, object], candidate: dict[str, object]) -> dict[str, dict[str, float]]:
    keys = (
        "combined_ranker_mean_normalized_rfs",
        "combined_ranker_mean_rfs",
        "combined_ranker_regret_to_oracle",
        "combined_ranker_top1_oracle_match_rate",
        "selected_kinematic_rate",
        "selected_learned_rate",
    )
    return {
        key: _metric_triplet(float(baseline[key]), float(candidate[key]))
        for key in keys
    }


def _slice_summary(baseline: dict[str, object], candidate: dict[str, object]) -> list[dict[str, object]]:
    baseline_slices = dict(baseline.get("long_tail_slices", {}))
    candidate_slices = dict(candidate.get("long_tail_slices", {}))
    rows: list[dict[str, object]] = []
    for key in DEFAULT_SLICE_KEYS:
        base = baseline_slices.get(key)
        cand = candidate_slices.get(key)
        if base is None or cand is None:
            continue
        rows.append(
            {
                "slice": key,
                "frames": int(base["frames"]),
                "selected_mean_rfs": _metric_triplet(
                    float(base["selected_mean_rfs"]),
                    float(cand["selected_mean_rfs"]),
                ),
                "mean_regret": _metric_triplet(
                    float(base["mean_regret"]),
                    float(cand["mean_regret"]),
                ),
            }
        )
    return rows


def _failure_case_summary(
    baseline: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, object]:
    baseline_buckets = list(baseline.get("failure_case_buckets", []))
    candidate_buckets = list(candidate.get("failure_case_buckets", []))
    return {
        "baseline_count": len(baseline_buckets),
        "candidate_count": len(candidate_buckets),
        "candidate_buckets": [
            {
                "bucket": str(bucket["bucket"]),
                "frames": int(bucket["frames"]),
                "total_regret": float(bucket["total_regret"]),
                "max_regret": float(bucket["max_regret"]),
                "examples": list(bucket.get("examples", []))[:3],
            }
            for bucket in candidate_buckets
        ],
    }


def _metric_triplet(baseline_value: float, candidate_value: float) -> dict[str, float]:
    return {
        "baseline": baseline_value,
        "candidate": candidate_value,
        "delta": candidate_value - baseline_value,
    }


def _markdown_summary(summary: dict[str, object]) -> str:
    overall = dict(summary["overall"])
    slice_rows = list(summary["long_tail_slices"])
    failure_cases = dict(summary["failure_case_buckets"])
    lines = [
        "# Long-Tail Evaluation Summary",
        "",
        "## Overall",
        "",
        "| Metric | Baseline | Candidate | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, metric in overall.items():
        metric_dict = dict(metric)
        lines.append(
            f"| `{key}` | {metric_dict['baseline']:.4f} | {metric_dict['candidate']:.4f} | {metric_dict['delta']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Long-Tail Slices",
            "",
            "| Slice | Frames | Selected RFS Baseline | Selected RFS Candidate | Delta | Regret Baseline | Regret Candidate | Delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in slice_rows:
        selected = dict(row["selected_mean_rfs"])
        regret = dict(row["mean_regret"])
        lines.append(
            "| `{slice}` | {frames} | {sb:.4f} | {sc:.4f} | {sd:+.4f} | {rb:.4f} | {rc:.4f} | {rd:+.4f} |".format(
                slice=row["slice"],
                frames=row["frames"],
                sb=selected["baseline"],
                sc=selected["candidate"],
                sd=selected["delta"],
                rb=regret["baseline"],
                rc=regret["candidate"],
                rd=regret["delta"],
            )
        )
    lines.extend(
        [
            "",
            "## Failure Cases",
            "",
            f"- baseline buckets: {failure_cases['baseline_count']}",
            f"- candidate buckets: {failure_cases['candidate_count']}",
        ]
    )
    for bucket in failure_cases["candidate_buckets"]:
        lines.extend(
            [
                "",
                f"### `{bucket['bucket']}`",
                "",
                f"- frames: {bucket['frames']}",
                f"- total regret: {bucket['total_regret']:.4f}",
                f"- max regret: {bucket['max_regret']:.4f}",
            ]
        )
        for example in bucket["examples"]:
            lines.append(
                "- `{frame}` | speed `{speed}` | selected `{selected}` -> oracle `{oracle}` | regret `{regret:.4f}`".format(
                    frame=example["frame_name"],
                    speed=example["speed_bin"],
                    selected=example["selected_candidate"],
                    oracle=example["oracle_candidate"],
                    regret=float(example["regret"]),
                )
            )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
