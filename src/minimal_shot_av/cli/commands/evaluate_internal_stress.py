from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from minimal_shot_av.simulator.wod_scenarios import WOD_E2E_CLUSTERS
from minimal_shot_av.stress import STRESS_LEVELS, aggregate_internal_stress, evaluate_internal_stress


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ego-relative internal stress sweeps for simulator scenarios.")
    parser.add_argument("--seed-start", type=int, default=1, help="First seed to evaluate, inclusive.")
    parser.add_argument("--seed-end", type=int, default=5, help="Last seed to evaluate, inclusive.")
    parser.add_argument(
        "--policy",
        choices=("baseline", "spotlight-reflex", "both"),
        default="both",
        help="Policy set to evaluate.",
    )
    parser.add_argument(
        "--clusters",
        default="intersection,spotlight,construction,cut-in,foreign object debris",
        help="Comma-separated WOD clusters to evaluate.",
    )
    parser.add_argument(
        "--levels",
        default="stress,audit",
        help="Comma-separated internal stress levels to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/internal_stress"),
        help="Output directory for JSON and CSV reports.",
    )
    args = parser.parse_args()

    clusters = tuple(_parse_selection(args.clusters, WOD_E2E_CLUSTERS))
    levels = tuple(_parse_selection(args.levels, tuple(STRESS_LEVELS)))
    rows = evaluate_internal_stress(clusters, args.seed_start, args.seed_end, levels, args.policy)
    summary = aggregate_internal_stress(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "internal_stress_runs.csv"
    json_path = args.output_dir / "internal_stress_summary.json"

    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(
        json.dumps(
            {
                "runs": rows,
                "summary": summary,
                "manifest": {
                    "clusters": clusters,
                    "levels": levels,
                    "seed_start": args.seed_start,
                    "seed_end": args.seed_end,
                    "policies": sorted({str(row["policy"]) for row in rows}),
                    "unit": "closed-loop internal stress rollout",
                },
            },
            indent=2,
        )
    )
    print(f"Wrote {csv_path} and {json_path}")


def _parse_selection(raw: str, valid: tuple[str, ...]) -> list[str]:
    selected = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = [item for item in selected if item not in valid]
    if invalid:
        valid_text = ", ".join(valid)
        raise ValueError(f"invalid selection {invalid}; expected values from: {valid_text}")
    return selected


if __name__ == "__main__":
    main()
