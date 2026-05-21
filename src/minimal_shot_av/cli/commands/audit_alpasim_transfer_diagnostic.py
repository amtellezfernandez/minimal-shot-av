from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minimal_shot_av.cli.commands.summarize_alpasim_episodes import summarize_run


BINARY_EPISODE_METRICS = (
    "raw_collision_any",
    "raw_offroad",
    "raw_wrong_lane",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit AlpaSim transfer diagnostics from completed run directories. "
            "For statistically cleaner veto estimates, pass one scene per run."
        )
    )
    parser.add_argument("run_dir", nargs="+", type=Path, help="Completed AlpaSim run directories.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2027)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = build_report(
        args.run_dir,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_markdown(report)


def build_report(
    run_dirs: list[Path],
    *,
    bootstrap_samples: int = 10000,
    seed: int = 2027,
) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    run_reports: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        run_dir = run_dir.resolve()
        episode_rows = summarize_run(run_dir, max_dist_to_gt=4.0)
        selection = _selection_stats(run_dir)
        run_reports.append(
            {
                "run_dir": str(run_dir),
                "run": run_dir.name,
                "episode_count": len(episode_rows),
                "clip_ids": sorted({str(row["clipgt_id"]) for row in episode_rows}),
                "episode_metrics": _episode_metric_summary(episode_rows),
                "selection": selection,
            }
        )
        episodes.extend(episode_rows)

    scene_veto_rates = [
        report["selection"]["veto_rate"]
        for report in run_reports
        if report["episode_count"] == 1 and report["selection"]["frame_count"] > 0
    ]
    all_selection = _combine_selection_stats([report["selection"] for report in run_reports])
    return {
        "run_count": len(run_reports),
        "episode_count": len(episodes),
        "runs": run_reports,
        "episode_metrics": _episode_metric_summary(episodes),
        "selection": all_selection,
        "per_scene_veto": _rate_sample_summary(
            scene_veto_rates,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        "notes": [
            "Per-scene veto statistics require one clip per run because the current AlpaSim PredictionInput does not expose scene_id to the model.",
            "Frame-level confidence intervals are descriptive only; frames within one rollout are temporally correlated.",
        ],
    }


def _selection_stats(run_dir: Path) -> dict[str, Any]:
    log_path = run_dir / "driver" / "selection-log.jsonl"
    if not log_path.is_file():
        return {
            "selection_log": None,
            "frame_count": 0,
            "veto_count": 0,
            "veto_rate": None,
            "veto_rate_wilson95": None,
            "fallback_count": 0,
            "decision_counts": {},
            "veto_reason_counts": {},
            "dagger_argmax_geo_gap": {},
        }
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    veto_rows = [row for row in rows if row.get("dagger_argmax_vetoed")]
    gaps = [
        float(row["dagger_argmax_geo_gap"])
        for row in veto_rows
        if row.get("dagger_argmax_geo_gap") is not None
    ]
    return {
        "selection_log": str(log_path),
        "frame_count": len(rows),
        "veto_count": len(veto_rows),
        "veto_rate": _safe_rate(len(veto_rows), len(rows)),
        "veto_rate_wilson95": _wilson_interval(len(veto_rows), len(rows)),
        "fallback_count": sum(1 for row in rows if row.get("used_fallback_geometric")),
        "dagger_win_count": sum(1 for row in rows if row.get("decision_type") == "dagger_wins"),
        "spotlight_win_count": sum(1 for row in rows if row.get("decision_type") == "spotlight_wins"),
        "decision_counts": dict(Counter(str(row.get("decision_type", "<missing>")) for row in rows)),
        "veto_reason_counts": dict(Counter(str(row.get("veto_reason", "<missing>")) for row in veto_rows)),
        "dagger_argmax_geo_gap": _numeric_summary(gaps),
        "proxy_visibility": _proxy_visibility_stats(rows),
    }


def _combine_selection_stats(stats: list[dict[str, Any]]) -> dict[str, Any]:
    frame_count = sum(int(item["frame_count"]) for item in stats)
    veto_count = sum(int(item["veto_count"]) for item in stats)
    fallback_count = sum(int(item.get("fallback_count", 0)) for item in stats)
    dagger_win_count = sum(int(item.get("dagger_win_count", 0)) for item in stats)
    spotlight_win_count = sum(int(item.get("spotlight_win_count", 0)) for item in stats)
    decisions: Counter[str] = Counter()
    veto_reasons: Counter[str] = Counter()
    proxy = Counter()
    for item in stats:
        decisions.update(item.get("decision_counts", {}))
        veto_reasons.update(item.get("veto_reason_counts", {}))
        proxy.update(item.get("proxy_visibility", {}))
    return {
        "frame_count": frame_count,
        "veto_count": veto_count,
        "veto_rate": _safe_rate(veto_count, frame_count),
        "veto_rate_wilson95": _wilson_interval(veto_count, frame_count),
        "fallback_count": fallback_count,
        "dagger_win_count": dagger_win_count,
        "spotlight_win_count": spotlight_win_count,
        "decision_counts": dict(decisions),
        "veto_reason_counts": dict(veto_reasons),
        "proxy_visibility": _proxy_visibility_rates(dict(proxy)),
    }


def _proxy_visibility_stats(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts["frame_count"] += 1
        signal = row.get("alpasim_signal") if isinstance(row.get("alpasim_signal"), dict) else {}
        hazards = signal.get("structured_hazards", []) if isinstance(signal, dict) else []
        if isinstance(hazards, list) and hazards:
            counts["structured_hazard_frames"] += 1
            if any(_is_moving_hazard(hazard) for hazard in hazards if isinstance(hazard, dict)):
                counts["moving_hazard_frames"] += 1
        if float(signal.get("visibility_risk", 0.0) or 0.0) > 0.0:
            counts["visibility_risk_frames"] += 1
        if float(signal.get("dynamics_risk", 0.0) or 0.0) > 0.0:
            counts["dynamics_risk_frames"] += 1
        if bool(signal.get("oracle_actor_proxy_enabled", False)):
            counts["oracle_actor_proxy_enabled_frames"] += 1
        if bool(signal.get("oracle_actor_proxy_hit", False)):
            counts["oracle_actor_proxy_hit_frames"] += 1
            counts["oracle_actor_proxy_hazard_total"] += int(signal.get("oracle_actor_proxy_count", 0) or 0)
        if _top_candidate_looks_clear(row):
            counts["top_candidate_clear_frames"] += 1
    return _proxy_visibility_rates(dict(counts))


def _proxy_visibility_rates(counts: dict[str, int | float]) -> dict[str, int | float | None]:
    frame_count = int(counts.get("frame_count", 0) or 0)
    output: dict[str, int | float | None] = {"frame_count": frame_count}
    for key in (
        "structured_hazard_frames",
        "moving_hazard_frames",
        "visibility_risk_frames",
        "dynamics_risk_frames",
        "top_candidate_clear_frames",
        "oracle_actor_proxy_enabled_frames",
        "oracle_actor_proxy_hit_frames",
    ):
        count = int(counts.get(key, 0) or 0)
        output[key] = count
        output[f"{key.removesuffix('_frames')}_rate"] = _safe_rate(count, frame_count)
    output["oracle_actor_proxy_hazard_total"] = int(counts.get("oracle_actor_proxy_hazard_total", 0) or 0)
    output["oracle_actor_proxy_mean_hazards_per_hit"] = _safe_rate(
        int(counts.get("oracle_actor_proxy_hazard_total", 0) or 0),
        int(counts.get("oracle_actor_proxy_hit_frames", 0) or 0),
    )
    return output


def _is_moving_hazard(hazard: dict[str, Any]) -> bool:
    return abs(float(hazard.get("vx", 0.0) or 0.0)) > 1e-6 or abs(float(hazard.get("vy", 0.0) or 0.0)) > 1e-6


def _top_candidate_looks_clear(row: dict[str, Any]) -> bool:
    candidates = row.get("spotlight_top_candidates")
    if not isinstance(candidates, list) or not candidates:
        return False
    top = candidates[0]
    if not isinstance(top, dict):
        return False
    action_clearance = _as_float(top.get("action_clearance_m"))
    horizon_clearance = _as_float(top.get("horizon_clearance_m"))
    safety_penalty = _as_float(top.get("safety_penalty"))
    return math.isinf(action_clearance) and math.isinf(horizon_clearance) and safety_penalty <= 0.0


def _as_float(value: Any) -> float:
    if isinstance(value, str) and value.lower() == "inf":
        return math.inf
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _episode_metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    summary: dict[str, Any] = {"episode_count": len(rows)}
    for metric in BINARY_EPISODE_METRICS:
        hits = sum(1 for row in rows if _truthy_metric(row.get(metric)))
        summary[metric] = {
            "count": hits,
            "rate": hits / len(rows),
            "wilson95": _wilson_interval(hits, len(rows)),
        }
    for metric in (
        "raw_dist_traveled_m",
        "raw_progress",
        "raw_dist_to_gt_trajectory",
        "metric_span_s",
    ):
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        summary[metric] = _numeric_summary(values)
    return summary


def _rate_sample_summary(
    values: list[float],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    if not values:
        return {"scene_count": 0}
    return {
        "scene_count": len(values),
        "min": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "max": max(values),
        "bootstrap_mean95": _bootstrap_mean_interval(values, bootstrap_samples=bootstrap_samples, seed=seed),
        "count_gt_50pct": sum(1 for value in values if value > 0.5),
        "count_gt_80pct": sum(1 for value in values if value > 0.8),
        "count_gt_90pct": sum(1 for value in values if value > 0.9),
        "sign_test_all_gt_50pct_p": _one_sided_sign_test(sum(1 for value in values if value > 0.5), len(values)),
    }


def _numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(ordered),
        "min": ordered[0],
        "p10": _percentile(ordered, 0.1),
        "p90": _percentile(ordered, 0.9),
        "max": ordered[-1],
    }


def _bootstrap_mean_interval(values: list[float], *, bootstrap_samples: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    n = len(values)
    if n == 1:
        return [values[0], values[0]]
    means = []
    for _ in range(max(1, bootstrap_samples)):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.fmean(sample))
    means.sort()
    return [_percentile(means, 0.025), _percentile(means, 0.975)]


def _one_sided_sign_test(successes: int, n: int) -> float | None:
    if n <= 0:
        return None
    return sum(math.comb(n, k) for k in range(successes, n + 1)) / (2**n)


def _wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    phat = successes / total
    denom = 1.0 + z * z / total
    centre = phat + z * z / (2.0 * total)
    spread = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * total)) / total)
    return [(centre - spread) / denom, (centre + spread) / denom]


def _percentile(ordered: list[float], q: float) -> float:
    if not ordered:
        raise ValueError("cannot compute percentile of empty list")
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _safe_rate(count: int, total: int) -> float | None:
    return None if total <= 0 else count / total


def _truthy_metric(value: Any) -> bool:
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def _print_markdown(report: dict[str, Any]) -> None:
    print(f"Runs: {report['run_count']}, episodes: {report['episode_count']}")
    selection = report["selection"]
    print(
        "Selection: "
        f"{selection['veto_count']}/{selection['frame_count']} vetoes "
        f"({selection['veto_rate']:.4f} rate)"
        if selection["frame_count"]
        else "Selection: no selection logs"
    )
    scene = report["per_scene_veto"]
    if scene.get("scene_count", 0):
        print(
            "Per-scene veto: "
            f"n={scene['scene_count']}, mean={scene['mean']:.4f}, "
            f"min={scene['min']:.4f}, median={scene['median']:.4f}, max={scene['max']:.4f}, "
            f"sign-test p={scene['sign_test_all_gt_50pct_p']:.6f}"
        )
    proxy = selection.get("proxy_visibility", {})
    if proxy.get("frame_count", 0):
        print(
            "Proxy visibility: "
            f"structured_hazard={_format_rate(proxy.get('structured_hazard_rate'))}, "
            f"moving_hazard={_format_rate(proxy.get('moving_hazard_rate'))}, "
            f"clear_top_candidate={_format_rate(proxy.get('top_candidate_clear_rate'))}"
        )
    metrics = report["episode_metrics"]
    for metric in BINARY_EPISODE_METRICS:
        if metric in metrics:
            item = metrics[metric]
            lo, hi = item["wilson95"]
            print(f"{metric}: {item['count']}/{metrics['episode_count']} ({item['rate']:.4f}, 95% CI [{lo:.4f}, {hi:.4f}])")


def _format_rate(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


if __name__ == "__main__":
    main()
