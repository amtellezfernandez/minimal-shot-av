#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sqlite3
from statistics import mean
from typing import Any

from minimal_shot_av.cli.commands.audit_nuplan_selected_token_realized_clearance import (
    DEFAULT_NEAR_MISS_THRESHOLD_M,
)
from minimal_shot_av.cli.commands.audit_nuplan_selected_token_realized_clearance import (
    compute_scene_log_replay_realized_fields,
)
from minimal_shot_av.cli.commands.audit_nuplan_selected_token_realized_clearance import (
    enrich_report_with_realized_clearance,
)
from minimal_shot_av.cli.commands.audit_nuplan_selected_token_realized_clearance import (
    markdown_report as replay_markdown,
)
from minimal_shot_av.cli.commands.fetch_nuplan_public_mini import PUBLIC_MINI_URL
from minimal_shot_av.cli.commands.fetch_nuplan_public_mini import _remotezip_import
from minimal_shot_av.cli.commands.fetch_nuplan_public_mini import fetch_public_mini_bundle
from minimal_shot_av.cli.commands.fetch_nuplan_public_mini import select_smallest_db_members
from minimal_shot_av.cli.commands.run_nuplan_maneuvertoken_rollout import _load_scenes
from minimal_shot_av.cli.commands.run_nuplan_maneuvertoken_rollout import markdown_report as rollout_markdown
from minimal_shot_av.cli.commands.run_nuplan_maneuvertoken_rollout import run_rollout
from minimal_shot_av.model.nuplan_maneuver_token_selector import load_selector


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BUNDLE_DIR = ROOT / "workspace" / "nuplan" / "public_replay_bundle"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "corl2027" / "nuplan_public_replay_study"
INTERACTION_RECOMMENDATION = "interaction"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch a larger public nuPlan mini bundle, run ManeuverToken rollout, and replay-audit it."
    )
    parser.add_argument("--url", default=PUBLIC_MINI_URL, help="Optional override for the public mini zip URL.")
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--db-count", type=int, default=20, help="How many public mini DBs to extract.")
    parser.add_argument(
        "--reuse-bundle",
        action="store_true",
        help="Use existing DB files under --bundle-dir instead of fetching or extracting the public archive.",
    )
    parser.add_argument("--scene-limit", type=int, default=250, help="Maximum scenes to analyze.")
    parser.add_argument("--selector-model", type=Path, help="Optional learned selector artifact.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--near-miss-threshold-m", type=float, default=DEFAULT_NEAR_MISS_THRESHOLD_M)
    parser.add_argument(
        "--sampling-mode",
        choices=("interaction", "simple"),
        default="interaction",
        help="Scene/DB sampling policy. `interaction` is the recommended CoRL path.",
    )
    parser.add_argument(
        "--max-scenes-per-db",
        type=int,
        help="Optional cap on selected scenes per DB. Recommended for interaction sampling.",
    )
    parser.add_argument(
        "--exclude-tiny-dbs-mb",
        type=float,
        help="Exclude DB members smaller than this size in MB before extraction.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = (
        existing_public_replay_bundle_manifest(args.bundle_dir)
        if bool(args.reuse_bundle)
        else prepare_public_replay_bundle(
            url=str(args.url),
            output_dir=args.bundle_dir,
            db_count=max(1, int(args.db_count)),
            sampling_mode=str(args.sampling_mode),
            exclude_tiny_dbs_mb=args.exclude_tiny_dbs_mb,
        )
    )
    (args.output_dir / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    study = build_public_replay_study(
        bundle_dir=args.bundle_dir,
        scene_limit=max(1, int(args.scene_limit)),
        selector_model=args.selector_model,
        near_miss_threshold_m=float(args.near_miss_threshold_m),
        sampling_mode=str(args.sampling_mode),
        max_scenes_per_db=args.max_scenes_per_db,
    )
    replay = study["replay_report"]
    rollout_report = study["rollout_report"]
    diagnostics = study["sampling_diagnostics"]
    (args.output_dir / "rollout.json").write_text(
        json.dumps(rollout_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "rollout.md").write_text(rollout_markdown(rollout_report) + "\n", encoding="utf-8")
    (args.output_dir / "replay.json").write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "replay.md").write_text(replay_markdown(replay) + "\n", encoding="utf-8")
    (args.output_dir / "sampling_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "sampling_diagnostics.md").write_text(
        markdown_sampling_diagnostics(diagnostics) + "\n",
        encoding="utf-8",
    )
    print(replay_markdown(replay))
    return 0


def prepare_public_replay_bundle(
    *,
    url: str,
    output_dir: Path,
    db_count: int,
    sampling_mode: str,
    exclude_tiny_dbs_mb: float | None,
) -> dict[str, Any]:
    if sampling_mode == "simple":
        return fetch_public_mini_bundle(url=url, output_dir=output_dir, db_count=db_count)
    minimum_mb = 10.0 if exclude_tiny_dbs_mb is None else max(0.0, float(exclude_tiny_dbs_mb))
    remotezip = _remotezip_import()
    output_dir.mkdir(parents=True, exist_ok=True)
    with remotezip.RemoteZip(url) as archive:
        selected = select_interaction_db_members(
            archive.infolist(),
            db_count=db_count,
            minimum_file_size_mb=minimum_mb,
        )
        extracted: list[dict[str, Any]] = []
        for member in selected:
            archive.extract(member["member"], path=output_dir)
            extracted_path = output_dir / member["member"]
            extracted.append(
                {
                    **member,
                    "selection_mode": "interaction",
                    "extracted_path": str(extracted_path),
                    "exists": extracted_path.is_file(),
                }
            )
    return {
        "schema": "nuplan_public_mini_bundle_v2",
        "selection_mode": "interaction",
        "url": url,
        "output_dir": str(output_dir),
        "db_count": len(extracted),
        "excluded_tiny_db_threshold_mb": minimum_mb,
        "members": extracted,
    }


def existing_public_replay_bundle_manifest(output_dir: Path) -> dict[str, Any]:
    db_files = sorted(output_dir.rglob("*.db"))
    return {
        "schema": "nuplan_public_mini_bundle_v2",
        "selection_mode": "existing_bundle",
        "url": None,
        "output_dir": str(output_dir),
        "db_count": len(db_files),
        "members": [
            {
                "member": str(path.relative_to(output_dir)),
                "file_size": path.stat().st_size,
                "selection_reason": "reuse_existing_bundle",
                "extracted_path": str(path),
                "exists": path.is_file(),
            }
            for path in db_files
        ],
    }


def select_interaction_db_members(
    file_infos: list[Any],
    *,
    db_count: int,
    minimum_file_size_mb: float,
) -> list[dict[str, Any]]:
    minimum_bytes = int(round(minimum_file_size_mb * 1024 * 1024))
    members = [
        {
            "member": str(info.filename),
            "file_size": int(info.file_size),
            "compress_size": int(info.compress_size),
        }
        for info in file_infos
        if str(info.filename).endswith(".db") and int(info.file_size) >= minimum_bytes
    ]
    if not members:
        members = select_smallest_db_members(file_infos, db_count=db_count)
        for row in members:
            row["selection_reason"] = "fallback_no_db_above_size_threshold"
        return members
    members.sort(key=lambda item: (-item["file_size"], item["compress_size"], item["member"]))
    selected = members[:db_count]
    for row in selected:
        row["selection_reason"] = "interaction_mode_exclude_tiny_and_prefer_larger_dbs"
    return selected


def build_public_replay_study(
    *,
    bundle_dir: Path,
    scene_limit: int,
    selector_model: Path | None,
    near_miss_threshold_m: float,
    sampling_mode: str,
    max_scenes_per_db: int | None,
) -> dict[str, object]:
    valid_db_files, invalid_db_files = validate_bundle_db_files(bundle_dir)
    load_args = argparse.Namespace(
        input=None,
        nuplan_data_root=None,
        nuplan_db_file=[str(path) for path in valid_db_files] if valid_db_files else [str(bundle_dir)],
        scenario_type=[],
        scenario_token=[],
        log_name=[],
        map_name=[],
        limit=None if sampling_mode == "interaction" else scene_limit,
    )
    candidate_scenes = _load_scenes(load_args)
    sampled_scenes, sampling_rows = sample_public_replay_scenes(
        candidate_scenes,
        scene_limit=scene_limit,
        sampling_mode=sampling_mode,
        max_scenes_per_db=max_scenes_per_db,
    )
    selector = None if selector_model is None else load_selector(selector_model)
    rollout_report = run_rollout(sampled_scenes, selector=selector)
    replay_report = enrich_report_with_realized_clearance(
        rollout_report,
        clearance_provider=compute_scene_log_replay_realized_fields,
        near_miss_threshold_m=float(near_miss_threshold_m),
    )
    diagnostics = build_sampling_diagnostics(
        candidate_scenes=candidate_scenes,
        sampled_scenes=sampled_scenes,
        rollout_report=rollout_report,
        sampling_rows=sampling_rows,
        sampling_mode=sampling_mode,
        max_scenes_per_db=max_scenes_per_db,
        invalid_db_files=invalid_db_files,
    )
    return {
        "rollout_report": rollout_report,
        "replay_report": replay_report,
        "sampling_diagnostics": diagnostics,
    }


def validate_bundle_db_files(bundle_dir: Path) -> tuple[list[Path], list[dict[str, str]]]:
    db_files = sorted(bundle_dir.rglob("*.db"))
    valid: list[Path] = []
    invalid: list[dict[str, str]] = []
    for path in db_files:
        try:
            with sqlite3.connect(str(path)) as connection:
                row = connection.execute("PRAGMA quick_check").fetchone()
            if row is None or str(row[0]).lower() != "ok":
                invalid.append({"db_file": str(path), "reason": str(row[0]) if row else "quick_check_failed"})
                continue
            valid.append(path)
        except sqlite3.DatabaseError as exc:
            invalid.append({"db_file": str(path), "reason": str(exc)})
    return valid, invalid


def sample_public_replay_scenes(
    candidate_scenes: list[dict[str, Any]],
    *,
    scene_limit: int,
    sampling_mode: str,
    max_scenes_per_db: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if sampling_mode == "simple":
        selected = candidate_scenes[:scene_limit]
        rows = [
            {
                "scene_id": scene.get("scene_id"),
                "source_db_file": scene.get("source_db_file"),
                "score": None,
                "sampling_reason": "simple_first_loaded_scene_order",
                "features": {},
            }
            for scene in selected
        ]
        return selected, rows
    per_db_cap = max_scenes_per_db if max_scenes_per_db is not None else max(1, math.ceil(scene_limit / 4))
    scored = [score_interaction_scene(scene) for scene in candidate_scenes]
    scored.sort(
        key=lambda row: (
            float(row["score"]),
            float(row["features"]["closing_risk_score"]),
            float(row["features"]["dynamic_actor_density"]),
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    db_counts: Counter[str] = Counter()
    for row in scored:
        db_key = str(row["source_db_file"])
        if db_counts[db_key] >= per_db_cap:
            continue
        selected.append(row["scene"])
        rows.append(
            {
                "scene_id": row["scene"]["scene_id"],
                "source_db_file": db_key,
                "score": row["score"],
                "sampling_reason": ", ".join(row["top_reasons"]),
                "features": row["features"],
            }
        )
        db_counts[db_key] += 1
        if len(selected) >= scene_limit:
            break
    return selected, rows


def score_interaction_scene(scene: dict[str, Any]) -> dict[str, Any]:
    actors = list(scene.get("actors", []))
    ego = dict(scene.get("ego_state", {}))
    route = dict(scene.get("route", {}))
    expert = list(scene.get("expert_trajectory", []))
    visible_actors = [actor for actor in actors if bool(actor.get("visible", True))]
    dynamic_actors = [
        actor
        for actor in visible_actors
        if math.hypot(float(actor.get("vx_mps", 0.0)), float(actor.get("vy_mps", 0.0))) >= 0.5
    ]
    min_distance = min(
        (
            math.hypot(float(actor.get("x_m", 0.0)), float(actor.get("y_m", 0.0)))
            for actor in visible_actors
        ),
        default=50.0,
    )
    dynamic_actor_density = sum(
        1
        for actor in dynamic_actors
        if math.hypot(float(actor.get("x_m", 0.0)), float(actor.get("y_m", 0.0))) <= 20.0
    )
    ttc_s, closing_risk_score = _scene_ttc_and_closing_risk(visible_actors)
    ego_speed_mps = float(ego.get("speed_mps", 0.0))
    intersection_context = _intersection_context_score(route=route, actors=visible_actors)
    braking_score, steering_score = _expert_control_scores(
        ego_speed_mps=ego_speed_mps,
        expert_trajectory=expert,
    )
    feature_scores = {
        "dynamic_actor_density": min(1.0, dynamic_actor_density / 6.0),
        "minimum_actor_distance_score": min(1.0, max(0.0, (15.0 - min_distance) / 15.0)),
        "closing_risk_score": closing_risk_score,
        "ego_speed_score": min(1.0, ego_speed_mps / 15.0),
        "intersection_context_score": intersection_context,
        "expert_braking_score": braking_score,
        "expert_steering_score": steering_score,
    }
    weights = {
        "dynamic_actor_density": 1.5,
        "minimum_actor_distance_score": 2.5,
        "closing_risk_score": 3.0,
        "ego_speed_score": 1.0,
        "intersection_context_score": 1.25,
        "expert_braking_score": 1.5,
        "expert_steering_score": 1.25,
    }
    contributions = {key: feature_scores[key] * weights[key] for key in feature_scores}
    top_reasons = [
        _reason_label(key, feature_scores[key], contributions[key], min_distance=min_distance, ttc_s=ttc_s)
        for key, _value in sorted(contributions.items(), key=lambda item: item[1], reverse=True)[:4]
    ]
    features = {
        "visible_actor_count": len(visible_actors),
        "dynamic_actor_count": len(dynamic_actors),
        "dynamic_actor_density": dynamic_actor_density,
        "minimum_actor_distance_m": round(min_distance, 6),
        "logged_ttc_s": None if not math.isfinite(ttc_s) else round(ttc_s, 6),
        "closing_risk_score": round(closing_risk_score, 6),
        "ego_speed_mps": round(ego_speed_mps, 6),
        "intersection_context_score": round(intersection_context, 6),
        "expert_braking_score": round(braking_score, 6),
        "expert_steering_score": round(steering_score, 6),
        "route_command": str(route.get("command", "straight")),
    }
    return {
        "scene": scene,
        "scene_id": scene.get("scene_id"),
        "source_db_file": scene.get("source_db_file"),
        "score": round(sum(contributions.values()), 6),
        "top_reasons": top_reasons,
        "features": features,
    }


def _scene_ttc_and_closing_risk(actors: list[dict[str, Any]]) -> tuple[float, float]:
    min_ttc_s = math.inf
    max_risk = 0.0
    for actor in actors:
        x_m = float(actor.get("x_m", 0.0))
        y_m = float(actor.get("y_m", 0.0))
        vx_mps = float(actor.get("vx_mps", 0.0))
        vy_mps = float(actor.get("vy_mps", 0.0))
        distance = math.hypot(x_m, y_m)
        if distance <= 1.0e-6:
            continue
        closing_speed = max(0.0, -(x_m * vx_mps + y_m * vy_mps) / distance)
        if closing_speed <= 1.0e-6:
            continue
        ttc_s = distance / closing_speed
        min_ttc_s = min(min_ttc_s, ttc_s)
        max_risk = max(max_risk, min(1.0, max(0.0, (6.0 - ttc_s) / 6.0)))
    return min_ttc_s, round(max_risk, 6)


def _intersection_context_score(*, route: dict[str, Any], actors: list[dict[str, Any]]) -> float:
    route_command = str(route.get("command", "straight")).lower()
    heading_error = abs(float(route.get("heading_error_rad", 0.0)))
    crossing = sum(
        1
        for actor in actors
        if abs(float(actor.get("vy_mps", 0.0))) > 1.0 and abs(float(actor.get("x_m", 0.0))) <= 15.0
    )
    score = 0.0
    if route_command in {"left", "right"}:
        score += 0.5
    score += min(0.3, heading_error / 1.0)
    score += min(0.2, crossing / 4.0)
    return round(min(1.0, score), 6)


def _expert_control_scores(*, ego_speed_mps: float, expert_trajectory: list[dict[str, Any]]) -> tuple[float, float]:
    if not expert_trajectory:
        return 0.0, 0.0
    dt_s = 0.5
    local_speeds = []
    heading_deltas = []
    previous = {"x_m": 0.0, "y_m": 0.0, "heading_rad": 0.0}
    for row in expert_trajectory:
        distance = math.hypot(
            float(row.get("x_m", 0.0)) - float(previous["x_m"]),
            float(row.get("y_m", 0.0)) - float(previous["y_m"]),
        )
        local_speeds.append(distance / dt_s)
        heading_deltas.append(abs(float(row.get("heading_rad", 0.0)) - float(previous["heading_rad"])))
        previous = row
    minimum_future_speed = min(local_speeds, default=ego_speed_mps)
    braking_score = min(1.0, max(0.0, ego_speed_mps - minimum_future_speed) / 4.0)
    steering_score = min(1.0, max(heading_deltas, default=0.0) / 0.4)
    return round(braking_score, 6), round(steering_score, 6)


def _reason_label(
    key: str,
    normalized_value: float,
    weighted_value: float,
    *,
    min_distance: float,
    ttc_s: float,
) -> str:
    if key == "minimum_actor_distance_score":
        return f"close_actor distance={min_distance:.2f}m weight={weighted_value:.2f}"
    if key == "closing_risk_score":
        ttc_text = "inf" if not math.isfinite(ttc_s) else f"{ttc_s:.2f}s"
        return f"closing_risk ttc={ttc_text} weight={weighted_value:.2f}"
    return f"{key} score={normalized_value:.2f} weight={weighted_value:.2f}"


def build_sampling_diagnostics(
    *,
    candidate_scenes: list[dict[str, Any]],
    sampled_scenes: list[dict[str, Any]],
    rollout_report: dict[str, Any],
    sampling_rows: list[dict[str, Any]],
    sampling_mode: str,
    max_scenes_per_db: int | None,
    invalid_db_files: list[dict[str, str]],
) -> dict[str, Any]:
    token_histogram = dict(rollout_report.get("selected_token_histogram", {}))
    actor_counts = [len(scene.get("actors", [])) for scene in sampled_scenes]
    distances = [
        float(row["features"]["minimum_actor_distance_m"])
        for row in sampling_rows
        if row["features"].get("minimum_actor_distance_m") is not None
    ]
    ttcs = [
        float(row["features"]["logged_ttc_s"])
        for row in sampling_rows
        if row["features"].get("logged_ttc_s") is not None
    ]
    db_distribution = []
    candidate_by_db = Counter(str(scene.get("source_db_file")) for scene in candidate_scenes)
    selected_by_db = Counter(str(scene.get("source_db_file")) for scene in sampled_scenes)
    for db_file in sorted(candidate_by_db):
        db_distribution.append(
            {
                "db_file": db_file,
                "candidate_scene_count": candidate_by_db[db_file],
                "selected_scene_count": selected_by_db.get(db_file, 0),
            }
        )
    return {
        "schema": "nuplan_public_replay_sampling_diagnostics_v1",
        "sampling_mode": sampling_mode,
        "recommended_corl_path": INTERACTION_RECOMMENDATION,
        "candidate_scene_count": len(candidate_scenes),
        "selected_scene_count": len(sampled_scenes),
        "max_scenes_per_db": max_scenes_per_db,
        "invalid_db_files": invalid_db_files,
        "db_distribution": db_distribution,
        "selected_token_histogram": token_histogram,
        "selected_token_entropy": _token_entropy(token_histogram),
        "actor_count_stats": _numeric_stats(actor_counts),
        "distance_stats_m": _numeric_stats(distances),
        "ttc_stats_s": _numeric_stats(ttcs),
        "sampling_rows": sampling_rows,
    }


def _numeric_stats(values: list[float | int]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "min": 0.0, "mean": 0.0, "max": 0.0}
    numeric = [float(value) for value in values]
    return {
        "count": float(len(numeric)),
        "min": round(min(numeric), 6),
        "mean": round(mean(numeric), 6),
        "max": round(max(numeric), 6),
    }


def _token_entropy(histogram: dict[str, int]) -> float:
    total = sum(int(value) for value in histogram.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in histogram.values():
        probability = float(count) / float(total)
        entropy -= probability * math.log(probability, 2)
    return round(entropy, 6)


def markdown_sampling_diagnostics(diagnostics: dict[str, Any]) -> str:
    lines = [
        "# nuPlan Public Replay Sampling Diagnostics",
        "",
        f"- Sampling mode: `{diagnostics['sampling_mode']}`",
        f"- Recommended CoRL path: `{diagnostics['recommended_corl_path']}`",
        f"- Candidate scenes: `{diagnostics['candidate_scene_count']}`",
        f"- Selected scenes: `{diagnostics['selected_scene_count']}`",
        f"- Max scenes per DB: `{diagnostics['max_scenes_per_db']}`",
        f"- Invalid DB files skipped: `{len(diagnostics['invalid_db_files'])}`",
        f"- Selected-token entropy: `{diagnostics['selected_token_entropy']:.3f}`",
        f"- Selected-token histogram: `{json.dumps(diagnostics['selected_token_histogram'], sort_keys=True)}`",
        "",
        "## DB Distribution",
        "",
        "| DB file | Candidate scenes | Selected scenes |",
        "|---|---:|---:|",
    ]
    for row in diagnostics["db_distribution"]:
        lines.append(
            f"| {Path(str(row['db_file'])).name} | {row['candidate_scene_count']} | {row['selected_scene_count']} |"
        )
    if diagnostics["invalid_db_files"]:
        lines.extend(
            [
                "",
                "## Invalid DB Files",
                "",
                "| DB file | Reason |",
                "|---|---|",
            ]
        )
        for row in diagnostics["invalid_db_files"]:
            lines.append(f"| {Path(str(row['db_file'])).name} | {row['reason']} |")
    lines.extend(
        [
            "",
            "## Scene Stats",
            "",
            f"- Actor count stats: `{json.dumps(diagnostics['actor_count_stats'], sort_keys=True)}`",
            f"- Distance stats (m): `{json.dumps(diagnostics['distance_stats_m'], sort_keys=True)}`",
            f"- TTC stats (s): `{json.dumps(diagnostics['ttc_stats_s'], sort_keys=True)}`",
            "",
            "## Sampling Reasons",
            "",
            "| Scene | DB | Score | Reason |",
            "|---|---|---:|---|",
        ]
    )
    for row in diagnostics["sampling_rows"]:
        score = "n/a" if row["score"] is None else f"{float(row['score']):.3f}"
        lines.append(
            f"| {row['scene_id']} | {Path(str(row['source_db_file'])).name} | {score} | {row['sampling_reason']} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
