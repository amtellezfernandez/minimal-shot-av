#!/usr/bin/env python3
import argparse
import json
import math
import pickle
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover - reviewer environment issue
    raise SystemExit("PyYAML is required to read NAVSIM split configs") from exc


PROMPT_SUFFIX = (
    "Output the reasoning in <think></think> and the predicted trajectory in "
    "<answer></answer>. For the content in <answer></answer>, Only generate "
    "the predicted future waypoints in pure text format: [x_1, y_1, heading_1], "
    "[x_2, y_2, heading_2], ..., [x_8, y_8, heading_8]. Output exactly 8 "
    "waypoints in the format [x, y, heading], with numbers in decimal form "
    "and exactly 2 digits after the decimal point. Separate waypoints using "
    "comma. Do not include extra text, brackets, or invalid values. The output "
    "must strictly follow this structure."
)


def load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def as_list(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return value


def yaw_from_quaternion(rotation) -> float:
    q = as_list(rotation)
    if len(q) != 4:
        raise ValueError(f"expected quaternion with 4 values, got {q}")
    w, x, y, z = [float(v) for v in q]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def pose_from_original_frame(frame: dict) -> list[float]:
    translation = as_list(frame["ego2global_translation"])
    yaw = yaw_from_quaternion(frame["ego2global_rotation"])
    return [float(translation[0]), float(translation[1]), yaw]


def pose_from_synthetic_frame(frame: dict) -> list[float]:
    ego_status = frame["ego_status"]
    pose = as_list(ego_status["ego_pose"])
    return [float(pose[0]), float(pose[1]), float(pose[2])]


def localize_pose(origin: list[float], pose: list[float]) -> list[float]:
    dx = pose[0] - origin[0]
    dy = pose[1] - origin[1]
    c = math.cos(origin[2])
    s = math.sin(origin[2])
    return [
        c * dx + s * dy,
        -s * dx + c * dy,
        normalize_angle(pose[2] - origin[2]),
    ]


def round_pose(pose: list[float]) -> list[float]:
    return [round(float(pose[0]), 2), round(float(pose[1]), 2), round(float(pose[2]), 2)]


def format_waypoints(points: list[list[float]]) -> str:
    return ", ".join(f"[{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}]" for p in points)


def resolve_log_pickle(logs_dir: Path, log_name: str) -> Path:
    direct = logs_dir / f"{log_name}.pkl"
    if direct.exists():
        return direct
    nested = logs_dir / "test" / f"{log_name}.pkl"
    if nested.exists():
        return nested
    raise FileNotFoundError(f"missing log pickle for {log_name} under {logs_dir}")


def build_original_token_index(logs_dir: Path, log_names: list[str]) -> dict[str, tuple[str, list[dict], int]]:
    index = {}
    for log_name in log_names:
        frames = load_pickle(resolve_log_pickle(logs_dir, log_name))
        for frame_idx, frame in enumerate(frames):
            index[frame["token"]] = (log_name, frames, frame_idx)
    return index


def candidate_stage_tokens(scene_filter_cfg: dict, max_stage_one: int | None, max_stage_two: int | None):
    stage_one = list(scene_filter_cfg.get("tokens") or [])
    stage_two = list(scene_filter_cfg.get("reactive_synthetic_initial_tokens") or [])
    if max_stage_one is not None:
        stage_one = stage_one[:max_stage_one]
    if max_stage_two is not None:
        stage_two = stage_two[:max_stage_two]
    return stage_one, stage_two


def ordered_unique(values: list[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def mapped_stage_tokens(train_test_split_cfg: dict, max_groups: int | None) -> tuple[list[str], list[str]]:
    mappings = list(train_test_split_cfg.get("reactive_all_mapping") or [])
    if max_groups is not None:
        mappings = mappings[:max_groups]

    stage_one = []
    stage_two = []
    for orig_token, prev_token, two_stage_pairs in mappings:
        stage_one.extend([orig_token, prev_token])
        for now_token, previous_token in two_stage_pairs:
            stage_two.extend([now_token, previous_token])
    return ordered_unique(stage_one), ordered_unique(stage_two)


def front_image_from_original(frame: dict, sensor_root: Path) -> str:
    return str((sensor_root / frame["cams"]["CAM_F0"]["data_path"]).resolve())


def front_image_from_synthetic(frame: dict, sensor_root: Path) -> str:
    camera_dict = frame.get("camera_dict") or frame.get("cams")
    camera = camera_dict.get("CAM_F0") or camera_dict.get("cam_f0")
    if camera is None:
        raise KeyError(f"front camera missing from synthetic camera keys: {sorted(camera_dict)}")
    return str((sensor_root / camera["data_path"]).resolve())


def ego_state_from_original(frame: dict) -> tuple[list[float], list[float]]:
    dyn = [float(v) for v in as_list(frame["ego_dynamic_state"])]
    return dyn[:2], dyn[2:]


def ego_state_from_synthetic(frame: dict) -> tuple[list[float], list[float]]:
    ego_status = frame["ego_status"]
    return (
        [float(v) for v in as_list(ego_status["ego_velocity"])],
        [float(v) for v in as_list(ego_status["ego_acceleration"])],
    )


def build_prompt(command_text: str, velocity: list[float], acceleration: list[float], history: list[list[float]]) -> str:
    vel = [round(float(velocity[0]), 2), round(float(velocity[1]), 2)]
    acc = [round(float(acceleration[0]), 2), round(float(acceleration[1]), 2)]
    hist = [round_pose(p) for p in history]
    return (
        f"<image> is the front view. Command: {command_text}. "
        f"Velocity: {vel}. Acceleration: {acc}. Historical trajectory: {hist}. "
        f"{PROMPT_SUFFIX}"
    )


def make_item(
    *,
    token: str,
    stage: str,
    log_name: str,
    image_path: str,
    command_text: str,
    velocity: list[float],
    acceleration: list[float],
    history_poses: list[list[float]],
    future_poses: list[list[float]],
    idx: int,
    include_gt: bool,
) -> dict:
    gt = format_waypoints([round_pose(p) for p in future_poses[:8]]) if include_gt and future_poses else ""
    return {
        "messages": [
            {
                "role": "user",
                "content": build_prompt(command_text, velocity, acceleration, history_poses),
            }
        ],
        "images": [image_path],
        "solution": "",
        "GT": gt,
        "idx": idx,
        "token": token,
        "stage": stage,
        "log_name": log_name,
    }


def build_original_item(
    token: str,
    token_index: dict[str, tuple[str, list[dict], int]],
    sensor_root: Path,
    command_text: str,
    idx: int,
    include_gt: bool,
) -> dict | None:
    log_name, frames, frame_idx = token_index[token]
    if frame_idx < 3:
        return None
    history_frames = frames[frame_idx - 3 : frame_idx + 1]
    current = history_frames[-1]
    origin = pose_from_original_frame(current)
    history = [localize_pose(origin, pose_from_original_frame(frame)) for frame in history_frames[:-1]]
    future_frames = frames[frame_idx + 1 : frame_idx + 9]
    future = [localize_pose(origin, pose_from_original_frame(frame)) for frame in future_frames]
    velocity, acceleration = ego_state_from_original(current)
    return make_item(
        token=token,
        stage="first",
        log_name=log_name,
        image_path=front_image_from_original(current, sensor_root),
        command_text=command_text,
        velocity=velocity,
        acceleration=acceleration,
        history_poses=history,
        future_poses=future,
        idx=idx,
        include_gt=include_gt,
    )


def resolve_synthetic_scene_path(synthetic_scenes_dir: Path, token: str) -> Path:
    direct = synthetic_scenes_dir / f"{token}.pkl"
    if direct.exists():
        return direct
    matches = list(synthetic_scenes_dir.glob(f"**/{token}.pkl"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"missing synthetic scene pickle for token {token} under {synthetic_scenes_dir}")


def synthetic_scene_index_cache_path(synthetic_scenes_dir: Path) -> Path:
    return synthetic_scenes_dir.parent / "synthetic_scene_pickles_token_index.json"


def load_synthetic_scene_index_cache(synthetic_scenes_dir: Path) -> dict[str, Path]:
    cache_path = synthetic_scene_index_cache_path(synthetic_scenes_dir)
    if not cache_path.exists():
        return {}
    with cache_path.open() as f:
        data = json.load(f)
    index = {}
    for token, filename in data.items():
        path = synthetic_scenes_dir / filename
        if path.exists():
            index[str(token)] = path
    return index


def write_synthetic_scene_index_cache(synthetic_scenes_dir: Path, index: dict[str, Path]) -> None:
    cache_path = synthetic_scene_index_cache_path(synthetic_scenes_dir)
    serializable = {token: path.name for token, path in sorted(index.items())}
    cache_path.write_text(json.dumps(serializable, indent=2))


def build_synthetic_scene_index(synthetic_scenes_dir: Path, requested_tokens: list[str] | None = None) -> dict[str, Path]:
    index = load_synthetic_scene_index_cache(synthetic_scenes_dir)
    requested = {str(token) for token in requested_tokens or []}
    if requested and requested.issubset(index):
        return index
    for path in synthetic_scenes_dir.glob("*.pkl"):
        if path.stem in index:
            continue
        scene_data = load_pickle(path)
        metadata = scene_data.get("scene_metadata", {})
        for key in ("initial_token", "scene_token"):
            token = metadata.get(key)
            if token:
                index[str(token)] = path
        index[path.stem] = path
        if requested and requested.issubset(index):
            break
    write_synthetic_scene_index_cache(synthetic_scenes_dir, index)
    return index


def build_synthetic_item(
    token: str,
    synthetic_scenes_dir: Path,
    synthetic_sensor_root: Path,
    synthetic_scene_index: dict[str, Path],
    command_text: str,
    idx: int,
    include_gt: bool,
) -> dict | None:
    scene_path = synthetic_scene_index.get(token)
    if scene_path is None:
        scene_path = resolve_synthetic_scene_path(synthetic_scenes_dir, token)
    scene_data = load_pickle(scene_path)
    metadata = scene_data["scene_metadata"]
    frames = scene_data["frames"]
    history_count = int(metadata.get("num_history_frames", 4))
    current_idx = history_count - 1
    if len(frames) <= current_idx:
        return None
    current = frames[current_idx]
    origin = pose_from_synthetic_frame(current)
    history = [localize_pose(origin, pose_from_synthetic_frame(frame)) for frame in frames[:current_idx]]
    future_frames = frames[current_idx + 1 : current_idx + 9]
    future = [localize_pose(origin, pose_from_synthetic_frame(frame)) for frame in future_frames]
    velocity, acceleration = ego_state_from_synthetic(current)
    return make_item(
        token=token,
        stage="second",
        log_name=metadata["log_name"],
        image_path=front_image_from_synthetic(current, synthetic_sensor_root),
        command_text=command_text,
        velocity=velocity,
        acceleration=acceleration,
        history_poses=history,
        future_poses=future,
        idx=idx,
        include_gt=include_gt,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-filter-yaml", type=Path, required=True)
    parser.add_argument("--navsim-logs-dir", type=Path, required=True)
    parser.add_argument("--original-sensor-root", type=Path, required=True)
    parser.add_argument("--synthetic-scenes-dir", type=Path, default=None)
    parser.add_argument("--synthetic-sensor-root", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--include-stage", choices=["first", "second", "both"], default="both")
    parser.add_argument("--max-stage-one", type=int, default=None)
    parser.add_argument("--max-stage-two", type=int, default=None)
    parser.add_argument("--train-test-split-yaml", type=Path, default=None)
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--command-text", default="MOVE FORWARD")
    parser.add_argument("--include-gt", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.scene_filter_yaml)
    if args.train_test_split_yaml is not None and args.max_groups is not None:
        split_cfg = load_yaml(args.train_test_split_yaml)
        stage_one_tokens, stage_two_tokens = mapped_stage_tokens(split_cfg, args.max_groups)
    else:
        stage_one_tokens, stage_two_tokens = candidate_stage_tokens(cfg, args.max_stage_one, args.max_stage_two)

    items = []
    idx = 0
    if args.include_stage in {"first", "both"}:
        token_index = build_original_token_index(args.navsim_logs_dir, list(cfg.get("log_names") or []))
        missing = sorted(set(stage_one_tokens) - set(token_index))
        if missing:
            raise RuntimeError(f"missing {len(missing)} original tokens in logs: {missing[:10]}")
        for token in stage_one_tokens:
            item = build_original_item(
                token,
                token_index,
                args.original_sensor_root,
                args.command_text,
                idx,
                args.include_gt,
            )
            if item is not None:
                items.append(item)
                idx += 1

    if args.include_stage in {"second", "both"}:
        if args.synthetic_scenes_dir is None or args.synthetic_sensor_root is None:
            raise ValueError("--synthetic-scenes-dir and --synthetic-sensor-root are required for second-stage data")
        synthetic_scene_index = build_synthetic_scene_index(args.synthetic_scenes_dir, stage_two_tokens)
        for token in stage_two_tokens:
            item = build_synthetic_item(
                token,
                args.synthetic_scenes_dir,
                args.synthetic_sensor_root,
                synthetic_scene_index,
                args.command_text,
                idx,
                args.include_gt,
            )
            if item is not None:
                items.append(item)
                idx += 1

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(items, indent=2))
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "items": len(items),
                "stage_one": sum(1 for item in items if item["stage"] == "first"),
                "stage_two": sum(1 for item in items if item["stage"] == "second"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
