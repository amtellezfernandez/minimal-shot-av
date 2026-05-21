#!/usr/bin/env python3
"""Smoke-test the official WOD-E2E TensorFlow/protobuf parser locally."""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=Path("waymo_open_dataset_end_to_end_camera_v_1_0_0/val"),
    )
    parser.add_argument("--max-shards", type=int, default=1)
    args = parser.parse_args()

    shards = sorted(args.val_dir.glob("val_*.tfrecord-*"))[: args.max_shards]
    if not shards:
        raise FileNotFoundError(f"no validation shards found under {args.val_dir}")

    records_seen = 0
    for shard in shards:
        dataset = tf.data.TFRecordDataset([str(shard)], compression_type="")
        for record_in_shard, record in enumerate(dataset, start=1):
            payload = record.numpy()
            records_seen += 1
            frame = wod_e2ed_pb2.E2EDFrame()
            frame.ParseFromString(payload)

            valid_preferences = [
                trajectory
                for trajectory in frame.preference_trajectories
                if trajectory.HasField("preference_score")
                and trajectory.preference_score >= 0
            ]
            if not valid_preferences:
                continue

            first = valid_preferences[0]
            print(f"tensorflow={tf.__version__}")
            print(f"parsed={shard.name}")
            print(f"record_in_shard={record_in_shard}")
            print(f"records_seen={records_seen}")
            print(f"context_name={frame.frame.context.name}")
            print(
                "preference_count="
                f"{len(frame.preference_trajectories)} valid_count={len(valid_preferences)}"
            )
            print(f"first_score={first.preference_score:.3f}")
            print(f"first_preference_len=({len(first.pos_x)},{len(first.pos_y)})")
            print(
                "future_states_len="
                f"({len(frame.future_states.pos_x)},{len(frame.future_states.pos_y)})"
            )
            return 0

    raise RuntimeError(
        f"parsed {records_seen} records from {len(shards)} shard(s), "
        "but found no valid preference trajectories"
    )


if __name__ == "__main__":
    raise SystemExit(main())
