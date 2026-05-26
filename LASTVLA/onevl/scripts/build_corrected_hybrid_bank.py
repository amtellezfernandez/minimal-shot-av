#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


METADATA_KEYS = ("idx", "token", "stage", "log_name")


def merge_metadata(item: dict, greedy_item: dict, sample_item: dict) -> None:
    for key in METADATA_KEYS:
        greedy_value = greedy_item.get(key)
        sample_value = sample_item.get(key)
        if greedy_value is not None and sample_value is not None and greedy_value != sample_value:
            raise ValueError(f"metadata mismatch for {key}: greedy={greedy_value} sample={sample_value}")
        value = sample_value if sample_value is not None else greedy_value
        if value is not None:
            item[key] = value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--greedy-json", type=Path, required=True)
    parser.add_argument("--sample-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--sample-count",
        type=int,
        default=3,
        help="How many sampled candidates to keep after the greedy candidate.",
    )
    args = parser.parse_args()

    greedy = load_json(args.greedy_json)
    sample = load_json(args.sample_json)
    if len(greedy) != len(sample):
        raise ValueError(f"mismatched lengths: greedy={len(greedy)} sample={len(sample)}")

    merged = []
    for greedy_item, sample_item in zip(greedy, sample):
        item = dict(sample_item)
        merge_metadata(item, greedy_item, sample_item)
        candidates = [dict(greedy_item["candidates"][0])]
        candidates[0]["candidate_id"] = 0
        candidates[0]["source"] = "onevl_top1"
        for new_id, candidate in enumerate(sample_item["candidates"][: args.sample_count], start=1):
            candidate_copy = dict(candidate)
            candidate_copy["candidate_id"] = new_id
            candidate_copy["source"] = "onevl_sample"
            candidates.append(candidate_copy)
        item["candidate_mode"] = "greedy_plus_sample"
        item["num_candidates"] = len(candidates)
        item["candidates"] = candidates
        merged.append(item)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(merged))
    print(json.dumps({"scene_count": len(merged), "candidate_count": merged[0]["num_candidates"]}, indent=2))


if __name__ == "__main__":
    main()
