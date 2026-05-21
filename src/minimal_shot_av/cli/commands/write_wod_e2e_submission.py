#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]

from minimal_shot_av.model.wod_submission import (
    WodSubmissionMetadata,
    load_frame_names,
    selected_predictions_from_jsonl,
    write_submission_tar,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Package WOD-E2E predictions as an official submission tar.gz.")
    parser.add_argument("--candidates", type=Path, required=True, help="JSONL predictions with trajectory_20wp_4hz.")
    parser.add_argument("--output", type=Path, required=True, help="Output .tar.gz path.")
    parser.add_argument("--frame-list", type=Path, help="Challenge JSON listing required test frame names.")
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Number of official proto shards to place in the tarball.",
    )
    parser.add_argument(
        "--score-field",
        help="Optional row field used to select the highest-scored candidate per frame.",
    )
    parser.add_argument("--candidate-name", help="Select this exact candidate_name for every frame.")
    parser.add_argument("--account-name", required=True)
    parser.add_argument("--unique-method-name", required=True)
    parser.add_argument("--authors", required=True, help="Comma-separated author names.")
    parser.add_argument("--affiliation", default="")
    parser.add_argument("--description", default="")
    parser.add_argument("--method-link", default="")
    parser.add_argument("--uses-public-model-pretraining", action="store_true")
    parser.add_argument("--public-model-names", default="", help="Comma-separated public model names.")
    parser.add_argument("--num-model-parameters", default="")
    args = parser.parse_args()

    required_frame_names = load_frame_names(args.frame_list) if args.frame_list else None
    predictions = selected_predictions_from_jsonl(
        args.candidates,
        required_frame_names=required_frame_names,
        score_field=args.score_field,
        candidate_name=args.candidate_name,
    )
    metadata = WodSubmissionMetadata(
        account_name=args.account_name,
        unique_method_name=args.unique_method_name,
        authors=_split_csv(args.authors),
        affiliation=args.affiliation,
        description=args.description,
        method_link=args.method_link,
        uses_public_model_pretraining=args.uses_public_model_pretraining,
        public_model_names=_split_csv(args.public_model_names),
        num_model_parameters=args.num_model_parameters,
    )
    output = write_submission_tar(predictions, args.output, metadata, num_shards=args.num_shards)
    print(json.dumps({"output": str(output), "predictions": len(predictions)}, indent=2))
    return 0


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
