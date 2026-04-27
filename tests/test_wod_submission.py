from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PROTOS = ROOT / ".wod-protos"
for path in (SRC, PROTOS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from minimal_shot_av.model.wod_submission import (
    WodSubmissionMetadata,
    WodTrajectoryPrediction,
    load_frame_names,
    read_submission_tar,
    read_submission_tar_shards,
    selected_predictions_from_jsonl,
    validate_submission_tar,
    write_submission_tar,
)


class WodSubmissionTests(unittest.TestCase):
    def test_load_frame_names_accepts_challenge_style_dicts(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "frames.json"
            path.write_text(
                json.dumps({"test_frames": [{"frame_name": "b"}, {"name": "a"}]}),
                encoding="utf-8",
            )

            self.assertEqual({"a", "b"}, load_frame_names(path))

    def test_selected_predictions_filters_required_frames_and_uses_score_field(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "candidates.jsonl"
            rows = [
                _row("keep", 0, 0.1, x=1.0),
                _row("keep", 1, 0.9, x=2.0),
                _row("skip", 0, 1.0, x=3.0),
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            predictions = selected_predictions_from_jsonl(
                path,
                required_frame_names={"keep"},
                score_field="ranker_score",
            )

            self.assertEqual(1, len(predictions))
            self.assertEqual("keep", predictions[0].frame_name)
            self.assertEqual((2.0, 0.0), predictions[0].trajectory[0])

    def test_selected_predictions_requires_all_frame_list_entries(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "candidates.jsonl"
            path.write_text(json.dumps(_row("present", 0, 0.0, x=1.0)) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing predictions"):
                selected_predictions_from_jsonl(path, required_frame_names={"present", "missing"})

    def test_selected_predictions_can_select_candidate_name_without_scores(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "candidates.jsonl"
            rows = [
                {**_row("frame", 0, 0.0, x=1.0), "candidate_name": "constant_velocity"},
                {**_row("frame", 1, 0.0, x=2.0), "candidate_name": "constant_acceleration"},
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            predictions = selected_predictions_from_jsonl(path, candidate_name="constant_acceleration")

            self.assertEqual(1, len(predictions))
            self.assertEqual((2.0, 0.0), predictions[0].trajectory[0])

    def test_write_submission_tar_round_trips_official_proto(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "submission.tar.gz"
            metadata = WodSubmissionMetadata(
                account_name="account",
                unique_method_name="ridge_non_text",
                authors=["A Dev"],
                affiliation="Independent",
                description="Fast non-text WOD-E2E baseline.",
                uses_public_model_pretraining=False,
                num_model_parameters="under 1M",
            )
            prediction = WodTrajectoryPrediction(
                frame_name="segment_frame",
                trajectory=[(float(index), float(-index)) for index in range(20)],
            )

            with patch("minimal_shot_av.model.wod_submission._import_submission_proto", return_value=FakeSubmissionPb2):
                write_submission_tar([prediction], output, metadata)
                parsed = read_submission_tar(output)

            self.assertEqual("account", parsed.account_name)
            self.assertEqual("ridge_non_text", parsed.unique_method_name)
            self.assertEqual(["A Dev"], list(parsed.authors))
            self.assertEqual(1, len(parsed.predictions))
            self.assertEqual("segment_frame", parsed.predictions[0].frame_name)
            self.assertEqual(20, len(parsed.predictions[0].trajectory.pos_x))
            self.assertEqual(19.0, parsed.predictions[0].trajectory.pos_x[-1])

    def test_validate_submission_tar_reports_required_frame_coverage(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "submission.tar.gz"
            metadata = WodSubmissionMetadata(
                account_name="account",
                unique_method_name="ridge_non_text",
                authors=["A Dev"],
            )
            prediction = WodTrajectoryPrediction(
                frame_name="present",
                trajectory=[(float(index), 0.0) for index in range(20)],
            )

            with patch("minimal_shot_av.model.wod_submission._import_submission_proto", return_value=FakeSubmissionPb2):
                write_submission_tar([prediction], output, metadata)
                report = validate_submission_tar(output, required_frame_names={"present", "missing"})

            self.assertFalse(report["valid"])
            self.assertEqual(1, report["prediction_count"])
            self.assertIn("missing 1 required frame predictions", report["errors"])

    def test_write_submission_tar_supports_official_style_shards(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "submission.tar.gz"
            metadata = WodSubmissionMetadata(
                account_name="account",
                unique_method_name="ridge_non_text",
                authors=["A Dev"],
            )
            predictions = [
                WodTrajectoryPrediction(
                    frame_name=f"frame_{index}",
                    trajectory=[(float(step), float(index)) for step in range(20)],
                )
                for index in range(3)
            ]

            with patch("minimal_shot_av.model.wod_submission._import_submission_proto", return_value=FakeSubmissionPb2):
                write_submission_tar(predictions, output, metadata, num_shards=2)
                shards = read_submission_tar_shards(output)
                report = validate_submission_tar(output)

            self.assertEqual(2, len(shards))
            self.assertEqual(["frame_0", "frame_1"], [prediction.frame_name for prediction in shards[0].predictions])
            self.assertEqual(["frame_2"], [prediction.frame_name for prediction in shards[1].predictions])
            self.assertTrue(report["valid"])
            self.assertEqual(2, report["shard_count"])
            self.assertEqual(3, report["prediction_count"])

    def test_write_submission_tar_keeps_requested_empty_shards(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "submission.tar.gz"
            metadata = WodSubmissionMetadata(
                account_name="account",
                unique_method_name="ridge_non_text",
                authors=["A Dev"],
            )
            prediction = WodTrajectoryPrediction(
                frame_name="frame_0",
                trajectory=[(float(step), 0.0) for step in range(20)],
            )

            with patch("minimal_shot_av.model.wod_submission._import_submission_proto", return_value=FakeSubmissionPb2):
                write_submission_tar([prediction], output, metadata, num_shards=2)
                shards = read_submission_tar_shards(output)
                report = validate_submission_tar(output)

            self.assertEqual(2, len(shards))
            self.assertEqual(["frame_0"], [item.frame_name for item in shards[0].predictions])
            self.assertEqual([], list(shards[1].predictions))
            self.assertTrue(report["valid"])
            self.assertEqual(2, report["shard_count"])
            self.assertEqual(1, report["prediction_count"])


def _row(frame_name: str, candidate_index: int, score: float, *, x: float) -> dict[str, object]:
    return {
        "frame_name": frame_name,
        "candidate_name": f"candidate_{candidate_index}",
        "candidate_index": candidate_index,
        "ranker_score": score,
        "trajectory_20wp_4hz": [[x, 0.0] for _ in range(20)],
    }


class FakeRepeatedPredictions(list):
    def add(self):
        prediction = FakeFrameTrajectoryPredictions()
        self.append(prediction)
        return prediction


class FakeTrajectoryPrediction:
    def __init__(self) -> None:
        self.pos_x: list[float] = []
        self.pos_y: list[float] = []


class FakeFrameTrajectoryPredictions:
    def __init__(self) -> None:
        self.frame_name = ""
        self.trajectory = FakeTrajectoryPrediction()


class FakeE2EDChallengeSubmission:
    E2ED_SUBMISSION = 1

    def __init__(self) -> None:
        self.predictions = FakeRepeatedPredictions()
        self.submission_type = 0
        self.account_name = ""
        self.unique_method_name = ""
        self.authors: list[str] = []
        self.affiliation = ""
        self.description = ""
        self.method_link = ""
        self.uses_public_model_pretraining = False
        self.public_model_names: list[str] = []
        self.num_model_parameters = ""

    def SerializeToString(self) -> bytes:
        payload = {
            "submission_type": self.submission_type,
            "account_name": self.account_name,
            "unique_method_name": self.unique_method_name,
            "authors": self.authors,
            "affiliation": self.affiliation,
            "description": self.description,
            "method_link": self.method_link,
            "uses_public_model_pretraining": self.uses_public_model_pretraining,
            "public_model_names": self.public_model_names,
            "num_model_parameters": self.num_model_parameters,
            "predictions": [
                {
                    "frame_name": prediction.frame_name,
                    "pos_x": prediction.trajectory.pos_x,
                    "pos_y": prediction.trajectory.pos_y,
                }
                for prediction in self.predictions
            ],
        }
        encoded = json.dumps(payload).encode("utf-8")
        return encoded

    def ParseFromString(self, data: bytes) -> None:
        payload = json.loads(data.decode("utf-8"))
        self.submission_type = int(payload["submission_type"])
        self.account_name = str(payload["account_name"])
        self.unique_method_name = str(payload["unique_method_name"])
        self.authors.extend(str(author) for author in payload["authors"])
        self.affiliation = str(payload["affiliation"])
        self.description = str(payload["description"])
        self.method_link = str(payload["method_link"])
        self.uses_public_model_pretraining = bool(payload["uses_public_model_pretraining"])
        self.public_model_names.extend(str(name) for name in payload["public_model_names"])
        self.num_model_parameters = str(payload["num_model_parameters"])
        for item in payload["predictions"]:
            prediction = self.predictions.add()
            prediction.frame_name = str(item["frame_name"])
            prediction.trajectory.pos_x.extend(float(value) for value in item["pos_x"])
            prediction.trajectory.pos_y.extend(float(value) for value in item["pos_y"])


class FakeSubmissionPb2:
    E2EDChallengeSubmission = FakeE2EDChallengeSubmission


if __name__ == "__main__":
    unittest.main()
