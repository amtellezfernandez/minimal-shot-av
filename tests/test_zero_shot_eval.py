from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.model.rfs_metric import RfsReference
from minimal_shot_av.model.wod_e2e import WodE2EPreferenceFrame
from minimal_shot_av.model.wod_ranker import WodPreferenceRanker
from minimal_shot_av.model.zero_shot_eval import (
    candidate_record_from_json,
    candidate_geometry_diagnostics,
    evaluate_zero_shot_candidate_groups,
    evaluate_zero_shot_candidates,
    load_candidate_record_groups,
    load_candidate_records,
    local_rfs_score,
)


def sample_frame(frame_name: str = "frame-a") -> WodE2EPreferenceFrame:
    trajectory = [(float(index), 0.0) for index in range(1, 21)]
    return WodE2EPreferenceFrame(
        frame_name=frame_name,
        past_trajectory=[(-1.0, 0.0), (0.0, 0.0)],
        future_trajectory=trajectory,
        intent=1,
        init_speed_mps=4.0,
        references=[RfsReference("human", trajectory, 9.0)],
    )


class ZeroShotEvalTests(unittest.TestCase):
    def test_candidate_record_accepts_20_waypoint_trajectory(self) -> None:
        payload = {
            "frame_name": "frame-a",
            "trajectory_20wp_4hz": [[float(index), 0.0] for index in range(1, 21)],
            "source": "numeric_candidate",
            "latency_ms": 42.5,
        }

        record = candidate_record_from_json(payload)

        self.assertEqual(record.frame_name, "frame-a")
        self.assertEqual(len(record.trajectory), 20)
        self.assertEqual(record.source, "numeric_candidate")
        self.assertEqual(record.latency_ms, 42.5)

    def test_candidate_record_accepts_64_waypoint_structured_field(self) -> None:
        record = candidate_record_from_json(
            {
                "frame_name": "frame-a",
                "trajectory_64wp_10hz": [[index / 10.0, 0.0] for index in range(1, 65)],
            }
        )

        self.assertEqual(len(record.trajectory), 20)
        self.assertAlmostEqual(record.trajectory[-1][0], 5.0)

    def test_candidate_record_rejects_free_form_response(self) -> None:
        response = json.dumps({"trajectory_64wp_10hz": [[index / 10.0, 0.0] for index in range(1, 65)]})

        with self.assertRaisesRegex(ValueError, "trajectory_20wp_4hz or trajectory_64wp_10hz"):
            candidate_record_from_json({"frame_name": "frame-a", "response": response})

    def test_evaluates_candidates_with_rfs_and_missing_accounting(self) -> None:
        frame = sample_frame("frame-a")
        record = candidate_record_from_json(
            {
                "frame_name": "frame-a",
                "trajectory_20wp_4hz": [[float(index), 0.0] for index in range(1, 21)],
                "latency_ms": 12.0,
            }
        )

        report = evaluate_zero_shot_candidates([frame, sample_frame("missing")], {"frame-a": record})

        self.assertEqual(report.evaluated_frames, 1)
        self.assertEqual(report.missing_candidate_frames, 0)
        self.assertEqual(report.mean_rfs, 9.0)
        self.assertEqual(report.mean_latency_ms, 12.0)
        self.assertEqual(report.score_backend, "local_rfs_metric")
        self.assertEqual(report.scanned_preference_frames, 1)
        self.assertEqual(report.evaluations[0].best_reference_label, "human")
        self.assertEqual(report.evaluations[0].best_reference_error_5s_m, 0.0)

    def test_evaluation_can_use_injected_official_scorer(self) -> None:
        frame = sample_frame("frame-a")
        record = candidate_record_from_json(
            {
                "frame_name": "frame-a",
                "trajectory_20wp_4hz": [[float(index), 0.0] for index in range(1, 21)],
            }
        )

        report = evaluate_zero_shot_candidates(
            [frame],
            {"frame-a": record},
            scorer=lambda candidate, preference_frame: 7.25,
            score_backend="official_waymo_rfs",
        )

        self.assertEqual(report.mean_rfs, 7.25)
        self.assertEqual(report.score_backend, "official_waymo_rfs")

    def test_candidate_geometry_diagnostics_explain_reference_miss(self) -> None:
        frame = WodE2EPreferenceFrame(
            frame_name="frame-a",
            past_trajectory=[(-1.0, 0.0), (0.0, 0.0)],
            future_trajectory=[(float(index), 0.0) for index in range(1, 21)],
            intent=1,
            init_speed_mps=0.0,
            references=[
                RfsReference("best", [(0.0, 0.0)] * 20, 10.0),
                RfsReference("closest_5s", [(0.0, 0.0)] * 19 + [(20.0, 0.0)], 6.0),
            ],
        )
        record = candidate_record_from_json(
            {
                "frame_name": "frame-a",
                "trajectory_20wp_4hz": [[float(index), 0.0] for index in range(1, 21)],
            }
        )

        diagnostics = candidate_geometry_diagnostics(record, frame)

        self.assertEqual(diagnostics["best_reference_label"], "best")
        self.assertEqual(diagnostics["best_reference_score"], 10.0)
        self.assertEqual(diagnostics["best_reference_error_3s_m"], 12.0)
        self.assertEqual(diagnostics["best_reference_error_5s_m"], 20.0)
        self.assertEqual(diagnostics["closest_5s_reference_label"], "closest_5s")
        self.assertEqual(diagnostics["closest_5s_error_m"], 0.0)

    def test_sparse_candidates_do_not_stop_on_early_missing_frames(self) -> None:
        late_frame = sample_frame("late")
        record = candidate_record_from_json(
            {
                "frame_name": "late",
                "trajectory_20wp_4hz": [[float(index), 0.0] for index in range(1, 21)],
            }
        )

        report = evaluate_zero_shot_candidates(
            [sample_frame("missing-a"), sample_frame("missing-b"), late_frame],
            {"late": record},
        )

        self.assertEqual(report.evaluated_frames, 1)
        self.assertEqual(report.missing_candidate_frames, 2)
        self.assertEqual(report.mean_rfs, local_rfs_score(record, late_frame))

    def test_load_candidate_records_keeps_invalid_lines_out_of_scoring(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidates.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "frame_name": "frame-a",
                                "trajectory_20wp_4hz": [[float(index), 0.0] for index in range(1, 21)],
                            }
                        ),
                        json.dumps({"frame_name": "bad"}),
                    ]
                ),
                encoding="utf-8",
            )

            candidates, invalid = load_candidate_records(path)

        self.assertIn("frame-a", candidates)
        self.assertEqual(len(invalid), 1)

    def test_load_candidate_record_groups_preserves_multiple_samples_per_frame(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidates.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "frame_name": "frame-a",
                                "source": "sample-a",
                                "trajectory_20wp_4hz": [[float(index), 1.0] for index in range(1, 21)],
                            }
                        ),
                        json.dumps(
                            {
                                "frame_name": "frame-a",
                                "source": "sample-b",
                                "trajectory_20wp_4hz": [[float(index), 0.0] for index in range(1, 21)],
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            groups, invalid = load_candidate_record_groups(path)

        self.assertEqual([], invalid)
        self.assertEqual(2, len(groups["frame-a"]))
        self.assertEqual([0, 1], [record.candidate_index for record in groups["frame-a"]])

    def test_multi_candidate_evaluation_reports_best_candidate_headroom(self) -> None:
        frame = sample_frame("frame-a")
        weak = candidate_record_from_json(
            {
                "frame_name": "frame-a",
                "source": "weak",
                "candidate_index": 0,
                "trajectory_20wp_4hz": [[float(index), 3.0] for index in range(1, 21)],
            }
        )
        strong = candidate_record_from_json(
            {
                "frame_name": "frame-a",
                "source": "strong",
                "candidate_index": 1,
                "trajectory_20wp_4hz": [[float(index), 0.0] for index in range(1, 21)],
            }
        )

        report = evaluate_zero_shot_candidate_groups([frame], {"frame-a": [weak, strong]})

        self.assertEqual(1, report.evaluated_frames)
        self.assertEqual(2, report.evaluations[0].candidate_count)
        self.assertEqual("weak", report.evaluations[0].source)
        self.assertEqual("strong", report.evaluations[0].best_candidate_source)
        self.assertEqual(1, report.evaluations[0].best_candidate_index)
        self.assertGreater(report.mean_selection_regret or 0.0, 0.0)

    def test_multi_candidate_oracle_selection_is_explicit(self) -> None:
        frame = sample_frame("frame-a")
        weak = candidate_record_from_json(
            {
                "frame_name": "frame-a",
                "source": "weak",
                "candidate_index": 0,
                "trajectory_20wp_4hz": [[float(index), 3.0] for index in range(1, 21)],
            }
        )
        strong = candidate_record_from_json(
            {
                "frame_name": "frame-a",
                "source": "strong",
                "candidate_index": 1,
                "trajectory_20wp_4hz": [[float(index), 0.0] for index in range(1, 21)],
            }
        )

        report = evaluate_zero_shot_candidate_groups(
            [frame],
            {"frame-a": [weak, strong]},
            selection_mode="best_validation",
        )

        self.assertEqual("strong", report.evaluations[0].source)
        self.assertEqual(0.0, report.evaluations[0].selection_regret)
        self.assertEqual(report.mean_rfs, report.mean_best_candidate_rfs)

    def test_ranker_candidate_selection_uses_saved_ranker(self) -> None:
        frame = sample_frame("frame-a")
        short = candidate_record_from_json(
            {
                "frame_name": "frame-a",
                "source": "short",
                "candidate_name": "short",
                "candidate_index": 0,
                "trajectory_20wp_4hz": [[float(index) * 0.25, 0.0] for index in range(1, 21)],
            }
        )
        matched = candidate_record_from_json(
            {
                "frame_name": "frame-a",
                "source": "matched",
                "candidate_name": "matched",
                "candidate_index": 1,
                "trajectory_20wp_4hz": [[float(index), 0.0] for index in range(1, 21)],
            }
        )
        ranker = WodPreferenceRanker(
            numeric_features=["x_5s"],
            candidate_names=["matched"],
            candidate_families=["matched"],
            feature_mean=[0.0, 0.0, 0.0],
            feature_scale=[1.0, 1.0, 1.0],
            weights=[-1.0, 0.0, 0.0],
            bias=0.0,
        )

        report = evaluate_zero_shot_candidate_groups(
            [frame],
            {"frame-a": [short, matched]},
            selection_mode="ranker",
            ranker=ranker,
        )

        self.assertEqual("short", report.evaluations[0].source)
        self.assertEqual("matched", report.evaluations[0].best_candidate_source)
        self.assertGreater(report.evaluations[0].selection_regret or 0.0, 0.0)
        self.assertEqual(["short"], report.ranker_unseen_candidate_names)
        self.assertEqual(["short"], report.ranker_unseen_candidate_families)

    def test_ranker_selection_requires_ranker_model(self) -> None:
        frame = sample_frame("frame-a")
        record = candidate_record_from_json(
            {
                "frame_name": "frame-a",
                "trajectory_20wp_4hz": [[float(index), 0.0] for index in range(1, 21)],
            }
        )

        with self.assertRaisesRegex(ValueError, "ranker selection requires"):
            evaluate_zero_shot_candidate_groups([frame], {"frame-a": [record]}, selection_mode="ranker")

    def test_cli_diagnostics_only_does_not_load_official_scorer(self) -> None:
        from minimal_shot_av.model import zero_shot_eval

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            candidate_path = tmp_path / "candidates.jsonl"
            output_path = tmp_path / "diagnostics.json"
            candidate_path.write_text(
                json.dumps(
                    {
                        "frame_name": "frame-a",
                        "trajectory_20wp_4hz": [[float(index), 0.0] for index in range(1, 21)],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            argv = [
                "zero_shot_eval",
                "--val-dir",
                str(tmp_path / "val"),
                "--candidate-jsonl",
                str(candidate_path),
                "--output",
                str(output_path),
                "--diagnostics-only",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch("minimal_shot_av.model.wod_e2e.load_preference_frames", return_value=[sample_frame()]),
            ):
                exit_code = zero_shot_eval.main()
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["score_backend"], "geometry_diagnostics_only")
        self.assertEqual(payload["evaluations"][0]["best_reference_error_5s_m"], 0.0)

    def test_cli_ranker_selection_loads_saved_ranker(self) -> None:
        from minimal_shot_av.model import zero_shot_eval

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            candidate_path = tmp_path / "candidates.jsonl"
            ranker_path = tmp_path / "ranker.json"
            output_path = tmp_path / "ranker_diagnostics.json"
            candidate_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "frame_name": "frame-a",
                                "source": "short",
                                "candidate_name": "short",
                                "candidate_index": 0,
                                "trajectory_20wp_4hz": [[float(index) * 0.25, 0.0] for index in range(1, 21)],
                            }
                        ),
                        json.dumps(
                            {
                                "frame_name": "frame-a",
                                "source": "matched",
                                "candidate_name": "matched",
                                "candidate_index": 1,
                                "trajectory_20wp_4hz": [[float(index), 0.0] for index in range(1, 21)],
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            ranker_path.write_text(
                json.dumps(
                    {
                        "numeric_features": ["x_5s"],
                        "candidate_names": ["matched"],
                        "candidate_families": ["matched"],
                        "feature_mean": [0.0, 0.0, 0.0],
                        "feature_scale": [1.0, 1.0, 1.0],
                        "weights": [-1.0, 0.0, 0.0],
                        "bias": 0.0,
                    }
                ),
                encoding="utf-8",
            )
            argv = [
                "zero_shot_eval",
                "--val-dir",
                str(tmp_path / "val"),
                "--candidate-jsonl",
                str(candidate_path),
                "--output",
                str(output_path),
                "--diagnostics-only",
                "--candidate-selection",
                "ranker",
                "--ranker",
                str(ranker_path),
            ]
            with (
                patch.object(sys, "argv", argv),
                patch("minimal_shot_av.model.wod_e2e.load_preference_frames", return_value=[sample_frame()]),
            ):
                exit_code = zero_shot_eval.main()
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["evaluations"][0]["source"], "short")
        self.assertEqual(payload["evaluations"][0]["best_candidate_source"], "matched")
        self.assertGreater(payload["evaluations"][0]["selection_regret"], 0.0)
        self.assertEqual(payload["ranker_unseen_candidate_names"], ["short"])
        self.assertEqual(payload["ranker_unseen_candidate_families"], ["short"])


if __name__ == "__main__":
    unittest.main()
