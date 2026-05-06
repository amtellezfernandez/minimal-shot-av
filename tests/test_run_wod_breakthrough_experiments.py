from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import unittest.mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_wod_breakthrough_experiments.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_wod_breakthrough_experiments", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WodBreakthroughExperimentRunnerTests(unittest.TestCase):
    def test_pilot_uses_local_rfs_full_matrix_and_pilot_slice(self) -> None:
        module = _load_module()
        args = argparse.Namespace(
            smoke=False,
            pilot=True,
            frame_cache=ROOT / "frames.json",
            external_embedding_cache=ROOT / "missing_external_cache.json",
            neural_candidate_model=None,
        )

        runs = module._experiment_matrix(args)
        command = module._command_for_run(runs[0], ROOT / "out.json", args)

        self.assertGreaterEqual(len(runs), 16)
        self.assertIn("champion_fastkin_scene_gate_rate020", {run["name"] for run in runs})
        self.assertIn("champion_fastkin_scene_gate_rate024", {run["name"] for run in runs})
        self.assertIn("fallback_local_selector_linear", {run["name"] for run in runs})
        self.assertIn("blend_mean_pairs", {run["name"] for run in runs})
        self.assertIn("blend_residual_pairs", {run["name"] for run in runs})
        self.assertIn("source_policy_speed_oracle_prior", {run["name"] for run in runs})
        self.assertIn("independent_kinematic_source_gate", {run["name"] for run in runs})
        self.assertIn("conservative_intent_source_gate", {run["name"] for run in runs})
        self.assertIn("conservative_intent_speed_source_gate", {run["name"] for run in runs})
        self.assertIn("zero_shot_reflex_kinematic_linear", {run["name"] for run in runs})
        self.assertIn("zero_shot_reflex_guarded_reactive", {run["name"] for run in runs})
        self.assertIn("zero_shot_reflex_fast_slow_scene_gate", {run["name"] for run in runs})
        self.assertIn("conservative_learned_cap_speed_fine", {run["name"] for run in runs})
        self.assertIn("pairwise_logistic_family_reliability_strong", {run["name"] for run in runs})
        self.assertIn("memory_pairwise_family_reliability_top5", {run["name"] for run in runs})
        self.assertIn("--rfs-backend", command)
        self.assertEqual("local", command[command.index("--rfs-backend") + 1])
        self.assertEqual("160", command[command.index("--max-preference-frames") + 1])
        self.assertEqual("4", command[command.index("--folds") + 1])

    def test_strong_pairwise_run_passes_tuned_pairwise_args(self) -> None:
        module = _load_module()
        args = argparse.Namespace(
            smoke=False,
            pilot=True,
            frame_cache=ROOT / "frames.json",
            external_embedding_cache=ROOT / "missing_external_cache.json",
            neural_candidate_model=None,
        )
        run = next(row for row in module._experiment_matrix(args) if row["name"].endswith("_strong"))

        command = module._command_for_run(run, ROOT / "out.json", args)

        self.assertEqual("900", command[command.index("--selector-pairwise-iterations") + 1])
        self.assertEqual("0.002", command[command.index("--selector-pairwise-l2") + 1])
        self.assertEqual("160", command[command.index("--selector-pairwise-max-pairs-per-frame") + 1])

    def test_source_gate_run_passes_gate_args(self) -> None:
        module = _load_module()
        args = argparse.Namespace(
            smoke=False,
            pilot=True,
            frame_cache=ROOT / "frames.json",
            external_embedding_cache=ROOT / "missing_external_cache.json",
            neural_candidate_model=None,
        )
        run = next(row for row in module._experiment_matrix(args) if row["name"] == "independent_kinematic_source_gate")

        command = module._command_for_run(run, ROOT / "out.json", args)

        self.assertEqual("independent_train_margin", command[command.index("--source-gate") + 1])
        self.assertEqual("kinematic", command[command.index("--source-gate-sources") + 1])
        self.assertEqual("speed", command[command.index("--source-gate-router") + 1])
        self.assertEqual("0.55", command[command.index("--source-gate-min-precision") + 1])

    def test_conservative_source_gate_runs_pass_intent_and_deny_args(self) -> None:
        module = _load_module()
        args = argparse.Namespace(
            smoke=False,
            pilot=True,
            frame_cache=ROOT / "frames.json",
            external_embedding_cache=ROOT / "missing_external_cache.json",
            neural_candidate_model=None,
        )
        runs = module._experiment_matrix(args)
        intent_run = next(row for row in runs if row["name"] == "conservative_intent_source_gate")
        learned_cap_run = next(row for row in runs if row["name"] == "conservative_learned_cap_speed_fine")

        intent_command = module._command_for_run(intent_run, ROOT / "intent.json", args)
        learned_cap_command = module._command_for_run(learned_cap_run, ROOT / "learned_cap.json", args)

        self.assertEqual("intent", intent_command[intent_command.index("--source-gate-router") + 1])
        self.assertEqual("kinematic,temporal", intent_command[intent_command.index("--source-gate-sources") + 1])
        self.assertIn("--source-gate-local-selector", intent_command)
        deny_index = learned_cap_command.index("--source-gate-deny-prefixes") + 1
        self.assertEqual("ridge_residual_pc2,ridge_residual_pc4,ridge_residual_pc5", learned_cap_command[deny_index])
        self.assertEqual("speed_fine", learned_cap_command[learned_cap_command.index("--source-gate-router") + 1])

    def test_champion_run_passes_proven_gate_recipe(self) -> None:
        module = _load_module()
        args = argparse.Namespace(
            smoke=False,
            pilot=True,
            frame_cache=ROOT / "frames.json",
            external_embedding_cache=ROOT / "missing_external_cache.json",
            neural_candidate_model=None,
        )
        run = next(
            row for row in module._experiment_matrix(args) if row["name"] == "champion_fastkin_scene_gate_rate020"
        )

        command = module._command_for_run(run, ROOT / "out.json", args)

        self.assertEqual("base", command[command.index("--kinematic-profile") + 1])
        self.assertEqual("3", command[command.index("--residual-modes") + 1])
        self.assertEqual("temporal_summary", command[command.index("--aux-feature-set") + 1])
        self.assertEqual("off", command[command.index("--residual-grouping") + 1])
        fallback_options_index = command.index("--selector-fallback-source-options") + 1
        self.assertEqual("kinematic;kinematic,temporal", command[fallback_options_index])
        self.assertEqual("train_margin", command[command.index("--source-gate") + 1])
        self.assertEqual("speed:fast", command[command.index("--source-gate-route-allowlist") + 1])
        self.assertEqual("train_margin", command[command.index("--scene-gate") + 1])
        self.assertEqual("0.12", command[command.index("--scene-gate-max-rate") + 1])
        self.assertEqual("external_embeddings", command[command.index("--scene-aux-feature-set") + 1])
        external_cache_index = command.index("--external-embedding-cache") + 1
        self.assertEqual(str(ROOT / "missing_external_cache.json"), command[external_cache_index])

    def test_fallback_local_selector_and_blend_args_are_passed(self) -> None:
        module = _load_module()
        args = argparse.Namespace(
            smoke=False,
            pilot=True,
            frame_cache=ROOT / "frames.json",
            external_embedding_cache=ROOT / "missing_external_cache.json",
            neural_candidate_model=None,
        )
        run = next(
            row
            for row in module._experiment_matrix(args)
            if row["name"] == "fallback_local_selector_blend_mean_pairs"
        )

        command = module._command_for_run(run, ROOT / "out.json", args)

        self.assertIn("--selector-fallback-local-selector", command)
        self.assertEqual("mean_pairs", command[command.index("--blend-candidates") + 1])

    def test_residual_pair_blend_run_passes_blend_arg(self) -> None:
        module = _load_module()
        args = argparse.Namespace(
            smoke=False,
            pilot=True,
            frame_cache=ROOT / "frames.json",
            external_embedding_cache=ROOT / "missing_external_cache.json",
            neural_candidate_model=None,
        )
        run = next(row for row in module._experiment_matrix(args) if row["name"] == "blend_residual_pairs")

        command = module._command_for_run(run, ROOT / "out.json", args)

        self.assertEqual("residual_pairs", command[command.index("--blend-candidates") + 1])

    def test_zero_shot_reflex_runs_use_reflex_kinematic_profile(self) -> None:
        module = _load_module()
        args = SimpleNamespace(
            smoke=False,
            pilot=False,
            frame_cache=ROOT / "frames.json",
            output_dir=ROOT / "artifacts",
            external_embedding_cache=ROOT / "cache.json",
            neural_candidate_model=None,
            transformer_candidate_model=None,
            promotion_baseline=ROOT / "baseline.json",
            min_rfs_gain=0.0,
            continue_on_error=False,
        )
        run = next(
            row for row in module._experiment_matrix(args) if row["name"] == "zero_shot_reflex_kinematic_linear"
        )
        command = module._command_for_run(run, ROOT / "out.json", args)

        self.assertEqual("reflex", command[command.index("--kinematic-profile") + 1])

    def test_zero_shot_reflex_guarded_run_only_gates_reflex_primitives(self) -> None:
        module = _load_module()
        args = SimpleNamespace(
            smoke=False,
            pilot=False,
            frame_cache=ROOT / "frames.json",
            output_dir=ROOT / "artifacts",
            external_embedding_cache=ROOT / "cache.json",
            neural_candidate_model=None,
            transformer_candidate_model=None,
            promotion_baseline=ROOT / "baseline.json",
            min_rfs_gain=0.0,
            continue_on_error=False,
        )
        run = next(
            row for row in module._experiment_matrix(args) if row["name"] == "zero_shot_reflex_guarded_reactive"
        )
        command = module._command_for_run(run, ROOT / "out.json", args)

        self.assertEqual("reflex", command[command.index("--kinematic-profile") + 1])
        self.assertEqual("train_margin", command[command.index("--source-gate") + 1])
        self.assertEqual("0.16", command[command.index("--source-gate-max-rate") + 1])
        prefix_index = command.index("--source-gate-candidate-prefixes") + 1
        self.assertEqual("yield_,lane_offset_,lane_change_,avoid_", command[prefix_index])

    def test_neural_candidate_model_adds_breakthrough_runs_and_args(self) -> None:
        module = _load_module()
        with unittest.mock.patch.object(Path, "is_file", return_value=True):
            args = argparse.Namespace(
                smoke=False,
                pilot=True,
                frame_cache=ROOT / "frames.json",
                external_embedding_cache=ROOT / "missing_external_cache.json",
                neural_candidate_model=ROOT / "neural.json",
                transformer_candidate_model=None,
            )
            runs = module._experiment_matrix(args)
            run = next(row for row in runs if row["name"] == "neural_residual_pairs_family_calibration")

        command = module._command_for_run(run, ROOT / "out.json", args)

        self.assertEqual(str(ROOT / "neural.json"), command[command.index("--neural-candidate-model") + 1])
        self.assertEqual("8", command[command.index("--neural-top-k") + 1])
        self.assertEqual("1", command[command.index("--neural-residual-modes-per-anchor") + 1])

    def test_neural_ensemble_adds_listwise_champion_recipe(self) -> None:
        module = _load_module()
        args = argparse.Namespace(
            smoke=False,
            pilot=True,
            frame_cache=ROOT / "frames.json",
            external_embedding_cache=ROOT / "missing_external_cache.json",
            neural_candidate_model=None,
            neural_candidate_models=f"{ROOT / 'neural_a.json'},{ROOT / 'neural_b.json'}",
            transformer_candidate_model=None,
        )
        with unittest.mock.patch.object(Path, "is_file", return_value=True):
            runs = module._experiment_matrix(args)
        run = next(
            row for row in runs if row["name"] == "neural_ensemble_listwise_temp062_safety_utility_familycal_veto"
        )

        command = module._command_for_run(run, ROOT / "out.json", args)

        self.assertEqual("listwise_softmax", command[command.index("--selector-model") + 1])
        self.assertEqual("100", command[command.index("--selector-ridge") + 1])
        self.assertEqual("900", command[command.index("--selector-listwise-iterations") + 1])
        self.assertEqual("0.18", command[command.index("--selector-listwise-lr") + 1])
        self.assertEqual("0.62", command[command.index("--selector-listwise-temperature") + 1])
        self.assertEqual("safety_utility", command[command.index("--selector-postprocess") + 1])
        self.assertEqual("3", command[command.index("--safety-utility-ridge") + 1])
        self.assertNotIn("--selector-route-targets", command)
        self.assertNotIn("--selector-route-router", command)
        self.assertEqual("speed_source_family", command[command.index("--selector-family-calibration") + 1])
        self.assertEqual("6", command[command.index("--selector-family-calibration-min-count") + 1])
        self.assertEqual("2", command[command.index("--neural-top-k") + 1])
        self.assertEqual("train_margin", command[command.index("--source-veto-gate") + 1])
        self.assertEqual("learned", command[command.index("--source-veto-sources") + 1])
        self.assertEqual("kinematic", command[command.index("--source-veto-fallback-sources") + 1])
        self.assertEqual("0.25", command[command.index("--source-veto-min-precision") + 1])

    def test_transformer_candidate_model_adds_breakthrough_runs_and_args(self) -> None:
        module = _load_module()
        with unittest.mock.patch.object(Path, "is_file", return_value=True):
            args = argparse.Namespace(
                smoke=False,
                pilot=True,
                frame_cache=ROOT / "frames.json",
                external_embedding_cache=ROOT / "missing_external_cache.json",
                neural_candidate_model=None,
                transformer_candidate_model=ROOT / "transformer.pt",
            )
            runs = module._experiment_matrix(args)
            run = next(row for row in runs if row["name"] == "transformer_residual_pairs_family_calibration")

        command = module._command_for_run(run, ROOT / "out.json", args)

        transformer_index = command.index("--transformer-candidate-model") + 1
        self.assertEqual(str(ROOT / "transformer.pt"), command[transformer_index])
        self.assertEqual("12", command[command.index("--transformer-top-k") + 1])

    def test_manifest_rows_get_baseline_deltas_and_pilot_pass_flag(self) -> None:
        module = _load_module()
        rows = [
            _manifest_row("baseline_recheck", rfs=7.0, normalized=8.0, regret=1.5),
            _manifest_row("candidate", rfs=7.2, normalized=8.1, regret=1.4),
            _manifest_row("worse_candidate", rfs=7.3, normalized=8.2, regret=1.8),
        ]

        module._add_baseline_comparisons(rows)

        self.assertAlmostEqual(0.2, rows[1]["baseline_selected_rfs_delta"])
        self.assertAlmostEqual(0.1, rows[1]["baseline_normalized_rfs_delta"])
        self.assertAlmostEqual(-0.1, rows[1]["baseline_worst_slice_regret_delta"])
        self.assertTrue(rows[1]["pilot_baseline_passed"])
        self.assertFalse(rows[2]["pilot_baseline_passed"])

    def test_promoted_run_requires_pilot_baseline_pass(self) -> None:
        module = _load_module()
        rows = [
            _manifest_row("baseline_recheck", rfs=7.0, normalized=8.0, regret=1.5),
            _manifest_row("best_observed", rfs=7.3, normalized=8.2, regret=1.8),
            _manifest_row("promoted", rfs=7.2, normalized=8.1, regret=1.4),
        ]
        module._add_baseline_comparisons(rows)

        promoted = module._promoted_run(rows)

        self.assertIsNotNone(promoted)
        self.assertEqual("promoted", promoted["name"])

    def test_validation_candidate_requires_official_mode(self) -> None:
        module = _load_module()
        rows = [
            _manifest_row("baseline_recheck", rfs=7.0, normalized=8.0, regret=1.5),
            _manifest_row("candidate", rfs=7.2, normalized=8.1, regret=1.4),
        ]
        module._add_baseline_comparisons(rows)

        self.assertIsNone(module._validation_candidate_run(rows, mode="pilot"))
        official = module._validation_candidate_run(rows, mode="official")

        self.assertIsNotNone(official)
        self.assertEqual("candidate", official["name"])

    def test_validation_candidate_requires_minimum_rfs_gain(self) -> None:
        module = _load_module()
        rows = [
            _manifest_row("baseline_recheck", rfs=7.0, normalized=8.0, regret=1.5),
            _manifest_row("too_small", rfs=7.05, normalized=8.1, regret=1.4),
        ]
        module._add_baseline_comparisons(rows)

        self.assertIsNone(module._validation_candidate_run(rows, mode="official", min_rfs_gain=0.1))

    def test_validation_candidate_requires_positive_rfs_gain_when_min_gain_is_zero(self) -> None:
        module = _load_module()
        rows = [
            _manifest_row("baseline_recheck", rfs=7.0, normalized=8.0, regret=1.5),
            _manifest_row("equal", rfs=7.0, normalized=8.1, regret=1.4),
        ]
        module._add_baseline_comparisons(rows)

        self.assertIsNone(module._validation_candidate_run(rows, mode="official", min_rfs_gain=0.0))

    def test_official_baseline_comparison_uses_fixed_promotion_report(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline_path = root / "promotion_baseline.json"
            baseline_path.write_text(
                """{
                  "combined_ranker_mean_rfs": 7.657,
                  "combined_ranker_mean_normalized_rfs": 7.98,
                  "combined_oracle_mean_rfs": 9.1,
                  "slices": {"intent:3": {"mean_regret": 1.7}}
                }""",
                encoding="utf-8",
            )
            rows = [
                _manifest_row("baseline_recheck", rfs=7.0, normalized=8.0, regret=1.5),
                _manifest_row("candidate", rfs=7.7, normalized=8.02, regret=1.6),
            ]

            module._add_baseline_comparisons(rows, baseline_report=baseline_path)

        self.assertAlmostEqual(0.043, rows[1]["baseline_selected_rfs_delta"])
        self.assertAlmostEqual(0.04, rows[1]["baseline_normalized_rfs_delta"])
        self.assertAlmostEqual(-0.1, rows[1]["baseline_worst_slice_regret_delta"])
        self.assertTrue(rows[1]["pilot_baseline_passed"])

    def test_clear_derived_reports_removes_stale_promotion_artifacts(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stale = root / "promoted_report.json"
            run_output = root / "baseline_recheck.json"
            stale.write_text("stale", encoding="utf-8")
            run_output.write_text("{}", encoding="utf-8")

            module._clear_derived_reports(root)

            self.assertFalse(stale.exists())
            self.assertTrue(run_output.exists())


def _manifest_row(name: str, *, rfs: float, normalized: float, regret: float) -> dict[str, object]:
    return {
        "name": name,
        "returncode": 0,
        "combined_ranker_mean_rfs": rfs,
        "combined_ranker_mean_normalized_rfs": normalized,
        "worst_slice": {"slice": "intent:3", "mean_regret": regret},
    }


if __name__ == "__main__":
    unittest.main()
