from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "tune_wod_direct_policy_optuna.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("tune_wod_direct_policy_optuna", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTrial:
    number = 7

    def __init__(self):
        self.attrs = {}

    def suggest_categorical(self, name, choices):
        if name == "direct_policy_model":
            return "random_fourier"
        if name == "direct_policy_identity_features":
            return "none"
        if name == "direct_policy_router":
            return "speed_fine"
        return choices[0]

    def suggest_float(self, _name, low, _high, log=False):
        return low

    def suggest_int(self, _name, low, _high, log=False):
        return low

    def set_user_attr(self, key, value):
        self.attrs[key] = value


class WodDirectPolicyOptunaTunerTests(unittest.TestCase):
    def test_suggested_trial_builds_direct_policy_command(self) -> None:
        module = _load_module()
        args = SimpleNamespace(
            python="python3",
            frame_cache=ROOT / "frames.json",
            external_embedding_cache=ROOT / "embeddings.json",
            neural_candidate_models="a.json,b.json",
            rfs_backend="local",
            max_preference_frames=40,
            folds=2,
            progress_every_fold=True,
            model_choices="linear,random_fourier",
            include_torch=False,
            identity_mode="both",
        )

        config = module._suggest_config(FakeTrial(), args)
        command = module._command_for_trial(config, ROOT / "out.json", args)

        self.assertEqual("random_fourier", config["direct_policy_model"])
        self.assertIn("--selector-postprocess", command)
        self.assertEqual("direct_policy", command[command.index("--selector-postprocess") + 1])
        self.assertEqual("random_fourier", command[command.index("--direct-policy-model") + 1])
        self.assertEqual("none", command[command.index("--direct-policy-identity-features") + 1])
        self.assertEqual("speed_fine", command[command.index("--direct-policy-router") + 1])
        self.assertEqual("selector", command[command.index("--direct-policy-feature-mode") + 1])
        self.assertEqual("0.0", command[command.index("--direct-policy-min-margin") + 1])
        self.assertEqual("0", command[command.index("--direct-policy-min-route-observations") + 1])
        self.assertEqual("0", command[command.index("--direct-policy-min-route-positives") + 1])
        self.assertEqual("0.0", command[command.index("--direct-policy-risk-weight") + 1])
        self.assertEqual("40", command[command.index("--max-preference-frames") + 1])
        self.assertEqual("2", command[command.index("--folds") + 1])
        self.assertIn("--progress-every-fold", command)

    def test_world_prior_feature_mode_enables_latent_predictive_prior(self) -> None:
        module = _load_module()
        args = SimpleNamespace(
            python="python3",
            frame_cache=ROOT / "frames.json",
            external_embedding_cache=ROOT / "embeddings.json",
            neural_candidate_models="",
            rfs_backend="local",
            max_preference_frames=None,
            folds=None,
            progress_every_fold=False,
        )
        config = {
            "direct_policy_model": "linear",
            "direct_policy_identity_features": "none",
            "direct_policy_router": "speed",
            "direct_policy_feature_mode": "relative_world_prior_contextual",
            "direct_policy_ridge": 1.0,
            "direct_policy_min_margin": 0.0,
            "direct_policy_max_rate": 0.1,
            "direct_policy_min_precision": 0.0,
            "direct_policy_conformal_alpha": 0.05,
            "direct_policy_risk_weight": 0.0,
            "scene_gate_ridge": 1.0,
            "scene_gate_max_rate": 0.1,
            "source_gate_ridge": 0.1,
            "source_gate_max_rate": 0.2,
            "source_gate_min_precision": 0.0,
            "world_prior_mask_mode": "mixed",
        }

        command = module._command_for_trial(config, ROOT / "out.json", args)

        self.assertEqual("relative_world_prior_contextual", command[command.index("--direct-policy-feature-mode") + 1])
        self.assertEqual("0.05", command[command.index("--direct-policy-conformal-alpha") + 1])
        self.assertEqual("latent_predictive", command[command.index("--world-prior") + 1])
        self.assertEqual("mixed", command[command.index("--world-prior-mask-mode") + 1])

    def test_reliability_gate_recipe_builds_v20_gate_command(self) -> None:
        module = _load_module()
        args = SimpleNamespace(
            python="python3",
            frame_cache=ROOT / "frames.json",
            external_embedding_cache=ROOT / "embeddings.json",
            neural_candidate_models="",
            rfs_backend="local",
            max_preference_frames=None,
            folds=None,
            progress_every_fold=False,
            learned_reliability_gate_feature_mode="learned_reliability_confidence_contextual",
        )
        config = {
            **module._gate_recipe_config("reliability"),
            "gate_recipe": "reliability",
            "direct_policy_model": "sklearn_hist_gradient_boosting",
            "direct_policy_identity_features": "none",
            "direct_policy_router": "speed",
            "direct_policy_feature_mode": "relative_world_prior_contextual",
            "direct_policy_ridge": 0.15,
            "direct_policy_min_margin": 0.0,
            "direct_policy_max_rate": 0.18,
            "direct_policy_min_precision": 0.45,
            "direct_policy_risk_weight": 0.15,
            "direct_policy_conformal_alpha": 0.05,
            "direct_policy_stump_iterations": 160,
            "direct_policy_stump_lr": 0.04,
            "direct_policy_stump_thresholds": 24,
        }

        command = module._command_for_trial(config, ROOT / "out.json", args)

        self.assertEqual("2.0", command[command.index("--scene-gate-margin") + 1])
        self.assertEqual("boosted_stumps", command[command.index("--source-gate-model") + 1])
        self.assertEqual("0.05", command[command.index("--source-gate-conformal-alpha") + 1])
        self.assertIn("--learned-reliability-gate-features", command)
        self.assertEqual(
            "learned_reliability_confidence_contextual",
            command[command.index("--learned-reliability-gate-feature-mode") + 1],
        )
        self.assertEqual("latent_predictive", command[command.index("--world-prior") + 1])

    def test_direct_policy_source_filters_are_passed_to_trials(self) -> None:
        module = _load_module()
        args = SimpleNamespace(
            python="python3",
            frame_cache=ROOT / "frames.json",
            external_embedding_cache=ROOT / "embeddings.json",
            neural_candidate_models="",
            rfs_backend="local",
            max_preference_frames=None,
            folds=None,
            progress_every_fold=False,
        )
        config = {
            "direct_policy_model": "linear",
            "direct_policy_identity_features": "none",
            "direct_policy_router": "speed",
            "direct_policy_feature_mode": "relative_contextual",
            "direct_policy_ridge": 1.0,
            "direct_policy_min_margin": 0.0,
            "direct_policy_max_rate": 0.1,
            "direct_policy_min_precision": 0.0,
            "direct_policy_conformal_alpha": 0.0,
            "direct_policy_risk_weight": 0.0,
            "direct_policy_candidate_sources": "internnav,learned",
            "direct_policy_baseline_sources": "scene,temporal",
            "scene_gate_ridge": 1.0,
            "scene_gate_max_rate": 0.1,
            "source_gate_ridge": 0.1,
            "source_gate_max_rate": 0.2,
            "source_gate_min_precision": 0.0,
        }

        command = module._command_for_trial(config, ROOT / "out.json", args)

        self.assertEqual("internnav,learned", command[command.index("--direct-policy-candidate-sources") + 1])
        self.assertEqual("scene,temporal", command[command.index("--direct-policy-baseline-sources") + 1])

    def test_objective_score_can_penalize_false_positive_loss(self) -> None:
        module = _load_module()
        report = {
            "combined_ranker_mean_rfs": 7.75,
            "frames": 100,
            "direct_policy_false_positive_loss_sum": 5.0,
        }

        score = module._objective_score(report, false_positive_penalty=0.2)

        self.assertAlmostEqual(7.74, score)

    def test_objective_score_can_penalize_false_positive_count(self) -> None:
        module = _load_module()
        report = {
            "combined_ranker_mean_rfs": 7.75,
            "frames": 100,
            "direct_policy_false_positive_count": 4,
        }

        score = module._objective_score(report, false_positive_penalty=0.0, false_positive_count_penalty=5.0)

        self.assertAlmostEqual(7.55, score)

    def test_objective_score_can_reward_realized_gain_and_penalize_inactivity(self) -> None:
        module = _load_module()
        report = {
            "combined_ranker_mean_rfs": 7.0,
            "frames": 20,
            "direct_policy_gain_sum": 4.0,
            "direct_policy_false_positive_loss_sum": 1.0,
            "direct_policy_override_count": 1,
        }

        score = module._objective_score(
            report,
            false_positive_penalty=2.0,
            gain_reward=3.0,
            min_direct_policy_overrides=3,
            min_direct_policy_selected_rate=0.0,
            underactive_penalty=10.0,
        )

        self.assertAlmostEqual(6.5, score)

    def test_objective_activity_uses_true_positive_overrides_when_available(self) -> None:
        module = _load_module()
        report = {
            "combined_ranker_mean_rfs": 7.0,
            "frames": 20,
            "direct_policy_override_count": 3,
            "direct_policy_true_positive_count": 0,
        }

        score = module._objective_score(
            report,
            false_positive_penalty=0.0,
            gain_reward=0.0,
            min_direct_policy_overrides=2,
            min_direct_policy_selected_rate=0.0,
            underactive_penalty=10.0,
        )

        self.assertAlmostEqual(6.0, score)

    def test_objective_score_can_penalize_fold_instability(self) -> None:
        module = _load_module()
        report = {
            "combined_ranker_mean_rfs": 8.0,
            "frames": 100,
            "folds_detail": [
                {"frames": 50, "combined_ranker_mean_rfs": 8.2},
                {"frames": 50, "combined_ranker_mean_rfs": 7.9},
            ],
        }

        score = module._objective_score(
            report,
            false_positive_penalty=0.0,
            fold_rfs_spread_penalty=0.5,
        )

        self.assertAlmostEqual(7.85, score)

    def test_objective_score_can_penalize_worst_fold_false_positive_loss(self) -> None:
        module = _load_module()
        report = {
            "combined_ranker_mean_rfs": 8.0,
            "frames": 100,
            "folds_detail": [
                {"frames": 50, "direct_policy_false_positive_loss_sum": 10.0},
                {"frames": 50, "direct_policy_false_positive_loss_sum": 1.0},
            ],
        }

        score = module._objective_score(
            report,
            false_positive_penalty=0.0,
            max_fold_false_positive_penalty=2.0,
        )

        self.assertAlmostEqual(7.6, score)

    def test_objective_score_can_penalize_active_fold_precision_shortfall(self) -> None:
        module = _load_module()
        report = {
            "combined_ranker_mean_rfs": 8.0,
            "frames": 100,
            "folds_detail": [
                {"direct_policy_override_count": 4, "direct_policy_precision": 0.25},
                {"direct_policy_override_count": 2, "direct_policy_precision": 0.75},
                {"direct_policy_override_count": 0, "direct_policy_precision": 0.0},
            ],
        }

        score = module._objective_score(
            report,
            false_positive_penalty=0.0,
            min_fold_direct_policy_precision=0.5,
            fold_precision_penalty=2.0,
            fold_precision_min_overrides=1,
        )
        ignored = module._objective_score(
            report,
            false_positive_penalty=0.0,
            min_fold_direct_policy_precision=0.5,
            fold_precision_penalty=2.0,
            fold_precision_min_overrides=5,
        )

        self.assertAlmostEqual(7.75, score)
        self.assertAlmostEqual(8.0, ignored)

    def test_float_range_validates_bounds(self) -> None:
        module = _load_module()

        self.assertEqual((0.02, 0.35), module._float_range("0.02,0.35", name="range"))
        with self.assertRaises(ValueError):
            module._float_range("-0.1,0.2", name="range", bounded_unit=True)
        with self.assertRaises(ValueError):
            module._float_range("-0.1,0.2", name="range", non_negative=True)

    def test_int_range_validates_bounds(self) -> None:
        module = _load_module()

        self.assertEqual((80, 240), module._int_range("80,240", name="range"))
        with self.assertRaises(ValueError):
            module._int_range("240,80", name="range")
        with self.assertRaises(ValueError):
            module._int_range("0,240", name="range")

    def test_int_choices_validate_and_deduplicate(self) -> None:
        module = _load_module()

        self.assertEqual([0, 6, 12], module._int_choices("0,6,6,12", name="choices"))
        with self.assertRaises(ValueError):
            module._int_choices("-1,2", name="choices")
        with self.assertRaises(ValueError):
            module._int_choices("", name="choices")

    def test_float_choices_validate_and_deduplicate(self) -> None:
        module = _load_module()

        self.assertEqual([0.0, 0.05], module._float_choices("0,0.05,0.05", name="choices", less_than=0.5))
        with self.assertRaises(ValueError):
            module._float_choices("0.5", name="choices", less_than=0.5)
        with self.assertRaises(ValueError):
            module._float_choices("-0.1", name="choices", non_negative=True)

    def test_source_set_choices_validate_and_normalize(self) -> None:
        module = _load_module()

        self.assertEqual(["", "internnav,learned"], module._source_set_choices("all;internnav,learned", name="sets"))
        with self.assertRaises(ValueError):
            module._source_set_choices("unknown", name="sets")

    def test_explicit_world_prior_feature_modes_are_not_expanded(self) -> None:
        module = _load_module()
        args = SimpleNamespace(
            direct_policy_feature_modes="relative_world_prior_contextual",
            include_world_prior_direct_policy_features=True,
        )

        self.assertEqual(["relative_world_prior_contextual"], module._direct_policy_feature_mode_choices(args))

    def test_objective_reuses_matching_artifact_without_rerun(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                python="python3",
                frame_cache=ROOT / "frames.json",
                external_embedding_cache=ROOT / "embeddings.json",
                neural_candidate_models="a.json,b.json",
                rfs_backend="local",
                max_preference_frames=40,
                folds=2,
                progress_every_fold=True,
                model_choices="linear,random_fourier",
                include_torch=False,
                identity_mode="both",
                rerun_existing=False,
                false_positive_penalty=0.0,
                output_dir=Path(tmpdir),
                trial_timeout=None,
            )
            trial = FakeTrial()
            config = module._suggest_config(trial, args)
            output = args.output_dir / "trial_0007.json"
            command = module._command_for_trial(config, output, args)
            fingerprint = module._trial_fingerprint(config, command)
            output.write_text(
                json.dumps({"combined_ranker_mean_rfs": 9.25, "frames": 10}) + "\n",
                encoding="utf-8",
            )
            module._trial_metadata_path(output).write_text(
                json.dumps({"schema": "wod_direct_policy_trial_metadata_v1", "fingerprint": fingerprint}) + "\n",
                encoding="utf-8",
            )

            called = {"count": 0}

            def _should_not_run(*_args, **_kwargs):
                called["count"] += 1
                raise AssertionError("cached artifact should have been reused")

            original_run_command = module._run_command
            module._run_command = _should_not_run
            try:
                score = module._objective(trial, args)
            finally:
                module._run_command = original_run_command

            self.assertEqual(0, called["count"])
            self.assertAlmostEqual(9.25, score)
            self.assertEqual(fingerprint, trial.attrs["fingerprint"])

    def test_objective_reruns_when_cached_artifact_does_not_match(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                python="python3",
                frame_cache=ROOT / "frames.json",
                external_embedding_cache=ROOT / "embeddings.json",
                neural_candidate_models="a.json,b.json",
                rfs_backend="local",
                max_preference_frames=40,
                folds=2,
                progress_every_fold=True,
                model_choices="linear,random_fourier",
                include_torch=False,
                identity_mode="both",
                rerun_existing=False,
                false_positive_penalty=0.0,
                output_dir=Path(tmpdir),
                trial_timeout=None,
            )
            trial = FakeTrial()
            config = module._suggest_config(trial, args)
            output = args.output_dir / "trial_0007.json"
            command = module._command_for_trial(config, output, args)
            output.write_text(
                json.dumps({"combined_ranker_mean_rfs": 8.0, "frames": 10}) + "\n",
                encoding="utf-8",
            )
            module._trial_metadata_path(output).write_text(
                json.dumps({"schema": "wod_direct_policy_trial_metadata_v1", "fingerprint": "stale"}) + "\n",
                encoding="utf-8",
            )

            called = {"count": 0}

            def _run_once(*_args, **_kwargs):
                called["count"] += 1
                return SimpleNamespace(returncode=0)

            original_run_command = module._run_command
            module._run_command = _run_once
            try:
                score = module._objective(trial, args)
            finally:
                module._run_command = original_run_command

            self.assertEqual(1, called["count"])
            self.assertAlmostEqual(8.0, score)
            self.assertEqual(module._trial_fingerprint(config, command), trial.attrs["fingerprint"])

    def test_objective_times_out_as_failed_trial(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                python="python3",
                frame_cache=ROOT / "frames.json",
                external_embedding_cache=ROOT / "embeddings.json",
                neural_candidate_models="a.json,b.json",
                rfs_backend="local",
                max_preference_frames=40,
                folds=2,
                progress_every_fold=True,
                model_choices="linear,random_fourier",
                include_torch=False,
                identity_mode="both",
                rerun_existing=False,
                false_positive_penalty=0.0,
                output_dir=Path(tmpdir),
                trial_timeout=1.0,
            )
            trial = FakeTrial()

            def _timeout(*_args, **_kwargs):
                raise subprocess.TimeoutExpired(cmd=["python3"], timeout=1.0)

            original_run_command = module._run_command
            module._run_command = _timeout
            try:
                score = module._objective(trial, args)
            finally:
                module._run_command = original_run_command

            self.assertEqual(-1.0e9, score)
            self.assertTrue(trial.attrs["timed_out"])
            self.assertEqual(124, trial.attrs["returncode"])


if __name__ == "__main__":
    unittest.main()
