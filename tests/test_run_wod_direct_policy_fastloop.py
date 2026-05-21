from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_wod_direct_policy_fastloop.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_wod_direct_policy_fastloop", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WodDirectPolicyFastloopTests(unittest.TestCase):
    def test_tuning_command_uses_short_local_slice_and_gain_aware_defaults(self) -> None:
        module = _load_module()
        args = argparse.Namespace(
            python="python",
            study_name="fastloop",
            trials=6,
            trial_timeout=1800.0,
            frame_cache=ROOT / "frames.json",
            external_embedding_cache=ROOT / "cache.json",
            neural_candidate_models="models.json",
            local_frames=20,
            local_folds=2,
            official_frames=20,
            official_folds=2,
            waymo_src=ROOT / "waymo",
            skip_official_preflight=False,
            full_official_output=None,
            gain_reward=4.0,
            false_positive_penalty=1.0,
            min_direct_policy_overrides=5,
            min_direct_policy_selected_rate=0.0,
            underactive_penalty=8.0,
            progress_every_fold=False,
        )

        command = module._tuning_command(args, ROOT / "optuna")

        self.assertEqual("python", command[0])
        self.assertIn("--rfs-backend", command)
        self.assertEqual("local", command[command.index("--rfs-backend") + 1])
        self.assertEqual("20", command[command.index("--max-preference-frames") + 1])
        self.assertEqual("2", command[command.index("--folds") + 1])
        self.assertEqual("4.0", command[command.index("--gain-reward") + 1])
        self.assertEqual("1.0", command[command.index("--false-positive-penalty") + 1])
        self.assertEqual("5", command[command.index("--min-direct-policy-overrides") + 1])
        self.assertEqual("full", command[command.index("--identity-mode") + 1])

    def test_official_command_rewrites_backend_output_and_waymo_source(self) -> None:
        module = _load_module()
        command = [
            "python",
            str(ROOT / "scripts" / "evaluate_wod_trajectory_model_cv.py"),
            "--output",
            "artifacts/local.json",
            "--rfs-backend",
            "local",
            "--max-preference-frames",
            "20",
            "--folds",
            "2",
        ]

        rewritten = module._official_preflight_command(
            command,
            output=ROOT / "official.json",
            waymo_src=ROOT / "workspace" / "waymo-open-dataset/src",
            frames=20,
            folds=2,
            progress_every_fold=True,
        )

        self.assertEqual("official", rewritten[rewritten.index("--rfs-backend") + 1])
        self.assertEqual("official.json", Path(rewritten[rewritten.index("--output") + 1]).name)
        self.assertEqual(str(ROOT / "workspace" / "waymo-open-dataset/src"), rewritten[rewritten.index("--waymo-src") + 1])
        self.assertEqual("20", rewritten[rewritten.index("--max-preference-frames") + 1])
        self.assertEqual("2", rewritten[rewritten.index("--folds") + 1])
        self.assertIn("--progress-every-fold", rewritten)

    def test_official_command_appends_missing_waymo_source(self) -> None:
        module = _load_module()
        command = ["python", "evaluate.py", "--output", "artifacts/local.json", "--rfs-backend", "local"]

        rewritten = module._official_preflight_command(
            command,
            output=ROOT / "official.json",
            waymo_src=ROOT / "workspace" / "waymo-open-dataset/src",
            frames=20,
            folds=2,
            progress_every_fold=False,
        )

        self.assertIn("--waymo-src", rewritten)
        self.assertIn("--folds", rewritten)
        self.assertIn("--max-preference-frames", rewritten)

    def test_official_sweep_builds_relaxed_threshold_variants(self) -> None:
        module = _load_module()
        base_command = [
            "python",
            str(ROOT / "scripts" / "evaluate_wod_trajectory_model_cv.py"),
            "--output",
            "artifacts/local.json",
            "--rfs-backend",
            "local",
            "--direct-policy-min-precision",
            "0.56",
            "--direct-policy-max-rate",
            "0.28",
        ]

        variants = module._official_sweep_commands(
            base_command,
            output_dir=ROOT / "artifacts" / "tmp_fastloop",
            waymo_src=ROOT / "workspace" / "waymo-open-dataset/src",
            frames=20,
            folds=2,
            progress_every_fold=True,
        )

        labels = [label for label, _, _ in variants]
        self.assertEqual(
            [
                "base",
                "relaxed",
                "permissive",
                "mild",
                "tight",
                "stricter",
                "ultra_tight",
                "risk_off",
                "risk_mid",
                "risk_high",
                "margin_light",
                "margin_safety",
            ],
            labels,
        )
        relaxed = variants[1][1]
        permissive = variants[2][1]
        mild = variants[3][1]
        tight = variants[4][1]
        stricter = variants[5][1]
        ultra_tight = variants[6][1]
        risk_off = variants[7][1]
        risk_mid = variants[8][1]
        risk_high = variants[9][1]
        margin_light = variants[10][1]
        margin_safety = variants[11][1]
        self.assertEqual("0.0", relaxed[relaxed.index("--direct-policy-min-precision") + 1])
        self.assertEqual("0.85", relaxed[relaxed.index("--direct-policy-max-rate") + 1])
        self.assertEqual("0.0", permissive[permissive.index("--direct-policy-min-precision") + 1])
        self.assertEqual("0.95", permissive[permissive.index("--direct-policy-max-rate") + 1])
        self.assertEqual("0.05", mild[mild.index("--direct-policy-min-precision") + 1])
        self.assertEqual("0.65", mild[mild.index("--direct-policy-max-rate") + 1])
        self.assertEqual("0.15", tight[tight.index("--direct-policy-min-precision") + 1])
        self.assertEqual("0.45", tight[tight.index("--direct-policy-max-rate") + 1])
        self.assertEqual("0.25", stricter[stricter.index("--direct-policy-min-precision") + 1])
        self.assertEqual("0.3", stricter[stricter.index("--direct-policy-max-rate") + 1])
        self.assertEqual("0.35", ultra_tight[ultra_tight.index("--direct-policy-min-precision") + 1])
        self.assertEqual("0.2", ultra_tight[ultra_tight.index("--direct-policy-max-rate") + 1])
        self.assertEqual("0.0", risk_off[risk_off.index("--direct-policy-risk-weight") + 1])
        self.assertEqual("0.75", risk_mid[risk_mid.index("--direct-policy-risk-weight") + 1])
        self.assertEqual("1.5", risk_high[risk_high.index("--direct-policy-risk-weight") + 1])
        self.assertEqual("0.25", margin_light[margin_light.index("--direct-policy-min-margin") + 1])
        self.assertEqual("0.5", margin_safety[margin_safety.index("--direct-policy-min-margin") + 1])

    def test_best_official_report_prefers_higher_rfs_then_overrides(self) -> None:
        module = _load_module()
        reports = [
            {
                "label": "base",
                "output": "a",
                "report": {
                    "combined_ranker_mean_rfs": 7.0,
                    "direct_policy_override_count": 0,
                    "direct_policy_gain_sum": 0.0,
                },
            },
            {
                "label": "relaxed",
                "output": "b",
                "report": {
                    "combined_ranker_mean_rfs": 7.0,
                    "direct_policy_override_count": 2,
                    "direct_policy_gain_sum": 1.0,
                },
            },
            {
                "label": "permissive",
                "output": "c",
                "report": {
                    "combined_ranker_mean_rfs": 6.5,
                    "direct_policy_override_count": 5,
                    "direct_policy_gain_sum": 3.0,
                },
            },
        ]

        best = module._best_official_report(reports)

        self.assertEqual("relaxed", best["label"])
        self.assertEqual("b", best["output"])


if __name__ == "__main__":
    unittest.main()
