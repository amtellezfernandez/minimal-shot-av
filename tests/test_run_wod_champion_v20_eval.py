from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_wod_champion_v20_eval.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_wod_champion_v20_eval", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WodChampionV20EvalTests(unittest.TestCase):
    def test_champion_command_contains_direct_policy_config(self) -> None:
        module = _load_module()
        args = argparse.Namespace(
            python="python",
            frame_cache=ROOT / "frames.json",
            external_embedding_cache=ROOT / "cache.json",
            neural_candidate_models="models.json",
            rfs_backend="local",
            waymo_src=None,
            folds=2,
            progress_every_fold=False,
        )

        command = module._command_for_variant(
            args,
            variant="champion",
            frames=120,
            output=ROOT / "champion.json",
        )

        self.assertEqual("direct_policy", command[command.index("--selector-postprocess") + 1])
        self.assertEqual("relative_contextual", command[command.index("--direct-policy-feature-mode") + 1])
        self.assertEqual("learned,temporal", command[command.index("--direct-policy-candidate-sources") + 1])
        self.assertEqual("temporal,kinematic", command[command.index("--direct-policy-baseline-sources") + 1])
        self.assertEqual("120", command[command.index("--max-preference-frames") + 1])

    def test_no_direct_command_disables_postprocess(self) -> None:
        module = _load_module()
        args = argparse.Namespace(
            python="python",
            frame_cache=ROOT / "frames.json",
            external_embedding_cache=ROOT / "cache.json",
            neural_candidate_models="models.json",
            rfs_backend="local",
            waymo_src=None,
            folds=2,
            progress_every_fold=True,
        )

        command = module._command_for_variant(
            args,
            variant="no_direct",
            frames=240,
            output=ROOT / "no_direct.json",
        )

        self.assertEqual("off", command[command.index("--selector-postprocess") + 1])
        self.assertEqual("240", command[command.index("--max-preference-frames") + 1])
        self.assertIn("--progress-every-fold", command)

    def test_clean_gate_reliability_uses_gate_feature_mode_without_direct_policy(self) -> None:
        module = _load_module()
        args = argparse.Namespace(
            python="python",
            frame_cache=ROOT / "frames.json",
            external_embedding_cache=ROOT / "cache.json",
            neural_candidate_models="models.json",
            rfs_backend="local",
            waymo_src=None,
            folds=2,
            progress_every_fold=False,
        )

        command = module._command_for_variant(
            args,
            variant="clean_gate_reliability",
            frames=479,
            output=ROOT / "clean.json",
        )

        self.assertEqual("off", command[command.index("--selector-postprocess") + 1])
        self.assertIn("--learned-reliability-gate-features", command)
        self.assertEqual(
            "learned_reliability_confidence_contextual",
            command[command.index("--learned-reliability-gate-feature-mode") + 1],
        )
        self.assertEqual(
            module.GATE_RELIABILITY_CONFIG["source_gate_max_rate"],
            command[command.index("--source-gate-max-rate") + 1],
        )
        self.assertEqual(
            "internnav_s2_waypoint_cautious,internnav_s1_stop_progress,internnav_s2_waypoint_progress,ensemble",
            command[command.index("--source-gate-candidate-prefixes") + 1],
        )
        self.assertEqual("2.0", command[command.index("--scene-gate-margin") + 1])
        self.assertEqual("boosted_stumps", command[command.index("--source-gate-model") + 1])
        self.assertEqual("160", command[command.index("--source-gate-stump-iterations") + 1])
        self.assertEqual("0.04", command[command.index("--source-gate-stump-lr") + 1])
        self.assertEqual("0.05", command[command.index("--source-gate-conformal-alpha") + 1])
        self.assertEqual("latent_predictive", command[command.index("--world-prior") + 1])
        self.assertEqual("mixed", command[command.index("--world-prior-mask-mode") + 1])

    def test_hgb_direct_variant_uses_v20_direct_policy(self) -> None:
        module = _load_module()
        args = argparse.Namespace(
            python="python",
            frame_cache=ROOT / "frames.json",
            external_embedding_cache=ROOT / "cache.json",
            neural_candidate_models="models.json",
            rfs_backend="local",
            waymo_src=None,
            folds=2,
            progress_every_fold=False,
        )

        command = module._command_for_variant(
            args,
            variant="hgb_direct",
            frames=479,
            output=ROOT / "hgb.json",
        )

        self.assertEqual("direct_policy", command[command.index("--selector-postprocess") + 1])
        self.assertEqual("sklearn_hist_gradient_boosting", command[command.index("--direct-policy-model") + 1])
        self.assertEqual("relative_world_prior_contextual", command[command.index("--direct-policy-feature-mode") + 1])
        self.assertEqual("0.0", command[command.index("--direct-policy-conformal-alpha") + 1])
        self.assertEqual(
            module.HGB_DIRECT_CONFIG["direct_policy_min_margin"],
            command[command.index("--direct-policy-min-margin") + 1],
        )
        self.assertEqual(
            module.HGB_DIRECT_CONFIG["direct_policy_min_precision"],
            command[command.index("--direct-policy-min-precision") + 1],
        )
        self.assertEqual(
            module.HGB_DIRECT_CONFIG["direct_policy_stump_thresholds"],
            command[command.index("--direct-policy-stump-thresholds") + 1],
        )
        self.assertEqual("learned,internnav", command[command.index("--direct-policy-candidate-sources") + 1])
        self.assertEqual("boosted_stumps", command[command.index("--source-gate-model") + 1])
        self.assertEqual("0.05", command[command.index("--source-gate-conformal-alpha") + 1])
        self.assertEqual("latent_predictive", command[command.index("--world-prior") + 1])
        self.assertIn("--learned-reliability-gate-features", command)

    def test_jepa_variant_enables_latent_predictive_prior(self) -> None:
        module = _load_module()
        args = argparse.Namespace(
            python="python",
            frame_cache=ROOT / "frames.json",
            external_embedding_cache=ROOT / "cache.json",
            neural_candidate_models="models.json",
            rfs_backend="local",
            waymo_src=None,
            folds=2,
            progress_every_fold=False,
        )

        command = module._command_for_variant(
            args,
            variant="jepa_precedent",
            frames=120,
            output=ROOT / "jepa.json",
        )

        self.assertEqual(
            "relative_precedent_latent_contextual", command[command.index("--direct-policy-feature-mode") + 1]
        )
        self.assertEqual("latent_predictive", command[command.index("--world-prior") + 1])
        self.assertEqual("mixed", command[command.index("--world-prior-mask-mode") + 1])

    def test_summary_adds_delta_against_matching_no_direct(self) -> None:
        module = _load_module()
        rows = [
            {"variant": "no_direct", "frames": 120, "combined_ranker_mean_rfs": 7.0},
            {"variant": "champion", "frames": 120, "combined_ranker_mean_rfs": 7.25},
        ]

        updated = module._add_ablation_deltas(rows)

        self.assertAlmostEqual(0.25, updated[1]["delta_vs_no_direct_rfs"])

    def test_parse_variants_rejects_unknown(self) -> None:
        module = _load_module()

        self.assertEqual(["champion", "no_direct"], module._parse_variants("champion,no_direct"))
        self.assertEqual(["hgb_direct"], module._parse_variants("hgb_direct"))
        self.assertEqual(["clean_gate_reliability"], module._parse_variants("clean_gate_reliability"))
        with self.assertRaises(ValueError):
            module._parse_variants("champion,unknown")


if __name__ == "__main__":
    unittest.main()
