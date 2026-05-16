from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_wod_grounding_ablation_table.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_wod_grounding_ablation_table", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuildWodGroundingAblationTableTests(unittest.TestCase):
    def test_markdown_table_contains_expected_row_names(self) -> None:
        module = _load_module()
        rows = [
            {
                "name": "Scalar / geometry only",
                "combined_ranker_mean_rfs": 7.0,
                "combined_oracle_mean_rfs": 9.0,
                "combined_ranker_regret_to_oracle": 2.0,
                "combined_ranker_top1_oracle_match_rate": 0.4,
                "embedding_source": None,
                "direct_policy_enabled": False,
            },
            {
                "name": "Cosmos 64d nonlinear head",
                "combined_ranker_mean_rfs": 7.8,
                "combined_oracle_mean_rfs": 9.2,
                "combined_ranker_regret_to_oracle": 1.4,
                "combined_ranker_top1_oracle_match_rate": 0.39,
                "embedding_source": "cosmos",
                "direct_policy_enabled": True,
            },
        ]

        table = module._markdown_table(rows)

        self.assertIn("Scalar / geometry only", table)
        self.assertIn("Cosmos 64d nonlinear head", table)
        self.assertIn("Cosmos + nonlinear head", table)


if __name__ == "__main__":
    unittest.main()
