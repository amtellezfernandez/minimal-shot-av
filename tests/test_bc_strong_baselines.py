from __future__ import annotations

import json
import unittest

import numpy as np
import tempfile
from pathlib import Path

from scripts.bc_collect_dagger_data import TOKEN_ORDER as DAGGER_TOKEN_ORDER
from scripts.bc_train import (
    GeomTransformerTokenBC,
    TOKEN_ORDER as TRAIN_TOKEN_ORDER,
    _classification_loss,
    _load_dagger_merge,
    _parse_source_weights,
    _token_checkpoint_payload,
    _validate_token_contract,
    build_history_windows,
)


class BCStrongBaselineTests(unittest.TestCase):
    def test_history_windows_left_pad_within_each_rollout(self) -> None:
        features = np.array(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
                [5.0, 50.0],
            ],
            dtype=np.float32,
        )
        rollout_id = np.array([0, 0, 0, 1, 1], dtype=np.int64)

        windows = build_history_windows(features, rollout_id, history_len=3)

        np.testing.assert_array_equal(windows[0], np.array([[1.0, 10.0], [1.0, 10.0], [1.0, 10.0]]))
        np.testing.assert_array_equal(windows[1], np.array([[1.0, 10.0], [1.0, 10.0], [2.0, 20.0]]))
        np.testing.assert_array_equal(windows[2], np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]))
        np.testing.assert_array_equal(windows[3], np.array([[4.0, 40.0], [4.0, 40.0], [4.0, 40.0]]))
        np.testing.assert_array_equal(windows[4], np.array([[4.0, 40.0], [4.0, 40.0], [5.0, 50.0]]))

    def test_dagger_token_order_matches_bc_contract(self) -> None:
        self.assertEqual(
            [
                "stop",
                "crawl",
                "maintain",
                "slow_yield",
                "nudge_left",
                "nudge_right",
                "evasive_left",
                "evasive_right",
                "lane_recover",
            ],
            DAGGER_TOKEN_ORDER,
        )
        self.assertEqual(tuple(DAGGER_TOKEN_ORDER), TRAIN_TOKEN_ORDER)

    def test_validate_token_contract_rejects_mismatched_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            bad_tokens = list(TRAIN_TOKEN_ORDER)
            bad_tokens[TRAIN_TOKEN_ORDER.index("maintain")] = "unknown_token"
            (data_dir / "meta.json").write_text(json.dumps({"token_names": bad_tokens}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "token_names do not match"):
                _validate_token_contract(data_dir, np.array([0, 1, 2], dtype=np.int64))

    def test_token_checkpoint_payload_persists_token_names(self) -> None:
        import torch

        model = torch.nn.Linear(1, 1)
        payload = _token_checkpoint_payload(
            model=model,
            feat_mean=np.zeros(10, dtype=np.float32),
            feat_std=np.ones(10, dtype=np.float32),
        )

        self.assertEqual(list(TRAIN_TOKEN_ORDER), payload["token_names"])
        self.assertEqual(len(TRAIN_TOKEN_ORDER), payload["n_tokens"])

    def test_load_dagger_merge_appends_multiple_relabel_pools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d1 = root / "d1"
            d2 = root / "d2"
            d1.mkdir()
            d2.mkdir()
            np.save(d1 / "features.npy", np.array([[10.0, 11.0]], dtype=np.float32))
            np.save(d1 / "token_idx.npy", np.array([3], dtype=np.int64))
            np.save(d1 / "rollout_id.npy", np.array([0], dtype=np.int64))
            (d1 / "meta.json").write_text('{"source_policy":"token_bc"}')
            np.save(d2 / "features.npy", np.array([[20.0, 21.0]], dtype=np.float32))
            np.save(d2 / "token_idx.npy", np.array([4], dtype=np.int64))
            np.save(d2 / "rollout_id.npy", np.array([0], dtype=np.int64))
            (d2 / "meta.json").write_text('{"source_policy":"token_dagger_iter2_bc"}')

            features, token_idx, source_ids, rollout_ids, summaries = _load_dagger_merge(
                np.array([[1.0, 2.0]], dtype=np.float32),
                np.array([0], dtype=np.int64),
                np.array([0], dtype=np.int64),
                [d1, d2],
            )

            np.testing.assert_array_equal(features[:, 0], np.array([1.0, 10.0, 20.0], dtype=np.float32))
            np.testing.assert_array_equal(token_idx, np.array([0, 3, 4], dtype=np.int64))
            np.testing.assert_array_equal(source_ids, np.array([0, 1, 2], dtype=np.int64))
            np.testing.assert_array_equal(rollout_ids, np.array([0, 1, 2], dtype=np.int64))
            self.assertEqual(["token_bc", "token_dagger_iter2_bc"], [row["source_policy"] for row in summaries])
            self.assertEqual([1, 2], [row["source_id"] for row in summaries])

    def test_parse_source_weights(self) -> None:
        self.assertEqual({0: 1.0, 1: 0.5, 3: 0.25}, _parse_source_weights("0:1,1:0.5,3:0.25"))

    def test_classification_loss_applies_source_weights(self) -> None:
        import torch

        logits = torch.tensor([[3.0, 0.0], [0.0, 3.0]], dtype=torch.float32)
        labels = torch.tensor([0, 0], dtype=torch.long)
        source_ids = torch.tensor([0, 1], dtype=torch.long)
        source_weights = torch.tensor([1.0, 0.0], dtype=torch.float32)

        unweighted = _classification_loss(logits, labels)
        weighted = _classification_loss(logits, labels, source_ids=source_ids, source_weights=source_weights)

        self.assertLess(weighted.item(), unweighted.item())

    def test_transformer_history_model_emits_token_logits(self) -> None:
        import torch

        model = GeomTransformerTokenBC(history_len=4)
        x = torch.zeros((3, 4, 10), dtype=torch.float32)
        logits = model(x)

        self.assertEqual((3, 9), tuple(logits.shape))


if __name__ == "__main__":
    unittest.main()
