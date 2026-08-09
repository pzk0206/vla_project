"""保护 rollout 拒绝旧 delta checkpoint 并执行冻结的 v2 语义。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from vla_project.training import rollout


def tokenizer_asset():
    return {
        "representation": "delta_q",
        "binning": "quantile",
        "requested_num_bins": 64,
        "edges": [[float(i) for i in range(65)] for _ in range(7)],
        "reconstruction_values": [
            [0.01 * (joint + 1) + 0.001 * token for token in range(64)]
            for joint in range(7)
        ],
    }


def delta_metadata():
    return {
        "action_semantics": "same_episode_saved_target_delta_v2",
        "training_split_sha256": "d" * 64,
        "tokenizer": tokenizer_asset(),
    }


class DeltaCheckpointContractTests(unittest.TestCase):
    def test_rejects_legacy_delta_checkpoint_before_model_construction(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "checkpoint.pt"
            torch.save(
                {
                    "action_representation": "delta_q_64",
                    "model_state_dict": {},
                    "metadata": {"val_loss": 1.0},
                },
                checkpoint,
            )
            with patch.object(rollout, "BCModel") as model:
                with self.assertRaisesRegex(
                    ValueError,
                    "invalid_reason=absolute_labels_encoded_as_delta",
                ):
                    rollout.load_model_for_rollout(checkpoint, "cpu")
                model.assert_not_called()

    def test_loads_complete_delta_v2_action_assets(self):
        metadata = delta_metadata()

        actual = rollout._load_checkpoint_action_assets(
            metadata,
            "delta_q_64",
        )

        self.assertEqual(actual, metadata["tokenizer"])

    def test_rejects_delta_metadata_without_split_hash_or_tokenizer(self):
        missing_split = delta_metadata()
        del missing_split["training_split_sha256"]
        missing_tokenizer = delta_metadata()
        del missing_tokenizer["tokenizer"]

        for metadata in (missing_split, missing_tokenizer):
            with self.subTest(metadata=metadata):
                with self.assertRaisesRegex(ValueError, "invalid delta checkpoint"):
                    rollout._load_checkpoint_action_assets(
                        metadata,
                        "delta_q_64",
                    )


class DeltaExecutionTests(unittest.TestCase):
    def test_adds_audited_decoded_delta_to_current_joint_state(self):
        tokenizer = tokenizer_asset()
        token_ids = np.asarray([0, 1, 2, 3, 4, 5, 6], dtype=np.int64)
        current_q = np.asarray([1.0, -1.0, 0.5, 0.0, 2.0, -2.0, 0.25])

        actual = rollout._classification_joint_targets(
            "delta_q_64",
            token_ids,
            tokenizer,
            current_q=current_q,
        )

        expected_delta = np.asarray(
            [0.01, 0.021, 0.032, 0.043, 0.054, 0.065, 0.076]
        )
        np.testing.assert_allclose(actual, current_q + expected_delta)

    def test_absolute_classification_does_not_add_current_state(self):
        tokenizer = tokenizer_asset()
        tokenizer["representation"] = "absolute_q"
        token_ids = np.zeros(7, dtype=np.int64)

        actual = rollout._classification_joint_targets(
            "absolute_q_32",
            token_ids,
            tokenizer,
            current_q=np.ones(7),
        )

        np.testing.assert_allclose(
            actual,
            [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
        )


if __name__ == "__main__":
    unittest.main()
