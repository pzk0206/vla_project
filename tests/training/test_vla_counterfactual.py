"""保护红蓝成对反事实的分母、分组、区间和结论规则。"""

import unittest

import torch

from vla_project.training import vla_counterfactual


def branch(color, *, success=False, nearest=None, system_error=None):
    return {
        "instruction_target": color,
        "success": success,
        "nearest_block": nearest,
        "system_error": system_error,
    }


class PairedSummaryTests(unittest.TestCase):
    def test_excludes_only_system_errors_and_counts_fixed_target_as_failure(self):
        pairs = [
            {
                "episode_idx": 0,
                "original_target": "red",
                "red": branch("red", success=True, nearest="red"),
                "blue": branch("blue", success=True, nearest="blue"),
            },
            {
                "episode_idx": 1,
                "original_target": "blue",
                "red": branch("red", success=True, nearest="red"),
                "blue": branch("blue", success=False, nearest="red"),
            },
            {
                "episode_idx": 2,
                "original_target": "red",
                "red": branch("red", system_error="scene_replay_mismatch"),
                "blue": branch("blue", success=True, nearest="blue"),
            },
        ]

        summary = vla_counterfactual.summarize_pairs(pairs)

        self.assertEqual(summary["num_pairs"], 3)
        self.assertEqual(summary["valid_pair_count"], 2)
        self.assertEqual(summary["excluded_pair_count"], 1)
        self.assertEqual(summary["paired_instruction_follow_count"], 1)
        self.assertEqual(summary["paired_instruction_follow_rate"], 0.5)
        self.assertEqual(summary["preference_switch_count"], 1)
        self.assertEqual(summary["groups"]["red_to_blue"]["valid_count"], 1)
        self.assertEqual(summary["groups"]["blue_to_red"]["valid_count"], 1)
        self.assertFalse(summary["supports_red_blue_instruction_recognition"])

    def test_fifty_following_pairs_support_recognition(self):
        pairs = [
            {
                "episode_idx": index,
                "original_target": "red" if index % 2 == 0 else "blue",
                "red": branch("red", success=True, nearest="red"),
                "blue": branch("blue", success=True, nearest="blue"),
            }
            for index in range(50)
        ]

        summary = vla_counterfactual.summarize_pairs(pairs)

        self.assertEqual(summary["valid_pair_count"], 50)
        self.assertEqual(summary["paired_instruction_follow_rate"], 1.0)
        self.assertGreater(
            summary["paired_instruction_follow_wilson_95"]["lower"],
            0.5,
        )
        self.assertTrue(summary["supports_red_blue_instruction_recognition"])


class InstructionEncodingAuditTests(unittest.TestCase):
    def test_reports_identical_tokens_and_embeddings(self):
        class FakeTextEncoder:
            def tokenize(self, texts):
                return {
                    "input_ids": torch.tensor([[101, 100, 102], [101, 100, 102]]),
                    "attention_mask": torch.ones((2, 3), dtype=torch.int64),
                }

        class FakeModel:
            text_encoder = FakeTextEncoder()

            def encode_texts(self, texts):
                return torch.tensor([[1.0, 2.0], [1.0, 2.0]])

        actual = vla_counterfactual.audit_instruction_encoding(FakeModel())

        self.assertTrue(actual["token_ids_equal"])
        self.assertTrue(actual["embeddings_equal"])
        self.assertEqual(actual["embedding_l2"], 0.0)
        self.assertEqual(actual["input_ids"][0], [101, 100, 102])
        self.assertEqual(len(actual["embedding_sha256"]), 2)


if __name__ == "__main__":
    unittest.main()
