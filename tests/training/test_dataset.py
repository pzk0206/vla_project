"""保护 BC delta 标签只来自同一 episode 的相邻绝对关节目标。"""

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from vla_project.training import dataset


def trajectory_row(episode_idx, step_idx, q_value, image_path="frame.png"):
    return {
        "episode_idx": episode_idx,
        "step_idx": step_idx,
        "image_path": image_path,
        "action": [q_value] * 7 + [1.0, 0],
    }


class DeltaRowTests(unittest.TestCase):
    def test_sorts_and_differences_only_adjacent_rows_within_episode(self):
        rows = [
            trajectory_row(1, 30, 0.3),
            trajectory_row(0, 40, 1.15),
            trajectory_row(0, 10, 1.0),
            trajectory_row(1, 5, -0.1),
            trajectory_row(0, 20, 1.2),
        ]

        actual = dataset._build_delta_rows(rows)

        self.assertEqual(
            [(row["episode_idx"], row["step_idx"]) for row in actual],
            [(0, 20), (0, 40), (1, 30)],
        )
        for value in actual[0]["_delta_q"]:
            self.assertAlmostEqual(value, 0.2)
        for value in actual[1]["_delta_q"]:
            self.assertAlmostEqual(value, -0.05)
        for value in actual[2]["_delta_q"]:
            self.assertAlmostEqual(value, 0.4)

    def test_rejects_duplicate_episode_step_key(self):
        rows = [
            trajectory_row(0, 10, 0.1),
            trajectory_row(0, 10, 0.2),
        ]

        with self.assertRaisesRegex(ValueError, "duplicate trajectory key"):
            dataset._build_delta_rows(rows)

    def test_rejects_action_without_seven_finite_joint_targets(self):
        wrong_length = trajectory_row(0, 10, 0.1)
        wrong_length["action"] = [0.1] * 6
        non_finite = trajectory_row(1, 10, 0.1)
        non_finite["action"][3] = float("nan")

        for row in (wrong_length, non_finite):
            with self.subTest(row=row):
                with self.assertRaisesRegex(ValueError, "invalid joint target"):
                    dataset._build_delta_rows([row])


class BCDatasetDeltaTests(unittest.TestCase):
    def test_encodes_true_delta_instead_of_absolute_target(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "frame.png"
            Image.new("RGB", (8, 8), color=(10, 20, 30)).save(image_path)
            (root / "episode_split.json").write_text(
                json.dumps({"train": [0], "val": [0]}),
                encoding="utf-8",
            )
            rows = [
                trajectory_row(0, 10, 0.8, str(image_path)),
                trajectory_row(0, 20, 0.6, str(image_path)),
            ]
            (root / "trajectory_expert.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            tokenizer_path = root / "audit.json"
            edges = [
                round(-1.0 + index / 32.0, 8)
                for index in range(65)
            ]
            reconstruction_values = [
                (edges[index] + edges[index + 1]) / 2.0
                for index in range(64)
            ]
            tokenizer_path.write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "representation": "delta_q",
                                "binning": "quantile",
                                "num_bins": 64,
                                "per_joint": [
                                    {
                                        "edges": edges,
                                        "reconstruction_values": (
                                            reconstruction_values
                                        ),
                                    }
                                    for _ in range(7)
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            actual = dataset.BCDataset(
                dataset_dir=root,
                split="train",
                action_representation="delta_q_64",
                tokenizer_audit_path=tokenizer_path,
            )
            _, tokens, auxiliary = actual[0]

        self.assertEqual(len(actual), 1)
        self.assertEqual(tokens.tolist(), [25] * 7)
        self.assertEqual(auxiliary.tolist(), [1.0, 0.0])


class VlaV2SemanticTests(unittest.TestCase):
    def test_rejects_instruction_that_disagrees_with_scene_target(self):
        row = trajectory_row(0, 0, 0.1)
        row.update(
            {
                "schema_version": "expert_multi_v2",
                "instruction": "悬停在蓝色积木上方",
                "scene_state": {"target_block": "red"},
            }
        )

        with self.assertRaisesRegex(ValueError, "instruction.*target_block"):
            dataset._validate_vla_v2_rows([row])

    def test_accepts_exact_red_and_blue_instruction_mapping(self):
        rows = []
        for episode_idx, color in enumerate(("red", "blue")):
            row = trajectory_row(episode_idx, 0, 0.1)
            row.update(
                {
                    "schema_version": "expert_multi_v2",
                    "instruction": f"悬停在{'红' if color == 'red' else '蓝'}色积木上方",
                    "scene_state": {"target_block": color},
                }
            )
            rows.append(row)

        dataset._validate_vla_v2_rows(rows)


if __name__ == "__main__":
    unittest.main()
