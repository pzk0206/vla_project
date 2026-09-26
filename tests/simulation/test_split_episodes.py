"""保护 expert_multi_v2 的红蓝平衡、场景语义与划分证据。"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from vla_project.simulation import split_episodes


def _make_v2_rows():
    rows = []
    for episode_idx in range(300):
        target_block = "red" if episode_idx % 2 == 0 else "blue"
        red_position = [
            -0.19 + 0.38 * ((episode_idx * 37) % 300) / 299,
            0.381 + 0.118 * ((episode_idx * 71) % 300) / 299,
            0.05,
        ]
        blue_position = [
            -0.19 + 0.38 * ((episode_idx * 97 + 11) % 300) / 299,
            0.381 + 0.118 * ((episode_idx * 43 + 17) % 300) / 299,
            0.05,
        ]
        target_position = red_position if target_block == "red" else blue_position
        rows.append(
            {
                "schema_version": "expert_multi_v2",
                "episode_idx": episode_idx,
                "random_seed": 1000 + episode_idx,
                "target_block": target_block,
                "task_instruction": f"悬停在{'红' if target_block == 'red' else '蓝'}色积木上方",
                "initial_block_pos": target_position,
                "initial_scene_state": {
                    "target_block": target_block,
                    "blocks": {
                        "red": {"position": red_position},
                        "blue": {"position": blue_position},
                    },
                },
            }
        )
    return rows


def _make_paired_rows():
    """构造成对数据：300 原始场景 + 300 反色补录，共 600 条。

    原始轨迹 is_paired_copy=False；补录轨迹 episode_idx=300+i、
    paired_with=i、is_paired_copy=True，target_block 为反色。
    """
    rows = []
    for episode_idx in range(300):
        original_target = "red" if episode_idx % 2 == 0 else "blue"
        paired_target = "blue" if original_target == "red" else "red"
        red_position = [
            -0.19 + 0.38 * ((episode_idx * 37) % 300) / 299,
            0.381 + 0.118 * ((episode_idx * 71) % 300) / 299,
            0.05,
        ]
        blue_position = [
            -0.19 + 0.38 * ((episode_idx * 97 + 11) % 300) / 299,
            0.381 + 0.118 * ((episode_idx * 43 + 17) % 300) / 299,
            0.05,
        ]

        def scene_state(target_block):
            return {
                "target_block": target_block,
                "blocks": {
                    "red": {"position": red_position},
                    "blue": {"position": blue_position},
                },
            }

        original_position = (
            red_position if original_target == "red" else blue_position
        )
        paired_position = (
            red_position if paired_target == "red" else blue_position
        )
        rows.append(
            {
                "schema_version": "expert_multi_v2",
                "episode_idx": episode_idx,
                "random_seed": 1000 + episode_idx,
                "target_block": original_target,
                "task_instruction": (
                    f"悬停在{'红' if original_target == 'red' else '蓝'}色积木上方"
                ),
                "initial_block_pos": original_position,
                "initial_scene_state": scene_state(original_target),
                "is_paired_copy": False,
            }
        )
        rows.append(
            {
                "schema_version": "expert_multi_v2",
                "episode_idx": 300 + episode_idx,
                "random_seed": 1000 + 300 + episode_idx,
                "target_block": paired_target,
                "task_instruction": (
                    f"悬停在{'红' if paired_target == 'red' else '蓝'}色积木上方"
                ),
                "initial_block_pos": paired_position,
                "initial_scene_state": scene_state(paired_target),
                "is_paired_copy": True,
                "paired_with": episode_idx,
            }
        )
    return rows


class PairedSplitTests(unittest.TestCase):
    """成对数据以场景为单位分层，保证同场景双指令不跨划分。"""

    def test_paired_split_keeps_scene_pairs_together(self):
        rows = _make_paired_rows()

        result = split_episodes._stratified_split(rows, is_paired=True)
        by_id = {row["episode_idx"]: row for row in rows}

        self.assertEqual(len(result["train"]), 500)
        self.assertEqual(len(result["val"]), 100)
        self.assertEqual(set(result["train"]) & set(result["val"]), set())
        self.assertEqual(
            set(result["train"]) | set(result["val"]), set(range(600))
        )
        # 每个原始场景与它的补录伴侣必须同 split
        for scene_index in result["scene_train"]:
            self.assertIn(scene_index, result["train"])
            self.assertIn(300 + scene_index, result["train"])
            self.assertNotIn(scene_index, result["val"])
            self.assertNotIn(300 + scene_index, result["val"])
        for scene_index in result["scene_val"]:
            self.assertIn(scene_index, result["val"])
            self.assertIn(300 + scene_index, result["val"])
        # 红蓝平衡：train 红250/蓝250，val 红50/蓝50
        train_counts = {
            color: sum(1 for i in result["train"] if by_id[i]["target_block"] == color)
            for color in ("red", "blue")
        }
        val_counts = {
            color: sum(1 for i in result["val"] if by_id[i]["target_block"] == color)
            for color in ("red", "blue")
        }
        self.assertEqual(train_counts, {"red": 250, "blue": 250})
        self.assertEqual(val_counts, {"red": 50, "blue": 50})

    def test_paired_detection_from_rows(self):
        rows = _make_paired_rows()
        self.assertTrue(split_episodes._detect_paired_dataset(rows))
        rows_v2 = _make_v2_rows()
        self.assertFalse(split_episodes._detect_paired_dataset(rows_v2))

    def test_run_split_records_v3_paired_provenance(self):
        rows = _make_paired_rows()
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            summary_path = dataset_dir / "episode_summary.jsonl"
            summary_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            manifest_path = dataset_dir / "dataset_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": "expert_multi_v2",
                        "paired_source_dir": "outputs/dataset/expert_multi_v2",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = split_episodes.run_split(dataset_dir)

        self.assertEqual(result["schema_version"], "episode_split_v3")
        self.assertEqual(result["train_count"], 500)
        self.assertEqual(result["val_count"], 100)
        self.assertEqual(result["task_counts"]["train"], {"red": 250, "blue": 250})
        self.assertEqual(result["task_counts"]["val"], {"red": 50, "blue": 50})
        self.assertEqual(
            result["stratification"]["method"],
            "paired_scene_then_target_block_then_quantile_5_bin_xy",
        )
        self.assertEqual(len(result["stratification"]["paired"]["scene_train"]), 250)
        self.assertEqual(len(result["stratification"]["paired"]["scene_val"]), 50)


class ExpertMultiV2SplitTests(unittest.TestCase):
    def test_split_balances_each_target_between_train_and_val(self):
        rows = _make_v2_rows()

        result = split_episodes._stratified_split(rows)
        by_id = {row["episode_idx"]: row for row in rows}

        self.assertEqual(len(result["train"]), 250)
        self.assertEqual(len(result["val"]), 50)
        self.assertEqual(
            [
                sum(by_id[index]["target_block"] == color for index in result["train"])
                for color in ("red", "blue")
            ],
            [125, 125],
        )
        self.assertEqual(
            [
                sum(by_id[index]["target_block"] == color for index in result["val"])
                for color in ("red", "blue")
            ],
            [25, 25],
        )

    def test_run_split_records_v2_scene_and_hash_provenance(self):
        rows = _make_v2_rows()
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            summary_path = dataset_dir / "episode_summary.jsonl"
            summary_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            manifest_path = dataset_dir / "dataset_manifest.json"
            manifest_path.write_text(
                json.dumps({"schema_version": "expert_multi_v2"}) + "\n",
                encoding="utf-8",
            )
            expected_summary_hash = hashlib.sha256(summary_path.read_bytes()).hexdigest()
            expected_manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

            result = split_episodes.run_split(dataset_dir)

        self.assertEqual(result["schema_version"], "episode_split_v2")
        self.assertEqual(result["task_counts"]["train"], {"red": 125, "blue": 125})
        self.assertEqual(result["task_counts"]["val"], {"red": 25, "blue": 25})
        self.assertEqual(
            result["stratification"]["fields"],
            ["target_block", "initial_scene_state.blocks[target_block].position.x/y"],
        )
        self.assertEqual(
            result["provenance"]["source_sha256"],
            expected_summary_hash,
        )
        self.assertEqual(
            result["provenance"]["manifest_sha256"],
            expected_manifest_hash,
        )


if __name__ == "__main__":
    unittest.main()
