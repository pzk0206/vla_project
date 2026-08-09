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
