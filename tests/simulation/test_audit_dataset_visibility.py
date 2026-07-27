"""测试 expert_v1 数据集的确定性重放可见性审计。"""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from vla_project.simulation.audit_dataset_visibility import (
    ReplayValidationError,
    build_visibility_rows,
    count_block_pixels,
    summarize_visibility,
    validate_replay_image,
)


class VisibilityMathTests(unittest.TestCase):
    def test_counts_block_object_id_while_ignoring_link_bits_and_background(self):
        block_id = 2
        segmentation = np.array(
            [
                [-1, block_id],
                [block_id | (3 << 24), 7],
            ],
            dtype=np.int64,
        )

        self.assertEqual(count_block_pixels(segmentation, block_id), 2)


class ReplayImageTests(unittest.TestCase):
    def test_accepts_same_image_after_default_jpeg_roundtrip(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[8:24, 8:24] = [0, 0, 255]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "frame.jpg"
            self.assertTrue(cv2.imwrite(str(path), image))

            result = validate_replay_image(path, image)

        self.assertTrue(result["replay_exact_match"])
        self.assertEqual(result["replay_pixel_mae"], 0.0)

    def test_rejects_replay_with_changed_pixels(self):
        original = np.zeros((32, 32, 3), dtype=np.uint8)
        replay = np.full((32, 32, 3), 255, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "frame.jpg"
            self.assertTrue(cv2.imwrite(str(path), original))

            with self.assertRaisesRegex(
                ReplayValidationError,
                "replay_image_mismatch",
            ):
                validate_replay_image(path, replay)


def observation(step_idx, pixels):
    return {
        "schema_version": "visibility_audit_v1",
        "episode_idx": 3,
        "random_seed": 1003,
        "step_idx": step_idx,
        "image_path": f"ep_3_step_{step_idx}.jpg",
        "visible_block_pixels": pixels,
        "replay_pixel_mae": 0.0,
        "replay_exact_match": True,
    }


class VisibilityRowTests(unittest.TestCase):
    def test_builds_rows_with_existing_visibility_boundaries(self):
        rows = build_visibility_rows(
            [
                observation(0, 0),
                observation(1, 25),
                observation(2, 75),
                observation(3, 100),
            ],
            reference_pixels=100,
        )

        self.assertEqual(
            [row["visibility_group"] for row in rows],
            ["severe", "partial", "clear", "clear"],
        )
        self.assertEqual(rows[1]["block_visibility_ratio"], 0.25)

    def test_rejects_nonpositive_reference_or_visible_above_reference(self):
        with self.assertRaises(ValueError):
            build_visibility_rows([observation(0, 0)], reference_pixels=0)
        with self.assertRaises(ValueError):
            build_visibility_rows(
                [observation(0, 101)],
                reference_pixels=100,
            )


class VisibilitySummaryTests(unittest.TestCase):
    def test_summarizes_groups_endpoints_and_longest_nonclear_run(self):
        rows = [
            {
                **observation(0, 100),
                "block_visibility_ratio": 1.0,
                "visibility_group": "clear",
                "reference_block_pixels": 100,
            },
            {
                **observation(24, 70),
                "block_visibility_ratio": 0.7,
                "visibility_group": "partial",
                "reference_block_pixels": 100,
            },
            {
                **observation(48, 20),
                "block_visibility_ratio": 0.2,
                "visibility_group": "severe",
                "reference_block_pixels": 100,
            },
            {
                **observation(72, 80),
                "block_visibility_ratio": 0.8,
                "visibility_group": "clear",
                "reference_block_pixels": 100,
            },
        ]

        summary = summarize_visibility(rows)

        self.assertEqual(
            summary["visibility_group_counts"],
            {"clear": 2, "partial": 1, "severe": 1},
        )
        self.assertEqual(summary["episodes_with_nonclear"], 1)
        self.assertEqual(summary["episodes_with_severe"], 1)
        self.assertEqual(
            summary["initial_visibility_group_counts"],
            {"clear": 1},
        )
        self.assertEqual(
            summary["terminal_visibility_group_counts"],
            {"clear": 1},
        )
        self.assertEqual(summary["longest_nonclear_saved_frame_run"], 2)
        self.assertEqual(
            summary["longest_nonclear_run"],
            {
                "episode_idx": 3,
                "start_step": 24,
                "end_step": 48,
                "num_frames": 2,
            },
        )
        self.assertEqual(
            summary["per_episode"][0]["longest_nonclear_run"],
            summary["longest_nonclear_run"],
        )


if __name__ == "__main__":
    unittest.main()
