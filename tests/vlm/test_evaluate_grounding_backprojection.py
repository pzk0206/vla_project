"""测试 grounding 框反投影的离线连接、错误隔离和指标统计。"""

import unittest

import numpy as np

from vla_project.simulation.camera_geometry import compute_camera_matrices, world_to_pixel
from vla_project.vlm.evaluate_grounding_backprojection import (
    evaluate_rows,
    summarize_results,
)


class GroundingBackprojectionEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = {
            "workspace_center": [0.0, 0.4, 0.0],
            "up_vector": [0.0, 1.0, 0.0],
            "image_width": 448,
            "image_height": 448,
            "fov": 45,
            "near_val": 0.1,
            "far_val": 100.0,
        }
        cls.view, cls.projection = compute_camera_matrices(
            config, [0.0, 0.4, 3.0]
        )

    def diagnostic(self, sample_id, block_pos, visibility_ratio=None):
        diagnostic = {
            "sample_id": sample_id,
            "block_pos": block_pos,
            "camera_eye": [0.0, 0.4, 3.0],
            "image_width": 448,
            "image_height": 448,
            "view_matrix": self.view,
            "projection_matrix": self.projection,
        }
        if visibility_ratio is not None:
            diagnostic.update(
                {
                    "block_visible_pixels": round(100 * visibility_ratio),
                    "block_reference_pixels": 100,
                    "block_visibility_ratio": visibility_ratio,
                }
            )
        return diagnostic

    def box_around_world_point(self, world):
        pixel_x, pixel_y = world_to_pixel(
            world, 448, 448, self.view, self.projection
        )
        normalized_x = pixel_x / 447.0 * 1000.0
        normalized_y = pixel_y / 447.0 * 1000.0
        return [
            normalized_x - 5.0,
            normalized_y - 5.0,
            normalized_x + 5.0,
            normalized_y + 5.0,
        ]

    def test_computes_zero_xy_error_for_matching_box_center(self):
        target = [0.08, 0.47, 0.0]
        predictions = [
            {
                "sample_id": "sample-1",
                "boxes": {"red_block": self.box_around_world_point(target)},
                "expected_direction": "left",
                "random_seed": 42,
                "latency_seconds": 1.2,
                "error_type": None,
                "error_message": None,
            }
        ]
        diagnostics = [
            self.diagnostic(
                "sample-1",
                [0.08, 0.47, 0.05],
                visibility_ratio=0.8,
            )
        ]

        result = evaluate_rows(predictions, diagnostics)[0]

        np.testing.assert_allclose(
            result["predicted_target_world"], target, atol=1e-6
        )
        self.assertAlmostEqual(result["localization_error_xy"], 0.0, places=6)
        self.assertEqual(result["expected_direction"], "left")
        self.assertEqual(result["random_seed"], 42)
        self.assertEqual(result["visibility_group"], "clear")
        self.assertIsNone(result["error_type"])

    def test_keeps_invalid_grounding_as_sample_error(self):
        predictions = [
            {
                "sample_id": "sample-1",
                "boxes": None,
                "error_type": "InvalidModelResponseError",
                "error_message": "missing boxes",
            }
        ]
        diagnostics = [self.diagnostic("sample-1", [0.08, 0.47, 0.05])]

        result = evaluate_rows(predictions, diagnostics)[0]

        self.assertEqual(result["error_type"], "InvalidModelResponseError")
        self.assertIsNone(result["predicted_target_world"])

    def test_rejects_duplicate_or_missing_diagnostics_before_evaluation(self):
        prediction = {"sample_id": "sample-1", "boxes": None}
        duplicate = self.diagnostic("sample-1", [0.0, 0.4, 0.05])

        with self.assertRaisesRegex(ValueError, "重复"):
            evaluate_rows([prediction], [duplicate, duplicate])
        with self.assertRaisesRegex(ValueError, "缺少诊断"):
            evaluate_rows([prediction], [])

    def test_summarizes_valid_errors_and_failure_types(self):
        results = [
            {
                "expected_direction": "left",
                "random_seed": 42,
                "box_center_pixel": [250.0, 190.0],
                "visibility_group": "clear",
                "localization_error_xy": 0.01,
                "signed_error_x": 0.006,
                "signed_error_y": -0.008,
                "error_type": None,
            },
            {
                "expected_direction": "right",
                "random_seed": 42,
                "box_center_pixel": [254.0, 193.0],
                "visibility_group": "partial",
                "localization_error_xy": 0.03,
                "signed_error_x": -0.018,
                "signed_error_y": 0.024,
                "error_type": None,
            },
            {
                "expected_direction": "left",
                "random_seed": 43,
                "box_center_pixel": None,
                "visibility_group": "severe",
                "localization_error_xy": None,
                "signed_error_x": None,
                "signed_error_y": None,
                "error_type": "InvalidModelResponseError",
            },
        ]

        summary = summarize_results(results)

        self.assertEqual(summary["num_samples"], 3)
        self.assertEqual(summary["num_valid"], 2)
        self.assertEqual(summary["num_failed"], 1)
        self.assertAlmostEqual(summary["localization_error_xy_mean"], 0.02)
        self.assertAlmostEqual(summary["localization_error_xy_median"], 0.02)
        self.assertAlmostEqual(summary["localization_error_xy_max"], 0.03)
        self.assertEqual(
            summary["error_type_counts"], {"InvalidModelResponseError": 1}
        )

    def test_summarizes_by_direction_and_seed_with_pixel_jitter(self):
        results = [
            {
                "expected_direction": "left",
                "random_seed": 42,
                "box_center_pixel": [250.0, 190.0],
                "visibility_group": "clear",
                "localization_error_xy": 0.01,
                "signed_error_x": -0.006,
                "signed_error_y": 0.008,
                "error_type": None,
            },
            {
                "expected_direction": "right",
                "random_seed": 42,
                "box_center_pixel": [254.0, 193.0],
                "visibility_group": "partial",
                "localization_error_xy": 0.03,
                "signed_error_x": -0.018,
                "signed_error_y": 0.024,
                "error_type": None,
            },
            {
                "expected_direction": "left",
                "random_seed": 43,
                "box_center_pixel": None,
                "visibility_group": "severe",
                "localization_error_xy": None,
                "signed_error_x": None,
                "signed_error_y": None,
                "error_type": "InvalidModelResponseError",
            },
        ]

        summary = summarize_results(results)

        self.assertEqual(summary["per_direction"]["left"]["num_samples"], 2)
        self.assertEqual(summary["per_direction"]["left"]["num_failed"], 1)
        self.assertAlmostEqual(
            summary["per_random_seed"]["42"]["box_center_pixel_span_x"],
            4.0,
        )
        self.assertAlmostEqual(
            summary["per_random_seed"]["42"]["box_center_pixel_span_y"],
            3.0,
        )
        self.assertAlmostEqual(
            summary["per_random_seed"]["42"]
            ["box_center_pixel_max_distance"],
            5.0,
        )
        self.assertEqual(
            summary["per_visibility_group"]["clear"]["num_valid"],
            1,
        )
        self.assertEqual(
            summary["per_visibility_group"]["severe"]["num_failed"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
