"""测试固定 grounding 偏差只能由校准集拟合并只读应用到验证集。"""

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vla_project.vlm.validate_grounding_calibration import (
    apply_frozen_calibration,
    fit_clear_calibration,
    run_validation,
    summarize_validation,
    write_outputs,
)


class OutputBoundaryTests(unittest.TestCase):
    def test_rejects_output_escape_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "vla_project.vlm.validate_grounding_calibration.read_jsonl"
        ) as read:
            with self.assertRaisesRegex(ValueError, "parent traversal"):
                run_validation(
                    "calibration.jsonl",
                    "validation.jsonl",
                    "outputs/vlm_evaluations/../../src",
                    project_root_override=temp_dir,
                )
            read.assert_not_called()


def result_row(
    sample_id,
    visibility_group="clear",
    predicted=(0.18, 0.42, 0.0),
    truth=(0.20, 0.40, 0.05),
    error_type=None,
):
    """构造一条与反投影脚本一致的最小结果。"""
    if predicted is None:
        return {
            "sample_id": sample_id,
            "visibility_group": visibility_group,
            "predicted_target_world": None,
            "true_block_pos": list(truth),
            "localization_error_xy": None,
            "signed_error_x": None,
            "signed_error_y": None,
            "error_type": error_type or "MissingGroundingBoxError",
            "error_message": "red_block box 缺失",
        }
    signed_x = predicted[0] - truth[0]
    signed_y = predicted[1] - truth[1]
    return {
        "sample_id": sample_id,
        "visibility_group": visibility_group,
        "predicted_target_world": list(predicted),
        "true_block_pos": list(truth),
        "localization_error_xy": math.hypot(signed_x, signed_y),
        "signed_error_x": signed_x,
        "signed_error_y": signed_y,
        "error_type": None,
        "error_message": None,
    }


class FitClearCalibrationTests(unittest.TestCase):
    def test_fits_only_clear_rows_and_uses_opposite_signed_mean(self):
        rows = [
            result_row("seed_42_left", predicted=(0.18, 0.43, 0.0)),
            result_row("seed_43_right", predicted=(0.16, 0.41, 0.0)),
            result_row(
                "seed_44_back",
                visibility_group="severe",
                predicted=(0.50, 0.10, 0.0),
            ),
        ]

        calibration = fit_clear_calibration(rows, "calibration.jsonl")

        self.assertEqual(calibration["calibration_source"], "calibration.jsonl")
        self.assertEqual(calibration["num_clear_calibration_samples"], 2)
        self.assertEqual(
            calibration["calibration_sample_ids"],
            ["seed_42_left", "seed_43_right"],
        )
        self.assertAlmostEqual(calibration["signed_error_x_mean"], -0.03)
        self.assertAlmostEqual(calibration["signed_error_y_mean"], 0.02)
        self.assertAlmostEqual(calibration["correction_x"], 0.03)
        self.assertAlmostEqual(calibration["correction_y"], -0.02)

    def test_rejects_empty_or_invalid_clear_calibration(self):
        with self.assertRaisesRegex(ValueError, "clear 校准样本不能为空"):
            fit_clear_calibration([], "empty.jsonl")
        with self.assertRaisesRegex(ValueError, "clear 校准样本定位无效"):
            fit_clear_calibration(
                [result_row("seed_42_left", predicted=None)],
                "invalid.jsonl",
            )

    def test_rejects_missing_truth_in_calibration_input(self):
        row = result_row("seed_42_left")
        row["true_block_pos"] = None
        with self.assertRaisesRegex(ValueError, "true_block_pos 非法"):
            fit_clear_calibration([row], "invalid-truth.jsonl")


class ApplyFrozenCalibrationTests(unittest.TestCase):
    def test_applies_frozen_xy_correction_without_changing_z_or_raw_fields(self):
        calibration = {
            "calibration_source": "old.jsonl",
            "calibration_sample_ids": ["seed_42_left"],
            "correction_x": 0.03,
            "correction_y": -0.02,
        }
        validation = [
            result_row(
                "seed_47_left",
                predicted=(0.17, 0.42, 0.0),
                truth=(0.20, 0.40, 0.05),
            )
        ]

        corrected = apply_frozen_calibration(calibration, validation)

        self.assertEqual(
            corrected[0]["predicted_target_world"],
            [0.17, 0.42, 0.0],
        )
        for actual, expected in zip(
            corrected[0]["corrected_target_world"],
            [0.20, 0.40, 0.0],
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(corrected[0]["corrected_error_xy"], 0.0)
        self.assertAlmostEqual(corrected[0]["correction_x_applied"], 0.03)
        self.assertAlmostEqual(corrected[0]["correction_y_applied"], -0.02)

    def test_rejects_overlapping_calibration_and_validation_ids(self):
        calibration = {
            "calibration_sample_ids": ["seed_42_left"],
            "correction_x": 0.03,
            "correction_y": -0.02,
        }
        with self.assertRaisesRegex(ValueError, "sample_id 重叠"):
            apply_frozen_calibration(
                calibration,
                [result_row("seed_42_left")],
            )

    def test_preserves_failed_validation_as_failed(self):
        calibration = {
            "calibration_sample_ids": ["seed_42_left"],
            "correction_x": 0.03,
            "correction_y": -0.02,
        }
        corrected = apply_frozen_calibration(
            calibration,
            [result_row("seed_47_left", predicted=None)],
        )
        self.assertIsNone(corrected[0]["corrected_target_world"])
        self.assertIsNone(corrected[0]["corrected_error_xy"])

    def test_rejects_missing_truth_even_when_prediction_failed(self):
        calibration = {
            "calibration_sample_ids": ["seed_42_left"],
            "correction_x": 0.03,
            "correction_y": -0.02,
        }
        row = result_row("seed_47_left", predicted=None)
        row["true_block_pos"] = None
        with self.assertRaisesRegex(ValueError, "true_block_pos 非法"):
            apply_frozen_calibration(calibration, [row])


class SummarizeValidationTests(unittest.TestCase):
    def calibration(self):
        return {
            "calibration_source": "old.jsonl",
            "num_clear_calibration_samples": 15,
            "calibration_sample_ids": ["seed_42_left"],
            "signed_error_x_mean": -0.03,
            "signed_error_y_mean": 0.02,
            "correction_x": 0.03,
            "correction_y": -0.02,
        }

    def corrected(self, rows):
        return apply_frozen_calibration(self.calibration(), rows)

    def test_passes_only_when_every_clear_sample_is_valid_and_within_3cm(self):
        rows = self.corrected(
            [
                result_row("seed_47_left", predicted=(0.17, 0.42, 0.0)),
                result_row("seed_47_right", predicted=(0.19, 0.42, 0.0)),
                result_row(
                    "seed_47_back",
                    visibility_group="severe",
                    predicted=(0.50, 0.10, 0.0),
                ),
            ]
        )

        summary = summarize_validation(rows, self.calibration())

        self.assertTrue(summary["passed"])
        self.assertEqual(summary["num_validation_samples"], 3)
        self.assertEqual(summary["overall"]["num_samples"], 3)
        clear = summary["per_visibility_group"]["clear"]
        self.assertEqual(clear["num_within_3cm"], 2)
        self.assertEqual(clear["within_3cm_rate"], 1.0)

    def test_fails_for_clear_error_over_3cm(self):
        rows = self.corrected(
            [result_row("seed_47_left", predicted=(0.10, 0.42, 0.0))]
        )

        summary = summarize_validation(rows, self.calibration())

        self.assertFalse(summary["passed"])
        self.assertIn(
            "clear 样本补偿后误差超过 0.03m",
            summary["failure_reasons"],
        )

    def test_fails_for_missing_or_invalid_clear_samples(self):
        no_clear = self.corrected(
            [result_row("seed_47_back", visibility_group="severe")]
        )
        invalid_clear = self.corrected(
            [result_row("seed_47_left", predicted=None)]
        )

        self.assertFalse(
            summarize_validation(no_clear, self.calibration())["passed"]
        )
        self.assertFalse(
            summarize_validation(invalid_clear, self.calibration())["passed"]
        )

    def test_writes_three_traceable_output_files(self):
        rows = self.corrected(
            [result_row("seed_47_left", predicted=(0.17, 0.42, 0.0))]
        )
        summary = summarize_validation(rows, self.calibration())

        with tempfile.TemporaryDirectory() as temp_dir:
            write_outputs(temp_dir, self.calibration(), rows, summary)
            output = Path(temp_dir)
            self.assertTrue((output / "calibration.json").is_file())
            self.assertTrue(
                (output / "corrected_validation_results.jsonl").is_file()
            )
            self.assertTrue(
                (output / "calibration_validation_summary.json").is_file()
            )
            saved = json.loads(
                (output / "calibration_validation_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(saved["pass_threshold_meters"], 0.03)


if __name__ == "__main__":
    unittest.main()
