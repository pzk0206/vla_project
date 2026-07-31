"""保护 expert_v1 动作表示只读审计契约。"""

import importlib
import math
import unittest
import warnings

from vla_project.simulation.audit_action_tokenization import (
    AuditValidationError,
    build_frame_analysis,
)


def action_row(episode_idx, step_idx, q, terminate=0):
    return {
        "schema_version": "expert_v1",
        "episode_idx": episode_idx,
        "step_idx": step_idx,
        "action": list(q) + [1.0, terminate],
    }


def visibility_row(episode_idx, step_idx, group="clear"):
    return {
        "episode_idx": episode_idx,
        "step_idx": step_idx,
        "visibility_group": group,
    }


class ActionInputTests(unittest.TestCase):
    def test_builds_frame_analysis_without_cross_episode_delta(self):
        try:
            module = importlib.import_module(
                "vla_project.simulation.audit_action_tokenization"
            )
        except ModuleNotFoundError:
            self.fail("action tokenization 审计模块尚未实现")

        manifest = {"schema_version": "expert_v1", "action_dim": 9}
        trajectories = [
            action_row(0, 0, [0.0] * 7),
            action_row(0, 24, [0.24] * 7, terminate=1),
            action_row(1, 0, [1.0] * 7),
            action_row(1, 12, [0.88] * 7, terminate=1),
        ]
        visibility = [
            visibility_row(0, 0),
            visibility_row(0, 24, "partial"),
            visibility_row(1, 0),
            visibility_row(1, 12, "severe"),
        ]

        rows = module.build_frame_analysis(
            manifest,
            trajectories,
            visibility,
        )

        self.assertIsNone(rows[0]["delta_q"])
        self.assertEqual(rows[1]["step_gap"], 24)
        self.assertEqual(rows[1]["delta_q"], [0.24] * 7)
        for value in rows[1]["delta_q_per_step"]:
            self.assertAlmostEqual(value, 0.01)
        self.assertIsNone(rows[2]["delta_q"])
        for value in rows[3]["delta_q"]:
            self.assertAlmostEqual(value, -0.12)
        self.assertEqual(rows[3]["visibility_group"], "severe")

    def test_rejects_unsupported_dataset_schema(self):
        manifest = {"schema_version": "expert_v2", "action_dim": 9}
        trajectories = [action_row(0, 0, [0.0] * 7, terminate=1)]
        visibility = [visibility_row(0, 0)]

        with self.assertRaises(AuditValidationError) as context:
            build_frame_analysis(manifest, trajectories, visibility)

        self.assertEqual(
            context.exception.reason,
            "unsupported_dataset_schema",
        )

    def test_rejects_unexpected_action_dimension(self):
        manifest = {"schema_version": "expert_v1", "action_dim": 8}
        trajectories = [action_row(0, 0, [0.0] * 7, terminate=1)]
        visibility = [visibility_row(0, 0)]

        with self.assertRaises(AuditValidationError) as context:
            build_frame_analysis(manifest, trajectories, visibility)

        self.assertEqual(context.exception.reason, "unexpected_action_dim")

    def test_rejects_duplicate_trajectory_key(self):
        manifest = {"schema_version": "expert_v1", "action_dim": 9}
        row = action_row(0, 0, [0.0] * 7, terminate=1)
        visibility = [visibility_row(0, 0)]

        with self.assertRaises(AuditValidationError) as context:
            build_frame_analysis(manifest, [row, dict(row)], visibility)

        self.assertEqual(context.exception.reason, "duplicate_trajectory_key")

    def test_rejects_duplicate_visibility_key(self):
        manifest = {"schema_version": "expert_v1", "action_dim": 9}
        trajectories = [action_row(0, 0, [0.0] * 7, terminate=1)]
        row = visibility_row(0, 0)

        with self.assertRaises(AuditValidationError) as context:
            build_frame_analysis(manifest, trajectories, [row, dict(row)])

        self.assertEqual(context.exception.reason, "duplicate_visibility_key")

    def test_rejects_missing_visibility_key(self):
        manifest = {"schema_version": "expert_v1", "action_dim": 9}
        trajectories = [action_row(0, 0, [0.0] * 7, terminate=1)]

        with self.assertRaises(AuditValidationError) as context:
            build_frame_analysis(manifest, trajectories, [])

        self.assertEqual(context.exception.reason, "missing_visibility_key")

    def test_rejects_extra_visibility_key(self):
        manifest = {"schema_version": "expert_v1", "action_dim": 9}
        trajectories = [action_row(0, 0, [0.0] * 7, terminate=1)]
        visibility = [visibility_row(0, 0), visibility_row(1, 0)]

        with self.assertRaises(AuditValidationError) as context:
            build_frame_analysis(manifest, trajectories, visibility)

        self.assertEqual(context.exception.reason, "extra_visibility_keys")

    def test_rejects_action_with_wrong_length(self):
        manifest = {"schema_version": "expert_v1", "action_dim": 9}
        row = action_row(0, 0, [0.0] * 7, terminate=1)
        row["action"].append(0.0)

        with self.assertRaises(AuditValidationError) as context:
            build_frame_analysis(manifest, [row], [visibility_row(0, 0)])

        self.assertEqual(context.exception.reason, "invalid_action")

    def test_rejects_boolean_joint_value(self):
        manifest = {"schema_version": "expert_v1", "action_dim": 9}
        row = action_row(0, 0, [False] + [0.0] * 6, terminate=1)

        with self.assertRaises(AuditValidationError) as context:
            build_frame_analysis(manifest, [row], [visibility_row(0, 0)])

        self.assertEqual(context.exception.reason, "invalid_action")

    def test_rejects_non_finite_action(self):
        manifest = {"schema_version": "expert_v1", "action_dim": 9}
        row = action_row(0, 0, [math.nan] + [0.0] * 6, terminate=1)

        with self.assertRaises(AuditValidationError) as context:
            build_frame_analysis(manifest, [row], [visibility_row(0, 0)])

        self.assertEqual(context.exception.reason, "non_finite_action")

    def test_rejects_decreasing_step_within_episode(self):
        manifest = {"schema_version": "expert_v1", "action_dim": 9}
        trajectories = [
            action_row(0, 24, [0.0] * 7),
            action_row(0, 12, [0.1] * 7, terminate=1),
        ]
        visibility = [visibility_row(0, 24), visibility_row(0, 12)]

        with self.assertRaises(AuditValidationError) as context:
            build_frame_analysis(manifest, trajectories, visibility)

        self.assertEqual(context.exception.reason, "non_increasing_step")

    def test_rejects_unknown_visibility_group(self):
        manifest = {"schema_version": "expert_v1", "action_dim": 9}
        trajectories = [action_row(0, 0, [0.0] * 7, terminate=1)]

        with self.assertRaises(AuditValidationError) as context:
            build_frame_analysis(
                manifest,
                trajectories,
                [visibility_row(0, 0, "hidden")],
            )

        self.assertEqual(context.exception.reason, "invalid_visibility_group")

    def test_rejects_non_terminal_stop_flag(self):
        manifest = {"schema_version": "expert_v1", "action_dim": 9}
        trajectories = [
            action_row(0, 0, [0.0] * 7, terminate=1),
            action_row(0, 24, [0.1] * 7, terminate=1),
        ]
        visibility = [visibility_row(0, 0), visibility_row(0, 24)]

        with self.assertRaises(AuditValidationError) as context:
            build_frame_analysis(manifest, trajectories, visibility)

        self.assertEqual(context.exception.reason, "invalid_episode_termination")

    def test_rejects_episode_without_terminal_stop_flag(self):
        manifest = {"schema_version": "expert_v1", "action_dim": 9}
        trajectories = [action_row(0, 0, [0.0] * 7)]

        with self.assertRaises(AuditValidationError) as context:
            build_frame_analysis(
                manifest,
                trajectories,
                [visibility_row(0, 0)],
            )

        self.assertEqual(context.exception.reason, "invalid_episode_termination")


class ActionStatisticsTests(unittest.TestCase):
    def test_numeric_stats_uses_population_std_and_linear_percentiles(self):
        module = importlib.import_module(
            "vla_project.simulation.audit_action_tokenization"
        )
        self.assertTrue(
            hasattr(module, "numeric_stats"),
            "numeric_stats 尚未实现",
        )

        result = module.numeric_stats([0.0, 1.0, 2.0, 3.0])

        self.assertEqual(result["count"], 4)
        self.assertEqual(result["mean"], 1.5)
        self.assertAlmostEqual(result["std"], 1.11803398875)
        self.assertAlmostEqual(result["p25"], 0.75)
        self.assertAlmostEqual(result["p99"], 2.97)

    def test_summarizes_special_dimensions_and_visibility_groups(self):
        module = importlib.import_module(
            "vla_project.simulation.audit_action_tokenization"
        )
        self.assertTrue(
            hasattr(module, "summarize_action_analysis"),
            "summarize_action_analysis 尚未实现",
        )
        manifest = {"schema_version": "expert_v1", "action_dim": 9}
        rows = build_frame_analysis(
            manifest,
            [
                action_row(0, 0, [0.0] * 7),
                action_row(0, 24, [0.24] * 7, terminate=1),
                action_row(1, 0, [1.0] * 7),
                action_row(1, 12, [0.88] * 7, terminate=1),
            ],
            [
                visibility_row(0, 0),
                visibility_row(0, 24, "partial"),
                visibility_row(1, 0),
                visibility_row(1, 12, "severe"),
            ],
        )

        summary = module.summarize_action_analysis(rows)

        self.assertEqual(summary["num_frames"], 4)
        self.assertEqual(summary["num_episodes"], 2)
        self.assertEqual(summary["num_transitions"], 2)
        self.assertTrue(summary["gripper"]["is_constant"])
        self.assertEqual(summary["gripper"]["counts"], {"1.0": 4})
        self.assertEqual(summary["terminate"]["counts"], {"0": 2, "1": 2})
        self.assertEqual(summary["step_gap"]["counts"], {"12": 1, "24": 1})
        self.assertEqual(summary["by_visibility"]["severe"]["num_frames"], 1)
        self.assertEqual(
            summary["by_visibility"]["severe"]["episode_count"],
            1,
        )
        self.assertTrue(
            summary["by_visibility"]["severe"][
                "insufficient_for_generalization"
            ]
        )


class BinningTests(unittest.TestCase):
    def test_evaluates_uniform_width_bins(self):
        module = importlib.import_module(
            "vla_project.simulation.audit_action_tokenization"
        )
        self.assertTrue(
            hasattr(module, "evaluate_binning"),
            "evaluate_binning 尚未实现",
        )

        result = module.evaluate_binning(
            [0.0, 1.0, 2.0, 3.0],
            "uniform_width",
            2,
        )

        self.assertEqual(result["edges"], [0.0, 1.5, 3.0])
        self.assertEqual(result["reconstruction_values"], [0.75, 2.25])
        self.assertEqual(result["counts"], [2, 2])
        self.assertAlmostEqual(result["mae"], 0.5)
        self.assertAlmostEqual(result["p95_absolute_error"], 0.75)
        self.assertAlmostEqual(result["normalized_entropy"], 1.0)

    def test_merges_duplicate_quantile_edges(self):
        module = importlib.import_module(
            "vla_project.simulation.audit_action_tokenization"
        )
        self.assertTrue(hasattr(module, "evaluate_binning"))

        result = module.evaluate_binning(
            [0.0, 0.0, 0.0, 1.0],
            "quantile",
            4,
        )

        self.assertLess(result["effective_num_bins"], 4)
        self.assertEqual(
            result["nonempty_bin_occupancy"],
            result["effective_num_bins"] / 4,
        )
        self.assertTrue(
            all(
                left < right
                for left, right in zip(result["edges"], result["edges"][1:])
            )
        )

    def test_quantile_empty_intervals_emit_null_without_warning(self):
        module = importlib.import_module(
            "vla_project.simulation.audit_action_tokenization"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            try:
                result = module.evaluate_binning(
                    [0.0, 0.2, 0.5, 1.0],
                    "quantile",
                    16,
                )
            except RuntimeWarning as exc:
                self.fail(f"空等频区间不应触发 warning: {exc}")

        empty_reconstruction_values = [
            reconstruction
            for count, reconstruction in zip(
                result["counts"],
                result["reconstruction_values"],
            )
            if count == 0
        ]
        self.assertTrue(empty_reconstruction_values)
        self.assertTrue(
            all(value is None for value in empty_reconstruction_values)
        )

    def test_evaluates_exactly_twelve_candidates(self):
        module = importlib.import_module(
            "vla_project.simulation.audit_action_tokenization"
        )
        self.assertTrue(
            hasattr(module, "evaluate_candidates"),
            "evaluate_candidates 尚未实现",
        )
        rows = build_frame_analysis(
            {"schema_version": "expert_v1", "action_dim": 9},
            [
                action_row(0, 0, [0.0] * 7),
                action_row(0, 24, [0.2] * 7),
                action_row(0, 48, [0.5] * 7, terminate=1),
                action_row(1, 0, [1.0] * 7),
                action_row(1, 24, [0.7] * 7),
                action_row(1, 48, [0.1] * 7, terminate=1),
            ],
            [
                visibility_row(0, 0),
                visibility_row(0, 24),
                visibility_row(0, 48),
                visibility_row(1, 0),
                visibility_row(1, 24),
                visibility_row(1, 48),
            ],
        )

        candidates = module.evaluate_candidates(rows)

        identities = {
            (
                candidate["representation"],
                candidate["binning"],
                candidate["num_bins"],
            )
            for candidate in candidates
        }
        expected = {
            (representation, method, num_bins)
            for representation in ("absolute_q", "delta_q")
            for method in ("uniform_width", "quantile")
            for num_bins in (16, 32, 64)
        }
        self.assertEqual(identities, expected)
        self.assertTrue(
            all(len(candidate["per_joint"]) == 7 for candidate in candidates)
        )
        self.assertTrue(
            all("worst_joint_metrics" in candidate for candidate in candidates)
        )

    def test_recommends_smallest_eligible_candidate_deterministically(self):
        module = importlib.import_module(
            "vla_project.simulation.audit_action_tokenization"
        )
        self.assertTrue(
            hasattr(module, "select_recommendations"),
            "select_recommendations 尚未实现",
        )

        def candidate(representation, method, bins, occupancy, count, error):
            return {
                "representation": representation,
                "binning": method,
                "num_bins": bins,
                "worst_joint_metrics": {
                    "nonempty_bin_occupancy": occupancy,
                    "minimum_nonempty_bin_count": count,
                    "normalized_p95_reconstruction_error": error,
                    "zero_robust_range": False,
                },
            }

        candidates = [
            candidate("absolute_q", "quantile", 16, 1.0, 30, 0.04),
            candidate("absolute_q", "uniform_width", 16, 1.0, 30, 0.04),
            candidate("absolute_q", "uniform_width", 32, 1.0, 40, 0.02),
            candidate("delta_q", "uniform_width", 16, 0.5, 50, 0.01),
        ]

        result = module.select_recommendations(candidates)

        self.assertTrue(result["continuous_regression"]["included"])
        self.assertEqual(result["absolute_q"]["status"], "selected")
        self.assertEqual(
            result["absolute_q"]["candidate"]["binning"],
            "uniform_width",
        )
        self.assertEqual(result["absolute_q"]["candidate"]["num_bins"], 16)
        self.assertEqual(result["delta_q"]["status"], "no_eligible_candidate")
        self.assertEqual(
            result["delta_q"]["ranked_candidates"][0]["rejection_reasons"],
            ["occupancy_below_threshold"],
        )

    def test_keeps_empty_visibility_groups_in_summary(self):
        module = importlib.import_module(
            "vla_project.simulation.audit_action_tokenization"
        )
        self.assertTrue(hasattr(module, "summarize_action_analysis"))
        rows = build_frame_analysis(
            {"schema_version": "expert_v1", "action_dim": 9},
            [action_row(0, 0, [0.0] * 7, terminate=1)],
            [visibility_row(0, 0, "clear")],
        )

        summary = module.summarize_action_analysis(rows)

        self.assertEqual(summary["by_visibility"]["partial"]["num_frames"], 0)
        self.assertEqual(summary["by_visibility"]["severe"]["num_frames"], 0)
        self.assertTrue(
            summary["by_visibility"]["severe"][
                "insufficient_for_generalization"
            ]
        )

if __name__ == "__main__":
    unittest.main()
