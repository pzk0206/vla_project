"""保护 expert_v1 动作表示只读审计契约。"""

import importlib
import math
import unittest

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


if __name__ == "__main__":
    unittest.main()
