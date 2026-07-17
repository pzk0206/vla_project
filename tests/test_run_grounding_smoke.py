"""验证 grounding smoke 闭环编排、证据留存和批次判定。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from grounding_targeting import SmokeSafetyAbort
from run_grounding_smoke import (
    SmokeDependencies,
    aggregate_smoke_summaries,
    run_control_loop,
)


CASE = {"episode_idx": 0, "seed": 52, "start_direction": "left"}
BASE_CONFIG = {
    "max_control_steps": 10,
    "clear_visibility_threshold": 0.75,
    "seeds": [52, 53, 54],
    "required_successes": 3,
    "max_total_api_calls": 30,
}


def observation(visibility_ratio=1.0):
    """构造不含红块真值的控制侧观测。"""
    return {
        "image_bgr": np.zeros((448, 448, 3), dtype=np.uint8),
        "view_matrix": [1.0] * 16,
        "projection_matrix": [1.0] * 16,
        "ee_pos": [0.0, 0.4, 0.2],
        "visibility": {
            "block_visible_pixels": round(378 * visibility_ratio),
            "block_reference_pixels": 378,
            "block_visibility_ratio": visibility_ratio,
        },
    }


def action(direction, distance=0.08):
    """构造策略层动作结果。"""
    return {
        "direction": direction,
        "box_center_pixel": [223.5, 223.5],
        "raw_target_world": [0.1, 0.4, 0.0],
        "corrected_target_world": [0.1, 0.4, 0.0],
        "predicted_distance_xy": distance,
        "target_jump_xy": 0.0,
        "safety_state": {
            "previous_target_xy": [0.1, 0.4],
            "previous_predicted_distance": distance,
            "no_progress_count": 0,
        },
    }


def scoring(distance):
    """真值只存在于独立评分返回值中。"""
    return {
        "true_block_pos": [0.1, 0.4, 0.05],
        "true_distance_xy": distance,
    }


class SmokeLoopTests(unittest.TestCase):
    def make_dependencies(
        self,
        *,
        observe=None,
        ground=None,
        compute_action=None,
        execute=None,
        score=None,
    ):
        return SmokeDependencies(
            observe=observe or Mock(return_value=observation()),
            ground=ground
            or Mock(return_value=({"red_block": [450, 450, 550, 550]}, "{}", 0.1)),
            compute_action=compute_action or Mock(return_value=action("stop", 0.01)),
            execute=execute or Mock(return_value={"target_pos": [0.02, 0.4, 0.2]}),
            score=score or Mock(return_value=scoring(0.01)),
            save_images=Mock(return_value=("raw.jpg", "annotated.jpg")),
        )

    def run_in_temp(self, dependencies, config=None):
        temp_dir = tempfile.TemporaryDirectory()
        episode_dir = Path(temp_dir.name) / "episode_000"
        summary = run_control_loop(
            config or BASE_CONFIG,
            CASE,
            {},
            episode_dir,
            dependencies,
        )
        trace_rows = [
            json.loads(line)
            for line in (episode_dir / "smoke_trace.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        temp_dir.cleanup()
        return summary, trace_rows

    def test_visibility_aborts_before_paid_grounding(self):
        ground = Mock()
        dependencies = self.make_dependencies(
            observe=Mock(return_value=observation(100 / 378)), ground=ground
        )

        summary, rows = self.run_in_temp(dependencies)

        self.assertEqual(summary["termination_reason"], "visibility_out_of_scope")
        self.assertEqual(rows[-1]["termination_reason"], "visibility_out_of_scope")
        self.assertEqual(summary["api_calls"], 0)
        ground.assert_not_called()

    def test_two_frame_success_regrounds_and_executes_once(self):
        ground = Mock(
            side_effect=[
                ({"red_block": [450, 450, 550, 550]}, "first", 0.1),
                ({"red_block": [451, 450, 551, 550]}, "second", 0.1),
            ]
        )
        compute_action = Mock(side_effect=[action("right", 0.08), action("stop", 0.01)])
        execute = Mock(return_value={"target_pos": [0.02, 0.4, 0.2]})
        score = Mock(side_effect=[scoring(0.10), scoring(0.08), scoring(0.02)])
        dependencies = self.make_dependencies(
            observe=Mock(side_effect=[observation(), observation()]),
            ground=ground,
            compute_action=compute_action,
            execute=execute,
            score=score,
        )

        summary, rows = self.run_in_temp(dependencies)

        self.assertTrue(summary["success"])
        self.assertEqual(summary["initial_true_distance_xy"], 0.10)
        self.assertEqual(summary["final_true_distance_xy"], 0.02)
        self.assertEqual(summary["num_actions"], 1)
        self.assertEqual(len(rows), 2)
        self.assertEqual(ground.call_count, 2)
        self.assertEqual(execute.call_count, 1)

    def test_action_call_never_receives_block_truth(self):
        compute_action = Mock(return_value=action("stop", 0.01))
        dependencies = self.make_dependencies(compute_action=compute_action)

        self.run_in_temp(dependencies)

        for call in compute_action.call_args_list:
            self.assertNotIn("block_pos", call.kwargs)
            self.assertNotIn("true_block_pos", call.kwargs)

    def test_false_stop_is_scored_only_after_predicted_stop(self):
        execute = Mock()
        dependencies = self.make_dependencies(
            compute_action=Mock(return_value=action("stop", 0.01)),
            execute=execute,
            score=Mock(side_effect=[scoring(0.10), scoring(0.031)]),
        )

        summary, rows = self.run_in_temp(dependencies)

        self.assertEqual(summary["termination_reason"], "false_stop")
        self.assertEqual(rows[-1]["true_distance_xy"], 0.031)
        execute.assert_not_called()

    def test_last_allowed_action_records_max_control_steps(self):
        config = dict(BASE_CONFIG, max_control_steps=2)
        dependencies = self.make_dependencies(
            observe=Mock(side_effect=[observation(), observation()]),
            ground=Mock(
                side_effect=[
                    ({"red_block": [450, 450, 550, 550]}, "one", 0.1),
                    ({"red_block": [450, 450, 550, 550]}, "two", 0.1),
                ]
            ),
            compute_action=Mock(side_effect=[action("right", 0.08), action("right", 0.06)]),
            score=Mock(side_effect=[scoring(0.10), scoring(0.08), scoring(0.06)]),
        )

        summary, rows = self.run_in_temp(dependencies, config)

        self.assertEqual(summary["termination_reason"], "max_control_steps")
        self.assertEqual(rows[-1]["termination_reason"], "max_control_steps")
        self.assertEqual(summary["num_actions"], 2)

    def test_safety_abort_reason_is_preserved_in_trace(self):
        for reason in ("target_jump", "target_out_of_workspace", "no_progress"):
            with self.subTest(reason=reason):
                dependencies = self.make_dependencies(
                    compute_action=Mock(
                        side_effect=SmokeSafetyAbort(reason, "test abort")
                    )
                )
                summary, rows = self.run_in_temp(dependencies)
                self.assertEqual(summary["termination_reason"], reason)
                self.assertEqual(rows[-1]["termination_reason"], reason)

    def test_grounding_exception_becomes_api_error(self):
        dependencies = self.make_dependencies(
            ground=Mock(side_effect=TimeoutError("timeout"))
        )

        summary, rows = self.run_in_temp(dependencies)

        self.assertEqual(summary["termination_reason"], "api_error")
        self.assertEqual(summary["api_calls"], 1)
        self.assertEqual(rows[-1]["termination_reason"], "api_error")

    def test_missing_red_box_becomes_invalid_box(self):
        dependencies = self.make_dependencies(
            ground=Mock(return_value=({}, "{}", 0.1))
        )

        summary, rows = self.run_in_temp(dependencies)

        self.assertEqual(summary["termination_reason"], "invalid_box")
        self.assertEqual(rows[-1]["termination_reason"], "invalid_box")

    def test_geometry_exception_becomes_backprojection_error(self):
        dependencies = self.make_dependencies(
            compute_action=Mock(side_effect=ValueError("bad matrix"))
        )

        summary, rows = self.run_in_temp(dependencies)

        self.assertEqual(summary["termination_reason"], "backprojection_error")
        self.assertEqual(rows[-1]["termination_reason"], "backprojection_error")

    def test_execute_exception_becomes_ik_error(self):
        dependencies = self.make_dependencies(
            compute_action=Mock(return_value=action("right", 0.08)),
            execute=Mock(side_effect=RuntimeError("ik failed")),
        )

        summary, rows = self.run_in_temp(dependencies)

        self.assertEqual(summary["termination_reason"], "ik_error")
        self.assertEqual(rows[-1]["termination_reason"], "ik_error")

    def test_batch_pass_requires_exact_three_clear_successes(self):
        episodes = [
            {
                "seed": seed,
                "success": True,
                "all_clear": True,
                "api_calls": 5,
                "termination_reason": "success",
            }
            for seed in (52, 53, 54)
        ]
        self.assertTrue(aggregate_smoke_summaries(episodes, BASE_CONFIG)["passed"])

        variants = [
            episodes[:2],
            [dict(episodes[0], success=False)] + episodes[1:],
            [dict(episodes[0], all_clear=False)] + episodes[1:],
            [dict(row, api_calls=11) for row in episodes],
        ]
        for rows in variants:
            with self.subTest(rows=rows):
                self.assertFalse(
                    aggregate_smoke_summaries(rows, BASE_CONFIG)["passed"]
                )


if __name__ == "__main__":
    unittest.main()
