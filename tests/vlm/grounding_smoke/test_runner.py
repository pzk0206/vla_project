"""验证 grounding smoke 闭环编排、证据留存和批次判定。"""

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from vla_project.simulation.stage3_probe import InvalidModelResponseError
from vla_project.vlm.grounding_smoke.targeting import (
    SmokeSafetyAbort,
    compute_action_from_world_target,
    compute_grounding_action,
)
from vla_project.vlm.grounding_smoke.runner import (
    SmokeDependencies,
    SmokePreflightError,
    _build_episode_dependencies,
    aggregate_smoke_summaries,
    build_smoke_cases,
    compute_visibility,
    make_run_dir,
    preflight_smoke_case,
    run_control_loop,
    run_smoke_batch,
)


CASE = {"episode_idx": 0, "seed": 52, "start_direction": "left"}
BASE_CONFIG = {
    "max_control_steps": 10,
    "max_stale_target_steps": 4,
    "clear_visibility_threshold": 0.75,
    "seeds": [52, 53, 54],
    "required_successes": 3,
    "max_total_api_calls": 30,
}


def batch_config(temp_dir):
    calibration_path = Path(temp_dir) / "calibration.json"
    calibration_path.write_text("{}", encoding="utf-8")
    config = {
        "connection_mode": "DIRECT",
        "gravity": [0, 0, -9.8],
        "enable_time_sleep": False,
        "simulation_hz": 240,
        "robot": {
            "ee_link_index": 6,
            "controlled_joints": 7,
        },
        "task": {"initial_settle_steps": 1},
        "dataset": {"instruction": "悬停在红色积木上方"},
        "probe": {
            "sim_steps_per_action": 1,
            "api": {},
        },
        "camera": {
            "workspace_center": [0.0, 0.4, 0.0],
            "near_val": 0.1,
            "far_val": 100.0,
        },
        "vlm_evaluation": {
            "balanced_pose_tolerance": 0.005,
            "balanced_pose_ik_iterations": 20,
            "camera_override": {
                "image_width": 448,
                "image_height": 448,
                "eye_offset_base": [0.0, 0.0, 3.0],
                "eye_offset_random_range": [0.0, 0.0],
                "up_vector": [0, 1, 0],
                "fov": 45,
            },
        },
        "grounding_smoke": {
            **BASE_CONFIG,
            "output_dir": temp_dir,
            "calibration_path": str(calibration_path),
            "expected_calibration_samples": 15,
            "expected_calibration_sample_ids": [
                f"seed_{seed}_d020_{direction}"
                for seed in range(42, 47)
                for direction in ("front", "left", "right")
            ],
            "api_max_retries": 0,
            "expected_correction_x": 0.02492227406480192,
            "expected_correction_y": -0.019467343494422532,
            "start_directions": ["left", "right", "front"],
            "start_offset_xy": 0.10,
            "hover_z": 0.20,
            "move_step_xy": 0.02,
            "visibility_reference_pixels": 378,
        },
    }
    fake_calibration = {
        "num_clear_calibration_samples": 15,
        "correction_x": 0.02492227406480192,
        "correction_y": -0.019467343494422532,
    }
    return config, fake_calibration


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
        compute_held_action=None,
        execute=None,
        score=None,
    ):
        return SmokeDependencies(
            observe=observe or Mock(return_value=observation()),
            ground=ground
            or Mock(return_value=({"red_block": [450, 450, 550, 550]}, "{}", 0.1)),
            compute_action=compute_action or Mock(return_value=action("stop", 0.01)),
            compute_held_action=compute_held_action
            or Mock(return_value=action("right", 0.06)),
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

    def test_visibility_without_cached_target_still_aborts(self):
        ground = Mock()
        held = Mock()
        dependencies = self.make_dependencies(
            observe=Mock(return_value=observation(0.50)),
            ground=ground,
            compute_held_action=held,
        )

        summary, rows = self.run_in_temp(dependencies)

        self.assertEqual(
            summary["termination_reason"], "visibility_out_of_scope"
        )
        self.assertIsNone(rows[-1]["decision_source"])
        self.assertFalse(rows[-1]["api_called"])
        ground.assert_not_called()
        held.assert_not_called()

    def test_low_visibility_reuses_cached_target_without_api_call(self):
        fresh = action("right", 0.08)
        held = action("right", 0.06)
        ground = Mock(
            return_value=(
                {"red_block": [450, 450, 550, 550]},
                "fresh",
                0.1,
            )
        )
        compute_held = Mock(return_value=held)
        dependencies = self.make_dependencies(
            observe=Mock(
                side_effect=[observation(1.0), observation(0.50)]
            ),
            ground=ground,
            compute_action=Mock(return_value=fresh),
            compute_held_action=compute_held,
            score=Mock(
                side_effect=[scoring(0.10), scoring(0.08), scoring(0.06)]
            ),
        )

        summary, rows = self.run_in_temp(
            dependencies,
            dict(BASE_CONFIG, max_control_steps=2),
        )

        self.assertEqual(summary["termination_reason"], "max_control_steps")
        self.assertEqual(ground.call_count, 1)
        self.assertEqual(rows[0]["decision_source"], "fresh_vlm")
        self.assertEqual(rows[0]["target_age_steps"], 0)
        self.assertEqual(rows[1]["decision_source"], "held_vlm_target")
        self.assertEqual(rows[1]["target_age_steps"], 1)
        self.assertFalse(rows[1]["api_called"])
        self.assertEqual(
            compute_held.call_args.kwargs["target_world"],
            fresh["corrected_target_world"],
        )

    def test_fifth_held_attempt_aborts_without_action(self):
        observations = [observation(1.0)] + [observation(0.50)] * 5
        dependencies = self.make_dependencies(
            observe=Mock(side_effect=observations),
            compute_action=Mock(return_value=action("right", 0.10)),
            compute_held_action=Mock(
                side_effect=[
                    action("right", 0.08),
                    action("right", 0.06),
                    action("right", 0.04),
                    action("right", 0.02),
                ]
            ),
            score=Mock(side_effect=[scoring(0.12)] * 7),
        )

        summary, rows = self.run_in_temp(
            dependencies,
            dict(BASE_CONFIG, max_control_steps=6),
        )

        self.assertEqual(summary["termination_reason"], "stale_target_limit")
        self.assertFalse(summary["task_success"])
        self.assertFalse(summary["autonomous_stop_success"])
        self.assertFalse(summary["success"])
        self.assertEqual(
            [row["target_age_steps"] for row in rows[:-1]],
            [0, 1, 2, 3, 4],
        )
        self.assertIsNone(rows[-1]["decision_source"])
        self.assertEqual(summary["num_held_target_steps"], 4)
        self.assertEqual(summary["num_actions"], 5)

    def test_new_fresh_target_resets_target_age(self):
        dependencies = self.make_dependencies(
            observe=Mock(
                side_effect=[
                    observation(1.0),
                    observation(0.50),
                    observation(1.0),
                ]
            ),
            ground=Mock(
                side_effect=[
                    ({"red_block": [450, 450, 550, 550]}, "one", 0.1),
                    ({"red_block": [451, 450, 551, 550]}, "two", 0.1),
                ]
            ),
            compute_action=Mock(
                side_effect=[action("right", 0.08), action("stop", 0.01)]
            ),
            compute_held_action=Mock(return_value=action("right", 0.04)),
            score=Mock(
                side_effect=[
                    scoring(0.10),
                    scoring(0.08),
                    scoring(0.04),
                    scoring(0.01),
                ]
            ),
        )

        summary, rows = self.run_in_temp(dependencies)

        self.assertTrue(summary["success"])
        self.assertEqual(
            [
                (row["decision_source"], row["target_age_steps"])
                for row in rows
            ],
            [
                ("fresh_vlm", 0),
                ("held_vlm_target", 1),
                ("fresh_vlm", 0),
            ],
        )

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

        self.assertTrue(summary["task_success"])
        self.assertTrue(summary["autonomous_stop_success"])
        self.assertTrue(summary["success"])
        self.assertEqual(summary["initial_true_distance_xy"], 0.10)
        self.assertEqual(summary["final_true_distance_xy"], 0.02)
        self.assertEqual(summary["num_actions"], 1)
        self.assertEqual(len(rows), 2)
        self.assertEqual(ground.call_count, 2)
        self.assertEqual(execute.call_count, 1)

    def test_stale_target_limit_counts_as_task_success_after_reaching_target(self):
        dependencies = self.make_dependencies(
            observe=Mock(
                side_effect=[observation(1.0)] + [observation(0.50)] * 5
            ),
            compute_action=Mock(return_value=action("right", 0.10)),
            compute_held_action=Mock(
                side_effect=[
                    action("right", 0.08),
                    action("right", 0.06),
                    action("right", 0.04),
                    action("right", 0.02),
                ]
            ),
            score=Mock(
                side_effect=[
                    scoring(0.12),
                    scoring(0.10),
                    scoring(0.08),
                    scoring(0.06),
                    scoring(0.04),
                    scoring(0.004),
                    scoring(0.004),
                ]
            ),
        )

        summary, _ = self.run_in_temp(
            dependencies,
            dict(BASE_CONFIG, max_control_steps=6),
        )

        self.assertEqual(summary["termination_reason"], "stale_target_limit")
        self.assertEqual(summary["final_true_distance_xy"], 0.004)
        self.assertTrue(summary["task_success"])
        self.assertFalse(summary["autonomous_stop_success"])
        self.assertTrue(summary["success"])

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

    def test_parser_rejection_becomes_invalid_box(self):
        dependencies = self.make_dependencies(
            ground=Mock(side_effect=InvalidModelResponseError("bad boxes"))
        )

        summary, rows = self.run_in_temp(dependencies)

        self.assertEqual(summary["termination_reason"], "invalid_box")
        self.assertEqual(rows[-1]["termination_reason"], "invalid_box")

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

    def test_held_compute_exception_becomes_held_target_error(self):
        dependencies = self.make_dependencies(
            observe=Mock(
                side_effect=[observation(1.0), observation(0.50)]
            ),
            compute_action=Mock(return_value=action("right", 0.08)),
            compute_held_action=Mock(
                side_effect=ValueError("malformed cached state")
            ),
            score=Mock(
                side_effect=[scoring(0.10), scoring(0.08), scoring(0.08)]
            ),
        )

        summary, rows = self.run_in_temp(
            dependencies,
            dict(BASE_CONFIG, max_control_steps=2),
        )

        self.assertEqual(summary["termination_reason"], "held_target_error")
        self.assertEqual(rows[-1]["termination_reason"], "held_target_error")
        self.assertFalse(rows[-1]["api_called"])

    def test_execute_exception_becomes_ik_error(self):
        dependencies = self.make_dependencies(
            compute_action=Mock(return_value=action("right", 0.08)),
            execute=Mock(side_effect=RuntimeError("ik failed")),
        )

        summary, rows = self.run_in_temp(dependencies)

        self.assertEqual(summary["termination_reason"], "ik_error")
        self.assertEqual(rows[-1]["termination_reason"], "ik_error")

    def test_system_errors_cannot_pass_despite_task_arrival(self):
        cases = {
            "api_error": self.make_dependencies(
                ground=Mock(side_effect=TimeoutError("timeout")),
                score=Mock(side_effect=[scoring(0.10), scoring(0.01)]),
            ),
            "backprojection_error": self.make_dependencies(
                compute_action=Mock(side_effect=ValueError("bad matrix")),
                score=Mock(side_effect=[scoring(0.10), scoring(0.01)]),
            ),
            "ik_error": self.make_dependencies(
                compute_action=Mock(return_value=action("right", 0.08)),
                execute=Mock(side_effect=RuntimeError("ik failed")),
                score=Mock(side_effect=[scoring(0.10), scoring(0.01)]),
            ),
        }

        for expected_reason, dependencies in cases.items():
            with self.subTest(expected_reason=expected_reason):
                summary, _ = self.run_in_temp(dependencies)
                self.assertEqual(summary["termination_reason"], expected_reason)
                self.assertTrue(summary["task_success"])
                self.assertFalse(summary["autonomous_stop_success"])
                self.assertFalse(summary["success"])

    def test_max_steps_cannot_pass_despite_task_arrival(self):
        dependencies = self.make_dependencies(
            compute_action=Mock(return_value=action("right", 0.08)),
            score=Mock(side_effect=[scoring(0.10), scoring(0.01)]),
        )

        summary, _ = self.run_in_temp(
            dependencies,
            dict(BASE_CONFIG, max_control_steps=1),
        )

        self.assertEqual(summary["termination_reason"], "max_control_steps")
        self.assertTrue(summary["task_success"])
        self.assertFalse(summary["autonomous_stop_success"])
        self.assertFalse(summary["success"])

    def test_batch_pass_requires_three_successful_recoverable_episodes(self):
        episodes = [
            {
                "seed": seed,
                "success": True,
                "api_calls": 2,
                "termination_reason": "success",
                "num_fresh_vlm_steps": 2,
                "num_held_target_steps": 3,
                "max_target_age_steps": 3,
                "recovered_from_occlusion": True,
            }
            for seed in (52, 53, 54)
        ]
        summary = aggregate_smoke_summaries(episodes, BASE_CONFIG)
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["fresh_vlm_steps"], 6)
        self.assertEqual(summary["held_target_steps"], 9)
        self.assertEqual(summary["recovered_episode_count"], 3)

        variants = [
            episodes[:2],
            [dict(episodes[0], success=False)] + episodes[1:],
            [dict(episodes[0], num_fresh_vlm_steps=0)] + episodes[1:],
            [dict(episodes[0], max_target_age_steps=5)] + episodes[1:],
            [dict(row, api_calls=11) for row in episodes],
        ]
        for rows in variants:
            with self.subTest(rows=rows):
                self.assertFalse(
                    aggregate_smoke_summaries(rows, BASE_CONFIG)["passed"]
                )


class SmokeBatchContractTests(unittest.TestCase):
    def make_preflight_config(self):
        return {
            "robot": {"ee_link_index": 6},
            "task": {"initial_settle_steps": 1},
            "camera": {"workspace_center": [0.0, 0.4, 0.0]},
            "vlm_evaluation": {
                "balanced_pose_tolerance": 0.005,
                "balanced_pose_ik_iterations": 20,
                "camera_override": {
                    "image_width": 448,
                    "image_height": 448,
                },
            },
            "grounding_smoke": {
                "hover_z": 0.20,
                "start_offset_xy": 0.10,
                "visibility_reference_pixels": 4,
                "clear_visibility_threshold": 0.75,
            },
        }

    def run_preflight(self, actual_start, segmentation):
        config = self.make_preflight_config()
        requested_start = [0.1, 0.4, 0.2]
        with (
            patch("vla_project.vlm.grounding_smoke.runner.connect_physics"),
            patch(
                "vla_project.vlm.grounding_smoke.runner.setup_world",
                return_value=(1, 10),
            ),
            patch(
                "vla_project.vlm.grounding_smoke.runner.load_block",
                return_value=20,
            ),
            patch("vla_project.vlm.grounding_smoke.runner.settle_object"),
            patch(
                "vla_project.vlm.grounding_smoke.runner.get_object_position",
                return_value=[0.0, 0.0, 0.05],
            ),
            patch(
                "vla_project.vlm.grounding_smoke.runner.build_balanced_ee_positions",
                return_value={"left": requested_start},
            ),
            patch(
                "vla_project.vlm.grounding_smoke.runner.reset_robot_to_target",
                return_value=actual_start,
            ),
            patch(
                "vla_project.vlm.grounding_smoke.runner.sample_camera_eye",
                return_value=[0.0, 0.4, 3.0],
            ),
            patch(
                "vla_project.vlm.grounding_smoke.runner.capture_rgb_and_segmentation",
                return_value=(
                    np.zeros((2, 2, 3), dtype=np.uint8),
                    np.asarray(segmentation, dtype=np.int64),
                ),
            ),
            patch(
                "vla_project.vlm.grounding_smoke.runner.p.disconnect"
            ) as disconnect,
        ):
            result = preflight_smoke_case(
                config,
                config["grounding_smoke"],
                CASE,
            )
        disconnect.assert_called_once_with()
        return result

    def test_preflight_records_visibility_even_when_start_pose_fails(self):
        result = self.run_preflight(
            [0.106, 0.4, 0.2],
            [[20, 20], [20, 20]],
        )
        self.assertFalse(result["qualified"])
        self.assertEqual(result["rejection_reason"], "start_pose_error")
        self.assertAlmostEqual(result["start_pose_error"], 0.006)
        self.assertEqual(result["block_visibility_ratio"], 1.0)

    def test_preflight_rejects_low_initial_visibility(self):
        result = self.run_preflight(
            [0.1, 0.4, 0.2],
            [[20, 20], [0, 0]],
        )
        self.assertFalse(result["qualified"])
        self.assertEqual(
            result["rejection_reason"],
            "visibility_below_threshold",
        )
        self.assertEqual(result["block_visibility_ratio"], 0.5)

    def test_preflight_accepts_equal_pose_and_visibility_boundaries(self):
        result = self.run_preflight(
            [0.105, 0.4, 0.2],
            [[20, 20], [20, 0]],
        )
        self.assertTrue(result["qualified"])
        self.assertIsNone(result["rejection_reason"])
        self.assertAlmostEqual(result["start_pose_error"], 0.005)
        self.assertEqual(result["block_visibility_ratio"], 0.75)

    def test_preflight_disconnects_when_setup_raises(self):
        config = self.make_preflight_config()
        with (
            patch("vla_project.vlm.grounding_smoke.runner.connect_physics"),
            patch(
                "vla_project.vlm.grounding_smoke.runner.setup_world",
                side_effect=RuntimeError("setup failed"),
            ),
            patch(
                "vla_project.vlm.grounding_smoke.runner.p.disconnect"
            ) as disconnect,
        ):
            with self.assertRaisesRegex(RuntimeError, "setup failed"):
                preflight_smoke_case(
                    config,
                    config["grounding_smoke"],
                    CASE,
                )
        disconnect.assert_called_once_with()

    def test_grounding_dependency_disables_retries_without_mutating_probe_config(self):
        config = {
            "robot": {"ee_link_index": 6},
            "dataset": {"instruction": "悬停在红色积木上方"},
            "probe": {"api": {"max_retries": 2}},
        }
        smoke_config = {
            "api_max_retries": 0,
            "visibility_reference_pixels": 378,
        }
        image = np.zeros((448, 448, 3), dtype=np.uint8)
        with patch(
            "vla_project.vlm.grounding_smoke.runner.call_openai_compatible_api",
            return_value=({"red_block": [1, 1, 2, 2]}, "{}"),
        ) as api:
            dependencies = _build_episode_dependencies(
                config,
                smoke_config,
                CASE,
                Path("episode_000"),
                10,
                20,
                {},
                [0.0, 0.4, 3.0],
                [1.0] * 16,
                [1.0] * 16,
            )
            dependencies.ground(image)

        self.assertEqual(api.call_args.args[1]["max_retries"], 0)
        self.assertEqual(config["probe"]["api"]["max_retries"], 2)

    def test_builds_exact_three_cases_without_back(self):
        self.assertEqual(
            build_smoke_cases(
                {
                    "seeds": [52, 53, 54],
                    "start_directions": ["left", "right", "front"],
                }
            ),
            [
                {"episode_idx": 0, "seed": 52, "start_direction": "left"},
                {"episode_idx": 1, "seed": 53, "start_direction": "right"},
                {"episode_idx": 2, "seed": 54, "start_direction": "front"},
            ],
        )

    def test_rejects_changed_case_set(self):
        for smoke_config in (
            {"seeds": [52, 53], "start_directions": ["left", "right"]},
            {
                "seeds": [52, 52, 54],
                "start_directions": ["left", "right", "front"],
            },
            {
                "seeds": [52, 53, 54],
                "start_directions": ["left", "right", "back"],
            },
        ):
            with self.subTest(smoke_config=smoke_config):
                with self.assertRaises(ValueError):
                    build_smoke_cases(smoke_config)

    def test_run_directory_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            created = make_run_dir(temp_dir, "run_fixed")
            self.assertTrue(created.is_dir())
            with self.assertRaises(FileExistsError):
                make_run_dir(temp_dir, "run_fixed")

    def test_visibility_decodes_pybullet_object_id(self):
        block_id = 7
        segmentation = np.array(
            [
                [-1, block_id],
                [block_id | (3 << 24), 2],
            ],
            dtype=np.int64,
        )

        metrics = compute_visibility(segmentation, block_id, reference_pixels=4)

        self.assertEqual(metrics["block_visible_pixels"], 2)
        self.assertEqual(metrics["block_reference_pixels"], 4)
        self.assertEqual(metrics["block_visibility_ratio"], 0.5)

    def test_batch_preflights_all_cases_before_rejecting_without_api(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config, fake_calibration = batch_config(temp_dir)
            rows = [
                {
                    "episode_idx": 0,
                    "seed": 52,
                    "start_direction": "left",
                    "qualified": False,
                    "rejection_reason": "visibility_below_threshold",
                },
                {
                    "episode_idx": 1,
                    "seed": 53,
                    "start_direction": "right",
                    "qualified": True,
                    "rejection_reason": None,
                },
                {
                    "episode_idx": 2,
                    "seed": 54,
                    "start_direction": "front",
                    "qualified": False,
                    "rejection_reason": "start_pose_error",
                },
            ]
            with (
                patch(
                    "vla_project.vlm.grounding_smoke.runner.load_frozen_calibration",
                    return_value=fake_calibration,
                ),
                patch(
                    "vla_project.vlm.grounding_smoke.runner.preflight_smoke_case",
                    side_effect=rows,
                ) as preflight,
                patch(
                    "vla_project.vlm.grounding_smoke.runner.run_control_loop"
                ) as loop,
                patch(
                    "vla_project.vlm.grounding_smoke.runner.call_openai_compatible_api"
                ) as api,
            ):
                with self.assertRaises(SmokePreflightError):
                    run_smoke_batch(config, run_name="run_rejected")

            self.assertEqual(preflight.call_count, 3)
            loop.assert_not_called()
            api.assert_not_called()
            run_dir = Path(temp_dir) / "run_rejected"
            payload = json.loads(
                (run_dir / "smoke_preflight.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(payload["num_cases"], 3)
            self.assertEqual(payload["qualified_count"], 1)
            self.assertFalse(payload["passed"])
            self.assertEqual(payload["cases"], rows)
            self.assertFalse((run_dir / "episode_summary.jsonl").exists())
            self.assertFalse((run_dir / "smoke_summary.json").exists())

    def test_batch_uses_three_isolated_cases_and_writes_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config, fake_calibration = batch_config(temp_dir)

            def fake_loop(smoke_config, case, calibration, episode_dir, dependencies):
                episode_dir.mkdir(parents=True, exist_ok=False)
                return {
                    **case,
                    "success": True,
                    "termination_reason": "success",
                    "api_calls": 1,
                    "all_clear": True,
                    "num_fresh_vlm_steps": 1,
                    "num_held_target_steps": 0,
                    "max_target_age_steps": 0,
                    "recovered_from_occlusion": False,
                    "num_control_steps": 1,
                    "final_true_distance_xy": 0.02,
                }

            with (
                patch("vla_project.vlm.grounding_smoke.runner.load_frozen_calibration", return_value=fake_calibration),
                patch(
                    "vla_project.vlm.grounding_smoke.runner.preflight_smoke_case",
                    side_effect=[
                        {
                            **case,
                            "qualified": True,
                            "rejection_reason": None,
                        }
                        for case in build_smoke_cases(
                            config["grounding_smoke"]
                        )
                    ],
                ) as preflight,
                patch("vla_project.vlm.grounding_smoke.runner.connect_physics"),
                patch("vla_project.vlm.grounding_smoke.runner.setup_world", return_value=(1, 10)),
                patch("vla_project.vlm.grounding_smoke.runner.load_block", return_value=20),
                patch("vla_project.vlm.grounding_smoke.runner.settle_object"),
                patch("vla_project.vlm.grounding_smoke.runner.get_object_position", return_value=[0.1, 0.45, 0.05]),
                patch("vla_project.vlm.grounding_smoke.runner.build_balanced_ee_positions", return_value={
                    "left": [0.2, 0.45, 0.20],
                    "right": [0.0, 0.45, 0.20],
                    "front": [0.1, 0.35, 0.20],
                }),
                patch(
                    "vla_project.vlm.grounding_smoke.runner.reset_robot_to_target",
                    side_effect=lambda robot_id, robot_config, target_pos, **kwargs: list(target_pos),
                ),
                patch("vla_project.vlm.grounding_smoke.runner.sample_camera_eye", return_value=[0.0, 0.4, 3.0]),
                patch("vla_project.vlm.grounding_smoke.runner.compute_camera_matrices", return_value=([1.0] * 16, [1.0] * 16)),
                patch("vla_project.vlm.grounding_smoke.runner.run_control_loop", side_effect=fake_loop) as loop,
                patch("vla_project.vlm.grounding_smoke.runner.call_openai_compatible_api") as api,
                patch("vla_project.vlm.grounding_smoke.runner.p.disconnect") as disconnect,
            ):
                batch_dir, summary = run_smoke_batch(config, run_name="run_test")

            self.assertEqual(preflight.call_count, 3)
            self.assertEqual(loop.call_count, 3)
            self.assertEqual(
                [call.args[1]["seed"] for call in loop.call_args_list],
                [52, 53, 54],
            )
            self.assertNotIn(
                "block_pos", inspect.signature(compute_grounding_action).parameters
            )
            self.assertEqual(disconnect.call_count, 3)
            api.assert_not_called()
            self.assertTrue((batch_dir / "smoke_preflight.json").is_file())
            preflight_payload = json.loads(
                (batch_dir / "smoke_preflight.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(preflight_payload["passed"])
            self.assertEqual(preflight_payload["qualified_count"], 3)
            self.assertTrue((batch_dir / "smoke_summary.json").is_file())
            self.assertTrue((batch_dir / "episode_summary.jsonl").is_file())
            self.assertTrue(summary["passed"])


if __name__ == "__main__":
    unittest.main()
