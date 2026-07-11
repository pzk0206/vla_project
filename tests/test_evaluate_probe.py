import tempfile
import unittest
from pathlib import Path

from evaluate_probe import aggregate_probe_summaries, cleanup_success_images


class AggregateTests(unittest.TestCase):
    def test_aggregates_success_failure_and_metrics(self):
        rows = [
            {"episode_idx": 0, "success": True, "termination_reason": "success", "final_distance": 0.02, "num_control_steps": 10, "direction_counts": {"front": 2}},
            {"episode_idx": 1, "success": False, "termination_reason": "max_control_steps", "final_distance": 0.5, "num_control_steps": 80, "direction_counts": {"left": 3}},
            {"episode_idx": 2, "success": False, "termination_reason": "error", "final_distance": None, "num_control_steps": 0, "direction_counts": {}},
        ]
        result = aggregate_probe_summaries(rows, {"probe": {}})
        self.assertEqual(result["success_count"], 1)
        self.assertEqual(result["failure_count"], 2)
        self.assertEqual(result["error_count"], 1)
        self.assertAlmostEqual(result["success_rate"], 1 / 3)
        self.assertEqual(result["final_distance_median"], 0.26)
        self.assertEqual(result["direction_counts"], {"front": 2, "left": 3})
        self.assertEqual(result["failed_episode_indices"], [1, 2])

    def test_success_cleanup_deletes_images_but_keeps_trace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)
            (path / "probe_step_00.jpg").write_bytes(b"x")
            (path / "probe_trace.jsonl").write_text("{}\n")
            cleanup_success_images(path, success=True, enabled=True)
            self.assertFalse((path / "probe_step_00.jpg").exists())
            self.assertTrue((path / "probe_trace.jsonl").exists())

    def test_failure_cleanup_keeps_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)
            image = path / "probe_step_00.jpg"
            image.write_bytes(b"x")
            cleanup_success_images(path, success=False, enabled=True)
            self.assertTrue(image.exists())
