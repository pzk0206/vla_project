import json
import tempfile
import unittest
from pathlib import Path

from control_arm import determine_termination, next_episode_index


class DetermineTerminationTests(unittest.TestCase):
    def test_success_includes_exact_distance_threshold(self):
        self.assertEqual(
            determine_termination(
                distance_to_target=0.03,
                recent_distances=[0.04, 0.03],
                step_idx=10,
                success_distance=0.03,
                stuck_window_steps=2,
                stuck_min_improvement=0.0005,
                force_terminal_after_step=976,
            ),
            "success",
        )

    def test_max_steps_is_not_masked_by_full_improving_stuck_window(self):
        self.assertEqual(
            determine_termination(
                distance_to_target=0.5,
                recent_distances=[0.6, 0.5],
                step_idx=976,
                success_distance=0.03,
                stuck_window_steps=2,
                stuck_min_improvement=0.0005,
                force_terminal_after_step=976,
            ),
            "max_steps",
        )

    def test_stuck_precedes_max_steps_when_both_apply(self):
        self.assertEqual(
            determine_termination(
                distance_to_target=0.5,
                recent_distances=[0.5001, 0.5],
                step_idx=976,
                success_distance=0.03,
                stuck_window_steps=2,
                stuck_min_improvement=0.0005,
                force_terminal_after_step=976,
            ),
            "stuck",
        )


class NextEpisodeIndexTests(unittest.TestCase):
    def test_missing_summary_starts_from_zero(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            summary_path = Path(temp_dir) / "episode_summary.jsonl"
            self.assertEqual(next_episode_index(summary_path), 0)

    def test_existing_summary_continues_after_largest_episode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            summary_path = Path(temp_dir) / "episode_summary.jsonl"
            rows = [{"episode_idx": 2}, {"episode_idx": 7}, {"episode_idx": 4}]
            summary_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            self.assertEqual(next_episode_index(summary_path), 8)


if __name__ == "__main__":
    unittest.main()
