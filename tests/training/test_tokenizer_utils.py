"""保护分类动作只使用审计报告冻结的边界与重建值。"""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vla_project.training import tokenizer_utils


def candidate(per_joint):
    return {
        "representation": "delta_q",
        "binning": "quantile",
        "num_bins": 2,
        "per_joint": per_joint,
    }


def joint(edges=None, reconstruction_values=None):
    return {
        "edges": edges if edges is not None else [0.0, 1.0, 3.0],
        "reconstruction_values": (
            reconstruction_values
            if reconstruction_values is not None
            else [0.1, 2.8]
        ),
    }


class QuantileTokenizerTests(unittest.TestCase):
    def write_report(self, root, per_joint):
        path = Path(root) / "audit.json"
        path.write_text(
            json.dumps({"candidates": [candidate(per_joint)]}),
            encoding="utf-8",
        )
        return path

    def test_loads_edges_and_audited_reconstruction_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self.write_report(temp_dir, [joint() for _ in range(7)])

            actual = tokenizer_utils.load_quantile_tokenizer(
                path,
                "delta_q",
                2,
            )

        self.assertEqual(actual["representation"], "delta_q")
        self.assertEqual(actual["binning"], "quantile")
        self.assertEqual(actual["requested_num_bins"], 2)
        self.assertEqual(actual["edges"], [[0.0, 1.0, 3.0]] * 7)
        self.assertEqual(
            actual["reconstruction_values"],
            [[0.1, 2.8]] * 7,
        )

    def test_decodes_exact_audited_values_not_bin_centers(self):
        reconstruction_values = [[0.1, 2.8] for _ in range(7)]

        actual = tokenizer_utils.decode_tokens_to_action(
            np.asarray([0, 1, 0, 1, 0, 1, 0]),
            reconstruction_values,
        )

        np.testing.assert_allclose(
            actual,
            [0.1, 2.8, 0.1, 2.8, 0.1, 2.8, 0.1],
        )
        self.assertNotEqual(actual[0], 0.5)
        self.assertNotEqual(actual[1], 2.0)

    def test_rejects_malformed_or_incomplete_audit_assets(self):
        malformed = {
            "six_joints": [joint() for _ in range(6)],
            "missing_reconstruction": [
                joint(reconstruction_values=[None, 2.8])
                for _ in range(7)
            ],
            "length_mismatch": [
                joint(reconstruction_values=[0.1])
                for _ in range(7)
            ],
            "non_finite_edge": [
                joint(edges=[0.0, math.inf, 3.0])
                for _ in range(7)
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            for name, per_joint in malformed.items():
                with self.subTest(name=name):
                    path = self.write_report(temp_dir, per_joint)
                    with self.assertRaisesRegex(ValueError, "invalid tokenizer"):
                        tokenizer_utils.load_quantile_tokenizer(
                            path,
                            "delta_q",
                            2,
                        )

    def test_rejects_out_of_range_token(self):
        with self.assertRaisesRegex(ValueError, "token id out of range"):
            tokenizer_utils.decode_tokens_to_action(
                [0, 1, 0, 1, 0, 1, 2],
                [[0.1, 2.8] for _ in range(7)],
            )


if __name__ == "__main__":
    unittest.main()
