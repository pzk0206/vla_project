"""加载 tokenizer 分箱边界，对连续动作做离散编码/解码。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_quantile_tokenizer(audit_path, representation, num_bins):
    """从审计报告加载并严格验证量化边界与重建值。"""
    with open(audit_path, encoding="utf-8") as fh:
        report = json.load(fh)
    candidates = report.get("candidates", [])
    selected = next(
        (
            candidate
            for candidate in candidates
            if candidate.get("representation") == representation
            and candidate.get("binning") == "quantile"
            and candidate.get("num_bins") == num_bins
        ),
        None,
    )
    if selected is None:
        raise ValueError(
            f"在 {audit_path} 中找不到 {representation}/{num_bins} 等频边界"
        )

    per_joint = selected.get("per_joint")
    if not isinstance(per_joint, list) or len(per_joint) != 7:
        raise ValueError("invalid tokenizer: expected seven joints")
    edges_list = []
    reconstruction_values = []
    for joint_index, joint in enumerate(per_joint):
        edges = np.asarray(joint.get("edges"), dtype=np.float64)
        reconstruction = np.asarray(
            joint.get("reconstruction_values"),
            dtype=np.float64,
        )
        if (
            edges.ndim != 1
            or reconstruction.ndim != 1
            or edges.size != num_bins + 1
            or reconstruction.size != num_bins
            or not np.isfinite(edges).all()
            or not np.isfinite(reconstruction).all()
            or not np.all(np.diff(edges) > 0)
        ):
            raise ValueError(
                f"invalid tokenizer for joint {joint_index}"
            )
        edges_list.append(edges.tolist())
        reconstruction_values.append(reconstruction.tolist())
    return {
        "representation": representation,
        "binning": "quantile",
        "requested_num_bins": num_bins,
        "edges": edges_list,
        "reconstruction_values": reconstruction_values,
    }


def load_quantile_edges(audit_path, representation, num_bins):
    """兼容接口：返回严格验证后的等频分箱边界。

    Returns:
        list[list[float]]: 每个关节的分箱边界 (num_joints × (num_bins+1))
    """
    return load_quantile_tokenizer(
        audit_path,
        representation,
        num_bins,
    )["edges"]


def encode_action_to_tokens(q_values, edges_list):
    """将连续关节角度编码为离散 token ids。

    Args:
        q_values: (7,) 或 (N, 7) 关节角度
        edges_list: list of 7 edge arrays, each shape (num_bins+1,)

    Returns:
        (7,) 或 (N, 7) int token ids
    """
    q = np.asarray(q_values)
    if q.ndim == 1:
        q = q[None, :]
    tokens = np.zeros(q.shape, dtype=np.int64)
    for j in range(q.shape[1]):
        edges = np.asarray(edges_list[j])
        # searchsorted 返回 1..num_bins-1，映射到 0..num_bins-1
        bin_idx = np.searchsorted(edges[1:-1], q[:, j], side="right")
        tokens[:, j] = bin_idx
    if q_values is not None and np.asarray(q_values).ndim == 1:
        return tokens[0]
    return tokens


def decode_tokens_to_action(tokens, reconstruction_values):
    """使用审计冻结的逐箱重建值解码离散 token ids。

    Args:
        tokens: (7,) 或 (N, 7) int token ids
        reconstruction_values: list of 7 reconstruction arrays

    Returns:
        (7,) 或 (N, 7) float 重建角度
    """
    tokens_arr = np.asarray(tokens)
    was_1d = tokens_arr.ndim == 1
    if was_1d:
        tokens_arr = tokens_arr[None, :]
    if (
        tokens_arr.ndim != 2
        or tokens_arr.shape[1] != 7
        or not np.issubdtype(tokens_arr.dtype, np.integer)
    ):
        raise ValueError("invalid token ids")
    if not isinstance(reconstruction_values, list) or len(
        reconstruction_values
    ) != 7:
        raise ValueError("invalid tokenizer reconstruction values")
    values = np.zeros(tokens_arr.shape, dtype=np.float64)
    for j in range(tokens_arr.shape[1]):
        recon = np.asarray(reconstruction_values[j], dtype=np.float64)
        if recon.ndim != 1 or not np.isfinite(recon).all():
            raise ValueError("invalid tokenizer reconstruction values")
        joint_tokens = tokens_arr[:, j]
        if np.any(joint_tokens < 0) or np.any(joint_tokens >= recon.size):
            raise ValueError("token id out of range")
        values[:, j] = recon[joint_tokens]
    if was_1d:
        return values[0]
    return values
