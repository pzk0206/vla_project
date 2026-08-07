"""加载 tokenizer 分箱边界，对连续动作做离散编码/解码。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_quantile_edges(audit_path, representation, num_bins):
    """从审计报告中提取指定表示和箱数的等频分箱边界。

    Returns:
        list[list[float]]: 每个关节的分箱边界 (num_joints × (num_bins+1))
    """
    with open(audit_path, encoding="utf-8") as fh:
        report = json.load(fh)
    candidates = report.get("candidates", [])
    for cand in candidates:
        if (
            cand.get("representation") == representation
            and cand.get("binning") == "quantile"
            and cand.get("num_bins") == num_bins
        ):
            per_joint = cand["per_joint"]
            edges = [joint["edges"] for joint in per_joint]
            return edges
    raise ValueError(
        f"在 {audit_path} 中找不到 {representation}/{num_bins} 等频边界"
    )


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


def decode_tokens_to_action(tokens, edges_list):
    """将离散 token ids 解码为连续关节角度（取箱内中位数）。

    Args:
        tokens: (7,) 或 (N, 7) int token ids
        edges_list: list of 7 edge arrays

    Returns:
        (7,) 或 (N, 7) float 重建角度
    """
    tokens_arr = np.asarray(tokens)
    was_1d = tokens_arr.ndim == 1
    if was_1d:
        tokens_arr = tokens_arr[None, :]
    values = np.zeros(tokens_arr.shape, dtype=np.float64)
    for j in range(tokens_arr.shape[1]):
        edges = np.asarray(edges_list[j])
        # 从 audit 代码复用的重建逻辑：取箱内 median
        num_bins = len(edges) - 1
        # 构建 reconstruction_values
        recon = _reconstruction_values_for_edges(edges, num_bins)
        for i in range(tokens_arr.shape[0]):
            tid = min(tokens_arr[i, j], len(edges) - 2)
            values[i, j] = recon[tid]
    if was_1d:
        return values[0]
    return values


def _reconstruction_values_for_edges(edges, num_bins):
    """为每箱计算重建值：取箱内中位数的近似值（简化版用箱中心点）。"""
    recon = []
    for b in range(num_bins):
        lo = edges[b]
        hi = edges[b + 1]
        recon.append((lo + hi) / 2.0)
    return recon
