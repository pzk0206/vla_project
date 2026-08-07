"""BC 数据集：从 trajectory + episode_split 加载 (image, action) 对。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

# ImageNet 归一化参数
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_DEFAULT_DATASET = "outputs/dataset/expert_scaling_v1"


def _load_episode_ids(split_path, split_key="train"):
    """从 episode_split.json 加载训练或验证的 episode 列表。"""
    with open(split_path, encoding="utf-8") as fh:
        doc = json.load(fh)
    ids = doc.get(split_key)
    if not isinstance(ids, list) or not ids:
        raise ValueError(f"episode_split.json 缺少 {split_key} 列表")
    return set(ids)


def _load_trajectory_rows(trajectory_path, episode_set):
    """从 trajectory JSONL 中加载属于给定 episode 集的帧记录。"""
    rows = []
    with open(trajectory_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("episode_idx") in episode_set:
                rows.append(row)
    return rows


class BCDataset(Dataset):
    """行为克隆数据集。

    Args:
        dataset_dir: 数据集目录 (含 trajectory_expert.jsonl 和 episode_split.json)
        split: "train" | "val"
        action_representation: "regression" | "absolute_q_32" | "delta_q_64"
        tokenizer_audit_path: 分类组 tokenizer 审计报告路径（regression 不需要）
        action_stats: (mean, std) tuple for regression normalization
    """

    def __init__(
        self,
        dataset_dir=_DEFAULT_DATASET,
        split="train",
        action_representation="regression",
        tokenizer_audit_path=None,
        action_stats=None,
        episode_ids=None,
    ):
        dataset_dir = Path(dataset_dir)
        self.action_representation = action_representation
        self._is_classification = action_representation != "regression"
        self._use_delta = action_representation == "delta_q_64"

        # 加载 episode 过滤
        split_path = dataset_dir / "episode_split.json"
        if not split_path.is_file():
            raise FileNotFoundError(f"找不到划分文件: {split_path}")
        episode_set = _load_episode_ids(split_path, split)

        # 额外的 episode 子集过滤（overfit 模式）
        if episode_ids is not None:
            extra_set = set(episode_ids)
            episode_set = episode_set & extra_set

        # 加载轨迹帧
        trajectory_path = dataset_dir / "trajectory_expert.jsonl"
        self.rows = _load_trajectory_rows(trajectory_path, episode_set)

        # 对于 delta_q，排除每 episode 的首帧（无前帧可做差分）
        if self._use_delta:
            first_per_ep = {}
            filtered = []
            for r in self.rows:
                ep = r["episode_idx"]
                if ep not in first_per_ep:
                    first_per_ep[ep] = r["step_idx"]
                if r["step_idx"] != first_per_ep[ep]:
                    filtered.append(r)
            self.rows = filtered

        # 加载 tokenizer 边界（分类组）
        if self._is_classification:
            if tokenizer_audit_path is None:
                raise ValueError("分类组需要 tokenizer_audit_path")
            from .tokenizer_utils import load_quantile_edges

            rep = "absolute_q" if not self._use_delta else "delta_q"
            num_bins = 32 if not self._use_delta else 64
            self._edges = load_quantile_edges(
                tokenizer_audit_path, rep, num_bins
            )
            self.num_bins = num_bins

        # 回归组归一化参数
        if not self._is_classification:
            if action_stats is None:
                raise ValueError("回归组需要 action_stats=(mean, std)")
            self._action_mean, self._action_std = (
                np.asarray(action_stats[0], dtype=np.float32),
                np.asarray(action_stats[1], dtype=np.float32),
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        # 加载图片
        image = Image.open(row["image_path"]).convert("RGB")
        image = image.resize((224, 224), Image.BILINEAR)
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        # ImageNet 归一化
        image_np = (image_np - IMAGENET_MEAN) / IMAGENET_STD
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)  # (C, H, W)

        # 目标
        action = np.asarray(row["action"], dtype=np.float32)
        q_target = action[:7].copy()
        gripper = np.float32(action[7])
        terminate = np.float32(action[8])

        if self._is_classification:
            from .tokenizer_utils import encode_action_to_tokens

            token_ids = encode_action_to_tokens(q_target, self._edges)
            return (
                image_tensor,
                torch.from_numpy(token_ids).long(),
                torch.tensor([gripper, terminate], dtype=torch.float32),
            )
        else:
            q_norm = (q_target - self._action_mean) / (self._action_std + 1e-8)
            return (
                image_tensor,
                torch.from_numpy(q_norm).float(),
                torch.tensor([gripper, terminate], dtype=torch.float32),
            )


def compute_action_stats(trajectory_rows):
    """计算 absolute_q 的 mean/std（只算前 7 维）。"""
    all_q = []
    for r in trajectory_rows:
        all_q.append(r["action"][:7])
    arr = np.array(all_q, dtype=np.float64)
    return arr.mean(axis=0).tolist(), arr.std(axis=0).tolist()


class VLADataset(Dataset):
    """VLA 数据集：返回 (image, instruction_text, action, aux)。

    仅支持 regression 动作表示。与 BCDataset 的区别是多了 instruction 字段。
    """

    def __init__(
        self,
        dataset_dir=_DEFAULT_DATASET,
        split="train",
        action_stats=None,
        episode_ids=None,
    ):
        dataset_dir = Path(dataset_dir)

        split_path = dataset_dir / "episode_split.json"
        if not split_path.is_file():
            raise FileNotFoundError(f"找不到划分文件: {split_path}")
        episode_set = _load_episode_ids(split_path, split)

        if episode_ids is not None:
            episode_set = episode_set & set(episode_ids)

        trajectory_path = dataset_dir / "trajectory_expert.jsonl"
        self.rows = _load_trajectory_rows(trajectory_path, episode_set)

        if action_stats is None:
            raise ValueError("VLADataset 需要 action_stats=(mean, std)")
        self._action_mean = np.asarray(action_stats[0], dtype=np.float32)
        self._action_std = np.asarray(action_stats[1], dtype=np.float32)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        image = Image.open(row["image_path"]).convert("RGB")
        image = image.resize((224, 224), Image.BILINEAR)
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        image_np = (image_np - IMAGENET_MEAN) / IMAGENET_STD
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)

        instruction = row.get("instruction", "悬停在红色积木上方")

        action = np.asarray(row["action"], dtype=np.float32)
        q_target = action[:7].copy()
        gripper = np.float32(action[7])
        terminate = np.float32(action[8])

        q_norm = (q_target - self._action_mean) / (self._action_std + 1e-8)
        return (
            image_tensor,
            instruction,
            torch.from_numpy(q_norm).float(),
            torch.tensor([gripper, terminate], dtype=torch.float32),
        )
