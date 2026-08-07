"""专家轨迹 → Qwen2-VL 微调数据集 (qwen_train.json)。

读取 trajectory_expert.jsonl + episode_split.json（仅取 train 划分），
逐帧生成一条 Qwen2-VL 对话样本：
  - human 输入: "<image>\\n{instruction}。输出下一步的7个关节角度。"
  - gpt 输出: 动作前 7 维关节角度，逗号+空格分隔，保留 4 位小数
输出为 JSON 数组，写入数据集目录下的 qwen_train.json。

CLI: python -m vla_project.training.convert_to_qwen_format
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_DEFAULT_DATASET_DIR = "outputs/dataset/expert_multi_v1"
_TRAJECTORY_NAME = "trajectory_expert.jsonl"
_SPLIT_NAME = "episode_split.json"
_OUTPUT_NAME = "qwen_train.json"
_TASK_PROMPT_SUFFIX = "。输出下一步的7个关节角度。"


def _load_train_episodes(dataset_dir):
    """从 episode_split.json 加载 train 划分的 episode 集合。"""
    split_path = dataset_dir / _SPLIT_NAME
    if not split_path.is_file():
        raise FileNotFoundError(f"找不到划分文件: {split_path}")
    with open(split_path, encoding="utf-8") as fh:
        doc = json.load(fh)
    ids = doc.get("train")
    if not isinstance(ids, list) or not ids:
        raise ValueError("episode_split.json 缺少 train 列表")
    return set(ids)


def _format_action(action):
    """取前 7 维关节角度，逗号+空格分隔，4 位小数。"""
    return ", ".join(f"{float(v):.4f}" for v in action[:7])


def _build_record(row):
    """将一条轨迹帧转换为 Qwen2-VL 对话样本。"""
    instruction = row["instruction"]
    human_value = f"<image>\n{instruction}{_TASK_PROMPT_SUFFIX}"
    gpt_value = _format_action(row["action"])
    return {
        "image": row["image_path"],
        "conversations": [
            {"from": "human", "value": human_value},
            {"from": "gpt", "value": gpt_value},
        ],
    }


def run_convert(dataset_dir):
    """执行转换，返回生成的样本列表。"""
    dataset_dir = Path(dataset_dir)
    trajectory_path = dataset_dir / _TRAJECTORY_NAME
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"找不到轨迹文件: {trajectory_path}")

    train_episodes = _load_train_episodes(dataset_dir)

    records = []
    episode_seen = set()
    with open(trajectory_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["episode_idx"] not in train_episodes:
                continue
            records.append(_build_record(row))
            episode_seen.add(row["episode_idx"])

    if not records:
        raise ValueError("train 划分内没有可用的轨迹帧")

    output_path = dataset_dir / _OUTPUT_NAME
    output_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Qwen 格式转换完成：帧数={len(records)}，episode 数={len(episode_seen)}")
    print(f"  输出: {output_path}")
    return records


def main():
    parser = argparse.ArgumentParser(
        description="专家轨迹 → Qwen2-VL 微调数据 (qwen_train.json)"
    )
    parser.add_argument("--dataset-dir", default=_DEFAULT_DATASET_DIR)
    args = parser.parse_args()
    try:
        run_convert(args.dataset_dir)
    except Exception as exc:
        print(f"转换失败: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
