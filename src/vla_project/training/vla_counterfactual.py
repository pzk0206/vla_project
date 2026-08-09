"""同一保存场景只交换红蓝指令的成对 VLA 反事实评估。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from vla_project.output_paths import resolve_new_output_directory
from vla_project.simulation.control_arm import load_config
from vla_project.training.vla_rollout import (
    _sha256,
    load_bound_vla_model,
    load_validation_summaries,
    run_saved_scene_branch,
)


def wilson_interval(successes, total, z=1.959963984540054):
    if total <= 0:
        return {"lower": None, "upper": None}
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return {
        "lower": max(0.0, center - margin),
        "upper": min(1.0, center + margin),
    }


def audit_instruction_encoding(model):
    """记录红蓝指令在冻结文本编码器中的 token 与向量可区分性。"""
    texts = ["悬停在红色积木上方", "悬停在蓝色积木上方"]
    tokenized = model.text_encoder.tokenize(texts)
    input_ids = tokenized["input_ids"].detach().cpu().tolist()
    attention_mask = tokenized["attention_mask"].detach().cpu().tolist()
    embeddings = (
        model.encode_texts(texts).detach().cpu().numpy().astype(np.float32)
    )
    difference = embeddings[0] - embeddings[1]
    denominator = float(np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
    cosine = (
        float(np.dot(embeddings[0], embeddings[1]) / denominator)
        if denominator > 0
        else None
    )
    return {
        "texts": texts,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_ids_equal": input_ids[0] == input_ids[1],
        "embeddings_equal": bool(np.array_equal(embeddings[0], embeddings[1])),
        "embedding_l2": float(np.linalg.norm(difference)),
        "embedding_max_abs_diff": float(np.max(np.abs(difference))),
        "embedding_cosine": cosine,
        "embedding_sha256": [
            hashlib.sha256(row.tobytes()).hexdigest() for row in embeddings
        ],
    }


def _pair_metrics(pair):
    red = pair["red"]
    blue = pair["blue"]
    valid = red.get("system_error") is None and blue.get("system_error") is None
    follows = valid and bool(red.get("success")) and bool(blue.get("success"))
    switches = (
        valid
        and red.get("nearest_block") == "red"
        and blue.get("nearest_block") == "blue"
    )
    return valid, follows, switches


def _summarize_subset(pairs):
    metrics = [_pair_metrics(pair) for pair in pairs]
    valid_count = sum(valid for valid, _, _ in metrics)
    follow_count = sum(follows for _, follows, _ in metrics)
    switch_count = sum(switches for _, _, switches in metrics)
    return {
        "pair_count": len(pairs),
        "valid_count": valid_count,
        "excluded_count": len(pairs) - valid_count,
        "paired_instruction_follow_count": follow_count,
        "paired_instruction_follow_rate": (
            follow_count / valid_count if valid_count else None
        ),
        "preference_switch_count": switch_count,
        "preference_switch_rate": switch_count / valid_count if valid_count else None,
    }


def summarize_pairs(pairs):
    overall = _summarize_subset(pairs)
    valid_count = overall["valid_count"]
    follow_count = overall["paired_instruction_follow_count"]
    interval = wilson_interval(follow_count, valid_count)
    groups = {
        "red_to_blue": _summarize_subset(
            [pair for pair in pairs if pair.get("original_target") == "red"]
        ),
        "blue_to_red": _summarize_subset(
            [pair for pair in pairs if pair.get("original_target") == "blue"]
        ),
    }
    valid_pairs = [pair for pair in pairs if _pair_metrics(pair)[0]]
    red_success_count = sum(bool(pair["red"].get("success")) for pair in valid_pairs)
    blue_success_count = sum(bool(pair["blue"].get("success")) for pair in valid_pairs)
    supports = (
        valid_count >= 40
        and interval["lower"] is not None
        and interval["lower"] > 0.5
    )
    return {
        "num_pairs": len(pairs),
        "valid_pair_count": valid_count,
        "excluded_pair_count": len(pairs) - valid_count,
        "paired_instruction_follow_count": follow_count,
        "paired_instruction_follow_rate": overall[
            "paired_instruction_follow_rate"
        ],
        "paired_instruction_follow_wilson_95": interval,
        "preference_switch_count": overall["preference_switch_count"],
        "preference_switch_rate": overall["preference_switch_rate"],
        "red_instruction_success_count": red_success_count,
        "red_instruction_success_rate": (
            red_success_count / valid_count if valid_count else None
        ),
        "blue_instruction_success_count": blue_success_count,
        "blue_instruction_success_rate": (
            blue_success_count / valid_count if valid_count else None
        ),
        "groups": groups,
        "recognition_rule": {
            "minimum_valid_pairs": 40,
            "required_wilson_lower_bound": 0.5,
            "pair_success_definition": "red branch reaches red and blue branch reaches blue",
        },
        "supports_red_blue_instruction_recognition": supports,
    }


def run_paired_counterfactual(
    *,
    checkpoint_path,
    config_path,
    dataset_dir,
    output_dir,
    max_pairs=None,
    max_steps=200,
    project_root_override=None,
):
    output_dir = resolve_new_output_directory(
        output_dir,
        allowed_root="outputs/rollout",
        project_root_override=project_root_override,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata, action_mean, action_std = load_bound_vla_model(
        checkpoint_path, dataset_dir, device
    )
    instruction_encoding = audit_instruction_encoding(model)
    config = load_config(config_path)
    summaries = load_validation_summaries(dataset_dir, max_pairs)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = []
    branch_records = []
    for summary in summaries:
        branches = {}
        for color in ("red", "blue"):
            branch_result = run_saved_scene_branch(
                model=model,
                device=device,
                action_mean=action_mean,
                action_std=action_std,
                config=config,
                episode_summary=summary,
                instruction_target=color,
                max_steps=max_steps,
            )
            branches[color] = branch_result
            branch_records.append(branch_result)
        valid, follows, switches = _pair_metrics(
            {"red": branches["red"], "blue": branches["blue"]}
        )
        pairs.append(
            {
                "episode_idx": summary["episode_idx"],
                "random_seed": summary["random_seed"],
                "original_target": summary["target_block"],
                "counterfactual_target": (
                    "blue" if summary["target_block"] == "red" else "red"
                ),
                "valid": valid,
                "paired_instruction_follow": follows,
                "preference_switch": switches,
                "red": {
                    key: value
                    for key, value in branches["red"].items()
                    if key != "trace"
                },
                "blue": {
                    key: value
                    for key, value in branches["blue"].items()
                    if key != "trace"
                },
            }
        )

    summary = summarize_pairs(pairs)
    summary["instruction_encoding_audit"] = instruction_encoding
    summary["recognition_rule"]["requires_distinct_instruction_embeddings"] = True
    summary["supports_red_blue_instruction_recognition"] = bool(
        summary["supports_red_blue_instruction_recognition"]
        and not instruction_encoding["embeddings_equal"]
    )
    summary.update(
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "dataset_dir": str(dataset_dir),
            "dataset_manifest_sha256": metadata["dataset_manifest_sha256"],
            "trajectory_expert_sha256": metadata["trajectory_expert_sha256"],
            "episode_summary_sha256": metadata["episode_summary_sha256"],
            "episode_split_sha256": metadata["episode_split_sha256"],
            "max_steps": max_steps,
            "sim_steps_per_action": 60,
        }
    )
    with (output_dir / "per_branch.jsonl").open("w", encoding="utf-8") as handle:
        for record in branch_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (output_dir / "per_pair.jsonl").open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="VLA 红蓝成对反事实评估")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="sim_config.yaml")
    parser.add_argument("--dataset-dir", default="outputs/dataset/expert_multi_v2")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args(argv)
    run_paired_counterfactual(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        max_pairs=args.max_pairs,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
