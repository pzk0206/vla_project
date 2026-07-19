# 生成图片与实验输出目录整理设计

## 目标

把主仓库中分散的生成图片统一收纳到 `outputs/`，并让后续运行直接写入新目录。图片对应的 JSON、JSONL、标注结果和汇总文件随实验目录一起迁移，避免图片与元数据失去关联。

本次只处理 `/home/pzk/vla_project` 主工作区，不处理 `.worktrees/` 中的独立实验副本。

## 目标目录

```text
outputs/
├── dataset/
├── probe/
├── probe_evaluations/
├── vlm_samples/
│   ├── default/
│   ├── 448/
│   ├── 448_multiseed_d020/
│   └── 448_calibration_validation_d020/
└── vlm_evaluations/
```

各目录职责：

- `dataset/`：Baseline 1 专家轨迹图片、动作记录和 episode 汇总。
- `probe/`：Stage 3 单次闭环探测图片及 trace。
- `probe_evaluations/`：Stage 3 批量评估的分批运行目录、失败图片、trace 和汇总。
- `vlm_samples/`：发送给 VLM 的离线样本图片、样本清单和诊断真值。
- `vlm_evaluations/`：方向预测、grounding 标注图、反投影结果和评估统计。

## 迁移映射

| 旧路径 | 新路径 |
| --- | --- |
| `dataset/` | `outputs/dataset/` |
| `probe_runs/` | `outputs/probe/` |
| `probe_eval_runs/` | `outputs/probe_evaluations/` |
| `vlm_eval_samples/` | `outputs/vlm_samples/default/` |
| `vlm_eval_samples_448/` | `outputs/vlm_samples/448/` |
| `vlm_eval_samples_448_multiseed_d020/` | `outputs/vlm_samples/448_multiseed_d020/` |
| `vlm_eval_samples_448_calibration_validation_d020/` | `outputs/vlm_samples/448_calibration_validation_d020/` |
| `vlm_eval_runs/` | `outputs/vlm_evaluations/` |

迁移采用移动而非复制。开始前必须检查目标路径不存在，任何冲突都应停止并报告，不覆盖已有文件。

## 代码与配置调整

- 修改 `sim_config.yaml` 中 dataset、probe、probe evaluation、VLM sample 和 VLM run 的输出根路径。
- 修改仍然硬编码旧目录的 Python 代码和测试，使其读取配置或使用新路径。
- 修改 `.gitignore`，统一忽略 `outputs/`；保留 `.worktrees/` 忽略规则。
- 修改 README、WORKLOG、BUGLOG 和当前有效说明中的运行路径。历史设计与历史实施计划保留当时路径，不进行全量改写，以免篡改历史记录；必要时在当前文档中说明新旧映射。
- 新生成记录中的 `image_path`、`annotated_path` 等路径字段必须直接写新路径。

## 已有记录路径修复

迁移后，递归处理被迁移目录中的 `.json` 和 `.jsonl`，仅替换已知旧路径前缀为对应新前缀。重点覆盖 `image_path`、`annotated_path` 及汇总文件中的输出目录字段。

路径修复后逐条解析所有 JSON/JSONL。迁移前已经存在的悬空图片引用要形成可诊断计数，迁移前后集合必须完全一致；迁移不得新增悬空引用。无法识别的旧路径不静默修改，而是列出后人工确认。

## 安全与回滚

- 实施前记录各源目录的文件数量、图片数量和图片文件总字节数。
- 先完成代码和配置修改，再执行目录迁移及记录路径修复。
- 不删除 `.worktrees/`，不清理任何实验批次，不合并同名实验。
- 使用 Git 管理代码与文档变更；生成数据被 Git 忽略，因此迁移前后要用数量、大小和路径完整性校验保证数据未丢失。
- 如果验证失败，在本次操作结束前按迁移映射反向移动目录并恢复记录路径。

## 验证标准

1. 原有 8 个生成输出目录均已迁移，旧路径不再存在。
2. 迁移前后文件总数、图片总数和图片文件总字节数一致。JSON/JSONL 因路径字符串变长，文件大小允许变化。
3. 主仓库所有生成图片都位于 `outputs/`；`.worktrees/` 明确排除。
4. 所有迁移后的 JSON、JSONL 均可解析；迁移前后的历史悬空图片引用集合完全一致，没有新增缺失。
5. `rg` 在有效代码、配置和当前文档中找不到应淘汰的输出根路径。
6. 配置契约测试、数据采集测试、Stage 3 测试和 VLM 评估相关测试通过。
7. 用最小临时配置执行一次无付费 API 的生成 smoke test，证明新图片写入 `outputs/`，且不会重新创建旧目录。

## 非目标

- 不处理 `.worktrees/` 中的图片。
- 不删除、压缩或去重历史实验图片。
- 不调用 Qwen API，不产生付费评估请求。
- 不改变控制算法、采样逻辑或实验结论。
