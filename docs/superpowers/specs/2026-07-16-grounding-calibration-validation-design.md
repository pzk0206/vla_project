# Grounding 定位独立校准验证设计

## 目标

验证五位置 grounding 实验中发现的固定偏差能否泛化到从未参与拟合的新红块位置。当前 seeds 42–46 的清晰样本只作为校准集；新 seeds 47–51 只作为验证集。验证过程中禁止使用新样本重新估计补偿。

本轮仍属于离线定位验收，不把补偿写入在线控制器，也不进入机械臂闭环。

## 为什么必须独立验证

当前 15 张清晰样本的平均有符号误差为：

```text
signed_error_x_mean = -0.02492227406480192 m
signed_error_y_mean = +0.019467343494422532 m
```

因此候选补偿为：

```text
correction_x = +0.02492227406480192 m
correction_y = -0.019467343494422532 m
```

在同一批校准数据上减去均值后，残差下降只能证明补偿具有潜力，不能证明它对新位置有效。只有在独立验证集上保持低误差，才可以考虑把补偿接入后续控制实验。

## 数据划分

校准集固定为现有结果：

```text
seeds: 42–46
样本目录: vlm_eval_samples_448_multiseed_d020/
反投影结果: vlm_eval_runs/grounding_qwen3_vl_flash_distance20_448_multiseed_v4/backprojection/backprojection_results.jsonl
```

验证集使用：

```text
seeds: 47–51
每个 seed: left/right/front/back 四种姿态
总样本数: 20
```

验证集使用独立样本目录、grounding 运行目录和反投影目录，不覆盖校准集。Stage 3 的 `probe_evaluation.random_seed` 继续保持 42；VLM 分层采样新增独立 seed 列表，避免改变既有闭环评估口径。

## 组件与数据流

### 1. 显式 VLM seed 列表

`vlm_evaluation.stratified_seeds` 保存 `[47, 48, 49, 50, 51]`。分层均衡采样优先使用该列表，而不是依赖 Stage 3 的全局起始 seed。采样器仍为每个 seed 生成四个方向，继续保存 segmentation 可见率诊断。

### 2. 校准参数提取

新增离线校准验证脚本，只从旧校准结果中筛选 `visibility_group == "clear"` 且定位有效的记录，计算 X/Y 平均有符号误差，并将补偿定义为其相反数。

脚本同时保存校准来源路径、校准样本数、原始偏差和最终补偿，保证结果可以追溯。校准集不得为空，也不得包含无效定位记录。

### 3. 独立验证

新验证图片先经过既有 Qwen grounding 和原始反投影，得到未补偿的验证结果。校准验证脚本随后对每条验证记录执行：

```text
corrected_x = predicted_x + correction_x
corrected_y = predicted_y + correction_y
corrected_error_xy = distance((corrected_x, corrected_y), true_xy)
```

原始预测、原始误差、补偿后坐标和补偿后误差必须同时保留，不能覆盖原始证据。

### 4. 可见率隔离

固定补偿只在 `clear` 验证样本上验收。`partial` 和 `severe` 样本仍应用相同补偿并报告结果，用于观察补偿是否会恶化遮挡样本，但它们不参与固定偏差通过判定。

## 输出

输出目录包含：

- `calibration.json`：校准来源、清晰校准样本数、X/Y 偏差和冻结补偿；
- `corrected_validation_results.jsonl`：逐样本原始与补偿后坐标、误差、方向、seed 和可见率；
- `calibration_validation_summary.json`：整体及逐可见率分组的原始/补偿后指标和通过判定。

摘要至少报告：

- 校准样本数和验证样本数；
- 验证集合法定位数和失败数；
- clear/partial/severe 各组数量；
- clear 组原始与补偿后的 mean、median、max；
- clear 组 `≤0.03m` 的样本数和比例；
- 固定补偿是否通过验收。

## 验收规则

只有同时满足以下条件，固定补偿才算通过：

1. seeds 47–51 共生成 20 张可读图片和 20 条匹配 diagnostics；
2. Qwen 对每条样本保存合法框或明确错误，不静默丢失；
3. 验证集中至少存在 1 条 `visibility_group == "clear"` 的样本；
4. 所有 clear 验证样本都必须定位有效，不能把定位失败的 clear 样本排除后再计算通过率；
5. 所有 clear 验证样本的补偿后误差均 `≤0.03m`；
6. 验证阶段没有重新计算或修改校准补偿；
7. partial/severe 样本单独报告，不用于掩盖 clear 组失败。

如果 clear 组任一样本超过 3cm，则本轮固定补偿不通过，不接入在线控制，并根据残差是否仍有方向性决定扩大校准模型还是回到 grounding 定位方法。

## 错误处理与断点续跑

- seed 列表必须包含五个唯一整数；重复、布尔值或空列表立即报错；
- 新样本和旧校准结果的 `sample_id` 不得重叠；
- grounding 继续按 `sample_id` 断点续跑，避免重复付费；
- 校准或验证输入存在重复 ID、缺失真值、非法可见率或无效坐标时明确失败；
- 原始反投影结果保持只读，补偿结果写入独立文件。

## 测试范围

- 配置契约测试固定验证 seeds 47–51 和独立输出目录；
- 采样测试验证显式 seed 列表产生 20 个唯一 ID、四方向各 5 张；
- 校准测试验证只使用 clear 校准记录，并正确取有符号误差的相反数；
- 验证测试证明新结果不会反向影响补偿参数；
- 分组摘要测试覆盖通过、超过 3cm、遮挡样本和无效定位四种情况；
- 最终完整测试继续使用 `conda run -n vla_env python -m unittest discover -s tests -v`。
