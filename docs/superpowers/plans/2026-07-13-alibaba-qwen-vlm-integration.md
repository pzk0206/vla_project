# Alibaba Qwen VLM Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 接入阿里云百炼华北 2（北京）的 `qwen3-vl-flash`，让模型仅根据相机图像和语言指令输出离散方向，并与当前 50/50 成功的 heuristic 基线进行可复现对比。

**Architecture:** 保留现有 `direction -> 世界坐标单步目标 -> IK -> PyBullet` 控制层，只替换决策来源。先用 heuristic trace 生成带标准方向标签的离线样本，验证 VLM 对方向语义和相机视角的理解，再进入在线闭环，避免一开始把视觉判断错误和机械臂控制错误混在一起。

**Tech Stack:** Python 3.10、PyBullet、OpenCV、阿里云百炼 OpenAI 兼容 Chat Completions API、`qwen3-vl-flash`、JSONL、unittest

## Global Constraints

- 阿里云地域固定为华北 2（北京）。
- Base URL 使用 `https://dashscope.aliyuncs.com/compatible-mode/v1`；如用户后续创建业务空间专属域名，仅替换环境变量，不修改控制代码。
- 模型固定为 `qwen3-vl-flash`，首轮不同时比较多个模型。
- API Key 只能通过环境变量传入，禁止写入 YAML、Python、Markdown、日志或 Git。
- VLM 在线决策不能读取 `block_pos`、`ee_pos` 等仿真真值；这些字段只用于离线评分和失败分析。
- heuristic 保持默认安全基线，不删除、不重写其控制逻辑。
- 第一轮 VLM 在线评估最多 20 个 episode；先通过单请求检查和 3 次 smoke test，避免无效调用造成费用浪费。
- 世界坐标控制参数保持基线：`max_control_steps=80`、`sim_steps_per_action=60`、`move_step_xy=0.03m`。

---

### Task 1: 百炼凭证与配置契约

**Files:**
- Modify: `sim_config.yaml`
- Modify: `tests/test_config_contract.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: 环境变量 `VLA_API_BASE_URL`、`VLA_API_KEY`、`VLA_MODEL_NAME`
- Produces: 可在不泄露密钥的前提下校验百炼 API 配置的项目约定

- [ ] **Step 1: 用户在阿里云百炼控制台完成外部准备**

开通华北 2（北京）模型服务并创建按量付费 API Key。不要在终端聊天、截图或项目文件中粘贴 Key。

- [ ] **Step 2: 在当前 shell 设置环境变量**

```bash
export VLA_API_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export VLA_API_KEY="<在本机终端填写真实 Key，不写入仓库>"
export VLA_MODEL_NAME="qwen3-vl-flash"
```

检查时只输出 SET/UNSET，不打印真实值：

```bash
for name in VLA_API_BASE_URL VLA_API_KEY VLA_MODEL_NAME; do
  if [ -n "${!name}" ]; then echo "$name=SET"; else echo "$name=UNSET"; fi
done
```

Expected:

```text
VLA_API_BASE_URL=SET
VLA_API_KEY=SET
VLA_MODEL_NAME=SET
```

- [ ] **Step 3: 先写配置契约测试**

在 `tests/test_config_contract.py` 增加：

```python
def test_probe_api_limits_are_valid(self):
    api = self.config["probe"]["api"]
    self.assertGreater(api["timeout_seconds"], 0)
    self.assertGreaterEqual(api["max_retries"], 0)
```

- [ ] **Step 4: 运行测试并确认先失败**

Run:

```bash
conda run -n vla_env python -m unittest tests.test_config_contract.ConfigContractTests.test_probe_api_limits_are_valid -v
```

Expected: 因缺少 `timeout_seconds` 而失败。

- [ ] **Step 5: 为 API 配置增加显式超时和重试上限**

在 `sim_config.yaml` 的 `probe.api` 下增加：

```yaml
    timeout_seconds: 30              # 单次百炼请求最长等待时间。
    max_retries: 2                   # 网络瞬时失败最多重试两次，避免无限扣费。
```

重新运行 Step 4 的命令，Expected: PASS。

- [ ] **Step 6: README 增加百炼配置说明**

只记录 Base URL、模型名和环境变量命令模板；Key 使用占位符，并明确 `.env` 若以后引入必须加入 `.gitignore`。

- [ ] **Step 7: 验证配置测试**

Run:

```bash
conda run -n vla_env python -m unittest tests.test_config_contract -v
```

Expected: 配置测试全部通过。

### Task 2: API 响应解析与错误可观测性

**Files:**
- Modify: `stage3_probe.py`
- Modify: `tests/test_stage3_probe.py`

**Interfaces:**
- Consumes: 百炼 OpenAI 兼容响应 `choices[0].message.content`
- Produces: 合法方向、原始回复、解析状态和明确的 API 错误类型

- [ ] **Step 1: 为合法、带解释和非法回复写测试**

在 `tests/test_stage3_probe.py` 增加：

```python
class ParseDirectionTests(unittest.TestCase):
    def test_exact_direction_is_valid(self):
        self.assertEqual(parse_direction("left"), "left")

    def test_direction_can_be_extracted_from_short_explanation(self):
        self.assertEqual(parse_direction("move right"), "right")

    def test_invalid_response_returns_none_instead_of_false_stop(self):
        self.assertIsNone(parse_direction("I cannot see the block"))
```

- [ ] **Step 2: 运行测试并确认非法回复测试失败**

Run:

```bash
conda run -n vla_env python -m unittest tests.test_stage3_probe.ParseDirectionTests -v
```

Expected: 第三个测试失败，因为当前实现把非法回复错误地当作 `stop`。

- [ ] **Step 3: 修改解析规则**

修改 `parse_direction()`：空回复或无合法方向时返回 `None`，不要伪装成 `stop`。`stop` 只允许来自模型明确输出。

- [ ] **Step 4: 增加 API 专用异常**

在 `stage3_probe.py` 定义：

```python
class InvalidModelResponseError(RuntimeError):
    """模型回复不包含任何允许的离散方向。"""
```

`call_openai_compatible_api()` 在解析结果为 `None` 时抛出该异常，并在异常消息中记录经过长度限制的原始回复，但绝不记录 API Key。

- [ ] **Step 5: 使用 YAML 超时并只重试网络瞬时错误**

把固定 `timeout=60` 改为 `api_config["timeout_seconds"]`。只对 `URLError` 和 HTTP 429/5xx 按 `max_retries` 重试；HTTP 400/401/403、JSON 结构错误和非法模型回复不重试。

- [ ] **Step 6: 为请求重试边界写 mock 测试**

使用 `unittest.mock.patch("stage3_probe.urllib.request.urlopen")` 验证：429 后成功会调用两次，401 只调用一次，非法回复不会变成 `stop`。

- [ ] **Step 7: 验证 Stage 3 单元测试**

Run:

```bash
conda run -n vla_env python -m unittest tests.test_stage3_probe -v
```

Expected: 全部通过，无真实网络请求。

### Task 3: 构建离线视觉方向评估集

**Files:**
- Create: `collect_vlm_eval_samples.py`
- Create: `tests/test_collect_vlm_eval_samples.py`
- Modify: `sim_config.yaml`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: heuristic 闭环中的相机图像、语言指令和 heuristic 标准方向
- Produces: `vlm_eval_samples/samples.jsonl` 及对应图片；该目录不提交 Git

- [ ] **Step 1: 定义样本 schema 测试**

每条样本至少包含：

```json
{
  "sample_id": "seed_42_step_000",
  "image_path": "vlm_eval_samples/images/seed_42_step_000.jpg",
  "instruction": "悬停在红色积木上方",
  "expected_direction": "right",
  "random_seed": 42,
  "control_step": 0,
  "camera_eye": [1.0, 0.4, 1.6]
}
```

测试应验证图片路径存在、方向属于五个合法值、样本 ID 唯一。`block_pos` 和 `ee_pos` 可以保存在单独诊断字段中，但不能进入发送给 VLM 的 prompt。

- [ ] **Step 2: 实现采样脚本**

复用 `run_probe_episode()` 或提取最小回调，在固定 seeds 42-46 的 heuristic 运行中保存图片与当步标准方向。每个 episode 均匀抽取最多 10 步，避免连续帧高度重复；目标约 50 个离线样本。

- [ ] **Step 3: 配置输出目录**

在 `sim_config.yaml` 增加：

```yaml
vlm_evaluation:
  sample_output_dir: "vlm_eval_samples"
  offline_num_episodes: 5
  max_samples_per_episode: 10
  online_smoke_episodes: 3
  online_eval_episodes: 20
```

- [ ] **Step 4: 忽略生成数据**

在 `.gitignore` 增加：

```gitignore
vlm_eval_samples/
vlm_eval_runs/
```

- [ ] **Step 5: 运行采样与校验**

Run:

```bash
conda run -n vla_env python collect_vlm_eval_samples.py
conda run -n vla_env python -m unittest tests.test_collect_vlm_eval_samples -v
```

Expected: 生成约 50 个有效样本，图片可读取，JSONL schema 全部通过。

### Task 4: 离线比较 Qwen 与 heuristic 标签

**Files:**
- Create: `evaluate_vlm_decisions.py`
- Create: `tests/test_evaluate_vlm_decisions.py`
- Modify: `sim_config.yaml`

**Interfaces:**
- Consumes: `vlm_eval_samples/samples.jsonl` 和对应图片
- Produces: `vlm_eval_runs/offline_*/predictions.jsonl`、`summary.json`

- [ ] **Step 1: 设计不泄露真值的 prompt**

发送内容只能包含图像和任务指令。Prompt 明确要求返回一个世界动作标签：

```text
你在控制 PyBullet 机械臂末端，使其移动到红色方块正上方。
只能输出一个标签：left、right、front、back、stop。
不要解释，不要输出标点或其他文字。
```

在离线校准阶段，需要额外用项目已知相机姿态解释这些标签对应的画面运动关系；禁止把当前样本的目标坐标或标准答案写入 prompt。

- [ ] **Step 2: 为汇总函数写测试**

汇总至少包含：总样本数、合法输出率、exact-match accuracy、每方向 precision/recall、混淆矩阵、平均 API 延迟、错误类型计数。

- [ ] **Step 3: 实现带断点续跑的离线评估**

每完成一个样本立即追加 `predictions.jsonl`。重新运行时按 `sample_id` 跳过已完成项，避免网络中断后重复付费。

- [ ] **Step 4: 先运行一个真实请求**

Run:

```bash
conda run -n vla_env python evaluate_vlm_decisions.py --limit 1
```

Expected: HTTP 成功、方向合法、响应和延迟写入 JSONL，日志不包含 API Key。

- [ ] **Step 5: 再运行 10 个样本检查方向语义**

Run:

```bash
conda run -n vla_env python evaluate_vlm_decisions.py --limit 10
```

Gate: 合法输出率必须为 100%；如果大量把同一对方向互换，先修正相机到世界动作的 prompt 校准，不进入在线控制。

- [ ] **Step 6: 完成约 50 个离线样本**

Run:

```bash
conda run -n vla_env python evaluate_vlm_decisions.py
```

Gate: exact-match accuracy 建议至少达到 80%，并且不存在某个关键方向 recall 为 0。未达到时只迭代 prompt 或相机校准，一次只改一个变量并记录结果。

### Task 5: 在线 3 次 smoke test

**Files:**
- Modify: `sim_config.yaml`
- Modify: `evaluate_probe.py`
- Modify: `stage3_probe.py`
- Modify: `tests/test_evaluate_probe.py`

**Interfaces:**
- Consumes: `probe.mode: api` 和百炼环境变量
- Produces: `vlm_eval_runs/online_smoke_*/` 中的逐步 trace、图片和摘要

- [ ] **Step 1: 为批量评估增加命令行覆盖参数**

使用 `argparse` 给 `evaluate_probe.py` 增加：

```text
--mode {heuristic,api}
--num-episodes N
--output-dir PATH
```

参数只覆盖本次内存中的配置副本，不回写 `sim_config.yaml`。在
`tests/test_evaluate_probe.py` 测试覆盖值正确传给 `run_batch()`，并验证非法
mode 被 argparse 拒绝。

- [ ] **Step 2: 让批量评估输出目录按决策来源分开**

heuristic 继续写 `probe_eval_runs/`；API/VLM 写 `vlm_eval_runs/`。摘要的 `config_snapshot` 必须记录模型名，但不能记录 API Key。

- [ ] **Step 3: trace 增加 VLM 诊断字段**

每步保留：

```text
decision_source
raw_response
response_valid
api_latency_seconds
api_attempt_count
direction
distance_before / distance_after / distance_delta
```

- [ ] **Step 4: 配置 smoke test**

临时运行配置：

```yaml
probe:
  mode: "api"
vlm_evaluation:
  online_smoke_episodes: 3
```

不要覆盖 heuristic 的 50 次正式基线结果。

- [ ] **Step 5: 运行 3 次在线闭环**

Run:

```bash
conda run -n vla_env python evaluate_probe.py --mode api --num-episodes 3 --output-dir vlm_eval_runs
```

Gate:

- 3 次都没有 HTTP/解析错误。
- 每一步都有图片、原始回复、方向和距离变化。
- 不出现连续非法回复导致的无限循环。
- 即使成功率不足，也必须能明确区分视觉方向错、API 错和控制执行错。

### Task 6: 20 次 Qwen 在线对比评估

**Files:**
- Modify: `evaluate_probe.py`
- Modify: `README.md`
- Modify: `docs/worklog/WORKLOG.md`
- Create: `docs/evaluation/vlm-vs-heuristic.md`

**Interfaces:**
- Consumes: 同一组 seeds 42-61、同一 PyBullet 控制参数、不同决策来源
- Produces: heuristic 与 `qwen3-vl-flash` 的公平对比报告

- [ ] **Step 1: 固定公平对比条件**

两组都使用：

```text
seeds = 42-61
max_control_steps = 80
sim_steps_per_action = 60
move_step_xy = 0.03m
success_distance = 0.03m
```

heuristic 可以复用已有相同 20 seeds 的修复后结果；如果代码路径发生影响控制行为的改动，则重新运行 heuristic 20 次。

- [ ] **Step 2: 运行 20 次 Qwen 闭环**

Run:

```bash
conda run -n vla_env python evaluate_probe.py --mode api --num-episodes 20 --output-dir vlm_eval_runs
```

- [ ] **Step 3: 生成对比指标**

报告至少包含：

```text
success_rate
final_distance mean / median / max
control_steps mean / median
distance_increase_steps
invalid_response_count
api_error_count
mean / p95 API latency
direction confusion（离线集）
```

- [ ] **Step 4: 写失败案例分析**

至少人工复盘 3 个失败或最低质量 episode；若 20 次全成功，则复盘最终距离最大的 3 个。每个案例说明：模型看到什么、输出什么、世界动作是什么、距离如何变化。

- [ ] **Step 5: 更新项目文档**

`README.md` 只写最终可复现指标；`WORKLOG.md` 写选择阿里百炼、离线校准、失败分析和结论；`docs/evaluation/vlm-vs-heuristic.md` 保存完整对比表。不要把“调用 VLM”夸写成端到端训练完成的 VLA。

### Task 7: 完整回归与阶段验收

**Files:**
- Test: `tests/`
- Verify: `README.md`
- Verify: `docs/worklog/WORKLOG.md`
- Verify: `docs/planning/vla_robotic_study_plan.md`
- Verify: `docs/evaluation/vlm-vs-heuristic.md`

**Interfaces:**
- Consumes: 所有实现和评估产物
- Produces: 是否进入专家数据规模化阶段的证据化结论

- [ ] **Step 1: 运行全部自动测试**

```bash
conda run -n vla_env python -m unittest discover -s tests -v
git diff --check
```

Expected: 全部测试通过、无空白错误。

- [ ] **Step 2: 检查密钥泄露**

只搜索 Key 的环境变量名称和常见前缀，不打印 shell 环境变量值：

```bash
git diff -- . ':!docs/superpowers/plans/2026-07-13-alibaba-qwen-vlm-integration.md'
rg -n "VLA_API_KEY\s*[:=]\s*['\"][^<]" . --glob '!probe_eval_runs/**' --glob '!vlm_eval_runs/**' --glob '!.git/**'
```

Expected: 项目文件中没有真实 Key。

- [ ] **Step 3: 验收判断**

满足以下条件即可认为“VLM 决策替换首版完成”：

- 离线合法输出率 100%，方向准确率至少 80%。
- 在线至少完成 20 个固定种子 episode。
- 有 heuristic 与 Qwen 的统一指标对比。
- 所有 API/解析/控制失败都能从 trace 中区分。
- README、WORKLOG 和学习计划与真实结果一致。

如果在线成功率低于 heuristic，不视为工程失败；只要差距能被方向混淆、视觉遮挡、响应错误或闭环振荡等证据解释，该阶段仍然形成了有效的 VLA 实验成果。

## Official Alibaba References

- OpenAI 兼容视觉接口：<https://help.aliyun.com/zh/model-studio/qwen-vl-compatible-with-openai>
- Base URL 与地域说明：<https://help.aliyun.com/zh/model-studio/base-url>
- 模型价格与免费额度：<https://help.aliyun.com/zh/model-studio/model-pricing>
