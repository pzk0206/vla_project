# Stage 3 Batch Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增可复现、可诊断的 heuristic 批量 probe 评估，并完成 20 个 episode 的第一批实验。

**Architecture:** 将 `stage3_probe.py` 的单次闭环提取为返回摘要的 `run_probe_episode`；`evaluate_probe.py` 负责批次目录、错误隔离、汇总统计和成功图片清理。单次入口与批量入口共享控制循环。

**Tech Stack:** Python 3、unittest、PyBullet、OpenCV、PyYAML、JSONL。

## Global Constraints

- 默认只运行 `probe.mode: heuristic`，不调用真实 API。
- 成功 episode 删除步骤图片但保留 trace；失败和异常保留图片。
- episode 级异常不得中断整批运行。
- 先运行单元测试和 3 次 smoke test，再运行 20 次评估。

---

### Task 1: 抽取可复用单次 probe

**Files:** Modify `stage3_probe.py`; Test `tests/test_stage3_probe.py`

**Interfaces:** Produces `summarize_probe_trace(rows, episode_idx, random_seed, trace_path) -> dict` and `run_probe_episode(config, episode_idx, episode_dir, random_seed) -> dict`.

- [ ] 写失败测试：给定两行 trace，摘要必须包含 success、步数、首末距离、方向计数、距离增大次数和最后 block 坐标。
- [ ] 运行 `conda run -n vla_env python -m unittest tests.test_stage3_probe -v`，确认因函数不存在失败。
- [ ] 实现 `summarize_probe_trace`，再把现有 `main` 循环移入 `run_probe_episode`；函数设置 `random.seed(random_seed)`、使用独立 `episode_dir`、在 finally 中清理 PyBullet，并返回摘要。
- [ ] `main()` 加载配置后调用 `run_probe_episode(config, 0, probe.output_dir, None)`，保持原命令可用。
- [ ] 重跑测试并确认通过。

### Task 2: 实现批次纯逻辑与图片策略

**Files:** Create `evaluate_probe.py`; Create `tests/test_evaluate_probe.py`

**Interfaces:** Produces `aggregate_probe_summaries(summaries, config_snapshot) -> dict`, `cleanup_success_images(episode_dir, success, enabled)`, `make_error_summary(...)`.

- [ ] 写失败测试，覆盖成功率、错误计数、均值/中位数/max、方向合计、失败索引，以及成功删除 JPG/保留 trace、失败保留 JPG。
- [ ] 运行 `conda run -n vla_env python -m unittest tests.test_evaluate_probe -v` 确认 RED。
- [ ] 使用 `statistics`、`collections.Counter` 和 `pathlib.Path.glob` 实现最小逻辑。
- [ ] 重跑测试确认 GREEN。

### Task 3: 实现批量运行入口与配置

**Files:** Modify `evaluate_probe.py`; Modify `sim_config.yaml`; Modify `tests/test_config_contract.py`

**Interfaces:** Consumes `probe_evaluation.{num_episodes,output_dir,save_failure_images_only,random_seed}`; writes timestamped batch directory, `episode_summary.jsonl`, `probe_eval_summary.json`.

- [ ] 先扩展配置契约测试，要求评估次数和种子有效、输出目录非空。
- [ ] 在 YAML 添加设计文档规定的四个配置字段。
- [ ] 实现 `main()`：拒绝非 heuristic mode；逐 episode 调用 `run_probe_episode`；异常转 error summary 并继续；逐行写摘要；最终写缩进 JSON 汇总。
- [ ] 运行全部单元测试。

### Task 4: Smoke test 与 20 次评估

**Files:** Generated `probe_eval_runs/**` (ignored); Inspect summary JSON.

- [ ] 临时以配置覆盖或函数调用运行 3 episode smoke test，验证目录隔离、成功图片删除和 JSON 字段。
- [ ] 使用默认 `num_episodes: 20` 运行 `conda run -n vla_env python evaluate_probe.py`。
- [ ] 校验 episode_summary 恰有 20 行，汇总计数相加为 20，成功率与行级结果一致。

### Task 5: 同步文档并完整验证

**Files:** Modify `README.md`, `WORKLOG.md`, `vla_robotic_study_plan.md`.

- [ ] README 增加批量评估命令、配置和输出说明。
- [ ] WORKLOG 记录 20 次的配置、成功率、距离、步数和失败类型。
- [ ] 学习计划把状态更新为“首批 20 次完成/仍待 50 次门槛实验”，不得夸大 Stage 3 完成。
- [ ] 运行 `conda run -n vla_env python -m unittest discover -s tests -v`。
- [ ] 运行 `conda run -n vla_env python -m py_compile control_arm.py stage3_probe.py evaluate_probe.py tests/*.py`。
- [ ] 运行 `git diff --check` 并审查最终 diff。
