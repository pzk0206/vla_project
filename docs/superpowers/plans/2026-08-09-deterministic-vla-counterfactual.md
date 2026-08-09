# 确定性 VLA 红蓝成对反事实实施计划

**目标：** 在通过门禁的 `expert_multi_v2` 上重新训练 regression VLA，并用相同保存场景、
只交换红蓝指令的成对 rollout，生成足以判断模型是否识别红蓝的可审计证据。

**依据：** 已批准的完整性恢复设计
`docs/superpowers/specs/2026-08-09-vla-data-integrity-recovery-design.md` 第6–7节。

**约束：** 继续使用斜视图；不复用非空训练/rollout目录；旧 v1 checkpoint 和结果只读；
所有行为修改先写失败测试；场景重放误差超过1mm时该对不可比较；系统错误不进入有效对分母，
但必须单独计数；在正式成对结果前不恢复旧70%结论。

## Task 1：expert_multi_v2 分层 Episode Split

**文件：**
- 修改 `src/vla_project/simulation/split_episodes.py`
- 新增/修改 `tests/simulation/test_split_episodes.py`

1. 写红测：v2 按 `target_block` 分为红蓝各150条，train/val 必须各为125/25；每个任务内按
   保存的目标块 initial scene-state XY 分层；split 保存 schema、任务计数、源摘要 SHA-256、
   数据 manifest SHA-256 和生成器版本。
2. 保留 expert_v1 的原250/50行为；拒绝缺 scene state、目标不一致、重复 episode 和非300条。
3. 运行定向测试后，在真实 v2 目录生成一次 `episode_split.json`，核对250/50、125/125与
   25/25，并记录 SHA-256。

## Task 2：VLA 训练语义与证据元数据

**文件：**
- 修改 `src/vla_project/training/vla_train.py`
- 修改 `src/vla_project/training/dataset.py`
- 新增 `tests/training/test_vla_train.py`
- 修改 `tests/training/test_dataset.py`

1. 写红测：只接受 `expert_multi_v2` 和对应 split；动作统计仅用250个 train episode；checkpoint
   保存 dataset manifest、trajectory、split SHA-256、action mean/std、训练 episode 列表、模型
   与文本编码器标识；新运行目录必须 `_v2` 且不可复用。
2. 固定 Python/NumPy/Torch/DataLoader seed，记录实际 device；验证指令与 target block 一致。
3. 先运行10 episode overfit，再运行250 episode full；新目录分别为
   `vla_regression_overfit_10_v2` 与 `vla_regression_full_v2`。

## Task 3：保存场景的确定性重放

**文件：**
- 重构 `src/vla_project/training/vla_rollout.py`
- 新增 `tests/training/test_vla_rollout.py`

1. 写红测：从 summary `initial_scene_state` 恢复红蓝 position/orientation、7维 home joints 与
   camera eye；不调用随机 `load_block`、`load_second_block` 或 `sample_camera_eye`。
2. 加载后落稳并捕获实际状态，红蓝位置/姿态与保存状态误差不超过1mm；否则写
   `scene_replay_mismatch` 且不执行模型动作。
3. checkpoint 在连接 PyBullet 前核对 dataset/split hash、action stats 和 v2 metadata。
4. 每步记录预测归一化动作、反归一化关节目标、terminate score、EE、红蓝目标距离和停止原因；
   正确保存实际执行步数，所有 body 与连接在 `finally` 中清理。

## Task 4：成对反事实评估器

**文件：**
- 修改 `src/vla_project/training/vla_rollout.py` 或新增同目录共享模块
- 新增 `src/vla_project/training/vla_counterfactual.py`
- 新增 `tests/training/test_vla_counterfactual.py`
- 修改 `pyproject.toml`

1. 每个 val episode 从同一保存状态分别运行红指令与蓝指令；两次只允许 instruction 与
   scoring target 不同，场景、相机、home pose、checkpoint 和控制协议必须相同。
2. 每个分支报告到红/蓝目标的最终距离、成功、最近目标、系统错误和逐步 trace；每对报告
   两分支是否都跟随指令、偏好是否切换、是否有效可比较。
3. 总结报告原任务成功率、红指令成功率、蓝指令成功率、有效对数、排除数、
   `paired_instruction_follow_rate`、红→蓝/蓝→红分组和 Wilson 95% 区间。
4. 判定规则：只有有效对中两条分支都分别到达所指颜色才算 instruction-follow；无切换、两边
   都失败或固定去同一块均算失败。有效对为0时结论必须是证据不足。

## Task 5：实验执行与结论

1. 全量测试、`compileall`、`git diff --check` 通过到既有允许基线后，执行 overfit/full GPU训练。
2. 用 best checkpoint 对固定50个 val episode 做100条成对 rollout，最多200步，每步60个
   仿真子步；新输出目录 `outputs/rollout/vla_regression_full_v2_paired/`。
3. 核对逐对 JSONL、总体 JSON、checkpoint/dataset/split hash、方向分组和置信区间。
4. 只有报告证据支持时才声明“能识别红蓝”；若不支持，则明确区分不会用语言、控制能力不足、
   scene replay错误和样本不确定性，不用旧70%覆盖新结果。
5. 更新 PROJECT_OVERVIEW、CURRENT_STATUS、WORKLOG、BUGLOG 和 README，提交实验恢复点。
