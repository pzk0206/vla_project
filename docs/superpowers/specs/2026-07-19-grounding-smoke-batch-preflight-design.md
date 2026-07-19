# Grounding Smoke 整批无 API 动态预检设计

**日期：** 2026-07-19  
**状态：** 已获用户批准，待书面规格复核  
**范围：** 只增加整批动态预检；不修改成功评分、目标保持算法、固定案例或真实 API 配置

## 背景

首轮真实 `grounding_smoke` 在三个固定案例全部完成动态资格检查之前，就按案例顺序进入
了闭环。实际结果是：

- `52-left` 首帧可见率为 `270/378=0.7143`，低于 `0.75`，在 API 前停止；
- `53-right` 进入闭环并产生1次真实 API 调用；
- `54-front` 起始姿态误差为 `0.053522m`，高于 `0.005m`，批次异常退出。

这说明静态配置和自动测试通过，并不能证明整批动态案例具备运行资格。后置案例尚未检查
时，前置案例已经可能产生付费请求。本设计在任何真实请求之前增加完整的两阶段门禁。

## 目标与非目标

目标：

1. 在第一条真实 API 请求之前，对三个固定案例全部完成起始姿态和首帧可见率检查。
2. 保存三个案例的结构化预检证据，不能遇到第一个失败就停止检查。
3. 任意案例不合格时拒绝整个批次，并证明 API 和控制循环调用次数都为0。
4. 三个案例全部合格时，保持现有闭环行为和固定 `3/3` 验收契约。

非目标：

- 不跳过不合格案例，也不自动替换 seed。
- 不修改 `success`、预测 `stop` 或事后真值评分规则。
- 不修改最近可靠目标、4步 held 上限或可见率阈值。
- 不在本轮解决普通闭环执行期间的案例级异常汇总。
- 不运行第二次真实 smoke；实现完成后只做无 API 验证。

## 方案选择

采用“两阶段重新建场景”：第一阶段逐个建立并销毁三个 PyBullet 场景，只做动态预检；
三个全部合格后，第二阶段再按现有流程重新建立场景并运行闭环。

没有选择同时保留三个 physics client，因为它会扩大连接状态和资源清理边界。没有选择
PyBullet 状态快照，因为快照生命周期与恢复语义会增加与本目标无关的复杂度。重复初始化
三次仿真的成本可接受，并能保持每个案例隔离。

## 组件与接口

### 单案例预检

在 `src/vla_project/vlm/grounding_smoke/runner.py` 中增加一个内部函数：

```python
def preflight_smoke_case(config, smoke_config, case):
    """无 API 地验证单个案例的起点和首帧可见率。"""
```

函数负责完整的单案例资源生命周期：

1. 以案例 seed 初始化随机状态并连接 `DIRECT` PyBullet。
2. 创建世界、加载并稳定红块。
3. 用红块真值构造固定10cm起点；真值只用于资格检查和场景搭建。
4. 重置机械臂，计算请求起点与实际末端位置的三维距离。
5. 构建固定 smoke 相机并捕获 RGB/segmentation；预检只消费 segmentation。
6. 计算红块可见像素和相对 `visibility_reference_pixels` 的比例。
7. 在 `finally` 中断开 physics 连接。

函数不得构建 grounding 依赖，不得调用 `call_openai_compatible_api()`，也不得创建 episode
闭环目录。它返回包含以下字段的字典：

```text
episode_idx
seed
start_direction
qualified
rejection_reason
requested_start_ee_pos
actual_start_ee_pos
start_pose_error
block_visible_pixels
block_reference_pixels
block_visibility_ratio
camera_eye
```

`rejection_reason` 只有以下稳定值：

- `start_pose_error`：起点误差严格大于 `balanced_pose_tolerance`；
- `visibility_below_threshold`：起点合格但可见率严格小于
  `clear_visibility_threshold`；
- `null`：两个条件均合格。

即使起点不合格，函数也继续捕获首帧并记录可见率，从而让每个案例产生完整、可比较的
预检记录。非预期的仿真或图像捕获异常仍向上传播，但 `finally` 必须断开连接。

### 整批门禁

`run_smoke_batch()` 保持现有静态配置和冻结校准检查，然后创建唯一运行目录。之后：

1. 对 `build_smoke_cases()` 返回的三个案例全部调用 `preflight_smoke_case()`。
2. 将批次级证据写入 `<run_dir>/smoke_preflight.json`。
3. 如果任意记录的 `qualified=false`，抛出明确的 `SmokePreflightError`，错误中列出失败
   案例和拒绝原因；不进入现有 episode 循环。
4. 只有全部合格，才执行现有三个案例的真实闭环。

`smoke_preflight.json` 使用以下顶层结构：

```json
{
  "num_cases": 3,
  "qualified_count": 1,
  "passed": false,
  "cases": []
}
```

该文件是预检权威证据。预检拒绝时不创建 `episode_summary.jsonl` 或
`smoke_summary.json`，以免把“尚未开始闭环”混同为在线 episode 失败。本轮不承担普通
闭环异常的完整 batch summary；该问题继续由 `BUG-006` 的后续工作处理。

## 数据与安全边界

- 预检允许读取 PyBullet 红块真值来构造固定起点和计算 segmentation 资格；这些数据不
  传入第二阶段的 grounding 或动作计算接口。
- 预检函数不接收 API callable，也不构建包含 `ground()` 的 `SmokeDependencies`。
- 任意案例不合格时，`run_control_loop()` 与 `call_openai_compatible_api()` 必须均为0次。
- 仍使用固定 seeds `[52, 53, 54]` 和方向 `[left, right, front]`，不静默改变实验对象。
- 每个预检案例和每个正式 episode 都拥有独立 physics 连接，并在 `finally` 中断开。

## 错误处理

新增 `SmokePreflightError(RuntimeError)`，只表示整批动态门禁未通过。错误消息包含
`episode_idx`、seed、方向和拒绝原因，便于 CLI 直接显示。预检证据必须在抛错前完成
JSON 写入。

非预期异常不伪装成资格拒绝原因；它保持原始异常类型向上传播。无论资格失败还是异常，
当前案例的 physics 连接都必须释放。

## 测试策略

使用 TDD 扩展 `tests/vlm/grounding_smoke/test_runner.py`：

1. 单案例起点误差超过容差时返回 `start_pose_error`，同时仍记录首帧可见率。
2. 起点合格但可见率低于阈值时返回 `visibility_below_threshold`。
3. 等于起点容差和等于可见率阈值时均视为合格，保护包含性边界。
4. 每个单案例无论合格或异常都断开 physics 连接。
5. 整批任意案例失败时，三个预检均执行，写出 `smoke_preflight.json`，随后抛
   `SmokePreflightError`。
6. 预检拒绝时，控制循环和 API 均未调用，且不创建 episode 或 smoke summary。
7. 三个案例全部合格时，才运行三次现有控制循环；现有批次汇总行为保持不变。

测试全部通过 mock/fake 边界运行，不访问网络或真实 API。完成后运行 grounding smoke
定向测试、配置/包契约测试和全量测试，并执行 `compileall` 与 `git diff --check`。

## 验收标准

- 当前固定案例执行 CLI 时，在产生任何真实请求之前完成三个案例预检。
- 当前已知结果应为整批预检失败：至少记录 `52-left` 的
  `visibility_below_threshold` 和 `54-front` 的 `start_pose_error`。
- 该无 API 验证的真实 API 调用数为0。
- `smoke_preflight.json` 包含恰好三个案例及可审计指标。
- 不修改成功评分、目标保持状态机、固定案例或 API 上限。
- 所有新增及既有测试通过，源码编译和补丁格式检查通过。
