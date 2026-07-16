# Camera Backprojection Design

## Goal

把 Qwen 返回的红块归一化框转换成 PyBullet 工作平面上的世界坐标，为后续
`api_grounded` 闭环提供可测试的目标感知接口。第一版只验收当前固定的
`448 x 448` 正俯视单相机，但底层几何函数使用完整相机矩阵，不写死俯视比例。

本阶段只完成几何模块和离线验证，不接入在线闭环，不新增付费 API 调用，也不改变
已经验证过的 `direction -> IK -> PyBullet` 控制语义。

## Chosen Approach

使用逆视图投影矩阵，把二维像素反投影成世界空间射线，再求射线与固定工作平面
`z = plane_z` 的交点。

不采用以下方案：

- 不按相机高度和 FOV 做俯视比例换算，因为相机一旦倾斜，该公式就会失效。
- 不依赖 PyBullet depth buffer，因为本阶段要验证的是
  `VLM grounding + 已知工作平面`，而不是让仿真深度替代目标定位。

## Architecture

新增 `camera_geometry.py`，集中维护相机矩阵和纯几何函数。现有图像采集和新增离线
验证共用同一套矩阵构造逻辑，避免渲染使用一套参数、反投影又复制一套参数。

数据流如下：

```text
Qwen red_block box (0-1000)
-> normalized_box_center_to_pixel(...)
-> pixel_to_world_on_plane(...)
-> predicted_target_world = [x, y, plane_z]
-> offline only: compare with PyBullet block truth
```

### Geometry API

`camera_geometry.py` 提供以下接口：

```python
compute_camera_matrices(camera_config, camera_eye) -> (view_matrix, projection_matrix)

normalized_box_center_to_pixel(box, image_width, image_height) -> (u, v)

pixel_to_world_on_plane(
    pixel_xy,
    image_width,
    image_height,
    view_matrix,
    projection_matrix,
    plane_z=0.0,
) -> [x, y, z]

world_to_pixel(
    world_xyz,
    image_width,
    image_height,
    view_matrix,
    projection_matrix,
) -> (u, v)
```

`world_to_pixel()` 是诊断接口，用于往返测试和离线可视化，不参与目标预测。

### Coordinate Conventions

- VLM 框格式为 `[x1, y1, x2, y2]`，范围为 `0..1000`。
- 像素原点位于图片左上角，`u` 向右增大，`v` 向下增大。
- 像素转换到 OpenGL NDC 时，x 映射到 `[-1, 1]`，图片 y 轴需要翻转。
- PyBullet 返回的 view/projection 数组按 OpenGL column-major 解释，使用
  `reshape((4, 4), order="F")` 恢复矩阵。
- near/far clip 点分别使用 NDC z `-1` 和 `1`，再乘
  `(projection @ view)^-1` 得到世界射线。
- 第一版工作平面明确设为 `z = 0.0`；参数保留 `plane_z`，以后可校准为其他固定高度。
- 第一版只验收 `vlm_evaluation.camera_override` 的固定 `448 x 448` 正俯视相机，
  不声称斜视相机已经通过真实数据验收。

## Offline Truth Isolation

运行时预测链路不得读取 `block_pos`。为了评估几何误差，采样器把仿真真值写入与
VLM 输入 manifest 分离的诊断文件：

```text
vlm_eval_samples_448/samples.jsonl       # 可发送给 VLM，不含 block_pos
vlm_eval_samples_448/diagnostics.jsonl   # 仅离线评分使用
```

`diagnostics.jsonl` 每行至少包含：

```json
{
  "sample_id": "seed_42_d020_left",
  "block_pos": [0.0, 0.44, 0.05],
  "camera_eye": [0.0, 0.4, 3.0],
  "view_matrix": [16 numbers],
  "projection_matrix": [16 numbers]
}
```

离线评分按 `sample_id` 连接 grounding prediction 和 diagnostics。任何发送给 Qwen 的
代码仍然只读取 `samples.jsonl` 中的图片与语言指令。矩阵随样本保存，确保修改 YAML
相机配置后仍能复现旧图片的反投影结果；评估器不得用当前 YAML 猜测旧样本相机参数。

## Offline Evaluation Output

新增离线反投影评估入口，读取已有 grounding 结果，不发起 API 请求。每个有效样本
记录：

- `red_block_box`
- `box_center_pixel`
- `predicted_target_world`
- `true_block_pos`
- `localization_error_xy`
- `error_type` 和 `error_message`

汇总至少记录：

- 有效反投影数量与失败数量
- `localization_error_xy` 的 mean、median 和 max
- x/y 有符号误差均值，用于识别系统性轴翻转或偏置
- 错误类型计数

如果现有样本没有对应的 `diagnostics.jsonl`，评估器必须明确报错，不能从标准方向或
文件名猜测真值。

## Error Handling

以下输入必须抛出带具体原因的 `ValueError`：

- 图片宽高不是正数
- 框不是四个有限数值、越过 `0..1000`，或边界顺序非法
- 像素位于图片范围外
- view/projection matrix 形状非法、包含非有限值或不可逆
- 反投影齐次坐标无法归一化
- 射线与工作平面平行
- 平面交点位于相机射线反方向

离线批量评估捕获单样本错误并写入结果，不能因一个坏框丢失整批诊断；输入文件缺失、
`sample_id` 重复或无法连接属于数据集契约错误，应在评估开始前整体失败。

## Testing Strategy

实现遵循 TDD，每项行为先写失败测试，再写最小实现。

### Pure Geometry Tests

- 归一化框中心正确映射到 `448 x 448` 的浮点像素坐标。
- 当前相机下，图片中心反投影到 `workspace_center` 的 x/y。
- 已知工作平面世界点执行 `world -> pixel -> world plane` 后，在数值容差内还原。
- 图片 y 轴翻转正确：画面上方对应当前配置的世界 `+Y`。
- 非法尺寸、非法框、奇异矩阵、平行射线和相机后方交点分别失败。

### Integration Tests

- `control_arm.capture_rgb()` 使用共享的相机矩阵构造函数，图像尺寸与类型保持不变。
- 采样器继续保证 `samples.jsonl` 不含 `block_pos`，并把真值写入独立 diagnostics。
- 离线评估能连接 grounding 与 diagnostics，计算逐样本误差和汇总指标。
- 缺失或重复 `sample_id` 会在任何指标计算前失败。

### Verification Commands

```bash
conda run -n vla_env python -m unittest tests.test_camera_geometry -v
conda run -n vla_env python -m unittest tests.test_collect_vlm_eval_samples -v
conda run -n vla_env python -m unittest discover -s tests -v
```

## Acceptance Criteria

- 当前固定俯视相机通过世界点往返测试，没有 x/y 轴翻转。
- `samples.jsonl` 仍然不包含 `block_pos` 或其他可泄露目标真值的字段。
- 离线评分无需网络，能够报告反投影有效率和 xy 定位误差统计。
- 所有失败都有可区分的错误类型，单样本失败不会中断整批评估。
- 现有完整测试套件继续通过。

定位误差是否足以进入 `api_grounded` 在线 smoke test，需要根据真实离线统计另行判断；
本设计不预先假定它一定低于当前 `0.03m` 成功阈值。

## Non-Goals

- 不实现 `api_grounded` 在线决策模式。
- 不运行新的 Qwen 请求。
- 不使用 depth buffer、双目或多视角三角测量。
- 不训练 action token、LoRA 或 QLoRA。
- 不宣称 VLM 框达到像素级精确定位。
