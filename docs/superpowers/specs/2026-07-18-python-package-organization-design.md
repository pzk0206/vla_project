# Python 代码包结构整理设计

## 目标

把仓库根目录的 11 个 Python 文件整理为标准 `src/` Python 包，同时让测试代码继续独立位于根目录 `tests/`。本次只改变文件位置、模块导入、命令入口和文档路径，不改变机械臂控制、采样、VLM、grounding、反投影或校准算法。

## 最终目录

```text
vla_project/
├── src/
│   └── vla_project/
│       ├── __init__.py
│       ├── simulation/
│       │   ├── __init__.py
│       │   ├── camera_geometry.py
│       │   ├── control_arm.py
│       │   ├── stage3_probe.py
│       │   └── evaluate_probe.py
│       ├── vlm/
│       │   ├── __init__.py
│       │   ├── collect_vlm_eval_samples.py
│       │   ├── evaluate_vlm_decisions.py
│       │   ├── diagnose_vlm_grounding.py
│       │   ├── evaluate_ground_then_decide.py
│       │   ├── evaluate_grounding_backprojection.py
│       │   └── validate_grounding_calibration.py
│       └── tools/
│           ├── __init__.py
│           └── migrate_generated_outputs.py
├── tests/
│   ├── __init__.py
│   ├── simulation/
│   ├── vlm/
│   ├── tools/
│   └── test_config_contract.py
├── outputs/
├── docs/
├── pyproject.toml
├── requirements.txt
├── sim_config.yaml
└── README.md
```

测试子目录包含 `__init__.py`，保证 Python 3.10 的 `unittest discover` 能递归导入测试模块。

## 文件映射

### Simulation

| 旧路径 | 新路径 |
| --- | --- |
| `camera_geometry.py` | `src/vla_project/simulation/camera_geometry.py` |
| `control_arm.py` | `src/vla_project/simulation/control_arm.py` |
| `stage3_probe.py` | `src/vla_project/simulation/stage3_probe.py` |
| `evaluate_probe.py` | `src/vla_project/simulation/evaluate_probe.py` |

### VLM

| 旧路径 | 新路径 |
| --- | --- |
| `collect_vlm_eval_samples.py` | `src/vla_project/vlm/collect_vlm_eval_samples.py` |
| `evaluate_vlm_decisions.py` | `src/vla_project/vlm/evaluate_vlm_decisions.py` |
| `diagnose_vlm_grounding.py` | `src/vla_project/vlm/diagnose_vlm_grounding.py` |
| `evaluate_ground_then_decide.py` | `src/vla_project/vlm/evaluate_ground_then_decide.py` |
| `evaluate_grounding_backprojection.py` | `src/vla_project/vlm/evaluate_grounding_backprojection.py` |
| `validate_grounding_calibration.py` | `src/vla_project/vlm/validate_grounding_calibration.py` |

### Tools

| 旧路径 | 新路径 |
| --- | --- |
| `migrate_generated_outputs.py` | `src/vla_project/tools/migrate_generated_outputs.py` |

## 测试映射

- `test_camera_geometry.py`、`test_control_arm.py`、`test_stage3_probe.py`、`test_evaluate_probe.py` 移到 `tests/simulation/`。
- `test_collect_vlm_eval_samples.py`、`test_evaluate_vlm_decisions.py`、`test_diagnose_vlm_grounding.py`、`test_evaluate_ground_then_decide.py`、`test_evaluate_grounding_backprojection.py`、`test_validate_grounding_calibration.py` 移到 `tests/vlm/`。
- `test_migrate_generated_outputs.py` 移到 `tests/tools/`。
- `test_config_contract.py` 保留在 `tests/`，因为它同时保护 simulation、VLM 和全局配置。

测试继续使用临时目录和 mock，不触碰 `outputs/` 中的真实数据，不启动完整 PyBullet episode，不调用 Qwen。

## 包与导入规则

- 包名固定为 `vla_project`，源码只从 `src/vla_project/` 导入。
- 跨子包导入使用绝对包路径，例如：

```python
from vla_project.simulation.control_arm import CONFIG_PATH, load_config
from vla_project.simulation.stage3_probe import call_openai_compatible_api
from vla_project.vlm.collect_vlm_eval_samples import read_jsonl
```

- 同一子包内部也统一使用完整包路径，避免脚本直接运行与包运行产生两套导入行为。
- 删除根目录旧 `.py` 文件，不保留重复 wrapper；所有正式入口通过 `pyproject.toml` console scripts 提供。
- `sim_config.yaml` 仍从当前工作目录读取，因此正式命令应在仓库根目录运行。此次不增加配置搜索或环境变量机制。

## `pyproject.toml`

使用 setuptools 的 `src` layout：

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "vla-project"
version = "0.1.0"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["src"]
```

运行环境继续由 `requirements.txt` 管理。安装项目本身使用：

```bash
conda run -n vla_env pip install -e . --no-deps
```

`.gitignore` 增加 `*.egg-info/`，避免 editable install 元数据进入 Git。

## 命令入口

`pyproject.toml` 提供以下命令：

| 命令 | Python 入口 |
| --- | --- |
| `vla-collect` | `vla_project.simulation.control_arm:main` |
| `vla-probe` | `vla_project.simulation.stage3_probe:main` |
| `vla-evaluate-probe` | `vla_project.simulation.evaluate_probe:main` |
| `vla-collect-vlm-samples` | `vla_project.vlm.collect_vlm_eval_samples:main` |
| `vla-evaluate-vlm-decisions` | `vla_project.vlm.evaluate_vlm_decisions:main` |
| `vla-diagnose-grounding` | `vla_project.vlm.diagnose_vlm_grounding:main` |
| `vla-ground-then-decide` | `vla_project.vlm.evaluate_ground_then_decide:main` |
| `vla-evaluate-backprojection` | `vla_project.vlm.evaluate_grounding_backprojection:main` |
| `vla-validate-grounding-calibration` | `vla_project.vlm.validate_grounding_calibration:main` |
| `vla-migrate-generated-outputs` | `vla_project.tools.migrate_generated_outputs:main` |

这些命令保持现有 `main()` 行为。可能调用 Qwen 的命令只验证入口可导入，不在重构验证中真实执行。

## 文档同步

- README 的文件树、安装步骤和四个常用运行命令更新为新结构。
- WORKLOG、BUGLOG 和 Markdown 学习计划中对源代码路径的当前引用更新为 `src/vla_project/...`。
- 历史 `docs/superpowers/specs/` 和 `docs/superpowers/plans/` 保留当时文件路径和命令，避免篡改历史实施记录。
- 原始 PDF 不修改。

## 安全与验证

1. 移动前运行当前完整测试套件，基线应为 106 个测试通过。
2. 先添加 `pyproject.toml` 和包骨架，再逐组移动 simulation、VLM、tools 及对应测试。
3. 每组移动后修改 imports 并运行该组测试，失败时不继续下一组。
4. editable install 后验证所有 10 个 console scripts 都已注册并指向预期入口；不执行会启动仿真或调用 API 的命令。
5. 最终运行 `conda run -n vla_env python -m unittest discover -s tests -v`，测试数量不得少于移动前的 106。
6. 使用 `rg` 确认正式代码和实时文档不存在旧根目录模块导入或旧 `python xxx.py` 命令。
7. `outputs/` 文件数、图片数和图片字节数在整理前后保持一致。
8. Git 状态只包含计划内源码、测试、配置和文档变更，不包含 `outputs/` 或 editable install 元数据。

## 非目标

- 不拆分现有大文件或重命名其中的函数、类和常量。
- 不修改算法、配置数值、输出数据格式或实验结果。
- 不重新生成、移动或删除 `outputs/`。
- 不调用 Qwen 或其他付费 API。
- 不引入 pytest、Poetry 或新的运行时依赖。
