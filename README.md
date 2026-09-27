# Molecular HOMO–LUMO Gap Prediction

基于 Chemprop D-MPNN，从分子 SMILES 预测 HOMO–LUMO 能隙。Chemprop 作为固定的第三方依赖；本仓库只管理数据、配置、实验、评估与后续自定义模块。

## 当前范围

项目已经接通可复现的数据管线和三阶段训练接口：

1. Phase 1：Chemprop 官方 CLI，`SMILES -> delta_e`。
2. Phase 2：Chemprop 官方 CLI，`SMILES -> homo, lumo, delta_e`，三任务等权。
3. Phase 3：Chemprop Python API + 项目内三个独立 head，并加入
   `delta_e ≈ lumo - homo` 的一致性损失。

三个阶段复用同一份划分。Chemprop 在训练时把 SMILES 转为分子图，并仅用训练集拟合目标
Z-score scaler；CSV 始终保留 Hartree 原始标签。本项目不导出分子图、嵌入或特征文件，也不修改
Chemprop 源码或 `site-packages`。

## 环境

Chemprop 2.3.1 要求 Python 3.11–3.14，本项目通过 `.python-version` 固定使用 Python 3.11。本机 CUDA 为 12.4，因此 `pyproject.toml` 将 PyTorch 固定为官方 `torch==2.6.0` 的 `cu124` wheel，并使用显式的 PyTorch 索引；不要用普通 PyPI 的 CUDA 13 构建替换它。

```bash
uv sync
```

以后添加普通运行依赖时直接使用：

```bash
uv add <package>
```

添加开发或分析依赖时使用对应的依赖组：

```bash
uv add --dev <package>
uv add --group analysis <package>
```

这些命令会同时更新 `pyproject.toml`、`uv.lock` 和 `.venv`。本项目不使用手写 `requirements.txt`，避免它与锁文件记录不同的软件包来源。

安装完成后可验证 wheel 与 WSL GPU 是否都可用：

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

预期前两项为 `2.6.0+cu124` 和 `12.4`。最后一项由 Windows NVIDIA 驱动与 WSL GPU 透传决定；本项目不会安装或修改系统 CUDA/Windows 驱动。

开发与测试依赖默认会安装。需要 notebook 绘图环境时：

```bash
uv sync --group analysis
```

## 数据预处理

原始数据保持只读：

- `data/raw/qm9_chn.csv`：完整数据集，配置为 `configs/data/qm9_full.yaml`
- `data/raw/QM9_CN.csv`：完整数据集的 CN 子集，配置为 `configs/data/qm9_cn_subset.yaml`

两个数据源必须独立处理，不能拼接。生成完整数据集的确定性随机划分：

```bash
uv run molgap-prepare --config configs/data/qm9_full.yaml
```

产物位于 `data/processed/qm9_full/random/seed3407/`。其中 `train.csv`、`val.csv`、
`test.csv` 的列为 `sample_id,smiles,homo,lumo,delta_e`。Phase 1 通过
`--target-columns delta_e` 只读取 gap；Phase 2/3 使用全部三个标签。目录同时保存重复记录、
拒绝记录、逐字段变更、处理日志、目标统计、完整 lineage 和 SHA-256。

需要 scaffold 划分时复制数据配置并设置 `split.method: scaffold_balanced`。所有参数通过 YAML 修改，不修改 Python 源码。

## 快速开始

准备数据并检查三个阶段的训练计划：

```bash
uv run molgap-prepare --config configs/data/qm9_full.yaml
uv run molgap-train --config configs/baseline.yaml --dry-run
uv run molgap-train --config configs/multitask.yaml --dry-run
uv run molgap-train --config configs/consistency.yaml --dry-run
```

去掉 `--dry-run` 启动所选阶段。例如先运行单任务 baseline：

```bash
uv run molgap-train --config configs/baseline.yaml
uv run molgap-predict --config configs/baseline.yaml
uv run molgap-evaluate --config configs/baseline.yaml
```

Phase 3 训练结束后会直接生成最佳 checkpoint、便携 Chemprop `.pt`、测试集预测和指标。

每次运行写入 `outputs/<experiment>/seed<seed>/`。配置快照、命令、日志、模型、预测和指标保存在同一个实验目录内。重复训练同一实验默认拒绝覆盖；确认需要替换时显式传入 `--overwrite`。

## 目录

```text
.
├── configs/                 # 可审计的实验配置
├── data/
│   ├── raw/                 # 原始数据，不提交 Git
│   ├── processed/           # 清洗后的 train/val/test
│   └── splits/              # 每次划分的行级清单
├── notebooks/               # 探索性分析
├── outputs/                 # 模型、日志、预测、指标
├── scripts/                 # 面向使用者的薄入口
├── src/molgap/
│   ├── data/                # 验证、预处理和划分
│   ├── metrics/             # 独立回归指标
│   ├── models/              # Phase 3 多头模型与物理一致性 loss
│   ├── training/            # CLI/Python API 训练、预测与评估
│   └── utils/               # 文件和随机种子工具
└── tests/                   # 不依赖真实数据的单元测试
```

## 实验约定

所有训练参数都在 YAML 中，包括网络结构、学习率、设备、任务权重和一致性权重。Phase 3
消融模板位于 `configs/experiments/exp003_consistency_0.yaml` 与
`exp004_consistency_01.yaml`；二者只改变 `loss.weights.consistency` 和实验名。

`baseline.yaml` 使用 seed 3407 的随机划分。论文结果应至少运行多个 seed，并额外比较
scaffold split；新 seed 必须先用对应数据配置生成划分，再修改训练配置中的 processed_dir、
`data.split.seed` 和 `training.seed`。

每次正式运行会记录解析后的配置、命令（Phase 1/2）、Python/Chemprop/Torch/CUDA 环境、
随机种子以及输入划分和 lineage 的 SHA-256。指标按 `sample_id` 对齐后计算，包含每个目标的
MAE、RMSE、R²；多任务额外报告预测一致性误差。

预处理会规范化 SMILES，在 `1.1e-4` Hartree 容差内聚合重复标签，超过容差时停止并写入待确认清单。它保证同一规范 SMILES 不跨数据集，并通过 `split_manifest.csv` 记录每个样本的去向。

## 质量检查

```bash
uv run ruff check .
uv run pytest
```

## 字段解释
+ source 数据来自哪个数据集
+ soure_id 原始数据中的id
+ sample_id 本项目中的id
+ source_row 原始文件的哪一行
+ split_position 在划分文件的哪个位置

## 训练指令
```
uv run molgap-train --config configs/baseline.yaml
uv run molgap-train --config configs/multitask.yaml
uv run molgap-train --config configs/consistency.yaml
```
