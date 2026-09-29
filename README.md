# Molecular HOMO–LUMO Gap Prediction

基于 Chemprop D-MPNN，从分子 SMILES 预测 HOMO–LUMO 能隙。Chemprop 作为固定的第三方依赖；本仓库只管理数据、配置、实验、评估与后续自定义模块。

## 当前范围

项目已经接通可复现的数据管线、三阶段 Chemprop 接口和一组序列模型对照实验：

1. Phase 1：Chemprop 官方 CLI，`SMILES -> delta_e`。
2. Phase 2：Chemprop 官方 CLI，`SMILES -> homo, lumo, delta_e`，三任务等权。
3. Phase 3：Chemprop Python API + 项目内三个独立 head，并加入
   `delta_e ≈ lumo - homo` 的一致性损失。
4. SMILES Transformer：`canonical SMILES -> tokenizer -> Transformer encoder -> 回归头 -> delta_e`。

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

默认 random 配置并非纯随机置乱：它先按 `delta_e` 建立 200 个分位数箱，再在箱内执行
确定性分层抽样，从而保持 train/val/test 的标签分布近似一致。实际箱数及每行箱号记录在
`split_manifest.csv`。需要 scaffold 划分时复制数据配置，设置
`split.method: scaffold_balanced` 并删除 `split.stratify`；骨架隔离和标签分层不能同时严格保证。

重新检查任意处理目录的分布并生成三条同图 KDE 曲线：

```bash
uv run molgap-audit-splits \
  --processed-dir data/processed/qm9_full/random/seed3407 \
  --output-dir reports/split_distribution/qm9_full/manual_audit \
  --dataset-name qm9_full \
  --stage manual_audit
```

完整集和 CN 子集的修改前后统计与图片位于 `reports/split_distribution/`。

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

检查并运行 SMILES Transformer 对照实验：

```bash
uv run molgap-train-transformer --config configs/transformer.yaml --dry-run
uv run molgap-train-transformer --config configs/transformer.yaml
```

该入口直接读取 Chemprop baseline 使用的 `train.csv/val.csv/test.csv`，不会重新划分。启动时会
逐行核对三个 CSV 的 `sample_id`、SMILES 和顺序是否与 `split_manifest.csv` 一致，并检查
`configs/baseline.yaml` 是否指向同一处理目录和同一 split 配置。验证结果和各文件 SHA-256
写入 `split_identity.json`。

Tokenizer 使用 `configs/transformer.yaml` 中的 SMILES 正则，词表只从训练集构建，并加入
`[PAD]`、`[UNK]`、`[CLS]`。`max_length: 40` 包含 `[CLS]`；任何未被正则覆盖或超长的 SMILES
都会终止运行。默认网络为 4 层、hidden 256、8 heads，共 2,160,641 个可训练参数。训练采用
AdamW、warmup + cosine decay 和验证集 Hartree MAE early stopping，最终报告 MAE、RMSE、MAPE
与 R²。标签 CSV 始终保留原值；损失内部的 Z-score 只用训练集拟合。

每次运行写入 `outputs/<experiment>/seed<training_seed>/`。配置快照、命令、日志、模型、预测和指标保存在同一个实验目录内。重复训练同一实验默认拒绝覆盖；确认需要替换时显式传入 `--overwrite`。

## 五个随机种子与单卡并行

批量配置位于 `configs/multiseed.yaml`。所有实验固定复用 `split_seed: 3407` 对应的同一份
train/val/test；`training_seeds: [3407, 42, 2026, 7, 123]` 只控制模型初始化、训练集
shuffle、Dropout、worker 和框架随机状态。四个模型共执行 20 次训练，但数据划分始终只有一份。

数据已经准备好时，分四次顺序运行四个模型：

```bash
uv run molgap-run-multiseed \
  --config configs/multiseed.yaml \
  --model chemprop_single_gap \
  --jobs 2

uv run molgap-run-multiseed \
  --config configs/multiseed.yaml \
  --model chemprop_multitask \
  --jobs 2

uv run molgap-run-multiseed \
  --config configs/multiseed.yaml \
  --model chemprop_consistency \
  --jobs 2

uv run molgap-run-multiseed \
  --config configs/multiseed.yaml \
  --model smiles_transformer \
  --jobs 2
```

`--jobs x` 表示同一时刻最多有 x 个训练进程，所有进程共享 `gpu: "0"` 指定的单张 GPU。
显存不足时使用 `--jobs 1`；显存和算力允许时可增加。也可以直接修改 YAML 中的
`parallel_jobs`。先查看将要生成的 20 个 model/training-seed 任务：

```bash
uv run molgap-run-multiseed --config configs/multiseed.yaml --jobs 2 --dry-run
```

如果固定划分尚未生成，可单独执行一次
`uv run molgap-prepare --config configs/data/qm9_full.yaml`；批量命令中的 `--prepare-data`
也只会检查或生成这一份固定划分，不会按训练 seed 重新划分。

只运行一个模型时使用 `--model` 过滤：

```bash
uv run molgap-run-multiseed \
  --config configs/multiseed.yaml \
  --model chemprop_single_gap \
  --jobs 2
```

调度器为每个 training seed 生成独立训练 YAML；所有 YAML 的 `data.processed_dir`、
`data.split.seed` 和 split manifest 都相同，只有 `training.seed` 改变。
Chemprop 任务依次运行 train、predict、evaluate；Transformer 在训练末尾完成预测和评估。
计划、配置、每个任务的 stdout/stderr 和最终状态位于
`outputs/batches/qm9_five_seed/`。已有训练结果默认不会覆盖；确需重跑时添加 `--overwrite`。

## 五 seed 汇总与模型比较

汇总单个模型：

```bash
uv run molgap-summarize \
  outputs/chemprop_single_gap \
  --seeds 3407 42 2026 7 123 \
  --output-dir outputs/summaries/chemprop_single_gap
```

比较全部模型：

```bash
uv run molgap-summarize \
  outputs/chemprop_single_gap \
  outputs/chemprop_multitask \
  outputs/chemprop_consistency \
  outputs/smiles_transformer \
  --seeds 3407 42 2026 7 123 \
  --output-dir outputs/summaries/qm9_model_comparison
```

汇总器检查每个模型是否恰好包含指定的五个 seeds，并针对 `delta_e` 计算 MAE、RMSE、MAPE、
R² 的均值、样本标准差、最小值和最大值。MAPE 的单位是百分比，绝对值不超过 `1e-12` 的
真实标签只从 MAPE 中排除并单独计数。输出包括 `runs.csv`、`summary.csv`、`summary.json`、
`summary.md`、带标准差误差条的 `model_comparison.png` 和 `model_comparison_table.png`。

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
│   ├── training/            # Chemprop 与 Transformer 训练、预测和评估入口
│   ├── transformer/         # SMILES tokenizer、共享划分校验、Dataset 与 Transformer
│   └── utils/               # 文件和随机种子工具
└── tests/                   # 不依赖真实数据的单元测试
```

## 实验约定

所有训练参数都在 YAML 中，包括网络结构、学习率、设备、任务权重和一致性权重。Phase 3
消融模板位于 `configs/experiments/exp003_consistency_0.yaml` 与
`exp004_consistency_01.yaml`；二者只改变 `loss.weights.consistency` 和实验名。

`baseline.yaml` 使用 split seed 3407 的固定随机划分。重复训练只修改 `training.seed`，不得
修改 `data.processed_dir` 或 `data.split.seed`。如果后续单独研究 random split 与 scaffold
split，应建立另一组明确命名的数据划分实验，不与本组训练随机性实验混合。

每次正式运行会记录解析后的配置、命令（Phase 1/2）、Python/Chemprop/Torch/CUDA 环境、
随机种子以及输入划分和 lineage 的 SHA-256。指标按 `sample_id` 对齐后计算，包含每个目标的
MAE、RMSE、MAPE、R²；多任务额外报告预测一致性误差。

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
uv run molgap-train-transformer --config configs/transformer.yaml
```
