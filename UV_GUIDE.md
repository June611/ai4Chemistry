# uv 环境速查（Python + PyTorch）

1. 在项目空目录中执行以下命令；`uv add` 会自动创建并同步 `.venv`：

```bash
uv init --package
uv python install 3.11
uv python pin 3.11
uv add "torch==2.6.0" --index pytorch-cu124=https://download.pytorch.org/whl/cu124
uv add chemprop pandas numpy scikit-learn
uv add --dev pytest ruff
uv sync --locked
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```
2. 先用 `nvidia-smi` 查看驱动支持的最高 CUDA 版本；本机显示 12.4，所以选官方 `cu124` wheel。
3. 在 PyTorch 官方版本表确认该 Torch 版本提供目标 CUDA wheel；纯 CPU 环境改用 `https://download.pytorch.org/whl/cpu`。
4. 普通依赖用 `uv add X`，开发依赖用 `uv add --dev X`，其他用途用 `uv add --group NAME X`。
5. `uv add` 同时更新 `pyproject.toml`、`uv.lock` 和 `.venv`；克隆项目后只需运行 `uv sync --locked`。
6. 提交 `.python-version`、`pyproject.toml` 和 `uv.lock`，不要再混用手写 `requirements.txt` 或 `uv pip install`。

## 本项目 RDKit 验证

在 `/home/june/deepLear_zone/Chem_proj/chempro` 下运行：

```bash
uv run --locked python scripts/check_rdkit.py
uv pip check
```

检查脚本默认读取 `data/processed/qm9_full/random/seed3407`，在训练、验证、测试集中各检查前 100 条分子，验证指定的 29 个描述符存在、计算结果有限，且直接调用与描述符计算器结果一致。可通过 `--rows-per-split 1000` 扩大检查量，通过 `--data-dir` 指定其他划分。默认只检查特征；增加 `--check-models` 时，在抽样训练数据上分别拟合 3 棵树的 XGBoost 和随机森林，再在抽样验证数据上检查预测，仅用于验证依赖兼容性，不产生正式实验指标或模型文件。

2026-09-30 验证通过：项目 `.venv` 使用 Python 3.11.9、RDKit 2026.3.6、XGBoost 2.0.3、scikit-learn 1.9.1，300 条分子的 29 个特征计算成功；RDKit 与 Chemprop 2.3.1 同进程导入成功，`uv pip check` 的 115 个包兼容性检查通过。该抽查不代表全量分子均已验证。

参考项目的 RDKit 2022.9.5 与本项目 `rdkit>=2024.3` 要求冲突，因此使用本项目 `uv.lock` 已锁定的兼容版本。后续实验应记录实际 RDKit 版本；同名描述符在不同版本间的数值一致性尚未验证。XGBoost 已在 WSL 本地通过 `uv add 'xgboost==2.0.3'` 添加；随机森林由现有 scikit-learn 提供。

## Git 同步到服务器

依赖在 WSL 修改、解析并验证，提交 `pyproject.toml` 和 `uv.lock`；服务器拉取后按锁文件安装，不复制本地 `.venv`。服务器在项目目录执行：

```bash
cd /root/lanyun-tmp/evan/ai4Chemistry
git pull --ff-only origin main
uv sync --locked
uv run --locked python scripts/check_rdkit.py --check-models
uv pip check
```

`uv sync --locked` 比直接 `uv sync` 多一项锁文件一致性检查；正常情况下安装同一套依赖。上述命令为服务器操作说明，本次仅在 WSL 安装和验证。

## 现有模型结果解析

服务器的 `outputs/batches/qm9_five_seed` 保存批次状态和生成配置；实际结果位于 `outputs/<model>/seed<training_seed>/`。四类模型为 `chemprop_single_gap`、`chemprop_multitask`、`chemprop_consistency` 和 `smiles_transformer`，各有 3407、42、2026、7、123 五个训练 seed，数据划分 seed 均为 3407。

每次运行包含 `config.yaml`、`metrics.json` 和 `predictions/test_predictions.csv`。Chemprop 指标位于 `targets.delta_e`，预测列为 `delta_e`；Transformer 指标为顶层字段，预测列为 `delta_e_pred`，其 CSV 中的 `delta_e` 是真实值。通用评估现在优先选择显式预测列，并拒绝非有限标签或预测。

在具有这些输出的项目目录内汇总四类模型的五个训练 seed：

```bash
uv run --locked molgap-summarize \
  outputs/chemprop_single_gap outputs/chemprop_multitask \
  outputs/chemprop_consistency outputs/smiles_transformer \
  --seeds 3407 42 2026 7 123 \
  --output-dir outputs/comparisons/qm9_full_five_seed
```

该命令读取已有指标，缺少必要指标时从预测文件重新计算，输出每次运行的 CSV、汇总表及图。`--seeds` 校验完整性，不用于筛选；只比较训练 seed3407 时应传入四个 `seed3407` 目录并使用 `--seeds 3407`。正式 baseline 训练和七模型比较仍在 TODO 中跟踪。
