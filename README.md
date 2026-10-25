# Molecular HOMO–LUMO Gap Prediction

## 新一轮200 epoch与可插拔模块审查

新正式比较采用batch256、最多200轮、patience20和42/2026/3407配对seed；历史100轮结果保留。
当前正式比较为 42 项：main.pdf 中 10 个神经模型的统一 gap 基线 30 项，加 DimeNet++/TorchMD-Net 两条 SOTA 路径上的 PINN/多任务模块 12 项；DeMol 与 Transformer-M 不入正式队列，传统 ML 沿用历史结果。
新增任务注意力七任务与物理约束十任务；逐标签训练集Z-score/MSE，联合任务按原单位Hartree验证gap MAE选模。
统一入口支持每50轮完整快照、每轮last和完整epoch边界恢复，保存优化器/LR/早停/RNG/loader/scaler与数据身份。
代码/协议测试与 ruff 已通过；历史 114 项烟测作为旧范围证据保留，当前 42 项统一队列尚待在 GPU0 完成 500 条/1 轮烟测。**尚未启动正式训练，全量最坏输入及并发/容量仍需按正式机器校准**。见[当前统一实验表](docs/features/qm9-epoch200-modular-study/queue_multitask_pinn.md)。
见[当前训练清单与交付](docs/features/qm9-epoch200-modular-study/README.md)与[运行/恢复指令](docs/features/qm9-epoch200-modular-study/operations.md)。

## QM9 六性质神经网络单任务扩展

十模型 × 六性质 × 三个配对 seed，共 180 项下游任务。任务清单与超参保存在
`configs/qm9_six_properties/tasks.yaml` 和 `models/`，训练集 Z-score、MSE、原单位 MAE 选模，
保持 256/100/20。生成计划不启动训练；各卡独立执行，失败隔离，结果与队列状态落盘。
安装、单模型/批量训练、恢复、Infomax 权重复用、汇总和 LaTeX 命令见
[六性质神经网络操作说明](docs/features/qm9-six-property-neural-baselines/README.md)。

现已提供 QM9 资源调度与飞书监测。在本地 WSL 项目目录运行下列命令，即可在既有服务器
后台调度全部启用模型，并通过本地 Lark CLI 同步资源、进度、指标和预测 CSV 附件：

```bash
bash scripts/train_all_auto.sh --remote --directory artifacts/auto_training/formal_20261012
```

默认 20 个下游模型 × 三组配对 seed（3407/42/2026），加三组 Infomax QM9 预训练，共63项。
神经网络保持256/100/20；Cormorant、LieTransformer不入队。默认最多两个GPU任务，大模型独占。
单任务失败后继续其他独立任务。服务器每5秒采样，本地后台程序约每5分钟同步飞书，不调用LLM；
使用tmux的一次启动方式见操作说明。
调度功能已通过本地141项测试、服务器相关32项测试和8项真实诊断，飞书指标与7份预测附件回读通过。
2026-10-12三组正式队列已启动，63项运行中；诊断与正式结果分开记录，尚未验收全部结果。
当前入口与启动证据见[正式运行记录](docs/features/qm9-resource-aware-training/formal_run.md)。详见[最终审查](docs/features/qm9-resource-aware-training/final_review.md)、
[操作说明](docs/features/qm9-resource-aware-training/operations.md)
和[实施TODO](docs/features/qm9-resource-aware-training/TODO.md)。
补查失败隔离、54个神经网络/预训练任务batch256与定时同步，见
[后续逻辑检查](docs/features/qm9-resource-aware-training/followup_checks.md)。

新增 SPE + Transformer 对照：使用官方 SmilesPE 和现有 Transformer 网络，采用三组
split/training seed 配对策略。配置为 `configs/spe_transformer.yaml`，批量配置为三份
`configs/spe_affinity_ablation_multiseed*.yaml`；详见 [SPE 训练指南](docs/guides/spe-training.md) 和
[执行清单](docs/features/spe-transformer/tasks.md)。

本项目从分子 SMILES 预测 HOMO–LUMO 能隙，以 Chemprop 为起点，持续接入其他方法，
形成共享数据与评估协议的多方法实验项目。Chemprop 保持为固定第三方依赖。

## QM9 十模型接入状态

十个 adapter、正式配置、固定 split 数据校验和服务器入口已完成本地验收。19 项 adapter
sanity tests 通过，十个模型在三套 split 上共完成 30 次全数据配置 dry-run；3D Infomax 的
PNA/Net3D 预训练与下游权重迁移链路完成 smoke 验证。启用模型的正式长训练已在GPU服务器排队执行，
Cormorant/LieTransformer保持暂停；正式结果完成后逐任务验收。

详细状态见 [十模型功能文档](docs/features/qm9-ten-models/README.md)、
[复现与公平性审查](docs/features/qm9-ten-models/reproduction-audit.md) 和
[任务清单](docs/features/qm9-ten-models/tasks.md)。

---

## 文档与复现方法管理

**以功能或研究任务（feature）为组织单位，不以 conversation 为目录单位。**
同一方法可以跨多次沟通完成；需求、实现、验证和实验结果应关联到稳定的功能档案。
这里的 feature 指项目功能或方法接入任务，不是分子描述符。

把多种方法接入同一项目是合理的，但应共享实验接口、保留方法独立性：

- 统一任务、样本划分、标签单位、预测文件格式和评估口径，记录输入 SHA-256。
- 各方法保留独立适配代码与 YAML，记录论文、上游源码版本、许可、依赖和预训练来源。
- 不强制所有方法使用 Chemprop 网络或同一组超参数；依赖冲突时允许独立环境，
  通过统一数据和预测文件接口接入。
- 明确区分“原论文设定复现”和“迁移到本项目 QM9 任务的适配”，分别配置、分别报告。
  记录额外监督、外部预训练和预算差异，不能仅凭误差高低判断实现是否正确。
- 分别验收代码适配、正确性检查、正式训练和结果报告；短跑通过不代表正式实验完成。

根目录仅维护两个项目级 Markdown 入口：本 README 和 [TODO.md](TODO.md)。
统一使用文件名 `TODO.md`，不再建立大小写不同的 `TODO.MD` 或功能专属根目录待办。

```text
docs/
├── guides/                       # 持续维护的环境、服务器和训练操作说明
├── features/
│   ├── model-benchmark/          # 多方法清单、接入验收及统一比较
│   │   └── inputs/               # 实际收到的模型 CSV 等原始需求输入
│   ├── qm9-ten-models/           # 十模型接入、逐项验收与交接
│   ├── spe-transformer/
│   ├── spe-multitask-affinity/
│   ├── paired-seed-100epoch-benchmark/
│   ├── traditional-baselines/
│   ├── smiles-transformer/
│   ├── chemprop-pipeline/
│   ├── data-splitting/
│   └── experiment-infrastructure/
└── templates/feature.md          # 新功能档案模板
```

每个功能目录包含 `README.md`（目标、状态与证据入口）、`tasks.md`（详细任务及验收证据）
和 `decisions.md`（关键需求、选择原因及范围变化）。根目录 TODO 仅保留活动工作摘要，
详细进度以各功能 tasks.md 为准；功能完成后移出根目录待办，保留功能内完成记录。

收到包含多个模型的 CSV 时，先保存原始输入，记录接收日期、SHA-256 和稳定模型 ID，
再在 [接入清单](docs/features/model-benchmark/tasks.md) 逐项追踪；复杂方法单独建功能目录。
尚未收到的文件不创建虚假条目。训练数据仍放 `data/`，论文放 `papers/`，
分析报告放 `reports/`，权重、预测和运行日志放 `outputs/`，通过链接关联而不重复复制。
若 outputs 未纳入 Git，交接时需另行记录和提供产物存储位置，不能只留下不可访问的链接。

沟通记录只提取影响实施的需求、决定、约束与验证结论，不用聊天全文代替技术文档。
决定按日期追加，变更注明替代关系；未知历史日期、提交号及实验结果必须如实标注。
文档和代码一起进行版本管理；路径调整需同步检查链接。

十模型接入的代码验收已经收口；后续按单项顺序在服务器执行正式训练，每次完成后记录
checkpoint、数据哈希、指标和运行环境。

文档入口：

- [十模型任务清单](docs/features/qm9-ten-models/tasks.md) · [复现与公平性审查](docs/features/qm9-ten-models/reproduction-audit.md) · [设计决定](docs/features/qm9-ten-models/decisions.md)

- [当前待办](TODO.md) · [多方法接入规范](docs/features/model-benchmark/README.md) · [新功能模板](docs/templates/feature.md)
- [SPE 功能档案](docs/features/spe-transformer/README.md) · [传统基线档案](docs/features/traditional-baselines/README.md)
- [SPE 多任务亲和力消融](docs/features/spe-multitask-affinity/README.md)
- [三组配对 seed、100-epoch 统一协议](docs/features/paired-seed-100epoch-benchmark/README.md)
- [环境指南](docs/guides/environment.md) · [服务器连接](docs/guides/server-access.md)
- [QM9 服务器部署与 GPU 环境](docs/features/qm9-server-deployment/README.md)
- [全模型一轮烟测与资源 CSV](docs/features/qm9-one-epoch-smoke/README.md)
- [飞书多维表格训练监测与历史结果同步](docs/features/feishu-training-monitoring/README.md)
- [SPE 训练指南](docs/guides/spe-training.md) · [传统基线训练指南](docs/guides/ml-training.md)

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

## 数据准备与还原

以下命令均在仓库根目录执行，使用同一代码版本及锁定环境（`uv sync --locked`）。
训练数据不随 Git 克隆；主 CSV 由项目提供者单独发送，三维结构从下述公开地址下载。
已有准备好的数据时先校验再复用，避免覆盖正在使用的划分。

### 1. 接收原始 CSV

向项目提供者获取 **`qm9_chn.csv` 完整文件**，保留文件内容和列名，放到
`data/raw/qm9_chn.csv`。这里不使用其他来源的同名 CSV 替代。

```bash
mkdir -p data/raw
# 将 /path/to 替换为实际收到文件的目录
cp /path/to/qm9_chn.csv data/raw/qm9_chn.csv

sha256sum --check <<'EOF'
00ebabed6fcb0008c81ba627c221c9bb7a713aae4c85201c5405fe2ef499e46b  data/raw/qm9_chn.csv
EOF
```

应显示 `OK`。该文件含 133885 条原始记录；除 SMILES、`mol_id` 外，还包含
`homo,lumo,delta_E,alpha,mu,r2,zpve` 等原始性质列。完整数据配置位于
[`configs/data/qm9_full.yaml`](configs/data/qm9_full.yaml)。历史 CN 子集
`data/raw/QM9_CN.csv` 使用独立配置 `configs/data/qm9_cn_subset.yaml`；当前完整 QM9
实验不需要该子集，不能将两个 CSV 拼接。

### 2. 预处理并生成三个固定划分

```bash
uv run --locked molgap-prepare --config configs/data/qm9_full.yaml
uv run --locked molgap-prepare --config configs/data/qm9_full_seed42.yaml
uv run --locked molgap-prepare --config configs/data/qm9_full_seed2026.yaml
```

入口为 [`src/molgap/data/preprocess.py`](src/molgap/data/preprocess.py)，SMILES 与重复记录
处理见 [`dataset.py`](src/molgap/data/dataset.py)，划分实现见
[`split.py`](src/molgap/data/split.py)。处理包括 SMILES 规范化并保留立体信息、标签合法性检查，
以及同一规范 SMILES 的重复标签聚合：在 `1.1e-4` Hartree 容差内取均值，超出容差则停止并
记录待确认项。当前完整数据从 133885 条记录聚合为 **133798 个唯一样本**，同一分子不会
跨 train/val/test。

三个 split seed 为 **3407、42、2026**，比例均为 **80% / 10% / 10%**，对应
**107038 / 13380 / 13380** 个样本。配置中的 `random` 实际先按 `delta_e` 建立
200 个分位数箱，再在箱内进行确定性分层抽样；不是直接随机置乱。三组使用同一批分子，
但集合归属不同；正式实验中模型初始化 seed 与 split seed 配对。

产物位于 `data/processed/qm9_full/random/seed<seed>/`：

- `train.csv`、`val.csv`、`test.csv`：列为 `sample_id,smiles,homo,lumo,delta_e`。
- `split_manifest.csv`：保存集合归属、行序、seed 和分层箱信息。
- `dataset.csv` 及其他审计文件：保存原始分子 ID 对应关系、重复记录、处理日志与校验信息。

用当前 WSL 的三个 manifest 校验是否还原了同一划分；任一项失败时先检查 CSV、代码版本和
环境，不继续训练或重新选择 seed：

```bash
sha256sum --check <<'EOF'
c23bfbb887e8fb7014353cea2154a623444defe3b356cee60350079b221ccd43  data/processed/qm9_full/random/seed3407/split_manifest.csv
cd291da6fcde7055870b30aa31a30f281ec92792881b57ebbee83c92ffcaf6cf  data/processed/qm9_full/random/seed42/split_manifest.csv
814583d9523c67427e3725cb7d991c8d47cd3a4f9280eaa39e7df1b598218c3c  data/processed/qm9_full/random/seed2026/split_manifest.csv
EOF
```

### 3. 补齐六项单任务性质

原 gap / 三任务训练直接使用上一步目录。训练新增的 HOMO、LUMO、极化率 alpha、偶极矩 mu、
电子空间范围 r2、零点振动能 zpve 时，在三个原划分均准备完成后执行一次：

```bash
uv run --locked python scripts/run_qm9_properties.py data
```

该入口调用 [`src/molgap/data/qm9_properties.py`](src/molgap/data/qm9_properties.py)，
按原始 ID 关联性质、复用已有重复样本分组的等权均值，保留三个划分的样本归属、行序及原能量标签。
输出为 `data/processed/qm9_single_properties_v1/random/seed<seed>/{train,val,test}.csv`，
根目录的 `data_identity.json` 记录来源、单位和文件哈希。已存在扩展目录时直接复用，生成命令
会拒绝覆盖；不会为每个性质重新划分数据。

磁盘 CSV 始终保存原物理量，**数据准备阶段不做标签 Z-score**。单位为：HOMO/LUMO/gap/ZPVE
使用 Hartree，alpha 使用 bohr³，mu 使用 Debye，电子空间范围 r2 使用 bohr²。六性质神经网络
在训练时仅用 train 拟合 Z-score，并逆变换后计算原单位误差；传统机器学习保留未缩放标签。
输入编码及特征缩放遵循各模型配置，SMILES token、原子类别不做连续特征 Z-score。

### 4. 下载并准备三维结构

仅运行 Chemprop、SMILES Transformer、SPE Transformer 时可跳过本步。需要 geometry 的
QM9 模型使用 **官方优化后的三维坐标**，不从 SMILES 随机生成构象。
原始 extended-XYZ 下载地址为
[QM9 `dsgdb9nsd.xyz.tar.bz2`](https://s3-eu-west-1.amazonaws.com/pstorage-npg-968563215/3195389/dsgdb9nsd.xyz.tar.bz2)，
来源说明见 [QM9 原始论文](https://www.nature.com/articles/sdata201422)。

```bash
uv sync --locked --extra qm9
uv run --locked python scripts/download_qm9_xyz.py --output-dir data/raw/qm9_xyz

uv run --locked python -m molgap.qm9_models.prepare \
  --config configs/qm9_models/egnn.yaml \
  --xyz-dir data/raw/qm9_xyz \
  --output data/geometry/qm9_full_seed3407.npz

for split_seed in 42 2026; do
  uv run --locked python scripts/reindex_qm9_geometry.py \
    --source data/geometry/qm9_full_seed3407.npz \
    --processed-dir "data/processed/qm9_full/random/seed${split_seed}" \
    --output "data/geometry/qm9_full_seed${split_seed}.npz"
done
```

[`download_qm9_xyz.py`](scripts/download_qm9_xyz.py) 自动下载压缩包到
`data/raw/dsgdb9nsd.xyz.tar.bz2`，校验 SHA-256
`3a63848ac80691bdb8d41834b575afad345b9300d7a2db0c38adb7f6eaa8360c`，并解压 133885 个
XYZ 文件到 `data/raw/qm9_xyz/`。网络慢时可手动下载到同一压缩包路径，再执行该命令。

[`prepare.py`](src/molgap/qm9_models/prepare.py) 通过原始分子 ID、原子组成与标签对应关系
匹配保留样本，按 train → val → test 的行序生成 NPZ；
[`reindex_qm9_geometry.py`](scripts/reindex_qm9_geometry.py) 对另外两套划分只重排同一份结构。
最终每个 seed 对应 `data/geometry/qm9_full_seed<seed>.npz` 和同名 `.json`，坐标单位为 Å，
包含样本 ID、原子序数、坐标与分子偏移量。六性质扩展继续复用这三份几何文件。
生成脚本拒绝覆盖现有几何文件；如直接收到准备好的文件，NPZ 和配套 JSON 必须一起复制。

DeMol / Transformer-M 还需要 `data/topology/qm9_full.npz` 和配套 JSON，以及生成它所用的
SDF；其额外准备步骤见[两模型数据输入与缓存说明](docs/features/qm9-demol-transformer-m/README.md#数据输入与一次性缓存)。

### 5. 还原后检查

在三份几何均准备完成后执行，不会启动训练：

```bash
uv run --locked python scripts/audit_qm9_split_seeds.py \
  --seeds 3407 42 2026 \
  --output reports/qm9_split_seed_audit.json
```

应得到 `status: pass`。该检查核对三个划分的样本数、集合互斥、行序、有限标签、几何哈希及
原子/坐标对应关系，并验证三套划分具有相同分子全集和相同三维结构。审计 JSON 中的时间戳和
机器绝对路径可能不同，不应据此判断训练数据不同。

如需绘制任一划分的分布图：

```bash
uv run --locked molgap-audit-splits \
  --processed-dir data/processed/qm9_full/random/seed3407 \
  --output-dir reports/split_distribution/qm9_full/manual_audit \
  --dataset-name qm9_full \
  --stage manual_audit
```

准备完成后的主要训练数据目录为：

```text
data/
├── raw/qm9_chn.csv                              # 单独收到的原始表
├── raw/dsgdb9nsd.xyz.tar.bz2                     # 下载的官方结构压缩包
├── raw/qm9_xyz/                                 # 解压后的原始 XYZ
├── processed/qm9_full/random/seed<seed>/         # 原 gap / 三任务数据与 manifest
├── processed/qm9_single_properties_v1/          # 六性质数据及身份记录
│   └── random/seed<seed>/
└── geometry/qm9_full_seed<seed>.{npz,json}       # 与各划分行序对应的几何
```

其中 `<seed>` 分别为 3407、42、2026。后续训练直接读取这些文件；更换机器不重新抽取划分。

## 快速开始

准备数据并检查三个阶段的训练计划（已完成上节数据准备时跳过第一条命令）：

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

## 训练吞吐与进度

调整前的五份配置完整保存在 `configs/archive/pre_throughput_20260929/`。当前四个主实验使用同一训练预算：batch size 256、最多 100 epochs、early-stopping patience 20、8 个 DataLoader workers。三组 Chemprop 实验因 batch 从 64 增加到 256，将学习率调度同步放大 4 倍至 `4e-4 / 4e-3 / 4e-4`；Transformer 保持已经与 batch 256 配套的 AdamW `lr: 3e-4`。

直接运行四个训练入口时，Lightning tqdm 会显示 epoch、batch、训练误差、验证误差、已耗时和 ETA。Phase 3 显式使用 pinned memory；Transformer 同时使用 pinned memory 与 persistent workers。Chemprop 2.3.1 CLI 没有公开 `pin_memory` 或 `precision` 参数，因此 Phase 1/2 保持库自身 DataLoader 行为，四模型比较统一采用全精度；没有向 YAML 写入不起作用的 AMP 参数。每次训练还会写出 `training_timing.json`。

单卡并行时，每个进程都会创建 8 个 worker。`--jobs 2` 因此最多使用约 16 个数据 worker；若服务器 CPU 核数不足或主机内存压力较大，先使用 `--jobs 1`，再依据 GPU 利用率逐步增加。

## 三组配对 seed 统一实验

当前协议固定三组 `(split seed, training seed)`：`(3407, 3407)`、`(42, 42)`、
`(2026, 2026)`。每次运行同时选择对应的数据划分和模型随机种子，不生成 3×3 交叉组合。
所有神经网络使用 batch size 256、最多 100 epochs、early-stopping patience 20。完整范围和
验收条件见 [统一协议](docs/features/paired-seed-100epoch-benchmark/README.md)。

当前所有下游神经网络统一按**原始 Hartree 的验证集 gap MAE**保存最佳checkpoint与早停，
min_delta为1e-5 Hartree。Chemprop通过项目内gap指标和scaler回调接入，保留官方网络/训练流程。
loss/scaler保留各方法既有配方：Chemprop/序列模型MSE与train-only Z-score，
QM9十模型MAE与train-only mean/MAD，传统基线不做scaler。它们是允许并披露的差异，
不影响共同gap MAE选模与测试口径；Infomax对比预训练单独使用对比验证loss。

本轮按 [实施计划](docs/features/paired-seed-100epoch-benchmark/standardization-plan.md) 和
[U系列TODO](docs/features/paired-seed-100epoch-benchmark/tasks.md) 执行。代码验收与服务器显存、
正式实验验收分开记录；历史审查保留，新审查见[最终TXT](docs/features/paired-seed-100epoch-benchmark/final-gap-mae-review.txt)。
正式协议ID为paired_split_gap_mae_v2，旧v1结果不进入新表。大文件只提供三种迁移方案。
三套数据的66次实际入口dry-run已通过；服务器先执行`bash scripts/qm9_server_preflight.sh`，
确认Torch/DGL的GPU支持，再逐模型检查batch256的显存；本地CPU检查不代表GPU验收。

Chemprop 三模型和原子 Transformer 使用三份矩阵配置：

```bash
for config in \
  configs/multiseed.yaml \
  configs/multiseed_seed42.yaml \
  configs/multiseed_seed2026.yaml
do
  uv run molgap-run-multiseed --config "$config" --jobs 1 --gpu 0
done
```

正式训练前把上述命令末尾替换为 `--jobs 1 --gpu 0 --dry-run`，检查生成计划。每个生成配置
都记录 `protocol_id: paired_split_gap_mae_v2`、split seed、training seed 和
`seed_pairing: matched`，结果统一写入 `outputs/paired_gap_mae_v2/`。

```bash
uv run molgap-run-multiseed --config configs/multiseed.yaml --jobs 1 --gpu 0 --dry-run
```

只运行一个模型时仍可使用 `--model` 过滤：

```bash
uv run molgap-run-multiseed \
  --config configs/multiseed.yaml \
  --model chemprop_single_gap \
  --jobs 1
```

## 三组 paired split-seed 汇总

比较模型时显式要求三组 seed：

```bash
uv run molgap-summarize \
  outputs/paired_gap_mae_v2/chemprop_single_gap \
  outputs/paired_gap_mae_v2/chemprop_multitask \
  outputs/paired_gap_mae_v2/chemprop_consistency \
  outputs/paired_gap_mae_v2/smiles_transformer \
  --seeds 3407 42 2026 \
  --output-dir outputs/paired_gap_mae_v2/summaries/main_models
```

汇总器校验三组运行均满足 `split_seed == training_seed`。三次运行同时改变数据划分和训练
随机性，因此标准差表示 paired split-seed 变化，不能解释为固定划分下的纯初始化方差。

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

所有训练参数都在 YAML 中，包括网络结构、学习率、设备、任务权重和一致性权重。当前协议
明确排除旧的 `configs/experiments/exp003_consistency_0.yaml` 和
`exp004_consistency_01.yaml`；它们只作为历史文件保留，不进入训练矩阵和结果汇总。

每组实验必须同时选择相同编号的数据划分和训练 seed。不得只修改 `training.seed` 后复用另一
个 split，也不得把三套 split 与三个 training seed 做交叉组合。

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

## output/workspace 目录

调度计划、生成配置和日志位于 `outputs/batches/<matrix-name>/`。模型产物统一写入
`outputs/paired_gap_mae_v2/<model>/seed<seed>/`，其中只允许 `seed3407`、`seed42` 和
`seed2026`。每个运行目录包含解析后的配置、数据身份、最佳 checkpoint、逐样本预测与指标。

## SPE + Transformer：在线算力服务器训练

本方法使用官方 `SmilesPE==0.0.3` 分词器，接入项目已有的 Transformer 回归网络。
SPE 合并规则和词表仅在训练集拟合；与原 Transformer 共享固定数据划分、网络结构和训练策略。
详细实现与来源见 [SPE 训练指南](docs/guides/spe-training.md)。

将项目代码、`pyproject.toml`、`uv.lock`、`configs/` 及现有
`data/processed/qm9_full/random/seed{3407,42,2026}/` 三个完整目录同步到服务器，
在项目根目录执行。十个 QM9 模型还需要对应三份 `data/geometry/*.npz`。
数据包和 SHA-256 校验方法见 [数据迁移指南](docs/guides/data-transfer.md)。
服务器需提供兼容项目 PyTorch CUDA 12.4 wheel 的 NVIDIA 驱动。
复用已有处理数据，无需重新划分。安装依赖并检查 GPU：

```bash
uv sync --locked
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

先检查分词、数据划分和模型配置（不训练）：

```bash
uv run molgap-train-transformer --config configs/spe_transformer.yaml --dry-run
```

三组配对实验分别运行对应矩阵：

```bash
for config in \
  configs/spe_affinity_ablation_multiseed.yaml \
  configs/spe_affinity_ablation_multiseed_seed42.yaml \
  configs/spe_affinity_ablation_multiseed_seed2026.yaml
do
  uv run molgap-run-multiseed --config "$config" --jobs 1 --gpu 0
done
```

每份矩阵只包含与 split seed 相同的一个 training seed，覆盖 SPE 单任务和四个多任务消融。
训练参数为 batch size 256、最多 100 epochs、patience 20、AdamW lr 3e-4、全精度。

需要断开 SSH 后继续训练时，使用下面的后台命令替代上面的前台命令：

```bash
mkdir -p outputs/batches/paired_protocol
nohup uv run molgap-run-multiseed \
  --config configs/spe_affinity_ablation_multiseed.yaml --jobs 1 --gpu 0 \
  > outputs/batches/paired_protocol/spe_seed3407.log 2>&1 &
```

查看第一个 seed 的训练进度：

```bash
tail -f outputs/batches/paired_protocol/spe_seed3407.log
```

如果只想用基础配置训练 seed 3407，使用以下命令。基础配置写入
`outputs/spe_transformer/seed3407/`；正式矩阵写入 `outputs/paired_gap_mae_v2/`，
汇总正式实验时只读取后者：

```bash
CUDA_VISIBLE_DEVICES=0 uv run molgap-train-transformer --config configs/spe_transformer.yaml
```

**本次本机试跑已按要求停止，保留了未完成的 seed3407 产物，没有最终测试结果。**
同步到服务器时不要复制本机的 `.venv/` 和 `outputs/spe_transformer/`，让服务器从空的
模型输出目录开始训练。如果已复制中断产物，可先将其改名保留，再运行上述命令：

```bash
mv outputs/spe_transformer outputs/spe_transformer_local_interrupted
```

该改名命令仅在源目录存在且目标目录不存在时执行。`--overwrite` 会删除相应运行目录并
重新训练，不是断点续训；默认不覆盖已有产物。

三组 seed 全部完成后汇总均值、标准差及比较图：

```bash
uv run molgap-summarize outputs/paired_gap_mae_v2/spe_transformer \
  --seeds 3407 42 2026 \
  --output-dir outputs/paired_gap_mae_v2/summaries/spe_transformer
```

每个 seed 自动保存最佳模型、分词词表与合并规则、逐样本测试预测，以及
MAE/RMSE（Hartree）、MAPE（%）、R²。将 `outputs/paired_gap_mae_v2/spe_transformer` 加入已有模型的
`molgap-summarize` 命令即可合并比较；参与比较的每个模型都需要完整三组配对结果。

## SPE 多任务、物理一致性与亲和力矩阵消融

在相同 SPE tokenizer 和 Transformer encoder 上新增 HOMO、LUMO、`delta_e` 三任务预测。
物理关系沿用 Phase 3：

```text
delta_e = lumo - homo
residual = predicted_delta_e - (predicted_lumo - predicted_homo)
```

完整模型在三个任务表示之间使用逐行 softmax 的可学习 3×3 亲和力矩阵。单次检查及训练：

```bash
uv run molgap-train-spe-multitask \
  --config configs/spe_multitask_affinity_consistency.yaml --dry-run
uv run molgap-train-spe-multitask \
  --config configs/spe_multitask_affinity_consistency.yaml
```

运行单任务基线和四组多任务 2×2 消融的 seed 3407 配对实验：

```bash
uv run molgap-run-multiseed \
  --config configs/spe_affinity_ablation_multiseed.yaml \
  --jobs 1 --gpu 0
```

该配置使用 split seed 3407。split seed 42 和 2026 分别使用
`configs/spe_affinity_ablation_multiseed_seed42.yaml` 与
`configs/spe_affinity_ablation_multiseed_seed2026.yaml`。详细设计、输出指标和执行状态见
[SPE 多任务亲和力融合](docs/features/spe-multitask-affinity/README.md)。

## QM9 十模型正式训练

**当前暂停 Cormorant 和 LieTransformer（2026-10-12 用户决定）**：两者在 RTX3090
24 GiB、batch256 下均 OOM；Cormorant 的 batch128/64 重试也 OOM，batch32 仅完成部分
训练后按要求停止，尚未验收；LieTransformer 降 batch 重试未启动。
`scripts/train_qm9_models.sh` 和全模型烟测默认跳过这两个模型，直接调用训练入口也会在
加载数据/GPU模型前拒绝；`--dry-run` 仍可检查配置。暂停名单统一保存在
[`paused_models.yaml`](configs/qm9_models/paused_models.yaml)。后续明确恢复任务后删除对应
名单项即可恢复入口，先验证显存，再运行正式实验。模型结构和原 batch256 配置保留，
其他已通过模型及 Infomax 预训练仍使用256，100 epochs、patience20和三组配对seed不变。
当前可运行范围为17个下游神经网络与3个传统基线，共60组配对实验；另有3组Infomax预训练。
停止记录与未完成事项见[显存适配档案](docs/features/qm9-model-batch-sizing/README.md)。

EGNN、SchNet、PaiNN、DimeNet++、ESA、TorchMD-Net、Cormorant、LieTransformer、ALIGNN
和 3D Infomax 已接入统一的固定划分训练入口。迁移到服务器后先运行：

```bash
bash scripts/qm9_server_preflight.sh
```

预检通过后运行三组配对 seed；脚本默认依次使用 3407、42、2026：

```bash
bash scripts/train_qm9_models.sh
```

若只运行一组，可覆盖配对 seed 列表：

```bash
QM9_PAIRED_SEEDS="3407" bash scripts/train_qm9_models.sh
```

3D Infomax 必须提供在 QM9 上完成对比预训练的 checkpoint；设置方式如下。未设置时训练脚本
会明确跳过该模型，其余七个未暂停的 QM9 模型继续执行。

```bash
bash scripts/pretrain_3dinfomax_paired.sh

INFOMAX_CHECKPOINT_TEMPLATE='outputs/paired_gap_mae_v2/3dinfomax_pretrain/splitseed{split_seed}/seed{split_seed}/best_checkpoint.pt' \
  bash scripts/train_qm9_models.sh
```

跨数据划分比较时，3D Infomax 必须在每套训练划分上分别预训练，避免预训练阶段看到该划分的
验证集或测试集。预训练同样使用 batch 256、最多 100 epochs、patience 20，且预训练 seed
严格等于对应 split seed。

三个传统基线通过同一配对协议运行：

```bash
bash scripts/train_ml_paired.sh
```

XGBoost、Random Forest 和训练均值没有 mini-batch、epoch 与 early stopping 概念，审计记录为
N/A；它们仍使用相同数据划分和配对 seed。

完整状态、数据例外和本地验收证据见
[QM9 十模型功能文档](docs/features/qm9-ten-models/README.md)。
