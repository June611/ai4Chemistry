# QM9 六性质神经网络单任务扩展

范围：Chemprop、SchNet、3D Infomax、EGNN、PaiNN、DimeNet++、SMILES Transformer、SPE Transformer、ESA、TorchMD-Net。

新增 homo、lumo、alpha、mu、r2（电子空间范围）、zpve，每项独立训练，seed 42/2026/3407，共 180 次下游实验。原 gap 及多任务结果保持原协议。

训练集 Z-score、MSE、原单位验证 MAE 选模；batch 256、100 epochs、patience 20，新任务 min_delta=0。输入沿用各架构的编码与坐标单位。

任务定义保存在 configs/qm9_six_properties/tasks.yaml；展开配置、计划、状态、日志、权重和预测全部落盘。正式训练由用户执行命令启动。

实现、验证、提交进度见 TODO.md；接口与验收记录在 tasks.md，协议取舍在 decisions.md。

## 操作命令

在项目根目录执行，先用 `uv sync --extra qm9` 安装环境。以下命令不需要 LLM 保持在线。
原始 QM9、旧三组 split、geometry 文件通过 scp 等方式另行传输；Git 不包含这些大文件。
首次部署还需获取固定上游源码（已有对应 .external 目录时会核验版本并复用）：

```bash
uv run molgap-bootstrap-qm9 --source egnn --source 3dinfomax --source torchmd-net
```

已有 `data/processed/qm9_single_properties_v1/data_identity.json` 时直接复用；缺失时先运行：

```bash
uv run python scripts/run_qm9_properties.py data
```

准备所有 180 项下游任务，两张卡独立运行、每卡一个任务：

```bash
uv run python scripts/run_qm9_properties.py prepare \
  --directory artifacts/qm9_six_properties/formal01 --gpus 0 1
uv run python scripts/run_qm9_properties.py run \
  --directory artifacts/qm9_six_properties/formal01
```

单卡服务器准备时使用 `--gpus 0`。建议在 tmux 中运行 run 命令；退出 Codex 不影响训练。
prepare 会冻结磁盘配置和任务计划，不启动训练。正式 batch 固定 256，不自动缩小。
历史显存值用于准入，不能保证新设备或不同库版本一定不 OOM；实际失败单独记录并继续其他独立任务。

单模型、单性质、单 seed 示例：

```bash
uv run python scripts/run_qm9_properties.py prepare \
  --directory artifacts/qm9_six_properties/chemprop_mu42 \
  --models chemprop_single_gap --targets mu --seeds 42 --gpus 0
uv run python scripts/run_qm9_properties.py run \
  --directory artifacts/qm9_six_properties/chemprop_mu42
```

模型名称：`chemprop_single_gap smiles_transformer spe_transformer schnet 3dinfomax egnn painn dimenet_pp esa torchmd_net`。
不传 targets/seeds 表示该模型的六性质、三 seed。`--models` 可传多个名称。
GPU 选择在 prepare 时写入计划，run 时不重新指定。机器迁移后重新 prepare，避免复用另一机器的绝对路径。

## Infomax 前置权重

在 tasks.yaml 的 `pretraining.checkpoints` 中为 42/2026/3407 分别填写已有 QM9 自监督 checkpoint 路径。
默认 `auto` 会在本项目既有 auto_training 和 paired_gap_mae_v2 预训练目录中查找并核验可复用权重；
没有匹配权重时生成该 seed 的预训练任务。显式路径不匹配时直接报错，不静默退回其他权重。
prepare 会核对 seed、原 split manifest、几何文件 hash；不匹配则拒绝复用。
留 null 时为该 seed 新增一次预训练任务（256/100/20），供对应的六项下游任务共享。
因此全部计划是 **180 项下游，另有 0–3 项预训练**。预训练失败只阻断依赖它的下游任务。
已有历史权重若使用旧参数，报告中保留其来源，不能宣称重新按新预算预训练。

## 状态、恢复和结果

```bash
uv run python scripts/run_qm9_properties.py status --directory artifacts/qm9_six_properties/formal01
uv run python scripts/run_qm9_properties.py summarize --directory artifacts/qm9_six_properties/formal01
uv run python scripts/run_qm9_properties.py verify --directory artifacts/qm9_six_properties/formal01
```

summarize 可生成部分进度；verify 要求所选任务全部完成，并从预测 CSV 重算指标。
输出在 `outputs/summaries/qm9_results/batches/<批次目录名>/`。
三 seed 均值和样本标准差按同一性质分组；各性质比较文件按 RMSE 降序排列。
内存/显存是进程树定时采样峰值，可能遗漏短瞬间峰值；原始采样在各 lane 的 runtime/*.jsonl。

再次执行 run 会跳过已完成和已失败任务，继续 pending 任务。
中断或失败的任务需在旧 controller/worker 退出后，显式指定 ID 重试：

```bash
uv run python scripts/run_qm9_properties.py retry \
  --directory artifacts/qm9_six_properties/formal01 --jobs chemprop_single_gap_mu_seed42
uv run python scripts/run_qm9_properties.py run --directory artifacts/qm9_six_properties/formal01
```

重试会重命名保存旧输出，从头训练该项；不是恢复优化器状态。依赖任务被 blocked 后也需显式重试。
修改 tasks.yaml 或模型模板后，用新的批次目录 prepare；运行中的任务及历史结果不覆盖。
修改已冻结 YAML 会触发 hash 检查，不会静默使用新参数。

完整的十模型六性质三 seed 完成后，生成新的 LaTeX 文件，保留已有 ML/gap/时间列：

```bash
uv run python scripts/run_qm9_properties.py latex \
  --directory artifacts/qm9_six_properties/formal01 \
  --source-tex /tmp/qm9_single_task_rmse.tex \
  --output-tex /tmp/qm9_single_task_rmse_neural.tex
```

## 对比口径

原 CSV 标签不缩放；每个训练任务另存训练集 mean/std、目标与物理单位。
新六性质统一 Z-score+MSE；已有七个 QM9 适配器的历史 gap 实验实际上为 MAD+MAE，保留原记录，不能表述为全表统一了 loss/scaler。
所有目标均在原单位下报告测试误差。R2 目标指电子空间范围，r2 指标指回归决定系数。
不同性质和不同单位的数值不能直接混排。Gap time/seed 只代表历史 gap，不代表六项新任务的耗时。
