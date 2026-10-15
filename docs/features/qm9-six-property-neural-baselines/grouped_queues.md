# 高、中、低三组跨机训练

每个模型为六性质 × 三配对 seed，即 18 项单任务实验。三台机器使用同一版本代码和三份 YAML，
每台只启动分配给它的组。组内每张 GPU 一个独立任务，无 DDP，batch256/100epochs/patience20。

| 组 | 模型 | 下游实验数 | 可编辑任务清单 | 启动脚本 |
|---|---|---:|---|---|
| high | DimeNet++、3D Infomax、TorchMD-Net | 54 | groups/high.yaml | scripts/train_qm9_high.sh |
| medium | ESA、PaiNN、EGNN、SchNet | 72 | groups/medium.yaml | scripts/train_qm9_medium.sh |
| low | Chemprop、SMILES Transformer、SPE Transformer | 54 | groups/low.yaml | scripts/train_qm9_low.sh |

三个 YAML 均在 configs/qm9_six_properties/ 下。
分组模式读取 groups/*.yaml；原全量入口读取 tasks.yaml，两者不自动联动。
学习率和架构修改 models/<模型名>.yaml，分组、workers、预训练路径修改各组 YAML；GPU 编号由启动参数指定。

## 服务器准备

先安装 uv 环境，原 QM9 三组数据及几何文件按原目录传输。低组可使用 `uv sync`，
高中组使用 `uv sync --extra qm9`。Git 不包含数据、几何文件、权重、outputs。
如扩展的六性质数据尚未传输，但原始 QM9 与原三组 split 已齐全，可以生成：

```bash
uv run python scripts/run_qm9_properties.py data
```

高组获取固定上游源码：

```bash
uv run molgap-bootstrap-qm9 --source 3dinfomax --source torchmd-net
```

中组获取：

```bash
uv run molgap-bootstrap-qm9 --source egnn
```

Infomax 优先复用匹配 seed/split/geometry 的预训练权重。可以按原目录传输旧权重，或在
high.yaml 的 pretraining.checkpoints 指定路径。auto 找不到时，另加最多三项预训练；
预训练失败只阻断依赖它的 Infomax 下游任务。

## 启动

在项目根目录执行，建议先进入 tmux。下面分别在三台机器执行：

```bash
# 高组机器：示例有两张卡；单卡改为 --gpus 0
bash scripts/train_qm9_high.sh start --gpus 0 1

# 中组机器
bash scripts/train_qm9_medium.sh start --gpus 0

# 低组机器
bash scripts/train_qm9_low.sh start --gpus 0
```

默认目录分别是 artifacts/qm9_six_properties/{high,medium,low}_formal01。
start 首次生成计划后启动；已有计划则恢复 pending 队列，不重训 completed/failed。
运行中的同一队列有进程锁。每组只启动一份，不要同时再启动旧的全量 180 项队列。
本方案使用静态互斥分组，没有跨主机中心锁；合并时检查重复项。

只制作任务、不启动训练：

```bash
bash scripts/train_qm9_high.sh prepare --gpus 0 1
bash scripts/train_qm9_medium.sh prepare --gpus 0
bash scripts/train_qm9_low.sh prepare --gpus 0
```

prepare 与 start 二选一作为首次操作；prepare 后执行 run 即可开始。
修改任务清单后生成新批次，例如：

```bash
bash scripts/train_qm9_high.sh prepare \
  --directory artifacts/qm9_six_properties/high_formal02 --gpus 0 1
bash scripts/train_qm9_high.sh run \
  --directory artifacts/qm9_six_properties/high_formal02
```

自定义目录名以 high_ / medium_ / low_ 开头，避免三个来源的汇总互相覆盖。
在目标机器重新 prepare，不直接复制 WSL 的 plan.json，其中含机器绝对路径。
配置、模板快照、状态、日志、权重、预测均落盘，修改源 YAML 不会热改运行中任务。
可用 QM9_PYTHON 指定另一套已配置的 Python，默认 .venv/bin/python，不自动下载或升级环境。

## 调度依据

保留用户指定组别。各组沿用已有队列的 LPT 分卡及长任务优先：估计负载大的任务先分配给
当前估计负载较小的 GPU。Infomax 同 seed 的六项任务与预训练依赖放在同一队列，避免跨卡依赖。

调度权重默认是已有 3090 单 epoch 烟测 fit_seconds × 100，见 group_smoke_reference.csv。
原始证据为 docs/features/qm9-one-epoch-smoke/verification/resources_complete.csv，YAML 保存 SHA256。
Chemprop 的 fit_pipeline_seconds 包含启动及特征化，其他模型为 fit_seconds。
这些值只用于优先级和分卡，不是新性质、其他 GPU 或早停后的准确耗时预测。
内存/显存准入沿用上次功能中较保守的历史峰值，不用较低的单 epoch 峰值替代。
三组负载不相等，卡数可以不同，任务数相近不代表同时完成。

## 状态和重试

```bash
bash scripts/train_qm9_high.sh status
bash scripts/train_qm9_medium.sh status
bash scripts/train_qm9_low.sh status

# 旧 controller/worker 退出后，指定失败 ID 重试；旧输出会保留
bash scripts/train_qm9_low.sh retry --jobs chemprop_single_gap_mu_seed42
bash scripts/train_qm9_low.sh run
```

run/status/retry 使用磁盘计划，不接受重新指定 gpus/seeds/targets。
start 可重复携带原参数，但不同于冻结值时拒绝启动。更换参数或 GPU 使用新批次。

## 结果导出和合并

训练中可用 summarize 查看部分结果。正式结束后，在每台来源机器执行对应 verify：

```bash
bash scripts/train_qm9_high.sh verify
bash scripts/train_qm9_medium.sh verify
bash scripts/train_qm9_low.sh verify
```

输出在 outputs/summaries/qm9_results/batches/{high,medium,low}_formal01，包含 runs.csv、
summary.csv、verification.json、group_report.json。verify 在来源机器从完整预测重算原单位
MAE/RMSE；合并要求各组完整成功。将三个报告目录复制到 WSL 后合并：

```bash
bash scripts/train_qm9_group.sh merge --reports \
  outputs/summaries/qm9_results/batches/high_formal01 \
  outputs/summaries/qm9_results/batches/medium_formal01 \
  outputs/summaries/qm9_results/batches/low_formal01
```

默认生成带 UTC 时间的新汇总目录。合并无需服务器路径、权重或 GPU，检查来源文件哈希、
数据身份、单位、180 项唯一记录和均值/样本标准差。保留来源训练时间和资源峰值，
同一性质内按 RMSE 降序导出 CSV。不同机器耗时不是同硬件速度基准，不混排不同性质的误差。

如需同时生成补齐后的 LaTeX，在上述 merge 命令后加：

```bash
  --source-tex /tmp/qm9_single_task_rmse.tex \
  --output-tex /tmp/qm9_single_task_rmse_neural.tex
```

source-tex 为已有原表，output-tex 必须是新文件，保留原 ML、gap、时间列和模型分类。
