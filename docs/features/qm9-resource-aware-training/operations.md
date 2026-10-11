# 一键调度与飞书监测

适用：本项目 WSL、已有服务器 uv 环境和 RTX 3090。服务器保持现有 QM9 数据和模型环境，
无需再下载依赖。飞书凭据保留在 WSL 的 Lark CLI 加密存储中；服务器只训练、采样和保存结果。

## 启动

在 WSL 项目目录执行：

```bash
bash scripts/train_all_auto.sh --remote --directory artifacts/auto_training/formal_20261012
```

该命令在服务器后台运行队列，在本地持续同步飞书。关闭本地同步进程不停止服务器训练；
飞书自动刷新需要 WSL 同步进程在线。断开后可单独恢复同步：

```bash
bash scripts/train_all_auto.sh --sync --remote --directory artifacts/auto_training/formal_20261012
```

只在服务器训练、不接飞书时，在服务器项目目录运行：

```bash
nohup bash scripts/train_all_auto.sh --directory artifacts/auto_training/formal_20261012 > auto-training.log 2>&1 &
```

只生成计划，不执行训练：

```bash
bash scripts/train_all_auto.sh --plan --directory artifacts/auto_training/preview
```

默认计划为 63 项：20 个启用下游模型 × 3407/42/2026 三组配对 seed，加 3 次 Infomax QM9
预训练。Cormorant/LieTransformer 由中央暂停清单排除；明确指定它们也会报错。
神经网络保持 batch256、100epochs 上限、patience20；早停可以提前结束。传统模型一次拟合。
保留现有方法的损失、标准化、LR、模型和辅助监督规则，不宣称所有架构科研配方完全相同。
每个 GPU 任务使用 4 个 data workers、2 个 OMP/MKL 线程，这是资源配置改变。

诊断示例（完整固定划分；神经网络 1 epoch，batch256）：

```bash
bash scripts/train_all_auto.sh --remote --directory artifacts/auto_training/my_diagnostic --models chemprop_single_gap smiles_transformer training_mean --seeds 3407 --diagnostic-epochs 1
```

诊断时若 Chemprop CLI 的 epoch 不大于 warmup，生成配置将 warmup 改为 0；正式配置不改。
指定 Infomax 下游时，自动加入同 split/seed 的 QM9 预训练，等待预训练产物验收后启动下游。

## 查看与停止

以下两条在服务器项目目录执行：

```bash
bash scripts/train_all_auto.sh --status --directory artifacts/auto_training/formal_20261012
bash scripts/train_all_auto.sh --stop --directory artifacts/auto_training/formal_20261012
```

停止只作用于该队列自己的进程组。重新运行同一目录会跳过完成任务、继续未启动任务；
正在运行的控制器和遗留活跃 worker 受检查保护，避免重复训练。已中断或失败的任务不自动
从 checkpoint 恢复，不覆盖旧产物；重试时使用新的目录和 `--models/--seeds` 选择。

云端目标保存在本地 `configs/feishu_training.json`，只包含目标表 ID，登录由 Lark CLI 管理。
它不提交 Git、不复制服务器；换 WSL 时根据已有表链接填入 `base_token/table_id/hardware_table_id`。
命令格式与现有登录由官方 CLI 内置 `lark-shared`、`lark-base`、`lark-doc` 技能核验。

迁移本地同步端时可复制模板：

```bash
cp configs/feishu_training.example.json configs/feishu_training.json
# 按下方已有表链接填写三个ID，使用已授权的 lark-cli 用户登录
```

远端启动命令先校验本地表配置、登录和训练表schema，再请求服务器运行。
服务器环境与源码需提前部署；一键入口不会重新安装依赖或传输数据。

- [训练记录](https://my.feishu.cn/base/Nckeb46n6anmx9shVAXcnul6nsh?table=tblgtkzmgVWdVrWu)
- [硬件采样](https://my.feishu.cn/base/Nckeb46n6anmx9shVAXcnul6nsh?table=tblADD5ee9TfNKbM)

## 记录与测量口径

| 内容 | 存储与解释 |
|---|---|
| 每次运行 | 飞书训练记录一行，按运行 ID 更新，不按每个 batch 新增 |
| 模型结果 | 原始 Hartree gap 验证 MAE；测试 MAE/MSE/RMSE/MAPE/R²；MSE 单位 Hartree² |
| 数值精度 | 数字列最多4位小数；“指标完整精度_JSON”、原始JSON和CSV保留完整精度 |
| 推理速度 | 实际预测耗时、样本/秒、摊销毫秒/样本；记录计时范围 |
| 显存 | nvidia-smi 进程树显存与整卡显存；共享环境下整卡占用不归因于单模型 |
| 内存 | 进程树 RSS/PSS、cgroup 可用内存；PSS 更适合观察共享内存后的占用 |
| GPU 利用率 | 整卡 nvidia-smi utilization.gpu；不是精确 SM occupancy |
| 预测 | 产物验收后将完整测试预测 CSV 上传到飞书行的“预测结果附件” |
| 完整采样 | 队列目录 `runtime/hardware.jsonl`、逐任务 `runtime/<id>.jsonl`，默认约 5 秒采样 |
| 汇总 CSV | 队列目录 `runtime/summary.csv`，运行中刷新；服务器真实资源证据 |
| 日志与状态 | `runtime/<id>.log`、`runtime/state.json`、`runtime/report.json` |
| 云端采样 | 本地桥默认每轮同步后等待 60 秒；硬件表每分钟更新一条快照 |

FLOPs 表示计算量，不等于推理速度。本次没有完整覆盖所有架构的 FLOPs 测量，留空并标记
`not_measured`。缺失的历史推理时间、验证 MSE/RMSE 不由训练时间或 MAE 推算。
图模型的计时包含测试加载与前向；Transformer/SPE 还包含最佳 checkpoint 恢复；Chemprop CLI
包含进程启动、模型加载和特征准备；传统模型的计时只包含预计算特征上的 predict。
摊销时间不是单请求延迟，跨架构比较应匹配计时范围及并发条件。共享 GPU 的利用率也不能
作为单个模型独立的算力占用。采样峰值会遗漏瞬时尖峰，不是显存或内存硬上界。

## 简单准入规则

默认最多 2 个 GPU 任务、1 个 CPU 任务。共享任务显存预算为烟测峰值的 1.2 倍或峰值加
512MiB 中较大者，再给整卡留 2GiB。ALIGNN、DimeNet++ 独占 GPU，按测量峰值另留 1GiB，
避免重复膨胀预算导致它们永远不能进入 24GiB 显卡。使用实时 free 显存，包含驱动保留空间。

同时检查 CPU 预算、cgroup 内存、磁盘和最近 GPU 利用率；新任务间隔默认 30 秒，减少尚未
分配显存时连续启动的风险。内存预留 20GiB，数据盘预留 10GiB。GPU 利用率达到 85% 时
等待，不因显存剩余就强行塞入模型。任务按估计单轮耗时降序调度可运行项。
这是一种保守的启发式调度；不能保证绝不 OOM，也不能保证全局最短时间。并发收益待实测校准。

训练失败记录返回码和日志，继续其他任务；依赖失败阻止对应下游。飞书错误不会结束服务器
训练，修复网络或登录后重跑同步可回填。磁盘低于预留值暂停新任务，已经运行的任务继续。
配置文件路径、模型和 seed 在生成计划后冻结；同一目录不接受新的模型选择或 epoch 修改。
新计划保存生成配置SHA-256，配置内容变更后拒绝启动，需使用新目录；历史无哈希计划仍可读取。
没有CSV/JSON训练历史的入口不会解析TensorBoard猜测验证指标；缺失字段留空。
