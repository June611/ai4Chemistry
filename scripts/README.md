# 脚本索引

> 约定：脚本只做"入口/编排"，可复用逻辑放 `src/molgap/`。新增脚本请在此登记。

## 当前主线（fresh-200）

| 脚本 | 用途 |
|---|---|
| `run_qm9_formal_4gpu.py` | 正式 42 项队列，4 条 GPU lane 各自顺序执行（幂等：COMPLETED 跳过、`last.ckpt` 续训） |
| `run_extension_chain.py` | 扩展/补跑队列：等触发任务完成 + 显存空闲后顺序执行，供新实验接力 |
| `prepare_qm9_epoch200_review.py` | 生成 fresh-200 配置、队列 CSV、阶段命令与 `queue_summary.json`（从不训练） |
| `smoke_qm9_epoch200.py` | 用真实队列配置跑隔离的 CUDA 烟测（诊断产物，不污染正式目录） |
| `preflight_qm9_epoch200.py` | 训练前的输入/依赖/建模自检 |
| `resume_qm9_fresh200.py` | fresh-200 断点续训 |
| `resume_qm9_epoch200.py` | 旧 100→200 Lightning 续训（历史口径，与 fresh-200 分开） |
| `export_qm9_epoch200_tables.py` | 从 `artifacts/` 汇总生成论文表格（Markdown/CSV/XLSX） |
| `aux_wait_and_run.sh` | 等触发任务完成后在空闲 GPU 上跑辅助任务（lane 补充） |

## 审计与校验

| 脚本 | 用途 |
|---|---|
| `audit_paired_seed_protocol.py` | 校验三配对 seed 协议一致性 |
| `audit_qm9_split_seeds.py` | 校验划分成员、标签、血缘与几何 |
| `audit_qm9_training_scope.py` | 校验训练只用项目固定划分 |
| `audit_gap_runtime_plans.py` | 解析三 seed 计划（不训练） |
| `check_qm9_device.py` | 目标设备上的 Torch/DGL 探测 |
| `check_rdkit.py` | RDKit-29 描述符契约检查 |

## 数据准备

| 脚本 | 用途 |
|---|---|
| `prepare_data.py` / `prepare_qm9_epoch200_review.py` | 划分与配置准备 |
| `download_qm9_xyz.py` | 下载并安全解包 QM9 扩展 XYZ |
| `reindex_qm9_geometry.py` | 几何产物按划分 seed 重排 |
| `package_qm9_data.py` | 数据打包/校验（与 Git 分离传输） |
| `create_test_subset.py` | 小样本子集（EGNN 验证用） |
| `pretrain_3dinfomax_paired.sh` | 3D Infomax 分 seed 预训练 |
| `run_qm9_properties.py` | 六性质任务 CLI（任务定义在 `configs/qm9_six_properties/tasks.yaml`） |

## 历史入口（旧协议，保留可追溯）

`train_qm9_models.sh`、`train_ml_paired.sh`、`train_all_auto.sh`、`train_all_hybrid.sh`、
`train_multitask.py`、`train_single.py`、`run_multiseed.py`、`summarize_results.py`、
`run_one_epoch_smoke.py`、`run_qm9_dual_reproduction.py`、`train_qm9_{low,medium,high,group}.sh`、
`monitor_smoke_resources.py`、`complete_smoke_resource_report.py`、`qm9_server_preflight.sh`、
`evaluate.py`、`predict_library.py`。

> 这些脚本服务于 100-epoch 或更早的探索期流程，结果不与 fresh-200 混表。

## 运行纪律

- 一卡一任务；OOM 不静默降 batch。
- 长跑用链式脚本，断点续训幂等；杀进程与启动命令分两个 shell 调用（避免模式串自匹配）。
- 产物写 `artifacts/qm9_epoch200/<stage>_<date>/{runs,logs,chain_logs}`。
