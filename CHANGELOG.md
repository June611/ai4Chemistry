# 变更记录

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号沿用语义化的
“协议版本”语义：不兼容的实验口径变更记为主版本，新增模块/基线记为次版本。

## [Unreleased]

### Added
- 逐目标单任务基线配置 36 项（`dimenet_pp` / `torchmd_net` × homo/lumo/alpha/mu/r2/zpve × 3 seed），
  用于回答"每个学习目标上多任务/物理约束是否优于单任务"。
- 工程规范：本文件、`CONTRIBUTING.md`、`Makefile`、`.pre-commit-config.yaml`、
  GitHub Actions lint 工作流、`docs/README.md`、`configs/README.md`、`scripts/README.md`。

### Changed
- 统一格式化（`ruff format`），补充 `pyproject.toml` 的格式/覆盖率配置。

## [0.3.0] - 2026-10-25

### Added
- 多任务基线 MLoRE（低秩专家混合 + 噪声 top-k 路由 + 共享路径 detach + CV² 均衡损失），
  替换旧 task_attention7 路线（旧配置归档至 `configs/qm9_epoch200/archive_task_attention7_20261022/`）。
- DeMol 骨干接入与 DeMol + PINN 扩展实验；逐目标结果表。
- 扩展实验汇总 `docs/features/qm9-epoch200-modular-study/extension_results.md`（16 项扩展、20 组全场排名）。
- 结果工作簿：`reports/high_group_experiment_status.xlsx`、`reports/high_group_final_results.xlsx`。

### Changed
- 训练队列产物按 MLoRE 路线重新生成（`training_queue.csv`、`fresh_commands.sh`、`stage2_commands.sh`、
  `queue_summary.json`）。
- 测试适配新变体（`MLoREHead` 路由/均衡断言、队列契约校验）。

### Fixed
- 清理测试中残留的 `TaskAttentionHead` 引用与失效的 task_attention7 配置路径。

## [0.2.0] - 2026-10-20

### Added
- fresh-200 统一协议（200 epoch / batch 256 / patience 20 / train-only Z-score / test-only 终评），
  42 项正式矩阵（30 项单任务基线 + 6 项 PINN + 6 项多任务）。
- 4-GPU lane 运行器 `scripts/run_qm9_formal_4gpu.py` 与扩展链 `scripts/run_extension_chain.py`。
- 协议守卫 `molgap/training/protocol.py` 与确定性运行时策略 `molgap/qm9_models/runtime.py`。

### Deprecated
- 旧 100-epoch 配对协议冻结为历史证据（`docs/features/paired-seed-100epoch-benchmark/`）。

## [0.1.0] - 2026-09

### Added
- Chemprop / 传统 ML / SMILES·SPE Transformer / 10 个 QM9 神经模型的统一训练与评估入口。
- 共享数据划分、审计脚本与按功能组织的文档体系。
