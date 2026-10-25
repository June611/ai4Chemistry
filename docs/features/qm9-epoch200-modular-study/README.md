# QM9：200轮统一神经基线与可插拔模块

## 当前正式范围

本轮统一整理为与 `main.pdf` 对齐的 **42 个评估任务**，DeMol 不运行，Transformer-M 不列入 main.pdf 神经基线表，传统机器学习结果沿用历史结果不重跑。

| 组成 | 数量 | 说明 |
|---|---:|---|
| main.pdf 神经单任务基线 | 30 | 10 个模型 × 3 个配对 seed，仅统一重跑 `delta_e`/gap |
| PINN 模块 | 6 | DimeNet++、TorchMD-Net × `physics_thermo10` × 3 seed |
| 多任务模块 | 6 | DimeNet++、TorchMD-Net × `task_attention7` × 3 seed |
| **合计** | **42** | 10 个基线模型 + 2 条 SOTA 模块路径 |

main.pdf 的神经基线模型分组：

- 图结构：Chemprop、SchNet、3D Infomax、EGNN、PaiNN、DimeNet++。
- Transformer-based：SMILES Transformer、SPE Transformer、ESA、TorchMD-Net。
- DeMol 不在 main.pdf 表中，且 SDF 拓扑预处理耗时高，明确排除。
- 传统 ML（Ridge、Linear、HGB、LightGBM、XGBoost、Random Forest、Extra Trees）已有结果，不进入本轮 GPU 队列。

## 两章模块结构

| 章节 | 图路径 | Transformer 路径 |
|---|---|---|
| PINN 物理约束 | DimeNet++ + `physics_thermo10` | TorchMD-Net + `physics_thermo10` |
| 多任务学习 | DimeNet++ + `task_attention7` | TorchMD-Net + `task_attention7` |

DimeNet++ 与 TorchMD-Net 是沿用 main.pdf 100 轮结果选择的两类 SOTA 基础模型。模块任务额外预测七/十标签，且 PINN 有物理残差，因此模块结果不能直接伪装成单任务严格消融；所有比较必须明确记录监督标签和损失差异。

## 统一训练协议

- 配对 seed：`(42,42)`、`(2026,2026)`、`(3407,3407)`，split seed 与 training seed 一一对应。
- 最多 200 epochs，batch 256，patience 20，train-only target scaling，标准化 MSE。
- 单任务与联合任务均只用验证集选择 checkpoint；统一以原单位 Hartree `val_delta_e_mae` 作为 gap 选择标准。
- 测试集只在最终 checkpoint 上评估，报告原单位 MAE/RMSE；main.pdf 的旧 100 epoch RMSE 仅作历史参照，不与 fresh-200 数值混表。
- 每轮保存 last，每 50 轮保存完整快照，支持完整 epoch 边界恢复；TF32 关闭，CuBLAS workspace 固定。
- 3D Infomax 的三个 split-bound 预训练 checkpoint 是额外的准备任务，不计入 42 个下游评估任务；不在本轮重复预训练。

## 状态与验证

`training_queue.csv` 是正式执行唯一清单；不要递归运行 `configs/qm9_epoch200/fresh/` 下的全部 YAML，因为其中保留了扩展池、历史配置和非正式变体。正式训练前先执行配置/数据预检，再按 CSV 逐项运行。

当前代码生成器已输出 42 项、阶段 1 为 30 个单任务基线、阶段 2 为 12 个模块任务。DeMol 与 Transformer-M 在正式 queue 外。测试与 ruff 已通过；GPU0 上 42/42 项 500 条/1 轮烟测通过，记录见 `a100_smoke_unified_gpu0/` 和本目录追加证据。正式训练尚未启动。

## 文件入口

- [统一42项实验表](queue_multitask_pinn.md)
- [机器可读队列](../../../artifacts/qm9_epoch200/queue_multitask_pinn.csv)
- [正式逐项队列](training_queue.csv)
- [阶段1基线命令](fresh_commands.sh)
- [阶段2模块命令](stage2_commands.sh)
- [统一训练/恢复操作](operations.md)
- [任务与决策](tasks.md) · [TODO](TODO.md) · [决策](decisions.md)
- 历史 114 项烟测报告仍保留为历史证据，不代表当前正式队列范围。

刷新清单而不训练：

```bash
uv run --no-sync python scripts/prepare_qm9_epoch200_review.py
```
