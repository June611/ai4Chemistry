# 统一实验表：main.pdf 神经基线 + PINN/多任务章节

本表对应正式 `training_queue.csv` 的 **42 个评估任务**：main.pdf 中 10 个神经基线各重跑 gap 单任务 3 个配对 seed（30 项），再在 DimeNet++/TorchMD-Net 两个 SOTA 基础模型上分别增加多任务与 PINN（12 项）。传统 ML 结果沿用 main.pdf；DeMol 和 Transformer-M 不运行。GPU0 上 500 条/1 轮 smoke 已完成 **42/42 通过**。

## 1. 主表对照：10 个神经基线（30 项）

| 家族 | 模型 | main.pdf gap RMSE | 输入/差异 | Seed | 任务 ID | 配置 | 预计时长(h) |
|---|---|---:|---|---:|---|---|---:|
| 图结构 | Chemprop | 0.007018 Ha | 2D | 42 | `chemprop_single_task_delta_e_seed42` | `configs/qm9_epoch200/fresh/chemprop/delta_e_seed42.yaml` | 0.57 |
| 图结构 | Chemprop | 0.007018 Ha | 2D | 2026 | `chemprop_single_task_delta_e_seed2026` | `configs/qm9_epoch200/fresh/chemprop/delta_e_seed2026.yaml` | 0.57 |
| 图结构 | Chemprop | 0.007018 Ha | 2D | 3407 | `chemprop_single_task_delta_e_seed3407` | `configs/qm9_epoch200/fresh/chemprop/delta_e_seed3407.yaml` | 0.57 |
| 图结构 | SchNet | 0.007320 Ha | 3D | 42 | `schnet_single_task_delta_e_seed42` | `configs/qm9_epoch200/fresh/schnet/single_task_seed42.yaml` | 3.88 |
| 图结构 | SchNet | 0.007320 Ha | 3D | 2026 | `schnet_single_task_delta_e_seed2026` | `configs/qm9_epoch200/fresh/schnet/single_task_seed2026.yaml` | 3.88 |
| 图结构 | SchNet | 0.007320 Ha | 3D | 3407 | `schnet_single_task_delta_e_seed3407` | `configs/qm9_epoch200/fresh/schnet/single_task_seed3407.yaml` | 3.88 |
| 图结构 | 3D Infomax | 0.006295 Ha | 3D / pretrain | 42 | `3dinfomax_single_task_delta_e_seed42` | `configs/qm9_epoch200/fresh/3dinfomax/single_task_seed42.yaml` | 6.59 |
| 图结构 | 3D Infomax | 0.006295 Ha | 3D / pretrain | 2026 | `3dinfomax_single_task_delta_e_seed2026` | `configs/qm9_epoch200/fresh/3dinfomax/single_task_seed2026.yaml` | 6.59 |
| 图结构 | 3D Infomax | 0.006295 Ha | 3D / pretrain | 3407 | `3dinfomax_single_task_delta_e_seed3407` | `configs/qm9_epoch200/fresh/3dinfomax/single_task_seed3407.yaml` | 6.59 |
| 图结构 | EGNN | 0.003545 Ha | 3D | 42 | `egnn_single_task_delta_e_seed42` | `configs/qm9_epoch200/fresh/egnn/single_task_seed42.yaml` | 3.59 |
| 图结构 | EGNN | 0.003545 Ha | 3D | 2026 | `egnn_single_task_delta_e_seed2026` | `configs/qm9_epoch200/fresh/egnn/single_task_seed2026.yaml` | 3.59 |
| 图结构 | EGNN | 0.003545 Ha | 3D | 3407 | `egnn_single_task_delta_e_seed3407` | `configs/qm9_epoch200/fresh/egnn/single_task_seed3407.yaml` | 3.59 |
| 图结构 | PaiNN | 0.003407 Ha | 3D | 42 | `painn_single_task_delta_e_seed42` | `configs/qm9_epoch200/fresh/painn/single_task_seed42.yaml` | 4.27 |
| 图结构 | PaiNN | 0.003407 Ha | 3D | 2026 | `painn_single_task_delta_e_seed2026` | `configs/qm9_epoch200/fresh/painn/single_task_seed2026.yaml` | 4.27 |
| 图结构 | PaiNN | 0.003407 Ha | 3D | 3407 | `painn_single_task_delta_e_seed3407` | `configs/qm9_epoch200/fresh/painn/single_task_seed3407.yaml` | 4.27 |
| 图结构 | DimeNet++ | 0.002731 Ha | 3D | 42 | `dimenet_pp_single_task_delta_e_seed42` | `configs/qm9_epoch200/fresh/dimenet_pp/single_task_seed42.yaml` | 6.95 |
| 图结构 | DimeNet++ | 0.002731 Ha | 3D | 2026 | `dimenet_pp_single_task_delta_e_seed2026` | `configs/qm9_epoch200/fresh/dimenet_pp/single_task_seed2026.yaml` | 6.95 |
| 图结构 | DimeNet++ | 0.002731 Ha | 3D | 3407 | `dimenet_pp_single_task_delta_e_seed3407` | `configs/qm9_epoch200/fresh/dimenet_pp/single_task_seed3407.yaml` | 6.95 |
| Transformer-based | SMILES Transformer | 0.009764 Ha | 2D | 42 | `smiles_transformer_single_task_delta_e_seed42` | `configs/qm9_epoch200/fresh/smiles_transformer/single_task_seed42.yaml` | 0.90 |
| Transformer-based | SMILES Transformer | 0.009764 Ha | 2D | 2026 | `smiles_transformer_single_task_delta_e_seed2026` | `configs/qm9_epoch200/fresh/smiles_transformer/single_task_seed2026.yaml` | 0.90 |
| Transformer-based | SMILES Transformer | 0.009764 Ha | 2D | 3407 | `smiles_transformer_single_task_delta_e_seed3407` | `configs/qm9_epoch200/fresh/smiles_transformer/single_task_seed3407.yaml` | 0.90 |
| Transformer-based | SPE Transformer | 0.009453 Ha | 2D | 42 | `spe_transformer_single_task_delta_e_seed42` | `configs/qm9_epoch200/fresh/spe_transformer/single_task_seed42.yaml` | 0.92 |
| Transformer-based | SPE Transformer | 0.009453 Ha | 2D | 2026 | `spe_transformer_single_task_delta_e_seed2026` | `configs/qm9_epoch200/fresh/spe_transformer/single_task_seed2026.yaml` | 0.92 |
| Transformer-based | SPE Transformer | 0.009453 Ha | 2D | 3407 | `spe_transformer_single_task_delta_e_seed3407` | `configs/qm9_epoch200/fresh/spe_transformer/single_task_seed3407.yaml` | 0.92 |
| Transformer-based | ESA | 0.008452 Ha | 2D | 42 | `esa_single_task_delta_e_seed42` | `configs/qm9_epoch200/fresh/esa/single_task_seed42.yaml` | 5.85 |
| Transformer-based | ESA | 0.008452 Ha | 2D | 2026 | `esa_single_task_delta_e_seed2026` | `configs/qm9_epoch200/fresh/esa/single_task_seed2026.yaml` | 5.85 |
| Transformer-based | ESA | 0.008452 Ha | 2D | 3407 | `esa_single_task_delta_e_seed3407` | `configs/qm9_epoch200/fresh/esa/single_task_seed3407.yaml` | 5.85 |
| Transformer-based | TorchMD-Net | 0.004056 Ha | 3D | 42 | `torchmd_net_single_task_delta_e_seed42` | `configs/qm9_epoch200/fresh/torchmd_net/single_task_seed42.yaml` | 5.84 |
| Transformer-based | TorchMD-Net | 0.004056 Ha | 3D | 2026 | `torchmd_net_single_task_delta_e_seed2026` | `configs/qm9_epoch200/fresh/torchmd_net/single_task_seed2026.yaml` | 5.84 |
| Transformer-based | TorchMD-Net | 0.004056 Ha | 3D | 3407 | `torchmd_net_single_task_delta_e_seed3407` | `configs/qm9_epoch200/fresh/torchmd_net/single_task_seed3407.yaml` | 5.84 |

> main.pdf 数值是历史 100 epoch RMSE；本轮 fresh-200 统一选择/报告使用原单位 `val_delta_e_mae` 和最终测试 MAE/RMSE，不能把两套数值直接混成一张结果表。

## 2. PINN 章节：SOTA 基础模型 + PhysicsThermo10（6 项）

十标签为七个 QM9 属性加 `u0/u298/h298`，物理残差为 `gap=LUMO-HOMO`、`H298-U298≈kBT`、`U298≥U0`；权重 0.1，warmup 10 + ramp 10。

| 路径 | 模型 | Seed | 任务 ID | 配置 | 预计时长(h) |
|---|---|---:|---|---|---:|
| 图结构 | DimeNet++ | 42 | `dimenet_pp_physics_thermo10_seed42` | `configs/qm9_epoch200/fresh/dimenet_pp/physics_thermo10_seed42.yaml` | 7.30 |
| 图结构 | DimeNet++ | 2026 | `dimenet_pp_physics_thermo10_seed2026` | `configs/qm9_epoch200/fresh/dimenet_pp/physics_thermo10_seed2026.yaml` | 7.30 |
| 图结构 | DimeNet++ | 3407 | `dimenet_pp_physics_thermo10_seed3407` | `configs/qm9_epoch200/fresh/dimenet_pp/physics_thermo10_seed3407.yaml` | 7.30 |
| Transformer-based | TorchMD-Net | 42 | `torchmd_net_physics_thermo10_seed42` | `configs/qm9_epoch200/fresh/torchmd_net/physics_thermo10_seed42.yaml` | 6.13 |
| Transformer-based | TorchMD-Net | 2026 | `torchmd_net_physics_thermo10_seed2026` | `configs/qm9_epoch200/fresh/torchmd_net/physics_thermo10_seed2026.yaml` | 6.13 |
| Transformer-based | TorchMD-Net | 3407 | `torchmd_net_physics_thermo10_seed3407` | `configs/qm9_epoch200/fresh/torchmd_net/physics_thermo10_seed3407.yaml` | 6.13 |

## 3. 多任务学习章节：SOTA 基础模型 + TaskAttention7（6 项）

七标签联合预测 `homo/lumo/delta_e/alpha/mu/r2/zpve`，TaskAttentionHead 为 hidden 128、4 heads、dropout 0.1。

| 路径 | 模型 | Seed | 任务 ID | 配置 | 预计时长(h) |
|---|---|---:|---|---|---:|
| 图结构 | DimeNet++ | 42 | `dimenet_pp_task_attention7_seed42` | `configs/qm9_epoch200/fresh/dimenet_pp/task_attention7_seed42.yaml` | 7.30 |
| 图结构 | DimeNet++ | 2026 | `dimenet_pp_task_attention7_seed2026` | `configs/qm9_epoch200/fresh/dimenet_pp/task_attention7_seed2026.yaml` | 7.30 |
| 图结构 | DimeNet++ | 3407 | `dimenet_pp_task_attention7_seed3407` | `configs/qm9_epoch200/fresh/dimenet_pp/task_attention7_seed3407.yaml` | 7.30 |
| Transformer-based | TorchMD-Net | 42 | `torchmd_net_task_attention7_seed42` | `configs/qm9_epoch200/fresh/torchmd_net/task_attention7_seed42.yaml` | 6.13 |
| Transformer-based | TorchMD-Net | 2026 | `torchmd_net_task_attention7_seed2026` | `configs/qm9_epoch200/fresh/torchmd_net/task_attention7_seed2026.yaml` | 6.13 |
| Transformer-based | TorchMD-Net | 3407 | `torchmd_net_task_attention7_seed3407` | `configs/qm9_epoch200/fresh/torchmd_net/task_attention7_seed3407.yaml` | 6.13 |

## 4. 统一运行协议

| 项目 | 统一要求 |
|---|---|
| 配对 seed | `(42,42)`, `(2026,2026)`, `(3407,3407)` |
| 训练 | fresh-200，batch 256，patience 20，train-only scaling，标准化 MSE |
| 选择 | 验证集原单位 Hartree `val_delta_e_mae`；测试集仅最终 checkpoint |
| 保存 | 每轮 last，每50轮完整快照，支持完整 epoch 边界恢复 |
| DeMol | 排除，不运行；Transformer-M 不是 main.pdf 正式神经基线，暂不运行 |
| 3D Infomax | 3 个 split-bound 预训练 checkpoint 为额外准备任务，不计入 42 项评估任务 |
| 传统 ML | 沿用 main.pdf 历史结果，不进入本轮 GPU 队列 |

## 5. 时间与执行统计

| 分组 | 项数 | 预计 GPU·h（170 epoch/项） |
|---|---:|---:|
| 10 个神经单任务基线 | 30 | 118.1 |
| PINN 模块 | 6 | 40.3 |
| 多任务模块 | 6 | 40.3 |
| **合计评估任务** | **42** | **198.6** |

4 卡理想负载均衡约 **49.7 h（2.1 天）**；实际需加 I/O、早停分布、Infomax CPU 开销和 lane 不均衡。

## 6. 执行边界

只运行 `training_queue.csv` 中的 42 行。不要递归执行 `configs/qm9_epoch200/fresh/`，因为其中还保留扩展池、历史配置和非正式变体。
