> [!WARNING]
> **历史协议（已废弃，2026-10-20 归档）**：本 feature 记录旧 100-epoch（main.pdf 时代）协议。
> 神经基线已由 fresh-200 研究（200 epoch 上限）全部重训，见
> `docs/features/qm9-epoch200-modular-study/`。两套结果口径不同，**严禁混入同一张表**。
> 旧结果数据已移至 `outputs/summaries/archive_100epoch_qm9_results/`。

# 三组配对 seed、100-epoch 统一实验协议

2026-10-12执行范围更新：用户暂停Cormorant、LieTransformer，注册的22个评估模型仍保留，
实际启用20个模型、60组三组配对实验，另含3组Infomax预训练。预算256/100/20保持；
两个暂停模型不自动训练。[暂停决定与证据](../qm9-model-batch-sizing/README.md)

状态：统一gap MAE选模与早停已实现，loss/scaler按最新决定保留；66次全数据运行计划通过；代码最终审查见下方TXT，GPU与正式实验待服务器验收。

2026-10-11最新需求的统计、最小改动方案与交接TODO已建立：
[统一训练口径计划](standardization-plan.md)、[当前超参数清单](hyperparameter-inventory.csv)、
[U系列任务](tasks.md)。本轮已执行项目适配与配置变更；历史验证不代替本轮验收。

本 feature 将 Chemprop、SMILES/SPE Transformer、SPE 多任务消融、QM9 十模型和传统基线
统一到三组配对实验：

| 组 | split seed | training/model seed |
|---:|---:|---:|
| 1 | 3407 | 3407 |
| 2 | 42 | 42 |
| 3 | 2026 | 2026 |

神经网络的统一预算为 `batch_size=256`、`epochs=100`、early-stopping patience 20。
训练仍按验证集指标 early stop 并保存最佳 checkpoint。

按当前建议范围共 22 个评估模型：19 个神经网络和 3 个传统基线。每个模型运行三组，共
66 次评估训练；3D Infomax 另有三个与 split 严格绑定的预训练任务。

当前所有下游神经网络按原始Hartree的validation gap MAE选模，patience20，min_delta1e-5。
loss/scaler保留模型配方；额外监督与预训练单独披露。协议ID为paired_split_gap_mae_v2，
输出根目录outputs/paired_gap_mae_v2；旧v1结果不混入正式汇总。
2026-10-11旧审查 [review-20261011.txt](review-20261011.txt) 仅作历史记录，
本轮结果以[最终TXT审查](final-gap-mae-review.txt)与U系列TODO为准。

详细实施与审查步骤见 [tasks.md](tasks.md)，协议边界见 [decisions.md](decisions.md)，纳入模型见
[inventory.md](inventory.md)。
