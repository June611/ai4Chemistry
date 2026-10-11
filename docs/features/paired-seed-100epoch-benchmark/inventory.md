# 协议纳入清单

更新时间：2026-10-10

统一协议 ID：`paired_split_gap_mae_v2`。三组运行是 `(3407,3407)`、`(42,42)`、
`(2026,2026)`；括号内依次为 split seed 与 training/model seed。

| 类别 | 稳定模型 ID | 数量 | 执行入口 |
|---|---|---:|---|
| Chemprop | `chemprop_single_gap`、`chemprop_multitask`、`chemprop_consistency` | 3 | `molgap-run-multiseed` |
| 序列模型 | `smiles_transformer`、`spe_transformer` | 2 | `molgap-run-multiseed` |
| SPE 三任务消融 | `spe_multitask`、`spe_multitask_affinity`、`spe_multitask_consistency`、`spe_multitask_affinity_consistency` | 4 | `molgap-run-multiseed` |
| QM9 方法 | `egnn`、`schnet`、`painn`、`dimenet_pp`、`esa`、`torchmd_net`、`cormorant`、`lie_transformer`、`alignn`、`3dinfomax` | 10 | `scripts/train_qm9_models.sh` |
| 传统基线 | `xgboost_rdkit29`、`random_forest_rdkit29`、`training_mean` | 3 | `scripts/train_ml_paired.sh` |

共 22 个评估模型、每个模型三组运行，共 66 次评估训练。3D Infomax 另有三次与 split
严格绑定的辅助预训练。神经网络预算为 batch 256、最多 100 epochs、patience 20；传统模型
没有这些概念，记录为 N/A。

以下配置不属于本协议：`configs/archive/**`、`configs/qm9_models/egnn_test.yaml`、
`configs/spe_transformer_10epoch.yaml`、`configs/experiments/exp003_consistency_0.yaml` 和
`configs/experiments/exp004_consistency_01.yaml`。

机器可读审计结果位于
[`reports/paired_seed_100epoch_protocol_audit.json`](../../../reports/paired_seed_100epoch_protocol_audit.json)。
