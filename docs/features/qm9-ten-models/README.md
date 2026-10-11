# QM9 十模型接入

本功能把 [输入清单](inputs/qm9_only_10_official_models.csv) 中的十种方法接入项目统一的
`delta_e` 回归协议。所有方法可选择 split seed `3407`、`42` 或 `2026`，并复用对应划分、
训练集拟合的目标缩放、验证集选模和独立测试集终评。

| 模型 | 正式配置 | 方法适配 | 当前结论 |
|---|---|---|---|
| EGNN | `egnn.yaml` | 官方 E(n)-GNN | 可训练 |
| SchNet | `schnet.yaml` | SchNetPack | 可训练 |
| PaiNN | `painn.yaml` | SchNetPack | 可训练 |
| DimeNet++ | `dimenet_pp.yaml` | PyG DimeNet++ | 可训练 |
| ESA | `esa.yaml` | 官方 edge attention + PMA | 可训练 |
| TorchMD-Net | `torchmd_net.yaml` | 官方 Equivariant Transformer | 可训练 |
| Cormorant | `cormorant.yaml` | 官方 Cormorant | 可训练 |
| LieTransformer | `lie_transformer.yaml` | 官方 LieTransformer | 可训练 |
| ALIGNN | `alignn.yaml` | 官方 ALIGNN | 可训练 |
| 3D Infomax | `3dinfomax.yaml` | 官方 2D PNA 下游编码器 | 需先提供 QM9 预训练权重 |

服务器首次运行：

```bash
bash scripts/qm9_server_preflight.sh
```

迁移时不要复制本地 `.venv/`、`.external/` 和 `outputs/`。可复制已验证的
`data/geometry/qm9_full_seed*.npz` 及同名 JSON；若不复制 seed 3407 artifact，预检脚本会从固定
URL 下载原始 XYZ 并重建；后两套 geometry 缺失时会从 seed 3407 按 manifest 无损重排。三个
固定 split 目录必须随代码一起迁移。

该命令安装锁定依赖、拉取固定上游提交、检查 Torch/DGL 的 GPU 设备支持、
在缺少几何时下载并校验 QM9 原始 XYZ、运行
19 项 adapter sanity tests，审计三套 split，并对十个模型执行正式配置 dry-run。3D Infomax
的 dry-run 需要
与当前 split 匹配的预训练 checkpoint；若已有 seed 3407 的检查点：

```bash
QM9_PREFLIGHT_SPLIT_SEEDS=3407 \
INFOMAX_CHECKPOINT=/path/to/best_checkpoint.pt \
  bash scripts/qm9_server_preflight.sh
```

三套 split 都已有独立 checkpoint 时，使用

```bash
INFOMAX_CHECKPOINT_TEMPLATE='/path/to/splitseed{split_seed}/best_checkpoint.pt' \
  bash scripts/qm9_server_preflight.sh
```

如尚无检查点，先为三套 split 分别运行对比预训练：

```bash
bash scripts/pretrain_3dinfomax_paired.sh
```

正式运行三组配对 seed：

```bash
bash scripts/train_qm9_models.sh
```

脚本默认依次运行 `(3407,3407)`、`(42,42)`、`(2026,2026)`，前一项是 split seed，后一项
是训练 seed。若只运行一组，设置 `QM9_PAIRED_SEEDS="3407"`。3D Infomax 用模板提供三个
相互独立的 checkpoint：

```bash
INFOMAX_CHECKPOINT_TEMPLATE='outputs/paired_gap_mae_v2/3dinfomax_pretrain/splitseed{split_seed}/seed{seed}/best_checkpoint.pt' \
  bash scripts/train_qm9_models.sh
```

本地验收结果及尚待服务器运行的正式训练见 [tasks.md](tasks.md)，设计依据见
[decisions.md](decisions.md)，逐项源码与公平性结论见
[reproduction-audit.md](reproduction-audit.md)。

本地 CPU sanity/dry-run 不代表 GPU 验收。预检中的 `check_qm9_device.py --require-cuda`
在 CUDA 不可用或 DGL 创图转移到 GPU 失败时停止；需在服务器确认 PyTorch/CUDA/DGL 的
兼容构建，不能只凭 `torch.cuda.is_available()` 判断 ALIGNN/Infomax 可运行。
该探测只验证基础设备操作，batch 256 的完整前向/反向和显存仍待逐模型检查。
