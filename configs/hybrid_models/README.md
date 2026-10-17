# Hybrid Models Training Guide

## 快速开始

### 1. 训练单个配置

```bash
# Work 1: MPNN + Transformer (2D)
python -m molgap.training.train_hybrid \
    --data_dir ~/data/qm9 \
    --output_dir ./artifacts/hybrid_models/work1/hybrid_2d/seed_42 \
    --model_type work1 \
    --n_tasks 1 \
    --seed 42 \
    --gpus 1
```

### 2. 训练所有配置（18 runs）

```bash
bash scripts/train_all_hybrid.sh
```

预计时间：~12 小时（单卡 3090）

---

## 配置文件说明

### Work 1: 2D 混合架构

| 配置文件 | 模型 | 任务 | Seeds | 说明 |
|---|---|---|---|---|
| `work1/mpnn_only.yaml` | MPNN only | 1 (gap) | 3 | 验证 MPNN 复现 |
| `work1/hybrid_2d.yaml` | MPNN+Transformer | 1 (gap) | 3 | **工作一主模型** |

### Work 2: 3D 几何注入 + 多任务

| 配置文件 | λ | PE类型 | 任务 | Seeds | 说明 |
|---|---|---|---|---|---|
| `work2/lambda0.yaml` | 0.0 | MLP | 7 | 3 | 纯 2D（验证退化） |
| `work2/lambda05.yaml` | 0.5 | MLP | 7 | 3 | 中等几何注入 |
| `work2/lambda1_mlp.yaml` | 1.0 | MLP | 7 | 3 | 全几何（MLP PE） |
| `work2/lambda1_fourier.yaml` | 1.0 | Fourier | 7 | 3 | 全几何（Fourier PE） |

---

## 参数说明

### 模型参数

```yaml
# MPNN 参数（与 Chemprop 一致）
d_h: 300                # MPNN 隐藏层维度
mpnn_depth: 3           # MPNN 消息传递深度

# Transformer 参数
d_model: 256            # 必须能被 nhead 整除
nhead: 8                # 注意力头数
num_layers: 4           # Transformer 层数
dim_feedforward: 512    # FFN 维度
dropout: 0.1            # Dropout 率
```

### 3D 参数（仅 Work 2）

```yaml
pe_type: mlp            # 'mlp' 或 'fourier'
lambda_geometry: 1.0    # 几何注入强度 [0, 1]
learnable_lambda: false # 是否可学习
num_frequencies: 50     # Fourier PE 频率数
```

### 多任务参数

```yaml
n_tasks: 7              # 任务数（1=单任务，7=多任务）
use_uncertainty_weighting: true  # 不确定性加权损失
```

### 训练参数

```yaml
lr: 0.001               # 学习率
weight_decay: 0.0       # L2 正则化
warmup_epochs: 2        # Warmup epochs
max_epochs: 100         # 最大 epochs
batch_size: 32          # Batch size
seeds: [42, 3407, 12345]  # 随机种子
```

---

## 命令行参数（覆盖 YAML）

```bash
python -m molgap.training.train_hybrid \
    --data_dir ~/data/qm9 \
    --output_dir ./output \
    --model_type work2 \
    --pe_type fourier \
    --lambda_geometry 0.5 \
    --learnable_lambda \
    --n_tasks 7 \
    --use_uncertainty_weighting \
    --seed 42 \
    --max_epochs 100 \
    --batch_size 32 \
    --gpus 1
```

---

## 输出结构

```
artifacts/hybrid_models/
├── work1/
│   ├── mpnn_only/
│   │   ├── seed_42/
│   │   │   ├── checkpoints/
│   │   │   ├── logs/
│   │   │   ├── config_seed42.json
│   │   │   └── train.log
│   │   ├── seed_3407/
│   │   └── seed_12345/
│   └── hybrid_2d/
│       └── ...
└── work2/
    ├── lambda0/
    ├── lambda05/
    ├── lambda1_mlp/
    └── lambda1_fourier/
```

---

## 监控训练

### TensorBoard

```bash
tensorboard --logdir artifacts/hybrid_models/work1/hybrid_2d/seed_42/logs
```

### 查看日志

```bash
tail -f artifacts/hybrid_models/work1/hybrid_2d/seed_42/train.log
```

### 检查不确定性参数（多任务）

```bash
grep "sigma" artifacts/hybrid_models/work2/lambda1_mlp/seed_42/train.log
```

---

## 常见问题

### Q1: 如何只跑特定的 λ 值？

修改 `scripts/train_all_hybrid.sh`，注释掉不需要的行：

```bash
# train_config configs/hybrid_models/work2/lambda0.yaml  # 注释掉
train_config configs/hybrid_models/work2/lambda1_mlp.yaml  # 只跑这个
```

### Q2: 如何修改 seed？

编辑 YAML 文件的 `seeds` 字段：

```yaml
seeds: [42]  # 只跑一个 seed
```

### Q3: 如何在 CPU 上训练？

```bash
python -m molgap.training.train_hybrid ... --gpus 0
```

### Q4: 如何恢复训练？

Lightning 自动保存 checkpoint，使用 `--resume_from_checkpoint` 参数：

```bash
python -m molgap.training.train_hybrid \
    ... \
    --resume_from_checkpoint artifacts/.../checkpoints/best-epoch=50.ckpt
```

---

## 实验矩阵总结

| 工作 | 配置数 | Seeds | 总 runs | 预计时间（单卡） |
|---|---|---|---|---|
| Work 1 | 2 | 3 | 6 | ~3h |
| Work 2 | 4 | 3 | 12 | ~9h |
| **总计** | 6 | 3 | **18** | **~12h** |

---

## 下一步

训练完成后，参考 `docs/features/qm9-six-property-neural-baselines/` 的评估方法：

1. 提取所有 runs 的结果
2. 计算每个配置的平均值和标准差
3. 绘制帕累托前沿（精度 vs 时间）
4. 生成对比表格
