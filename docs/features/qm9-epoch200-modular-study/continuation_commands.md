# 100→200续训指令（独立补充实验）

先将本地当前代码部署到独立epoch200 checkout，原source-run保持只读。下列默认新checkout路径尚未在服务器创建。无卡旧westc只能预检，执行训练需挂GPU或完整搬迁原数据与权重。默认预检不训练，增加`--execute`才训练。不能将其混入fresh200结果。

## nmb2: chemprop_single_gap_homo_seed42

```bash
# 在单独部署的epoch200项目中执行，原source-run目录只读保留。
cd /root/autodl-tmp/evan/ai4Chemistry_epoch200
uv run python scripts/resume_qm9_epoch200.py --backend chemprop --source-run /root/autodl-tmp/evan/ai4Chemistry_qm9_low_20261016/artifacts/qm9_six_properties/low_formal01/runs/chemprop_single_gap/homo/seed42 --checkpoint /root/autodl-tmp/evan/ai4Chemistry_qm9_low_20261016/artifacts/qm9_six_properties/low_formal01/runs/chemprop_single_gap/homo/seed42/chemprop/model_0/checkpoints/last.ckpt --output-dir /root/autodl-tmp/evan/ai4Chemistry_qm9_low_20261016/artifacts/qm9_epoch200/continuation/chemprop_single_gap_homo_seed42
# 预检通过且GPU可用时才执行：
uv run python scripts/resume_qm9_epoch200.py --backend chemprop --source-run /root/autodl-tmp/evan/ai4Chemistry_qm9_low_20261016/artifacts/qm9_six_properties/low_formal01/runs/chemprop_single_gap/homo/seed42 --checkpoint /root/autodl-tmp/evan/ai4Chemistry_qm9_low_20261016/artifacts/qm9_six_properties/low_formal01/runs/chemprop_single_gap/homo/seed42/chemprop/model_0/checkpoints/last.ckpt --output-dir /root/autodl-tmp/evan/ai4Chemistry_qm9_low_20261016/artifacts/qm9_epoch200/continuation/chemprop_single_gap_homo_seed42 --execute
```

## nmb2: chemprop_single_gap_zpve_seed3407

```bash
# 在单独部署的epoch200项目中执行，原source-run目录只读保留。
cd /root/autodl-tmp/evan/ai4Chemistry_epoch200
uv run python scripts/resume_qm9_epoch200.py --backend chemprop --source-run /root/autodl-tmp/evan/ai4Chemistry_qm9_low_20261016/artifacts/qm9_six_properties/low_formal01/runs/chemprop_single_gap/zpve/seed3407 --checkpoint /root/autodl-tmp/evan/ai4Chemistry_qm9_low_20261016/artifacts/qm9_six_properties/low_formal01/runs/chemprop_single_gap/zpve/seed3407/chemprop/model_0/checkpoints/last.ckpt --output-dir /root/autodl-tmp/evan/ai4Chemistry_qm9_low_20261016/artifacts/qm9_epoch200/continuation/chemprop_single_gap_zpve_seed3407
# 预检通过且GPU可用时才执行：
uv run python scripts/resume_qm9_epoch200.py --backend chemprop --source-run /root/autodl-tmp/evan/ai4Chemistry_qm9_low_20261016/artifacts/qm9_six_properties/low_formal01/runs/chemprop_single_gap/zpve/seed3407 --checkpoint /root/autodl-tmp/evan/ai4Chemistry_qm9_low_20261016/artifacts/qm9_six_properties/low_formal01/runs/chemprop_single_gap/zpve/seed3407/chemprop/model_0/checkpoints/last.ckpt --output-dir /root/autodl-tmp/evan/ai4Chemistry_qm9_low_20261016/artifacts/qm9_epoch200/continuation/chemprop_single_gap_zpve_seed3407 --execute
```

## nmb2: smiles_transformer_homo_seed2026

```bash
# 在单独部署的epoch200项目中执行，原source-run目录只读保留。
cd /root/autodl-tmp/evan/ai4Chemistry_epoch200
uv run python scripts/resume_qm9_epoch200.py --backend sequence --source-run /root/autodl-tmp/evan/ai4Chemistry_qm9_low_20261016/artifacts/qm9_six_properties/low_formal01/runs/smiles_transformer/homo/seed2026 --checkpoint /root/autodl-tmp/evan/ai4Chemistry_qm9_low_20261016/artifacts/qm9_six_properties/low_formal01/runs/smiles_transformer/homo/seed2026/checkpoints/best-epoch=099-val_mae=0.004325.ckpt --output-dir /root/autodl-tmp/evan/ai4Chemistry_qm9_low_20261016/artifacts/qm9_epoch200/continuation/smiles_transformer_homo_seed2026
# 预检通过且GPU可用时才执行：
uv run python scripts/resume_qm9_epoch200.py --backend sequence --source-run /root/autodl-tmp/evan/ai4Chemistry_qm9_low_20261016/artifacts/qm9_six_properties/low_formal01/runs/smiles_transformer/homo/seed2026 --checkpoint /root/autodl-tmp/evan/ai4Chemistry_qm9_low_20261016/artifacts/qm9_six_properties/low_formal01/runs/smiles_transformer/homo/seed2026/checkpoints/best-epoch=099-val_mae=0.004325.ckpt --output-dir /root/autodl-tmp/evan/ai4Chemistry_qm9_low_20261016/artifacts/qm9_epoch200/continuation/smiles_transformer_homo_seed2026 --execute
```

## nmb2: spe_transformer_zpve_seed3407

```bash
# 在单独部署的epoch200项目中执行，原source-run目录只读保留。
cd /root/autodl-tmp/evan/ai4Chemistry_epoch200
uv run python scripts/resume_qm9_epoch200.py --backend sequence --source-run /root/autodl-tmp/evan/ai4Chemistry_qm9_low_20261016/artifacts/qm9_six_properties/low_formal01/runs/spe_transformer/zpve/seed3407 --checkpoint /root/autodl-tmp/evan/ai4Chemistry_qm9_low_20261016/artifacts/qm9_six_properties/low_formal01/runs/spe_transformer/zpve/seed3407/checkpoints/best-epoch=099-val_mae=0.000619.ckpt --output-dir /root/autodl-tmp/evan/ai4Chemistry_qm9_low_20261016/artifacts/qm9_epoch200/continuation/spe_transformer_zpve_seed3407
# 预检通过且GPU可用时才执行：
uv run python scripts/resume_qm9_epoch200.py --backend sequence --source-run /root/autodl-tmp/evan/ai4Chemistry_qm9_low_20261016/artifacts/qm9_six_properties/low_formal01/runs/spe_transformer/zpve/seed3407 --checkpoint /root/autodl-tmp/evan/ai4Chemistry_qm9_low_20261016/artifacts/qm9_six_properties/low_formal01/runs/spe_transformer/zpve/seed3407/checkpoints/best-epoch=099-val_mae=0.000619.ckpt --output-dir /root/autodl-tmp/evan/ai4Chemistry_qm9_low_20261016/artifacts/qm9_epoch200/continuation/spe_transformer_zpve_seed3407 --execute
```

## westc: chemprop_mtl7_affinity_seed42

```bash
# 在单独部署的epoch200项目中执行，原source-run目录只读保留。
cd /root/autodl-tmp/evan/ai4Chemistry_epoch200
uv run python scripts/resume_qm9_epoch200.py --backend multitask7 --source-run /root/autodl-tmp/evan/ai4Chemistry_mtl7_phase1_20261016/artifacts/qm9_multitask7_phase1_formal01/runs/chemprop_mtl7_affinity/seed42 --checkpoint /root/autodl-tmp/evan/ai4Chemistry_mtl7_phase1_20261016/artifacts/qm9_multitask7_phase1_formal01/runs/chemprop_mtl7_affinity/seed42/checkpoints/last.ckpt --output-dir /root/autodl-tmp/evan/ai4Chemistry_mtl7_phase1_20261016/artifacts/qm9_epoch200/continuation/chemprop_mtl7_affinity_seed42
# 预检通过且GPU可用时才执行：
uv run python scripts/resume_qm9_epoch200.py --backend multitask7 --source-run /root/autodl-tmp/evan/ai4Chemistry_mtl7_phase1_20261016/artifacts/qm9_multitask7_phase1_formal01/runs/chemprop_mtl7_affinity/seed42 --checkpoint /root/autodl-tmp/evan/ai4Chemistry_mtl7_phase1_20261016/artifacts/qm9_multitask7_phase1_formal01/runs/chemprop_mtl7_affinity/seed42/checkpoints/last.ckpt --output-dir /root/autodl-tmp/evan/ai4Chemistry_mtl7_phase1_20261016/artifacts/qm9_epoch200/continuation/chemprop_mtl7_affinity_seed42 --execute
```

## westc: spe_transformer_seed3407

```bash
# 在单独部署的epoch200项目中执行，原source-run目录只读保留。
cd /root/autodl-tmp/evan/ai4Chemistry_epoch200
uv run python scripts/resume_qm9_epoch200.py --backend sequence --source-run /root/autodl-tmp/evan/ai4Chemistry/artifacts/auto_training/chemprop_spe_formal_20261014/gpu1/runs/spe_transformer/seed3407 --checkpoint /root/autodl-tmp/evan/ai4Chemistry/artifacts/auto_training/chemprop_spe_formal_20261014/gpu1/runs/spe_transformer/seed3407/checkpoints/best-epoch=099-val_mae=0.006052.ckpt --output-dir /root/autodl-tmp/evan/ai4Chemistry/artifacts/qm9_epoch200/continuation/spe_transformer_seed3407
# 预检通过且GPU可用时才执行：
uv run python scripts/resume_qm9_epoch200.py --backend sequence --source-run /root/autodl-tmp/evan/ai4Chemistry/artifacts/auto_training/chemprop_spe_formal_20261014/gpu1/runs/spe_transformer/seed3407 --checkpoint /root/autodl-tmp/evan/ai4Chemistry/artifacts/auto_training/chemprop_spe_formal_20261014/gpu1/runs/spe_transformer/seed3407/checkpoints/best-epoch=099-val_mae=0.006052.ckpt --output-dir /root/autodl-tmp/evan/ai4Chemistry/artifacts/qm9_epoch200/continuation/spe_transformer_seed3407 --execute
```
