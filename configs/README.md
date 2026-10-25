# 配置目录约定

```
configs/
├── qm9_epoch200/                 # fresh-200 统一协议（当前主线）
│   ├── fresh/<model>/            # 可执行配置：<variant>_seed<seed>.yaml
│   │                             #   单任务: single_task_seed<seed>.yaml（gap）
│   │                             #   逐目标: <target>_seed<seed>.yaml（homo/lumo/alpha/mu/r2/zpve）
│   │                             #   模块:   mlore7_seed<seed>.yaml, physics_thermo10_seed<seed>.yaml
│   ├── archive_<route>_<date>/   # 被取代路线的归档（保留历史，不参与运行）
│   └── ...                       # 旧 114 项探索期配置（历史，部分已不可运行）
├── qm9_multitask7/               # 旧多任务 7 目标定义（历史）
├── qm9_six_properties/           # 六性质单任务任务定义
├── multiseed*.yaml               # 旧 100-epoch 配对协议矩阵（历史，见 docs/features/paired-seed-100epoch-benchmark）
└── spe_affinity_ablation_*.yaml  # 旧 SPE 消融（历史）
```

## 规则

1. **可执行配置自包含**：`training` 段固定 200 epoch / batch 256 / patience 20 /
   `train_only_per_target_zscore` / `val_<target>_mae` 选型；`module` 段声明变体与超参。
2. **新增模型或变体**：用 `scripts/prepare_qm9_epoch200_review.py` 的
   `make_executable_config(model, variant, seed, target)` 生成，勿手写 YAML。
3. **路线更替**：把旧配置 `git mv` 到 `archive_<route>_<date>/` 并写 `README.md` 说明取代原因，
   旧结果保留在 `artifacts/` 与对应 feature 文档中，不删除。
4. **历史配置不删**：旧协议（100-epoch）与探索期配置保留可追溯性，但不得与新协议结果混表；
   文件头用注释标注 `[HISTORICAL ...]`。
5. **命名**：`<model>_<variant>[_<target>]_seed<seed>.yaml`；`experiment.name` 与文件名一致。
