# Experiment configs

每次实验使用独立 YAML 和唯一的 `experiment.name`。Phase 3 的两个示例分别设置
`loss.weights.consistency: 0.0` 和 `0.1`，用于物理一致性约束消融。更换数据划分 seed 时，
必须先生成对应 processed split，并同步修改 `data.processed_dir`、`data.split.seed` 和
`training.seed`。
