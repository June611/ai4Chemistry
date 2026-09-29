# Experiment configs

每次实验使用独立 YAML 和唯一的 `experiment.name`。Phase 3 的两个示例分别设置
`loss.weights.consistency: 0.0` 和 `0.1`，用于物理一致性约束消融。重复训练使用同一份
processed split，只修改 `training.seed`；`data.processed_dir` 和 `data.split.seed` 必须保持
不变。不同数据划分方法应作为独立实验设计，不得混入训练 seed 重复实验。
