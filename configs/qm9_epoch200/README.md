# 当前200轮配置

唯一执行清单为`docs/features/qm9-epoch200-modular-study/training_queue.csv`，114项均接入`molgap.modular.train`。不要递归遍历所有YAML：旧B/C、references和planned文件仍留作历史记录。

第一阶段33项指令在fresh_commands.sh，后续81项在stage2_commands.sh。新入口统一200/256/20、训练集逐标签Z-score/MSE、原单位验证MAE选模；source段是原模型参考快照，不使用其中历史100轮或loss执行新实验。

输入、可选依赖、Infomax预训练文件、GPU与磁盘容量仍需预检。完整用法见`docs/features/qm9-epoch200-modular-study/operations.md`。

2026-10-19：114项已在3090完成500条/1轮/batch256的GPU烟测，见docs/features/qm9-epoch200-modular-study/smoke114_report.md。YAML/status与CSV/status保留生成时的待验收标记，不改动本轮实际受测配置的hash；当前验收以verification/smoke114_nmb2_20261019.json及其配置hash为准。重新生成queue_summary会重置生成时的验收标记，不会删除已归档证据。
