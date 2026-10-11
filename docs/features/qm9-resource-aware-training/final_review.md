# 调度系统最终审查：2026-10-12

结论：资源调度、监测、一键入口和飞书同步的实现验收通过，可提交为feature。
正式100epoch训练与并发性能校准属于后续实验，没有启动，也不计入本次完成范围。
本次提交包含此前暂停的监测代码及恢复后修复；未更改模型架构和科学训练配方。

## 验收证据

| 检查 | 实际结果 |
|---|---|
| 全套本地回归 | `pytest -q`：141 passed，37 warnings，191.62s |
| 服务器相关回归 | scheduling/tracking/metrics：32 passed，2.29s |
| 静态检查 | `ruff check .`、`bash -n scripts/train_all_auto.sh`通过 |
| 默认计划 | 20个启用下游模型×3组配对seed + 3组QM9 Infomax预训练，共63项 |
| 预算与暂停 | 3407/42/2026；NN 256/100/20；Cormorant/LieTransformer拒绝入队 |
| 真实诊断 | 8项completed；7个下游测试各13380样本；Infomax预训练不伪造gap指标 |
| 产物检查 | 固定测试样本身份与覆盖、有限预测值、重算gap MAE、模型/checkpoint检查通过 |
| 一键入口 | 对已完成诊断目录再次执行；跳过训练，飞书同步8项、0错误 |
| 云端回读 | 8条completed，7条预测附件，8条完整精度JSON与本地一致；训练表60字段 |
| 数字显示 | 8个指标/LR列precision=4；JSON/CSV保留完整数值，不以显示值做科研比较 |
| 停止及容错 | 实际子进程停止、CSV/JSON终态一致、失败隔离、监测读取故障、重复任务保护通过 |
| 配置保护 | 新计划保存配置SHA-256；内容被修改后拒绝运行；远端路径越界拒绝 |

完整机器可读结论见[final_acceptance.json](verification/final_acceptance.json)，
8项诊断的可移植资源摘要见[diagnostic_acceptance.csv](verification/diagnostic_acceptance.csv)。
服务器完整CSV：`/root/autodl-tmp/evan/ai4Chemistry/artifacts/auto_training/diagnostic_20261012/runtime/summary.csv`。
逐任务日志、完整采样和状态在同一目录`runtime/`；完整预测与checkpoint保留在服务器`runs/`。
本地云端原始回执在`artifacts/auto_training/feishu/`，按既有规则不纳入Git。

## 实现边界

- GPU准入同时检查预测预算与实时空闲，独占任务使用独立余量；CPU、RAM、磁盘和GPU利用率
  参与准入。已观察到Infomax预训练、SchNet和CPU训练均值重叠，但没有测量相同任务集的
  串行对照，不能声称最短完成时间或固定加速比。
- nvidia-smi利用率为整卡`utilization.gpu`，不是SM occupancy；共享GPU利用率不归因于单模型。
  峰值来自5秒离散采样，不能保证没有瞬时OOM；小于采样间隔的CPU任务可能无内存峰值记录。
- 推理耗时、样本/秒和摊销时间已实测；计时范围按入口记录，不能直接当成单请求延迟。
  FLOPs尚未覆盖所有模型，留空并明确`not_measured`。
- 缺失的验证MSE/RMSE或Chemprop CLI CSV训练进度不由测试结果推算；没有原始来源就留空。
  现有模型loss/scaler/LR保持各自冻结配置，不能宣称所有架构科研配方完全相同。
- 中断/失败保留原产物，但不提供所有模型统一checkpoint续训；重试使用新目录。
  已完成状态基于任务结束时的产物验收；重复运行不会重新校验这些文件是否被事后修改。
  生成配置改变需新计划；远端入口对已有目录直接恢复原计划，选择覆盖仅在创建新目录时生效。
- 飞书只使用本地已授权CLI登录；服务器无需应用密钥。飞书同步中断不会停止远端训练，
  重新运行同步可回填；刷新期间WSL同步端需在线。
- 正式训练仍需逐任务确认长训练资源峰值、早停/选模、三组划分隔离、Infomax正式预训练来源
  和完整结果。诊断权重与误差不用于正式论文比较。

## 交付与后续

本feature提交保存源码、配置模板、测试、操作指南与小型验收证据；数据、权重、登录缓存、
实际飞书目标配置和原始运行日志不进入Git。服务器源码通过安全部署包同步，数据集沿用现有
QM9文件，不重复上传。后续待办保留在[TODO.md](TODO.md)；暂停时快照保留在[PAUSED.md](PAUSED.md)。
