# QM9：200轮训练与可插拔模块

审查报告列出的功能已经编码并完成本地检查：任务注意力七任务、物理约束十任务、十标签数据扩展、统一训练入口、50轮完整快照与中断恢复、114项配置/指令。**3090上114/114项实际一轮GPU烟测通过（各分区500条、batch256），拟提交干净副本73项测试通过。未启动正式训练；全量最大输入batch和正式并发/容量仍待校准。**

## 当前任务

| 阶段 | 内容 | 三seed作业数 | 实现状态 |
|---|---|---:|---|
| 1 | Chemprop七性质独立单任务 | 21 | 统一入口，保留原BondMPNN和原MLP |
| 1 | Chemprop D1/D2 | 6 | 复用原七头/亲和力/UW结构及损失 |
| 1 | TaskAttention7 / PhysicsThermo10 | 6 | 已实现，CPU恢复及3090真实数据前缀诊断通过 |
| 2 | 其他11骨干的两个模块 | 66 | 3090上全部66项一轮GPU烟测通过；其他机器仍需环境/资产预检 |
| 2 | 五个图骨干gap基础模型 | 15 | 复用原骨干预测结构，统一Z-score/MSE和200轮 |
| 合计 | 全阶段候选 | 114 | 全部有配置与命令，114项一轮GPU烟测通过；正式训练未执行 |

B/C、细分消融、已有Transformer基础重训仍不安排。3D Infomax复用对应seed的QM9训练分区预训练权重，不自动重做预训练。历史100轮入口和结果不改。

## 阅读顺序

- [114项GPU烟测报告及资源CSV](smoke114_report.md)
- [本轮实现交付及验收表](implementation_delivery.md)
- [模型、优先级及详细MTL/PINN方案](model_and_module_lists.md)
- [运行与恢复指令](operations.md)
- [逐项CSV](training_queue.csv) · [第一阶段33项指令](fresh_commands.sh) · [后续81项指令](stage2_commands.sh)
- [TODO](TODO.md) · [任务](tasks.md) · [决策](decisions.md)
- [实现前审查记录](implementation_double_check_20261019.md)

刷新清单而不训练：`uv run python scripts/prepare_qm9_epoch200_review.py`。

所有当前配置使用`molgap.modular.train`，不再把新模块交给旧七任务入口。顶层`training`控制200/256/20、MAE选模与标准化；`source`是原模型参考配置快照，其中历史100轮/MAD/MAE等字段用于追溯，训练器只从该段读取骨干、原优化器/LR、预热和梯度裁剪设置，不拿历史预算或loss执行新训练。

## 统一协议

七/十标签逐目标仅用完整训练集Z-score；训练为标准化MSE，D1/D2复用原UW；物理方案在逆变换后的预测值上用float64计算残差并归一化。联合模型按原单位Hartree验证gap MAE选唯一模型及早停，单任务按该性质原单位MAE。测试指标恢复原单位，不参与选模。

每轮更新完整last，每50轮另存完整快照，结束保存terminal。完整状态含模型、优化器、scheduler、早停进度、RNG、loader generator、配置/scaler/数据指纹及历史最佳权重。仅支持**完整epoch边界**恢复；进程在某轮中间中断时重做该未完成轮，不宣称从任意batch逐位恢复。当前确定性数据加载器的双worker恢复已在沙箱外本地验证；不涵盖未来新增的随机数据增强或不同GPU间逐位一致性。

历史100轮、不同loss/scaler的结果仍只能作为带协议标记的历史参照。新模块效果是否提升尚需正式训练证明。
