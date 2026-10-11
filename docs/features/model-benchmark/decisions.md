# 多方法接入：关键决定

## 2026-10-09：按功能组织，一个仓库接入多个方法

来源：用户希望将其他复现方法全部接入当前项目，并按 feature 保存任务及关键沟通信息。

决定：共享数据、评估与产物接口，各方法独立实现和配置。目录采用稳定功能名，
不按 conversation 切割任务；对话只提取需求、约束、范围变更及验证结论。

原因：同一方法可能跨多次对话；功能档案可以连接输入、代码、实验和证据。

影响：根目录保留 README.md 和 TODO.md；tasks.md 保存详细进度，decisions.md 追加决定。
历史记录不得编造日期、提交号、输入文件或实验结果。

## 2026-10-09：区分论文复现与统一任务适配

决定：两种实验分别声明目标和配置。统一 QM9 对照不等同于复现原论文数值；
SPE 原论文的 pIC50/AWD-LSTM 实验与本项目 delta_e/Transformer 不直接比较误差大小。

原因：数据、单位、模型、预训练和训练预算不同都会影响结果。

影响：正式比较记录偏离、额外监督和预训练来源；性能较好或较差都不能单独证明实现正确。

## 2026-10-09：方法独立性与可比性

决定：统一任务、划分、输出单位和评估协议；学习率、架构及归一化等按方法约定保留。
依赖冲突时允许仓库内独立环境，通过统一文件接口交换预测。

原因：强制所有算法采用同一组超参数或同一依赖环境，可能改变方法或破坏已验证结果。

影响：先验证接入正确性，再执行约定预算；完成状态分为适配、验证、正式训练和报告。

## 2026-10-09：根目录文档迁移

按用户要求将组织原则写入根 README，建立功能档案和统一待办。原任务文件迁移如下，
内容中的历史验收与约束保留，不仅凭复选框完成就丢弃正文中的未完成工作。

| 原文件 | 当前位置（相对项目根目录） |
| --- | --- |
| TODO.md 的原三阶段记录 | docs/features/chemprop-pipeline/tasks.md |
| TRANSFORMER_TODO.md | docs/features/smiles-transformer/tasks.md |
| SPLIT_DISTRIBUTION_TODO.md | docs/features/data-splitting/tasks.md |
| MULTISEED_EXPERIMENT_TODO.md | docs/features/experiment-infrastructure/tasks.md |
| TODOLIST_TRAINING_THROUGHPUT.md | docs/features/experiment-infrastructure/throughput-audit.md |
| ML_IMPLEMENTATION_TODO.md | docs/features/traditional-baselines/tasks.md |
| BASELINE_SEED3407_TODO.md | docs/features/traditional-baselines/baseline-audit.md |
| SPE_TRANSFORMER_TODO.md | docs/features/spe-transformer/tasks.md |
| ML_TRAINING.md / SPE_TRAINING.md | docs/guides/ml-training.md / spe-training.md |
| UV_GUIDE.md / SSH_REVERSE_PROXY.md | docs/guides/environment.md / server-access.md |

根 TODO.md 重建为活动工作索引。未变动代码、数据划分、训练配置和已有产物；未启动训练。
迁移验证：检查根 README、TODO 和 docs 下 Markdown 的本地链接目标及 git diff --check。
