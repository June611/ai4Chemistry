# 决策记录

## 2026-10-12：烟测口径

按用户要求只用一个配对 seed 3407、一个 epoch，保留正式 batch 256。使用完整固定 QM9 划分，检查正式规模的数据读取与预测闭环；预训练仍使用原配置的训练集 50000 条子集。

现有 `--smoke` 会将 batch 降为 8 并截取数据，不能验证 batch 256 显存，因此使用复制配置并正常调用真实入口。模型和训练器未改动。

当前部分入口严格要求 `paired_split_gap_mae_v2` 标识；复制配置保留该字段，增加 `smoke_suite.purpose=diagnostic_only`，并将全部输出隔离到本功能 artifacts 目录。正式汇总不得收录本次产物。

初始计划保持各模型 warmup 和 LR 不变，一轮烟测仅用于检查执行链路，不用于比较模型精度。传统基线没有 epoch，按原有树数量拟合一次。Infomax 必须先完成本次 QM9 预训练，失败时下游标为前置步骤阻塞。

遇到显存不足记录 batch 256 下的失败，不暗中缩小 batch；失败模型不会阻止后续模型检查。

## 2026-10-12：资源采样

用户追加要求为每个模型保存显存与主机内存。使用 nvidia-smi 和 Linux /proc，按模型进程树每秒采样并保存 CSV，同时记录 RSS 与 PSS，避免将共享内存的重复计数误认为独占需求。

监测加入时 EGNN、SchNet、PaiNN 已完成，DimeNet++ 正在运行。先记录其缺失/部分覆盖状态；主烟测结束后，在独立目录补跑这些模型的一轮资源测量，再合并证据。其余模型从启动前开始采样。

## 2026-10-12：一轮 Chemprop CLI warmup 冲突

实际执行证实官方 CLI validate_train_args 要求 epochs > warmup_epochs，single gap 和普通 multitask 在优化前退出。仅在复制的诊断配置中设 warmup0 后重试，保留 LR 数值、batch256 和正式配置 warmup2；Python consistency 入口没有此 CLI 参数限制，保留原 warmup。

原始 status.json 与失败日志不改写。最终 final_status.json 保存修正重试并嵌入 initial_attempt，resources_complete.csv 保存初次状态、错误、最终状态和对应测量目录。
