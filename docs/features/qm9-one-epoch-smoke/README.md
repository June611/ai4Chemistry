# QM9 全模型一轮烟测

日期：2026-10-12。范围：当前正式清单中的 19 个下游神经网络、3 个传统基线，另含 3D Infomax QM9 预训练前置步骤。

已完成：20 个下游成功、2 个 batch256 显存不足（Cormorant、LieTransformer），Infomax 预训练成功。所有成功模型均保存模型并生成 13380 条有限测试预测；23 行资源记录均完成采样覆盖。

最终文件：[资源 CSV](verification/resources_complete.csv)、[逐模型结果](verification/final_results.md)、[状态与首次失败历史](verification/final_status.json)、[审查 TXT](review.txt)。

在 RTX 3090 服务器串行运行，固定 split seed = training seed = 3407，神经网络 batch 256、epochs 1，保留原有架构、loss、scaler、LR 和 patience 20。读取已有完整 QM9 划分：107038/13380/13380；Infomax 预训练仍从训练集抽取 50000 条。

Chemprop CLI 要求 epochs > warmup_epochs，正式 warmup2 与一轮烟测冲突。因此仅在 single gap 和普通 multitask 的诊断配置中设 warmup0；Python consistency 入口保留原设置。正式配置不变，首次参数校验失败与修正重试均保存。

独立复制配置和输出，正式 256/100/20 配置不变。传统模型不适用 epoch/batch，按原配置拟合一次。Infomax 只使用本次 QM9 预训练 checkpoint。

成功要求：命令正常退出、完成一轮训练、保存模型、生成覆盖 13380 条固定测试样本的有限预测，并能重新计算 gap MAE。预训练检查一轮 history、有限验证损失和 checkpoint；失败保留日志，继续下一个模型，不自动减小 batch。

服务器运行：

```bash
cd /root/autodl-tmp/evan/ai4Chemistry
source artifacts/server_deployment/server-env.sh
python scripts/run_one_epoch_smoke.py --output artifacts/one_epoch_smoke_20261012
```

结果：指定目录内的 `status.json`、`results.md`、`resources.csv`、逐模型 `logs/`、复制的 `configs/` 和隔离的 `runs/`。这些是诊断产物，不能用于正式性能结论或论文结果。

资源 CSV 以 MiB 记录模型进程树的 GPU 显存峰值、整卡峰值、主机 RSS/PSS 峰值、采样数量与完整性。每秒采样，短瞬时峰值可能漏采；RSS 会重复统计共享页，PSS 按进程分摊共享页。资源范围包括训练器、数据加载子进程和入口的少量管理开销。失败模型的峰值只代表失败前尝试，不能视为成功训练所需上限。

本次服务器配置：RTX3090 24576MiB，PyTorch2.6.0+cu124。容器 CPU 配额为 15 核、内存上限 90GiB；`free` 和 `os.cpu_count()` 返回的宿主机资源不能作为当前容器配额。硬件证据保存在 `resource_hardware.json`。

任务：[tasks.md](tasks.md)；决策：[decisions.md](decisions.md)。

中断恢复与 Codex 自动审批配置：[operator-notes.md](operator-notes.md)。

首次启动应直接使用本入口的内置资源采样。只有本次早期缺失覆盖的情况才需要额外启动 `scripts/complete_smoke_resource_report.py`；它等待主烟测结束，串行补测缺失/部分覆盖模型，并生成 `resources_complete.csv`，保留补测目录与原结果的来源关系。
