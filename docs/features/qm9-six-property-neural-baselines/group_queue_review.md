# 三组队列审查

- 高组：DimeNet++、3D Infomax、TorchMD-Net，54 项下游；中组：ESA、PaiNN、EGNN、SchNet，72 项；低组：Chemprop、SMILES Transformer、SPE Transformer，54 项。
- 在本地通过真实 prepare 入口展开全部 180 个任务，核对三组无交集、无遗漏、无重复目录，所有状态为 pending，未创建训练运行目录。
- 每份展开 YAML 的 hash、目标物理单位、split/init 配对 seed、Z-score、batch256/100epochs/patience20 已核对。既有训练入口和模型代码未修改。
- 调度优先级改用已保存的同一张 3090 单 epoch 烟测，原 CSV SHA256 固定；保留 Chemprop 含启动/特征化的计时口径说明。组内沿用已有 LPT 分卡和独立任务失败隔离。
- 本地三份 Infomax 权重核验通过，准备阶段未新增预训练；其他机器缺权重时，高组可能增加最多三项前置预训练。
- 20 项新测试通过，覆盖静态分组、参数/路径引用、配置漂移、冻结计划恢复、报告完整性/去重/单位/多 seed 统计、跨机合并及填入原 LaTeX。另有 32 项既有流程回归检查通过，共覆盖 52 项不同检查。
- Bash 语法、Ruff、差异空白、源文件大小及敏感信息扫描通过。仅用待提交补丁在干净 HEAD 副本运行 20 项检查，同样全部通过。

## 边界

本次只生成队列，不在本地或服务器启动正式训练，不重新跑烟测。
默认每卡一任务、无 DDP；各机器独立运行，没有跨机中心锁。同一组只启动一次，不与全量 180 队列重叠执行。
分组脚本启动时用目标机器的本地路径重新 prepare；源 YAML 修改后用新批次，运行中的配置不热更新。
合并只接受三组完整结果，先在来源机器 verify 原始预测，再在本地校验报告文件 hash、同一 QM9 数据身份、180 个唯一任务和均值/样本标准差。
GPU batch256 峰值和新设备吞吐仍需目标服务器验收。烟测权重只用于排队，不代表实际完工时间。

完整配置审计见 group_queue_audit.json，逐模型计时依据见 group_smoke_reference.csv，操作见 grouped_queues.md。
