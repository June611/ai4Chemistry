# 项目当前待办

更新时间：2026-10-12。仅列跨功能的未完成工作；详细子任务及证据以对应功能 tasks.md 为准。
这里不因代码可运行而标记正式实验完成；服务器状态未经本轮核验时，保留“待核验”。

- [ ] 资源调度与飞书同步实现验收已完成（本地141项、服务器32项测试，8项诊断及7份预测附件）；
  2026-10-12正式三组63项队列已启动，持续验收正式产物并实测校准并发收益。
  [运行记录](docs/features/qm9-resource-aware-training/formal_run.md) · [一键调度 TODO](docs/features/qm9-resource-aware-training/TODO.md)

- [ ] 三组 paired split/training seed 的 256/100/20 配置与审查已完成；服务器GPU环境、QM9数据/源码、95项测试和十模型batch2检查通过。
  单seed一轮烟测已完成，20个下游成功、2个OOM；Cormorant/LieTransformer按用户要求暂停，
  完成Infomax三组正式QM9预训练及正式产物验收。[统一协议 TODO](docs/features/paired-seed-100epoch-benchmark/tasks.md)
- [ ] Cormorant、LieTransformer 显存适配已暂停，默认入口跳过且直接训练受保护；
  当前重试进程已停止，不自动继续降batch。后续明确恢复后另行验收。
  [显存适配与停止记录](docs/features/qm9-model-batch-sizing/README.md)
- [ ] 冻结十模型的 controlled-adapted 超参数配置，在服务器逐模型完成三组配对训练；3D Infomax
  先按每个 split 分别运行 QM9 对比预训练。
  [十模型功能](docs/features/qm9-ten-models/tasks.md)
- [ ] 在服务器完成 SPE + Transformer 三组 paired split-seed 正式训练和汇总；本机已有的
  10-epoch 结果仅作历史诊断。[SPE 任务](docs/features/spe-transformer/tasks.md)
- [ ] 在服务器完成 SPE 三任务 affinity × consistency 消融；先运行 split seed 3407，再分别运行
  42 和 2026。[SPE 多任务任务](docs/features/spe-multitask-affinity/TODO.md)
- [ ] 完成 XGBoost、Random Forest、训练均值的三组配对训练；均值的历史 seed3407 单次结果
  不替代当前三组验收。[传统基线](docs/features/traditional-baselines/tasks.md)
- [ ] 导入并核验已有服务器模型结果，检查划分、标签单位和预测样本身份。[统一比较](docs/features/model-benchmark/tasks.md)
- [ ] 汇总可比模型结果，分别报告单次与多 seed 结果，并注明预训练和监督信息差异。[统一比较](docs/features/model-benchmark/tasks.md)
- [ ] 飞书CLI自动同步与服务器真实显存验收已通过；后续检查正式三组训练的最终结果完整性。
  [接入步骤与验收](docs/features/feishu-training-monitoring/README.md)

文档组织及后续工作规则见 [README](README.md#文档与复现方法管理)。
已完成的细项保留在功能档案，不在根目录继续新增功能专属 TODO。
