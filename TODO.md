# 项目当前待办

更新时间：2026-10-11。仅列跨功能的未完成工作；详细子任务及证据以对应功能 tasks.md 为准。
这里不因代码可运行而标记正式实验完成；服务器状态未经本轮核验时，保留“待核验”。

- [ ] 三组 paired split/training seed 的 256/100/20 配置、入口与自动口径审查已完成；本地66次运行计划和95项测试通过；待服务器
  完成GPU依赖检查、batch256显存检查和正式产物验收。[统一协议 TODO](docs/features/paired-seed-100epoch-benchmark/tasks.md)
- [ ] 选择大文件迁移方案：直接SCP/SFTP、已有网盘或对象存储；当前只提供方案。
  [三方案比较](docs/features/paired-seed-100epoch-benchmark/standardization-plan.md#大文件迁移三选一仅方案)
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

文档组织及后续工作规则见 [README](README.md#文档与复现方法管理)。
已完成的细项保留在功能档案，不在根目录继续新增功能专属 TODO。
