# 一轮烟测任务

- [x] S01 确认服务器 GPU 空闲，清点 19 个神经网络及 3 个传统基线。
- [x] S02 编写串行入口，隔离一轮配置与产物，保留正式训练配置。
- [x] S03 运行 EGNN：完整一轮及 13380 条测试预测通过，资源待补测。
- [x] S04 运行 SchNet：完整一轮及 13380 条测试预测通过，资源待补测。
- [x] S05 运行 PaiNN：完整一轮及 13380 条测试预测通过，资源待补测。
- [x] S06 运行 DimeNet++：完整一轮及 13380 条测试预测通过，资源待完整补测。
- [x] S07 运行 ESA：完整一轮及 13380 条测试预测通过，资源采样完整。
- [x] S08 运行 TorchMD-Net：完整一轮及 13380 条测试预测通过。
- [x] S09 运行 Cormorant：batch256 在 CUDA 前向中显存不足，保存失败证据。
- [x] S10 运行 LieTransformer：batch256 在 CUDA 前向中显存不足，保存失败证据。
- [x] S11 运行 ALIGNN：完整一轮及 13380 条测试预测通过。
- [x] S12 运行 3D Infomax QM9 预训练与下游训练：两阶段均通过，未使用外部权重。
- [x] S13 Chemprop consistency 已通过；single gap、普通 multitask 的诊断 warmup0 重试均通过，保留首次失败证据。
- [x] S14 运行普通 SMILES Transformer 与 SPE Transformer：均完成一轮及完整测试预测。
- [x] S15 运行 SPE multitask 的四种 affinity/consistency 组合：均完成一轮及完整测试预测。
- [x] S16 Training Mean、XGBoost、Random Forest 均拟合并生成完整测试预测。
- [x] S17 逐模型核验 checkpoint、完整测试预测和有限指标：20 个下游成功，2 个 batch256 显存不足；另含 Infomax 预训练成功。
- [x] S18 结果和日志已回传；86 文件证据包 SHA256 一致，轻量报告保存在 verification/。
- [x] S19 23 行资源记录均具备完整尝试采样；4 个早期模型已补测，CSV 标明失败、初次状态和测量目录。

每项即使失败也记录为已检查；成功与失败以结果表为准，不以勾选替代。
