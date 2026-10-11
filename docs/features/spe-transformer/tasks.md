# SPE + Transformer 对比实验

当前状态（2026-10-11 复审）：适配及历史 10-epoch 诊断已完成；三组配对正式实验未完成。
以下历史验证数字对应当时版本，本轮文档整理未重新运行训练或模型测试。

- [x] 核查当前划分、Transformer 网络、训练预算与多 seed 调度。
- [x] 阅读本地 SPE 论文：论文预测网络为 AWD-LSTM。
- [x] 核查作者仓库与示例，记录版本和源码许可。
- [x] 接入官方 SPE 学习/分词实现，合并规则和词表仅拟合训练集。
- [x] 复用项目 Transformer、目标标准化、早停、预测与评估流程。
- [x] 添加独立配置、多 seed 入口、可恢复分词器和使用说明。
- [x] 验证分词正确性、数据隔离、模型保存恢复与训练链路。
- [x] 按用户最新要求停止本机训练，将服务器训练与汇总命令追加至 README。
- [ ] 三组 paired split-seed 正式训练与结果汇总：转移至在线算力服务器执行。

当前实验约定：QM9 full 三套 split seeds 3407、42、2026，training seed 与 split seed 一一匹配；
batch 256；最多 100 epochs；patience 20；
AdamW lr 3e-4；全精度；以验证集 Hartree MAE 选模。
保留当前工作区已有修改，不重新生成数据划分，不覆盖旧模型产物。

全量预检：107038/13380/13380 行，词表 803 项，最长序列 12 tokens（不含 CLS），
全部划分未知 token 数为零；可训练参数 2357761。
预检及源码来源记录：`reports/spe_transformer/`。

验证：全套原有及首批 SPE 测试 53 passed；补充 SPE 优化器更新与模型保存恢复测试后，
SPE 专项 5 passed；Ruff 和 git diff --check 均通过。
正式训练于 2026-10-08 在本机 GTX 1050 Ti 上启动，随后按用户要求停止。
保留正式五 seed 批次的 seed3407 中断产物；该正式批次尚无最终测试指标。
独立 10-epoch 实验已完成，其结果见下文。
批次日志：`outputs/batches/qm9_spe_five_seed/logs/spe_transformer/`。

## 10-epoch 实现检查（用户追加）

- [x] 建立独立 `configs/spe_transformer_10epoch.yaml`，保留原正式配置。
- [x] 本机 GPU 跑完 seed3407 的 10 epochs，并使用最佳验证 checkpoint 评估测试集。
- [x] 与同一测试集的训练集均值基线比较，检查学习曲线及模型保存恢复。
- [x] 核对论文可比性：论文为 24 个靶标的 pIC50 回归及分子生成，不是 QM9 delta_e。
- [x] 记录结果及结论边界：优于或劣于论文指标均不能单独证明实现正确。

10-epoch 已完成：测试 MAE 0.00785194 Hartree，RMSE 0.01109081 Hartree，R² 0.945409。
报告：`reports/spe_transformer/10epoch/README.md`；原五 seed 正式训练仍保持停止状态。
