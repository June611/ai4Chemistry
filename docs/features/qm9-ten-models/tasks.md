# 十模型任务与验收状态

更新时间：2026-10-10

## 已完成

- [x] 保存十模型 CSV 输入及固定上游提交清单。
- [x] 十个模型均有 adapter、factory 注册和独立 YAML。
- [x] 修复 ESA 非法层类型和错误批输出，采用含 PMA 的图回归结构。
- [x] 修复 TorchMD-Net 3.0.3 / PyTorch 2.6 custom-op 类型兼容与当前 forward API。
- [x] 对 LieConv 的 Python 3.11 语法进行可复现补丁，保持上游 checkout 干净。
- [x] 将 3D Infomax 修正为论文的 2D PNA 下游迁移路径；禁止无预训练权重冒充该方法。
- [x] 增加固定训练 split 上的 PNA/Net3D 对比预训练入口，并验证其 checkpoint 可被下游加载。
- [x] 恢复 Chemprop 固定划分：107,038 / 13,380 / 13,380。
- [x] 新增 split seed 42 和 2026；两套均为 107,038 / 13,380 / 13,380，且与 seed 3407
  使用相同样本全集、分层规则和预处理协议。
- [x] 为 split seed 42 和 2026 重排官方几何 artifact，并增加独立 `--split-seed` 训练参数，
  防止与 training seed 混淆。
- [x] 从 QM9 原始 extended-XYZ 重建 133,798 条几何，SHA-256 为
  `b9ca7d881ec66a554fe747b376bf0229f2b34ba7d39f16bb2487fe266da29b19`。
- [x] 数据加载强制核对 split manifest、顺序、无泄漏和 Chemprop 数据协议。
- [x] 十个 adapter 的 19 项前向/有限值/批形状/反向梯度测试通过。
- [x] 十个模型的正式 YAML 在三个完整 split 上均通过数据加载 dry-run；3D Infomax 使用各
  split 对应的 smoke checkpoint 验证两阶段路由。
- [x] 完成官方源码、三套 split 和公平性逐项审查，结果见
  [reproduction-audit.md](reproduction-audit.md)。
- [x] 修正 ESA 的 QM9 原子/键特征为官方 35/13 维编码，edge 输入为 83 维，并恢复官方
  QM9 的 M/S/P 与 PMA 配置。
- [x] 3D Infomax 下游强制校验预训练 checkpoint 的 split manifest 与 geometry 哈希，禁止
  跨 split 复用。
- [x] 增加服务器预检与顺序训练脚本，训练 CLI 支持 seed 和 3D Infomax checkpoint 覆盖。

## 服务器正式训练前仍需完成

- [ ] 将代码、`uv.lock`、固定 split 文件及几何 artifact 迁移到服务器，运行
  `bash scripts/qm9_server_preflight.sh`。
- [ ] 在服务器运行 `scripts/pretrain_3dinfomax_paired.sh`，为三套 split 分别完成 batch 256、
  100 epochs、patience 20 的预训练；本地只完成 32 条样本的端到端 smoke run。
- [ ] 先运行 split seed 3407 / training seed 3407，检查训练 loss、验证 MAE、checkpoint 恢复
  和测试预测文件。
- [ ] 基线通过后运行配对的 seed 42 和 2026；汇总明确标记为 paired split-seed runs。
- [x] 冻结统一预算协议：神经网络 batch 256、最多 100 epochs、patience 20，split seed 与
  training seed 相等。

## 验收边界

当前结论是“代码、数据和启动链路已通过本地 sanity check”，不是“十个模型已经完成正式训练”。
随机初始化的合成批测试不能证明论文指标可复现；论文级结论必须来自完整训练和多 seed 结果。
