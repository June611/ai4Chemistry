# 审查缺口的实现交付

本轮已补齐代码、配置、数据准备与训练/恢复指令；未启动正式200轮训练队列。代码验收、前缀诊断与正式实验分开记录。

| 原审查项 | 当前实现 | 验收证据 |
|---|---|---|
| TaskAttention7 | 七任务投影、128维/4头自注意力、LayerNorm、零初始化可学习残差门、七独立头 | 门为0时退化、门/注意力梯度、12骨干CPU前后向及全部对应GPU任务；原单位指标 |
| PhysicsThermo10 | 十独立头；标准化监督MSE＋原单位gap/H−U/热增量约束 | 容差内为0、违约方向、逆变换/梯度、float64能量差；β前10轮0，11–20轮升至0.1 |
| 十标签数据 | 原七标签增加U0/U298/H298；按原source_id组求均值，不重划/丢弃样本 | 三seed数据已在WSL生成，原七标签/manifest/hash验证、重复组均值及源标签篡改拒绝测试 |
| 原基线 | Chemprop原BondMPNN/mean pooling/MLP；D1/D2直接复用原模型类和UW损失 | 结构不叠加新模块；新200轮从头训练，统一新的采样/保存执行器 |
| 推广接口 | 12骨干均有分子表示适配；图/序列/几何原输入分别复用 | 全部小模型CPU前后向与3D刚体不变性检查；3090全114项通过，Infomax三个seed真实权重已核验 |
| 配置与命令 | 当前114项全部有可校验配置和命令；第一阶段33项、后续81项分开 | CSV与脚本逐行一致；114项真实GPU通过；单项失败不终止后续任务 |
| 状态保存 | 每轮last、每50轮完整快照、独立terminal、最终best推理导出 | 权重、优化器、scheduler、epoch/step、早停历史、RNG、loader/词表、scaler、数据身份和历史best均保存 |
| 中断恢复 | 从任意已完成epoch边界恢复，不重新warmup/清空patience；复用保存的scaler/词表 | 五种Chemprop连续/恢复一致；50→52恢复；双worker；错误配置/完成任务拒绝；序列恢复不再fit词表 |
| 标准化与选模 | 每目标train-only Z-score；联合任务选Hartree验证gap MAE，单任务选本性质MAE | 同一best输出七性质，逆变换后计算误差；不使用测试集选epoch |
| 结果兼容 | 变体独立model_id、原单位MAE/MSE/RMSE/MAPE/R²、预测CSV、时间、allocator峰值、注意力/UW | 已接已有summary：过滤未训练目标，拒绝diagnostic混入正式比较，非能量标签正确标单位 |

## 代码入口

- `src/molgap/modular/heads.py`：独立头与TaskAttentionHead。
- `src/molgap/modular/physics.py`：QM9PhysicsLoss，两个等式与一个不等式。
- `src/molgap/modular/data.py`：十标签扩展与逐列训练scaler。
- `src/molgap/modular/backbones.py`、`inputs.py`：骨干/输入适配，恢复时加载词表。
- `src/molgap/modular/train.py`：统一执行器、MAE选模、完整保存和恢复。
- `scripts/resume_qm9_fresh200.py`、`preflight_qm9_epoch200.py`：恢复及预检工具。

新执行器保留骨干/头的指定结构，但统一使用Torch DataLoader的带状态generator采样；不声称其采样序列等于旧Chemprop CLI的NumPy SeededSampler。旧100轮仍为历史参照，不能将差异只归因于epoch。

## 本地验收

最终测试的逐项汇总与源码/config SHA256见[交付manifest](verification/implementation_manifest.json)。初轮三份JUnit合计72个独立用例（含1个重叠队列测试）；本轮拟提交干净副本重跑这72项并新增1项烟测调度测试，73项全部通过。选用证据：

- [核心与历史回归](verification/implementation_regression.xml)
- [恢复、数据和序列状态](verification/modular_pipeline.xml)
- [汇总兼容及单位](verification/modular_summary.xml)
- [队列故障隔离](verification/implementation_queue_failures.json)：首项强制失败，33/81项仍全部被调用，均最终返回非零；无真实训练。
- [真实QM9 CPU前缀记录](verification/real_data_diagnostics.json)：每分区前16条，scaler仍使用完整107038条训练记录；仅诊断。

旧`modules_implementation.xml`保存初次测试的ESA缩小fixture失败；原因是其上游BN固定32个池化token。fixture恢复原32约定后已在最终回归通过，没有修改其原模型算法。

本地沙箱禁止DataLoader多进程IPC套接字；双worker测试经沙箱外运行通过。该限制不标为模型失败。

## 全114项GPU验收更新

见[本轮完整报告](smoke114_report.md)：3090上114/114项均一次通过，每分区500条、batch256、1轮，2次更新；全部预测回算与完整状态审计通过。成功诊断权重已按用户授权清理，日志/指标/预测/状态hash已回收到WSL。没有模型失败，因此未缩小batch或改变架构。73项干净副本测试通过。

下面的2/12轮专项记录仍保留，用于覆盖一轮烟测没有启用的物理约束与恢复路径。

## GPU及正式实验边界

3090独立诊断记录见[GPU JSON](verification/gpu_probe_nmb2_20261019.json)和[完整日志](verification/gpu_probe_nmb2_20261019.log)，服务器8个新模块源码hash均与本地一致。注意力2轮、物理2轮后恢复至12轮；物理第11/12轮权重分别为0.01/0.02。两个模块训练allocator峰值约275MiB；小前缀不用于正式显存/吞吐估计。诊断只使用一个seed、每分区256条，不能用其误差证明改进有效，单轮时间也不是全量epoch时间。物理模块运行到预热启用后，并测试一次GPU恢复；最终best若仍在预热阶段，`physics_active_at_best=false`会如实记录。

FP32使用highest matmul precision、关闭TF32、CuBLAS workspace默认`:4096:8`。这些数值设置和PyTorch版本进入恢复上下文；不同GPU仍不承诺逐位一致。

待完成的是其他正式服务器的数据/可选依赖/预训练资产预检，全量最大输入batch、并发与磁盘校准，以及正式33项及后续推广训练。本轮3090已覆盖全部114项的基础GPU运行链路，不等于200轮精度或长期资源验收。现有3090数据盘在部署诊断前只剩约5.4GB，不能假定能容纳全部114项完整状态。

初始十标签/注意力参数是固定初始方案，不是调优后的最优值；物理方案同时增加辅助监督，当前不做细分控制，提升若出现应归于整体方案。

## 执行

完整数据、GPU诊断、分阶段正式训练及恢复命令见[运行指南](operations.md)。

```bash
# 不训练，先检查：
uv run python scripts/preflight_qm9_epoch200.py --stage 1
# 确认环境/显存/容量后，执行第一阶段33项：
CUDA_VISIBLE_DEVICES=0 bash docs/features/qm9-epoch200-modular-study/fresh_commands.sh
```
