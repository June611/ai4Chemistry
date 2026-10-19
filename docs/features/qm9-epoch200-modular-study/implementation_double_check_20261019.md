# 实际代码Double Check：方案不等于已实现

> 实现前审查快照。相关缺口现已编码，最新状态见[实现交付](implementation_delivery.md)；以下历史结论不代表当前未实现。

本次只检查本地代码、逐任务配置、命令覆盖率和CPU合成测试；没有登录服务器、启动正式QM9训练或实现新模块。下表的“已实现”仅指注明范围，不代表全部114项候选可运行。

## 审查结果

| 检查项 | 当前实现 | 本次核查证据 / 尚缺内容 |
|---|---|---|
| 原Chemprop七性质单任务、D1/D2 | 已有入口，待GPU验收 | 21＋6份配置均可校验；模型仍使用原七任务头/亲和力/UW实现 |
| 新TaskAttention7 | 仅详细设计，未实现 | `src/molgap`没有`TaskAttentionHead`或该变体的训练分发；4头注意力、128维、残差门尚未接通 |
| 新PhysicsThermo10 | 仅详细设计，未实现 | 没有`QM9PhysicsLoss`、十头网络或十标签数据入口；旧三任务物理loss不等于此十任务模块 |
| 训练队列 | 清单完成，执行链未全部完成 | CSV有114项：第一阶段33项，后续81项；其中87项仍为planned且command为空 |
| 训练指令 | 部分完成 | fresh_commands.sh的27条命令与已接通配置完全对应；不包含第一阶段6项新模块，更不包含全部114项 |
| 失败后继续其他任务 | 27项参考脚本已验证 | 用替身uv强制首项退出13，后26项均调用，脚本最终退出1报告有失败；没有真实训练 |
| 200/256/20和三seed配对 | 已接通配置通过 | 27份配置校验；其余87项只通过计划文件检查，不代表实际训练支持 |
| 权重/优化器/LR调度/轮次保存 | 已接通200轮入口部分验收 | Lightning完整ckpt含state_dict、optimizer_states、lr_schedulers、loops/global_step及callback状态；200轮入口另存terminal_fit.ckpt |
| 配置/scaler/数据身份保存 | 已有七任务实现 | config.yaml、scalers.json、split_identity.json等是配套文件；复制ckpt时应保留这些资产 |
| 每50轮独立完整checkpoint | 未实现 | 当前相关回调没有every_n_epochs=50或等效周期保存；best/last/terminal不能替代50/100/150/200轮独立快照 |
| 任意epoch中断恢复 | 未完成统一入口 | 历史resume脚本只接受原100轮全状态；fresh200入口没有通用resume参数，不能直接恢复新200轮任务的50/150轮状态 |
| 随机与DataLoader状态恢复 | 不完整 | 新200轮callback保存rng_snapshot，但没有对应读取恢复代码；worker/采样器状态未完整验收，不承诺逐位连续 |
| 七标签标准化 | 已实现并测试 | 每目标只在训练集拟合Z-score，val/test复用；监督与UW均在标准化空间工作 |
| 十标签标准化 | 仅配置合同 | 当前TASK_NAMES与数据/验证器固定七任务；不能把七任务通过推定为十任务已实现 |
| 验证MAE选最佳模型及早停 | 已接通入口已实现 | D1/D2用val_delta_e_mae；单任务CLI显式tracking metric为gap_mae/property_mae；不以loss选模 |
| 原单位指标与七性质报告 | 已有七任务实现 | 标准化误差乘训练scale计算验证MAE，测试预测逐标签逆变换；同一best gap MAE checkpoint报告全部七项 |
| 新模块MAE/物理loss/梯度行为 | 尚未实现验收 | 计划YAML有合同；注意力梯度、十任务逆变换、物理残差/预热权重尚不能运行测试 |

“eval MAE”在本项目指**验证集MAE**。联合模型用Hartree单位的gap MAE；其他单任务用对应性质自己的单位。测试集MAE/RMSE只用于最终报告，不能用于选epoch。训练损失保持标准化MSE或其UW/物理增强形式。

## 代码定位

- 模型分发：`src/molgap/multitask7/models.py`，现有构造器只有B/C/D1/D2。
- 七标签拟合与复用：`src/molgap/multitask7/data.py`，`fit_scalers`、`load_splits`。
- 七任务验证误差与best/early stop：`src/molgap/training/multitask7.py`，`validation_step`及训练回调。
- 单任务CLI实际tracking metric：`src/molgap/training/commands.py`、`gap_metric.py`；已检查本地安装的Chemprop CLI确实将tracking metric交给两个回调。
- 终态与随机快照：`src/molgap/training/terminal_checkpoint.py`。只有保存钩子，没有恢复rng_snapshot的钩子。
- 历史续训限制：`scripts/resume_qm9_epoch200.py`要求原batch256/100轮与完整100轮进度；不等于通用200轮恢复入口。

## 本次实际做了什么

1. 对照详细方案检查模型类、数据任务数、训练分发和实际回调；没有仅凭YAML判断实现完成。
2. 对114项CSV逐条分类；核对27条脚本命令、配置路径和可导入训练模块，静态结果见[JSON](verification/double_check_static_20261019.json)。
3. 用替身执行器检验单项失败后继续的行为，证据见[故障隔离JSON](verification/double_check_queue_failure_20261019.json)。
4. 增强`tests/test_multitask7_pipeline.py`，在真实CPU合成fit产生的terminal_fit.ckpt中检查优化器、scheduler、训练进度、UW参数与RNG快照，并检查配套scaler和身份文件。
5. 复跑29项本地检查，结果见[本轮JUnit](verification/double_check_20261019.xml)。包括标签量级隔离、MAE原单位、best/early stop和官方CLI预测对应性；不能替代GPU或新模块验收。

`best_checkpoint.pt`是带模型/配置/scaler的推理导出，不含优化器和scheduler，不能替代Lightning完整ckpt续训。

## 阻止宣布“全部准备就绪”的缺口

1. 两个新模块及十标签数据路径尚未编码。
2. 每50轮完整快照、fresh200任意轮次恢复及随机/采样状态恢复尚未闭环。
3. 第一阶段6项新模块无可执行指令，后续81项未接通；新模块也未做GPU烟测。

应先补齐上述内容，再宣告完整第一阶段或全推广队列可一键训练。当前27项已有入口也仅通过本地检查，不把待GPU验收标为完成。
