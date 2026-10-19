# 训练范围、MAE选模与标签尺度审查

> 实现前审查快照。相关缺口现已编码，最新状态见[实现交付](implementation_delivery.md)；以下历史结论不代表当前未实现。

## 结论与范围

当前候选114项，替代原441项：第一阶段Chemprop 33项（21个单任务＋D1/D2六项＋两个新模块六项），后续81项（11骨干两个模块66项＋五个图骨干gap基线15项）。已接通入口27项，待实现87项；未启动服务器或真实QM9训练。

B/C、MTL7单独控制、PhysicsGap7、MTL10-control不进入队列。六类Transformer基础单任务默认不重训，对应两个模块均保留。后续图骨干只跑gap基线，不扩六个其他性质单任务。原结果不删除，不选择性替换三seed中的单条结果。

## 当前入口实际行为

| 入口 | 训练目标 | best/early-stop指标 | 原单位处理 |
|---|---|---|---|
| Chemprop gap CLI | 训练集标准化标签上的MSE | `val/gap_mae` | `GapMetricScale`从训练scaler取得gap尺度，恢复验证误差单位 |
| Chemprop六个其他单任务 | 同上，每个性质独立训练 | `val/property_mae` | 复用同一回调；只有一个目标，使用该目标自己的scale |
| Chemprop D1/D2 | 每标签标准化MSE，再做不确定性加权 | `val_delta_e_mae` | `MultiTask7LightningModule`以各标签scale还原误差，再按样本数汇总 |
| 已有SMILES/SPE单任务 | 标准化MSE | `val_mae` | forward输出已逆变换；验证目标保持原单位 |
| 新注意力/物理模块 | 计划为逐目标标准化MSE，物理方案另加正则 | 规定为`val_delta_e_mae`，早停相同 | 尚未接通训练；不能把配置合同视为运行验证 |

检查路径：`src/molgap/training/commands.py`、`gap_metric.py`、`multitask7.py`、`transformer.py`。当前这几条已接通入口无需为了“val MAE选模”更换损失；训练MSE和验证MAE各司其职。最终报告仍可使用测试RMSE，不能用测试RMSE挑选epoch。

## 七/十标签的尺度合同

1. 每个目标单独从训练集拟合`mean_j`与`scale_j=std(ddof=0)`，训练使用`z_j=(y_j-mean_j)/scale_j`。val/test复用原scaler，禁止重新拟合。
2. 基础监督为逐标签标准化MSE的平均；D1/D2在此基础上学习不确定性权重。标准化消除单位量级差异，但不能保证消除任务难度差异或负迁移。
3. `scalers.json`保留任务名、单位、mean/scale、拟合集、样本数和ddof。标准差为零、非有限标签应报错，不能默默传入训练。
4. MAE/RMSE按各性质原单位计算。七任务共享一个按gap MAE选择的checkpoint，其他六性质指标不分别挑不同epoch拼表。
5. 十标签扩展仍保持原七标签、sample_id、split和重复来源聚合规则；增加U0/U298/H298，最后仍汇报原七性质。十标签数据和模型入口尚未实现。
6. 物理残差需先逐标签逆变换到原单位，再在float64求能量差；各关系分别用仅训练集或预设正尺度归一化，再与监督loss组合。禁止直接混合原始总能量MSE或用近零恒等式残差标准差作分母。关系权重、容差、精确归一化规则待实现前冻结。
7. 离散原子/键特征和token ID不做数值Z-score；可学习embedding与连续描述符不是同一种特征。连续特征若需scaler只拟合训练集；3D坐标/距离保持骨干物理约定。

已有七任务数据实现为`src/molgap/multitask7/data.py`；指标逆变换为`metrics.py`；不确定性损失为`losses.py`。新增24份模块设计YAML已写入上述核心合同，但这些配置均标记`planned_not_executable`。

## 本地证据

- [逐配置审查JSON](verification/scope_policy_review_20261019.json)：检查27份配置的epoch/batch/patience、实际Chemprop命令的tracking metric及配置SHA256。
- [训练与尺度测试JUnit](verification/scope_and_label_policy_20261019.xml)：21项通过，包括历史100轮/新200轮合同、CPU完整checkpoint/预测/评测链路、任务单位跨度达1e16的标准化与梯度检查、多批次样本加权的原单位MAE、拒绝loss/跨标签平均MAE选模。
- [MAE入口回归JUnit](verification/mae_selection_20261019.xml)：检验gap指标不受辅助任务权重影响、best/early-stop遵循gap，以及官方CLI最佳分数与原单位验证预测的MAE一致；8项通过。
- 修改文件的Ruff检查与参考脚本`bash -n`检查通过。

CPU合成测试不是正式200轮QM9训练，不产生新模型精度结论。本轮不修改历史训练配置或LaTeX结果，不上传服务器、不提交代码。

## 未完成边界

注意力模块、物理loss、十标签数据扩展、各骨干适配、每50轮全状态checkpoint验收、GPU烟测与并发校准仍在TODO。后续模块必须满足本合同，不能仅凭当前七任务测试推定十任务已正确实现。

旧100轮与新200轮、旧MAD/MAE与新Z-score/MSE对比须保留协议标签；默认不补Transformer基础重训时，只能评价整体方案变化，不单独归因于某个模块。
