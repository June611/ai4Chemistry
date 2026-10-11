# 三组配对 seed、100-epoch 协议边界

## 当前执行状态（2026-10-12）

用户暂停Cormorant、LieTransformer，取消继续降低batch；配置保持256，不启动两个模型。
中央暂停名单控制批量和直接训练入口，dry-run保留。当前启用20个评估模型，共60组配对实验；
下文66组是含两个暂停模型的注册范围，不代表当前全部可训练。
[完整暂停记录](../qm9-model-batch-sizing/README.md)

## 已明确的口径

- 三组运行固定为 3407、42、2026。
- 每组的 split seed 和模型初始化/training seed 相同。
- 不运行 seed 7、123，也不做 split 与 training seed 的笛卡尔积。
- 神经网络统一 batch 256、最多 100 epochs、patience 20。
- 允许验证指标连续 20 epochs 无改进后提前停止，并恢复最佳验证 checkpoint。

## 参数适用性

XGBoost、Random Forest 和训练均值没有 mini-batch 与 epoch 概念，当前也没有 early stopping。
这些字段应记为 N/A。公平性来自相同的三套数据、配对随机种子、预处理、标签单位和测试口径，
而不是人为把 `n_estimators` 等同于 epoch。

## batch 256 的风险处理

DimeNet++、Cormorant 等 3D 模型当前 batch 明显小于 256。实现阶段按需求写入 256；实验阶段
若 OOM，记录为该统一协议在当前硬件上不可行并回到协议审核。不能自动梯度累积或降低 batch，
因为两者都会改变已批准口径。

## 结果解释

三次运行同时改变数据划分和模型随机性，因此测量的是“配对 split-seed 变化”，不是固定数据
划分下纯初始化方差。结果汇总必须保留每组值，并明确说明标准差同时包含划分与训练随机性。

## 已批准决定

1. 3D Infomax 的 2D/3D 预训练阶段也强制采用 batch 256、100 epochs、patience 20。
2. 排除 `configs/experiments/exp003_consistency_0.yaml` 和
   `exp004_consistency_01.yaml`；旧文件仅保留作历史记录。

## 2026-10-11：提交前复审

用户要求“口径达成一致后”才提交并 push。此次审查仅确认预算、数据身份与 seed 配对一致，
选模指标、训练 loss、标签标准化有方法族差异，且 P27/P28/A05 未全部完成。因此本次保存
本地 TXT 和可迁移数据包，不执行 commit/push，也不自动更改科研配方。

数据独立于 Git 搬运。恢复 `.gitignore`，记录压缩包与每个源文件的 SHA-256；已跟踪文件
需要在后续提交前显式从索引移除。历史数据不做未经授权的重写或 force push。

## 2026-10-11：统一训练口径的新实施计划

用户要求先统计，再以最少改动标准化训练规则，评估batch/LR公平性，最终审查后交接服务器。
本轮建立U系列TODO；数据迁移仅比较SCP/SFTP、网盘、对象存储三方案，不实施上传。

推荐所有下游神经网络采用train-only Z-score、基础监督MSE和Hartree validation gap MAE，
保留架构、辅助监督、物理约束与各架构优化配方；预训练保留对比loss，传统模型记N/A。
loss/scaler范围通过异步问题征求选择，未返回前为建议，非已实施/已验收决定。
公平性主张限于固定配方、共同数据/监督规则与预算，不声称同LR或各模型最优调参。
具体方案与验收见 [standardization-plan.md](standardization-plan.md)。

## 2026-10-11：最新确认，执行gap MAE选模

根据最新沟通，统一原始Hartree的validation delta_e MAE作为下游checkpoint与早停指标，
保持三组配对seed与256/100/20；撤回强制统一MSE/Z-score的建议，保留各模型loss/scaler。
min_delta统一1e-5 Hartree，单GPU；预训练保留对比loss和256/100/20。
新协议paired_split_gap_mae_v2写outputs/paired_gap_mae_v2，隔离旧选模产物。
Chemprop使用项目内注册指标与scaler回调，不改site-packages；普通多任务平均MAE不代替gap。
Infomax lr_patience25大于earlystop20，最小调整为10，threshold_mode=abs；基础LR不变。
本轮完成必要短跑与预检，审查后保存TXT并提交代码；正式训练与服务器部署另行验收。

## 2026-10-11：代码交接验收与设备边界

66次完整数据运行计划通过，包含三组Infomax smoke checkpoint路由；这些权重不用于正式结果。
Chemprop归一化验证路径下，Phase3克隆物理criterion的诊断残差也在Hartree计算；
仅调整验证回调的单位转换状态，训练loss、系数、模型与最终raw测试行为保持。
GPU验收独立：服务器预检必须运行Torch和DGL设备探测；本地CUDA不可用记PENDING。
基础DGL创图成功仍不等于所有架构batch256的前向/反向与显存验收。
本轮审查满足代码提交条件；之前“先不提交”的历史记录由此决定替代。
数据取消Git索引跟踪、保留本地，不重写历史；数据上传方式仍由用户三选一。
