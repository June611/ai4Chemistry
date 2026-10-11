# 训练口径统一与服务器交接计划

日期：2026-10-11。状态：按最新需求执行；统一gap MAE选模，保留原loss/scaler，本地66次运行计划已通过；最终审查记录见final-gap-mae-review.txt，服务器验收独立进行。
本计划属于既有 paired-seed feature，逐项执行记录见 [tasks.md](tasks.md) 的 U 系列任务。
历史结果与旧审查不改写为新协议通过。

## 目标与边界

以最少配置与项目适配层改动统一可控训练规则，为在线算力服务器交接准备可审查版本。
范围：19 个神经网络、3 个传统基线，各三组配对；Infomax 另有三次预训练。
不改变网络关键机制、重新切分数据或重新实现上游模型，不改 Chemprop 的安装源码。
本轮修改代码与配置并做必要短跑/预检；不启动正式实验、部署或数据上传。

## 当前统计

[hyperparameter-inventory.csv](hyperparameter-inventory.csv) 记录 23 份基础配置
（19 个下游神经网络、1 份预训练、3 个传统基线）的当前值；最终实施仍需检查三组展开配置。

| 项目 | 修改前状态 | 已冻结规则 |
|---|---|---|
| 数据/标签 | 三套共同样本、Hartree、训练集拟合统计量 | 保持；重新审计哈希与样本互斥 |
| seed | 3407/42/2026，split 与 training 配对 | 保持，不做交叉组合 |
| 神经网络预算 | batch 256 / 最多 100 epochs / patience 20 | 保持；固定单 GPU，明确全局 batch |
| checkpoint 与早停 | Chemprop val_loss；其他下游 gap MAE | 统一到逆变换后的 validation delta_e MAE |
| early-stop min_delta | Chemprop 默认 0，其他下游 1e-5 | 建议统一为 1e-5 Hartree，mode=min |
| 标签 scaler | Chemprop/序列 Z-score；QM9 mean/MAD | 保持既有train-only Z-score / mean-MAD |
| 基础监督 loss | Chemprop/序列 MSE；QM9 MAE | 保持既有MSE / MAE及辅助目标 |
| 多任务与物理约束 | HOMO/LUMO/gap、亲和力、一致性 | 保留现有监督、结构、权重与 Hartree 物理关系 |
| 输入特征 | graph/SMILES/3D/RDKit，处理方式不同 | 保留模型机制；统一防泄漏要求，不强制统一特征 scaler |
| 测试 | 原始 Hartree 的 gap 指标 | 保持统一预测格式、逐样本复算与三组汇总 |

最新决定：统一选模与评估，保留各模型现有loss/scaler。此前强制统一MSE/Z-score的建议撤回。
不同loss/scaler是明确披露的方法配方，不作为“选模口径未统一”的阻塞理由。

Z-score 使用训练集均值与标准差（总体标准差 ddof=0）；验证/测试不重新拟合。
多任务按各目标分别标准化，基础 loss 按已声明任务权重归一，检查有效样本分母与 batch reduction。
物理残差仍为 `predicted_delta_e - (predicted_lumo - predicted_homo)`，在 Hartree 空间计算，
其系数及量纲需记录并核对，不能因 scaler 变更意外改变物理项。
树模型保持无需 scaler 的既有机制，batch/epochs/patience 为 N/A。
Infomax 预训练保留对比 loss 和其预训练选模指标，不改成 gap MSE；预算仍为 256/100/20。

## 最小实施路径

1. 冻结协议字段与差异清单，逐模型明确主目标、标准化、监督 loss、监控指标、单位和预算。
2. 保留现有QM9 mean-MAD/MAE与其他模型Z-score/MSE，只改选模、单卡和协议字段；
   不修改模型表示、邻居构图、官方层或读出。
3. Chemprop 单任务可用官方 tracking metric；多任务必须只跟踪 delta_e 的原始单位 MAE。
   普通 `--tracking-metric mae` 不足以证明只监控 gap，不能拿三个目标的平均 MAE 替代。
   注册项目内GapMAE与scaler回调，通过薄CLI入口接入；保留官方CLI/API既有网络和训练流程。
4. Phase 3 新增或接通独立 gap validation MAE，checkpoint 与 early stop 共同监控它；
   val_loss 继续记录用于诊断，物理损失仍参与训练。
5. 核对序列与 QM9 对 `min_delta`、最佳 checkpoint 的语义：
   保存严格最优 gap MAE；只有超过统一 min_delta 的改进才重置早停计数。
   用相同合成验证序列检查 checkpoint epoch 和 stop epoch，避免只有参数名相同。
6. 更新所有基础/生成配置、运行计划、审计器、README 和配置快照，隔离旧产物。
   选模规则变化，使用paired_split_gap_mae_v2与outputs/paired_gap_mae_v2；旧v1结果不迁入新表。

## 超参数公平性评估

公平性不能仅由“LR 数字相同”判断。固定数据、预算、选模和测试规程，保留并冻结架构适合的
优化器、LR 与 scheduler 是本次最少改动的建议；这属于固定配方比较，不宣称所有模型充分调优。

| 模型/类别 | 当前 LR | 审查重点 |
|---|---|---|
| Chemprop 三模型 | init/final 4e-4，peak 4e-3 | 检查历史 batch 扩大四倍后的 LR 放大依据与 warmup，不直接判为错误 |
| 原子/SPE 单任务与消融 | 3e-4 | 同 encoder 的消融保持相同 LR、optimizer、scheduler、weight decay |
| EGNN、PaiNN | 5e-4 | 记录 AdamW、模型容量、梯度裁剪与100-epoch调度 |
| DimeNet++、Cormorant、ALIGNN | 1e-3 | 记录 optimizer 差异，不用 LR 大小直接推断公平性 |
| SchNet、ESA、TorchMD-Net、Infomax 下游 | 1e-4 | 既有scaler/loss保留，服务器复查学习稳定性 |
| LieTransformer | 3e-4 | 记录模型特有 optimizer 与调度 |
| Infomax 预训练 | 8e-5 | 预训练 train 子集最多50000；额外预算单列；核对 warmup与调度 |
| XGBoost | 0.032845274 | boosting shrinkage，不与神经网络 LR 数值比较 |
| Random Forest/均值 | N/A | 树数不等于 epoch |

batch 256 统一的是一次参数更新处理的分子数，不是各架构耗时、显存或 FLOPs。
在107038个训练样本、无drop_last的单卡设定下，每epoch最多419次更新；早停后实际次数仍需记录。
Infomax预训练最多50000个train样本，其每epoch更新预算不同，不能与下游或无预训练方法混为一谈。
正式单卡下明确实际 batch、最后不足256的 batch、drop_last 和更新次数；禁止默默改成多卡全局
batch 512/1024。OOM 记为阻塞项，不在个别模型上静默缩小 batch 或自动梯度累积。

Chemprop 的四倍 LR 是当前设置，需要验证支持。经典 linear scaling + warmup 的证据来自
ImageNet/ResNet 的 SGD 实验，不能据此直接推出本项目 Adam/AdamW 与分子模型的最优 LR。
参考 [Goyal et al.](https://arxiv.org/abs/1706.02677)。

预训练旧设置lr_patience25大于早停20；本轮最小调整为lr_patience10，并设置absolute threshold，
使plateau判断与earlystop使用相同绝对min_delta。基础LR、warmup、loss和模型均保留。

本轮不开展 LR 搜索或增加训练次数。如后续要比较调优后的性能，需要所有模型采用预先固定的
相同候选数量与验证预算，只读 train/val，冻结后再测试；不能仅给创新模型额外挑 LR。
服务器少量稳定性预检若用于挑参数，也属于调参，应计入并公开预算，而非称为无成本检查。

## 最终审查与交接标准

- 静态审查：正式纳入名单、三组配对、全局batch256、100epoch上限、patience20、
  原始单位gap MAE监控、min_delta、scaler/loss及允许差异明确；无归档配置混入。
- 行为审查：改变held-out标签不改变训练scaler/词表；多任务HOMO/LUMO变化不影响gap选模；
  有效样本加权的整集MAE、逆变换、checkpoint恢复和earlystop行为正确。
- 配置预检：三套split的19个神经网络与3个传统模型逐个dry-run；缺Infomax匹配checkpoint记PENDING。
- 服务器检查：锁定依赖、固定上游commit、指定GPU/显存；batch256可行性与实际更新日志。
  尚无服务器地址、GPU/显存和调参预算，暂记部署阶段输入，不阻止本轮规划。
- 交接前代码审查：代码/文档/配置一致，必需测试、Ruff、shell语法、数据哈希通过。
  保存最终TXT与差异，检查数据未被新提交携带，再按既有授权commit/push；不自动启动云训练。
- 报告分为“配置与代码通过”和“服务器运行/正式实验通过”；未运行项不得标PASS。
- 完整训练后才复算预测指标并汇总三组均值、样本SD；同时披露3D输入、额外HOMO/LUMO监督和预训练。

冻结后的首表是统一可控规则的适配比较。仅SMILES与使用3D几何的方法，应注明输入信息差异；
gap-only与多任务方法，应注明额外监督，不能把这些实验解释为仅网络结构不同。

## 大文件迁移三选一（仅方案）

已有数据包约162.13 MiB，源数据约499.53 MiB，SHA-256清单保留在代码侧。
代码经Git交接，数据、权重与日志独立搬运；34个已跟踪旧数据已取消索引跟踪，保留全部本地文件和.gitkeep。
此次不新建云资源、不上传文件、不实现任何方案。

| 方案 | 适用情况与理由 | 成本/局限 |
|---|---|---|
| A：本地直接SCP/SFTP到算力服务器（推荐） | 本次单服务器迁移链路短，不需增加网盘账户；SCP通过SSH认证传输 | 依赖本地上行和SSH连通；SCP不提供自动续传，网络不稳可选择支持续传的SFTP |
| B：已有网盘保存数据包，服务器下载 | 可作独立备份、便于人工分享和后续换服务器 | Linux端是否易下载、链接有效期、限速/会员取决于网盘；先确认服务器下载能力再选，不预设某家可用 |
| C：对象存储作为长期数据源 | 多服务器/多次实验时上传一次后重复获取，便于自动化交接与数据版本管理 | 需账号/访问控制，可能产生存储、请求及流量费用；本次小规模单次迁移配置成本较高 |

SCP与SFTP能力参考 [OpenSSH SCP](https://man.openbsd.org/scp)、
[OpenSSH SFTP](https://man.openbsd.org/sftp)；对象存储/网盘跨后端传输工具能力可参考
[rclone文档](https://rclone.org/docs/)。这里不承诺任何供应商价格、速度或账号可用性。
三种方案均使用同一归档与SHA-256，接收端解压后核对逐文件清单与split/geometry哈希。
首选A完成本次迁移；已有网盘适合B作备份；需要长期多服务器复用时再考虑C。
