# 缩减后的重训与模块清单

## 全部训练任务总览：12个骨干，114次训练

下表数字均为**独立训练次数**，不是不同模型架构的数量。每个“变体＋目标组合”使用42、2026、3407三个seed各训练一次，模型初始化seed与数据划分seed配对。第一阶段为Chemprop的33次，第二阶段为其余11个骨干的81次。

| 基础模型 | 原始单任务 | D1：七任务＋不确定性加权 | D2：七任务＋亲和力＋不确定性加权 | TaskAttention7：七任务自注意力 | PhysicsThermo10：十任务＋物理约束 | 合计 |
|---|---:|---:|---:|---:|---:|---:|
| Chemprop（第一阶段） | 21（七性质分别训练） | 3 | 3 | 3 | 3 | **33** |
| DimeNet++ | 3（仅gap） | — | — | 3 | 3 | 9 |
| DeMol | — | — | — | 3 | 3 | 6 |
| PaiNN | 3（仅gap） | — | — | 3 | 3 | 9 |
| EGNN | 3（仅gap） | — | — | 3 | 3 | 9 |
| TorchMD-Net | — | — | — | 3 | 3 | 6 |
| Transformer-M | — | — | — | 3 | 3 | 6 |
| 3D Infomax下游 | 3（仅gap） | — | — | 3 | 3 | 9 |
| SchNet | 3（仅gap） | — | — | 3 | 3 | 9 |
| ESA | — | — | — | 3 | 3 | 6 |
| SPE Transformer | — | — | — | 3 | 3 | 6 |
| SMILES Transformer | — | — | — | 3 | 3 | 6 |
| **第二阶段小计（不含Chemprop）** | **15** | **0** | **0** | **33** | **33** | **81** |
| **全部合计** | **36** | **3** | **3** | **36** | **36** | **114** |

各列对应的改进点与预测目标：

- **原始单任务**：保留原骨干和原预测结构，一个模型一次只学习一个性质。Chemprop对HOMO、LUMO、gap、alpha、mu、R2、ZPVE分别训练，故为7×3＝21次；其余五个入列骨干只重训gap，各3次。
- **D1**：仅Chemprop；共享编码器、七个独立任务头，使用不确定性加权MSE动态分配任务权重，不加亲和力矩阵。
- **D2**：仅Chemprop；在七任务结构中保留额外任务投影、全局可学习7×7亲和力融合，再使用不确定性加权MSE。D1/D2复用已有实现，不叠加下述新模块。
- **TaskAttention7**：全部12个骨干；保留原分子编码与聚合，新增七组任务投影、样本相关的4头自注意力、可学习残差门和独立输出头；联合预测原七性质，监督为等权标准化MSE。
- **PhysicsThermo10**：全部12个骨干；保留原分子编码与聚合，使用十个独立任务头；原七性质增加U0、U298、H298辅助标签，在标准化监督MSE之外加入gap＝LUMO−HOMO、H298−U298≈kBT、U298≥U0的物理软约束。该方案与TaskAttention7平行，不叠加任务注意力。

**统一训练口径**：所有新任务从头训练，上限200轮、batch_size＝256、patience＝20；标签逐目标仅用训练集拟合Z-score，计算指标前逆变换。联合任务按原单位Hartree验证gap MAE选模与早停，单任务按对应性质原单位验证MAE；同一个联合模型的全部性质使用同一个最佳checkpoint。每轮保存last，每50轮另存完整状态，支持完整epoch边界恢复。

“—”表示本轮不安排该组合，不表示模型不能使用该方法。3D Infomax只训练下游，复用对应seed的QM9训练分区预训练权重。新模块沿用骨干原输入要求，不会让2D模型额外依赖3D坐标。

总览与[逐项训练队列](training_queue.csv)一致；运行指令见[第一阶段33项](fresh_commands.sh)和[第二阶段81项](stage2_commands.sh)。这些是任务与实现清单，不表示正式实验已经完成；验收范围见[实现交付报告](implementation_delivery.md)。

## 配置与详细方案

全部新任务200轮上限、256 batch、patience20、三配对seed。当前CSV引用的`fresh/`全部由统一训练入口校验；`planned/`仅留作历史设计，不执行。以下路径相对于`configs/qm9_epoch200/`，逐seed展开见training_queue.csv。

## 第一阶段：Chemprop，33项

| 模型/变体 | 目标 | 作业数 | 配置 | 改动 |
|---|---|---:|---|---|
| Chemprop原始单任务 | HOMO/LUMO/gap/alpha/mu/R2/ZPVE分别训练 | 21 | `fresh/chemprop/<目标>_seed<seed>.yaml` | 原架构；fresh200；原单位对应性质MAE选模 |
| D1 | 七性质联合 | 3 | `fresh/chemprop_variants/D1_seed<seed>.yaml` | 原不确定性加权，不加亲和力 |
| D2 | 七性质联合 | 3 | `fresh/chemprop_variants/D2_seed<seed>.yaml` | 原额外投影、亲和力融合、不确定性加权 |
| TaskAttention7 | 七性质联合 | 3 | `fresh/chemprop/task_attention7_seed<seed>.yaml` | 样本条件化的任务自注意力；等权标准化MSE |
| PhysicsThermo10 | 七性质＋U0/U298/H298 | 3 | `fresh/chemprop/physics_thermo10_seed<seed>.yaml` | 独立任务头＋物理约束；与注意力方案平行 |

B/C不重训，原结果和旧配置保留。D1/D2不是已确认超越单任务的最优模型：当前gap RMSE分别0.007200/0.007060 Ha，单任务为0.007018 Ha。

## 两个新模块具体怎么实现

以下结构和参数已经编码到`src/molgap/modular/`并写入当前可执行配置。它们是预先固定的初始方案，不宣称最优；CPU数值、梯度和恢复测试已通过，3090上全部114项一轮GPU烟测已通过，正式精度仍待验证。训练数量保持不变。

| 对比项 | MTL：TaskAttention7 | 物理约束：PhysicsThermo10（本项目暂称PINN） |
|---|---|---|
| 输入 | 原骨干分子表示，不额外引入真实标签或新3D数据 | 同左，标签仅用于训练监督 |
| 骨干 | 原编码器和原分子聚合保留 | 同左 |
| 输出 | 七性质，每个性质一个输出头 | 十性质，每个性质一个输出头；新增三种热化学辅助标签 |
| 新增机制 | 对七个任务特征做样本相关的自注意力 | 对预测值施加两条等式、一条不等式的软约束 |
| 监督损失 | 七个标准化MSE等权平均 | 十个标准化MSE等权平均 |
| 是否使用亲和力矩阵/UW | 不使用原全局亲和力参数，不加不确定性加权 | 不加注意力、亲和力或不确定性加权 |
| 推理结果 | 报告七性质，gap由自己的头直接预测 | 仍报告原七性质；辅助头无需真实热化学标签 |
| 选模和早停 | 原单位验证gap MAE | 同左，不用物理loss挑checkpoint |

这里的“PINN”具体指**物理约束增强的性质回归**，不是求解偏微分方程的经典PINN。它也需要联合预测相关物理量，否则无法从一个gap输出构造这些约束；与MTL路线的区别是独立任务头加物理loss，不使用任务注意力。

### MTL：七任务自注意力，不改变分子编码器

处理顺序：`分子输入 → 原骨干/原聚合 → 分子向量h → 七个任务投影 → 任务自注意力 → 七个独立标量头`。

1. 原骨干产生`h:[batch,D]`。Chemprop仍用BondMPNN和mean pooling；SPE/SMILES Transformer仍使用原分词和原序列聚合。
2. 七个独立投影`P_j(h)`产生`T:[batch,7,d]`。建议起点为`d=128`、投影`Linear(D,128)→ReLU→Dropout(0.1)`，任务身份由不同的投影参数区分，不再额外增加任务embedding。
3. 用4头自注意力作用于**七个任务token**，不是重新对原子或SMILES做一遍注意力。每个头的注意力为：

   $$A_b=\operatorname{softmax}_{\mathrm{key}}\left(\frac{Q_bK_b^\top}{\sqrt{d_k}}\right),\qquad Q,K,V=\operatorname{Linear}(\operatorname{LayerNorm}(T)).$$

   各头输出拼接并经过输出投影得到`O:[batch,7,128]`。`A:[batch,heads,7,7]`随分子变化，区别于C/D2共享所有分子的可学习全局7×7矩阵。保存注意力时要注明head平均/样本平均方式，不将权重直接解释成因果关系。
4. 使用残差门`T'=T+γO`，`γ`为可学习标量、建议初始化0；让模型从独立任务特征开始学习。这里退化为新的共享编码器七头模型，**不保证等于旧单任务Chemprop**，因为任务头结构已变化。零门初始化时注意力支路第一步梯度可为0，门更新后应恢复梯度，需要测试。
5. 七个独立`Linear(128,1)`输出标准化预测`z_hat:[batch,7]`。任务顺序固定为HOMO、LUMO、gap、alpha、mu、R2、ZPVE。

监督损失为：

$$L_{\mathrm{MTL}}=\frac{1}{7}\sum_{j=1}^{7}\frac{1}{N}\sum_{i=1}^{N}(\hat z_{ij}-z_{ij})^2.$$

每个标签独立用训练集Z-score；不把数值很大的R2/alpha直接与Hartree能量的原始MSE相加。验证时逆变换，并且只用gap的原单位MAE选择同一个checkpoint。所有七个测试指标都来自这个checkpoint，不逐任务挑epoch拼表。

已新增可复用`TaskAttentionHead(input_dim,n_tasks,hidden_dim,num_heads)`，输入分子表示，输出标准化预测，并可导出注意力权重。它不负责分词、图构造或优化器，因此可通过骨干适配器接入其他模型。

### PINN：十任务独立输出＋物理软约束

处理顺序：`分子输入 → 原骨干/原聚合 → 十个独立任务头 → 标准化监督loss + 原单位物理loss`。

1. 原七性质之外增加`u0、u298、h298`，分别为0K内能、298.15K内能和焓；均为QM9**总分子能量、Hartree**，不混用原子化能。样本、重复来源聚合、原始split和顺序保持不变。
2. 建议任务头仍为`Linear(D,128)→ReLU→Dropout(0.1)→Linear(128,1)`，十个头独立；输出`z_hat:[batch,10]`。没有任务注意力，也不强行从LUMO−HOMO计算gap替代独立gap输出。
3. 十个标签各自训练集Z-score，基础监督为十个标准化MSE的平均。使用保存的每列`mean/scale`，在保留梯度的情况下逆变换：

   $$\hat y_j=\sigma_j\hat z_j+\mu_j.$$

   物理差值用float64；不能调用`detach()`、转NumPy或在逆变换时切断梯度。
4. 用**预测值**构造三个残差，真实HOMO/LUMO/U/H不作为模型输入：

| 关系 | 预测残差/违约量 | 对gap的作用路径 |
|---|---|---|
| gap=LUMO−HOMO | `r_g=gap_hat−(LUMO_hat−HOMO_hat)` | 直接联系三个轨道性质的头 |
| H298−U298≈kBT | `r_h=H298_hat−U298_hat−kBT` | 约束辅助头，通过共享编码器间接影响gap |
| U298≥U0 | `v_t=ReLU(U0_hat−U298_hat)` | 同上；这是不等式，不要求U298=U0 |

温度与常数使用[已审计物理约定](physics_specification.md)，`kBT≈0.000944186 Ha`。不能写成gap=HOMO−LUMO，也不使用缺少电子能项的U0=ZPVE关系。

5. 为适应原始标签舍入，对等式使用容差带：`v_g=ReLU(|r_g|−δ_g)`，`v_h=ReLU(|r_h|−δ_h)`。建议起点`δ_g=1e-4 Ha`、`δ_h=1e-6 Ha`来自已有标签审计，属于数据表示容差，不是已证明最优的正则超参。不等式暂不设额外容差。
6. 三项物理罚项必须先变为无量纲。建议的稳定起点为：

   $$s_g=\sqrt{\sigma_{gap}^2+\sigma_{HOMO}^2+\sigma_{LUMO}^2},\quad s_h=\sqrt{\sigma_{H298}^2+\sigma_{U298}^2},\quad s_t=\sqrt{\sigma_{U0}^2+\sigma_{U298}^2}.$$

   所有`σ`仅来自当前seed训练集；对每个尺度设置`1e-8 Ha`正下界。这里用的是各物理量的边际标准差，**不是**几乎恒为零的真实关系残差标准差。它能避免量级失衡，但不保证三项梯度自动等强，也不保证达到舍入级的物理精度。

   $$L_{\mathrm{phys}}=\frac{1}{3}\left[\operatorname{mean}\left[(v_g/s_g)^2\right]+\operatorname{mean}\left[(v_h/s_h)^2\right]+\operatorname{mean}\left[(v_t/s_t)^2\right]\right],$$

   其中每项均指**先逐样本平方，再对样本平均**，不是把均值平方。实现写作`((v/s)**2).mean()`。

   $$L_{\mathrm{total}}=\frac{1}{10}\sum_{j=1}^{10}\operatorname{MSE}(\hat z_j,z_j)+\beta(e)L_{\mathrm{phys}}.$$

7. 建议起点：前10轮`β=0`，第11至20轮线性增加至0.1，此后保持0.1。预热轮数和最终权重已写入每份实际配置；先做数值/梯度烟测，若需调整只使用验证集，并记录调参成本。patience20仍按gap MAE运行，不为确保物理项启用而重置早停；若最佳epoch在预热阶段，须如实报告。

已新增可复用`QM9PhysicsLoss(z_pred,train_scalers,epoch)`，返回监督、三项物理loss与total loss；记录原单位残差的均值/分位数、违约率和实际β。物理loss降低不等于gap预测提升，最终仍由原单位gap MAE/RMSE判断。

### 接入骨干及验收要求

两个模块共用`encode(batch)→h:[batch,D]`适配接口，只替换预测头或附加loss，保留原数据输入、图/序列处理和优化器策略。3D模型应使用已有旋转不变的分子读出；如PaiNN含向量特征，不能直接展平xyz分量作为标量性质的任务token。没有现成分子向量的骨干需要单独审查适配，不声称“一行替换已完成”。

| 验收 | 必须检查的行为 |
|---|---|
| 张量与梯度 | 输出七/十列、固定任务顺序；残差门与注意力支路可学习；物理项梯度能回传到相关头和骨干 |
| 尺度与数据 | 仅训练集拟合七/十个scaler；val/test不参与拟合；数据扩展不改原划分和身份 |
| 物理数值 | 构造满足关系的预测，容差内loss为0；分别扰动gap/H/U时相应项响应；大总能量差值不溢出 |
| 推理隔离 | 推理只提供分子输入，不能需要真实辅助标签；3D需求由原骨干决定 |
| 指标与保存 | 同一best gap MAE checkpoint报告原七性质MAE/RMSE；保存每项loss、scaler、模块超参及完整训练状态 |

以上维度、head数、β与预热已编码并做CPU检查，不宣称已验证最优。两个新模块还没有正式训练结果；详细物理适用边界见[物理规范](physics_specification.md)，选模与尺度合同见[口径审查](protocol_review_20261019.md)。

## 推广顺序及基础重训范围

顺序依据当前LaTeX三seed测试gap RMSE（Hartree）预先确定；只用于安排预算，不用于挑选checkpoint或调超参。Chemprop因模块先行验证置首。

| 顺序 | 骨干 | 历史gap RMSE | 基础模型重训 | 两模块新增作业 |
|---:|---|---:|---|---:|
| 1 | Chemprop | 0.007018 | 七性质21项＋D1/D2六项 | 6 |
| 2 | DimeNet++ | 0.002731 | gap三seed，已接通 | 6 |
| 3 | DeMol | 0.003179 | 默认不重训，历史参照 | 6 |
| 4 | PaiNN | 0.003407 | gap三seed，已接通 | 6 |
| 5 | EGNN | 0.003545 | gap三seed，已接通 | 6 |
| 6 | TorchMD-Net | 0.004056 | 默认不重训，历史参照 | 6 |
| 7 | Transformer-M | 0.004938 | 默认不重训，历史参照 | 6 |
| 8 | 3D Infomax下游 | 0.006295 | gap三seed；预训练不默认重训 | 6 |
| 9 | SchNet | 0.007320 | gap三seed，已接通 | 6 |
| 10 | ESA | 0.008452 | 默认不重训，历史参照 | 6 |
| 11 | SPE Transformer | 0.009453 | 默认不重训，历史参照 | 6 |
| 12 | SMILES Transformer | 0.009764 | 默认不重训，历史参照 | 6 |

其余11骨干的模块配置通式为`fresh/<骨干>/task_attention7_seed<seed>.yaml`、`fresh/<骨干>/physics_thermo10_seed<seed>.yaml`；五个图骨干的gap基线为`fresh/<骨干>/single_task_seed<seed>.yaml`。上述后续81项代码/配置/指令已接通，输入与GPU验收及第一阶段分析仍在执行前完成，不自动启动。DeMol与TorchMD-Net沿用本项目当前分类，不据分类名推断其全部架构性质。

## 不再默认重训

- 所有B/C，以及细分MTL7、PhysicsGap7、MTL10-control对照。
- 已有Transformer基础模型的全部单任务；其模块改进仍必须保留。
- 五个后续图骨干的六项非gap单任务，暂不扩展这90项训练。
- 3D Infomax预训练、传统机器学习、旧失败Hybrid配置、历史重复机器复现。

这是一轮整体效果验证。物理方案同时增加额外标签，若改善只能先归于整体方案；没有独立对照不能声称物理损失的独立贡献。现有旧MAD/MAE结果不能伪装成新Z-score/MSE的同口径消融。
