# MTL 章节技术路线更换：task_attention7 → MLoRE（2026-10-22）

## 用户决策
"MIL技术路线改成基于MLoRE技术路线的，修改代码，去掉之前的实现，重新跑MIL实验"。

## 参考方法
MLoRE = Multi-Task Dense Prediction via Mixture of Low-Rank Experts, CVPR 2024
(arXiv:2403.17749, 官方代码 github.com/YuqiYang213/MLoRE)。参考实现已核对：
models/transformers/MLoRE.py 的 MOEBlock/LoraBlock 与 losses/loss_schemes.py 的
MultiTaskLoss（CV² 负载均衡），超参取自 configs/nyud/nyud_vitLp16_MLoRE.yml。

## 忠实保留的核心机制
1. 三路径求和融合：`S^t = BN(共享低秩专家混合) + 任务专属低秩专家×prob_mix门控 + 通用共享路径(out.detach() 梯度截断)`
2. 15 个任务共享低秩专家，秩逐专家递增（多样性），全部线性无激活
3. 每任务独立路由器：输出 logits + 可学习噪声 std + prob_mix；训练时加噪，
   pre-softmax top-k 屏蔽（desert_k 个最小 logit 置 -1e10 后 softmax），top-k=9/15
4. 融合后 BN → GELU → 输出投影；两个 MLoRE 模块顺序堆叠（module2 吃 module1 的任务特征）
5. CV² 负载均衡损失：每模块对任务×样本平均的专家路由概率算 (std/mean)²，两模块求和
   × weight 0.0005，加入总损失（记录为 train_load_balancing）

## 场景适配（dense-prediction → 分子向量回归）
- 1×1/3×3 Conv2d → Linear（每分子单向量、无空间维）
- SpatialAtt 空间上下文路由 → 任务向量上的 Linear 路由（无空间可池化）
- 多尺度特征 + task_mask 注意力加权 → 单尺度，退化为恒等（省略）
- BatchNorm2d → BatchNorm1d；rank_list 按 C=128 等比缩放为 [4..60]（步长4，均 < C/2 保持低秩性）；
  spe_rank=4（= rank_list[0]，同参考比例）
- 路由诊断替代注意力诊断：attention.json 存 [模块×任务×专家] 平均路由概率

## 超参（写入 6 份 fresh 配置的 module 段）
hidden_dim 128 / rank_list [4,8,...,60] / spe_rank 4 / topk 9 / pre_softmax true /
num_modules 2 / balance_weight 0.0005；头部参数 ~1.17M（dimenet 输入 512 维时）。
训练协议与正式 42 项完全一致：200ep/batch256/patience20/配对 seed/TF32-off/val gap MAE 选型。

## 移除与归档
- `TaskAttentionHead` 从 heads.py 删除；train.py/config.py/生成器中 task_attention7 全部替换为 mlore7
- 旧配置归档至 configs/qm9_epoch200/archive_task_attention7_20261022/（含 README）
- 旧 ta7 六项已完成结果保留在 artifacts/qm9_epoch200/formal_4gpu/runs/，论文中作"被取代路线"历史证据，不与 mlore7 混表

## 实验矩阵（extension_20261022）
dimenet_pp × mlore7 × seed{42,2026,3407} + torchmd_net × mlore7 × seed{42,2026,3407}；
烟测通过（2ep×5120行：路由概率和=1、均衡损失生效、诊断 COMPLETED）。
对照基线 = 正式队列已完成的同骨干 single_task（3 seed 全齐）。

## 结果（2026-10-25，6/6 完成，test gap MAE/RMSE Hartree）

| 骨干 | seed42 | seed2026 | seed3407 | 3-seed MAE | 3-seed RMSE |
|---|---|---|---|---|---|
| DimeNet++ | 0.003409/0.004553 (33*) | 0.003730/0.005064 (29*) | 0.003879/0.005255 (30*) | 0.003672±0.000240 | 0.004957±0.000363 |
| TorchMD-Net | 0.002312/0.003299 (172*) | 0.002357/0.003404 (200) | 0.002311/0.003172 (193*) | 0.002327±0.000026 | 0.003292±0.000116 |

配对判决（RMSE）：
- vs 单任务：DimeNet++ **+90.1%** ❌❌（st 0.002608）；TorchMD-Net **−0.5%** ≈ 打平（st 0.003308，MAE +4.8%）
- vs 旧 ta7：DimeNet++ +15.4%（更差）；TorchMD-Net **−24.6%**（ta7 0.004367 → 大幅修复）

结论：MLoRE 修复了 transformer 路径上 ta7 的灾难性负迁移（+32%→打平），但两骨干均未跑赢单任务；
跨骨干相差 ~90 pp，训练形态两极（DimeNet ep~30 崩盘早停 vs TorchMD 平均 188 ep 长跑、
RMSE SD ±0.000116 全场最稳之一）。MTL 章以"负面/中性结果 + 强骨干依赖性"定稿。
运行明细与全场排名：docs/features/qm9-epoch200-modular-study/extension_results.md。
