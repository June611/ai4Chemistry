# TODO：实现与验收

## 2026-10-20 统一基线矩阵（当前状态）

- [x] 按 main.pdf 恢复 10 个神经单任务 gap 基线：Chemprop、SchNet、3D Infomax、EGNN、PaiNN、DimeNet++、SMILES Transformer、SPE Transformer、ESA、TorchMD-Net，各 3 个配对 seed，共 30 项。
- [x] 按论文两章保留两条 SOTA 模块路径：DimeNet++/TorchMD-Net × task_attention7、physics_thermo10，共 12 项。
- [x] DeMol 移出正式队列；Transformer-M 不在 main.pdf 正式神经基线表，均保留为扩展池，不运行。
- [x] `prepare_qm9_epoch200_review.py` / `smoke_qm9_epoch200.py` / 协议测试同步更新；42 项队列生成完成。
- [x] 统一实验表与 CSV 已按 main.pdf 分组、RMSE 历史列和 fresh-200 任务列整理。
- [x] 42 项 GPU 烟测（500 条/1 轮）已完成：`expected=42, passed=42, failed=0`，记录于 `a100_smoke_unified_gpu0/`；3D Infomax 预训练 checkpoint 仅作额外准备任务，不计入 42 项下游评估。
- [ ] 全量最大输入 batch 和各正式机器的容量/并发校准。
- [ ] 42 项正式训练（用户指令后启动）；结果按 3 seed 均值±样本 SD 报告，模块增益相对同路径单任务基线。

## 历史记录（114 项范围，已被上面收敛取代）

- [x] 缩减为114项，第一阶段33项；B/C与细分消融不重训。
- [x] TaskAttentionHead：七任务投影、4头自注意力、128维、零初始化残差门。
- [x] QM9PhysicsLoss：十标签监督、两等式一不等式、逐列逆变换、float64残差、容差与归一化、10＋10轮权重预热。
- [x] 十标签扩展保留原sample_id、重复来源均值、顺序和三个split；本地真实数据已生成。
- [x] 12骨干特征适配；全部小模型CPU前向/反向检查，3D适配器检查刚体不变性。
- [x] 统一200轮入口，保留原Chemprop基础及D1/D2结构；生成全部114份配置和指令。
- [x] 每轮last、每50轮完整快照、terminal；保存优化器/scheduler/RNG/loader/早停/历史best/scaler和身份。
- [x] 新200轮任意完整epoch边界恢复入口；配置或数据变化拒绝恢复。
- [x] 五种Chemprop配置连续与中断恢复参数、预测一致；50轮快照与双worker恢复通过。
- [x] 原单位验证MAE选模；测试CSV回读核对；保存七性质、辅助任务、注意力/UW、时间及资源记录。
- [x] 72项独立本地测试通过；真实QM9前缀16条检查通过，均标记diagnostic。
- [x] 3090独立目录部署；两个Chemprop新模块GPU前缀256条诊断通过，物理模块恢复至第12轮，11/12轮约束启用。未启动正式训练。
- [x] 全114项在3090上以500条/1轮/batch256实际GPU运行，通过状态及原单位预测回算审计；73项干净副本测试通过；A100 上 114/114 复测通过（`a100_smoke_gpu{0,1,2}`）。
