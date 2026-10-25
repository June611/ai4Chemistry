# 决策与证据边界（2026-10-19）

## 实现交付更新

新200轮114项统一由molgap.modular.train执行；第一阶段33项全部有命令。下文“新模块未实现/27项可校验”属于实现前决策快照，现以implementation_delivery.md与operations.md为准。旧100轮入口不改，新模块、十标签、周期快照和新协议恢复已完成CPU检查，两个Chemprop新模块3090专项诊断和本轮全部114项GPU烟测均通过；正式收益待验收。

## 实现前决策快照（保留过程）

- 当前以整体改进有效性为目标，取消B/C重训和细分消融；保留原结果，不宣称B/C无价值。
- 第一阶段33项；全推广候选114项，替代原441项。27项配置可校验，两个新模块未实现。
- 已有Transformer基础单任务默认不重训；模块实验不取消。五个其他图骨干仅重训gap三seed，不扩全部七个单任务。3D Infomax预训练不默认重训。
- 全部新任务fresh200、batch256、patience20，三组seed42/2026/3407配对；不把旧100轮和新200轮seed混算。200是上限。
- 最佳epoch≥90仅为优先级信号，须结合验证曲线、真实停止轮数和学习率；不是收敛证明。选择某模型目标后，三个seed一并处理。
- 联合训练best和early stopping固定原单位Hartree验证gap MAE；单任务固定本性质原单位验证MAE。训练MSE与验证MAE承担不同作用，不把损失改成MAE，也不跨单位平均MAE。
- 七/十个标签逐列只在训练集拟合Z-score，保存mean/scale/ddof/单位/样本数。D1/D2的不确定性权重作用于标准化MSE，不能代替标准化。
- 图原子/键的离散编码、SMILES/SPE token ID不进行数值Z-score；连续特征若有scaler也仅用训练集。坐标/距离保留各骨干物理约定，不能任意逐坐标标准化破坏几何性质。
- 两个新模块是平行方案：七任务注意力；独立十任务头＋物理损失。暂不加第三种二者叠加方案。
- 物理残差先逐标签逆变换到原单位，以float64求能量差，再用训练集或预设的正尺度归一化。不能用接近零的真实恒等式残差标准差作分母；权重和舍入容差需实现前冻结。
- 热化学十标签保留原split/sample_id/顺序/重复来源均值，不用val/test拟合scaler、残差尺度或调参。
- 旧100轮、旧MAD/MAE与新200轮Z-score/MSE比较是历史参照，不是严格模块消融；如论文需要同预算归因，再增补对应骨干gap基线。
- 历史checkpoint审计与续训工具保留供追溯，本轮全部从头训练。已有terminal_fit终态保存不等于每50轮周期保存；后者仍待检查验收。
- PINN此处为物理约束正则化：gap=LUMO−HOMO；H298−U298≈kBT；U298≥U0。两等式一不等式，不造独立熵约束，不混用原子化能与总能量。
- 未启动服务器训练、未上传或提交当前改动；本地检查不能替代新模块GPU精度证据。

## 2026-10-20 服务器重启前的安全停机决策

- 用户因延长租约重启服务器；重启前对四条正式训练 lane 做优雅停机（先 lane 父进程后训练进程），避免硬重启时 checkpoint 半写损坏。
- 每轮 epoch 边界都写 last.ckpt，停机恢复损失 < 1 epoch；已完成任务靠 COMPLETED.json 幂等跳过，在训任务靠 --resume 从 last.ckpt 全状态续训（优化器/scheduler/RNG/loader/best），该链路已在烟测与正式运行中验证。
- 停机前留存最终快照 pre_restart_state_20261020_220551.json 与逐任务 CSV；8/42 完成（dimenet_pp 全部模块组 + single_task seed42 + 3dinfomax seed3407），4 项在训停于 ep45–162，lane 失败计数全程为 0。
- 中间结果：physics_thermo10 三 seed 跑满 200 epoch，test MAE 0.001486±0.0000195 / RMSE 0.002393±0.000132；task_attention7 全部早停于 34–61 epoch，test MAE 0.003167±0.000230。full-200 基线深收敛（single_task seed42 跑满 200），模块增益必须等 42 项全部完成后按配对 seed 同口径比较，不在此阶段下结论。
- 重启后恢复命令与核验清单见 formal_run_status_20261021.md；沿用同一 output-root，不动 runs/ 下任何文件。

## 2026-10-20 22:38 重启后恢复验证

服务器 22:11 重启完成（延长租约），22:38 用既定恢复命令重启四条 lane。恢复核验通过：8 个已完成任务自动 SKIP；4 个在训任务全部从 last.ckpt 无缝续跑（3dinfomax_2026 ep117→119、torchmd_ta7_3407 ep45→47、dimenet_st_3407 ep150→151、dimenet_st_2026 ep162→163），bad_epochs 与验证指标连续，无进度损失（< 1 epoch）。GPU0 的 3dinfomax 低利用率为已知 CPU 瓶颈，非异常。证据：monitor_round13_postrestart.json。

## 2026-10-20 旧 100-epoch 记录归档（防口径混淆）

- 旧结果目录 `outputs/summaries/qm9_results/` 更名为 `outputs/summaries/archive_100epoch_qm9_results/`，内置 ARCHIVE_NOTICE.md 溯源说明；内容为旧单属性协议（paired_split_qm9_neural_single_property_v1 等）汇总，与 fresh-200 不得混表。
- 旧 feature `docs/features/paired-seed-100epoch-benchmark/` 的 README/tasks 顶部加"已废弃"横幅，指向本 fresh-200 研究。
- 10 个旧矩阵配置（multiseed*.yaml、spe_affinity_ablation_multiseed*.yaml、spe_multiseed.yaml、training_mean*.yaml）顶部加 HISTORICAL 注释；不移动文件——旧 audit 脚本与根 README 仍按原路径引用，物理移动会破坏历史工具链；YAML 解析已验证不受影响。
- 传统 ML（XGBoost/RF/训练均值）结果沿用历史不重训，记录保留在归档与 reports/。
- 残留待办：根 README 的旧入口章节与 ai4Chemistry/CLAUDE.md 的 "Formal matrices" 行仍指向旧路径，属项目文档级修改，待用户确认后再改。

## 2026-10-21 21:45 队列重平衡（用户批准："可以重新平衡计算"）

- 动因：早停使原 LPT 分配失衡——lane1 于 ~22:10 跑完全部任务（GPU1 将闲置 ~17 h），lane2 于周四 ~03:00 跑完（GPU2 闲置 ~12 h），而 lane0/lane3 各压 ~20 h。
- 机制：runner 新增 --jobs/--skip-jobs（在 assign() 之后、所选 lane 内部过滤，保证与仍在运行的 lane1/lane2 分配完全一致，杜绝同任务双跑）；aux 运行时不覆盖 lane_manifest.json。
- 操作：TERM lane0/lane3 runner 及两个 torchmd pt10 trainer（ckpt ep122/ep91，损失 <2 epoch），带 skip 重启并 RESUME 成功（ep125/ep94 续训验证）；3 个等待器（scripts/aux_wait_and_run.sh）在触发任务 COMPLETED + GPU 空闲后自动接力。
- 任务迁移：torchmd_st_3407、schnet_42 → aux-A(GPU1)；schnet_3407 → aux-B(GPU2)；painn_42、schnet_2026 → aux-C(GPU0)。lane0 保留 pt10_3407+spe/smiles/chemprop_2026；lane3 保留 pt10_2026+esa_3407+spe/smiles/chemprop_3407。
- 42 项集合不变、每项恰好运行一次（skip 与 aux 列表互补，COMPLETED.json 幂等兜底）。
- 预期完工：由周四 ~17:00 提前至周四 ~11:30（关键路径 aux-C 的 schnet_2026）。

## 2026-10-22 02:25 接力计划二次微调

- spe_2026 未如预期早停（ep87 仍在改进），lane0 尾部顺延；利用 GPU3 在 lane3 完成（~07:30）后的空窗，新增 aux-D（触发 chemprop_3407，GPU3，跑 schnet_2026），aux-C 缩减为只跑 painn_42。
- 调整后关键路径 = aux-D 的 schnet_2026，预计周四 ~11:30 全部完成（维持原目标，避免滑向 ~13:30）。
- torchmd pt10 组 3/3 齐：MAE 0.002639 ± 0.000022，RMSE 0.003705 ± 0.000018（ep200/200/200）。对比 torchmd single_task（暂 2/3：0.002209/0.003297）：PINN 在 TorchMD-Net 上一致劣化（MAE +19%，RMSE +12%），与 DimeNet++ 上 RMSE −8.2% 的增益相反 → "模块收益依赖骨干"是论文可用结论。

## 2026-10-22 02:55 DeMol+PINN 扩展实验（用户指示："加上pinn试一下，找到一个能够提点的transformer类架构"）

- 架构判定：DeMol 是 transformer 族（双通道原子+键自注意力、32 头、跨图 attention，arXiv 2603.00568 论文重实现，adapter 已在本仓库 src/molgap/qm9_models/demol.py + modular/backbones.py 注册）。旧 review 中"无 adapter"说法已过时。
- A100 实测（诊断 smoke，TF32-off 协议口径）：~2.03 s/batch 完美线性（5120 行 41 s/ep、10240 行 81 s/ep），GPU-bound 99-100%，显存峰值 ~20 GiB；419 批全量 ≈ 850 s/ep → 200 ep ≈ 47 h/任务。4090 历史值 314 s/ep 因其未关 TF32；A100 FP32 矩阵乘慢 ~2.7×。
- 配置：新增 fresh/demol/single_task_seed{42,2026,3407}.yaml（生成器 make_executable_config，validate 通过，epochs 200/batch 256/patience 20/paired seeds 与正式队列同口径）；demol pt10 fresh 配置已存在且对齐。
- 烟测：demol st/pt10、spe pt10、smiles pt10 四路接线全部通过（diagnostic=true，不污染正式目录，存于 artifacts/qm9_epoch200/demol_smoke/）。
- 分级投入决策：①免费档——spe/smiles transformer 的 pt10×3（6 任务，~1.1 h/个，共 ~7 GPU·h），低成本先回答"是否存在能被 PINN 提点的 transformer 架构"；②demol 先跑 seed42+seed2026 两对（st+pt10 共 4 任务，4 卡并行，~47 h/个），seed3407 押后，视两对结果决定是否补齐（若 PINN 无效可省 ~94 GPU·h）。
- 调度：scripts/run_extension_chain.py（复用正式 runner 的 run_row，skip/resume/env 一致）× 4 条链，触发 = 各 GPU 最后一个正式/aux 任务的 COMPLETED.json + 显存清空；输出根 artifacts/qm9_epoch200/extension_20261022/（与 formal_4gpu 分离，聚合时不混表）。
- 链布局：GPU2: spe_pt10_42→smiles_pt10_42→spe_pt10_2026→demol_st_42；GPU1: smiles_pt10_2026→spe_pt10_3407→demol_pt10_42；GPU0: smiles_pt10_3407→demol_st_2026；GPU3: demol_pt10_2026。
- 预期：便宜档周四 ~12:10 全齐；demol 两对周六 ~10:30-11:15 出结果；若续跑 seed3407 对，全矩阵周一中午前完成。
- TF32 提速选项（未采用）：demol 单独开 TF32 可提速 ~3-4×（47 h→~12 h/任务），但需在正式队列结束后改 trainer 并作为记录在案的口径偏离；当前坚持协议一致（TF32-off），保证与 42 项正式结果同口径可比。

## 2026-10-22 15:30 扩展链修复与 demol 全矩阵排程

- 事故：本机为多容器共享 GPU 环境，GPU0/GPU2 存在容器外租户的小额显存占用（654+1590 / 1592 MiB，nvidia-smi 显示宿主 PID，/proc 不可见）；chain-0/2 的"GPU 空闲 <1000 MiB"判据永不满足，卡等 ~6 h（未跑任何任务，无数据损失）。
- 修复：run_extension_chain.py 增加 --mem-threshold 与 --trigger-root；以阈值 3000 MiB 重启 chain-0b/2b（外来占用 ≤2.3 GiB，40 GiB 卡余量充足；demol ~20 GiB 可安全共存）。chain-1/3（GPU1/3 无外来占用）不受影响、未中断。
- 顺序调整：chain-0b/2b 改为 demol single_task 基线优先（配对信号的关键路径），便宜档 spe/smiles pt10 排在其后。
- 追加：chain-1b/3b 以扩展 pt10 任务为触发，接力 demol seed3407 对（st_3407@GPU1、pt10_3407@GPU3）——GPU1/3 周五下午起本会闲置，直接排满；若周六两对 seed 显示 PINN 无效，可随时终止 seed3407 对止损。
- demol 实测提速：独占 GPU 时 ~543 s/ep（烟测 850 s/ep 因与 4 条正式 lane 争抢），200 ep ≈ 30 h。
- 新时间表：demol pt10_2026/pt10_42 → 周五 ~15:00/16:00；demol st_2026/st_42（15:25 启动）→ 周六 ~21:40；seed3407 对 → 周六 ~22:00 / 周日 ~22:00（pt10_3407 最晚）。便宜档剩余 4 项 → 周六深夜~周日凌晨。若周六两对信号明确为负，砍掉 seed3407 对后全部结束于周六 ~22:00。
- pkill 教训（再次）：杀进程与含 '--gpu 0' 字面量的启动命令不可同命令行混排（正则跨文本自匹配会误杀自身 shell）；按 PID 分步执行。

## 2026-10-22 18:00 MTL 路线更换为 MLoRE + 外部清扫事故恢复

- 事故：17:05-17:11 全部 4 个 demol trainer + 4 条链进程被外部清扫杀死（非 OOM：480 GB 内存仅用 8 GB、dmesg 无当日记录；uptime 显示 3 用户在线，机器为 Nomad 托管共享环境）。checkpoint 完好（last.ckpt ep48/53/6/6），进度零丢失。
- 恢复：18:00 以 4 条新链（chain_gpu{0,1,2,3}c）重启全部工作，run_row 自动 RESUME；新增每小时健康检查提醒（自动恢复守护）。
- MTL 路线更换（用户指令）：task_attention7 → mlore7（MLoRE, CVPR'24 arXiv:2403.17749，低秩专家混合 + 通用共享路径）。实现细节与适配映射见 docs/features/qm9-mlore-multitask/README.md。旧 TaskAttentionHead 代码删除、6 份旧配置归档、旧结果保留作历史证据。
- mlore7 六项烟测通过后全部排入链：GPU1: demol_pt10_42(续) → dimenet_mlore_42 → dimenet_mlore_2026 → demol_st_42(续)；GPU3: demol_pt10_2026(续) → torchmd_mlore_42 → torchmd_mlore_2026 → demol_st_2026(续)；GPU0: dimenet_mlore_3407 → torchmd_mlore_3407；GPU2: spe_pt10_42 → smiles_pt10_42 → spe_pt10_2026 → smiles_pt10_3407。
- mlore7 速度：头部 210 个小算子 → 每 batch 开销约翻倍（dimenet ~1.2 s/batch vs st 0.59）；ta7 先例（34-109 ep 早停）预期单任务 5-15 h。demol st 在租户卡（GPU0/2 曾 ~980 s/ep）改到净卡 GPU1/3 续跑（~545 s/ep）。
- 新时间表：便宜档 4 项今晚 ~23:30 齐；MLoRE 6/6 最晚周六上午（早停预期）；demol pt10 两 seed 周五 16-17 点；demol st → demol 两对配对信号周日下午。demol seed3407 对仍押后待试点信号。

## 2026-10-25 21:10 扩展实验全部收官（58/58）——两章终判

- demol_st_2026（ep200，0.001711/0.003183）为最后一项；全部 4 条链 failed=0 自然退出，无残留进程。终版数据表、配对判决与全场排名见 extension_results.md。
- MTL 章（MLoRE 6/6）：DimeNet++ +90.1% RMSE ❌❌（比旧 ta7 还差 15%）；TorchMD-Net −0.5% RMSE / +4.8% MAE ≈ 打平（比 ta7 好 ~25%，修复负迁移但未胜单任务）。跨骨干差 ~90 pp，训练形态两极（ep~30 崩盘 vs 188 ep 长跑）。章结论 = 诚实负面/中性结果 + 强骨干依赖性；ta7 降为历史对照。
- PINN 章（DeMol 3×2 补齐）：seed42 +3.5% ❌ / seed2026 −23.8% ✅ / seed3407 −18.1% ✅ → 均值 −12.7% RMSE（MAE −2.1%）。demol_pt10_3407 = 0.002361 为全体系最佳单次（低于 dimenet_pt10 均值 0.002393）；demol 单任务 0.003060 亦为最强 transformer 基线。"能提点的 transformer 架构" = DeMol+PINN（家族最佳、单次达图级）。
- PINN×骨干终版记分牌（RMSE 3-seed）：SMILES −13.6% ✅ / DeMol −12.7% ✅ / DimeNet++ −8.2% ✅ / SPE −4.9% ✅ / TorchMD +12.0% ❌ → 4/5 为正；绝对最佳仍为 dimenet_pt10（0.002393±0.000132）。
- DeMol 风险注记：种子方差大（pt10 RMSE SD ±0.00048，dimenet ±0.00013 的 3.6 倍）；seed42 PINN 早停（ep71）为唯一败票。论文中 DeMol 结论应带方差披露。
- 守护：10-22 18:56 起每小时健康检查 75 轮至全部收官，无第二次外部清扫；提醒已删除。GPU0/2 租户拖速（demol ~1000-1090 s/ep vs 净卡 ~540）仅影响工期不影响结果。
- 未做/不做：不动 results.md 正式 42 项主表（扩展数据单独成文）；不 git 提交（未获授权）；旧 ta7 结果只作历史证据不混表。
