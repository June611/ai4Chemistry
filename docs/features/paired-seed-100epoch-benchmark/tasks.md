# 三组配对 seed、100-epoch 协议 TODO

更新时间：2026-10-11

## U：最小改动统一训练口径与服务器交接

最新需求：先统计与标准化训练口径，评估 batch/LR 公平性，最后审查；大文件仅提供三个选择，
本轮执行代码与配置审查；数据只提供方案，不上传或部署。详细目标与范围见 [standardization-plan.md](standardization-plan.md)。
以下是新任务；既有 P/A 勾选只代表此前预算协议验收，不能视作本轮完整训练口径已通过。

- [x] U01 读取代码、基础配置和历史审查，整理当前预算、loss、scaler、选模与LR差异。
- [x] U02 保存23份基础配置的超参数清单（19下游神经网络、1预训练、3传统基线）。
- [x] U03 制定最小修改方案、公平性边界、验收条件和三个数据迁移备选方案。
- [x] U04 按最新沟通采用gap MAE选模；保留各方法现有loss/scaler、辅助任务与物理约束，
  不强制改成共同MSE/Z-score。上一次统一loss/scaler的推荐已撤回。
- [x] U05 冻结统一选模指标：逆变换后的Hartree validation delta_e MAE；
  checkpoint与earlystop共同监控，patience20、mode=min、min_delta=1e-5 Hartree。
- [x] U06 Chemprop单任务/多任务通过官方注册机制接入项目内gap指标；Phase3接通独立gap MAE，
  不用三任务平均MAE代替，不修改site-packages或改变模型结构。
- [x] U07 保留QM9的MAE/mean-MAD及其他模型的MSE/Z-score；只统一选模、预算和评估，
  保持辅助监督、物理项单位与权重，检查差异披露。
- [x] U08 对所有下游入口统一min_delta、最佳checkpoint和早停计数语义；
  预训练保留对比指标并单列为非gap任务。
- [x] U09 评估batch256的全局/单卡语义、最后不足batch、drop_last和参数更新次数；
  OOM显式记录，不单独缩batch、不默认梯度累积。
- [x] U10 审查各模型optimizer/LR/warmup/scheduler/weight_decay来源；重点检查Chemprop
  四倍LR依据与Infomax lr_patience25/早停20的关系；本轮不进行LR搜索。
- [x] U11 冻结并披露各架构的LR与优化配方；如要调参，另设所有模型相同候选数/验证预算。
- [x] U12 活动配置切换为paired_split_gap_mae_v2与outputs/paired_gap_mae_v2，
  隔离旧val_loss选模结果；三组矩阵仍严格一一配对。
- [x] U13 增强协议审计器，逐模型验证预算、配对、监控目标/单位、min_delta、loss/scaler
  与必要例外，不能只检查配置字段而忽略入口实际使用值。
- [x] U14 添加必要行为回归：train-only统计、只按gap选模、完整验证集加权MAE、
  同一验证序列的checkpoint/stop epoch、逆变换与模型恢复。
- [x] U15 完成三套数据的正式配置dry-run；缺Infomax匹配checkpoint记PENDING。
- [x] U16 最终代码/配置/文档审查，运行必要pytest、Ruff、shell语法和数据/geometry哈希审计。
- [x] U17 输出最终审查表，区分PASS/PENDING/FAILED/OOM/N/A；服务器未运行不记通过。
- [x] U18 本地审查通过，保存最终TXT/差异、清理旧data索引并执行代码交接；
  提交/push最终事实以Git HEAD与origin/main和artifacts/gap_mae_commit_receipt_20261011.txt为准。
- [x] U19 用户选择SCP；QM9数据归档已迁移至指定服务器并通过86文件哈希核验。
- [ ] U20 部署阶段取得服务器连接信息、GPU/显存，完成环境和batch256预检，
  再按3407→42→2026逐模型正式训练；环境部署进度见[服务器部署功能](../qm9-server-deployment/tasks.md)，
  batch256显存与正式训练仍待实验阶段验收。

## P0：冻结范围与基准表

- [x] P01 建立纳入模型清单和稳定模型 ID：
  Chemprop 单任务/多任务/物理一致性、原子 Transformer、SPE 单任务、四个 SPE 多任务消融、
  QM9 十模型、XGBoost、Random Forest、训练均值。
- [x] P02 排除历史归档和诊断配置：`configs/archive/**`、`egnn_test.yaml`、
  `spe_transformer_10epoch.yaml`、`configs/experiments/exp003/004`。
- [x] P03 保存协议纳入清单，记录模型 ID、预算适用性、seed 关系、训练入口和排除项。
- [x] P04 固定三组一一配对关系：
  `(3407,3407)`、`(42,42)`、`(2026,2026)`；禁止生成 3×3 交叉组合。

## P1：统一配置

- [x] P05 Chemprop 三模型统一为 batch 256、100 epochs、patience 20，并为三个 split 生成配对配置。
- [x] P06 原子 Transformer、SPE 单任务和四个 SPE 多任务消融统一为 batch 256、100 epochs、
  patience 20。
- [x] P07 QM9 十模型的下游训练统一为 batch 256、100 epochs、patience 20，不静默回退到较小
  batch；显存失败在实验阶段记录为协议不可行。
- [x] P08 3D Infomax 对每个 split 独立预训练；预训练 seed 与 split seed 配对，并采用
  batch 256、100 epochs、patience 20。
- [x] P09 XGBoost、Random Forest 和训练均值改为支持三个 split，并将随机种子绑定到 split seed。
  batch、epoch、early stopping 标记为 N/A，不把树数量伪装成 epoch。
- [x] P10 所有运行配置写入 `protocol_id`、`split_seed`、`training_seed` 和
  `seed_pairing: matched`，输出目录同时区分模型与 split。

## P2：执行器

- [x] P11 扩展现有批处理合同，支持 `seed_pairing: matched`；每份矩阵只允许一个与 split seed
  相同的 training seed。
- [x] P12 配对模式为每个 seed 自动选择对应 data config、Chemprop reference config 和处理目录。
- [x] P13 为 Chemprop、Transformer/SPE、传统基线生成统一配对运行计划。
- [x] P14 修改 QM9 顺序训练脚本，使 split seed 和 training seed 按位置配对，而不是双重循环。
- [x] P15 3D Infomax 为三个 split 分别选择匹配 manifest/geometry 哈希的预训练 checkpoint。
- [x] P16 防止旧五 seed 配置被误用于新协议；历史配置保留用于追溯，但从当前 README 训练入口移除。

## P3：静态审查

- [x] P17 新增协议审计器，逐个解析最终配置和生成计划。
- [x] P18 审计所有神经网络运行均为 batch 256、100 epochs、patience 20。
- [x] P19 审计恰好三组运行，且每组 `split_seed == training_seed`，不存在 seed 7/123。
- [x] P20 审计三套 split manifest/geometry 哈希、样本数量、互斥性和数据身份继续通过。
- [x] P21 审计验证集选模和测试集终评逻辑不因预算修改而改变。
- [x] P22 审计标签单位、train-only scaler、预测列、指标结构和结果汇总兼容性。
- [x] P23 对不适用 batch/epoch/patience 的传统模型生成显式 N/A 记录，而不是让审计器漏检。
- [x] P24 审计汇总报告将三次结果标记为“paired split-seed runs”，不能表述为固定 split 下的
  三次初始化重复。

## P4：实现验证

- [x] P25 更新配置/执行器单元测试：配对数量、seed 相等、路径选择、输出目录和非法交叉组合。
- [x] P26 对 Chemprop 三模型执行三组 dry-run，检查命令行参数。
- [x] P27 对原子 Transformer、SPE 单任务和四个 SPE 多任务消融共六个模型执行三组完整数据
  dry-run。
- [x] P28 对 QM9 十模型执行三组完整数据 dry-run；3D Infomax 使用各 split 匹配的 smoke
  checkpoint 验证路由。
- [x] P29 对三个传统基线执行三组数据加载/拟合计划 dry-run。
- [x] P30 运行 Ruff、相关 pytest、shell 语法检查和协议审计器，要求全部通过。

## P5：实验阶段验收

- [ ] P31 先在 split/training seed 3407 上逐模型启动，验证 batch 256 的显存可行性。
- [ ] P32 对 OOM 模型保留失败日志并提交协议复审；未经批准不得单独降低 batch。
- [ ] P33 单组通过后运行 seed 42、2026，核对各自产物中的 split/training seed 和数据哈希。
- [ ] P34 每个运行核对最多执行 100 epochs、early stopping 行为、最佳验证 checkpoint 可恢复、
  测试预测完整且有限。
- [ ] P35 从逐样本预测重新计算指标，汇总三组 paired split-seed 的均值、样本标准差和单组结果。
- [ ] P36 输出最终口径审查报告，逐模型给出 PASS / N/A / OOM / FAILED，不将未运行项标记通过。

## 完成标准

- [x] A01 所有可迭代神经网络训练计划严格满足 256 / 100 / 20。
- [x] A02 每个纳入模型恰好对应三组 seed 配对，没有五 seed 或交叉组合残留。
- [x] A03 传统模型的不可适用参数得到明确记录，数据与 seed 协议仍保持一致。
- [x] A04 3D Infomax 预训练和下游 checkpoint 均与各自 split 严格绑定。
- [ ] A05 自动审计与实际产物审查结论一致，README 命令可直接在服务器执行。

## 本地证据

- 历史测试记录见此前审查；本轮证据以 final-gap-mae-review.txt 为准。
- 独立数据包约 162.13 MiB；压缩包 SHA-256、86 个本地源文件及归档内容逐项验证通过。

- 2026-10-11：重新核查完整训练口径，预算与配对 seed 一致；选模指标、loss、标签 scaler
  仍存在方法族差异，不能用静态审计 PASS 声称全部配方一致。见 `review-20261011.txt`。
- [x] 修正 QM9 CLI seed 覆盖引起的模型名称变化；同模型三组按稳定 `model_id` 汇总。
- [x] 汇总产物保留 `run_mode`，明确三组标准差同时包含划分和初始化变化。
- [x] Chemprop与其他下游模型统一原始Hartree gap MAE选模；见U系列和新审查记录。
- [x] 保留Z-score/mean-MAD、MSE/MAE为方法特有配方，明确固定配方比较边界。
- [x] 完成 P27/P28：66次全数据计划通过；Infomax使用配对smoke权重检查路由。
- [ ] A05服务器显存可行性与完整产物验收仍待执行；本地dry-run不代替服务器GPU检查。
- [x] 本轮代码审查通过；最终TXT已保存，正式训练不作为代码交接的前置条件。
- [x] 归档及86源文件校验通过；34个旧data条目已取消索引跟踪，保留所有本地文件和`.gitkeep`；
  不重写历史，不forcepush。

- 协议审计：`reports/paired_seed_100epoch_protocol_audit.json`，状态 PASS，66 次评估训练计划，
  另含 3 次 3D Infomax 预训练。
- 数据审计：`reports/qm9_split_seed_audit.json`，三套 split 状态 PASS。
- 六份 Chemprop/Transformer/SPE 矩阵均已成功展开；Chemprop 三模型的 9 个配对配置和传统
  基线的 9 个配对配置均通过全量数据 dry-run。
- `tests/test_multiseed.py`、`tests/test_summary.py`、`tests/test_ml.py`、
  `tests/test_3dinfomax_adapter.py`、`tests/test_qm9_adapters.py` 共 34 项通过；Ruff 与四个 shell
  入口语法检查通过。
- QM9 seed 42 的 EGNN 全数据 dry-run 显示 107038/13380/13380 行且预算为 256/100/20；
  非配对的 split 42 / training 3407 被入口拒绝。

## 本轮最终验收证据

- 完整pytest：95 passed、31 warnings；Ruff、四个shell语法、git diff --check通过。
- 66个运行计划PASS：9个Chemprop命令/API计划，57个其他方法全数据加载/计划；
  数据本身由split/geometry独立审计，不能称Chemprop dry-run已完成全量训练数据加载。
- 三个Infomax smoke预训练/匹配checkpoint路由通过，正式预训练和GPU仍PENDING。
- 行为回归覆盖gap-only选模、样本数聚合、逆变换、物理诊断单位、恢复和earlystop。
- 最终结论与交接：[final-gap-mae-review.txt](final-gap-mae-review.txt)。
