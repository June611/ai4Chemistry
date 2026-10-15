# QM9 DeMol 与 Transformer-M：实现及训练

目标：把两个方法接入项目既有固定划分、单任务gap训练和资源记录流程；仅QM9，从零训练，不下载外部预训练权重。数据/输出/权重不进Git。

服务器共享原环境时，先在新项目根目录执行 `export PYTHONPATH="$PWD/src"`，确保解释器加载本次代码。

## 来源与复现范围

| 模型 | 代码/结构来源 | 本次范围 |
|---|---|---|
| Transformer-M | [官方仓库](https://github.com/lsj2408/Transformer-M)，commit `45647e143a5282e0e97117969396446084bcf1ab`；MIT | vendored官方QM9 encoder/attention/GBF等模块及对应CLS头；移除Fairseq环境依赖 |
| DeMol | [论文](https://arxiv.org/abs/2603.00568)，本地PDF hash见sources文件；§3.2公式5–10、附录B/D/H | 按论文重实现，不称作者官方代码；尚未找到论文或会议页面链接的作者仓库 |

来源锁：[demol_transformer_m.sources.json](../../../configs/qm9_models/demol_transformer_m.sources.json)。Transformer-M原文件/修改后hash、MIT许可证和移植改动在[src/molgap/_vendor/transformer_m](../../../src/molgap/_vendor/transformer_m/PROVENANCE.md)。下面命令均从项目根执行。

两个模型采用论文/脚本的12层、768维、128个Gaussian核；Transformer-M官方QM9脚本为32头、FFN768、CLS读出、`no_2d=True`，所以本次不启用预训练时的随机模态切换，不把仅SMILES输入当作Transformer-M。

DeMol保留原子/键两个通道、原子自注意力/键自注意力、两方向incident跨图注意力、原子SPD与距离、键中点距离与方向余弦编码、5Å原子mask、键相邻/共轭mask。化学键来自身份核验过的QM9 SDF，不从属性标签推断。

### 论文未公开或未明确的实现选择

DeMol没有可核查的作者代码；以下是重实现选择，不能声称为作者精确配置：32头、FFN768；post-norm与GELU；Gaussian距离宽度8Å，均值/方差按论文固定；两个通道最终mean pooling后拼接MLP；跨图位置偏置采用原子到键中点的距离；每条化学键使用两个方向的键节点，避免方向余弦依赖任意原子编号；键共轭关系采用相邻共轭键的连通分量。

结构mask采用dense张量及PyTorch SDPA执行，保留允许的交互关系；没有实现专用稀疏attention kernel，不宣称达到论文讨论的稀疏线性复杂度。

论文将方向向量余弦称为角/二面角编码；本实现遵循公式中的向量余弦，**不声称它等价于通用四原子二面角计算**。Gaussian使用通常的正值概率核。SPD断连采用独立远距离桶；padding和无incident键行显式处理。以上细节与未来作者代码对比后仍可能修订，需版本化、不能覆盖已训练结果。

共价半径BCE头与`1.15*(r_i+r_j)`目标已实现并测试，可通过`bond_regularization: true`及训练`auxiliary_bond_weight`启用。半径使用RDKit固定周期表、单位Å。主配置关闭该预训练辅助目标：论文将它列在预训练策略，当前任务明确排除外部预训练，QM9从零监督训练不额外假定作者沿用该损失；主loss与既有Top5一致。开启此项必须作为单独标注的QM9辅助监督实验，不能偷偷改变主配置。

## 固定实验协议

- 现有133798个样本与三split；107038/13380/13380。seed3407、42、2026同时用于划分与初始化。保留当前队列既有样本，不为新模型静默过滤异常SDF或官方uncharacterized样本。
- 每任务batch256、最多100epoch、patience20，min_delta=1e-5 Hartree；checkpoint保存验证gap MAE最低点。
- 缩放只拟合train，沿用本项目现有3D候选的MAD+MAE，原单位gap MAE选模；保存逆变换后的Hartree预测。比较MAE/RMSE时乘27211.386245981得到meV，MSE乘该因子平方。
- 两模型LR均7e-5、clip5、weight_decay0，依据论文QM9/官方脚本；DeMol AdamW，Transformer-M Adam。调度器沿用统一入口的100epoch cosine，不套用超出当前训练步数的60k warmup。
- 这与论文的外部预训练、600k步微调、batch128、MSE/standard等设置不同；属于**项目统一协议下的QM9从零训练比较**，不能把对文献的误差差值全归因实现好坏。
- 默认FP32，不启用额外AMP/TF32策略；开启activation checkpointing降低激活内存，不改变batch或模型宽度。测试覆盖checkpoint前后输出/梯度一致。

## 数据输入与一次性缓存

Transformer-M沿用官方QM9 `removeHs=True`：只选择现有坐标中的重原子，OGB九维原子类别，3D距离编码。它不读取真实HOMO/LUMO/gap作为输入，也不在线构建RDKit图。DeMol保留现有几何的全部原子及SDF键、OGB原子/键类别和SPD。

```bash
uv run --no-sync python -m molgap.qm9_models.topology
```

缓存位于忽略的`data/topology/qm9_full.npz`，配套JSON记录SDF/拓扑hash、三几何文件hash与部分sanitization标记。准备时逐样本核验原ID、原子序列、坐标，并证明三份几何只改变顺序；不读取性质标签。异常价态保留并记录部分sanitization，而非改变固定队列的样本。不能把这个缓存套用到未知几何文件。

## 双卡启动与观察

```bash
# 必须选择全新目录；两个模型每卡一个任务，不使用DDP。
uv run --no-sync python scripts/run_qm9_dual_reproduction.py \
  --directory artifacts/auto_training/qm9_demol_transformer_m_YYYYMMDD_HHMMSS --prepare

# 真实完整训练集1epoch，batch256，不是默认batch8的小烟测。
uv run --no-sync python scripts/run_qm9_dual_reproduction.py \
  --directory artifacts/auto_training/qm9_demol_transformer_m_YYYYMMDD_HHMMSS --phase smoke

# 只有两项烟测经工件校验通过才准入；按实测epoch时间分配六个独立任务。
uv run --no-sync python scripts/run_qm9_dual_reproduction.py \
  --directory artifacts/auto_training/qm9_demol_transformer_m_YYYYMMDD_HHMMSS --phase formal
```

烟测时GPU0为DeMol，GPU1为Transformer-M；正式启动前用实测完整epoch耗时按长任务优先分配六个独立任务到两卡，不将同一个模型固定在一张卡导致负载倾斜。实际归属和粗略ETA记录在manifest.json。正式任务之间无依赖，某seed失败后继续下个seed；烟测失败则禁止启动未验收正式批次。通过既有持久队列记录每5秒资源、任务状态和独立日志，不依赖LLM轮询。每卡最多一个任务，不预设可三seed同卡。

查看`<directory>/formal_gpu0/runtime/state.json`与`formal_gpu1/runtime/state.json`、对应`runtime/*.log`。每个run的`history.csv`记录epoch、MAE、训练/epoch耗时、CUDA allocator峰值；队列资源采样记录进程显存、主机RSS/PSS和GPU状态，`training_timing.json`与`inference_timing.json`记录fit和测试吞吐。FLOPs未计数则留空，不用samples/s冒充FLOPs。

正式结果位于`<directory>/formal_gpu*/runs/*/seed*/`，遵循项目已有metrics.json、predictions/test_predictions.csv、model.pt格式，后续按三seed汇总。首次实际运行路径、烟测资源与启动结果追加到[operations.md](operations.md)。

## 验收状态

实现通过37项针对性集成/回归检查，包含两个模型反向传播、旋转/平移/原子置换、padding、单样本与batch独立性、checkpoint往返、activation checkpointing梯度、官方collator数值对照、三seed调度/烟测门禁及原QM9适配器/调度行为。lint通过；完整数据缓存、GPU烟测与正式训练状态见[TODO.md](TODO.md)。未完成训练不能在论文或正式比较表中填造成绩。
