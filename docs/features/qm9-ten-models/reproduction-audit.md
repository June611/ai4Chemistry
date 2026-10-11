# QM9 十模型复现与公平性审查

更新时间：2026-10-11

本报告的结构/数据审查沿用历史证据；当前预算以
[三组配对 100-epoch 协议](../paired-seed-100epoch-benchmark/README.md) 为准。
下面历史 30 次 dry-run 不替代新路由与 checkpoint 门禁下的 P28 验收。

结论标记：`PASS` 表示满足当前协议；`ADAPTED` 表示保留论文关键思想，但接口、容量或训练配方
按本项目改写；`CONDITIONAL` 表示代码可训练，但正式比较前需要冻结实验政策。

## TODO 验收清单

### A. 官方实现与关键思想

- [x] A01 核对 CSV 的十个模型、官方仓库 URL 和固定 commit。`PASS`
- [x] A02 EGNN：输入特征、E(n) 等变消息传递、坐标和读出。`ADAPTED`
- [x] A03 SchNet：连续滤波卷积、径向基、截断半径和原子求和读出。`ADAPTED`
- [x] A04 PaiNN：标量/向量特征、等变消息传递和原子求和读出。`ADAPTED`
- [x] A05 DimeNet++：邻居、三元组、角度、径向/球基和输出块。`ADAPTED`
- [x] A06 ESA：QM9 原子/边特征、edge set attention、PMA 与图回归读出。`ADAPTED，已修复`
- [x] A07 TorchMD-Net：Equivariant Transformer、邻居策略、输出头和先验。`ADAPTED`
- [x] A08 Cormorant：SO(3) 表示、Clebsch-Gordan 层、QM9 特征与读出。`ADAPTED`
- [x] A09 LieTransformer：SE(3) lift、等变注意力、掩码、中心化与池化。`ADAPTED`
- [x] A10 ALIGNN：原子图、线图、键长/键角基和 ALIGNN/gated-GCN 更新。`ADAPTED`
- [x] A11 3D Infomax：QM9 内部 2D/3D 对比预训练、PNA 迁移和 gap 微调。`PASS`
- [x] A12 运行十模型结构、形状、有限值和反向传播 sanity tests。`PASS`

### B. 三个 split seed

- [x] B01 核对 seed 3407、42、2026 的配置、比例和分层策略。`PASS`
- [x] B02 核对每套 train/val/test 数量、互斥性、全集覆盖和 manifest 顺序。`PASS`
- [x] B03 核对三套划分使用相同样本全集且成员分配确实不同。`PASS`
- [x] B04 核对三套标签与数据集内容一致，没有因 seed 改变预处理结果。`PASS`
- [x] B05 核对三套 3D 几何的样本身份、顺序、坐标内容和哈希 lineage。`PASS`
- [x] B06 用正式配置对三套划分执行数据加载 dry-run。`PASS`

### C. Pipeline 公平性

- [x] C01 原始数据清洗、canonical SMILES、重复样本和标签单位统一。`PASS`
- [x] C02 标签标准化只拟合训练集，并在验证/测试预测后正确逆变换。`PASS`
- [x] C03 输入特征标准化遵循各模型官方机制，不引入标签或跨 split 统计量。`PASS`
- [x] C04 损失、优化器、学习率、batch size、epoch 和 early stopping 的可比性。`CONDITIONAL`
- [x] C05 所有模型只用验证集选模，测试集只做一次终评。`PASS`
- [x] C06 指标、单位、预测文件和样本身份一致。`PASS`
- [x] C07 3D Infomax 每个 split 独立预训练，不接触对应验证/测试样本。`PASS，已加门禁`
- [x] C08 training seed 与 split seed 独立记录，输出目录不得冲突。`PASS`
- [x] C09 汇总阻塞问题、允许的模型特有差异和正式服务器训练协议。`PASS`

## 官方源码对照

`configs/qm9_models/sources.lock.json` 固定了所有主仓库及 LieTransformer 依赖的 LieConv commit。
`.external/qm9_models/` 中的 checkout 均处于对应 commit，且工作树干净。实现定位如下：

| 模型 | 关键思想核验 | 与官方实现的关系 | 结论 |
|---|---|---|---|
| EGNN | 官方电荷幂特征、全连接 masked graph、E(n) 消息传递和图读出 | 直接调用官方 `qm9.models.EGNN`；容量和优化配方有调整 | ADAPTED |
| SchNet | Gaussian RBF、连续滤波卷积、cutoff、原子求和 | vendored SchNetPack 核心仅改导入路径；训练循环统一改写 | ADAPTED |
| PaiNN | 标量/向量通道、等变交互、原子求和 | vendored SchNetPack 核心仅改导入路径；训练循环统一改写 | ADAPTED |
| DimeNet++ | radius graph、三元组、角度、Bessel/球基与 output blocks | 参数结构对齐官方 `config_pp.yaml`，用纯 Torch/PyG 路径避开旧索引和编译扩展 | ADAPTED |
| ESA | edge tokens、M/S blocks、PMA 和回归头 | 已恢复官方 QM9 的 35 维原子、13 维键、83 维 edge 编码及主要网络默认值 | ADAPTED |
| TorchMD-Net | 官方 Equivariant Transformer 与 scalar output head | 直接调用固定版本官方构造器；容量、邻居参数和无 atomref 设置面向 gap 调整 | ADAPTED |
| Cormorant | SO(3) 表示和 Clebsch-Gordan 层 | 直接调用官方 `CormorantQM9`；channel/level 配置有调整 | ADAPTED |
| LieTransformer | SE(3) lift、等变 attention、mask 和 mean pooling | 使用官方 LieTransformer/LieConv；hidden size 与 lift samples 有调整 | ADAPTED |
| ALIGNN | 原子图、line graph、键长和键角更新 | 直接调用官方 ALIGNN；把晶体邻居构图适配为非周期分子 | ADAPTED |
| 3D Infomax | 同一分子的 2D PNA/3D Net3D 对比学习，再迁移 PNA | 预训练和迁移结构、主要配置与官方一致；仅使用对应 QM9 train split | PASS |

这里的 `ADAPTED` 不表示实现错误。它表示模型关键机制保留，但不能用当前配置直接宣称“精确复现
论文数值”。尤其是 TorchMD-Net、Cormorant、LieTransformer 和 EGNN 的容量或训练配方与官方
示例不同。DimeNet++ 的旧数据偏移问题由基于 `sample_id` 的官方 XYZ 对齐规避。

本轮发现并修复了两项实现问题：ESA 原先使用简化的 26 维 edge 特征，现在与官方 QM9 编码
一致；EGNN 官方仓库与 3D Infomax 都暴露顶层 `models` 包，现已隔离模块缓存，消除同一进程中
的导入冲突。

## 三套数据划分

可重复审查命令：

```bash
uv run python scripts/audit_qm9_split_seeds.py
```

机器可读结果位于 `reports/qm9_split_seed_audit.json`。三套划分均使用
`quantile_stratified`、200 bins，行数均为 107,038 / 13,380 / 13,380，总计 133,798；每套内部
互斥且完整覆盖同一数据集。相对 seed 3407，seed 42 有 45,641 条样本改变 split，seed 2026
有 45,559 条改变 split，因此不是重复划分。

| split seed | manifest SHA-256 | geometry SHA-256 |
|---:|---|---|
| 3407 | `c23bfbb887e8fb7014353cea2154a623444defe3b356cee60350079b221ccd43` | `b9ca7d881ec66a554fe747b376bf0229f2b34ba7d39f16bb2487fe266da29b19` |
| 42 | `cd291da6fcde7055870b30aa31a30f281ec92792881b57ebbee83c92ffcaf6cf` | `0f2d284e59242ae71493f15aea876c444a7e8b0baa607c8b2e9380caf0e6b210` |
| 2026 | `814583d9523c67427e3725cb7d991c8d47cd3a4f9280eaa39e7df1b598218c3c` | `442448d38c3606cc0fe4d5f305679d738c97db4fd75fec18b5a527d0599bc076` |

每套 geometry 均通过样本身份、顺序、原子序数和坐标逐分子核对。十个模型在三个完整 split 上
共完成 30 次正式配置 dry-run；3D Infomax 对 seed 42/2026 使用各自的 smoke 预训练 checkpoint
验证路由，不代表完成正式预训练。

## 公平性结论

当前 pipeline 可以用于统一数据和统一评估协议下的模型比较：

- 原始标签均为 Hartree，清洗、canonical SMILES、重复样本处理和 split 生成完全共享。
- 十个 QM9 adapter 均只从 train 拟合 `delta_e` 的 mean/MAD，训练时变换标签，验证选模和测试
  指标在逆变换后的 Hartree 空间计算。scaler 会写入 checkpoint 和 `target_scaler.json`。
- Chemprop 与 SMILES Transformer 同样只用 train 拟合 Z-score。树模型保持原始 RDKit 特征和
  原始标签；这是配置中明确记录的模型特有处理，不能与神经网络的 scaler 混淆。
- 坐标不做跨数据集标准化；距离/RBF、类别 embedding、固定元素属性和模型内部 BatchNorm/
  LayerNorm 都属于各架构自身机制，不读取验证或测试统计量。
- 十个 QM9 模型的最佳 checkpoint 只按 validation MAE 选择，加载后才生成 test 预测与指标。
  Chemprop现已接通原始Hartree的gap MAE，跨方法选模策略统一；loss/scaler保留方法配方。
  training seed 和 split seed 分开写入配置与数据哈希；三组正式运行按 seed 子目录隔离。
- 3D Infomax checkpoint 现在强制匹配当前 split manifest 与 geometry 哈希，跨 split 权重会立即
  报错，因此预训练不会把另一个划分的验证/测试分子带入训练。

C04 为有条件通过：十模型当前共享 MAE、100 epoch 上限、batch 256、patience 20、
cosine scheduler、validation MAE 选模和 train-only mean/MAD，但 optimizer、学习率、
weight decay 和网络容量按模型不同。
这对“适配实现比较”是合理的，对“精确论文复现”或“完全相同训练预算”还不够。

正式服务器实验采用以下冻结协议：

1. 把当前结果命名为 **controlled adapted benchmark**，固定现有 YAML、三套 split 和评估代码；
   batch 256、100 epochs、patience 20；显存不足时记录 OOM 并复审，不能自动降低 batch，
   禁止查看 test 后调参。此前允许模型特有 batch 的建议被此协议替代。
2. 每个 split 单独完成 3D Infomax 预训练，再运行对应下游任务；checkpoint 哈希检查不得关闭。
3. 先完成 split seed 3407 / training seed 3407 的预检，再扩展到三个 split；若比较训练随机性，
   当前每个 split 只运行同编号 training seed，不运行五 seed 或交叉组合。
4. 每个模型保存配置快照、上游 commit、split/geometry SHA-256、scaler、最佳 validation MAE、
   test 预测和运行环境。测试集只在配置冻结后终评一次。
5. 若另做官方配方复现，建立独立的 `official-recipe` 结果表，逐模型对齐官方容量和优化设置，
   不与本表混合。

## 验证记录

```text
adapter suite: 19 passed
split audit: PASS (3/3 seeds)
full-data dry-run: PASS (10 models x 3 split seeds)
3D Infomax wrong-split checkpoint: correctly rejected
```

本审查证明代码结构、数据隔离和启动链路适合迁移到服务器；它不替代完整训练，也不证明论文
指标已经复现。
