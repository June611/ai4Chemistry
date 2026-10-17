# 混合消息传递与 Transformer 架构（MPNN-Transformer Hybrid）

**特性状态**：设计阶段（2026-01）  
**工作类型**：研究生毕业论文双工作（工作一 2D + 工作二 3D）  
**目标**：统一 2D 拓扑和 3D 几何的分子性质预测框架

---

## 研究动机

### 现有方法的问题

从已有 24 模型对比表（`outputs/summaries/qm9_results/tables/`）观察到：

1. **单一架构的局限性**：
   - 纯 MPNN（Chemprop）：111.5 meV，快（20 min）但精度受限于局部感受野
   - 纯 3D 方法（SchNet）：138.0 meV，慢（2.28 h）却**输给 2D**（距离不带角度不够）
   - 高阶 3D（DimeNet++）：42.5 meV，准但极慢（5.39 h）且显存需求大（23.2 GiB）

2. **信息层次的断层**：
   - 2D 方法（Chemprop / SPE Transformer）完全忽略几何
   - 3D 方法重新编码整个分子，丢弃 2D 拓扑的归纳偏置

3. **缺少统一框架**：
   - GraphGPS (2022) 提出 MPNN + Transformer 混合架构，但未用于量子化学性质
   - Transformer-M (2022) 统一 2D/3D，但是纯 Transformer，无 MPNN 的化学键归纳偏置

### 我们的解决方案

**核心思想**：MPNN 擅长局部化学环境（沿键传播），Transformer 擅长全局依赖（长程相互作用），**两者互补**。

**两个工作的关系**：
- **工作一**：MPNN + Transformer 混合架构，纯 2D 拓扑，证明混合优于单一
- **工作二**：同架构，几何信息按需注入 Transformer（λ 门控），实现 2D/3D 统一

**子集关系**：工作二在 λ=0 时严格退化为工作一（架构一致，信息层次递进）

---

## 文献继承链条

### 直接继承的三个核心组件

| 组件 | 来源 | 我们的用法 |
|---|---|---|
| **BondMessagePassing** | Yang et al. (2019), Chemprop 2.x | 有向边消息传递编码局部化学环境 |
| **Transformer** | Vaswani et al. (2017) | 标准 encoder 捕捉全局交互 |
| **混合架构** | Rampášek et al. (2022), GraphGPS | MPNN + Transformer，但我们用串行而非交替层 |

**实现说明**：我们直接使用 Chemprop 2.x 的 `BondMessagePassing` 模块（不做修改），专注于混合架构的设计。该模块在边空间进行消息传递，最后通过 scatter aggregation 聚合为原子表示。

### 工作二额外继承

| 组件 | 来源 | 我们的用法 |
|---|---|---|
| **3D 位置编码** | Shi et al. (2022), Graphormer-3D | 坐标 → 可学习编码，但我们用 PE 而非 attention bias |
| **不确定性加权** | Kendall et al. (2018) | 多任务损失自动平衡 |

### 与同期工作的关系

- **DeMol (2026)**：双通道 Transformer（原子 + 键），键的一等地位。我们的 D-MPNN 也体现键中心思想（有向边消息），但通过 MPNN-Transformer 混合实现，兼容 Chemprop 生态。
- **Transformer-M (2022)**：纯 Transformer 统一 2D/3D。我们的工作二思想相似，但保留 MPNN 的化学键归纳偏置。

**定位**：我们是 GraphGPS 思想在量子化学领域的首次应用 + Graphormer-3D 几何编码的混合架构实现。

---

## 方法概述

### 工作一：2D 混合架构

```
SMILES → RDKit Mol → MolGraph
                         ↓
            ┌────────────────────────┐
            │ BondMessagePassing     │  ← Chemprop 2.x
            │ 局部编码                │
            │ • 有向边消息传递        │
            │ • depth=3, d_h=300     │
            └────────────────────────┘
                         ↓
            原子表示 {h_i^local}
                         ↓
            批处理 + Padding (bmg.batch)
                         ↓
            ┌────────────────────────┐
            │ Transformer 全局编码   │  ← Vaswani (2017)
            │ • L=4 层, 8 头         │
            │ • 自注意力 + FFN        │
            └────────────────────────┘
                         ↓
            {h_i^global}
                         ↓
            Attention Pooling → z
                         ↓
            MLP → ŷ (单任务 gap)
```

**关键技术**：
- `bmg.batch` 提供原子到分子的映射
- Padding mask 确保不同大小分子可批处理
- Attention pooling 自动聚合重要原子

### 工作二：3D 几何注入 + 多任务

在工作一基础上：

1. **3D 位置编码注入**（Transformer 之前）：
   ```
   h^local + λ · MLP_PE(r_i)  或  λ · FourierFeatures(r_i)
   ```
   
2. **多任务扩展**（7 个量子化学性质）：
   - homo, lumo, gap, μ, α, ⟨R²⟩, zpve
   - 不确定性加权损失

3. **λ 门控消融**：
   - λ=0: 纯 2D（工作一）
   - λ=0.5: 中等几何注入
   - λ=1.0: 全强度
   - λ 可学习: 端到端优化

---

## 实验设计

### 工作一：2D 单任务

**目标**：证明混合架构优于单一 MPNN

| 配置 | MPNN depth | Transformer layers | 对标 |
|---|---|---|---|
| `chemprop_baseline` | 3 | 0 | 111.5 meV（已有） |
| `mpnn_only` | 3 | 0 | 验证 MPNN 复现 |
| `mpnn_transformer` | 3 | 4 | **主模型** |

**预期**：`mpnn_transformer` < 111.5 meV（混合优于单一）

### 工作二：3D 多任务

**目标**：量化几何增益 + 多任务收益

| 配置 | λ | 任务数 | 说明 |
|---|---|---|---|
| `hybrid_2d` | 0.0 | 1 | 工作一 |
| `hybrid_3d_lambda0.5` | 0.5 | 7 | 中等几何 |
| `hybrid_3d_lambda1.0` | 1.0 | 7 | 全几何 |
| `hybrid_3d_learned` | 可学习 | 7 | 端到端 |

**核心图表**：
- λ vs MAE vs 时间（帕累托前沿）
- 7 任务相关矩阵热力图
- 与 SchNet / EGNN / DimeNet++ 对比

---

## 创新点（答辩 PPT 用）

1. **首次将 GraphGPS 混合架构用于量子化学性质预测**，结合 Chemprop 局部编码和 Transformer 全局建模
2. **提出可调节几何注入强度的统一 2D/3D 框架**（λ 门控）
3. **两种 3D 位置编码方案**（MLP 和 Fourier Features）的对比实验
4. **多任务不确定性加权在量子化学 7 性质上的应用**
5. **LLM 驱动的智能筛选系统**，在电镀添加剂筛选场景验证方法泛化能力

---

## 与已有代码的关系

**复用**：
- `chemprop.nn.BondMessagePassing`（Chemprop 局部编码模块）
- `torch.nn.TransformerEncoder`（标准 Transformer）
- `multitask7.py`（多任务头和损失）

**新增**：
- `models/mpnn_transformer.py`：工作一核心
- `models/mpnn_transformer_3d.py`：工作二核心
- `nn/batch_utils.py`：原子批处理工具
- `nn/positional_encoding.py`：3D PE 实现

**兼容**：
- `phase3.py` 新增 `coords` 参数，向后兼容原 Chemprop 路径

---

## 下一步

1. ✅ 设计文档完成
2. ⏳ 实现 `mpnn_transformer.py`（工作一）
3. ⏳ 实现 `mpnn_transformer_3d.py`（工作二）
4. ⏳ 实验：工作一 2D 单任务
5. ⏳ 实验：工作二 3D 多任务
6. ⏳ 论文撰写

---

## 参考文献（BibTeX 见 `architecture.md`）

- Yang et al. (2019): Chemprop / D-MPNN
- Vaswani et al. (2017): Transformer
- Rampášek et al. (2022): GraphGPS
- Shi et al. (2022): Graphormer-3D
- Kendall et al. (2018): 不确定性加权
- Li et al. (2026): DeMol（对比参照）
- Luo et al. (2022): Transformer-M（对比参照）
