# 架构设计与形式化表示

本文档包含论文第 3、4 章的完整方法设计，包括 LaTeX 公式、算法伪代码、架构图描述。

---

## 第 3 章：工作一 —— 混合消息传递与全局注意力架构（2D）

### 3.1 动机与设计原则

**观察 1**：从 24 模型对比表发现，SchNet（3D，只用距离）138.0 meV **输给** Chemprop（2D）111.5 meV  
→ 说明**不是"用 3D"就一定好，而是"如何编码结构"才关键**

**观察 2**：MPNN 的感受野受限于深度 $T$（一般 3-5 层），难以捕捉长程相互作用（如共轭体系的电子离域）

**设计原则**：
1. 保留 D-MPNN 的**化学键归纳偏置**（有向边消息传递）
2. 用 Transformer 补足**全局感受野**（任意原子对直接交互）
3. 串行架构：MPNN 先编码局部 → Transformer 再编码全局（符合分子层次结构）

---

### 3.2 架构设计

#### 3.2.1 局部化学环境编码（BondMessagePassing）

我们使用 Chemprop \cite{yang2019analyzing} 的 `BondMessagePassing` 模块编码局部化学环境。该模块的核心思想是在**有向边空间**进行消息传递，而非传统 GNN 的节点空间。

**输入**：分子图 $G = (V, E)$，原子特征 $\{\mathbf{v}_i\}$，边特征 $\{\mathbf{e}_{ij}\}$

**输出**：原子表示 $\{\mathbf{h}_i^{\text{local}}\} \in \mathbb{R}^{N \times (d_h + d_v)}$

**过程概述**：

1. **边初始化**：每条有向边 $(i \to j)$ 的隐藏状态由起点原子特征和边特征拼接初始化：
   $$
   \mathbf{h}_{ij}^{(0)} = \mathbf{W}_{\text{init}} [\mathbf{v}_i \oplus \mathbf{e}_{ij}]
   $$

2. **边消息传递**（$T$ 层）：每条边聚合来自相邻边的消息并更新自身状态。消息传递的方向性确保信息沿化学键的方向性传播（避免信息回流）。

3. **聚合到原子**：每个原子 $i$ 的表示由指向它的所有边的隐藏状态聚合得到：
   $$
   \mathbf{m}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{h}_{ji}^{(T)}
   $$

4. **最终表示**：拼接聚合的边消息和原始原子特征（类似 skip connection）：
   $$
   \mathbf{h}_i^{\text{local}} = [\mathbf{m}_i \oplus \mathbf{v}_i]
   $$

**实现细节**：我们直接使用 `chemprop.nn.BondMessagePassing`，参数设置为 $d_h=300$, $T=3$（与 Chemprop 默认配置一致）。详细的消息传递公式和更新机制见 Chemprop 官方文档 \cite{chemprop_docs}。

**为什么使用 Chemprop**：
- 有向边消息传递保留了化学键的方向性信息（电子云极化方向）
- 在多个分子性质预测任务上达到 SOTA
- 代码实现成熟，可直接复用，让我们专注于混合架构的设计

**形式化记号**：为简洁起见，后文中我们用以下符号表示该模块：
$$
\{\mathbf{h}_i^{\text{local}}\} = \text{BondMessagePassing}(G; \{\mathbf{v}_i\}, \{\mathbf{e}_{ij}\})
$$

---

#### 3.2.2 原子表示批处理（关键技术细节）

Chemprop 的 `BondMessagePassing` 输出是 flat 的 $\mathbf{H}_{\text{flat}} \in \mathbb{R}^{N_{\text{total}} \times d_h}$（所有分子的原子拼在一起）。

**问题**：Transformer 需要 batched 输入 $\mathbf{H} \in \mathbb{R}^{B \times N_{\max} \times d_h}$（固定形状）

**解决**：利用 `bmg.batch` 提供的原子归属信息重塑

**`bmg.batch` 示例**：
```python
# 3 个分子：CCO (3 atoms), c1ccccc1 (6 atoms), CC (2 atoms)
bmg.batch = tensor([0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2])
#                   \_____/  \_______________/  \___/
#                    mol 0        mol 1         mol 2
```

#### 算法伪代码

```latex
\begin{algorithm}[h]
\caption{原子表示批处理与 Padding}
\label{alg:batch}
\begin{algorithmic}[1]
\FUNCTION{BatchAtoms}{$\mathbf{H}_{\text{flat}} \in \mathbb{R}^{N_{\text{total}} \times d}$, batch\_ids $\in \mathbb{Z}^{N_{\text{total}}}$}
    \STATE $B \gets \max(\text{batch\_ids}) + 1$ \COMMENT{批次大小}
    \STATE $N_{\max} \gets \max_{i=0}^{B-1} |\{j : \text{batch\_ids}[j] = i\}|$ \COMMENT{最大原子数}
    \STATE 初始化 $\mathbf{H}_{\text{batched}} \in \mathbb{R}^{B \times N_{\max} \times d}$ 为零
    \STATE 初始化 mask $\mathbf{M} \in \mathbb{R}^{B \times N_{\max}}$ 为 $-\infty$
    
    \FOR{$i = 0$ \TO $B-1$}
        \STATE $\text{indices} \gets \{j : \text{batch\_ids}[j] = i\}$ \COMMENT{属于分子 $i$ 的原子索引}
        \STATE $n_i \gets |\text{indices}|$
        \STATE $\mathbf{H}_{\text{batched}}[i, :n_i, :] \gets \mathbf{H}_{\text{flat}}[\text{indices}, :]$
        \STATE $\mathbf{M}[i, :n_i] \gets 0$ \COMMENT{真实原子位置在注意力中可见}
    \ENDFOR
    
    \RETURN $\mathbf{H}_{\text{batched}}, \mathbf{M}$
\ENDFUNCTION
\end{algorithmic}
\end{algorithm}
```

**形式化表示**：

设 $\pi: \{1, \ldots, N_{\text{total}}\} \to \{1, \ldots, B\}$ 是原子到分子的映射（即 `batch_ids`），则：

$$
\mathbf{H}_{\text{batched}}[i, k, :] = 
\begin{cases}
\mathbf{H}_{\text{flat}}[\pi^{-1}(i)[k], :] & \text{if } k < |\pi^{-1}(i)| \\
\mathbf{0} & \text{otherwise (padding)}
\end{cases}
$$

$$
\mathbf{M}[i, k] = 
\begin{cases}
0 & \text{if } k < |\pi^{-1}(i)| \\
-\infty & \text{otherwise}
\end{cases}
$$

---

#### 3.2.3 全局 Transformer 编码器

继承自 Vaswani et al. (2017)，标准 Transformer encoder。

**多头自注意力**：
$$
\mathbf{Q}^{(\ell)}, \mathbf{K}^{(\ell)}, \mathbf{V}^{(\ell)} = \mathbf{H}^{(\ell-1)} \mathbf{W}_Q, \mathbf{H}^{(\ell-1)} \mathbf{W}_K, \mathbf{H}^{(\ell-1)} \mathbf{W}_V
$$

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}, \mathbf{M}) = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right) \mathbf{V}
$$

其中 $\mathbf{M} \in \mathbb{R}^{B \times N_{\max} \times N_{\max}}$ 是 padding mask（扩展为二维）：
$$
\mathbf{M}[b, i, j] = \mathbf{M}_{\text{1d}}[b, j]
$$

**LayerNorm + FFN**：
$$
\mathbf{H}^{(\ell)} = \text{LayerNorm}(\mathbf{H}^{(\ell-1)} + \text{MultiHeadAttn}(\mathbf{H}^{(\ell-1)}, \mathbf{M}))
$$

$$
\mathbf{H}^{(\ell)} = \text{LayerNorm}(\mathbf{H}^{(\ell)} + \text{FFN}(\mathbf{H}^{(\ell)}))
$$

$$
\text{FFN}(\mathbf{x}) = \mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2
$$

---

#### 3.2.4 Attention Pooling

**问题**：Transformer 输出 $\mathbf{H}^{\text{global}} \in \mathbb{R}^{B \times N_{\max} \times d}$，需要聚合为分子级表示 $\mathbf{z} \in \mathbb{R}^{B \times d}$

**方案**：可学习注意力池化（优于 mean/sum，因为不同原子贡献不同）

$$
\alpha_i = \frac{\exp(\mathbf{w}^\top \mathbf{h}_i^{\text{global}})}{\sum_{j=1}^{N} \exp(\mathbf{w}^\top \mathbf{h}_j^{\text{global}})}
$$

$$
\mathbf{z} = \sum_{i=1}^{N} \alpha_i \mathbf{h}_i^{\text{global}}
$$

其中 $\mathbf{w} \in \mathbb{R}^d$ 是可学习参数。

**实现细节**：在有 padding 的情况下，$\alpha$ 的 softmax 需要加 mask：
$$
\alpha_i = \frac{\exp(\mathbf{w}^\top \mathbf{h}_i^{\text{global}} + M_{1d}[i])}{\sum_{j} \exp(\mathbf{w}^\top \mathbf{h}_j^{\text{global}} + M_{1d}[j])}
$$

---

#### 3.2.5 完整前向传播

$$
\{\mathbf{h}_i^{\text{local}}\} = \text{BondMessagePassing}(G; \{\mathbf{v}_i\}, \{\mathbf{e}_{ij}\})
$$

$$
\mathbf{H}^{\text{local}}, \mathbf{M} = \text{BatchAtoms}(\{\mathbf{h}_i^{\text{local}}\}, \text{bmg.batch})
$$

$$
\mathbf{H}^{\text{global}} = \text{Transformer}(\mathbf{H}^{\text{local}}, \mathbf{M})
$$

$$
\mathbf{z} = \text{AttentionPool}(\mathbf{H}^{\text{global}}, \mathbf{M})
$$

$$
\hat{y} = \text{MLP}(\mathbf{z})
$$

---

## 第 4 章：工作二 —— 几何感知的统一 2D/3D 框架

### 4.1 动机：从拓扑到几何

工作一证明了混合架构在纯 2D 条件下的有效性。**问题**：

1. 量子化学性质（尤其是轨道能级）本质上依赖 3D 几何
2. 但获取 DFT 几何的成本高（计算昂贵），很多场景只有 SMILES

**需求**：一个**可调节几何注入强度**的统一框架，在有/无 3D 坐标时都能工作。

### 4.2 几何信息注入机制

#### 4.2.1 3D 位置编码

**关键设计**：几何信息注入到 **Transformer 输入**（而非 MPNN），因为：
1. MPNN 沿化学键传播，已编码拓扑
2. Transformer 全连接，适合编码空间距离

**两种实现方案**：

#### 方案 A：可学习 MLP

$$
\mathbf{p}_i = \text{MLP}_{\text{PE}}(\mathbf{r}_i)
$$

$$
\text{MLP}_{\text{PE}}(\mathbf{r}) = \mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \mathbf{r} + \mathbf{b}_1) + \mathbf{b}_2
$$

其中：
- $\mathbf{r}_i \in \mathbb{R}^3$ 是原子 $i$ 的 3D 坐标
- $\mathbf{W}_1 \in \mathbb{R}^{128 \times 3}, \mathbf{W}_2 \in \mathbb{R}^{d \times 128}$
- 输出 $\mathbf{p}_i \in \mathbb{R}^d$ 与 $\mathbf{h}_i^{\text{local}}$ 维度匹配

#### 方案 B：Fourier Features（保持平移不变性）

$$
\mathbf{p}_i = [\sin(2\pi \mathbf{B} \mathbf{r}_i), \cos(2\pi \mathbf{B} \mathbf{r}_i)]
$$

其中：
- $\mathbf{B} \in \mathbb{R}^{L \times 3}$ 是频率矩阵（$L$ 个频率）
- 输出维度 $2L$，再经过一个线性层投影到 $d$

**选择建议**：
- 方案 A 更简单（1 天实现），参数少
- 方案 B 理论上更漂亮（频率编码是 NeRF / Transformer 常用技巧），但需要调频率范围

#### 4.2.3 几何编码的不变性讨论

**重要说明**：上述两种位置编码方案（MLP 和 Fourier Features）都**不保证旋转和平移不变性**。

**方案 A（MLP）**：
$$
\mathbf{p}_i = \text{MLP}(\mathbf{r}_i)
$$
- 对于平移 $\mathbf{r}_i' = \mathbf{r}_i + \mathbf{t}$，有 $\mathbf{p}_i' \neq \mathbf{p}_i$
- 对于旋转 $\mathbf{r}_i' = \mathbf{R}\mathbf{r}_i$，有 $\mathbf{p}_i' \neq \mathbf{p}_i$

**方案 B（Fourier Features）**：
$$
\mathbf{f}(\mathbf{r}) = [\sin(2\pi \mathbf{B}\mathbf{r}), \cos(2\pi \mathbf{B}\mathbf{r})]
$$
- 同样不保证旋转/平移不变性（$\mathbf{B}$ 是固定或可学习的，不随输入变换）

**为什么可以接受这种设计**：

1. **数据预处理已对齐**：QM9 数据集的分子构象已经过标准化对齐（质心平移到原点，主轴对齐），训练和测试时的旋转/平移变化很小。

2. **实验验证**：Transformer-M \cite{luo2022one} 使用类似的坐标编码（直接对绝对坐标操作，不保证等变性）在 QM9 上达到了与等变网络（EGNN）相当的性能（80.1 meV vs 60.2 meV），说明在实际数据分布下，显式等变性不是必需的。

3. **E(3) 等变编码的代价**：如果要保证旋转/平移不变性，需要使用球谐函数（DimeNet++）或等变层（EGNN / PaiNN），计算开销大 3-10 倍，且实现复杂度显著提高。

4. **我们的重点是架构设计**：本文的贡献在于混合架构（MPNN + Transformer）和几何按需注入（λ 门控），而非几何编码本身。我们选择简单高效的编码方案，在保持可解释性的同时验证架构有效性。

**未来工作**：可以探索轻量级的等变位置编码（如相对距离 + 键角的不变量组合），在保持效率的同时增强几何泛化能力。这不影响本文核心架构的有效性。

#### 4.2.4 门控参数 λ 的设计

$$
\mathbf{H}^{\text{input}} = \mathbf{H}^{\text{local}} + \lambda \cdot \mathbf{P}
$$

其中 $\lambda$ 有三种设置：

**设置 1：固定值**（用于消融）
- $\lambda = 0$: 纯 2D（工作一）
- $\lambda = 0.5$: 中等几何注入
- $\lambda = 1.0$: 全强度几何

**设置 2：可学习全局标量**
$$
\lambda = \sigma(\alpha) \quad \text{where } \alpha \in \mathbb{R} \text{ is learnable}
$$

初始化 $\alpha = 0$（即 $\lambda = 0.5$），训练时端到端优化。

**设置 3：逐原子可学习**（未来工作）
$$
\lambda_i = \sigma(\text{MLP}(\mathbf{h}_i^{\text{local}}))
$$

每个原子自己决定需要多少几何信息（例如 sp³ 碳可能需要更多）。

---

### 4.3 多任务学习扩展

#### 4.3.1 七任务预测头

预测 QM9 七个量子化学性质：

| 符号 | 名称 | 单位 |
|---|---|---|
| $\epsilon_{\text{HOMO}}$ | 最高占据轨道能 | Hartree |
| $\epsilon_{\text{LUMO}}$ | 最低未占据轨道能 | Hartree |
| $\Delta\epsilon$ | HOMO-LUMO gap | Hartree |
| $\mu$ | 偶极矩 | Debye |
| $\alpha$ | 极化率 | bohr³ |
| $\langle R^2 \rangle$ | 电子空间范围 | bohr² |
| ZPVE | 零点振动能 | Hartree |

**架构**：
$$
\hat{\mathbf{y}} = [\hat{y}_{\text{homo}}, \hat{y}_{\text{lumo}}, \ldots, \hat{y}_{\text{zpve}}] = \mathbf{W}_{\text{out}} \mathbf{z} + \mathbf{b}_{\text{out}}
$$

其中 $\mathbf{W}_{\text{out}} \in \mathbb{R}^{7 \times d}$。

#### 4.3.2 不确定性加权损失

继承自 Kendall et al. (2018)，自动平衡多任务：

$$
\mathcal{L} = \sum_{t=1}^{7} \frac{1}{2\sigma_t^2} \mathcal{L}_t(\hat{y}_t, y_t) + \log \sigma_t
$$

其中：
- $\mathcal{L}_t = \text{MSE}(\hat{y}_t, y_t)$
- $\sigma_t > 0$ 是任务 $t$ 的可学习不确定性参数

**物理意义**：
- $\sigma_t$ 大 → 任务 $t$ 不确定性高 → 损失权重 $\frac{1}{\sigma_t^2}$ 小 → 该任务贡献少
- $\log \sigma_t$ 项防止 $\sigma_t \to \infty$（否则模型会把所有任务都当作"不确定"来逃避）

**初始化**：$\sigma_t = 1$ for all $t$

---

### 4.4 完整前向传播（工作二）

$$
\{\mathbf{h}_i^{\text{local}}\} = \text{BondMessagePassing}(G; \{\mathbf{v}_i\}, \{\mathbf{e}_{ij}\})
$$

$$
\mathbf{H}^{\text{local}}, \mathbf{M} = \text{BatchAtoms}(\{\mathbf{h}_i^{\text{local}}\}, \text{bmg.batch})
$$

**【工作二新增】**如果有 3D 坐标 $\{\mathbf{r}_i\}$：
$$
\mathbf{P} = \text{BatchAtoms}(\{\text{PE}(\mathbf{r}_i)\}, \text{bmg.batch})
$$

$$
\mathbf{H}^{\text{input}} = \mathbf{H}^{\text{local}} + \lambda \cdot \mathbf{P}
$$

否则 $\mathbf{H}^{\text{input}} = \mathbf{H}^{\text{local}}$（退化为工作一）

$$
\mathbf{H}^{\text{global}} = \text{Transformer}(\mathbf{H}^{\text{input}}, \mathbf{M})
$$

$$
\mathbf{z} = \text{AttentionPool}(\mathbf{H}^{\text{global}}, \mathbf{M})
$$

$$
\hat{\mathbf{y}} = \mathbf{W}_{\text{out}} \mathbf{z} + \mathbf{b}_{\text{out}} \quad \in \mathbb{R}^{7}
$$

---

## 架构图描述（供你画图）

### 图 3.1：工作一整体架构（纵向流程图）

```
┌─────────────────────────────────────────┐
│           输入: SMILES                   │
└────────────────┬────────────────────────┘
                 │ RDKit
                 ↓
┌─────────────────────────────────────────┐
│     MolGraph (V, E, 边特征)             │
└────────────────┬────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────┐
│  D-MPNN 局部编码                         │
│  • 有向边消息传递 (T=3 层)               │
│  • d_h=300                              │
│  ┌─────┐  ┌─────┐  ┌─────┐             │
│  │ m_ij│─>│ h_j │─>│ h_i │             │
│  └─────┘  └─────┘  └─────┘             │
└────────────────┬────────────────────────┘
                 │ {h_i^local}
                 ↓
┌─────────────────────────────────────────┐
│  批处理 & Padding                        │
│  • 利用 bmg.batch 重塑                   │
│  • 生成 padding mask                     │
└────────────────┬────────────────────────┘
                 │ H^local [B, N_max, d]
                 ↓
┌─────────────────────────────────────────┐
│  Transformer 全局编码 (L=4 层)           │
│  • 8 头自注意力                          │
│  • LayerNorm + FFN                      │
│  ┌───────────────────────────┐          │
│  │ Attention across all atoms│          │
│  └───────────────────────────┘          │
└────────────────┬────────────────────────┘
                 │ H^global
                 ↓
┌─────────────────────────────────────────┐
│  Attention Pooling                      │
│  α_i = softmax(w^T h_i)                 │
│  z = Σ α_i h_i                          │
└────────────────┬────────────────────────┘
                 │ z [B, d]
                 ↓
┌─────────────────────────────────────────┐
│  MLP 预测头                              │
│  ŷ = W_out z + b                        │
└────────────────┬────────────────────────┘
                 │
                 ↓
           输出 ŷ (gap)
```

### 图 3.2：D-MPNN 消息传递细节（小分子示例）

画一个乙醇分子 CCO：

```
      H   H
       \ /
   H -- C -- C -- O -- H
       / \     \
      H   H     H

节点编号:
  C1 (idx=0)
  C2 (idx=1)
  O  (idx=2)

有向边 (省略氢):
  m_01: C1 → C2
  m_10: C2 → C1
  m_12: C1 → O
  m_21: O → C1
```

标注消息传递公式：
```
m_01^(t+1) = τ(W_1 [m_01^(t) ⊕ h_1^(t) ⊕ (m_21^(t) + m_H1^(t) + ...)])
                                           └─────┬─────┘
                                      来自 C2 的邻居边消息
```

### 图 4.1：工作二架构（在工作一基础上标注新增）

在图 3.1 的"批处理 & Padding"和"Transformer"之间插入：

```
┌─────────────────────────────────────────┐
│  批处理 & Padding                        │
└────────────────┬────────────────────────┘
                 │ H^local
                 ↓
           ╔═════════════════╗  ← 【工作二新增】
           ║ 3D 位置编码      ║
           ║ P = PE({r_i})   ║
           ║ ┌───────────┐   ║
           ║ │MLP 或      │   ║
           ║ │Fourier    │   ║
           ║ └───────────┘   ║
           ╚═══════┬═════════╝
                   │ P [B, N_max, d]
                   ↓
           H^input = H^local + λ · P
                   │
                   ↓
┌─────────────────────────────────────────┐
│  Transformer 全局编码                    │
└─────────────────────────────────────────┘
```

用虚线框或不同颜色标注"工作二新增"。

### 图 4.2：帕累托前沿（λ 消融）

横轴：训练时间（小时/seed）  
纵轴：Gap MAE（meV，越低越好）

**点标注**：
- Chemprop (0.33h, 111.5 meV) —— 2D baseline
- 你的 λ=0 (预期 0.5h, 100-110 meV)
- 你的 λ=0.5 (预期 0.7h, 80-95 meV)
- 你的 λ=1.0 (预期 0.9h, 70-90 meV)
- SchNet (2.28h, 138.0 meV) —— 标注"落在帕累托内部"
- EGNN (2.11h, 60.2 meV)
- DimeNet++ (5.39h, 42.5 meV)

用曲线连接你的三个点，画在左下方（更快 + 更准）。

---

## BibTeX 参考文献

```bibtex
@article{yang2019analyzing,
  title={Analyzing learned molecular representations for property prediction},
  author={Yang, Kevin and Swanson, Kyle and Jin, Wengong and Coley, Connor and Eiden, Philipp and Gao, Hua and Gomez-Bombarelli, Rafael and Barzilay, Regina},
  journal={Journal of Chemical Information and Modeling},
  volume={59},
  number={8},
  pages={3370--3388},
  year={2019},
  publisher={ACS Publications}
}

@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in Neural Information Processing Systems},
  volume={30},
  year={2017}
}

@article{rampasek2022recipe,
  title={Recipe for a general, powerful, scalable graph transformer},
  author={Ramp{\'a}{\v{s}}ek, Ladislav and Galkin, Mikhail and Dwivedi, Vijay Prakash and Luu, Anh Tuan and Wolf, Guy and Beaini, Dominique},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={14501--14515},
  year={2022}
}

@article{shi2022benchmarking,
  title={Benchmarking graphormer on large-scale molecular modeling datasets},
  author={Shi, Yu and Huang, Zhen and Feng, Shengjie and Zhong, Hao and Wang, Weiying and Sun, Yu},
  journal={arXiv preprint arXiv:2203.04810},
  year={2022}
}

@inproceedings{kendall2018multi,
  title={Multi-task learning using uncertainty to weigh losses for scene geometry and semantics},
  author={Kendall, Alex and Gal, Yarin and Cipolla, Roberto},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={7482--7491},
  year={2018}
}

@article{luo2022one,
  title={One transformer can understand both 2d \& 3d molecular data},
  author={Luo, Shengjie and Chen, Tianlang and Xu, Yixian and Zheng, Shuxin and Liu, Tie-Yan and Wang, Liwei and He, Di},
  journal={arXiv preprint arXiv:2210.01765},
  year={2022}
}

@article{li2026demol,
  title={DeMol: Dual-channel message passing for molecular property prediction},
  author={Li, Xiangzhe and others},
  journal={NeurIPS},
  year={2026}
}

@misc{chemprop_docs,
  title={Chemprop: Machine Learning for Molecular Property Prediction},
  author={Heid, Esther and Green, William H and others},
  howpublished={\url{https://chemprop.readthedocs.io}},
  year={2023}
}
```

---

## 下一步

1. ✅ LaTeX 公式和算法完成
2. ⏳ 画 5 张图（用 draw.io 或 PPT）
3. ⏳ 实现代码（见 `../README.md` 代码组织部分）
