# 学习路径与模块理解指南

本文档帮助你深入理解每个模块的实现，包括形式化表示、公式推导、实现要点。

---

## 学习路径概览

```
第 1 周：理解基础组件
  Day 1-2: Chemprop BondMessagePassing 机制
  Day 3-4: Transformer 自注意力机制
  Day 5-7: 批处理与 Padding 的技术细节

第 2 周：实现工作一
  Day 1-3: 实现 MPNNTransformerHybrid
  Day 4-5: 数据管道改造（坐标传入）
  Day 6-7: 调试 + 单分子前向测试

第 3 周：实现工作二
  Day 1-2: 3D 位置编码（两种方案）
  Day 3-4: λ 门控 + 多任务头
  Day 5-7: 实验：工作一 baseline

第 4 周：实现工作三
  Day 1-3: LLM Agent 框架
  Day 4-5: 知识库 + 路由策略
  Day 6-7: 端到端测试
```

---

## 模块 1：Chemprop BondMessagePassing（必须深入理解）

### 1.0 重要前置说明

**我们的方法直接使用 Chemprop 2.x 的 `BondMessagePassing` 模块，不做任何修改。**

这一节的目的是帮助你理解 Chemprop 的实现原理，以便：
1. 调试时知道 MPNN 输出的是什么
2. 论文里正确引用 Chemprop
3. 答辩时能回答"为什么不自己实现 D-MPNN"

**我们不是"改进 Chemprop"，而是"在 Chemprop 基础上加 Transformer"。**

我们的贡献是：
- ✅ 混合架构设计（MPNN + Transformer）
- ✅ 3D 几何按需注入（λ 门控）
- ✅ 在电镀添加剂筛选场景的应用验证

不是：
- ❌ 改进消息传递机制
- ❌ 发明新的边编码方式

---

### 1.1 为什么是"有向边"？

**问题**：传统 GNN（如 GCN）是节点到节点的消息传递：
$$
\mathbf{h}_i^{(t+1)} = \sigma \left( \mathbf{W} \sum_{j \in \mathcal{N}(i)} \frac{\mathbf{h}_j^{(t)}}{|\mathcal{N}(i)|} \right)
$$

这在分子图上有什么问题？

**答案**：丢失了**边的方向性信息**。例如：
```
   C1 -- C2 -- C3
```
节点 C2 收到来自 C1 和 C3 的消息，但无法区分：
- "C1 通过 C2 传给 C3 的信息"
- "C3 通过 C2 传给 C1 的信息"

在化学中，这两个方向的相互作用是不同的（电子云极化方向）。

**D-MPNN（Yang et al. 2019）的解决**：把边作为消息载体，每条化学键有两个方向：
```
   m_12: C1 → C2  (正向)
   m_21: C2 → C1  (反向)
```

### 1.2 消息传递公式的直觉理解

**注意**：下面的公式是 D-MPNN 论文（Yang et al. 2019）中的原始版本。Chemprop 2.x 的实现有细节差异（例如使用残差连接而非 GRU），但核心思想一致。我们展示原始公式是为了帮助理解**有向边消息传递**的本质。

$$
\mathbf{m}_{ij}^{(t+1)} = \tau \left( \mathbf{W}_1 \underbrace{[\mathbf{m}_{ij}^{(t)} \oplus \mathbf{h}_j^{(t)} \oplus \sum_{k \neq i} \mathbf{m}_{kj}^{(t)}]}_{\text{三部分信息}} \right)
$$

**三部分信息**：
1. $\mathbf{m}_{ij}^{(t)}$：边 $i \to j$ 自己的历史信息
2. $\mathbf{h}_j^{(t)}$：目标节点 $j$ 的状态
3. $\sum_{k \neq i} \mathbf{m}_{kj}^{(t)}$：从 $j$ 的**其他邻居**传来的消息（排除回传）

**为什么排除 $k=i$？**  
防止信息"往回流"：如果不排除，$m_{ji}$ 会被包含进来，导致 $i \to j \to i$ 的循环。

**Chemprop 2.x 的实际实现**：
- 边更新使用**残差连接** `H_t = tau(H_0 + W_h(M_t))`，而非上述公式中的 GRU
- 没有显式的节点隐藏状态更新循环，所有消息传递在边空间完成
- 最后通过 `scatter_reduce` 聚合到原子

### 1.3 实现要点（Chemprop 代码对应）

**Chemprop 的边表示**：
```python
# chemprop/nn/message_passing/base.py
# 每条化学键存储为两条有向边
edges = []
for bond in mol.GetBonds():
    i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    edges.append((i, j))  # 正向
    edges.append((j, i))  # 反向
```

**邻居边聚合**（对应公式中的 $\sum_{k \neq i}$）：
```python
# 伪代码
def message_passing_step(m_ij, h_j, neighbor_messages):
    # neighbor_messages = [m_kj for k in N(j) if k != i]
    aggregated = torch.sum(neighbor_messages, dim=0)
    concat = torch.cat([m_ij, h_j, aggregated], dim=-1)
    return activation(W @ concat)
```

**你需要读的源码**：
1. `chemprop/nn/message_passing/base.py` 的 `BondMessagePassing.forward()`
2. 找到 `for _ in range(depth):` 循环（消息传递 T 层）
3. 理解 `a_message = self.W_a(...)` 对应公式的哪一部分

---

## 模块 2：Transformer 自注意力（与 NLP Transformer 的区别）

### 2.1 标准 Transformer 回顾

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right) \mathbf{V}
$$

**输入**：$\mathbf{H} \in \mathbb{R}^{B \times N \times d}$（批次 × 序列长度 × 特征维度）

**关键点**：
- $\mathbf{Q} = \mathbf{H}\mathbf{W}_Q$：Query（问）
- $\mathbf{K} = \mathbf{H}\mathbf{W}_K$：Key（被问）
- $\mathbf{V} = \mathbf{H}\mathbf{W}_V$：Value（返回值）

### 2.2 为什么需要 Padding Mask？

在 NLP 中，句子长度不同：
```
Batch 1: "The cat sat on the mat"       (7 tokens)
Batch 2: "Hello"                        (1 token)
```

Padding 后：
```
Batch 1: "The cat sat on the mat" [PAD] [PAD]
Batch 2: "Hello" [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]
```

**问题**：`[PAD]` 不是真实 token，不应该参与注意力计算。

**解决**：在 softmax 之前，把 `[PAD]` 位置的 attention score 设为 $-\infty$：
$$
\text{Attention} = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right) \mathbf{V}
$$

其中：
$$
\mathbf{M}[i, j] = 
\begin{cases}
0 & \text{if token } j \text{ 是真实的} \\
-\infty & \text{if token } j \text{ 是 padding}
\end{cases}
$$

### 2.3 在分子图上的应用

分子 = NLP 句子  
原子 = token  
分子大小不同 → 需要 padding

**关键区别**：
- NLP：token 是离散符号（词表索引）
- 分子：token 是连续向量（来自 D-MPNN 的原子表示）

**Mask 的构造**（见 `architecture.md` 算法 2）：
```python
def create_padding_mask(batch_ids, max_atoms):
    """
    batch_ids: [0,0,0, 1,1,1,1,1,1, 2,2] (总共 11 个原子)
    max_atoms: 6 (最大分子有 6 个原子)
    
    返回:
    mask[0] = [0, 0, 0, -inf, -inf, -inf]  (分子 0 有 3 个原子)
    mask[1] = [0, 0, 0, 0, 0, 0]           (分子 1 有 6 个原子)
    mask[2] = [0, 0, -inf, -inf, -inf, -inf]  (分子 2 有 2 个原子)
    """
    B = batch_ids.max() + 1
    mask = torch.full((B, max_atoms), float('-inf'))
    for i in range(B):
        n_atoms = (batch_ids == i).sum()
        mask[i, :n_atoms] = 0
    return mask
```

### 2.4 实现要点

**PyTorch 实现**：
```python
import torch.nn as nn

encoder_layer = nn.TransformerEncoderLayer(
    d_model=300,
    nhead=8,
    dim_feedforward=512,
    dropout=0.1,
    batch_first=True  # ← 重要！让输入是 [B, N, d] 而非 [N, B, d]
)

encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

# 前向传播
output = encoder(
    src=H_batched,           # [B, N_max, d]
    src_key_padding_mask=mask_1d  # [B, N_max]，True 表示 padding
)
```

**注意**：PyTorch 的 `src_key_padding_mask` 用 **True/False** 而非 $-\infty/0$，且 **True = padding**。

---

## 模块 3：批处理与反批处理（技术细节）

### 3.1 问题的本质

Chemprop 的 `BondMessagePassing` 输出：
```python
H_flat: [11, 300]  # 11 = 3+6+2（三个分子的原子总数）
batch_ids: [0,0,0, 1,1,1,1,1,1, 2,2]
```

Transformer 需要：
```python
H_batched: [3, 6, 300]  # 3 个分子，每个 pad 到 6 个原子
mask: [3, 6]
```

### 3.2 为什么不能简单 reshape？

**错误做法**：
```python
H_batched = H_flat.reshape(B, -1, d)  # ❌ 形状不匹配
```

**原因**：分子大小不同，无法简单切分。

**正确做法**：按 `batch_ids` 分组，然后 pad。

### 3.3 实现（两种方案）

#### 方案 A：循环（简单但慢）

```python
def batch_atoms_loop(H_flat, batch_ids):
    B = batch_ids.max().item() + 1
    max_atoms = max((batch_ids == i).sum() for i in range(B))
    
    H_batched = torch.zeros(B, max_atoms, H_flat.size(-1))
    mask = torch.ones(B, max_atoms, dtype=torch.bool)
    
    for i in range(B):
        indices = (batch_ids == i).nonzero(as_tuple=True)[0]
        n = len(indices)
        H_batched[i, :n] = H_flat[indices]
        mask[i, :n] = False  # 真实原子
    
    return H_batched, mask
```

#### 方案 B：scatter（快，PyG 风格）

```python
from torch_scatter import scatter

def batch_atoms_scatter(H_flat, batch_ids, max_atoms):
    B = batch_ids.max().item() + 1
    d = H_flat.size(-1)
    
    # 为每个原子分配在 batched 张量中的位置
    # 需要计算每个原子在其所属分子内的索引
    atom_idx_in_mol = torch.zeros_like(batch_ids)
    for i in range(B):
        mask = (batch_ids == i)
        atom_idx_in_mol[mask] = torch.arange(mask.sum())
    
    # 计算 flat index
    flat_idx = batch_ids * max_atoms + atom_idx_in_mol
    
    # Scatter
    H_batched_flat = torch.zeros(B * max_atoms, d, device=H_flat.device)
    H_batched_flat[flat_idx] = H_flat
    H_batched = H_batched_flat.reshape(B, max_atoms, d)
    
    # Mask
    mask = torch.ones(B, max_atoms, dtype=torch.bool, device=H_flat.device)
    mask_flat = torch.zeros(B * max_atoms, dtype=torch.bool, device=H_flat.device)
    mask_flat[flat_idx] = False
    mask = mask_flat.reshape(B, max_atoms)
    
    return H_batched, mask
```

**建议**：先用方案 A（易懂易调试），优化时再换方案 B。

---

## 模块 4：3D 位置编码（两种方案详解）

### 4.1 为什么需要位置编码？

Transformer 的自注意力是**排列不变**的：
$$
\text{Attention}(\mathbf{H}) = \text{Attention}(\mathbf{P}\mathbf{H})
$$

对任意排列矩阵 $\mathbf{P}$。

**问题**：分子中原子的空间位置很重要，但 Transformer 看不到。

**解决**：把 3D 坐标编码为向量，加到原子表示上。

### 4.2 方案 A：可学习 MLP

$$
\text{PE}(\mathbf{r}) = \mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \mathbf{r} + \mathbf{b}_1) + \mathbf{b}_2
$$

**实现**：
```python
class MLPPositionalEncoding(nn.Module):
    def __init__(self, d_model=300):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, d_model)
        )
    
    def forward(self, coords):
        """
        coords: [N_total, 3]（flat 坐标）
        返回: [N_total, d_model]
        """
        return self.mlp(coords)
```

**优点**：简单，参数少（约 40K）  
**缺点**：不保证平移不变性（$\text{PE}(\mathbf{r} + \mathbf{t}) \neq \text{PE}(\mathbf{r})$）

### 4.3 方案 B：Fourier Features

$$
\mathbf{f}(\mathbf{r}) = [\sin(2\pi \mathbf{B}\mathbf{r}), \cos(2\pi \mathbf{B}\mathbf{r})]
$$

其中 $\mathbf{B} \in \mathbb{R}^{L \times 3}$ 是频率矩阵。

**两种初始化**：

**B1：固定频率**（类似 Transformer 原文的 sinusoidal PE）
```python
L = 50  # 频率数
freq = torch.exp(torch.linspace(0, np.log(10), L))  # [1, 1.05, ..., 10]
B = freq[:, None] * torch.randn(L, 3)  # [L, 3]
```

**B2：可学习频率**
```python
B = nn.Parameter(torch.randn(L, 3) * 0.1)
```

**实现**：
```python
class FourierPositionalEncoding(nn.Module):
    def __init__(self, num_frequencies=50, d_model=300, learnable=False):
        super().__init__()
        self.num_frequencies = num_frequencies
        
        if learnable:
            self.B = nn.Parameter(torch.randn(num_frequencies, 3) * 0.1)
        else:
            freq = torch.exp(torch.linspace(0, np.log(10), num_frequencies))
            B = freq[:, None] * torch.randn(num_frequencies, 3)
            self.register_buffer('B', B)
        
        # 输出维度 2*num_frequencies，投影到 d_model
        self.proj = nn.Linear(2 * num_frequencies, d_model)
    
    def forward(self, coords):
        """
        coords: [N, 3]
        返回: [N, d_model]
        """
        # [N, 3] @ [3, L] = [N, L]
        x = 2 * np.pi * coords @ self.B.T
        # [N, 2L]
        fourier = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return self.proj(fourier)
```

**优点**：理论优雅，频率编码是 NeRF / Transformer 标准做法  
**缺点**：多一个超参（频率数 L），需要调

### 4.4 选择建议

**第一次实现**：用方案 A（1 天）  
**论文对比实验**：实现方案 B，做消融（方案 A vs 方案 B），各跑 3 seed

---

## 模块 5：不确定性加权（公式推导）

### 5.1 为什么需要加权？

7 个任务，损失量纲不同：
- gap: ~0.01 Hartree²
- μ: ~1 Debye²
- α: ~100 bohr⁶

如果直接求和：
$$
\mathcal{L} = \mathcal{L}_{\text{gap}} + \mathcal{L}_{\mu} + \mathcal{L}_{\alpha}
$$

α 的梯度会主导，gap 几乎不被优化。

### 5.2 Kendall 公式的推导（简化版）

假设模型输出 $\hat{y}_t$ 服从高斯分布：
$$
p(y_t | \hat{y}_t, \sigma_t) = \mathcal{N}(y_t; \hat{y}_t, \sigma_t^2)
$$

负对数似然：
$$
-\log p = \frac{1}{2\sigma_t^2} (y_t - \hat{y}_t)^2 + \log \sigma_t + \text{const}
$$

忽略常数项，多任务损失：
$$
\mathcal{L} = \sum_{t=1}^{T} \left[ \frac{1}{2\sigma_t^2} \mathcal{L}_t + \log \sigma_t \right]
$$

**物理意义**：
- $\sigma_t$ 大 → 任务 $t$ 不确定性高 → 权重 $\frac{1}{\sigma_t^2}$ 小
- $\log \sigma_t$ 项防止 $\sigma_t \to \infty$（否则模型会逃避所有任务）

### 5.3 实现

```python
class UncertaintyWeightedLoss(nn.Module):
    def __init__(self, num_tasks=7):
        super().__init__()
        # log(σ) 而非 σ，确保 σ > 0
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, preds, targets):
        """
        preds, targets: [B, num_tasks]
        """
        mse_per_task = (preds - targets).pow(2).mean(dim=0)  # [num_tasks]
        precision = torch.exp(-self.log_vars)  # 1/σ²
        loss = (precision * mse_per_task + self.log_vars).sum()
        return loss
```

**调试技巧**：在训练时 print `exp(log_vars)`，应该看到不同任务的 σ 差异。

---

## 模块 6：LLM Agent 编排（工作三）

### 6.1 Function Calling 机制

**原理**：LLM 生成 JSON 格式的工具调用请求，你解析并执行。

**示例对话**：

```
用户：帮我预测 CCO 的 HOMO 能级

LLM 输出（OpenAI 格式）：
{
  "function_call": {
    "name": "predict_molecular_properties",
    "arguments": {
      "smiles": "CCO",
      "use_3d": false
    }
  }
}

你执行：
result = predict("CCO", coords=None)
# {'homo': -6.2, 'lumo': -1.8, ...}

你返回给 LLM：
"预测结果：HOMO = -6.2 eV"

LLM 生成自然语言回复：
"CCO (乙醇) 的 HOMO 能级为 -6.2 eV，
 这表明它是一个较弱的电子供体..."
```

### 6.2 实现框架（OpenAI / Anthropic 通用）

```python
import openai

tools = [
    {
        "type": "function",
        "function": {
            "name": "predict_molecular_properties",
            "description": "预测分子的量子化学性质",
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles": {"type": "string"},
                    "use_3d": {"type": "boolean", "default": False}
                },
                "required": ["smiles"]
            }
        }
    }
]

def run_agent(user_message):
    messages = [{"role": "user", "content": user_message}]
    
    while True:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tools
        )
        
        message = response.choices[0].message
        messages.append(message)
        
        # 如果 LLM 要调用工具
        if message.tool_calls:
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                
                # 执行工具
                if func_name == "predict_molecular_properties":
                    result = predict(args["smiles"], args.get("use_3d", False))
                
                # 返回结果给 LLM
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })
        else:
            # LLM 不再调用工具，返回最终回复
            return message.content
```

---

## 下一步执行清单

**现在你应该做的**（按顺序）：

1. ✅ **阅读完这个文档**

2. **画 5 张架构图**（用 PPT / draw.io）：
   - 图 3.1：工作一整体流程（纵向）
   - 图 3.2：D-MPNN 消息传递细节（小分子示例）
   - 图 4.1：工作二架构（在工作一基础上标注）
   - 图 4.2：帕累托前沿（λ 消融）
   - 图 5.1：工作三系统架构（LLM + 工具链）

3. **写 LaTeX 第 2 章文献综述**（参照 `architecture.md` 的 BibTeX）

4. **写 LaTeX 第 3.2 节方法设计**（直接复制 `architecture.md` 的算法伪代码）

5. **通知我**："图画完了，LaTeX 写完了，可以开始写代码"

**我下一步会做**（等你完成 1-5）：

1. 实现 `src/molgap/models/mpnn_transformer.py`（工作一核心）
2. 实现 `src/molgap/nn/batch_utils.py`（批处理工具）
3. 实现 `src/molgap/nn/pooling.py`（attention pooling）
4. 改 `src/molgap/training/phase3.py`（兼容坐标传入）
5. 写测试脚本验证前向传播

---

**预计你完成 1-5 需要 2-3 天，我完成代码框架需要 1 天，然后我们一起调试。**
