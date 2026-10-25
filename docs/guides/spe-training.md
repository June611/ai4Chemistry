# SPE + Transformer

本实验使用作者的 `SmilesPE==0.0.3`，接入项目已有的 Transformer encoder 和
delta_e 回归头。作者仓库未包含 Transformer 回归模型；论文使用 AWD-LSTM/MolPMoFiT。
本实验是 SPE 分词方法在当前任务上的受控对比，不是论文完整预训练流程的复现。

## 来源与设置

- 仓库：https://github.com/XinhaoLi74/SmilesPE
- 核查 commit：`e5f27dfea0778966818ac0a9dd23ac646c62707d`；Apache-2.0。
- 已逐文件比对安装包和仓库的 learner、tokenizer、pretokenizer，源码完全相同。
  哈希记录位于 `reports/spe_transformer/upstream_provenance.json`。
- README 链接的 Hugging Face Colab 需要登录，未将其当作已核实的网络实现。
- 与 `configs/transformer.yaml` 共享网络结构、数据配置和所有训练超参数。
  词嵌入参数量随 SPE 词表大小变化，实际参数量记录在每次运行的 `model_summary.json`。
- 合并规则上限 3000、最低频次 100，事先固定，不使用测试集调参。
  这是针对当前 QM9 训练语料的设置；论文的 ChEMBL 语料使用上限 30000、频次 2000。
- 仅从固定划分的 `train.csv` 学习规则及原子词表；没有外部预训练或 SMILES 增广。
  词表包括训练集中出现的原子及所有已学习合并符号，保留中间合并符号以支持新上下文。
- 与原 Transformer 一样保留最多 40 个位置（含 CLS），全字符覆盖校验，禁止静默截断。
- 三组 split/training seed 一一配对：3407、42、2026。
  batch 256、100 epochs、patience 20、AdamW lr 3e-4、32 位精度。

## 运行

```bash
uv sync
uv run molgap-train-transformer --config configs/spe_transformer.yaml --dry-run
uv run molgap-train-transformer --config configs/spe_transformer.yaml
```

运行三组配对实验：

```bash
for config in configs/spe_affinity_ablation_multiseed.yaml \
  configs/spe_affinity_ablation_multiseed_seed42.yaml \
  configs/spe_affinity_ablation_multiseed_seed2026.yaml; do
  uv run molgap-run-multiseed --config "$config" --jobs 1
done
uv run molgap-summarize outputs/paired_gap_mae_v2/spe_transformer \
  --seeds 3407 42 2026 \
  --output-dir outputs/paired_gap_mae_v2/summaries/spe_transformer
```

单次入口与配对矩阵可能共享同一运行目录；已经完成相同 seed 时不要再次覆盖。默认拒绝覆盖已有结果。
可按服务器显存调整 `--jobs`，不必修改数据划分或训练预算。

## 产物与推理

每个 `outputs/paired_gap_mae_v2/spe_transformer/seed<seed>/` 保存：配置、环境、split SHA-256、
`spe_merges.txt`、`vocabulary.json`、分词统计、最佳 checkpoint、`model.pt`、
测试集逐样本预测、Hartree MAE/RMSE、百分比 MAPE、R² 和训练时间。
三组配对结果复用现有汇总器，也可加入统一模型比较命令。

便携模型内含完整合并规则、词表、网络参数及训练集 scaler，推理无需重新学习 SPE：

```python
import torch
from molgap.transformer.model import SmilesTransformerRegressor
from molgap.transformer.tokenizer import SmilesTokenizer

payload = torch.load(
    "outputs/spe_transformer/seed3407/model.pt", map_location="cpu", weights_only=False
)
tokenizer = SmilesTokenizer.from_dict(payload["tokenizer"])
model = SmilesTransformerRegressor(
    len(tokenizer.vocabulary), tokenizer.max_length, payload["model_config"]
)
model.load_state_dict(payload["state_dict"])
model.eval()
encoded = tokenizer.encode("CCO")
with torch.inference_mode():
    normalized = model(torch.tensor([encoded.input_ids]), torch.tensor([encoded.attention_mask]))
    scaler = payload["target_scaler"]
    delta_e = normalized * scaler["scale_hartree"] + scaler["mean_hartree"]
print(delta_e.item())  # Hartree
```
