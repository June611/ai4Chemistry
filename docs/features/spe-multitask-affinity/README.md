# SPE 多任务亲和力融合

本功能在现有 train-only Smiles Pair Encoding 和 Transformer 编码器上同时预测
`homo`、`lumo`、`delta_e`。物理关系沿用项目 Phase 3：

```text
delta_e = lumo - homo
consistency residual = predicted_delta_e - (predicted_lumo - predicted_homo)
```

三个任务先生成独立表示，再通过逐行 softmax 的可学习 3×3 亲和力矩阵融合。实验保留原有
SPE 单任务基线，并采用 2×2 消融分别打开亲和力融合和物理一致性损失。

详细实施状态见 [TODO.md](TODO.md)，关键选择见 [decisions.md](decisions.md)。

## 消融配置

| 实验 | 亲和力矩阵 | 一致性损失 | 配置 |
|---|---:|---:|---|
| 现有 SPE 单任务 | 不适用 | 不适用 | `configs/spe_transformer.yaml` |
| SPE 三任务 | 关闭 | 0 | `configs/spe_multitask.yaml` |
| SPE 三任务 + affinity | 开启 | 0 | `configs/spe_multitask_affinity.yaml` |
| SPE 三任务 + consistency | 关闭 | 0.1 | `configs/spe_multitask_consistency.yaml` |
| 完整模型 | 开启 | 0.1 | `configs/spe_multitask_affinity_consistency.yaml` |

单次训练：

```bash
uv run molgap-train-spe-multitask \
  --config configs/spe_multitask_affinity_consistency.yaml
```

五 training-seed 完整消融：

```bash
uv run molgap-run-multiseed \
  --config configs/spe_affinity_ablation_multiseed.yaml \
  --jobs 1 --gpu 0
```

另外两套数据划分分别使用
`configs/spe_affinity_ablation_multiseed_seed42.yaml` 和
`configs/spe_affinity_ablation_multiseed_seed2026.yaml`。三个 matrix 配置必须分开运行，
避免把 split seed 和 training seed 混为一类重复实验。

每次运行按验证集直接 `delta_e` MAE 选模。测试产物同时包含三个直接预测、
`pred_lumo - pred_homo` 派生的 gap、两种 gap 的一致性残差、最终亲和力矩阵和三个
train-only scaler。
