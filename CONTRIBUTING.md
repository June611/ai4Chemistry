# 参与开发

本仓库研究 QM9 HOMO–LUMO gap（及其他性质）的神经基线、物理约束模块（PINN）与多任务学习模块，
实验协议以 fresh-200（200 epoch / batch 256 / patience 20 / 三配对 seed）为准。

## 环境

```bash
uv sync --locked --extra qm9      # 完整环境（含 DGL / TorchMD-Net 等可选骨干）
uv run --no-sync pytest -q        # 测试
uv run --no-sync ruff check .     # 静态检查
```

- Python 3.11；依赖锁定在 `uv.lock`，不要手工升级 Chemprop / CUDA wheel。
- 数据与产物不入库：`data/`、`artifacts/`、`outputs/` 由 `.gitignore` 忽略。

## 提交流程

1. **改动前**：读 `docs/features/<相关 feature>/README.md` 与 `decisions.md`，确认协议口径。
2. **改代码**：遵循现有分层
   - `src/molgap/modular/`：fresh-200 模块化训练（head、loss、数据、骨干适配）
   - `src/molgap/qm9_models/`：QM9 模型与训练入口
   - `src/molgap/training/`、`src/molgap/experiments/`：通用训练与实验编排
3. **补测试**：新行为加到 `tests/`，跑 `uv run --no-sync pytest -q` 全绿。
4. **写文档**：同步 `docs/features/<feature>/` 下的 README / tasks.md / decisions.md；
   结果以表格回写（含 seed 均值 ± 标准差与配对 delta）。
5. **提交**：Conventional Commits，scope 用功能名，例如
   ```
   feat(qm9): add per-target single-task baselines for dimenet_pp and torchmd_net
   fix(modular): keep route probabilities in float32 during evaluation
   docs(qm9): record extension results and MLoRE verdict
   ```

## 实验纪律

- 一次只跑一个任务/卡；OOM 不静默降 batch，先记录再决策。
- 三配对 seed（42 / 2026 / 3407），报告均值 ± 标准差；选型只用验证集，test 只做终评。
- fresh-200 结果不与历史 100-epoch 结果混表；被取代的路线保留为历史证据。
- 长跑任务用 `scripts/run_extension_chain.py`（幂等：COMPLETED 跳过、`last.ckpt` 断点续训）。
- 产物目录命名：`artifacts/qm9_epoch200/<stage>_<date>/{runs,logs,chain_logs}`。

## 目录速查

| 路径 | 用途 |
|---|---|
| `configs/qm9_epoch200/fresh/<model>/` | fresh-200 可执行配置（按模型分目录） |
| `configs/qm9_epoch200/archive_*/` | 被取代路线的归档配置（不参与运行） |
| `scripts/` | 入口脚本（见 [scripts/README.md](../scripts/README.md)） |
| `docs/features/` | 按功能组织的需求/任务/决策文档（见 [docs/README.md](../docs/README.md)） |
| `reports/` | 审计与结果表格（JSON / XLSX） |
