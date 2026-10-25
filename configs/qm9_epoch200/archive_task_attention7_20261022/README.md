# 归档：task_attention7（旧 MTL 技术路线，2026-10-22 被 MLoRE 取代）

- 用户决定：MIL/MTL 章节技术路线改为 MLoRE（CVPR'24, arXiv:2403.17749），旧 task-attention 实现已从
  `src/molgap/modular/heads.py`（TaskAttentionHead 删除）与 `train.py`/`config.py`/生成器中移除。
- 本目录保存 6 份旧配置仅作审计；对应已完成实验结果保留在
  `artifacts/qm9_epoch200/formal_4gpu/runs/{dimenet_pp,torchmd_net}_task_attention7_seed*`，
  论文中作为"被取代路线"的历史证据，不与新 mlore7 结果混表。
- 新路线配置：`configs/qm9_epoch200/fresh/{dimenet_pp,torchmd_net}/mlore7_seed*.yaml`。
