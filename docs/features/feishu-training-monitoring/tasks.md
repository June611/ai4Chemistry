# 实施与验收

- [x] 确认 WSL2、项目训练入口、本地指标/耗时格式及已有资源采样方案。
- [x] 用户确认国内飞书，并确认尚未创建应用和表格。
- [x] 公开域名连通性：open.feishu.cn DNS 成功，公开 HTTPS 返回 200。
- [x] 完成 schema/check/init-table/run/watch/sync 命令和本地运行记录。
- [x] 接入 Linux 进程树 RSS/PSS、nvidia-smi、当前/峰值资源、流水线和 fit 耗时。
- [x] 实现后台同步、token 刷新、记录更新、UUID 幂等键和补传。
- [x] 补齐新测试结果的 MSE，兼容历史平铺/嵌套指标，并拒绝混用物理单位。
- [x] 添加 Transformer/SPE CSV 训练日志及中文配置说明、凭证模板和忽略规则。
- [x] 初次接入相关回归共 61 项通过，覆盖 tracking、metrics、evaluate、Transformer、
  SPE multitask、Phase3 CPU 训练、ML 和结果汇总；随后对读取边界和响应丢失重试
  再运行 tracking/metrics/evaluate，25 项通过。当时未执行全套；恢复后的141项见下文。
- [x] `.venv/bin/ruff check .` 与 `git diff --check` 通过；
  `uv run --locked --no-sync python -m molgap.tracking schema` 可运行，当时输出43个字段。
- [x] CLI 实测：`run --local-only` 启动合成产物生成脚本（非模型训练），
  得到 completed 状态、RSS/PSS、11 次采样统计、约 2.2 秒流水线时间、
  epoch/loss/验证 MAE 与 MSE/MAE/RMSE/R²；`sync --dry-run` 可读回同一运行。
  初始化阶段目录尚未创建的样本只计入 state 的统计/峰值，创建后开始写 samples.jsonl。
  本地证据：`artifacts/feishu-monitor-smoke/verification.json`（Git 忽略的本机产物）。
- [x] 读取本轮真实 Phase3 CPU 测试产物，当前 epoch 与 fit 完成 epoch 均为 1，
  后续 test 日志不误计额外 epoch。模拟验证创建响应丢失后，持久化 UUID 重用、
  重试不新增重复行并更新为最新状态。
- [x] 进一步检查 WSL，找到官方 lark-cli 1.0.97、已配置应用及加密用户登录；
  `auth status --json --verify` 通过，实际调用使用 user 身份。
- [x] 真实账号测试：创建独立测试 Base 和 7 字段数据表，新增/修改模拟指标记录，按 ID 读回一致。
- [x] 真实账号测试：创建云文档、追加正文、替换文字并读回；最终 revision=4。
  链接和证据见 [WSL 真实读写验收](wsl-cli-verification.md)。
- [x] 采用本地Lark CLI用户登录同步桥；服务器仅训练和采样，无需复制App ID/Secret。
- [x] 创建真实训练记录表与硬件采样表，CLI schema预检和实际读写通过。
- [x] 集成持久化资源队列，8项服务器诊断完成；真实显存、PSS/RSS及推理计时同步通过。
- [x] 重新执行已完成诊断的一键入口，训练跳过，飞书同步8项、0错误。
- [x] 回读8条completed、7份13380样本预测附件、8条完整精度JSON；训练表60字段。
- [x] 修正CLI精度限制：指标/LR列precision=4，完整数值另存JSON/CSV。
- [x] 恢复后本地全套141项、服务器调度/监测/指标32项测试通过；监测容错与停止终态一致通过。
- [ ] 正式100epoch、三组训练与结果完整性验收；不把诊断误差作为正式性能。

初次WSL显存读取受操作系统限制；服务器GPU读取现已验收。
当前证据见[资源调度最终审查](../qm9-resource-aware-training/final_review.md)。
