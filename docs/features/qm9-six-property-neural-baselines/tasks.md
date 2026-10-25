# 实现及验收

2026-10-16：开始实现。当前工作区含历史未提交改动，已保留原始差异快照；仅纳入本功能及必需前置依赖。

验收：180 组唯一配置；训练集统计量隔离；原单位 MAE/RMSE；三组原划分不变；模型权重可加载；失败不阻断其他任务；旧 gap/多任务兼容。

已完成：三条公共训练路径扩展、磁盘清单与模板快照、按卡独立队列、显式重试、按性质汇总和 LaTeX 导出。
分组检查覆盖 75 项不同测试，另在干净 HEAD 加待提交补丁的副本上通过 36 项检查。
审查及 GPU 验收边界见 REVIEW.md；正式任务均未启动。

功能提交 fb883517f2a30604d1e4e8d24449ec8bef0579d0 已推送至 GitHub origin/main，正常快进 5a3857d→fb88351。
该快进也同步了当前 main 已存在的五条本地提交；未改写远端历史。15 份其他历史 tracked 修改及 README 既有内容已逐项确认保留。

三组跨机扩展：独立 groups/*.yaml 和四个 Bash 入口已完成；真实展开 54/72/54 个下游任务，无重复。
20 项新增测试及 32 项旧流程回归通过。结果可在来源机器 verify 后复制并合并至本地，合并后可填写原 LaTeX。
执行及审查见 grouped_queues.md、group_queue_review.md；本次没有启动正式训练。

三组队列功能提交 82326c9 已成功推送，origin/main 正常快进 6e089cc→82326c9。
原有模型代码和其他历史工作区改动未纳入此次提交；本地展开任务均为 pending。

## 2026-10-16：A100 高组工程化与正式启动

- 本机确认 4×NVIDIA A100-SXM4-40GB；高组定义保持 DimeNet++、3D Infomax、
  TorchMD-Net，三组 matched seed、六性质，共54项下游，另有三项 split-specific
  Infomax 预训练。
- 从项目提供的原始 CSV 重新生成三套固定 split，三个 manifest SHA-256 与 README
  基线完全一致；下载并校验133885个官方 XYZ，生成三份 geometry，split/geometry 审计通过。
- 完成不改变模型数学和正式协议的低风险优化：共享 DataLoader runtime、CUDA pinned memory、
  persistent workers、prefetch、non-blocking H2D；GeometryDataset 去除逐样本 pandas/NumPy copy；
  epoch内 loss 设备侧累加；验证/测试批量汇总；DimeNet++/SchNet/PaiNN 半径图批量向量化；
  model factory 改为 lazy registry；新增 formal protocol 执行边界校验。
- DimeNet++ 和 TorchMD-Net 在 A100 上完成真实 CUDA smoke；TorchMD 首次 Warp JIT 约5.4秒，
  之后复用缓存。DimeNet++ batch256 正式首轮观测约3.17 step/s、GPU利用率100%、约18GiB。
  Infomax 预训练约2.5 step/s但 GPU利用率约8–27%，确认主要瓶颈仍是逐SMILES RDKit/DGL构图；
  本轮未冒险改变官方图特征/边顺序，后续应以离线图缓存加精确等价门禁处理。
- 正式冻结目录为 `artifacts/qm9_six_properties/high_formal_a100_v1`，4条GPU lane均已启动：
  三张卡先运行 seed42/2026/3407 Infomax预训练，第四张卡运行 DimeNet++。OOM策略仍为失败，
  不降低batch。状态、日志和5秒资源采样保存在各 lane 的 `runtime/`。
