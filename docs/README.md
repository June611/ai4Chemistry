# 文档索引

本仓库文档按 **feature（功能）** 组织：每个 feature 一个 `docs/features/<name>/` 目录，内含
`README.md`（需求与结论）、`tasks.md`（子任务与状态）、`decisions.md`（决策与证据）。

## 当前主线

| Feature | 内容 | 状态 |
|---|---|---|
| [qm9-epoch200-modular-study](features/qm9-epoch200-modular-study/README.md) | fresh-200 统一协议：42 项正式矩阵 + 扩展（PINN / MLoRE / DeMol / 逐目标基线）；结果见 [results.md](features/qm9-epoch200-modular-study/results.md) 与 [extension_results.md](features/qm9-epoch200-modular-study/extension_results.md) | 进行中（逐目标 ST 基线 36 项在跑） |
| [qm9-mlore-multitask](features/qm9-mlore-multitask/README.md) | MTL 章技术路线：task_attention7 → MLoRE 的适配映射、超参与判决 | 已定稿（6/6 完成） |

## 历史与支撑

| Feature | 内容 | 状态 |
|---|---|---|
| [paired-seed-100epoch-benchmark](features/paired-seed-100epoch-benchmark/README.md) | 旧 100-epoch 配对协议（batch 256 / patience 20 / 三配对 seed） | 已被 fresh-200 取代（归档） |
| [qm9-ten-models](features/qm9-ten-models/README.md) | 十模型接入 | 历史 |
| [qm9-six-property-neural-baselines](features/qm9-six-property-neural-baselines/README.md) | 六性质单任务扩展（100 epoch） | 历史，未完成 |
| [qm9-resource-aware-training](features/qm9-resource-aware-training/README.md) | 资源感知调度与正式队列 | 历史（被 4-GPU lane 取代） |
| [qm9-demol-transformer-m](features/qm9-demol-transformer-m/README.md) | DeMol / Transformer-M 实现 | 部分并入主线（DeMol + PINN） |
| [model-benchmark](features/model-benchmark/README.md) | 多方法统一比较 | 历史 |
| [traditional-baselines](features/traditional-baselines/README.md) | 传统 ML 基线 | 历史（结果保留） |
| [smiles-transformer](features/smiles-transformer/README.md) / [spe-transformer](features/spe-transformer/README.md) | 序列 Transformer 骨干 | 历史（骨干并入主线） |
| [spe-multitask-affinity](features/spe-multitask-affinity/README.md) | SPE 多任务亲和力 | 历史 |
| [hybrid-mpnn-transformer](features/hybrid-mpnn-transformer/README.md) | MPNN-Transformer 混合 | 历史 |
| [chemprop-pipeline](features/chemprop-pipeline/README.md) | Chemprop 三阶段训练 | 历史 |
| [data-splitting](features/data-splitting/README.md) | 共享数据划分 | 支撑（协议基础） |
| [experiment-infrastructure](features/experiment-infrastructure/README.md) | 调度与结果汇总 | 支撑 |
| [qm9-server-deployment](features/qm9-server-deployment/README.md) | 服务器部署 | 支撑 |
| [qm9-model-batch-sizing](features/qm9-model-batch-sizing/README.md) | 显存适配与暂停记录 | 历史 |
| [qm9-one-epoch-smoke](features/qm9-one-epoch-smoke/README.md) | 一轮烟测 | 历史 |
| [feishu-training-monitoring](features/feishu-training-monitoring/README.md) | 飞书训练监测 | 支撑（可选） |

## 指南

- [environment](guides/environment.md)：Python/uv 环境与依赖
- [server-access](guides/server-access.md)：算力服务器访问
- [data-transfer](guides/data-transfer.md)：数据/产物的迁移与校验
- [ml-training](guides/ml-training.md)：传统 ML 训练入口
- [spe-training](guides/spe-training.md)：SPE 训练入口

## 约定

1. **代码与文档同源**：改动训练入口或协议必须同步 `docs/features/<name>/` 下的 README/tasks/decisions。
2. **产物不入库**：`artifacts/`、`data/`、`outputs/` 由 `.gitignore` 忽略，结果以表格形式回写到 feature 文档。
3. **协议不混表**：fresh-200 与历史 100-epoch 结果永不同表比较；被取代的路线标为"历史证据"。
4. **文档语言**：面向中文读者，命令与字段名保留英文原文。
