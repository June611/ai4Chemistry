# 双4090运行交接：DeMol / Transformer-M

训练代码提交：`8380b39e1d3f127790e2089e651da3b2638b300b`。初始实现`bbc115c`，耗时分配`85cf299`，Transformer-M原点边界修复`8380b39`。本地37项测试与lint通过。

服务器项目：`/root/autodl-tmp/evan/ai4Chemistry_demol_transformer_m_20261015`。独立Git checkout复用原项目data子目录及.venv；运行环境使用`PYTHONPATH=<新项目>/src`，没有覆盖旧项目工作区。

正式队列：`/root/autodl-tmp/evan/ai4Chemistry_demol_transformer_m_20261015/artifacts/auto_training/qm9_demol_transformer_m_20261015_165545`。后台dispatcher PID：259387；SSH或本地Codex退出不终止独立队列。

## 实际验收的完整一轮烟测

均为seed3407、batch256、完整107038/13380/13380，419个训练批次。下列是烟测成本，非正式100epoch成绩。

| 模型 | epoch秒 | 采样进程显存峰值GiB | 主机PSS峰值GiB | 参数量 |
|---|---:|---:|---:|---:|
| demol | 314.32 | 18.447 | 2.807 | 176825914 |
| transformer_m | 43.98 | 1.527 | 2.059 | 47797473 |

显存/主机峰值是每5秒进程树采样；另在CSV保存CUDA allocated/reserved峰值，二者不可混称。完整预测、模型权重、原单位指标和逐行MAE核验均通过。

Transformer-M首次烟测保留在原smoke_gpu1目录。原点单重原子边界修复后，重新完成完整一轮烟测于`..._165545_transformer_m_origin_fix/smoke_gpu1`；manifest的validated_smoke_sources明确指向复验工件。DeMol的adapter/trainer/data/config未随此修复改变，其完整烟测复用原位置；不是重复训练一遍DeMol。

## 正式任务实际分配

| GPU | 顺序 | 当前状态 |
|---|---|---|
| 0 | demol_seed3407 → demol_seed2026 | demol_seed3407: running；demol_seed2026: pending |
| 1 | demol_seed42 → transformer_m_seed3407 → transformer_m_seed42 → transformer_m_seed2026 | demol_seed42: running；transformer_m_seed3407: pending；transformer_m_seed42: pending；transformer_m_seed2026: pending |

两卡已经分别进入DeMol seed3407与seed42的Epoch 1，运行快照时已完成约80/419批。每任务单卡，不用DDP，每卡最多一个；正式seed之间无依赖，某seed失败继续后续任务。

按烟测完整epoch×100外推：GPU0约17.46小时，GPU1约12.40小时；并行总墙钟参考约17.5小时，建议预留17–20小时。该估计不保证收敛、早停或外部负载；不是三个seed合并成一次训练。

## 查看日志

```bash
cd /root/autodl-tmp/evan/ai4Chemistry_demol_transformer_m_20261015
tail -f /root/autodl-tmp/evan/ai4Chemistry_demol_transformer_m_20261015/artifacts/auto_training/qm9_demol_transformer_m_20261015_165545/formal_gpu0/runtime/demol_seed3407.log
# 另一张卡：
tail -f /root/autodl-tmp/evan/ai4Chemistry_demol_transformer_m_20261015/artifacts/auto_training/qm9_demol_transformer_m_20261015_165545/formal_gpu1/runtime/demol_seed42.log
```

实际run在`formal_gpu*/runs/*/seed*/`：history.csv包含epoch/MAE/耗时/allocator显存，metrics.json为Hartree，predictions/test_predictions.csv为原单位预测，model.pt为最佳验证checkpoint。训练尚未结束时不能把缺失metrics当作零误差或已完成。

WSL证据目录：`/home/june/deepLear_zone/Chem_proj/chempro/outputs/summaries/qm9_demol_transformer_m_launch_20261015_165545_UTC8`。smoke_resources.csv保存资源与烟测指标，formal_assignment.csv保存任务分配，launch_snapshot.json保存原始状态和来源，preflight.json保存六项身份/配置校验。

尚未完成：六组正式训练、三seed均值/SD、最终多模型比较、正式训练全程峰值和实际总时间。完成后追加验收，不能用单epoch烟测指标对文献下结论。

补充进度：最新检查原两卡DeMol3407/42各完成8轮，正式仍0/6完成，无失败。另在用户指定的nmb2服务器（实际RTX3090）启动Transformer-M3407跨机器复现，已完成5轮；独立角色与路径见[cross_gpu_seed3407.md](cross_gpu_seed3407.md)，原4090队列未改。
