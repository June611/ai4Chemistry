# SPE + Transformer

更新时间：2026-10-09（文档整理日）。

状态：适配与历史 10-epoch 诊断完成；当前三组配对 split/training seed 正式训练待服务器执行。

## 目标与范围

使用官方 SmilesPE 分词，接入项目 Transformer，研究分词方法对 delta_e 预测的影响。

## 进度与决策

- [详细任务](tasks.md)：此功能进度的权威记录。
- [关键决定](decisions.md)：需求约束及其原因。
- [全项目待办](../../../TODO.md)：跨功能工作入口。

## 验收原则

代码适配、正确性检查、正式训练、结果比较分别验收；每项完成需要配置、测试或报告证据。
历史测试记录只说明当时版本，不代表本轮重新执行了测试。

## 实现与证据

- [训练指南](../../../docs/guides/spe-training.md)
- [分词适配](../../../src/molgap/transformer/spe.py)
- [正式配置](../../../configs/spe_transformer.yaml)
- [短跑报告](../../../reports/spe_transformer/10epoch/README.md)
- [上游核查记录](../../../reports/spe_transformer/upstream_provenance.json)
