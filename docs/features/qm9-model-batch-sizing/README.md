# QM9 两模型显存适配与暂停

状态：2026-10-12 用户要求停止重试，暂不运行 Cormorant、LieTransformer。服务器本轮
重试进程树已终止，GPU已释放；原模型结构和 batch256 正式配置保持。

此前用户选择按模型分别设置 batch、其他模型保留256；随后明确暂停两个模型，当前以暂停
决定为准，不自动继续重试。

| 模型 | 已有证据 | 本轮结果 | 当前状态 |
|---|---|---|---|
| Cormorant | batch256 CUDA OOM | batch128/64 OOM；batch32 部分训练后主动停止 | 暂停，未通过完整一轮 |
| LieTransformer | batch256 CUDA OOM | 降batch重试未开始 | 暂停 |

RTX3090为24GiB。原全量烟测采用QM9 seed3407（107038/13380/13380），新重试也保持全量；
不能把部分训练或失败过程的显存峰值当作正式可运行预算。

暂停名单统一保存在 [`paused_models.yaml`](../../../configs/qm9_models/paused_models.yaml)：
默认批量正式训练与烟测跳过两个模型，直接训练/`--smoke`也会在数据加载之前拒绝。
`--dry-run`仍允许核查模型配置。后续用户明确恢复后移除名单项，再验证显存；本次不执行恢复。

其他17个下游神经网络及3D Infomax预训练保持batch256/最多100epochs/patience20，三组seed
仍为3407、42、2026，初始化种子与split配对；传统基线batch不适用。

原batch256证据见[原烟测档案](../qm9-one-epoch-smoke/README.md)。本轮服务器日志在
`artifacts/batch_sizing_20261012/`；小型停止/资源记录保存到本feature的`verification/`，
原日志不覆盖。详细任务见[tasks.md](tasks.md)，决策见[decisions.md](decisions.md)。
