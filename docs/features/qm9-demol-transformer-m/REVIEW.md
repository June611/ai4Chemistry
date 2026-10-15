# 实现审查

2026-10-15；本文件记录审查范围，后续GPU证据另见operations.md。

- 来源：Transformer-M源码锁定commit并保留MIT；Fairseq兼容层不全局注入模块，未实现的quantization/layerdrop显式拒绝。修复官方QM9返回未定义atom_output；activation/normalization顺序保留官方QM9默认。
- DeMol身份：论文公式重实现，未知头结构/FFN/池化/角度定义选择均公开记录，不把它称为作者官方实现。共价半径预训练辅助目标可用，当前无预训练主配置不暗中启用。
- 输入：SDF按原始ID关联，核对原子与几何顺序，三split坐标逐分子一致性检查；拓扑不读取标签。Transformer-M选择重原子并保留官方OGB偏移，DeMol保留全部几何原子。
- 隔离：沿用manifest/order验证与训练分区scaler，不改既有数据与旧结果。新数据字段仅在显式topology配置中加载，其他适配器继续使用原字段。
- 模型：两个模型通过反向、旋转/平移/原子置换、padding/批次独立性及state_dict回载；完整模型结构没有为了batch256缩成小模型。测试中的32维实例只用于低成本数学/接口检查，正式配置仍768维12层。
- 单位：网络输出标准化gap；统一trainer逆变换到Hartree用于验证/测试。比较换算为meV，不改变训练真值单位。
- 预算：固定256/100/20、配对三seed；新配置的loss/scaler复用现有3D基线MAD+MAE，与原论文MSE/standard区别明示。LR7e-5与clip5有来源，scheduler复用项目cosine。
- 调度：复用既有持久队列与资源记录；每卡最多一个任务；三个正式seed无依赖边，某seed失败继续后续。正式入口检查两个batch256完整一轮烟测工件；一轮419批，不能默认降成batch8。
- 内存：activation checkpointing的输出与梯度对照通过；不预设GPU显存已通过。DeMol dense mask不声称实现论文的稀疏复杂度。
- 版本与交付：仅本功能必要代码、配置、测试及文档进入提交；outputs/data/artifacts/.external权重与大文件被忽略。服务器独立目录保留旧工作区。

验证命令：

```bash
.venv/bin/pytest -q tests/test_demol_transformer_m.py tests/test_qm9_adapters.py \
  tests/test_paused_qm9_models.py tests/test_scheduling.py
```

结果：36 passed。新增及修改的非vendor源码ruff检查通过；git diff --check通过。GPU完整数据烟测和三组正式训练不能由这些CPU测试替代，仍需另验收。
