# 暂停交接：2026-10-12

本文件是暂停时的历史快照。用户随后授权恢复检查、完成实现并提交推送；下文待办与测试数量
不表示当前状态。恢复后的结果见[最终审查](final_review.md)与[TODO](TODO.md)。

用户要求暂时暂停。停止后续修改、部署、实验和云端操作。已启动的一轮诊断和本轮飞书
同步命令均已结束，没有启动100轮正式训练；没有需要继续等待的本轮训练或同步进程。

## 已完成

- 本地实现资源准入、持久化队列、依赖检查、失败隔离、状态查询、停止和 Lark CLI 同步桥。
- 展开默认63项计划：20个启用下游模型三组配对seed，加3组Infomax QM9预训练。
  暂停清单排除Cormorant/LieTransformer。
- 本地全套测试134项通过；服务器调度/监测/指标测试25项通过。
- 服务器`artifacts/auto_training/diagnostic_20261012`的8项一轮诊断全部完成并通过产物检查：
  Infomax预训练、Infomax下游、SchNet、Chemprop单任务、Chemprop物理约束、SMILES
  Transformer、SPE多任务亲和力+一致性、训练均值。神经网络batch256、1epoch；
  Chemprop CLI诊断warmup0。正式预算仍为256/100/20。
- 飞书创建训练记录表与硬件采样表；最终同步8项、0错误，云端回读8项均为completed。
  下游预测CSV通过附件同步，预训练不混入gap测试指标。
- 创建并回读飞书操作说明；写入README与本功能操作文档。

飞书入口：

- [训练记录](https://my.feishu.cn/base/Nckeb46n6anmx9shVAXcnul6nsh?table=tblgtkzmgVWdVrWu)
- [硬件采样](https://my.feishu.cn/base/Nckeb46n6anmx9shVAXcnul6nsh?table=tblADD5ee9TfNKbM)
- [操作说明](https://my.feishu.cn/docx/SLSedXUDao610nxZAH4cqg5onYA)

## 恢复后待完成

1. 最终审查，补充必要的边界验证与验收记录；不把诊断结果视作正式模型效果。
2. 将最后新增的说明、脚本status/stop/sync入口和观测容错修改再次同步服务器。
   当前服务器已通过的部署包与本地最后几处修改尚未完全相同。
3. 飞书指标显示精度修改收到API错误`800010701`，尚未完成；保存的数值和CSV不因此
   改变。恢复后核对CLI/API支持的precision范围再修改，不盲目重试。
4. 更新旧功能文档中“自动同步待接通/调度器未实现”的历史表述，完成最终使用口径审查。
5. 继续校准并发收益；本轮观察到预训练与SchNet并发，但没有证明全局最短时间。
6. 正式100轮、三组训练尚未启动。代码修改尚未提交commit或push。

## 本地与服务器证据

- 服务器项目：`/root/autodl-tmp/evan/ai4Chemistry`。
- 服务器诊断CSV：
  `/root/autodl-tmp/evan/ai4Chemistry/artifacts/auto_training/diagnostic_20261012/runtime/summary.csv`。
- 服务器诊断状态、完整采样、逐任务日志同目录`runtime/`。
- 本地飞书回执：`artifacts/auto_training/feishu/`。
- 本地最终镜像：`artifacts/auto_training/feishu_bridge/diagnostic_20261012/report.json`。
- 本地飞书目标配置：`configs/feishu_training.json`，已加入gitignore，不复制服务器。
- 本轮服务器部署包：`/tmp/qm9-auto-training-server.tar.gz`，不含飞书目标配置与凭据。

首次含飞书目标配置的SCP被自动审批拦截；去除服务器无需的配置后，安全部署包传输成功。
未向服务器复制Lark CLI登录凭据。现有工作树含之前的监测修改，本轮没有重置或丢弃它们。
