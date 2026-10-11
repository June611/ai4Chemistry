# QM9 20 模型正式训练与资源安排

状态：2026-10-12恢复工作后，调度系统实现、边界测试与一键入口验收已完成。
本地全套141项测试、服务器相关32项测试通过；8项代表性诊断均完成，飞书回读一致，
7份完整测试预测CSV附件可见。随后用户授权启动三组正式训练，63项队列已于2026-10-12启动，
当前正式结果尚未完成验收，见[正式运行记录](formal_run.md)。
最终结论与限制见[最终审查](final_review.md)，机器可读证据见
[final_acceptance.json](verification/final_acceptance.json)。[暂停交接](PAUSED.md)保留为历史记录。
操作入口见[一键使用说明](operations.md)，实施进度见[TODO](TODO.md)。
GPU为RTX3090 24GiB，容器CPU配额15核、内存90GiB；服务器部署从提交3c85976继续，
新增程序通过不含数据、模型权重和飞书登录凭据的部署包同步；本次feature提交作为代码基线。

## 磁盘检查（2026-10-12）

实测系统盘30GiB，overlay可写层使用约498MiB；数据盘`/root/autodl-tmp`50GiB，已用约24GiB，
可用26.39GiB（df四舍五入显示27G）。项目data约500MiB，环境.venv约7.2GiB，官方源码约1.2GiB，
历史artifacts约1.7GiB，uv缓存目录约14GiB。du存在硬链接共享，不能把目录值简单相加，或把缓存
目录大小当作必然可回收空间；实际用量以文件系统可用字节为准。

20个成功下游与Infomax预训练的一组产物合计约0.672GiB；三组按当前体积外推约2.02GiB。
checkpoint策略是最佳模型（Chemprop CLI另存last），不保存100份逐轮权重；因此不会按epoch100
将模型产物体积放大100倍。正式日志、指标和临时空间会增长，建议暂为本轮产物预留5–10GiB，
这只是存储预算估算，不是实测正式训练占用或精确上限。

当前不需要扩容。数据盘空闲低于10GiB时停止准入新任务，先重新核算产物与缓存，再决定清理或
扩容；调度器现已实现磁盘准入监测。本轮不清理缓存、历史模型或数据，不执行扩容。
磁盘证据见[storage_audit.json](verification/storage_audit.json)。

## 两个Chemprop模型的真实状态

首次失败均是参数校验：`epochs=1 <= warmup_epochs=2`，没有因显存不足失败。
一轮诊断将warmup改为0后，两者均完成全量107038/13380/13380数据的训练、验证和测试预测。
单任务GPU峰值0.645GiB，多任务0.652GiB。正式配置仍是100epochs、warmup2、batch256、patience20。
三组3407/42/2026共六个正式命令已在服务器展开检查，epoch大于warmup且配对seed正确。

这是正式配置可启动的证据，不是100epochs和所有划分均可成功的保证。正式启动后先监测前3轮
跨过warmup，再继续同一任务，避免重复执行全量训练。没有预先运行新的3epoch诊断。

`resources.csv`保存首次尝试；`resources_complete.csv`的`status`是最终结果，`initial_status`
和`initial_error`保留历史失败。仅含20个成功下游模型的当前表为
[resources_active_20.csv](verification/resources_active_20.csv)，列`final_status`不混入历史状态。
原证据保留不覆盖。表中`concurrent_measurement=not_run`，不能把单进程峰值当作并发验证。

## 调度计划

正式范围：17个下游神经网络和3个传统基线，每个模型三个配对seed，共60次评估；Infomax额外
三个split内QM9预训练。模型训练参数保持256/100/20，传统模型按一次拟合、不适用该预算。

| 任务组 | 已测单进程GPU峰值GiB | 首选安排 |
|---|---:|---|
| ALIGNN、DimeNet++ | 22.234、21.605 | GPU独占；可按CPU余量穿插传统基线 |
| TorchMD-Net、EGNN | 12.881、9.201 | 先单任务；试配一个Chemprop/序列任务，通过吞吐验证才保留 |
| SchNet、PaiNN、ESA、Infomax下游 | 3.072、4.658、3.113、2.139 | 最多两个起步，实测是否比串行快 |
| Chemprop三模型、SMILES/SPE六模型 | 0.645–1.188 | 最多两个起步，同时限制数据worker及CPU线程 |
| 训练均值、XGBoost、RandomForest | 0 | CPU队列，按剩余CPU/RAM资源与GPU任务重叠 |

单卡GPU没有因显存小而天然空闲：GPU计算利用率、数据等待、每步时间与最终完成时间都需要实测。
保留能减少同一组任务总墙钟时间的并发组合；若单任务已充分使用GPU，或并发拖慢完成时间，回到串行。
不承诺未经测量的并发一定加速，也不承诺全局最短时间。

共享任务的准入暂定为每模型`max(1.2*单进程峰值, 峰值+0.5GiB)`，合计不超过整卡容量减2GiB；
同时查询实际空闲显存。ALIGNN/DimeNet++不套用共享准入，预留整卡；其单独峰值接近上限，
仍需监测形状变化和后续轮次。不得将两者与其他GPU任务同跑；也不将TorchMD-Net与EGNN同时开跑。
该阈值是保守计划，尚未通过并发实验验证，不能消除漏采瞬时峰值的风险。

CPU/RAM一并准入：初始总GPU任务最多2个；为并行任务复制运行配置，将worker暂定为每任务4，
OMP/MKL线程各2，避免两个原8-worker任务就超过15核配额。并发CPU传统任务按剩余配额限制n_jobs，
没有余量时排队。RAM暂留至少20GiB余量，以cgroup实际占用和进程树PSS校准，不能只看宿主机free。
这些worker/线程运行覆盖已在生成配置中实施；架构、batch、LR、loss/scaler、训练数据及seed不改变。

动态队列优先就绪的长任务，单卡独占任务之间再填共享组合。Infomax每个split先完成正式预训练，
核对该split/geometry哈希及seed后才释放对应下游任务；烟测checkpoint不作为正式预训练权重。
运行进程、配置、状态、日志和资源记录已持久化；失败任务单独收尾，避免一个失败终止整个队列。
当前训练入口没有统一自动checkpoint恢复，不能声称中断会无缝续训；完成的任务必须通过产物检查
才在新调度器中跳过，失败重试使用新目录，不覆盖原记录。

现有`scripts/train_qm9_models.sh`串行执行且遇错误退出，`molgap-run-multiseed --jobs`仅限制进程数；
两者都没有本计划要求的资源准入和跨架构调度，不能直接提高`--jobs`后称为已经安全加速。

实现和验收见[tasks.md](tasks.md)，决定见[decisions.md](decisions.md)。
