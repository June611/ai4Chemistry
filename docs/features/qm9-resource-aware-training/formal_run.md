# 三组QM9正式训练运行记录

状态：2026-10-12用户明确授权启动。服务器队列已启动，ALIGNN seed3407实际进入第1轮；
Random Forest seed3407并行计算RDKit特征。启动观察时2项running、61项pending、0项失败。
该状态是启动快照，不代表当前实时状态或全部实验完成；实时结果以服务器状态和飞书为准。
机器可读启动快照见[formal_launch.json](verification/formal_launch.json)。

## 启动前验收

- 冻结训练源码提交：`6ee081e908d884482090cbbaeabf9c46e7551eb9`。
  已核对188个训练源码、配置和锁文件SHA-256与本地提交一致。
- 三组split seed 3407/42/2026审查通过：每组107038训练、13380验证、13380测试；组内无交叉，
  三组共享同一预处理数据集合，但划分不同；几何内容、哈希和顺序均匹配。
- 63项正式计划：51个下游神经网络任务、3个Infomax预训练任务、9个传统基线。
  所有54项神经网络/预训练均为batch256、100epoch上限、patience20；训练seed与split seed配对。
  早停可能在100轮前结束，不修改既有loss/scaler/LR和辅助监督配置。
- Infomax使用同seed的QM9训练子集预训练（保持既有num_train=50000），对应下游等待正式checkpoint。
  不复用诊断checkpoint，不引入外部预训练数据；Cormorant/LieTransformer不入队。
- Torch2.6.0+cu124与DGL CUDA可用，活跃旧控制器检查为空。GPU24GiB、15核CPU、90GiB内存；
  启动前磁盘可用26.26GiB、高于10GiB保留线，内存可用约73.75GiB、高于20GiB保留线。
- CPU基线按实际training.n_jobs预留：Random Forest/XGBoost各4核、训练均值1核。
  该资源记账修复后的36项专项回归通过；正式单独资源准入检查全部通过。
- 飞书训练表与硬件表CLI登录/字段检查通过；首个正式行回读为running、batch256、100epochs、
  paired seed3407、diagnostic=no、unit=hartree。

ALIGNN实际启动时间：`2026-10-11T20:43:02.741647+00:00`，即新加坡时间2026-10-12 04:43:02。
服务器Git HEAD保留初次部署基线`3c85976`，源码通过部署包同步；正式来源以冻结提交、
188项source_manifest与生成配置为准，不把服务器未更新的Git HEAD误作本次源码版本。

## 持续运行与查看

服务器项目：`/root/autodl-tmp/evan/ai4Chemistry`。
正式队列：`artifacts/auto_training/formal_20261012`。
本地tmux会话：`qm9-training`；服务器通过nohup后台执行，WSL会话持续同步飞书。
服务器每5秒采样，本地同步目标起始周期300秒；保持电脑不休眠、WSL与网络可用。
程序不调用LLM；关闭本次Codex对话不停止队列或同步。

- [飞书训练记录](https://my.feishu.cn/base/Nckeb46n6anmx9shVAXcnul6nsh?table=tblgtkzmgVWdVrWu)
- [飞书硬件快照](https://my.feishu.cn/base/Nckeb46n6anmx9shVAXcnul6nsh?table=tblADD5ee9TfNKbM)
- 本地同步日志：`artifacts/auto_training/formal_launch_20261012/bridge.log`。
- 服务器汇总CSV：`/root/autodl-tmp/evan/ai4Chemistry/artifacts/auto_training/formal_20261012/runtime/summary.csv`。
- 服务器完整状态、日志与资源采样：同目录`runtime/`；单任务产物在`runs/<model>/seed<seed>/`。
- 启动前审查：队列目录`preflight.json`与`source_manifest.json`；三组数据审查在
  `artifacts/auto_training/formal_launch_20261012/split_audit.json`。
  本地审查副本及首个云端回读保存在同名本地`formal_launch_20261012`目录，Git忽略原始运行产物。

在服务器项目目录查看：

```bash
bash scripts/train_all_auto.sh --status --directory artifacts/auto_training/formal_20261012
```

WSL查看实时同步输出：

```bash
tail -f artifacts/auto_training/formal_launch_20261012/bridge.log
```

不要另建目录重复启动同一批实验。重新执行同一目录会恢复原计划并跳过已完成项；已有活跃
控制器会拒绝重复启动。需要停止时按[操作说明](operations.md)停止服务器队列；关闭本地
tmux仅停止本地同步，不停止远端训练。

## 尚未验收

全部63项正式产物、三seed汇总及两Chemprop跨过warmup的实际表现尚未完成。
GPU重任务使用独占，小任务按资源余量共享，CPU传统基线可重叠；采样峰值不保证绝不OOM。
已有ALIGNN与CPU任务重叠的事实，但没有正式串行对照；并发收益、不同组合及总墙钟时间
仍待校准，不能宣称当前安排是全局最快。相关任务保留在[TODO](TODO.md)。
