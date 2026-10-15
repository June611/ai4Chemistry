# Transformer-M seed3407跨机器复现

用户要求在新增“3080”机器只训练一个seed，用于检查相同模型/配置在不同机器上的训练与预测差异。用户随后明确提供了第二台服务器地址；实际nvidia-smi返回 **NVIDIA GeForce RTX3090**，GPU空闲。因此本实验必须标为 **RTX3090 ↔ RTX4090**，不能写作3080结果。

只复现Transformer-M、split3407及初始化seed3407。保留双4090原正式队列中的Transformer-M3407，不替换、不取消；目标机器的结果是独立cross_gpu_reproduction，不计作第四个正式seed，也不重复纳入三seed平均。

使用4090真实冻结任务配置，而非重新从base配置推测：12层/768维/32头/128核、batch256、最多100epoch、patience20、MAD+MAE、LR7e-5、Adam、cosine、clip5、workers2，以及相同代码、OGB特征、重原子选择和原单位逆变换。仅改变输出位置、实验名称和复现元数据。

准备时对比关键Python包、模型/训练源码hash、五份split文件hash、几何/拓扑hash及相同seed的CPU初始化state_dict SHA256。CPU身份探针不占4090显存，不中断原训练。目标卡先做独立一次真实batch256前向/反向探针，探针更新丢弃，正式训练从相同seed重新初始化。

输出单独放入新目录的artifacts/reproductions，沿用既有单任务队列每5秒采样GPU显存、主机PSS/RSS，保存history、Hartree预测、MAE/RMSE、最佳验证epoch与耗时。

两边100epoch上限实验都结束后：对比MAE/RMSE、每轮验证轨迹、最佳epoch、逐sample_id预测差值和权重差异。固定checkpoint在两种GPU交叉推理，可进一步区分训练路径与推理数值差异。单seed只支持这个案例，不能据此证明所有硬件等价；驱动/OS差异照实记录，不把所有差异仅归因GPU。

状态：已部署、核验并启动。目标卡最新已完成5/100轮，仍在继续训练；当前4090参考任务仍在排队，暂时没有可比较的两机正式结果。

## 实际部署与启动

目标项目：`/root/autodl-tmp/evan/ai4Chemistry_transformer_m_cross_gpu_20261015`。运行代码为`2b0a952`（模型/训练代码自`8380b39`未变），与4090相同。复用目标机已有uv环境与固定数据；原项目未覆盖。

目标队列：`/root/autodl-tmp/evan/ai4Chemistry_transformer_m_cross_gpu_20261015/artifacts/reproductions/transformer_m_cross_gpu_seed3407_20261015_173755`。dispatcher PID `574837`，队列worker PID `574842`（GPU trainer为其子进程）；仅一个正式任务，已运行。

冻结配置：`/root/autodl-tmp/evan/ai4Chemistry_transformer_m_cross_gpu_20261015/artifacts/reproductions/transformer_m_cross_gpu_seed3407_20261015_173755/config.yaml`。输出：`/root/autodl-tmp/evan/ai4Chemistry_transformer_m_cross_gpu_20261015/artifacts/reproductions/transformer_m_cross_gpu_seed3407_20261015_173755/runs/transformer_m_qm9_scratch_cross_gpu_reproduction/seed3407`。experiment_role=cross_gpu_reproduction，不能与正式三seed结果混合。

源代码、关键包、CUDA12.4、精度/环境标志、五个split文件及geometry/topology哈希均匹配。CPU初始state_dict及目标GPU加载后的state_dict SHA256同为`195b7d632b68f3f420fec7168ba6381213aadb0750bc979a1dc93645f7958299`；训练mean/MAD完全一致。

Python补丁版本为4090侧3.11.17、3090侧3.11.15；驱动分别595.71.05与570.124.04。其余核对的关键包版本相同；不是严格只改变GPU的因果实验，这些环境差异保留。

首轮耗时45.48秒；当前采样进程显存峰值1.377GiB、主机PSS峰值2.031GiB，均为进行中采样，不是最终峰值。

```bash
cd /root/autodl-tmp/evan/ai4Chemistry_transformer_m_cross_gpu_20261015
tail -f /root/autodl-tmp/evan/ai4Chemistry_transformer_m_cross_gpu_20261015/artifacts/reproductions/transformer_m_cross_gpu_seed3407_20261015_173755/runtime/transformer_m_seed3407_cross_gpu_reproduction.log
```

WSL证据：`/home/june/deepLear_zone/Chem_proj/chempro/outputs/summaries/transformer_m_cross_gpu_seed3407_20261015_173755_UTC8`，含两机身份、初始化hash、单batch GPU探针、运行快照、进行中资源CSV和4090最新队列进度。

独立探针更新丢弃，正式进程重新初始化。首次探针准备曾因DataLoader提前消耗CPU RNG被初始权重断言拦截，尚未执行更新；调整为与不变的正式trainer相同顺序后通过。准备记录保留，不计为正式失败或额外seed。

尚未完成：目标100epoch上限训练；4090参考3407训练；最终两机MAE/RMSE、逐样本预测/权重对照及必要的固定checkpoint跨GPU推理。当前仅有一组硬件复现，不能先声称两机等价。

最新交接快照：3090复现已完成5轮、当前running且无error；每轮约45.6秒，GPU进程采样峰值约1.379GiB。4090上两项DeMol各完成8轮，四项排队且无失败。当前仍不能给出两机最终测试误差差值。
