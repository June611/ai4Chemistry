# 数据准备、训练和恢复

## 1. 环境和数据

沿用项目README的uv环境及QM9原始/三维数据准备流程。保留原三个seed的数据目录和`qm9_single_properties_v1`七标签扩展。额外十标签数据：

```bash
uv run python -m molgap.modular.data --output data/processed/qm9_thermo10_v1
```

输入为`data/raw/qm9_chn.csv`、`data/processed/qm9_full/random/seed{seed}/dataset.csv`及已有七标签划分。输出为`data/processed/qm9_thermo10_v1/random/seed{seed}/{train,val,test}.csv`和身份审计JSON；文件保留原七列，再增加u0/u298/h298。遇到已存在目录会拒绝覆盖，应先核对其身份而非盲目重建。WSL本地已生成这份数据；数据不提交Git，迁移时一并复制或用相同输入重建。

3D模型仍读取`data/geometry/qm9_full_seed{seed}.npz`及其JSON；DeMol/Transformer-M还需要原`data/topology/qm9_full.npz`。两个2D序列模型仍从每个seed的完整训练集拟合原tokenizer，不使用验证/测试学习词表。Chemprop不引入3D。

Infomax配置默认要求`artifacts/qm9_epoch200/pretrained/3dinfomax_seed{seed}.pt`，需复制对应旧QM9训练分区权重或编辑相应配置的source.model.params.pretrain_checkpoint。训练前检查manifest、预训练seed、几何hash及实际文件hash，不允许缺失时静默从头训练。

## 2. 刷新和预检（不训练）

```bash
uv run python scripts/prepare_qm9_epoch200_review.py
uv run python scripts/preflight_qm9_epoch200.py --stage 1 --data-only
uv run python scripts/preflight_qm9_epoch200.py --stage 1
# 后续按骨干预检，不自动开训：
uv run python scripts/preflight_qm9_epoch200.py --stage 2 --model dimenet_pp
```

预检会逐项记录错误并继续检查，最后只要有失败就返回非零，结果在`artifacts/qm9_epoch200/preflight.json`。完整预检包括CPU构造模型/输入；不是GPU显存验收。

## 3. GPU诊断烟测与正式指令

```bash
# 独立输出，保留200轮epoch/LR比例规划，但诊断loader步数更少，只跑一轮/小前缀；不混入正式结果。
CUDA_VISIBLE_DEVICES=0 uv run python -m molgap.modular.train \
  --config configs/qm9_epoch200/fresh/chemprop/task_attention7_seed42.yaml \
  --diagnostic-epochs 1 --limit-rows 256 \
  --output-dir artifacts/qm9_epoch200/gpu_smoke/attention42

# 单个正式任务（确认GPU与容量后执行）：
CUDA_VISIBLE_DEVICES=0 uv run python -m molgap.modular.train \
  --config configs/qm9_epoch200/fresh/chemprop/task_attention7_seed42.yaml

# 第一阶段33项，顺序参考脚本；当前并未执行：
CUDA_VISIBLE_DEVICES=0 bash docs/features/qm9-epoch200-modular-study/fresh_commands.sh
# 后续81项单列，第一阶段分析后再执行：
CUDA_VISIBLE_DEVICES=0 bash docs/features/qm9-epoch200-modular-study/stage2_commands.sh
```

输出为`artifacts/qm9_epoch200/fresh/<experiment.name>/seed<seed>/`。不要同时对同一个任务启动多个实例；已有结果目录会拒绝覆盖。两个脚本均为顺序执行，不包含未经GPU实测的并发推荐；不要在两张卡上重复执行同一完整列表。

## 4. 从任意完整epoch边界恢复

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/resume_qm9_fresh200.py \
  --run-dir artifacts/qm9_epoch200/fresh/chemprop_task_attention7/seed42
# 指定50轮快照，保留原200轮LR和早停历史，不重新warmup：
CUDA_VISIBLE_DEVICES=0 uv run python scripts/resume_qm9_fresh200.py \
  --run-dir artifacts/qm9_epoch200/fresh/chemprop_task_attention7/seed42 \
  --checkpoint epoch0050.ckpt
```

默认恢复`checkpoints/last.ckpt`。也可对训练入口使用`--resume <完整ckpt> --output-dir <原run>`；不得把`best_checkpoint.pt`推理导出传给resume。新工具不接收旧100轮协议文件，旧续训工具仍独立保留。

严格检查配置、scaler、数据身份、诊断上下文和可见GPU数量；scaler先核对训练统计后复用保存系数，序列词表/merge从状态恢复，不重新拟合。恢复时保留相同CUDA_VISIBLE_DEVICES数量。变更batch、划分、LR或头结构应新建实验，不伪装成同次续训。已因patience20提前停止的任务拒绝静默重置早停；已到200轮的任务不会重复启动。

任意batch中断时从上一完整epoch重做该未完成轮。当前数据加载器没有随机增强，保存worker基准seed的generator和采样状态；双worker连续/恢复已测试。未来若增加worker内部随机增强，必须另加状态测试。不同GPU之间不承诺逐位一致。

诊断恢复必须明确传`--diagnostic-epochs`，并保持原limit_rows与FP32/CuBLAS数值策略；工具会从checkpoint读取后者，不能把16条诊断续训成正式全量。

FP32固定highest matmul precision、关闭TF32，CuBLAS workspace默认为`:4096:8`。PyTorch版本和这些数值策略一并保存并核对；不能在恢复时悄悄更改精度设置。

## 5. 保存内容及磁盘

- `checkpoints/last.ckpt`：每轮完整训练状态；`epoch0050/0100/0150/0200.ckpt`：达到该轮才保存，不为产生快照强行越过早停。
- `terminal_fit.ckpt`：训练结束完整状态，与推理使用的best分离。
- 完整ckpt包含参数、Adam状态、scheduler、完整轮次/更新数、早停计数、RNG、三个loader的generator、配置/scaler/数据指纹及当时最佳模型权重。
- `best_checkpoint.pt`：按验证MAE选择的最终推理导出，不含优化器。即使从较早快照回退，也使用该快照嵌入的历史best，避免误用未来分支的best文件。
- 恢复时旧日志及旧best.ckpt移入`resume_backups/`。之后只有再次出现更优验证指标才产生新的best.ckpt；正式预测以`best_checkpoint.pt`为准。
- `scalers.json/config.yaml/split_identity.json/environment.json`、`training_log.csv`、`test_predictions.csv/test_metrics.json`、`training_timing.json/inference_timing.json`及注意力/UW诊断均落盘。

完整状态比仅权重文件大：Adam两份状态、当前权重、历史best，以及多个周期/last/terminal副本都占空间。需要按骨干实际参数量检查数据盘，不把原50GB数据盘默认视为足够保存全部114项。不要只复制推理模型就删除完整状态；先将完成实验归档到本地/其他存储。GPU内存记录为PyTorch训练allocator峰值，推理耗时范围明确含loader和原单位指标处理，不能直接冒充FLOPs或单查询延迟。


## 6. 全114项GPU烟测（500条、1轮）

```bash
# 独立目录，实际使用CUDA；每个分区前500条，每轮256+244两个batch。
CUDA_VISIBLE_DEVICES=0 uv run python scripts/smoke_qm9_epoch200.py \
  --output artifacts/qm9_epoch200/smoke114 --rows 500 \
  --prune-successful-checkpoints
# 只修复/检查某个骨干；同目录会重测失败项，保留已通过且源码/config hash相同的项：
CUDA_VISIBLE_DEVICES=0 uv run python scripts/smoke_qm9_epoch200.py \
  --output artifacts/qm9_epoch200/smoke114 --rows 500 \
  --model dimenet_pp --prune-successful-checkpoints
```

脚本每项起独立子进程，不复用CUDA上下文；当前实现顺序执行，不做并发校准。原配置仍为200轮，诊断运行上限另设1轮。完整训练分区用于scaler和词表拟合，val/test不参与拟合。结果在`results.csv/results.json`，每项配置、原单位指标、预测CSV、日志及`smoke_audit.json`单独落盘。主机内存是按秒采样的进程树RSS求和，共享页可能重复，不能当成精确独占内存。

`--prune-successful-checkpoints`是用户授权的烟测节省空间选项：先检查实际训练epoch/更新次数、完整状态、有限参数以及预测CSV回算指标，保存权重文件的大小/hash，再移除本次成功诊断的权重及checkpoint。默认不删除；失败现场和旧正式实验不删除。该诊断记录不能再作为续训资产，正式训练入口不受此选项影响。

更有效的测试采用分层方式：

1. 配置/数据身份/单位预检，不占GPU。
2. 每个骨干一个代表变体，优先发现可选依赖、维度、几何和CUDA问题。
3. 全114项实际一轮，覆盖全部seed、头与输入组合，独立验证失败隔离。
4. 专项测试覆盖一轮烟测无法触达的分支：第11轮后的物理权重、注意力梯度、50轮周期保存、完整状态恢复及错误数据拒绝。
5. 正式服务器再做全量数据最大分子/最大token长度batch检查与少量并发测量。一轮500条通过不证明全数据、最坏显存、长期收敛或精度提升。
