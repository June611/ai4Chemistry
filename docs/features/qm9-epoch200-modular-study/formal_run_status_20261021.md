# 42 项正式训练运行状态与交接（2026-10-21 启动）

## ⚠️ 当前状态：已为服务器重启做安全停机（2026-10-20 22:05 UTC 左右）

用户因延长租约需要重启服务器。已于重启前完成：

1. 全量状态快照：`artifacts/qm9_epoch200/formal_4gpu/pre_restart_state_20261020_220551.json` + `pre_restart_runs_20261020_220551.csv`。
2. 优雅停止全部 lane 与训练进程（先 lane 父进程、后训练进程），四卡显存清零，`sync` 已刷盘。
3. 完整性核验：8 个 `COMPLETED.json`，12 个 `last.ckpt` 全部非零大小，无损坏迹象。

## 重启后恢复命令（在 ai4Chemistry 目录下执行）

```bash
# 与首次启动完全相同的命令；runner 自动跳过已完成任务、自动从 last.ckpt 续训
for lane in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$lane nohup uv run --no-sync python scripts/run_qm9_formal_4gpu.py \
    --gpus 0,1,2,3 --lane "$lane" \
    --output-root artifacts/qm9_epoch200/formal_4gpu \
    > "artifacts/qm9_epoch200/formal_4gpu/launcher_logs/lane${lane}_restart.log" 2>&1 &
done
```

恢复机制（已在烟测与本轮运行中验证）：
- 已完成任务（`runs/<id>/COMPLETED.json` 且非 diagnostic）自动跳过。
- 中断任务自动带 `--resume checkpoints/last.ckpt`，从最近 epoch 边界恢复（优化器/scheduler/RNG/loader/best 全状态），每 epoch 写一次 checkpoint，损失 < 1 epoch。
- 若 lane 启动失败，可单独重跑 `--lane N`。

**注意**：lane_manifest.json 的 lane→GPU 映射不变（lane0→GPU0 … lane3→GPU3）；恢复用同一 output-root，禁止改动 `runs/` 下任何文件。

## 停机时点的结果（8/42 完成）

### 已完成组聚合（test，Hartree）

| 组 | n | MAE mean ± SD | RMSE mean ± SD |
|---|---:|---|---|
| dimenet_pp / physics_thermo10 | 3 | **0.001486 ± 0.0000195** | **0.002393 ± 0.000132** |
| dimenet_pp / single_task | 1 (seed42) | 0.001515 | 0.002748 |
| dimenet_pp / task_attention7 | 3 | 0.003167 ± 0.000230 | 0.004297 ± 0.000246 |
| 3dinfomax / single_task | 1 (seed3407) | 0.003523 | 0.006036 |

单任务明细（`runs/<id>/test_metrics.json` 的 `targets.delta_e`）：
- physics_thermo10：seed42 MAE 0.001500/RMSE 0.002536（跑满 200）；seed2026 0.001464/0.002370（best ep197）；seed3407 0.001495/0.002274（best ep199）。
- task_attention7：seed42 0.003151/0.004290（ep61 早停）；seed2026 0.002946/0.004054（ep44 早停）；seed3407 0.003405/0.004547（ep34 早停）。
- dimenet_pp single_task seed42：0.001515/0.002748（跑满 200，best ep195）。
- 3dinfomax single_task seed3407：0.003523/0.006036（ep99 早停，best ep79；seed42 在 lane3 尚未开跑）。

### 停机时在训任务（恢复后从此处续跑）

| 任务 | 已到 epoch | 恢复点 | 备注 |
|---|---:|---|---|
| dimenet_pp single_task seed2026 | 162 | last.ckpt（30.5 MB） | best val 0.001681@ep108 |
| dimenet_pp single_task seed3407 | 150 | last.ckpt（30.5 MB） | best val 0.001742@ep98 |
| 3dinfomax single_task seed2026 | 117 | last.ckpt（79.9 MB） | CPU 瓶颈模型，进度慢属预期 |
| torchmd_net task_attention7 seed3407 | 45 | last.ckpt（23.3 MB） | lane1 新接任务 |

## 运行方式（背景）

- 四条独立 GPU lane（GPU0–3），每卡一次一个任务，LPT 负载均衡；`scripts/run_qm9_formal_4gpu.py`。
- 失败隔离：单项失败不终止 lane；lane 结束时返回失败计数。全程 lane 失败计数 0。
- 协议核验（round4 checkpoint 审计）：全部任务 `diagnostic=false`、200 epoch 上限/batch 256/patience 20、配对 seed (42,42)/(2026,2026)/(3407,3407)、TF32 关闭、CuBLAS `:4096:8`。
- 本 feature 的 200 epoch 协议为该研究专门设定；与 `paired-seed-100epoch-benchmark` 的 100 epoch 正式矩阵相互独立，结果不混用。

## 交接自查命令

```bash
# 队列总览（completed/active/pending + 每任务进度）
python - <<'PY'
import csv,os,json
root='artifacts/qm9_epoch200/formal_4gpu'
for r in csv.DictReader(open(root+'/queue_snapshot.csv')):
    d=root+'/runs/'+r['id']
    done=os.path.exists(d+'/COMPLETED.json')
    if done:
        m=json.load(open(d+'/test_metrics.json'))['targets']['delta_e']
        print('DONE',r['id'],'testMAE',round(m['mae'],6))
    elif os.path.exists(d+'/training_log.csv'):
        rows=list(csv.DictReader(open(d+'/training_log.csv')))
        print('RUN ',r['id'],'ep',rows[-1]['epoch'],'bad',rows[-1]['bad_epochs'])
    else: print('PEND',r['id'])
PY

# GPU 状态
nvidia-smi

# lane 失败计数
grep -c FAILED artifacts/qm9_epoch200/formal_4gpu/launcher_logs/lane*.log

# 单任务训练日志
tail -5 artifacts/qm9_epoch200/formal_4gpu/logs/<job_id>.log
```

## 全部完成后的汇总入口

每项结果位于 `runs/<job_id>/test_metrics.json`（原单位 MAE/RMSE/R²，`targets.delta_e` 为主表口径）。完成后需按 3 seed 均值 ± 样本 SD 汇总，模块增益相对同路径 single_task 基线（配对 seed）计算，不得混用历史 100 epoch 结果。fresh-200 基线深收敛显著（single_task seed42 跑满 200 才到 best），因此模块对比必须用同口径 fresh-200 基线。

## 监控快照索引

- `artifacts/qm9_epoch200/formal_4gpu/monitor_round{3..12}.json`：各轮队列状态、epoch、best 指标与 checkpoint 审计。
- `artifacts/qm9_epoch200/formal_4gpu/pre_restart_state_20261020_220551.json`：重启前最终快照（含恢复协议与逐任务状态）。
- `artifacts/qm9_epoch200/formal_4gpu/checkpoint_audit_round4.json`：正式协议 checkpoint 审计证据。

## 重平衡（2026-10-21 21:45，用户批准）

- lane0/lane3 已带 --skip-jobs 重启并 RESUME（pt10_3407 ep122→125、pt10_2026 ep91→94）。
- 接力等待器：aux-A(GPU1，egnn_3407 完成后跑 torchmd_st_3407+schnet_42)、aux-B(GPU2，chemprop_42 后跑 schnet_3407)、aux-C(GPU0，chemprop_2026 后跑 painn_42+schnet_2026)。
- 日志：launcher_logs/lane{0,3}_rebalance.log、aux{A,B,C}_gpu*.log；快照 monitor_round14_rebalance.json。
- 自检：`pgrep -af 'aux_wait_and_ru[n]'` 应有 3 个等待器；完工判断仍以 42 个 COMPLETED.json 为准。
- 预期全部完成：周四（10-22）~11:30。
