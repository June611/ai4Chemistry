"""Per-seed and aggregate Phase 1 reports; never automatically launch Phase 2."""

from pathlib import Path

import numpy as np
import pandas as pd

from molgap.experiments.neural_properties import lane_states
from molgap.multitask7.config import TASK_NAMES, TASK_UNITS, VARIANTS
from molgap.multitask7.evaluation import verify_run
from molgap.tracking.records import read_json, write_json


def summarize(directory, *, strict=False):
    directory = Path(directory).resolve()
    plan = read_json(directory / "plan.json")
    root = Path(plan["root"])
    jobs = [j for _, state in lane_states(directory) for j in state["jobs"]]
    output = root / "outputs/summaries/qm9_results/batches" / directory.name
    output.mkdir(parents=True, exist_ok=True)
    rows, wide, available = [], [], []
    baseline = read_json(
        root / "docs/features/qm9-seven-task-mtl-dynamic-loss/baseline_results.json"
    )
    baseline_by_seed = {r["seed"]: r for r in baseline.get("records", [])}
    issues = []
    for job in jobs:
        row = {k: job.get(k) for k in ("id", "model", "seed", "status", "error", "run_dir")}
        row["variant"] = VARIANTS[job["model"]][0]
        if job["status"] == "completed":
            try:
                verify_run(job["run_dir"], root)
            except Exception as exc:
                row.update(status="verification_failed", error=str(exc))
                issues.append(f"{job['id']}: {exc}")
            else:
                available.append(job)
                run = Path(job["run_dir"])
                metrics = read_json(run / "test_metrics.json")["targets"]
                timing = read_json(run / "training_timing.json")
                row.update(timing)
                peaks = job.get("peaks", {})
                row.update({k: peaks.get(k) for k in ("gpu_process_mib", "rss_mib", "pss_mib")})
                for task in TASK_NAMES:
                    record = dict(
                        model=job["model"],
                        variant=row["variant"],
                        seed=job["seed"],
                        target=task,
                        **metrics[task],
                    )
                    rows.append(record)
                    row.update(
                        {
                            f"{task}_{metric}": metrics[task][metric]
                            for metric in ("mae", "mse", "rmse", "r2")
                        }
                    )
                previous = (
                    baseline_by_seed.get(job["seed"], {})
                    .get("targets", {})
                    .get("delta_e", {})
                    .get("mae")
                )
                row["delta_e_improvement_pct_vs_A"] = (
                    100 * (previous - metrics["delta_e"]["mae"]) / previous if previous else None
                )
        elif strict:
            issues.append(f"{job['id']}: {job['status']}")
        wide.append(row)
    pd.DataFrame(wide).to_csv(output / "phase1_comparison.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "per_seed_metrics.csv", index=False)
    grouped = []
    for name in VARIANTS:
        for task in TASK_NAMES:
            subset = [r for r in rows if r["model"] == name and r["target"] == task]
            if not subset:
                continue
            row = dict(
                model=name,
                variant=VARIANTS[name][0],
                target=task,
                unit=TASK_UNITS[task],
                seeds=len(subset),
            )
            for metric in ("mae", "mse", "rmse", "r2"):
                values = [r[metric] for r in subset]
                row[metric + "_mean"] = float(np.mean(values))
                row[metric + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            grouped.append(row)
    pd.DataFrame(grouped).to_csv(output / "aggregate_metrics.csv", index=False)
    a = [r["targets"]["delta_e"]["mae"] for r in baseline_by_seed.values()]
    gate = {
        "phase2_auto_start": False,
        "scope": "exploratory gate based on three-seed test means; not statistical significance",
        "complete": len(available) == 12,
        "status": "PENDING",
        "candidates": [],
    }
    if len(available) == 12 and len(a) == 3:
        a_mean, a_std = float(np.mean(a)), float(np.std(a, ddof=1))
        for row in grouped:
            if row["target"] != "delta_e" or row["seeds"] != 3:
                continue
            improvement = 100 * (a_mean - row["mae_mean"]) / a_mean
            passed = improvement > 2 or (improvement > 1.5 and row["mae_std"] < a_std)
            gate["candidates"].append(
                dict(
                    model=row["model"],
                    improvement_pct=improvement,
                    std=row["mae_std"],
                    passes_exploratory_gate=passed,
                )
            )
        gate["status"] = (
            "PASS" if any(c["passes_exploratory_gate"] for c in gate["candidates"]) else "FAIL"
        )
        gate.update(baseline_mean=a_mean, baseline_std=a_std)
    write_json(output / "phase1_gate.json", gate)
    summary_lines = [
        "# Phase 1 Chemprop 七任务结果",
        "",
        f"已验证完成 {len(available)}/12；探索性扩展门槛：{gate['status']}。Phase 2 不会自动启动。",
        "",
        "主要口径：配对划分/初始化seed42、2026、3407；256/100/20；"
        "train-only Z-score；原单位验证ΔE MAE选模。",
        "C/D2按用户选择保留额外投影层。C对B同时反映容量与融合变化，不能单独归因于49个矩阵参数。",
        "",
        "| 变体 | Seed数 | ΔE MAE mean | std（ddof=1）| 单位 |",
        "|---|---:|---:|---:|---|",
    ]
    for row in grouped:
        if row["target"] == "delta_e":
            std = f"{row['mae_std']:.8f}" if row["mae_std"] is not None else "待补"
            summary_lines.append(
                f"| {row['variant']} | {row['seeds']} | {row['mae_mean']:.8f} | {std} | Hartree |"
            )
    summary_lines.extend(
        [
            "",
            "完整逐seed结果见phase1_comparison.csv，七性质长表见per_seed_metrics.csv。",
            "历史A的min_delta=1e-5；新规范为0。历史A均实际跑满100轮，比较时保留此差异说明。",
            "辅助性质单任务A仍待从现有队列导入；缺失时不能计算对应负迁移百分比。",
        ]
    )
    (output / "phase1_results.md").write_text("\n".join(summary_lines) + "\n")
    _analyses(output, available, grouped, baseline_by_seed)
    write_json(
        output / "verification.json",
        dict(verified_runs=len(available), total_jobs=len(jobs), issues=issues),
    )
    if strict and (issues or len(available) != 12):
        raise ValueError(f"Phase 1 incomplete or invalid: {issues}")
    return output


def _analyses(output, jobs, grouped, baseline):
    # Always produce the five required analysis files, explicitly distinguishing missing evidence.
    affinity, uncertainty, resources = [], [], []
    for job in jobs:
        run = Path(job["run_dir"])
        a, u = read_json(run / "affinity_matrix.json"), read_json(run / "uncertainty_params.json")
        if a:
            affinity.append(dict(model=job["model"], seed=job["seed"], **a))
        if u:
            uncertainty.append(dict(model=job["model"], seed=job["seed"], **u))
        resources.append(
            dict(
                model=job["model"],
                seed=job["seed"],
                **read_json(run / "training_timing.json"),
                peaks=job.get("peaks", {}),
            )
        )
    write_json(
        output / "analysis_data.json",
        dict(affinity=affinity, uncertainty=uncertainty, resources=resources),
    )
    notes = {
        "negative_transfer_analysis.md": (
            "七性质指标已分别汇总。辅助性质的单任务对照尚需导入，缺失时不判定其负迁移。"
            "跨性质误差不能按不同原始单位直接排序。"
        ),
        "affinity_matrix_analysis.md": (
            "矩阵行是接收任务，列是提供任务。最佳checkpoint的矩阵见analysis_data.json；"
            "每轮轨迹见各run的task_dynamics.jsonl。权重不是物理因果证据。"
            "C/D2还增加了投影容量。"
        ),
        "uncertainty_analysis.md": (
            "导出s=log_vars与exp(-s)。高precision表示更大的标准化损失系数；"
            "较大的标准化残差通常对应较低precision。"
            "不能将该量直接视为校准后的分子预测置信区间，"
            "也不能按各性质原单位MAE直接比较任务难度。"
        ),
        "training_dynamics_analysis.md": (
            "训练曲线使用每个run的training_log.csv；统一按验证ΔE MAE选模。"
            "记录最佳epoch与停止epoch，观察其他任务继续改善时ΔE是否退化。"
            "训练objective含可变权重，不能只用其数值比较四个变体。"
        ),
        "resource_efficiency_analysis.md": (
            "参数量、fit/pipeline时间与采样资源见phase1_comparison.csv、analysis_data.json。"
            "并发任务耗时包含资源争用；与历史单任务时间比较时应注明调度条件，"
            "不能宣称受控架构加速。"
        ),
    }
    for filename, note in notes.items():
        (output / filename).write_text(
            f"# {filename.removesuffix('.md')}\n\n当前已验证 {len(jobs)}/12 项。\n\n{note}\n"
        )
    if not jobs:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    gaps = sorted(
        [r for r in grouped if r["target"] == "delta_e"], key=lambda r: r["mae_mean"], reverse=True
    )
    bars = axes[0].bar(
        [r["variant"] for r in gaps],
        [r["mae_mean"] for r in gaps],
        yerr=[r["mae_std"] or 0 for r in gaps],
        capsize=5,
        color="#4477AA",
    )
    axes[0].bar_label(bars, labels=[f"{r['mae_mean']:.6f}" for r in gaps], padding=5, fontsize=8)
    axes[0].set_ylim(0, max(r["mae_mean"] + (r["mae_std"] or 0) for r in gaps) * 1.3)
    if len(baseline) == 3:
        axes[0].axhline(
            np.mean([r["targets"]["delta_e"]["mae"] for r in baseline.values()]),
            color="#CC6677",
            linestyle="--",
            label="A: single task",
        )
        axes[0].legend()
    axes[0].set(ylabel="Test gap MAE (Hartree)", title="Mean ± sample standard deviation")
    for job in jobs:
        log = pd.read_csv(Path(job["run_dir"]) / "training_log.csv")
        axes[1].plot(
            log.epoch,
            log.val_delta_e_mae,
            linewidth=1,
            alpha=0.65,
            label=f"{VARIANTS[job['model']][0]}/{job['seed']}",
        )
    axes[1].set(xlabel="Epoch", ylabel="Validation gap MAE (Hartree)", title="Validation curves")
    axes[1].legend(fontsize=7, ncol=2)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"phase1_comparison.{suffix}", dpi=200)
    plt.close(fig)
    if affinity:
        fig, axes = plt.subplots(
            len(affinity),
            1,
            figsize=(7, max(3, len(affinity) * 3.4)),
            squeeze=False,
            layout="constrained",
        )
        for ax, record in zip(axes[:, 0], affinity, strict=True):
            matrix = np.asarray(record["matrix"])
            im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
            ax.set_xticks(range(7), TASK_NAMES, rotation=30, ha="right")
            ax.set_yticks(range(7), TASK_NAMES)
            ax.set_title(
                f"{VARIANTS[record['model']][0]} seed {record['seed']}: rows receive from columns"
            )
            fig.colorbar(im, ax=ax, shrink=0.8)
        fig.savefig(output / "affinity_matrices.png", dpi=180)
        plt.close(fig)
