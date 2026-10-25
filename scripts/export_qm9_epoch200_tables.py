# ruff: noqa: E501
"""Export all fresh-200 QM9 runs to per-target tables (CSV / XLSX / Markdown).

Reads COMPLETED runs from the formal queue and the extension chain, aggregates the
seven property metrics over the three paired seeds, and writes machine-readable
tables plus the Markdown appendix consumed by the results documentation.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "artifacts/qm9_epoch200/formal_4gpu/runs"
EXTENSION = ROOT / "artifacts/qm9_epoch200/extension_20261022/runs"
OUT = ROOT / "artifacts/qm9_epoch200/summary"

TARGETS = ("homo", "lumo", "delta_e", "alpha", "mu", "r2", "zpve")
UNITS = {
    "homo": "hartree",
    "lumo": "hartree",
    "delta_e": "hartree",
    "alpha": "bohr^3",
    "mu": "debye",
    "r2": "bohr^2",
    "zpve": "hartree",
}
TARGET_CN = {
    "homo": "HOMO 能级",
    "lumo": "LUMO 能级",
    "delta_e": "gap（主指标）",
    "alpha": "各向同性极化率",
    "mu": "偶极矩",
    "r2": "电子空间延展",
    "zpve": "零点振动能",
}
VARIANTS = {
    "single_task_delta_e": "单任务（st）",
    "task_attention7": "旧 MTL（ta7）",
    "mlore7": "MLoRE（mtl）",
    "physics_thermo10": "PINN（pt10）",
}
VARIANT_ORDER = ("single_task_delta_e", "task_attention7", "mlore7", "physics_thermo10")
MODEL_CN = {
    "dimenet_pp": "DimeNet++",
    "torchmd_net": "TorchMD-Net",
    "painn": "PaiNN",
    "egnn": "EGNN",
    "schnet": "SchNet",
    "3dinfomax": "3DInfomax",
    "chemprop": "Chemprop",
    "esa": "ESA",
    "smiles_transformer": "SMILES Tr",
    "spe_transformer": "SPE Tr",
    "demol": "DeMol",
}
MODEL_ORDER = (
    "dimenet_pp",
    "torchmd_net",
    "painn",
    "egnn",
    "schnet",
    "3dinfomax",
    "chemprop",
    "esa",
    "smiles_transformer",
    "spe_transformer",
    "demol",
)
SEEDS = ("42", "2026", "3407")
ID_RE = re.compile(r"^(?P<model>.+)_(?P<variant>" + "|".join(VARIANTS) + r")_seed(?P<seed>\d+)$")


def collect_runs():
    runs = {}
    for root in (FORMAL, EXTENSION):
        for d in sorted(root.glob("*/")):
            if not (d / "COMPLETED.json").exists():
                continue
            m = ID_RE.match(d.name)
            if m is None:
                raise ValueError(f"Unrecognised run id: {d.name}")
            runs[(m["model"], m["variant"], m["seed"])] = d
    return runs


def run_rows(runs):
    rows = []
    for (model, variant, seed), d in sorted(
        runs.items(),
        key=lambda kv: (MODEL_ORDER.index(kv[0][0]), VARIANT_ORDER.index(kv[0][1]), int(kv[0][2])),
    ):
        metrics = json.loads((d / "test_metrics.json").read_text())
        done = json.loads((d / "COMPLETED.json").read_text())
        for t in TARGETS:
            if t not in metrics["targets"]:
                continue
            m = metrics["targets"][t]
            rows.append(
                {
                    "model": model,
                    "variant": variant,
                    "seed": int(seed),
                    "root": "formal" if d.parent == FORMAL else "extension",
                    "target": t,
                    "unit": UNITS[t],
                    "n": m["n"],
                    "mae": m["mae"],
                    "rmse": m["rmse"],
                    "mse": m["mse"],
                    "r2": m["r2"],
                    "mape": m.get("mape"),
                    "epochs": done["epochs_completed"],
                    "early_stopped": done["early_stopped"],
                    "best_epoch": done["best_epoch"],
                }
            )
    return rows


def aggregate(rows):
    cells = {}
    for r in rows:
        key = (r["model"], r["variant"], r["target"])
        cells.setdefault(key, []).append(r)
    out = {}
    for key, group in cells.items():
        out[key] = {
            "n_seeds": len(group),
            "mae_mean": float(np.mean([g["mae"] for g in group])),
            "mae_sd": float(np.std([g["mae"] for g in group], ddof=1)) if len(group) > 1 else 0.0,
            "rmse_mean": float(np.mean([g["rmse"] for g in group])),
            "rmse_sd": float(np.std([g["rmse"] for g in group], ddof=1)) if len(group) > 1 else 0.0,
            "r2_mean": float(np.mean([g["r2"] for g in group])),
            "epochs_mean": float(np.mean([g["epochs"] for g in group])),
            "any_early_stop": any(g["early_stopped"] for g in group),
        }
    return out


def write_long_csv(rows):
    path = OUT / "per_target_metrics.csv"
    fields = [
        "model",
        "variant",
        "seed",
        "root",
        "target",
        "unit",
        "n",
        "mae",
        "rmse",
        "mse",
        "r2",
        "mape",
        "epochs",
        "early_stopped",
        "best_epoch",
    ]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    return path


def write_agg_csv(agg):
    path = OUT / "per_target_aggregates.csv"
    fields = [
        "model",
        "variant",
        "target",
        "unit",
        "n_seeds",
        "mae_mean",
        "mae_sd",
        "rmse_mean",
        "rmse_sd",
        "r2_mean",
        "epochs_mean",
        "any_early_stop",
    ]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for (model, variant, target), a in sorted(
            agg.items(),
            key=lambda kv: (
                MODEL_ORDER.index(kv[0][0]),
                VARIANT_ORDER.index(kv[0][1]),
                TARGETS.index(kv[0][2]),
            ),
        ):
            w.writerow(
                {"model": model, "variant": variant, "target": target, "unit": UNITS[target], **a}
            )
    return path


def write_xlsx(long_rows, agg):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    head_fill = PatternFill("solid", fgColor="1F4E78")
    head_font = Font(color="FFFFFF", bold=True)

    def sheet(title, headers, data, widths=None):
        ws = wb.create_sheet(title)
        ws.append(headers)
        for cell in ws[1]:
            cell.fill = head_fill
            cell.font = head_font
        for row in data:
            ws.append(row)
        for j, w in enumerate(widths or [], start=1):
            ws.column_dimensions[get_column_letter(j)].width = w
        ws.freeze_panes = "A2"
        return ws

    readme = wb.active
    readme.title = "README"
    for line in [
        ("fresh-200 QM9 gap 研究 — 全部实验结果（60 项：正式 42 + 扩展 18）", True),
        ("", False),
        ("协议：200 epoch 上限 / batch 256 / patience 20 / 仅训练集 Z-score /", False),
        ("验证集 delta_e MAE 选型 / 仅测试集最终评估 / 配对种子 42,2026,3407 / 5 折 CV。", False),
        ("", False),
        ("工作表：", True),
        (
            "  per_seed_long   — 每一行的粒度：run × target 的完整指标（含逐 seed、epoch、早停标记）",
            False,
        ),
        ("  agg_per_target  — model × variant × target 的 3-seed 均值±SD", False),
        (
            "  delta_e_groups  — 主指标 delta_e 的分组汇总（按 RMSE 升序，含 epoch 与早停信息）",
            False,
        ),
        ("  gains           — 模块增益（逐 target 配对差值百分比）", False),
        ("", False),
        (
            "单位：homo/lumo/delta_e/zpve = hartree；alpha = bohr^3；mu = debye；r2 = bohr^2。",
            False,
        ),
        ("口径：mean±SD 对 3 个配对 seed；早停 run 的指标为其 best_epoch 处的测试值。", False),
    ]:
        readme.append([line[0]])
        readme.cell(readme.max_row, 1).font = Font(bold=line[1])
    readme.column_dimensions["A"].width = 110

    sheet(
        "per_seed_long",
        [
            "model",
            "variant",
            "seed",
            "root",
            "target",
            "unit",
            "n",
            "mae",
            "rmse",
            "mse",
            "r2",
            "mape",
            "epochs",
            "early_stopped",
            "best_epoch",
        ],
        [
            [
                r["model"],
                r["variant"],
                r["seed"],
                r["root"],
                r["target"],
                r["unit"],
                r["n"],
                r["mae"],
                r["rmse"],
                r["mse"],
                r["r2"],
                r["mape"],
                r["epochs"],
                r["early_stopped"],
                r["best_epoch"],
            ]
            for r in long_rows
        ],
        [14, 22, 6, 10, 9, 9, 7, 12, 12, 12, 9, 9, 8, 13, 10],
    )
    sheet(
        "agg_per_target",
        [
            "model",
            "variant",
            "target",
            "unit",
            "n_seeds",
            "mae_mean",
            "mae_sd",
            "rmse_mean",
            "rmse_sd",
            "r2_mean",
            "epochs_mean",
            "any_early_stop",
        ],
        [
            [
                m,
                v,
                t,
                UNITS[t],
                a["n_seeds"],
                a["mae_mean"],
                a["mae_sd"],
                a["rmse_mean"],
                a["rmse_sd"],
                a["r2_mean"],
                a["epochs_mean"],
                a["any_early_stop"],
            ]
            for (m, v, t), a in sorted(
                agg.items(),
                key=lambda kv: (
                    MODEL_ORDER.index(kv[0][0]),
                    VARIANT_ORDER.index(kv[0][1]),
                    TARGETS.index(kv[0][2]),
                ),
            )
        ],
        [14, 22, 9, 9, 8, 12, 11, 12, 11, 9, 11, 14],
    )
    groups = [(k, a) for k, a in agg.items() if k[2] == "delta_e"]
    groups.sort(key=lambda kv: kv[1]["rmse_mean"])
    sheet(
        "delta_e_groups",
        [
            "rank",
            "model",
            "variant",
            "n_seeds",
            "mae_mean",
            "mae_sd",
            "rmse_mean",
            "rmse_sd",
            "r2_mean",
            "epochs_mean",
            "any_early_stop",
        ],
        [
            [
                i + 1,
                m,
                v,
                a["n_seeds"],
                a["mae_mean"],
                a["mae_sd"],
                a["rmse_mean"],
                a["rmse_sd"],
                a["r2_mean"],
                a["epochs_mean"],
                a["any_early_stop"],
            ]
            for i, ((m, v, _), a) in enumerate(groups)
        ],
        [6, 14, 22, 8, 12, 11, 12, 11, 9, 11, 14],
    )
    gains = gain_rows(agg)
    sheet(
        "gains",
        ["model", "target", "comparison", "delta_mae_pct", "delta_rmse_pct"],
        [[g["model"], g["target"], g["comparison"], g["mae_pct"], g["rmse_pct"]] for g in gains],
        [14, 9, 34, 14, 14],
    )
    path = OUT / "qm9_epoch200_all_results.xlsx"
    wb.save(path)
    return path


def gain_rows(agg):
    rows = []
    pairs = (
        ("mlore7", "task_attention7", "MLoRE vs 旧 ta7"),
        ("mlore7", "physics_thermo10", "MLoRE vs PINN"),
        ("task_attention7", "physics_thermo10", "旧 ta7 vs PINN"),
        ("physics_thermo10", "single_task_delta_e", "PINN vs 单任务"),
        ("mlore7", "single_task_delta_e", "MLoRE vs 单任务"),
        ("task_attention7", "single_task_delta_e", "旧 ta7 vs 单任务"),
    )
    for model in MODEL_ORDER:
        for target in TARGETS:
            for new, base, label in pairs:
                a = agg.get((model, new, target))
                b = agg.get((model, base, target))
                if not a or not b:
                    continue
                rows.append(
                    {
                        "model": model,
                        "target": target,
                        "comparison": label,
                        "mae_pct": round((a["mae_mean"] / b["mae_mean"] - 1) * 100, 2),
                        "rmse_pct": round((a["rmse_mean"] / b["rmse_mean"] - 1) * 100, 2),
                    }
                )
    return rows


def fmt(v, digits=6):
    return f"{v:.{digits}f}"


def fmt_pm(mean, sd, digits=6):
    return f"{mean:.{digits}f}±{sd:.{digits}f}"


def md_target_table(agg, model, metric):
    """Wide table: rows = variants, columns = targets (mean±SD over seeds)."""
    lines = [
        "| 目标 | " + " | ".join(f"{t} " + UNITS[t] for t in TARGETS) + " |",
        "|---|" + "---:|" * len(TARGETS),
    ]
    for variant in VARIANT_ORDER:
        cells = []
        for t in TARGETS:
            a = agg.get((model, variant, t))
            if a is None:
                cells.append("—")
            elif metric == "mae":
                cells.append(fmt_pm(a["mae_mean"], a["mae_sd"]))
            elif metric == "rmse":
                cells.append(fmt_pm(a["rmse_mean"], a["rmse_sd"]))
            else:
                cells.append(fmt(a["r2_mean"], 4) if a["n_seeds"] >= 1 else "—")
        lines.append(f"| {VARIANTS[variant]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def md_delta_e_ranking(agg):
    groups = [(k, a) for k, a in agg.items() if k[2] == "delta_e"]
    groups.sort(key=lambda kv: kv[1]["rmse_mean"])
    lines = [
        "| # | 骨干 | 方法 | MAE（3-seed） | RMSE（3-seed） | R² | epochs（均值） | 早停 |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for i, ((model, variant, _), a) in enumerate(groups, 1):
        es = "是" if a["any_early_stop"] else "否"
        lines.append(
            f"| {i} | {MODEL_CN[model]} | {VARIANTS[variant]} | "
            f"{fmt_pm(a['mae_mean'], a['mae_sd'])} | {fmt_pm(a['rmse_mean'], a['rmse_sd'])} | "
            f"{fmt(a['r2_mean'], 4)} | {a['epochs_mean']:.0f} | {es} |"
        )
    return "\n".join(lines)


def md_gain_table(agg, model, new, base, label, digits=1):
    lines = ["| 目标 | ΔMAE | ΔRMSE |", "|---|---:|---:|"]
    for t in TARGETS:
        a = agg.get((model, new, t))
        b = agg.get((model, base, t))
        if not a or not b:
            lines.append(f"| {t} | — | — |")
            continue
        dm = (a["mae_mean"] / b["mae_mean"] - 1) * 100
        dr = (a["rmse_mean"] / b["rmse_mean"] - 1) * 100
        lines.append(f"| {t} | {dm:+.{digits}f}% | {dr:+.{digits}f}% |")
    return "\n".join(lines)


def write_markdown(agg, rows):
    src = FORMAT_MD_TEMPLATE.format(
        ranking=md_delta_e_ranking(agg),
        dimenet_mae=md_target_table(agg, "dimenet_pp", "mae"),
        dimenet_rmse=md_target_table(agg, "dimenet_pp", "rmse"),
        dimenet_r2=md_target_table(agg, "dimenet_pp", "r2"),
        torchmd_mae=md_target_table(agg, "torchmd_net", "mae"),
        torchmd_rmse=md_target_table(agg, "torchmd_net", "rmse"),
        torchmd_r2=md_target_table(agg, "torchmd_net", "r2"),
        demol_mae=md_target_table(agg, "demol", "mae"),
        demol_rmse=md_target_table(agg, "demol", "rmse"),
        demol_r2=md_target_table(agg, "demol", "r2"),
        dimenet_gain_mlore_ta7=md_gain_table(
            agg, "dimenet_pp", "mlore7", "task_attention7", "MLoRE vs 旧 ta7"
        ),
        dimenet_gain_mlore_pt10=md_gain_table(
            agg, "dimenet_pp", "mlore7", "physics_thermo10", "MLoRE vs PINN"
        ),
        torchmd_gain_mlore_ta7=md_gain_table(
            agg, "torchmd_net", "mlore7", "task_attention7", "MLoRE vs 旧 ta7"
        ),
        torchmd_gain_mlore_pt10=md_gain_table(
            agg, "torchmd_net", "mlore7", "physics_thermo10", "MLoRE vs PINN"
        ),
        torchmd_gain_pt10_st=md_gain_table(
            agg, "torchmd_net", "physics_thermo10", "single_task_delta_e", "PINN vs 单任务"
        ),
        dimenet_gain_pt10_st=md_gain_table(
            agg, "dimenet_pp", "physics_thermo10", "single_task_delta_e", "PINN vs 单任务"
        ),
    )
    path = ROOT / "docs/features/qm9-epoch200-modular-study/per-target-results.md"
    path.write_text(src)
    return path


FORMAT_MD_TEMPLATE = """# QM9 gap 模块化研究：逐 target 完整结果（fresh-200）

> 本文件由 `scripts/export_qm9_epoch200_tables.py` 从 60 个已完成 run 的 `test_metrics.json`
> 自动生成（正式 42 项 + 扩展 18 项，含 demol、MLoRE 全量）。生成于 2026-10-26（全部 60 项实验完成后）。

## 1. 协议与统计口径

- **数据/划分**：QM9 全量 133,885 分子；随机 80/10/10 划分（split seed 与训练 seed 配对：
  42/2026/3407），测试集 13,380 行（10%，quantile 分层 200 bin）。本报告全部指标均为该
  **测试集**指标（最终评估只在测试集做一次）。
- **训练协议**：200 epoch 上限，batch 256，patience 20（验证集 delta_e MAE 无改善即早停），
  `val_delta_e_mae` 选型，仅训练集上拟合的逐目标 Z-score 反标准化，TF32 关闭，确定性 seeds。
- **种子**：配对种子 42 / 2026 / 3407；表中为 **3-seed 均值±SD**（SD 为样本标准差）。
- **早停**：`epochs` 列为实际训练 epoch；早停 run 的指标取 best_epoch 处的测试值（与正式协议一致）。
- **单任务基线限制**：`single_task_delta_e` 变体只训练 delta_e 一个目标，因此**只有 delta_e 列有
  单任务对照**；其余 6 个目标（homo/lumo/alpha/mu/r2/zpve）仅多任务变体（ta7 / mlore7 / pt10）有
  预测，逐 target 对比只能在多任务方法之间进行。DeMol 骨干的 st 基线同样只覆盖 delta_e。
- **单位**：homo / lumo / delta_e / zpve = hartree；alpha = bohr^3；mu = debye；r2 = bohr^2。

## 2. 数据文件

| 文件 | 内容 |
|---|---|
| `artifacts/qm9_epoch200/summary/per_target_metrics.csv` | 逐 run × 逐 target 的原始指标（含 seed、epoch、早停标记） |
| `artifacts/qm9_epoch200/summary/per_target_aggregates.csv` | model × variant × target 的 3-seed 均值±SD |
| `artifacts/qm9_epoch200/summary/qm9_epoch200_all_results.xlsx` | 上述数据的 Excel 版（4 个工作表 + README） |
| 本文件 | 说明 + 全部汇总表 |

## 3. 主指标 delta_e 总排名（3-seed 均值±SD，按 RMSE 升序）

{ranking}

## 4. 逐 target 明细（MAE / RMSE / R²）

### 4.1 DimeNet++（图骨干；四个变体齐全）

**MAE（mean±SD）**

{dimenet_mae}

**RMSE（mean±SD）**

{dimenet_rmse}

**R²（3-seed 均值）**

{dimenet_r2}

### 4.2 TorchMD-Net（3D transformer；四个变体齐全）

**MAE（mean±SD）**

{torchmd_mae}

**RMSE（mean±SD）**

{torchmd_rmse}

**R²（3-seed 均值）**

{torchmd_r2}

### 4.3 DeMol（3D transformer，扩展；st 仅 delta_e，pt10 全 7 目标）

**MAE（mean±SD）**

{demol_mae}

**RMSE（mean±SD）**

{demol_rmse}

**R²（3-seed 均值）**

{demol_r2}

### 4.4 其余骨干（仅 delta_e 单任务，扩展 PINN 的 SMILES/SPE 见 §5）

其余 6 个骨干只有单任务 delta_e 结果（painn 0.002042±0.000015 / egnn 0.002109±0.000039 /
3dinfomax 0.003457±0.000058 / schnet 0.003713±0.000124 / chemprop 0.003978±0.000096 /
esa 0.005169±0.000112 MAE；RMSE 见 §3 表），其扩展 PINN 配对（smiles/spe）在 §5 给出。

## 5. 模块增益（逐 target 配对差值百分比，负值=新方法更好）

> 约定：Δ = (新 − 对照)/对照 × 100%。**只有 delta_e 列**有单任务对照；其余目标以 ta7 或
> pt10 作为多任务/多目标对照。

### 5.1 PINN vs 单任务（仅 delta_e 有 st 对照）

DimeNet++：

{dimenet_gain_pt10_st}

TorchMD-Net：

{torchmd_gain_pt10_st}

SMILES Tr / SPE Tr 的 delta_e 配对（扩展，3-seed）：
- SMILES Tr：st 0.006355 / 0.009619 → PINN 0.005408 / 0.008311（ΔMAE −14.9%、ΔRMSE −13.6% ✅）
- SPE Tr：st 0.005969 / 0.009622 → PINN 0.005899 / 0.009149（ΔMAE −1.2%、ΔRMSE −4.9% ✅）
- DeMol（3-seed 完整配对）：st 0.001698±0.000042/0.003060±0.000158 → PINN
  0.001662±0.000419/0.002670±0.000481，配对 **ΔMAE 均值 −2.4%、ΔRMSE 均值 −12.8%**
  （seed42 +23.9%/+3.5% ❌（ep71 早停）；seed2026 −16.7%/−23.8% ✅；seed3407 −14.4%/−18.1% ✅）
  → **2/3 seed 提点、均值正向**，DeMol 是第 4 个被 PINN 提点的骨干

### 5.2 MLoRE vs 旧 ta7（7 目标）

DimeNet++：

{dimenet_gain_mlore_ta7}

TorchMD-Net：

{torchmd_gain_mlore_ta7}

### 5.3 MLoRE vs PINN（7 目标）

DimeNet++：

{dimenet_gain_mlore_pt10}

TorchMD-Net：

{torchmd_gain_mlore_pt10}

## 6. 关键观察（逐 target 视角）

1. **目标族分化明确**：homo/lumo/delta_e/mu 四个"电子结构同源"目标在 TorchMD+MLoRE 上取得
   最优（相对 PINN 的 delta_e −11.8%、homo −8.1%、lumo −7.6%、mu −3.4%）；alpha/r2/zpve 三个
   "热力学/物理量"目标由 PINN 主导（dimenet 上 pt10 对 ta7 三项全胜，r2 尤其悬殊）。
2. **DimeNet++ 上 PINN 对旧 ta7 7/7 目标全胜**（MAE 与 RMSE 均全胜），7 目标无短板——与
   delta_e 主指标第 1 名的结论一致；这也是"物理辅助目标不会牺牲任一单目标"的直接证据。
3. **MLoRE 在 DimeNet++ 上对旧 ta7 仅 lumo（−9%）与 zpve（−47%）两项占优**，其余 5 目标下降，
   r2 目标崩坏最严重（+222%）——MLoRE 的低秩专家路由难以在强 GNN 特征上共享有用信息。
4. **MLoRE 在 TorchMD-Net 上对旧 ta7 7/7 目标全胜**，且对 PINN 在 4/7 目标上占优——说明该
   方法在 transformer 骨干上确有价值，但整体仍未跑赢 delta_e 单任务基线（−0.5% RMSE 属噪声）。
5. **R² 的伪饱和**：dimenet pt10 的 7 目标 R² 全部 ≥0.9946（r2 目标 0.99929），单看 R² 会低估
   误差差异；比较应以 MAE/RMSE 为主。
6. **DeMol 上 PINN 的净效应为正但方差极大**：3-seed 配对均值 ΔRMSE −12.8%（2/3 seed 提点，
   seed2026 达 −23.8%），唯一劣化 seed42 由 PINN 早停（ep71）造成；该骨干单 seed 结论不可靠，
   必须用配对均值，且要在论文里披露 pt10 的 RMSE SD（±0.00048）远大于 st（±0.00016）。
7. **PINN 章最终判决修正为 4/5 骨干提点**（DimeNet −8.2%、SMILES −13.6%、SPE −4.9%、DeMol −12.8%），
   仅 TorchMD-Net +12.0% 劣化——PI 章"物理约束对强 3D 等变 transformer 过正则"的假说仍成立，
   但 DeMol 不是反例（它的 PINN 变体在 2/3 seed 上更强）。

## 7. 局限与备注

- 单任务基线只预测 delta_e：**无法评估"加入 6 个辅助目标对辅助目标本身的收益"**（缺少 7 目标
  单任务对照），这是 fresh-200 协议的结构性缺口；逐 target 增益只作方法间对比。
- demol 的 st 三个 run 均跑满 200 ep；pt10 仅 seed42 在 ep71 早停（指标取 best_epoch，明显偏弱）。
  因此 DeMol 结论以 3-seed 配对均值为准（ΔRMSE −12.8%）；单 seed 或 ratio-of-means 会被
  pt10 的高方差误读（pt10 RMSE SD ±0.00048 vs st ±0.00016）。
- 旧 ta7（task attention）结果保留作历史证据；当前 MTL 路线为 MLoRE，论文主表应使用 mlore7。
- r2 目标（bohr^2）数值量级为 10^-1~10^1，alpha 为 10^-1~10^0，与 hartree 目标不可直接比大小。

## 8. 复现

```bash
uv run --no-sync python scripts/export_qm9_epoch200_tables.py
```

脚本从 `artifacts/qm9_epoch200/{{formal_4gpu,extension_20261022}}/runs/*/test_metrics.json` 读取，
重新生成 §2 中全部文件（含本 Markdown）。
"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = collect_runs()
    rows = run_rows(runs)
    agg = aggregate(rows)
    long_csv = write_long_csv(rows)
    agg_csv = write_agg_csv(agg)
    xlsx = write_xlsx(rows, agg)
    md = write_markdown(agg, rows)
    print(f"runs: {len(runs)}, long rows: {len(rows)}, groups: {len(agg)}")
    for p in (long_csv, agg_csv, xlsx, md):
        print("written:", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
