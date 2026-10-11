"""Read saved experiment artifacts and produce one comparable row per run."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import socket
from datetime import UTC, datetime
from pathlib import Path

import yaml

# API field types: 1 = text, 2 = number. ISO timestamps use text for portability.
TEXT_FIELDS = (
    "运行ID",
    "模型",
    "协议",
    "状态",
    "目标",
    "单位",
    "主机",
    "运行目录",
    "开始时间",
    "结束时间",
    "更新时间",
    "采样覆盖",
    "显存可用性",
    "诊断运行",
    "标签缩放",
    "损失函数",
    "推理计时范围",
    "FLOPs测量状态",
    "错误说明",
    "指标完整精度_JSON",
)
NUMBER_FIELDS = (
    "训练seed",
    "划分seed",
    "batch_size",
    "epoch预算",
    "学习率",
    "流水线耗时_s",
    "训练耗时_s",
    "已完成epoch",
    "当前epoch",
    "训练loss",
    "验证MAE_Hartree",
    "测试样本数",
    "测试MSE_Hartree2",
    "测试MAE_Hartree",
    "测试RMSE_Hartree",
    "测试MAPE_pct",
    "测试R2",
    "进程树RSS_MiB",
    "峰值RSS_MiB",
    "进程树PSS_MiB",
    "峰值PSS_MiB",
    "进程显存_MiB",
    "峰值进程显存_MiB",
    "整卡显存_MiB",
    "峰值整卡显存_MiB",
    "采样数",
    "采样间隔_s",
    "GPU利用率_pct",
    "峰值GPU利用率_pct",
    "平均GPU利用率_pct",
    "系统可用内存_MiB",
    "数据盘剩余_GiB",
    "推理耗时_s",
    "推理吞吐_样本每秒",
    "摊销推理时间_ms每样本",
    "模型参数量",
    "FLOPs每样本",
    "验证MSE_Hartree2",
    "验证RMSE_Hartree",
)
SCHEMA = {**dict.fromkeys(TEXT_FIELDS, 1), **dict.fromkeys(NUMBER_FIELDS, 2)}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_progress(run_dir: Path) -> dict:
    """Read QM9/Lightning CSV history; validation MAE remains in physical units."""
    progress = {}
    history = run_dir / "history.json"
    if history.is_file():
        try:
            rows = json.loads(history.read_text())
            if isinstance(rows, list) and rows:
                progress.update(
                    epoch=finite(rows[-1].get("epoch")),
                    train_loss=finite(rows[-1].get("train_loss")),
                )
        except (OSError, ValueError, TypeError):
            pass
    paths = [run_dir / "history.csv", *sorted(run_dir.glob("**/metrics.csv"))]
    for path in paths:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                training_row = False
                for dest, names in (
                    ("train_loss", ("train_loss", "train_loss_epoch", "train/loss")),
                    ("val_mae", ("val_mae", "val_delta_e_mae", "val/gap_mae")),
                ):
                    for name in names:
                        value = finite(row.get(name))
                        if value is not None:
                            progress[dest] = value
                            training_row = True
                            break
                # Lightning test rows can have the next epoch number after fitting stops.
                epoch = finite(row.get("epoch"))
                if training_row and epoch is not None:
                    progress["epoch"] = epoch + (path.name != "history.csv")
    return progress


def build_record(run_dir: Path, state: dict | None = None, target: str = "delta_e") -> dict:
    run_dir = run_dir.resolve()
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        config_path = run_dir / "resolved_config.yaml"
    config = (
        yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    )
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config: {config_path}")
    state = state if state is not None else read_json(run_dir / "tracking/state.json")
    if not config and state.get("config"):
        config = yaml.safe_load(Path(state["config"]).read_text())
    payload = read_json(run_dir / "metrics.json")
    source = payload.get("targets", {}).get(target, {}) if "targets" in payload else payload
    if "targets" not in payload and payload.get("target", target) != target:
        source = {}
    source = source if isinstance(source, dict) else {}
    timing = read_json(run_dir / "training_timing.json")
    pretrain_timing = read_json(run_dir / "timing.json")
    experiment, training, data = (config.get(key, {}) for key in ("experiment", "training", "data"))
    training = training or config.get("pretraining", {})
    unit = str(payload.get("unit", data.get("label_unit", "hartree"))).lower()
    if unit != "hartree":
        raise ValueError("The Feishu metric columns require raw Hartree labels")
    latest, peaks = state.get("latest", {}), state.get("peaks", {})
    progress = read_progress(run_dir)
    host = state.get("host", socket.gethostname())
    run_id = state.get("run_id") or hashlib.sha256(f"{host}:{run_dir}".encode()).hexdigest()[:32]
    status = state.get("status", "completed" if payload else "unknown")
    diagnostic = bool(payload.get("smoke") or config.get("smoke_suite") or state.get("diagnostic"))
    rmse, mse = finite(source.get("rmse")), finite(source.get("mse"))
    if mse is None and rmse is not None:
        mse = finite(rmse**2)
    fields = {
        "运行ID": run_id,
        "模型": str(experiment.get("model_id", experiment.get("name", run_dir.parent.name))),
        "协议": str(experiment.get("protocol_id") or "unspecified"),
        "状态": status,
        "目标": target,
        "单位": unit,
        "主机": host,
        "运行目录": str(run_dir),
        "更新时间": utc_now(),
        "开始时间": state.get("started_at"),
        "结束时间": state.get("ended_at"),
        "采样覆盖": state.get("coverage", "not_measured"),
        "显存可用性": "available" if latest.get("gpu_process_mib") is not None else "unavailable",
        "诊断运行": "yes" if diagnostic else "no",
        "标签缩放": str(
            training.get(
                "target_scaling", config.get("normalization", {}).get("method", "unspecified")
            )
        ),
        "损失函数": str(
            training.get("loss", config.get("loss", {}).get("function", "unspecified"))
        ),
        "训练seed": finite(training.get("seed")),
        "划分seed": finite(data.get("split", {}).get("seed")),
        "batch_size": finite(training.get("batch_size")),
        "epoch预算": finite(training.get("epochs")),
        "学习率": finite(training.get("lr", training.get("init_lr"))),
        "流水线耗时_s": finite(state.get("elapsed_seconds")),
        "训练耗时_s": finite(
            timing.get(
                "fit_seconds", timing.get("fit_pipeline_seconds", pretrain_timing.get("seconds"))
            )
        ),
        "已完成epoch": finite(timing.get("epochs_completed")),
        "当前epoch": progress.get("epoch"),
        "训练loss": progress.get("train_loss"),
        "验证MAE_Hartree": progress.get(
            "val_mae",
            finite(payload.get("best_validation_mae", payload.get("best_validation_delta_e_mae"))),
        ),
        "测试样本数": finite(source.get("n", payload.get("rows"))),
        "测试MSE_Hartree2": mse,
        "测试MAE_Hartree": finite(source.get("mae")),
        "测试RMSE_Hartree": rmse,
        "测试MAPE_pct": finite(source.get("mape")),
        "测试R2": finite(source.get("r2")),
        "采样数": finite(state.get("sample_count")),
        "采样间隔_s": finite(state.get("interval_seconds")),
    }
    inference = read_json(run_dir / "inference_timing.json")
    validation = read_json(run_dir / "validation_metrics.json")
    validation = validation.get("targets", {}).get(target, validation)
    fields.update(
        {
            "推理计时范围": inference.get("scope"),
            "FLOPs测量状态": inference.get("flops_status", "not_measured"),
            "错误说明": state.get("error"),
            "GPU利用率_pct": finite(latest.get("gpu_util_pct")),
            "峰值GPU利用率_pct": finite(peaks.get("gpu_util_pct")),
            "平均GPU利用率_pct": finite(state.get("gpu_util_mean_pct")),
            "系统可用内存_MiB": finite(latest.get("ram_available_mib")),
            "数据盘剩余_GiB": finite(latest.get("disk_free_gib")),
            "推理耗时_s": finite(inference.get("seconds")),
            "推理吞吐_样本每秒": finite(inference.get("samples_per_second")),
            "摊销推理时间_ms每样本": finite(inference.get("amortized_ms_per_sample")),
            "模型参数量": finite(inference.get("parameters")),
            "FLOPs每样本": finite(inference.get("flops")),
            "验证MSE_Hartree2": finite(validation.get("mse")),
            "验证RMSE_Hartree": finite(validation.get("rmse")),
        }
    )
    if fields["验证MAE_Hartree"] is None:
        fields["验证MAE_Hartree"] = finite(validation.get("mae"))
    fields["指标完整精度_JSON"] = json.dumps(
        {
            name: fields[name]
            for name in (
                "验证MAE_Hartree",
                "测试MAE_Hartree",
                "测试MSE_Hartree2",
                "测试RMSE_Hartree",
                "学习率",
            )
        },
        ensure_ascii=False,
        allow_nan=False,
    )
    for metric, current, peak in (
        ("rss_mib", "进程树RSS_MiB", "峰值RSS_MiB"),
        ("pss_mib", "进程树PSS_MiB", "峰值PSS_MiB"),
        ("gpu_process_mib", "进程显存_MiB", "峰值进程显存_MiB"),
        ("gpu_device_mib", "整卡显存_MiB", "峰值整卡显存_MiB"),
    ):
        fields[current], fields[peak] = finite(latest.get(metric)), finite(peaks.get(metric))
    return {
        "run_id": run_id,
        "fields": fields,
        "mse_source": "metrics"
        if finite(source.get("mse")) is not None
        else "rmse_squared"
        if mse is not None
        else "missing",
    }


def find_runs(paths: list[Path]) -> list[Path]:
    found = set()
    for path in paths:
        path = path.resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        if (path / "config.yaml").is_file() or (path / "tracking/state.json").is_file():
            found.add(path)
        for name in ("metrics.json", "tracking/state.json"):
            for item in path.rglob(name):
                run_dir = item.parent if name == "metrics.json" else item.parent.parent
                if (run_dir / "config.yaml").is_file() or (
                    run_dir / "tracking/state.json"
                ).is_file():
                    found.add(run_dir)
    if not found:
        raise FileNotFoundError(
            "No run config/metrics/tracking state found under the supplied paths"
        )
    return sorted(found)
