"""Generate and execute reproducible multi-seed experiments with bounded concurrency."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from molgap.config import ConfigError, find_project_root, resolve_path


@dataclass(frozen=True)
class RunSpec:
    """One model/training-seed subprocess pipeline on one fixed data split."""

    model: str
    trainer: str
    training_seed: int
    split_seed: int
    config_path: Path
    log_path: Path
    commands: tuple[tuple[str, ...], ...]


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ConfigError(f"YAML root must be a mapping: {path}")
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _validate_matrix(matrix: dict[str, Any], matrix_path: Path) -> None:
    required = {
        "name",
        "split_seed",
        "training_seeds",
        "parallel_jobs",
        "gpu",
        "data_config",
        "models",
        "output",
    }
    missing = sorted(required - set(matrix))
    if missing:
        raise ConfigError(f"Missing multi-seed setting(s): {', '.join(missing)}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(matrix["name"])):
        raise ConfigError("name may contain only letters, numbers, dot, underscore, and hyphen.")
    training_seeds = matrix["training_seeds"]
    if (
        not isinstance(training_seeds, list)
        or len(training_seeds) != 5
        or len(set(map(int, training_seeds))) != 5
    ):
        raise ConfigError("training_seeds must contain exactly five unique integers.")
    try:
        int(matrix["split_seed"])
    except (TypeError, ValueError) as error:
        raise ConfigError("split_seed must be an integer.") from error
    if int(matrix["parallel_jobs"]) <= 0:
        raise ConfigError("parallel_jobs must be positive.")
    models = matrix["models"]
    if not isinstance(models, list) or not models:
        raise ConfigError("models must be a non-empty list.")
    names = [str(item.get("name", "")) for item in models if isinstance(item, dict)]
    if (
        len(names) != len(models)
        or len(set(names)) != len(names)
        or any(not name for name in names)
    ):
        raise ConfigError("Every model requires a unique non-empty name.")
    for model in models:
        if model.get("trainer") not in {"chemprop", "transformer"}:
            raise ConfigError(
                f"Unsupported trainer for {model.get('name')}: {model.get('trainer')}"
            )
        if "config" not in model:
            raise ConfigError(f"Model {model['name']} has no config path.")
    if not isinstance(matrix["output"], dict) or "root" not in matrix["output"]:
        raise ConfigError("output.root is required.")
    root = find_project_root(matrix_path)
    for value in [matrix["data_config"], *(model["config"] for model in models)]:
        if not resolve_path(root, value).is_file():
            raise FileNotFoundError(f"Configured file does not exist: {resolve_path(root, value)}")


def _processed_dir(data_config: dict[str, Any], root: Path) -> Path:
    output_root = resolve_path(root, data_config["output"]["root"])
    return (
        output_root
        / str(data_config["dataset"]["name"])
        / str(data_config["split"]["method"])
        / f"seed{int(data_config['split']['seed'])}"
    )


def _training_seed_config(
    base: dict[str, Any],
    model_name: str,
    training_seed: int,
    data_config: dict[str, Any],
    processed_dir: Path,
    root: Path,
) -> dict[str, Any]:
    generated = copy.deepcopy(base)
    generated["experiment"]["name"] = model_name
    generated["data"]["processed_dir"] = _relative_or_absolute(processed_dir, root)
    generated["data"]["split"] = copy.deepcopy(data_config["split"])
    generated["training"]["seed"] = training_seed
    return generated


def _pipeline_commands(
    trainer: str, config_path: Path, overwrite: bool
) -> tuple[tuple[str, ...], ...]:
    config_args = ("--config", str(config_path))
    if trainer == "transformer":
        command = [sys.executable, "-m", "molgap.training.transformer", *config_args]
        if overwrite:
            command.append("--overwrite")
        return (tuple(command),)
    train = [sys.executable, "-m", "molgap.training.train", *config_args]
    if overwrite:
        train.append("--overwrite")
    return (
        tuple(train),
        (sys.executable, "-m", "molgap.training.predict", *config_args),
        (sys.executable, "-m", "molgap.training.evaluate", *config_args),
    )


def expand_matrix(
    matrix_path: str | Path,
    *,
    overwrite: bool = False,
    selected_models: set[str] | None = None,
) -> tuple[dict[str, Any], Path, Path, list[Path], list[RunSpec]]:
    """Expand base YAML files into training-seed configs sharing one fixed split."""
    matrix_path = Path(matrix_path).resolve()
    matrix = _load_yaml(matrix_path)
    _validate_matrix(matrix, matrix_path)
    root = find_project_root(matrix_path)
    batch_root = resolve_path(root, matrix["output"]["root"]) / str(matrix["name"])
    config_root = batch_root / "generated_configs"
    log_root = batch_root / "logs"
    data_base_path = resolve_path(root, matrix["data_config"])
    data_base = _load_yaml(data_base_path)
    split_seed = int(matrix["split_seed"])
    configured_split_seed = int(data_base["split"]["seed"])
    if configured_split_seed != split_seed:
        raise ConfigError(
            f"split_seed={split_seed} does not match {data_base_path} "
            f"split.seed={configured_split_seed}. Refusing to select a different data split."
        )
    training_seeds = [int(seed) for seed in matrix["training_seeds"]]
    processed_dir = _processed_dir(data_base, root)

    fixed_data_path = config_root / "data" / f"split_seed{split_seed}.yaml"
    _write_yaml(fixed_data_path, copy.deepcopy(data_base))
    preparation_paths = [fixed_data_path]

    specs: list[RunSpec] = []
    for model_entry in matrix["models"]:
        model_name = str(model_entry["name"])
        if selected_models and model_name not in selected_models:
            continue
        trainer = str(model_entry["trainer"])
        base_path = resolve_path(root, model_entry["config"])
        base = _load_yaml(base_path)
        for training_seed in training_seeds:
            generated = _training_seed_config(
                base, model_name, training_seed, data_base, processed_dir, root
            )
            if trainer == "transformer":
                reference_base_path = resolve_path(root, base["data"]["chemprop_reference_config"])
                reference_base = _load_yaml(reference_base_path)
                reference = _training_seed_config(
                    reference_base,
                    str(reference_base["experiment"]["name"]),
                    training_seed,
                    data_base,
                    processed_dir,
                    root,
                )
                reference_path = (
                    config_root / "references" / f"chemprop_training_seed{training_seed}.yaml"
                )
                _write_yaml(reference_path, reference)
                generated["data"]["chemprop_reference_config"] = _relative_or_absolute(
                    reference_path, root
                )
            config_path = config_root / model_name / f"training_seed{training_seed}.yaml"
            _write_yaml(config_path, generated)
            specs.append(
                RunSpec(
                    model=model_name,
                    trainer=trainer,
                    training_seed=training_seed,
                    split_seed=split_seed,
                    config_path=config_path,
                    log_path=log_root / model_name / f"training_seed{training_seed}.log",
                    commands=_pipeline_commands(trainer, config_path, overwrite),
                )
            )
    if selected_models:
        unknown = selected_models - {str(model["name"]) for model in matrix["models"]}
        if unknown:
            raise ConfigError(f"Unknown model selection: {sorted(unknown)}")
    return matrix, root, batch_root, preparation_paths, specs


def _splits_ready(data_config_path: Path, root: Path) -> bool:
    config = _load_yaml(data_config_path)
    processed = _processed_dir(config, root)
    return all(
        (processed / name).is_file()
        for name in ("train.csv", "val.csv", "test.csv", "split_manifest.csv")
    )


def _prepare_splits(paths: list[Path], root: Path, overwrite_data: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        split_seed = int(_load_yaml(path)["split"]["seed"])
        if _splits_ready(path, root) and not overwrite_data:
            records.append(
                {"split_seed": split_seed, "status": "skipped_existing", "config": str(path)}
            )
            continue
        command = [sys.executable, "-m", "molgap.data.preprocess", "--config", str(path)]
        completed = subprocess.run(command, cwd=root, check=False)
        if completed.returncode:
            raise RuntimeError(f"Data preparation failed for split seed {split_seed}.")
        records.append({"split_seed": split_seed, "status": "completed", "config": str(path)})
    return records


def _assert_splits(paths: list[Path], root: Path) -> None:
    missing = [str(path) for path in paths if not _splits_ready(path, root)]
    if missing:
        raise FileNotFoundError(
            "The fixed data split is missing. Re-run with --prepare-data. Config: "
            + ", ".join(missing)
        )


def _run_spec(spec: RunSpec, root: Path, gpu: str | None) -> dict[str, Any]:
    started_at = _timestamp()
    spec.log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    if gpu is not None and str(gpu).lower() not in {"", "none"}:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    command_records: list[dict[str, Any]] = []
    return_code = 0
    with spec.log_path.open("w", encoding="utf-8") as log:
        log.write(
            f"started_at={started_at}\nmodel={spec.model}\n"
            f"training_seed={spec.training_seed}\nsplit_seed={spec.split_seed}\n"
        )
        for command in spec.commands:
            log.write("command=" + json.dumps(command, ensure_ascii=False) + "\n")
            log.flush()
            completed = subprocess.run(
                command,
                cwd=root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            command_records.append({"argv": list(command), "return_code": completed.returncode})
            if completed.returncode:
                return_code = completed.returncode
                break
    return {
        "model": spec.model,
        "trainer": spec.trainer,
        "training_seed": spec.training_seed,
        "split_seed": spec.split_seed,
        "config": str(spec.config_path),
        "log": str(spec.log_path),
        "gpu": gpu,
        "started_at": started_at,
        "finished_at": _timestamp(),
        "status": "completed" if return_code == 0 else "failed",
        "return_code": return_code,
        "commands": command_records,
    }


def run_matrix(
    matrix_path: str | Path,
    *,
    jobs: int | None = None,
    gpu: str | None = None,
    prepare_data: bool = False,
    overwrite_data: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
    selected_models: set[str] | None = None,
) -> dict[str, Any]:
    """Reuse one fixed split and execute model/training-seed pipelines in parallel."""
    matrix, root, batch_root, preparation_paths, specs = expand_matrix(
        matrix_path, overwrite=overwrite, selected_models=selected_models
    )
    concurrency = int(jobs if jobs is not None else matrix["parallel_jobs"])
    if concurrency <= 0:
        raise ConfigError("--jobs must be positive.")
    selected_gpu = str(gpu if gpu is not None else matrix["gpu"])
    plan = {
        "name": matrix["name"],
        "split_seed": int(matrix["split_seed"]),
        "training_seeds": [int(seed) for seed in matrix["training_seeds"]],
        "parallel_jobs": concurrency,
        "gpu": selected_gpu,
        "prepare_data": prepare_data,
        "runs": [
            {
                **asdict(spec),
                "config_path": str(spec.config_path),
                "log_path": str(spec.log_path),
                "commands": [list(command) for command in spec.commands],
            }
            for spec in specs
        ],
    }
    batch_root.mkdir(parents=True, exist_ok=True)
    with (batch_root / "plan.json").open("w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    if dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return plan

    preparations = _prepare_splits(preparation_paths, root, overwrite_data) if prepare_data else []
    _assert_splits(preparation_paths, root)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(_run_spec, spec, root, selected_gpu): spec for spec in specs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"[{result['status'].upper()}] model={result['model']} "
                f"training_seed={result['training_seed']} "
                f"split_seed={result['split_seed']} log={result['log']}"
            )
    results.sort(key=lambda item: (item["model"], item["training_seed"]))
    status = {
        "name": matrix["name"],
        "created_at": _timestamp(),
        "split_seed": int(matrix["split_seed"]),
        "training_seeds": [int(seed) for seed in matrix["training_seeds"]],
        "parallel_jobs": concurrency,
        "gpu": selected_gpu,
        "preparations": preparations,
        "runs": results,
        "completed": sum(item["status"] == "completed" for item in results),
        "failed": sum(item["status"] == "failed" for item in results),
    }
    with (batch_root / "status.json").open("w", encoding="utf-8") as handle:
        json.dump(status, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    if status["failed"]:
        raise RuntimeError(
            f"{status['failed']} training run(s) failed; see {batch_root / 'status.json'}"
        )
    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/multiseed.yaml", help="Matrix YAML")
    parser.add_argument("--jobs", type=int, help="Maximum simultaneous runs on the shared GPU")
    parser.add_argument("--gpu", help="CUDA_VISIBLE_DEVICES value; defaults to matrix YAML")
    parser.add_argument("--model", action="append", dest="models", help="Run only this model")
    parser.add_argument(
        "--prepare-data", action="store_true", help="Build the fixed split if it is missing"
    )
    parser.add_argument(
        "--overwrite-data", action="store_true", help="Rebuild the fixed split even if it exists"
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing model runs")
    parser.add_argument(
        "--dry-run", action="store_true", help="Generate configs and print the plan"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_matrix(
        args.config,
        jobs=args.jobs,
        gpu=args.gpu,
        prepare_data=args.prepare_data,
        overwrite_data=args.overwrite_data,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        selected_models=set(args.models) if args.models else None,
    )


if __name__ == "__main__":
    main()
