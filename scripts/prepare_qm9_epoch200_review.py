"""Generate the 114 executable fresh200 configs and explicit stage scripts; never train."""

import copy
import csv
import json
import shlex
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "configs/qm9_epoch200"
DOC = ROOT / "docs/features/qm9-epoch200-modular-study"
SEEDS = (42, 2026, 3407)
TARGETS = ("homo", "lumo", "delta_e", "alpha", "mu", "r2", "zpve")
BACKBONES = {
    "chemprop": ("graph", "configs/baseline.yaml", "2D"),
    "schnet": ("graph", "configs/qm9_models/schnet.yaml", "3D"),
    "3dinfomax": ("graph", "configs/qm9_models/3dinfomax.yaml", "3D pretrain / 2D inference"),
    "egnn": ("graph", "configs/qm9_models/egnn.yaml", "3D"),
    "painn": ("graph", "configs/qm9_models/painn.yaml", "3D"),
    "dimenet_pp": ("graph", "configs/qm9_models/dimenet_pp.yaml", "3D"),
    "smiles_transformer": ("transformer", "configs/transformer.yaml", "2D"),
    "spe_transformer": ("transformer", "configs/spe_transformer.yaml", "2D"),
    "esa": ("transformer", "configs/qm9_models/esa.yaml", "2D graph"),
    "transformer_m": ("transformer", "configs/qm9_models/transformer_m.yaml", "3D"),
    "torchmd_net": ("transformer", "configs/qm9_models/torchmd_net.yaml", "3D"),
    "demol": ("transformer", "configs/qm9_models/demol.yaml", "3D"),
}
PRIORITY = (
    "chemprop",
    "dimenet_pp",
    "demol",
    "painn",
    "egnn",
    "torchmd_net",
    "transformer_m",
    "3dinfomax",
    "schnet",
    "esa",
    "spe_transformer",
    "smiles_transformer",
)
MODULES = ("task_attention7", "physics_thermo10")


def make_executable_config(model, variant, seed, target=None):
    from molgap.modular.config import PHYSICS, PROTOCOL, validate
    from molgap.modular.physics import TASKS7, TASKS10, UNITS10

    family, source_path, input_type = BACKBONES[model]
    if variant in {"D1", "D2"}:
        old = (
            "chemprop_mtl7_uncertainty" if variant == "D1" else "chemprop_mtl7_affinity_uncertainty"
        )
        source_path = f"configs/qm9_multitask7/models/{old}.yaml"
    source = yaml.safe_load((ROOT / source_path).read_text())
    if model == "3dinfomax":
        source["model"]["params"]["pretrain_checkpoint"] = (
            f"artifacts/qm9_epoch200/pretrained/3dinfomax_seed{seed}.pt"
        )
    tasks = (
        [target]
        if variant == "single_task"
        else list(TASKS10 if variant == "physics_thermo10" else TASKS7)
    )
    selection = target if variant == "single_task" else "delta_e"
    split = copy.deepcopy(source["data"]["split"])
    split["seed"] = seed
    data_base = "qm9_thermo10_v1" if variant == "physics_thermo10" else "qm9_single_properties_v1"
    cfg = dict(
        status="implemented_pending_gpu_smoke",
        experiment=dict(
            name=f"{model}_{variant}" + (f"_{target}" if target else ""),
            model_id=f"{model}_{variant}",
            protocol_id=PROTOCOL,
        ),
        backbone=dict(model=model, family=family, input=input_type),
        source_config=source_path,
        source=source,
        data=dict(
            smiles_column="smiles",
            id_column="sample_id",
            processed_dir=f"data/processed/{data_base}/random/seed{seed}",
            split=split,
            targets=tasks,
            units={t: UNITS10[t] for t in tasks},
            selection_target=selection,
        ),
        training=dict(
            epochs=200,
            batch_size=256,
            patience=20,
            min_delta=0.0,
            checkpoint_every_n_epochs=50,
            seed=seed,
            split_seed=seed,
            run_mode="paired_split_seed",
            seed_pairing="matched",
            num_workers=2,
            pin_memory=True,
            target_scaling="train_only_per_target_zscore",
            supervised_loss="mse",
            selection_metric=f"val_{selection}_mae",
        ),
        module=dict(
            variant=variant, hidden_dim=128, num_heads=4, dropout=0.1, physics=dict(PHYSICS)
        ),
        output=dict(root="artifacts/qm9_epoch200/fresh"),
        evaluation=dict(
            report_tasks=[t for t in tasks if t in TASKS7],
            metrics=["mae", "mse", "rmse", "r2"],
            original_units=True,
        ),
    )
    validate(cfg)
    return cfg


def write_yaml(path, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    return str(path.relative_to(ROOT))


def main():
    rows = []

    def add(model, variant, seed, target=None):
        stage = 1 if model == "chemprop" else 2
        cfg = make_executable_config(model, variant, seed, target)
        if model == "chemprop" and variant == "single_task":
            path = DEST / "fresh/chemprop" / f"{target}_seed{seed}.yaml"
        elif model == "chemprop" and variant in {"D1", "D2"}:
            path = DEST / "fresh/chemprop_variants" / f"{variant}_seed{seed}.yaml"
        else:
            path = DEST / "fresh" / model / f"{variant}_seed{seed}.yaml"
        path = write_yaml(path, cfg)
        command = shlex.join(
            ["uv", "run", "python", "-m", "molgap.modular.train", "--config", path]
        )
        rows.append(
            dict(
                id=f"{model}_{variant}" + (f"_{target}" if target else "") + f"_seed{seed}",
                stage=stage,
                model=model,
                variant=variant,
                target=target or ("ten_joint" if variant == "physics_thermo10" else "seven_joint"),
                seed=seed,
                epochs=200,
                config=path,
                status="ready_pending_gpu_smoke",
                change=(
                    "original architecture; train Z-score/MSE; fresh200"
                    if variant == "single_task"
                    else "preserve historical head/UW"
                    if variant in {"D1", "D2"}
                    else "task attention on seven task tokens"
                    if variant == "task_attention7"
                    else "independent ten heads + original-unit physical residuals"
                ),
                command=command,
                backbone_priority=PRIORITY.index(model) + 1,
                selection_metric=cfg["training"]["selection_metric"],
                requirements=(
                    "paired QM9 pretrain checkpoint, pinned sources, optional GPU dependencies"
                    if model == "3dinfomax"
                    else "label data; source-specific inputs/dependencies; GPU smoke"
                ),
                method_spec="docs/features/qm9-epoch200-modular-study/model_and_module_lists.md",
            )
        )

    for model in PRIORITY:
        if model == "chemprop":
            for target in TARGETS:
                for seed in SEEDS:
                    add(model, "single_task", seed, target)
            for variant in ("D1", "D2"):
                for seed in SEEDS:
                    add(model, variant, seed)
        elif BACKBONES[model][0] == "graph":
            for seed in SEEDS:
                add(model, "single_task", seed, "delta_e")
        for variant in MODULES:
            for seed in SEEDS:
                add(model, variant, seed)
    DOC.mkdir(parents=True, exist_ok=True)
    with (DOC / "training_queue.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for stage, filename in [(1, "fresh_commands.sh"), (2, "stage2_commands.sh")]:
        selected = [r for r in rows if r["stage"] == stage]
        (DOC / filename).write_text(
            "#!/usr/bin/env bash\nset -uo pipefail\n"
            "# One task per GPU; set CUDA_VISIBLE_DEVICES=1 to select the second card.\n"
            'export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"\n'
            'export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"\n'
            "# Explicit stage reference; failures do not cancel later tasks.\n"
            "failed=0\nrun_job() {\n"
            '  if "$@"; then return 0; else failed=$((failed+1)); return 0; fi\n}\n'
            + "\n".join("run_job " + r["command"] for r in selected)
            + "\n((failed == 0))\n"
        )
    summary = dict(
        total_planned_jobs=len(rows),
        stage1_jobs=sum(r["stage"] == 1 for r in rows),
        ready_configs=len(rows),
        stage2_jobs=sum(r["stage"] == 2 for r in rows),
        stage2_graph_gap_baselines=15,
        stage2_plugin_jobs=66,
        transformer_base_retraining_default=False,
        excluded_variants=["B", "C", "mtl7", "physics_gap7", "mtl10_control"],
        stage2_requires_review=True,
        all_epochs=200,
        modules_implemented=True,
        gpu_training_verified=False,
        training_started=False,
        inputs_and_dependencies_require_preflight=True,
    )
    (DOC / "queue_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    main()
