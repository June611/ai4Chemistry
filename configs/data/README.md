# Data configurations

`qm9_full.yaml` prepares the full 133,885-row dataset. `qm9_cn_subset.yaml` prepares the
14,202-row CN subset independently; the two sources must never be concatenated.

Run preprocessing with:

```bash
uv run molgap-prepare --config configs/data/qm9_full.yaml
```

The default random configurations use deterministic quantile stratification on `delta_e` so each
split receives approximately the same regression-target distribution. `split.stratify.bins`
controls the requested number of quantile bins; the effective number is recorded in the manifest.

Copy a configuration and set `split.method: scaffold_balanced`, removing `split.stratify`, for a
scaffold split. Scaffold groups are indivisible, so target stratification is not combined with
scaffold isolation. Labels are always written in unscaled Hartree values; Chemprop owns target
scaling during model training.
