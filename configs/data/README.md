# Data configurations

`qm9_full.yaml` prepares the full 133,885-row dataset. `qm9_cn_subset.yaml` prepares the
14,202-row CN subset independently; the two sources must never be concatenated.

Run preprocessing with:

```bash
uv run molgap-prepare --config configs/data/qm9_full.yaml
```

Copy a configuration and set `split.method: scaffold_balanced` for a scaffold split. Labels are
always written in unscaled Hartree values; Chemprop owns target scaling during model training.
