# Delta-E split distribution audit

## Scope

This audit compares the original seeded row permutation with deterministic quantile-stratified
random splitting. Both methods use seed 3407 and exact `[0.8, 0.1, 0.1]` sizes. Labels remain in
Hartree and are never transformed. The revised method uses 200 `delta_e` quantile bins.

## Maximum pairwise distances

| Dataset | Stage | KS distance | Standardized mean difference | Jensen-Shannon distance |
|---|---:|---:|---:|---:|
| qm9_full | plain random | 0.008146 | 0.008361 | 0.050537 |
| qm9_full | quantile stratified | 0.001570 | 0.001365 | 0.038967 |
| qm9_cn_subset | plain random | 0.039232 | 0.032841 | 0.089185 |
| qm9_cn_subset | quantile stratified | 0.007564 | 0.001933 | 0.048127 |

The maximum KS distance decreased by about 80.7% for both datasets. The CN subset changed from
failing the audit threshold (`KS <= 0.03`) to passing it. Rare tail observations can still make
the minimum and maximum differ between splits; stratification aligns distributions rather than
copying identical label sets.

## Final descriptive statistics

| Dataset | Split | Rows | Mean | Std | Median | Q05 | Q95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| qm9_full | train | 107,038 | 0.251139 | 0.047491 | 0.249500 | 0.174100 | 0.326600 |
| qm9_full | val | 13,380 | 0.251181 | 0.047674 | 0.249500 | 0.174095 | 0.326700 |
| qm9_full | test | 13,380 | 0.251117 | 0.047470 | 0.249400 | 0.174000 | 0.326600 |
| qm9_cn_subset | train | 11,340 | 0.248202 | 0.043301 | 0.250500 | 0.169195 | 0.314900 |
| qm9_cn_subset | val | 1,417 | 0.248285 | 0.043105 | 0.250500 | 0.169400 | 0.315040 |
| qm9_cn_subset | test | 1,418 | 0.248228 | 0.043464 | 0.250450 | 0.168955 | 0.314930 |

## Artifacts

- `qm9_full/before_random/`: original report and KDE plot.
- `qm9_full/after_quantile_stratified/`: revised report and KDE plot.
- `qm9_cn_subset/before_random/`: original report and KDE plot.
- `qm9_cn_subset/after_quantile_stratified/`: revised report and KDE plot.

`split_manifest.csv` was byte-identical after a complete repeated preprocessing run for both
datasets. Canonical SMILES remain isolated across splits. Scaffold splitting remains available as
a separate mode and continues to enforce scaffold isolation; it does not apply target
stratification because scaffold groups are indivisible.
