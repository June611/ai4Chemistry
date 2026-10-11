# QM9 final one-epoch smoke results

Seed/split3407; neural batch256, 1epoch; full fixed QM9 split.
CLI Chemprop uses warmup0 for this diagnostic only. Initial failed attempts are retained.

| Model | Final status | GPU peak GiB | RAM peak PSS GiB | Note |
|---|---|---:|---:|---|
| egnn | success | 9.201 | 1.521 |  |
| schnet | success | 3.072 | 1.604 |  |
| painn | success | 4.658 | 1.603 |  |
| dimenet_pp | success | 21.605 | 1.73 |  |
| esa | success | 3.113 | 1.688 |  |
| torchmd_net | success | 12.881 | 1.736 |  |
| cormorant | failed | 19.15 | 1.238 | Pipeline command exited with code 1; torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 5.94 GiB. GPU 0 has a total capacity of 23.57 GiB of which 4.41 GiB is free. Including non-PyTorch memory, this process has 19.15 GiB memory in use. Of the allocated memory 16.36 GiB is allocated by PyTorch, and 2.50 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables) |
| lie_transformer | failed | 22.859 | 1.333 | Pipeline command exited with code 1; torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.32 GiB. GPU 0 has a total capacity of 23.57 GiB of which 718.81 MiB is free. Including non-PyTorch memory, this process has 22.86 GiB memory in use. Of the allocated memory 22.45 GiB is allocated by PyTorch, and 108.14 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables) |
| alignn | success | 22.234 | 2.171 |  |
| 3dinfomax_pretrain | success | 2.195 | 2.267 |  |
| 3dinfomax | success | 2.139 | 2.165 |  |
| chemprop_single_gap | success | 0.645 | 4.262 | Passed after diagnostic CLI warmup0 retry |
| chemprop_multitask | success | 0.652 | 4.751 | Passed after diagnostic CLI warmup0 retry |
| chemprop_consistency | success | 0.656 | 3.166 |  |
| smiles_transformer | success | 1.184 | 1.912 |  |
| spe_transformer | success | 1.188 | 2.107 |  |
| spe_multitask | success | 1.188 | 2.024 |  |
| spe_multitask_affinity | success | 1.188 | 2.017 |  |
| spe_multitask_consistency | success | 1.188 | 2.023 |  |
| spe_multitask_affinity_consistency | success | 1.188 | 2.031 |  |
| training_mean | success | 0.0 | 0.807 |  |
| xgboost_rdkit29 | success | 0.0 | 0.906 |  |
| random_forest_rdkit29 | success | 0.0 | 2.43 |  |
