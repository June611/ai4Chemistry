# Pre-throughput configuration snapshot

These files preserve the experiment settings before the RTX 4090 throughput update on 2026-09-29.
They are reference snapshots and are not used by `configs/multiseed.yaml`.

The active configs increase the three Chemprop batch sizes from 64 to 256, scale their learning-rate
schedule by four, use eight loader workers, and align all early-stopping patience values at 20.
