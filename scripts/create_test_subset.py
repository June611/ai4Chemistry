"""Prepare a small subset of QM9 data for EGNN validation (100 molecules)."""
import shutil
from pathlib import Path

import pandas as pd

# Paths
processed_dir = Path("data/processed/qm9_full/random/seed3407")
test_dir = Path("data/processed/qm9_test_100/random/seed3407")

print("Creating small test dataset (100 molecules)...")

# Create test directory
test_dir.mkdir(parents=True, exist_ok=True)

# Load split manifest to get consistent sample_ids
manifest = pd.read_csv(processed_dir / "split_manifest.csv")

# Sample from manifest first to maintain consistency
train_manifest = manifest[manifest["split"] == "train"].head(80).copy()
val_manifest = manifest[manifest["split"] == "val"].head(10).copy()
test_manifest = manifest[manifest["split"] == "test"].head(10).copy()

# Reset split_position to be contiguous from 0 for each split
train_manifest["split_position"] = range(len(train_manifest))
val_manifest["split_position"] = range(len(val_manifest))
test_manifest["split_position"] = range(len(test_manifest))

# Combine and save updated manifest
small_manifest = pd.concat([train_manifest, val_manifest, test_manifest])
small_manifest.to_csv(test_dir / "split_manifest.csv", index=False)
print(f"✓ Split manifest: {len(small_manifest)} samples")

# Now load CSV files and filter by sample_ids
train_df = pd.read_csv(processed_dir / "train.csv")
val_df = pd.read_csv(processed_dir / "val.csv")
test_df = pd.read_csv(processed_dir / "test.csv")

# Filter by sample_ids from manifest
train_ids = train_manifest["sample_id"].tolist()
val_ids = val_manifest["sample_id"].tolist()
test_ids = test_manifest["sample_id"].tolist()

train_small = train_df[train_df["sample_id"].isin(train_ids)]
val_small = val_df[val_df["sample_id"].isin(val_ids)]
test_small = test_df[test_df["sample_id"].isin(test_ids)]

# Reorder to match manifest
train_small = train_small.set_index("sample_id").loc[train_ids].reset_index()
val_small = val_small.set_index("sample_id").loc[val_ids].reset_index()
test_small = test_small.set_index("sample_id").loc[test_ids].reset_index()

# Save
train_small.to_csv(test_dir / "train.csv", index=False)
val_small.to_csv(test_dir / "val.csv", index=False)
test_small.to_csv(test_dir / "test.csv", index=False)

print(f"✓ Train: {len(train_small)} samples")
print(f"✓ Val: {len(val_small)} samples")
print(f"✓ Test: {len(test_small)} samples")

# Copy metadata files (but not split_manifest, we already created it)
for file in ["dataset.csv", "data_lineage.json", "target_statistics.json"]:
    src = processed_dir / file
    if src.exists():
        shutil.copy2(src, test_dir / file)
        print(f"✓ Copied {file}")

print(f"\n✅ Test dataset created at {test_dir}")
print(f"   Train positions: 0-{len(train_small)-1}")
print(f"   Val positions: 0-{len(val_small)-1}")
print(f"   Test positions: 0-{len(test_small)-1}")
