#!/bin/bash
# Batch training script for all hybrid model configurations
# Usage: bash scripts/train_all_hybrid.sh

set -e

DATA_DIR="${DATA_DIR:-~/data/qm9}"
OUTPUT_BASE="${OUTPUT_BASE:-./artifacts/hybrid_models}"
GPUS="${GPUS:-1}"

echo "=========================================="
echo "Hybrid Model Training Queue"
echo "=========================================="
echo "Data directory: $DATA_DIR"
echo "Output base: $OUTPUT_BASE"
echo "GPUs: $GPUS"
echo ""

# Function to train one config
train_config() {
    local config_file=$1
    local config_name=$(basename $config_file .yaml)
    local work_dir=$(dirname $config_file | xargs basename)

    echo "=========================================="
    echo "Training: $work_dir/$config_name"
    echo "Config: $config_file"
    echo "=========================================="

    # Read seeds from YAML
    seeds=$(grep "seeds:" $config_file | sed 's/seeds: \[\(.*\)\]/\1/' | tr ',' ' ')

    for seed in $seeds; do
        echo "  → Seed $seed"

        output_dir="$OUTPUT_BASE/$work_dir/$config_name/seed_$seed"

        python -m molgap.training.train_hybrid \
            --data_dir "$DATA_DIR" \
            --output_dir "$output_dir" \
            --config "$config_file" \
            --seed "$seed" \
            --gpus "$GPUS" \
            2>&1 | tee "$output_dir/train.log"

        echo "  ✓ Seed $seed completed"
    done

    echo "✓ $work_dir/$config_name completed"
    echo ""
}

# Work 1 configs
echo "=========================================="
echo "WORK 1: 2D MPNN + Transformer"
echo "=========================================="

train_config configs/hybrid_models/work1/mpnn_only.yaml
train_config configs/hybrid_models/work1/hybrid_2d.yaml

# Work 2 configs
echo "=========================================="
echo "WORK 2: 3D Geometry Injection"
echo "=========================================="

train_config configs/hybrid_models/work2/lambda0.yaml
train_config configs/hybrid_models/work2/lambda05.yaml
train_config configs/hybrid_models/work2/lambda1_mlp.yaml
train_config configs/hybrid_models/work2/lambda1_fourier.yaml

echo "=========================================="
echo "🎉 All training completed!"
echo "=========================================="
echo "Results saved to: $OUTPUT_BASE"
