#!/bin/bash
# Run training for 3 seeds (42, 570, 1073) and aggregate results
#
# Usage: ./run_three_seeds.sh [train.py arguments]
#
# Example:
#   ./run_three_seeds.sh --backbone densenet201 --training-mode frozen --split-mode page_disjoint

SEEDS=(42 570 1073)

echo "========================================"
echo "Running training for 3 seeds: ${SEEDS[@]}"
echo "========================================"

# Run training for each seed
for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "========================================"
    echo "Starting seed $SEED"
    echo "========================================"
    
    python train.py "$@" --seed $SEED
    
    if [ $? -ne 0 ]; then
        echo "Error: Training failed for seed $SEED"
        exit 1
    fi
done

echo ""
echo "========================================"
echo "All seeds completed successfully!"
echo "========================================"
echo ""
echo "Aggregating results..."

# Aggregate results for the configuration
python aggregate_results.py --all

echo ""
echo "========================================"
echo "Complete! Check Results/ for aggregated results"
echo "========================================"
