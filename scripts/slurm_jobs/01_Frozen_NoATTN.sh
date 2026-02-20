#!/bin/bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=500:00:00
#SBATCH --job-name=Frozen_NoATTN
#SBATCH --output=logs/Frozen_NoATTN_%j.out

# Configuration: Frozen_NoATTN
# Running: 4 backbones × 3 seeds = 12 total runs

echo "=========================================="
echo "SLURM Job: Frozen_NoATTN"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="

# Initialize conda
source ~/.bashrc

# Activate environment
conda activate /shared/b00090279/PROJtfgpu310

# Lock in the interpreter
ENV_PY="$CONDA_PREFIX/bin/python"

# Verify activation
if [[ "$CONDA_DEFAULT_ENV" == "/shared/b00090279/PROJtfgpu310" || "$CONDA_DEFAULT_ENV" == "b00090279" ]]; then
    echo "[YES] Conda environment activated: $CONDA_DEFAULT_ENV"
else
    echo "[NO] Conda environment did NOT activate."
    echo "Current python: $(which python)"
    exit 1
fi

echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "Using Python: $ENV_PY"

# Sanity check
"$ENV_PY" -c "import sys, tensorflow, os; \
print('sys.executable:', sys.executable); \
print('CONDA_PREFIX:', os.environ.get('CONDA_PREFIX'));"

# Create logs directory
mkdir -p logs

# Change to scripts directory
cd Muharaf-Writer-Identification/scripts

# GPU cleanup function
cleanup_gpu() {
    echo "Cleaning up GPU memory..."
    
    # Force TensorFlow to release all GPU memory by resetting default graph
    "$ENV_PY" << 'CLEANUP_SCRIPT'
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# Reset TensorFlow state
tf.keras.backend.clear_session()

# Force garbage collection
import gc
gc.collect()

# Reset CUDA memory allocator by creating and destroying a tiny session
# This forces CUDA to defragment and release memory back to the driver
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        # Create minimal tensor to initialize CUDA context, then clear
        _ = tf.constant([[1.0]])
        tf.keras.backend.clear_session()
        gc.collect()
        print("GPU memory cleanup completed")
    except Exception as e:
        print(f"GPU cleanup warning: {e}")
CLEANUP_SCRIPT
    
    # Give CUDA driver time to complete cleanup
    sleep 5
    
    # Show current GPU memory usage for monitoring
    echo "Current GPU memory usage:"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 || true
    echo ""
}

# Common training arguments
COMMON_ARGS="\
  --training-mode frozen \
   \
  --split-mode page_disjoint \
  --disjoint-mode page \
  --writer-policy require_3way \
  --split-dir ../splits \
  --batch-size 256 \
  --epochs 450"

echo ""
echo "=========================================="
echo "Configuration: Frozen_NoATTN"
echo "=========================================="

echo ""
echo "----------------------------------------"
echo "Backbone: resnet50"
echo "----------------------------------------"

echo "  Seed: 42"
"$ENV_PY" -u train.py \
  --backbone resnet50 \
  $COMMON_ARGS \
  --seed 42

if [ $? -ne 0 ]; then
    echo "ERROR: Training failed for resnet50 seed 42"
    exit 1
fi

cleanup_gpu

echo "  Seed: 570"
"$ENV_PY" -u train.py \
  --backbone resnet50 \
  $COMMON_ARGS \
  --seed 570

if [ $? -ne 0 ]; then
    echo "ERROR: Training failed for resnet50 seed 570"
    exit 1
fi

cleanup_gpu

echo "  Seed: 1073"
"$ENV_PY" -u train.py \
  --backbone resnet50 \
  $COMMON_ARGS \
  --seed 1073

if [ $? -ne 0 ]; then
    echo "ERROR: Training failed for resnet50 seed 1073"
    exit 1
fi

cleanup_gpu

echo ""
echo "  Aggregating results for resnet50 - Frozen_NoATTN..."
"$ENV_PY" -u aggregate_results.py Results/resnet50/PgDisj_resnet50_Frozen_NoATTN

echo ""
echo "----------------------------------------"
echo "Backbone: densenet201"
echo "----------------------------------------"

echo "  Seed: 42"
"$ENV_PY" -u train.py \
  --backbone densenet201 \
  $COMMON_ARGS \
  --seed 42

if [ $? -ne 0 ]; then
    echo "ERROR: Training failed for densenet201 seed 42"
    exit 1
fi

cleanup_gpu

echo "  Seed: 570"
"$ENV_PY" -u train.py \
  --backbone densenet201 \
  $COMMON_ARGS \
  --seed 570

if [ $? -ne 0 ]; then
    echo "ERROR: Training failed for densenet201 seed 570"
    exit 1
fi

cleanup_gpu

echo "  Seed: 1073"
"$ENV_PY" -u train.py \
  --backbone densenet201 \
  $COMMON_ARGS \
  --seed 1073

if [ $? -ne 0 ]; then
    echo "ERROR: Training failed for densenet201 seed 1073"
    exit 1
fi

cleanup_gpu

echo ""
echo "  Aggregating results for densenet201 - Frozen_NoATTN..."
"$ENV_PY" -u aggregate_results.py Results/densenet201/PgDisj_densenet201_Frozen_NoATTN

echo ""
echo "----------------------------------------"
echo "Backbone: xception"
echo "----------------------------------------"

echo "  Seed: 42"
"$ENV_PY" -u train.py \
  --backbone xception \
  $COMMON_ARGS \
  --seed 42

if [ $? -ne 0 ]; then
    echo "ERROR: Training failed for xception seed 42"
    exit 1
fi

cleanup_gpu

echo "  Seed: 570"
"$ENV_PY" -u train.py \
  --backbone xception \
  $COMMON_ARGS \
  --seed 570

if [ $? -ne 0 ]; then
    echo "ERROR: Training failed for xception seed 570"
    exit 1
fi

cleanup_gpu

echo "  Seed: 1073"
"$ENV_PY" -u train.py \
  --backbone xception \
  $COMMON_ARGS \
  --seed 1073

if [ $? -ne 0 ]; then
    echo "ERROR: Training failed for xception seed 1073"
    exit 1
fi

cleanup_gpu

echo ""
echo "  Aggregating results for xception - Frozen_NoATTN..."
"$ENV_PY" -u aggregate_results.py Results/xception/PgDisj_xception_Frozen_NoATTN

echo ""
echo "----------------------------------------"
echo "Backbone: mobilenetv3"
echo "----------------------------------------"

echo "  Seed: 42"
"$ENV_PY" -u train.py \
  --backbone mobilenetv3 \
  $COMMON_ARGS \
  --seed 42

if [ $? -ne 0 ]; then
    echo "ERROR: Training failed for mobilenetv3 seed 42"
    exit 1
fi

cleanup_gpu

echo "  Seed: 570"
"$ENV_PY" -u train.py \
  --backbone mobilenetv3 \
  $COMMON_ARGS \
  --seed 570

if [ $? -ne 0 ]; then
    echo "ERROR: Training failed for mobilenetv3 seed 570"
    exit 1
fi

cleanup_gpu

echo "  Seed: 1073"
"$ENV_PY" -u train.py \
  --backbone mobilenetv3 \
  $COMMON_ARGS \
  --seed 1073

if [ $? -ne 0 ]; then
    echo "ERROR: Training failed for mobilenetv3 seed 1073"
    exit 1
fi

cleanup_gpu

echo ""
echo "  Aggregating results for mobilenetv3 - Frozen_NoATTN..."
"$ENV_PY" -u aggregate_results.py Results/mobilenetv3/PgDisj_mobilenetv3_Frozen_NoATTN

echo ""
echo "=========================================="
echo "All backbones completed!"
echo "=========================================="

echo ""
echo "=========================================="
echo "Job Frozen_NoATTN completed successfully!"
echo "Finished: $(date)"
echo "=========================================="
