#!/bin/bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=500:00:00
#SBATCH --job-name=Scratch_ATTN_r50_s1073
#SBATCH --output=logs/Scratch_ATTN_resnet50_seed1073_%j.out

# Configuration: Scratch_ATTN - ResNet50 seed 1073 only
# Running: 1 backbone × 1 seed = 1 total run

echo "=========================================="
echo "SLURM Job: Scratch_ATTN (ResNet50 seed 1073 only)"
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
  --training-mode scratch \
  --use-attention \
  --split-mode page_disjoint \
  --disjoint-mode page \
  --writer-policy require_3way \
  --split-dir ../splits \
  --batch-size 32 \
  --epochs 450"

echo ""
echo "=========================================="
echo "Configuration: Scratch_ATTN"
echo "=========================================="

echo ""
echo "----------------------------------------"
echo "Backbone: resnet50"
echo "----------------------------------------"

echo "  Seed: 1073"
# Disable XLA to avoid compilation overhead and potential memory issues
export TF_XLA_FLAGS=""
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
echo "  Aggregating results for resnet50 - Scratch_ATTN..."
"$ENV_PY" -u aggregate_results.py Results/resnet50/PgDisj_resnet50_Scratch_ATTN

echo ""
echo "=========================================="
echo "ResNet50 seed 1073 completed!"
echo "=========================================="
echo "Job Scratch_ATTN (ResNet50 seed 1073) completed successfully!"
echo "Finished: $(date)"
echo "=========================================="
