#!/bin/bash

# Local runner for the remaining configurations (no SLURM, no ENV_PY).
# You should manually activate your environment first so that `python`
# points to the correct interpreter.
#
# Runs:
#   1) Finetune all + No Attention  : xception (all seeds)
#   2) Finetune all + Attention     : densenet201, xception (all seeds)
#   3) Scratch + No Attention       : densenet201, xception (all seeds)
#   4) Scratch + Attention          : densenet201, xception (all seeds)
#
# Usage (from repo root):
#   bash scripts/run_remaining.sh
# Optional:
#   BATCH_SIZE=128 bash scripts/run_remaining.sh

set -e

echo "=========================================="
echo "Local Job: Remaining configurations"
echo "Started: $(date)"
echo "=========================================="

# Move to scripts directory relative to this file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs

# Simple TF / GPU cleanup between runs
cleanup_gpu() {
    echo "Cleaning up GPU / TF memory..."
    python - << 'CLEANUP_PY'
import tensorflow as tf
import gc

try:
    tf.keras.backend.clear_session()
except Exception as e:
    print(f"TF cleanup warning: {e}")

gc.collect()
CLEANUP_PY
    sleep 3
}

# Allow overriding batch size via environment variable
BATCH_SIZE="${BATCH_SIZE:-256}"
SEEDS=(42 570 1073)

echo ""
echo "Using batch size: ${BATCH_SIZE}"
echo "Seeds: ${SEEDS[*]}"
echo ""

run_config() {
    local config_name="$1"
    local common_args="$2"
    shift 2
    local backbones=("$@")

    echo ""
    echo "=========================================="
    echo "Configuration: ${config_name}"
    echo "Backbones: ${backbones[*]}"
    echo "=========================================="

    for backbone in "${backbones[@]}"; do
        echo ""
        echo "----------------------------------------"
        echo "Backbone: ${backbone}  (${config_name})"
        echo "----------------------------------------"

        for seed in "${SEEDS[@]}"; do
            echo "  Seed: ${seed}"
            python train.py \
              --backbone "${backbone}" \
              ${common_args} \
              --batch-size "${BATCH_SIZE}" \
              --seed "${seed}"

            if [ $? -ne 0 ]; then
                echo "ERROR: Training failed for ${config_name}, backbone ${backbone}, seed ${seed}"
                exit 1
            fi

            cleanup_gpu
        done

        # Aggregation path mirrors the SLURM jobs
        agg_path=""
        case "${config_name}|${backbone}" in
            "FinetuneAll_NoATTN|xception")
                agg_path="Results/xception/PgDisj_xception_Finetune_all_NoATTN"
                ;;
            "FinetuneAll_ATTN|densenet201")
                agg_path="Results/densenet201/PgDisj_densenet201_Finetune_all_ATTN"
                ;;
            "FinetuneAll_ATTN|xception")
                agg_path="Results/xception/PgDisj_xception_Finetune_all_ATTN"
                ;;
            "Scratch_NoATTN|densenet201")
                agg_path="Results/densenet201/PgDisj_densenet201_Scratch_NoATTN"
                ;;
            "Scratch_NoATTN|xception")
                agg_path="Results/xception/PgDisj_xception_Scratch_NoATTN"
                ;;
            "Scratch_ATTN|densenet201")
                agg_path="Results/densenet201/PgDisj_densenet201_Scratch_ATTN"
                ;;
            "Scratch_ATTN|xception")
                agg_path="Results/xception/PgDisj_xception_Scratch_ATTN"
                ;;
        esac

        if [ -n "${agg_path}" ]; then
            echo ""
            echo "  Aggregating results for ${backbone} - ${config_name}..."
            python aggregate_results.py "${agg_path}"
        else
            echo "  (No aggregation path defined for ${config_name} / ${backbone}, skipping aggregate_results.py)"
        fi
    done
}

###############################################################################
# 1) Finetune all + No Attention (xception only)
###############################################################################
FINETUNE_ALL_NOATTN_ARGS="\
  --training-mode finetune_all \
  --split-mode page_disjoint \
  --disjoint-mode page \
  --writer-policy require_3way \
  --split-dir ../splits \
  --epochs 450"

run_config "FinetuneAll_NoATTN" "${FINETUNE_ALL_NOATTN_ARGS}" xception

###############################################################################
# 2) Finetune all + Attention (densenet201, xception)
###############################################################################
FINETUNE_ALL_ATTN_ARGS="\
  --training-mode finetune_all \
  --use-attention \
  --split-mode page_disjoint \
  --disjoint-mode page \
  --writer-policy require_3way \
  --split-dir ../splits \
  --epochs 450"

run_config "FinetuneAll_ATTN" "${FINETUNE_ALL_ATTN_ARGS}" densenet201 xception

###############################################################################
# 3) Scratch + No Attention (densenet201, xception)
###############################################################################
SCRATCH_NOATTN_ARGS="\
  --training-mode scratch \
  --split-mode page_disjoint \
  --disjoint-mode page \
  --writer-policy require_3way \
  --split-dir ../splits \
  --epochs 450"

run_config "Scratch_NoATTN" "${SCRATCH_NOATTN_ARGS}" densenet201 xception

###############################################################################
# 4) Scratch + Attention (densenet201, xception)
###############################################################################
SCRATCH_ATTN_ARGS="\
  --training-mode scratch \
  --use-attention \
  --split-mode page_disjoint \
  --disjoint-mode page \
  --writer-policy require_3way \
  --split-dir ../splits \
  --epochs 450"

run_config "Scratch_ATTN" "${SCRATCH_ATTN_ARGS}" densenet201 xception

echo ""
echo "=========================================="
echo "All requested remaining configurations completed!"
echo "Finished: $(date)"
echo "=========================================="

