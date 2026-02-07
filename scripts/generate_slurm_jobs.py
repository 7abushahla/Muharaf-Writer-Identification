#!/usr/bin/env python3
"""
Generate SLURM batch scripts for the full experimental grid.

Creates one job per configuration (14 total), each running:
- 4 backbones (ResNet50, DenseNet201, Xception, MobileNetV3-Large)
- 3 seeds (42, 570, 1073)
- Automatic GPU memory cleanup between runs
- Automatic result aggregation at the end
"""

import os
from pathlib import Path


# Configuration matrix
CONFIGS = [
    {"name": "Frozen_NoATTN", "mode": "frozen", "layers": None, "attention": False},
    {"name": "Frozen_ATTN", "mode": "frozen", "layers": None, "attention": True},
    {"name": "FinetuneLast1_NoATTN", "mode": "finetune_last_n", "layers": 1, "attention": False},
    {"name": "FinetuneLast1_ATTN", "mode": "finetune_last_n", "layers": 1, "attention": True},
    {"name": "FinetuneLast5_NoATTN", "mode": "finetune_last_n", "layers": 5, "attention": False},
    {"name": "FinetuneLast5_ATTN", "mode": "finetune_last_n", "layers": 5, "attention": True},
    {"name": "FinetuneLast10_NoATTN", "mode": "finetune_last_n", "layers": 10, "attention": False},
    {"name": "FinetuneLast10_ATTN", "mode": "finetune_last_n", "layers": 10, "attention": True},
    {"name": "FinetuneLast25_NoATTN", "mode": "finetune_last_n", "layers": 25, "attention": False},
    {"name": "FinetuneLast25_ATTN", "mode": "finetune_last_n", "layers": 25, "attention": True},
    {"name": "FinetuneAll_NoATTN", "mode": "finetune_all", "layers": None, "attention": False},
    {"name": "FinetuneAll_ATTN", "mode": "finetune_all", "layers": None, "attention": True},
    {"name": "Scratch_NoATTN", "mode": "scratch", "layers": None, "attention": False},
    {"name": "Scratch_ATTN", "mode": "scratch", "layers": None, "attention": True},
]

BACKBONES = ["resnet50", "densenet201", "xception", "mobilenetv3"]
SEEDS = [42, 570, 1073]


def generate_slurm_script(config, config_idx, output_dir="slurm_jobs", 
                          batch_size=256, epochs=450, 
                          account="acc-mialhajri",
                          partition="gpu", 
                          time_limit="500:00:00",
                          conda_env="/shared/b00090279/PROJtfgpu310"):
    """Generate a SLURM batch script for one configuration"""
    
    config_name = config["name"]
    mode = config["mode"]
    layers = config["layers"]
    attention = config["attention"]
    
    # Build training command args
    mode_args = f"--training-mode {mode}"
    if layers is not None:
        mode_args += f" --num-trainable-layers {layers}"
    
    attention_flag = "--use-attention" if attention else ""
    
    # Build config directory name (same logic as train.py generate_experiment_name)
    split_prefix = "PgDisj_"  # Using page_disjoint mode
    attention_str = "ATTN" if attention else "NoATTN"
    
    if mode == 'finetune_last_n':
        mode_str = f"Finetune{layers}Layers"
    else:
        mode_str = mode.capitalize()
    
    # Config name template: {split_prefix}{backbone}_{mode_str}_{attention_str}
    # Will be filled in for each backbone
    
    # Create script content
    script = f"""#!/bin/bash
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time={time_limit}
#SBATCH --job-name={config_name}
#SBATCH --output=logs/{config_name}_%j.out

# Configuration: {config_name}
# Running: {len(BACKBONES)} backbones × {len(SEEDS)} seeds = {len(BACKBONES) * len(SEEDS)} total runs

echo "=========================================="
echo "SLURM Job: {config_name}"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="

# Initialize conda
source ~/.bashrc

# Activate environment
conda activate {conda_env}

# Lock in the interpreter
ENV_PY="$CONDA_PREFIX/bin/python"

# Verify activation
if [[ "$CONDA_DEFAULT_ENV" == "{conda_env}" || "$CONDA_DEFAULT_ENV" == "b00090279" ]]; then
    echo "[YES] Conda environment activated: $CONDA_DEFAULT_ENV"
else
    echo "[NO] Conda environment did NOT activate."
    echo "Current python: $(which python)"
    exit 1
fi

echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "Using Python: $ENV_PY"

# Sanity check
"$ENV_PY" -c "import sys, tensorflow, os; \\
print('sys.executable:', sys.executable); \\
print('CONDA_PREFIX:', os.environ.get('CONDA_PREFIX'));"

# Create logs directory
mkdir -p logs

# Change to scripts directory
cd Muharaf-Writer-Identification/scripts

# GPU cleanup function
cleanup_gpu() {{
    echo "Cleaning up GPU memory..."
    "$ENV_PY" -c "import tensorflow as tf; tf.keras.backend.clear_session()" 2>/dev/null || true
    "$ENV_PY" -c "import gc; gc.collect()" 2>/dev/null || true
    sleep 5
}}

# Common training arguments
COMMON_ARGS="\\
  {mode_args} \\
  {attention_flag} \\
  --split-mode page_disjoint \\
  --disjoint-mode page \\
  --writer-policy require_3way \\
  --split-dir ../splits \\
  --batch-size {batch_size} \\
  --epochs {epochs}"

echo ""
echo "=========================================="
echo "Configuration: {config_name}"
echo "=========================================="
"""

    # Add training runs for each backbone and seed
    for backbone in BACKBONES:
        script += f"""
echo ""
echo "----------------------------------------"
echo "Backbone: {backbone}"
echo "----------------------------------------"
"""
        
        for seed in SEEDS:
            script += f"""
echo "  Seed: {seed}"
"$ENV_PY" -u train.py \\
  --backbone {backbone} \\
  $COMMON_ARGS \\
  --seed {seed}

if [ $? -ne 0 ]; then
    echo "ERROR: Training failed for {backbone} seed {seed}"
    exit 1
fi

cleanup_gpu
"""
        
        # Add aggregation after each backbone completes all seeds
        # Build the specific config directory for this backbone
        config_dir_name = f"{split_prefix}{backbone}_{mode_str}_{attention_str}"
        script += f"""
echo ""
echo "  Aggregating results for {backbone} - {config_name}..."
"$ENV_PY" -u aggregate_results.py Results/{backbone}/{config_dir_name}
"""
    
    # Add final message
    script += f"""
echo ""
echo "=========================================="
echo "All backbones completed!"
echo "=========================================="

echo ""
echo "=========================================="
echo "Job {config_name} completed successfully!"
echo "Finished: $(date)"
echo "=========================================="
"""
    
    # Write script to file
    os.makedirs(output_dir, exist_ok=True)
    script_path = os.path.join(output_dir, f"{config_idx:02d}_{config_name}.sh")
    
    with open(script_path, 'w') as f:
        f.write(script)
    
    # Make executable
    os.chmod(script_path, 0o755)
    
    return script_path


def generate_master_submit_script(output_dir="slurm_jobs"):
    """Generate a master script to submit all jobs"""
    
    script = """#!/bin/bash
# Master script to submit all SLURM jobs for the experimental grid

echo "=========================================="
echo "Submitting all SLURM jobs"
echo "=========================================="

# Create logs directory
mkdir -p logs

# Submit all jobs
"""
    
    for idx, config in enumerate(CONFIGS, start=1):
        config_name = config["name"]
        script += f"""
echo "Submitting job {idx}/{len(CONFIGS)}: {config_name}"
sbatch {idx:02d}_{config_name}.sh
"""
    
    script += """
echo ""
echo "=========================================="
echo "All jobs submitted!"
echo "=========================================="
echo ""
echo "Monitor jobs with: squeue -u $USER"
echo "Check logs in: logs/"
echo ""
"""
    
    # Write script
    script_path = os.path.join(output_dir, "submit_all.sh")
    with open(script_path, 'w') as f:
        f.write(script)
    
    os.chmod(script_path, 0o755)
    
    return script_path


def generate_monitor_script(output_dir="slurm_jobs"):
    """Generate a script to monitor job progress"""
    
    script = """#!/bin/bash
# Monitor SLURM job progress

echo "=========================================="
echo "SLURM Job Status"
echo "=========================================="
squeue -u $USER
echo ""

echo "=========================================="
echo "Completed Jobs (last 20)"
echo "=========================================="
sacct -u $USER --format=JobID,JobName,State,Elapsed,MaxRSS,MaxVMSize -n | tail -20
echo ""

echo "=========================================="
echo "Recent Log Output"
echo "=========================================="
for log in $(ls -t logs/*.out 2>/dev/null | head -5); do
    echo "--- $log (last 5 lines) ---"
    tail -5 "$log"
    echo ""
done
"""
    
    script_path = os.path.join(output_dir, "monitor_jobs.sh")
    with open(script_path, 'w') as f:
        f.write(script)
    
    os.chmod(script_path, 0o755)
    
    return script_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate SLURM batch scripts')
    parser.add_argument('--output-dir', type=str, default='slurm_jobs',
                        help='Output directory for SLURM scripts')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=450,
                        help='Number of epochs')
    parser.add_argument('--account', type=str, default='acc-mialhajri',
                        help='SLURM account')
    parser.add_argument('--partition', type=str, default='gpu',
                        help='SLURM partition')
    parser.add_argument('--time-limit', type=str, default='500:00:00',
                        help='Time limit per job (HH:MM:SS)')
    parser.add_argument('--conda-env', type=str, default='/shared/b00090279/PROJtfgpu310',
                        help='Conda environment path')
    args = parser.parse_args()
    
    print(f"Generating SLURM scripts in: {args.output_dir}/")
    print(f"Configuration: {len(CONFIGS)} configs × {len(BACKBONES)} backbones × {len(SEEDS)} seeds")
    print(f"Total runs per job: {len(BACKBONES) * len(SEEDS)}")
    print()
    
    # Generate individual job scripts
    for idx, config in enumerate(CONFIGS, start=1):
        script_path = generate_slurm_script(
            config, idx, 
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            account=args.account,
            partition=args.partition,
            time_limit=args.time_limit,
            conda_env=args.conda_env
        )
        print(f"  [{idx:2d}/{len(CONFIGS)}] Generated: {script_path}")
    
    print()
    print("========================================")
    print("Setup complete!")
    print("========================================")
    print()
    print("To submit a job:")
    print(f"  cd {args.output_dir}")
    print(f"  sbatch 01_Frozen_NoATTN.sh")
    print()
    print("To monitor jobs:")
    print(f"  squeue -u $USER")
    print(f"  tail -f logs/Frozen_NoATTN_*.out")
    print()


if __name__ == '__main__':
    main()
