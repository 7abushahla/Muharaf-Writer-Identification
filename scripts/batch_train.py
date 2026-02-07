#!/usr/bin/env python
"""
Batch training script to train multiple configurations
This script can be used to replicate the original experiments
"""
import subprocess
import itertools
from pathlib import Path


def run_experiment(config):
    """Run a single training experiment"""
    cmd = ['python', 'train.py']
    
    # Add all configuration parameters
    for key, value in config.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        else:
            cmd.extend([f'--{key}', str(value)])
    
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd)
    return result.returncode


def main():
    """Run batch experiments"""
    
    # Define experiment configurations
    # You can customize this based on what you want to run
    
    backbones = ['resnet50', 'densenet201', 'xception', 'mobilenetv3']
    seeds = [42, 570, 1073]
    
    # Example configurations to replicate original experiments
    training_configs = [
        {'training-mode': 'frozen', 'use-attention': False},
        {'training-mode': 'frozen', 'use-attention': True},
        {'training-mode': 'scratch', 'use-attention': False},
        {'training-mode': 'scratch', 'use-attention': True},
        {'training-mode': 'finetune_last_n', 'num-trainable-layers': 1, 'use-attention': False},
        {'training-mode': 'finetune_last_n', 'num-trainable-layers': 1, 'use-attention': True},
        {'training-mode': 'finetune_last_n', 'num-trainable-layers': 5, 'use-attention': False},
        {'training-mode': 'finetune_last_n', 'num-trainable-layers': 5, 'use-attention': True},
        {'training-mode': 'finetune_last_n', 'num-trainable-layers': 10, 'use-attention': False},
        {'training-mode': 'finetune_last_n', 'num-trainable-layers': 10, 'use-attention': True},
        {'training-mode': 'finetune_last_n', 'num-trainable-layers': 25, 'use-attention': False},
        {'training-mode': 'finetune_last_n', 'num-trainable-layers': 25, 'use-attention': True},
        {'training-mode': 'finetune_all', 'use-attention': False},
        {'training-mode': 'finetune_all', 'use-attention': True},
    ]
    
    # Generate all combinations
    experiments = []
    for backbone in backbones:
        for seed in seeds:
            for train_config in training_configs:
                config = {
                    'backbone': backbone,
                    'seed': seed,
                    **train_config
                }
                experiments.append(config)
    
    print(f"Total experiments to run: {len(experiments)}")
    
    # Run experiments
    failed_experiments = []
    for i, config in enumerate(experiments, 1):
        print(f"\n{'#'*80}")
        print(f"# Experiment {i}/{len(experiments)}")
        print(f"{'#'*80}")
        
        returncode = run_experiment(config)
        
        if returncode != 0:
            failed_experiments.append((i, config))
            print(f"\n[ERROR] Experiment {i} failed with return code {returncode}")
        else:
            print(f"\n[SUCCESS] Experiment {i} completed successfully")
    
    # Summary
    print(f"\n{'='*80}")
    print("BATCH TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"Total experiments: {len(experiments)}")
    print(f"Successful: {len(experiments) - len(failed_experiments)}")
    print(f"Failed: {len(failed_experiments)}")
    
    if failed_experiments:
        print("\nFailed experiments:")
        for exp_num, config in failed_experiments:
            print(f"  Experiment {exp_num}: {config}")
    
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
