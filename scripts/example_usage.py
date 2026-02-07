#!/usr/bin/env python
"""
Example usage of the refactored training scripts

This script demonstrates various ways to use the training framework
"""

# Example 1: Train a frozen ResNet50 without attention (like the original models)
print("Example 1: Frozen ResNet50")
print("-" * 80)
print("""
python train.py \\
    --backbone resnet50 \\
    --training-mode frozen \\
    --seed 42 \\
    --epochs 450 \\
    --batch-size 256
""")

# Example 2: Train Xception from scratch with attention
print("\nExample 2: Xception from Scratch with Attention")
print("-" * 80)
print("""
python train.py \\
    --backbone xception \\
    --training-mode scratch \\
    --use-attention \\
    --seed 570 \\
    --epochs 450
""")

# Example 3: Finetune last 10 layers of DenseNet201 with attention
print("\nExample 3: DenseNet201 Finetuned (Last 10 Layers) with Attention")
print("-" * 80)
print("""
python train.py \\
    --backbone densenet201 \\
    --training-mode finetune_last_n \\
    --num-trainable-layers 10 \\
    --use-attention \\
    --seed 1073
""")

# Example 4: Quick test run with smaller settings
print("\nExample 4: Quick Test Run")
print("-" * 80)
print("""
python train.py \\
    --backbone resnet50 \\
    --training-mode frozen \\
    --epochs 5 \\
    --batch-size 32 \\
    --verbose
""")

# Example 5: Custom experiment with CPU
print("\nExample 5: CPU Training")
print("-" * 80)
print("""
python train.py \\
    --backbone mobilenetv3 \\
    --training-mode frozen \\
    --disable-gpu \\
    --epochs 10 \\
    --batch-size 16
""")

# Example 6: All variations for one backbone and seed (useful for ablation studies)
print("\nExample 6: Batch Training - Single Backbone, Multiple Configs")
print("-" * 80)
print("""
# Edit batch_train.py and set:
backbones = ['resnet50']
seeds = [42]
# Then run:
python batch_train.py
""")

# Example 7: Using custom data paths
print("\nExample 7: Custom Data Paths")
print("-" * 80)
print("""
python train.py \\
    --backbone resnet50 \\
    --training-mode frozen \\
    --data-dir /path/to/your/Lines \\
    --csv-file /path/to/your/labels.csv \\
    --output-dir /path/to/save/results
""")

# Example 8: Hyperparameter tuning
print("\nExample 8: Custom Hyperparameters")
print("-" * 80)
print("""
python train.py \\
    --backbone resnet50 \\
    --training-mode finetune_last_n \\
    --num-trainable-layers 5 \\
    --learning-rate 0.0005 \\
    --batch-size 128 \\
    --early-stop-patience 30 \\
    --lr-patience 5 \\
    --epochs 300
""")

# Example 9: Comparing with and without attention
print("\nExample 9: Attention Ablation Study")
print("-" * 80)
print("""
# Without attention:
python train.py --backbone resnet50 --training-mode frozen --seed 42

# With attention:
python train.py --backbone resnet50 --training-mode frozen --use-attention --seed 42
""")

# Example 10: Progressive training strategy
print("\nExample 10: Progressive Training (Multi-Stage)")
print("-" * 80)
print("""
# Stage 1: Train with frozen backbone
python train.py \\
    --backbone resnet50 \\
    --training-mode frozen \\
    --epochs 100 \\
    --experiment-name resnet50_stage1

# Stage 2: Finetune last 5 layers
python train.py \\
    --backbone resnet50 \\
    --training-mode finetune_last_n \\
    --num-trainable-layers 5 \\
    --epochs 100 \\
    --experiment-name resnet50_stage2

# Stage 3: Finetune all layers
python train.py \\
    --backbone resnet50 \\
    --training-mode finetune_all \\
    --epochs 50 \\
    --learning-rate 0.0001 \\
    --experiment-name resnet50_stage3
""")

print("\n" + "=" * 80)
print("For more information, see scripts/README.md")
print("=" * 80)
