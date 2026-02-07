# Quick Start Guide

Get started with the refactored Writer Identification training scripts in 5 minutes!

## 1. Installation

First, install the required dependencies:

```bash
cd scripts/
pip install -r requirements.txt
```

Or if you're using conda:

```bash
conda activate your_environment_name
pip install -r requirements.txt
```

## 2. Verify Setup

Check that everything is installed correctly:

```bash
python validate_setup.py
```

If you see all green checkmarks (✓), you're ready to go!

## 3. Run Your First Training

Train a simple model to test the setup:

```bash
python train.py \
    --backbone resnet50 \
    --training-mode frozen \
    --epochs 5 \
    --batch-size 32 \
    --verbose
```

This will train for just 5 epochs with a small batch size to verify everything works.

## 4. Run a Full Experiment

Once you've verified the setup works, run a full experiment:

```bash
python train.py \
    --backbone resnet50 \
    --training-mode frozen \
    --seed 42 \
    --epochs 450 \
    --batch-size 256
```

## 5. Check Results

Results are saved in the `Results/{backbone}/` directory:

```bash
ls ../Results/resnet50/
```

You'll find:
- Model checkpoints (`.keras` files)
- Training history (`.pkl` file)
- Test metrics (`.json` file)
- Plots (`.png` files)
- Classification report (`.csv` file)

## Common Commands

### View All Options
```bash
python train.py --help
```

### Train Different Backbones
```bash
# ResNet50
python train.py --backbone resnet50 --training-mode frozen

# DenseNet201
python train.py --backbone densenet201 --training-mode frozen

# Xception
python train.py --backbone xception --training-mode frozen

# MobileNetV3
python train.py --backbone mobilenetv3 --training-mode frozen
```

### Different Training Strategies
```bash
# Frozen (transfer learning)
python train.py --training-mode frozen

# From scratch
python train.py --training-mode scratch

# Finetune last 10 layers
python train.py --training-mode finetune_last_n --num-trainable-layers 10

# Finetune all layers
python train.py --training-mode finetune_all
```

### With/Without Attention
```bash
# Without attention
python train.py --backbone resnet50 --training-mode frozen

# With attention
python train.py --backbone resnet50 --training-mode frozen --use-attention
```

### Different Seeds
```bash
python train.py --seed 42
python train.py --seed 570
python train.py --seed 1073
```

## Batch Training

To run multiple experiments automatically:

```bash
python batch_train.py
```

**Warning:** This will run many experiments and take a long time!

## Replicate Original Experiments

To replicate a specific experiment from the `models/` folder, see `MIGRATION_GUIDE.md`.

For example:
- **Old:** `models/ResNet50/ResNet50 + Frozen + No Attention/...seed 42...py`
- **New:** `python train.py --backbone resnet50 --training-mode frozen --seed 42`

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python train.py --batch-size 64  # or even smaller
```

### No GPU
```bash
# Use CPU
python train.py --disable-gpu --batch-size 32
```

### Data Not Found
```bash
# Specify custom paths
python train.py --data-dir /path/to/Lines --csv-file /path/to/merged_writer.csv
```

## Next Steps

1. **Read the full documentation:** `README.md`
2. **See usage examples:** `example_usage.py`
3. **Learn migration from old scripts:** `MIGRATION_GUIDE.md`

## Summary

```bash
# Quick test (5 minutes)
python train.py --epochs 5 --batch-size 32

# Full experiment (several hours)
python train.py --backbone resnet50 --training-mode frozen --seed 42

# All experiments (days/weeks)
python batch_train.py
```

That's it! You're ready to train writer identification models. 🚀
