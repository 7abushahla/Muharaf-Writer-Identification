# Migration Guide: From Old Scripts to New Scripts

This guide shows how to replicate the original experiments from the `models/` folder using the new refactored scripts.

## Quick Reference

### Original Structure → New Command

The original files followed this naming pattern:
```
{Backbone} {Training Type} {Attention}/{Backbone} - E2E WriterIdent - {Details} - Seed {X} 450 Epochs EarlyStop 50.py
```

The new script uses command-line arguments:
```bash
python train.py --backbone {backbone} --training-mode {mode} [--use-attention] --seed {X}
```

## Training Mode Mapping

| Original Folder Name | New Command Argument |
|---------------------|---------------------|
| `+ Frozen + No Attention` | `--training-mode frozen` |
| `+ Frozen + Attention` | `--training-mode frozen --use-attention` |
| `+ From Scratch + No Attention` | `--training-mode scratch` |
| `+ From Scratch + Attention` | `--training-mode scratch --use-attention` |
| `+ Finetuned ImageNet Last Layer + No Attention` | `--training-mode finetune_last_n --num-trainable-layers 1` |
| `+ Finetuned ImageNet Last Layer + Attention` | `--training-mode finetune_last_n --num-trainable-layers 1 --use-attention` |
| `+ Finetuned ImageNet Last 5 Layers + No Attention` | `--training-mode finetune_last_n --num-trainable-layers 5` |
| `+ Finetuned ImageNet Last 5 Layers + Attention` | `--training-mode finetune_last_n --num-trainable-layers 5 --use-attention` |
| `+ Finetuned ImageNet Last 10 Layers + No Attention` | `--training-mode finetune_last_n --num-trainable-layers 10` |
| `+ Finetuned ImageNet Last 10 Layers + Attention` | `--training-mode finetune_last_n --num-trainable-layers 10 --use-attention` |
| `+ Finetuned ImageNet Last 25 Layers + No Attention` | `--training-mode finetune_last_n --num-trainable-layers 25` |
| `+ Finetuned ImageNet Last 25 Layers + Attention` | `--training-mode finetune_last_n --num-trainable-layers 25 --use-attention` |
| `+ Finetuned All ImageNet + No Attention` | `--training-mode finetune_all` |
| `+ Finetuned All ImageNet + Attention` | `--training-mode finetune_all --use-attention` |

## Backbone Mapping

| Original Name | New Argument |
|--------------|-------------|
| `ResNet50` | `--backbone resnet50` |
| `DenseNet201` | `--backbone densenet201` |
| `Xception` | `--backbone xception` |
| `MobileNetV3` | `--backbone mobilenetv3` |

## Complete Examples

### Example 1: ResNet50 Frozen No Attention

**Original:**
```
models/ResNet50/ResNet50 + Frozen + No Attention/ResNet50 - E2E WriterIdent - Seed 42 450 Epochs EarlyStop 50.py
```

**New:**
```bash
python train.py --backbone resnet50 --training-mode frozen --seed 42
```

### Example 2: Xception From Scratch with Attention

**Original:**
```
models/Xception/Xception + From Scratch + Attention/Xception - E2E WriterIdent - From Scratch - Seed 570 450 Epochs EarlyStop 50.py
```

**New:**
```bash
python train.py --backbone xception --training-mode scratch --use-attention --seed 570
```

### Example 3: DenseNet201 Finetuned Last 10 Layers with Attention

**Original:**
```
models/DenseNet201/DenseNet201 + Finetuned ImageNet Last 10 Layers + Attention/DenseNet201 - E2E WriterIdent - Finetuned Last 10 ImageNet Layers - ATTN - Seed 1073 450 Epochs EarlyStop 50.py
```

**New:**
```bash
python train.py --backbone densenet201 --training-mode finetune_last_n --num-trainable-layers 10 --use-attention --seed 1073
```

## Batch Replication

To replicate ALL original experiments at once:

```bash
python batch_train.py
```

This will run all combinations automatically.

## Output Comparison

### Original Output Files

The original scripts saved files with names like:
```
{Backbone}_classification_report_{seed}.csv
{Backbone}_history_seed_{seed}.pkl
{Backbone}_test_metrics_seed_{seed}.json
```

### New Output Files

The new scripts save files with more descriptive names:
```
{backbone}_{mode}_{attention}_{seed}_classification_report.csv
{backbone}_{mode}_{attention}_{seed}_history.pkl
{backbone}_{mode}_{attention}_{seed}_test_metrics.json
```

Example:
- Original: `ResNet50_classification_report_42.csv`
- New: `resnet50_Frozen_NoATTN_seed42_classification_report.csv`

The new naming is more explicit and prevents filename collisions.

## Key Differences

### Advantages of New Scripts

1. **No code duplication**: One script instead of 149 files
2. **Easy to modify**: Change hyperparameters via command line
3. **Better organization**: Consistent structure across all experiments
4. **Version control friendly**: Smaller repository
5. **Easier debugging**: Fix bugs in one place
6. **Extensibility**: Easy to add new features
7. **Documentation**: Self-documenting via `--help`

### Maintained Compatibility

- Same model architecture
- Same training procedure
- Same data preprocessing
- Same metrics and callbacks
- Same output format (JSON, CSV, PKL)
- Same random seeding

The results should be **identical** to the original scripts (given the same seed).

## Verification

To verify the new scripts work correctly:

1. **Run a single experiment:**
```bash
python train.py --backbone resnet50 --training-mode frozen --seed 42 --epochs 10
```

2. **Compare output shapes:**
Check that the model summary matches the original.

3. **Verify metrics:**
The metrics format should match the original JSON files.

4. **Test all backbones:**
```bash
for backbone in resnet50 densenet201 xception mobilenetv3; do
    python train.py --backbone $backbone --training-mode frozen --epochs 1 --batch-size 32
done
```

## Troubleshooting

### "Module not found" Error

Make sure you're in the `scripts/` directory:
```bash
cd scripts/
python train.py ...
```

Or add the scripts directory to your Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/scripts"
```

### Different Results

If you get different results than the original:
1. Check that you're using the same seed
2. Verify TensorFlow version matches
3. Ensure data preprocessing is identical
4. Check GPU determinism settings

### Performance Issues

If training is slower:
- The new scripts include the same optimizations
- Check GPU is being used: `--verbose` flag will show GPU info
- Verify batch size is the same

## Need Help?

Check:
1. `README.md` - Full documentation
2. `example_usage.py` - Usage examples
3. `python train.py --help` - All available options

## Summary

**Old way (models/ folder):**
- 149 separate Python files
- Manual editing required for each configuration
- Hard to maintain

**New way (scripts/ folder):**
- 1 main script
- Command-line arguments for all options
- Easy to maintain and extend

The new scripts provide **the same functionality** with **much better usability**.
