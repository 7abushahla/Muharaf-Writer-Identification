# Writer Identification Training Scripts

This directory contains refactored, modular training scripts for the Writer Identification project. All the functionality from the individual model files in the `models/` folder has been consolidated into these reusable scripts with command-line argument support.

## Structure

```
scripts/
├── train.py                  # Main training script
├── aggregate_results.py      # Aggregate results across multiple seeds
├── run_three_seeds.sh        # Helper script to run all 3 seeds + aggregate
├── batch_train.py            # Batch training for multiple experiments
├── model_builder.py          # Model architecture builder
├── data_utils.py             # Data loading and preprocessing
├── custom_layers.py          # Custom Keras layers (SPP, NetVLAD, etc.)
├── custom_metrics.py         # Custom metrics (Macro F1, Precision, Recall)
├── custom_callbacks.py       # Custom training callbacks
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Or if using conda:

```bash
conda env create -f ../PROJtfgpu310.yml
conda activate your_env_name
```

## Usage

### Single Training Run

The main training script supports comprehensive command-line arguments:

```bash
python train.py --backbone resnet50 \
                --training-mode frozen \
                --seed 42 \
                --epochs 450 \
                --batch-size 256
```

### Command-Line Arguments

#### Model Configuration
- `--backbone`: Choose backbone architecture
  - Options: `resnet50`, `densenet201`, `xception`, `mobilenetv3`
  - Default: `resnet50`

- `--training-mode`: Training strategy
  - Options:
    - `frozen`: Freeze all backbone layers (transfer learning)
    - `scratch`: Train from scratch (no pretrained weights)
    - `finetune_last_n`: Finetune last N layers
    - `finetune_all`: Finetune all layers
  - Default: `frozen`

- `--num-trainable-layers`: Number of layers to finetune
  - Required when using `finetune_last_n` mode
  - Example: `--num-trainable-layers 10`

- `--use-attention`: Enable attention mechanism
  - Flag (no value needed)
  - Default: False

- `--num-clusters`: Number of clusters for NetVLAD
  - Default: `64`

#### Data Configuration
- `--data-dir`: Directory containing image data
  - Default: `./Lines`

- `--csv-file`: CSV file with writer labels
  - Default: `merged_writer.csv`

- `--image-size`: Image size (square)
  - Default: `224`

- `--num-classes`: Number of writer classes
  - Default: Auto-detected from data
  - Usually `179`, but may vary with page-disjoint splits

- `--split-mode`: Data splitting strategy
  - Options:
    - `line`: Line-level splits (70/15/15 random split)
    - `page_disjoint`: Page-disjoint or document-disjoint splits
  - Default: `line`
  - See [PAGE_DISJOINT_GUIDE.md](PAGE_DISJOINT_GUIDE.md) for details

- `--split-dir`: Directory containing disjoint split files
  - Default: `./splits`
  - Required when using `--split-mode page_disjoint`

- `--disjoint-mode`: Disjoint splitting granularity
  - Options:
    - `page`: Page-disjoint (all lines from same page stay together)
    - `document`: Document-disjoint (all pages from same document stay together)
  - Default: `page`
  - Only used when `--split-mode page_disjoint`

- `--writer-policy`: Writer filtering policy for splits
  - Options:
    - `None`: No filtering (default)
    - `require_3way`: Only writers with >=3 pages/documents
    - `drop_if_lt2`: Drop writers with <2 pages/documents
    - `drop_if_lt3`: Drop writers with <3 pages/documents
    - `allow_train_test_only`: Allow writers with 2 pages/documents
  - Default: `None`
  - Only used when `--split-mode page_disjoint`

#### Training Configuration
- `--seed`: Random seed for reproducibility
  - Default: `42`

- `--epochs`: Number of training epochs
  - Default: `450`

- `--batch-size`: Batch size
  - Default: `256`

- `--learning-rate`: Initial learning rate
  - Default: `0.001`

- `--early-stop-patience`: Early stopping patience (epochs)
  - Default: `50`

- `--lr-patience`: Learning rate reduction patience (epochs)
  - Default: `10`

#### GPU Configuration
- `--gpu`: GPU device ID to use
  - Default: `0`

- `--disable-gpu`: Disable GPU and use CPU
  - Flag

#### Output Configuration
- `--output-dir`: Directory to save results
  - Default: `./Results`

- `--save-freq`: Save model every N epochs
  - Default: `50`

- `--experiment-name`: Custom experiment name
  - Auto-generated if not specified

#### Other Options
- `--verbose`: Enable verbose output
  - Flag

- `--no-plots`: Skip generating plots
  - Flag

## Multi-Seed Training and Aggregation

For robust evaluation, it's recommended to run each experiment configuration with **3 random seeds** (42, 570, 1073) and report aggregated results.

### Quick Start: Run All 3 Seeds

Use the helper script to automatically run all 3 seeds and aggregate results:

```bash
./run_three_seeds.sh \
  --backbone densenet201 \
  --training-mode frozen \
  --split-mode page_disjoint \
  --disjoint-mode page
```

This will:
1. Train with seeds 42, 570, and 1073
2. Automatically aggregate results
3. Compute mean ± std (population std with N denominator)

### Manual Seed Runs

If you prefer to run seeds individually:

```bash
# Train with each seed
python train.py --backbone densenet201 --training-mode frozen --seed 42
python train.py --backbone densenet201 --training-mode frozen --seed 570
python train.py --backbone densenet201 --training-mode frozen --seed 1073

# Then aggregate results
python aggregate_results.py --all
```

### Aggregation Script Usage

**Aggregate all configurations:**
```bash
python aggregate_results.py --all
```

**Aggregate specific backbone:**
```bash
python aggregate_results.py --backbone densenet201
```

**Aggregate specific configuration:**
```bash
python aggregate_results.py Results/densenet201/PgDisj_densenet201_Frozen_NoATTN
```

### Output Structure

Results are organized by configuration, with each seed in its own subdirectory:

```
Results/
  densenet201/
    PgDisj_densenet201_Frozen_NoATTN/
      ├── seed_42/
      │   ├── best_model.keras
      │   ├── test_metrics.json
      │   ├── classification_report.csv
      │   └── ... (plots, history, etc.)
      ├── seed_570/
      │   └── ...
      ├── seed_1073/
      │   └── ...
      └── aggregated/
          ├── aggregated_metrics.json          # Mean ± Std for all metrics
          └── classification_report_aggregated.csv
```

### Aggregated Metrics Format

**`aggregated_metrics.json`** contains:
- `metrics_mean`: Mean values across seeds
- `metrics_std`: Population standard deviation (N denominator)
- `metrics_formatted`: Human-readable "mean ± std" format

Example:
```json
{
  "seeds": ["42", "570", "1073"],
  "num_seeds": 3,
  "metrics_formatted": {
    "test_accuracy": "0.2446 ± 0.0123",
    "test_f1_score": "0.1387 ± 0.0098",
    "test_precision": "0.0985 ± 0.0067",
    "test_recall": "0.2342 ± 0.0145"
  }
}
```

**Terminal Output:**
```
================================================================================
AGGREGATED RESULTS ACROSS 3 SEEDS: 42, 570, 1073
================================================================================

Test Metrics (Mean ± Std):
--------------------------------------------------
test_accuracy       : 0.2446 ± 0.0123
test_top_5_accuracy : 0.4419 ± 0.0201
test_precision      : 0.0985 ± 0.0067
test_recall         : 0.2342 ± 0.0145
test_f1_score       : 0.1387 ± 0.0098
test_loss           : 3.3812 ± 0.1523
================================================================================
```

## Examples

### Example 1: Frozen ResNet50 without Attention

```bash
python train.py --backbone resnet50 \
                --training-mode frozen \
                --seed 42
```

### Example 2: From Scratch Xception with Attention

```bash
python train.py --backbone xception \
                --training-mode scratch \
                --use-attention \
                --seed 570
```

### Example 3: Finetune Last 10 Layers of DenseNet201

```bash
python train.py --backbone densenet201 \
                --training-mode finetune_last_n \
                --num-trainable-layers 10 \
                --use-attention \
                --seed 1073
```

### Example 4: MobileNetV3 with All Layers Finetuned

```bash
python train.py --backbone mobilenetv3 \
                --training-mode finetune_all \
                --epochs 300 \
                --batch-size 128 \
                --learning-rate 0.0005
```

### Example 5: Page-Disjoint Splits (Realistic Evaluation)

**Important:** First generate the splits using `page_disjoint_splits.py` in the project root:

```bash
# From project root
cd ..
python page_disjoint_splits.py --seed 42
cd scripts/
```

Then train with page-disjoint mode:

```bash
python train.py --backbone resnet50 \
                --training-mode frozen \
                --use-attention \
                --seed 42 \
                --split-mode page_disjoint \
                --split-dir ../splits
```

The experiment name will be prefixed with `PD_` to indicate page-disjoint mode:
- Output: `PD_resnet50_Frozen_ATTN_seed42_best_model.keras`

See [PAGE_DISJOINT_GUIDE.md](PAGE_DISJOINT_GUIDE.md) for complete documentation.

### Example 5: Quick Test Run

```bash
python train.py --backbone resnet50 \
                --training-mode frozen \
                --epochs 10 \
                --batch-size 64 \
                --verbose
```

## Batch Training

To replicate all experiments from the original `models/` folder:

```bash
python batch_train.py
```

This will run all combinations of:
- 4 backbones (ResNet50, DenseNet201, Xception, MobileNetV3)
- 3 seeds (42, 570, 1073)
- Multiple training configurations (frozen, scratch, finetune variations)
- With and without attention

**Note:** This will take a very long time! You can edit `batch_train.py` to run a subset of experiments.

### Batch Training with Page/Document-Disjoint Splits

If you want **all batch experiments** to use page- or document-disjoint splits, edit
the `split_config` block in `batch_train.py`:

```python
split_config = {
    'split-mode': 'page_disjoint',
    'disjoint-mode': 'document',  # or 'page'
    'writer-policy': 'require_3way',
    'split-dir': '../splits',
}
```

This appends the split options to every experiment (across all backbones/seeds).

## Output Files

Each training run creates files organized by configuration and seed:

```
Results/{backbone}/{config_name}/
├── seed_{seed}/
│   ├── best_model.keras               # Best model checkpoint
│   ├── last_saved.keras               # Last saved checkpoint
│   ├── history.pkl                    # Training history
│   ├── test_metrics.json              # Test set metrics
│   ├── elapsed_time.txt               # Training time
│   ├── config.json                    # Experiment configuration
│   ├── classification_report.csv      # Per-class metrics
│   ├── confusion_matrix.png           # Confusion matrix plot
│   ├── accuracy.png                   # Accuracy plot
│   ├── top_5_accuracy.png             # Top-5 accuracy plot
│   ├── loss.png                       # Loss plot
│   ├── f1_score.png                   # F1 score plot
│   ├── precision.png                  # Precision plot
│   └── recall.png                     # Recall plot
└── aggregated/                        # Created by aggregate_results.py
    ├── aggregated_metrics.json        # Mean ± Std across seeds
    └── classification_report_aggregated.csv

Example:
Results/densenet201/PgDisj_densenet201_Frozen_NoATTN/
├── seed_42/
├── seed_570/
├── seed_1073/
└── aggregated/
```

## Experiment Naming

Experiments are automatically named based on configuration:

**Configuration name (folder):**
```
[{split_prefix}]{backbone}_{training_mode}_{attention}
```

**Full experiment name (for logs):**
```
[{split_prefix}]{backbone}_{training_mode}_{attention}_seed{seed}
```

Where `split_prefix` is:
- `PgDisj_` for page-disjoint splits (default policy: require_3way, not shown)
- `DocDisj_` for document-disjoint splits
- `PgDisj_{policy}_` for non-default writer policies (e.g., `PgDisj_drop_if_lt3_`)
- Empty for line-level splits

Examples:
- `PgDisj_densenet201_Frozen_NoATTN` (config folder)
  - `seed_42/`, `seed_570/`, `seed_1073/` (subdirectories)
- `DocDisj_resnet50_Scratch_ATTN` (document-disjoint)
- `PgDisj_drop_if_lt3_xception_Finetune10Layers_ATTN` (non-default policy)
- `mobilenetv3_Frozen_NoATTN` (line-level split)

You can override the full name with `--experiment-name`:

```bash
python train.py --experiment-name my_custom_experiment ...
```

## Advantages Over Original Scripts

1. **Single codebase**: All functionality in reusable modules
2. **Command-line interface**: No need to edit Python files
3. **Consistent structure**: All experiments follow the same pattern
4. **Easy experimentation**: Try new configurations instantly
5. **Batch processing**: Run multiple experiments automatically
6. **Better organization**: Structured output files
7. **Reproducibility**: Configuration saved with each experiment
8. **Maintainability**: Fix bugs or add features in one place

## Comparison with Original Structure

### Original (`models/` folder):
- 149 separate Python files
- Each file = one configuration (backbone + training mode + seed)
- Hard to modify or maintain
- Lots of code duplication

### Refactored (`scripts/` folder):
- 8 modular Python files
- All configurations via command-line arguments
- Easy to extend and maintain
- DRY principle (Don't Repeat Yourself)

## Troubleshooting

### Out of Memory Error
Reduce batch size:
```bash
python train.py --batch-size 128  # or even smaller
```

### GPU Not Found
Use CPU:
```bash
python train.py --disable-gpu
```

Or specify different GPU:
```bash
python train.py --gpu 1
```

### Data Not Found
Specify correct paths:
```bash
python train.py --data-dir /path/to/Lines \
                --csv-file /path/to/merged_writer.csv
```

## Citation

If you use this code, please cite the original paper (add citation here).

## License

(Add license information here)
