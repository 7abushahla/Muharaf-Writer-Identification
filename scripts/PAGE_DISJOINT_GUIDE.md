# Page-Disjoint & Document-Disjoint Splits Guide

This guide explains how to use page-disjoint and document-disjoint splits with the refactored training code.

## Overview

The training system supports two data splitting modes:

1. **Line-level splits** (`--split-mode line`): Traditional 70/15/15 random split at the line level
2. **Page/Document-disjoint splits** (`--split-mode page_disjoint`): Splits where all lines from the same page/document stay together in the same set

Page/document-disjoint splits are more realistic for real-world scenarios since they prevent data leakage from different lines of the same document appearing in both training and test sets.

## Disjoint Modes

### Page-Disjoint Mode
- All lines from the same **page** stay together in one split (train/val/test)
- Each page image is treated as an independent unit
- More granular splitting

### Document-Disjoint Mode
- All pages (and their lines) from the same **document** stay together in one split
- Multiple pages that belong to the same document are kept together
- More realistic evaluation (entire documents are unseen during test)

## Workflow

### Step 1: Generate Disjoint Splits

First, you need to generate the split files using `page_disjoint_splits.py` script:

#### Option A: Page-Disjoint Mode (Default)

```bash
# Navigate to project root
cd /Users/hamza/Documents/GitHub/Muharaf-Writer-Identification

# Generate page-disjoint splits with default policy
python scripts/page_disjoint_splits.py \
    --lines-dir Lines \
    --disjoint-mode page \
    --seeds 42,570,1073
```

#### Option B: Page-Disjoint with Strict Writer Policy

```bash
# Only include writers with >=3 pages (train+val+test)
python scripts/page_disjoint_splits.py \
    --lines-dir Lines \
    --disjoint-mode page \
    --writer-policy require_3way \
    --seeds 42,570,1073
```

#### Option C: Document-Disjoint Mode

```bash
# All pages from same document stay together
python scripts/page_disjoint_splits.py \
    --lines-dir Lines \
    --disjoint-mode document \
    --documents-dir Documents \
    --writer-policy require_3way \
    --seeds 42,570,1073
```

**Writer Policy Options:**
- `allow_train_only` (default): Allow writers with only 1 page/document (train only)
- `require_3way`: Only include writers with >=3 pages/documents (ensures train+val+test presence)
- `drop_if_lt2`: Drop writers with <2 pages/documents
- `drop_if_lt3`: Drop writers with <3 pages/documents
- `allow_train_test_only`: Allow writers with 2 pages/documents (train+test, no val)

**Output Files:**

The script generates files in `splits/` directory with names like:
- Page mode: `page_disjoint_seed_{seed}.csv`
- Page mode with policy: `page_disjoint_{policy}_seed_{seed}.csv`
- Document mode: `page_disjoint_document_seed_{seed}.csv`
- Document mode with policy: `page_disjoint_document_{policy}_seed_{seed}.csv`

Each CSV file contains:
- `page_id`: The page identifier
- `writer`: Writer name
- `split`: Assignment to "train", "val", or "test"
- `line_count`: Number of lines from that page

### Step 2: Train with Disjoint Splits

Use the `--split-mode page_disjoint` flag when training:

#### Example 1: Page-Disjoint Training

```bash
cd scripts/

python train.py \
    --backbone xception \
    --training-mode frozen \
    --use-attention \
    --split-mode page_disjoint \
    --disjoint-mode page \
    --split-dir ../splits \
    --seed 42
```

#### Example 2: Page-Disjoint with Writer Policy

```bash
python train.py \
    --backbone densenet201 \
    --training-mode finetune_last_n \
    --num-trainable-layers 5 \
    --use-attention \
    --split-mode page_disjoint \
    --disjoint-mode page \
    --writer-policy require_3way \
    --split-dir ../splits \
    --seed 42
```

#### Example 3: Document-Disjoint Training

```bash
python train.py \
    --backbone resnet50 \
    --training-mode finetune_all \
    --use-attention \
    --split-mode page_disjoint \
    --disjoint-mode document \
    --writer-policy require_3way \
    --split-dir ../splits \
    --seed 42
```

### Step 3: Run Multiple Seeds

To run experiments across multiple seeds:

```bash
# Page-disjoint
for seed in 42 570 1073; do
    python train.py \
        --backbone xception \
        --training-mode frozen \
        --use-attention \
        --split-mode page_disjoint \
        --disjoint-mode page \
        --split-dir ../splits \
        --seed $seed
done

# Document-disjoint with policy
for seed in 42 570 1073; do
    python train.py \
        --backbone densenet201 \
        --training-mode frozen \
        --use-attention \
        --split-mode page_disjoint \
        --disjoint-mode document \
        --writer-policy require_3way \
        --split-dir ../splits \
        --seed $seed
done
```

## Important Notes

1. **Split Files Must Exist First**: Always generate split files using `page_disjoint_splits.py` before training.

2. **Matching Parameters**: The `--disjoint-mode` and `--writer-policy` arguments in `train.py` must match those used when generating splits.

3. **Seed Consistency**: Use the same seeds across all experiments for reproducibility.

4. **Number of Classes**: The refactored code auto-detects the number of classes from the data. With strict writer policies (like `require_3way`), some writers may be excluded, changing the total number of classes.

5. **Experiment Naming**: 
   - Page-disjoint experiments are prefixed with `PgDisj_`
   - Document-disjoint experiments are prefixed with `DocDisj_`
   - Writer policy is included in the name if specified

## Comparing Line-Level vs Page-Disjoint

To compare both approaches:

```bash
# Line-level (default)
python train.py --backbone xception --training-mode frozen --use-attention --seed 42

# Page-disjoint
python train.py --backbone xception --training-mode frozen --use-attention \
    --split-mode page_disjoint --disjoint-mode page --seed 42

# Document-disjoint
python train.py --backbone xception --training-mode frozen --use-attention \
    --split-mode page_disjoint --disjoint-mode document --writer-policy require_3way --seed 42
```

Page/document-disjoint splits typically show slightly lower performance due to the more realistic evaluation setup, but provide better estimates of real-world generalization.

## Verification

To verify your splits were loaded correctly, check the training output:

```
Loading dataset...
Loaded 45123 images from 179 writers

Splitting dataset (mode: page_disjoint)...
Loaded split file: page_disjoint_document_require_3way_seed_42.csv
Page-disjoint split pages: train=1234, val=234, test=245
Page-disjoint split lines: train=31586, val=6768, test=6769
```

The output shows:
- Which split file was loaded
- Number of pages in each split
- Number of lines in each split

## Troubleshooting

**Error: "Split file not found"**
- Make sure you've generated splits using `page_disjoint_splits.py` first
- Verify the `--split-dir` path is correct
- Check that `--disjoint-mode` and `--writer-policy` match the generated file names

**Error: "Page ID not found in split map"**
- Your split file may be outdated or generated with different data
- Regenerate splits with the current data

**Different number of classes**
- Normal when using writer policies like `require_3way`
- The code auto-detects the correct number of classes from loaded data
