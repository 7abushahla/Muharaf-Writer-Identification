# Writer Identification for Historical Arabic Manuscripts

This repository contains the full implementation and supplementary materials for the paper: **"Different Strokes for Different Folks: Writer Identification for Historical Arabic Manuscripts"**

**Authors:** Hamza Ahmed Abushahla, Ariel Justine Navarro Panopio, Layth Al-Khairulla, Mohamed I. AlHajri

This repository showcases the tools, methods, and results behind our work on writer identification in historical Arabic manuscripts, using the **[Muharaf Dataset](https://github.com/MehreenMehreen/muharaf)**. It provides everything needed to reproduce our experiments and extend the methodology.

### What’s Included:
- A **manual labeling process** showcasing how the dataset was prepared and optimized for the writer identification task, including methods to clean and expand the metadata and ensure consistency.
- **Data preprocessing pipelines** describing the steps to prepare the dataset for training, including image resizing, normalization, data augmentation, and train-validation-test splitting.
- Comprehensive implementations of **baseline and advanced models**, showcasing the architectures used and enhancements like self-attention and cross-attention mechanisms for improved writer identification performance.
- Scripts demonstrating the process for **model training and evaluation**, with detailed metric computations and aids for reproducibility to facilitate experimentation.
- **Benchmark results** summarizing the performance of models and configurations on writer identification tasks, providing insights and pre-trained checkpoints for further research.  
 
This repository serves as a comprehensive resource for researchers interested in handwriting recognition, writer identification, or historical document analysis.

---

## Table of Contents

1. [Manual Labeling Process](#1-manual-labeling-process)  
    - [Dataset Metadata and Excel File](#dataset-metadata-and-excel-file)  
    - [Duplicate Detection](#duplicate-detection)  
    - [Error Corrections and Final Dataset](#error-corrections-and-final-dataset)  
2. [Data Preprocessing](#2-data-preprocessing)  
    - [Resizing and Normalization](#resizing-and-normalization)  
    - [Augmentation Techniques](#augmentation-techniques)  
    - [Train-Test Split](#train-test-split)  
3. [Models and Architectures](#3-models-and-architectures)  
    - [Baseline Model](#baseline-model)  
    - [Variations: Feature Extractors and Attention](#variations-feature-extractors-and-attention)  
4. [Evaluation Metrics](#4-evaluation-metrics)  
5. [Experimental Results](#5-experimental-results)  
6. [Reproducibility and Setup](#6-reproducibility-and-setup)  
    - [Setup and Installation](#setup-and-installation)  
    - [Reproducing Results](#reproducing-results)  
7. [Repository File Structure](#7-repository-file-structure)  
8. [Citation](#8-citation)  
9. [Contact](#9-contact)  

---


## 1. Manual Labeling Process

### Dataset Metadata and Excel File

The manual labeling process used the **Muharaf Dataset**, which originally contained partially labeled and noisy metadata. To address this, we consolidated all metadata into a structured Excel file.  

**File Location**: `manual_labeling/`  
**File Details**:
- `Image Filename`: Unique identifier for each line image.  
- `Writer Name (Arabic)`: Original Arabic names.  
- `Writer Name (English)`: Transliteration for consistency.  

| Image Filename    | Writer Name (Arabic) | Writer Name (English) |
|-------------------|-----------------------|------------------------|
| line001.png       | خالد البصري           | Khalid Al-Basri        |
| line002.png       | يوسف جابر            | Yousef Jaber           |

This file serves as the foundation for preprocessing, splitting, and training.

### Duplicate Detection

To detect and handle duplicate writer names:  
- **Tool**: `fuzzywuzzy` library for string similarity.  
- **Threshold**: Similarity scores ≥85% flagged for manual review.  

**Example**:  
- "Boutros Hassan" and "Botros Hassan" → 98% similarity → Merged.

**Code Reference**:  
`manual_labeling/fuzzy_matching.py`

### Error Corrections and Final Dataset

Corrections included:
1. Resolving ambiguous writer names using context.  
2. Standardizing transliterations (e.g., "Khaleel" → "Khalil").  

**Final Dataset Statistics**:
| Metric                  | Value            |
|-------------------------|------------------|
| Total Lines             | 36,311           |
| Total Labeled Lines     | 21,249           |
| Unique Writers (Classes)| 179              |

**Code Reference**:  
`manual_labeling/corrections.py`

---

## 2. Data Preprocessing

### Resizing and Normalization

Images were resized to **224x224 pixels** and normalized to pixel intensity values between [0, 1].  

**Code Reference**:  
`data_preprocessing/`

### Augmentation Techniques

To improve generalization and balance the dataset:  
- **Rotation**: ±15°.  
- **Zoom**: ±30%.  
- **Width/Height Shifts**: ±20%.  

**Code Reference**:  
`data_preprocessing/augmentation.py`

### Train-Test Split

Dataset split into:
- **70% Training**, **15% Validation**, **15% Testing**.  
- **Seed Values**: 42, 570, 1073 (for reproducibility).  

**Code Reference**:  
`data_preprocessing/split.py`

---

## 3. Models and Architectures

### Baseline Model

**Architecture**: ResNet50  
- Pretrained on ImageNet.  
- Optimized using Adam (LR=0.001).  
- Loss: Categorical Cross-Entropy.  

**Code Reference**:  
`models/`

### Variations: Feature Extractors and Attention

#### Feature Extractors
- **DenseNet201**: Encourages feature reuse.  
- **Xception**: Efficient depthwise separable convolutions.  
- **MobileNetV2**: Lightweight model for edge applications.

#### Attention Mechanisms
- **Self-Attention**: Captures spatial relationships.  
- **Cross-Attention**: Fuses local and global features.  

**Code Reference**:  
`models/architectural_variations.py`

---

## 4. Evaluation Metrics
The following metrics were used for evaluation:
1. **Accuracy**: Proportion of correct classifications.  
2. **Macro F1-Score**: Harmonic mean of precision and recall, averaged across classes.  
3. **Top-5 Accuracy**: Probability of the correct class being in the top-5 predictions.  

**Code Reference**:  
`evaluation/`

---

## 5. Experimental Results

| **Model**           | **Accuracy (%)** | **Macro F1** | **Top-5 Accuracy (%)** |
|----------------------|------------------|--------------|-------------------------|
| ResNet50 (Baseline) | 85.3             | 84.1         | 93.4                   |
| DenseNet201         | 87.5             | 86.3         | 94.8                   |
| Xception            | 88.2             | 87.8         | 95.2                   |

---

## 6. Reproducibility and Setup

### Setup and Installation

```bash
conda env create -f environment.yaml
conda activate writer-id
```

### Reproducing Results

#### Preprocessing
```bash
python data_preprocessing/preprocess_pipeline.py
```

#### Training
```bash
python models/train.py --config 5_configs/train_config.yaml
```

#### Evaluation
```bash
python evaluation/evaluate.py --config 5_configs/eval_config.yaml
```

---

## 7. Repository File Structure

```
Writer-Identification/
├── manual_labeling/
├── data_preprocessing/
├── models/
├── evaluation/
├── environment.yaml
├── requirements.txt
├── README.md
├── LICENSE
```
---

## 8. Citation

```bibtex
@article{abushahla2024writerid,
  title={Different Strokes for Different Folks: Writer Identification for Historical Arabic Manuscripts},
  author={Hamza A. Abushahla, Ariel J.N. Panopio, Layth M. Al-Khairulla, Mohamed I. AlHajri},
  journal={IEEE Access},
  year={2025}
}
```

---

## 9. Contact

For inquiries, contact:  
**Hamza Ahmed Abushahla**  
Email: [b00090279@aus.edu](mailto:b00090279@aus.edu)



