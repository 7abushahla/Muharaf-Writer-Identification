# Writer Identification for Historical Arabic Manuscripts

This repository contains the full implementation and supplementary materials for the paper: **[Different Strokes for Different Folks: Writer Identification for Historical Arabic Manuscripts](url)**

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

1. [Data Labeling and Preparation](#1-Data-Labeling-and-Preparation)  
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
6. [Repository File Structure](#6-repository-file-structure)  
7. [Reproducibility and Setup](#7-reproducibility-and-setup)  
    - [Setup and Installation](#setup-and-installation)  
    - [Reproducing Results](#reproducing-results)  
8. [Citation](#8-citation)  
9. [Contact](#9-contact)  

---


## 1. Data Labeling and Preparation

### Overview

The manual labeling process used the **public part of the Muharaf Dataset ([Muharaf-public](https://zenodo.org/records/11492215))**, which consisted of 1,216 pages. Of these, 309 pages had a writer tag, corresponding to 6,858 text lines and 94 unique writers. The remaining 907 pages, containing 17,637 text lines, lacked writer tags, necessitating manual intervention to expand the labeled data.

**Writer Status in the Muharaf Dataset (Public):**

| Writer Status | No. of Pages | % of Pages | No. of Lines | % of Lines |
|---------------|--------------|------------|--------------|------------|
| Present       | 309          | 25.41%     | 6,858        | 28.0%      |
| Not Present   | 907          | 74.59%     | 17,637       | 72.0%      |
| **Total**     | **1,216**    | **100%**   | **24,495**   | **100%**   |

### Step 1: Manual Labeling

At this stage, we created an Excel sheet (`manual_labeling/manual_labeling.xlsx`) to organize and annotate the data. The columns in the sheet are:

- `Image Filename`: Refers to the page-level images in the dataset.
- `Prefix`: Collection tag from the dataset.
- `Writer Name (English)`: Transliterated writer name.
- `Writer Name (Arabic)`: Original Arabic writer name.

**Example Rows (non-consecutive):**

| Image Filename    | Prefix | Writer Name (English)  | Writer Name (Arabic)    |
|-------------------|--------|------------------------|--------------------------|
| AF_304_01r        | AF     | Your son George        | ولدكم جورج              |
| AR51_008          | AR     | Ameen Rihani           | أمين الريحاني           |
| AR56_01_003_1     | AR     | Shibli N. Damus        | شبل نصيف دموس           |

Manually labeling each page involved identifying writers based on document types and context. For personal letters, identifying the writer was straightforward as the sender’s name was often explicitly stated in the first line or included as a signature. These annotations were directly linked to the corresponding handwriting samples.

For pages without clear signatures or identifying features, handwriting styles were compared with known samples to infer the writer’s identity where possible. Relational identifiers, such as 'Your nephew' or 'Your son,' were preserved without further disambiguation. In some cases, historical and literary context played a critical role:

- **Amin Rihani Collection**: Letters signed as "May" were attributed to May Ziadeh, a poet and author, based on historical correspondence with Rihani ([Link 1](https://lebanesestudies.omeka.chass.ncsu.edu/items/show/87673#?c=&m=&s=&cv=), [Link 2](https://books.google.ae/books?id=MMfEoQEACAAJ)).
- **Elias Abu Shabaki Collection**: Poems like the one on page `EAC_A_039_059r` were matched to online archives (e.g., [arabic-poetry.net](https://arabic-poetry.net/poem/88042-في-ربيع-الحياة-حلو-الخصال)) to confirm authorship.
- **Salah Tizani Collection**: Scripts like `ST1A_197_01` were attributed to Tizani after verifying character names against known TV and theater productions (e.g., [Wikipedia](https://ar.wikipedia.org/wiki/أبو_سليم_الطبل_(مسلسل))).

We also referred to the [Moise A. Khayrallah Center for Lebanese Diaspora Studies Archive](https://lebanesestudies.omeka.chass.ncsu.edu) to find writers for the pages in the Muharaf Dataset, specifically for those included in collections provided by the Khayrallah Center. Using their advanced search by 'Identifier,' we retrieved many writer names, such as those in the Ellis Family collection.

To facilitate line-level processing, we created a folder for each set of lines extracted from the page, named after the page image. All lines within the same folder are assumed to be written by the same author.


### Step 2: Label Verification

We merged the newly labeled data with the previously labeled portion (`writer_filled.csv`), creating a consolidated CSV file (`writer_merged.csv`). The process included manually reviewing potential matches flagged by the `manual_labeling/fuzzy_matching.ipynb` Python script. The script used fuzzy string matching (via Levenshtein distance) to calculate similarity scores and flag matches within thresholds (85%-95%), which were manually reviewed. Matches were flagged by the script and reviewed manually to ensure accuracy and consistency in labeling. During this process, we:


1. Standardized writer names by aligning transliterations and formatting.
2. Employed **fuzzy string matching** (using `fuzzywuzzy`) to identify and resolve potential duplicate writer names.
   - **Similarity Thresholds:** Selected manually based on the specific requirements of each review step, lowering it to 85% to capture less obvious duplicates or raising it to 95% to focus on highly confident matches.
   - Example Matches:
     - "Botros Hassan" and "Boutros Hassan" → 98% similarity → Merged.
     - "Botros Hassan" and "Botros Hasan" → 97% similarity → Flagged for manual review.

**CSV Files:**
- `manual_labeling.csv`: Original Excel file converted to CSV.
- `writer_filled.csv`: Original labled portion of the dataset.
- `merged_writer.csv`: Consolidated labeled data.



### Step 3: Error Corrections

We identified errors in the original dataset during label verification. For example:
- A labeled page attributed to "Father Youssef Baissary" (`JoM_Kobayat_002`) was corrected to "Father Youhanna Habib Baissary" after verifying handwriting and cross-referencing with other pages.
- When cross-referencing this with other pages from the unlabeled portion of the dataset, we found identical handwriting and signature instances. For example:
  - `JoM_Kobayat_0567`: Example of a similar signature transcribed as "al-Khoori Youhanna Habib" in the unlabeled portion.
  - `JoM_Kobayat_0548`: Example of the full name and title "al-Khoori Youhanna Habib al-Baissary" found in the unlabeled portion.

Additionally, transliterations were aligned with biblical origins (e.g., "Yousef" corresponds to "Joseph," not "John") to ensure consistency and accuracy in the dataset.


### Step 4: Dataset Preparation

Afterward, line-level images were mapped to their corresponding writers. Out of the 24,495 public text lines, 21,249 lines were successfully labeled, increasing the number of identified writers from 94 to 179. These lines were filtered to remove non-handwritten content, resulting in 18,987 usable lines for the dataset.

**Writer Status After Manual Labeling:**

| Writer Status | No. of Pages | % of Pages | No. of Lines | % of Lines |
|---------------|--------------|------------|--------------|------------|
| Labeled       | 1,015        | 83.5%      | 21,249       | 86.75%     |
| Unlabeled     | 201          | 16.5%      | 3,246        | 13.25%     |
| **Total**     | **1,216**    | **100%**   | **24,495**   | **100%**   |

**Filtered Dataset After Manual Labeling and Excluding Non-Handwritten Content:**

| Metric                     | Value   |
|----------------------------|---------|
| Total lines used           | 18,987  |
| Total lines unused         | 2,262   |
| % of lines used            | 77.51%  |
| Total writers (classes)    | 179     |
| Maximum images per writer  | 949     |
| Minimum images per writer  | 10      |
| Mean images per writer     | 106.07  |
| Standard deviation         | 183.29  |



<!-- - `Image Filename`: Refers to the page-level images in the dataset. 
- `Prefix`: Collection tag from the dataset.
- `Writer Name (English)`: Transliterated writer name.
- `Writer Name (Arabic)`: Original Arabic writer name.

**Example Rows (non-consecutive):**

| Image Filename    | Prefix | Writer Name (English)  | Writer Name (Arabic)    |
|-------------------|--------|------------------------|--------------------------|
| AF_304_01r        | AF     | Your son George        | ولدكم جورج              |
| AR51_008          | AR     | Ameen Rihani           | أمين الريحاني           |
| AR56_01_003_1     | AR     | Shibli N. Damus        | شبل نصيف دموس           |  


We created a folder for each set of lines extracted from the page, named after the page image. All lines within the same folder are assumed to be written by the same author.

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
`manual_labeling/corrections.py` -->

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

## 6. Repository File Structure

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

## 7. Reproducibility and Setup

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
python models/train.py --config configs/train_config.yaml
```

#### Evaluation
```bash
python evaluation/evaluate.py --config configs/eval_config.yaml
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
**Ariel Justine Panopio**  
Email: [b00088568@aus.edu](mailto:b00088568@aus.edu)



