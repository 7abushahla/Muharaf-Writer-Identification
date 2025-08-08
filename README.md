# Different Strokes for Different Folks: Writer Identification for Historical Arabic Manuscripts

_Hamza A. Abushahla, Ariel Justine Navarro Panopio, Layth Al-Khairulla, and Dr. Mohamed I. AlHajri_

This repository contains the full implementation and supplementary materials for our paper, **Different Strokes for Different Folks: Writer Identification for Historical Arabic Manuscripts**. It includes all code, configurations, and documentation needed to reproduce the experiments and extend the methodology — as well as full access to our manually curated dataset and labeling effort.

<div align="center">
  <img src="figures/proposed_model.jpg" height="350px" alt="E2E" />
</div>
<p align="center"><em>Figure 1: Proposed end-to-end architecture illustrating both the attention-based and no-attention variants. The dashed
blocks and arrows represent the optional attention path, which is active only in the attention-based version.</em></p>


## 📌 Overview

This work presents the **first** application of the **[Muharaf dataset](https://github.com/MehreenMehreen/muharaf)** for **writer identification** on historical Arabic manuscripts. Our contributions can be summarized as follows:

- We manually verified and labeled a substantial chunk of the public portion of the Muharaf dataset, significantly expanding the amount of labeled data from 6,858 lines (28.00\%) to 21,249 lines (86.75\%), thereby enhancing its applicability for supervised learning and writer identification.
- We developed an end-to-end CNN-based DL system with attention mechanisms for line-level writer identification in historical Arabic handwritten manuscripts, accommodating up to two authors per line.
- We demonstrate that fine-tuning pre-trained feature extractors achieves performance that matches or surpasses that of non-fine-tuned models while significantly reducing training time.
- We provide an in-depth analysis of the optimal number of layers to unfreeze during fine-tuning, showing that when executed correctly, fine-tuning far surpasses training from scratch---thereby highlighting the benefits of transfer learning.
- We highlight the challenges and potential of leveraging partially annotated datasets, such as Muharaf, for writer identification, offering valuable insights for future research in writer identification and related domains.
  
---


## 📂 1. Dataset and Manual Labeling Effort

We provide our cleaned and labeled dataset through Google Drive:

- 📦 **Lines folder**: [Download here](https://drive.google.com/file/d/1nyuI01sw17BObYh6N-eLFe4-1m-p2lq_/view?usp=sharing)
- 📑 **Writer labels (`merged_writer.csv`)**: [Download here](https://drive.google.com/file/d/1phlCpHYRqa-9z8YQ5Wewg5VtkjjkfQFG/view?usp=share_link)

After downloading, the folder structure should look something like:

```bash
Writer-Identification/
├── Lines/
│   ├── AF_279r/
│   │   ├── AF_279r-1.png
│   │   ├── AF_279r-2.png
│   │   └── ...
│   ├── AR51_008/
│   │   ├── AR51_008-1.png
│   │   ├── AR51_008-2.png
│   │   └── ...
│   └── ...
├── merged_writer.csv
└── ...
```

Each folder contains line-level images extracted from a page, and all lines in the same folder are assumed to be written by the same author. The `merged_writer.csv` file maps each image to its writer label.

---

### 🖊️ Dataset Labeling and Preparation

The manual labeling process was conducted on the **public portion of the Muharaf dataset** ([Zenodo](https://zenodo.org/records/11492215)), which consists of 1,216 scanned manuscript pages. Initially, only 309 pages (25.4%) contained writer tags, covering 6,858 line images. The remaining 907 pages (74.6%), comprising 17,637 lines, lacked writer annotations and required manual labeling. The table below shows how writer metadata was distributed across pages and line images in the original dataset.

**Table 1. Writer Status in the Muharaf Dataset (Public)**

| Writer Status | No. of Pages | % of Pages | No. of Lines | % of Lines |
|---------------|--------------|------------|--------------|------------|
| Present       | 309          | 25.41%     | 6,858        | 28.0%      |
| Not Present   | 907          | 74.59%     | 17,637       | 72.0%      |
| **Total**     | **1,216**    | **100%**   | **24,495**   | **100%**   |

### Step 1: Manual Labeling

To annotate the previously unlabeled portion, we created an Excel spreadsheet (`manual_labeling/manual_labeling.xlsx`) with structured metadata fields. Each row includes the filename of the page-level image, a collection prefix, and the writer’s name in both English transliteration and Arabic script.

**Table 2. Example Rows from the Manual Labeling Sheet (non-consecutive):**

| Image Filename    | Prefix | Writer Name (English)  | Writer Name (Arabic)    |
|-------------------|--------|------------------------|--------------------------|
| AF_304_01r        | AF     | Your son George        | ولدكم جورج              |
| AR51_008          | AR     | Ameen Rihani           | أمين الريحاني           |
| EAC_A_039_059r    | EAC    | Elias Abu Shabaki      |إلياس أبو شبكة           |

Manually labeling each page involved identifying writers based on document types and context. For personal letters, identifying the writer was straightforward as the sender’s name was often explicitly stated in the first line or included as a signature. These annotations were directly linked to the corresponding handwriting samples.

For pages without clear signatures or identifying features, handwriting styles were compared with known samples to infer the writer’s identity where possible. Relational identifiers, such as 'Your nephew' or 'Your son,' were preserved without further disambiguation. In some cases, historical and literary context played a critical role:

- **Amin Rihani Collection**: Letters signed as "May" were attributed to May Ziadeh, a poet and author, based on historical correspondence with Rihani ([Link 1](https://lebanesestudies.omeka.chass.ncsu.edu/items/show/87673#?c=&m=&s=&cv=), [Link 2](https://books.google.ae/books?id=MMfEoQEACAAJ)).
- **Elias Abu Shabaki Collection**: Poems like the one on page `EAC_A_039_059r` were matched to online archives (e.g., [arabic-poetry.net](https://arabic-poetry.net/poem/88042-في-ربيع-الحياة-حلو-الخصال)) to confirm authorship.
- **Salah Tizani Collection**: Scripts like `ST1A_197_01` were attributed to Tizani after verifying character names against known TV and theater productions (e.g., [Wikipedia](https://ar.wikipedia.org/wiki/أبو_سليم_الطبل_(مسلسل))).

We also referred to the [Moise A. Khayrallah Center for Lebanese Diaspora Studies Archive](https://lebanesestudies.omeka.chass.ncsu.edu) to find writers for the pages in the Muharaf Dataset, specifically for those included in collections provided by the Khayrallah Center. Using their advanced search by 'Identifier,' we retrieved many writer names, such as those in the Ellis Family collection.

To facilitate line-level processing, we created a folder for each set of lines extracted from the page, named after the page image. All lines within the same folder are assumed to be written by the same author. This is available in the `Lines` folder.


### Step 2: Label Verification

After labeling, we merged the new annotations with the previously labeled subset (`writer_filled.csv`) into a unified file: `merged_writer.csv`. We validated the labels through semi-automated duplicate detection and manual review. We used fuzzy string matching (via Levenshtein distance in `manual_labeling/fuzzy_matching.ipynb`) to detect inconsistent transliterations and possible duplicate entries. To illustrate, Table 3 shows example matches and decisions:

**Table 3. Sample Fuzzy Matching Decisions**

| Name A         | Name B           | Similarity | Action         |
|----------------|------------------|------------|----------------|
| Botros Hassan  | Boutros Hassan   | 98%        | Merged         |
| Botros Hassan  | Botros Hasan     | 97%        | Manual Review  |


Key steps included:

1. Standardized writer names by aligning transliterations and formatting.
2. Using multiple thresholds (85–95%) to balance recall and precision.
3. Reviewing all borderline cases manually.

### Step 3: Error Corrections

We identified errors in the original dataset during label verification. For example:
- A labeled page attributed to "Father Youssef Baissary" (`JoM_Kobayat_002`) was corrected to "Father Youhanna Habib Baissary" after verifying handwriting and cross-referencing with other pages.
- When cross-referencing this with other pages from the unlabeled portion of the dataset, we found identical handwriting and signature instances. For example:
  - `JoM_Kobayat_0567`: Example of a similar signature transcribed as "Al-Khoori Youhanna Habib" in the unlabeled portion.
  - `JoM_Kobayat_0548`: Example of the full name and title "Al-Khoori Youhanna Habib Al-Baissary" found in the unlabeled portion.

Additionally, transliterations were aligned with biblical origins (e.g., "Yousef" corresponds to "Joseph," not "John") to ensure consistency and accuracy in the dataset.


### Step 4: Dataset Preparation

Once verified, labled line-level images were mapped to their corresponding writers. From the original 24,495 lines, we successfully labeled 21,249 lines, increasing the number of identified writers from 94 to 179. Table 4 summarizes the post-labeling status:

**Table 4. Writer Status After Manual Labeling**

| Writer Status | No. of Pages | % of Pages | No. of Lines | % of Lines |
|---------------|--------------|------------|--------------|------------|
| Labeled       | 1,015        | 83.5%      | 21,249       | 86.75%     |
| Unlabeled     | 201          | 16.5%      | 3,246        | 13.25%     |
| **Total**     | **1,216**    | **100%**   | **24,495**   | **100%**   |

These lines were filtered to remove non-handwritten content, resulting in 18,987 usable lines for the dataset. The final statistics of the cleaned dataset are shown below:

**Table 5. Filtered Dataset Summary (Post-Cleanup)**

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

This manual labeling process significantly increased the usability of the Muharaf dataset for writer identification. However, the dataset remains highly imbalanced, with certain classes having a disproportionately large number of labeled samples compared to others. For example, the top three classes include "Ameen Rihani" with 949 images, "Hanna Ghayth" with 934 images, and "Hanna Moussa" with 876 images. Conversely, the lowest classes include "Nehme Elias Mikhail" with 12 images, "Shibli Barakat Witnesses" with 11 images, and "Father Elias" with only 10 images. The mean number of images per writer is 106.07, with a standard deviation of 183.29, reflecting the highly skewed distribution of labeled samples (see the histogram in the `Supplemental Results` folder).


### Step 5: Dataset Preprocessing

To prepare the data for training, we used the filtered line-level images and corresponding writer labels in a 70-15-15 train-validation-test split. To address the severe class imbalance, we employed Keras' data augmentation tool, `ImageDataGenerator`, to increase the size of the dataset and the number of instances per writer. Our data augmentations included image rotation, zoom, shear, width and height shifts, and fill mode set to nearest. Moreover, we do not binarize the images because we utilized the preprocessing functions provided by each of the feature extractors that we used. The exact parameter values used for these augmentations are described in Table 6 below.

**Table 6. Data Augmentation Parameters for Training**

| **Augmentation Parameter** | **Value**                    |
|----------------------------|------------------------------|
| Rotation Range             | ±15°                         |
| Zoom Range                 | ±30%                         |
| Shear Range                | ±30%                         |
| Width Shift Range          | ±20% of image width          |
| Height Shift Range         | ±20% of image height         |
| Fill Mode                  | Nearest                      |

These data transformations were applied exclusively to the training set to avoid data leakage and to make sure that the validation and test sets accurately reflect the model's performance on real-world data. Furthermore, to prevent order bias or the bias in which the sequence in the training data is seen by a model, which may influence the model's learning process, we only shuffle the training set and not the validation or test set. We should not shuffle the validation or test set to ensure reproducibility and consistency during model evaluation and inference. In our case, the labels, the writers' names, are one-hot-encoded for later use with our categorical cross-entropy loss function.

---

## 🧱 2. Proposed Architecture

This section describes our proposed architecture for writer identification, including both the standard (no-attention) and attention-enhanced variants. The pipeline is illustrated in Figure 1, and shows both versions. The overall pipeline consists of three main stages: feature extraction, encoding, and classification, integrated into a single end-to-end system.

### 2.1 Overall Design

Our architecture builds on the Deep-TEN framework by Chammas et al.[^1], which we recreated and modified for improved generalization and deployment efficiency. The final model is composed of the following stages:

1. **Convolutional Backbone**: A CNN (ResNet50, DenseNet201, or Xception) extracts hierarchical features from the 224×224 input image.
2. **L2-Normalization**: Applied to ensure scale-invariant features.
3. **Spatial Pyramid Pooling (SPP)**: Provides fixed-size feature maps by aggregating local features across multiple scales.
4. **Feature Aggregation**:

   * **NetVLAD** aggregates local descriptors into a compact global representation.
   * **Attention Mechanisms** (for attention-based models only) further refine the global features through self- and cross-attention.
5. **Dense Layers**: A fully connected layer (512 units), followed by dropout and L2-normalization, generates the final feature embedding.
6. **Classification Head**: Outputs writer probabilities using a softmax activation function with categorical cross-entropy loss.

### 2.2 Key Modifications from Deep-TEN
- Repositioned the Spatial Pyramid Pooling (SPP) layer after the L2-normalization layer for improved generalization.
- Used categorical cross-entropy instead of triplet loss for training efficiency and stability.
* Introduced L2-regularization to mitigate overfitting.
* Added a compact dense layer and dropout before the classification head.

### 2.3 Attention-Based Enhancements

To better capture the contextual and sequential nature of handwriting, we introduced the following attention modules:

1. **Self-Attention I:**  After refining local features through convolution, L2-normalization, and Layer Normalization.
2. **Self-Attention II**: After layer-normalized outputs of the SPP layer.
3. **Cross-Attention**: Between NetVLAD and the base CNN features using dense QKV projections.

Dense layers were introduced to align queries ($Q$), keys ($K$), and values ($V$) for cross-attention. The architecture integrates ResNet50, DenseNet201, and Xception as feature extractors.


### 2.4 Custom Modifications to SPP and NetVLAD
To adapt the architecture for better performance and compatibility with attention modules, we introduce key modifications to both the Spatial Pyramid Pooling (SPP) and NetVLAD layers.

* **Modified SPP**: In the original SPP design[^2], multi-scale max pooling is applied across the feature map, and the resulting pooled outputs are flattened and concatenated into a single vector. In contrast, our modified version retains the spatial structure of the pooled outputs. After applying max pooling at different scales (such as 1×1, 2×2, and 4×4), we upsample each pooled output back to the original spatial resolution. These upsampled outputs are then concatenated along the channel dimension, forming a multi-scale feature map that maintains consistent spatial dimensions across all inputs. This design is particularly beneficial for attention-based models, as it preserves the alignment required for self-attention and cross-attention operations, while also capturing both fine and coarse-grained context.

* **Modified NetVLAD**: The original NetVLAD layer[^3] computes soft assignments using a learnable 1×1 convolution layer, which introduces extra parameters and computational cost. In our modification, we simplify this by removing the learnable assignment layer and instead computing similarity scores using a dot product between each local feature descriptor and a set of learnable cluster centers. Both the features and cluster centers are normalized beforehand, so this dot product effectively measures cosine similarity. These similarity scores are softmax-normalized to produce soft assignments indicating how much each feature belongs to each cluster. Residuals between features and their assigned centers are then aggregated, followed by normalization steps to yield a compact and discriminative global descriptor. This approach not only simplifies the computation but also improves compatibility with deployment on resource-constrained devices.

Figures illustrating the updated SPP and NetVLAD modules are provided below.

<table>
  <tr>
    <td align="center" width="60%">
      <img src="figures/SPP.jpg" height="320px" alt="Modified SPP Layer" /><br/>
      <em>Figure 2: Modified SPP module. Unlike the original, our variant upsamples pooled outputs to 7×7 and concatenates them channel-wise to preserve spatial structure and enable attention integration.</em>
    </td>
    <td align="center" width="40%">
      <img src="figures/NetVLAD.jpg" height="220px" alt="Modified NetVLAD Layer" /><br/>
      <em>Figure 3: Modified NetVLAD module. We replace learned convolutional assignments with cosine similarity via dot product between descriptors and cluster centers, reducing parameters and improving deployment readiness.</em>
    </td>
  </tr>
</table>



### 2.5 Model Configurations and Training Regimes

We evaluated the following backbones:

* ResNet50
* DenseNet201
* Xception
* MobileNetV3-Large (for future deployment on edge devices)

We evaluated 14 architectural variations combining different levels of attention, training regimes, and fine-tuning depths. The configurations, as used in experiments, are listed below in their exact form:

1. Frozen + No Attention (Baseline)  
2. Frozen + Attention  
3. Fine-tuned + Last Layer + No Attention  
4. Fine-tuned + Last Layer + Attention  
5. Fine-tuned + Last 5 Layers + No Attention  
6. Fine-tuned + Last 5 Layers + Attention  
7. Fine-tuned + Last 10 Layers + No Attention  
8. Fine-tuned + Last 10 Layers + Attention  
9. Fine-tuned + Last 25 Layers + No Attention  
10. Fine-tuned + Last 25 Layers + Attention  
11. Fine-tuned + No Attention  
12. Fine-tuned + Attention  
13. From Scratch + No Attention  
14. From Scratch + Attention

Each configuration was evaluated across three random seeds (42, 570, 1073) to ensure statistical robustness.


### 2.6 Training Setup and Hyperparameters

The model was trained on 70% of the data, validated on 15%, and tested on the remaining 15% to ensure unseen test data (This train-val-test split was achieved using Scikit-learn's `train_test_split` function, where we first split the data into a 70-30 split, then split the 30\% into half for validation and testing data). Key training hyperparameters are summarized below:

**Table 7. General Hyperparameters**

| Parameter                    | Value                     |
| ---------------------------- | ------------------------- |
| Optimizer                    | Adam                      |
| Loss Function                | Categorical Cross-Entropy |
| Initial Learning Rate        | $1 × 10^{-3}$                  |
| Batch Size                   | 256                       |
| Number of Clusters (NetVLAD) | 64                        |
| Number of Epochs             | 450                       |
| Dropout Rate                 | 0.5                       |
| L2-Regularization            | $1 × 10^{-4}$                  |

**Table 8. Learning Rate Scheduler**

| Parameter                  | Value             |
| -------------------------- | ----------------- |
| Learning Rate Scheduler    | ReduceLROnPlateau |
| Scheduler Reduction Factor | 0.5               |
| Scheduler Patience         | 10 epochs         |
| Mode                       | Max               |
| Minimum Learning Rate      | $1 × 10^{-8}$          |

**Table 9. Early Stopping**

| Parameter               | Value               |
| ----------------------- | ------------------- |
| Early Stopping Metric   | Validation F1-Score |
| Early Stopping Patience | 50 epochs           |
| Mode                    | Max                 |

**Additional Callbacks**:
To complement the training process, we employed several callback functions. These included periodic model checkpointing every 50 epochs to save intermediate models, outputting training metrics for each epoch, and clearing displayed outputs in the Jupyter Notebook every 10 epochs to improve readability during training.


### 2.7 Experimental Setup

All experiments were conducted using two NVIDIA A100 (SXM4) Tensor Core GPUs, each equipped with 80 GB of memory. One GPU was accessed remotely through the American University of Sharjah's (AUS) Computer Science and Engineering (CSE) Artificial Intelligence (AI) Lab, while the other was rented via the RunPod platform to facilitate parallel runs of different models and seeds. The training setup utilized Python 3 and TensorFlow tools and libraries, running on Ubuntu 24.04.1. CUDA version 12.4 and cuDNN were employed to optimize GPU computations, ensuring efficient use of resources. Training times varied depending on factors such as the model's configuration and the sharing of CPU usage on the AUS CSE AI Lab's A100. These details are summarized in Table 10 below.

**Table 10. Average Training Time per Configuration**

| **Model Configuration**                    | **ResNet50**        | **DenseNet201**     | **Xception**           | **MobileNetV3-Large**  |
|:-------------------------------------------|:--------------------|:--------------------|:------------------------|:------------------------|
| Frozen + No Attention (Baseline)           | 8 hrs 19 min 29 sec | 8 hrs 50 min 46 sec | 7 hrs 25 min 29 sec     | 7 hrs 37 min 23 sec     |
| Frozen + Attention                         | 7 hrs 3 min 13 sec  | 7 hrs 19 min 12 sec | 7 hrs 46 min 16 sec     | 7 hrs 5 min 10 sec      |
| Fine-tuned + Last Layer + No Attention     | 8 hrs 26 min 17 sec | 8 hrs 27 min 51 sec | 8 hrs 14 min 52 sec     | 7 hrs 33 min 5 sec      |
| Fine-tuned + Last Layer + Attention        | 7 hrs 20 min 2 sec  | 7 hrs 54 min 17 sec | 7 hrs 55 min 2 sec      | 6 hrs 44 min 52 sec     |
| Fine-tuned + Last 5 Layers + No Attention  | 8 hrs 12 min 24 sec | 8 hrs 43 min 54 sec | 8 hrs 24 min 3 sec      | 7 hrs 49 min 52 sec     |
| Fine-tuned + Last 5 Layers + Attention     | 7 hrs 20 min 28 sec | 7 hrs 3 min 0 sec   | 8 hrs 37 min 11 sec     | 6 hrs 18 min 10 sec     |
| Fine-tuned + Last 10 Layers + No Attention | 6 hrs 53 min 30 sec | 8 hrs 18 min 45 sec | 6 hrs 51 min 26 sec     | 7 hrs 34 min 4 sec      |
| Fine-tuned + Last 10 Layers + Attention    | 5 hrs 58 min 40 sec | 6 hrs 49 min 27 sec | 5 hrs 36 min 10 sec     | 6 hrs 35 min 25 sec     |
| Fine-tuned + Last 25 Layers + No Attention | 6 hrs 58 min 23 sec | 7 hrs 54 min 12 sec | 6 hrs 50 min 25 sec     | 7 hrs 33 min 52 sec     |
| Fine-tuned + Last 25 Layers + Attention    | 5 hrs 59 min 11 sec | 6 hrs 24 min 6 sec  | 6 hrs 8 min 58 sec      | 6 hrs 15 min 58 sec     |
| Fine-tuned + No Attention                  | 7 hrs 7 min 2 sec   | 7 hrs 45 min 56 sec | 8 hrs 40 min 30 sec     | 8 hrs 21 min 34 sec     |
| Fine-tuned + Attention                     | 5 hrs 55 min 0 sec  | 6 hrs 50 min 51 sec | 6 hrs 55 min 40 sec     | 6 hrs 30 min 3 sec      |
| From Scratch + No Attention                | 8 hrs 18 min 32 sec | 7 hrs 56 min 10 sec | 8 hrs 49 min 11 sec     | 5 hrs 29 min 50 sec     |
| From Scratch + Attention                   | 9 hrs 5 min 55 sec  | 7 hrs 9 min 54 sec  | 7 hrs 35 min 20 sec     | 1 hrs 31 min 21 sec     |


### 2.8 Custom Layer Serialization

A notable challenge we encountered during experimentation was ensuring that full models could be loaded seamlessly to either resume training or obtain evaluation metrics. This difficulty arose because we defined custom layers for `SPP`, `NetVLAD`, and `L2Normalization`, which required careful handling to guarantee compatibility with TensorFlow’s serialization and deserialization mechanisms.

To address this, each custom layer was implemented with the necessary methods to support full serialization. The `get_config()` method was defined to store initialization parameters and ensure that layer configurations could be correctly reconstructed when loading the model. We also used the `@tf.keras.utils.register_keras_serializable()` decorator to make the custom layers recognizable by TensorFlow's saving and loading routines.

The `call()` method defined the forward pass logic, ensuring consistency during both training and inference. For layers with trainable parameters—such as `NetVLAD`—the `build()` method was used to initialize weights appropriately, enabling proper reconstruction during deserialization. By adhering to these practices, our custom layers were seamlessly integrated into TensorFlow workflows, supporting robust training, evaluation, and deployment across different environments.

---

## 🛠️ 3. Reproducibility and Setup

### Setup and Installation

```bash
conda env create -f PROJtfgpu310.yml
conda activate PROJtfgpu310.yml
```


### 🏋️ Training Instructions

All training scripts are organized under the `models/` directory. Each model configuration has its own subfolder, named clearly to reflect the architecture, training strategy, and attention mechanism.

For example:

```
models/
├── ResNet50/
│   ├── ResNet50 + From Scratch + No Attention/
│   │   ├── ResNet50 - E2E WriterIdent - From Scratch - Seed 42 450 Epochs EarlyStop 50.py
│   │   ├── ResNet50 - E2E WriterIdent - From Scratch - Seed 570 450 Epochs EarlyStop 50.py
│   │   └── ResNet50 - E2E WriterIdent - From Scratch - Seed 1073 450 Epochs EarlyStop 50.py
│   ├── ResNet50 + Frozen + No Attention/
│   │   ├── ...
│   └── ...
├── Xception/
│   ├── Xception + Frozen + No Attention/
│   │   ├── ...
│   └── ...
├── DenseNet201/
│   └── ...
├── MobileNetV3-Large/
│   └── ...
```

The naming format of the files follows this pattern:
```
<ModelName> - E2E WriterIdent - <Training Strategy> - Seed <Seed> <Epochs> Epochs EarlyStop <Patience>.py
```
Each script includes the full training pipeline with:

- Model initialization (feature extractor, NetVLAD/SPP/L2Norm layers)
- Loss function and optimizer setup
- Dataset loading and augmentation
- Callbacks (EarlyStopping, ReduceLROnPlateau, Checkpointing, Logging)
- Evaluation and metrics tracking

This modular layout makes it easy to locate and reproduce any experiment.

### 🔁 Batch Training Utility

To streamline experimentation and avoid manual starting/stopping, we also provide a script `run_files.py` that allows running multiple training scripts one after another while logging their output and releasing GPU memory.

#### 📌 Example Usage

```bash
(PROJtfgpu310) python run_files.py "models/ResNet50 + From Scratch + No Attention/ResNet50 - E2E WriterIdent - From Scratch - Seed 42 450 Epochs EarlyStop 50.py" "models/ResNet50 + From Scratch + No Attention/ResNet50 - E2E WriterIdent - From Scratch - Seed 570 450 Epochs EarlyStop 50.py"
```

Each script's output is logged to a separate `.log` file with the same base name. The script automatically handles:

- Execution time logging (start, end, and duration).
- Output redirection to log files.
- GPU memory release by terminating the completed process.

---


## 4. Experimental Results

All primary evaluation metrics—**Top-1 Accuracy**, **Top-5 Accuracy**, and **F1-Score**—averaged over three random seeds, are reported in the tables within the main paper. Additional metrics such as **Precision**, **Recall**, and **Test Loss** are included in the supplementary materials (see *Online Resource 2*) and can also be found in the `Supplemental Results` folder.

Moreover, we provide **per-seed results** and full **classification reports** for each model configuration in the `Results` directory. Pretrained model checkpoints may be released in the future to facilitate further evaluation or fine-tuning by other researchers.

----

## 📚 Citation & Reaching out
If you find our work useful in your research, please consider citing it using the following:
```bibtex
@article{abushahla2025writerid,
  title={Different Strokes for Different Folks: Writer Identification for Historical Arabic Manuscripts},
  author={Hamza A. Abushahla, Ariel J.N. Panopio, Layth M. Al-Khairulla, Mohamed I. AlHajri},
  journal={},
  year={2025}
}
```

For questions, collaborations, or feedback, feel free to reach out via email:

* **Hamza Abushahla** — `b00090279@alumni.aus.edu`
* **Ariel Justine Panopio** — `b00088568@alumni.aus.edu`
* **Layth Al-Khairulla** — `b00087225@alumni.aus.edu`
* **Dr. Mohamed AlHajri** — `mialhajri@aus.edu`



[^1]: https://link.springer.com/article/10.1007/s11042-023-17303-8
[^2]: https://ieeexplore.ieee.org/abstract/document/7005506
[^3]: https://ieeexplore.ieee.org/abstract/document/7937898
