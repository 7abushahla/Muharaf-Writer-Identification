#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import math
import pickle
import json
# Suppress TensorFlow INFO, WARNING, and ERROR messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# # Enable XLA
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'


# In[2]:


import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
import random
from IPython.display import clear_output
import gc
from PIL import Image

import tensorflow as tf

from tensorflow.keras.applications import MobileNetV3Large #import 

from tensorflow.keras.layers import Input, Lambda, Dense, Layer, Conv2D, Dropout

from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import layers, Model
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback


# In[3]:


# Set the seed for reproducibility
SEED = 1073
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


# In[4]:


# # Disable GPU
# tf.config.set_visible_devices([], 'GPU')


# In[5]:


# Check if TensorFlow can access the GPU
print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")


# In[6]:


# Limit TensorFlow to only use the first GPU
tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[0], 'GPU')

# Verify the configuration
logical_gpus = tf.config.list_logical_devices('GPU')
print("Visible GPUs:", logical_gpus)


# In[7]:


# Parallelized SPP
@tf.keras.utils.register_keras_serializable()
class SpatialPyramidPooling(layers.Layer):
    def __init__(self, pool_sizes, **kwargs):
        super(SpatialPyramidPooling, self).__init__(**kwargs)
        self.pool_sizes = pool_sizes
        self.pool_layers = [
            tf.keras.layers.MaxPooling2D(
                pool_size=(size, size),
                strides=(size, size),
                padding='same'
            ) for size in self.pool_sizes
        ]

    def call(self, inputs):
        h = tf.shape(inputs)[1]
        w = tf.shape(inputs)[2]

        pooled_outputs = [pool(inputs) for pool in self.pool_layers]
        # Upsample all pooled outputs back to (h, w)
        upsampled = [tf.image.resize(pooled, (h, w), method='bilinear') for pooled in pooled_outputs]
        # Concatenate along the channel dimension
        output = tf.concat(upsampled, axis=-1)
        return output

    def compute_output_shape(self, input_shape):
        # input_shape: (batch_size, h, w, c)
        if input_shape[3] is None:
            c_new = None
        else:
            c_new = input_shape[3] * len(self.pool_sizes)
        return (input_shape[0], input_shape[1], input_shape[2], c_new)

    def get_config(self):
        config = super(SpatialPyramidPooling, self).get_config()
        config.update({
            'pool_sizes': self.pool_sizes,
        })
        return config


# In[8]:


# Parallelized NetVLAD
@tf.keras.utils.register_keras_serializable()
class NetVLADLayer(layers.Layer):
    def __init__(self, num_clusters, **kwargs):
        super(NetVLADLayer, self).__init__(**kwargs)
        self.num_clusters = num_clusters

    def build(self, input_shape):
        self.feature_dim = input_shape[-1]
        # Initialize cluster centers with Xavier/Glorot initialization for better convergence
        self.cluster_centers = self.add_weight(
            shape=(self.num_clusters, self.feature_dim),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
            name="cluster_centers"
        )
        super(NetVLADLayer, self).build(input_shape)

    def call(self, inputs):
        # Inputs shape: (batch_size, H, W, D)
        batch_size = tf.shape(inputs)[0]
        H = tf.shape(inputs)[1]
        W = tf.shape(inputs)[2]
        D = self.feature_dim

        # Reshape inputs to (batch_size, H*W, D)
        inputs_reshaped = tf.reshape(inputs, [batch_size, -1, D])

        # L2-normalize along the feature dimension
        inputs_norm = tf.nn.l2_normalize(inputs_reshaped, axis=-1)

        # Compute similarities (assignments) to cluster centers
        similarities = tf.matmul(inputs_norm, self.cluster_centers, transpose_b=True)  # Shape: (batch_size, H*W, num_clusters)
        assignments = tf.nn.softmax(similarities, axis=-1)  # Shape: (batch_size, H*W, num_clusters)

        # Compute residuals
        residuals = tf.expand_dims(inputs_reshaped, axis=2) - self.cluster_centers  # Shape: (batch_size, H*W, num_clusters, D)
        residuals_weighted = residuals * tf.expand_dims(assignments, axis=-1)  # Shape: (batch_size, H*W, num_clusters, D)

        # Aggregate residuals
        vlad = tf.reduce_sum(residuals_weighted, axis=1)  # Shape: (batch_size, num_clusters, D)

        # Flatten and L2-normalize the VLAD descriptors
        vlad = tf.reshape(vlad, [batch_size, -1])  # Shape: (batch_size, num_clusters * D)
        vlad = tf.nn.l2_normalize(vlad, axis=-1)

        return vlad

    def compute_output_shape(self, input_shape):
        # input_shape: (batch_size, H, W, D)
        batch_size = input_shape[0]
        output_dim = self.num_clusters * input_shape[3]
        return (batch_size, output_dim)

    def get_config(self):
        config = super(NetVLADLayer, self).get_config()
        config.update({
            'num_clusters': self.num_clusters,
        })
        return config


# In[9]:


@tf.keras.utils.register_keras_serializable()
class L2Normalization(layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super(L2Normalization, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.nn.l2_normalize(inputs, axis=self.axis)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(L2Normalization, self).get_config()
        config.update({
            'axis': self.axis,
        })
        return config


# In[10]:


def build_writer_identification_model(input_shape, num_clusters, num_classes):
    # Base model using MobileNetV3
    
    # For now, we fine-tune imagenet weights
    base_model = MobileNetV3Large( 
        include_top=False, weights='imagenet', input_shape=input_shape, pooling=None, alpha=1.0, name="MobileNetV3"
    )
    for layer in base_model.layers:
        layer.trainable = True  # Make backbone trainable

    # Input layer
    inputs = Input(shape=input_shape)
    x = base_model(inputs)
    print('Shape after base_model:', x.shape)

    # 1x1 Convolution to reduce channels from 2048 to 64
    x = Conv2D(64, kernel_size=(1, 1), activation='relu', kernel_regularizer=l2(1e-4))(x)
    print('Shape after Conv2D:', x.shape)

    # L2-normalize to obtain compact local descriptors
    x = L2Normalization(axis=-1)(x)
    print('Shape after L2-normalization:', x.shape)

    # Spatial Pyramid Pooling
    x = SpatialPyramidPooling(pool_sizes=[1, 2, 4])(x)
    print('Shape after SpatialPyramidPooling:', x.shape)

    # NetVLAD Layer
    x = NetVLADLayer(num_clusters=num_clusters)(x)
    print('Shape after NetVLADLayer:', x.shape)

    # L2-normalize the output of NetVLAD
    x = L2Normalization(axis=-1)(x)
    print('Shape after final L2-normalization:', x.shape)

    # Fully connected layers to produce the embedding vector
    x = Dense(512, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.5)(x)
    x = L2Normalization(axis=-1)(x)

    # Classification head
    classification_output = Dense(num_classes, activation='softmax', name='classification_output')(x)
    print('Shape after classification_output:', classification_output.shape)

    # Define the model
    model = Model(inputs, classification_output)

    return model


# In[11]:


# Define the path to the main directory containing writer folders
main_dir = './Lines'
csv_file = 'merged_writer.csv'  # Update with the actual path to the CSV file
DIMENSIONS = 224
EPOCHS = 450
BATCH_SIZE = 256
BACKBONE = "MobileNetV3"
FINETUNE = "_Finetuned_All_ImageNet_"
ATTN = ""
image_size = (DIMENSIONS, DIMENSIONS)  # Set the desired image size for resizing
num_classes = 179  # Number of unique writers

# In[12]:


# Load the CSV data
writer_data = pd.read_csv(csv_file)


# In[13]:


# Dictionary to map writer names (folder names) to unique integer labels
writer_to_label = {}
current_label = 0


# In[14]:


# Initialize lists to store image data and labels
images = []
labels = []


# In[15]:


writer_data.head()


# In[16]:


writer_data.info()


# In[17]:


def block_processor_opencv(img_path, target_size=(224, 224)):
    """
    Optimized version of the block_processor function using OpenCV.

    Args:
        img_path (str): Path to the image file.
        target_size (tuple): Desired output image size.

    Returns:
        PIL.Image: Processed image.
    """
    try:
        # Read the image with OpenCV
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Image not found or unable to read: {img_path}")
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise ValueError(f"Error opening image {img_path}: {e}")

    original_height, original_width = img.shape[:2]

    # Resize while maintaining aspect ratio with height=64
    new_height = 64
    aspect_ratio = original_width / original_height
    new_width = int(new_height * aspect_ratio)
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Note: Removed horizontal flip to prevent augmentation during preprocessing

    # Create a new black image with target_size
    final_width, final_height = target_size
    new_img = np.zeros((final_height, final_width, 3), dtype=np.uint8)

    if new_width <= final_width:
        # Center the image horizontally
        left = (final_width - new_width) // 2
        new_img[0:new_height, left:left + new_width, :] = resized_img
    else:
        # Calculate the number of segments that fit vertically
        max_segments = final_height // new_height
        # Calculate total segments needed
        num_segments = (new_width + final_width - 1) // final_width
        # Limit the number of segments to fit within the final height
        for i in range(min(num_segments, max_segments)):
            left_crop = i * final_width
            right_crop = min(left_crop + final_width, new_width)
            segment = resized_img[0:new_height, left_crop:right_crop, :]
            new_img[i * new_height:i * new_height + new_height, 0:right_crop - left_crop, :] = segment

    # Convert back to PIL Image for consistency with the rest of the pipeline
    return Image.fromarray(new_img)


# In[18]:


from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

# Initialize lists to store images and labels
images = []
labels = []

# Loop through each row in the CSV file
for index, row in writer_data.iterrows():
    image_filename = row['Image Filename']
    writer_name = row['Writer Name (English)']

    # Assign a unique label to each writer if not already assigned
    if writer_name not in writer_to_label:
        writer_to_label[writer_name] = current_label
        current_label += 1

    # Construct the full path to the folder containing the image
    folder_path = os.path.join(main_dir, image_filename)

    # Check if the folder exists
    if os.path.isdir(folder_path):
        # Look for images within the folder
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpeg', '.jpg')):
                try:
                    # print(f"Image file was found: {filename.lower()}")    
                    img_path = os.path.join(folder_path, filename)
                    # Process the image using BlockProcessor
                    processed_img = block_processor_opencv(img_path, target_size=image_size)
                    
                    # Convert PIL Image to NumPy array
                    img_array = np.array(processed_img)
                    
                    # Ensure the image is in RGB format
                    if img_array.shape[-1] != 3:
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                    
                    # Preprocess the image for ResNet50
                    processed_img = preprocess_input(img_array)
                    
                    # Append the image and the corresponding label
                    images.append(processed_img)
                    labels.append(writer_to_label[writer_name])

                    del img_array, processed_img
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")             
    else:
        print(f"Folder not found: {folder_path}")

# Validate dataset sizes
if len(images) != len(labels):
    raise ValueError(f"Number of images ({len(images)}) does not match number of labels ({len(labels)})")


# In[19]:


# Create a reverse mapping from label indices to writer names
label_to_writer = {label: writer for writer, label in writer_to_label.items()}


# In[20]:


gc.collect()


# In[21]:


# Convert lists to numpy arrays if images were loaded
if images:
    images = np.array(images).reshape(-1, image_size[0], image_size[1], 3)  # Add channel dimension
    num_classes = len(writer_to_label)  # Set num_classes to the total number of unique writers
    labels = to_categorical(labels, num_classes=num_classes)  # One-hot encode labels

    # Check the unique values of labels to confirm the range
    unique_labels = np.unique(labels)
    print(f"Unique labels: {unique_labels} and the number of labels we have is {len(unique_labels)}")
    print(f"The number of classes we have is {num_classes}")
else:
    print("No images were loaded. Please check the folder structure and file paths.")

# Ensure no label is equal to or greater than num_classes
if max(unique_labels) >= num_classes:
    print(f"Warning: Found labels out of range. Maximum label found is {max(unique_labels)} with num_classes set to {num_classes}.")
    # Optionally, handle out-of-range labels if any are found


# In[22]:


print("Number of non-augmented data (total):", len(images))


# In[23]:


# First split: 70% training and 30% (validation + test)
train_images, temp_images, train_labels, temp_labels = train_test_split(
    images, labels, test_size=0.3, random_state=SEED, stratify=labels
)


# In[24]:


# Second split: Split the temporary set into 50% validation and 50% test
# Since the temporary set is 30% of the original data,
# this results in 15% validation and 15% test
val_images, test_images, val_labels, test_labels = train_test_split(
    temp_images, temp_labels, test_size=0.5, random_state=SEED, stratify=temp_labels
)


# In[25]:


print(f"Shape of one-hot encoded labels: {labels.shape}")
unique_classes = np.unique(np.argmax(labels, axis=1))
print(f"Unique labels after one-hot encoding: {unique_classes}")
print(f"Number of unique labels after encoding: {len(unique_classes)}")


# In[26]:


print(f"Total images: {len(images)}")
print(f"Training images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")
print(f"Test images: {len(test_images)}")
print(f"BATCH_SIZE: {BATCH_SIZE}")


# In[27]:


# Check the number of unique classes in each split
train_classes = np.unique(np.argmax(train_labels, axis=1))
val_classes = np.unique(np.argmax(val_labels, axis=1))
test_classes = np.unique(np.argmax(test_labels, axis=1))

print(f"Number of unique classes in training set: {len(train_classes)}")
print(f"Number of unique classes in validation set: {len(val_classes)}")
print(f"Number of unique classes in test set: {len(test_classes)}")

# Optional: Print to see which classes are present
print(f"Training set classes: {train_classes}")
print(f"Validation set classes: {val_classes}")
print(f"Test set classes: {test_classes}")

# Verify no missing classes
assert len(train_classes) <= num_classes, "Training set has more classes than expected!"
assert len(val_classes) <= num_classes, "Validation set has more classes than expected!"
assert len(test_classes) <= num_classes, "Test set has more classes than expected!"


# In[28]:


del images, labels, writer_data, csv_file, temp_images, temp_labels


# In[29]:


import tensorflow.keras.backend as K


# In[30]:


@tf.keras.utils.register_keras_serializable()
class MacroPrecision(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='macro_precision', **kwargs):
        super(MacroPrecision, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.true_positives = self.add_weight(
            name='tp', shape=(num_classes,), initializer='zeros', dtype=tf.float32
        )
        self.false_positives = self.add_weight(
            name='fp', shape=(num_classes,), initializer='zeros', dtype=tf.float32
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert predictions and true labels to class indices
        y_pred_indices = tf.argmax(y_pred, axis=-1)
        y_true_indices = tf.argmax(y_true, axis=-1)

        # One-hot encode the indices
        y_pred_one_hot = tf.one_hot(y_pred_indices, depth=self.num_classes)
        y_true_one_hot = tf.one_hot(y_true_indices, depth=self.num_classes)

        # Calculate true positives and false positives per class
        tp = tf.reduce_sum(y_true_one_hot * y_pred_one_hot, axis=0)
        fp = tf.reduce_sum(y_pred_one_hot * (1 - y_true_one_hot), axis=0)

        # Update the state variables
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)

    def result(self):
        # Compute per-class precision and macro-average it
        precision_per_class = self.true_positives / (
            self.true_positives + self.false_positives + tf.keras.backend.epsilon()
        )
        macro_precision = tf.reduce_mean(precision_per_class)
        return macro_precision

    def reset_states(self):
        self.true_positives.assign(tf.zeros_like(self.true_positives))
        self.false_positives.assign(tf.zeros_like(self.false_positives))

    def get_config(self):
        base_config = super(MacroPrecision, self).get_config()
        base_config.update({'num_classes': self.num_classes})
        return base_config


# In[31]:


@tf.keras.utils.register_keras_serializable()
class MacroRecall(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='macro_recall', **kwargs):
        super(MacroRecall, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.true_positives = self.add_weight(
            name='tp', shape=(num_classes,), initializer='zeros', dtype=tf.float32
        )
        self.false_negatives = self.add_weight(
            name='fn', shape=(num_classes,), initializer='zeros', dtype=tf.float32
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert predictions and true labels to class indices
        y_pred_indices = tf.argmax(y_pred, axis=-1)
        y_true_indices = tf.argmax(y_true, axis=-1)

        # One-hot encode the indices
        y_pred_one_hot = tf.one_hot(y_pred_indices, depth=self.num_classes)
        y_true_one_hot = tf.one_hot(y_true_indices, depth=self.num_classes)

        # Calculate true positives and false negatives per class
        tp = tf.reduce_sum(y_true_one_hot * y_pred_one_hot, axis=0)
        fn = tf.reduce_sum(y_true_one_hot * (1 - y_pred_one_hot), axis=0)

        # Update the state variables
        self.true_positives.assign_add(tp)
        self.false_negatives.assign_add(fn)

    def result(self):
        # Compute per-class recall and macro-average it
        recall_per_class = self.true_positives / (
            self.true_positives + self.false_negatives + tf.keras.backend.epsilon()
        )
        macro_recall = tf.reduce_mean(recall_per_class)
        return macro_recall

    def reset_states(self):
        self.true_positives.assign(tf.zeros_like(self.true_positives))
        self.false_negatives.assign(tf.zeros_like(self.false_negatives))

    def get_config(self):
        base_config = super(MacroRecall, self).get_config()
        base_config.update({'num_classes': self.num_classes})
        return base_config


# In[32]:


@tf.keras.utils.register_keras_serializable()
class MacroF1Score(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='macro_f1_score', **kwargs):
        super(MacroF1Score, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.precision_metric = MacroPrecision(num_classes)
        self.recall_metric = MacroRecall(num_classes)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision_metric.update_state(y_true, y_pred)
        self.recall_metric.update_state(y_true, y_pred)

    def result(self):
        # Compute the macro-averaged precision and recall
        precision = self.precision_metric.result()
        recall = self.recall_metric.result()
        # Compute the macro-averaged F1 score
        f1_score = 2 * (precision * recall) / (
            precision + recall + tf.keras.backend.epsilon()
        )
        return f1_score

    def reset_states(self):
        self.precision_metric.reset_states()
        self.recall_metric.reset_states()

    def get_config(self):
        base_config = super(MacroF1Score, self).get_config()
        base_config.update({'num_classes': self.num_classes})
        return base_config


# In[33]:


class ClearOutputEveryNEpochs(Callback):
    def __init__(self, n=10):
        super().__init__()
        self.n = n

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.n == 0:
            clear_output(wait=True)  # Clears the output but waits to prevent flickering
            print(f"Cleared output at epoch {epoch + 1}")


# In[34]:


class PeriodicModelCheckpoint(Callback):
    """
    Custom callback to save the model at periodic intervals (e.g., every N epochs).
    """
    def __init__(self, filepath, save_freq_epochs=30, save_best_only=False, verbose=1):
        """
        Initializes the PeriodicModelCheckpoint callback.
        
        Args:
            filepath (str): Path where the model will be saved. Can include formatting options like {epoch}.
            save_freq_epochs (int): Frequency in epochs to save the model.
            save_best_only (bool): If True, saves the model only if the monitored metric improves.
            verbose (int): Verbosity mode. 0 = silent, 1 = messages.
        """
        super(PeriodicModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.save_freq_epochs = save_freq_epochs
        self.save_best_only = save_best_only
        self.verbose = verbose
        if self.save_best_only:
            # Initialize best metric based on 'val_f1'
            self.best = -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch += 1  # Epoch indexing starts at 0

        # Save every 'save_freq_epochs' epochs
        if epoch % self.save_freq_epochs == 0:
            if self.save_best_only:
                current = logs.get('val_f1_score')
                if current is None:
                    if self.verbose > 0:
                        print(f"Validation F1-score ('val_f1_score') is not available. Skipping save.")
                    return
                if current > self.best:
                    self.best = current
                    filepath = self.filepath.format(epoch=epoch, **logs)
                    self.model.save(filepath)
                    if self.verbose > 0:
                        print(f"\nEpoch {epoch}: 'val_f1' improved to {current:.4f}. Saving model to {filepath}")
            else:
                # If not saving based on a metric, save unconditionally
                filepath = self.filepath.format(epoch=epoch, **logs)
                self.model.save(filepath)
                if self.verbose > 0:
                    print(f"\nEpoch {epoch}: Saving model to {filepath}")


# In[35]:


# Number of original training samples
num_non_augmented_samples = len(train_images)  # Assuming train_images is your original dataset
print("Number of non-augmented training data samples:", num_non_augmented_samples)


# In[36]:


# Define the ImageDataGenerator with enhanced augmentations
train_datagen = ImageDataGenerator(
    rotation_range=15,             # Increased rotation
    zoom_range=0.3,                # Increased zoom
    shear_range=0.3,               # Increased shear
    width_shift_range=0.2,         # Increased width shift
    height_shift_range=0.2,        # Increased height shift
    # brightness_range=[1.01, 1.05],    # Adjust brightness
    fill_mode='nearest',           # Fill mode
    horizontal_flip=False,         # No horizontal flip
    vertical_flip=False            # No vertical flip
)

# Create training generator
train_generator = train_datagen.flow(
    train_images, train_labels, batch_size=BATCH_SIZE, shuffle=True
)


# In[37]:


# Define ImageDataGenerator for validation (no augmentation)
validation_test_datagen = ImageDataGenerator()

# Create validation generator
validation_generator = validation_test_datagen.flow(
    val_images, val_labels,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Create test generator
test_generator = validation_test_datagen.flow(
    test_images, test_labels,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# In[38]:


def infinite_generator(generator):
    while True:
        for batch in generator:
            yield batch


# In[39]:


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Now limit TensorFlow to the first GPU
        tf.config.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        print(e)


# In[40]:


from tensorflow.keras.metrics import TopKCategoricalAccuracy


# In[41]:


# Define input shape and other parameters
input_shape = (DIMENSIONS, DIMENSIONS, 3)
num_clusters = 64
num_classes = 179

model = build_writer_identification_model(input_shape=input_shape, num_clusters=num_clusters, num_classes=num_classes)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        TopKCategoricalAccuracy(k=5, name='top_5_accuracy'),  # Top-5 Accuracy
        MacroPrecision(num_classes=num_classes, name='precision'),
        MacroRecall(num_classes=num_classes, name='recall'),
        MacroF1Score(num_classes=num_classes, name='f1_score')
    ]
)

# Display the model summary
model.summary()


# In[42]:


# Define callbacks
# Instantiate the callback
clear_output_callback = ClearOutputEveryNEpochs(n=10)

# Define ModelCheckpoint to save the best model based on F1-score
checkpoint_path = f"{ATTN}{BACKBONE}{FINETUNE}E2E_WriterIdent_best_model_val_f1_{SEED}.keras"
checkpoint_callback = ModelCheckpoint(
    checkpoint_path, monitor='val_f1_score', save_best_only=True, mode='max', verbose=1
)

# Define the file path template for periodic checkpointing
# Including epoch number in the filename to differentiate saved models
periodic_checkpoint_path = f"{ATTN}{BACKBONE}{FINETUNE}E2E_WriterIdent_Last_Saved_Epoch_{SEED}.keras"

# Initialize the PeriodicModelCheckpoint callback
periodic_checkpoint_callback = PeriodicModelCheckpoint(
    filepath=periodic_checkpoint_path,
    save_freq_epochs=50,    # Save every 50 epochs
    save_best_only=False,   # Set to True if you want to save only when 'val_f1_score' improves
    verbose=1
)

# Define additional callbacks
lr_reduce = ReduceLROnPlateau(
    monitor='val_f1_score', 
    factor=0.5, 
    patience=10, 
    verbose=1, 
    mode='max',
    min_lr=1e-8
)

early_stop = EarlyStopping(
    monitor='val_f1_score', 
    patience=50, 
    restore_best_weights=True, 
    mode='max',
    verbose=1
)


# In[43]:


import math
steps_per_epoch = math.ceil(len(train_images) / BATCH_SIZE)
validation_steps = math.ceil(len(val_images) / BATCH_SIZE)


# In[44]:


# Create training generator
train_generator_infinite = infinite_generator(train_generator)

# Create validation generator
validation_generator_infinite = infinite_generator(validation_generator)

# Create test generator
test_generator_infinite = infinite_generator(test_generator)


# In[45]:


import time


# In[46]:


# Start the timer
start_time = time.time()

history = model.fit(
    train_generator_infinite,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator_infinite,
    validation_steps=validation_steps,
    epochs=EPOCHS,
    callbacks=[checkpoint_callback, periodic_checkpoint_callback, lr_reduce, early_stop, clear_output_callback],
    verbose=1
)

# Stop the timer
end_time = time.time()


# In[47]:


# Calculate the elapsed time
elapsed_time = end_time - start_time

# Convert elapsed time to hours, minutes, and seconds
hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = elapsed_time % 60

# Print the elapsed time in seconds, minutes, and hours
print(f"Total training time: {elapsed_time:.6f} seconds")
print(f"Or equivalently: {hours} hours, {minutes} minutes, and {seconds:.6f} seconds")


# In[48]:


# OR
# Convert elapsed time to hours, minutes, and seconds
hours = elapsed_time / 3600  # Total time in hours (float)
minutes = elapsed_time / 60  # Total time in minutes (float)
seconds = elapsed_time       # Total time in seconds

# Print the elapsed time in all units
print(f"Total training time:")
print(f"- Seconds: {seconds:.6f}")
print(f"- Minutes: {minutes:.6f}")
print(f"- Hours: {hours:.6f}")


# In[49]:


K.clear_session()
gc.collect()


# In[50]:


# Reset the generator before evaluation to ensure it starts from the beginning
test_generator.reset()

test_results = model.evaluate(
    test_generator, 
    steps=math.ceil(len(test_images) / BATCH_SIZE),
    verbose=1
)
# Unpack the results
test_loss = test_results[0]
test_accuracy = test_results[1]
test_top_5_accuracy = test_results[2]  # Top-5 Accuracy
test_precision = test_results[3]
test_recall = test_results[4]
test_f1_score = test_results[5]

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Top-5 Accuracy: {test_top_5_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1 Score: {test_f1_score:.4f}")


# In[51]:


# Reset the test generator before prediction to ensure alignment
test_generator.reset()

# Get predictions
test_preds = model.predict(
    test_generator, 
    steps=math.ceil(len(test_images) / BATCH_SIZE), 
    verbose=0
)

# Convert predictions to class indices
test_pred_classes = np.argmax(test_preds, axis=1)
test_true_classes = np.argmax(test_labels, axis=1)  # Convert one-hot to integer labels


# In[52]:


# Compute Top-5 Predictions
top_5_preds = np.argsort(test_preds, axis=1)[:, -5:]

# Calculate Top-5 Accuracy
top_5_correct = 0
for i in range(len(test_true_classes)):
    if test_true_classes[i] in top_5_preds[i]:
        top_5_correct += 1

top_5_accuracy_manual = top_5_correct / len(test_true_classes)
print(f"Manual Top-5 Accuracy: {top_5_accuracy_manual:.4f}")


# In[53]:


# Generate a list of class names ordered by label index
target_names = [label_to_writer[i] for i in range(num_classes)]


# In[54]:


# Generate and print the final classification report
print("Final Classification Report on Test Set:")
print(classification_report(test_true_classes, test_pred_classes, target_names=target_names))


# In[55]:

# Generate the classification report as a dictionary
report_dict = classification_report(test_true_classes, test_pred_classes, target_names=target_names, output_dict=True)

# Convert the dictionary to a pandas DataFrame
report_df = pd.DataFrame(report_dict).transpose()

# Save the DataFrame to a CSV file
report_df.to_csv(f'{ATTN}{BACKBONE}{FINETUNE}classification_report_{SEED}.csv', index=True)

print(f"Final Classification Report saved as '{ATTN}{BACKBONE}{FINETUNE}classification_report_{SEED}.csv'.")


# In[ ]:


# Save the history.history dictionary to a file
with open(f'{ATTN}{BACKBONE}{FINETUNE}history_seed_{SEED}.pkl', 'wb') as f:
    pickle.dump(history.history, f)

# Save the evaluation metrics to a JSON file, including top-5 accuracy
test_metrics = {
    'test_loss': test_loss,
    'test_accuracy': test_accuracy,
    'test_top_5_accuracy': test_top_5_accuracy,  # Added top-5 accuracy
    'test_precision': test_precision,
    'test_recall': test_recall,
    'test_f1_score': test_f1_score
}
with open(f'{ATTN}{BACKBONE}{FINETUNE}test_metrics_seed_{SEED}.json', 'w') as f:
    json.dump(test_metrics, f)

# Save the elapsed time to a file
with open(f'{ATTN}{BACKBONE}{FINETUNE}elapsed_time_seed_{SEED}.txt', 'w') as f:
    f.write(str(elapsed_time))


# In[ ]:


gc.collect()


# In[ ]:


# Generate the confusion matrix
cm = confusion_matrix(test_true_classes, test_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(24, 24))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=target_names,
    yticklabels=target_names
)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix on Test Set')
# plt.show()
plt.savefig(f'{ATTN}{BACKBONE}{FINETUNE}confusion_matrix_{SEED}.png')
# plt.close()


# In[ ]:


gc.collect()


# In[ ]:


# Plot training & validation Accuracy
plt.figure(figsize=(12, 6))

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.axhline(y=test_accuracy, color='c', linestyle='--', label='Test Accuracy')  # Add Test Accuracy
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(f'{ATTN}{BACKBONE}{FINETUNE}accuracy_{SEED}.png')
plt.show()
plt.close()


# In[ ]:


# Plot training & validation Top-5 Accuracy
plt.figure(figsize=(12, 6))

plt.plot(history.history['top_5_accuracy'], label='Train Top-5 Accuracy')
plt.plot(history.history['val_top_5_accuracy'], label='Validation Top-5 Accuracy')
plt.axhline(y=test_top_5_accuracy, color='c', linestyle='--', label='Test Top-5 Accuracy')  # Add Test Top-5 Accuracy
plt.title('Model Top-5 Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Top-5 Accuracy')
plt.legend()
plt.savefig(f'{ATTN}{BACKBONE}{FINETUNE}top_5_accuracy_{SEED}.png')
plt.show()
plt.close()


# In[ ]:


# Plot training & validation loss
plt.figure(figsize=(12, 6))

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.axhline(y=test_loss, color='c', linestyle='--', label='Test Loss')  # Add Test Loss
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'{ATTN}{BACKBONE}{FINETUNE}loss_{SEED}.png')
plt.show()
plt.close()


# In[ ]:


# Plot training & validation F1 Score
plt.figure(figsize=(12, 6))

plt.plot(history.history['f1_score'], label='Train F1 Score')
plt.plot(history.history['val_f1_score'], label='Validation F1 Score')
plt.axhline(y=test_f1_score, color='c', linestyle='--', label='Test F1 Score')  # Add Test F1 Score
plt.title('Model F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()
plt.savefig(f'{ATTN}{BACKBONE}{FINETUNE}f1_score_{SEED}.png')
plt.show()
plt.close()


# In[ ]:


# Plot training & validation precision
plt.figure(figsize=(12, 6))

plt.plot(history.history['precision'], label='Train Precision')
plt.plot(history.history['val_precision'], label='Validation Precision')
plt.axhline(y=test_precision, color='c', linestyle='--', label='Test Precision')  # Add Test Precision
plt.title('Model Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend()
plt.savefig(f'{ATTN}{BACKBONE}{FINETUNE}precision_{SEED}.png')
plt.show()
plt.close()


# In[ ]:


# Plot training & validation recall
plt.figure(figsize=(12, 6))

plt.plot(history.history['recall'], label='Train Recall')
plt.plot(history.history['val_recall'], label='Validation Recall')
plt.axhline(y=test_recall, color='c', linestyle='--', label='Test Recall')  # Add Test Recall
plt.title('Model Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend()
plt.savefig(f'{ATTN}{BACKBONE}{FINETUNE}recall_{SEED}.png')
plt.show()
plt.close()

