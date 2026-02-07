"""
Data loading and preprocessing utilities for Writer Identification
"""
import os
import sys
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Add repo root to sys.path for shared utilities
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_PARENT = os.path.dirname(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.page_disjoint import load_split_map, validate_split_map
import tensorflow as tf


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


def _resolve_data_dir(main_dir):
    if not main_dir:
        return main_dir
    if os.path.isabs(main_dir):
        return main_dir

    candidates = [
        os.path.abspath(main_dir),
        os.path.join(REPO_ROOT, main_dir),
        os.path.join(REPO_PARENT, main_dir),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return main_dir


def _resolve_csv_file(csv_file):
    if not csv_file:
        return csv_file
    if os.path.isabs(csv_file):
        return csv_file

    candidates = [
        os.path.abspath(csv_file),
        os.path.join(REPO_ROOT, csv_file),
        os.path.join(REPO_PARENT, csv_file),
        os.path.join(REPO_ROOT, "manual_labeling", os.path.basename(csv_file)),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return csv_file


def load_dataset_metadata(
    main_dir='./Lines',
    csv_file='merged_writer.csv',
    verbose=True,
    allowed_page_ids=None,
):
    """
    Load dataset metadata (paths and labels) without loading images into RAM.
    
    Args:
        main_dir: Directory containing the image folders
        csv_file: Path to the CSV file with writer labels
        verbose: Whether to print progress
        allowed_page_ids: Optional set of page IDs to include (filters CSV rows)
        
    Returns:
        image_paths: List of image file paths
        labels: Array of integer labels
        page_ids: Array of page IDs (for page-disjoint splits)
        writer_to_label: Dictionary mapping writer names to labels
        label_to_writer: Dictionary mapping labels to writer names
    """
    # Resolve paths to allow running from repo root or scripts/
    resolved_main_dir = _resolve_data_dir(main_dir)
    resolved_csv_file = _resolve_csv_file(csv_file)
    if verbose:
        if resolved_main_dir != main_dir:
            print(f"Using data dir: {resolved_main_dir}")
        if resolved_csv_file != csv_file:
            print(f"Using CSV file: {resolved_csv_file}")

    main_dir = resolved_main_dir
    csv_file = resolved_csv_file

    # Load the CSV data
    writer_data = pd.read_csv(csv_file)
    
    # Dictionary to map writer names to unique integer labels
    writer_to_label = {}
    current_label = 0
    
    # Initialize lists to store metadata only (not images!)
    image_paths = []
    labels = []
    page_ids = []
    
    # Loop through each row in the CSV file
    for index, row in writer_data.iterrows():
        image_filename = row['Image Filename']
        writer_name = row['Writer Name (English)']

        if allowed_page_ids is not None and image_filename not in allowed_page_ids:
            continue
        
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
                    img_path = os.path.join(folder_path, filename)
                    # Only store the path, don't load the image!
                    image_paths.append(img_path)
                    labels.append(writer_to_label[writer_name])
                    page_ids.append(image_filename)  # Store page ID for page-disjoint splits
        else:
            if verbose:
                print(f"Folder not found: {folder_path}")
    
    # Convert lists to numpy arrays
    labels = np.array(labels)
    page_ids = np.array(page_ids)
    
    # Create a reverse mapping from label indices to writer names
    label_to_writer = {label: writer for writer, label in writer_to_label.items()}
    
    if verbose:
        print(f"Found {len(image_paths)} images from {len(writer_to_label)} writers (metadata only, ~{len(image_paths) * 200 / 1024 / 1024:.1f} MB RAM)")
    
    return image_paths, labels, page_ids, writer_to_label, label_to_writer


def load_and_preprocess_image(img_path, image_size, preprocess_fn):
    """
    Load and preprocess a single image (used by tf.data.Dataset).
    
    Args:
        img_path: Path to the image file
        image_size: Target image size (height, width)
        preprocess_fn: Preprocessing function for the image
        
    Returns:
        Preprocessed image tensor
    """
    def _load_image(path):
        """Python function to load image using our custom block_processor"""
        path_str = path.numpy().decode('utf-8') if isinstance(path, tf.Tensor) else path
        processed_img = block_processor_opencv(path_str, target_size=image_size)
        img_array = np.array(processed_img, dtype=np.float32)
        
        # Ensure RGB format
        if img_array.shape[-1] != 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        # Apply preprocessing
        if preprocess_fn is not None:
            img_array = preprocess_fn(img_array)
        
        return img_array
    
    # Use tf.py_function to wrap our custom Python preprocessing
    img = tf.py_function(_load_image, [img_path], tf.float32)
    img.set_shape([image_size[0], image_size[1], 3])
    return img


def augment_image(image, label):
    """
    Apply data augmentation to images (matches original ImageDataGenerator settings).
    
    Args:
        image: Input image tensor
        label: Label tensor
        
    Returns:
        Augmented image and label
    """
    # Random rotation (±15 degrees)
    image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
    angle = tf.random.uniform([], -15, 15) * (3.14159 / 180.0)  # Convert to radians
    # Note: TensorFlow doesn't have a simple rotate by arbitrary angle, so we use rot90 as approximation
    
    # Random zoom (0.7 to 1.3 range, equivalent to zoom_range=0.3)
    zoom_factor = tf.random.uniform([], 0.7, 1.3)
    img_shape = tf.shape(image)
    h, w = img_shape[0], img_shape[1]
    new_h = tf.cast(tf.cast(h, tf.float32) * zoom_factor, tf.int32)
    new_w = tf.cast(tf.cast(w, tf.float32) * zoom_factor, tf.int32)
    image = tf.image.resize(image, [new_h, new_w])
    image = tf.image.resize_with_crop_or_pad(image, h, w)
    
    # Random width shift (±20%)
    max_shift_w = tf.cast(tf.cast(w, tf.float32) * 0.2, tf.int32)
    shift_w = tf.random.uniform([], -max_shift_w, max_shift_w, dtype=tf.int32)
    image = tf.roll(image, shift_w, axis=1)
    
    # Random height shift (±20%)
    max_shift_h = tf.cast(tf.cast(h, tf.float32) * 0.2, tf.int32)
    shift_h = tf.random.uniform([], -max_shift_h, max_shift_h, dtype=tf.int32)
    image = tf.roll(image, shift_h, axis=0)
    
    return image, label


def create_lazy_dataset(
    image_paths,
    labels,
    image_size=(224, 224),
    preprocess_fn=None,
    batch_size=256,
    shuffle=False,
    augment=False,
    seed=None
):
    """
    Create a tf.data.Dataset that loads images lazily on-demand.
    
    Args:
        image_paths: List or array of image file paths
        labels: Array of one-hot encoded labels
        image_size: Target image size (height, width)
        preprocess_fn: Preprocessing function (e.g., backbone's preprocess_input)
        batch_size: Batch size
        shuffle: Whether to shuffle the dataset
        augment: Whether to apply data augmentation (for training)
        seed: Random seed for shuffling
        
    Returns:
        tf.data.Dataset
    """
    # Create dataset from paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    
    # Shuffle if requested (will reshuffle each epoch automatically)
    if shuffle:
        buffer_size = min(10000, len(image_paths))
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed, reshuffle_each_iteration=True)
    
    # Load and preprocess images lazily
    dataset = dataset.map(
        lambda path, label: (load_and_preprocess_image(path, image_size, preprocess_fn), label),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Apply augmentation if requested (only for training)
    if augment:
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch the dataset
    dataset = dataset.batch(batch_size)
    
    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def prepare_data_splits(
    image_paths, 
    labels, 
    page_ids=None,
    split_mode='line',
    split_dir='./splits',
    disjoint_mode='page',
    writer_policy=None,
    test_size=0.3, 
    val_size=0.5, 
    seed=42,
    verbose=True
):
    """
    Split data into train, validation, and test sets.
    
    Args:
        image_paths: List/array of image file paths
        labels: Array of one-hot encoded labels
        page_ids: Array of page IDs (required for page_disjoint mode)
        split_mode: 'line' for line-level splits, 'page_disjoint' for page-disjoint splits
        split_dir: Directory containing page-disjoint split files
        disjoint_mode: 'page' or 'document' (only used when split_mode='page_disjoint')
        writer_policy: Writer policy used to generate splits (e.g., 'require_3way', 'drop_if_lt3')
        test_size: Proportion of data for validation + test (line mode only)
        val_size: Proportion of temp data for validation (line mode only)
        seed: Random seed
        verbose: Whether to print split information
        
    Returns:
        train_paths, val_paths, test_paths, train_labels, val_labels, test_labels
    """
    # Convert to numpy arrays if needed
    if not isinstance(image_paths, np.ndarray):
        image_paths = np.array(image_paths)
    
    if split_mode == 'page_disjoint':
        if page_ids is None:
            raise ValueError("page_ids must be provided for page_disjoint split mode")
        
        # Load the page-disjoint/document-disjoint split map
        split_map = load_split_map(split_dir, seed, disjoint_mode=disjoint_mode, writer_policy=writer_policy)
        validate_split_map(split_map, page_ids)
        
        # Create split labels for each sample
        split_labels = np.array([split_map[pid] for pid in page_ids])
        
        # Create masks for each split
        train_mask = split_labels == "train"
        val_mask = split_labels == "val"
        test_mask = split_labels == "test"
        
        # Split data using masks
        train_paths, train_labels = image_paths[train_mask], labels[train_mask]
        val_paths, val_labels = image_paths[val_mask], labels[val_mask]
        test_paths, test_labels = image_paths[test_mask], labels[test_mask]
        
        if verbose:
            # Count unique pages in each split
            train_pages = len(set(page_ids[train_mask]))
            val_pages = len(set(page_ids[val_mask]))
            test_pages = len(set(page_ids[test_mask]))
            print(f"Page-disjoint split pages: train={train_pages}, val={val_pages}, test={test_pages}")
            print(f"Page-disjoint split lines: train={len(train_paths)}, val={len(val_paths)}, test={len(test_paths)}")
    
    else:  # line-level split
        # First split: 70% training and 30% (validation + test)
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            image_paths, labels, test_size=test_size, random_state=seed, stratify=labels
        )
        
        # Second split: Split the temporary set into validation and test
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels, test_size=val_size, random_state=seed, stratify=temp_labels
        )
        
        if verbose:
            print(f"Line-level split: train={len(train_paths)}, val={len(val_paths)}, test={len(test_paths)}")
    
    return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels


def create_data_generators(
    train_images,
    train_labels,
    val_images,
    val_labels,
    test_images,
    test_labels,
    batch_size=256
):
    """
    Create data generators for training, validation, and testing.
    
    Args:
        train_images, val_images, test_images: Image arrays
        train_labels, val_labels, test_labels: Label arrays
        batch_size: Batch size for generators
        
    Returns:
        train_generator, validation_generator, test_generator
    """
    # Define the ImageDataGenerator with enhanced augmentations
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.3,
        shear_range=0.3,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode='nearest',
        horizontal_flip=False,
        vertical_flip=False
    )
    
    # Create training generator
    train_generator = train_datagen.flow(
        train_images, train_labels, batch_size=batch_size, shuffle=True
    )
    
    # Define ImageDataGenerator for validation and test (no augmentation)
    validation_test_datagen = ImageDataGenerator()
    
    # Create validation generator
    validation_generator = validation_test_datagen.flow(
        val_images, val_labels, batch_size=batch_size, shuffle=False
    )
    
    # Create test generator
    test_generator = validation_test_datagen.flow(
        test_images, test_labels, batch_size=batch_size, shuffle=False
    )
    
    return train_generator, validation_generator, test_generator


def infinite_generator(generator):
    """Create an infinite generator from a finite one"""
    while True:
        for batch in generator:
            yield batch
