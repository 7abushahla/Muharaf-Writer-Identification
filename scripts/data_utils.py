"""
Data loading and preprocessing utilities for Writer Identification
"""
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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


def load_dataset(
    main_dir='./Lines',
    csv_file='merged_writer.csv',
    image_size=(224, 224),
    preprocess_fn=None,
    verbose=True
):
    """
    Load the writer identification dataset.
    
    Args:
        main_dir: Directory containing the image folders
        csv_file: Path to the CSV file with writer labels
        image_size: Target image size
        preprocess_fn: Preprocessing function for images
        verbose: Whether to print progress
        
    Returns:
        images: Array of preprocessed images
        labels: Array of integer labels
        writer_to_label: Dictionary mapping writer names to labels
        label_to_writer: Dictionary mapping labels to writer names
    """
    # Load the CSV data
    writer_data = pd.read_csv(csv_file)
    
    # Dictionary to map writer names to unique integer labels
    writer_to_label = {}
    current_label = 0
    
    # Initialize lists to store image data and labels
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
                        img_path = os.path.join(folder_path, filename)
                        # Process the image using BlockProcessor
                        processed_img = block_processor_opencv(img_path, target_size=image_size)
                        
                        # Convert PIL Image to NumPy array
                        img_array = np.array(processed_img)
                        
                        # Ensure the image is in RGB format
                        if img_array.shape[-1] != 3:
                            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                        
                        # Preprocess the image
                        if preprocess_fn is not None:
                            processed_img_array = preprocess_fn(img_array)
                        else:
                            processed_img_array = img_array
                        
                        # Append the image and the corresponding label
                        images.append(processed_img_array)
                        labels.append(writer_to_label[writer_name])
                        
                        del img_array, processed_img_array
                    except Exception as e:
                        if verbose:
                            print(f"Error processing image {img_path}: {e}")
        else:
            if verbose:
                print(f"Folder not found: {folder_path}")
    
    # Validate dataset sizes
    if len(images) != len(labels):
        raise ValueError(f"Number of images ({len(images)}) does not match number of labels ({len(labels)})")
    
    # Convert lists to numpy arrays
    images = np.array(images).reshape(-1, image_size[0], image_size[1], 3)
    labels = np.array(labels)
    
    # Create a reverse mapping from label indices to writer names
    label_to_writer = {label: writer for writer, label in writer_to_label.items()}
    
    if verbose:
        print(f"Loaded {len(images)} images from {len(writer_to_label)} writers")
    
    return images, labels, writer_to_label, label_to_writer


def prepare_data_splits(images, labels, test_size=0.3, val_size=0.5, seed=42):
    """
    Split data into train, validation, and test sets.
    
    Args:
        images: Array of images
        labels: Array of one-hot encoded labels
        test_size: Proportion of data for validation + test
        val_size: Proportion of temp data for validation (0.5 means equal val and test)
        seed: Random seed
        
    Returns:
        train_images, val_images, test_images, train_labels, val_labels, test_labels
    """
    # First split: 70% training and 30% (validation + test)
    train_images, temp_images, train_labels, temp_labels = train_test_split(
        images, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    
    # Second split: Split the temporary set into validation and test
    val_images, test_images, val_labels, test_labels = train_test_split(
        temp_images, temp_labels, test_size=val_size, random_state=seed, stratify=temp_labels
    )
    
    return train_images, val_images, test_images, train_labels, val_labels, test_labels


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
