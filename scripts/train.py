#!/usr/bin/env python
"""
Main training script for Writer Identification Model
Supports multiple backbones, training strategies, and configurations via command-line arguments
"""
import os
import argparse
import json
import pickle
import math
import time
import gc
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras import backend as K
from sklearn.metrics import classification_report, confusion_matrix

# Import custom modules
from model_builder import build_writer_identification_model
from data_utils import (
    load_dataset_metadata, 
    prepare_data_splits, 
    create_lazy_dataset
)
from custom_metrics import MacroPrecision, MacroRecall, MacroF1Score
from custom_callbacks import ClearOutputEveryNEpochs, PeriodicModelCheckpoint
from utils.page_disjoint import load_split_map


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Train Writer Identification Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet50', 'densenet201', 'xception', 'mobilenetv3'],
                        help='Backbone architecture')
    parser.add_argument('--training-mode', type=str, default='frozen',
                        choices=['frozen', 'scratch', 'finetune_last_n', 'finetune_all'],
                        help='Training strategy')
    parser.add_argument('--num-trainable-layers', type=int, default=None,
                        help='Number of layers to finetune (required for finetune_last_n mode)')
    parser.add_argument('--use-attention', action='store_true',
                        help='Use attention mechanism')
    parser.add_argument('--num-clusters', type=int, default=64,
                        help='Number of clusters for NetVLAD')
    
    # Data configuration
    parser.add_argument('--data-dir', type=str, default='./Lines',
                        help='Directory containing image data')
    parser.add_argument('--csv-file', type=str, default='merged_writer.csv',
                        help='CSV file with writer labels')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Image size (square)')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Number of writer classes (auto-detected if not specified)')
    parser.add_argument('--split-mode', type=str, default='line',
                        choices=['line', 'page_disjoint'],
                        help='Data splitting mode: line-level or page-disjoint')
    parser.add_argument('--split-dir', type=str, default='./splits',
                        help='Directory containing page-disjoint split files')
    parser.add_argument('--disjoint-mode', type=str, default='page',
                        choices=['page', 'document'],
                        help='Disjoint mode: page or document (only used with --split-mode page_disjoint)')
    parser.add_argument('--writer-policy', type=str, default=None,
                        choices=[None, 'require_3way', 'drop_if_lt2', 'drop_if_lt3', 'allow_train_test_only'],
                        help='Writer policy suffix for split files (e.g., require_3way)')
    
    # Training configuration
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--epochs', type=int, default=450,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--early-stop-patience', type=int, default=50,
                        help='Early stopping patience')
    parser.add_argument('--lr-patience', type=int, default=10,
                        help='Learning rate reduction patience')
    
    # GPU configuration
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID to use')
    parser.add_argument('--disable-gpu', action='store_true',
                        help='Disable GPU and use CPU')
    
    # Output configuration
    parser.add_argument('--output-dir', type=str, default='./Results',
                        help='Directory to save results')
    parser.add_argument('--save-freq', type=int, default=50,
                        help='Save model every N epochs')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Custom experiment name (auto-generated if not specified)')
    
    # Other options
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Validation
    if args.training_mode == 'finetune_last_n' and args.num_trainable_layers is None:
        parser.error("--num-trainable-layers is required when using finetune_last_n mode")
    
    return args


def setup_gpu(args):
    """Configure GPU settings"""
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    if args.disable_gpu:
        tf.config.set_visible_devices([], 'GPU')
        print("GPU disabled. Using CPU.")
    else:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Use specified GPU
                if args.gpu < len(gpus):
                    tf.config.set_visible_devices(gpus[args.gpu], 'GPU')
                    tf.config.experimental.set_memory_growth(gpus[args.gpu], True)
                    print(f"Using GPU {args.gpu}: {gpus[args.gpu]}")
                else:
                    print(f"Warning: GPU {args.gpu} not found. Using GPU 0.")
                    tf.config.set_visible_devices(gpus[0], 'GPU')
                    tf.config.experimental.set_memory_growth(gpus[0], True)
            except RuntimeError as e:
                print(f"GPU setup error: {e}")
        else:
            print("No GPU found. Using CPU.")


def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    import random
    random.seed(seed)


def generate_experiment_name(args):
    """Generate experiment name from arguments"""
    if args.experiment_name:
        return args.experiment_name
    
    # Build name from configuration
    split_prefix = ""
    if args.split_mode == 'page_disjoint':
        if args.disjoint_mode == 'document':
            split_prefix = "DocDisj_"
        else:
            split_prefix = "PgDisj_"
        
        # Add writer policy if specified
        if args.writer_policy:
            split_prefix += f"{args.writer_policy}_"
    
    attention_str = "ATTN" if args.use_attention else "NoATTN"
    
    if args.training_mode == 'finetune_last_n':
        mode_str = f"Finetune{args.num_trainable_layers}Layers"
    else:
        mode_str = args.training_mode.capitalize()
    
    name = f"{split_prefix}{args.backbone}_{mode_str}_{attention_str}_seed{args.seed}"
    return name


def save_results(args, history, test_metrics, elapsed_time, experiment_name, output_dir):
    """Save training results to files"""
    # Save history
    history_path = os.path.join(output_dir, f'{experiment_name}_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"Saved training history to {history_path}")
    
    # Save test metrics
    metrics_path = os.path.join(output_dir, f'{experiment_name}_test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=4)
    print(f"Saved test metrics to {metrics_path}")
    
    # Save elapsed time
    time_path = os.path.join(output_dir, f'{experiment_name}_elapsed_time.txt')
    with open(time_path, 'w') as f:
        f.write(f"Total training time: {elapsed_time:.2f} seconds\n")
        f.write(f"Hours: {elapsed_time / 3600:.4f}\n")
        f.write(f"Minutes: {elapsed_time / 60:.4f}\n")
    print(f"Saved elapsed time to {time_path}")
    
    # Save configuration
    config_path = os.path.join(output_dir, f'{experiment_name}_config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(f"Saved configuration to {config_path}")


def plot_metrics(history, test_metrics, experiment_name, output_dir):
    """Generate and save training metric plots"""
    metrics_to_plot = [
        ('accuracy', 'Accuracy'),
        ('top_5_accuracy', 'Top-5 Accuracy'),
        ('loss', 'Loss'),
        ('f1_score', 'F1 Score'),
        ('precision', 'Precision'),
        ('recall', 'Recall')
    ]
    
    for metric_key, metric_name in metrics_to_plot:
        if metric_key in history.history:
            plt.figure(figsize=(12, 6))
            
            plt.plot(history.history[metric_key], label=f'Train {metric_name}')
            plt.plot(history.history[f'val_{metric_key}'], label=f'Validation {metric_name}')
            
            # Add test metric line
            test_key = f'test_{metric_key}'
            if test_key in test_metrics:
                plt.axhline(
                    y=test_metrics[test_key], 
                    color='c', 
                    linestyle='--', 
                    label=f'Test {metric_name}'
                )
            
            plt.title(f'Model {metric_name}')
            plt.xlabel('Epoch')
            plt.ylabel(metric_name)
            plt.legend()
            
            plot_path = os.path.join(output_dir, f'{experiment_name}_{metric_key}.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved {metric_name} plot to {plot_path}")


def save_classification_report(y_true, y_pred, label_to_writer, num_classes, experiment_name, output_dir):
    """Generate and save classification report"""
    # Generate target names
    target_names = [label_to_writer[i] for i in range(num_classes)]
    
    # Print classification report
    print("\nClassification Report on Test Set:")
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    # Save as CSV
    report_dict = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    
    report_path = os.path.join(output_dir, f'{experiment_name}_classification_report.csv')
    report_df.to_csv(report_path, index=True)
    print(f"Saved classification report to {report_path}")


def save_confusion_matrix(y_true, y_pred, label_to_writer, num_classes, experiment_name, output_dir):
    """Generate and save confusion matrix"""
    target_names = [label_to_writer[i] for i in range(num_classes)]
    
    cm = confusion_matrix(y_true, y_pred)
    
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
    
    cm_path = os.path.join(output_dir, f'{experiment_name}_confusion_matrix.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved confusion matrix to {cm_path}")


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    # Setup
    setup_gpu(args)
    set_seed(args.seed)
    
    # Load data first to determine num_classes
    print("\nLoading dataset metadata...")
    
    # We need to build model first to get preprocess_fn, but use dummy num_classes
    # Then rebuild after knowing actual num_classes
    from model_builder import get_backbone_model
    _, preprocess_fn = get_backbone_model(args.backbone, (args.image_size, args.image_size, 3))
    
    allowed_page_ids = None
    if args.split_mode == 'page_disjoint':
        split_map = load_split_map(
            args.split_dir,
            args.seed,
            disjoint_mode=args.disjoint_mode,
            writer_policy=args.writer_policy
        )
        allowed_page_ids = set(split_map.keys())
        print(f"Filtering dataset to {len(allowed_page_ids)} pages from split file")

    image_paths, labels, page_ids, writer_to_label, label_to_writer = load_dataset_metadata(
        main_dir=args.data_dir,
        csv_file=args.csv_file,
        verbose=args.verbose,
        allowed_page_ids=allowed_page_ids
    )
    
    # Determine num_classes from the data
    actual_num_classes = len(writer_to_label)
    if args.num_classes is None:
        args.num_classes = actual_num_classes
        print(f"Auto-detected {args.num_classes} writer classes")
    elif args.num_classes != actual_num_classes:
        print(f"Warning: Specified num_classes ({args.num_classes}) differs from actual ({actual_num_classes})")
        print(f"Using actual: {actual_num_classes}")
        args.num_classes = actual_num_classes
    
    # Generate experiment name
    experiment_name = generate_experiment_name(args)
    print(f"\n{'='*80}")
    print(f"Experiment: {experiment_name}")
    print(f"{'='*80}\n")
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, args.backbone)
    os.makedirs(output_dir, exist_ok=True)
    
    # Build model
    print("Building model...")
    model, preprocess_fn = build_writer_identification_model(
        backbone_name=args.backbone,
        input_shape=(args.image_size, args.image_size, 3),
        num_clusters=args.num_clusters,
        num_classes=args.num_classes,
        training_mode=args.training_mode,
        num_trainable_layers=args.num_trainable_layers,
        use_attention=args.use_attention
    )
    
    if args.verbose:
        model.summary()
    
    # Compile model
    print("\nCompiling model...")
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            TopKCategoricalAccuracy(k=5, name='top_5_accuracy'),
            MacroPrecision(num_classes=args.num_classes, name='precision'),
            MacroRecall(num_classes=args.num_classes, name='recall'),
            MacroF1Score(num_classes=args.num_classes, name='f1_score')
        ]
    )
    
    # One-hot encode labels
    labels_categorical = to_categorical(labels, num_classes=args.num_classes)
    
    # Split data
    print(f"\nSplitting dataset (mode: {args.split_mode})...")
    train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = prepare_data_splits(
        image_paths, 
        labels_categorical,
        page_ids=page_ids,
        split_mode=args.split_mode,
        split_dir=args.split_dir,
        disjoint_mode=args.disjoint_mode,
        writer_policy=args.writer_policy,
        seed=args.seed,
        verbose=args.verbose
    )
    
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    print(f"Test samples: {len(test_paths)}")
    
    # Clean up memory
    del image_paths, labels_categorical
    gc.collect()
    
    # Create lazy data loaders (images loaded on-demand)
    print("\nCreating lazy data loaders...")
    train_dataset = create_lazy_dataset(
        train_paths, train_labels,
        image_size=(args.image_size, args.image_size),
        preprocess_fn=preprocess_fn,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed
    )
    
    val_dataset = create_lazy_dataset(
        val_paths, val_labels,
        image_size=(args.image_size, args.image_size),
        preprocess_fn=preprocess_fn,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    test_dataset = create_lazy_dataset(
        test_paths, test_labels,
        image_size=(args.image_size, args.image_size),
        preprocess_fn=preprocess_fn,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Calculate steps
    steps_per_epoch = math.ceil(len(train_paths) / args.batch_size)
    validation_steps = math.ceil(len(val_paths) / args.batch_size)
    
    # Setup callbacks
    print("\nSetting up callbacks...")
    callbacks = []
    
    # Model checkpoint
    checkpoint_path = os.path.join(output_dir, f'{experiment_name}_best_model.keras')
    checkpoint_callback = ModelCheckpoint(
        checkpoint_path,
        monitor='val_f1_score',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    callbacks.append(checkpoint_callback)
    
    # Periodic checkpoint
    periodic_checkpoint_path = os.path.join(output_dir, f'{experiment_name}_last_saved.keras')
    periodic_checkpoint_callback = PeriodicModelCheckpoint(
        filepath=periodic_checkpoint_path,
        save_freq_epochs=args.save_freq,
        save_best_only=False,
        verbose=1
    )
    callbacks.append(periodic_checkpoint_callback)
    
    # Learning rate reduction
    lr_reduce = ReduceLROnPlateau(
        monitor='val_f1_score',
        factor=0.5,
        patience=args.lr_patience,
        verbose=1,
        mode='max',
        min_lr=1e-8
    )
    callbacks.append(lr_reduce)
    
    # Early stopping
    early_stop = EarlyStopping(
        monitor='val_f1_score',
        patience=args.early_stop_patience,
        restore_best_weights=True,
        mode='max',
        verbose=1
    )
    callbacks.append(early_stop)
    
    # Optional: Clear output callback (for notebooks)
    # callbacks.append(ClearOutputEveryNEpochs(n=10))
    
    # Train model
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    history = model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1 if args.verbose else 2
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\nTraining completed in {elapsed_time / 3600:.2f} hours")
    
    # Evaluate on test set
    print("\n" + "="*80)
    print("Evaluating on test set...")
    print("="*80 + "\n")
    
    test_generator.reset()
    test_results = model.evaluate(
        test_dataset,
        steps=math.ceil(len(test_paths) / args.batch_size),
        verbose=1
    )
    
    test_metrics = {
        'test_loss': test_results[0],
        'test_accuracy': test_results[1],
        'test_top_5_accuracy': test_results[2],
        'test_precision': test_results[3],
        'test_recall': test_results[4],
        'test_f1_score': test_results[5]
    }
    
    print("\nTest Results:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Generate predictions
    print("\nGenerating predictions...")
    test_generator.reset()
    test_preds = model.predict(
        test_dataset,
        steps=math.ceil(len(test_paths) / args.batch_size),
        verbose=0
    )
    
    test_pred_classes = np.argmax(test_preds, axis=1)
    test_true_classes = np.argmax(test_labels, axis=1)
    
    # Save results
    print("\nSaving results...")
    save_results(args, history, test_metrics, elapsed_time, experiment_name, output_dir)
    
    # Generate and save plots
    if not args.no_plots:
        print("\nGenerating plots...")
        plot_metrics(history, test_metrics, experiment_name, output_dir)
        
        # Save classification report
        save_classification_report(
            test_true_classes, test_pred_classes, 
            label_to_writer, args.num_classes,
            experiment_name, output_dir
        )
        
        # Save confusion matrix
        save_confusion_matrix(
            test_true_classes, test_pred_classes,
            label_to_writer, args.num_classes,
            experiment_name, output_dir
        )
    
    # Clean up
    K.clear_session()
    gc.collect()
    
    print("\n" + "="*80)
    print(f"Experiment '{experiment_name}' completed successfully!")
    print(f"Results saved to: {output_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
