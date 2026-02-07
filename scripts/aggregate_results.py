"""
Aggregate results across multiple seeds for writer identification experiments.

Computes mean and population standard deviation (N denominator) across seeds.
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def find_seed_directories(config_dir):
    """Find all seed_* subdirectories in a configuration directory"""
    seed_dirs = []
    for item in os.listdir(config_dir):
        item_path = os.path.join(config_dir, item)
        if os.path.isdir(item_path) and item.startswith('seed_'):
            seed_dirs.append(item_path)
    return sorted(seed_dirs)


def load_test_metrics(seed_dirs):
    """Load test metrics from all seed directories"""
    all_metrics = []
    seeds = []
    
    for seed_dir in seed_dirs:
        metrics_path = os.path.join(seed_dir, 'test_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                all_metrics.append(metrics)
                
                # Extract seed number from directory name
                seed_num = os.path.basename(seed_dir).replace('seed_', '')
                seeds.append(seed_num)
    
    return all_metrics, seeds


def aggregate_metrics(all_metrics):
    """Compute mean and std across metrics"""
    if not all_metrics:
        return None, None
    
    # Collect all metric keys
    metric_keys = set()
    for metrics in all_metrics:
        metric_keys.update(metrics.keys())
    
    # Compute mean and std for each metric
    aggregated_mean = {}
    aggregated_std = {}
    
    for key in metric_keys:
        values = [m[key] for m in all_metrics if key in m]
        if values:
            aggregated_mean[key] = float(np.mean(values))
            aggregated_std[key] = float(np.std(values, ddof=0))  # Population std (N denominator)
    
    return aggregated_mean, aggregated_std


def load_classification_reports(seed_dirs):
    """Load classification reports from all seed directories"""
    all_reports = []
    
    for seed_dir in seed_dirs:
        report_path = os.path.join(seed_dir, 'classification_report.csv')
        if os.path.exists(report_path):
            df = pd.read_csv(report_path)
            all_reports.append(df)
    
    return all_reports


def aggregate_classification_reports(all_reports):
    """Average classification reports across seeds with mean ± std formatting"""
    if not all_reports:
        return None
    
    # Assuming all reports have the same structure
    base_df = all_reports[0].copy()
    
    # Get numeric columns (precision, recall, f1-score, support)
    numeric_cols = ['precision', 'recall', 'f1-score']
    
    # Average across seeds for each class
    for col in numeric_cols:
        if col in base_df.columns:
            values = np.array([df[col].values for df in all_reports])
            means = np.mean(values, axis=0)
            stds = np.std(values, axis=0, ddof=0)  # Population std
            
            # Format as "mean ± std"
            formatted = [f"{m:.4f} ± {s:.4f}" for m, s in zip(means, stds)]
            base_df[col] = formatted
    
    # Support should be the same across seeds (just take from first)
    if 'support' in base_df.columns:
        base_df['support'] = all_reports[0]['support']
    
    return base_df


def save_aggregated_results(config_dir, aggregated_mean, aggregated_std, seeds):
    """Save aggregated metrics to JSON"""
    output_dir = os.path.join(config_dir, 'aggregated')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create combined results with mean and std
    combined = {
        'seeds': seeds,
        'num_seeds': len(seeds),
        'metrics_mean': aggregated_mean,
        'metrics_std': aggregated_std,
        'metrics_formatted': {}
    }
    
    # Format as "mean ± std"
    for key in aggregated_mean.keys():
        mean_val = aggregated_mean[key]
        std_val = aggregated_std[key]
        combined['metrics_formatted'][key] = f"{mean_val:.4f} ± {std_val:.4f}"
    
    # Save to JSON
    output_path = os.path.join(output_dir, 'aggregated_metrics.json')
    with open(output_path, 'w') as f:
        json.dump(combined, f, indent=2)
    
    print(f"Saved aggregated metrics to {output_path}")
    return output_path


def save_aggregated_classification_report(config_dir, aggregated_report):
    """Save aggregated classification report to CSV"""
    if aggregated_report is None:
        return None
    
    output_dir = os.path.join(config_dir, 'aggregated')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'classification_report_aggregated.csv')
    aggregated_report.to_csv(output_path, index=False)
    
    print(f"Saved aggregated classification report to {output_path}")
    return output_path


def print_summary(aggregated_mean, aggregated_std, seeds):
    """Print formatted summary of aggregated results"""
    print("\n" + "="*80)
    print(f"AGGREGATED RESULTS ACROSS {len(seeds)} SEEDS: {', '.join(seeds)}")
    print("="*80)
    
    print("\nTest Metrics (Mean ± Std):")
    print("-" * 50)
    
    # Order metrics nicely
    metric_order = ['test_accuracy', 'test_top_5_accuracy', 'test_precision', 
                    'test_recall', 'test_f1_score', 'test_loss']
    
    for key in metric_order:
        if key in aggregated_mean:
            mean_val = aggregated_mean[key]
            std_val = aggregated_std[key]
            print(f"{key:20s}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Print any remaining metrics
    for key in sorted(aggregated_mean.keys()):
        if key not in metric_order:
            mean_val = aggregated_mean[key]
            std_val = aggregated_std[key]
            print(f"{key:20s}: {mean_val:.4f} ± {std_val:.4f}")
    
    print("="*80 + "\n")


def aggregate_config_directory(config_dir, verbose=True):
    """Aggregate all results in a configuration directory"""
    if verbose:
        print(f"\nProcessing configuration: {os.path.basename(config_dir)}")
    
    # Find seed directories
    seed_dirs = find_seed_directories(config_dir)
    
    if not seed_dirs:
        print(f"  No seed directories found in {config_dir}")
        return False
    
    if verbose:
        print(f"  Found {len(seed_dirs)} seed directories: {[os.path.basename(d) for d in seed_dirs]}")
    
    # Load and aggregate test metrics
    all_metrics, seeds = load_test_metrics(seed_dirs)
    
    if not all_metrics:
        print(f"  No test metrics found")
        return False
    
    aggregated_mean, aggregated_std = aggregate_metrics(all_metrics)
    
    # Save aggregated metrics
    save_aggregated_results(config_dir, aggregated_mean, aggregated_std, seeds)
    
    # Load and aggregate classification reports
    all_reports = load_classification_reports(seed_dirs)
    if all_reports:
        aggregated_report = aggregate_classification_reports(all_reports)
        save_aggregated_classification_report(config_dir, aggregated_report)
    
    # Print summary
    if verbose:
        print_summary(aggregated_mean, aggregated_std, seeds)
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Aggregate results across seeds')
    parser.add_argument('config_dir', type=str, nargs='?',
                        help='Path to configuration directory (e.g., Results/densenet201/PgDisj_densenet201_Frozen_NoATTN)')
    parser.add_argument('--results-dir', type=str, default='./Results',
                        help='Root results directory (default: ./Results)')
    parser.add_argument('--backbone', type=str,
                        help='Specific backbone to process (processes all if not specified)')
    parser.add_argument('--all', action='store_true',
                        help='Process all configurations in results directory')
    args = parser.parse_args()
    
    if args.config_dir:
        # Process single configuration directory
        if not os.path.isdir(args.config_dir):
            print(f"Error: Directory not found: {args.config_dir}")
            return 1
        
        success = aggregate_config_directory(args.config_dir, verbose=True)
        return 0 if success else 1
    
    elif args.all or args.backbone:
        # Process multiple configurations
        if not os.path.isdir(args.results_dir):
            print(f"Error: Results directory not found: {args.results_dir}")
            return 1
        
        # Get backbones to process
        if args.backbone:
            backbones = [args.backbone]
        else:
            backbones = [d for d in os.listdir(args.results_dir) 
                        if os.path.isdir(os.path.join(args.results_dir, d))]
        
        total_processed = 0
        for backbone in backbones:
            backbone_dir = os.path.join(args.results_dir, backbone)
            if not os.path.isdir(backbone_dir):
                continue
            
            print(f"\n{'='*80}")
            print(f"Processing backbone: {backbone}")
            print('='*80)
            
            # Find all configuration directories
            config_dirs = [os.path.join(backbone_dir, d) 
                          for d in os.listdir(backbone_dir)
                          if os.path.isdir(os.path.join(backbone_dir, d)) 
                          and not d.startswith('.')]
            
            for config_dir in sorted(config_dirs):
                if aggregate_config_directory(config_dir, verbose=True):
                    total_processed += 1
        
        print(f"\nProcessed {total_processed} configurations")
        return 0
    
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
