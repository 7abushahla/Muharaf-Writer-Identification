#!/usr/bin/env python3
"""
Generate results table in CSV format from aggregated results.

Creates tables with one per metric (F1, Accuracy, etc.) with rows for each
configuration and columns for each backbone.
"""

import os
import json
import pandas as pd
import argparse
from pathlib import Path


# Mapping from config names to display names
CONFIG_DISPLAY_NAMES = {
    "Frozen_NoATTN": "Frozen + No Attention (Baseline) ",
    "Frozen_ATTN": "Frozen + Attention",
    "Scratch_NoATTN": "From Scratch + No Attention",
    "Scratch_ATTN": "From Scratch + Attention",
    "FinetuneAll_NoATTN": "Fine-tuned + No Attention",
    "FinetuneAll_ATTN": "Fine-tuned + Attention",
    "FinetuneLast1_NoATTN": "Fine-tuned + Last Layer + No Attention",
    "FinetuneLast1_ATTN": "Fine-tuned + Last Layer + Attention",
    "FinetuneLast5_NoATTN": "Fine-tuned + Last 5 Layers + No Attention",
    "FinetuneLast5_ATTN": "Fine-tuned + Last 5 Layers + Attention",
    "FinetuneLast10_NoATTN": "Fine-tuned + Last 10 Layers + No Attention",
    "FinetuneLast10_ATTN": "Fine-tuned + Last 10 Layers + Attention",
    "FinetuneLast25_NoATTN": "Fine-tuned + Last 25 Layers + No Attention",
    "FinetuneLast25_ATTN": "Fine-tuned + Last 25 Layers + Attention",
}

# Order of configurations (rows in table) - matches 01-14 .sh file order
CONFIG_ORDER = [
    "Frozen_NoATTN",           # 01
    "Frozen_ATTN",             # 02
    "FinetuneLast1_NoATTN",    # 03
    "FinetuneLast1_ATTN",      # 04
    "FinetuneLast5_NoATTN",    # 05
    "FinetuneLast5_ATTN",      # 06
    "FinetuneLast10_NoATTN",   # 07
    "FinetuneLast10_ATTN",     # 08
    "FinetuneLast25_NoATTN",   # 09
    "FinetuneLast25_ATTN",     # 10
    "FinetuneAll_NoATTN",      # 11
    "FinetuneAll_ATTN",        # 12
    "Scratch_NoATTN",          # 13
    "Scratch_ATTN",            # 14
]

# Backbone names (columns in table)
BACKBONES = ["resnet50", "densenet201", "xception", "mobilenetv3"]
BACKBONE_DISPLAY = {
    "resnet50": "ResNet50",
    "densenet201": "DenseNet201",
    "xception": "Xception",
    "mobilenetv3": "MobileNetV3"
}

# Metrics to create tables for
METRICS = {
    "test_f1_score": "F1 Score",
    "test_accuracy": "Top-1 Accuracy",
    "test_top_5_accuracy": "Top-5 Accuracy",
    "test_loss": "Loss",
    "test_precision": "Precision",
    "test_recall": "Recall",
    "elapsed_time_hours": "Training Time (Hours)",
}


def extract_config_name(dir_name):
    """Extract config name from directory name like 'PgDisj_resnet50_Frozen_NoATTN'"""
    # Remove prefix and backbone name
    parts = dir_name.split('_')
    # Find the config part (after backbone)
    if 'PgDisj' in parts[0]:
        # Format: PgDisj_backbone_config_parts
        return '_'.join(parts[2:])
    return None


def load_aggregated_results(results_dir):
    """Load all aggregated results from Results directory"""
    results = {}
    
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return results
    
    # Iterate through backbones
    for backbone in BACKBONES:
        backbone_dir = results_path / backbone
        if not backbone_dir.exists():
            continue
        
        # Iterate through config directories
        for config_dir in backbone_dir.iterdir():
            if not config_dir.is_dir():
                continue
            
            # Look for aggregated results
            agg_file = config_dir / 'aggregated' / 'aggregated_metrics.json'
            if not agg_file.exists():
                continue
            
            # Load the aggregated metrics
            with open(agg_file, 'r') as f:
                data = json.load(f)
            
            # Extract config name
            config_name = extract_config_name(config_dir.name)
            if not config_name:
                continue
            
            # Store results
            if config_name not in results:
                results[config_name] = {}
            results[config_name][backbone] = data
    
    return results


def create_metric_table(results, metric_key):
    """Create a pandas DataFrame for a specific metric"""
    # Create full table with all configurations, even if data is missing
    rows = []
    
    for config in CONFIG_ORDER:
        row_data = {
            'Model': CONFIG_DISPLAY_NAMES.get(config, config)
        }
        
        for backbone in BACKBONES:
            # Check if we have data for this config and backbone
            if config in results and backbone in results[config]:
                # Handle elapsed_time specially - it's not in metrics_formatted
                if metric_key == 'elapsed_time_hours':
                    formatted_value = results[config][backbone].get('elapsed_time_hours_formatted', '')
                else:
                    formatted_value = results[config][backbone]['metrics_formatted'].get(metric_key, '')
                row_data[BACKBONE_DISPLAY[backbone]] = formatted_value
            else:
                # Leave empty if no data available yet
                row_data[BACKBONE_DISPLAY[backbone]] = ''
        
        rows.append(row_data)
    
    df = pd.DataFrame(rows)
    return df


def generate_results_tables(results_dir, output_file):
    """Generate CSV file with all metrics tables"""
    print(f"Loading aggregated results from: {results_dir}")
    results = load_aggregated_results(results_dir)
    
    if not results:
        print("No aggregated results found - generating empty table template")
    else:
        print(f"Found results for {len(results)}/{len(CONFIG_ORDER)} configurations")
    
    # Create dataframes for each metric
    all_tables = []
    
    for metric_key, metric_name in METRICS.items():
        print(f"Creating table for: {metric_name}")
        
        # Create metric title rows
        title_df = pd.DataFrame([[metric_name, '', '', '', '']], 
                                columns=['Model', 'ResNet50', 'DenseNet201', 'Xception', 'MobileNetV3'])
        
        # Create metric table
        metric_df = create_metric_table(results, metric_key)
        
        # Combine title and data
        combined = pd.concat([title_df, metric_df], ignore_index=True)
        
        # Add empty rows after each table
        empty_rows = pd.DataFrame([['', '', '', '', ''] for _ in range(3)],
                                 columns=['Model', 'ResNet50', 'DenseNet201', 'Xception', 'MobileNetV3'])
        combined = pd.concat([combined, empty_rows], ignore_index=True)
        
        all_tables.append(combined)
    
    # Combine all tables
    final_df = pd.concat(all_tables, ignore_index=True)
    
    # Save to CSV
    final_df.to_csv(output_file, index=False)
    print(f"Saved CSV to: {output_file}")
    
    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description='Generate results table from aggregated results')
    parser.add_argument('--results-dir', type=str, default='./Results',
                        help='Directory containing aggregated results')
    parser.add_argument('--output', type=str, default='./Results/page_disjoint_results.csv',
                        help='Output CSV file path')
    args = parser.parse_args()
    
    generate_results_tables(args.results_dir, args.output)


if __name__ == '__main__':
    main()
