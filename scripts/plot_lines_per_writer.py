#!/usr/bin/env python
"""
Generate a horizontal bar chart showing lines (images) per writer for all 179 writers.
Matches the style of the pages-per-writer chart.
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os

# Set style
plt.style.use('default')
# Convert 163mm x 300mm to inches (1 inch = 25.4mm)
fig_width = 163 / 25.4  # 6.417 inches
fig_height = 300 / 25.4  # 11.811 inches
plt.rcParams['figure.figsize'] = (fig_width, fig_height)
plt.rcParams['font.size'] = 7

# Resolve path - handle running from repo root or scripts/
if os.path.exists('stats/writer_line_counts_sorted.csv'):
    csv_path = 'stats/writer_line_counts_sorted.csv'
elif os.path.exists('../stats/writer_line_counts_sorted.csv'):
    csv_path = '../stats/writer_line_counts_sorted.csv'
else:
    raise FileNotFoundError("Cannot find writer_line_counts_sorted.csv")

# Load data
df = pd.read_csv(csv_path)

# Sort by number of lines (ascending for horizontal bar chart)
df = df.sort_values('lines_total', ascending=True)

print(f"Found {len(df)} writers")
print(f"Line count range: {df['lines_total'].min()} to {df['lines_total'].max()}")
print(f"Total lines: {df['lines_total'].sum()}")
print(f"Figure size: {fig_width:.2f} x {fig_height:.2f} inches (163mm x 300mm)")

# Create horizontal bar chart (163mm x 300mm)
fig, ax = plt.subplots(figsize=(fig_width, fig_height))

# Create bars
bars = ax.barh(df['writer'], df['lines_total'], 
               color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5, height=0.8)

# Add value labels on bars
for i, (writer, lines) in enumerate(zip(df['writer'], df['lines_total'])):
    ax.text(lines + 5, i, str(int(lines)), 
            va='center', ha='left', fontsize=4.5, fontweight='bold')

# Formatting
ax.set_xlabel('Number of Lines', fontsize=10, fontweight='bold')
ax.set_ylabel('Writers', fontsize=10, fontweight='bold')
ax.set_title('Number of Lines Per Writer\n(All 179 Writers)', 
             fontsize=11, fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Remove whitespace at top and bottom
ax.set_ylim(-0.5, len(df) - 0.5)

# Adjust y-axis labels (writer names) to be more readable - smaller to avoid touching
ax.tick_params(axis='y', labelsize=4.8)
ax.tick_params(axis='x', labelsize=8)

# Tight layout
plt.tight_layout()

# Save figure
output_path = 'stats/lines_per_writer_all179.png'
if not os.path.exists('stats'):
    output_path = '../stats/lines_per_writer_all179.png'

plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nSaved chart to: {output_path}")

# Also save as PDF for publication quality
pdf_path = output_path.replace('.png', '.pdf')
plt.savefig(pdf_path, bbox_inches='tight')
print(f"Saved PDF to: {pdf_path}")

plt.close()

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Number of writers: {len(df)}")
print(f"Total lines: {df['lines_total'].sum()}")
print(f"Mean lines per writer: {df['lines_total'].mean():.2f}")
print(f"Median lines per writer: {df['lines_total'].median():.0f}")
print(f"Min lines: {df['lines_total'].min()}")
print(f"Max lines: {df['lines_total'].max()}")
print(f"Std dev: {df['lines_total'].std():.2f}")
print("="*60)

# Print top 10 writers by line count
print("\nTop 10 writers by line count:")
print(df.nlargest(10, 'lines_total')[['writer', 'lines_total']])

# Print bottom 10 writers by line count
print("\nBottom 10 writers by line count:")
print(df.nsmallest(10, 'lines_total')[['writer', 'lines_total']])
