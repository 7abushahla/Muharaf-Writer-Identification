#!/usr/bin/env python
"""
Generate a horizontal bar chart showing pages per writer for the 71-writer subset (>=3 pages).
Similar to the lines-per-writer histogram but for pages.
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os

# Set style
plt.style.use('default')
# Convert 163mm x 300mm to inches (1 inch = 25.4mm) - same as lines chart
fig_width = 163 / 25.4  # 6.417 inches
fig_height = 300 / 25.4  # 11.811 inches
plt.rcParams['figure.figsize'] = (fig_width, fig_height)
plt.rcParams['font.size'] = 7

# Resolve path - handle running from repo root or scripts/
if os.path.exists('stats/page_disjoint_writer_eligibility.csv'):
    csv_path = 'stats/page_disjoint_writer_eligibility.csv'
elif os.path.exists('../stats/page_disjoint_writer_eligibility.csv'):
    csv_path = '../stats/page_disjoint_writer_eligibility.csv'
else:
    raise FileNotFoundError("Cannot find page_disjoint_writer_eligibility.csv")

# Load data
df = pd.read_csv(csv_path)

# Filter to writers with >=3 pages (eligibility = 'train_val_test')
df_eligible = df[df['eligibility'] == 'train_val_test'].copy()

# Sort by number of pages (ascending for horizontal bar chart)
df_eligible = df_eligible.sort_values('pages_total', ascending=True)

print(f"Found {len(df_eligible)} writers with >=3 pages")
print(f"Page count range: {df_eligible['pages_total'].min()} to {df_eligible['pages_total'].max()}")
print(f"Total pages: {df_eligible['pages_total'].sum()}")
print(f"Total lines: {df_eligible['lines_total'].sum()}")
print(f"Figure size: {fig_width:.2f} x {fig_height:.2f} inches (163mm x 300mm)")

# Create horizontal bar chart (163mm x 300mm)
fig, ax = plt.subplots(figsize=(fig_width, fig_height))

# Create bars
bars = ax.barh(df_eligible['writer'], df_eligible['pages_total'], 
               color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5, height=0.8)

# Add value labels on bars
for i, (writer, pages) in enumerate(zip(df_eligible['writer'], df_eligible['pages_total'])):
    ax.text(pages + 0.5, i, str(int(pages)), 
            va='center', ha='left', fontsize=6.5, fontweight='bold')

# Formatting
ax.set_xlabel('Number of Pages', fontsize=10, fontweight='bold')
ax.set_ylabel('Writers', fontsize=10, fontweight='bold')
ax.set_title('Number of Pages Per Writer\n(71 Writers)', 
             fontsize=11, fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Remove whitespace at top and bottom
ax.set_ylim(-0.5, len(df_eligible) - 0.5)

# Adjust y-axis labels (writer names) to be more readable - larger since only 71 writers
ax.tick_params(axis='y', labelsize=7)
ax.tick_params(axis='x', labelsize=8)

# Tight layout
plt.tight_layout()

# Save figure
output_path = 'stats/pages_per_writer_71subset.png'
if not os.path.exists('stats'):
    output_path = '../stats/pages_per_writer_71subset.png'

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
print(f"Number of writers: {len(df_eligible)}")
print(f"Total pages: {df_eligible['pages_total'].sum()}")
print(f"Total lines: {df_eligible['lines_total'].sum()}")
print(f"Mean pages per writer: {df_eligible['pages_total'].mean():.2f}")
print(f"Median pages per writer: {df_eligible['pages_total'].median():.0f}")
print(f"Min pages: {df_eligible['pages_total'].min()}")
print(f"Max pages: {df_eligible['pages_total'].max()}")
print(f"Std dev: {df_eligible['pages_total'].std():.2f}")
print("="*60)

# Print top 10 writers by page count
print("\nTop 10 writers by page count:")
print(df_eligible.nlargest(10, 'pages_total')[['writer', 'pages_total', 'lines_total']])
