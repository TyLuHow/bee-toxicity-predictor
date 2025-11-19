#!/usr/bin/env python3
"""
Create exploratory visualizations for the bee toxicity dataset.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os

# Create output directory
os.makedirs('outputs/visualizations', exist_ok=True)

# Set visualization style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10

print("Loading dataset...")
df = pd.read_csv('data/raw/dataset_with_descriptors.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}\n")

# ============================================================================
# 1. Target Distribution with Chemical Types
# ============================================================================
print("Creating visualization 1: Target Distribution by Chemical Type...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Overall distribution
label_counts = df['label'].value_counts().sort_index()
colors = ['#3498db', '#e74c3c']
axes[0].bar(['Non-toxic', 'Toxic'], label_counts.values, color=colors, alpha=0.8, edgecolor='black')
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Overall Toxicity Distribution', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)
for i, v in enumerate(label_counts.values):
    axes[0].text(i, v + 10, f'{v}\n({v/len(df)*100:.1f}%)', ha='center', fontweight='bold')

# By chemical type
chem_types = ['herbicide', 'fungicide', 'insecticide', 'other_agrochemical']
type_data = []
for ct in chem_types:
    if ct in df.columns:
        toxic = df[(df[ct] == 1) & (df['label'] == 1)].shape[0]
        non_toxic = df[(df[ct] == 1) & (df['label'] == 0)].shape[0]
        type_data.append([ct.replace('_', ' ').title(), non_toxic, toxic])

type_df = pd.DataFrame(type_data, columns=['Type', 'Non-toxic', 'Toxic'])
x = np.arange(len(type_df))
width = 0.35
axes[1].bar(x - width/2, type_df['Non-toxic'], width, label='Non-toxic', color=colors[0], alpha=0.8)
axes[1].bar(x + width/2, type_df['Toxic'], width, label='Toxic', color=colors[1], alpha=0.8)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Toxicity by Chemical Type', fontsize=14, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(type_df['Type'], rotation=45, ha='right')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

# Toxicity rate by type
type_df['Toxic_Rate'] = type_df['Toxic'] / (type_df['Toxic'] + type_df['Non-toxic']) * 100
axes[2].barh(type_df['Type'], type_df['Toxic_Rate'], color='#e67e22', alpha=0.8, edgecolor='black')
axes[2].set_xlabel('Toxicity Rate (%)', fontsize=12)
axes[2].set_title('Toxicity Rate by Chemical Type', fontsize=14, fontweight='bold')
axes[2].grid(axis='x', alpha=0.3)
for i, v in enumerate(type_df['Toxic_Rate']):
    axes[2].text(v + 1, i, f'{v:.1f}%', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/visualizations/1_target_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: outputs/visualizations/1_target_distribution.png\n")

# ============================================================================
# 2. Molecular Properties Distribution
# ============================================================================
print("Creating visualization 2: Molecular Properties Distribution...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

key_features = ['MolecularWeight', 'LogP', 'TPSA', 'HeavyAtomCount', 'NumRotatableBonds', 'NumAromaticRings']

for idx, feature in enumerate(key_features):
    if feature in df.columns:
        data = df[feature].dropna()
        axes[idx].hist(data, bins=40, color='steelblue', alpha=0.7, edgecolor='black')
        axes[idx].axvline(data.mean(), color='red', linestyle='--', linewidth=2,
                         label=f'Mean: {data.mean():.2f}')
        axes[idx].axvline(data.median(), color='green', linestyle='--', linewidth=2,
                         label=f'Median: {data.median():.2f}')
        axes[idx].set_xlabel(feature, fontsize=11)
        axes[idx].set_ylabel('Frequency', fontsize=11)
        axes[idx].set_title(f'{feature} Distribution', fontsize=12, fontweight='bold')
        axes[idx].legend(fontsize=9)
        axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/visualizations/2_molecular_properties.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: outputs/visualizations/2_molecular_properties.png\n")

# ============================================================================
# 3. Toxic vs Non-Toxic Comparison
# ============================================================================
print("Creating visualization 3: Toxic vs Non-Toxic Comparison...")
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

comparison_features = ['MolecularWeight', 'LogP', 'TPSA', 'HeavyAtomCount',
                       'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds', 'NumAromaticRings']

for idx, feature in enumerate(comparison_features):
    if feature in df.columns:
        toxic_data = df[df['label'] == 1][feature].dropna()
        non_toxic_data = df[df['label'] == 0][feature].dropna()

        # Create violin plot
        parts = axes[idx].violinplot([non_toxic_data, toxic_data],
                                     positions=[1, 2],
                                     showmeans=True,
                                     showmedians=True)

        # Color the violins
        for pc, color in zip(parts['bodies'], [colors[0], colors[1]]):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        axes[idx].set_xticks([1, 2])
        axes[idx].set_xticklabels(['Non-toxic', 'Toxic'])
        axes[idx].set_ylabel(feature, fontsize=11)
        axes[idx].set_title(f'{feature}', fontsize=12, fontweight='bold')
        axes[idx].grid(axis='y', alpha=0.3)

        # Add statistical info
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(non_toxic_data, toxic_data)
        sig_text = '***' if p_value < 0.001 else ('**' if p_value < 0.01 else ('*' if p_value < 0.05 else 'ns'))
        axes[idx].text(0.5, 0.95, f'p={p_value:.4f} {sig_text}',
                      transform=axes[idx].transAxes,
                      ha='center', va='top', fontsize=9,
                      bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))

plt.tight_layout()
plt.savefig('outputs/visualizations/3_toxic_vs_nontoxic.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: outputs/visualizations/3_toxic_vs_nontoxic.png\n")

# ============================================================================
# 4. Feature Correlation Heatmap
# ============================================================================
print("Creating visualization 4: Feature Correlation Analysis...")
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Select numeric features
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'label' in numeric_cols:
    numeric_cols.remove('label')
if 'CID' in numeric_cols:
    numeric_cols.remove('CID')

# Correlation with target
correlations = df[numeric_cols].corrwith(df['label']).sort_values(ascending=False)
top_features = correlations.abs().sort_values(ascending=False).head(15)

axes[0].barh(range(len(top_features)), top_features.values, color='teal', alpha=0.8)
axes[0].set_yticks(range(len(top_features)))
axes[0].set_yticklabels(top_features.index, fontsize=10)
axes[0].set_xlabel('Absolute Correlation with Toxicity', fontsize=12)
axes[0].set_title('Top 15 Features Correlated with Toxicity', fontsize=14, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)
axes[0].invert_yaxis()

# Correlation heatmap of top features
top_feature_names = correlations.abs().sort_values(ascending=False).head(10).index.tolist()
corr_matrix = df[top_feature_names].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            ax=axes[1], square=True, linewidths=1, cbar_kws={'label': 'Correlation'})
axes[1].set_title('Top 10 Features Correlation Heatmap', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/visualizations/4_feature_correlations.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: outputs/visualizations/4_feature_correlations.png\n")

# ============================================================================
# 5. Time Trends and Sources
# ============================================================================
print("Creating visualization 5: Temporal and Source Analysis...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Year distribution
if 'year' in df.columns:
    year_counts = df.groupby('year').size()
    axes[0, 0].plot(year_counts.index, year_counts.values, marker='o', linewidth=2, markersize=4, color='darkblue')
    axes[0, 0].fill_between(year_counts.index, year_counts.values, alpha=0.3, color='darkblue')
    axes[0, 0].set_xlabel('Year', fontsize=12)
    axes[0, 0].set_ylabel('Number of Compounds', fontsize=12)
    axes[0, 0].set_title('Temporal Distribution of Compounds', fontsize=14, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)

# Toxicity over time
if 'year' in df.columns:
    year_toxicity = df.groupby('year')['label'].agg(['sum', 'count'])
    year_toxicity['rate'] = (year_toxicity['sum'] / year_toxicity['count'] * 100)

    # Only plot years with sufficient data
    year_toxicity_filtered = year_toxicity[year_toxicity['count'] >= 5]

    axes[0, 1].scatter(year_toxicity_filtered.index, year_toxicity_filtered['rate'],
                      s=year_toxicity_filtered['count']*3, alpha=0.6, color='crimson', edgecolors='black')
    axes[0, 1].set_xlabel('Year', fontsize=12)
    axes[0, 1].set_ylabel('Toxicity Rate (%)', fontsize=12)
    axes[0, 1].set_title('Toxicity Rate Over Time\n(bubble size = sample count)', fontsize=14, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)

# Source distribution
if 'source' in df.columns:
    source_counts = df['source'].value_counts()
    axes[1, 0].pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%',
                   startangle=90, colors=sns.color_palette('pastel'))
    axes[1, 0].set_title('Distribution by Data Source', fontsize=14, fontweight='bold')

# Toxicity by source
if 'source' in df.columns:
    source_tox = df.groupby(['source', 'label']).size().unstack(fill_value=0)
    source_tox.plot(kind='bar', stacked=True, ax=axes[1, 1], color=colors, alpha=0.8, edgecolor='black')
    axes[1, 1].set_xlabel('Source', fontsize=12)
    axes[1, 1].set_ylabel('Count', fontsize=12)
    axes[1, 1].set_title('Toxicity Distribution by Source', fontsize=14, fontweight='bold')
    axes[1, 1].legend(['Non-toxic', 'Toxic'])
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45, ha='right')
    axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/visualizations/5_temporal_source_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: outputs/visualizations/5_temporal_source_analysis.png\n")

# ============================================================================
# 6. Summary Statistics
# ============================================================================
print("Creating visualization 6: Summary Statistics Dashboard...")
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

# Overall stats
ax1 = fig.add_subplot(gs[0, :])
ax1.axis('off')
stats_text = f"""
DATASET SUMMARY STATISTICS

Total Compounds: {len(df)}
Toxic Compounds: {(df['label']==1).sum()} ({(df['label']==1).sum()/len(df)*100:.1f}%)
Non-toxic Compounds: {(df['label']==0).sum()} ({(df['label']==0).sum()/len(df)*100:.1f}%)
Imbalance Ratio: {(df['label']==0).sum() / (df['label']==1).sum():.2f}:1

Year Range: {df['year'].min():.0f} - {df['year'].max():.0f}
Number of Features: {len(df.columns)}
Data Sources: {', '.join(df['source'].unique())}
"""
ax1.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
ax1.set_title('Dataset Overview', fontsize=16, fontweight='bold', pad=20)

# Top correlated features
ax2 = fig.add_subplot(gs[1, :2])
top_10_corr = correlations.abs().sort_values(ascending=True).tail(10)
ax2.barh(range(len(top_10_corr)), top_10_corr.values, color='coral', alpha=0.8)
ax2.set_yticks(range(len(top_10_corr)))
ax2.set_yticklabels(top_10_corr.index, fontsize=10)
ax2.set_xlabel('Absolute Correlation', fontsize=11)
ax2.set_title('Top 10 Most Correlated Features', fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
for i, v in enumerate(top_10_corr.values):
    ax2.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

# Chemical type breakdown
ax3 = fig.add_subplot(gs[1, 2])
chem_totals = [df[ct].sum() for ct in chem_types if ct in df.columns]
chem_labels = [ct.replace('_', '\n').title() for ct in chem_types if ct in df.columns]
ax3.bar(range(len(chem_totals)), chem_totals, color=['#1abc9c', '#9b59b6', '#f39c12', '#34495e'], alpha=0.8)
ax3.set_xticks(range(len(chem_labels)))
ax3.set_xticklabels(chem_labels, fontsize=9)
ax3.set_ylabel('Count', fontsize=11)
ax3.set_title('Chemical Type Distribution', fontsize=12, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
for i, v in enumerate(chem_totals):
    ax3.text(i, v + 5, str(int(v)), ha='center', fontsize=9, fontweight='bold')

# Molecular weight distribution by toxicity
ax4 = fig.add_subplot(gs[2, :])
if 'MolecularWeight' in df.columns:
    bins = np.linspace(df['MolecularWeight'].min(), df['MolecularWeight'].max(), 40)
    ax4.hist(df[df['label']==0]['MolecularWeight'], bins=bins, alpha=0.6,
            label='Non-toxic', color=colors[0], edgecolor='black')
    ax4.hist(df[df['label']==1]['MolecularWeight'], bins=bins, alpha=0.6,
            label='Toxic', color=colors[1], edgecolor='black')
    ax4.set_xlabel('Molecular Weight (g/mol)', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('Molecular Weight Distribution by Toxicity', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(axis='y', alpha=0.3)

plt.savefig('outputs/visualizations/6_summary_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: outputs/visualizations/6_summary_dashboard.png\n")

print("="*70)
print("ALL VISUALIZATIONS CREATED SUCCESSFULLY!")
print("="*70)
print("\nGenerated visualizations:")
print("  1. outputs/visualizations/1_target_distribution.png")
print("  2. outputs/visualizations/2_molecular_properties.png")
print("  3. outputs/visualizations/3_toxic_vs_nontoxic.png")
print("  4. outputs/visualizations/4_feature_correlations.png")
print("  5. outputs/visualizations/5_temporal_source_analysis.png")
print("  6. outputs/visualizations/6_summary_dashboard.png")
print("\nView these images to explore the dataset!")
