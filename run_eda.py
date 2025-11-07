#!/usr/bin/env python3
"""
Run Exploratory Data Analysis and generate all visualizations.
This script executes the EDA notebook as a Python script.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import os

# Create output directory if it doesn't exist
os.makedirs('outputs/figures', exist_ok=True)

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Set random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("="*80)
print("PHASE 1: EXPLORATORY DATA ANALYSIS")
print("ApisTox Dataset - Honey Bee Pesticide Toxicity Prediction")
print("="*80)

# 1. Load Dataset
print("\n1. Loading dataset...")
df = pd.read_csv('outputs/dataset_final.csv')
print(f"✓ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# 2. Basic Info
print("\n2. Dataset info:")
print(f"   Columns: {', '.join(df.columns.tolist())}")
print(f"   Missing values: {df.isnull().sum().sum()}")

# 3. Target Analysis
print("\n3. Target variable analysis...")
label_counts = df['label'].value_counts().sort_index()
print(f"   Non-toxic (0): {label_counts[0]} ({label_counts[0]/len(df)*100:.1f}%)")
print(f"   Toxic (1): {label_counts[1]} ({label_counts[1]/len(df)*100:.1f}%)")
print(f"   Imbalance ratio: {label_counts[0]/label_counts[1]:.2f}:1")

# Target distribution visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(['Non-toxic (0)', 'Toxic (1)'], label_counts.values, color=['#2ecc71', '#e74c3c'])
axes[0].set_ylabel('Count')
axes[0].set_title('Target Variable Distribution', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)
for i, v in enumerate(label_counts.values):
    axes[0].text(i, v + 10, str(v), ha='center', fontweight='bold')

axes[1].pie(label_counts.values, labels=['Non-toxic (0)', 'Toxic (1)'], 
            autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], startangle=90)
axes[1].set_title('Target Variable Proportion', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/figures/target_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: outputs/figures/target_distribution.png")

# 4. Molecular Descriptor Extraction
print("\n4. Extracting molecular descriptors from SMILES...")

def calculate_molecular_descriptors(smiles):
    """Calculate molecular descriptors from SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    descriptors = {
        'MolecularWeight': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'NumAromaticRings': Descriptors.NumAromaticRings(mol),
        'TPSA': Descriptors.TPSA(mol),
        'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
        'NumRings': Descriptors.RingCount(mol),
        'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
        'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
        'FractionCSP3': Descriptors.FractionCSP3(mol),
        'MolarRefractivity': Descriptors.MolMR(mol),
        'BertzCT': Descriptors.BertzCT(mol),
        'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
    }
    return descriptors

descriptors_list = []
invalid_count = 0

for idx, row in df.iterrows():
    desc = calculate_molecular_descriptors(row['SMILES'])
    if desc is not None:
        desc['index'] = idx
        descriptors_list.append(desc)
    else:
        invalid_count += 1

df_descriptors = pd.DataFrame(descriptors_list)
df_descriptors = df_descriptors.set_index('index')
df_with_descriptors = pd.concat([df, df_descriptors], axis=1)

print(f"   ✓ Calculated descriptors for {len(df_descriptors)} compounds")
print(f"   ✗ Invalid SMILES: {invalid_count}")
print(f"   New shape: {df_with_descriptors.shape}")

# 5. Save processed dataset
os.makedirs('data/raw', exist_ok=True)
df_with_descriptors.to_csv('data/raw/dataset_with_descriptors.csv', index=False)
print(f"   ✓ Saved: data/raw/dataset_with_descriptors.csv")

# 6. Molecular descriptor distributions
print("\n5. Visualizing molecular descriptors...")
key_descriptors = ['MolecularWeight', 'LogP', 'NumHDonors', 'NumHAcceptors', 
                   'NumRotatableBonds', 'NumAromaticRings', 'TPSA', 'HeavyAtomCount']

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.flatten()

for idx, desc in enumerate(key_descriptors):
    if desc in df_descriptors.columns:
        axes[idx].hist(df_descriptors[desc], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        axes[idx].set_title(f'{desc} Distribution', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel(desc)
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(axis='y', alpha=0.3)
        mean_val = df_descriptors[desc].mean()
        axes[idx].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        axes[idx].legend()

axes[8].axis('off')
plt.tight_layout()
plt.savefig('outputs/figures/molecular_descriptors.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: outputs/figures/molecular_descriptors.png")

# 7. Feature correlations
print("\n6. Analyzing feature correlations...")
correlations = df_descriptors.corrwith(df_with_descriptors['label']).sort_values(ascending=False)
print("   Top 5 correlated features:")
for feat, corr in correlations.head(5).items():
    print(f"     {feat:25s}: {corr:+.4f}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
correlations_sorted = correlations.abs().sort_values(ascending=False)
axes[0].barh(range(len(correlations_sorted)), correlations_sorted.values)
axes[0].set_yticks(range(len(correlations_sorted)))
axes[0].set_yticklabels(correlations_sorted.index, fontsize=9)
axes[0].set_xlabel('Absolute Correlation with Toxicity')
axes[0].set_title('Feature Importance (Correlation)', fontsize=12, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)
axes[0].invert_yaxis()

top_features = correlations_sorted.head(10).index.tolist()
corr_matrix = df_descriptors[top_features].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            ax=axes[1], square=True, linewidths=1)
axes[1].set_title('Top 10 Feature Correlation Heatmap', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/figures/feature_correlations.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: outputs/figures/feature_correlations.png")

# 8. Toxicity comparison
print("\n7. Comparing toxic vs non-toxic compounds...")
key_desc = ['MolecularWeight', 'LogP', 'TPSA', 'HeavyAtomCount', 
            'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds', 'NumAromaticRings']

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for idx, desc in enumerate(key_desc):
    if desc in df_with_descriptors.columns:
        toxic_data = df_with_descriptors[df_with_descriptors['label'] == 1][desc].dropna()
        non_toxic_data = df_with_descriptors[df_with_descriptors['label'] == 0][desc].dropna()
        
        bp = axes[idx].boxplot([non_toxic_data, toxic_data], 
                              labels=['Non-toxic', 'Toxic'],
                              patch_artist=True,
                              showmeans=True)
        
        bp['boxes'][0].set_facecolor('#2ecc71')
        bp['boxes'][1].set_facecolor('#e74c3c')
        
        axes[idx].set_title(f'{desc}', fontsize=11, fontweight='bold')
        axes[idx].set_ylabel(desc)
        axes[idx].grid(axis='y', alpha=0.3)
        
        t_stat, p_value = stats.ttest_ind(non_toxic_data, toxic_data)
        axes[idx].text(0.5, 0.95, f'p={p_value:.4f}', 
                      transform=axes[idx].transAxes, 
                      ha='center', va='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('outputs/figures/toxicity_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: outputs/figures/toxicity_comparison.png")

# Summary
print("\n" + "="*80)
print("EDA COMPLETE - SUMMARY")
print("="*80)
print(f"Dataset Shape: {df_with_descriptors.shape}")
print(f"Features: {len(df_descriptors.columns)} molecular descriptors + {len(df.columns)} original")
print(f"Target: Binary (Non-toxic: {label_counts[0]}, Toxic: {label_counts[1]})")
print(f"Imbalance: {label_counts[0]/label_counts[1]:.2f}:1")
print(f"\nGenerated Figures:")
print(f"  - outputs/figures/target_distribution.png")
print(f"  - outputs/figures/molecular_descriptors.png")
print(f"  - outputs/figures/feature_correlations.png")
print(f"  - outputs/figures/toxicity_comparison.png")
print(f"\nProcessed Data:")
print(f"  - data/raw/dataset_with_descriptors.csv")
print("\n✓ Phase 1 Complete - Ready for modeling!")

