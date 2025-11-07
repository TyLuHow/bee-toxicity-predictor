#!/usr/bin/env python3
"""Quick dataset exploration script."""
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('outputs/dataset_final.csv')

print("="*80)
print("DATASET OVERVIEW")
print("="*80)
print(f"\nDataset Shape: {df.shape}")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n" + "="*80)
print("COLUMN NAMES AND TYPES")
print("="*80)
for col, dtype in df.dtypes.items():
    print(f"{col:30s} {str(dtype):15s}")

print("\n" + "="*80)
print("MISSING VALUES")
print("="*80)
missing = df.isnull().sum()
if missing.sum() == 0:
    print("No missing values!")
else:
    print(missing[missing > 0])

print("\n" + "="*80)
print("TARGET VARIABLE: label (Binary Classification)")
print("="*80)
print(df['label'].value_counts().sort_index())
print(f"\nClass Balance:")
print(f"Non-toxic (0): {(df['label']==0).sum()} ({(df['label']==0).sum()/len(df)*100:.1f}%)")
print(f"Toxic (1):     {(df['label']==1).sum()} ({(df['label']==1).sum()/len(df)*100:.1f}%)")

print("\n" + "="*80)
print("PPDB LEVEL DISTRIBUTION (Multi-class Alternative)")
print("="*80)
print(df['ppdb_level'].value_counts().sort_index())

print("\n" + "="*80)
print("CATEGORICAL FEATURES")
print("="*80)
print("\nSource Distribution:")
print(df['source'].value_counts())

print("\nToxicity Type Distribution:")
print(df['toxicity_type'].value_counts())

print("\n" + "="*80)
print("AGROCHEMICAL TYPE FLAGS")
print("="*80)
for col in ['herbicide', 'fungicide', 'insecticide', 'other_agrochemical']:
    print(f"{col:25s}: {df[col].sum():4d} ({df[col].sum()/len(df)*100:.1f}%)")

print("\n" + "="*80)
print("YEAR RANGE")
print("="*80)
print(f"First publication year range: {df['year'].min()} - {df['year'].max()}")
print(f"Mean year: {df['year'].mean():.0f}")
print(f"Median year: {df['year'].median():.0f}")

print("\n" + "="*80)
print("SAMPLE RECORDS")
print("="*80)
print(df.head(10))

print("\n" + "="*80)
print("DATASET SUMMARY")
print("="*80)
print(df.describe(include='all'))

