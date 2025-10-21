# -*- coding: utf-8 -*-
"""
Diagnostic Analysis: Why Synthetic Fraud Data Has No Patterns

Investigating whether data imputation destroyed patterns in the original data.
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

print("="*70)
print("DIAGNOSING SYNTHETIC FRAUD DATASET GENERATION")
print("="*70)

# Load the dataset
df = pd.read_csv("synthetic_fraud_dataset.csv")

print(f"\nDataset shape: {df.shape}")
print(f"Fraud rate: {df['Fraud_Label'].mean():.2%}")

# Check for missing values
print("\n" + "="*70)
print("MISSING VALUES CHECK")
print("="*70)
missing = df.isnull().sum()
if missing.sum() > 0:
    print("\n⚠️  Current dataset has missing values:")
    for col, count in missing[missing > 0].items():
        print(f"  {col}: {count:,} ({count/len(df)*100:.2f}%)")
else:
    print("\n✓ No missing values in current dataset")

# Check for imputation artifacts
print("\n" + "="*70)
print("IMPUTATION ARTIFACT DETECTION")
print("="*70)

print("\nLooking for signs of median/mode imputation...")

# For continuous features, check if there are suspicious peaks at median
continuous_cols = ['Transaction_Amount', 'Account_Balance', 'Daily_Transaction_Count',
                   'Avg_Transaction_Amount_7d', 'Failed_Transaction_Count_7d', 'Card_Age',
                   'Transaction_Distance', 'Risk_Score']

print("\n1. Continuous Features - Checking for median clustering:")
for col in continuous_cols:
    if col in df.columns:
        median_val = df[col].median()
        # Count how many values are exactly at median
        at_median = (df[col] == median_val).sum()
        pct_at_median = at_median / len(df) * 100
        
        # Also check for "too uniform" distribution
        std = df[col].std()
        mean = df[col].mean()
        cv = std / mean if mean != 0 else 0  # Coefficient of variation
        
        print(f"\n  {col}:")
        print(f"    Median: {median_val:.2f}")
        print(f"    Values at median: {at_median} ({pct_at_median:.2f}%)")
        print(f"    Coefficient of variation: {cv:.4f}")
        
        if pct_at_median > 5:
            print(f"    ⚠️  SUSPICIOUS: {pct_at_median:.1f}% of values at median!")

# For categorical features, check for mode clustering
categorical_cols = ['Transaction_Type', 'Device_Type', 'Location', 'Merchant_Category',
                    'Card_Type', 'Authentication_Method']

print("\n2. Categorical Features - Checking for mode clustering:")
for col in categorical_cols:
    if col in df.columns:
        value_counts = df[col].value_counts()
        mode_val = value_counts.index[0]
        mode_count = value_counts.iloc[0]
        pct_at_mode = mode_count / len(df) * 100
        
        print(f"\n  {col}:")
        print(f"    Mode: '{mode_val}'")
        print(f"    Values at mode: {mode_count} ({pct_at_mode:.2f}%)")
        print(f"    Unique values: {df[col].nunique()}")
        print(f"    Distribution: {value_counts.to_dict()}")
        
        if pct_at_mode > 50:
            print(f"    ⚠️  SUSPICIOUS: {pct_at_mode:.1f}% of values are mode!")

# Check if data appears "too uniform"
print("\n" + "="*70)
print("UNIFORMITY ANALYSIS")
print("="*70)

print("\nChecking if categorical features have suspiciously uniform distributions...")

for col in categorical_cols:
    if col in df.columns:
        value_counts = df[col].value_counts()
        # Calculate chi-square for uniformity
        expected = len(df) / len(value_counts)
        chi2 = ((value_counts - expected) ** 2 / expected).sum()
        
        # For uniform distribution, chi2 should be close to 0
        print(f"\n  {col}:")
        print(f"    Chi-square test for uniformity: {chi2:.2f}")
        print(f"    Distribution:")
        for val, count in value_counts.items():
            pct = count / len(df) * 100
            print(f"      {val}: {count:,} ({pct:.2f}%)")
        
        if chi2 < 100:  # Very low chi-square = very uniform
            print(f"    ⚠️  HIGHLY UNIFORM DISTRIBUTION (Chi² = {chi2:.2f})")

# Check for relationships between features and target
print("\n" + "="*70)
print("FEATURE-TARGET RELATIONSHIPS")
print("="*70)

print("\nAnalyzing if imputation destroyed predictive patterns...")

def cramers_v(x, y):
    """Calculate Cramér's V statistic for categorical association."""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1
    return np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

print("\nCramér's V scores (0 = no association, 1 = perfect association):")
for col in categorical_cols:
    if col in df.columns:
        v = cramers_v(df[col], df['Fraud_Label'])
        print(f"  {col}: {v:.6f}")
        if v < 0.01:
            print(f"    → Extremely weak association!")

# Check if the data was generated randomly
print("\n" + "="*70)
print("SYNTHETIC DATA GENERATION HYPOTHESIS")
print("="*70)

print("\nPossible scenarios:")
print("\n1. ❌ Original data had missing values, imputation destroyed patterns")
print("   • Median/mode filling creates artificial uniformity")
print("   • Original correlations lost")
print("   • Explains weak Cramér's V scores")

print("\n2. ❌ Data was generated with random sampling")
print("   • Features randomly sampled from uniform distributions")
print("   • No real relationships between features and fraud")
print("   • Explains uniform chi-square scores")

print("\n3. ✓ Data generation needs fraud-correlated rules")
print("   • Example: High transaction amounts → higher fraud rate")
print("   • Example: Certain locations/devices → suspicious patterns")
print("   • Example: Failed transactions → fraud indicator")

# Recommendations
print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)

print("\nTo fix this dataset, you should:")
print("\n1. Generate synthetic data WITH correlations:")
print("   • Define rules: e.g., 'High amount + Weekend = 60% fraud chance'")
print("   • Use conditional probabilities for feature generation")
print("   • Ensure features have predictive relationships with fraud")

print("\n2. Avoid imputation that destroys patterns:")
print("   • If data has missing values, use more sophisticated methods")
print("   • Or generate complete data from the start")
print("   • Never fill with simple median/mode if it loses information")

print("\n3. Validate data quality:")
print("   • Check Cramér's V > 0.05 for at least some features")
print("   • Verify fraud rates vary across feature categories")
print("   • Test with simple models (logistic regression) first")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

# Final verdict
max_cramers_v = 0
for col in categorical_cols:
    if col in df.columns:
        v = cramers_v(df[col], df['Fraud_Label'])
        max_cramers_v = max(max_cramers_v, v)

print(f"\nMaximum Cramér's V: {max_cramers_v:.6f}")

if max_cramers_v < 0.02:
    print("\n⚠️  VERDICT: Dataset has NO MEANINGFUL PATTERNS")
    print("   • All Cramér's V scores < 0.02 (extremely weak)")
    print("   • Data appears randomly generated or heavily imputed")
    print("   • Bayesian Networks CANNOT learn from random noise")
    print("   • Need to regenerate data with actual correlations")
else:
    print("\n✓ Dataset has some patterns (though may be weak)")
    print("   • Consider feature engineering to amplify signals")

print("\n" + "="*70)
