"""
Bayesian Network Optimization Analysis
Finds optimal discretization bins and best feature subset for fraud detection.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, chi2, SelectKBest
from scipy.stats import chi2_contingency
import json

# Load data
print("Loading dataset...")
df = pd.read_csv("synthetic_fraud_dataset.csv")

print(f"Dataset: {len(df)} rows, {len(df.columns)} columns")
print(f"Fraud rate: {df['Fraud_Label'].mean():.2%}\n")

# Separate features by type
categorical_features = [
    'Transaction_Type', 'Device_Type', 'Location', 
    'Merchant_Category', 'IP_Address_Flag', 'Previous_Fraudulent_Activity',
    'Card_Type', 'Authentication_Method'
]

numerical_features = [
    'Transaction_Amount', 'Account_Age_Days', 'Transaction_Hour',
    'Days_Since_Last_Transaction', 'Average_Transaction_Amount',
    'Transaction_Count', 'Failed_Transactions'
]

# ============================================================================
# PART 1: ANALYZE CATEGORICAL FEATURES
# ============================================================================
print("="*70)
print("PART 1: CATEGORICAL FEATURE ANALYSIS")
print("="*70)

categorical_scores = []

for col in categorical_features:
    # Calculate mutual information with target
    contingency = pd.crosstab(df[col], df['Fraud_Label'])
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency)
    
    # Cramér's V (effect size for chi-squared)
    n = contingency.sum().sum()
    cramers_v = np.sqrt(chi2_stat / (n * (min(contingency.shape) - 1)))
    
    # Fraud rate variance across categories
    fraud_rates = df.groupby(col)['Fraud_Label'].mean()
    rate_variance = fraud_rates.var()
    rate_range = fraud_rates.max() - fraud_rates.min()
    
    categorical_scores.append({
        'Feature': col,
        'Chi2': chi2_stat,
        'P_Value': p_value,
        'Cramers_V': cramers_v,
        'Rate_Variance': rate_variance,
        'Rate_Range': rate_range,
        'N_Categories': df[col].nunique()
    })
    
    print(f"\n{col}:")
    print(f"  Categories: {df[col].nunique()}")
    print(f"  Chi² statistic: {chi2_stat:.2f} (p={p_value:.4f})")
    print(f"  Cramér's V: {cramers_v:.4f} (effect size)")
    print(f"  Fraud rate range: {rate_range:.4f}")
    print(f"  Fraud rates by category:")
    for cat, rate in fraud_rates.sort_values(ascending=False).head(5).items():
        print(f"    {cat}: {rate:.2%}")

cat_df = pd.DataFrame(categorical_scores).sort_values('Cramers_V', ascending=False)

print("\n" + "="*70)
print("CATEGORICAL FEATURES RANKED BY PREDICTIVE POWER (Cramér's V):")
print("="*70)
print(cat_df.to_string(index=False))

# ============================================================================
# PART 2: ANALYZE NUMERICAL FEATURES & FIND OPTIMAL BINS
# ============================================================================
print("\n" + "="*70)
print("PART 2: NUMERICAL FEATURE DISCRETIZATION ANALYSIS")
print("="*70)

numerical_scores = []

for col in numerical_features:
    if col not in df.columns:
        continue
    
    print(f"\n{col}:")
    print(f"  Range: [{df[col].min():.2f}, {df[col].max():.2f}]")
    print(f"  Mean: {df[col].mean():.2f}, Std: {df[col].std():.2f}")
    
    # Test different numbers of bins
    best_score = -1
    best_n_bins = 2
    bin_results = []
    
    for n_bins in [2, 3, 4, 5, 6, 8, 10]:
        try:
            # Discretize using quantiles
            binned = pd.qcut(df[col], q=n_bins, labels=False, duplicates='drop')
            
            # Calculate chi-squared
            contingency = pd.crosstab(binned, df['Fraud_Label'])
            chi2_stat, p_value, dof, expected = chi2_contingency(contingency)
            
            # Cramér's V
            n = contingency.sum().sum()
            cramers_v = np.sqrt(chi2_stat / (n * (min(contingency.shape) - 1)))
            
            # Fraud rate variance
            fraud_rates = df.groupby(binned)['Fraud_Label'].mean()
            rate_variance = fraud_rates.var()
            rate_range = fraud_rates.max() - fraud_rates.min()
            
            bin_results.append({
                'n_bins': n_bins,
                'chi2': chi2_stat,
                'cramers_v': cramers_v,
                'rate_variance': rate_variance,
                'rate_range': rate_range
            })
            
            if cramers_v > best_score:
                best_score = cramers_v
                best_n_bins = n_bins
        except Exception as e:
            pass
    
    print(f"\n  Binning results:")
    for result in bin_results:
        marker = " ✓ BEST" if result['n_bins'] == best_n_bins else ""
        print(f"    {result['n_bins']} bins: Cramér's V = {result['cramers_v']:.4f}, "
              f"Rate range = {result['rate_range']:.4f}{marker}")
    
    numerical_scores.append({
        'Feature': col,
        'Best_N_Bins': best_n_bins,
        'Best_Cramers_V': best_score,
        'Best_Rate_Range': bin_results[best_n_bins-2]['rate_range'] if best_n_bins-2 < len(bin_results) else 0
    })

num_df = pd.DataFrame(numerical_scores).sort_values('Best_Cramers_V', ascending=False)

print("\n" + "="*70)
print("NUMERICAL FEATURES RANKED BY PREDICTIVE POWER:")
print("="*70)
print(num_df.to_string(index=False))

# ============================================================================
# PART 3: COMBINED FEATURE SELECTION
# ============================================================================
print("\n" + "="*70)
print("PART 3: OPTIMAL FEATURE SUBSET SELECTION")
print("="*70)

# Combine all features with their scores
all_features = []

for _, row in cat_df.iterrows():
    all_features.append({
        'Feature': row['Feature'],
        'Type': 'Categorical',
        'Score': row['Cramers_V'],
        'N_Bins': row['N_Categories']
    })

for _, row in num_df.iterrows():
    all_features.append({
        'Feature': row['Feature'],
        'Type': 'Numerical',
        'Score': row['Best_Cramers_V'],
        'N_Bins': row['Best_N_Bins']
    })

all_df = pd.DataFrame(all_features).sort_values('Score', ascending=False)

print("\nALL FEATURES RANKED BY PREDICTIVE POWER:")
print(all_df.to_string(index=False))

# Recommend feature subsets
print("\n" + "="*70)
print("RECOMMENDATIONS:")
print("="*70)

print("\n✓ TOP 5 FEATURES (Best discriminative power):")
top5 = all_df.head(5)
for _, row in top5.iterrows():
    bins_info = f"{row['N_Bins']} categories" if row['Type'] == 'Categorical' else f"{int(row['N_Bins'])} bins"
    print(f"  - {row['Feature']} ({row['Type']}, {bins_info}, score={row['Score']:.4f})")

print("\n✓ TOP 7 FEATURES (Balanced subset):")
top7 = all_df.head(7)
for _, row in top7.iterrows():
    bins_info = f"{row['N_Bins']} categories" if row['Type'] == 'Categorical' else f"{int(row['N_Bins'])} bins"
    print(f"  - {row['Feature']} ({row['Type']}, {bins_info}, score={row['Score']:.4f})")

print("\n✓ OPTIMAL BINNING CONFIGURATION:")
print("\nCategorical features (use as-is):")
for _, row in cat_df.head(7).iterrows():
    print(f"  - {row['Feature']}: {int(row['N_Categories'])} categories")

print("\nNumerical features (discretize with these bins):")
for _, row in num_df.head(7).iterrows():
    print(f"  - {row['Feature']}: {int(row['Best_N_Bins'])} bins (quantile strategy)")

# ============================================================================
# PART 4: SAVE CONFIGURATION
# ============================================================================
print("\n" + "="*70)
print("SAVING OPTIMIZATION CONFIG...")
print("="*70)

config = {
    'top_features': list(all_df.head(7)['Feature']),
    'binning_config': {}
}

for _, row in num_df.iterrows():
    config['binning_config'][row['Feature']] = int(row['Best_N_Bins'])

import json
with open('bn_optimization_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("✓ Configuration saved to bn_optimization_config.json")
print("\nDone! Use this config to update fraud_detection_pipeline.py")
