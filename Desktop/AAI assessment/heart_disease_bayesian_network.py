# -*- coding: utf-8 -*-
"""
Heart Disease Prediction with Bayesian Networks
Testing BN performance on REAL medical dataset (UCI Heart Disease)

This will show if the poor BN performance was due to:
1. Synthetic/random fraud data (hypothesis)
2. Bayesian Network limitations (alternative)

Dataset: 1025 patients, 13 clinical features, binary heart disease target
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, brier_score_loss)
from scipy.stats import chi2_contingency
import time

# pgmpy for Bayesian Networks
try:
    from pgmpy.estimators import HillClimbSearch, PC, TreeSearch, BayesianEstimator, BIC, AIC
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.inference import VariableElimination
    PGMPY_AVAILABLE = True
except Exception as e:
    PGMPY_AVAILABLE = False
    print(f"❌ pgmpy not available. Error: {e}")
    exit(1)


def load_and_prepare_heart_data(csv_path):
    """
    Load heart disease dataset and discretize features.
    
    Feature descriptions (UCI Heart Disease):
    - age: Age in years
    - sex: Sex (1 = male, 0 = female)
    - cp: Chest pain type (0-3)
    - trestbps: Resting blood pressure
    - chol: Serum cholesterol
    - fbs: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
    - restecg: Resting ECG results (0-2)
    - thalach: Maximum heart rate achieved
    - exang: Exercise induced angina (1 = yes, 0 = no)
    - oldpeak: ST depression induced by exercise
    - slope: Slope of peak exercise ST segment (0-2)
    - ca: Number of major vessels colored by fluoroscopy (0-3)
    - thal: Thalassemia (0-3)
    - target: Heart disease (1 = yes, 0 = no)
    """
    print(f"\nLoading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Handle missing values by dropping rows (preserves data patterns)
    initial_rows = len(df)
    df = df.dropna()
    dropped = initial_rows - len(df)
    if dropped > 0:
        print(f"⚠️  Dropped {dropped} rows with missing values ({dropped/initial_rows*100:.2f}%)")
    
    print(f"Dataset: {len(df)} patients, {len(df.columns)} features")
    print(f"Target distribution:")
    print(f"  No heart disease (0): {(df['target']==0).sum()} ({(df['target']==0).mean():.1%})")
    print(f"  Heart disease (1): {(df['target']==1).sum()} ({(df['target']==1).mean():.1%})")
    
    # Analyze feature associations with target using Cramér's V
    print("\nAnalyzing feature associations with heart disease...")
    print("="*70)
    
    cramers_v_scores = []
    
    for col in df.columns:
        if col == 'target':
            continue
            
        # Calculate Cramér's V
        contingency = pd.crosstab(df[col], df['target'])
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        n = contingency.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
        
        cramers_v_scores.append({
            'Feature': col,
            'Cramers_V': cramers_v,
            'Chi2': chi2,
            'P_Value': p_value,
            'Unique_Values': df[col].nunique()
        })
    
    cv_df = pd.DataFrame(cramers_v_scores).sort_values('Cramers_V', ascending=False)
    
    print("\nTop 10 features by predictive power (Cramér's V):")
    print(cv_df.head(10).to_string(index=False))
    
    # Discretize continuous features into bins
    print("\nDiscretizing continuous features...")
    df_bn = df.copy()
    
    # Age: quartiles
    df_bn['age'] = pd.qcut(df['age'], q=4, labels=['Young', 'Middle_Age', 'Senior', 'Elderly'], duplicates='drop')
    
    # Blood pressure: tertiles
    df_bn['trestbps'] = pd.qcut(df['trestbps'], q=3, labels=['Low_BP', 'Normal_BP', 'High_BP'], duplicates='drop')
    
    # Cholesterol: tertiles
    df_bn['chol'] = pd.qcut(df['chol'], q=3, labels=['Low_Chol', 'Normal_Chol', 'High_Chol'], duplicates='drop')
    
    # Max heart rate: tertiles
    df_bn['thalach'] = pd.qcut(df['thalach'], q=3, labels=['Low_HR', 'Normal_HR', 'High_HR'], duplicates='drop')
    
    # ST depression: tertiles (skip if all same value)
    if df['oldpeak'].nunique() > 3:
        df_bn['oldpeak'] = pd.qcut(df['oldpeak'], q=3, labels=['No_Depression', 'Mild', 'Severe'], duplicates='drop')
    else:
        df_bn['oldpeak'] = df['oldpeak'].astype(str)
    
    # Convert all to strings (pgmpy requirement)
    for col in df_bn.columns:
        df_bn[col] = df_bn[col].astype(str)
    
    print(f"✓ Prepared {len(df_bn.columns)-1} features for Bayesian Network")
    
    return df_bn, cv_df


def test_structure_learning_algorithms(df, target='target'):
    """Test multiple structure learning algorithms."""
    print("\n" + "="*70)
    print("TESTING STRUCTURE LEARNING ALGORITHMS")
    print("="*70)
    
    algorithms = {}
    
    # 1. Hill Climbing + AIC
    print("\n1. Hill Climbing + AIC...")
    try:
        start = time.time()
        hc = HillClimbSearch(df)
        hc_model = hc.estimate(scoring_method=AIC(df))
        hc_time = time.time() - start
        hc_fraud_edges = sum(1 for edge in hc_model.edges() if target in edge)
        algorithms['HillClimb_AIC'] = {
            'model': hc_model,
            'edges': len(hc_model.edges()),
            'target_edges': hc_fraud_edges,
            'time': hc_time
        }
        print(f"   Edges: {len(hc_model.edges())}, Target-connected: {hc_fraud_edges}, Time: {hc_time:.2f}s")
    except Exception as e:
        print(f"   Failed: {e}")
    
    # 2. Hill Climbing + BIC
    print("\n2. Hill Climbing + BIC...")
    try:
        start = time.time()
        hc_bic = HillClimbSearch(df)
        hc_bic_model = hc_bic.estimate(scoring_method=BIC(df))
        hc_bic_time = time.time() - start
        hc_bic_fraud_edges = sum(1 for edge in hc_bic_model.edges() if target in edge)
        algorithms['HillClimb_BIC'] = {
            'model': hc_bic_model,
            'edges': len(hc_bic_model.edges()),
            'target_edges': hc_bic_fraud_edges,
            'time': hc_bic_time
        }
        print(f"   Edges: {len(hc_bic_model.edges())}, Target-connected: {hc_bic_fraud_edges}, Time: {hc_bic_time:.2f}s")
    except Exception as e:
        print(f"   Failed: {e}")
    
    # 3. TreeSearch TAN
    print("\n3. TreeSearch TAN...")
    try:
        start = time.time()
        tree = TreeSearch(df)
        tree_model = tree.estimate(estimator_type='tan', class_node=target)
        tree_time = time.time() - start
        tree_fraud_edges = sum(1 for edge in tree_model.edges() if target in edge)
        algorithms['TreeSearch_TAN'] = {
            'model': tree_model,
            'edges': len(tree_model.edges()),
            'target_edges': tree_fraud_edges,
            'time': tree_time
        }
        print(f"   Edges: {len(tree_model.edges())}, Target-connected: {tree_fraud_edges}, Time: {tree_time:.2f}s")
    except Exception as e:
        print(f"   Failed: {e}")
    
    # Select best
    print("\n" + "="*70)
    if not algorithms:
        print("❌ All algorithms failed!")
        return None, None
    
    best_algo = max(algorithms.items(), key=lambda x: (x[1]['target_edges'], x[1]['edges']))
    algo_name, algo_data = best_algo
    
    print(f"✓ Selected: {algo_name}")
    print(f"  Total edges: {algo_data['edges']}")
    print(f"  Target-connected edges: {algo_data['target_edges']}")
    print(f"  Learning time: {algo_data['time']:.2f}s")
    
    if algo_data['edges'] > 0:
        target_edges = [e for e in algo_data['model'].edges() if target in e]
        other_edges = [e for e in algo_data['model'].edges() if target not in e]
        if target_edges:
            print(f"\n  Predictive edges (→ {target}):")
            for edge in target_edges[:10]:  # Show first 10
                print(f"    {edge}")
        if other_edges:
            print(f"\n  Feature relationships: {len(other_edges)} edges")
    
    return algo_name, algo_data['model']


def evaluate_bayesian_network(model, df, target='target'):
    """Evaluate Bayesian Network with cross-validation."""
    print("\n" + "="*70)
    print("EVALUATING BAYESIAN NETWORK")
    print("="*70)
    
    # 5-fold CV
    print("\nPerforming 5-fold stratified cross-validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    all_probs = []
    all_labels = []
    fold_times = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df, df[target]), 1):
        fold_start = time.time()
        print(f"  Fold {fold_idx}/5...", end=' ')
        
        df_train = df.iloc[train_idx]
        df_test = df.iloc[test_idx]
        
        # Learn parameters
        try:
            bn = DiscreteBayesianNetwork(model.edges())
            bn.fit(df_train, estimator=BayesianEstimator, prior_type='BDeu')
            
            # Inference
            inference = VariableElimination(bn)
            
            # Predict
            test_features = df_test.drop(columns=[target])
            probs = []
            
            for idx in range(len(test_features)):
                evidence = test_features.iloc[idx].to_dict()
                try:
                    result = inference.query(variables=[target], evidence=evidence)
                    prob_1 = result.values[1]
                    probs.append(prob_1)
                except:
                    probs.append(0.5)
            
            all_probs.extend(probs)
            all_labels.extend(df_test[target].astype(int).values)
            
            fold_time = time.time() - fold_start
            fold_times.append(fold_time)
            print(f"completed in {fold_time:.2f}s")
            
        except Exception as e:
            print(f"failed: {e}")
            continue
    
    # Metrics
    if len(all_probs) == 0:
        print("❌ No predictions made!")
        return
    
    y_proba = np.array(all_probs)
    y_true = np.array(all_labels)
    
    threshold = 0.5  # Standard threshold for balanced dataset
    y_pred = (y_proba >= threshold).astype(int)
    
    print("\n" + "="*70)
    print("BAYESIAN NETWORK RESULTS (HEART DISEASE)")
    print("="*70)
    
    print(f"\nProbability distribution:")
    print(f"  Range: [{y_proba.min():.4f}, {y_proba.max():.4f}]")
    print(f"  Mean: {y_proba.mean():.4f}")
    print(f"  Std: {y_proba.std():.4f}")
    
    # Check for discrimination
    prob_range = y_proba.max() - y_proba.min()
    if prob_range > 0.3:
        print(f"  ✓ Good discrimination! (range = {prob_range:.4f})")
    elif prob_range > 0.1:
        print(f"  ⚠ Moderate discrimination (range = {prob_range:.4f})")
    else:
        print(f"  ❌ Poor discrimination (range = {prob_range:.4f})")
    
    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_proba)
    except:
        auc = 0.5
    
    try:
        brier = brier_score_loss(y_true, y_proba)
    except:
        brier = 0.25
    
    print(f"\nUsing threshold: {threshold:.4f}")
    print("\nBayesian Network Performance Metrics:")
    print("-" * 50)
    print(f"  Accuracy:        {acc:.4f}")
    print(f"  Precision:       {prec:.4f}")
    print(f"  Recall:          {rec:.4f}")
    print(f"  F1-Score:        {f1:.4f}")
    print(f"  ROC AUC:         {auc:.4f}")
    print(f"  Brier Score:     {brier:.4f}")
    
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"                Predicted")
    print(f"              0        1")
    print(f"  Actual 0  {cm[0,0]:6d}   {cm[0,1]:6d}")
    print(f"         1  {cm[1,0]:6d}   {cm[1,1]:6d}")
    
    print(f"\n  Total Training Time: {sum(fold_times):.2f}s")
    
    # Comparison
    print("\n" + "="*70)
    print("COMPARISON: HEART DISEASE vs FRAUD DETECTION")
    print("="*70)
    print("\nFraud Dataset (synthetic, uniform distributions):")
    print("  Cramér's V: < 0.014 for all features")
    print("  Accuracy: 50.23%, Precision: 32.19%, Recall: 49.62%")
    print("  ROC AUC: 0.5007 (barely above random)")
    print("  Probability range: [0.189, 0.449]")
    
    print(f"\nHeart Disease Dataset (real medical data):")
    print(f"  Cramér's V: (see analysis above)")
    print(f"  Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
    print(f"  ROC AUC: {auc:.4f}")
    print(f"  Probability range: [{y_proba.min():.4f}, {y_proba.max():.4f}]")
    
    if auc > 0.70:
        print("\n✓✓✓ EXCELLENT! Bayesian Network performs well on real data!")
        print("    This confirms the fraud dataset was the problem, not BN itself.")
    elif auc > 0.60:
        print("\n✓✓ GOOD! Bayesian Network shows improvement on real data!")
        print("   Real patterns lead to better BN performance.")
    elif auc > 0.55:
        print("\n✓ MODERATE improvement over fraud dataset.")
        print("  Real data helps but still limited.")
    else:
        print("\n→ Similar to fraud dataset - BN struggles with both.")


def main():
    """Main execution."""
    print("="*70)
    print("HEART DISEASE PREDICTION WITH BAYESIAN NETWORKS")
    print("Testing BN on REAL medical data vs synthetic fraud data")
    print("="*70)
    
    if not PGMPY_AVAILABLE:
        return
    
    # Load data
    csv_path = Path("heart.csv")
    if not csv_path.exists():
        print(f"❌ Dataset not found: {csv_path}")
        return
    
    df_bn, cramers_v_df = load_and_prepare_heart_data(csv_path)
    
    # Test algorithms
    best_algo, best_model = test_structure_learning_algorithms(df_bn)
    
    if best_model is None:
        print("❌ No successful algorithm found!")
        return
    
    # Evaluate
    evaluate_bayesian_network(best_model, df_bn)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nKEY TAKEAWAY:")
    print("If BN performs well here, it proves the issue was the synthetic")
    print("fraud dataset (uniform random distributions), not Bayesian Networks!")


if __name__ == "__main__":
    main()
