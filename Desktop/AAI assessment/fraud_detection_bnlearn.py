# -*- coding: utf-8 -*-
"""
Fraud Detection using bnlearn (Alternative BN implementation)
Testing if bnlearn's algorithms can achieve better Bayesian Network performance.

bnlearn offers additional structure learning algorithms:
- Constraint-based: PC, GS, IAMB, Fast-IAMB, Inter-IAMB
- Score-based: HC (Hill Climbing), Tabu, Chow-Liu
- Hybrid: MMHC, H2PC, ARACNE
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix)
import time

# Try importing bnlearn
try:
    import bnlearn as bn
    BNLEARN_AVAILABLE = True
    print("✓ bnlearn imported successfully")
except ImportError:
    BNLEARN_AVAILABLE = False
    print("✗ bnlearn not available")
    print("  Install via: pip install bnlearn")
    exit(1)


def load_and_prepare_data(csv_path):
    """Load dataset and prepare for Bayesian Network."""
    print(f"\nLoading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"Fraud rate: {df['Fraud_Label'].mean():.2%}")
    
    # Select optimized features based on Cramér's V analysis
    # (from bn_optimization_analysis.py results)
    optimized_features = [
        'Transaction_Amount',
        'Authentication_Method',
        'Location',
        'Merchant_Category',
        'Device_Type',
        'Card_Type',
        'Transaction_Type'
    ]
    
    # Discretize Transaction_Amount into 10 bins
    print("\nDiscretizing Transaction_Amount into 10 quantile bins...")
    df['Transaction_Amount_Binned'] = pd.qcut(
        df['Transaction_Amount'], 
        q=10, 
        labels=[f'Bin_{i}' for i in range(10)],
        duplicates='drop'
    )
    
    # Prepare features
    categorical_features = [
        'Transaction_Amount_Binned',
        'Authentication_Method',
        'Location',
        'Merchant_Category',
        'Device_Type',
        'Card_Type',
        'Transaction_Type'
    ]
    
    # Create BN dataframe with only selected features
    df_bn = df[categorical_features + ['Fraud_Label']].copy()
    
    # Convert to strings (bnlearn requirement)
    for col in df_bn.columns:
        df_bn[col] = df_bn[col].astype(str)
    
    print(f"\n✓ Prepared {len(categorical_features)} features for Bayesian Network")
    return df_bn, categorical_features


def test_bnlearn_algorithms(df, target='Fraud_Label'):
    """
    Test multiple bnlearn structure learning algorithms.
    
    bnlearn offers more algorithms than pgmpy:
    - Constraint-based: PC, GS (Grow-Shrink), IAMB variants
    - Score-based: hc (Hill Climbing), tabu, chow-liu (tree structure)
    - Hybrid: mmhc, h2pc, aracne
    """
    print("\n" + "="*70)
    print("TESTING BNLEARN STRUCTURE LEARNING ALGORITHMS")
    print("="*70)
    
    algorithms = {}
    
    # 1. Hill Climbing (Score-based)
    print("\n1. Hill Climbing (hc - score-based)...")
    try:
        start = time.time()
        model_hc = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic')
        hc_time = time.time() - start
        hc_edges = len(model_hc['model'].edges()) if model_hc['model'] is not None else 0
        hc_fraud_edges = sum(1 for edge in model_hc['model'].edges() if target in edge) if model_hc['model'] is not None else 0
        algorithms['HC_BIC'] = {
            'model': model_hc,
            'edges': hc_edges,
            'fraud_edges': hc_fraud_edges,
            'time': hc_time
        }
        print(f"   Edges: {hc_edges}, Fraud-connected: {hc_fraud_edges}, Time: {hc_time:.2f}s")
    except Exception as e:
        print(f"   Failed: {e}")
    
    # 2. Tabu Search (Score-based)
    print("\n2. Tabu Search (score-based with tabu list)...")
    try:
        start = time.time()
        model_tabu = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic', tabu_length=10)
        tabu_time = time.time() - start
        tabu_edges = len(model_tabu['model'].edges()) if model_tabu['model'] is not None else 0
        tabu_fraud_edges = sum(1 for edge in model_tabu['model'].edges() if target in edge) if model_tabu['model'] is not None else 0
        algorithms['Tabu_BIC'] = {
            'model': model_tabu,
            'edges': tabu_edges,
            'fraud_edges': tabu_fraud_edges,
            'time': tabu_time
        }
        print(f"   Edges: {tabu_edges}, Fraud-connected: {tabu_fraud_edges}, Time: {tabu_time:.2f}s")
    except Exception as e:
        print(f"   Failed: {e}")
    
    # 3. Chow-Liu (Tree structure - optimal for classification)
    print("\n3. Chow-Liu Tree (optimal tree structure)...")
    try:
        start = time.time()
        # Chow-Liu builds tree structure, similar to TAN
        model_cl = bn.structure_learning.fit(df, methodtype='cl', root_node=target)
        cl_time = time.time() - start
        cl_edges = len(model_cl['model'].edges()) if model_cl['model'] is not None else 0
        cl_fraud_edges = sum(1 for edge in model_cl['model'].edges() if target in edge) if model_cl['model'] is not None else 0
        algorithms['ChowLiu'] = {
            'model': model_cl,
            'edges': cl_edges,
            'fraud_edges': cl_fraud_edges,
            'time': cl_time
        }
        print(f"   Edges: {cl_edges}, Fraud-connected: {cl_fraud_edges}, Time: {cl_time:.2f}s")
    except Exception as e:
        print(f"   Failed: {e}")
    
    # 4. Naive Bayes (All features → target)
    print("\n4. Naive Bayes (all features → target)...")
    try:
        start = time.time()
        model_nb = bn.structure_learning.fit(df, methodtype='naivebayes', root_node=target)
        nb_time = time.time() - start
        nb_edges = len(model_nb['model'].edges()) if model_nb['model'] is not None else 0
        nb_fraud_edges = sum(1 for edge in model_nb['model'].edges() if target in edge) if model_nb['model'] is not None else 0
        algorithms['NaiveBayes'] = {
            'model': model_nb,
            'edges': nb_edges,
            'fraud_edges': nb_fraud_edges,
            'time': nb_time
        }
        print(f"   Edges: {nb_edges}, Fraud-connected: {nb_fraud_edges}, Time: {nb_time:.2f}s")
    except Exception as e:
        print(f"   Failed: {e}")
    
    # 5. Tree-Augmented Naive Bayes (TAN)
    print("\n5. TAN (Tree-Augmented Naive Bayes)...")
    try:
        start = time.time()
        model_tan = bn.structure_learning.fit(df, methodtype='tan', root_node=target, class_node=target)
        tan_time = time.time() - start
        tan_edges = len(model_tan['model'].edges()) if model_tan['model'] is not None else 0
        tan_fraud_edges = sum(1 for edge in model_tan['model'].edges() if target in edge) if model_tan['model'] is not None else 0
        algorithms['TAN'] = {
            'model': model_tan,
            'edges': tan_edges,
            'fraud_edges': tan_fraud_edges,
            'time': tan_time
        }
        print(f"   Edges: {tan_edges}, Fraud-connected: {tan_fraud_edges}, Time: {tan_time:.2f}s")
    except Exception as e:
        print(f"   Failed: {e}")
    
    # Select best algorithm
    print("\n" + "="*70)
    if not algorithms:
        print("❌ All algorithms failed!")
        return None, None
    
    best_algo = max(algorithms.items(), key=lambda x: (x[1]['fraud_edges'], x[1]['edges']))
    algo_name, algo_data = best_algo
    
    print(f"✓ Selected: {algo_name}")
    print(f"  Total edges: {algo_data['edges']}")
    print(f"  Fraud-connected edges: {algo_data['fraud_edges']}")
    print(f"  Learning time: {algo_data['time']:.2f}s")
    
    return algo_name, algo_data['model']


def evaluate_bnlearn_model(model, df, target='Fraud_Label'):
    """
    Evaluate bnlearn Bayesian Network using cross-validation.
    """
    print("\n" + "="*70)
    print("EVALUATING BNLEARN BAYESIAN NETWORK")
    print("="*70)
    
    # Learn parameters from full dataset
    print("\nLearning parameters...")
    try:
        model_fitted = bn.parameter_learning.fit(model, df)
    except Exception as e:
        print(f"Parameter learning failed: {e}")
        return
    
    # 5-fold cross-validation
    print("\nPerforming 5-fold stratified cross-validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    all_probs = []
    all_labels = []
    fold_times = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df, df[target]), 1):
        fold_start = time.time()
        print(f"  Fold {fold_idx}/5...", end=' ')
        
        # Split data
        df_train = df.iloc[train_idx].copy()
        df_test = df.iloc[test_idx].copy()
        
        # Learn parameters on training set
        try:
            model_fold = bn.parameter_learning.fit(model, df_train)
            
            # Make predictions on test set
            # bnlearn's predict returns most likely class
            # We need probabilities for proper evaluation
            test_features = df_test.drop(columns=[target])
            
            # For each test instance, query P(Fraud_Label | evidence)
            probs = []
            for idx in range(len(test_features)):
                try:
                    # Create evidence dictionary for this instance
                    evidence = test_features.iloc[idx].to_dict()
                    
                    # Query probability - bnlearn returns distribution
                    result = bn.inference.fit(model_fold, variables=[target], evidence=evidence)
                    
                    # Extract P(Fraud_Label=1)
                    if result is not None and 'p' in result:
                        prob_df = result['p']
                        # Find probability for Fraud_Label='1'
                        prob_1 = prob_df[prob_df[target] == '1']['p'].values
                        prob = prob_1[0] if len(prob_1) > 0 else 0.5
                    else:
                        prob = 0.5  # Default if inference fails
                    
                    probs.append(prob)
                except:
                    probs.append(0.5)  # Default to uncertain
            
            all_probs.extend(probs)
            all_labels.extend(df_test[target].astype(int).values)
            
            fold_time = time.time() - fold_start
            fold_times.append(fold_time)
            print(f"completed in {fold_time:.2f}s")
            
        except Exception as e:
            print(f"failed: {e}")
            continue
    
    # Calculate metrics
    if len(all_probs) == 0:
        print("❌ No predictions made!")
        return
    
    y_proba = np.array(all_probs)
    y_true = np.array(all_labels)
    
    # Use base fraud rate as threshold
    threshold = y_true.mean()
    y_pred = (y_proba >= threshold).astype(int)
    
    print("\n" + "="*70)
    print("BNLEARN BAYESIAN NETWORK RESULTS")
    print("="*70)
    
    print(f"\nProbability distribution:")
    print(f"  Range: [{y_proba.min():.4f}, {y_proba.max():.4f}]")
    print(f"  Mean: {y_proba.mean():.4f}")
    print(f"  Std: {y_proba.std():.4f}")
    
    print(f"\nUsing threshold: {threshold:.4f} (base fraud rate)")
    
    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_proba)
    except:
        auc = 0.5
    
    print("\nBNLEARN Bayesian Network (5-fold CV) Performance Metrics:")
    print("-" * 50)
    print(f"  Accuracy:        {acc:.4f}")
    print(f"  Precision:       {prec:.4f}")
    print(f"  Recall:          {rec:.4f}")
    print(f"  F1-Score:        {f1:.4f}")
    print(f"  ROC AUC:         {auc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"                Predicted")
    print(f"              0        1")
    print(f"  Actual 0  {cm[0,0]:6d}   {cm[0,1]:6d}")
    print(f"         1  {cm[1,0]:6d}   {cm[1,1]:6d}")
    
    print(f"\n  Total Training Time: {sum(fold_times):.2f}s")
    print(f"  Average Fold Time: {np.mean(fold_times):.2f}s")
    
    # Comparison
    print("\n" + "="*70)
    print("COMPARISON WITH PGMPY RESULTS")
    print("="*70)
    print("\nPGMPY (TreeSearch TAN):")
    print("  Accuracy: 0.5023, Precision: 0.3219, Recall: 0.4962, F1: 0.3905")
    print("  ROC AUC: 0.5007")
    print(f"\nBNLEARN (Current Algorithm):")
    print(f"  Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print(f"  ROC AUC: {auc:.4f}")
    
    if auc > 0.5007:
        print("\n✓ BNLEARN shows improvement over pgmpy!")
    else:
        print("\n→ Similar performance to pgmpy (dataset limitation)")


def main():
    """Main execution."""
    print("="*70)
    print("FRAUD DETECTION WITH BNLEARN - Alternative BN Implementation")
    print("="*70)
    
    if not BNLEARN_AVAILABLE:
        return
    
    # Load data
    csv_path = Path("synthetic_fraud_dataset.csv")
    if not csv_path.exists():
        print(f"❌ Dataset not found: {csv_path}")
        return
    
    df_bn, features = load_and_prepare_data(csv_path)
    
    # Test algorithms
    best_algo, best_model = test_bnlearn_algorithms(df_bn)
    
    if best_model is None:
        print("❌ No successful algorithm found!")
        return
    
    # Evaluate
    evaluate_bnlearn_model(best_model, df_bn)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
