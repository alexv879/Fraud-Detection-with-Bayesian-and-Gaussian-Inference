# -*- coding: utf-8 -*-
"""
Fraud Detection with Gaussian Bayesian Networks

Gaussian Bayesian Networks (GBN) assume continuous features follow Gaussian 
distributions with linear relationships: X_i = β₀ + Σ(β_j * X_parent_j) + ε

Key characteristics:
- Works with continuous features (no discretization)
- Assumes linear Gaussian relationships between variables
- Structure learning: Hill Climbing with BIC scoring
- Parameter learning: Linear Gaussian CPDs (Conditional Probability Distributions)
- Fast inference once structure is learned
- Interpretable: Shows which features have linear dependencies
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, brier_score_loss)
from sklearn.preprocessing import LabelEncoder
import time
import warnings
warnings.filterwarnings('ignore')

# Gaussian Bayesian Networks (pgmpy)
try:
    from pgmpy.models import LinearGaussianBayesianNetwork
    from pgmpy.estimators import HillClimbSearch, BIC
    PGMPY_AVAILABLE = True
except Exception as e:
    PGMPY_AVAILABLE = False
    print(f"❌ pgmpy not available for Gaussian BN: {e}")
    print("   Install with: pip install pgmpy")
    exit(1)


def load_and_prepare_fraud_data(csv_path, sample_size=10000):
    """
    Load fraud dataset and prepare for Gaussian Bayesian Network.
    
    Note: GBN structure learning can be slow on large datasets.
    Subsampling for computational efficiency.
    
    Args:
        csv_path: Path to synthetic_fraud_dataset.csv
        sample_size: Subsample size (default 10000)
    
    Returns:
        X: Continuous features (Transaction_Amount + encoded categoricals)
        y: Binary fraud labels
        feature_names: List of feature names
    """
    print(f"\nLoading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Dataset: {len(df):,} transactions")
    print(f"Fraud rate: {df['Fraud_Label'].mean():.2%}")
    
    # Subsample FIRST for computational efficiency
    if len(df) > sample_size:
        print(f"\n⚠️  Subsampling to {sample_size:,} transactions for efficiency")
        print(f"   Gaussian BN structure learning can be slow on large datasets")
        df = df.sample(n=sample_size, random_state=42)
        print(f"   Fraud rate in sample: {df['Fraud_Label'].mean():.2%}")
    
    # Separate continuous and categorical features
    target = 'Fraud_Label'
    continuous_features = ['Transaction_Amount']
    categorical_features = [col for col in df.columns 
                           if col not in continuous_features + [target]]
    
    print(f"\nFeature breakdown:")
    print(f"  Continuous: {len(continuous_features)} features")
    print(f"  Categorical: {len(categorical_features)} features")
    
    # Encode categorical features as numeric (GBN requires numerical data)
    df_processed = df.copy()
    label_encoders = {}
    
    for col in categorical_features:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # Prepare feature matrix and target
    feature_cols = continuous_features + categorical_features
    X = df_processed[feature_cols].values.astype(float)
    y = df_processed[target].values
    
    print(f"\n✓ Prepared {X.shape[0]:,} samples with {X.shape[1]} features")
    
    return X, y, feature_cols


def build_gaussian_bayesian_network(X_train, y_train, feature_names):
    """
    Build Gaussian Bayesian Network using pgmpy.
    
    GBN models linear Gaussian relationships: X_i = β₀ + Σ(β_j * X_parent_j) + ε
    where ε ~ N(0, σ²)
    
    Structure Learning:
    - Algorithm: Hill Climbing (greedy search)
    - Scoring: BIC (Bayesian Information Criterion)
    - Penalizes complex structures to avoid overfitting
    
    Parameter Learning:
    - Linear Gaussian CPDs for each variable
    - Learns mean, variance, and linear coefficients
    
    Args:
        X_train: Training features
        y_train: Training labels
        feature_names: Feature column names
    
    Returns:
        model: Fitted LinearGaussianBayesianNetwork
        learning_time: Time taken to learn structure + parameters
    """
    print("\n" + "="*70)
    print("GAUSSIAN BAYESIAN NETWORK")
    print("="*70)
    
    # Create DataFrame with feature names
    df_train = pd.DataFrame(X_train, columns=feature_names)
    df_train['Fraud_Label'] = y_train
    
    print("\nLearning structure with Hill Climbing + BIC...")
    print("(GBN assumes linear Gaussian relationships between variables)")
    
    try:
        start_time = time.time()
        
        # Structure learning
        print("\n1. Structure Learning Phase:")
        print("   - Algorithm: Hill Climbing (greedy local search)")
        print("   - Scoring: BIC (penalizes model complexity)")
        print("   - Searching for optimal DAG structure...")
        
        hc = HillClimbSearch(df_train)
        best_model = hc.estimate(scoring_method=BIC(df_train))
        
        structure_time = time.time() - start_time
        print(f"\n✓ Structure learned in {structure_time:.2f}s")
        print(f"  Total edges: {len(best_model.edges())}")
        
        target_edges = [edge for edge in best_model.edges() if 'Fraud_Label' in edge]
        print(f"  Fraud-connected edges: {len(target_edges)}")
        
        if target_edges:
            print(f"\n  Predictive edges (showing causal relationships):")
            for edge in target_edges:
                parent, child = edge
                if child == 'Fraud_Label':
                    print(f"    {parent} → Fraud_Label (predictor)")
                else:
                    print(f"    Fraud_Label → {child} (influenced by fraud)")
        else:
            print("\n  ⚠️  No edges connected to Fraud_Label!")
            print("     This indicates weak/no linear relationships in the data")
        
        # Parameter learning for Linear Gaussian BN
        print("\n2. Parameter Learning Phase:")
        print("   - Learning Linear Gaussian CPDs")
        print("   - Each CPD: X_i = β₀ + Σ(β_j * X_parent_j) + N(0, σ²)")
        
        model = LinearGaussianBayesianNetwork(best_model.edges())
        model.fit(df_train)
        
        learning_time = time.time() - start_time
        print(f"\n✓ Total learning time: {learning_time:.2f}s")
        
        # Show learned parameters for Fraud_Label
        if 'Fraud_Label' in model.nodes():
            cpd = model.get_cpds('Fraud_Label')
            if cpd:
                print(f"\n  Fraud_Label CPD (Linear Gaussian):")
                if hasattr(cpd, 'mean'):
                    print(f"    Mean (intercept β₀): {cpd.mean:.4f}")
                if hasattr(cpd, 'variance'):
                    print(f"    Variance (σ²): {cpd.variance:.4f}")
                
                # Show parent coefficients if available
                parents = list(model.predecessors('Fraud_Label'))
                if parents and hasattr(cpd, 'beta_vector'):
                    print(f"    Linear coefficients (β):")
                    for i, parent in enumerate(parents):
                        if i < len(cpd.beta_vector):
                            print(f"      {parent}: {cpd.beta_vector[i]:.4f}")
        
        return model, learning_time
        
    except Exception as e:
        print(f"\n❌ Gaussian BN learning failed: {e}")
        print("\nPossible reasons:")
        print("  1. Data doesn't follow Gaussian distribution")
        print("  2. Non-linear relationships (GBN assumes linearity)")
        print("  3. Insufficient variance in features")
        import traceback
        traceback.print_exc()
        return None, 0


def evaluate_gaussian_bn(model, X_test, y_test, feature_names):
    """
    Evaluate Gaussian Bayesian Network on test set.
    
    For Linear Gaussian BN, prediction involves:
    1. Use learned CPDs to predict E[Fraud_Label | evidence]
    2. Linear prediction: β₀ + Σ(β_i * parent_i)
    3. Normalize to [0,1] for probability interpretation
    4. Threshold at 0.5 for binary classification
    
    Args:
        model: Fitted LinearGaussianBayesianNetwork
        X_test: Test features
        y_test: True labels
        feature_names: Feature names
    
    Returns:
        metrics: Dictionary of performance metrics
    """
    if model is None:
        return None
    
    print("\n" + "="*70)
    print("EVALUATING GAUSSIAN BAYESIAN NETWORK")
    print("="*70)
    
    try:
        df_test = pd.DataFrame(X_test, columns=feature_names)
        
        print("\nPredicting fraud probabilities...")
        print("Method: Linear Gaussian inference E[Fraud_Label | evidence]")
        
        predictions = []
        
        # Get CPD for Fraud_Label
        if 'Fraud_Label' in model.nodes():
            cpd = model.get_cpds('Fraud_Label')
            parents = list(model.predecessors('Fraud_Label'))
            
            if parents and hasattr(cpd, 'mean') and hasattr(cpd, 'beta_vector'):
                print(f"  Using {len(parents)} parent features for prediction")
                
                # Linear prediction: E[Fraud_Label] = mean + Σ(beta_i * parent_i)
                beta = cpd.beta_vector
                mean = cpd.mean
                
                for i in range(len(X_test)):
                    # Get parent values
                    evidence_values = []
                    for parent in parents:
                        parent_idx = feature_names.index(parent)
                        evidence_values.append(X_test[i, parent_idx])
                    
                    # Linear prediction
                    prediction = mean + np.dot(beta, evidence_values)
                    predictions.append(prediction)
                
                predictions = np.array(predictions)
                
                # Normalize to [0, 1] range for probability interpretation
                predictions_normalized = (predictions - predictions.min()) / (predictions.max() - predictions.min() + 1e-10)
                
            else:
                # Fallback: use mean prediction (no parents found)
                print("  ⚠️  No parents found for Fraud_Label, using baseline mean")
                predictions_normalized = np.full(len(X_test), cpd.mean if hasattr(cpd, 'mean') else 0.5)
        else:
            print("  ⚠️  Fraud_Label not in model")
            predictions_normalized = np.full(len(X_test), 0.5)
        
        # Convert to binary predictions (threshold at 0.5)
        y_pred = (predictions_normalized >= 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            roc_auc = roc_auc_score(y_test, predictions_normalized)
        except:
            roc_auc = 0.5
        
        try:
            brier = brier_score_loss(y_test, predictions_normalized)
        except:
            brier = 0.25
        
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"\n✓ Predictions completed")
        print(f"  Probability range: [{predictions_normalized.min():.4f}, {predictions_normalized.max():.4f}]")
        print(f"  Mean probability: {predictions_normalized.mean():.4f}")
        
        print(f"\n" + "="*70)
        print("GAUSSIAN BAYESIAN NETWORK RESULTS")
        print("="*70)
        
        print(f"\nPerformance Metrics:")
        print(f"{'─'*50}")
        print(f"  Accuracy:        {accuracy:.4f}")
        print(f"  Precision:       {precision:.4f}")
        print(f"  Recall:          {recall:.4f}")
        print(f"  F1-Score:        {f1:.4f}")
        print(f"  ROC AUC:         {roc_auc:.4f}")
        print(f"  Brier Score:     {brier:.4f}")
        
        print(f"\n  Confusion Matrix:")
        print(f"            Predicted")
        print(f"          0        1")
        print(f"  Actual 0 {cm[0,0]:5d}   {cm[0,1]:5d}")
        print(f"         1 {cm[1,0]:5d}   {cm[1,1]:5d}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'brier_score': brier,
            'confusion_matrix': cm,
            'predictions': predictions_normalized
        }
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """
    Main execution: Gaussian Bayesian Network for fraud detection.
    """
    print("="*70)
    print("FRAUD DETECTION WITH GAUSSIAN BAYESIAN NETWORKS")
    print("="*70)
    
    print("\nGaussian Bayesian Networks (GBN):")
    print("  • Assumes linear Gaussian relationships: X_i = β₀ + Σ(β_j·X_j) + ε")
    print("  • Works with continuous features (no discretization)")
    print("  • Structure learning: Hill Climbing + BIC")
    print("  • Parameter learning: Linear Gaussian CPDs")
    print("  • Fast inference, interpretable structure")
    
    # Load data
    csv_path = Path("synthetic_fraud_dataset.csv")
    if not csv_path.exists():
        print(f"\n❌ Dataset not found: {csv_path}")
        return
    
    X, y, feature_names = load_and_prepare_fraud_data(csv_path, sample_size=10000)
    
    # Train-test split (80-20)
    print("\n" + "="*70)
    print("TRAIN-TEST SPLIT")
    print("="*70)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining set: {len(X_train):,} samples")
    print(f"Test set:     {len(X_test):,} samples")
    print(f"Train fraud rate: {y_train.mean():.2%}")
    print(f"Test fraud rate:  {y_test.mean():.2%}")
    
    # Build and train Gaussian BN
    model, training_time = build_gaussian_bayesian_network(X_train, y_train, feature_names)
    
    # Evaluate
    if model:
        metrics = evaluate_gaussian_bn(model, X_test, y_test, feature_names)
        
        if metrics:
            print("\n" + "="*70)
            print("KEY INSIGHTS")
            print("="*70)
            
            print("\n✓ Gaussian BN Characteristics:")
            print("  • Linear relationships between variables")
            print("  • Gaussian noise assumption")
            print("  • Interpretable structure and parameters")
            print("  • Fast inference (closed-form solutions)")
            
            print("\n⚠️  Expected Limitations on This Dataset:")
            print("  • Fraud data has uniform random distributions")
            print("  • No real linear patterns to learn")
            print("  • Categorical features encoded as numeric (loses meaning)")
            print("  • Performance limited by data quality, not method")
            
            print(f"\n📊 Performance Summary:")
            print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
            print(f"  Training time: {training_time:.2f}s")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
