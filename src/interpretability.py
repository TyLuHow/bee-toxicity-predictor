#!/usr/bin/env python3
"""
Model Interpretability Module
===============================

This module provides comprehensive model interpretability using:
- SHAP (SHapley Additive exPlanations) for global and local interpretability
- LIME (Local Interpretable Model-agnostic Explanations) for individual predictions
- Feature importance analysis
- Partial dependence plots

Author: IME 372 Project Team
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime import lime_tabular
import joblib
import warnings
warnings.filterwarnings('ignore')

class ModelInterpreter:
    """
    Comprehensive model interpretability framework.
    """
    
    def __init__(self, model, X_train, feature_names=None):
        """
        Initialize the interpreter.
        
        Args:
            model: Trained model
            X_train: Training data for background samples
            feature_names: List of feature names
        """
        self.model = model
        self.X_train = X_train if isinstance(X_train, np.ndarray) else X_train.values
        self.feature_names = feature_names if feature_names is not None else \
                            (X_train.columns.tolist() if hasattr(X_train, 'columns') else 
                             [f'Feature_{i}' for i in range(X_train.shape[1])])
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
        
    def setup_shap(self, explainer_type='tree', n_samples=100):
        """
        Setup SHAP explainer.
        
        Args:
            explainer_type: 'tree' for tree-based models, 'kernel' for any model
            n_samples: Number of background samples for KernelExplainer
        """
        print(f"Setting up SHAP explainer (type: {explainer_type})...")
        
        try:
            if explainer_type == 'tree':
                # TreeExplainer for tree-based models (faster)
                self.shap_explainer = shap.TreeExplainer(self.model)
                print("✓ SHAP TreeExplainer initialized")
            else:
                # KernelExplainer for any model (slower but model-agnostic)
                background = shap.sample(self.X_train, n_samples)
                self.shap_explainer = shap.KernelExplainer(
                    self.model.predict_proba, 
                    background
                )
                print("✓ SHAP KernelExplainer initialized")
        except Exception as e:
            print(f"Warning: Could not initialize {explainer_type} explainer: {e}")
            print("Falling back to KernelExplainer...")
            background = shap.sample(self.X_train, min(n_samples, len(self.X_train)))
            self.shap_explainer = shap.KernelExplainer(
                self.model.predict_proba, 
                background
            )
            print("✓ SHAP KernelExplainer initialized (fallback)")
    
    def setup_lime(self, mode='classification'):
        """
        Setup LIME explainer.
        
        Args:
            mode: 'classification' or 'regression'
        """
        print("Setting up LIME explainer...")
        
        self.lime_explainer = lime_tabular.LimeTabularExplainer(
            self.X_train,
            feature_names=self.feature_names,
            class_names=['Non-toxic', 'Toxic'],
            mode=mode,
            random_state=42
        )
        print("✓ LIME explainer initialized")
    
    def calculate_shap_values(self, X, max_samples=None):
        """
        Calculate SHAP values for given data.
        
        Args:
            X: Data to explain
            max_samples: Maximum number of samples to compute (for speed)
            
        Returns:
            SHAP values array
        """
        if self.shap_explainer is None:
            raise ValueError("SHAP explainer not initialized! Call setup_shap() first.")
        
        X_array = X if isinstance(X, np.ndarray) else X.values
        
        if max_samples is not None and len(X_array) > max_samples:
            print(f"Computing SHAP values for {max_samples} samples (out of {len(X_array)})...")
            X_array = X_array[:max_samples]
        else:
            print(f"Computing SHAP values for {len(X_array)} samples...")
        
        shap_values = self.shap_explainer.shap_values(X_array)
        
        # For tree explainer, shap_values might be a list [class_0_values, class_1_values]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class (toxic)
        
        print("✓ SHAP values computed")
        return shap_values
    
    def plot_shap_summary(self, X, shap_values, output_path='outputs/figures/shap_summary.png'):
        """
        Create SHAP summary plot (beeswarm plot).
        
        Args:
            X: Input data
            shap_values: SHAP values
            output_path: Path to save figure
        """
        print("Creating SHAP summary plot...")
        
        X_array = X if isinstance(X, np.ndarray) else X.values
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, 
            X_array[:len(shap_values)],
            feature_names=self.feature_names,
            show=False
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ SHAP summary plot saved: {output_path}")
    
    def plot_shap_bar(self, X, shap_values, output_path='outputs/figures/shap_importance.png'):
        """
        Create SHAP bar plot showing feature importance.
        
        Args:
            X: Input data
            shap_values: SHAP values
            output_path: Path to save figure
        """
        print("Creating SHAP bar plot...")
        
        X_array = X if isinstance(X, np.ndarray) else X.values
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X_array[:len(shap_values)],
            feature_names=self.feature_names,
            plot_type='bar',
            show=False
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ SHAP bar plot saved: {output_path}")
    
    def plot_shap_waterfall(self, X, shap_values, instance_idx=0, 
                           output_path='outputs/figures/shap_waterfall.png'):
        """
        Create SHAP waterfall plot for a single prediction.
        
        Args:
            X: Input data
            shap_values: SHAP values
            instance_idx: Index of instance to explain
            output_path: Path to save figure
        """
        print(f"Creating SHAP waterfall plot for instance {instance_idx}...")
        
        X_array = X if isinstance(X, np.ndarray) else X.values
        
        # Create explanation object
        base_value = self.shap_explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[1]
        
        explanation = shap.Explanation(
            values=shap_values[instance_idx],
            base_values=base_value,
            data=X_array[instance_idx],
            feature_names=self.feature_names
        )
        
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, show=False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ SHAP waterfall plot saved: {output_path}")
    
    def explain_with_lime(self, instance, instance_idx=0, num_features=10,
                         output_path='outputs/figures/lime_explanation.png'):
        """
        Explain a single prediction using LIME.
        
        Args:
            instance: Single instance to explain
            instance_idx: Index for labeling
            num_features: Number of top features to show
            output_path: Path to save figure
            
        Returns:
            LIME explanation object
        """
        if self.lime_explainer is None:
            raise ValueError("LIME explainer not initialized! Call setup_lime() first.")
        
        print(f"Creating LIME explanation for instance {instance_idx}...")
        
        instance_array = instance if isinstance(instance, np.ndarray) else instance.values
        if len(instance_array.shape) > 1:
            instance_array = instance_array.flatten()
        
        # Generate explanation
        explanation = self.lime_explainer.explain_instance(
            instance_array,
            self.model.predict_proba,
            num_features=num_features
        )
        
        # Save figure
        fig = explanation.as_pyplot_figure()
        fig.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ LIME explanation saved: {output_path}")
        
        return explanation
    
    def get_feature_importance(self, shap_values):
        """
        Calculate global feature importance from SHAP values.
        
        Args:
            shap_values: SHAP values array
            
        Returns:
            DataFrame with feature importance
        """
        # Calculate mean absolute SHAP value for each feature
        importance = np.abs(shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df


def run_interpretability_analysis(
    model_path='outputs/models/best_model_xgboost.pkl',
    data_path='data/raw/dataset_with_descriptors.csv',
    n_samples_shap=100,
    n_instances_explain=5
):
    """
    Run complete interpretability analysis.
    
    Args:
        model_path: Path to trained model
        data_path: Path to dataset
        n_samples_shap: Number of samples for SHAP analysis
        n_instances_explain: Number of individual instances to explain
    """
    print("="*80)
    print("PHASE 4: MODEL INTERPRETABILITY ANALYSIS")
    print("="*80)
    
    # Load model
    print("\n1. Loading trained model...")
    model = joblib.load(model_path)
    print(f"✓ Model loaded from {model_path}")
    
    # Load preprocessor and data
    print("\n2. Loading preprocessor and data...")
    from preprocessing import create_preprocessing_pipeline
    
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = create_preprocessing_pipeline(
        data_path=data_path,
        apply_scaling=True,
        apply_feature_selection=False,
        handle_imbalance=False,  # No resampling for interpretability
        save_path=None  # Don't overwrite
    )
    
    feature_names = X_train.columns.tolist()
    print(f"✓ Data loaded: {X_test.shape[0]} test samples")
    
    # Initialize interpreter
    print("\n3. Initializing interpreter...")
    interpreter = ModelInterpreter(model, X_train, feature_names)
    
    # Setup SHAP
    print("\n4. Setting up SHAP explainer...")
    interpreter.setup_shap(explainer_type='tree', n_samples=100)
    
    # Calculate SHAP values
    print("\n5. Calculating SHAP values...")
    shap_values = interpreter.calculate_shap_values(X_test, max_samples=n_samples_shap)
    
    # Create SHAP visualizations
    print("\n6. Creating SHAP visualizations...")
    interpreter.plot_shap_summary(X_test, shap_values)
    interpreter.plot_shap_bar(X_test, shap_values)
    
    # Waterfall plots for specific instances
    for idx in range(min(n_instances_explain, len(shap_values))):
        interpreter.plot_shap_waterfall(
            X_test, shap_values, instance_idx=idx,
            output_path=f'outputs/figures/shap_waterfall_instance_{idx}.png'
        )
    
    # Get feature importance
    importance_df = interpreter.get_feature_importance(shap_values)
    print("\n7. Top 10 Most Important Features (by SHAP):")
    print(importance_df.head(10).to_string(index=False))
    
    # Save feature importance
    importance_df.to_csv('outputs/metrics/feature_importance_shap.csv', index=False)
    print("\n✓ Feature importance saved: outputs/metrics/feature_importance_shap.csv")
    
    # Setup LIME
    print("\n8. Setting up LIME explainer...")
    interpreter.setup_lime(mode='classification')
    
    # LIME explanations for specific instances
    print("\n9. Creating LIME explanations...")
    for idx in range(min(n_instances_explain, len(X_test))):
        interpreter.explain_with_lime(
            X_test.iloc[idx],
            instance_idx=idx,
            num_features=10,
            output_path=f'outputs/figures/lime_explanation_instance_{idx}.png'
        )
    
    print("\n" + "="*80)
    print("✓ PHASE 4 COMPLETE - INTERPRETABILITY ANALYSIS")
    print("="*80)
    print("\nGenerated visualizations:")
    print("  - outputs/figures/shap_summary.png")
    print("  - outputs/figures/shap_importance.png")
    print(f"  - {n_instances_explain} SHAP waterfall plots")
    print(f"  - {n_instances_explain} LIME explanation plots")
    print("  - outputs/metrics/feature_importance_shap.csv")


if __name__ == "__main__":
    import os
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    run_interpretability_analysis(
        n_samples_shap=100,
        n_instances_explain=3
    )

