#!/usr/bin/env python3
"""
Fast model training script with reduced hyperparameter search.
For production use, run src/models.py with full hyperparameter tuning.
"""

import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

from src.preprocessing import create_preprocessing_pipeline
from src.models import ModelTrainer
import joblib
import json

print("="*80)
print("FAST MODEL TRAINING - PHASE 3")
print("="*80)

# Load data
X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = create_preprocessing_pipeline(
    data_path='data/raw/dataset_with_descriptors.csv',
    apply_scaling=True,
    apply_feature_selection=False,
    handle_imbalance=True,
    imbalance_method='smote'
)

# Initialize trainer
trainer = ModelTrainer(random_state=42)

# Train models with minimal hyperparameter tuning for speed
models_to_train = ['logistic', 'random_forest', 'xgboost']

all_results = {}
for model_name in models_to_train:
    print(f"\nTraining {model_name}...")
    result = trainer.train_model(
        model_name,
        X_train, y_train,
        X_val, y_val,
        tune_hyperparams=False,  # Skip tuning for speed
        cv_folds=3
    )
    all_results[model_name] = result

# Compare models
trainer._print_model_comparison(all_results)

# Select best and evaluate on test set
trainer._select_best_model(all_results, metric='f1')
test_metrics = trainer.evaluate_on_test(X_test, y_test)

# Save best model
best_model_name = [name for name, result in all_results.items() 
                   if result['model'] == trainer.best_model][0]
os.makedirs('outputs/models', exist_ok=True)
trainer.save_model(best_model_name, f'outputs/models/best_model_{best_model_name}.pkl')

# Save all results
os.makedirs('outputs/metrics', exist_ok=True)
trainer.save_results('outputs/metrics/training_results.json')

print("\n" + "="*80)
print("PHASE 3 COMPLETE!")
print("="*80)
print(f"Best Model: {best_model_name}")
print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
print(f"Test F1 Score: {test_metrics['f1']:.4f}")
print(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")

