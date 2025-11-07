#!/usr/bin/env python3
"""
Unit Tests for Model Training Module
=====================================

Tests cover:
- Model initialization
- Model training and prediction
- Hyperparameter tuning
- Model evaluation metrics
- Model persistence
- Cross-validation

Author: IME 372 Project Team
Date: November 2025
"""

import pytest
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Import the module to test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.models import ModelTrainer


class TestModelTrainer:
    """Test suite for ModelTrainer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create synthetic classification dataset."""
        X, y = make_classification(
            n_samples=200,
            n_features=15,
            n_informative=10,
            n_redundant=3,
            n_classes=2,
            weights=[0.7, 0.3],  # Imbalanced
            random_state=42
        )
        
        # Split into train/val/test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    @pytest.fixture
    def trainer(self):
        """Create ModelTrainer instance."""
        return ModelTrainer(random_state=42)
    
    def test_initialization(self, trainer):
        """Test trainer initialization."""
        assert trainer.random_state == 42
        assert trainer.models == {}
        assert trainer.results == {}
        assert trainer.best_model_name is None
        assert trainer.best_model is None
    
    def test_logistic_regression_training(self, trainer, sample_data):
        """Test logistic regression model training."""
        X_train, X_val, X_test, y_train, y_val, y_test = sample_data
        
        # Train model
        trainer.train_logistic_regression(X_train, y_train, X_val, y_val)
        
        # Check model was saved
        assert 'logistic_regression' in trainer.models
        assert 'logistic_regression' in trainer.results
        
        # Check model can predict
        y_pred = trainer.models['logistic_regression'].predict(X_test)
        assert len(y_pred) == len(y_test)
        assert set(y_pred).issubset({0, 1})
    
    def test_random_forest_training(self, trainer, sample_data):
        """Test random forest model training."""
        X_train, X_val, X_test, y_train, y_val, y_test = sample_data
        
        # Train model
        trainer.train_random_forest(X_train, y_train, X_val, y_val)
        
        # Check model was saved
        assert 'random_forest' in trainer.models
        assert 'random_forest' in trainer.results
        
        # Check model can predict
        y_pred = trainer.models['random_forest'].predict(X_test)
        assert len(y_pred) == len(y_test)
    
    def test_xgboost_training(self, trainer, sample_data):
        """Test XGBoost model training."""
        X_train, X_val, X_test, y_train, y_val, y_test = sample_data
        
        # Train model
        trainer.train_xgboost(X_train, y_train, X_val, y_val)
        
        # Check model was saved
        assert 'xgboost' in trainer.models
        assert 'xgboost' in trainer.results
        
        # Check model can predict
        y_pred = trainer.models['xgboost'].predict(X_test)
        assert len(y_pred) == len(y_test)
    
    def test_prediction_probabilities(self, trainer, sample_data):
        """Test that models can predict probabilities."""
        X_train, X_val, X_test, y_train, y_val, y_test = sample_data
        
        # Train a model
        trainer.train_logistic_regression(X_train, y_train, X_val, y_val)
        
        # Get probability predictions
        model = trainer.models['logistic_regression']
        y_pred_proba = model.predict_proba(X_test)
        
        # Check shape and values
        assert y_pred_proba.shape == (len(X_test), 2)
        assert np.all((y_pred_proba >= 0) & (y_pred_proba <= 1))
        assert np.allclose(y_pred_proba.sum(axis=1), 1.0)
    
    def test_model_evaluation_metrics(self, trainer, sample_data):
        """Test that evaluation metrics are computed correctly."""
        X_train, X_val, X_test, y_train, y_val, y_test = sample_data
        
        # Train model
        trainer.train_logistic_regression(X_train, y_train, X_val, y_val)
        
        # Check that metrics are stored
        results = trainer.results['logistic_regression']
        
        assert 'val_accuracy' in results
        assert 'val_precision' in results
        assert 'val_recall' in results
        assert 'val_f1' in results
        assert 'val_roc_auc' in results
        
        # Check metric ranges
        assert 0 <= results['val_accuracy'] <= 1
        assert 0 <= results['val_f1'] <= 1
        assert 0 <= results['val_roc_auc'] <= 1
    
    def test_best_model_selection(self, trainer, sample_data):
        """Test best model selection based on metrics."""
        X_train, X_val, X_test, y_train, y_val, y_test = sample_data
        
        # Train multiple models
        trainer.train_logistic_regression(X_train, y_train, X_val, y_val)
        trainer.train_random_forest(X_train, y_train, X_val, y_val)
        
        # Select best model
        trainer.select_best_model(metric='val_f1')
        
        # Check best model was selected
        assert trainer.best_model_name is not None
        assert trainer.best_model is not None
        assert trainer.best_model_name in trainer.models
    
    def test_model_save_load(self, trainer, sample_data, tmp_path):
        """Test saving and loading trained models."""
        X_train, X_val, X_test, y_train, y_val, y_test = sample_data
        
        # Train model
        trainer.train_logistic_regression(X_train, y_train, X_val, y_val)
        
        # Save model
        model_path = tmp_path / "test_model.pkl"
        joblib.dump(trainer.models['logistic_regression'], model_path)
        
        # Load model
        loaded_model = joblib.load(model_path)
        
        # Make predictions
        y_pred_original = trainer.models['logistic_regression'].predict(X_test)
        y_pred_loaded = loaded_model.predict(X_test)
        
        # Should be identical
        np.testing.assert_array_equal(y_pred_original, y_pred_loaded)
    
    def test_cross_validation(self, trainer, sample_data):
        """Test cross-validation functionality."""
        X_train, _, _, y_train, _, _ = sample_data
        
        from sklearn.model_selection import cross_val_score
        from sklearn.linear_model import LogisticRegression
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
        
        # Check scores
        assert len(scores) == 3
        assert np.all((scores >= 0) & (scores <= 1))
    
    def test_confusion_matrix(self, trainer, sample_data):
        """Test confusion matrix computation."""
        from sklearn.metrics import confusion_matrix
        
        X_train, X_val, X_test, y_train, y_val, y_test = sample_data
        
        # Train model
        trainer.train_logistic_regression(X_train, y_train, X_val, y_val)
        
        # Make predictions
        y_pred = trainer.models['logistic_regression'].predict(X_test)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Check shape
        assert cm.shape == (2, 2)
        assert cm.sum() == len(y_test)
    
    def test_reproducibility(self, sample_data):
        """Test that training is reproducible with same random seed."""
        X_train, X_val, X_test, y_train, y_val, y_test = sample_data
        
        # Train two models with same seed
        trainer1 = ModelTrainer(random_state=42)
        trainer1.train_logistic_regression(X_train, y_train, X_val, y_val)
        
        trainer2 = ModelTrainer(random_state=42)
        trainer2.train_logistic_regression(X_train, y_train, X_val, y_val)
        
        # Predictions should be identical
        y_pred1 = trainer1.models['logistic_regression'].predict(X_test)
        y_pred2 = trainer2.models['logistic_regression'].predict(X_test)
        
        np.testing.assert_array_equal(y_pred1, y_pred2)
    
    def test_feature_importance(self, trainer, sample_data):
        """Test feature importance extraction from tree-based models."""
        X_train, X_val, X_test, y_train, y_val, y_test = sample_data
        
        # Train random forest
        trainer.train_random_forest(X_train, y_train, X_val, y_val)
        
        # Get feature importance
        model = trainer.models['random_forest']
        importance = model.feature_importances_
        
        # Check shape and values
        assert len(importance) == X_train.shape[1]
        assert np.all(importance >= 0)
        assert np.sum(importance) > 0  # At least some features are important


class TestModelComparison:
    """Test model comparison functionality."""
    
    @pytest.fixture
    def trained_models(self):
        """Create and train multiple models."""
        X, y = make_classification(
            n_samples=200, n_features=10, n_classes=2, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier
        
        models = {
            'logistic': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=50),
            'decision_tree': DecisionTreeClassifier(random_state=42)
        }
        
        # Train all models
        for name, model in models.items():
            model.fit(X_train, y_train)
        
        return models, X_train, X_test, y_train, y_test
    
    def test_model_comparison_table(self, trained_models):
        """Test creating model comparison table."""
        models, X_train, X_test, y_train, y_test = trained_models
        
        results = {}
        for name, model in models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
        
        # Convert to dataframe
        df_results = pd.DataFrame(results).T
        
        # Checks
        assert len(df_results) == 3
        assert 'accuracy' in df_results.columns
        assert 'f1' in df_results.columns
        assert 'roc_auc' in df_results.columns


class TestIntegration:
    """Integration tests with actual dataset."""
    
    def test_full_training_pipeline(self):
        """Test complete training pipeline with actual dataset."""
        if os.path.exists('data/raw/dataset_with_descriptors.csv'):
            # Load data
            df = pd.read_csv('data/raw/dataset_with_descriptors.csv')
            
            # Prepare features and target
            feature_cols = [col for col in df.columns if col not in ['label', 'name', 'SMILES', 'CAS', 'CID']]
            X = df[feature_cols].select_dtypes(include=[np.number])
            y = df['label']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.125, random_state=42, stratify=y_train
            )
            
            # Train model
            trainer = ModelTrainer(random_state=42)
            trainer.train_logistic_regression(X_train.values, y_train.values, X_val.values, y_val.values)
            
            # Make predictions
            y_pred = trainer.models['logistic_regression'].predict(X_test.values)
            
            # Check accuracy
            accuracy = accuracy_score(y_test, y_pred)
            assert accuracy > 0.5  # Better than random
            
            print(f"✓ Integration test passed!")
            print(f"  Train size: {len(X_train)}")
            print(f"  Test accuracy: {accuracy:.4f}")
        else:
            pytest.skip("Dataset file not found")


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

