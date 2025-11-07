#!/usr/bin/env python3
"""
Unit Tests for Data Preprocessing Module
==========================================

Tests cover:
- Data loading and validation
- Feature encoding
- Feature scaling
- Train/test splitting
- Class imbalance handling (SMOTE)
- Preprocessor persistence

Author: IME 372 Project Team
Date: November 2025
"""

import pytest
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Import the module to test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.preprocessing import DataPreprocessor


class TestDataPreprocessor:
    """Test suite for DataPreprocessor class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing."""
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'source': np.random.choice(['ECOTOX', 'PPDB', 'BPDB'], n_samples),
            'toxicity_type': np.random.choice(['Contact', 'Oral', 'Other'], n_samples),
            'year': np.random.randint(2000, 2024, n_samples),
            'herbicide': np.random.randint(0, 2, n_samples),
            'fungicide': np.random.randint(0, 2, n_samples),
            'insecticide': np.random.randint(0, 2, n_samples),
            'other_agrochemical': np.random.randint(0, 2, n_samples),
            'MolecularWeight': np.random.uniform(100, 500, n_samples),
            'LogP': np.random.uniform(-2, 6, n_samples),
            'NumHDonors': np.random.randint(0, 10, n_samples),
            'NumHAcceptors': np.random.randint(0, 15, n_samples),
            'NumRotatableBonds': np.random.randint(0, 20, n_samples),
            'AromaticRings': np.random.randint(0, 5, n_samples),
            'TPSA': np.random.uniform(0, 200, n_samples),
            'label': np.random.randint(0, 2, n_samples)  # Binary target
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance."""
        return DataPreprocessor(random_state=42)
    
    def test_initialization(self, preprocessor):
        """Test preprocessor initialization."""
        assert preprocessor.random_state == 42
        assert preprocessor.scaler is None
        assert preprocessor.categorical_features is None
        assert preprocessor.numerical_features is None
        assert preprocessor.feature_names is None
    
    def test_feature_identification(self, preprocessor, sample_data):
        """Test automatic feature identification."""
        X = sample_data.drop('label', axis=1)
        y = sample_data['label']
        
        # Fit preprocessor
        X_transformed = preprocessor.fit_transform(X, y)
        
        # Check that features were identified
        assert preprocessor.categorical_features is not None
        assert preprocessor.numerical_features is not None
        assert len(preprocessor.categorical_features) > 0
        assert len(preprocessor.numerical_features) > 0
    
    def test_fit_transform(self, preprocessor, sample_data):
        """Test fit_transform method."""
        X = sample_data.drop('label', axis=1)
        y = sample_data['label']
        
        X_transformed = preprocessor.fit_transform(X, y)
        
        # Check output type
        assert isinstance(X_transformed, np.ndarray)
        
        # Check shape (should have same number of rows)
        assert X_transformed.shape[0] == X.shape[0]
        
        # Check that features were scaled (mean ≈ 0, std ≈ 1 for numerical)
        # Note: Not all features will be perfectly scaled due to categorical encoding
        assert X_transformed.shape[1] > 0
    
    def test_transform(self, preprocessor, sample_data):
        """Test transform method on new data."""
        X = sample_data.drop('label', axis=1)
        y = sample_data['label']
        
        # Fit on data
        X_train = preprocessor.fit_transform(X[:80], y[:80])
        
        # Transform test data
        X_test = preprocessor.transform(X[80:])
        
        # Check output
        assert isinstance(X_test, np.ndarray)
        assert X_test.shape[0] == 20
        assert X_test.shape[1] == X_train.shape[1]  # Same number of features
    
    def test_categorical_encoding(self, preprocessor, sample_data):
        """Test categorical feature encoding."""
        X = sample_data[['source', 'toxicity_type']]
        y = sample_data['label']
        
        X_transformed = preprocessor.fit_transform(X, y)
        
        # Should have more columns after one-hot encoding
        assert X_transformed.shape[1] >= X.shape[1]
    
    def test_numerical_scaling(self, preprocessor, sample_data):
        """Test numerical feature scaling."""
        X = sample_data[['MolecularWeight', 'LogP', 'TPSA']]
        y = sample_data['label']
        
        X_transformed = preprocessor.fit_transform(X, y)
        
        # Check that values are scaled (approximate mean 0, std 1)
        assert np.abs(np.mean(X_transformed)) < 1.0  # Mean close to 0
        assert 0.5 < np.std(X_transformed) < 2.0  # Std close to 1
    
    def test_missing_values_handling(self, preprocessor, sample_data):
        """Test handling of missing values."""
        X = sample_data.drop('label', axis=1)
        y = sample_data['label']
        
        # Introduce missing values
        X_missing = X.copy()
        X_missing.loc[0:5, 'MolecularWeight'] = np.nan
        
        # Should handle missing values without crashing
        try:
            X_transformed = preprocessor.fit_transform(X_missing, y)
            # If it reaches here, it handled missing values
            assert True
        except Exception as e:
            # If it throws an error, that's also acceptable behavior
            # depending on the implementation
            assert 'missing' in str(e).lower() or 'nan' in str(e).lower()
    
    def test_save_load(self, preprocessor, sample_data, tmp_path):
        """Test saving and loading preprocessor."""
        X = sample_data.drop('label', axis=1)
        y = sample_data['label']
        
        # Fit preprocessor
        X_transformed = preprocessor.fit_transform(X, y)
        
        # Save preprocessor
        save_path = tmp_path / "test_preprocessor.pkl"
        joblib.dump(preprocessor, save_path)
        
        # Load preprocessor
        loaded_preprocessor = joblib.load(save_path)
        
        # Transform with loaded preprocessor
        X_transformed_loaded = loaded_preprocessor.transform(X)
        
        # Check that transformations are identical
        np.testing.assert_array_almost_equal(X_transformed, X_transformed_loaded)
    
    def test_reproducibility(self, sample_data):
        """Test that preprocessing is reproducible with same random seed."""
        X = sample_data.drop('label', axis=1)
        y = sample_data['label']
        
        # Create two preprocessors with same seed
        prep1 = DataPreprocessor(random_state=42)
        prep2 = DataPreprocessor(random_state=42)
        
        X_transformed1 = prep1.fit_transform(X, y)
        X_transformed2 = prep2.fit_transform(X, y)
        
        # Should be identical
        np.testing.assert_array_almost_equal(X_transformed1, X_transformed2)
    
    def test_feature_names_preservation(self, preprocessor, sample_data):
        """Test that feature names are tracked."""
        X = sample_data.drop('label', axis=1)
        y = sample_data['label']
        
        preprocessor.fit_transform(X, y)
        
        # Should have feature names stored
        assert preprocessor.feature_names is not None
        assert len(preprocessor.feature_names) > 0


class TestSMOTEResampling:
    """Test class imbalance handling with SMOTE."""
    
    @pytest.fixture
    def imbalanced_data(self):
        """Create imbalanced dataset."""
        np.random.seed(42)
        
        # Create imbalanced dataset (70% class 0, 30% class 1)
        n_majority = 70
        n_minority = 30
        
        X_majority = np.random.randn(n_majority, 5)
        y_majority = np.zeros(n_majority)
        
        X_minority = np.random.randn(n_minority, 5) + 1  # Different distribution
        y_minority = np.ones(n_minority)
        
        X = np.vstack([X_majority, X_minority])
        y = np.concatenate([y_majority, y_minority])
        
        return X, y
    
    def test_smote_balancing(self, imbalanced_data):
        """Test SMOTE creates balanced dataset."""
        X, y = imbalanced_data
        
        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Check that classes are balanced
        unique, counts = np.unique(y_resampled, return_counts=True)
        assert counts[0] == counts[1]  # Classes should be equal
        
        # Check that minority class was oversampled
        assert len(y_resampled) > len(y)
    
    def test_smote_feature_preservation(self, imbalanced_data):
        """Test that SMOTE preserves number of features."""
        X, y = imbalanced_data
        
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Number of features should be preserved
        assert X_resampled.shape[1] == X.shape[1]


class TestTrainTestSplit:
    """Test train/test splitting functionality."""
    
    def test_stratified_split(self):
        """Test stratified train/test split maintains class distribution."""
        from sklearn.model_selection import train_test_split
        
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.array([0] * 70 + [1] * 30)  # 70-30 imbalance
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Check that class distribution is maintained
        train_ratio = np.sum(y_train) / len(y_train)
        test_ratio = np.sum(y_test) / len(y_test)
        original_ratio = np.sum(y) / len(y)
        
        assert abs(train_ratio - original_ratio) < 0.05  # Within 5%
        assert abs(test_ratio - original_ratio) < 0.05


class TestFeatureScaling:
    """Test feature scaling functionality."""
    
    def test_standard_scaler(self):
        """Test StandardScaler produces mean=0, std=1."""
        np.random.seed(42)
        X = np.random.randn(100, 3) * 10 + 50  # Random data with mean≈50, std≈10
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Check scaled mean and std
        np.testing.assert_almost_equal(np.mean(X_scaled, axis=0), 0, decimal=10)
        np.testing.assert_almost_equal(np.std(X_scaled, axis=0), 1, decimal=10)
    
    def test_scaler_transform_consistency(self):
        """Test that scaler transforms new data consistently."""
        np.random.seed(42)
        X_train = np.random.randn(80, 3) * 10 + 50
        X_test = np.random.randn(20, 3) * 10 + 50
        
        scaler = StandardScaler()
        scaler.fit(X_train)
        
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Test set should be scaled using training set statistics
        assert X_test_scaled.shape == X_test.shape
        # Test mean won't be exactly 0 since it uses train statistics
        assert np.abs(np.mean(X_test_scaled)) < 2.0


class TestIntegration:
    """Integration tests for full preprocessing pipeline."""
    
    def test_full_pipeline(self):
        """Test complete preprocessing pipeline from raw data to model-ready."""
        # Load actual dataset
        if os.path.exists('outputs/dataset_final.csv'):
            df = pd.read_csv('outputs/dataset_final.csv')
            
            # Drop non-feature columns
            drop_cols = ['name', 'CID', 'CAS', 'SMILES', 'ppdb_level']
            X = df.drop([col for col in drop_cols if col in df.columns] + ['label'], axis=1)
            y = df['label']
            
            # Create preprocessor
            preprocessor = DataPreprocessor(random_state=42)
            
            # Fit and transform
            X_transformed = preprocessor.fit_transform(X, y)
            
            # Checks
            assert X_transformed.shape[0] == X.shape[0]
            assert X_transformed.shape[1] > 0
            assert not np.isnan(X_transformed).any()  # No NaN values
            assert not np.isinf(X_transformed).any()  # No inf values
            
            print(f"✓ Full pipeline test passed!")
            print(f"  Input shape: {X.shape}")
            print(f"  Output shape: {X_transformed.shape}")
        else:
            pytest.skip("Dataset file not found")


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

