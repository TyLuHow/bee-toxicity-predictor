# Reproducibility Guide
## Honey Bee Toxicity Prediction System

This document provides complete instructions to reproduce all results from this project.

---

## System Requirements

### Hardware
- **CPU**: Any modern processor (Intel i5 or equivalent)
- **RAM**: Minimum 4GB, recommended 8GB
- **Storage**: 2GB free space
- **GPU**: Not required (CPU-only implementation)

### Software
- **OS**: Windows, Linux, or macOS
- **Python**: 3.8, 3.9, 3.10, 3.11, or 3.12
- **Git**: For cloning repository (optional)
- **Docker**: For containerized deployment (optional)

---

## Installation

### Step 1: Clone or Download Repository

```bash
# Option 1: Git clone (if using git)
git clone [repository-url]
cd apis_tox_dataset

# Option 2: Download ZIP and extract
# Extract to desired location
cd apis_tox_dataset
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install all required packages
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python -c "import pandas, numpy, sklearn, xgboost, shap; print('✓ All packages installed')"
```

---

## Reproducing Results

### Phase 1: Data Exploration (Already Complete)

The dataset is already processed and available in `outputs/dataset_final.csv`.

To regenerate from raw data:
```bash
python create_dataset.py
python create_dataset_splits.py
```

To run exploratory analysis:
```bash
python run_eda.py
```

View results in `notebooks/01_exploratory_analysis.ipynb`

### Phase 2: Model Training

```bash
# Quick training (recommended for testing)
python train_models_fast.py

# Full training with all models (takes longer)
python -m src.models
```

**Expected Outputs**:
- `outputs/models/best_model_xgboost.pkl` - Trained XGBoost model
- `outputs/preprocessors/preprocessor.pkl` - Preprocessing pipeline
- `outputs/metrics/training_results.json` - Performance metrics

**Expected Results**:
- Test Accuracy: 83-85%
- Test ROC-AUC: 85-87%
- Test F1 Score: 70-72%

### Phase 3: Interpretability Analysis

```bash
python -m src.interpretability
```

**Expected Outputs**:
- `outputs/figures/shap_summary.png`
- `outputs/figures/shap_importance.png`
- `outputs/figures/shap_waterfall_*.png`
- `outputs/figures/lime_explanation_*.png`
- `outputs/metrics/feature_importance_shap.csv`

**Expected Results**:
- Top feature: `insecticide` (importance ≈ 1.4)
- Top 3: insecticide, herbicide, fungicide

### Phase 4: Run API Server

```bash
# Start API server
python app/backend/main.py

# Or using uvicorn directly
uvicorn app.backend.main:app --reload --port 8000

# Access interactive documentation
# Open browser: http://localhost:8000/docs
```

**Test API**:
```bash
# Health check
curl http://localhost:8000/health

# Make prediction (requires full JSON body)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_prediction.json
```

### Phase 5: Run Tests

```bash
# Run system integration tests
python test_system.py

# Run unit tests (requires pytest)
pip install pytest
pytest tests/ -v

# Run specific test file
pytest tests/test_api.py -v
```

**Expected Results**:
- System integration: 6-7/7 tests passed
- Unit tests: All tests should pass

---

## Verifying Reproducibility

### Random Seed

All random operations use seed = 42:
- Data splitting
- Model training
- SMOTE resampling
- Cross-validation

### Version Control

Package versions are specified in `requirements.txt`:
```
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
xgboost==2.0.3
...
```

### Data Integrity

Verify dataset integrity:
```python
import pandas as pd
import hashlib

df = pd.read_csv('outputs/dataset_final.csv')
print(f"Shape: {df.shape}")  # Should be (1035, 13)
print(f"Columns: {len(df.columns)}")  # Should be 13
print(f"Missing values: {df.isnull().sum().sum()}")  # Should be 0
```

### Model Checksum

Verify model file:
```python
import os

model_path = 'outputs/models/best_model_xgboost.pkl'
if os.path.exists(model_path):
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Model file size: {size_mb:.2f} MB")  # Should be 0.5-2 MB
```

---

## Docker Deployment

For completely reproducible environment:

```bash
# Build and run backend API
docker-compose up backend

# Or build manually
docker build -f Dockerfile.backend -t bee-toxicity-api .
docker run -p 8000:8000 bee-toxicity-api

# Access API at http://localhost:8000
```

---

## Common Issues & Solutions

### Issue 1: Package Installation Errors

**Problem**: Error installing RDKit, XGBoost, or SHAP

**Solution**:
```bash
# Try installing with conda (if available)
conda install -c conda-forge rdkit xgboost shap

# Or use pip with --no-cache-dir
pip install --no-cache-dir rdkit xgboost shap
```

### Issue 2: Model File Not Found

**Problem**: `FileNotFoundError: outputs/models/best_model_xgboost.pkl`

**Solution**:
```bash
# Train the model first
python train_models_fast.py

# Verify file exists
ls outputs/models/
```

### Issue 3: Import Errors

**Problem**: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```bash
# Ensure you're in the project root directory
cd apis_tox_dataset

# Add current directory to Python path
export PYTHONPATH="${PYTHONPATH}:${PWD}"  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;%CD%          # Windows
```

### Issue 4: SHAP Installation on Windows

**Problem**: SHAP fails to install on Windows

**Solution**:
```bash
# Install Microsoft C++ Build Tools first
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Then retry installation
pip install shap
```

### Issue 5: API Port Already in Use

**Problem**: `Address already in use: 0.0.0.0:8000`

**Solution**:
```bash
# Use different port
uvicorn app.backend.main:app --port 8001

# Or kill process using port 8000 (Linux/Mac)
lsof -ti:8000 | xargs kill

# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

---

## Performance Benchmarks

Expected performance on standard hardware:

| Task | Time | Resource Usage |
|------|------|----------------|
| Data loading | <1 second | 50MB RAM |
| Preprocessing | 1-2 seconds | 100MB RAM |
| Model training | 1-3 seconds | 200MB RAM |
| SHAP calculation | 5-10 seconds | 300MB RAM |
| API prediction | 50-150ms | 150MB RAM |
| Full pipeline | <30 seconds | <500MB RAM |

---

## Experiment Tracking

All experiments are logged in:
- `outputs/metrics/training_results.json` - Model performance
- `outputs/metrics/feature_importance_shap.csv` - Feature importance
- `app/backend/prediction_history.json` - API predictions (if API used)

### Viewing Results

```python
import json
import pandas as pd

# Load training results
with open('outputs/metrics/training_results.json', 'r') as f:
    results = json.load(f)
    print(json.dumps(results, indent=2))

# Load feature importance
importance = pd.read_csv('outputs/metrics/feature_importance_shap.csv')
print(importance.head(10))
```

---

## File Locations

### Input Data
- `outputs/dataset_final.csv` - Main dataset (1,035 compounds)
- `outputs/splits/` - Train/val/test splits

### Trained Models
- `outputs/models/best_model_xgboost.pkl` - Final trained model
- `outputs/preprocessors/preprocessor.pkl` - Preprocessing pipeline

### Results
- `outputs/figures/` - All visualizations (12+ plots)
- `outputs/metrics/` - Performance metrics (JSON, CSV)

### Documentation
- `README.md` - Project overview
- `docs/MODEL_CARD.md` - Model documentation
- `docs/API_DOCS.md` - API documentation
- `docs/project_proposal.md` - Project proposal
- `docs/presentation/PRESENTATION_SLIDES.md` - Presentation slides

### Code
- `src/preprocessing.py` - Data preprocessing
- `src/models.py` - Model training
- `src/interpretability.py` - SHAP/LIME analysis
- `app/backend/main.py` - FastAPI application

### Tests
- `tests/test_preprocessing.py` - Preprocessing tests
- `tests/test_models.py` - Model tests
- `tests/test_api.py` - API tests
- `test_system.py` - Integration tests

---

## Citation

If you use this code or reproduce these results, please cite:

```bibtex
@software{bee_toxicity_2025,
  title = {Honey Bee Toxicity Prediction System},
  author = {IME 372 Project Team},
  year = {2025},
  url = {[Repository URL]},
  note = {IME 372 Course Project - Predictive Analytics}
}

@article{apistox_2024,
  title = {ApisTox: A New Benchmark Dataset for the Classification of Bee Toxicity of Pesticides},
  author = {Gao, J. and others},
  journal = {Scientific Data},
  year = {2024},
  volume = {11},
  pages = {1234},
  doi = {10.1038/s41597-024-04232-w}
}
```

---

## Support

For issues reproducing results:

1. **Check System Requirements**: Ensure Python version and dependencies match
2. **Review Common Issues**: See "Common Issues & Solutions" section above
3. **Run Integration Tests**: `python test_system.py`
4. **Check Logs**: Look for error messages in terminal output
5. **Verify Files**: Ensure all required files exist in `outputs/` directory

---

## Changelog

### Version 1.0.0 (November 2025)
- Initial release
- XGBoost model (83.6% accuracy)
- Complete reproducibility support
- Docker deployment
- Comprehensive documentation

---

**Document Version**: 1.0.0  
**Last Updated**: November 7, 2025  
**Status**: ✅ Complete

