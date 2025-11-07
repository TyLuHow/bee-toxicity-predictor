# Honey Bee Toxicity Prediction System
## IME 372 Course Project - Predictive Analytics

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-orange)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-CC--BY--NC--4.0-yellow)](https://creativecommons.org/licenses/by-nc/4.0/)

**A comprehensive machine learning system for predicting pesticide toxicity to honey bees using molecular descriptors and agrochemical properties.**

---

## 🎯 Project Overview

This project implements an end-to-end machine learning pipeline to predict whether pesticides are toxic to honey bees, addressing a critical agricultural and environmental challenge. The system achieves **83.6% accuracy** and **85.8% ROC-AUC** on the test set.

### Key Features

- ✅ **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost, LightGBM
- ✅ **Model Interpretability**: SHAP and LIME explanations
- ✅ **REST API**: FastAPI backend for production deployment
- ✅ **Comprehensive EDA**: Statistical analysis and visualizations
- ✅ **Feature Engineering**: Molecular descriptors from SMILES
- ✅ **Class Imbalance Handling**: SMOTE resampling
- ✅ **Academic Documentation**: Proposal, presentation, and technical reports

---

## 📊 Project Results

### Model Performance (Test Set)

| Metric | Score |
|--------|-------|
| **Accuracy** | 83.57% |
| **F1 Score** | 70.18% |
| **ROC-AUC** | 85.83% |
| **Precision (Toxic)** | 72.73% |
| **Recall (Toxic)** | 67.80% |

### Top Predictive Features (SHAP Analysis)

1. **Insecticide** (1.366) - Chemical type flag
2. **Herbicide** (1.054) - Chemical type flag
3. **Fungicide** (0.740) - Chemical type flag
4. **Year** (0.641) - Publication year
5. **LogP** (0.474) - Lipophilicity

---

## 🗂️ Project Structure

```
apis_tox_dataset/
├── app/
│   └── backend/
│       └── main.py                 # FastAPI REST API
├── data/
│   └── raw/
│       └── dataset_with_descriptors.csv
├── notebooks/
│   └── 01_exploratory_analysis.ipynb  # EDA notebook
├── src/
│   ├── preprocessing.py            # Data preprocessing pipeline
│   ├── models.py                   # Model training and evaluation
│   └── interpretability.py         # SHAP/LIME analysis
├── outputs/
│   ├── models/                     # Trained model files
│   ├── preprocessors/              # Saved preprocessors
│   ├── figures/                    # Generated visualizations
│   └── metrics/                    # Performance metrics
├── tests/
│   └── test_*.py                   # Unit tests
├── docs/
│   ├── project_proposal.md         # Project proposal
│   └── presentation/               # Presentation materials
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip or conda
- 4GB RAM minimum
- Internet connection (for package installation)

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd apis_tox_dataset
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

Or using the existing poetry/requirements:
```bash
poetry install --no-root
# OR
pip install -r requirements.txt
```

3. **Install additional ML packages**:
```bash
pip install seaborn shap lime lightgbm imbalanced-learn rdkit
```

### Quick Start

1. **Run Exploratory Data Analysis**:
```bash
python run_eda.py
```

2. **Train Models**:
```bash
python train_models_fast.py
```

3. **Generate Interpretability Analysis**:
```bash
python src/interpretability.py
```

4. **Start API Server**:
```bash
python app/backend/main.py
```

The API will be available at `http://localhost:8000`

5. **View API Documentation**:
Open browser: `http://localhost:8000/docs`

---

## 📈 Dataset Information

**Source**: ApisTox Dataset  
**Size**: 1,035 pesticide compounds  
**Target**: Binary classification (0=non-toxic, 1=toxic)  
**Features**: 24 total (after preprocessing)
- 15 molecular descriptors (from SMILES)
- 7 agrochemical flags
- 2 temporal/source features

**Class Distribution**:
- Non-toxic: 739 (71.4%)
- Toxic: 296 (28.6%)
- Imbalance ratio: 2.50:1

---

## 🔬 Methodology

### 1. Data Preprocessing
- ✅ No missing values
- ✅ Molecular descriptor extraction using RDKit
- ✅ One-hot encoding for categorical features
- ✅ StandardScaler for numerical features
- ✅ Stratified train/val/test split (70/10/20)
- ✅ SMOTE resampling for class imbalance

### 2. Model Development
- **Baseline**: Logistic Regression
- **Ensemble**: Random Forest, XGBoost, LightGBM
- **SVM**: Support Vector Machine
- **Neural Network**: Multi-Layer Perceptron

**Best Model**: XGBoost Classifier with hyperparameter tuning

### 3. Model Interpretability
- **SHAP**: Global and local feature importance
- **LIME**: Individual prediction explanations
- **Feature Analysis**: Correlation and dependency plots

### 4. Production Deployment
- **FastAPI**: RESTful API with automatic documentation
- **Model Serving**: Joblib persistence
- **Prediction History**: SQLite/JSON storage
- **CORS**: Frontend integration ready

---

## 🔌 API Endpoints

### Health Check
```bash
GET /health
```

### Make Prediction
```bash
POST /predict
Content-Type: application/json

{
  "source": "PPDB",
  "year": 2020,
  "toxicity_type": "Contact",
  "insecticide": 1,
  "MolecularWeight": 350.0,
  "LogP": 3.5,
  ...
}
```

### Get Model Information
```bash
GET /model/info
```

### Feature Importance
```bash
GET /feature/importance
```

### Prediction History
```bash
GET /history?limit=10
```

---

## 📊 Visualizations

The project generates comprehensive visualizations:

1. **EDA Visualizations**:
   - Target distribution (bar and pie charts)
   - Molecular descriptor distributions
   - Feature correlations heatmap
   - Toxicity comparison boxplots

2. **Model Interpretability**:
   - SHAP summary plots (beeswarm)
   - SHAP feature importance (bar chart)
   - SHAP waterfall plots (individual predictions)
   - LIME explanations

All visualizations saved in `outputs/figures/`

---

## 🧪 Testing

Run unit tests:
```bash
pytest tests/ -v
```

Test API endpoints:
```bash
python test_api.py
```

---

## 📚 Academic Deliverables

### 1. Project Proposal (2-3 pages)
- **Location**: `docs/project_proposal.md`
- **Contents**: Problem statement, methodology, timeline, team roles

### 2. Presentation Materials (12-15 minutes)
- **Location**: `docs/presentation/`
- **Contents**: Slides covering all project phases with visualizations

### 3. Technical Documentation
- **README.md**: Project overview and usage guide
- **Notebooks**: Interactive Jupyter analysis
- **Code Comments**: Comprehensive docstrings

---

## 🌍 Ethical Considerations

### Environmental Impact
- **Pollinator Conservation**: Models inform safe pesticide use
- **Bee Population Health**: Predictions support regulatory decisions
- **Agricultural Sustainability**: Balance crop protection with bee safety

### Model Limitations
- **Data Bias**: Dataset may not represent all pesticide types
- **Probabilistic**: Predictions are not definitive assessments
- **Transparency**: Full interpretability provided via SHAP/LIME
- **Precautionary Principle**: Uncertainty should favor bee safety

### Responsible Use
- Tool aids conservation, not harmful development
- Stakeholders: farmers, beekeepers, regulators, manufacturers
- Data lineage: public chemical data (no privacy concerns)

---

## 🏆 Key Achievements

✅ **Data Analysis**: Comprehensive EDA with 1,035 pesticide compounds  
✅ **Feature Engineering**: 15 molecular descriptors extracted from SMILES  
✅ **Model Training**: 4 ML algorithms trained and compared  
✅ **Performance**: 83.6% accuracy, 85.8% ROC-AUC  
✅ **Interpretability**: SHAP and LIME explanations implemented  
✅ **API Development**: Production-ready FastAPI backend  
✅ **Documentation**: Academic-grade reports and visualizations  
✅ **Reproducibility**: All results documented with random seeds  

---

## 📖 References

1. **ApisTox Dataset**: [Scientific Data (2024)](https://www.nature.com/articles/s41597-024-04232-w)
2. **SHAP**: Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions"
3. **XGBoost**: Chen & Guestrin (2016). "XGBoost: A Scalable Tree Boosting System"
4. **RDKit**: Open-source cheminformatics toolkit
5. **SMOTE**: Chawla et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique"

---

## 👥 Team Information

**Course**: IME 372 - Predictive Analytics  
**Institution**: [University Name]  
**Semester**: Fall 2025  
**Project Type**: Machine Learning Classification with Interpretability  

---

## 📝 License

This project uses the ApisTox dataset, which is licensed under [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/).

**Non-Commercial Use Only**: This project is for educational and research purposes.

---

## 🆘 Support

For questions or issues:
1. Check the documentation in `docs/`
2. Review API documentation at `/docs` endpoint
3. Examine example notebooks in `notebooks/`
4. Contact project team

---

## 🎓 Acknowledgments

- **ApisTox Team**: For the comprehensive dataset
- **Scientific Community**: For RDKit, SHAP, and ML libraries
- **Course Instructors**: For project guidance and support

---

## 📈 Future Enhancements

- [ ] **Frontend**: React/TypeScript web application
- [ ] **Database**: PostgreSQL for production storage
- [ ] **Monitoring**: MLflow for experiment tracking
- [ ] **Deployment**: Docker containerization
- [ ] **CI/CD**: Automated testing and deployment
- [ ] **Additional Models**: Deep learning (Graph Neural Networks)
- [ ] **Real-time API**: Streaming predictions
- [ ] **Mobile App**: iOS/Android applications

---

**Built with ❤️ for honey bees and sustainable agriculture**
