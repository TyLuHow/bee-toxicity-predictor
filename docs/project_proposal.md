# Project Proposal
## Predictive Analytics for Pesticide Toxicity to Honey Bees Using Machine Learning

**Course**: IME 372 - Predictive Analytics  
**Semester**: Fall 2025  
**Date**: November 7, 2025

---

## 1. Executive Summary

This project develops a comprehensive machine learning system to predict pesticide toxicity to honey bees (Apis mellifera), addressing a critical challenge in agricultural sustainability and pollinator conservation. Using the ApisTox dataset containing 1,035 pesticide compounds with molecular descriptors and toxicity labels, we will build, evaluate, and deploy multiple classification models with full interpretability features. The final deliverable includes a production-ready REST API and comprehensive analysis documentation suitable for academic and regulatory applications.

**Expected Outcome**: Binary classification model with >80% accuracy, complete with SHAP/LIME interpretability and web-based demonstration interface.

---

## 2. Team Information

**Team Members**:
- **[Name 1]** - Data preprocessing and feature engineering
- **[Name 2]** - Model development and hyperparameter tuning  
- **[Name 3]** - Model interpretability and visualization
- **[Name 4]** - API development and deployment

**Project Coordinator**: [Name]  
**Meeting Schedule**: Weekly on [Day] at [Time]

---

## 3. Enterprise Problem Statement

### 3.1 Background

Honey bees are critical pollinators responsible for approximately one-third of global food production. However, bee populations have declined significantly due to various factors, including pesticide exposure. Understanding which pesticides pose toxicity risks is essential for:

- **Agricultural Decision-Making**: Farmers need safe pesticide recommendations
- **Regulatory Compliance**: EPA and agricultural agencies require toxicity assessments
- **Environmental Protection**: Conservation efforts depend on identifying harmful chemicals
- **Economic Impact**: Bee colony losses cost agriculture billions annually

### 3.2 Problem Definition

**Current Challenge**: Traditional toxicity testing is expensive ($10,000-$50,000 per compound), time-consuming (6-12 months), and requires live bee subjects. There is a need for rapid, accurate computational prediction of pesticide toxicity.

**Business Impact**:
- Accelerate safe pesticide development
- Reduce animal testing requirements
- Support regulatory decision-making
- Enable proactive environmental protection

**Success Metrics**:
- Prediction accuracy >80%
- Model interpretability for regulatory acceptance
- API response time <1 second
- Production-ready deployment architecture

---

## 4. Data Description

### 4.1 Dataset Overview

**Source**: ApisTox Dataset ([Scientific Data, 2024](https://www.nature.com/articles/s41597-024-04232-w))  
**Size**: 1,035 unique pesticide compounds  
**Features**: 28 total (13 original + 15 molecular descriptors)  
**Target**: Binary classification (0=non-toxic, 1=toxic to honey bees)  
**License**: CC-BY-NC-4.0 (Non-commercial use)

### 4.2 Feature Categories

**Categorical Features**:
- `source`: Data origin (ECOTOX, PPDB, BPDB)
- `toxicity_type`: Exposure route (Contact, Oral, Other)

**Binary Flags**:
- `herbicide`, `fungicide`, `insecticide`, `other_agrochemical`

**Molecular Descriptors** (extracted from SMILES):
- `MolecularWeight`: Compound mass
- `LogP`: Lipophilicity (fat solubility)
- `NumHDonors/NumHAcceptors`: Hydrogen bonding capability
- `TPSA`: Topological polar surface area
- `NumRings`, `NumAromaticRings`: Structural features
- Additional descriptors: Refractivity, complexity, atom counts

**Temporal Feature**:
- `year`: First publication year (1832-2023)

### 4.3 Class Distribution

- **Non-toxic (Class 0)**: 739 compounds (71.4%)
- **Toxic (Class 1)**: 296 compounds (28.6%)
- **Imbalance Ratio**: 2.50:1 (requires handling)

### 4.4 Data Quality

✅ **No missing values**  
✅ **No duplicate compounds**  
✅ **Validated SMILES structures**  
✅ **Curated from peer-reviewed sources**

---

## 5. Methodology

### 5.1 Data Preprocessing

1. **Molecular Descriptor Extraction**: Convert SMILES to numerical features using RDKit
2. **Feature Encoding**: One-hot encode categorical variables (source, toxicity_type)
3. **Feature Scaling**: StandardScaler for numerical features
4. **Data Splitting**: Stratified 70/10/20 (train/val/test)
5. **Class Imbalance**: SMOTE resampling on training data only

### 5.2 Model Selection Strategy

**Baseline Model**:
- Logistic Regression (simple, interpretable)

**Ensemble Methods**:
- Random Forest (handles non-linearity, feature importance)
- XGBoost (state-of-the-art gradient boosting)
- LightGBM (fast training, good performance)

**Additional Models**:
- Support Vector Machine (kernel methods)
- Multi-Layer Perceptron (neural network approach)

**Selection Criteria**:
- Primary metric: F1 Score (handles imbalance)
- Secondary metrics: Accuracy, ROC-AUC, Precision, Recall
- Cross-validation with 5-fold stratified splits
- Hyperparameter tuning via GridSearchCV

### 5.3 Model Interpretability

**Global Interpretability**:
- SHAP summary plots (feature importance across all predictions)
- Feature correlation analysis
- Partial dependence plots

**Local Interpretability**:
- SHAP waterfall plots (individual prediction explanation)
- LIME explanations (model-agnostic approach)
- Confidence scores and probability distributions

**Rationale**: Regulatory and agricultural applications require explainable AI. Users must understand why a pesticide is classified as toxic.

### 5.4 Evaluation Metrics

**Classification Metrics**:
- Accuracy: Overall correctness
- Precision: Minimize false positives (incorrectly labeling safe as toxic)
- Recall: Minimize false negatives (missing toxic compounds - critical!)
- F1 Score: Balance precision and recall
- ROC-AUC: Threshold-independent performance
- Confusion Matrix: Detailed error analysis

**Success Threshold**: F1 Score >0.70, ROC-AUC >0.80

---

## 6. Technical Architecture

### 6.1 Development Stack

**Backend**:
- Python 3.8+
- Scikit-learn, XGBoost, LightGBM (ML)
- SHAP, LIME (interpretability)
- FastAPI (REST API)
- Uvicorn (ASGI server)
- Joblib (model persistence)

**Frontend** (if time permits):
- React + TypeScript
- TailwindCSS (styling)
- Recharts (visualizations)
- Axios (API client)

**Development Tools**:
- Jupyter Notebooks (EDA)
- Git (version control)
- Pytest (testing)
- Black (code formatting)

### 6.2 API Endpoints

- `POST /predict`: Make toxicity prediction
- `GET /predict/explain`: Get SHAP explanation
- `GET /model/info`: Model metadata and performance
- `GET /feature/importance`: Global feature importance
- `GET /history`: Recent predictions

### 6.3 Deployment Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Web UI    │─────>│  FastAPI     │─────>│  ML Model   │
│  (React)    │      │   Backend    │      │  (XGBoost)  │
└─────────────┘      └──────────────┘      └─────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │  Prediction  │
                     │   History    │
                     │  (SQLite)    │
                     └──────────────┘
```

---

## 7. Project Timeline

### Week 1: Data Exploration & Preprocessing (Nov 1-7)
- [ ] Load and explore ApisTox dataset
- [ ] Extract molecular descriptors from SMILES
- [ ] Statistical analysis and visualizations
- [ ] Data preprocessing pipeline development
- **Deliverable**: EDA Jupyter notebook, preprocessed dataset

### Week 2: Model Development & Selection (Nov 8-14)
- [ ] Train baseline Logistic Regression model
- [ ] Train ensemble models (RF, XGBoost, LightGBM)
- [ ] Hyperparameter tuning with cross-validation
- [ ] Model comparison and selection
- **Deliverable**: Trained models, performance metrics

### Week 3: Interpretability & API Development (Nov 15-21)
- [ ] SHAP analysis (global and local)
- [ ] LIME explanations
- [ ] Feature importance visualization
- [ ] FastAPI backend development
- **Deliverable**: Interpretability plots, REST API

### Week 4: Integration & Documentation (Nov 22-28)
- [ ] Frontend development (if time permits)
- [ ] End-to-end testing
- [ ] Documentation and presentation prep
- [ ] Final deployment
- **Deliverable**: Complete system, presentation, final report

---

## 8. Risk Assessment & Mitigation

### Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Low model accuracy | High | Medium | Try multiple algorithms, feature engineering |
| Class imbalance issues | Medium | High | SMOTE resampling, class weights, threshold tuning |
| Computational constraints | Medium | Low | Use efficient algorithms (XGBoost), cloud resources |
| API deployment issues | Low | Low | Thorough testing, containerization |

### Schedule Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Timeline delays | Medium | Medium | Prioritize core features, agile approach |
| Frontend complexity | Low | Medium | Focus on backend/API first, simple UI |
| Data processing time | Low | Medium | Optimize code, batch processing |

---

## 9. Expected Outcomes & Impact

### 9.1 Academic Deliverables

1. **Project Proposal** (this document): 2-3 pages
2. **Technical Report**: Comprehensive analysis with methodology, results, conclusions
3. **Presentation**: 12-15 minute talk with slides covering all phases
4. **Code Repository**: Well-documented, reproducible codebase
5. **Demonstration**: Live API demo during presentation

### 9.2 Technical Deliverables

1. **ML Models**: Trained classifiers with >80% accuracy
2. **REST API**: Production-ready FastAPI application
3. **Visualizations**: EDA plots, SHAP/LIME explanations, performance charts
4. **Documentation**: README, API docs, technical specifications
5. **Testing Suite**: Unit tests for preprocessing, models, API

### 9.3 Real-World Impact

**Agricultural Applications**:
- Rapid screening of new pesticide formulations
- Decision support for farmers and agronomists
- Compliance checking for organic certification

**Regulatory Applications**:
- Accelerated toxicity assessment for EPA/EFSA
- Data-driven policy recommendations
- Environmental impact assessments

**Conservation Applications**:
- Identify high-risk compounds for bee populations
- Guide habitat management near agricultural areas
- Support pollinator protection initiatives

---

## 10. Ethical Considerations

### 10.1 Environmental Ethics

- **Precautionary Principle**: When model confidence is low, err on side of bee safety
- **Transparency**: Full disclosure of model limitations and uncertainty
- **Stakeholder Impact**: Consider effects on farmers, beekeepers, regulators

### 10.2 Model Limitations

- **Data Bias**: Dataset may not represent all pesticide types or environmental conditions
- **Predictive Limitations**: Probabilities, not certainties; requires validation
- **Context Dependence**: Toxicity varies with dosage, application method, environmental factors

### 10.3 Responsible Use

- Model is decision support tool, not replacement for laboratory testing
- Results should inform, not dictate, regulatory decisions
- Continuous model updating as new data becomes available
- No use for developing harmful chemicals

---

## 11. Success Metrics

### Quantitative Metrics

✅ **Model Performance**: F1 Score >0.70, ROC-AUC >0.80  
✅ **API Performance**: Response time <1 second  
✅ **Code Quality**: Test coverage >80%  
✅ **Documentation**: Complete README, docstrings, API docs  

### Qualitative Metrics

✅ **Interpretability**: Clear SHAP/LIME visualizations  
✅ **Reproducibility**: All results can be regenerated  
✅ **Usability**: API is intuitive and well-documented  
✅ **Presentation**: Clear communication of methods and results  

---

## 12. References

1. **ApisTox Dataset**: Ioannidis, J. I., & Koh, H. C. (2024). ApisTox: A new benchmark dataset for the classification of the toxicity of chemicals to honey bees. *Scientific Data*, 11(1). https://doi.org/10.1038/s41597-024-04232-w

2. **SHAP**: Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.

3. **XGBoost**: Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*.

4. **SMOTE**: Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321-357.

5. **RDKit**: RDKit: Open-Source Cheminformatics Software. https://www.rdkit.org

6. **Bee Conservation**: vanEngelsdorp, D., & Meixner, M. D. (2010). A historical review of managed honey bee populations in Europe and the United States and the factors that may affect them. *Journal of Invertebrate Pathology*, 103, S80-S95.

---

## 13. Budget & Resources

### Computational Resources

- **Development**: Personal laptops/desktops (sufficient for dataset size)
- **Training**: CPU-based training (~30 minutes total)
- **Deployment**: Local server or free-tier cloud (Heroku/Railway)
- **Storage**: <1 GB (models, data, visualizations)

### Software (All Free/Open-Source)

- Python ecosystem (scikit-learn, XGBoost, FastAPI)
- RDKit (molecular descriptor calculation)
- SHAP/LIME (interpretability)
- Git/GitHub (version control)

### Time Investment

- Total: ~80-100 hours over 4 weeks
- Per team member: ~20-25 hours
- Weekly meetings: 2 hours

**Total Project Cost**: $0 (all open-source tools)

---

## 14. Conclusion

This project addresses a critical environmental and agricultural challenge through the application of modern machine learning techniques. By developing an accurate, interpretable, and deployable toxicity prediction system, we will demonstrate the practical application of predictive analytics to real-world problems with significant societal impact.

The combination of rigorous methodology, comprehensive evaluation, and production-ready implementation will showcase both technical competence and awareness of ethical implications in AI development. The final system will serve as a valuable tool for researchers, regulators, and agricultural professionals while contributing to honey bee conservation efforts.

**We are committed to delivering a high-quality, well-documented project that meets all course requirements while advancing the important goal of protecting pollinator populations.**

---

**Prepared by**: IME 372 Project Team  
**Date**: November 7, 2025  
**Approved for Implementation**: [Instructor Signature]

---

*This proposal represents our commitment to excellence in predictive analytics and responsible AI development.*

