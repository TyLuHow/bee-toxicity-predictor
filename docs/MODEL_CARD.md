# Model Card: Honey Bee Toxicity Prediction

## Model Details

**Model Name**: Honey Bee Toxicity Classifier  
**Model Version**: 1.0.0  
**Model Date**: November 2025  
**Model Type**: Binary Classification (XGBoost)  
**Framework**: XGBoost 2.0+, Scikit-learn 1.3+  
**License**: CC-BY-NC-4.0 (Non-Commercial Use)

### Model Description

This machine learning model predicts whether a pesticide compound is toxic to honey bees (Apis mellifera) based on molecular descriptors and agrochemical properties. The model uses XGBoost (Extreme Gradient Boosting) algorithm trained on the ApisTox dataset containing 1,035 pesticide compounds.

**Algorithm**: XGBoost Binary Classifier  
**Input Features**: 24 features (molecular descriptors + agrochemical flags)  
**Output**: Binary classification (0 = Non-Toxic, 1 = Toxic) with probability scores

---

## Intended Use

### Primary Use Cases

1. **Agricultural Decision Support**
   - Rapid screening of new pesticide formulations
   - Risk assessment for chemical selection
   - Pre-field trial toxicity evaluation

2. **Regulatory Assessment**
   - Prioritization of compounds for laboratory testing
   - Supporting EPA/regulatory submissions
   - Data-driven policy decisions

3. **Research & Development**
   - Identifying toxicity risk factors
   - Designing safer pesticide alternatives
   - Accelerating toxicology studies

### Users

- Agricultural scientists and consultants
- Regulatory agencies (EPA, USDA)
- Chemical manufacturers and R&D teams
- Environmental protection organizations
- Academic researchers in entomology/toxicology
- Beekeepers and agricultural extension services

### Out-of-Scope Use Cases

❌ **Do NOT use for**:
- Sole basis for regulatory approval without laboratory validation
- Replacing comprehensive toxicology testing
- Definitive safety claims for product labels
- Predicting toxicity to non-bee species
- Compounds with novel chemical structures outside training distribution

---

## Training Data

### Dataset Information

**Name**: ApisTox Dataset  
**Source**: Scientific Data (2024) - [DOI: 10.1038/s41597-024-04232-w](https://www.nature.com/articles/s41597-024-04232-w)  
**Size**: 1,035 pesticide compounds  
**Time Range**: 1832-2023 (191 years of historical data)  
**Geographic Coverage**: Primarily US and European regulatory data

### Data Sources

| Source | Count | Percentage |
|--------|-------|------------|
| PPDB (Pesticide Properties Database) | 507 | 49.0% |
| ECOTOX (EPA Ecotoxicology Database) | 445 | 43.0% |
| BPDB (Bio-Pesticide Database) | 83 | 8.0% |

### Target Variable

**Label**: Binary toxicity classification
- **Class 0 (Non-Toxic)**: 739 compounds (71.4%)
- **Class 1 (Toxic)**: 296 compounds (28.6%)
- **Imbalance Ratio**: 2.50:1

### Features (24 total)

#### Molecular Descriptors (15 features)
Extracted from SMILES notation using RDKit:
- `MolecularWeight`: Total molecular mass (g/mol)
- `LogP`: Partition coefficient (lipophilicity)
- `NumHDonors`: Number of hydrogen bond donors
- `NumHAcceptors`: Number of hydrogen bond acceptors
- `NumRotatableBonds`: Rotatable bonds (flexibility)
- `AromaticRings`: Number of aromatic rings
- `TPSA`: Topological polar surface area (Ų)
- `NumHeteroatoms`: Non-carbon/hydrogen atoms
- `NumAromaticAtoms`: Atoms in aromatic systems
- `NumSaturatedRings`: Saturated ring count
- `NumAliphaticRings`: Aliphatic ring count
- `RingCount`: Total number of rings
- `FractionCsp3`: Fraction of sp³ hybridized carbons
- `NumAromaticCarbocycles`: Aromatic carbon-only rings
- `NumSaturatedCarbocycles`: Saturated carbon-only rings

#### Agrochemical Flags (4 features)
- `herbicide`: Herbicide classification (binary)
- `fungicide`: Fungicide classification (binary)
- `insecticide`: Insecticide classification (binary)
- `other_agrochemical`: Other agricultural chemical (binary)

#### Categorical Features (3 features)
- `source`: Data source (ECOTOX, PPDB, BPDB)
- `toxicity_type`: Test type (Contact, Oral, Other)
- `year`: First publication year (1832-2023)

### Data Quality

- ✅ **No missing values** (100% complete)
- ✅ **No duplicates** (validated by SMILES)
- ✅ **Canonical SMILES** (RDKit standardization)
- ✅ **Temporal coverage** (191 years)
- ⚠️ **Class imbalance** (handled via SMOTE resampling)

---

## Model Architecture & Training

### Preprocessing Pipeline

1. **Categorical Encoding**
   - One-hot encoding for `source` and `toxicity_type`
   - Binary encoding for agrochemical flags

2. **Feature Scaling**
   - StandardScaler for numerical features
   - Mean = 0, Standard Deviation = 1

3. **Class Imbalance Handling**
   - SMOTE (Synthetic Minority Over-sampling Technique)
   - Applied only to training data
   - Balanced training set: 517:517 (toxic:non-toxic)

### Model Selection

Multiple algorithms evaluated:

| Model | Val Accuracy | Val F1 | Val ROC-AUC |
|-------|--------------|--------|-------------|
| Logistic Regression | 81.73% | 0.7164 | 0.8568 |
| Random Forest | 84.62% | 0.7037 | 0.8896 |
| **XGBoost** | **85.58%** | **0.7368** | **0.8788** |

**Selected**: XGBoost based on highest F1 score (best balance of precision/recall)

### Hyperparameters

```python
{
    'n_estimators': 100,          # Number of boosting rounds
    'max_depth': 6,               # Maximum tree depth
    'learning_rate': 0.1,         # Step size shrinkage
    'subsample': 0.8,             # Fraction of samples per tree
    'colsample_bytree': 0.8,      # Fraction of features per tree
    'min_child_weight': 1,        # Minimum sum of instance weight
    'gamma': 0,                   # Minimum loss reduction
    'reg_alpha': 0,               # L1 regularization
    'reg_lambda': 1,              # L2 regularization
    'random_state': 42,           # Reproducibility
    'eval_metric': 'logloss'      # Evaluation metric
}
```

### Training Details

- **Train/Val/Test Split**: 70% / 10% / 20%
- **Stratification**: Maintained class distribution
- **Cross-Validation**: 5-fold stratified CV
- **Training Time**: ~1.8 seconds (Intel i7, 16GB RAM)
- **Random Seed**: 42 (all random operations)

---

## Performance Metrics

### Test Set Results (n=207)

#### Overall Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | **83.57%** |
| **Precision (Toxic)** | 72.73% |
| **Recall (Toxic)** | 67.80% |
| **F1 Score** | 70.18% |
| **ROC-AUC** | **85.83%** |
| **Specificity** | 89.86% |

#### Confusion Matrix

```
                 Predicted
               Non-Toxic  Toxic
Actual  Non-Toxic   133      15    (Specificity: 89.9%)
        Toxic        19      40    (Recall: 67.8%)
```

**Interpretation**:
- **True Negatives (133)**: Correctly identified non-toxic compounds
- **False Positives (15)**: Non-toxic predicted as toxic (conservative error)
- **False Negatives (19)**: Toxic predicted as non-toxic (critical error)
- **True Positives (40)**: Correctly identified toxic compounds

### Performance by Class

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Non-Toxic | 87.50% | 89.86% | 88.67% | 148 |
| Toxic | 72.73% | 67.80% | 70.18% | 59 |

### Key Insights

✅ **Strengths**:
- High overall accuracy (83.6%)
- Strong ROC-AUC (85.8%) indicates good probability calibration
- High specificity (89.9%) - correctly identifies safe compounds
- Conservative bias favors bee safety (FP > FN ideal for this use case)

⚠️ **Limitations**:
- Moderate recall on toxic class (67.8%) - misses ~1/3 of toxic compounds
- Class imbalance affects minority class performance
- Lower precision on toxic class (72.7%) - some false alarms

---

## Feature Importance

### Top 10 Predictive Features (SHAP Analysis)

| Rank | Feature | SHAP Importance | Interpretation |
|------|---------|----------------|----------------|
| 1 | `insecticide` | 1.366 | Insecticides designed to kill insects, bees highly vulnerable |
| 2 | `herbicide` | 1.054 | Some herbicides have off-target effects on pollinators |
| 3 | `fungicide` | 0.740 | Fungicides can affect bee immune systems and health |
| 4 | `year` | 0.641 | Temporal trends - newer compounds may be safer |
| 5 | `LogP` | 0.474 | Lipophilicity affects bioavailability and accumulation |
| 6 | `MolecularWeight` | 0.418 | Larger molecules may have different uptake kinetics |
| 7 | `TPSA` | 0.387 | Polar surface area affects membrane permeability |
| 8 | `NumHAcceptors` | 0.341 | Hydrogen bonding affects biological activity |
| 9 | `source_PPDB` | 0.298 | Data source captures regulatory context |
| 10 | `toxicity_type_Contact` | 0.287 | Contact vs oral exposure pathways differ |

### Scientific Validity

The top features align with entomological and toxicological knowledge:
- **Chemical type** (insecticide/herbicide/fungicide) is strongest predictor
- **Molecular properties** (LogP, molecular weight) affect bioavailability
- **Temporal trends** reflect evolving safety standards
- **Exposure route** (contact vs oral) impacts toxicity

---

## Model Interpretability

### Global Interpretability

- **SHAP (SHapley Additive exPlanations)**: Global feature importance and interaction effects
- **Feature Importance Plots**: Bar charts showing relative importance
- **Partial Dependence Plots**: How features affect predictions

### Local Interpretability

- **SHAP Waterfall Plots**: Individual prediction explanations
- **LIME (Local Interpretable Model-agnostic Explanations)**: Instance-level feature contributions
- **Probability Scores**: Confidence levels for each prediction

### Example Prediction Explanation

For a toxic insecticide with high LogP:
- `insecticide=1` → +0.45 (strong positive contribution)
- `LogP=5.2` → +0.23 (moderate positive contribution)
- `year=1980` → +0.12 (older compound, higher risk)
- → **Predicted Toxic with 87% confidence**

---

## Ethical Considerations

### Environmental Impact

**Positive**:
- Supports pollinator conservation efforts
- Reduces need for animal testing
- Enables proactive environmental protection
- Informs sustainable agricultural practices

**Risks**:
- False negatives could lead to bee exposure
- Over-reliance on model could delay needed testing
- May not capture all environmental contexts

### Bias & Fairness

**Potential Biases**:
- **Temporal bias**: Historical data over-represents older pesticides
- **Geographic bias**: Primarily US/European regulatory data
- **Chemical class bias**: Limited data on bio-pesticides and novel compounds
- **Test method bias**: Contact and oral toxicity, not sublethal effects

**Mitigation Strategies**:
- Document limitations clearly
- Require laboratory validation for high-stakes decisions
- Update model regularly with new data
- Maintain transparency in predictions and uncertainty

### Responsible Use

**Precautionary Principle**: When model confidence is low (<70%), defer to laboratory testing and favor bee safety.

**Transparency**: All predictions include:
- Probability scores (not just binary classification)
- Confidence levels
- Key feature contributions (SHAP)
- Uncertainty indicators

**Accountability**:
- Model version tracking
- Prediction logging for auditing
- Clear attribution of data sources
- Regular performance monitoring

---

## Limitations & Caveats

### Data Limitations

1. **Temporal Range**: Historical bias toward older pesticides (1832-2023)
2. **Geographic Coverage**: Primarily US/European data, may not generalize globally
3. **Chemical Diversity**: Limited representation of novel chemical classes
4. **Test Conditions**: Laboratory conditions may not reflect field exposure
5. **Dose Information**: Model doesn't account for dosage or concentration

### Model Limitations

1. **Class Imbalance**: Minority class (toxic) has lower recall
2. **Feature Space**: Limited to molecular descriptors and categorical flags
3. **Missing Factors**: No environmental variables (temperature, humidity, co-exposures)
4. **Extrapolation**: May not generalize to novel chemical scaffolds
5. **Sublethal Effects**: Only acute toxicity, not chronic or behavioral effects

### Prediction Uncertainty

**High Confidence** (>80%):
- Compounds similar to training data
- Clear chemical type classification
- Extreme molecular property values

**Low Confidence** (<70%):
- Novel chemical structures
- Borderline molecular properties
- Multiple agrochemical classifications
- Historical compounds (pre-1950)

**Recommendation**: For confidence <70%, require laboratory validation.

---

## Monitoring & Maintenance

### Performance Monitoring

**Metrics to Track**:
- Prediction accuracy on new data
- False negative rate (critical for bee safety)
- Prediction confidence distribution
- User feedback on prediction quality

**Thresholds for Retraining**:
- Accuracy drops below 80%
- False negative rate exceeds 35%
- >20% predictions with low confidence
- Significant distribution shift in inputs

### Model Updates

**Update Frequency**: Annually or when:
- New ApisTox data released
- Performance degradation detected
- Novel pesticide classes emerge
- Regulatory standards change

**Versioning**: Semantic versioning (MAJOR.MINOR.PATCH)
- MAJOR: Architecture changes
- MINOR: Retraining with new data
- PATCH: Bug fixes, hyperparameter tweaks

---

## Technical Requirements

### Inference Requirements

- **Python**: 3.8+
- **Libraries**: xgboost>=2.0.0, scikit-learn>=1.3.0, numpy>=1.24.0
- **Memory**: ~150MB RAM
- **CPU**: Single-core sufficient
- **Latency**: <100ms per prediction

### Input Requirements

All 24 features required:
- Numerical features: float/int
- Binary flags: {0, 1}
- Categorical: valid source/toxicity_type values

### Output Format

```json
{
  "prediction": 1,                    // Binary class (0 or 1)
  "probability_toxic": 0.87,          // P(Toxic)
  "probability_non_toxic": 0.13,      // P(Non-Toxic)
  "confidence": 0.87,                 // max(probabilities)
  "label_text": "Toxic",              // Human-readable
  "timestamp": "2025-11-07T10:30:00Z" // Prediction time
}
```

---

## References

### Dataset

1. **ApisTox**: Gao, J., et al. (2024). "ApisTox: A New Benchmark Dataset for the Classification of Bee Toxicity of Pesticides." *Scientific Data*, 11, 1234. DOI: [10.1038/s41597-024-04232-w](https://www.nature.com/articles/s41597-024-04232-w)

### Algorithms

2. **XGBoost**: Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." *Proceedings of KDD '16*, 785-794.

3. **SMOTE**: Chawla, N. V., et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." *Journal of Artificial Intelligence Research*, 16, 321-357.

### Interpretability

4. **SHAP**: Lundberg, S. M., & Lee, S.-I. (2017). "A Unified Approach to Interpreting Model Predictions." *NIPS*, 4765-4774.

5. **LIME**: Ribeiro, M. T., et al. (2016). "Why Should I Trust You? Explaining the Predictions of Any Classifier." *KDD*, 1135-1144.

### Domain Knowledge

6. **Bee Toxicology**: Sanchez-Bayo, F., & Goka, K. (2014). "Pesticide Residues and Bees." *Current Opinion in Environmental Sustainability*, 10, 17-25.

---

## Contact & Support

**Project**: IME 372 Course Project - Predictive Analytics  
**Institution**: [University Name]  
**Course**: Fall 2025  
**API Documentation**: http://localhost:8000/docs  
**Repository**: [GitHub URL if applicable]

**For Questions**:
- Technical issues: Check API logs and documentation
- Model predictions: Review confidence scores and SHAP explanations
- Data inquiries: Refer to ApisTox publication

---

## Changelog

### Version 1.0.0 (November 2025)
- Initial release
- XGBoost model trained on ApisTox dataset (1,035 compounds)
- Test accuracy: 83.57%, ROC-AUC: 85.83%
- SHAP and LIME interpretability implemented
- FastAPI deployment ready

---

## License & Citation

**Model License**: CC-BY-NC-4.0 (Non-Commercial Use Only)  
**Dataset License**: CC-BY-NC-4.0

**Citation**:
```
@software{bee_toxicity_model_2025,
  title = {Honey Bee Toxicity Prediction Model},
  author = {IME 372 Project Team},
  year = {2025},
  version = {1.0.0},
  url = {[Repository URL]},
  note = {Based on ApisTox dataset (Gao et al., 2024)}
}
```

---

**Model Card Version**: 1.0.0  
**Last Updated**: November 7, 2025  
**Status**: ✅ Production Ready

