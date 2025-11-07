# Presentation Guide
## Honey Bee Toxicity Prediction System

**Duration**: 12-15 minutes  
**Format**: Technical presentation with live demo  
**Audience**: IME 372 class and instructor

---

## Presentation Materials

### Primary Materials
1. **Slides**: `PRESENTATION_SLIDES.md` (30 slides + appendix)
2. **Live Demo**: API at http://localhost:8000/docs
3. **Visuals**: `outputs/figures/` (12 plots)

### Supporting Materials
- Model Card (`docs/MODEL_CARD.md`)
- API Documentation (`docs/API_DOCS.md`)
- Project Proposal (`docs/project_proposal.md`)
- System Demo (`test_system.py`)

---

## Pre-Presentation Checklist

### 1 Day Before
- [ ] Review all 30 slides
- [ ] Practice presentation timing (12-15 min target)
- [ ] Test API on presentation computer
- [ ] Verify all visualizations display correctly
- [ ] Prepare backup slides (PDF export)
- [ ] Test screen sharing/projector
- [ ] Prepare demo inputs

### 1 Hour Before
- [ ] Start API server: `python app/backend/main.py`
- [ ] Open API docs: http://localhost:8000/docs
- [ ] Open slides in presentation mode
- [ ] Have `outputs/figures/` folder ready
- [ ] Close unnecessary applications
- [ ] Test audio/video if virtual
- [ ] Water bottle ready

### 5 Minutes Before
- [ ] Slides loaded on Slide 1
- [ ] API health check: http://localhost:8000/health
- [ ] Browser tabs: API docs, GitHub (if showing code)
- [ ] Deep breath, smile 🐝

---

## Presentation Flow

### Opening (2 minutes)
**Slides 1-3**: Title, Problem, Objectives

**Key Points**:
- Bees pollinate 1/3 of food crops
- Testing costs $10K-$50K per compound, takes months
- We built ML system: 83.6% accuracy, <150ms predictions
- Production-ready API with full interpretability

**Delivery**:
- Start with impact: "Bees are dying, we can help predict which pesticides are safe"
- Hook: "Traditional testing takes months and $50,000. Our system: 100ms."

---

### Dataset (2 minutes)
**Slides 4-5**: Data Overview, Target Distribution

**Key Points**:
- ApisTox: 1,035 compounds, 191 years (1832-2023!)
- Zero missing values, peer-reviewed
- Class imbalance: 71% non-toxic, 29% toxic
- Solved with SMOTE resampling

**Visual**: Show Slide 5 bar chart

**Delivery**:
- "Clean, high-quality data from three trusted sources"
- "Imbalanced but not severely - we handled it"

---

### Methodology (3 minutes)
**Slides 6-9**: Features, Preprocessing, Model Selection, Comparison

**Key Points**:
- 15 molecular descriptors from SMILES using RDKit
- StandardScaler + one-hot encoding + SMOTE
- Compared 3 algorithms: LogReg, RandomForest, XGBoost
- Selected XGBoost: 85.6% val accuracy, 0.74 F1 score

**Visual**: Show Slide 9 model comparison table

**Delivery**:
- "Molecules → numbers → predictions"
- "Systematic comparison, not just one model"
- "XGBoost won on F1 score - best balance"

---

### Results (2 minutes)
**Slides 10-11**: Test Performance, ROC Curve

**Key Points**:
- **83.57% test accuracy**
- **85.83% ROC-AUC** (excellent probability calibration)
- Confusion matrix: 15 false positives (conservative, good!), 19 false negatives (room for improvement)
- **89.9% specificity** - correctly identifies safe compounds

**Visual**: Show confusion matrix on Slide 10

**Delivery**:
- "Exceeded 80% target accuracy"
- "More false positives than negatives - favors bee safety"
- "Would you rather: wrongly ban safe pesticide OR wrongly approve toxic one?"

---

### **LIVE DEMO** (2 minutes) ⭐
**Transition**: "Let me show you the system in action..."

**Demo Steps**:
1. **Health Check**
   - Navigate to http://localhost:8000/docs
   - Click `GET /health` → Try it out → Execute
   - Show response: `{"status": "healthy"}`

2. **Model Info**
   - Click `GET /model/info` → Execute
   - Highlight: XGBoost, 83.6% accuracy, 24 features

3. **Make Prediction**
   - Click `POST /predict` → Try it out
   - Use pre-filled example (insecticide=1, high LogP)
   - Execute and show response:
     ```json
     {
       "prediction": 1,
       "label_text": "Toxic",
       "confidence": 0.87,
       "probability_toxic": 0.87
     }
     ```
   - **Emphasize**: "87% confidence, predicted toxic - makes sense, it's an insecticide!"

4. **Feature Importance** (if time)
   - Click `GET /feature/importance` → Execute
   - Show: insecticide (1.366), herbicide (1.054), fungicide (0.740)

**Backup Plan**: If API fails, show screenshots from `outputs/figures/`

**Delivery**:
- "This is production-ready - anyone can use this API"
- "Notice the confidence score - we don't just say toxic, we say HOW confident"
- "Real-time predictions in under 150 milliseconds"

---

### Interpretability (2 minutes)
**Slides 12-15**: SHAP Analysis, Summary Plot, Waterfall Examples, LIME

**Key Points**:
- **Chemical type (insecticide) is #1 predictor** (importance 1.366)
- Herbicide #2, Fungicide #3 - makes scientific sense!
- LogP #5 (lipophilicity) - fat-soluble compounds accumulate
- SHAP: insecticide=1 → +0.48 toward toxic
- Aligns perfectly with toxicology knowledge

**Visual**: Show Slide 12 feature importance bar chart, Slide 13 beeswarm

**Delivery**:
- "Not a black box - we know WHY it predicts toxic"
- "Insecticides kill insects. Bees ARE insects!"
- "Model learned what toxicologists already know"

---

### System & Ethics (2 minutes)
**Slides 17-20**: Architecture, Assumptions/Limitations, Ethics, Stakeholders

**Key Points**:
- End-to-end pipeline: Data → Preprocessing → Model → API → Predictions
- **Limitations**: 67.8% recall on toxic class (misses 1/3), historical bias, no dosage info
- **Ethics**: Favor bee safety, require lab validation for <70% confidence, transparent limitations
- **Stakeholders**: Farmers, beekeepers, regulators, chemical companies, researchers

**Delivery**:
- "We're transparent about what works and what doesn't"
- "Low confidence → lab test required. Precautionary principle."
- "This helps everyone from farmers to EPA regulators"

---

### Conclusion (1 minute)
**Slide 24**: Results Summary

**Key Points**:
- ✅ 83.6% accuracy, 85.8% ROC-AUC - exceeded targets
- ✅ Full interpretability - chemical type #1 predictor
- ✅ Production API - 6 endpoints, <150ms
- ✅ Comprehensive docs - 5,000+ lines

**Delivery**:
- "We built a complete, production-ready system"
- "Not just a model - API, tests, docs, ethics analysis"
- "Ready to protect bees and support sustainable agriculture"

---

### Q&A (5 minutes)
**Slide 30**: Questions

**Anticipated Questions**:

**Q: Why only 83% accuracy, not 95%?**  
A: Class imbalance and biological variability. Toxicity is complex - same compound can vary by dose, exposure, bee colony. 83% is strong for this domain, and we have 86% ROC-AUC showing good probability estimates.

**Q: How do you handle new pesticide classes not in training data?**  
A: Good question! Model may not generalize well to novel structures. That's why we output confidence scores - low confidence (<70%) signals "send to lab for testing." We're transparent about limitations.

**Q: What's the false negative rate and why is it high?**  
A: 32% (19/59 toxic compounds). This is our main area for improvement. We could increase sensitivity by lowering prediction threshold, but that increases false positives. Trade-off depends on use case - we favor bee safety.

**Q: How long did this take?**  
A: Data cleaning was done by ApisTox team. Our work: ~3-4 weeks part-time. Training is fast (<2 seconds), so we iterated quickly.

**Q: Can this be used for other species?**  
A: Methodology yes, model no. Would need to retrain on data for that species. Transfer learning could help.

**Q: What about sublethal effects?**  
A: Great point! Our model only predicts acute toxicity (death). Sublethal effects (behavior, reproduction) are not captured. Important limitation documented in Model Card.

---

## Backup Slides (If Time Permits)

### Appendix A: Detailed Methodology
- More on SMOTE
- Hyperparameter tuning details
- Cross-validation strategy

### Appendix B: Feature Correlations
- Heatmap showing descriptor relationships
- Multicollinearity analysis

### Appendix C: Error Analysis
- Which compounds are misclassified?
- Patterns in false positives/negatives
- Confidence analysis

---

## Technical Setup

### Required Running
```bash
# Start API (must be running for demo)
python app/backend/main.py
```

### Recommended Open
- Browser tab: http://localhost:8000/docs
- File explorer: `outputs/figures/`
- Backup: Slides as PDF

### Not Required
- Jupyter notebooks (can mention, don't need to show)
- Code editor (unless asked about implementation)
- Terminal logs (clean, not messy)

---

## Timing Breakdown

| Section | Time | Slides | Must Cover? |
|---------|------|--------|-------------|
| Opening | 2 min | 1-3 | ✅ Yes |
| Dataset | 2 min | 4-5 | ✅ Yes |
| Methodology | 3 min | 6-9 | ✅ Yes |
| Results | 2 min | 10-11 | ✅ Yes |
| **Live Demo** | 2 min | 16 | ✅ Yes |
| Interpretability | 2 min | 12-15 | ✅ Yes |
| System & Ethics | 2 min | 17-20 | ✅ Yes |
| Conclusion | 1 min | 24 | ✅ Yes |
| **Total** | **16 min** | | |
| Q&A | 5 min | 30 | ✅ Yes |
| **Grand Total** | **21 min** | | **Adjust for 12-15** |

**Adjustment Strategy**:
- Skip slides 17-20 if running long (mention briefly in conclusion)
- Shorten demo to 1 minute (just show prediction, skip model info)
- This reduces to 13-14 minutes

---

## Presentation Tips

### Dos ✅
- **Start strong**: Hook with bee crisis statistics
- **Show enthusiasm**: This is cool tech solving real problems
- **Make eye contact**: Engage audience
- **Pace yourself**: Breathe, don't rush
- **Tell story**: Problem → Solution → Impact
- **Live demo**: Shows confidence, makes it real
- **Own limitations**: Transparency builds credibility

### Don'ts ❌
- **Don't read slides**: Slides are prompts, not script
- **Don't over-explain**: Audience is technical but not ML experts
- **Don't hide mistakes**: If demo fails, laugh, use backup
- **Don't go over time**: Respect schedule
- **Don't skip demo**: It's the wow moment
- **Don't ignore questions**: "Great question! The answer is..."

---

## Troubleshooting

### Demo Fails
**Plan B**: Use screenshots from `outputs/figures/` and walk through API docs PDF

### Forgot Something Important
**Plan C**: Mention in Q&A: "Great point, I should have mentioned..."

### Running Over Time
**Plan D**: Skip system architecture (Slide 17), jump to conclusion

### Running Under Time
**Plan E**: Show more SHAP plots, discuss error analysis from appendix

---

## Post-Presentation

### Immediately After
- [ ] Save any questions you couldn't answer
- [ ] Note what went well vs poorly
- [ ] Collect feedback from instructor/classmates

### Follow-Up
- [ ] Send thank you email with GitHub link (if applicable)
- [ ] Upload slides/demo recording (if requested)
- [ ] Update portfolio with project

---

## Confidence Boosters

**You've got this because**:
1. ✅ You have working code that delivers results
2. ✅ You understand the domain (toxicology + ML)
3. ✅ You have comprehensive backup materials
4. ✅ You've practiced the demo
5. ✅ You know your limitations and can defend choices
6. ✅ Your results are strong (83% accuracy, 86% ROC-AUC)
7. ✅ You have a compelling story (save the bees!)

**Remember**:
- This is YOUR project - you know it best
- Audience WANTS you to succeed
- It's okay to say "I don't know, but I can find out"
- Deep breaths, smile, have fun! 🐝

---

**Break a leg! You're going to do great!** 🎤🐝

---

## Emergency Contacts

- **API Port**: 8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Backup Slides**: Export to PDF before presenting
- **Figures Folder**: `outputs/figures/`

