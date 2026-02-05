# Safety Pharmacology ML Model Assessment
## Comprehensive Framework for Predictive Toxicology

---

## 📋 Project Overview

This repository contains a complete, scientifically rigorous framework for:
1. Extracting bioactivity data from ChEMBL for 11 key safety pharmacology targets
2. Assessing machine learning models (RF, XGBoost, Neural Networks, Deep Learning, GANs)
3. Quantifying prediction uncertainty using multiple methods
4. Statistically comparing models with appropriate corrections
5. Selecting the best model based on multi-criteria decision analysis

**Goal**: Predict binding/inhibition probability with quantified uncertainty for safety-critical targets

---

## 🎯 Safety Targets (11 Total)

### Cardiac Safety (3)
- **hERG (CHEMBL240)**: K+ channel - primary cardiac liability
- **Cav1.2 (CHEMBL1940)**: L-type Ca2+ channel
- **Nav1.5 (CHEMBL1993)**: Cardiac Na+ channel

### Hepatotoxicity (5)
- **CYP3A4 (CHEMBL340)**: Major drug metabolizing enzyme (50% of drugs)
- **CYP2D6 (CHEMBL289)**: Polymorphic enzyme with genetic variability
- **CYP2C9 (CHEMBL3397)**: Warfarin metabolism, drug-drug interactions
- **CYP1A2 (CHEMBL3356)**: Caffeine metabolism
- **CYP2C19 (CHEMBL3622)**: Clopidogrel activation

### Other Safety (2)
- **P-glycoprotein (CHEMBL4302)**: Efflux transporter, affects BBB penetration
- **BSEP (CHEMBL4105)**: Bile salt export pump, cholestasis risk

---

## 📁 Repository Structure

```
.
├── README.md                           # This file
├── EXECUTIVE_SUMMARY.py                # Comprehensive methodology (23KB)
├── retrieve_chembl_safety_data.py      # Practical ChEMBL API script (13KB)
├── retrieve_safety_data.py             # Alternative data retrieval framework (8KB)
├── model_assessment_plan.py            # Complete assessment framework (37KB)
└── deep_learning_assessment.py         # Deep learning & uncertainty (21KB)
```

---

## 🚀 Quick Start

### 1. Data Retrieval

```bash
# Install dependencies
pip install requests pandas numpy scikit-learn rdkit torch xgboost lightgbm

# Run ChEMBL data retrieval
python retrieve_chembl_safety_data.py
```

**Expected output:**
- `safety_targets_bioactivity.csv` - Combined dataset (~50K-100K compounds)
- `summary_statistics.csv` - Data quality metrics by target

**Expected data volume:**
- Total measurements: 200K-500K
- Unique compounds: 50K-100K
- File size: 50-200 MB

### 2. Model Assessment

The complete assessment workflow is documented in `model_assessment_plan.py`:

```python
from model_assessment_plan import ModelAssessmentPlan

# Initialize framework
assessment = ModelAssessmentPlan(random_state=42)

# Load your data
# X_train, y_train, X_val, y_val, X_test, y_test = load_data()

# Get model configurations
models = assessment.get_model_configurations()

# Hyperparameter optimization
results = assessment.hyperparameter_optimization(
    X_train, y_train, 
    model_name='XGBoost',
    optimization_method='random_search',
    n_iter=50
)

# Cross-validation evaluation
cv_results = assessment.stratified_cross_validation(
    X_train, y_train, 
    model,
    n_splits=5, 
    n_repeats=3
)

# External validation
test_results = assessment.external_validation(
    model, 
    X_train, y_train,
    X_test, y_test
)

# Uncertainty quantification
uncertainty = assessment.conformal_prediction(
    model,
    X_train, y_train,
    X_calibration, y_calibration,
    X_test,
    confidence_level=0.95
)
```

### 3. Deep Learning & Uncertainty

For deep learning approaches with advanced uncertainty quantification:

```python
from deep_learning_assessment import (
    EnsembleDeepNetwork,
    MolecularGAN,
    assess_deep_learning_models
)

# Train ensemble with uncertainty
ensemble = EnsembleDeepNetwork(
    input_dim=2048,
    n_models=5,
    hidden_dims=[512, 256, 128]
)

ensemble.train_ensemble(X_train, y_train, X_val, y_val, epochs=100)

# Predictions with epistemic + aleatoric uncertainty
predictions = ensemble.predict_with_uncertainty(X_test, n_iterations=100)

print(f"Mean prediction: {predictions['mean']}")
print(f"Total uncertainty: {predictions['total_uncertainty']}")
print(f"Epistemic uncertainty: {predictions['epistemic_uncertainty']}")
print(f"Aleatoric uncertainty: {predictions['aleatoric_uncertainty']}")
```

---

## 📊 Model Comparison Framework

### Models Included

1. **Random Forest** (Baseline)
   - Fast, robust, interpretable
   - Good for imbalanced data
   - Expected ROC-AUC: 0.75-0.85

2. **XGBoost** (Best Performance)
   - State-of-the-art gradient boosting
   - Built-in regularization
   - Expected ROC-AUC: 0.80-0.90

3. **LightGBM** (Fast Training)
   - Efficient for large datasets
   - Lower memory usage than XGBoost
   - Expected ROC-AUC: 0.78-0.88

4. **Neural Networks** (Flexible)
   - MLP with dropout
   - MC Dropout for uncertainty
   - Expected ROC-AUC: 0.75-0.85

5. **Ensemble Deep Networks** (Best Uncertainty)
   - 5 models with bootstrap sampling
   - Epistemic + aleatoric uncertainty decomposition
   - Expected ROC-AUC: 0.78-0.88

6. **SVM** (High-dimensional)
   - Effective in high-dimensional spaces
   - Memory efficient
   - Expected ROC-AUC: 0.72-0.82

7. **GAN** (Data Augmentation)
   - Address class imbalance
   - Generate synthetic compounds
   - Improves minority class performance

### Evaluation Metrics

**Performance Metrics:**
- ROC-AUC: Overall discrimination ability
- PR-AUC: Performance on imbalanced data
- MCC: Matthews Correlation Coefficient (balanced metric)
- Sensitivity @ 90% Specificity
- Specificity @ 90% Sensitivity

**Calibration Metrics:**
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Brier Score
- Calibration curves

**Uncertainty Metrics:**
- Prediction intervals (conformal prediction)
- Epistemic uncertainty (model uncertainty)
- Aleatoric uncertainty (data noise)
- Sharpness and resolution

---

## 🔬 Statistical Rigor

### Validation Strategy
- **Cross-validation**: 5-fold stratified × 3 repeats = 15 evaluations
- **External validation**: 20% held-out test set (scaffold split)
- **Applicability domain**: Leverage approach + LOF

### Statistical Tests
- Paired t-tests with Bonferroni correction (multiple comparisons)
- Effect size calculations (Cohen's d)
- McNemar's test for classifier comparison
- Friedman test for multiple models

### Uncertainty Quantification
1. **Conformal Prediction** (Recommended)
   - Coverage guarantee: P(y_true ∈ prediction_set) ≥ 95%
   - Model-agnostic, non-parametric
   
2. **Monte Carlo Dropout**
   - 100 forward passes with dropout enabled
   - Estimates epistemic uncertainty
   
3. **Ensemble Uncertainty**
   - Variance across 5+ models
   - Decomposes epistemic + aleatoric uncertainty

---

## 📈 Expected Performance

Based on literature and similar studies:

| Target | Expected ROC-AUC | Data Quality | Comments |
|--------|------------------|--------------|----------|
| hERG | 0.80-0.90 | High | Well-studied, good data |
| CYP3A4 | 0.75-0.85 | Medium | Substrate-dependent |
| CYP2D6 | 0.75-0.85 | Medium | Polymorphic |
| P-gp | 0.70-0.80 | Medium | Complex mechanism |
| Others | 0.70-0.85 | Variable | Depends on data volume |

**Best overall models (predicted):**
1. XGBoost: Highest average performance
2. Ensemble DNN: Best uncertainty quantification
3. Random Forest: Best balance (performance/speed/interpretability)

---

## ⚙️ Technical Requirements

### Python Dependencies
```bash
pip install numpy pandas scikit-learn
pip install xgboost lightgbm
pip install torch torchvision  # For deep learning
pip install rdkit  # For molecular descriptors/fingerprints
pip install matplotlib seaborn  # For visualization
pip install scipy  # For statistical tests
pip install requests  # For ChEMBL API
```

### Computational Resources
- **Minimum**: 16GB RAM, 4 CPU cores
- **Recommended**: 32GB+ RAM, 8+ CPU cores, GPU (for deep learning)
- **Storage**: 10GB for data and models

### Estimated Runtime
- Data retrieval: 2-4 hours
- Feature engineering: 1-2 hours
- Model training (per model): 1-4 hours
- Cross-validation: 5-10 hours per model
- Total: ~5-10 days (part-time) or 2-3 days (full-time)

---

## 📋 Implementation Checklist

### Phase 1: Data Preparation (Weeks 1-2)
- [ ] Extract bioactivity data from ChEMBL (IC50, Ki)
- [ ] Clean data (remove duplicates, handle missing values)
- [ ] Standardize chemical structures (RDKit)
- [ ] Generate molecular descriptors (200-300 features)
- [ ] Generate molecular fingerprints (2048-4096 bits)
- [ ] Feature selection (variance threshold, correlation filter)
- [ ] Train/val/test split (60/20/20, scaffold-based)

### Phase 2: Model Training (Weeks 3-4)
- [ ] Implement Random Forest baseline
- [ ] Implement XGBoost with hyperparameter tuning
- [ ] Implement LightGBM
- [ ] Implement Neural Network (MLP)
- [ ] Implement Ensemble DNN (5 models)
- [ ] (Optional) Train GAN for data augmentation

### Phase 3: Model Evaluation (Weeks 5-6)
- [ ] 5-fold CV × 3 repeats for each model
- [ ] Calculate performance metrics (ROC-AUC, PR-AUC, MCC, etc.)
- [ ] External validation on test set
- [ ] Generate ROC and PR curves
- [ ] Calibration assessment (ECE, Brier score)
- [ ] Applicability domain analysis

### Phase 4: Uncertainty Quantification (Week 7)
- [ ] Implement conformal prediction
- [ ] Implement MC Dropout (for neural networks)
- [ ] Implement ensemble uncertainty
- [ ] Calculate prediction intervals
- [ ] Assess uncertainty quality

### Phase 5: Statistical Comparison (Week 8)
- [ ] Paired t-tests between models
- [ ] Bonferroni correction for multiple comparisons
- [ ] Effect size calculations (Cohen's d)
- [ ] Multi-criteria decision analysis
- [ ] Model ranking and selection

### Phase 6: Deliverables (Week 9-10)
- [ ] Generate performance plots
- [ ] Create statistical comparison tables
- [ ] Write technical report
- [ ] Prepare model cards
- [ ] Package trained models
- [ ] Create prediction pipeline script

---

## 📖 Documentation Files

### EXECUTIVE_SUMMARY.py (23KB)
Complete methodology document including:
- Detailed phase-by-phase plan
- Model architectures and hyperparameters
- Evaluation protocols
- Statistical testing procedures
- Expected results and recommendations
- Implementation checklist

### model_assessment_plan.py (37KB)
Comprehensive Python framework with:
- Feature engineering functions
- Model training classes
- Hyperparameter optimization
- Cross-validation functions
- Uncertainty quantification methods
- Statistical comparison tools
- Visualization generators

### deep_learning_assessment.py (21KB)
Specialized deep learning module:
- Ensemble Deep Neural Networks
- Monte Carlo Dropout implementation
- GAN for data augmentation
- VAE for representation learning
- Epistemic + aleatoric uncertainty decomposition
- Calibration metrics

### retrieve_chembl_safety_data.py (13KB)
Practical data retrieval script:
- ChEMBL REST API interface
- Robust error handling
- Data quality filtering
- Summary statistics generation
- CSV export

---

## 🎓 Key Concepts

### Uncertainty Quantification
- **Epistemic Uncertainty**: Model uncertainty, reducible with more data
- **Aleatoric Uncertainty**: Data noise, irreducible
- **Conformal Prediction**: Provides coverage guarantees (e.g., 95%)

### Applicability Domain
- Identifies predictions outside model's training space
- Uses Mahalanobis distance or Local Outlier Factor
- Flags compounds for manual review

### Calibration
- Ensures predicted probabilities match observed frequencies
- ECE < 0.05: Excellent calibration
- ECE > 0.10: Poor calibration, needs recalibration

### Model Selection Criteria
1. **Performance**: ROC-AUC, PR-AUC, MCC
2. **Calibration**: ECE, Brier score
3. **Robustness**: Low variance across CV folds
4. **Efficiency**: Training/inference time
5. **Interpretability**: Feature importance availability
6. **Uncertainty**: Prediction interval quality

---

## 🔍 References and Resources

### ChEMBL Database
- Website: https://www.ebi.ac.uk/chembl/
- API Documentation: https://www.ebi.ac.uk/chembl/api/data/docs
- REST API: https://www.ebi.ac.uk/chembl/api/data

### Key Papers
1. Conformal Prediction: Vovk et al. (2005) "Algorithmic Learning in a Random World"
2. Uncertainty Quantification: Gal & Ghahramani (2016) "Dropout as a Bayesian Approximation"
3. Safety Pharmacology: Bowes et al. (2012) "Reducing safety-related attrition"

### RDKit (Molecular Descriptors)
- Documentation: https://www.rdkit.org/docs/
- Fingerprints: Morgan, MACCS, Topological

---

## 📞 Support

For questions or issues:
1. Review EXECUTIVE_SUMMARY.py for detailed methodology
2. Check model_assessment_plan.py for implementation examples
3. Examine deep_learning_assessment.py for uncertainty quantification

---

## 📜 License

This framework is provided for research and educational purposes.
ChEMBL data is licensed under CC BY-SA 3.0.

---

## ✅ Summary

This repository provides:
- ✅ Complete data retrieval pipeline from ChEMBL
- ✅ 7 different model architectures (RF, XGBoost, LightGBM, NN, Ensemble, SVM, GAN)
- ✅ Rigorous validation strategy (15-fold repeated CV + external test set)
- ✅ Multiple uncertainty quantification methods (conformal, MC dropout, ensemble)
- ✅ Statistical model comparison with Bonferroni correction
- ✅ Multi-criteria decision analysis for model selection
- ✅ Comprehensive documentation and implementation guides
- ✅ Scientifically rigorous, evidence-based approach

**Timeline**: 10 weeks (part-time) or 5 weeks (full-time)

**Expected Outcome**: High-performance models (ROC-AUC 0.75-0.90) with reliable uncertainty estimates

---

**Ready to start? Run:** `python retrieve_chembl_safety_data.py`
