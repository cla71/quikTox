# Research Project Summary: Safety Target Bioactivity Workflow

## Overview
This run follows the EXECUTIVE_SUMMARY.txt workflow using `/safety_targets_bioactivity.csv` as both the training and test source. The workflow includes data cleaning, scaffold splitting, feature engineering (descriptors + Morgan fingerprints), model training, evaluation, calibration checks, and uncertainty quantification. Outputs are stored in `OFFTOXv2/workflow_outputs/`. 

## Data Preparation
- **Input dataset**: 2,111 compound-target records across 6 safety targets after filtering for exact measurements, valid pChEMBL values (≥4), and canonical SMILES.
- **Activity labeling**: Compounds labeled active at pChEMBL ≥ 6.
- **Split strategy**: Scaffold-based 60/20/20 split into train/validation/test sets.

## Feature Engineering
- **Descriptors**: Basic physicochemical descriptors (MW, LogP, HBA, HBD, TPSA, rotatable bonds, aromatic rings, heavy atoms, FractionCSP3, MolMR).
- **Fingerprints**: Morgan fingerprints (radius 2, 2048 bits).
- **Target encoding**: One-hot encoding of the target name, combined with descriptors and fingerprints.
- **Feature selection**: Variance threshold filtering (0.01).

## Model Training and Validation
- **Models evaluated**: Random Forest, XGBoost, and LightGBM.
- **Hyperparameter search**: RandomizedSearchCV (5 iterations, 3-fold CV).
- **Cross-validation**: Repeated Stratified K-Fold (3 folds × 2 repeats) to estimate stability.

## Evaluation Results
- **Cross-validation ROC-AUC (mean ± SD)**:
  - XGBoost: 0.861 ± 0.018
  - LightGBM: 0.857 ± 0.015
  - Random Forest: 0.853 ± 0.017
- **Best model (CV)**: XGBoost.
- **Test set performance (best model)**:
  - ROC-AUC: 0.810
  - PR-AUC: 0.692
  - MCC: 0.481
- **Calibration**:
  - ECE: 0.047 (good calibration)
  - MCE: 0.174

## Uncertainty Quantification
- **Conformal prediction** (95% nominal coverage):
  - Empirical coverage: 0.950
  - Average prediction set size: 1.57

## Applicability Domain
- **k-NN distance** (5 neighbors):
  - ~14.9% of test compounds flagged as outside the applicability domain.

## Statistical Comparison & Model Selection
- Paired t-tests were computed using fold-level ROC-AUC scores for each model pair.
- Multi-criteria decision analysis (MCDA) used weighted criteria (ROC-AUC, PR-AUC, calibration, robustness, efficiency, interpretability) to rank models.

## Overall Assessment
- The workflow successfully executed the primary steps of the EXECUTIVE_SUMMARY plan using the provided dataset.
- The top-performing model (XGBoost) met the primary discrimination and calibration targets (ROC-AUC > 0.75, ECE < 0.10).
- Calibration and conformal prediction results indicate reliable uncertainty estimates for this dataset.
- The current run uses a reduced CV configuration (3 folds × 2 repeats) and a trimmed model list for runtime efficiency; a full 5×3 CV and additional model types (MLP, SVM, DNN ensembles) would further align with the complete plan.

## Generated Artifacts
Key outputs in `OFFTOXv2/workflow_outputs/`:
- `cv_summary.csv`, `test_metrics.csv`, `mcda_ranking.csv`
- `workflow_summary.json`, `conformal_summary.json`, `statistical_comparison.csv`
- `conformal_set_sizes.csv`
