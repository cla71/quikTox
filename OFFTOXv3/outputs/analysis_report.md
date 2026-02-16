# OFFTOXv3 Analysis Report

**Generated:** 2026-02-16 17:34
**Best Model:** XGBoost
**Targets:** 24 safety pharmacology targets

## 1. Dataset Summary

- **Total training compounds:** 2665
- **Unique targets:** 24
- **Train/Val/Test split:** 1599/533/533 (scaffold-based)
- **Feature dimensions:** 863

### Class Distribution

| Class | Label | Count | Percentage |
|-------|-------|------:|----------:|
| 0 | inactive | 696 | 26.1% |
| 1 | less_potent | 596 | 22.4% |
| 2 | potent | 1373 | 51.5% |

### Per-Target Compound Counts

| Target | Category | Total | Potent | Less Potent | Inactive |
|--------|----------|------:|-------:|------------:|---------:|
| AR | Nuclear Hormone Receptor | 34 | 8 | 2 | 24 |
| BSEP | Transporter | 36 | 7 | 2 | 27 |
| CAR | Nuclear Hormone Receptor | 38 | 7 | 2 | 29 |
| CYP1A2 | Hepatotoxicity | 279 | 134 | 106 | 39 |
| CYP2C19 | Hepatotoxicity | 34 | 6 | 2 | 26 |
| CYP2C9 | Hepatotoxicity | 35 | 7 | 1 | 27 |
| CYP2D6 | Hepatotoxicity | 443 | 272 | 135 | 36 |
| CYP3A4 | Hepatotoxicity | 378 | 222 | 122 | 34 |
| Cav1.2 | Cardiac Safety | 205 | 123 | 50 | 32 |
| ER_beta | Nuclear Hormone Receptor | 38 | 9 | 3 | 26 |
| ERa | Nuclear Hormone Receptor | 34 | 7 | 3 | 24 |
| FXR | Nuclear Hormone Receptor | 34 | 6 | 2 | 26 |
| GR | Nuclear Hormone Receptor | 36 | 5 | 3 | 28 |
| LXRa | Nuclear Hormone Receptor | 33 | 7 | 2 | 24 |
| LXRb | Nuclear Hormone Receptor | 38 | 6 | 2 | 30 |
| MR | Nuclear Hormone Receptor | 40 | 9 | 3 | 28 |
| Nav1.5 | Cardiac Safety | 350 | 246 | 70 | 34 |
| P-gp | Transporter | 42 | 8 | 3 | 31 |
| PPARg | Nuclear Hormone Receptor | 35 | 5 | 1 | 29 |
| PR | Nuclear Hormone Receptor | 36 | 8 | 2 | 26 |
| PXR | Nuclear Hormone Receptor | 38 | 7 | 1 | 30 |
| RXRa | Nuclear Hormone Receptor | 35 | 4 | 1 | 30 |
| VDR | Nuclear Hormone Receptor | 33 | 6 | 1 | 26 |
| hERG | Cardiac Safety | 361 | 254 | 77 | 30 |

## 2. Cross-Validation Results

| Model | ROC-AUC | PR-AUC | MCC |
|-------|--------:|-------:|----:|
| XGBoost | 0.9306 +/- 0.0084 | 0.8304 +/- 0.0168 | 0.7302 +/- 0.0168 |
| LightGBM | 0.9293 +/- 0.0096 | 0.8325 +/- 0.0171 | 0.7229 +/- 0.0217 |
| RandomForest | 0.9285 +/- 0.0069 | 0.8295 +/- 0.0157 | 0.7229 +/- 0.0190 |

**Selected model:** XGBoost (highest ROC-AUC)

## 3. Internal Test Set Performance (Scaffold Split)

| Metric | Value |
|--------|------:|
| ROC-AUC (macro) | 0.8748 |
| PR-AUC (macro) | 0.7481 |
| MCC | 0.5204 |
| ECE (calibrated) | 0.3792 |
| MCE (calibrated) | 0.9440 |

### Confusion Matrix

| | Pred: inactive | Pred: less_potent | Pred: potent |
|---|---:|---:|---:|
| **inactive** | 74 | 30 | 14 |
| **less_potent** | 4 | 57 | 58 |
| **potent** | 2 | 40 | 254 |

## 4. Uncertainty Quantification

- **Conformal coverage:** 0.9418 (target: 0.95)
- **Average prediction set size:** 2.02
- **AD threshold (95th pct k-NN):** 16.1152
- **Out-of-domain rate:** 27.58%

## 5. Held-Out Test Set Evaluation

- **Test compounds:** 452
- **ROC-AUC (macro):** 0.9149
- **MCC:** 0.6984
- **Accuracy:** 0.8119

### Per-Target Test Performance

| Target | N | Accuracy | MCC |
|--------|--:|---------:|----:|
| AR | 10 | 0.900 | 0.728 |
| BSEP | 7 | 0.857 | 0.715 |
| CAR | 4 | 1.000 | 1.000 |
| CYP1A2 | 37 | 0.568 | 0.182 |
| CYP2C19 | 9 | 1.000 | 1.000 |
| CYP2C9 | 9 | 0.667 | 0.447 |
| CYP2D6 | 63 | 0.778 | 0.588 |
| CYP3A4 | 62 | 0.742 | 0.563 |
| Cav1.2 | 34 | 0.735 | 0.586 |
| ER_beta | 7 | 1.000 | 1.000 |
| ERa | 12 | 0.833 | 0.726 |
| FXR | 7 | 0.857 | 0.500 |
| GR | 7 | 0.857 | 0.772 |
| LXRa | 8 | 1.000 | 0.000 |
| LXRb | 2 | 1.000 | 0.000 |
| MR | 4 | 1.000 | 0.000 |
| Nav1.5 | 66 | 0.773 | 0.434 |
| P-gp | 2 | 1.000 | 1.000 |
| PPARg | 7 | 0.857 | 0.783 |
| PR | 6 | 1.000 | 0.000 |
| PXR | 3 | 1.000 | 1.000 |
| RXRa | 5 | 0.800 | 0.722 |
| VDR | 8 | 1.000 | 1.000 |
| hERG | 73 | 0.945 | 0.876 |

## 6. Statistical Model Comparison

Bonferroni-corrected alpha = 0.0167

| Model A | Model B | t-stat | p-value | Cohen's d | Significant |
|---------|---------|-------:|--------:|----------:|:-----------:|
| XGBoost | LightGBM | 0.7783 | 0.471601 | 0.1477 | No |
| XGBoost | RandomForest | 0.7118 | 0.508396 | 0.2709 | No |
| LightGBM | RandomForest | 0.2270 | 0.829396 | 0.0920 | No |

## 7. MCDA Ranking

| Rank | Model | Composite Score |
|-----:|-------|----------------:|
| 1 | XGBoost | 0.6862 |
| 2 | LightGBM | 0.5912 |
| 3 | RandomForest | 0.3500 |

## 8. Target Panel Reference

| # | Target | ChEMBL ID | Category |
|--:|--------|-----------|----------|
| 1 | ERa | CHEMBL206 | Nuclear Hormone Receptor |
| 2 | ER_beta | CHEMBL242 | Nuclear Hormone Receptor |
| 3 | AR | CHEMBL1871 | Nuclear Hormone Receptor |
| 4 | GR | CHEMBL2034 | Nuclear Hormone Receptor |
| 5 | PR | CHEMBL208 | Nuclear Hormone Receptor |
| 6 | MR | CHEMBL1994 | Nuclear Hormone Receptor |
| 7 | PPARg | CHEMBL235 | Nuclear Hormone Receptor |
| 8 | PXR | CHEMBL3401 | Nuclear Hormone Receptor |
| 9 | CAR | CHEMBL2248 | Nuclear Hormone Receptor |
| 10 | LXRa | CHEMBL5231 | Nuclear Hormone Receptor |
| 11 | LXRb | CHEMBL4309 | Nuclear Hormone Receptor |
| 12 | FXR | CHEMBL2001 | Nuclear Hormone Receptor |
| 13 | RXRa | CHEMBL2061 | Nuclear Hormone Receptor |
| 14 | VDR | CHEMBL1977 | Nuclear Hormone Receptor |
| 15 | hERG | CHEMBL240 | Cardiac Safety |
| 16 | Cav1.2 | CHEMBL1940 | Cardiac Safety |
| 17 | Nav1.5 | CHEMBL1993 | Cardiac Safety |
| 18 | CYP3A4 | CHEMBL340 | Hepatotoxicity |
| 19 | CYP2D6 | CHEMBL289 | Hepatotoxicity |
| 20 | CYP2C9 | CHEMBL3397 | Hepatotoxicity |
| 21 | CYP1A2 | CHEMBL3356 | Hepatotoxicity |
| 22 | CYP2C19 | CHEMBL3622 | Hepatotoxicity |
| 23 | P-gp | CHEMBL4302 | Transporter |
| 24 | BSEP | CHEMBL4105 | Transporter |

## 9. Output Files

| File | Description |
|------|-------------|
| `01_data_exploration.png` | Class distribution, per-target breakdown, pChEMBL histogram |
| `02_roc_curves.png` | Per-class ROC curves |
| `03_pr_curves.png` | Per-class Precision-Recall curves |
| `04_confusion_matrix.png` | Confusion matrices (counts and percentages) |
| `05_calibration_curves.png` | Per-class calibration curves |
| `06_feature_importance.png` | Top 20 feature importances |
| `07_uncertainty.png` | Conformal sets, AD distances, reliability diagram |
| `08_test_set_results.png` | Held-out test set results |
| `workflow_summary.json` | Machine-readable summary |
| `test_set_predictions.csv` | Test set predictions with probabilities |
| `analysis_report.md` | This report |
