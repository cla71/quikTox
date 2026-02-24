# OFFTOXv3 Analysis Report

**Generated:** 2026-02-22 23:28
**Best Model:** LightGBM
**Targets:** 24 safety pharmacology targets
**Classification:** Binary (binding vs non-binding at 10 uM threshold)

## 1. Dataset Summary

- **Total training compounds:** 19696
- **Unique targets:** 23
- **Train/Val/Test split:** 11817/3939/3940 (scaffold-based)
- **Feature dimensions:** 1039 (10 descriptors + 2048 Morgan FP + target encoding)

### Class Distribution

| Class | Label | Count | Percentage |
|-------|-------|------:|----------:|
| 0 | non_binding | 3144 | 16.0% |
| 1 | binding | 16552 | 84.0% |

### Per-Target Compound Counts

| Target | Category | Total | Binding | Non-Binding |
|--------|----------|------:|--------:|------------:|
| AR | Nuclear Hormone Receptor | 1277 | 1226 | 51 |
| BSEP | Transporter | 141 | 90 | 51 |
| CAR | Nuclear Hormone Receptor | 3 | 0 | 3 |
| CYP1A2 | Hepatotoxicity | 996 | 630 | 366 |
| CYP2C19 | Hepatotoxicity | 1013 | 607 | 406 |
| CYP2C9 | Hepatotoxicity | 1096 | 686 | 410 |
| CYP2D6 | Hepatotoxicity | 1082 | 737 | 345 |
| CYP3A4 | Hepatotoxicity | 1059 | 689 | 370 |
| Cav1.2 | Cardiac Safety | 223 | 149 | 74 |
| ER_beta | Nuclear Hormone Receptor | 1199 | 1128 | 71 |
| ERa | Nuclear Hormone Receptor | 1330 | 1249 | 81 |
| FXR | Nuclear Hormone Receptor | 1108 | 1046 | 62 |
| GR | Nuclear Hormone Receptor | 1295 | 1281 | 14 |
| LXRb | Nuclear Hormone Receptor | 34 | 17 | 17 |
| MR | Nuclear Hormone Receptor | 940 | 930 | 10 |
| Nav1.5 | Cardiac Safety | 425 | 328 | 97 |
| P-gp | Transporter | 769 | 569 | 200 |
| PPARg | Nuclear Hormone Receptor | 1352 | 1284 | 68 |
| PR | Nuclear Hormone Receptor | 1208 | 1183 | 25 |
| PXR | Nuclear Hormone Receptor | 208 | 193 | 15 |
| RXRa | Nuclear Hormone Receptor | 911 | 888 | 23 |
| VDR | Nuclear Hormone Receptor | 383 | 312 | 71 |
| hERG | Cardiac Safety | 1644 | 1330 | 314 |

## 2. Cross-Validation Results

| Model | ROC-AUC | PR-AUC | MCC |
|-------|--------:|-------:|----:|
| LightGBM | 0.9237 +/- 0.0035 | 0.9839 +/- 0.0009 | 0.5847 +/- 0.0151 |
| XGBoost | 0.9242 +/- 0.0052 | 0.9841 +/- 0.0012 | 0.5745 +/- 0.0215 |
| RandomForest | 0.9148 +/- 0.0035 | 0.9819 +/- 0.0007 | 0.5401 +/- 0.0168 |
| GNN | 0.8791 +/- 0.0022 | 0.9733 +/- 0.0004 | 0.4790 +/- 0.0058 |

**Selected model:** LightGBM (highest ROC-AUC)

## 3. Internal Test Set Performance (Scaffold Split)

| Metric | Value |
|--------|------:|
| ROC-AUC | 0.8895 |
| PR-AUC | 0.9721 |
| MCC | 0.4858 |
| ECE (calibrated) | 0.2008 |
| MCE (calibrated) | 0.9471 |

### Confusion Matrix

| | Pred: non_binding | Pred: binding |
|---|---:|---:|
| **non_binding** | 322 | 393 |
| **binding** | 142 | 3083 |

## 4. Uncertainty Quantification

- **Conformal coverage:** 0.9500 (target: 0.95)
- **Average prediction set size:** 1.25
- **AD threshold (95th pct k-NN):** 10.3700
- **Out-of-domain rate:** 35.20%

## 5. Held-Out Test Set Evaluation

- **Test compounds:** 452
- **ROC-AUC:** 0.9341
- **MCC:** 0.7519
- **Accuracy:** 0.8739

### Per-Target Test Performance

| Target | N | Accuracy | MCC |
|--------|--:|---------:|----:|
| AR | 10 | 0.400 | 0.218 |
| BSEP | 7 | 1.000 | 1.000 |
| CAR | 4 | 0.750 | 0.000 |
| CYP1A2 | 37 | 0.946 | 0.892 |
| CYP2C19 | 9 | 0.778 | 0.357 |
| CYP2C9 | 9 | 0.667 | -0.189 |
| CYP2D6 | 63 | 0.841 | 0.680 |
| CYP3A4 | 62 | 0.871 | 0.744 |
| Cav1.2 | 34 | 0.971 | 0.942 |
| ER_beta | 7 | 0.714 | 0.471 |
| ERa | 12 | 0.917 | 0.816 |
| FXR | 7 | 0.714 | -0.167 |
| GR | 7 | 0.429 | 0.000 |
| LXRa | 8 | 1.000 | 0.000 |
| LXRb | 2 | 1.000 | 0.000 |
| MR | 4 | 0.000 | 0.000 |
| Nav1.5 | 66 | 0.939 | 0.831 |
| P-gp | 2 | 1.000 | 1.000 |
| PPARg | 7 | 0.714 | 0.417 |
| PR | 6 | 0.833 | 0.000 |
| PXR | 3 | 0.667 | 0.000 |
| RXRa | 5 | 0.800 | 0.667 |
| VDR | 8 | 1.000 | 1.000 |
| hERG | 73 | 0.973 | 0.929 |

## 6. Statistical Model Comparison

Bonferroni-corrected alpha = 0.0083

| Model A | Model B | t-stat | p-value | Cohen's d | Significant |
|---------|---------|-------:|--------:|----------:|:-----------:|
| LightGBM | XGBoost | -0.5743 | 0.590666 | -0.1063 | No |
| LightGBM | RandomForest | 6.0907 | 0.001726 | 1.5680 | Yes |
| LightGBM | GNN | 38.0157 | 0.000000 | 1.9828 | Yes |
| XGBoost | RandomForest | 5.2987 | 0.003196 | 1.4506 | Yes |
| XGBoost | GNN | 26.7896 | 0.000001 | 1.9692 | Yes |
| RandomForest | GNN | 45.5566 | 0.000000 | 1.9732 | Yes |

## 7. MCDA Ranking

| Rank | Model | Composite Score |
|-----:|-------|----------------:|
| 1 | LightGBM | 0.9528 |
| 2 | XGBoost | 0.8269 |
| 3 | RandomForest | 0.6412 |
| 4 | GNN | 0.1000 |

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
| `workflow_summary.json` | Machine-readable summary of all metrics |
| `test_set_predictions.csv` | Held-out test set predictions with probabilities |
| `analysis_report.md` | This report |
