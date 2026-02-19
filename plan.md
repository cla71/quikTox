# OFFTOXv3 Improvement Plan

## Problem Analysis

### 1. Inactive Compounds Are Not Drug-Like
The current "universal negatives" pool consists of 47 simple molecules: amino acids (glycine, alanine, etc.), sugars (glucose, sucrose), organic acids (citric, lactic), and other small metabolites (urea, glycerol, ethanol). These are structurally trivial compared to real drug candidates. The model is learning to distinguish drug-like molecules from non-drug-like simple molecules rather than learning to distinguish binders from non-binders at safety targets. This is a fundamental data quality problem.

### 2. Hit/Miss Analysis from the Report

**Confusion Matrix (Internal Test Set):**
| Actual \ Predicted | inactive | less_potent | potent |
|---|---|---|---|
| **inactive** | 74 (62.7%) | 30 | 14 |
| **less_potent** | 4 | 57 (47.9%) | 58 |
| **potent** | 2 | 40 | 254 (85.8%) |

Key observations:
- **less_potent class is heavily confused**: Only 47.9% accuracy. 58 of 119 less_potent compounds are predicted as potent. The boundary between less_potent (4.0-5.0 pChEMBL) and potent (≥5.0 pChEMBL) is the primary source of errors.
- **inactive class leaks to less_potent**: 30 inactive compounds predicted as less_potent, likely because the simple-molecule negatives are too easy to separate from drug-like actives.
- **Calibration is poor**: ECE=0.379, MCE=0.944 — the model is severely overconfident.
- **Average conformal prediction set = 2.02 classes**: Nearly every prediction spans 2 of the 3 classes, providing little discrimination.
- **OOD rate = 27.6%**: Over a quarter of test compounds fall outside the applicability domain, partly because the inactive training set is structurally dissimilar from drug-like test compounds.

**Per-Target Failures:**
- CYP1A2: 56.8% accuracy, MCC 0.182 — worst target, heavily confused between less_potent/potent
- Nav1.5: 77.3% accuracy, MCC 0.434 — potent/less_potent confusion
- Cav1.2: 73.5% accuracy, MCC 0.586 — similar pattern
- LXRa, LXRb, MR, PR: 100% accuracy but MCC 0.000 — only predict one class due to data imbalance (nearly all inactive)

### 3. 2-Class System Recommendation

**Yes, switching to a 2-class (binding vs. non-binding) system is recommended.** Rationale:
- The less_potent class (pChEMBL 4.0-5.0) accounts for 48% of all classification errors
- The 10 µM cutoff (pChEMBL = 5.0) is the standard pharmacological relevance threshold
- Regulatory safety screening fundamentally asks "does it bind?" not "how potently does it bind?"
- Eliminates the most confused class boundary, will improve MCC and calibration significantly
- Reduces conformal prediction set sizes (2 classes max instead of 3)
- Binary classification models are better calibrated and more robust with limited data

---

## Implementation Plan

### Step 1: Replace Inactive Compounds with Drug-Like Non-Binders (`build_24target_dataset.py`)

Replace the `UNIVERSAL_NEGATIVES` pool of 47 simple molecules with ~60 well-known, drug-like compounds that have established pharmacological profiles but are confirmed inactive at the relevant safety targets. Categories:

- **Antibiotics** (structurally complex, no safety-target activity): Amoxicillin, Cephalexin, Metronidazole, Ciprofloxacin, Azithromycin, Doxycycline, Trimethoprim, Nitrofurantoin, Linezolid, Meropenem
- **Antivirals**: Acyclovir, Oseltamivir, Tenofovir, Sofosbuvir, Remdesivir
- **NSAIDs/Analgesics** (known selectivity, well-characterized profiles): Acetaminophen, Naproxen, Celecoxib, Meloxicam, Aspirin
- **Antidiabetics** (non-CYP, non-ion channel): Metformin, Sitagliptin, Empagliflozin, Pioglitazone
- **GI/Respiratory** (target-specific, no safety-target activity): Omeprazole, Montelukast, Loratadine, Ranitidine, Loperamide
- **Vitamins/Well-characterized** (with drug-like properties): Riboflavin, Pyridoxine, Thiamine, Biotin
- **Miscellaneous approved drugs** (well-characterized safety profiles): Methotrexate, Allopurinol, Levetiracetam, Gabapentin, Topiramate, Sumatriptan, Finasteride, Tamsulosin, Sildenafil, Hydroxychloroquine

This ensures the model learns structural patterns of binding vs. non-binding rather than "drug-like vs. sugar."

Note: Some of these drugs DO bind to certain targets (e.g., loperamide binds hERG). The per-target negative assignment logic (`get_negatives_for_target`) already handles this by cycling the pool — but we will add explicit exclusion logic for known cross-reactivity.

### Step 2: Convert to 2-Class System (`run_pipeline.py`)

**Classification:**
- Class 0 (**non-binding**): pChEMBL < 5.0 (≥ 10 µM) or confirmed inactive
- Class 1 (**binding**): pChEMBL ≥ 5.0 (< 10 µM)

**Changes in `run_pipeline.py`:**
- Update `ACTIVITY_CLASS_MAP` → `{0: "non_binding", 1: "binding"}`
- Update `NUM_CLASSES` → 2
- Update `CLASS_COLORS` → 2 colors
- Update `load_and_clean_data()`: Map activity_class 0 and 1 → 0 (non-binding), 2 → 1 (binding)
- Update model definitions: XGBoost `objective` → `binary:logistic`, remove `num_class`; LightGBM `objective` → `binary`
- Update evaluation: Use binary ROC-AUC (simpler), binary PR-AUC, binary MCC
- Update all visualization code: 2 subplots instead of 3 for ROC/PR/calibration
- Update confusion matrix: 2x2
- Update report generation text/tables

### Step 3: Update `build_24target_dataset.py` for 2-Class Labels

- In `make_row()`: Update activity class assignment logic for 2 classes
- Update `activity_class_label` mapping: `{0: "non_binding", 1: "binding"}`
- Merge the old "less_potent" curated compounds into "non_binding" (class 0) since pChEMBL 4.0-5.0 is now non-binding

### Step 4: Update `generate_report.py`

- Update HTML report template for 2 classes
- Update Plotly visualizations for binary classification
- Update interpretation text

### Step 5: Validate and Test

- Run the full pipeline end-to-end
- Compare metrics with previous 3-class results
- Verify improved calibration and reduced conformal set sizes

---

## Expected Outcomes

| Metric | Current (3-class) | Expected (2-class + better inactives) |
|--------|-------------------|---------------------------------------|
| ROC-AUC (internal) | 0.875 | >0.90 |
| MCC | 0.520 | >0.70 |
| ECE | 0.379 | <0.10 |
| Avg conformal set | 2.02 | <1.5 |
| OOD rate | 27.6% | <15% |
| CYP1A2 accuracy | 56.8% | >75% |

The 2-class system removes the most confused boundary and the drug-like inactives ensure the applicability domain covers real drug candidates.
