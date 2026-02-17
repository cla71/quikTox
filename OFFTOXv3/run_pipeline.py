#!/usr/bin/env python3
"""
Run the OFFTOXv3 24-target safety pharmacology pipeline.

Executes the same workflow as the notebook but as a standalone script.
Produces all outputs (PNGs, CSVs, JSON, analysis_report.md) in outputs/.
"""

import json
import csv
import time
import warnings
import pickle
import hashlib
import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, MolSurf, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    classification_report,
)
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.1)

# ── Paths ────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "data" / "safety_targets_bioactivity.csv"
TEST_PATH = SCRIPT_DIR / "data" / "test_compounds.csv"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
MODEL_DIR  = SCRIPT_DIR / "model"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# ── 24-Target Safety Panel ───────────────────────────────────────────
TARGET_PANEL = {
    "ERa":     {"chembl_id": "CHEMBL206",  "category": "Nuclear Hormone Receptor"},
    "ER_beta": {"chembl_id": "CHEMBL242",  "category": "Nuclear Hormone Receptor"},
    "AR":      {"chembl_id": "CHEMBL1871", "category": "Nuclear Hormone Receptor"},
    "GR":      {"chembl_id": "CHEMBL2034", "category": "Nuclear Hormone Receptor"},
    "PR":      {"chembl_id": "CHEMBL208",  "category": "Nuclear Hormone Receptor"},
    "MR":      {"chembl_id": "CHEMBL1994", "category": "Nuclear Hormone Receptor"},
    "PPARg":   {"chembl_id": "CHEMBL235",  "category": "Nuclear Hormone Receptor"},
    "PXR":     {"chembl_id": "CHEMBL3401", "category": "Nuclear Hormone Receptor"},
    "CAR":     {"chembl_id": "CHEMBL2248", "category": "Nuclear Hormone Receptor"},
    "LXRa":    {"chembl_id": "CHEMBL5231", "category": "Nuclear Hormone Receptor"},
    "LXRb":    {"chembl_id": "CHEMBL4309", "category": "Nuclear Hormone Receptor"},
    "FXR":     {"chembl_id": "CHEMBL2001", "category": "Nuclear Hormone Receptor"},
    "RXRa":    {"chembl_id": "CHEMBL2061", "category": "Nuclear Hormone Receptor"},
    "VDR":     {"chembl_id": "CHEMBL1977", "category": "Nuclear Hormone Receptor"},
    "hERG":    {"chembl_id": "CHEMBL240",  "category": "Cardiac Safety"},
    "Cav1.2":  {"chembl_id": "CHEMBL1940", "category": "Cardiac Safety"},
    "Nav1.5":  {"chembl_id": "CHEMBL1993", "category": "Cardiac Safety"},
    "CYP3A4":  {"chembl_id": "CHEMBL340",  "category": "Hepatotoxicity"},
    "CYP2D6":  {"chembl_id": "CHEMBL289",  "category": "Hepatotoxicity"},
    "CYP2C9":  {"chembl_id": "CHEMBL3397", "category": "Hepatotoxicity"},
    "CYP1A2":  {"chembl_id": "CHEMBL3356", "category": "Hepatotoxicity"},
    "CYP2C19": {"chembl_id": "CHEMBL3622", "category": "Hepatotoxicity"},
    "P-gp":    {"chembl_id": "CHEMBL4302", "category": "Transporter"},
    "BSEP":    {"chembl_id": "CHEMBL4105", "category": "Transporter"},
}

RANDOM_STATE = 42
ACTIVITY_CLASS_MAP = {0: "inactive", 1: "less_potent", 2: "potent"}
CLASS_COLORS = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}
NUM_CLASSES = 3


# ══════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════
@dataclass
class SplitData:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


# ══════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════
def load_and_clean_data(path: Path) -> List[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            smi = row.get("canonical_smiles")
            if not smi:
                continue
            raw_class = row.get("activity_class", "")
            if raw_class == "0" or row.get("activity_class_label") == "inactive":
                row["pchembl_value"] = None
                row["activity_class"] = 0
                rows.append(row)
                continue
            if row.get("standard_relation") != "=":
                continue
            if not row.get("pchembl_value"):
                continue
            try:
                pchembl = float(row["pchembl_value"])
            except ValueError:
                continue
            if pchembl < 4.0:
                continue
            row["pchembl_value"] = pchembl
            row["activity_class"] = 2 if pchembl >= 5.0 else 1
            rows.append(row)

    deduped: dict = {}
    for row in rows:
        key = (row.get("molecule_chembl_id"), row.get("target_chembl_id"))
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = row
        else:
            existing_p = existing.get("pchembl_value")
            current_p = row.get("pchembl_value")
            if current_p is not None and (existing_p is None or current_p > existing_p):
                deduped[key] = row
    return list(deduped.values())


# ══════════════════════════════════════════════════════════════════════
# Feature engineering
# ══════════════════════════════════════════════════════════════════════
def compute_descriptors(smiles: List[str]) -> Tuple[np.ndarray, List[str]]:
    descriptor_functions = {
        "MW": Descriptors.MolWt, "LogP": Crippen.MolLogP,
        "HBA": Lipinski.NumHAcceptors, "HBD": Lipinski.NumHDonors,
        "TPSA": MolSurf.TPSA, "RotatableBonds": Lipinski.NumRotatableBonds,
        "AromaticRings": Lipinski.NumAromaticRings,
        "HeavyAtoms": Lipinski.HeavyAtomCount,
        "FractionCSP3": Lipinski.FractionCSP3, "MolMR": Crippen.MolMR,
    }
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append([np.nan] * len(descriptor_functions))
        else:
            rows.append([func(mol) for func in descriptor_functions.values()])
    return np.array(rows, dtype=float), list(descriptor_functions.keys())


def compute_morgan_fingerprints(smiles: List[str], n_bits: int = 2048) -> np.ndarray:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    fps = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(np.zeros(n_bits, dtype=int))
        else:
            fps.append(np.array(gen.GetFingerprint(mol)))
    return np.array(fps)


def build_feature_matrix(rows, selected_columns=None):
    smiles = [row["canonical_smiles"] for row in rows]
    targets = [row.get("target_common_name", row.get("target", "")) for row in rows]
    labels = np.array([row.get("activity_class", -1) for row in rows], dtype=int)

    descriptors, desc_names = compute_descriptors(smiles)
    fingerprints = compute_morgan_fingerprints(smiles)
    fp_names = [f"FP_{i}" for i in range(fingerprints.shape[1])]

    target_names = sorted({t for t in targets if t})
    target_map = {name: idx for idx, name in enumerate(target_names)}
    target_matrix = np.zeros((len(rows), len(target_names)), dtype=float)
    for idx, target in enumerate(targets):
        if target in target_map:
            target_matrix[idx, target_map[target]] = 1.0

    feature_matrix = np.concatenate([descriptors, fingerprints, target_matrix], axis=1)
    columns = desc_names + fp_names + [f"target_{n}" for n in target_names]

    if selected_columns is None:
        variances = np.nanvar(feature_matrix, axis=0)
        mask = variances > 0.01
        feature_matrix = np.nan_to_num(feature_matrix[:, mask], nan=0.0)
        selected_columns = [col for col, keep in zip(columns, mask) if keep]
    else:
        col_index = {col: idx for idx, col in enumerate(columns)}
        aligned = np.zeros((len(rows), len(selected_columns)), dtype=float)
        for out_idx, col in enumerate(selected_columns):
            if col in col_index:
                aligned[:, out_idx] = np.nan_to_num(
                    feature_matrix[:, col_index[col]], nan=0.0)
        feature_matrix = aligned

    return feature_matrix, labels, selected_columns


# ══════════════════════════════════════════════════════════════════════
# Scaffold split
# ══════════════════════════════════════════════════════════════════════
def scaffold_split(smiles, y, random_state=42):
    scaffolds: Dict[str, List[int]] = {}
    for idx, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        scaffold = "" if mol is None else MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        scaffolds.setdefault(scaffold, []).append(idx)
    scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)
    rng = np.random.default_rng(random_state)
    rng.shuffle(scaffold_sets)
    n_total = len(smiles)
    n_train = int(0.6 * n_total)
    n_val = int(0.2 * n_total)
    train_idx, val_idx, test_idx = [], [], []
    for group in scaffold_sets:
        if len(train_idx) + len(group) <= n_train:
            train_idx.extend(group)
        elif len(val_idx) + len(group) <= n_val:
            val_idx.extend(group)
        else:
            test_idx.extend(group)
    return SplitData(np.array(train_idx), np.array(val_idx), np.array(test_idx))


# ══════════════════════════════════════════════════════════════════════
# Models
# ══════════════════════════════════════════════════════════════════════
def ece_score_fn(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    ece, mce = 0.0, 0.0
    for i in range(n_bins):
        mask = binids == i
        if not np.any(mask):
            continue
        avg_conf = y_prob[mask].mean()
        avg_acc = y_true[mask].mean()
        gap = abs(avg_conf - avg_acc)
        ece += gap * mask.mean()
        mce = max(mce, gap)
    return ece, mce


def get_models(random_state):
    return {
        "RandomForest": (
            Pipeline([("scaler", StandardScaler(with_mean=False)),
                       ("model", RandomForestClassifier(random_state=random_state, n_jobs=-1))]),
            {"model__n_estimators": [200, 500], "model__max_depth": [10, 20, None],
             "model__min_samples_split": [2, 5, 10], "model__max_features": ["sqrt", "log2", 0.3],
             "model__class_weight": ["balanced", None]},
        ),
        "XGBoost": (
            Pipeline([("scaler", StandardScaler(with_mean=False)),
                       ("model", XGBClassifier(random_state=random_state, objective="multi:softprob",
                                               num_class=NUM_CLASSES, eval_metric="mlogloss",
                                               n_jobs=-1, verbosity=0))]),
            {"model__n_estimators": [200, 500], "model__max_depth": [3, 5, 7],
             "model__learning_rate": [0.01, 0.05, 0.1], "model__subsample": [0.6, 0.8, 1.0],
             "model__colsample_bytree": [0.6, 0.8, 1.0]},
        ),
        "LightGBM": (
            Pipeline([("scaler", StandardScaler(with_mean=False)),
                       ("model", LGBMClassifier(random_state=random_state, n_jobs=-1, verbose=-1,
                                                objective="multiclass", num_class=NUM_CLASSES))]),
            {"model__n_estimators": [200, 500], "model__max_depth": [-1, 5, 10],
             "model__learning_rate": [0.01, 0.05, 0.1], "model__num_leaves": [31, 63, 127],
             "model__subsample": [0.6, 0.8, 1.0]},
        ),
    }


# ══════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("OFFTOXv3 — 24-Target Safety Pharmacology Pipeline")
    print("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────
    print("\n[1/12] Loading data...")
    data = load_and_clean_data(DATA_PATH)
    labels_all = np.array([row["activity_class"] for row in data], dtype=int)
    targets_all = [row.get("target_common_name", "unknown") for row in data]
    print(f"  Loaded {len(data)} records, {len(set(targets_all))} targets")
    for cls in sorted(ACTIVITY_CLASS_MAP):
        n = int((labels_all == cls).sum())
        print(f"  {cls} ({ACTIVITY_CLASS_MAP[cls]:>12s}): {n:>5d}  ({100*n/len(data):.1f}%)")

    # ── 2. Exploratory viz ────────────────────────────────────────────
    print("\n[2/12] Generating data exploration plot...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    class_counts = Counter(labels_all)
    bars = axes[0].bar(
        [ACTIVITY_CLASS_MAP[c] for c in sorted(class_counts)],
        [class_counts[c] for c in sorted(class_counts)],
        color=[CLASS_COLORS[c] for c in sorted(class_counts)], edgecolor="black")
    for bar, c in zip(bars, sorted(class_counts)):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height()+10,
                     str(class_counts[c]), ha="center", fontweight="bold")
    axes[0].set_title("Class Distribution"); axes[0].set_ylabel("Count")

    target_class_df = pd.DataFrame({"target": targets_all, "class": labels_all})
    target_order = sorted(set(targets_all))
    class_by_target = target_class_df.groupby(["target", "class"]).size().unstack(fill_value=0)
    class_by_target = class_by_target.reindex(columns=[0, 1, 2], fill_value=0)
    class_by_target.columns = [ACTIVITY_CLASS_MAP[c] for c in class_by_target.columns]
    class_by_target.loc[target_order].plot.barh(
        stacked=True, ax=axes[1],
        color=[CLASS_COLORS[0], CLASS_COLORS[1], CLASS_COLORS[2]], edgecolor="black")
    axes[1].set_title("Compounds per Target"); axes[1].set_xlabel("Count")
    axes[1].legend(title="Class", loc="lower right")

    pchembl_vals = [float(row["pchembl_value"]) for row in data if row["pchembl_value"] is not None]
    axes[2].hist(pchembl_vals, bins=30, color="#3498db", edgecolor="black", alpha=0.8)
    axes[2].axvline(5.0, color="red", ls="--", lw=2, label="Potent (5.0)")
    axes[2].axvline(4.0, color="orange", ls="--", lw=2, label="Less-potent (4.0)")
    axes[2].set_title("pChEMBL Value Distribution"); axes[2].set_xlabel("pChEMBL")
    axes[2].set_ylabel("Count"); axes[2].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_data_exploration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 01_data_exploration.png")

    # ── 3. Features ───────────────────────────────────────────────────
    print("\n[3/12] Computing features...")
    t0 = time.time()
    features, labels, selected_columns = build_feature_matrix(data)
    print(f"  Done in {time.time()-t0:.1f}s — shape: {features.shape}")

    # ── 4. Scaffold split ─────────────────────────────────────────────
    print("\n[4/12] Scaffold split...")
    split = scaffold_split([row["canonical_smiles"] for row in data], labels, RANDOM_STATE)
    X_train, y_train = features[split.train_idx], labels[split.train_idx]
    X_val, y_val = features[split.val_idx], labels[split.val_idx]
    X_test, y_test = features[split.test_idx], labels[split.test_idx]
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ── 5. Train models ──────────────────────────────────────────────
    print("\n[5/12] Training models...")
    models = get_models(RANDOM_STATE)
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=RANDOM_STATE)
    cv_summary = []
    best_estimators = {}
    calibration_metrics = {}
    fold_scores: Dict[str, List[float]] = {}
    train_times = {}

    for name, (pipeline, param_grid) in models.items():
        print(f"  Training {name}...", end=" ", flush=True)
        search = RandomizedSearchCV(
            pipeline, param_distributions=param_grid, n_iter=5,
            scoring="roc_auc_ovr", cv=3, random_state=RANDOM_STATE, n_jobs=1)
        t0 = time.time()
        search.fit(X_train, y_train)
        train_times[name] = time.time() - t0
        best_estimators[name] = search.best_estimator_

        scores, pr_scores, mcc_scores = [], [], []
        for train_idx, test_idx in cv.split(X_train, y_train):
            X_tr, X_te = X_train[train_idx], X_train[test_idx]
            y_tr, y_te = y_train[train_idx], y_train[test_idx]
            est = search.best_estimator_
            est.fit(X_tr, y_tr)
            probs = est.predict_proba(X_te)
            preds = est.predict(X_te)
            scores.append(roc_auc_score(y_te, probs, multi_class="ovr", average="macro"))
            pr_per = [average_precision_score((y_te == c).astype(int), probs[:, c])
                      for c in range(NUM_CLASSES) if (y_te == c).sum() > 0]
            pr_scores.append(float(np.mean(pr_per)) if pr_per else 0.0)
            mcc_scores.append(matthews_corrcoef(y_te, preds))

        cv_summary.append({
            "model": name,
            "roc_auc_mean": np.mean(scores), "roc_auc_std": np.std(scores),
            "pr_auc_mean": np.mean(pr_scores), "pr_auc_std": np.std(pr_scores),
            "mcc_mean": np.mean(mcc_scores), "mcc_std": np.std(mcc_scores),
        })
        fold_scores[name] = scores

        val_probs = search.best_estimator_.predict_proba(X_val)
        val_probs_true = val_probs[np.arange(len(y_val)), y_val]
        ece_val, _ = ece_score_fn(np.ones(len(y_val)), val_probs_true)
        calibration_metrics[name] = ece_val

        print(f"ROC-AUC={np.mean(scores):.4f} ({train_times[name]:.0f}s)")

    # ── 6. Best model & test eval ─────────────────────────────────────
    print("\n[6/12] Evaluating best model...")
    cv_summary_sorted = sorted(cv_summary, key=lambda r: r["roc_auc_mean"], reverse=True)
    best_model_name = cv_summary_sorted[0]["model"]
    best_model = best_estimators[best_model_name]
    best_model.fit(np.vstack([X_train, X_val]), np.hstack([y_train, y_val]))
    print(f"  Best model: {best_model_name}")

    test_probs = best_model.predict_proba(X_test)
    test_preds = best_model.predict(X_test)
    test_roc = roc_auc_score(y_test, test_probs, multi_class="ovr", average="macro")
    pr_per = [average_precision_score((y_test == c).astype(int), test_probs[:, c])
              for c in range(NUM_CLASSES) if (y_test == c).sum() > 0]
    test_pr = float(np.mean(pr_per)) if pr_per else 0.0
    test_mcc = matthews_corrcoef(y_test, test_preds)

    calibrated = CalibratedClassifierCV(best_model, method="isotonic", cv=3)
    calibrated.fit(np.vstack([X_train, X_val]), np.hstack([y_train, y_val]))
    cal_probs = calibrated.predict_proba(X_test)
    cal_probs_true = cal_probs[np.arange(len(y_test)), y_test]
    ece, mce = ece_score_fn(np.ones(len(y_test)), cal_probs_true)

    print(f"  ROC-AUC={test_roc:.4f}, PR-AUC={test_pr:.4f}, MCC={test_mcc:.4f}")
    print(f"  ECE={ece:.4f}, MCE={mce:.4f}")

    # ── 7. Visualization plots ────────────────────────────────────────
    print("\n[7/12] Generating evaluation plots...")

    # ROC curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for cls in range(NUM_CLASSES):
        label = ACTIVITY_CLASS_MAP[cls]
        binary_true = (y_test == cls).astype(int)
        if binary_true.sum() == 0:
            axes[cls].set_title(f"ROC - {label} (no samples)"); continue
        fpr, tpr, _ = roc_curve(binary_true, test_probs[:, cls])
        auc_val = roc_auc_score(binary_true, test_probs[:, cls])
        axes[cls].plot(fpr, tpr, color=CLASS_COLORS[cls], lw=2, label=f"AUC = {auc_val:.3f}")
        axes[cls].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        axes[cls].set_xlabel("FPR"); axes[cls].set_ylabel("TPR")
        axes[cls].set_title(f"ROC - {label}"); axes[cls].legend(loc="lower right")
    fig.suptitle(f"Per-Class ROC ({best_model_name})", fontsize=14, y=1.02)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "02_roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # PR curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for cls in range(NUM_CLASSES):
        label = ACTIVITY_CLASS_MAP[cls]
        binary_true = (y_test == cls).astype(int)
        if binary_true.sum() == 0:
            axes[cls].set_title(f"PR - {label} (no samples)"); continue
        prec, rec, _ = precision_recall_curve(binary_true, test_probs[:, cls])
        ap = average_precision_score(binary_true, test_probs[:, cls])
        axes[cls].plot(rec, prec, color=CLASS_COLORS[cls], lw=2, label=f"AP = {ap:.3f}")
        baseline = binary_true.mean()
        axes[cls].axhline(baseline, color="gray", ls="--", lw=1, alpha=0.5, label=f"Baseline = {baseline:.3f}")
        axes[cls].set_xlabel("Recall"); axes[cls].set_ylabel("Precision")
        axes[cls].set_title(f"PR - {label}"); axes[cls].legend(loc="upper right")
    fig.suptitle(f"Per-Class PR ({best_model_name})", fontsize=14, y=1.02)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "03_pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Confusion matrix
    cm = confusion_matrix(y_test, test_preds, labels=list(range(NUM_CLASSES)))
    cm_pct = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=[ACTIVITY_CLASS_MAP[c] for c in range(NUM_CLASSES)],
                yticklabels=[ACTIVITY_CLASS_MAP[c] for c in range(NUM_CLASSES)])
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual"); axes[0].set_title("Counts")
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues", ax=axes[1],
                xticklabels=[ACTIVITY_CLASS_MAP[c] for c in range(NUM_CLASSES)],
                yticklabels=[ACTIVITY_CLASS_MAP[c] for c in range(NUM_CLASSES)])
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual"); axes[1].set_title("% per row")
    fig.suptitle(f"Confusion Matrix ({best_model_name})", fontsize=14, y=1.02)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "04_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Calibration
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for cls in range(NUM_CLASSES):
        label = ACTIVITY_CLASS_MAP[cls]
        binary_true = (y_test == cls).astype(int)
        cls_cal_probs = cal_probs[:, cls]
        if binary_true.sum() == 0:
            axes[cls].set_title(f"Cal - {label} (no samples)"); continue
        prob_true, prob_pred = calibration_curve(binary_true, cls_cal_probs, n_bins=10)
        axes[cls].plot(prob_pred, prob_true, "o-", color=CLASS_COLORS[cls], lw=2, label=label)
        axes[cls].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        axes[cls].set_xlabel("Mean Pred Prob"); axes[cls].set_ylabel("Fraction Pos")
        axes[cls].set_title(f"Calibration - {label}"); axes[cls].legend()
    fig.suptitle(f"Calibration ({best_model_name})", fontsize=14, y=1.02)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "05_calibration_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Feature importance
    if hasattr(best_model.named_steps["model"], "feature_importances_"):
        importances = best_model.named_steps["model"].feature_importances_
        top_k = 20
        indices = np.argsort(importances)[-top_k:]
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(range(top_k), importances[indices], color="#3498db", edgecolor="black")
        ax.set_yticks(range(top_k))
        ax.set_yticklabels([selected_columns[i] for i in indices])
        ax.set_xlabel("Importance"); ax.set_title(f"Top {top_k} Features ({best_model_name})")
        fig.tight_layout(); fig.savefig(OUTPUT_DIR / "06_feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print("  Saved: 02-06 plots")

    # ── 8. Statistical comparison ─────────────────────────────────────
    print("\n[8/12] Statistical comparison...")
    model_names = [row["model"] for row in cv_summary_sorted]
    stat_rows = []
    for i, model_a in enumerate(model_names):
        for model_b in model_names[i+1:]:
            sa = np.array(fold_scores.get(model_a, []))
            sb = np.array(fold_scores.get(model_b, []))
            if len(sa) == 0 or len(sb) == 0:
                continue
            t_stat, p_val = stats.ttest_rel(sa, sb)
            pooled = np.std(np.concatenate([sa, sb]))
            cohen_d = (sa.mean()-sb.mean()) / pooled if pooled else 0.0
            stat_rows.append({"Model A": model_a, "Model B": model_b,
                              "t-stat": t_stat, "p-value": p_val, "Cohen's d": cohen_d})
    if stat_rows:
        bonferroni = 0.05 / len(stat_rows)
        for r in stat_rows:
            r["Significant"] = "Yes" if r["p-value"] < bonferroni else "No"
            r["Bonferroni alpha"] = bonferroni

    # MCDA
    mcda_rows = []
    for row in cv_summary_sorted:
        name = row["model"]
        mcda_rows.append({
            "model": name, "roc_auc": row["roc_auc_mean"], "pr_auc": row["pr_auc_mean"],
            "calibration": max(0.0, 1-calibration_metrics.get(name, ece)),
            "robustness": max(0.0, 1-row["roc_auc_std"]),
            "efficiency": 1.0/(1.0+train_times.get(name, 1.0)),
            "interpretability": 1.0 if name in {"RandomForest","LightGBM","XGBoost"} else 0.5,
        })
    weights = {"roc_auc": 0.25, "pr_auc": 0.20, "calibration": 0.20,
               "robustness": 0.15, "efficiency": 0.10, "interpretability": 0.10}
    for metric in weights:
        vals = [r[metric] for r in mcda_rows]
        mn, mx = min(vals), max(vals)
        for r in mcda_rows:
            r[metric] = (r[metric]-mn)/(mx-mn) if mx > mn else 1.0
    for r in mcda_rows:
        r["composite"] = sum(r[m]*w for m, w in weights.items())
    mcda_rows = sorted(mcda_rows, key=lambda r: r["composite"], reverse=True)
    print(f"  MCDA winner: {mcda_rows[0]['model']}")

    # ── 9. Uncertainty ────────────────────────────────────────────────
    print("\n[9/12] Uncertainty quantification...")
    def conformal_prediction(probs, y_true, alpha=0.05):
        scores = 1.0 - probs[np.arange(len(y_true)), y_true]
        q = np.quantile(scores, 1-alpha, method="higher")
        pred_sets = probs >= (1.0-q)
        cov = pred_sets[np.arange(len(y_true)), y_true].mean()
        return pred_sets, cov, q

    pred_sets, coverage, q_threshold = conformal_prediction(cal_probs, y_test)
    set_sizes = pred_sets.sum(axis=1)

    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(X_train)
    train_dists = nn.kneighbors(X_train)[0].mean(axis=1)
    ad_threshold = np.percentile(train_dists, 95)
    test_dists = nn.kneighbors(X_test)[0].mean(axis=1)
    ood_rate = (test_dists > ad_threshold).mean()
    print(f"  Coverage={coverage:.4f}, Avg set size={set_sizes.mean():.2f}, OOD={ood_rate:.2%}")

    # Uncertainty plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    unique_sizes, counts = np.unique(set_sizes, return_counts=True)
    axes[0].bar(unique_sizes.astype(str), counts, color="#9b59b6", edgecolor="black")
    axes[0].set_xlabel("Set Size"); axes[0].set_ylabel("Count")
    axes[0].set_title(f"Conformal Sets (cov={coverage:.2%})")

    axes[1].hist(test_dists, bins=30, color="#1abc9c", edgecolor="black", alpha=0.8)
    axes[1].axvline(ad_threshold, color="red", ls="--", lw=2, label=f"AD={ad_threshold:.2f}")
    axes[1].set_xlabel("Mean k-NN Dist"); axes[1].set_ylabel("Count")
    axes[1].set_title(f"AD (OOD={ood_rate:.1%})"); axes[1].legend()

    max_probs = test_probs.max(axis=1)
    correct = (test_preds == y_test)
    bins_edge = np.linspace(0, 1, 11)
    bin_accs, bin_confs = [], []
    for lo, hi in zip(bins_edge[:-1], bins_edge[1:]):
        mask = (max_probs >= lo) & (max_probs < hi)
        if mask.sum() > 0:
            bin_accs.append(correct[mask].mean())
            bin_confs.append(max_probs[mask].mean())
    axes[2].plot(bin_confs, bin_accs, "o-", color="#e67e22", lw=2, label="Model")
    axes[2].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    axes[2].set_xlabel("Confidence"); axes[2].set_ylabel("Accuracy")
    axes[2].set_title("Reliability"); axes[2].legend()
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "07_uncertainty.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 07_uncertainty.png")

    # ── 10. Save model ────────────────────────────────────────────────
    print("\n[10/12] Saving model artifacts...")
    model_artifacts = {
        "best_model": best_model, "calibrated_model": calibrated,
        "selected_columns": selected_columns, "activity_class_map": ACTIVITY_CLASS_MAP,
        "num_classes": NUM_CLASSES, "ad_threshold": ad_threshold,
        "nn_model": nn, "conformal_q": q_threshold, "best_model_name": best_model_name,
    }
    model_path = MODEL_DIR / "safety_model.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump(model_artifacts, fh)

    summary = {
        "n_compounds": len(data), "targets": sorted(set(targets_all)),
        "class_distribution": {ACTIVITY_CLASS_MAP[c]: int((labels==c).sum()) for c in range(NUM_CLASSES)},
        "train_size": len(X_train), "val_size": len(X_val), "test_size": len(X_test),
        "best_model": best_model_name,
        "test_metrics": {"roc_auc_macro": test_roc, "pr_auc_macro": test_pr,
                         "mcc": test_mcc, "ece": ece, "mce": mce},
        "conformal_coverage": coverage,
        "avg_prediction_set_size": float(set_sizes.mean()), "out_of_domain_rate": ood_rate,
    }
    with open(OUTPUT_DIR / "workflow_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  Model saved: {model_path}")

    # ── 11. Held-out test set ─────────────────────────────────────────
    print("\n[11/12] Evaluating held-out test set...")
    ext_roc = ext_mcc = ext_acc = float("nan")
    test_target_metrics = {}
    y_test_ext_valid = np.array([])

    if TEST_PATH.exists():
        test_df = pd.read_csv(TEST_PATH)
        print(f"  Loaded {len(test_df)} test compounds")

        test_rows_for_features = []
        for _, row in test_df.iterrows():
            kc = row.get("known_class", -1)
            test_rows_for_features.append({
                "canonical_smiles": row["smiles"],
                "target_common_name": row["target"],
                "activity_class": int(kc) if pd.notna(kc) and str(kc).strip() != "" else -1,
            })
        X_test_ext, y_test_ext, _ = build_feature_matrix(test_rows_for_features,
                                                          selected_columns=selected_columns)
        valid_mask = y_test_ext >= 0
        X_test_ext_valid = X_test_ext[valid_mask]
        y_test_ext_valid = y_test_ext[valid_mask]
        test_df_valid = test_df[valid_mask].copy()

        if len(y_test_ext_valid) > 0:
            ext_probs = best_model.predict_proba(X_test_ext_valid)
            ext_preds = best_model.predict(X_test_ext_valid)
            ext_dists = nn.kneighbors(X_test_ext_valid)[0].mean(axis=1)
            ext_in_domain = ext_dists <= ad_threshold

            test_ext_classes = sorted(set(y_test_ext_valid))
            if len(test_ext_classes) >= 2:
                try:
                    ext_roc = roc_auc_score(y_test_ext_valid, ext_probs,
                                            multi_class="ovr", average="macro")
                except ValueError:
                    ext_roc = float("nan")
            ext_mcc = matthews_corrcoef(y_test_ext_valid, ext_preds)
            ext_acc = (ext_preds == y_test_ext_valid).mean()

            print(f"  ROC-AUC={ext_roc:.4f}, MCC={ext_mcc:.4f}, Acc={ext_acc:.4f}")
            print(f"  In-domain: {ext_in_domain.sum()}/{len(ext_in_domain)}")

            for target in sorted(test_df_valid["target"].unique()):
                tmask = test_df_valid["target"].values == target
                if tmask.sum() < 2:
                    continue
                t_true = y_test_ext_valid[tmask]
                t_pred = ext_preds[tmask]
                t_acc = (t_pred == t_true).mean()
                t_mcc = matthews_corrcoef(t_true, t_pred) if len(set(t_true)) > 1 else 0.0
                test_target_metrics[target] = {"n": int(tmask.sum()), "acc": t_acc, "mcc": t_mcc}

            # Test set results plot
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            actual_counts = Counter(y_test_ext_valid)
            pred_counts_ext = Counter(ext_preds)
            x_labels = [ACTIVITY_CLASS_MAP[c] for c in range(NUM_CLASSES)]
            x = np.arange(NUM_CLASSES); w = 0.35
            axes[0].bar(x-w/2, [actual_counts.get(c, 0) for c in range(NUM_CLASSES)],
                        w, color=[CLASS_COLORS[c] for c in range(NUM_CLASSES)],
                        edgecolor="black", alpha=0.7, label="Actual")
            axes[0].bar(x+w/2, [pred_counts_ext.get(c, 0) for c in range(NUM_CLASSES)],
                        w, color=[CLASS_COLORS[c] for c in range(NUM_CLASSES)],
                        edgecolor="black", alpha=0.4, label="Predicted", hatch="//")
            axes[0].set_xticks(x); axes[0].set_xticklabels(x_labels)
            axes[0].set_title("Actual vs Predicted"); axes[0].legend()

            if test_target_metrics:
                ts = sorted(test_target_metrics.keys(),
                            key=lambda t: test_target_metrics[t]["acc"], reverse=True)
                accs = [test_target_metrics[t]["acc"] for t in ts]
                ns = [test_target_metrics[t]["n"] for t in ts]
                colors = ["#3498db" if a >= 0.5 else "#e74c3c" for a in accs]
                axes[1].barh(range(len(ts)), accs, color=colors, edgecolor="black")
                axes[1].set_yticks(range(len(ts)))
                axes[1].set_yticklabels([f"{t} (n={n})" for t, n in zip(ts, ns)])
                axes[1].axvline(0.5, color="gray", ls="--"); axes[1].set_xlabel("Accuracy")
                axes[1].set_title("Per-Target Accuracy"); axes[1].set_xlim([0, 1.05])

            axes[2].hist(ext_probs.max(axis=1), bins=20, color="#9b59b6", edgecolor="black", alpha=0.8)
            axes[2].axvline(0.5, color="red", ls="--", lw=1.5, label="50%")
            axes[2].set_xlabel("Max Confidence"); axes[2].set_title("Confidence"); axes[2].legend()
            fig.suptitle("Held-Out Test Set Results", fontsize=14, y=1.02)
            fig.tight_layout()
            fig.savefig(OUTPUT_DIR / "08_test_set_results.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

            # Save predictions
            test_df_valid["predicted_class"] = ext_preds
            test_df_valid["predicted_label"] = [ACTIVITY_CLASS_MAP.get(int(p), "?") for p in ext_preds]
            test_df_valid["correct"] = ext_preds == y_test_ext_valid
            test_df_valid["in_domain"] = ext_in_domain
            for c in range(NUM_CLASSES):
                test_df_valid[f"prob_{ACTIVITY_CLASS_MAP[c]}"] = ext_probs[:, c]
            test_df_valid["max_confidence"] = ext_probs.max(axis=1)
            test_df_valid.to_csv(OUTPUT_DIR / "test_set_predictions.csv", index=False)
            print("  Saved: 08_test_set_results.png, test_set_predictions.csv")
    else:
        print(f"  Test file not found: {TEST_PATH}")

    # ── 12. Analysis report ───────────────────────────────────────────
    print("\n[12/12] Generating analysis report...")
    lines = []
    lines.append("# OFFTOXv3 Analysis Report")
    lines.append(f"\n**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Best Model:** {best_model_name}")
    lines.append(f"**Targets:** {len(TARGET_PANEL)} safety pharmacology targets\n")

    lines.append("## 1. Dataset Summary\n")
    lines.append(f"- **Total training compounds:** {len(data)}")
    lines.append(f"- **Unique targets:** {len(set(targets_all))}")
    lines.append(f"- **Train/Val/Test split:** {len(X_train)}/{len(X_val)}/{len(X_test)} (scaffold-based)")
    lines.append(f"- **Feature dimensions:** {features.shape[1]}\n")

    lines.append("### Class Distribution\n")
    lines.append("| Class | Label | Count | Percentage |")
    lines.append("|-------|-------|------:|----------:|")
    for cls in range(NUM_CLASSES):
        n = int((labels_all == cls).sum())
        lines.append(f"| {cls} | {ACTIVITY_CLASS_MAP[cls]} | {n} | {100*n/len(data):.1f}% |")

    lines.append("\n### Per-Target Compound Counts\n")
    lines.append("| Target | Category | Total | Potent | Less Potent | Inactive |")
    lines.append("|--------|----------|------:|-------:|------------:|---------:|")
    tdf = pd.DataFrame({"target": targets_all, "class": labels_all})
    for t in sorted(set(targets_all)):
        cat = TARGET_PANEL.get(t, {}).get("category", "?")
        td = tdf[tdf["target"] == t]
        lines.append(f"| {t} | {cat} | {len(td)} | {int((td['class']==2).sum())} | "
                     f"{int((td['class']==1).sum())} | {int((td['class']==0).sum())} |")

    lines.append("\n## 2. Cross-Validation Results\n")
    lines.append("| Model | ROC-AUC | PR-AUC | MCC |")
    lines.append("|-------|--------:|-------:|----:|")
    for row in cv_summary_sorted:
        lines.append(f"| {row['model']} | {row['roc_auc_mean']:.4f} +/- {row['roc_auc_std']:.4f} | "
                     f"{row['pr_auc_mean']:.4f} +/- {row['pr_auc_std']:.4f} | "
                     f"{row['mcc_mean']:.4f} +/- {row['mcc_std']:.4f} |")
    lines.append(f"\n**Selected model:** {best_model_name} (highest ROC-AUC)\n")

    lines.append("## 3. Internal Test Set Performance (Scaffold Split)\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|------:|")
    lines.append(f"| ROC-AUC (macro) | {test_roc:.4f} |")
    lines.append(f"| PR-AUC (macro) | {test_pr:.4f} |")
    lines.append(f"| MCC | {test_mcc:.4f} |")
    lines.append(f"| ECE (calibrated) | {ece:.4f} |")
    lines.append(f"| MCE (calibrated) | {mce:.4f} |")

    lines.append("\n### Confusion Matrix\n")
    cm_r = confusion_matrix(y_test, test_preds, labels=list(range(NUM_CLASSES)))
    lines.append("| | Pred: inactive | Pred: less_potent | Pred: potent |")
    lines.append("|---|---:|---:|---:|")
    for i, lab in enumerate(["inactive", "less_potent", "potent"]):
        lines.append(f"| **{lab}** | {cm_r[i,0]} | {cm_r[i,1]} | {cm_r[i,2]} |")

    lines.append("\n## 4. Uncertainty Quantification\n")
    lines.append(f"- **Conformal coverage:** {coverage:.4f} (target: 0.95)")
    lines.append(f"- **Average prediction set size:** {set_sizes.mean():.2f}")
    lines.append(f"- **AD threshold (95th pct k-NN):** {ad_threshold:.4f}")
    lines.append(f"- **Out-of-domain rate:** {ood_rate:.2%}\n")

    lines.append("## 5. Held-Out Test Set Evaluation\n")
    if not np.isnan(ext_roc):
        lines.append(f"- **Test compounds:** {len(y_test_ext_valid)}")
        lines.append(f"- **ROC-AUC (macro):** {ext_roc:.4f}")
        lines.append(f"- **MCC:** {ext_mcc:.4f}")
        lines.append(f"- **Accuracy:** {ext_acc:.4f}\n")
        if test_target_metrics:
            lines.append("### Per-Target Test Performance\n")
            lines.append("| Target | N | Accuracy | MCC |")
            lines.append("|--------|--:|---------:|----:|")
            for target, m in sorted(test_target_metrics.items()):
                lines.append(f"| {target} | {m['n']} | {m['acc']:.3f} | {m['mcc']:.3f} |")
    else:
        lines.append("No held-out test set available.\n")

    lines.append("\n## 6. Statistical Model Comparison\n")
    if stat_rows:
        lines.append(f"Bonferroni-corrected alpha = {bonferroni:.4f}\n")
        lines.append("| Model A | Model B | t-stat | p-value | Cohen's d | Significant |")
        lines.append("|---------|---------|-------:|--------:|----------:|:-----------:|")
        for r in stat_rows:
            cohens_d = r["Cohen's d"]
            lines.append(f"| {r['Model A']} | {r['Model B']} | {r['t-stat']:.4f} | "
                         f"{r['p-value']:.6f} | {cohens_d:.4f} | {r['Significant']} |")

    lines.append("\n## 7. MCDA Ranking\n")
    lines.append("| Rank | Model | Composite Score |")
    lines.append("|-----:|-------|----------------:|")
    for i, r in enumerate(mcda_rows, 1):
        lines.append(f"| {i} | {r['model']} | {r['composite']:.4f} |")

    lines.append("\n## 8. Target Panel Reference\n")
    lines.append("| # | Target | ChEMBL ID | Category |")
    lines.append("|--:|--------|-----------|----------|")
    for i, (tname, tinfo) in enumerate(TARGET_PANEL.items(), 1):
        lines.append(f"| {i} | {tname} | {tinfo['chembl_id']} | {tinfo['category']} |")

    lines.append("\n## 9. Output Files\n")
    lines.append("| File | Description |")
    lines.append("|------|-------------|")
    lines.append("| `01_data_exploration.png` | Class distribution, per-target breakdown, pChEMBL histogram |")
    lines.append("| `02_roc_curves.png` | Per-class ROC curves |")
    lines.append("| `03_pr_curves.png` | Per-class Precision-Recall curves |")
    lines.append("| `04_confusion_matrix.png` | Confusion matrices (counts and percentages) |")
    lines.append("| `05_calibration_curves.png` | Per-class calibration curves |")
    lines.append("| `06_feature_importance.png` | Top 20 feature importances |")
    lines.append("| `07_uncertainty.png` | Conformal sets, AD distances, reliability diagram |")
    lines.append("| `08_test_set_results.png` | Held-out test set results |")
    lines.append("| `workflow_summary.json` | Machine-readable summary |")
    lines.append("| `test_set_predictions.csv` | Test set predictions with probabilities |")
    lines.append("| `analysis_report.md` | This report |")
    lines.append("")

    report_path = OUTPUT_DIR / "analysis_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {report_path}")

    print("\n" + "=" * 70)
    print("Pipeline complete!")
    print(f"  Outputs: {OUTPUT_DIR}")
    print(f"  Model: {model_path}")
    print(f"  Report: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
