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

from rdkit.Chem.MolStandardize import rdMolStandardize

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
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
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ── Optional: SMOTE for class imbalance ──────────────────────────────
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

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
ACTIVITY_CLASS_MAP = {0: "non_binding", 1: "binding"}
CLASS_COLORS = {0: "#2ecc71", 1: "#e74c3c"}
NUM_CLASSES = 2
ACTIVITY_THRESHOLD_UM = 10.0        # pChEMBL >= 5.0 <=> IC50/Ki <= 10 µM
ACTIVITY_THRESHOLD_PCHEMBL = 5.0
SEVERE_OOD_THRESHOLD = 100.0       # k-NN distance beyond which predictions are highly uncertain

# ── SMILES standardizer ───────────────────────────────────────────────
_standardizer = rdMolStandardize.Standardizer()
_largest_fragment = rdMolStandardize.LargestFragmentChooser()
_uncharger = rdMolStandardize.Uncharger()

def standardize_smiles(smi):
    if not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        mol = _standardizer.standardize(mol)
        mol = _largest_fragment.choose(mol)
        mol = _uncharger.uncharge(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return Chem.MolToSmiles(mol)


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
    """Load training CSV and assign 2-class labels.

    Class 0 (non_binding): confirmed inactive, or pChEMBL < 5.0 (>= 10 µM)
    Class 1 (binding):     pChEMBL >= 5.0 (< 10 µM)
    """
    rows = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            smi = row.get("canonical_smiles")
            if not smi:
                continue
            raw_class = row.get("activity_class", "")
            raw_label = row.get("activity_class_label", "")
            # Confirmed inactive or non_binding → class 0
            if raw_class == "0" or raw_label in ("inactive", "non_binding"):
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
            # 2-class: binding (>= 5.0) vs non_binding (< 5.0)
            row["activity_class"] = 1 if pchembl >= 5.0 else 0
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


class WLGraphClassifier:
    """Weisfeiler-Lehman graph kernel + Logistic Regression (GNN proxy)."""

    def __init__(self, n_iterations=4, n_hash_buckets=4096, C=1.0,
                 class_weight="balanced", random_state=42):
        self.n_iterations = n_iterations
        self.n_hash_buckets = n_hash_buckets
        self.C = C
        self.class_weight = class_weight
        self.random_state = random_state
        self._clf = None

    def _mol_to_wl_vector(self, smi):
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            return np.zeros(self.n_hash_buckets * (self.n_iterations + 1))
        n_atoms = mol.GetNumAtoms()
        node_labels = {}
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            node_labels[idx] = hash((
                atom.GetAtomicNum(), atom.GetTotalNumHs(), atom.GetDegree(),
                int(atom.GetIsAromatic()), int(atom.IsInRing()), int(atom.GetFormalCharge()),
            )) % self.n_hash_buckets
        adj = {i: [] for i in range(n_atoms)}
        for bond in mol.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            adj[u].append(v); adj[v].append(u)
        feat = np.zeros(self.n_hash_buckets * (self.n_iterations + 1))
        for lbl in node_labels.values():
            feat[lbl] += 1
        for iteration in range(1, self.n_iterations + 1):
            new_labels = {}
            for idx in range(n_atoms):
                nb_labels = tuple(sorted(node_labels[nb] for nb in adj[idx]))
                new_labels[idx] = hash((node_labels[idx],) + nb_labels) % self.n_hash_buckets
            node_labels = new_labels
            offset = iteration * self.n_hash_buckets
            for lbl in node_labels.values():
                feat[offset + lbl] += 1
        return feat

    def _transform(self, smiles_list):
        X = np.array([self._mol_to_wl_vector(s) for s in smiles_list])
        return normalize(X, norm="l2")

    def fit(self, smiles_list, y):
        X = self._transform(smiles_list)
        self._clf = LogisticRegression(
            C=self.C, class_weight=self.class_weight,
            max_iter=1000, random_state=self.random_state, solver="saga")
        self._clf.fit(X, y)
        return self

    def predict(self, smiles_list):
        return self._clf.predict(self._transform(smiles_list))

    def predict_proba(self, smiles_list):
        return self._clf.predict_proba(self._transform(smiles_list))


def get_models(random_state, y_train_ref=None):
    pos_weight_est = 1.0
    if y_train_ref is not None:
        n_neg = int((y_train_ref == 0).sum())
        n_pos = int((y_train_ref == 1).sum())
        pos_weight_est = n_neg / max(n_pos, 1)

    return {
        "RandomForest": (
            Pipeline([("scaler", StandardScaler(with_mean=False)),
                       ("model", RandomForestClassifier(random_state=random_state, n_jobs=-1))]),
            {"model__n_estimators": [200, 500], "model__max_depth": [10, 20, None],
             "model__min_samples_split": [2, 5, 10], "model__max_features": ["sqrt", "log2", 0.3],
             "model__class_weight": ["balanced", {0: 1, 1: int(pos_weight_est)}, None]},
        ),
        "XGBoost": (
            Pipeline([("scaler", StandardScaler(with_mean=False)),
                       ("model", XGBClassifier(
                           random_state=random_state, objective="binary:logistic",
                           eval_metric="logloss", n_jobs=-1, verbosity=0,
                           scale_pos_weight=pos_weight_est))]),
            {"model__n_estimators": [200, 500], "model__max_depth": [3, 5, 7],
             "model__learning_rate": [0.01, 0.05, 0.1], "model__subsample": [0.6, 0.8, 1.0],
             "model__colsample_bytree": [0.6, 0.8, 1.0],
             "model__scale_pos_weight": [1.0, pos_weight_est]},
        ),
        "LightGBM": (
            Pipeline([("scaler", StandardScaler(with_mean=False)),
                       ("model", LGBMClassifier(
                           random_state=random_state, n_jobs=-1, verbose=-1,
                           objective="binary", is_unbalance=True))]),
            {"model__n_estimators": [200, 500], "model__max_depth": [-1, 5, 10],
             "model__learning_rate": [0.01, 0.05, 0.1], "model__num_leaves": [31, 63, 127],
             "model__subsample": [0.6, 0.8, 1.0], "model__is_unbalance": [True],
             "model__min_child_samples": [10, 20, 50]},
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
    class_by_target = class_by_target.reindex(columns=list(range(NUM_CLASSES)), fill_value=0)
    class_by_target.columns = [ACTIVITY_CLASS_MAP[c] for c in class_by_target.columns]
    class_by_target.loc[target_order].plot.barh(
        stacked=True, ax=axes[1],
        color=[CLASS_COLORS[c] for c in range(NUM_CLASSES)], edgecolor="black")
    axes[1].set_title("Compounds per Target"); axes[1].set_xlabel("Count")
    axes[1].legend(title="Class", loc="lower right")

    pchembl_vals = [float(row["pchembl_value"]) for row in data if row["pchembl_value"] is not None]
    axes[2].hist(pchembl_vals, bins=30, color="#3498db", edgecolor="black", alpha=0.8)
    axes[2].axvline(5.0, color="red", ls="--", lw=2, label="Binding threshold (5.0)")
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
    smiles_all   = [row["canonical_smiles"] for row in data]
    smiles_train = [smiles_all[i] for i in split.train_idx]
    smiles_val   = [smiles_all[i] for i in split.val_idx]
    smiles_test  = [smiles_all[i] for i in split.test_idx]
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    # Per-target imbalance check
    for target in sorted(set(targets_all)):
        t_idx = [i for i in split.train_idx if targets_all[i] == target]
        if not t_idx:
            continue
        t_labels = labels[t_idx]
        pct_bind = 100 * int((t_labels == 1).sum()) / len(t_labels)
        if pct_bind > 75 or pct_bind < 25:
            print(f"  IMBALANCE {target}: {pct_bind:.0f}% binders")

    # ── 5. Train models ──────────────────────────────────────────────
    print("\n[5/12] Training models (RF, XGBoost, LightGBM, GNN)...")
    models = get_models(RANDOM_STATE, y_train_ref=y_train)
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
            scoring="roc_auc", cv=3, random_state=RANDOM_STATE, n_jobs=1)
        t0 = time.time()
        search.fit(X_train, y_train)
        train_times[name] = time.time() - t0
        best_estimators[name] = search.best_estimator_

        scores, pr_scores, mcc_scores = [], [], []
        for tr_idx, te_idx in cv.split(X_train, y_train):
            X_tr, X_te = X_train[tr_idx], X_train[te_idx]
            y_tr, y_te = y_train[tr_idx], y_train[te_idx]
            # Apply SMOTE within each CV fold
            if HAS_SMOTE and len(np.unique(y_tr)) > 1:
                k_n = min(5, np.bincount(y_tr).min() - 1)
                if k_n >= 1:
                    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_n)
                    X_tr, y_tr = smote.fit_resample(X_tr, y_tr)
            est = search.best_estimator_
            est.fit(X_tr, y_tr)
            probs = est.predict_proba(X_te)
            preds = est.predict(X_te)
            scores.append(roc_auc_score(y_te, probs[:, 1]))
            pr_scores.append(average_precision_score(y_te, probs[:, 1]))
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

        print(f"ROC-AUC={np.mean(scores):.4f} MCC={np.mean(mcc_scores):.4f} ({train_times[name]:.0f}s)")

    # ── Train GNN (WL Graph Classifier) ──────────────────────────────
    print(f"  Training GNN (WL graph kernel)...", end=" ", flush=True)
    t0 = time.time()
    gnn = WLGraphClassifier(n_iterations=4, class_weight="balanced", random_state=RANDOM_STATE)
    gnn.fit(smiles_train, y_train)
    train_times["GNN"] = time.time() - t0
    best_estimators["GNN"] = gnn

    scores_gnn, pr_gnn, mcc_gnn = [], [], []
    for tr_idx, te_idx in cv.split(X_train, y_train):
        smi_tr = [smiles_train[i] for i in tr_idx]
        smi_te = [smiles_train[i] for i in te_idx]
        y_tr, y_te = y_train[tr_idx], y_train[te_idx]
        est_gnn = WLGraphClassifier(n_iterations=4, class_weight="balanced", random_state=RANDOM_STATE)
        est_gnn.fit(smi_tr, y_tr)
        gnn_probs = est_gnn.predict_proba(smi_te)
        gnn_preds = est_gnn.predict(smi_te)
        scores_gnn.append(roc_auc_score(y_te, gnn_probs[:, 1]))
        pr_gnn.append(average_precision_score(y_te, gnn_probs[:, 1]))
        mcc_gnn.append(matthews_corrcoef(y_te, gnn_preds))

    cv_summary.append({
        "model": "GNN",
        "roc_auc_mean": np.mean(scores_gnn), "roc_auc_std": np.std(scores_gnn),
        "pr_auc_mean": np.mean(pr_gnn), "pr_auc_std": np.std(pr_gnn),
        "mcc_mean": np.mean(mcc_gnn), "mcc_std": np.std(mcc_gnn),
    })
    fold_scores["GNN"] = scores_gnn
    gnn_val_probs = gnn.predict_proba(smiles_val)
    gnn_val_true = gnn_val_probs[np.arange(len(y_val)), y_val]
    ece_gnn, _ = ece_score_fn(np.ones(len(y_val)), gnn_val_true)
    calibration_metrics["GNN"] = ece_gnn
    print(f"ROC-AUC={np.mean(scores_gnn):.4f} MCC={np.mean(mcc_gnn):.4f} ({train_times['GNN']:.0f}s)")

    # ── 6. MCDA-based model selection & test eval ─────────────────────
    print("\n[6/12] Selecting best model (MCDA composite, MCC-weighted)...")
    # Compute composite MCDA score: MCC=40%, ROC-AUC=25%, PR-AUC=20%, calibration=10%, efficiency=5%
    sel_rows = []
    for row in cv_summary:
        n = row["model"]
        sel_rows.append({
            "model": n,
            "mcc": row["mcc_mean"],
            "roc_auc": row["roc_auc_mean"],
            "pr_auc": row["pr_auc_mean"],
            "calibration": max(0.0, 1.0 - calibration_metrics.get(n, 0.5)),
            "efficiency": 1.0 / (1.0 + train_times.get(n, 1.0)),
        })
    sel_weights = {"mcc": 0.40, "roc_auc": 0.25, "pr_auc": 0.20, "calibration": 0.10, "efficiency": 0.05}
    for metric in sel_weights:
        vals = [r[metric] for r in sel_rows]
        mn, mx = min(vals), max(vals)
        for r in sel_rows:
            r[f"{metric}_norm"] = (r[metric] - mn) / (mx - mn) if mx > mn else 1.0
    for r in sel_rows:
        r["composite"] = sum(r[f"{m}_norm"] * w for m, w in sel_weights.items())
    sel_rows = sorted(sel_rows, key=lambda r: r["composite"], reverse=True)

    best_model_name = sel_rows[0]["model"]
    best_model = best_estimators[best_model_name]

    # Top-3 unique models for consensus
    top3_models = []
    for r in sel_rows:
        if r["model"] not in top3_models:
            top3_models.append(r["model"])
        if len(top3_models) == 3:
            break

    print(f"  Best model (MCDA): {best_model_name}")
    print(f"  Top-3 consensus  : {top3_models}")
    for r in sel_rows:
        print(f"    {r['model']:<15s} MCC={r['mcc']:.4f} ROC-AUC={r['roc_auc']:.4f} composite={r['composite']:.4f}")

    # Refit best model with SMOTE
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.hstack([y_train, y_val])
    if best_model_name == "GNN":
        smiles_combined = smiles_train + smiles_val
        best_model.fit(smiles_combined, y_combined)
    else:
        if HAS_SMOTE and len(np.unique(y_combined)) > 1:
            k_n = min(5, np.bincount(y_combined).min() - 1)
            if k_n >= 1:
                smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_n)
                X_refit, y_refit = smote.fit_resample(X_combined, y_combined)
                print(f"  SMOTE: {len(y_combined)} → {len(y_refit)} samples")
            else:
                X_refit, y_refit = X_combined, y_combined
        else:
            X_refit, y_refit = X_combined, y_combined
        best_model.fit(X_refit, y_refit)

    # Refit other top-3 on same augmented data
    for m_name in top3_models:
        if m_name == best_model_name:
            continue
        if m_name == "GNN":
            smiles_combined = smiles_train + smiles_val
            best_estimators[m_name].fit(smiles_combined, y_combined)
        else:
            best_estimators[m_name].fit(X_refit if HAS_SMOTE else X_combined,
                                        y_refit if HAS_SMOTE else y_combined)

    cv_summary_sorted = sorted(cv_summary, key=lambda r: r["mcc_mean"], reverse=True)

    if best_model_name == "GNN":
        test_probs = best_model.predict_proba(smiles_test)
        test_preds = best_model.predict(smiles_test)
        calibrated = best_model
        cal_probs = test_probs
    else:
        test_probs = best_model.predict_proba(X_test)
        test_preds = best_model.predict(X_test)
        calibrated = CalibratedClassifierCV(best_model, method="isotonic", cv=3)
        calibrated.fit(X_refit if (HAS_SMOTE and best_model_name != "GNN") else X_combined,
                       y_refit if (HAS_SMOTE and best_model_name != "GNN") else y_combined)
        cal_probs = calibrated.predict_proba(X_test)

    test_roc = roc_auc_score(y_test, test_probs[:, 1])
    test_pr = average_precision_score(y_test, test_probs[:, 1])
    test_mcc = matthews_corrcoef(y_test, test_preds)
    cal_probs_true = cal_probs[np.arange(len(y_test)), y_test]
    ece, mce = ece_score_fn(np.ones(len(y_test)), cal_probs_true)

    print(f"  ROC-AUC={test_roc:.4f}, PR-AUC={test_pr:.4f}, MCC={test_mcc:.4f}")
    print(f"  ECE={ece:.4f}, MCE={mce:.4f}")
    if ece > 0.15:
        print(f"  NOTE: High ECE={ece:.3f} — use conformal prediction sets, not raw probabilities")

    # ── 7. Visualization plots ────────────────────────────────────────
    print("\n[7/12] Generating evaluation plots...")

    # ROC curves (binary: one curve for binding class)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for cls in range(NUM_CLASSES):
        label = ACTIVITY_CLASS_MAP[cls]
        fpr, tpr, _ = roc_curve(y_test, test_probs[:, 1], pos_label=1)
        auc_val = roc_auc_score(y_test, test_probs[:, 1])
        if cls == 0:
            axes[cls].plot(1-fpr, 1-tpr, color=CLASS_COLORS[cls], lw=2, label=f"AUC = {auc_val:.3f}")
        else:
            axes[cls].plot(fpr, tpr, color=CLASS_COLORS[cls], lw=2, label=f"AUC = {auc_val:.3f}")
        axes[cls].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        axes[cls].set_xlabel("FPR"); axes[cls].set_ylabel("TPR")
        axes[cls].set_title(f"ROC - {label}"); axes[cls].legend(loc="lower right")
    fig.suptitle(f"ROC Curves ({best_model_name})", fontsize=14, y=1.02)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "02_roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # PR curves (binary)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    prec_b, rec_b, _ = precision_recall_curve(y_test, test_probs[:, 1])
    ap_b = average_precision_score(y_test, test_probs[:, 1])
    axes[1].plot(rec_b, prec_b, color=CLASS_COLORS[1], lw=2, label=f"AP = {ap_b:.3f}")
    axes[1].axhline(y_test.mean(), color="gray", ls="--", lw=1, alpha=0.5,
                     label=f"Baseline = {y_test.mean():.3f}")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("PR - binding"); axes[1].legend(loc="upper right")
    prec_nb, rec_nb, _ = precision_recall_curve(1-y_test, test_probs[:, 0])
    ap_nb = average_precision_score(1-y_test, test_probs[:, 0])
    axes[0].plot(rec_nb, prec_nb, color=CLASS_COLORS[0], lw=2, label=f"AP = {ap_nb:.3f}")
    axes[0].axhline(1-y_test.mean(), color="gray", ls="--", lw=1, alpha=0.5,
                     label=f"Baseline = {1-y_test.mean():.3f}")
    axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
    axes[0].set_title("PR - non_binding"); axes[0].legend(loc="upper right")
    fig.suptitle(f"PR Curves ({best_model_name})", fontsize=14, y=1.02)
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

    # Calibration (binary)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
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

    # Feature importance (only for sklearn pipeline models, not GNN)
    if best_model_name != "GNN" and hasattr(best_model, "named_steps") and hasattr(best_model.named_steps.get("model", None), "feature_importances_"):
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

    # MCDA (MCC-weighted for class imbalance)
    mcda_rows = []
    for row in cv_summary_sorted:
        name = row["model"]
        mcda_rows.append({
            "model": name,
            "mcc": row["mcc_mean"],
            "roc_auc": row["roc_auc_mean"], "pr_auc": row["pr_auc_mean"],
            "calibration": max(0.0, 1-calibration_metrics.get(name, ece)),
            "robustness": max(0.0, 1-row["roc_auc_std"]),
            "efficiency": 1.0/(1.0+train_times.get(name, 1.0)),
            "interpretability": 1.0 if name in {"RandomForest","LightGBM","XGBoost"} else 0.6,
        })
    weights = {"mcc": 0.30, "roc_auc": 0.20, "pr_auc": 0.18,
               "calibration": 0.15, "robustness": 0.10, "efficiency": 0.04, "interpretability": 0.03}
    for metric in weights:
        vals = [r[metric] for r in mcda_rows]
        mn, mx = min(vals), max(vals)
        for r in mcda_rows:
            r[f"{metric}_norm"] = (r[metric]-mn)/(mx-mn) if mx > mn else 1.0
    for r in mcda_rows:
        r["composite"] = sum(r[f"{m}_norm"]*w for m, w in weights.items())
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
    sas_cores = getattr(model_artifacts_tmp := {}, "sas_cores", {}) if False else {}
    model_artifacts = {
        "best_model": best_model, "calibrated_model": calibrated,
        "all_estimators": best_estimators,
        "top3_models": top3_models,
        "selected_columns": selected_columns, "activity_class_map": ACTIVITY_CLASS_MAP,
        "num_classes": NUM_CLASSES, "ad_threshold": ad_threshold,
        "severe_ood_threshold": SEVERE_OOD_THRESHOLD,
        "nn_model": nn, "conformal_q": q_threshold, "best_model_name": best_model_name,
        "sas_cores": sas_cores,
        "smiles_train": smiles_train,
        "target_panel": TARGET_PANEL,
        "mcda_ranking": [{"model": r["model"], "composite": r["composite"],
                          "mcc": r["mcc"], "roc_auc": r["roc_auc"]}
                         for r in mcda_rows],
    }
    model_path = MODEL_DIR / "safety_model.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump(model_artifacts, fh)

    summary = {
        "n_compounds": len(data), "targets": sorted(set(targets_all)),
        "class_distribution": {ACTIVITY_CLASS_MAP[c]: int((labels_all==c).sum()) for c in range(NUM_CLASSES)},
        "train_size": len(X_train), "val_size": len(X_val), "test_size": len(X_test),
        "best_model": best_model_name,
        "top3_models": top3_models,
        "model_selection_method": "MCDA composite (MCC=40%, ROC-AUC=25%, PR-AUC=20%)",
        "test_metrics": {"roc_auc_macro": test_roc, "pr_auc_macro": test_pr,
                         "mcc": test_mcc, "ece": ece, "mce": mce},
        "conformal_coverage": coverage,
        "avg_prediction_set_size": float(set_sizes.mean()),
        "out_of_domain_rate": ood_rate,
        "calibration_warning": ece > 0.15,
        "smote_applied": HAS_SMOTE,
        "mcda_ranking": [{"model": r["model"], "composite": r["composite"]} for r in mcda_rows],
    }
    with open(OUTPUT_DIR / "workflow_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  Model saved: {model_path}")
    print(f"  Best model (MCDA): {best_model_name}")
    print(f"  Top-3 consensus  : {top3_models}")

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
            if pd.notna(kc) and str(kc).strip() != "":
                kc_int = int(kc)
                # Map old 3-class labels to 2-class: 0→0, 1→0 (non_binding), 2→1 (binding)
                if kc_int == 2:
                    kc_mapped = 1
                elif kc_int in (0, 1):
                    kc_mapped = 0
                else:
                    kc_mapped = -1
            else:
                kc_mapped = -1
            test_rows_for_features.append({
                "canonical_smiles": row["smiles"],
                "target_common_name": row["target"],
                "activity_class": kc_mapped,
            })
        X_test_ext, y_test_ext, _ = build_feature_matrix(test_rows_for_features,
                                                          selected_columns=selected_columns)
        valid_mask = y_test_ext >= 0
        X_test_ext_valid = X_test_ext[valid_mask]
        y_test_ext_valid = y_test_ext[valid_mask]
        test_df_valid = test_df[valid_mask].copy()

        smiles_test_ext = [r["canonical_smiles"] for r, valid in
                           zip(test_rows_for_features, valid_mask) if valid]

        if len(y_test_ext_valid) > 0:
            if best_model_name == "GNN":
                ext_probs = best_model.predict_proba(smiles_test_ext)
                ext_preds = best_model.predict(smiles_test_ext)
            else:
                ext_probs = best_model.predict_proba(X_test_ext_valid)
                ext_preds = best_model.predict(X_test_ext_valid)

            # Consensus from top-3 models
            con_probs_list = []
            for m_name in top3_models:
                est_m = best_estimators[m_name]
                try:
                    if m_name == "GNN":
                        mp = est_m.predict_proba(smiles_test_ext)
                    else:
                        mp = est_m.predict_proba(X_test_ext_valid)
                    con_probs_list.append(mp)
                except Exception:
                    pass
            consensus_probs = np.mean(np.stack(con_probs_list, axis=0), axis=0) if con_probs_list else ext_probs
            consensus_preds = np.argmax(consensus_probs, axis=1)

            ext_dists = nn.kneighbors(X_test_ext_valid)[0].mean(axis=1)
            ext_in_domain = ext_dists <= ad_threshold
            ext_severe_ood = ext_dists > SEVERE_OOD_THRESHOLD

            test_ext_classes = sorted(set(y_test_ext_valid))
            if len(test_ext_classes) >= 2:
                try:
                    ext_roc = roc_auc_score(y_test_ext_valid, ext_probs[:, 1])
                    con_roc = roc_auc_score(y_test_ext_valid, consensus_probs[:, 1])
                except ValueError:
                    ext_roc = con_roc = float("nan")
            ext_mcc = matthews_corrcoef(y_test_ext_valid, ext_preds)
            con_mcc = matthews_corrcoef(y_test_ext_valid, consensus_preds)
            ext_acc = (ext_preds == y_test_ext_valid).mean()

            print(f"  Best model  — ROC-AUC={ext_roc:.4f}, MCC={ext_mcc:.4f}, Acc={ext_acc:.4f}")
            print(f"  Consensus   — ROC-AUC={con_roc:.4f}, MCC={con_mcc:.4f}")
            print(f"  In-domain: {ext_in_domain.sum()}/{len(ext_in_domain)}, "
                  f"Severe OOD (>{SEVERE_OOD_THRESHOLD:.0f}): {ext_severe_ood.sum()}")

            suspicious_targets = []
            for target in sorted(test_df_valid["target"].unique()):
                tmask = test_df_valid["target"].values == target
                if tmask.sum() < 2:
                    continue
                t_true = y_test_ext_valid[tmask]
                t_pred = ext_preds[tmask]
                t_acc = (t_pred == t_true).mean()
                t_mcc = matthews_corrcoef(t_true, t_pred) if len(set(t_true)) > 1 else float("nan")
                is_suspicious = t_mcc == 1.0
                if is_suspicious:
                    suspicious_targets.append(target)
                test_target_metrics[target] = {
                    "n": int(tmask.sum()), "acc": t_acc,
                    "mcc": t_mcc if not np.isnan(t_mcc) else 0.0,
                    "suspicious_mcc1": is_suspicious,
                }
                flag = " *** MCC=1 SUSPECT" if is_suspicious else ""
                print(f"    {target:<12s} n={tmask.sum()} acc={t_acc:.3f} mcc={t_mcc:.3f}{flag}"
                      if not np.isnan(t_mcc) else
                      f"    {target:<12s} n={tmask.sum()} acc={t_acc:.3f} mcc=N/A")

            if suspicious_targets:
                print(f"  WARNING: {len(suspicious_targets)} target(s) with MCC=1 — "
                      f"possible data leakage: {suspicious_targets}")

            def _activity_lbl(p):
                return f"Active (<{ACTIVITY_THRESHOLD_UM}µM)" if p == 1 else f"Inactive(>={ACTIVITY_THRESHOLD_UM}µM)"

            test_df_valid = test_df_valid.copy()
            test_df_valid["predicted_class"]      = ext_preds
            test_df_valid["predicted_label"]      = [ACTIVITY_CLASS_MAP.get(int(p), "?") for p in ext_preds]
            test_df_valid["predicted_activity"]   = [_activity_lbl(p) for p in ext_preds]
            test_df_valid["consensus_class"]      = consensus_preds
            test_df_valid["consensus_activity"]   = [_activity_lbl(p) for p in consensus_preds]
            test_df_valid["in_domain"]            = ext_in_domain
            test_df_valid["severe_ood"]           = ext_severe_ood
            test_df_valid["knn_distance"]         = ext_dists
            test_df_valid["prob_binding"]         = ext_probs[:, 1]
            test_df_valid["consensus_prob_binding"] = consensus_probs[:, 1]
            test_df_valid["target_suspicious_mcc1"] = test_df_valid["target"].isin(suspicious_targets)
            test_df_valid.to_csv(OUTPUT_DIR / "test_set_predictions.csv", index=False)

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

            # test_set_predictions.csv already saved above
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
    lines.append("| Target | Category | Total | Binding | Non-Binding |")
    lines.append("|--------|----------|------:|--------:|------------:|")
    tdf = pd.DataFrame({"target": targets_all, "class": labels_all})
    for t in sorted(set(targets_all)):
        cat = TARGET_PANEL.get(t, {}).get("category", "?")
        td = tdf[tdf["target"] == t]
        lines.append(f"| {t} | {cat} | {len(td)} | {int((td['class']==1).sum())} | "
                     f"{int((td['class']==0).sum())} |")

    lines.append("\n## 2. Cross-Validation Results\n")
    lines.append("| Model | ROC-AUC | PR-AUC | MCC |")
    lines.append("|-------|--------:|-------:|----:|")
    for row in cv_summary_sorted:
        lines.append(f"| {row['model']} | {row['roc_auc_mean']:.4f} +/- {row['roc_auc_std']:.4f} | "
                     f"{row['pr_auc_mean']:.4f} +/- {row['pr_auc_std']:.4f} | "
                     f"{row['mcc_mean']:.4f} +/- {row['mcc_std']:.4f} |")
    lines.append(f"\n**Selected model:** {best_model_name} (MCDA composite: MCC=40%, ROC-AUC=25%, PR-AUC=20%)")
    lines.append(f"\n**Top-3 consensus models:** {', '.join(top3_models)}")
    lines.append(f"\n**SMOTE applied:** {'Yes (imbalanced-learn)' if HAS_SMOTE else 'No (install imbalanced-learn)'}\n")

    lines.append("## 3. Internal Test Set Performance (Scaffold Split)\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|------:|")
    lines.append(f"| ROC-AUC | {test_roc:.4f} |")
    lines.append(f"| PR-AUC | {test_pr:.4f} |")
    lines.append(f"| MCC | {test_mcc:.4f} |")
    lines.append(f"| ECE (calibrated) | {ece:.4f} |")
    lines.append(f"| MCE (calibrated) | {mce:.4f} |")

    lines.append("\n### Confusion Matrix\n")
    cm_r = confusion_matrix(y_test, test_preds, labels=list(range(NUM_CLASSES)))
    lines.append("| | Pred: non_binding | Pred: binding |")
    lines.append("|---|---:|---:|")
    for i, lab in enumerate(["non_binding", "binding"]):
        lines.append(f"| **{lab}** | {cm_r[i,0]} | {cm_r[i,1]} |")

    lines.append("\n## 4. Uncertainty Quantification\n")
    lines.append(f"- **Conformal coverage:** {coverage:.4f} (target: 0.95)")
    lines.append(f"- **Average prediction set size:** {set_sizes.mean():.2f}")
    lines.append(f"- **AD threshold (95th pct k-NN):** {ad_threshold:.4f}")
    lines.append(f"- **Out-of-domain rate:** {ood_rate:.2%}\n")

    lines.append("## 5. Held-Out Test Set Evaluation\n")
    if not np.isnan(ext_roc):
        lines.append(f"- **Test compounds:** {len(y_test_ext_valid)}")
        lines.append(f"- **ROC-AUC:** {ext_roc:.4f}")
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
