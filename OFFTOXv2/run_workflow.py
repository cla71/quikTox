"""
Run the safety pharmacology ML workflow using safety_targets_bioactivity.csv.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import csv
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, MolSurf, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "safety_targets_bioactivity.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "workflow_outputs"
RANDOM_STATE = 42


@dataclass
class SplitData:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


ACTIVITY_CLASS_MAP = {0: "inactive", 1: "less_potent", 2: "potent"}
NUM_CLASSES = 3


def load_and_clean_data(path: Path) -> List[dict]:
    """Load training data and assign 3-class activity labels.

    Classes
    -------
    2 – potent:      pChEMBL >= 5.0  (< 10 µM)
    1 – less_potent: 4.0 <= pChEMBL < 5.0  (10-100 µM)
    0 – inactive:    confirmed-inactive compounds (activity_class == 0 in
                     source data, no measurable pChEMBL)
    """
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            smi = row.get("canonical_smiles")
            if not smi:
                continue

            # --- Handle confirmed-inactive compounds ---
            # These may have activity_class already set to '0' by the
            # retrieval script and may lack a pchembl_value.
            raw_class = row.get("activity_class", "")
            if raw_class == "0" or row.get("activity_class_label") == "inactive":
                row["pchembl_value"] = None
                row["activity_class"] = 0
                rows.append(row)
                continue

            # --- Active / less-potent compounds ---
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
            # 3-class assignment
            row["activity_class"] = 2 if pchembl >= 5.0 else 1
            rows.append(row)

    # Deduplicate: keep highest pchembl per (molecule, target) pair.
    # For inactive compounds (pchembl is None), keep one entry.
    deduped: dict = {}
    for row in rows:
        key = (row.get("molecule_chembl_id"), row.get("target_chembl_id"))
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = row
        else:
            # Prefer the record with a real pchembl measurement
            existing_p = existing.get("pchembl_value")
            current_p = row.get("pchembl_value")
            if current_p is not None and (existing_p is None or current_p > existing_p):
                deduped[key] = row
    return list(deduped.values())


def compute_descriptors(smiles: List[str]) -> Tuple[np.ndarray, List[str]]:
    descriptor_functions = {
        "MW": Descriptors.MolWt,
        "LogP": Crippen.MolLogP,
        "HBA": Lipinski.NumHAcceptors,
        "HBD": Lipinski.NumHDonors,
        "TPSA": MolSurf.TPSA,
        "RotatableBonds": Lipinski.NumRotatableBonds,
        "AromaticRings": Lipinski.NumAromaticRings,
        "HeavyAtoms": Lipinski.HeavyAtomCount,
        "FractionCSP3": Lipinski.FractionCSP3,
        "MolMR": Crippen.MolMR,
    }
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append([np.nan for _ in descriptor_functions])
            continue
        rows.append([func(mol) for func in descriptor_functions.values()])
    return np.array(rows, dtype=float), list(descriptor_functions.keys())


def compute_morgan_fingerprints(smiles: List[str], n_bits: int = 2048) -> np.ndarray:
    fingerprints = []
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fingerprints.append(np.zeros(n_bits, dtype=int))
            continue
        fp = generator.GetFingerprint(mol)
        fingerprints.append(np.array(fp))
    return np.array(fingerprints)


def scaffold_split(smiles: List[str], y: np.ndarray, random_state: int = 42) -> SplitData:
    scaffolds: Dict[str, List[int]] = {}
    for idx, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            scaffold = ""
        else:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
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

    return SplitData(
        train_idx=np.array(train_idx),
        val_idx=np.array(val_idx),
        test_idx=np.array(test_idx),
    )


def build_feature_matrix(
    rows: List[dict],
    selected_columns: List[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    smiles = [row["canonical_smiles"] for row in rows]
    targets = [row["target_common_name"] for row in rows]
    labels = np.array([row["activity_class"] for row in rows], dtype=int)
    descriptors, descriptor_names = compute_descriptors(smiles)
    fingerprints = compute_morgan_fingerprints(smiles)
    fingerprint_names = [f"FP_{i}" for i in range(fingerprints.shape[1])]
    target_names = sorted({t for t in targets if t})
    target_map = {name: idx for idx, name in enumerate(target_names)}
    target_matrix = np.zeros((len(rows), len(target_names)), dtype=float)
    for idx, target in enumerate(targets):
        if target in target_map:
            target_matrix[idx, target_map[target]] = 1.0
    feature_matrix = np.concatenate([descriptors, fingerprints, target_matrix], axis=1)
    columns = descriptor_names + fingerprint_names + [f"target_{name}" for name in target_names]
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
                aligned[:, out_idx] = np.nan_to_num(feature_matrix[:, col_index[col]], nan=0.0)
        feature_matrix = aligned
    return feature_matrix, labels, selected_columns


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[float, float]:
    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    mce = 0.0
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


def conformal_prediction(probs: np.ndarray, y_true: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, float]:
    scores = 1.0 - probs[np.arange(len(y_true)), y_true]
    q = np.quantile(scores, 1 - alpha, method="higher")
    prediction_sets = probs >= (1.0 - q)
    coverage = prediction_sets[np.arange(len(y_true)), y_true].mean()
    return prediction_sets, coverage


def get_models(random_state: int) -> Dict[str, Tuple[Pipeline, Dict[str, List]]]:
    models = {
        "RandomForest": (
            Pipeline([
                ("scaler", StandardScaler(with_mean=False)),
                ("model", RandomForestClassifier(
                    random_state=random_state, n_jobs=-1,
                )),
            ]),
            {
                "model__n_estimators": [200, 500],
                "model__max_depth": [10, 20, None],
                "model__min_samples_split": [2, 5, 10],
                "model__max_features": ["sqrt", "log2", 0.3],
                "model__class_weight": ["balanced", None],
            },
        ),
        "XGBoost": (
            Pipeline([
                ("scaler", StandardScaler(with_mean=False)),
                ("model", XGBClassifier(
                    random_state=random_state,
                    objective="multi:softprob",
                    num_class=NUM_CLASSES,
                    eval_metric="mlogloss",
                    n_jobs=-1,
                    verbosity=0,
                )),
            ]),
            {
                "model__n_estimators": [200, 500],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__subsample": [0.6, 0.8, 1.0],
                "model__colsample_bytree": [0.6, 0.8, 1.0],
            },
        ),
        "LightGBM": (
            Pipeline([
                ("scaler", StandardScaler(with_mean=False)),
                ("model", LGBMClassifier(
                    random_state=random_state, n_jobs=-1, verbose=-1,
                    objective="multiclass",
                    num_class=NUM_CLASSES,
                )),
            ]),
            {
                "model__n_estimators": [200, 500],
                "model__max_depth": [-1, 5, 10],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__num_leaves": [31, 63, 127],
                "model__subsample": [0.6, 0.8, 1.0],
            },
        ),
    }
    return models


def run_workflow() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_clean_data(DATA_PATH)
    features, labels, selected_columns = build_feature_matrix(df)
    split = scaffold_split([row["canonical_smiles"] for row in df], labels, random_state=RANDOM_STATE)

    X_train = features[split.train_idx]
    y_train = labels[split.train_idx]
    X_val = features[split.val_idx]
    y_val = labels[split.val_idx]
    X_test = features[split.test_idx]
    y_test = labels[split.test_idx]

    models = get_models(RANDOM_STATE)
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=RANDOM_STATE)
    cv_summary = []
    best_estimators = {}
    calibration_metrics = {}
    fold_scores: Dict[str, List[float]] = {}
    train_times = {}

    for name, (pipeline, param_grid) in models.items():
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            n_iter=5,
            scoring="roc_auc_ovr",
            cv=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        start = time.time()
        search.fit(X_train, y_train)
        train_times[name] = time.time() - start
        best_estimators[name] = search.best_estimator_

        scores = []
        pr_scores = []
        mcc_scores = []
        for train_idx, test_idx in cv.split(X_train, y_train):
            X_tr, X_te = X_train[train_idx], X_train[test_idx]
            y_tr, y_te = y_train[train_idx], y_train[test_idx]
            estimator = search.best_estimator_
            estimator.fit(X_tr, y_tr)
            probs = estimator.predict_proba(X_te)
            preds = estimator.predict(X_te)
            scores.append(roc_auc_score(y_te, probs, multi_class="ovr", average="macro"))
            # Per-class average precision, then macro-average
            pr_auc_per_class = []
            for cls in range(NUM_CLASSES):
                if (y_te == cls).sum() > 0:
                    pr_auc_per_class.append(average_precision_score(
                        (y_te == cls).astype(int), probs[:, cls]
                    ))
            pr_scores.append(float(np.mean(pr_auc_per_class)) if pr_auc_per_class else 0.0)
            mcc_scores.append(matthews_corrcoef(y_te, preds))

        cv_summary.append({
            "model": name,
            "roc_auc_mean": np.mean(scores),
            "roc_auc_std": np.std(scores),
            "pr_auc_mean": np.mean(pr_scores),
            "pr_auc_std": np.std(pr_scores),
            "mcc_mean": np.mean(mcc_scores),
            "mcc_std": np.std(mcc_scores),
        })
        fold_scores[name] = scores

        val_probs = search.best_estimator_.predict_proba(X_val)
        # ECE computed on the predicted probability of the true class
        val_probs_true_class = val_probs[np.arange(len(y_val)), y_val]
        calibration_metrics[name] = ece_score(
            np.ones(len(y_val)), val_probs_true_class
        )[0]

    cv_summary_sorted = sorted(cv_summary, key=lambda row: row["roc_auc_mean"], reverse=True)
    with (OUTPUT_DIR / "cv_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cv_summary_sorted[0].keys()))
        writer.writeheader()
        writer.writerows(cv_summary_sorted)

    best_model_name = cv_summary_sorted[0]["model"]
    best_model = best_estimators[best_model_name]
    best_model.fit(np.vstack([X_train, X_val]), np.hstack([y_train, y_val]))

    test_probs = best_model.predict_proba(X_test)  # shape (n, 3)
    test_preds = best_model.predict(X_test)
    test_roc_auc = roc_auc_score(y_test, test_probs, multi_class="ovr", average="macro")
    pr_auc_per_class = []
    for cls in range(NUM_CLASSES):
        if (y_test == cls).sum() > 0:
            pr_auc_per_class.append(average_precision_score(
                (y_test == cls).astype(int), test_probs[:, cls]
            ))
    test_pr_auc = float(np.mean(pr_auc_per_class)) if pr_auc_per_class else 0.0
    test_metrics = {
        "model": best_model_name,
        "roc_auc_macro": test_roc_auc,
        "pr_auc_macro": test_pr_auc,
        "mcc": matthews_corrcoef(y_test, test_preds),
    }

    calibrated = CalibratedClassifierCV(best_model, method="isotonic", cv=3)
    calibrated.fit(np.vstack([X_train, X_val]), np.hstack([y_train, y_val]))
    calibrated_probs = calibrated.predict_proba(X_test)  # shape (n, 3)
    # ECE on the probability assigned to the true class
    calibrated_probs_true = calibrated_probs[np.arange(len(y_test)), y_test]
    ece, mce = ece_score(np.ones(len(y_test)), calibrated_probs_true)
    test_metrics["ece"] = ece
    test_metrics["mce"] = mce

    with (OUTPUT_DIR / "test_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(test_metrics.keys()))
        writer.writeheader()
        writer.writerow(test_metrics)

    cm = confusion_matrix(y_test, test_preds, labels=list(range(NUM_CLASSES)))

    # Per-class ROC and PR curves (one-vs-rest)
    for cls in range(NUM_CLASSES):
        cls_label = ACTIVITY_CLASS_MAP[cls]
        binary_true = (y_test == cls).astype(int)
        cls_probs = test_probs[:, cls]
        if binary_true.sum() == 0:
            continue
        fpr, tpr, roc_thresholds = roc_curve(binary_true, cls_probs)
        with (OUTPUT_DIR / f"roc_curve_{cls_label}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["fpr", "tpr", "threshold"])
            writer.writeheader()
            for fpr_val, tpr_val, threshold in zip(fpr, tpr, roc_thresholds):
                writer.writerow({"fpr": fpr_val, "tpr": tpr_val, "threshold": threshold})

        pr_precision, pr_recall, pr_thresholds = precision_recall_curve(binary_true, cls_probs)
        with (OUTPUT_DIR / f"pr_curve_{cls_label}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["precision", "recall", "threshold"])
            writer.writeheader()
            thresholds_list = list(pr_thresholds) + [np.nan]
            for precision, recall, threshold in zip(pr_precision, pr_recall, thresholds_list):
                writer.writerow({"precision": precision, "recall": recall, "threshold": threshold})

    # 3-class confusion matrix
    with (OUTPUT_DIR / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = [""] + [f"pred_{ACTIVITY_CLASS_MAP[c]}" for c in range(NUM_CLASSES)]
        writer.writerow(header)
        for i in range(NUM_CLASSES):
            row_data = [f"actual_{ACTIVITY_CLASS_MAP[i]}"] + [int(cm[i, j]) for j in range(NUM_CLASSES)]
            writer.writerow(row_data)

    # Calibration curve per class
    for cls in range(NUM_CLASSES):
        cls_label = ACTIVITY_CLASS_MAP[cls]
        binary_true = (y_test == cls).astype(int)
        cls_cal_probs = calibrated_probs[:, cls]
        if binary_true.sum() == 0:
            continue
        prob_true, prob_pred = calibration_curve(binary_true, cls_cal_probs, n_bins=10)
        with (OUTPUT_DIR / f"calibration_curve_{cls_label}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["mean_predicted_prob", "fraction_positives"])
            writer.writeheader()
            for mean_pred, frac_pos in zip(prob_pred, prob_true):
                writer.writerow({"mean_predicted_prob": mean_pred, "fraction_positives": frac_pos})

    if hasattr(best_model.named_steps["model"], "feature_importances_"):
        importances = best_model.named_steps["model"].feature_importances_
        indices = np.argsort(importances)[-20:]
        feature_rows = [
            {"feature": selected_columns[idx], "importance": float(importances[idx])}
            for idx in indices
        ]
        feature_rows = sorted(feature_rows, key=lambda row: row["importance"], reverse=True)
        with (OUTPUT_DIR / "feature_importance.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["feature", "importance"])
            writer.writeheader()
            writer.writerows(feature_rows)

    calibration_probs = calibrated.predict_proba(X_test)
    prediction_sets, coverage = conformal_prediction(calibration_probs, y_test)
    set_sizes = prediction_sets.sum(axis=1)
    with (OUTPUT_DIR / "conformal_set_sizes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["set_size"])
        writer.writeheader()
        for size in set_sizes:
            writer.writerow({"set_size": int(size)})
    with open(OUTPUT_DIR / "conformal_summary.json", "w", encoding="utf-8") as handle:
        json.dump({"coverage": coverage, "avg_set_size": float(set_sizes.mean())}, handle, indent=2)

    unique_sizes, counts = np.unique(set_sizes, return_counts=True)
    with (OUTPUT_DIR / "conformal_set_sizes_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["set_size", "count"])
        writer.writeheader()
        for size, count in zip(unique_sizes, counts):
            writer.writerow({"set_size": int(size), "count": int(count)})

    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(X_train)
    distances, _ = nn.kneighbors(X_train)
    train_dist = distances.mean(axis=1)
    threshold = np.percentile(train_dist, 95)
    test_distances, _ = nn.kneighbors(X_test)
    test_dist = test_distances.mean(axis=1)
    out_of_domain = (test_dist > threshold).mean()

    stats_rows = []
    models_list = [row["model"] for row in cv_summary_sorted]
    for i, model_a in enumerate(models_list):
        for model_b in models_list[i + 1 :]:
            scores_a = np.array(fold_scores.get(model_a, []))
            scores_b = np.array(fold_scores.get(model_b, []))
            if len(scores_a) == 0 or len(scores_b) == 0:
                continue
            t_stat, p_val = stats.ttest_rel(scores_a, scores_b)
            pooled_std = np.std(np.concatenate([scores_a, scores_b]))
            cohen_d = (scores_a.mean() - scores_b.mean()) / pooled_std if pooled_std else 0.0
            stats_rows.append({
                "model_a": model_a,
                "model_b": model_b,
                "t_stat": t_stat,
                "p_value": p_val,
                "cohen_d": cohen_d,
            })

    if stats_rows:
        bonferroni = 0.05 / len(stats_rows)
        for row in stats_rows:
            row["bonferroni_alpha"] = bonferroni
        with (OUTPUT_DIR / "statistical_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(stats_rows[0].keys()))
            writer.writeheader()
            writer.writerows(stats_rows)
    else:
        with (OUTPUT_DIR / "statistical_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
            handle.write("")

    mcda_rows = []
    for row in cv_summary_sorted:
        name = row["model"]
        metric_row = {
            "model": name,
            "roc_auc": row["roc_auc_mean"],
            "pr_auc": row["pr_auc_mean"],
            "calibration": max(0.0, 1 - calibration_metrics.get(name, ece)),
            "robustness": max(0.0, 1 - row["roc_auc_std"]),
            "efficiency": 1.0 / (1.0 + train_times.get(name, 1.0)),
            "interpretability": 1.0 if name in {"RandomForest", "LightGBM", "XGBoost"} else 0.5,
        }
        mcda_rows.append(metric_row)

    weights = {
        "roc_auc": 0.25,
        "pr_auc": 0.20,
        "calibration": 0.20,
        "robustness": 0.15,
        "efficiency": 0.10,
        "interpretability": 0.10,
    }
    for metric in weights:
        values = [row[metric] for row in mcda_rows]
        min_val = min(values)
        max_val = max(values)
        for row in mcda_rows:
            if max_val > min_val:
                row[metric] = (row[metric] - min_val) / (max_val - min_val)
            else:
                row[metric] = 1.0
    for row in mcda_rows:
        row["composite_score"] = sum(row[m] * w for m, w in weights.items())
    mcda_rows = sorted(mcda_rows, key=lambda row: row["composite_score"], reverse=True)
    with (OUTPUT_DIR / "mcda_ranking.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mcda_rows[0].keys()))
        writer.writeheader()
        writer.writerows(mcda_rows)

    class_counts = {ACTIVITY_CLASS_MAP[c]: int((labels == c).sum()) for c in range(NUM_CLASSES)}
    summary = {
        "n_compounds": len(df),
        "targets": len({row["target_common_name"] for row in df}),
        "class_distribution": class_counts,
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "best_model": best_model_name,
        "test_metrics": test_metrics,
        "conformal_coverage": coverage,
        "avg_prediction_set_size": float(set_sizes.mean()),
        "out_of_domain_rate": out_of_domain,
    }
    with open(OUTPUT_DIR / "workflow_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    public_path = ROOT / "public_test_set.csv"
    if public_path.exists():
        public_rows = []
        with public_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row["activity_class"] = int(row["activity_class"])
                public_rows.append(row)
        overlap = set(zip([row["canonical_smiles"] for row in public_rows],
                          [row["target_common_name"] for row in public_rows])) & set(
            zip([row["canonical_smiles"] for row in df], [row["target_common_name"] for row in df])
        )
        if overlap:
            raise ValueError(f"Public test set overlaps training data: {overlap}")
        public_features, public_labels, _ = build_feature_matrix(
            public_rows,
            selected_columns=selected_columns,
        )
        public_probs = best_model.predict_proba(public_features)  # shape (n, 3)
        public_preds = best_model.predict(public_features)
        with (OUTPUT_DIR / "public_test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            prob_fields = [f"prob_{ACTIVITY_CLASS_MAP[c]}" for c in range(NUM_CLASSES)]
            fieldnames = list(public_rows[0].keys()) + prob_fields + ["predicted_class", "predicted_label"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row, probs_row, pred in zip(public_rows, public_probs, public_preds):
                output_row = dict(row)
                for c in range(NUM_CLASSES):
                    output_row[f"prob_{ACTIVITY_CLASS_MAP[c]}"] = float(probs_row[c])
                output_row["predicted_class"] = int(pred)
                output_row["predicted_label"] = ACTIVITY_CLASS_MAP.get(int(pred), "unknown")
                writer.writerow(output_row)
        if len(np.unique(public_labels)) > 1:
            public_roc = roc_auc_score(public_labels, public_probs, multi_class="ovr", average="macro")
        else:
            public_roc = float("nan")
        public_metrics = {
            "roc_auc_macro": public_roc,
            "mcc": matthews_corrcoef(public_labels, public_preds),
            "confusion_matrix": confusion_matrix(
                public_labels, public_preds, labels=list(range(NUM_CLASSES))
            ).tolist(),
        }
        with open(OUTPUT_DIR / "public_test_metrics.json", "w", encoding="utf-8") as handle:
            json.dump(public_metrics, handle, indent=2)


if __name__ == "__main__":
    run_workflow()
