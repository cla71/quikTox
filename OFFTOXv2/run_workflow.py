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
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, MolSurf, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "safety_targets_bioactivity.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "workflow_outputs"
RANDOM_STATE = 42


@dataclass
class SplitData:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def load_and_clean_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["standard_relation"] == "="]
    df = df[df["pchembl_value"].notna()]
    df = df[df["pchembl_value"] >= 4.0]
    df = df[df["canonical_smiles"].notna()]
    df = df.sort_values("pchembl_value", ascending=False)
    df = df.drop_duplicates(subset=["molecule_chembl_id", "target_chembl_id"], keep="first")
    df = df.copy()
    df["is_active"] = (df["pchembl_value"] >= 6.0).astype(int)
    df = df.reset_index(drop=True)
    return df


def compute_descriptors(smiles: List[str]) -> pd.DataFrame:
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
            rows.append({name: np.nan for name in descriptor_functions})
            continue
        rows.append({name: func(mol) for name, func in descriptor_functions.items()})
    return pd.DataFrame(rows)


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


def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    descriptors = compute_descriptors(df["canonical_smiles"].tolist())
    fingerprints = compute_morgan_fingerprints(df["canonical_smiles"].tolist())
    fingerprint_df = pd.DataFrame(fingerprints, columns=[f"FP_{i}" for i in range(fingerprints.shape[1])])
    target_df = pd.get_dummies(df["target_common_name"], prefix="target")
    features = pd.concat([descriptors, fingerprint_df, target_df], axis=1)
    features = features.fillna(features.median())
    selector = VarianceThreshold(threshold=0.01)
    features_selected = selector.fit_transform(features)
    selected_columns = features.columns[selector.get_support()]
    return pd.DataFrame(features_selected, columns=selected_columns), df["is_active"].values


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
                ("model", RandomForestClassifier(random_state=random_state, n_jobs=-1)),
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
                    eval_metric="logloss",
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
                ("model", LGBMClassifier(random_state=random_state, n_jobs=-1, verbose=-1)),
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
    features, labels = build_feature_matrix(df)
    split = scaffold_split(df["canonical_smiles"].tolist(), labels, random_state=RANDOM_STATE)

    X_train = features.iloc[split.train_idx].values
    y_train = labels[split.train_idx]
    X_val = features.iloc[split.val_idx].values
    y_val = labels[split.val_idx]
    X_test = features.iloc[split.test_idx].values
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
            scoring="roc_auc",
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
            probs = estimator.predict_proba(X_te)[:, 1]
            preds = estimator.predict(X_te)
            scores.append(roc_auc_score(y_te, probs))
            pr_scores.append(average_precision_score(y_te, probs))
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

        val_probs = search.best_estimator_.predict_proba(X_val)[:, 1]
        calibration_metrics[name] = ece_score(y_val, val_probs)[0]

    cv_summary_df = pd.DataFrame(cv_summary).sort_values("roc_auc_mean", ascending=False)
    cv_summary_df.to_csv(OUTPUT_DIR / "cv_summary.csv", index=False)

    best_model_name = cv_summary_df.iloc[0]["model"]
    best_model = best_estimators[best_model_name]
    best_model.fit(np.vstack([X_train, X_val]), np.hstack([y_train, y_val]))

    test_probs = best_model.predict_proba(X_test)[:, 1]
    test_preds = best_model.predict(X_test)
    test_metrics = {
        "model": best_model_name,
        "roc_auc": roc_auc_score(y_test, test_probs),
        "pr_auc": average_precision_score(y_test, test_probs),
        "mcc": matthews_corrcoef(y_test, test_preds),
    }

    calibrated = CalibratedClassifierCV(best_model, method="isotonic", cv=3)
    calibrated.fit(np.vstack([X_train, X_val]), np.hstack([y_train, y_val]))
    calibrated_probs = calibrated.predict_proba(X_test)[:, 1]
    ece, mce = ece_score(y_test, calibrated_probs)
    test_metrics["ece"] = ece
    test_metrics["mce"] = mce

    pd.DataFrame([test_metrics]).to_csv(OUTPUT_DIR / "test_metrics.csv", index=False)

    fpr, tpr, _ = roc_curve(y_test, test_probs)
    pr_precision, pr_recall, _ = precision_recall_curve(y_test, test_probs)
    cm = confusion_matrix(y_test, test_preds)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{best_model_name} (AUC={test_metrics['roc_auc']:.2f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(pr_recall, pr_precision, label=f"{best_model_name} (AP={test_metrics['pr_auc']:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pr_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=300)
    plt.close()

    prob_true, prob_pred = calibration_curve(y_test, calibrated_probs, n_bins=10)
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker="o", label="Calibrated")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "calibration_curve.png", dpi=300)
    plt.close()

    if hasattr(best_model.named_steps["model"], "feature_importances_"):
        importances = best_model.named_steps["model"].feature_importances_
        indices = np.argsort(importances)[-20:]
        plt.figure(figsize=(8, 6))
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), features.columns[indices])
        plt.title("Top 20 Feature Importances")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=300)
        plt.close()

    calibration_probs = calibrated.predict_proba(X_test)
    prediction_sets, coverage = conformal_prediction(calibration_probs, y_test)
    set_sizes = prediction_sets.sum(axis=1)
    pd.DataFrame({"set_size": set_sizes}).to_csv(OUTPUT_DIR / "conformal_set_sizes.csv", index=False)
    with open(OUTPUT_DIR / "conformal_summary.json", "w", encoding="utf-8") as handle:
        json.dump({"coverage": coverage, "avg_set_size": float(set_sizes.mean())}, handle, indent=2)

    plt.figure(figsize=(6, 4))
    sns.countplot(x=set_sizes)
    plt.title("Conformal Prediction Set Sizes")
    plt.xlabel("Set Size")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "conformal_set_sizes.png", dpi=300)
    plt.close()

    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(X_train)
    distances, _ = nn.kneighbors(X_train)
    train_dist = distances.mean(axis=1)
    threshold = np.percentile(train_dist, 95)
    test_distances, _ = nn.kneighbors(X_test)
    test_dist = test_distances.mean(axis=1)
    out_of_domain = (test_dist > threshold).mean()

    stats_rows = []
    models_list = cv_summary_df["model"].tolist()
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

    stats_df = pd.DataFrame(stats_rows)
    if not stats_df.empty:
        bonferroni = 0.05 / len(stats_df)
        stats_df["bonferroni_alpha"] = bonferroni
    stats_df.to_csv(OUTPUT_DIR / "statistical_comparison.csv", index=False)

    mcda_rows = []
    for _, row in cv_summary_df.iterrows():
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

    mcda_df = pd.DataFrame(mcda_rows)
    weights = {
        "roc_auc": 0.25,
        "pr_auc": 0.20,
        "calibration": 0.20,
        "robustness": 0.15,
        "efficiency": 0.10,
        "interpretability": 0.10,
    }
    for metric in weights:
        min_val = mcda_df[metric].min()
        max_val = mcda_df[metric].max()
        if max_val > min_val:
            mcda_df[metric] = (mcda_df[metric] - min_val) / (max_val - min_val)
        else:
            mcda_df[metric] = 1.0
    mcda_df["composite_score"] = sum(mcda_df[m] * w for m, w in weights.items())
    mcda_df = mcda_df.sort_values("composite_score", ascending=False)
    mcda_df.to_csv(OUTPUT_DIR / "mcda_ranking.csv", index=False)

    summary = {
        "n_compounds": len(df),
        "targets": df["target_common_name"].nunique(),
        "actives": int(df["is_active"].sum()),
        "inactives": int((df["is_active"] == 0).sum()),
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


if __name__ == "__main__":
    run_workflow()
