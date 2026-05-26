#!/usr/bin/env python3
"""Run the OFFTOXv3 24-target safety-pharmacology workflow as a script.

This is a thin orchestrator over ``offtox_core`` (the single source of truth,
also used by ``safety_pharmacology_workflow.ipynb``). It produces every output
the notebook does: the eight PNG figures, ``workflow_summary.json``,
``test_set_predictions.csv``, ``standalone_predictions.csv`` and
``analysis_report.md`` (consumed downstream by ``generate_report.py``).

Examples
--------
    python run_pipeline.py                 # full dataset
    python run_pipeline.py --fast          # quick end-to-end smoke run
    python run_pipeline.py --no-automl      # skip Hyperopt-sklearn
    python run_pipeline.py --no-cdk         # skip PaDEL CDK fingerprints
"""

import argparse
import pickle
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (classification_report, matthews_corrcoef,
                             roc_auc_score, average_precision_score)

import offtox_core as C

warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser(description="OFFTOXv3 safety pharmacology pipeline")
    p.add_argument("--fast", action="store_true", help="quick smoke run (subsample + small search)")
    p.add_argument("--sample", type=int, default=None, help="cap number of training rows")
    p.add_argument("--no-automl", action="store_true", help="disable Hyperopt-sklearn AutoML")
    p.add_argument("--no-cdk", action="store_true", help="disable PaDEL CDK fingerprints")
    p.add_argument("--no-smote", action="store_true", help="disable SMOTE resampling")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = C.RunConfig(fast_mode=args.fast, sample_size=args.sample,
                      use_automl=not args.no_automl, use_smote=not args.no_smote)
    cfg.features.use_cdk = not args.no_cdk
    paths = C.get_paths()
    cfg.features.cache_dir = paths.cache
    rs = cfg.random_state

    print("=" * 70)
    print(f"OFFTOXv3 pipeline | fast={cfg.fast_mode} automl={cfg.use_automl and C.HAS_AUTOML} "
          f"cdk={cfg.features.use_cdk and C.HAS_PADEL} smote={cfg.use_smote and C.HAS_SMOTE}")
    print("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────
    data = C.load_and_clean_data(paths.data)
    data = C.subsample_rows(data, cfg.sample_size, rs)
    labels_all = np.array([r["activity_class"] for r in data], dtype=int)
    targets_all = [r.get("target_common_name", "unknown") for r in data]
    print(f"[1] Loaded {len(data)} records | "
          f"binding={int((labels_all==1).sum())} non_binding={int((labels_all==0).sum())}")

    # ── 2. Features (descriptors + Morgan + MACCS + CDK family) ───────
    print("[2] Building features ...")
    cdk_cache, cdk_names = None, None
    if cfg.features.use_cdk:
        try:
            uniq = list(dict.fromkeys(r["canonical_smiles"] for r in data))
            cdk_cache, cdk_names = C.compute_cdk_fingerprints(uniq, paths.cache)
        except Exception as exc:
            print(f"    WARNING: CDK fingerprints unavailable ({exc!r}) — disabling.")
            cfg.features.use_cdk = False
    features, labels, columns = C.build_feature_matrix(
        data, cfg.features, cdk_cache=cdk_cache, cdk_names=cdk_names, verbose=True)
    print(f"    feature matrix: {features.shape}")

    # ── 3. Scaffold split ─────────────────────────────────────────────
    smiles_all = [r["canonical_smiles"] for r in data]
    split = C.scaffold_split(smiles_all, rs)
    X_tr, y_tr = features[split.train_idx], labels[split.train_idx]
    X_va, y_va = features[split.val_idx], labels[split.val_idx]
    X_te, y_te = features[split.test_idx], labels[split.test_idx]
    smi_tr = [smiles_all[i] for i in split.train_idx]
    smi_va = [smiles_all[i] for i in split.val_idx]
    smi_te = [smiles_all[i] for i in split.test_idx]
    print(f"[3] Split train/val/test = {len(X_tr)}/{len(X_va)}/{len(X_te)}")

    # ── 4. Train base models ──────────────────────────────────────────
    res = C.train_models(X_tr, y_tr, smi_tr, X_va, y_va, smi_va, cfg)
    ranking = C.mcda_rank(res.cv_summary, res.calibration, res.train_times)
    best_name = ranking[0]["model"]
    best_model = res.best_estimators[best_name]
    consensus_members = [r["model"] for r in ranking[:3]]
    print(f"\n[4] Best model: {best_name} | consensus members: {consensus_members}")

    # ── 5. Refit best + ensemble on train+val ─────────────────────────
    X_comb = np.vstack([X_tr, X_va])
    y_comb = np.hstack([y_tr, y_va])
    smi_comb = smi_tr + smi_va
    for name, est in res.best_estimators.items():
        if name == "GNN":
            est.fit(smi_comb, y_comb)
        elif name == "AutoML":
            pass  # AutoML refit is expensive; keep the train-fit estimator
        else:
            est.fit(X_comb, y_comb)

    stacking = C.build_stacking_ensemble(res.best_estimators, X_comb, y_comb, cfg)
    if stacking is not None:
        res.best_estimators["Stacking"] = stacking

    # Isotonic-calibrated probabilities for tree models (CalibratedClassifierCV
    # clones + refits internally with cv=3). GNN/AutoML keep raw probabilities.
    calibrated = None
    if best_name in ("RandomForest", "XGBoost", "LightGBM"):
        calibrated = CalibratedClassifierCV(best_model, method="isotonic", cv=3)
        calibrated.fit(X_comb, y_comb)

    # ── 6. Internal test-set metrics ──────────────────────────────────
    if best_name == "GNN":
        test_probs = best_model.predict_proba(smi_te)
    elif calibrated is not None:
        test_probs = calibrated.predict_proba(X_te)
    else:
        test_probs = best_model.predict_proba(X_te)
    test_preds = test_probs.argmax(axis=1)
    test_roc = roc_auc_score(y_te, test_probs[:, 1])
    test_pr = average_precision_score(y_te, test_probs[:, 1])
    test_mcc = matthews_corrcoef(y_te, test_preds)
    ece, mce = C.ece_mce(y_te, test_probs[:, 1])
    print(f"[6] Test ROC={test_roc:.3f} PR={test_pr:.3f} MCC={test_mcc:.3f} "
          f"ECE={ece:.3f} MCE={mce:.3f}")
    print(classification_report(y_te, test_preds,
          target_names=[C.ACTIVITY_CLASS_MAP[c] for c in range(C.NUM_CLASSES)], digits=3))

    cv_sorted = sorted(res.cv_summary, key=lambda r: r["mcc_mean"], reverse=True)

    # ── 7. Uncertainty: conformal + applicability domain ──────────────
    cal_probs = test_probs
    pred_sets, coverage, q = C.conformal_prediction(cal_probs, y_te)
    set_sizes = pred_sets.sum(axis=1)
    nn, ad_threshold = C.fit_applicability_domain(X_tr)
    SEVERE_OOD = 100.0
    test_dists = nn.kneighbors(X_te)[0].mean(axis=1)
    ood_rate = float((test_dists > ad_threshold).mean())
    severe_ood_rate = float((test_dists > SEVERE_OOD).mean())
    print(f"[7] Conformal coverage={coverage:.3f} avg_set={set_sizes.mean():.2f} "
          f"OOD={ood_rate:.1%}")

    # ── 8. SAS cores ──────────────────────────────────────────────────
    train_rows = [data[i] for i in split.train_idx]
    sas_cores = C.compute_sas_cores(train_rows, y_tr)

    # ── 9. Figures ────────────────────────────────────────────────────
    print("[9] Rendering figures ...")
    import matplotlib.pyplot as plt
    pchembl_vals = [float(r["pchembl_value"]) for r in data if r.get("pchembl_value") is not None]
    C.plot_data_exploration(labels_all, targets_all, pchembl_vals, paths.output / "01_data_exploration.png")
    C.plot_roc(y_te, test_probs, best_name, paths.output / "02_roc_curves.png")
    C.plot_pr(y_te, test_probs, best_name, paths.output / "03_pr_curves.png")
    C.plot_confusion(y_te, test_preds, best_name, paths.output / "04_confusion_matrix.png")
    C.plot_calibration(y_te, cal_probs, best_name, paths.output / "05_calibration_curves.png")
    C.plot_feature_importance(best_model, columns, best_name, paths.output / "06_feature_importance.png")
    C.plot_uncertainty(set_sizes, coverage, test_dists, ad_threshold, SEVERE_OOD,
                       ood_rate, severe_ood_rate, test_probs.max(axis=1),
                       test_preds == y_te, ece, mce, paths.output / "07_uncertainty.png")
    plt.close("all")

    # ── 10. Save model artifacts ──────────────────────────────────────
    art = {
        "best_model": best_model,
        "calibrated_model": calibrated,
        "all_estimators": res.best_estimators,
        "consensus_members": consensus_members,
        "best_model_name": best_name,
        "selected_columns": columns,
        "feature_config": cfg.features,
        "cdk_cache": cdk_cache,
        "cdk_names": cdk_names,
        "activity_class_map": C.ACTIVITY_CLASS_MAP,
        "num_classes": C.NUM_CLASSES,
        "ad_threshold": ad_threshold,
        "severe_ood_threshold": SEVERE_OOD,
        "nn_model": nn,
        "conformal_q": q,
        "sas_cores": sas_cores,
        "target_panel": C.TARGET_PANEL,
        "mcda_ranking": [{"model": r["model"], "composite": r["composite"],
                          "mcc": r["mcc"], "roc_auc": r["roc_auc"]} for r in ranking],
    }
    model_path = paths.model / "safety_model.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump(art, fh)
    print(f"[10] Saved model -> {model_path}")

    # ── 11. Held-out test_compounds.csv evaluation ────────────────────
    ext = C.evaluate_test_compounds(paths.test, art, paths.output)
    if ext.get("available"):
        print(f"[11] Held-out n={ext['n']} ROC={ext['roc']:.3f} MCC={ext['mcc']:.3f} "
              f"acc={ext['acc']:.3f}")
        if ext.get("suspicious"):
            print(f"     suspicious MCC=1 targets (excluded): {ext['suspicious']}")

    # ── 12. Standalone predictions on withdrawn drugs ─────────────────
    withdrawn = paths.base / "data" / "withdrawn_terminated_drugs_2021_2026.csv"
    if withdrawn.exists():
        print("[12] Predicting novel (withdrawn) compounds ...")
        preds = C.predict_compounds(withdrawn, model_path)
        preds.to_csv(paths.output / "standalone_predictions.csv", index=False)
        print(f"     -> {paths.output / 'standalone_predictions.csv'} ({len(preds)} rows)")

    # ── 13. Statistics + summary + markdown report ────────────────────
    stat_rows, bonferroni = C.paired_ttests(cv_sorted, res.fold_scores)
    R = {
        "n_compounds": len(data), "targets": sorted(set(targets_all)),
        "class_distribution": {C.ACTIVITY_CLASS_MAP[c]: int((labels_all == c).sum())
                               for c in range(C.NUM_CLASSES)},
        "train_size": len(X_tr), "val_size": len(X_va), "test_size": len(X_te),
        "n_features": len(columns), "best_model": best_name,
        "consensus_members": consensus_members, "feature_blocks": C.feature_blocks(cfg),
        "test_metrics": {"roc_auc_macro": test_roc, "pr_auc_macro": test_pr,
                         "mcc": test_mcc, "ece": ece, "mce": mce},
        "conformal_coverage": coverage, "avg_prediction_set_size": float(set_sizes.mean()),
        "out_of_domain_rate": ood_rate, "smote_applied": bool(cfg.use_smote and C.HAS_SMOTE),
        "automl_used": bool(cfg.use_automl and C.HAS_AUTOML),
        "cv_sorted": cv_sorted, "y_test": y_te, "test_preds": test_preds,
        "ad_threshold": ad_threshold, "ext": ext, "stat_rows": stat_rows,
        "bonferroni": bonferroni, "ranking": ranking,
    }
    C.write_workflow_summary(paths.output / "workflow_summary.json", R)
    C.write_analysis_markdown(paths.output / "analysis_report.md", R)
    print("[13] Saved workflow_summary.json + analysis_report.md")

    print("\n" + "=" * 70)
    print(f"Pipeline complete. Outputs in {paths.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
