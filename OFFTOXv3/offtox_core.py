#!/usr/bin/env python3
"""OFFTOXv3 core engine — single source of truth for the 24-target safety
pharmacology / NHR binding workflow.

Both ``safety_pharmacology_workflow.ipynb`` and ``run_pipeline.py`` import from
this module so the analysis logic lives in exactly one place.

Pipeline summary
----------------
* 2-class binding scheme: binding (pChEMBL >= 5.0, < 10 uM) vs non_binding.
* Diverse feature representation: RDKit physicochemical descriptors +
  Morgan/ECFP + MACCS keys + CDK, CDK-extended and Substructure-count
  fingerprints (PaDEL). CDK fingerprints are cached to disk (computed once).
* Diverse model ensemble: RandomForest, XGBoost, LightGBM, a Weisfeiler-Lehman
  graph classifier (GNN proxy) and a Hyperopt-sklearn AutoML model, combined
  through a stacking ensemble and a soft-voting consensus.
* Class-imbalance handling (class weights + optional SMOTE), scaffold split,
  isotonic calibration, conformal prediction and applicability domain.

Optional heavy dependencies degrade loudly, never silently:
* ``padelpy`` + a Java runtime  -> CDK / CDK-extended / Substructure-count FPs.
* ``hpsklearn`` + ``hyperopt``   -> AutoML model.
If either is missing the pipeline still runs and prints a clear warning.
"""

from __future__ import annotations

import csv
import pickle
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")  # silence per-molecule parse/sanitize log spam
from rdkit.Chem import (
    Crippen,
    Descriptors,
    Lipinski,
    MACCSkeys,
    MolSurf,
    rdFingerprintGenerator,
)
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.DataStructs import TanimotoSimilarity

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# ── Optional dependencies (loud, not silent, when missing) ────────────────
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    HAS_SMOTE = True
except ImportError:  # pragma: no cover
    from sklearn.pipeline import Pipeline as ImbPipeline  # type: ignore

    SMOTE = None  # type: ignore
    HAS_SMOTE = False

try:
    from padelpy import padeldescriptor

    HAS_PADEL = True
except ImportError:  # pragma: no cover
    padeldescriptor = None  # type: ignore
    HAS_PADEL = False

try:
    from hpsklearn import HyperoptEstimator, any_classifier
    from hyperopt import tpe

    HAS_AUTOML = True
except Exception:  # pragma: no cover  (hyperopt import can fail, not just ImportError)
    HyperoptEstimator = None  # type: ignore
    HAS_AUTOML = False


# ══════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════

RANDOM_STATE = 42
NUM_CLASSES = 2
ACTIVITY_CLASS_MAP = {0: "non_binding", 1: "binding"}
CLASS_COLORS = {0: "#2ecc71", 1: "#e74c3c"}
CHEMBL_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"

# Activity definitions (kept explicit to avoid the mislabelling bug where the
# binding boundary was reported as 100 uM instead of 10 uM):
BINDING_PCHEMBL = 5.0          # pChEMBL >= 5.0  <=> IC50/Ki < 10 uM  -> binding
BINDING_THRESHOLD_UM = 10.0    # potency boundary for the *binding* class
DATA_PCHEMBL_FLOOR = 4.0       # outer data boundary (100 uM) for ingestion
DATA_BOUNDARY_UM = 100.0

TARGET_PANEL: Dict[str, Dict[str, str]] = {
    # Nuclear Hormone Receptors (14)
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
    # Cardiac Safety (3)
    "hERG":    {"chembl_id": "CHEMBL240",  "category": "Cardiac Safety"},
    "Cav1.2":  {"chembl_id": "CHEMBL1940", "category": "Cardiac Safety"},
    "Nav1.5":  {"chembl_id": "CHEMBL1993", "category": "Cardiac Safety"},
    # Hepatotoxicity / CYP (5)
    "CYP3A4":  {"chembl_id": "CHEMBL340",  "category": "Hepatotoxicity"},
    "CYP2D6":  {"chembl_id": "CHEMBL289",  "category": "Hepatotoxicity"},
    "CYP2C9":  {"chembl_id": "CHEMBL3397", "category": "Hepatotoxicity"},
    "CYP1A2":  {"chembl_id": "CHEMBL3356", "category": "Hepatotoxicity"},
    "CYP2C19": {"chembl_id": "CHEMBL3622", "category": "Hepatotoxicity"},
    # Transporters (2)
    "P-gp":    {"chembl_id": "CHEMBL4302", "category": "Transporter"},
    "BSEP":    {"chembl_id": "CHEMBL4105", "category": "Transporter"},
}


@dataclass
class Paths:
    base: Path
    data: Path
    test: Path
    output: Path
    model: Path
    cache: Path


def get_paths(base_dir: Optional[Path] = None) -> Paths:
    """Resolve the standard project paths, creating output dirs as needed."""
    base = Path(base_dir) if base_dir else Path(__file__).resolve().parent
    paths = Paths(
        base=base,
        data=base / "data" / "safety_targets_bioactivity.csv",
        test=base / "data" / "test_compounds.csv",
        output=base / "outputs",
        model=base / "model",
        cache=base / "data" / "cache",
    )
    for d in (paths.output, paths.model, paths.cache):
        d.mkdir(parents=True, exist_ok=True)
    return paths


@dataclass
class FeatureConfig:
    """Which molecular representations to combine into the feature matrix."""
    use_descriptors: bool = True
    use_morgan: bool = True
    morgan_bits: int = 2048
    morgan_radius: int = 2
    use_maccs: bool = True
    use_cdk: bool = True          # CDK + CDK-extended + Substructure-count (PaDEL)
    variance_threshold: float = 1e-4
    cache_dir: Optional[Path] = None


@dataclass
class RunConfig:
    """Knobs controlling cost vs fidelity. ``fast_mode`` shrinks everything so
    the whole workflow runs end-to-end in a couple of minutes for testing."""
    fast_mode: bool = False
    sample_size: Optional[int] = None     # cap rows (None = all); auto-set in fast_mode
    use_smote: bool = True
    use_automl: bool = True
    automl_max_evals: int = 15
    automl_trial_timeout: int = 120
    search_n_iter: int = 8                 # RandomizedSearchCV iterations per model
    wl_buckets: int = 1024                 # WL/GNN hash buckets (memory vs detail)
    wl_iterations: int = 3
    random_state: int = RANDOM_STATE
    features: FeatureConfig = field(default_factory=FeatureConfig)

    def __post_init__(self):
        if self.fast_mode:
            self.sample_size = self.sample_size or 2500
            self.automl_max_evals = min(self.automl_max_evals, 5)
            self.automl_trial_timeout = min(self.automl_trial_timeout, 45)
            self.search_n_iter = min(self.search_n_iter, 4)
            self.wl_buckets = min(self.wl_buckets, 512)


# ══════════════════════════════════════════════════════════════════════════
# SMILES standardisation & data loading
# ══════════════════════════════════════════════════════════════════════════

_largest_fragment = rdMolStandardize.LargestFragmentChooser()
_uncharger = rdMolStandardize.Uncharger()


def standardize_smiles(smi: Optional[str]) -> Optional[str]:
    """Cleanup -> largest fragment -> uncharge -> canonical SMILES.

    None if the SMILES is unparseable. Uses the version-stable functional API
    (``Cleanup``) rather than the ``Standardizer`` class removed in newer RDKit.
    """
    if not smi or not isinstance(smi, str):
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = _largest_fragment.choose(mol)
        mol = _uncharger.uncharge(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return Chem.MolToSmiles(mol)


def load_and_clean_data(path: Path) -> List[dict]:
    """Load the bioactivity CSV and assign 2-class activity labels.

    1 = binding     : pChEMBL >= 5.0 (< 10 uM)
    0 = non_binding : pChEMBL < 5.0, or a record flagged inactive/non_binding.
    Records are deduplicated per (molecule, target), keeping the most potent.
    """
    rows: List[dict] = []
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            smi = row.get("canonical_smiles")
            if not smi:
                continue

            raw_class = row.get("activity_class", "")
            label = row.get("activity_class_label", "")
            if raw_class == "0" or label in ("inactive", "non_binding"):
                row["pchembl_value"] = None
                row["activity_class"] = 0
                rows.append(row)
                continue

            if row.get("standard_relation") != "=" or not row.get("pchembl_value"):
                continue
            try:
                pchembl = float(row["pchembl_value"])
            except ValueError:
                continue
            if pchembl < DATA_PCHEMBL_FLOOR:
                continue
            row["pchembl_value"] = pchembl
            row["activity_class"] = 1 if pchembl >= BINDING_PCHEMBL else 0
            rows.append(row)

    deduped: dict = {}
    for row in rows:
        key = (row.get("molecule_chembl_id"), row.get("target_chembl_id"))
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = row
        else:
            ep, cp = existing.get("pchembl_value"), row.get("pchembl_value")
            if cp is not None and (ep is None or cp > ep):
                deduped[key] = row
    return list(deduped.values())


def subsample_rows(rows: List[dict], n: int, random_state: int) -> List[dict]:
    """Stratified-ish subsample preserving the binding/non_binding ratio."""
    if n is None or len(rows) <= n:
        return rows
    rng = np.random.default_rng(random_state)
    labels = np.array([r.get("activity_class", 0) for r in rows])
    keep: List[int] = []
    for cls in (0, 1):
        idx = np.where(labels == cls)[0]
        take = max(1, round(n * len(idx) / len(rows)))
        keep.extend(rng.choice(idx, size=min(take, len(idx)), replace=False).tolist())
    rng.shuffle(keep)
    return [rows[i] for i in keep]


# ══════════════════════════════════════════════════════════════════════════
# ChEMBL retrieval (optional refresh of the bioactivity CSV)
# ══════════════════════════════════════════════════════════════════════════

_CSV_FIELDS = [
    "molecule_chembl_id", "canonical_smiles", "standard_type", "standard_relation",
    "standard_value", "standard_units", "pchembl_value", "activity_comment",
    "assay_chembl_id", "assay_type", "target_chembl_id", "target_pref_name",
    "document_chembl_id", "src_id", "data_validity_comment", "safety_category",
    "target_common_name", "activity_class", "activity_class_label",
]


def _chembl_get(params, retries=4):
    import requests
    for attempt in range(retries):
        try:
            resp = requests.get(f"{CHEMBL_BASE_URL}/activity.json", params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
    return None


def fetch_chembl_activities(target_chembl_id, target_name, category,
                            activity_types=("IC50", "Ki"), pchembl_min=DATA_PCHEMBL_FLOOR,
                            max_records=2000):
    """Potent + low-potent binders (pChEMBL >= 4.0). Labels assigned by caller."""
    records, limit = [], 600
    for act_type in activity_types:
        offset = 0
        while offset < max_records:
            data = _chembl_get({
                "target_chembl_id": target_chembl_id, "standard_type": act_type,
                "pchembl_value__gte": pchembl_min, "standard_relation": "=",
                "limit": limit, "offset": offset, "format": "json"})
            acts = (data or {}).get("activities", [])
            if not acts:
                break
            for a in acts:
                smi, pval = a.get("canonical_smiles"), a.get("pchembl_value")
                if not smi or not pval:
                    continue
                records.append({"molecule_chembl_id": a.get("molecule_chembl_id", ""),
                                "canonical_smiles": smi, "standard_type": a.get("standard_type", act_type),
                                "standard_relation": "=", "standard_value": a.get("standard_value", ""),
                                "standard_units": a.get("standard_units", "nM"), "pchembl_value": pval,
                                "assay_chembl_id": a.get("assay_chembl_id", ""),
                                "target_chembl_id": target_chembl_id,
                                "safety_category": category, "target_common_name": target_name})
            if len(acts) < limit:
                break
            offset += limit
            time.sleep(0.5)
    return records


def fetch_chembl_inactives(target_chembl_id, target_name, category, min_inactive=50,
                           max_records=2000):
    """Confirmed inactives: right-censored (>) records at >= 100 uM -> non_binding."""
    inactive, limit, offset, seen = [], 500, 0, set()
    while offset < max_records and len(inactive) < min_inactive * 3:
        data = _chembl_get({"target_chembl_id": target_chembl_id, "standard_relation": ">",
                            "standard_type__in": "IC50,Ki", "limit": limit,
                            "offset": offset, "format": "json"})
        acts = (data or {}).get("activities", [])
        if not acts:
            break
        for a in acts:
            smi, val, mid = a.get("canonical_smiles"), a.get("standard_value"), a.get("molecule_chembl_id", "")
            if not smi or mid in seen:
                continue
            try:
                if float(val) >= DATA_BOUNDARY_UM * 1000:  # uM -> nM
                    seen.add(mid)
                    inactive.append({"molecule_chembl_id": mid, "canonical_smiles": smi,
                                     "standard_type": a.get("standard_type", ""), "standard_relation": ">",
                                     "standard_value": val, "standard_units": "nM",
                                     "activity_comment": "Confirmed inactive (> 100 uM)",
                                     "target_chembl_id": target_chembl_id, "safety_category": category,
                                     "target_common_name": target_name,
                                     "activity_class": "0", "activity_class_label": "non_binding"})
            except (ValueError, TypeError):
                continue
        if len(acts) < limit:
            break
        offset += limit
        time.sleep(0.5)
    return inactive[:min_inactive + 10]


def fetch_chembl_broad_nonbinders(target_chembl_id, target_name, category, min_count=40):
    """CEREP-style 'Not Active' / 'Inactive' comment records -> diverse non_binders."""
    nonbinders, seen = [], set()
    for comment in ("Not Active", "inactive", "Inactive", "not active"):
        if len(nonbinders) >= min_count * 2:
            break
        data = _chembl_get({"target_chembl_id": target_chembl_id,
                            "activity_comment__icontains": comment, "limit": 400,
                            "offset": 0, "format": "json"}, retries=3)
        for a in (data or {}).get("activities", []):
            smi, mid = a.get("canonical_smiles"), a.get("molecule_chembl_id", "")
            if not smi or mid in seen:
                continue
            seen.add(mid)
            nonbinders.append({"molecule_chembl_id": mid, "canonical_smiles": smi,
                               "standard_type": a.get("standard_type", ""),
                               "activity_comment": a.get("activity_comment", comment),
                               "target_chembl_id": target_chembl_id, "safety_category": category,
                               "target_common_name": target_name,
                               "activity_class": "0", "activity_class_label": "non_binding"})
        time.sleep(0.3)
    return nonbinders[:min_count]


def refresh_chembl_dataset(out_path: Path, panel=TARGET_PANEL, verbose=True) -> int:
    """Pull the full 24-target panel from ChEMBL and write the bioactivity CSV.

    Strategy: pChEMBL >= 5.0 -> binding; 4.0-5.0 (10-100 uM) -> non_binding;
    plus confirmed inactives (> 100 uM) and broad-screen 'Not Active' decoys.
    """
    all_rows = []
    for name, info in panel.items():
        cid, cat = info["chembl_id"], info["category"]
        if verbose:
            print(f"  {name} ({cid}) ...")
        active = fetch_chembl_activities(cid, name, cat)
        for r in active:
            try:
                p = float(r["pchembl_value"])
            except (ValueError, TypeError):
                continue
            r["activity_class"] = "1" if p >= BINDING_PCHEMBL else "0"
            r["activity_class_label"] = "binding" if p >= BINDING_PCHEMBL else "non_binding"
        all_rows += active
        all_rows += fetch_chembl_inactives(cid, name, cat)
        all_rows += fetch_chembl_broad_nonbinders(cid, name, cat)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in _CSV_FIELDS})
    if verbose:
        print(f"  Saved {len(all_rows)} records -> {out_path}")
    return len(all_rows)


# ══════════════════════════════════════════════════════════════════════════
# Molecular features (descriptors + multiple fingerprint families)
# ══════════════════════════════════════════════════════════════════════════

_DESCRIPTORS = {
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
    "NumRings": Descriptors.RingCount,
    "NumHeteroatoms": Lipinski.NumHeteroatoms,
    "NHOHCount": Lipinski.NHOHCount,
    "NOCount": Lipinski.NOCount,
    "QED": Descriptors.qed,
    "BalabanJ": Descriptors.BalabanJ,
    "BertzCT": Descriptors.BertzCT,
}


def compute_descriptors(smiles: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Physicochemical descriptor block (one row per molecule)."""
    out = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            out.append([np.nan] * len(_DESCRIPTORS))
        else:
            row = []
            for func in _DESCRIPTORS.values():
                try:
                    row.append(float(func(mol)))
                except Exception:
                    row.append(np.nan)
            out.append(row)
    return np.asarray(out, dtype=np.float32), list(_DESCRIPTORS.keys())


def compute_morgan(smiles: List[str], n_bits: int, radius: int) -> Tuple[np.ndarray, List[str]]:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        fps.append(np.zeros(n_bits, dtype=np.float32) if mol is None
                   else np.asarray(gen.GetFingerprint(mol), dtype=np.float32))
    names = [f"Morgan_{i}" for i in range(n_bits)]
    return np.asarray(fps, dtype=np.float32), names


def compute_maccs(smiles: List[str]) -> Tuple[np.ndarray, List[str]]:
    fps = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        fps.append(np.zeros(167, dtype=np.float32) if mol is None
                   else np.asarray(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32))
    names = [f"MACCS_{i}" for i in range(167)]
    return np.asarray(fps, dtype=np.float32), names


# ── CDK / CDK-extended / Substructure-count fingerprints (PaDEL, cached) ───

_CDK_XML = """<?xml version="1.0" ?>
<Root>
  <Group name="Fingerprint">
    <Descriptor name="Fingerprinter" value="true"/>
    <Descriptor name="ExtendedFingerprinter" value="true"/>
    <Descriptor name="SubstructureFingerprintCount" value="true"/>
  </Group>
</Root>
"""


def _cdk_cache_files(cache_dir: Path) -> Tuple[Path, Path]:
    return cache_dir / "cdk_fp_cache.pkl", cache_dir / "cdk_fp_names.json"


def compute_cdk_fingerprints(
    unique_smiles: List[str], cache_dir: Path, verbose: bool = True
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """CDK (1024) + CDK-extended (1024) + Substructure-count (307) via PaDEL.

    Returns ``({smiles: vector}, column_names)``. Results are cached to
    ``cache_dir`` so each unique SMILES is only ever computed once. If PaDEL or
    Java is unavailable this raises ``RuntimeError`` (callers decide whether to
    skip CDK features) — it never silently returns empty data.
    """
    import json

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path, names_path = _cdk_cache_files(cache_dir)

    cache: Dict[str, np.ndarray] = {}
    names: List[str] = []
    if cache_path.exists() and names_path.exists():
        with open(cache_path, "rb") as fh:
            cache = pickle.load(fh)
        names = json.loads(names_path.read_text())

    todo = [s for s in dict.fromkeys(unique_smiles) if s and s not in cache]
    if todo:
        if not HAS_PADEL:
            raise RuntimeError("padelpy not installed — cannot compute CDK fingerprints.")
        if verbose:
            print(f"  CDK fingerprints: computing {len(todo)} new SMILES "
                  f"({len(cache)} cached) via PaDEL ...")
        new_names = _run_padel(todo, cache, cache_dir, verbose=verbose)
        if new_names:
            names = new_names
        with open(cache_path, "wb") as fh:
            pickle.dump(cache, fh)
        names_path.write_text(json.dumps(names))
    elif verbose:
        print(f"  CDK fingerprints: all {len(set(unique_smiles))} SMILES served from cache.")

    if not names and cache:
        names = [f"CDK_{i}" for i in range(len(next(iter(cache.values()))))]
    return cache, names


def _run_padel(todo: List[str], cache: Dict[str, np.ndarray], cache_dir: Path,
               verbose: bool = True) -> List[str]:
    """Run PaDEL on the ``todo`` SMILES, updating ``cache`` in place."""
    import tempfile

    import pandas as pd

    xml_path = cache_dir / "cdk_fp_types.xml"
    xml_path.write_text(_CDK_XML)

    names: List[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        smi_path = Path(tmp) / "input.smi"
        out_path = Path(tmp) / "output.csv"
        id_to_smi = {}
        with open(smi_path, "w") as fh:
            for i, smi in enumerate(todo):
                mol_id = f"M{i}"
                id_to_smi[mol_id] = smi
                fh.write(f"{smi}\t{mol_id}\n")

        t0 = time.time()
        padeldescriptor(
            mol_dir=str(smi_path), d_file=str(out_path),
            descriptortypes=str(xml_path), fingerprints=True,
            retainorder=True, removesalt=True, standardizenitro=True,
            threads=-1, sp_timeout=3600,
        )
        res = pd.read_csv(out_path)
        names = [c for c in res.columns if c != "Name"]
        res = res.set_index("Name")
        for mol_id, smi in id_to_smi.items():
            if mol_id in res.index:
                vec = res.loc[mol_id, names].to_numpy(dtype=np.float32)
                vec = np.nan_to_num(vec, nan=0.0)
            else:
                vec = np.zeros(len(names), dtype=np.float32)
            cache[smi] = vec
        if verbose:
            print(f"    PaDEL done: {len(todo)} SMILES, {len(names)} columns "
                  f"in {time.time() - t0:.1f}s")
    return names


def ensure_cdk_cache(smiles_list, cdk_cache, cdk_names, feat_cfg, verbose=True):
    """Top up the CDK fingerprint cache with any SMILES not seen at train time.

    Returns the (possibly extended) ``(cache, names)``. If PaDEL is unavailable
    the missing rows fall back to zeros — and we say so loudly, never silently.
    """
    if not feat_cfg.use_cdk or cdk_cache is None:
        return cdk_cache, cdk_names
    missing = [s for s in dict.fromkeys(smiles_list) if s and s not in cdk_cache]
    if not missing:
        return cdk_cache, cdk_names
    cache_dir = feat_cfg.cache_dir or (Path(__file__).resolve().parent / "data" / "cache")
    if HAS_PADEL:
        return compute_cdk_fingerprints(list(cdk_cache.keys()) + missing, cache_dir, verbose=verbose)
    if verbose:
        print(f"  WARNING: {len(missing)} SMILES lack CDK fingerprints and PaDEL is "
              f"unavailable — those columns default to 0 for those compounds.")
    return cdk_cache, cdk_names


def build_feature_matrix(
    rows: List[dict],
    cfg: FeatureConfig,
    selected_columns: Optional[List[str]] = None,
    cdk_cache: Optional[Dict[str, np.ndarray]] = None,
    cdk_names: Optional[List[str]] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Assemble the combined feature matrix.

    Training mode (``selected_columns is None``): build everything, drop
    near-constant columns, return the surviving column names.
    Prediction mode: align to ``selected_columns`` exactly.
    """
    smiles = [r["canonical_smiles"] for r in rows]
    targets = [r.get("target_common_name", r.get("target", "")) for r in rows]
    labels = np.array([r.get("activity_class", -1) for r in rows], dtype=int)

    blocks, columns = [], []

    if cfg.use_descriptors:
        d, dn = compute_descriptors(smiles)
        blocks.append(d)
        columns += dn
    if cfg.use_morgan:
        m, mn = compute_morgan(smiles, cfg.morgan_bits, cfg.morgan_radius)
        blocks.append(m)
        columns += mn
    if cfg.use_maccs:
        k, kn = compute_maccs(smiles)
        blocks.append(k)
        columns += kn
    if cfg.use_cdk:
        if cdk_cache is None:
            cache_dir = cfg.cache_dir or (Path(__file__).resolve().parent / "data" / "cache")
            cdk_cache, cdk_names = compute_cdk_fingerprints(smiles, cache_dir, verbose=verbose)
        n_cdk = len(cdk_names) if cdk_names else (
            len(next(iter(cdk_cache.values()))) if cdk_cache else 0)
        cdk_names = cdk_names or [f"CDK_{i}" for i in range(n_cdk)]
        zero = np.zeros(n_cdk, dtype=np.float32)
        cdk_block = np.vstack([cdk_cache.get(s, zero) for s in smiles]).astype(np.float32)
        blocks.append(cdk_block)
        columns += list(cdk_names)

    # Target one-hot encoding
    target_names = sorted({t for t in targets if t})
    target_idx = {n: i for i, n in enumerate(target_names)}
    target_block = np.zeros((len(rows), len(target_names)), dtype=np.float32)
    for i, t in enumerate(targets):
        if t in target_idx:
            target_block[i, target_idx[t]] = 1.0
    blocks.append(target_block)
    columns += [f"target_{n}" for n in target_names]

    matrix = np.concatenate(blocks, axis=1).astype(np.float32)

    if selected_columns is None:
        matrix = np.nan_to_num(matrix, nan=0.0)
        variances = matrix.var(axis=0)
        mask = variances > cfg.variance_threshold
        matrix = matrix[:, mask]
        selected_columns = [c for c, keep in zip(columns, mask) if keep]
    else:
        col_index = {c: i for i, c in enumerate(columns)}
        aligned = np.zeros((len(rows), len(selected_columns)), dtype=np.float32)
        for j, col in enumerate(selected_columns):
            if col in col_index:
                aligned[:, j] = np.nan_to_num(matrix[:, col_index[col]], nan=0.0)
        matrix = aligned

    return matrix, labels, selected_columns


# ══════════════════════════════════════════════════════════════════════════
# Scaffold split & metrics
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class SplitData:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def scaffold_split(smiles: List[str], random_state: int = RANDOM_STATE) -> SplitData:
    """60/20/20 split grouped by Bemis-Murcko scaffold (no scaffold leakage)."""
    scaffolds: Dict[str, List[int]] = {}
    for idx, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        scaffold = "" if mol is None else MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        scaffolds.setdefault(scaffold, []).append(idx)

    groups = sorted(scaffolds.values(), key=len, reverse=True)
    rng = np.random.default_rng(random_state)
    rng.shuffle(groups)

    n = len(smiles)
    n_train, n_val = int(0.6 * n), int(0.2 * n)
    train, val, test = [], [], []
    for g in groups:
        if len(train) + len(g) <= n_train:
            train.extend(g)
        elif len(val) + len(g) <= n_val:
            val.extend(g)
        else:
            test.extend(g)
    return SplitData(np.array(train), np.array(val), np.array(test))


def ece_mce(y_true: np.ndarray, prob_pos: np.ndarray, n_bins: int = 10) -> Tuple[float, float]:
    """Standard Expected/Maximum Calibration Error for the positive class.

    ``y_true`` is the binary outcome (1 = binding), ``prob_pos`` the predicted
    probability of binding. Each bin compares mean predicted probability with
    the empirical binding rate.
    """
    y_true = np.asarray(y_true, dtype=float)
    prob_pos = np.asarray(prob_pos, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = mce = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (prob_pos >= lo) & (prob_pos < hi) if i < n_bins - 1 else \
               (prob_pos >= lo) & (prob_pos <= hi)
        if not mask.any():
            continue
        gap = abs(prob_pos[mask].mean() - y_true[mask].mean())
        ece += gap * (mask.sum() / n)
        mce = max(mce, gap)
    return float(ece), float(mce)


# ══════════════════════════════════════════════════════════════════════════
# Models
# ══════════════════════════════════════════════════════════════════════════

class WLGraphClassifier(BaseEstimator, ClassifierMixin):
    """Weisfeiler-Lehman graph-kernel + logistic regression (GNN proxy).

    Captures molecular topology through k rounds of neighbourhood hashing, a
    message-passing GNN analogue that needs only RDKit + scikit-learn.
    """

    def __init__(self, n_iterations: int = 3, n_hash_buckets: int = 1024,
                 C: float = 1.0, class_weight="balanced", random_state: int = RANDOM_STATE):
        self.n_iterations = n_iterations
        self.n_hash_buckets = n_hash_buckets
        self.C = C
        self.class_weight = class_weight
        self.random_state = random_state

    def _vector(self, smi: str) -> np.ndarray:
        width = self.n_hash_buckets * (self.n_iterations + 1)
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            return np.zeros(width, dtype=np.float32)

        labels = {}
        for atom in mol.GetAtoms():
            key = (atom.GetAtomicNum(), atom.GetTotalNumHs(), atom.GetDegree(),
                   int(atom.GetIsAromatic()), int(atom.IsInRing()), atom.GetFormalCharge())
            labels[atom.GetIdx()] = hash(key) % self.n_hash_buckets

        adj: Dict[int, List[int]] = {a.GetIdx(): [] for a in mol.GetAtoms()}
        for bond in mol.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            adj[u].append(v)
            adj[v].append(u)

        feat = np.zeros(width, dtype=np.float32)
        for lbl in labels.values():
            feat[lbl] += 1
        for it in range(1, self.n_iterations + 1):
            new = {}
            for idx in adj:
                agg = (labels[idx],) + tuple(sorted(labels[nb] for nb in adj[idx]))
                new[idx] = hash(agg) % self.n_hash_buckets
            labels = new
            off = it * self.n_hash_buckets
            for lbl in labels.values():
                feat[off + lbl] += 1
        return feat

    def _transform(self, smiles_list: List[str]) -> np.ndarray:
        X = np.vstack([self._vector(s) for s in smiles_list])
        return normalize(X, norm="l2")

    def fit(self, smiles_list, y):
        self.classes_ = np.unique(y)
        self._clf = LogisticRegression(
            C=self.C, class_weight=self.class_weight, max_iter=1000,
            random_state=self.random_state, solver="saga")
        self._clf.fit(self._transform(smiles_list), y)
        return self

    def predict(self, smiles_list):
        return self._clf.predict(self._transform(smiles_list))

    def predict_proba(self, smiles_list):
        return self._clf.predict_proba(self._transform(smiles_list))


class AutoMLClassifier(BaseEstimator, ClassifierMixin):
    """Thin wrapper around Hyperopt-sklearn's ``HyperoptEstimator``.

    Searches ~40 algorithms automatically, optimising balanced accuracy (robust
    to the binding/non_binding imbalance). To stay tractable on the wide
    combined-fingerprint matrix, the features are first scaled and reduced with
    Truncated SVD (when high-dimensional) before the search. Always yields a
    probability-like output so it can join the ensemble.
    """

    def __init__(self, max_evals: int = 15, trial_timeout: int = 120,
                 n_components: int = 60, random_state: int = RANDOM_STATE):
        self.max_evals = max_evals
        self.trial_timeout = trial_timeout
        self.n_components = n_components
        self.random_state = random_state

    @staticmethod
    def _loss(y_true, y_pred):
        return 1.0 - balanced_accuracy_score(y_true, y_pred)

    def _reduce_fit(self, X):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.pipeline import Pipeline as SkPipeline

        if X.shape[1] <= self.n_components:
            self._reducer = StandardScaler(with_mean=False)
        else:
            self._reducer = SkPipeline([
                ("scale", StandardScaler(with_mean=False)),
                ("svd", TruncatedSVD(n_components=self.n_components,
                                     random_state=self.random_state))])
        return self._reducer.fit_transform(X)

    def fit(self, X, y):
        if not HAS_AUTOML:
            raise RuntimeError("hpsklearn/hyperopt not installed — AutoML unavailable.")
        self.classes_ = np.unique(y)
        Xr = self._reduce_fit(np.asarray(X, dtype=float))
        self._est = HyperoptEstimator(
            classifier=any_classifier("automl_clf"), preprocessing=[],
            algo=tpe.suggest, max_evals=self.max_evals,
            trial_timeout=self.trial_timeout, loss_fn=self._loss,
            seed=self.random_state, n_jobs=1, verbose=False,
        )
        self._est.fit(Xr, np.asarray(y))
        self._learner = self._est.best_model()["learner"]
        return self

    def predict(self, X):
        return self._learner.predict(self._reducer.transform(np.asarray(X, dtype=float)))

    def predict_proba(self, X):
        Xr = self._reducer.transform(np.asarray(X, dtype=float))
        if hasattr(self._learner, "predict_proba"):
            return self._learner.predict_proba(Xr)
        if hasattr(self._learner, "decision_function"):
            s = self._learner.decision_function(Xr)
            p = 1.0 / (1.0 + np.exp(-s))
            return np.column_stack([1 - p, p])
        pred = self._learner.predict(Xr)
        proba = np.zeros((len(pred), len(self.classes_)))
        for i, c in enumerate(self.classes_):
            proba[pred == c, i] = 1.0
        return proba


def make_pipeline(estimator, use_smote: bool, minority_count: int,
                  random_state: int = RANDOM_STATE, scale: bool = True):
    """Wrap a tabular estimator with optional in-fold SMOTE + scaling.

    SMOTE runs inside the pipeline, so cross-validation never leaks resampled
    rows across folds. It is skipped (with the class staying weighted) when the
    minority class is too small for k-neighbours.
    """
    steps = []
    if use_smote and HAS_SMOTE and minority_count > 6:
        steps.append(("smote", SMOTE(random_state=random_state,
                                     k_neighbors=min(5, minority_count - 1))))
    if scale:
        steps.append(("scaler", StandardScaler(with_mean=False)))
    steps.append(("model", estimator))
    return ImbPipeline(steps)


def build_tabular_models(random_state: int, y_train: np.ndarray, cfg: RunConfig) -> dict:
    """RF / XGBoost / LightGBM pipelines with imbalance-aware search spaces."""
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    minority = min(n_neg, n_pos)
    pos_weight = n_neg / max(n_pos, 1)
    use_smote = cfg.use_smote
    n_est = [200, 400] if not cfg.fast_mode else [120]

    return {
        "RandomForest": (
            make_pipeline(
                RandomForestClassifier(random_state=random_state, n_jobs=-1,
                                       class_weight="balanced"),
                use_smote, minority, random_state),
            {
                "model__n_estimators": n_est,
                "model__max_depth": [10, 20, None],
                "model__min_samples_split": [2, 5, 10],
                "model__max_features": ["sqrt", "log2", 0.3],
                "model__class_weight": ["balanced", "balanced_subsample"],
            },
        ),
        "XGBoost": (
            make_pipeline(
                XGBClassifier(random_state=random_state, objective="binary:logistic",
                              eval_metric="logloss", n_jobs=-1, verbosity=0,
                              scale_pos_weight=pos_weight),
                use_smote, minority, random_state),
            {
                "model__n_estimators": n_est,
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.03, 0.1],
                "model__subsample": [0.7, 1.0],
                "model__colsample_bytree": [0.7, 1.0],
                "model__scale_pos_weight": [1.0, pos_weight],
            },
        ),
        "LightGBM": (
            make_pipeline(
                LGBMClassifier(random_state=random_state, n_jobs=-1, verbose=-1,
                               objective="binary", class_weight="balanced"),
                use_smote, minority, random_state),
            {
                "model__n_estimators": n_est,
                "model__num_leaves": [31, 63],
                "model__learning_rate": [0.03, 0.1],
                "model__min_child_samples": [10, 30],
            },
        ),
    }


# ══════════════════════════════════════════════════════════════════════════
# Training, ensemble, selection
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class TrainResult:
    cv_summary: List[dict]
    best_estimators: dict          # name -> fitted estimator
    fold_scores: Dict[str, List[float]]
    train_times: Dict[str, float]
    calibration: Dict[str, float]


def train_models(X_train, y_train, smiles_train, X_val, y_val, smiles_val,
                 cfg: RunConfig) -> TrainResult:
    """Hyperparameter-search + cross-validate every base model.

    Tabular models (RF/XGB/LGBM) are tuned with RandomizedSearchCV; the GNN and
    the AutoML model are evaluated directly. Returns fitted estimators plus CV
    metrics used for MCDA selection.
    """
    rs = cfg.random_state
    cv_eval = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=rs)
    tabular = build_tabular_models(rs, y_train, cfg)

    cv_summary, best_estimators, fold_scores, train_times, calibration = [], {}, {}, {}, {}

    for name, (pipe, grid) in tabular.items():
        print(f"\n{'=' * 56}\nTraining: {name}\n{'=' * 56}")
        # n_jobs=1 on the search: the estimators already use all cores, and
        # nesting search-level parallelism inside model-level parallelism
        # deadlocks LightGBM under loky.
        search = RandomizedSearchCV(pipe, grid, n_iter=cfg.search_n_iter,
                                    scoring="roc_auc", cv=3, random_state=rs, n_jobs=1)
        t0 = time.time()
        search.fit(X_train, y_train)
        train_times[name] = time.time() - t0
        best = search.best_estimator_
        best_estimators[name] = best
        print(f"  best params: {search.best_params_}")

        roc, pr, mcc = _crossval_tabular(best, X_train, y_train, cv_eval)
        cv_summary.append(_summary_row(name, roc, pr, mcc))
        fold_scores[name] = roc
        calibration[name] = _val_ece(best, X_val, y_val)
        _print_cv(name, roc, pr, mcc, train_times[name])

    # GNN (Weisfeiler-Lehman) -------------------------------------------------
    print(f"\n{'=' * 56}\nTraining: GNN (Weisfeiler-Lehman)\n{'=' * 56}")
    gnn = WLGraphClassifier(n_iterations=cfg.wl_iterations, n_hash_buckets=cfg.wl_buckets,
                            random_state=rs)
    t0 = time.time()
    gnn.fit(smiles_train, y_train)
    train_times["GNN"] = time.time() - t0
    best_estimators["GNN"] = gnn
    roc, pr, mcc = _crossval_gnn(smiles_train, y_train, cv_eval, cfg)
    cv_summary.append(_summary_row("GNN", roc, pr, mcc))
    fold_scores["GNN"] = roc
    calibration["GNN"] = _val_ece(gnn, smiles_val, y_val, is_smiles=True)
    _print_cv("GNN", roc, pr, mcc, train_times["GNN"])

    # AutoML (Hyperopt-sklearn) ----------------------------------------------
    if cfg.use_automl and HAS_AUTOML:
        print(f"\n{'=' * 56}\nTraining: AutoML (Hyperopt-sklearn)\n{'=' * 56}")
        try:
            automl = AutoMLClassifier(max_evals=cfg.automl_max_evals,
                                      trial_timeout=cfg.automl_trial_timeout, random_state=rs)
            t0 = time.time()
            automl.fit(X_train, y_train)
            train_times["AutoML"] = time.time() - t0
            best_estimators["AutoML"] = automl
            print(f"  selected learner: {type(automl._learner).__name__}")
            roc, pr, mcc = _eval_holdout(automl, X_val, y_val)
            cv_summary.append(_summary_row("AutoML", [roc], [pr], [mcc]))
            fold_scores["AutoML"] = [roc]
            calibration["AutoML"] = _val_ece(automl, X_val, y_val)
            _print_cv("AutoML", [roc], [pr], [mcc], train_times["AutoML"])
        except Exception as exc:
            print(f"  WARNING: AutoML failed ({exc!r}) — continuing without it.")
    elif cfg.use_automl:
        print("\nWARNING: AutoML requested but hpsklearn/hyperopt unavailable — skipped.")

    print(f"\nTraining complete: {len(cv_summary)} models evaluated.")
    return TrainResult(cv_summary, best_estimators, fold_scores, train_times, calibration)


def _crossval_tabular(estimator, X, y, cv):
    roc, pr, mcc = [], [], []
    for tr, te in cv.split(X, y):
        est = clone(estimator)
        est.fit(X[tr], y[tr])
        proba = est.predict_proba(X[te])[:, 1]
        roc.append(roc_auc_score(y[te], proba))
        pr.append(average_precision_score(y[te], proba))
        mcc.append(matthews_corrcoef(y[te], est.predict(X[te])))
    return roc, pr, mcc


def _crossval_gnn(smiles, y, cv, cfg):
    smiles = list(smiles)
    roc, pr, mcc = [], [], []
    for tr, te in cv.split(np.zeros(len(y)), y):
        est = WLGraphClassifier(n_iterations=cfg.wl_iterations,
                                n_hash_buckets=cfg.wl_buckets, random_state=cfg.random_state)
        est.fit([smiles[i] for i in tr], y[tr])
        smi_te = [smiles[i] for i in te]
        proba = est.predict_proba(smi_te)[:, 1]
        roc.append(roc_auc_score(y[te], proba))
        pr.append(average_precision_score(y[te], proba))
        mcc.append(matthews_corrcoef(y[te], est.predict(smi_te)))
    return roc, pr, mcc


def _eval_holdout(estimator, X, y):
    proba = estimator.predict_proba(X)[:, 1]
    return (roc_auc_score(y, proba), average_precision_score(y, proba),
            matthews_corrcoef(y, estimator.predict(X)))


def _summary_row(name, roc, pr, mcc):
    return {
        "model": name,
        "roc_auc_mean": float(np.mean(roc)), "roc_auc_std": float(np.std(roc)),
        "pr_auc_mean": float(np.mean(pr)), "pr_auc_std": float(np.std(pr)),
        "mcc_mean": float(np.mean(mcc)), "mcc_std": float(np.std(mcc)),
    }


def _val_ece(estimator, X_val, y_val, is_smiles: bool = False):
    if len(y_val) == 0:
        return 0.5
    proba = estimator.predict_proba(X_val)[:, 1]
    return ece_mce(y_val, proba)[0]


def _print_cv(name, roc, pr, mcc, t):
    print(f"  CV ROC-AUC: {np.mean(roc):.4f} +/- {np.std(roc):.4f}")
    print(f"  CV PR-AUC : {np.mean(pr):.4f} +/- {np.std(pr):.4f}")
    print(f"  CV MCC    : {np.mean(mcc):.4f} +/- {np.std(mcc):.4f}  ({t:.1f}s)")


def mcda_rank(cv_summary, calibration, train_times) -> List[dict]:
    """Multi-criteria ranking. MCC is weighted heavily because it is the metric
    that exposes minority-class (non_binder) discrimination under imbalance."""
    weights = {"mcc": 0.35, "roc_auc": 0.22, "pr_auc": 0.18,
               "calibration": 0.13, "robustness": 0.07,
               "efficiency": 0.03, "interpretability": 0.02}
    interp = {"RandomForest", "LightGBM", "XGBoost"}
    rows = []
    for r in cv_summary:
        name = r["model"]
        rows.append({
            "model": name,
            "mcc": r["mcc_mean"], "roc_auc": r["roc_auc_mean"], "pr_auc": r["pr_auc_mean"],
            "calibration": max(0.0, 1 - calibration.get(name, 0.5)),
            "calibration_raw": calibration.get(name, 0.5),
            "robustness": max(0.0, 1 - r["roc_auc_std"]),
            "efficiency": 1.0 / (1.0 + train_times.get(name, 1.0)),
            "interpretability": 1.0 if name in interp else 0.6,
        })
    for metric in weights:
        vals = [row[metric] for row in rows]
        lo, hi = min(vals), max(vals)
        for row in rows:
            row[f"{metric}_norm"] = (row[metric] - lo) / (hi - lo) if hi > lo else 1.0
    for row in rows:
        row["composite"] = sum(row[f"{m}_norm"] * w for m, w in weights.items())
    return sorted(rows, key=lambda r: r["composite"], reverse=True)


def build_stacking_ensemble(best_estimators, X, y, cfg: RunConfig):
    """Stack the tuned tabular learners with a logistic-regression meta-model.

    Uses out-of-fold predictions (cv=3) so the meta-learner never sees a base
    model's in-sample predictions. The GNN and AutoML members are excluded here
    (they have bespoke inputs / interfaces) but still contribute to the
    soft-voting consensus.
    """
    estimators = [(n, clone(best_estimators[n]))
                  for n in ("RandomForest", "XGBoost", "LightGBM") if n in best_estimators]
    if len(estimators) < 2:
        return None
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced"),
        cv=3, stack_method="predict_proba", n_jobs=1, passthrough=False)
    stack.fit(X, y)
    return stack


def consensus_proba(estimators: dict, members: List[str], X, smiles):
    """Soft-voting: mean predicted probability across the named members."""
    probs = []
    for name in members:
        est = estimators.get(name)
        if est is None:
            continue
        try:
            p = est.predict_proba(smiles if name == "GNN" else X)
            probs.append(p)
        except Exception:
            continue
    if not probs:
        return None
    return np.mean(np.stack(probs, axis=0), axis=0)


# ══════════════════════════════════════════════════════════════════════════
# Uncertainty: conformal prediction + applicability domain
# ══════════════════════════════════════════════════════════════════════════

def conformal_prediction(probs, y_true, alpha=0.05):
    """Split-conformal prediction sets with marginal coverage ~1-alpha."""
    scores = 1.0 - probs[np.arange(len(y_true)), y_true]
    q = np.quantile(scores, 1 - alpha, method="higher")
    pred_sets = probs >= (1.0 - q)
    coverage = pred_sets[np.arange(len(y_true)), y_true].mean()
    return pred_sets, float(coverage), float(q)


def resolve_conformal(prob_row, q, act_map=ACTIVITY_CLASS_MAP):
    """Return (set_string, is_ambiguous) for one probability row."""
    members = [act_map[c] for c in range(len(prob_row)) if prob_row[c] >= (1.0 - q)]
    if len(members) == 0:
        return "{} (fallback: " + act_map[int(np.argmax(prob_row))] + ")", True
    if len(members) == 1:
        return "{" + members[0] + "}", False
    return "{ambiguous}", True


def fit_applicability_domain(X_train, n_neighbors=5):
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(X_train)
    dists = nn.kneighbors(X_train)[0].mean(axis=1)
    return nn, float(np.percentile(dists, 95))


# ══════════════════════════════════════════════════════════════════════════
# SAS cores (maximum common substructure of training actives per target)
# ══════════════════════════════════════════════════════════════════════════

def compute_sas_cores(rows: List[dict], labels: np.ndarray, verbose=True) -> Dict[str, Optional[str]]:
    from rdkit.Chem import rdFMCS

    targets = sorted({r.get("target_common_name", "") for r in rows} - {""})
    cores: Dict[str, Optional[str]] = {}
    for target in targets:
        mols = [Chem.MolFromSmiles(r["canonical_smiles"])
                for r, lbl in zip(rows, labels)
                if r.get("target_common_name") == target and lbl == 1]
        mols = [m for m in mols if m is not None]
        if len(mols) < 2:
            cores[target] = None
            continue
        try:
            res = rdFMCS.FindMCS(mols, timeout=15, completeRingsOnly=True,
                                 atomCompare=rdFMCS.AtomCompare.CompareElements,
                                 bondCompare=rdFMCS.BondCompare.CompareOrder)
            cores[target] = res.smartsString if res.numAtoms > 0 else None
        except Exception:
            cores[target] = None
    if verbose:
        print(f"  SAS cores: {sum(v is not None for v in cores.values())}/{len(cores)} targets")
    return cores


# ══════════════════════════════════════════════════════════════════════════
# Prediction on new compounds
# ══════════════════════════════════════════════════════════════════════════

def _morgan_fp_from_smarts(smarts, n_bits=2048):
    try:
        mol = Chem.MolFromSmarts(smarts)
        if mol is None:
            return None
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
        return gen.GetFingerprint(mol)
    except Exception:
        return None


def activity_label(pred_class) -> str:
    return (f"Active (< {BINDING_THRESHOLD_UM:.0f} uM)" if int(pred_class) == 1
            else f"Inactive (>= {BINDING_THRESHOLD_UM:.0f} uM)")


def predict_compounds(input_csv: Path, model_pkl: Path, verbose=True):
    """Predict binding for every compound in ``input_csv`` using a saved model.

    Required columns: ``compound_id``, ``smiles``. Optional ``target`` — if
    absent, each compound is scored against all 24 panel targets. SMILES are
    standardised; out-of-domain and ambiguous-conformal rows are flagged.
    """
    import pandas as pd

    with open(model_pkl, "rb") as fh:
        art = pickle.load(fh)

    feat_cfg: FeatureConfig = art["feature_config"]
    sel_cols = art["selected_columns"]
    act_map = art["activity_class_map"]
    nn = art["nn_model"]
    ad_thresh = art["ad_threshold"]
    severe_ood = art.get("severe_ood_threshold", 100.0)
    q = art["conformal_q"]
    estimators = art["all_estimators"]
    best_name = art["best_model_name"]
    consensus_members = art.get("consensus_members", list(estimators.keys()))
    panel = art.get("target_panel", TARGET_PANEL)
    cdk_cache = art.get("cdk_cache")
    cdk_names = art.get("cdk_names")
    sas_cores = art.get("sas_cores", {})
    best_model = estimators.get(best_name) or art.get("best_model")
    calibrated = art.get("calibrated_model")

    core_fps = {t: _morgan_fp_from_smarts(s) for t, s in sas_cores.items() if s}

    df_raw = pd.read_csv(input_csv, encoding="utf-8-sig")  # tolerate UTF-8 BOM
    if "compound_id" not in df_raw.columns or "smiles" not in df_raw.columns:
        raise ValueError("input_csv must have columns: compound_id, smiles")

    df_raw["smiles_standardized"] = df_raw["smiles"].apply(standardize_smiles)
    n_invalid = int(df_raw["smiles_standardized"].isna().sum())
    if n_invalid and verbose:
        print(f"  WARNING: {n_invalid} SMILES could not be standardised — skipped.")
    df_valid = df_raw.dropna(subset=["smiles_standardized"]).copy()
    if df_valid.empty:
        raise ValueError("No valid SMILES after standardisation.")

    if "target" not in df_valid.columns:
        if verbose:
            print(f"  No 'target' column — scoring all {len(panel)} panel targets.")
        df = pd.DataFrame([{**row.to_dict(), "target": t}
                           for _, row in df_valid.iterrows() for t in panel])
    else:
        df = df_valid.copy()

    rows = [{"canonical_smiles": r["smiles_standardized"],
             "target_common_name": r["target"], "activity_class": -1}
            for _, r in df.iterrows()]

    # Ensure CDK features for any new SMILES are available before building X.
    cdk_cache, cdk_names = ensure_cdk_cache(
        [r["canonical_smiles"] for r in rows], cdk_cache, cdk_names, feat_cfg, verbose=verbose)

    X, _, _ = build_feature_matrix(rows, feat_cfg, selected_columns=sel_cols,
                                   cdk_cache=cdk_cache, cdk_names=cdk_names)
    smiles_list = [r["canonical_smiles"] for r in rows]

    if best_name == "GNN":
        probs = best_model.predict_proba(smiles_list)
    elif calibrated is not None:
        probs = calibrated.predict_proba(X)
    else:
        probs = best_model.predict_proba(X)
    preds = probs.argmax(axis=1)

    dists = nn.kneighbors(X)[0].mean(axis=1)
    in_dom = dists <= ad_thresh
    sev = dists > severe_ood

    consensus = consensus_proba(estimators, consensus_members, X, smiles_list)
    if consensus is None:
        consensus = probs
    consensus_preds = consensus.argmax(axis=1)

    conf_sets, conf_amb = [], []
    for i, row in enumerate(probs):
        s, amb = resolve_conformal(row, q, act_map)
        conf_sets.append(s)
        conf_amb.append(amb or bool(sev[i]))

    bind_idx = next(c for c, l in act_map.items() if l == "binding")
    nonbind_idx = next(c for c, l in act_map.items() if l == "non_binding")

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    sas = []
    for _, r in df.iterrows():
        core = core_fps.get(r["target"])
        mol = Chem.MolFromSmiles(r["smiles_standardized"])
        sas.append(TanimotoSimilarity(gen.GetFingerprint(mol), core)
                   if (mol is not None and core is not None) else float("nan"))

    out = df.copy()
    out["prob_non_binding"] = probs[:, nonbind_idx]
    out["prob_binding"] = probs[:, bind_idx]
    out["predicted_class"] = preds
    out["predicted_label"] = [act_map.get(int(p), "?") for p in preds]
    out["predicted_activity"] = [activity_label(p) for p in preds]
    out["consensus_class"] = consensus_preds
    out["consensus_label"] = [act_map.get(int(p), "?") for p in consensus_preds]
    out["consensus_activity"] = [activity_label(p) for p in consensus_preds]
    out["consensus_prob_binding"] = consensus[:, bind_idx]
    out["max_confidence"] = probs.max(axis=1)
    out["conformal_set"] = conf_sets
    out["conformal_ambiguous"] = conf_amb
    out["in_domain"] = in_dom
    out["severe_ood"] = sev
    out["knn_distance"] = dists
    out["risk_score"] = probs[:, bind_idx]
    out["similarity_to_training"] = 1.0 / (1.0 + dists)
    out["sas_score"] = sas
    return out


# ══════════════════════════════════════════════════════════════════════════
# Plotting (shared by the notebook and run_pipeline.py)
# ══════════════════════════════════════════════════════════════════════════

def plot_data_exploration(labels_all, targets_all, pchembl_vals, path):
    import matplotlib.pyplot as plt
    import pandas as pd
    from collections import Counter

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    counts = Counter(labels_all)
    order = sorted(counts)
    bars = ax[0].bar([ACTIVITY_CLASS_MAP[c] for c in order], [counts[c] for c in order],
                     color=[CLASS_COLORS[c] for c in order], edgecolor="black")
    for b, c in zip(bars, order):
        ax[0].text(b.get_x() + b.get_width() / 2, b.get_height(), str(counts[c]),
                   ha="center", va="bottom", fontweight="bold")
    ax[0].set_title("Class Distribution")
    ax[0].set_ylabel("Count")

    tdf = pd.DataFrame({"target": targets_all, "class": labels_all})
    by_t = tdf.groupby(["target", "class"]).size().unstack(fill_value=0)
    by_t = by_t.reindex(columns=list(range(NUM_CLASSES)), fill_value=0)
    by_t.columns = [ACTIVITY_CLASS_MAP[c] for c in by_t.columns]
    by_t.loc[sorted(set(targets_all))].plot.barh(
        stacked=True, ax=ax[1], color=[CLASS_COLORS[c] for c in range(NUM_CLASSES)],
        edgecolor="black")
    ax[1].set_title("Compounds per Target")
    ax[1].set_xlabel("Count")

    ax[2].hist(pchembl_vals, bins=30, color="#3498db", edgecolor="black", alpha=0.8)
    ax[2].axvline(BINDING_PCHEMBL, color="red", ls="--", lw=2,
                  label=f"Binding threshold ({BINDING_PCHEMBL})")
    ax[2].set_title("pChEMBL Distribution")
    ax[2].set_xlabel("pChEMBL")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_roc(y_test, probs, best_name, path):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for cls in range(NUM_CLASSES):
        yt = (y_test == cls).astype(int)
        if yt.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(yt, probs[:, cls])
        ax[cls].plot(fpr, tpr, color=CLASS_COLORS[cls], lw=2,
                     label=f"AUC = {roc_auc_score(yt, probs[:, cls]):.3f}")
        ax[cls].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax[cls].set(xlabel="False Positive Rate", ylabel="True Positive Rate",
                    title=f"ROC — {ACTIVITY_CLASS_MAP[cls]}")
        ax[cls].legend(loc="lower right")
    fig.suptitle(f"Per-Class ROC ({best_name})", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_pr(y_test, probs, best_name, path):
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for cls in range(NUM_CLASSES):
        yt = (y_test == cls).astype(int)
        if yt.sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(yt, probs[:, cls])
        ax[cls].plot(rec, prec, color=CLASS_COLORS[cls], lw=2,
                     label=f"AP = {average_precision_score(yt, probs[:, cls]):.3f}")
        ax[cls].axhline(yt.mean(), color="gray", ls="--", lw=1, alpha=0.5,
                        label=f"Baseline = {yt.mean():.3f}")
        ax[cls].set(xlabel="Recall", ylabel="Precision",
                    title=f"PR — {ACTIVITY_CLASS_MAP[cls]}")
        ax[cls].legend(loc="upper right")
    fig.suptitle(f"Per-Class Precision-Recall ({best_name})", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_confusion(y_test, preds, best_name, path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, preds, labels=list(range(NUM_CLASSES)))
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1) * 100
    labels = [ACTIVITY_CLASS_MAP[c] for c in range(NUM_CLASSES)]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax[0],
                xticklabels=labels, yticklabels=labels)
    ax[0].set(xlabel="Predicted", ylabel="Actual", title="Confusion (counts)")
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues", ax=ax[1],
                xticklabels=labels, yticklabels=labels)
    ax[1].set(xlabel="Predicted", ylabel="Actual", title="Confusion (% per row)")
    fig.suptitle(f"Test Confusion Matrix ({best_name})", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_calibration(y_test, cal_probs, best_name, path):
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for cls in range(NUM_CLASSES):
        yt = (y_test == cls).astype(int)
        if yt.sum() == 0 or yt.sum() == len(yt):
            ax[cls].set_title(f"Calibration — {ACTIVITY_CLASS_MAP[cls]} (n/a)")
            continue
        pt, pp = calibration_curve(yt, cal_probs[:, cls], n_bins=10)
        ax[cls].plot(pp, pt, "o-", color=CLASS_COLORS[cls], lw=2)
        ax[cls].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax[cls].set(xlabel="Mean Predicted Probability", ylabel="Fraction of Positives",
                    title=f"Calibration — {ACTIVITY_CLASS_MAP[cls]}")
    fig.suptitle(f"Calibration Curves (isotonic, {best_name})", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_feature_importance(best_model, columns, best_name, path, top_k=20):
    import matplotlib.pyplot as plt

    model = best_model
    if hasattr(best_model, "named_steps"):
        model = best_model.named_steps.get("model", best_model)
    if not hasattr(model, "feature_importances_"):
        return None
    imp = model.feature_importances_
    if len(imp) != len(columns):
        return None
    idx = np.argsort(imp)[-top_k:]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(idx)), imp[idx], color="#3498db", edgecolor="black")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([columns[i] for i in idx])
    ax.set(xlabel="Importance", title=f"Top {top_k} Feature Importances ({best_name})")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_uncertainty(set_sizes, coverage, test_dists, ad_threshold, severe_ood_threshold,
                     ood_rate, severe_ood_rate, max_probs, correct, ece, mce, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    sizes, counts = np.unique(set_sizes, return_counts=True)
    ax[0].bar(sizes.astype(str), counts, color="#9b59b6", edgecolor="black")
    ax[0].set(xlabel="Prediction Set Size", ylabel="Count",
              title=f"Conformal Set Sizes (coverage={coverage:.2%})")

    ax[1].hist(test_dists, bins=30, color="#1abc9c", edgecolor="black", alpha=0.8)
    ax[1].axvline(ad_threshold, color="red", ls="--", lw=2, label=f"AD ({ad_threshold:.2f})")
    ax[1].axvline(severe_ood_threshold, color="darkred", ls=":", lw=2,
                  label=f"Severe OOD ({severe_ood_threshold:.0f})")
    ax[1].set(xlabel="Mean k-NN Distance", ylabel="Count",
              title=f"Applicability Domain (OOD={ood_rate:.1%})")
    ax[1].legend()

    edges = np.linspace(0, 1, 11)
    accs, confs = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (max_probs >= lo) & (max_probs < hi)
        if m.sum():
            accs.append(correct[m].mean())
            confs.append(max_probs[m].mean())
    ax[2].plot(confs, accs, "o-", color="#e67e22", lw=2, label="Model")
    ax[2].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect")
    ax[2].set(xlabel="Mean Confidence", ylabel="Accuracy",
              title=f"Reliability (ECE={ece:.3f}, MCE={mce:.3f})")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Held-out evaluation, statistics & report writers (shared)
# ══════════════════════════════════════════════════════════════════════════

def evaluate_test_compounds(test_path, art, output_dir, save_fig=True, verbose=True) -> dict:
    """Score the labelled held-out ``test_compounds.csv`` with the trained model.

    Returns a metrics dict, writes ``test_set_predictions.csv`` and (optionally)
    ``08_test_set_results.png``. Targets that score a perfect MCC=1 are flagged
    as suspicious (likely too easy / leakage) and excluded from benchmarking.
    """
    import pandas as pd
    from sklearn.metrics import matthews_corrcoef, roc_auc_score

    test_path, output_dir = Path(test_path), Path(output_dir)
    if not test_path.exists():
        if verbose:
            print("  No held-out test file — skipping.")
        return {"available": False}

    df = pd.read_csv(test_path)
    rows, smiles, ys = [], [], []
    for _, r in df.iterrows():
        kc = int(r["known_class"]) if pd.notna(r.get("known_class")) else -1
        ys.append(1 if kc == 2 else (0 if kc >= 0 else -1))
        rows.append({"canonical_smiles": r["smiles"], "target_common_name": r["target"],
                     "activity_class": ys[-1]})
        smiles.append(r["smiles"])

    cdk_cache, cdk_names = ensure_cdk_cache(
        [r["canonical_smiles"] for r in rows], art["cdk_cache"], art["cdk_names"],
        art["feature_config"], verbose=verbose)
    X, _, _ = build_feature_matrix(rows, art["feature_config"],
                                   selected_columns=art["selected_columns"],
                                   cdk_cache=cdk_cache, cdk_names=cdk_names)
    ys = np.array(ys)
    mask = ys >= 0
    X, ys = X[mask], ys[mask]
    smiles = [s for s, m in zip(smiles, mask) if m]
    df = df[mask].copy()
    if len(ys) == 0:
        return {"available": False}

    best = art["all_estimators"][art["best_model_name"]]
    probs = (best.predict_proba(smiles) if art["best_model_name"] == "GNN"
             else best.predict_proba(X))
    preds = probs.argmax(axis=1)
    consensus = consensus_proba(art["all_estimators"], art["consensus_members"], X, smiles)
    con_preds = consensus.argmax(axis=1) if consensus is not None else preds

    ext_roc = roc_auc_score(ys, probs[:, 1]) if len(set(ys)) > 1 else float("nan")
    ext_mcc = matthews_corrcoef(ys, preds)
    ext_acc = float((preds == ys).mean())

    per_target = {}
    for t in sorted(df["target"].unique()):
        tm = df["target"].values == t
        if tm.sum() < 2:
            continue
        tt, tp = ys[tm], preds[tm]
        mcc = matthews_corrcoef(tt, tp) if len(set(tt)) > 1 else float("nan")
        per_target[t] = {"n": int(tm.sum()), "acc": float((tp == tt).mean()),
                         "mcc": 0.0 if np.isnan(mcc) else float(mcc), "suspicious": mcc == 1.0}

    nn, ad = art["nn_model"], art["ad_threshold"]
    dists = nn.kneighbors(X)[0].mean(axis=1)
    df = df.copy()
    df["predicted_class"] = preds
    df["predicted_label"] = [ACTIVITY_CLASS_MAP[int(p)] for p in preds]
    df["predicted_activity"] = [activity_label(p) for p in preds]
    df["consensus_class"] = con_preds
    df["correct"] = preds == ys
    df["in_domain"] = dists <= ad
    df["severe_ood"] = dists > art.get("severe_ood_threshold", 100.0)
    df["knn_distance"] = dists
    for c in range(NUM_CLASSES):
        df[f"prob_{ACTIVITY_CLASS_MAP[c]}"] = probs[:, c]
    df["max_confidence"] = probs.max(axis=1)
    df.to_csv(output_dir / "test_set_predictions.csv", index=False)

    if save_fig:
        _plot_test_results(ys, preds, probs, per_target, output_dir / "08_test_set_results.png")

    return {"available": True, "n": int(len(ys)), "roc": float(ext_roc), "mcc": float(ext_mcc),
            "acc": ext_acc, "per_target": per_target,
            "suspicious": [t for t, m in per_target.items() if m["suspicious"]]}


def _plot_test_results(ys, preds, probs, per_target, path):
    import matplotlib.pyplot as plt
    from collections import Counter

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(NUM_CLASSES)
    ac, pc = Counter(ys), Counter(preds)
    ax[0].bar(x - 0.2, [ac.get(c, 0) for c in range(NUM_CLASSES)], 0.35,
              color=[CLASS_COLORS[c] for c in range(NUM_CLASSES)], label="Actual")
    ax[0].bar(x + 0.2, [pc.get(c, 0) for c in range(NUM_CLASSES)], 0.35, alpha=0.5,
              color=[CLASS_COLORS[c] for c in range(NUM_CLASSES)], hatch="//", label="Predicted")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([ACTIVITY_CLASS_MAP[c] for c in range(NUM_CLASSES)])
    ax[0].set_title("Actual vs Predicted")
    ax[0].legend()
    if per_target:
        order = sorted(per_target, key=lambda t: per_target[t]["acc"], reverse=True)
        ax[1].barh(range(len(order)), [per_target[t]["acc"] for t in order],
                   color=["#3498db" if per_target[t]["acc"] >= 0.5 else "#e74c3c" for t in order])
        ax[1].set_yticks(range(len(order)))
        ax[1].set_yticklabels([f"{t} (n={per_target[t]['n']})" for t in order])
        ax[1].axvline(0.5, color="gray", ls="--")
        ax[1].set(title="Per-Target Accuracy", xlim=(0, 1.05))
    ax[2].hist(probs.max(axis=1), bins=20, color="#9b59b6", edgecolor="black", alpha=0.8)
    ax[2].set(title="Confidence", xlabel="Max Confidence", ylabel="Count")
    fig.suptitle("Held-Out Test Set Results", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def paired_ttests(cv_sorted, fold_scores):
    """Paired t-tests on per-fold CV ROC-AUC, Bonferroni-corrected."""
    names = [r["model"] for r in cv_sorted]
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sa, sb = np.array(fold_scores.get(a, [])), np.array(fold_scores.get(b, []))
            if len(sa) < 2 or len(sa) != len(sb):
                continue
            t, p = stats.ttest_rel(sa, sb)
            pooled = np.std(np.concatenate([sa, sb]))
            rows.append({"Model A": a, "Model B": b, "t-stat": float(t), "p-value": float(p),
                         "Cohen's d": float((sa.mean() - sb.mean()) / pooled) if pooled else 0.0})
    bonf = 0.05 / len(rows) if rows else 0.05
    for r in rows:
        r["Significant"] = "Yes" if r["p-value"] < bonf else "No"
    return rows, bonf


def feature_blocks(cfg: RunConfig) -> List[str]:
    return [b for b, v in {"descriptors": cfg.features.use_descriptors,
                           "Morgan/ECFP": cfg.features.use_morgan,
                           "MACCS": cfg.features.use_maccs,
                           "CDK+ExtFP+SubFPC": cfg.features.use_cdk}.items() if v]


def write_workflow_summary(path, R: dict):
    """Write workflow_summary.json (the contract consumed by generate_report.py)."""
    import json

    summary = {
        "n_compounds": R["n_compounds"],
        "targets": R["targets"],
        "class_distribution": R["class_distribution"],
        "train_size": R["train_size"], "val_size": R["val_size"], "test_size": R["test_size"],
        "n_features": R["n_features"],
        "best_model": R["best_model"],
        "top3_models": R["consensus_members"],
        "model_selection_method": "MCDA composite (MCC 35% / ROC-AUC 22% / PR-AUC 18% / cal 13%)",
        "feature_blocks": R["feature_blocks"],
        "test_metrics": R["test_metrics"],
        "conformal_coverage": R["conformal_coverage"],
        "avg_prediction_set_size": R["avg_prediction_set_size"],
        "out_of_domain_rate": R["out_of_domain_rate"],
        "calibration_warning": bool(R["test_metrics"]["ece"] > 0.15),
        "smote_applied": R["smote_applied"],
        "automl_used": R["automl_used"],
    }
    Path(path).write_text(json.dumps(summary, indent=2))
    return summary


def write_analysis_markdown(path, R: dict):
    """Write analysis_report.md (consumed by generate_report.py)."""
    import datetime

    from sklearn.metrics import confusion_matrix

    cd = R["class_distribution"]
    n = R["n_compounds"]
    L = ["# OFFTOXv3 Analysis Report",
         f"\n**Generated:** {datetime.datetime.now():%Y-%m-%d %H:%M}",
         f"**Best Model:** {R['best_model']}",
         f"**Consensus models:** {', '.join(R['consensus_members'])}",
         f"**Targets:** {len(TARGET_PANEL)} safety pharmacology targets\n",
         "## 1. Dataset Summary\n",
         f"- **Total training compounds:** {n}",
         f"- **Train/Val/Test split:** {R['train_size']}/{R['val_size']}/{R['test_size']} (scaffold-based)",
         f"- **Feature dimensions:** {R['n_features']}",
         f"- **Feature blocks:** {', '.join(R['feature_blocks'])}\n",
         "### Class Distribution\n", "| Class | Label | Count | % |", "|--|--|--:|--:|"]
    for c in range(NUM_CLASSES):
        cnt = cd[ACTIVITY_CLASS_MAP[c]]
        L.append(f"| {c} | {ACTIVITY_CLASS_MAP[c]} | {cnt} | {100*cnt/n:.1f}% |")

    L += ["\n## 2. Cross-Validation Results\n", "| Model | ROC-AUC | PR-AUC | MCC |", "|--|--:|--:|--:|"]
    for r in R["cv_sorted"]:
        L.append(f"| {r['model']} | {r['roc_auc_mean']:.4f} ± {r['roc_auc_std']:.4f} | "
                 f"{r['pr_auc_mean']:.4f} ± {r['pr_auc_std']:.4f} | "
                 f"{r['mcc_mean']:.4f} ± {r['mcc_std']:.4f} |")
    tm = R["test_metrics"]
    L += [f"\n**Selected model:** {R['best_model']}",
          f"\n**SMOTE applied:** {'Yes' if R['smote_applied'] else 'No'} | "
          f"**AutoML:** {'Yes' if R['automl_used'] else 'No'}\n",
          "## 3. Internal Test Set (Scaffold Split)\n", "| Metric | Value |", "|--|--:|",
          f"| ROC-AUC | {tm['roc_auc_macro']:.4f} |", f"| PR-AUC | {tm['pr_auc_macro']:.4f} |",
          f"| MCC | {tm['mcc']:.4f} |", f"| ECE | {tm['ece']:.4f} |", f"| MCE | {tm['mce']:.4f} |",
          "\n### Confusion Matrix\n", "| | Pred non_binding | Pred binding |", "|--|--:|--:|"]
    cm = confusion_matrix(R["y_test"], R["test_preds"], labels=list(range(NUM_CLASSES)))
    for i, lab in enumerate(["non_binding", "binding"]):
        L.append(f"| **{lab}** | {cm[i,0]} | {cm[i,1]} |")

    L += ["\n## 4. Uncertainty Quantification\n",
          f"- **Conformal coverage:** {R['conformal_coverage']:.4f} (target 0.95)",
          f"- **Average prediction set size:** {R['avg_prediction_set_size']:.2f}",
          f"- **AD threshold (95th pct k-NN):** {R['ad_threshold']:.4f}",
          f"- **Out-of-domain rate:** {R['out_of_domain_rate']:.2%}\n",
          "## 5. Held-Out Test Set\n"]
    ext = R.get("ext", {})
    if ext.get("available"):
        L += [f"- **Compounds:** {ext['n']}", f"- **ROC-AUC:** {ext['roc']:.4f}",
              f"- **MCC:** {ext['mcc']:.4f}", f"- **Accuracy:** {ext['acc']:.4f}\n"]
        if ext.get("suspicious"):
            L.append(f"- **Suspicious (MCC=1) targets, excluded from benchmarking:** "
                     f"{', '.join(ext['suspicious'])}\n")
        if ext.get("per_target"):
            L += ["### Per-Target Performance\n", "| Target | N | Accuracy | MCC |", "|--|--:|--:|--:|"]
            for t, m in sorted(ext["per_target"].items()):
                L.append(f"| {t} | {m['n']} | {m['acc']:.3f} | {m['mcc']:.3f} |")
    else:
        L.append("No held-out test set available.\n")

    L += ["\n## 6. Statistical Model Comparison\n"]
    if R.get("stat_rows"):
        L += [f"Bonferroni alpha = {R['bonferroni']:.4f}\n",
              "| Model A | Model B | t | p | Cohen's d | Significant |", "|--|--|--:|--:|--:|:--:|"]
        for r in R["stat_rows"]:
            cohen = r["Cohen's d"]
            L.append(f"| {r['Model A']} | {r['Model B']} | {r['t-stat']:.3f} | "
                     f"{r['p-value']:.5f} | {cohen:.3f} | {r['Significant']} |")

    L += ["\n## 7. MCDA Ranking\n", "| Rank | Model | Composite |", "|--:|--|--:|"]
    for i, r in enumerate(R["ranking"], 1):
        L.append(f"| {i} | {r['model']} | {r['composite']:.4f} |")

    Path(path).write_text("\n".join(L) + "\n")

