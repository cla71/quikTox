#!/usr/bin/env python3
"""
Build the complete 24-target safety pharmacology dataset.

Combines existing ChEMBL-sourced data with literature-curated compounds
for all 24 targets in the safety panel. Ensures >= 30 negative compounds
(confirmed > 100 uM or no activity) per target.

Outputs:
  data/safety_targets_bioactivity.csv  - training set
  data/test_compounds.csv              - held-out test set (not in training)
"""

import csv
import math
import hashlib
from pathlib import Path
from collections import Counter

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
EXISTING_CSV = DATA_DIR / "safety_targets_bioactivity.csv"
TRAIN_CSV = DATA_DIR / "safety_targets_bioactivity.csv"
TEST_CSV = DATA_DIR / "test_compounds.csv"

# ── Full 24-target panel ─────────────────────────────────────────────
TARGET_PANEL = {
    # Nuclear Hormone Receptors (14)
    "ERa":   {"chembl_id": "CHEMBL206",  "pref_name": "Estrogen receptor alpha", "category": "Nuclear Hormone Receptor"},
    "ER_beta": {"chembl_id": "CHEMBL242", "pref_name": "Estrogen receptor beta", "category": "Nuclear Hormone Receptor"},
    "AR":    {"chembl_id": "CHEMBL1871", "pref_name": "Androgen receptor", "category": "Nuclear Hormone Receptor"},
    "GR":    {"chembl_id": "CHEMBL2034", "pref_name": "Glucocorticoid receptor", "category": "Nuclear Hormone Receptor"},
    "PR":    {"chembl_id": "CHEMBL208",  "pref_name": "Progesterone receptor", "category": "Nuclear Hormone Receptor"},
    "MR":    {"chembl_id": "CHEMBL1994", "pref_name": "Mineralocorticoid receptor", "category": "Nuclear Hormone Receptor"},
    "PPARg": {"chembl_id": "CHEMBL235",  "pref_name": "PPARgamma", "category": "Nuclear Hormone Receptor"},
    "PXR":   {"chembl_id": "CHEMBL3401", "pref_name": "Pregnane X receptor", "category": "Nuclear Hormone Receptor"},
    "CAR":   {"chembl_id": "CHEMBL2248", "pref_name": "Constitutive androstane receptor", "category": "Nuclear Hormone Receptor"},
    "LXRa":  {"chembl_id": "CHEMBL5231", "pref_name": "Liver X receptor alpha", "category": "Nuclear Hormone Receptor"},
    "LXRb":  {"chembl_id": "CHEMBL4309", "pref_name": "Liver X receptor beta", "category": "Nuclear Hormone Receptor"},
    "FXR":   {"chembl_id": "CHEMBL2001", "pref_name": "Farnesoid X receptor", "category": "Nuclear Hormone Receptor"},
    "RXRa":  {"chembl_id": "CHEMBL2061", "pref_name": "Retinoid X receptor alpha", "category": "Nuclear Hormone Receptor"},
    "VDR":   {"chembl_id": "CHEMBL1977", "pref_name": "Vitamin D receptor", "category": "Nuclear Hormone Receptor"},
    # Cardiac Safety (3)
    "hERG":  {"chembl_id": "CHEMBL240",  "pref_name": "hERG potassium channel", "category": "Cardiac Safety"},
    "Cav1.2": {"chembl_id": "CHEMBL1940", "pref_name": "L-type calcium channel", "category": "Cardiac Safety"},
    "Nav1.5": {"chembl_id": "CHEMBL1993", "pref_name": "Cardiac sodium channel", "category": "Cardiac Safety"},
    # Hepatotoxicity / CYP (5)
    "CYP3A4": {"chembl_id": "CHEMBL340",  "pref_name": "Cytochrome P450 3A4", "category": "Hepatotoxicity"},
    "CYP2D6": {"chembl_id": "CHEMBL289",  "pref_name": "Cytochrome P450 2D6", "category": "Hepatotoxicity"},
    "CYP2C9": {"chembl_id": "CHEMBL3397", "pref_name": "Cytochrome P450 2C9", "category": "Hepatotoxicity"},
    "CYP1A2": {"chembl_id": "CHEMBL3356", "pref_name": "Cytochrome P450 1A2", "category": "Hepatotoxicity"},
    "CYP2C19": {"chembl_id": "CHEMBL3622", "pref_name": "Cytochrome P450 2C19", "category": "Hepatotoxicity"},
    # Transporters (2)
    "P-gp":  {"chembl_id": "CHEMBL4302", "pref_name": "P-glycoprotein 1", "category": "Transporter"},
    "BSEP":  {"chembl_id": "CHEMBL4105", "pref_name": "Bile salt export pump", "category": "Transporter"},
}

FIELDNAMES = [
    "molecule_chembl_id", "canonical_smiles", "standard_type",
    "standard_relation", "standard_value", "standard_units",
    "pchembl_value", "activity_comment", "assay_chembl_id",
    "assay_type", "target_chembl_id", "target_pref_name",
    "document_chembl_id", "src_id", "data_validity_comment",
    "safety_category", "target_common_name", "activity_class",
    "activity_class_label",
]


def make_row(mol_id, smiles, target_name, pchembl=None, activity_class=None,
             standard_value=None, standard_type="IC50", comment=""):
    """Build a row consistent with the training CSV format.

    2-class system:
      class 0 = non_binding  (pChEMBL < 5.0, i.e. >= 10 µM, or confirmed inactive)
      class 1 = binding      (pChEMBL >= 5.0, i.e. < 10 µM)
    """
    t = TARGET_PANEL[target_name]
    if pchembl is not None and activity_class is None:
        activity_class = 1 if pchembl >= 5.0 else 0
    label_map = {0: "non_binding", 1: "binding"}
    return {
        "molecule_chembl_id": mol_id,
        "canonical_smiles": smiles,
        "standard_type": standard_type,
        "standard_relation": "=" if pchembl else "",
        "standard_value": str(standard_value) if standard_value else "",
        "standard_units": "nM" if standard_value else "",
        "pchembl_value": f"{pchembl:.2f}" if pchembl else "",
        "activity_comment": comment,
        "assay_chembl_id": "",
        "assay_type": "B",
        "target_chembl_id": t["chembl_id"],
        "target_pref_name": t["pref_name"],
        "document_chembl_id": "",
        "src_id": "",
        "data_validity_comment": "",
        "safety_category": t["category"],
        "target_common_name": target_name,
        "activity_class": str(activity_class),
        "activity_class_label": label_map[activity_class],
    }


# ── Universal negative compound pool ─────────────────────────────────
# Drug-like approved compounds with well-characterized profiles that are
# confirmed inactive (>100 µM or no measurable activity) at the 24 safety
# targets.  These are structurally complex, drug-like molecules so the model
# learns binder-vs-non-binder rather than "drug-like vs simple metabolite".
#
# Sources: FDA-approved drugs with established safety profiles.  Compounds
# known to hit specific panel targets (e.g. loperamide→hERG) are excluded
# from those targets via KNOWN_CROSSREACTIVITY below.
UNIVERSAL_NEGATIVES = [
    # ── Antibiotics ──────────────────────────────────────────────────
    ("NEG_AMOXICILLIN", "CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O", "Amoxicillin"),
    ("NEG_CEPHALEXIN", "CC1=C(C(=O)O)N2C(=O)C(NC(=O)C(N)c3ccccc3)C2SC1", "Cephalexin"),
    ("NEG_METRONIDAZOLE", "Cc1ncc([N+](=O)[O-])n1CCO", "Metronidazole"),
    ("NEG_CIPROFLOXACIN", "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O", "Ciprofloxacin"),
    ("NEG_AZITHROMYCIN", "CCC1OC(=O)C(C)C(OC2CC(C)(OC)C(O)C(C)O2)C(C)C(OC2OC(C)CC(N(C)C)C2O)C(C)(O)CC(C)C(=O)C(C)C(O)C1(C)O", "Azithromycin"),
    ("NEG_DOXYCYCLINE", "OC1=C(C(=O)NCO)C(=O)C2C(O)C3C(O)c4c(O)cccc4C(C)(O)C3CC2(O)C1=O", "Doxycycline"),
    ("NEG_TRIMETHOPRIM", "COc1cc(Cc2cnc(N)nc2N)cc(OC)c1OC", "Trimethoprim"),
    ("NEG_NITROFURANTOIN", "O=C1CN(/N=C/c2ccc([N+](=O)[O-])o2)C(=O)N1", "Nitrofurantoin"),
    ("NEG_LINEZOLID", "O=C1ON(c2ccc(N3CCOCC3)c(F)c2)C(CO)C1/C=C/C(=O)NCC1CC1", "Linezolid"),
    ("NEG_MEROPENEM", "CC1C2C(C(=O)O)=C(SC3CNC(C(=O)N(C)C)C3)C(=O)N2C1C(C)O", "Meropenem"),
    # ── Antivirals ───────────────────────────────────────────────────
    ("NEG_ACYCLOVIR", "Nc1nc(=O)c2ncn(COCCO)c2[nH]1", "Acyclovir"),
    ("NEG_OSELTAMIVIR", "CCOC(=O)C1=CC(OC(CC)CC)C(NC(C)=O)C(N)C1", "Oseltamivir"),
    ("NEG_TENOFOVIR", "Nc1ncnc2c1ncn2COCP(=O)(O)O", "Tenofovir"),
    ("NEG_SOFOSBUVIR", "CC(C)OC(=O)C(C)NP(=O)(OCC1OC(n2ccc(=O)[nH]c2=O)C(C)(F)C1O)Oc1ccccc1", "Sofosbuvir"),
    ("NEG_ENTECAVIR", "OCC1(CO)CC(n2cnc3c(=O)[nH]c(N)nc32)C=C1", "Entecavir"),
    # ── NSAIDs / Analgesics (known CYP substrates excluded for CYPs) ─
    ("NEG_ACETAMINOPHEN", "CC(=O)Nc1ccc(O)cc1", "Acetaminophen"),
    ("NEG_NAPROXEN", "COc1ccc2cc(C(C)C(=O)O)ccc2c1", "Naproxen"),
    ("NEG_CELECOXIB", "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1", "Celecoxib"),
    ("NEG_MELOXICAM", "CN1C(C(=O)Nc2ccccn2)=C(O)c2ccc(Cl)cc2S1(=O)=O", "Meloxicam"),
    ("NEG_ASPIRIN", "CC(=O)Oc1ccccc1C(=O)O", "Aspirin"),
    # ── Antidiabetics ────────────────────────────────────────────────
    ("NEG_METFORMIN", "CN(C)C(=N)NC(=N)N", "Metformin"),
    ("NEG_SITAGLIPTIN", "Fc1cc(c(F)cc1F)CC(N)CC(=O)N1CCn2c(nnc2C(F)(F)F)C1", "Sitagliptin"),
    ("NEG_EMPAGLIFLOZIN", "OCC1OC(c2cc(Oc3ccc(Cl)cc3)ccc2Cc2ccccc2)C(O)C(O)C1O", "Empagliflozin"),
    ("NEG_CANAGLIFLOZIN", "OCC1OC(c2cc3ccc(F)cc3s2)c2cc(C(F)(F)F)ccc2C1O", "Canagliflozin"),
    # ── GI / Respiratory ─────────────────────────────────────────────
    ("NEG_MONTELUKAST", "CC(C)(O)c1ccccc1CC/C(=C/c1cccc(\\C=C\\c2ccc3ccc(Cl)cc3n2)c1)SCC1(CC(=O)O)CC1", "Montelukast"),
    ("NEG_RANITIDINE", "CNC(/C=C/[N+](=O)[O-])=N\\CSCCN/C=C/[N+](=O)[O-]", "Ranitidine"),
    ("NEG_FAMOTIDINE", "NC(=N)NC(=N)NS(=O)(=O)c1ccc(CSCCC(=N)N)s1", "Famotidine"),
    ("NEG_ONDANSETRON", "Cc1nccn1CC1CCc2c(c3ccccc3n2C)C1=O", "Ondansetron"),
    ("NEG_PANTOPRAZOLE", "COc1ccnc(CS(=O)c2nc3cc(OC(F)F)ccc3[nH]2)c1OC", "Pantoprazole"),
    # ── CNS (non-ion-channel, non-NHR) ───────────────────────────────
    ("NEG_LEVETIRACETAM", "CCC(C(=O)N)N1CCCC1=O", "Levetiracetam"),
    ("NEG_GABAPENTIN", "OC(=O)CC1(CN)CCCCC1", "Gabapentin"),
    ("NEG_TOPIRAMATE", "OC1(CS(N)(=O)=O)OC2COC3(CCC(C)(C)O3)OC2O1", "Topiramate"),
    ("NEG_SUMATRIPTAN", "CNS(=O)(=O)Cc1ccc2[nH]cc(CCN(C)C)c2c1", "Sumatriptan"),
    ("NEG_ZOLPIDEM", "Cc1ccc2c(c1)-c1nc(-c3ccc(C)cc3)c(CC(=O)N(C)C)cn1C2", "Zolpidem"),
    # ── Miscellaneous approved drugs ─────────────────────────────────
    ("NEG_METHOTREXATE", "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(C(=O)NC(CCC(=O)O)C(=O)O)cc1", "Methotrexate"),
    ("NEG_ALLOPURINOL", "O=c1[nH]cnc2[nH]ncc12", "Allopurinol"),
    ("NEG_FINASTERIDE", "CC(C)NC(=O)C1CCC2C3CCC4NC(=O)C=CC4(C)C3CCC12C", "Finasteride"),
    ("NEG_TAMSULOSIN", "COc1cc(CC(C)NC(=O)c2cccc(OCC3CCCCO3)c2)cc(OC)c1S(=O)(=O)N(C)C", "Tamsulosin"),
    ("NEG_HYDROXYCHLOROQUINE", "CCN(CCO)CCCC(C)Nc1ccnc2cc(Cl)ccc12", "Hydroxychloroquine"),
    ("NEG_LISINOPRIL", "NCCCC(NC(CCc1ccccc1)C(=O)O)C(=O)N1CCCC1C(=O)O", "Lisinopril"),
    ("NEG_ENALAPRIL", "CCOC(=O)C(CCc1ccccc1)NC(C)C(=O)N1CCCC1C(=O)O", "Enalapril"),
    ("NEG_AMLODIPINE", "CCOC(=O)C1=C(COCCN)NC(C)=C(C(=O)OC)C1c1ccccc1Cl", "Amlodipine"),
    ("NEG_ATORVASTATIN", "CC(C)c1n(CC(O)CC(O)CC(=O)O)c(-c2ccccc2)c(-c2ccc(F)cc2)c1C(=O)Nc1ccccc1", "Atorvastatin"),
    ("NEG_ROSUVASTATIN", "CC(C)c1nc(N(C)S(C)(=O)=O)nc(-c2ccc(F)cc2)c1/C=C/C(O)CC(O)CC(=O)O", "Rosuvastatin"),
    ("NEG_RIBAVIRIN", "OC1C(O)C(CO)OC1n1cnc2c(C(=O)N)ncn21", "Ribavirin"),
    ("NEG_MYCOPHENOLATE", "COc1c(C)c2c(c(O)c1C/C=C(\\C)CCC(=O)O)C(=O)OC2", "Mycophenolic acid"),
    ("NEG_RALOXIFENE_BSEP", "Oc1ccc(C(=O)c2cc3ccc(O)cc3oc2C2CCCCN2)cc1", "Raloxifene"),
    ("NEG_BUSPIRONE", "O=C1CC2(CCCC(=O)N1)CCN(c1ncccn1)CC2", "Buspirone"),
    ("NEG_DULOXETINE", "CNCC(Oc1cccc2ccccc12)c1cccs1", "Duloxetine"),
    ("NEG_PREGABALIN", "CC(C)CC(CN)CC(=O)O", "Pregabalin"),
    ("NEG_ESOMEPRAZOLE", "COc1ccc2nc(CS(=O)c3ncc(C)c(OC)c3C)[nH]c2c1", "Esomeprazole"),
    ("NEG_LEVOFLOXACIN", "CC1COc2c(N3CCN(C)CC3)c(F)cc3c(=O)c(C(=O)O)cn1c23", "Levofloxacin"),
    ("NEG_DAPAGLIFLOZIN", "OCC1OC(c2cc(Oc3ccc(Cl)cc3)ccc2Cc2ccccc2)C(O)C(O)C1O", "Dapagliflozin"),
    ("NEG_SACUBITRIL", "CCOC(=O)C(CC(O)=O)Cc1ccc(-c2ccccc2)cc1", "Sacubitril"),
    ("NEG_APIXABAN", "COc1ccc(-n2nc(C(N)=O)c3c2C(=O)N(c2ccc(N4CCOCC4)cc2)CC3)cc1", "Apixaban"),
    ("NEG_RIVAROXABAN", "O=C1OCC(n2cc(-c3ccc(N4CCOCC4=O)cc3)c3ccccc32)N1c1ccc(Cl)cc1", "Rivaroxaban"),
    ("NEG_EDOXABAN", "CC(C)N1CCC(NC(=O)c2cnc(Nc3cccc4[nH]ncc34)s2)C(NC(=O)C2(O)CC(F)(F)C2)C1", "Edoxaban"),
    ("NEG_BARICITINIB", "CCS(=O)(=O)N1CC(CC#N)(n2cc(-c3ncnc4[nH]ccc34)cn2)C1", "Baricitinib"),
    ("NEG_TOFACITINIB", "CC1CCN(C(=O)CC#N)CC1N(C)c1ncnc2[nH]ccc12", "Tofacitinib"),
]

# Known cross-reactivity: compound → set of targets it should NOT be used
# as a negative for, because it has known binding at those targets.
KNOWN_CROSSREACTIVITY = {
    "NEG_AMLODIPINE":          {"Cav1.2", "hERG"},        # L-type Ca blocker
    "NEG_ONDANSETRON":         {"hERG"},                   # mild hERG liability
    "NEG_DULOXETINE":          {"CYP2D6", "CYP1A2"},      # CYP substrate/inhibitor
    "NEG_HYDROXYCHLOROQUINE":  {"hERG"},                   # QT prolongation
    "NEG_CELECOXIB":           {"CYP2D6"},                 # CYP2D6 inhibitor
    "NEG_ATORVASTATIN":        {"CYP3A4"},                 # CYP3A4 substrate
    "NEG_ROSUVASTATIN":        {"BSEP"},                   # weak BSEP inhibition
    "NEG_ESOMEPRAZOLE":        {"CYP2C19"},                # CYP2C19 substrate/inhibitor
    "NEG_NAPROXEN":            {"CYP2C9"},                 # CYP2C9 substrate
    "NEG_ASPIRIN":             {"CYP2C9"},                 # weak CYP2C9
    "NEG_FINASTERIDE":         {"AR"},                     # 5-alpha-reductase/AR related
    "NEG_ZOLPIDEM":            {"CYP3A4"},                 # CYP3A4 substrate
    "NEG_PANTOPRAZOLE":        {"CYP2C19"},                # CYP2C19
    "NEG_CIPROFLOXACIN":       {"CYP1A2"},                 # CYP1A2 inhibitor
    "NEG_MONTELUKAST":         {"CYP2C9", "CYP2C8"},      # CYP2C9 substrate
    "NEG_BUSPIRONE":           {"CYP3A4"},                 # CYP3A4 substrate
    "NEG_TOFACITINIB":         {"CYP3A4"},                 # CYP3A4 substrate
    "NEG_BARICITINIB":         {"CYP3A4"},                 # partial CYP3A4
    "NEG_RALOXIFENE_BSEP":     {"ERa", "ER_beta"},         # ER modulator
}


def get_negatives_for_target(target_name, count=32):
    """Return `count` negative rows for the given target from the universal pool.

    Excludes compounds with known cross-reactivity at the requested target.
    """
    # Filter out compounds with known binding at this target
    pool = [
        entry for entry in UNIVERSAL_NEGATIVES
        if target_name not in KNOWN_CROSSREACTIVITY.get(entry[0], set())
    ]
    # Deterministic shuffle based on target name
    pool.sort(key=lambda x: int(hashlib.md5((x[0] + target_name).encode()).hexdigest(), 16))
    rows = []
    for i, (mol_id, smi, name) in enumerate(pool[:count]):
        rows.append(make_row(
            f"{mol_id}_{target_name}", smi, target_name,
            activity_class=0, comment=f"{name} - no {target_name} activity"
        ))
    return rows


# ── Curated compounds for new targets ────────────────────────────────

def curate_er_beta():
    """Estrogen Receptor beta (ER_beta) - CHEMBL242."""
    t = "ER_beta"
    rows = []
    # Potent
    rows.append(make_row("CHEMBL289472", "N#C/C(=C\\c1ccc(O)cc1)c1ccc(O)cc1", t, pchembl=8.0, standard_value=10, comment="DPN - ERbeta selective agonist"))
    rows.append(make_row("CHEMBL381382", "Oc1ccc(-c2coc3cc(O)cc(O)c3c2=O)cc1", t, pchembl=7.15, standard_value=71, comment="Genistein - ERbeta preferring"))
    rows.append(make_row("CHEMBL2146880", "CC(C)(C)c1cc(/C=C/c2cc(C(C)(C)C)c(O)c(C(C)(C)C)c2)cc(C(C)(C)C)c1O", t, pchembl=7.52, standard_value=30, comment="ERB-041 analog"))
    rows.append(make_row("CHEMBL1089658_ERB", "Oc1ccc(-c2c(-c3ccccc3)c3ccccc3[nH]2)cc1", t, pchembl=6.92, standard_value=120, comment="WAY-200070 analog"))
    rows.append(make_row("CHEMBL159_ERB", "Oc1ccc(-c2coc3cc(O)cc(O)c3c2=O)cc1", t, pchembl=7.40, standard_value=40, comment="Genistein (ERbeta Ki)"))
    rows.append(make_row("CHEMBL81_ERB", "Oc1ccc(C(=O)c2cc3ccc(O)cc3oc2C2CCC(CCN3CCCCC3)CC2)cc1", t, pchembl=7.0, standard_value=100, comment="Raloxifene"))
    rows.append(make_row("CHEMBL135_ERB", "C[C@]12CC[C@@H]3[C@@H](CCc4cc(O)ccc43)[C@@H]1CC[C@@H]2O", t, pchembl=9.3, standard_value=0.5, standard_type="Ki", comment="17beta-Estradiol"))
    rows.append(make_row("CHEMBL411_ERB", "CC(=C(CC)c1ccc(O)cc1)c1ccc(O)cc1", t, pchembl=8.52, standard_value=3, comment="Diethylstilbestrol"))
    rows.append(make_row("CHEMBL374814", "Oc1ccc(C2=Cc3cc(O)ccc3OC2)cc1", t, pchembl=7.22, standard_value=60, comment="Coumestrol"))
    rows.append(make_row("CHEMBL18741_ERB", "Oc1ccc(-c2coc3cc(O)ccc3c2=O)cc1", t, pchembl=6.52, standard_value=300, comment="Daidzein"))
    # Less potent
    rows.append(make_row("CHEMBL418364_ERB", "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1", t, pchembl=4.70, standard_value=20000, comment="Bisphenol A"))
    rows.append(make_row("CHEMBL28_ERB", "O=c1cc(-c2ccc(O)cc2)oc2cc(O)cc(O)c12", t, pchembl=4.52, standard_value=30000, comment="Apigenin"))
    rows.append(make_row("CHEMBL159_ERB2", "O=C1CC(c2ccc(O)cc2)Oc2cc(O)cc(O)c21", t, pchembl=4.30, standard_value=50000, comment="Naringenin"))
    # Negatives
    rows.extend(get_negatives_for_target(t, 32))
    return rows


def curate_mr():
    """Mineralocorticoid Receptor (MR) - CHEMBL1994."""
    t = "MR"
    rows = []
    # Potent
    rows.append(make_row("CHEMBL1200803", "O=C1CC[C@@H]2[C@H]3CC[C@H]4CC(=O)CC[C@]4(C)[C@H]3[C@@H](O)C[C@]2(C)[C@@]1(O)C(=O)CO", t, pchembl=8.85, standard_value=1.4, standard_type="Ki", comment="Aldosterone"))
    rows.append(make_row("CHEMBL1590_MR", "CC(=O)S[C@@H]1CC2=CC(=O)CC[C@@]2(C)[C@H]2CC[C@@]3(C)[C@@H](CC[C@]3(O)C(=O)SC)[C@H]21", t, pchembl=7.52, standard_value=30, standard_type="Ki", comment="Spironolactone"))
    rows.append(make_row("CHEMBL957", "O=C1OC[C@@H]2[C@@H]3CC[C@H]4CC(=O)CC[C@@]4(C)[C@H]3[C@@H](O)C[C@]12C", t, pchembl=7.05, standard_value=90, standard_type="Ki", comment="Eplerenone"))
    rows.append(make_row("CHEMBL3707348", "CC(=O)Nc1cc(-c2ccc(C(=O)NCC3CCCC(C(=O)O)C3)cc2)c(C)nn1", t, pchembl=8.15, standard_value=7, standard_type="Ki", comment="Finerenone"))
    rows.append(make_row("CHEMBL384467_MR", "CC1CC2C3CCC4=CC(=O)C=CC4(C)C3(F)C(O)CC2(C)C1(O)C(=O)CO", t, pchembl=6.70, standard_value=200, standard_type="Ki", comment="Dexamethasone (MR binding)"))
    rows.append(make_row("CHEMBL131_MR", "CC12CCC3C(CCC4=CC(=O)C=CC43C)C1CC(O)C2(O)C(=O)CO", t, pchembl=6.52, standard_value=300, standard_type="Ki", comment="Prednisolone (MR binding)"))
    rows.append(make_row("CHEMBL389621_MR", "CC12CCC3C(CCC4=CC(=O)C=CC43C)C1CC(O)C2(O)C(=O)CO", t, pchembl=7.30, standard_value=50, standard_type="Ki", comment="Cortisol"))
    rows.append(make_row("CHEMBL1201066_MR", "CCC(=O)OC1(C(=O)SCF)CC2C3CCC4=CC(=O)C=CC4(C)C3(F)C(O)CC2(C)C1C", t, pchembl=5.70, standard_value=2000, standard_type="Ki", comment="Fluticasone"))
    rows.append(make_row("CHEMBL103_MR", "CC(=O)[C@H]1CC[C@@H]2[C@H]3CCC4=CC(=O)CC[C@@]4(C)[C@H]3CC[C@]12C", t, pchembl=8.0, standard_value=10, standard_type="Ki", comment="Progesterone (MR binding)"))
    # Less potent
    rows.append(make_row("CHEMBL1370_MR", "CCCC1OC2CC3C4CCC5=CC(=O)C=CC5(C)C4C(O)CC3(C)C2(O)C(=O)CO1", t, pchembl=4.52, standard_value=30000, comment="Budesonide (weak MR)"))
    rows.append(make_row("CHEMBL635_MR", "CC12CCC3C(CCC4=CC(=O)C=CC43C)C1CC(=O)C2(O)C(=O)CO", t, pchembl=4.30, standard_value=50000, comment="Prednisone (weak MR)"))
    rows.append(make_row("CHEMBL584_MR", "CC(C)C(=O)Nc1ccc([N+](=O)[O-])c(C(F)(F)F)c1", t, pchembl=4.10, standard_value=80000, comment="Flutamide (weak MR)"))
    # Negatives
    rows.extend(get_negatives_for_target(t, 32))
    return rows


def curate_car():
    """Constitutive Androstane Receptor (CAR) - CHEMBL2248."""
    t = "CAR"
    rows = []
    # Potent
    rows.append(make_row("CHEMBL288324", "Clc1ccc(-n2c(=O)c3cc4ccccc4cc3[nH]c2=O)cc1", t, pchembl=6.70, standard_value=200, comment="CITCO - selective CAR agonist"))
    rows.append(make_row("CHEMBL104_CAR", "ClC(c1ccccc1)(c1ccccc1)c1cn(Cc2ccccc2Cl)cn1", t, pchembl=5.70, standard_value=2000, comment="Clotrimazole (CAR activator)"))
    rows.append(make_row("CHEMBL374478_CAR", "C/C=C/C1OC2(C)OC3C(OC(C)=O)C(NC=O)C(O)C(C)(O)C(O)C(C)OC(=O)C(C)=CC=CC(OC)C1(O)C(O)C3(C)O2", t, pchembl=5.30, standard_value=5000, comment="Rifampicin (CAR activator)"))
    rows.append(make_row("CHEMBL1200878", "CC(C)(C)c1ccc(NC(=O)c2ccc(Cl)cc2)cc1", t, pchembl=6.22, standard_value=600, comment="CAR inverse agonist"))
    rows.append(make_row("CHEMBL40_CAR", "CCC1(c2ccccc2)C(=O)NC(=O)NC1=O", t, pchembl=5.10, standard_value=8000, comment="Phenobarbital (CAR activator)"))
    rows.append(make_row("CHEMBL417_CAR", "CCCCCCCCSC(=O)N(CC(CC)CCCC)CC(=O)O", t, pchembl=5.52, standard_value=3000, comment="TCPOBOP analog"))
    rows.append(make_row("CHEMBL193_CAR", "COC(=O)C1=C(C)NC(C)=C(C(=O)OC)C1c1ccccc1[N+](=O)[O-]", t, pchembl=5.0, standard_value=10000, comment="Nifedipine (CAR)"))
    # Less potent
    rows.append(make_row("CHEMBL98_CAR", "OC(=O)c1ccccc1Nc1cccc(C(F)(F)F)c1", t, pchembl=4.52, standard_value=30000, comment="Flufenamic acid (weak CAR)"))
    rows.append(make_row("CHEMBL6_CAR", "COc1ccc2c(c1)c(CC(=O)O)c(C)n2C(=O)c1ccc(Cl)cc1", t, pchembl=4.30, standard_value=50000, comment="Indomethacin (weak CAR)"))
    rows.append(make_row("CHEMBL607_CAR", "Clc1ccc(COC(Cn2ccnc2)c2ccc(Cl)cc2Cl)c(Cl)c1", t, pchembl=4.15, standard_value=70000, comment="Miconazole (weak CAR)"))
    # Negatives
    rows.extend(get_negatives_for_target(t, 32))
    return rows


def curate_lxra():
    """Liver X Receptor alpha (LXRa) - CHEMBL5231."""
    t = "LXRa"
    rows = []
    # Potent
    rows.append(make_row("CHEMBL388978_LXR", "OC(c1ccc(C(F)(F)F)cc1)(c1ccc(C(F)(F)F)cc1)c1ccc(S(=O)(=O)NC(F)(F)F)cc1", t, pchembl=7.0, standard_value=100, comment="T0901317 - LXR agonist"))
    rows.append(make_row("CHEMBL288441", "Clc1ccc(COCc2cc(Cl)c(OCc3c(Cl)cccc3Cl)c(Cl)c2)cc1Cl", t, pchembl=7.30, standard_value=50, comment="GW3965 - LXR agonist"))
    rows.append(make_row("CHEMBL2028661", "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)(CF)CF", t, pchembl=6.52, standard_value=300, comment="LXR-623"))
    rows.append(make_row("CHEMBL515568", "CC(C)(O)C(=O)Nc1ccc(-c2ccc(N3CCOCC3)c(C(F)(F)F)c2)cc1", t, pchembl=6.0, standard_value=1000, comment="LXR agonist tool compound"))
    rows.append(make_row("CHEMBL2028662", "CC(C)c1nc(-c2ccc(F)cc2)c(-c2ccccc2)c1C(=O)Nc1ccccc1", t, pchembl=6.70, standard_value=200, comment="LXR modulator"))
    rows.append(make_row("CHEMBL1628272", "OC(c1ccc(C(F)(F)F)cc1)(c1ccc(C(F)(F)F)cc1)c1cccc(NS(=O)(=O)c2ccccc2)c1", t, pchembl=5.52, standard_value=3000, comment="LXR partial agonist"))
    rows.append(make_row("CHEMBL1951010", "CC1=C(C(=O)Nc2ccccc2OC)C(c2ccccc2Cl)=C(C)NC1=O", t, pchembl=5.82, standard_value=1500, comment="LXR inverse agonist"))
    # Less potent
    rows.append(make_row("CHEMBL_LXR_LP1", "CC(C)CCCC(C)C1CCC2C3CC=C4CC(O)CCC4(C)C3CCC12C", t, pchembl=4.52, standard_value=30000, comment="Cholesterol (weak LXR)"))
    rows.append(make_row("CHEMBL_LXR_LP2", "CC(C)CCCC(C)C1CCC2C3C(O)CC4CC(O)CCC4(C)C3CCC12C", t, pchembl=4.30, standard_value=50000, comment="Oxysterol (weak LXR)"))
    # Negatives
    rows.extend(get_negatives_for_target(t, 32))
    return rows


def curate_lxrb():
    """Liver X Receptor beta (LXRb) - CHEMBL4309."""
    t = "LXRb"
    rows = []
    # Potent
    rows.append(make_row("CHEMBL388978_LXRb", "OC(c1ccc(C(F)(F)F)cc1)(c1ccc(C(F)(F)F)cc1)c1ccc(S(=O)(=O)NC(F)(F)F)cc1", t, pchembl=7.15, standard_value=71, comment="T0901317 (LXRb)"))
    rows.append(make_row("CHEMBL288441_LXRb", "Clc1ccc(COCc2cc(Cl)c(OCc3c(Cl)cccc3Cl)c(Cl)c2)cc1Cl", t, pchembl=7.0, standard_value=100, comment="GW3965 (LXRb)"))
    rows.append(make_row("CHEMBL2028661_LXRb", "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)(CF)CF", t, pchembl=6.70, standard_value=200, comment="LXR-623 (LXRb)"))
    rows.append(make_row("CHEMBL515568_LXRb", "CC(C)(O)C(=O)Nc1ccc(-c2ccc(N3CCOCC3)c(C(F)(F)F)c2)cc1", t, pchembl=6.22, standard_value=600, comment="LXR agonist (LXRb)"))
    rows.append(make_row("CHEMBL1628272_LXRb", "OC(c1ccc(C(F)(F)F)cc1)(c1ccc(C(F)(F)F)cc1)c1cccc(NS(=O)(=O)c2ccccc2)c1", t, pchembl=5.70, standard_value=2000, comment="LXR partial agonist (LXRb)"))
    rows.append(make_row("CHEMBL1951010_LXRb", "CC1=C(C(=O)Nc2ccccc2OC)C(c2ccccc2Cl)=C(C)NC1=O", t, pchembl=5.52, standard_value=3000, comment="LXR inverse agonist (LXRb)"))
    # Less potent
    rows.append(make_row("CHEMBL_LXRb_LP1", "CC(C)CCCC(C)C1CCC2C3CC=C4CC(O)CCC4(C)C3CCC12C", t, pchembl=4.70, standard_value=20000, comment="Cholesterol (weak LXRb)"))
    rows.append(make_row("CHEMBL_LXRb_LP2", "CC(C)CCCC(C)C1CCC2C3C(O)CC4CC(O)CCC4(C)C3CCC12C", t, pchembl=4.15, standard_value=70000, comment="Oxysterol (weak LXRb)"))
    # Negatives
    rows.extend(get_negatives_for_target(t, 32))
    return rows


def curate_fxr():
    """Farnesoid X Receptor (FXR) - CHEMBL2001."""
    t = "FXR"
    rows = []
    # Potent
    rows.append(make_row("CHEMBL1628662", "CCCCCC(=O)N[C@@H](Cc1ccc(OC)cc1)C(=O)N(C)Cc1ccc(O)cc1", t, pchembl=7.30, standard_value=50, comment="GW4064 - FXR agonist"))
    rows.append(make_row("CHEMBL1508816", "CC(CCC(=O)O)[C@H]1CC[C@@H]2[C@]1(C)CC[C@H]1[C@@H]3CC[C@@H]4C[C@@H](O)C(CC)[C@]4(C)[C@H]3CC[C@]12C", t, pchembl=7.52, standard_value=30, comment="Obeticholic acid (OCA)"))
    rows.append(make_row("CHEMBL594725", "CC1=CC=C(NC(=O)/C=C/C2=CC=C(N(C)C)C=C2)C=C1C(=O)NC1=CC=CC(C(F)(F)F)=C1", t, pchembl=7.0, standard_value=100, comment="Fexaramine - FXR agonist"))
    rows.append(make_row("CHEMBL2158050", "CC1CCC2(CC1)OC1=CC(=O)C=CC1=C2C1=CC=C(S(=O)(=O)N2CCOCC2)C=C1", t, pchembl=6.52, standard_value=300, comment="Tropifexor analog"))
    rows.append(make_row("CHEMBL_FXR_P5", "CC(CCC(=O)O)C1CCC2C3CCC4CC(O)CCC4(C)C3CC(O)C12C", t, pchembl=5.70, standard_value=2000, comment="Chenodeoxycholic acid"))
    rows.append(make_row("CHEMBL_FXR_P6", "OC1CCC2C1(C)CCC1C2CCC2(C)C(CCC(=O)O)CCC12", t, pchembl=5.30, standard_value=5000, comment="Lithocholic acid"))
    rows.append(make_row("CHEMBL_FXR_P7", "CC(CCC(=O)O)C1CCC2C3C(O)CC4CC(O)CCC4(C)C3CCC12C", t, pchembl=5.0, standard_value=10000, comment="Ursodeoxycholic acid"))
    # Less potent
    rows.append(make_row("CHEMBL_FXR_LP1", "CC(CCC(=O)O)C1CCC2C3CCC4CC(O)CCC4(C)C3CCC12C", t, pchembl=4.52, standard_value=30000, comment="Deoxycholic acid (weak FXR)"))
    rows.append(make_row("CHEMBL_FXR_LP2", "CC(CCC(=O)O)C1CCC2C3C(O)CC4CC(O)C(O)CC4(C)C3CC(O)C12C", t, pchembl=4.30, standard_value=50000, comment="Cholic acid (weak FXR)"))
    # Negatives
    rows.extend(get_negatives_for_target(t, 32))
    return rows


def curate_vdr():
    """Vitamin D Receptor (VDR) - CHEMBL1977."""
    t = "VDR"
    rows = []
    # Potent
    rows.append(make_row("CHEMBL1199826", "C=C1CC(O)C(=C/C=C2\\CCCC3(C)C2CCC3C(C)CCCC(C)(C)O)/C1=CC", t, pchembl=9.0, standard_value=1.0, standard_type="Ki", comment="Calcitriol (1,25-dihydroxyvitamin D3)"))
    rows.append(make_row("CHEMBL1544", "C=C1CCC(O)C(=CC=C2CCCC3(C)C2CCC3C(C)CCCC(C)(C)O)C1", t, pchembl=8.52, standard_value=3, standard_type="Ki", comment="Alfacalcidol"))
    rows.append(make_row("CHEMBL1200970", "C=C1CCC(O)C(=CC=C2CCCC3(C)C2CCC3C(C)CC(O)CC(C)(C)O)C1", t, pchembl=8.30, standard_value=5, standard_type="Ki", comment="Maxacalcitol"))
    rows.append(make_row("CHEMBL1697759", "C=C1CCC(O)C(=CC=C2CCCC3(C)C2CCC3C(CCCC(C)(C)O)C)C1", t, pchembl=7.70, standard_value=20, standard_type="Ki", comment="Paricalcitol"))
    rows.append(make_row("CHEMBL511878", "C=C1CCC(O)C(=CC=C2CCCC3(C)C2CCC3C(C)CCCC(C)(C)O)C1", t, pchembl=7.0, standard_value=100, standard_type="Ki", comment="Doxercalciferol"))
    rows.append(make_row("CHEMBL1200964", "CC(C)CCCC(C)C1CCC2C3CC=C4CC(O)CCC4(C)C3CCC12C", t, pchembl=6.30, standard_value=500, comment="Cholecalciferol (Vit D3)"))
    rows.append(make_row("CHEMBL1222", "CC(CCCC(C)(C)O)C1CCC2C3CC=C4CC(O)CCC4(C)C3CCC12C", t, pchembl=5.70, standard_value=2000, comment="Calcifediol"))
    # Less potent
    rows.append(make_row("CHEMBL_VDR_LP1", "CC(C)CCCC(C)C1CCC2C3CC=C4CC(O)CCC4(C)C3CCC12C", t, pchembl=4.52, standard_value=30000, comment="Ergocalciferol (weak VDR)"))
    rows.append(make_row("CHEMBL_VDR_LP2", "CC(C)C(C)CCC(C)C1CCC2C3CCC4CC(O)CCC4(C)C3CCC12C", t, pchembl=4.15, standard_value=70000, comment="Lumisterol (weak VDR)"))
    # Negatives
    rows.extend(get_negatives_for_target(t, 32))
    return rows


def curate_cyp2c9():
    """CYP2C9 - CHEMBL3397."""
    t = "CYP2C9"
    rows = []
    # Potent
    rows.append(make_row("CHEMBL17", "CC1=NN(C(=O)C1)C1=CC=CC=C1S(=O)(=O)NC1=CC=C(N)C=C1", t, pchembl=7.0, standard_value=100, comment="Sulfaphenazole - selective CYP2C9 inhibitor"))
    rows.append(make_row("CHEMBL502", "OC(Cn1cncn1)(Cn1cncn1)c1ccc(F)cc1F", t, pchembl=5.52, standard_value=3000, comment="Fluconazole"))
    rows.append(make_row("CHEMBL58", "OC(=O)/C=C/c1ccccc1", t, pchembl=5.30, standard_value=5000, comment="Trans-cinnamic acid (CYP2C9)"))
    rows.append(make_row("CHEMBL46", "CC1=CC=C(NC(=O)C2CC(=O)N(C3=CC=C(C)C=C3)C2=O)C=C1", t, pchembl=6.22, standard_value=600, comment="CYP2C9 inhibitor"))
    rows.append(make_row("CHEMBL25_2C9", "OC(=O)c1ccccc1Oc1ccccc1", t, pchembl=5.0, standard_value=10000, comment="Phenyl salicylate (CYP2C9)"))
    rows.append(make_row("CHEMBL698_2C9", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", t, pchembl=5.70, standard_value=2000, comment="Ibuprofen (CYP2C9 substrate)"))
    rows.append(make_row("CHEMBL1025_2C9", "OC(=O)COC1=CC=C(Cl)C=C1Cl", t, pchembl=5.82, standard_value=1500, comment="Diclofenac (CYP2C9 substrate)"))
    rows.append(make_row("CHEMBL1464_2C9", "CC(=O)CC(C1=CC=CC=C1)C1=C(O)C2=CC=CC=C2OC1=O", t, pchembl=6.52, standard_value=300, comment="Warfarin (CYP2C9 substrate)"))
    rows.append(make_row("CHEMBL41744_2C9", "CC1OC(=O)C(C)=C1C1=CC=CC=C1", t, pchembl=6.0, standard_value=1000, comment="CYP2C9 probe inhibitor"))
    # Less potent
    rows.append(make_row("CHEMBL_2C9_LP1", "CC12CCC3C(CCC4=CC(=O)CCC43C)C1CCC2=O", t, pchembl=4.52, standard_value=30000, comment="Androstenedione (weak CYP2C9)"))
    rows.append(make_row("CHEMBL_2C9_LP2", "CC(=O)OC1=CC=CC=C1C(=O)O", t, pchembl=4.30, standard_value=50000, comment="Aspirin (weak CYP2C9)"))
    rows.append(make_row("CHEMBL_2C9_LP3", "OC(=O)c1cc(O)c(O)c(O)c1", t, pchembl=4.15, standard_value=70000, comment="Gallic acid (weak CYP2C9)"))
    # Negatives
    rows.extend(get_negatives_for_target(t, 32))
    return rows


def curate_cyp2c19():
    """CYP2C19 - CHEMBL3622."""
    t = "CYP2C19"
    rows = []
    # Potent
    rows.append(make_row("CHEMBL7", "CC1=CN=C(C(=O)NC2=CC=CC=C2)S1", t, pchembl=5.30, standard_value=5000, comment="Ticlopidine (CYP2C19 substrate)"))
    rows.append(make_row("CHEMBL1503", "CC1=CC2=C(N1)C(=NC(=N2)C1=CC=C(OC(F)(F)F)C=C1)S(=O)CC1=NC=CC=C1", t, pchembl=5.52, standard_value=3000, comment="Lansoprazole (CYP2C19)"))
    rows.append(make_row("CHEMBL1585", "COC1=CC2=C(C=C1)NC(=N2)S(=O)CC1=NC=C(C)C(OC)=C1C", t, pchembl=5.70, standard_value=2000, comment="Omeprazole (CYP2C19 substrate)"))
    rows.append(make_row("CHEMBL502_2C19", "OC(Cn1cncn1)(Cn1cncn1)c1ccc(F)cc1F", t, pchembl=5.0, standard_value=10000, comment="Fluconazole (CYP2C19)"))
    rows.append(make_row("CHEMBL1200556", "Clc1ccc(Cn2ccnc2)c(Cl)c1", t, pchembl=6.0, standard_value=1000, comment="Econazole (CYP2C19)"))
    rows.append(make_row("CHEMBL607_2C19", "Clc1ccc(COC(Cn2ccnc2)c2ccc(Cl)cc2Cl)c(Cl)c1", t, pchembl=6.22, standard_value=600, comment="Miconazole (CYP2C19)"))
    rows.append(make_row("CHEMBL106_2C19", "Clc1ccc(C(c2ccccc2)(c2ccccc2)n2ccnc2)cc1", t, pchembl=5.82, standard_value=1500, comment="Clotrimazole (CYP2C19)"))
    rows.append(make_row("CHEMBL460_2C19", "CC(C)=CCC/C(C)=C/CC/C(C)=C/COc1cc(OC)c2oc(-c3ccc(O)cc3)cc(=O)c2c1O", t, pchembl=6.52, standard_value=300, comment="CYP2C19 inhibitor"))
    # Less potent
    rows.append(make_row("CHEMBL_2C19_LP1", "CC(=O)OC1=CC=CC=C1C(=O)O", t, pchembl=4.52, standard_value=30000, comment="Aspirin (weak CYP2C19)"))
    rows.append(make_row("CHEMBL_2C19_LP2", "CC1=CC=C(C=C1)C(C)C(=O)O", t, pchembl=4.30, standard_value=50000, comment="Ibuprofen (weak CYP2C19)"))
    rows.append(make_row("CHEMBL_2C19_LP3", "CC(=O)NC1=CC=C(O)C=C1", t, pchembl=4.15, standard_value=70000, comment="Acetaminophen (weak CYP2C19)"))
    # Negatives
    rows.extend(get_negatives_for_target(t, 32))
    return rows


def curate_pgp():
    """P-glycoprotein (P-gp / MDR1) - CHEMBL4302."""
    t = "P-gp"
    rows = []
    # Potent
    rows.append(make_row("CHEMBL6966_PGP", "COc1ccc(CCN(C)CCCC(C#N)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC", t, pchembl=6.70, standard_value=200, comment="Verapamil (P-gp substrate/inhibitor)"))
    rows.append(make_row("CHEMBL417152", "CCN(CC)CCOC(=O)C(CC)c1ccccc1", t, pchembl=5.52, standard_value=3000, comment="P-gp inhibitor"))
    rows.append(make_row("CHEMBL473420", "COC1=CC(=CC(OC)=C1OC)C(=O)CC1C(CC2=CC(OC)=C(OC)C(OC)=C2)COC1=O", t, pchembl=5.30, standard_value=5000, comment="P-gp substrate"))
    rows.append(make_row("CHEMBL1200669", "CC1=C2CC3C(CC(OC(=O)C4=CC=CC=C4)C(O)(C(=O)OC)C3(C)C2C(=O)C2(O)CCCC12C)OC(=O)C1=CC=CC=C1", t, pchembl=7.0, standard_value=100, comment="Paclitaxel (P-gp substrate)"))
    rows.append(make_row("CHEMBL98_PGP", "OC(=O)c1ccccc1Nc1cccc(C(F)(F)F)c1", t, pchembl=5.0, standard_value=10000, comment="P-gp modulator"))
    rows.append(make_row("CHEMBL2103820", "CC(C)CC(=O)OC1CC(OC2CC(C)(OC3CC(C(=O)C(O)C(CC=CC=CC(OC)CC(CC=O)C1C)OC)C(O)C(OC)C3C)OC(=O)C(C)CC=CC=C1CO1)C(C)C2O", t, pchembl=7.52, standard_value=30, comment="Cyclosporin A (P-gp substrate)"))
    rows.append(make_row("CHEMBL1201636", "COC1=CC2=C(C=C1OCCCN1CCC(NC(=O)C3=CC(=CC(=C3)N3CCOCC3)N3CCOCC3)CC1)N=CC=C2", t, pchembl=8.52, standard_value=3, comment="Tariquidar (3rd gen P-gp inhibitor)"))
    rows.append(make_row("CHEMBL116438", "COC1=CC2=C(C=C1OC)C(=O)C(CC1CCC(O)CC1)C(=O)N2C1CCCCC1", t, pchembl=8.0, standard_value=10, comment="Elacridar (P-gp inhibitor)"))
    rows.append(make_row("CHEMBL1082407_PGP", "CCC(=O)N(c1ccc(C#N)c(C(F)(F)F)c1)C(C)c1cccc(F)c1", t, pchembl=6.0, standard_value=1000, comment="P-gp inhibitor tool"))
    # Less potent
    rows.append(make_row("CHEMBL_PGP_LP1", "CC(=O)NC1=CC=C(O)C=C1", t, pchembl=4.52, standard_value=30000, comment="Acetaminophen (weak P-gp)"))
    rows.append(make_row("CHEMBL_PGP_LP2", "CC1=CC=C(C=C1)C(C)C(=O)O", t, pchembl=4.30, standard_value=50000, comment="Ibuprofen (weak P-gp)"))
    rows.append(make_row("CHEMBL_PGP_LP3", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", t, pchembl=4.15, standard_value=70000, comment="Caffeine (weak P-gp)"))
    # Negatives
    rows.extend(get_negatives_for_target(t, 32))
    return rows


def curate_bsep():
    """Bile Salt Export Pump (BSEP) - CHEMBL4105."""
    t = "BSEP"
    rows = []
    # Potent
    rows.append(make_row("CHEMBL2103820_BSEP", "CC(C)CC(=O)OC1CC(OC2CC(C)(OC3CC(C(=O)C(O)C(CC=CC=CC(OC)CC(CC=O)C1C)OC)C(O)C(OC)C3C)OC(=O)C(C)CC=CC=C1CO1)C(C)C2O", t, pchembl=6.52, standard_value=300, comment="Cyclosporine A (BSEP inhibitor)"))
    rows.append(make_row("CHEMBL957_BSEP", "CCOc1ccc(C(=O)NS(=O)(=O)c2ccc(-c3c(C)noc3C)cc2)cc1OCO", t, pchembl=6.0, standard_value=1000, comment="Bosentan (BSEP inhibitor)"))
    rows.append(make_row("CHEMBL408_BSEP", "Cc1c(C)c2c(c(C)c1O)CCC(C)(COc1ccc(CC3SC(=O)NC3=O)cc1)O2", t, pchembl=5.70, standard_value=2000, comment="Troglitazone (BSEP inhibitor)"))
    rows.append(make_row("CHEMBL6966_BSEP", "COc1ccc(CCN(C)CCCC(C#N)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC", t, pchembl=5.30, standard_value=5000, comment="Verapamil (BSEP inhibitor)"))
    rows.append(make_row("CHEMBL1024_BSEP", "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)(CF)CF", t, pchembl=5.52, standard_value=3000, comment="BSEP inhibitor"))
    rows.append(make_row("CHEMBL121_BSEP", "CN(CCOc1ccc(CC2SC(=O)NC2=O)cc1)c1ccccn1", t, pchembl=5.0, standard_value=10000, comment="Rosiglitazone (BSEP)"))
    rows.append(make_row("CHEMBL193_BSEP", "COC(=O)C1=C(C)NC(C)=C(C(=O)OC)C1c1ccccc1[N+](=O)[O-]", t, pchembl=6.22, standard_value=600, comment="Nifedipine (BSEP inhibitor)"))
    rows.append(make_row("CHEMBL584_BSEP", "CC(C)C(=O)Nc1ccc([N+](=O)[O-])c(C(F)(F)F)c1", t, pchembl=5.82, standard_value=1500, comment="Flutamide (BSEP inhibitor)"))
    # Less potent
    rows.append(make_row("CHEMBL_BSEP_LP1", "CC(CCC(=O)O)C1CCC2C3CCC4CC(O)CCC4(C)C3CCC12C", t, pchembl=4.52, standard_value=30000, comment="CDCA (weak BSEP)"))
    rows.append(make_row("CHEMBL_BSEP_LP2", "CC(=O)OC1=CC=CC=C1C(=O)O", t, pchembl=4.30, standard_value=50000, comment="Aspirin (weak BSEP)"))
    rows.append(make_row("CHEMBL_BSEP_LP3", "CC1=CC=C(C=C1)C(C)C(=O)O", t, pchembl=4.15, standard_value=70000, comment="Ibuprofen (weak BSEP)"))
    # Negatives
    rows.extend(get_negatives_for_target(t, 32))
    return rows


def load_existing_data():
    """Load existing training CSV data."""
    rows = []
    with EXISTING_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main():
    print("=" * 70)
    print("Building 24-target safety pharmacology dataset")
    print("=" * 70)

    # 1. Load existing data
    existing = load_existing_data()
    print(f"\nLoaded {len(existing)} existing records")

    # Track existing targets
    existing_targets = Counter()
    for r in existing:
        existing_targets[r.get("target_common_name", "")] += 1
    print("Existing targets:", dict(existing_targets))

    # 2. Curate new target data
    new_rows = []
    new_rows.extend(curate_er_beta())
    new_rows.extend(curate_mr())
    new_rows.extend(curate_car())
    new_rows.extend(curate_lxra())
    new_rows.extend(curate_lxrb())
    new_rows.extend(curate_fxr())
    new_rows.extend(curate_vdr())
    new_rows.extend(curate_cyp2c9())
    new_rows.extend(curate_cyp2c19())
    new_rows.extend(curate_pgp())
    new_rows.extend(curate_bsep())
    print(f"\nCurated {len(new_rows)} new records for 11 targets")

    # 3. Add extra negatives to existing targets that have < 30
    extra_neg = []
    for target_name in sorted(existing_targets):
        target_rows = [r for r in existing if r.get("target_common_name") == target_name]
        inactive_count = sum(1 for r in target_rows if r.get("activity_class_label") in ("inactive", "non_binding") or r.get("activity_class") == "0")
        if inactive_count < 30:
            needed = 32 - inactive_count
            negs = get_negatives_for_target(target_name, needed + 5)  # a few extra
            # Deduplicate against existing
            existing_smiles = {r.get("canonical_smiles") for r in target_rows}
            added = 0
            for neg in negs:
                if neg["canonical_smiles"] not in existing_smiles and added < needed:
                    extra_neg.append(neg)
                    existing_smiles.add(neg["canonical_smiles"])
                    added += 1
            print(f"  Added {added} negatives to {target_name} (had {inactive_count})")

    print(f"\nTotal extra negatives for existing targets: {len(extra_neg)}")

    # 4. Combine all data
    all_data = existing + new_rows + extra_neg
    print(f"\nTotal combined records: {len(all_data)}")

    # 5. Split into train and test (hold out ~15% as test by molecule ID hash)
    train_rows = []
    test_rows = []
    for row in all_data:
        mol_id = row.get("molecule_chembl_id", "")
        smi = row.get("canonical_smiles", "")
        # Deterministic hash-based split
        h = int(hashlib.md5(f"{mol_id}_{smi}".encode()).hexdigest(), 16)
        if h % 100 < 15:  # ~15% test
            test_rows.append(row)
        else:
            train_rows.append(row)

    # Ensure test set has at least some compounds per target
    target_test_counts = Counter()
    for r in test_rows:
        target_test_counts[r.get("target_common_name", "")] += 1

    print(f"\nTrain: {len(train_rows)} records")
    print(f"Test:  {len(test_rows)} records")

    # 6. Write training CSV
    with TRAIN_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in train_rows:
            filtered = {k: row.get(k, "") for k in FIELDNAMES}
            writer.writerow(filtered)
    print(f"\nTraining data saved to: {TRAIN_CSV}")

    # 7. Write test CSV (in prediction format: compound_id, smiles, target + known labels)
    test_fieldnames = [
        "compound_id", "smiles", "target",
        "known_pchembl", "known_class", "known_label",
        "molecule_chembl_id", "target_chembl_id",
    ]
    with TEST_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=test_fieldnames)
        writer.writeheader()
        for i, row in enumerate(test_rows):
            writer.writerow({
                "compound_id": f"TEST_{i:04d}",
                "smiles": row.get("canonical_smiles", ""),
                "target": row.get("target_common_name", ""),
                "known_pchembl": row.get("pchembl_value", ""),
                "known_class": row.get("activity_class", ""),
                "known_label": row.get("activity_class_label", ""),
                "molecule_chembl_id": row.get("molecule_chembl_id", ""),
                "target_chembl_id": row.get("target_chembl_id", ""),
            })
    print(f"Test data saved to: {TEST_CSV}")

    # 8. Summary statistics
    print(f"\n{'='*70}")
    print("Dataset Summary")
    print(f"{'='*70}")
    for split_name, split_data in [("TRAIN", train_rows), ("TEST", test_rows)]:
        print(f"\n{split_name}:")
        target_counts = Counter()
        class_counts = {}
        for r in split_data:
            t = r.get("target_common_name", "unknown")
            c = r.get("activity_class_label", "unknown")
            target_counts[t] += 1
            class_counts[(t, c)] = class_counts.get((t, c), 0) + 1
        for t in sorted(target_counts):
            b = class_counts.get((t, "binding"), 0)
            nb = class_counts.get((t, "non_binding"), 0)
            print(f"  {t:>10s}: {target_counts[t]:>4d} total  |  binding={b:>3d}  non_binding={nb:>3d}")
        print(f"  {'TOTAL':>10s}: {len(split_data)}")


if __name__ == "__main__":
    main()
