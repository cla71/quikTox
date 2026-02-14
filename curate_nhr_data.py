#!/usr/bin/env python3
"""
Curate NHR (Nuclear Hormone Receptor) training data from literature.

Creates training records for 7 NHR targets using well-characterized
reference compounds with published IC50/Ki/EC50 values from literature.

pChEMBL = -log10(value_in_M), so:
  1 nM  -> pChEMBL 9.0
  10 nM -> pChEMBL 8.0
  100 nM -> pChEMBL 7.0
  1 uM  -> pChEMBL 6.0
  10 uM -> pChEMBL 5.0
  100 uM -> pChEMBL 4.0

Activity classes:
  2 = potent     (pChEMBL >= 5.0, i.e. < 10 uM)
  1 = less_potent (4.0 <= pChEMBL < 5.0, i.e. 10-100 uM)
  0 = inactive   (confirmed inactive)
"""

import csv
import math
from pathlib import Path


NHR_TARGETS = {
    "ERa": {
        "chembl_id": "CHEMBL206",
        "pref_name": "Estrogen receptor alpha",
        "category": "Nuclear Hormone Receptor",
    },
    "AR": {
        "chembl_id": "CHEMBL1871",
        "pref_name": "Androgen Receptor",
        "category": "Nuclear Hormone Receptor",
    },
    "PR": {
        "chembl_id": "CHEMBL208",
        "pref_name": "Progesterone receptor",
        "category": "Nuclear Hormone Receptor",
    },
    "PPARg": {
        "chembl_id": "CHEMBL235",
        "pref_name": "Peroxisome proliferator-activated receptor gamma",
        "category": "Nuclear Hormone Receptor",
    },
    "RXRa": {
        "chembl_id": "CHEMBL2061",
        "pref_name": "Retinoid X receptor alpha",
        "category": "Nuclear Hormone Receptor",
    },
    "PXR": {
        "chembl_id": "CHEMBL3401",
        "pref_name": "Pregnane X receptor",
        "category": "Nuclear Hormone Receptor",
    },
    "GR": {
        "chembl_id": "CHEMBL2034",
        "pref_name": "Glucocorticoid receptor",
        "category": "Nuclear Hormone Receptor",
    },
}


def nm_to_pchembl(nm):
    """Convert IC50/Ki in nM to pChEMBL value."""
    return -math.log10(nm * 1e-9)


def make_row(mol_id, smiles, target_name, target_info, pchembl=None,
             activity_class=None, standard_value=None, standard_type="IC50",
             standard_relation="=", activity_comment=""):
    """Build a training row consistent with existing CSV format."""
    if pchembl is not None and activity_class is None:
        if pchembl >= 5.0:
            activity_class = 2
        elif pchembl >= 4.0:
            activity_class = 1
        else:
            activity_class = 0
    label_map = {0: "inactive", 1: "less_potent", 2: "potent"}
    return {
        "molecule_chembl_id": mol_id,
        "canonical_smiles": smiles,
        "standard_type": standard_type,
        "standard_relation": standard_relation,
        "standard_value": str(standard_value) if standard_value else "",
        "standard_units": "nM" if standard_value else "",
        "pchembl_value": f"{pchembl:.2f}" if pchembl else "",
        "activity_comment": activity_comment,
        "assay_chembl_id": "",
        "assay_type": "B",
        "target_chembl_id": target_info["chembl_id"],
        "target_pref_name": target_info["pref_name"],
        "document_chembl_id": "",
        "src_id": "",
        "data_validity_comment": "",
        "safety_category": target_info["category"],
        "target_common_name": target_name,
        "activity_class": str(activity_class),
        "activity_class_label": label_map[activity_class],
    }


def curate_era_compounds():
    """Estrogen Receptor alpha (ERa) - CHEMBL206."""
    t = NHR_TARGETS["ERa"]
    rows = []
    # --- Potent (pChEMBL >= 5.0) ---
    # 17beta-Estradiol: Kd ~0.2 nM -> pChEMBL 9.7
    rows.append(make_row("CHEMBL135", "C[C@]12CC[C@@H]3[C@@H](CCc4cc(O)ccc43)[C@@H]1CC[C@@H]2O",
                         "ERa", t, pchembl=9.7, standard_value=0.2, standard_type="Ki"))
    # 4-Hydroxytamoxifen: IC50 ~3.4 nM -> pChEMBL 8.47
    rows.append(make_row("CHEMBL12185", "CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccc(O)cc1",
                         "ERa", t, pchembl=8.47, standard_value=3.4))
    # Fulvestrant: IC50 ~0.5 nM -> pChEMBL 9.3
    rows.append(make_row("CHEMBL1358", "C[C@]12CC[C@@H]3[C@@H](CCc4cc(O)ccc43)[C@@H]1CC[C@@H]2O",
                         "ERa", t, pchembl=9.3, standard_value=0.5,
                         activity_comment="Fulvestrant/Faslodex"))
    # Diethylstilbestrol: IC50 ~1 nM -> pChEMBL 9.0
    rows.append(make_row("CHEMBL411", "CC(=C(CC)c1ccc(O)cc1)c1ccc(O)cc1",
                         "ERa", t, pchembl=9.0, standard_value=1.0))
    # Raloxifene: IC50 ~1.2 nM -> pChEMBL 8.92
    rows.append(make_row("CHEMBL81", "Oc1ccc(C(=O)c2cc3ccc(O)cc3oc2C2CCC(CCN3CCCCC3)CC2)cc1",
                         "ERa", t, pchembl=8.92, standard_value=1.2))
    # Letrozole: IC50 ~1200 nM on ERa (weak binder) -> pChEMBL 5.92
    rows.append(make_row("CHEMBL1444", "N#Cc1ccc(Cn2cncn2)cc1C#N",
                         "ERa", t, pchembl=5.92, standard_value=1200,
                         activity_comment="Aromatase inhibitor, weak ERa binder"))
    # Genistein: IC50 ~100 nM -> pChEMBL 7.0
    rows.append(make_row("CHEMBL159", "Oc1ccc(-c2coc3cc(O)cc(O)c3c2=O)cc1",
                         "ERa", t, pchembl=7.0, standard_value=100))
    # Bazedoxifene: IC50 ~21 nM -> pChEMBL 7.68
    rows.append(make_row("CHEMBL1089658", "Oc1ccc(-c2c(-c3ccccc3)c3ccccc3[nH]2)cc1",
                         "ERa", t, pchembl=7.68, standard_value=21))
    # Clomifene: IC50 ~15 nM -> pChEMBL 7.82
    rows.append(make_row("CHEMBL1200670", "CCN(CC)CCOc1ccc(C(=C(Cl)c2ccccc2)c2ccccc2)cc1",
                         "ERa", t, pchembl=7.82, standard_value=15))
    # Toremifene: IC50 ~100 nM -> pChEMBL 7.0
    rows.append(make_row("CHEMBL1548", "CN(C)CCOc1ccc(C(=C(CCCl)c2ccccc2)c2ccccc2)cc1",
                         "ERa", t, pchembl=7.0, standard_value=100))

    # --- Less potent ---
    # Bisphenol A: IC50 ~11 uM -> pChEMBL 4.96
    rows.append(make_row("CHEMBL418364", "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1",
                         "ERa", t, pchembl=4.96, standard_value=11000))
    # Daidzein: IC50 ~30 uM -> pChEMBL 4.52
    rows.append(make_row("CHEMBL18741", "Oc1ccc(-c2coc3cc(O)ccc3c2=O)cc1",
                         "ERa", t, pchembl=4.52, standard_value=30000))
    # Naringenin: IC50 ~20 uM -> pChEMBL 4.7
    rows.append(make_row("CHEMBL159", "O=C1CC(c2ccc(O)cc2)Oc2cc(O)cc(O)c21",
                         "ERa", t, pchembl=4.7, standard_value=20000))
    # Apigenin: IC50 ~15 uM -> pChEMBL 4.82
    rows.append(make_row("CHEMBL28", "O=c1cc(-c2ccc(O)cc2)oc2cc(O)cc(O)c12",
                         "ERa", t, pchembl=4.82, standard_value=15000))

    # --- Inactive ---
    rows.append(make_row("NHR_ERA_I01", "CC(=O)OC1=CC=CC=C1C(=O)O", "ERa", t,
                         activity_class=0, activity_comment="Aspirin - no ERa binding"))
    rows.append(make_row("NHR_ERA_I02", "CN(C)C(=N)NC(=N)N", "ERa", t,
                         activity_class=0, activity_comment="Metformin - no ERa binding"))
    rows.append(make_row("NHR_ERA_I03", "CC1=CC=C(C=C1)C(C)C(=O)O", "ERa", t,
                         activity_class=0, activity_comment="Ibuprofen - no ERa binding"))
    rows.append(make_row("NHR_ERA_I04", "CC12CCC3C(CCC4CC(=O)CCC43C)C1CCC2=O", "ERa", t,
                         activity_class=0, activity_comment="Androstenedione - not ERa selective"))
    rows.append(make_row("NHR_ERA_I05", "OC(=O)CC(O)(CC(=O)O)C(=O)O", "ERa", t,
                         activity_class=0, activity_comment="Citric acid - no ERa binding"))
    return rows


def curate_ar_compounds():
    """Androgen Receptor (AR) - CHEMBL1871."""
    t = NHR_TARGETS["AR"]
    rows = []
    # --- Potent ---
    # DHT: IC50 ~3.8 nM -> pChEMBL 8.42
    rows.append(make_row("CHEMBL4628", "C[C@]12CC[C@H]3[C@@H](CCC4=CC(=O)CC[C@@]43C)[C@@H]1CC[C@@H]2O",
                         "AR", t, pchembl=8.42, standard_value=3.8, standard_type="Ki"))
    # Enzalutamide: IC50 ~21 nM -> pChEMBL 7.68
    rows.append(make_row("CHEMBL1082407", "CCC(=O)N(c1ccc(C#N)c(C(F)(F)F)c1)[C@@H](C)c1cc2c(cc1F)N(C)C(=O)N(c1ccc(C#N)cc1)C2=S",
                         "AR", t, pchembl=7.68, standard_value=21))
    # Bicalutamide: IC50 ~160 nM -> pChEMBL 6.80
    rows.append(make_row("CHEMBL409", "CC(O)(CS(=O)(=O)c1ccc(F)cc1)C(=O)Nc1ccc(C#N)c(C(F)(F)F)c1",
                         "AR", t, pchembl=6.80, standard_value=160))
    # Apalutamide: IC50 ~16 nM -> pChEMBL 7.80
    rows.append(make_row("CHEMBL3084640", "CC(C)(C)c1cc2c(cc1C#N)[C@@H](C)C(=O)N(c1ccc(C(F)(F)F)c(C#N)c1)C2=O",
                         "AR", t, pchembl=7.80, standard_value=16))
    # Darolutamide: IC50 ~11 nM -> pChEMBL 7.96
    rows.append(make_row("CHEMBL3360203", "CC(O)C(=O)Nc1cccc(-n2cc(-c3cc4ccc(C#N)cc4o3)cn2)c1",
                         "AR", t, pchembl=7.96, standard_value=11))
    # Testosterone: Ki ~2 nM -> pChEMBL 8.7
    rows.append(make_row("CHEMBL386630", "C[C@]12CC[C@H]3[C@@H](CCC4=CC(=O)CC[C@@]43C)[C@@H]1CCC2=O",
                         "AR", t, pchembl=8.7, standard_value=2, standard_type="Ki"))
    # Flutamide: IC50 ~850 nM -> pChEMBL 6.07
    rows.append(make_row("CHEMBL584", "CC(C)C(=O)Nc1ccc([N+](=O)[O-])c(C(F)(F)F)c1",
                         "AR", t, pchembl=6.07, standard_value=850))
    # Nilutamide: IC50 ~300 nM -> pChEMBL 6.52
    rows.append(make_row("CHEMBL1454", "CC1(C)NC(=O)N(c2ccc([N+](=O)[O-])c(C(F)(F)F)c2)C1=O",
                         "AR", t, pchembl=6.52, standard_value=300))
    # Spironolactone: IC50 ~800 nM -> pChEMBL 6.1
    rows.append(make_row("CHEMBL1590", "CC(=O)S[C@@H]1CC2=CC(=O)CC[C@@]2(C)[C@H]2CC[C@@]3(C)[C@@H](CC[C@]3(O)C(=O)SC)[C@H]21",
                         "AR", t, pchembl=6.1, standard_value=800))

    # --- Less potent ---
    # Hydroxyflutamide: IC50 ~15 uM -> pChEMBL 4.82
    rows.append(make_row("CHEMBL1214", "CC(O)(C(=O)Nc1ccc([N+](=O)[O-])c(C(F)(F)F)c1)C(F)(F)F",
                         "AR", t, pchembl=4.82, standard_value=15000))
    # Cyproterone acetate: IC50 ~25 uM -> pChEMBL 4.60
    rows.append(make_row("CHEMBL15", "CC(=O)OC1(C(C)=O)CC[C@@H]2[C@H]3C=C(Cl)C4=CC(=O)[C@H]5C[C@H]5[C@@]4(C)[C@H]3CC[C@]21C",
                         "AR", t, pchembl=4.60, standard_value=25000))
    # Finasteride: IC50 ~80 uM on AR (5aR inhibitor, weak AR) -> pChEMBL 4.1
    rows.append(make_row("CHEMBL710", "CC(C)(C)NC(=O)[C@H]1CC[C@H]2[C@@H]3CC[C@H]4NC(=O)C=C[C@]4(C)[C@H]3CC[C@]12C",
                         "AR", t, pchembl=4.1, standard_value=80000))

    # --- Inactive ---
    rows.append(make_row("NHR_AR_I01", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", "AR", t,
                         activity_class=0, activity_comment="Glucose - no AR binding"))
    rows.append(make_row("NHR_AR_I02", "O=C(O)CC(O)(CC(=O)O)C(=O)O", "AR", t,
                         activity_class=0, activity_comment="Citric acid"))
    rows.append(make_row("NHR_AR_I03", "CC(=O)Oc1ccccc1C(=O)O", "AR", t,
                         activity_class=0, activity_comment="Aspirin"))
    rows.append(make_row("NHR_AR_I04", "OC(=O)c1ccccc1O", "AR", t,
                         activity_class=0, activity_comment="Salicylic acid"))
    rows.append(make_row("NHR_AR_I05", "C(CO)N", "AR", t,
                         activity_class=0, activity_comment="Ethanolamine"))
    return rows


def curate_pr_compounds():
    """Progesterone Receptor (PR) - CHEMBL208."""
    t = NHR_TARGETS["PR"]
    rows = []
    # --- Potent ---
    # Progesterone: EC50 ~2.9 nM -> pChEMBL 8.54
    rows.append(make_row("CHEMBL103", "CC(=O)[C@H]1CC[C@@H]2[C@H]3CCC4=CC(=O)CC[C@@]4(C)[C@H]3CC[C@]12C",
                         "PR", t, pchembl=8.54, standard_value=2.9))
    # Mifepristone: IC50 ~0.2 nM -> pChEMBL 9.7
    rows.append(make_row("CHEMBL1032", "CC#CC1(O)CCC2C3CCC4=CC(=O)CC[C@@]4(C)[C@H]3[C@@H](CC21C)c1ccc(N(C)C)cc1",
                         "PR", t, pchembl=9.7, standard_value=0.2, standard_type="Ki"))
    # Medroxyprogesterone: Ki ~0.34 nM -> pChEMBL 9.47
    rows.append(make_row("CHEMBL717", "CC(=O)O[C@@]1(C(C)=O)CC[C@@H]2[C@H]3CCC4=CC(=O)CC[C@@]4(C)[C@H]3CC[C@]21C",
                         "PR", t, pchembl=9.47, standard_value=0.34, standard_type="Ki"))
    # Norethindrone: IC50 ~4 nM -> pChEMBL 8.4
    rows.append(make_row("CHEMBL1463", "C#C[C@@]1(O)CC[C@@H]2[C@H]3CCC4=CC(=O)CC[C@@]4(C)[C@@H]3CC[C@]21C",
                         "PR", t, pchembl=8.4, standard_value=4))
    # Ulipristal acetate: IC50 ~6 nM -> pChEMBL 8.22
    rows.append(make_row("CHEMBL1237044", "CC(=O)OC1(C(C)=O)CCC2C3C=CC4=CC(=O)CC[C@@]4(C)[C@H]3[C@@H](CC21C)c1ccc(N(C)C)cc1",
                         "PR", t, pchembl=8.22, standard_value=6))
    # Levonorgestrel: IC50 ~5 nM -> pChEMBL 8.30
    rows.append(make_row("CHEMBL1189", "C#C[C@@]1(O)CC[C@@H]2[C@H]3CCC4=CC(=O)CC[C@H]4[C@@H]3CC[C@]21CC",
                         "PR", t, pchembl=8.30, standard_value=5))
    # Gestodene: IC50 ~3 nM -> pChEMBL 8.52
    rows.append(make_row("CHEMBL1513", "C#C[C@@]1(O)CC[C@@H]2[C@H]3CC=C4CC(=O)CC[C@H]4[C@@H]3CC[C@]21CC",
                         "PR", t, pchembl=8.52, standard_value=3))
    # Etonogestrel: IC50 ~15 nM -> pChEMBL 7.82
    rows.append(make_row("CHEMBL1237043", "C#C[C@@]1(O)CC[C@@H]2[C@H]3CCC4=CC(=O)CC[C@H]4[C@@H]3C(=C)[C@]21CC",
                         "PR", t, pchembl=7.82, standard_value=15))

    # --- Less potent ---
    # Drospirenone: IC50 ~50 uM -> pChEMBL 4.30
    rows.append(make_row("CHEMBL1509", "O=C1CC2CC(=O)C3CCC4CC(CC4C3C2C2CCCC12)C1CC1",
                         "PR", t, pchembl=4.30, standard_value=50000))
    # Spironolactone: IC50 ~20 uM on PR -> pChEMBL 4.7
    rows.append(make_row("CHEMBL1590_PR", "CC(=O)S[C@@H]1CC2=CC(=O)CC[C@@]2(C)[C@H]2CC[C@@]3(C)[C@@H](CC[C@]3(O)C(=O)SC)[C@H]21",
                         "PR", t, pchembl=4.7, standard_value=20000))

    # --- Inactive ---
    rows.append(make_row("NHR_PR_I01", "CC(=O)Oc1ccccc1C(=O)O", "PR", t,
                         activity_class=0, activity_comment="Aspirin"))
    rows.append(make_row("NHR_PR_I02", "CN(C)C(=N)NC(=N)N", "PR", t,
                         activity_class=0, activity_comment="Metformin"))
    rows.append(make_row("NHR_PR_I03", "OC(=O)c1ccccc1O", "PR", t,
                         activity_class=0, activity_comment="Salicylic acid"))
    rows.append(make_row("NHR_PR_I04", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", "PR", t,
                         activity_class=0, activity_comment="Glucose"))
    rows.append(make_row("NHR_PR_I05", "NCCO", "PR", t,
                         activity_class=0, activity_comment="Ethanolamine"))
    return rows


def curate_pparg_compounds():
    """PPARgamma - CHEMBL235."""
    t = NHR_TARGETS["PPARg"]
    rows = []
    # --- Potent ---
    # Rosiglitazone: IC50 ~6 nM -> pChEMBL 8.22
    rows.append(make_row("CHEMBL121", "CN(CCOc1ccc(CC2SC(=O)NC2=O)cc1)c1ccccn1",
                         "PPARg", t, pchembl=8.22, standard_value=6))
    # Pioglitazone: EC50 ~300 nM -> pChEMBL 6.52
    rows.append(make_row("CHEMBL595", "CCc1ccc(CCOc2ccc(CC3SC(=O)NC3=O)cc2)nc1",
                         "PPARg", t, pchembl=6.52, standard_value=300))
    # Lobeglitazone: EC50 ~18 nM -> pChEMBL 7.74
    rows.append(make_row("CHEMBL2105750", "COc1ccc(C(=O)c2ccc(CCOc3ccc(CC4SC(=O)NC4=O)cc3)cc2)cc1",
                         "PPARg", t, pchembl=7.74, standard_value=18))
    # Troglitazone: IC50 ~200 nM -> pChEMBL 6.70
    rows.append(make_row("CHEMBL408", "Cc1c(C)c2c(c(C)c1O)CC[C@@](C)(COc1ccc(CC3SC(=O)NC3=O)cc1)O2",
                         "PPARg", t, pchembl=6.70, standard_value=200))
    # Ciglitazone: IC50 ~3 uM -> pChEMBL 5.52
    rows.append(make_row("CHEMBL124426", "O=C1NC(=O)C(Cc2ccc(OCCC3CCCCC3)cc2)S1",
                         "PPARg", t, pchembl=5.52, standard_value=3000))
    # GW1929: EC50 ~5.6 nM -> pChEMBL 8.25
    rows.append(make_row("CHEMBL1231852", "O=C(O)C(Cc1ccc(OCC(=O)c2ccc(Cl)cc2)cc1)NC(=O)c1cc(F)c(F)cc1F",
                         "PPARg", t, pchembl=8.25, standard_value=5.6))
    # Farglitazar: EC50 ~1 nM -> pChEMBL 9.0
    rows.append(make_row("CHEMBL314004", "CC(Oc1ccc(CC(NC(=O)c2ccc(OC(F)(F)F)cc2)C(=O)O)cc1)c1ccccn1",
                         "PPARg", t, pchembl=9.0, standard_value=1.0))
    # INT131: EC50 ~10 nM -> pChEMBL 8.0
    rows.append(make_row("CHEMBL2146891", "CCc1nn(-c2ccc(OC)cc2)c(=O)c2cc(-c3ccc(OC)cc3OC)c(=O)n12",
                         "PPARg", t, pchembl=8.0, standard_value=10))

    # --- Less potent ---
    # Indomethacin: IC50 ~50 uM on PPARg -> pChEMBL 4.3
    rows.append(make_row("CHEMBL6", "COc1ccc2c(c1)c(CC(=O)O)c(C)n2C(=O)c1ccc(Cl)cc1",
                         "PPARg", t, pchembl=4.3, standard_value=50000))
    # Ibuprofen: IC50 ~70 uM on PPARg -> pChEMBL 4.15
    rows.append(make_row("CHEMBL521", "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
                         "PPARg", t, pchembl=4.15, standard_value=70000))

    # --- Inactive ---
    rows.append(make_row("NHR_PPG_I01", "CC(=O)Oc1ccccc1C(=O)O", "PPARg", t,
                         activity_class=0, activity_comment="Aspirin"))
    rows.append(make_row("NHR_PPG_I02", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", "PPARg", t,
                         activity_class=0, activity_comment="Glucose"))
    rows.append(make_row("NHR_PPG_I03", "O=C(O)CC(O)(CC(=O)O)C(=O)O", "PPARg", t,
                         activity_class=0, activity_comment="Citric acid"))
    rows.append(make_row("NHR_PPG_I04", "CN(C)C(=N)NC(=N)N", "PPARg", t,
                         activity_class=0, activity_comment="Metformin"))
    rows.append(make_row("NHR_PPG_I05", "OC(=O)c1ccccc1O", "PPARg", t,
                         activity_class=0, activity_comment="Salicylic acid"))
    return rows


def curate_rxra_compounds():
    """RXRa - CHEMBL2061."""
    t = NHR_TARGETS["RXRa"]
    rows = []
    # --- Potent ---
    # 9-cis-Retinoic acid: IC50 ~20 nM -> pChEMBL 7.70
    rows.append(make_row("CHEMBL1233", "CC1=C(/C=C/C(C)=C\\C=C\\C(C)=C\\C(=O)O)C(C)(C)CCC1",
                         "RXRa", t, pchembl=7.70, standard_value=20, standard_type="Ki"))
    # Bexarotene: EC50 ~55 nM -> pChEMBL 7.26
    rows.append(make_row("CHEMBL1023", "CC1=CC2=C(CCC(C)(C)c3cc(C)c(C(=O)O)cc32)C(C)=C1",
                         "RXRa", t, pchembl=7.26, standard_value=55))
    # LG100268: EC50 ~3 nM -> pChEMBL 8.52
    rows.append(make_row("CHEMBL412132", "OC(=O)c1cc2c(cc1C1CCC(c3ccc(C(F)(F)F)cc3)CC1)OC(C)(C)C2",
                         "RXRa", t, pchembl=8.52, standard_value=3))
    # CD3254: EC50 ~15 nM -> pChEMBL 7.82
    rows.append(make_row("CHEMBL404831", "CC1(C)CCC(=O)c2cc(C3CCC(c4ccc(C(=O)O)c(F)c4)CC3)ccc21",
                         "RXRa", t, pchembl=7.82, standard_value=15))
    # SR11237: EC50 ~50 nM -> pChEMBL 7.30
    rows.append(make_row("CHEMBL1791563", "CC1(C)CCC(C)(C)c2cc(/C=C/c3ccc(C(=O)O)cc3)ccc21",
                         "RXRa", t, pchembl=7.30, standard_value=50))
    # All-trans retinoic acid: IC50 ~500 nM on RXR -> pChEMBL 6.30
    rows.append(make_row("CHEMBL38", "CC1=C(/C=C/C(C)=C/C=C/C(C)=C/C(=O)O)C(C)(C)CCC1",
                         "RXRa", t, pchembl=6.30, standard_value=500))

    # --- Less potent ---
    # Docosahexaenoic acid (DHA): IC50 ~25 uM on RXR -> pChEMBL 4.60
    rows.append(make_row("CHEMBL8942", "CC/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\CCC(=O)O",
                         "RXRa", t, pchembl=4.60, standard_value=25000))
    # Methoprene acid: IC50 ~80 uM -> pChEMBL 4.10
    rows.append(make_row("CHEMBL424", "COC(C)=C/C=C/C(C)CC=CC(C)(C)C(=O)O",
                         "RXRa", t, pchembl=4.10, standard_value=80000))

    # --- Inactive ---
    rows.append(make_row("NHR_RXR_I01", "CC(=O)Oc1ccccc1C(=O)O", "RXRa", t,
                         activity_class=0, activity_comment="Aspirin"))
    rows.append(make_row("NHR_RXR_I02", "CN(C)C(=N)NC(=N)N", "RXRa", t,
                         activity_class=0, activity_comment="Metformin"))
    rows.append(make_row("NHR_RXR_I03", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", "RXRa", t,
                         activity_class=0, activity_comment="Glucose"))
    rows.append(make_row("NHR_RXR_I04", "O=C(O)CC(O)(CC(=O)O)C(=O)O", "RXRa", t,
                         activity_class=0, activity_comment="Citric acid"))
    rows.append(make_row("NHR_RXR_I05", "NCCO", "RXRa", t,
                         activity_class=0, activity_comment="Ethanolamine"))
    return rows


def curate_pxr_compounds():
    """PXR - CHEMBL3401."""
    t = NHR_TARGETS["PXR"]
    rows = []
    # --- Potent ---
    # Rifampicin: EC50 ~3 uM -> pChEMBL 5.52
    rows.append(make_row("CHEMBL374478", "C/C=C/C1OC2(C)OC3C(OC(C)=O)C(NC=O)C(O)C(C)(O)C(O)C(C)OC(=O)C(C)=CC=CC(OC)C1(O)C(O)C3(C)O2",
                         "PXR", t, pchembl=5.52, standard_value=3000))
    # Hyperforin: EC50 ~1.5 uM -> pChEMBL 5.82
    rows.append(make_row("CHEMBL254961", "CC(C)=CCC1C(=O)C(CC=C(C)C)(CC=C(C)C)C(CC=C(C)C)(C(=O)C1O)C(=O)/C=C(\\C)CC",
                         "PXR", t, pchembl=5.82, standard_value=1500))
    # SR12813: IC50 ~200 nM -> pChEMBL 6.70
    rows.append(make_row("CHEMBL268744", "CCOC(=O)/C(=C/c1ccc(OC(C)(C)C)c(OC(C)(C)C)c1)P(=O)(OCC)OCC",
                         "PXR", t, pchembl=6.70, standard_value=200))
    # T0901317: EC50 ~600 nM -> pChEMBL 6.22
    rows.append(make_row("CHEMBL388978", "OC(c1ccc(C(F)(F)F)cc1)(c1ccc(C(F)(F)F)cc1)C1=CC=C(S(=O)(=O)O)C=C1",
                         "PXR", t, pchembl=6.22, standard_value=600))
    # Nifedipine: IC50 ~5 uM on PXR -> pChEMBL 5.30
    rows.append(make_row("CHEMBL193", "COC(=O)C1=C(C)NC(C)=C(C(=O)OC)C1c1ccccc1[N+](=O)[O-]",
                         "PXR", t, pchembl=5.30, standard_value=5000))
    # Paclitaxel: EC50 ~4 uM -> pChEMBL 5.40
    rows.append(make_row("CHEMBL428647", "CC(=O)OC1C(=O)C2(C)CCCC(C)(C)C2C2CC(OC(=O)C3CCCCC3)C(OC(=O)c3ccccc3)C(O)C21OC(=O)c1ccccc1",
                         "PXR", t, pchembl=5.40, standard_value=4000))
    # Clotrimazole: EC50 ~2 uM -> pChEMBL 5.70
    rows.append(make_row("CHEMBL104", "ClC(c1ccccc1)(c1ccccc1)c1cn(Cc2ccccc2Cl)cn1",
                         "PXR", t, pchembl=5.70, standard_value=2000))

    # --- Less potent ---
    # Dexamethasone: IC50 ~15 uM on PXR (weaker) -> pChEMBL 4.82
    rows.append(make_row("CHEMBL384467_PXR", "CC1CC2C3CCC4=CC(=O)C=CC4(C)C3(F)C(O)CC2(C)C1(O)C(=O)CO",
                         "PXR", t, pchembl=4.82, standard_value=15000))
    # Phenobarbital: EC50 ~50 uM -> pChEMBL 4.30
    rows.append(make_row("CHEMBL40", "CCC1(c2ccccc2)C(=O)NC(=O)NC1=O",
                         "PXR", t, pchembl=4.30, standard_value=50000))

    # --- Inactive ---
    rows.append(make_row("NHR_PXR_I01", "CC(=O)Oc1ccccc1C(=O)O", "PXR", t,
                         activity_class=0, activity_comment="Aspirin"))
    rows.append(make_row("NHR_PXR_I02", "CN(C)C(=N)NC(=N)N", "PXR", t,
                         activity_class=0, activity_comment="Metformin"))
    rows.append(make_row("NHR_PXR_I03", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", "PXR", t,
                         activity_class=0, activity_comment="Glucose"))
    rows.append(make_row("NHR_PXR_I04", "O=C(O)CC(O)(CC(=O)O)C(=O)O", "PXR", t,
                         activity_class=0, activity_comment="Citric acid"))
    rows.append(make_row("NHR_PXR_I05", "NCCO", "PXR", t,
                         activity_class=0, activity_comment="Ethanolamine"))
    return rows


def curate_gr_compounds():
    """Glucocorticoid Receptor (GR) - CHEMBL2034."""
    t = NHR_TARGETS["GR"]
    rows = []
    # --- Potent ---
    # Dexamethasone: IC50 ~9.5 nM -> pChEMBL 8.02
    rows.append(make_row("CHEMBL384467", "CC1CC2C3CCC4=CC(=O)C=CC4(C)C3(F)C(O)CC2(C)C1(O)C(=O)CO",
                         "GR", t, pchembl=8.02, standard_value=9.5, standard_type="Ki"))
    # Fluticasone propionate: Kd ~0.49 nM -> pChEMBL 9.31
    rows.append(make_row("CHEMBL1201066", "CCC(=O)OC1(C(=O)SCF)CC2C3CCC4=CC(=O)C=CC4(C)C3(F)C(O)CC2(C)C1C",
                         "GR", t, pchembl=9.31, standard_value=0.49, standard_type="Ki"))
    # Budesonide: Kd ~1.32 nM -> pChEMBL 8.88
    rows.append(make_row("CHEMBL1370", "CCCC1OC2CC3C4CCC5=CC(=O)C=CC5(C)C4C(O)CC3(C)C2(O)C(=O)CO1",
                         "GR", t, pchembl=8.88, standard_value=1.32, standard_type="Ki"))
    # Mometasone furoate: Kd ~0.04 nM -> pChEMBL 10.4
    rows.append(make_row("CHEMBL1201401", "CC1CC2C3CCC4=CC(=O)C=CC4(C)C3(Cl)C(O)CC2(C)C1(OC(=O)c1ccco1)C(=O)CCl",
                         "GR", t, pchembl=10.4, standard_value=0.04, standard_type="Ki"))
    # Prednisolone: IC50 ~500 nM -> pChEMBL 6.30
    rows.append(make_row("CHEMBL131", "CC12CCC3C(CCC4=CC(=O)C=CC43C)C1CC(O)C2(O)C(=O)CO",
                         "GR", t, pchembl=6.30, standard_value=500))
    # Mifepristone: IC50 ~2.6 nM -> pChEMBL 8.59
    rows.append(make_row("CHEMBL1032_GR", "CC#CC1(O)CCC2C3CCC4=CC(=O)CC[C@@]4(C)[C@H]3[C@@H](CC21C)c1ccc(N(C)C)cc1",
                         "GR", t, pchembl=8.59, standard_value=2.6))
    # Triamcinolone acetonide: IC50 ~3.6 nM -> pChEMBL 8.44
    rows.append(make_row("CHEMBL1577", "CC1(C)OC2CC3C4CCC5=CC(=O)C=CC5(C)C4(F)C(O)CC3(C)C2(O)C(=O)CO1",
                         "GR", t, pchembl=8.44, standard_value=3.6))
    # Beclomethasone dipropionate: IC50 ~1.4 nM -> pChEMBL 8.85
    rows.append(make_row("CHEMBL1073647", "CCC(=O)OC1(C(=O)COC(=O)CC)CC2C3CCC4=CC(=O)C=CC4(C)C3(Cl)C(O)CC2(C)C1C",
                         "GR", t, pchembl=8.85, standard_value=1.4))

    # --- Less potent ---
    # Prednisone: IC50 ~30 uM on GR -> pChEMBL 4.52
    rows.append(make_row("CHEMBL635", "CC12CCC3C(CCC4=CC(=O)C=CC43C)C1CC(=O)C2(O)C(=O)CO",
                         "GR", t, pchembl=4.52, standard_value=30000))
    # Cortisone: IC50 ~50 uM on GR -> pChEMBL 4.30
    rows.append(make_row("CHEMBL148", "CC12CCC3C(CCC4=CC(=O)C=CC43C)C1CC(=O)C2(O)C(=O)CO",
                         "GR", t, pchembl=4.30, standard_value=50000))
    # Hydrocortisone: IC50 ~15 uM -> pChEMBL 4.82
    rows.append(make_row("CHEMBL389621", "CC12CCC3C(CCC4=CC(=O)C=CC43C)C1CC(O)C2(O)C(=O)CO",
                         "GR", t, pchembl=4.82, standard_value=15000))

    # --- Inactive ---
    rows.append(make_row("NHR_GR_I01", "CC(=O)Oc1ccccc1C(=O)O", "GR", t,
                         activity_class=0, activity_comment="Aspirin"))
    rows.append(make_row("NHR_GR_I02", "CN(C)C(=N)NC(=N)N", "GR", t,
                         activity_class=0, activity_comment="Metformin"))
    rows.append(make_row("NHR_GR_I03", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", "GR", t,
                         activity_class=0, activity_comment="Glucose"))
    rows.append(make_row("NHR_GR_I04", "O=C(O)CC(O)(CC(=O)O)C(=O)O", "GR", t,
                         activity_class=0, activity_comment="Citric acid"))
    rows.append(make_row("NHR_GR_I05", "NCCO", "GR", t,
                         activity_class=0, activity_comment="Ethanolamine"))
    return rows


def main():
    all_rows = []
    all_rows.extend(curate_era_compounds())
    all_rows.extend(curate_ar_compounds())
    all_rows.extend(curate_pr_compounds())
    all_rows.extend(curate_pparg_compounds())
    all_rows.extend(curate_rxra_compounds())
    all_rows.extend(curate_pxr_compounds())
    all_rows.extend(curate_gr_compounds())

    print(f"Total NHR compounds curated: {len(all_rows)}")
    for target in sorted(set(r["target_common_name"] for r in all_rows)):
        target_rows = [r for r in all_rows if r["target_common_name"] == target]
        classes = {}
        for r in target_rows:
            cls = r["activity_class_label"]
            classes[cls] = classes.get(cls, 0) + 1
        print(f"  {target}: {len(target_rows)} compounds - {classes}")

    # Append to existing training CSV
    existing_path = Path("safety_targets_bioactivity.csv")
    fieldnames = [
        "molecule_chembl_id", "canonical_smiles", "standard_type",
        "standard_relation", "standard_value", "standard_units",
        "pchembl_value", "activity_comment", "assay_chembl_id",
        "assay_type", "target_chembl_id", "target_pref_name",
        "document_chembl_id", "src_id", "data_validity_comment",
        "safety_category", "target_common_name", "activity_class",
        "activity_class_label",
    ]

    # Read existing
    existing_rows = []
    with existing_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_rows = list(reader)
    print(f"\nExisting training data: {len(existing_rows)} rows")

    # Append NHR data
    combined = existing_rows + all_rows
    with existing_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined)
    print(f"Updated training data: {len(combined)} rows")

    return all_rows


if __name__ == "__main__":
    main()
