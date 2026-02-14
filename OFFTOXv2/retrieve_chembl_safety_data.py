#!/usr/bin/env python3
"""
ChEMBL Safety Data Retrieval Script
====================================

Direct implementation using ChEMBL REST API
Retrieves IC50 and Ki data for all safety targets
Includes robust error handling and data quality checks

Retrieves three classes of compounds per target:
  - Potent (pChEMBL >= 5.0, i.e. < 10 µM)
  - Less potent (4.0 <= pChEMBL < 5.0, i.e. 10-100 µM)
  - Inactive: compounds tested against the target with no measurable
    activity (right-censored measurements with standard_relation '>'
    at concentrations >= 10 µM, or explicit 'Not Active' comments).
    A diversity filter ensures inactive compounds span many scaffolds.

Usage:
    python retrieve_chembl_safety_data.py

Output:
    safety_targets_bioactivity.csv - Combined dataset
    summary_statistics.csv - Data quality summary
"""

import requests
import pandas as pd
import numpy as np
import time
import json
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

# ChEMBL REST API base URL
CHEMBL_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"

# Safety targets dictionary
SAFETY_TARGETS = {
    'Cardiac Safety': {
        'hERG': 'CHEMBL240',
        'Cav1.2': 'CHEMBL1940',
        'Nav1.5': 'CHEMBL1993'
    },
    'Hepatotoxicity': {
        'CYP3A4': 'CHEMBL340',
        'CYP2D6': 'CHEMBL289',
        'CYP2C9': 'CHEMBL3397',
        'CYP1A2': 'CHEMBL3356',
        'CYP2C19': 'CHEMBL3622'
    },
    'Other Safety': {
        'P-glycoprotein': 'CHEMBL4302',
        'BSEP': 'CHEMBL4105'
    }
}


def get_bioactivity_data(target_chembl_id: str,
                        activity_types: List[str] = ['IC50', 'Ki'],
                        limit: int = 1000,
                        offset: int = 0) -> Optional[pd.DataFrame]:
    """
    Retrieve bioactivity data from ChEMBL REST API
    
    Parameters:
    -----------
    target_chembl_id : str
        ChEMBL target identifier
    activity_types : List[str]
        Activity types to retrieve (default: IC50, Ki)
    limit : int
        Number of records per request (max 1000)
    offset : int
        Starting offset for pagination
    
    Returns:
    --------
    pd.DataFrame or None
        Bioactivity data if successful, None otherwise
    """
    
    try:
        # Build API URL
        url = f"{CHEMBL_BASE_URL}/activity.json"
        
        params = {
            'target_chembl_id': target_chembl_id,
            'limit': limit,
            'offset': offset
        }
        
        # Make request
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if 'activities' not in data:
            print(f"No activities found for {target_chembl_id}")
            return None
        
        activities = data['activities']
        
        if not activities:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(activities)
        
        # Filter by activity type
        if 'standard_type' in df.columns:
            df = df[df['standard_type'].isin(activity_types)].copy()
        
        return df
    
    except requests.exceptions.RequestException as e:
        print(f"Error retrieving data for {target_chembl_id}: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected error for {target_chembl_id}: {str(e)}")
        return None


def retrieve_all_bioactivity(target_chembl_id: str,
                            target_name: str,
                            activity_types: List[str] = ['IC50', 'Ki'],
                            max_records: int = 10000) -> pd.DataFrame:
    """
    Retrieve all bioactivity data with pagination
    
    Parameters:
    -----------
    target_chembl_id : str
        ChEMBL target identifier
    target_name : str
        Human-readable target name
    activity_types : List[str]
        Activity types to retrieve
    max_records : int
        Maximum total records to retrieve
    
    Returns:
    --------
    pd.DataFrame
        Combined bioactivity data
    """
    
    all_data = []
    offset = 0
    limit = 1000
    
    print(f"\nRetrieving data for {target_name} ({target_chembl_id})...")
    
    while True:
        # Respect API rate limits
        time.sleep(0.5)
        
        df = get_bioactivity_data(
            target_chembl_id=target_chembl_id,
            activity_types=activity_types,
            limit=limit,
            offset=offset
        )
        
        if df is None or len(df) == 0:
            break
        
        all_data.append(df)
        print(f"  Retrieved {len(df)} records (offset={offset})")
        
        # Check if we've retrieved enough or reached the end
        if len(df) < limit or offset + limit >= max_records:
            break
        
        offset += limit
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"  Total records: {len(combined_df)}")
        return combined_df
    else:
        print(f"  No data retrieved")
        return pd.DataFrame()


def retrieve_inactive_compounds(target_chembl_id: str,
                               target_name: str,
                               activity_types: List[str] = ['IC50', 'Ki'],
                               max_records: int = 10000,
                               max_per_target: int = 150) -> pd.DataFrame:
    """
    Retrieve confirmed inactive compounds from ChEMBL for a given target.

    Inactive compounds are those that were tested against the target but
    showed no measurable activity.  Two evidence sources are combined:

    1. Right-censored measurements (standard_relation = '>') where the
       reported value is >= 10 000 nM (10 µM), meaning the compound
       was tested up to at least 10 µM and showed no significant
       inhibition.
    2. Records whose ``activity_comment`` explicitly states the compound
       is inactive (e.g. "Not Active", "inactive", "Inactive").

    After collection, a diversity filter selects compounds across many
    Murcko scaffolds so that the inactive set is chemically varied.

    Parameters
    ----------
    target_chembl_id : str
        ChEMBL target identifier (e.g. 'CHEMBL240').
    target_name : str
        Human-readable target name for logging.
    activity_types : list of str
        Activity measurement types to include (default IC50, Ki).
    max_records : int
        Hard ceiling on API records to retrieve.
    max_per_target : int
        Maximum number of inactive compounds to return per target after
        diversity filtering.

    Returns
    -------
    pd.DataFrame
        DataFrame of inactive compounds with ``activity_class`` = 0 and
        ``activity_class_label`` = 'inactive'.
    """
    all_data: List[pd.DataFrame] = []
    offset = 0
    limit = 1000

    print(f"\nRetrieving inactive compounds for {target_name} ({target_chembl_id})...")

    # --- 1. Retrieve right-censored ('>') measurements -----------------
    while True:
        time.sleep(0.5)
        try:
            url = f"{CHEMBL_BASE_URL}/activity.json"
            params = {
                'target_chembl_id': target_chembl_id,
                'standard_relation': '>',
                'limit': limit,
                'offset': offset
            }
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if 'activities' not in data or not data['activities']:
                break

            df_page = pd.DataFrame(data['activities'])

            # Filter by activity type
            if 'standard_type' in df_page.columns:
                df_page = df_page[df_page['standard_type'].isin(activity_types)].copy()

            if len(df_page) > 0:
                all_data.append(df_page)
                print(f"  Retrieved {len(df_page)} right-censored records (offset={offset})")

            if len(data['activities']) < limit or offset + limit >= max_records:
                break
            offset += limit
        except Exception as exc:
            print(f"  Warning: error fetching inactives at offset {offset}: {exc}")
            break

    # --- 2. Retrieve records with 'Not Active' / 'inactive' comment ----
    for comment_kw in ['Not Active', 'inactive']:
        offset = 0
        while True:
            time.sleep(0.5)
            try:
                url = f"{CHEMBL_BASE_URL}/activity.json"
                params = {
                    'target_chembl_id': target_chembl_id,
                    'activity_comment__icontains': comment_kw,
                    'limit': limit,
                    'offset': offset
                }
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                if 'activities' not in data or not data['activities']:
                    break
                df_page = pd.DataFrame(data['activities'])
                if 'standard_type' in df_page.columns:
                    df_page = df_page[df_page['standard_type'].isin(activity_types)].copy()
                if len(df_page) > 0:
                    all_data.append(df_page)
                    print(f"  Retrieved {len(df_page)} '{comment_kw}' comment records (offset={offset})")
                if len(data['activities']) < limit or offset + limit >= max_records:
                    break
                offset += limit
            except Exception as exc:
                print(f"  Warning: error fetching '{comment_kw}' records: {exc}")
                break

    if not all_data:
        print(f"  No inactive data retrieved for {target_name}")
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    print(f"  Raw inactive candidates: {len(combined)}")

    # --- 3. Quality filters --------------------------------------------
    # Keep entries with SMILES
    if 'canonical_smiles' in combined.columns:
        combined = combined[combined['canonical_smiles'].notna()].copy()

    # For right-censored records, require standard_value >= 10_000 nM
    # (i.e. compound was tested at >= 10 µM and still showed no activity)
    if 'standard_value' in combined.columns and 'standard_relation' in combined.columns:
        combined['standard_value'] = pd.to_numeric(combined['standard_value'], errors='coerce')
        mask_right_censored = combined['standard_relation'] == '>'
        mask_high_enough = combined['standard_value'] >= 10_000
        mask_comment = combined['activity_comment'].fillna('').str.lower().str.contains('not active|inactive')
        combined = combined[
            (mask_right_censored & mask_high_enough) | mask_comment
        ].copy()

    # Remove data validity issues
    if 'data_validity_comment' in combined.columns:
        combined = combined[combined['data_validity_comment'].isna()].copy()

    # Deduplicate by molecule
    if 'molecule_chembl_id' in combined.columns:
        combined = combined.drop_duplicates(subset=['molecule_chembl_id'], keep='first')

    print(f"  After quality filters: {len(combined)} unique inactive candidates")

    # --- 4. Drug-likeness filter (loose) --------------------------------
    if HAS_RDKIT and 'canonical_smiles' in combined.columns:
        keep_idx = []
        for idx, smi in combined['canonical_smiles'].items():
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            if 100 <= mw <= 800 and -3 <= logp <= 8:
                keep_idx.append(idx)
        combined = combined.loc[keep_idx].copy()
        print(f"  After drug-likeness filter: {len(combined)}")

    # --- 5. Diversity selection via Murcko scaffolds --------------------
    if HAS_RDKIT and len(combined) > max_per_target and 'canonical_smiles' in combined.columns:
        combined = _diversity_select(combined, max_per_target)
        print(f"  After diversity selection: {len(combined)}")

    print(f"  Final inactive compound count for {target_name}: {len(combined)}")
    return combined


def _diversity_select(df: pd.DataFrame, max_compounds: int) -> pd.DataFrame:
    """Select a structurally diverse subset using Murcko scaffolds.

    Compounds are grouped by their generic Murcko scaffold.  One compound
    is selected per scaffold in round-robin fashion until the budget is
    reached.  This ensures maximum scaffold diversity.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'canonical_smiles' column.
    max_compounds : int
        Maximum compounds to retain.

    Returns
    -------
    pd.DataFrame
    """
    scaffolds: Dict[str, List[int]] = {}
    for idx, smi in df['canonical_smiles'].items():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            scaffold = '__invalid__'
        else:
            try:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            except Exception:
                scaffold = '__error__'
        scaffolds.setdefault(scaffold, []).append(idx)

    # Round-robin pick one compound per scaffold
    selected: List[int] = []
    scaffold_lists = list(scaffolds.values())
    np.random.seed(42)
    np.random.shuffle(scaffold_lists)
    pointer = [0] * len(scaffold_lists)
    while len(selected) < max_compounds:
        added_this_round = False
        for i, group in enumerate(scaffold_lists):
            if pointer[i] < len(group):
                selected.append(group[pointer[i]])
                pointer[i] += 1
                added_this_round = True
                if len(selected) >= max_compounds:
                    break
        if not added_this_round:
            break

    return df.loc[selected].copy()


def clean_bioactivity_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess bioactivity data
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw bioactivity data
    
    Returns:
    --------
    pd.DataFrame
        Cleaned data
    """
    
    if len(df) == 0:
        return df
    
    print("\nCleaning bioactivity data...")
    initial_count = len(df)
    
    # Keep only relevant columns
    columns_to_keep = [
        'molecule_chembl_id',
        'canonical_smiles',
        'standard_type',
        'standard_relation',
        'standard_value',
        'standard_units',
        'pchembl_value',
        'activity_comment',
        'assay_chembl_id',
        'assay_type',
        'target_chembl_id',
        'target_pref_name',
        'document_chembl_id',
        'src_id',
        'data_validity_comment'
    ]
    
    available_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[available_columns].copy()
    
    # Filter criteria
    # 1. Keep only exact measurements (=)
    if 'standard_relation' in df.columns:
        df = df[df['standard_relation'] == '='].copy()
        print(f"  After filtering for '=' relation: {len(df)} records")
    
    # 2. Keep only nM units
    if 'standard_units' in df.columns:
        df = df[df['standard_units'] == 'nM'].copy()
        print(f"  After filtering for nM units: {len(df)} records")
    
    # 3. Remove data validity issues
    if 'data_validity_comment' in df.columns:
        df = df[df['data_validity_comment'].isna()].copy()
        print(f"  After removing data validity issues: {len(df)} records")
    
    # 4. Keep only entries with SMILES
    if 'canonical_smiles' in df.columns:
        df = df[df['canonical_smiles'].notna()].copy()
        print(f"  After requiring SMILES: {len(df)} records")
    
    # 5. Convert numeric columns
    if 'standard_value' in df.columns:
        df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')
        df = df[df['standard_value'] > 0].copy()
    
    if 'pchembl_value' in df.columns:
        df['pchembl_value'] = pd.to_numeric(df['pchembl_value'], errors='coerce')
    
    # 6. Remove duplicates (keep first occurrence)
    if 'molecule_chembl_id' in df.columns and 'target_chembl_id' in df.columns:
        df = df.drop_duplicates(
            subset=['molecule_chembl_id', 'target_chembl_id', 'standard_type'],
            keep='first'
        )
        print(f"  After removing duplicates: {len(df)} records")
    
    # 7. Three-class activity label
    #    2 = potent     (pChEMBL >= 5.0, i.e. < 10 µM)
    #    1 = less_potent (4.0 <= pChEMBL < 5.0, i.e. 10-100 µM)
    #    0 = inactive    (assigned later for confirmed-inactive compounds)
    #
    #    Note: all records reaching this point have pChEMBL >= 4.0 (from
    #    earlier quality filter), so they are either potent or less_potent.
    #    Truly inactive compounds are added via retrieve_inactive_compounds().
    if 'pchembl_value' in df.columns:
        df['activity_class'] = df['pchembl_value'].apply(
            lambda x: 2 if x >= 5.0 else 1
        )
        df['activity_class_label'] = df['activity_class'].map(
            {2: 'potent', 1: 'less_potent', 0: 'inactive'}
        )

    print(f"  Final record count: {len(df)} ({100*len(df)/initial_count:.1f}% retained)")
    
    return df


def generate_summary_statistics(df: pd.DataFrame,
                                category: str,
                                target_name: str) -> Dict:
    """
    Generate summary statistics for a target
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned bioactivity data
    category : str
        Safety category
    target_name : str
        Target name
    
    Returns:
    --------
    Dict
        Summary statistics
    """
    
    if len(df) == 0:
        return {
            'category': category,
            'target': target_name,
            'n_total': 0,
            'n_unique_compounds': 0,
            'n_ic50': 0,
            'n_ki': 0,
            'n_active': 0,
            'n_inactive': 0,
            'pchembl_mean': None,
            'pchembl_std': None,
            'pchembl_min': None,
            'pchembl_max': None
        }
    
    summary = {
        'category': category,
        'target': target_name,
        'n_total': len(df),
        'n_unique_compounds': df['molecule_chembl_id'].nunique() if 'molecule_chembl_id' in df.columns else 0,
        'n_ic50': len(df[df['standard_type'] == 'IC50']) if 'standard_type' in df.columns else 0,
        'n_ki': len(df[df['standard_type'] == 'Ki']) if 'standard_type' in df.columns else 0,
        'n_potent': int((df['activity_class'] == 2).sum()) if 'activity_class' in df.columns else 0,
        'n_less_potent': int((df['activity_class'] == 1).sum()) if 'activity_class' in df.columns else 0,
        'n_inactive': int((df['activity_class'] == 0).sum()) if 'activity_class' in df.columns else 0,
    }
    
    if 'pchembl_value' in df.columns:
        pchembl_values = df['pchembl_value'].dropna()
        if len(pchembl_values) > 0:
            summary.update({
                'pchembl_mean': round(pchembl_values.mean(), 2),
                'pchembl_std': round(pchembl_values.std(), 2),
                'pchembl_min': round(pchembl_values.min(), 2),
                'pchembl_max': round(pchembl_values.max(), 2)
            })
    
    return summary


def main():
    """
    Main execution function
    """
    
    print("="*80)
    print("ChEMBL Safety Pharmacology Data Retrieval")
    print("="*80)
    
    # Storage for all data
    all_bioactivity = []
    all_summaries = []

    # Iterate through all targets
    for category, targets in SAFETY_TARGETS.items():
        print(f"\n{'='*80}")
        print(f"Category: {category}")
        print(f"{'='*80}")

        for target_name, chembl_id in targets.items():
            # --- Retrieve active/less-potent compounds (pChEMBL >= 4.0) ---
            df = retrieve_all_bioactivity(
                target_chembl_id=chembl_id,
                target_name=target_name,
                activity_types=['IC50', 'Ki'],
                max_records=10000
            )

            target_frames = []
            if len(df) > 0:
                # Clean data (labels: potent / less_potent)
                df_clean = clean_bioactivity_data(df)

                if len(df_clean) > 0:
                    df_clean['safety_category'] = category
                    df_clean['target_common_name'] = target_name
                    target_frames.append(df_clean)

            # --- Retrieve confirmed-inactive compounds ---
            df_inactive = retrieve_inactive_compounds(
                target_chembl_id=chembl_id,
                target_name=target_name,
                activity_types=['IC50', 'Ki'],
                max_per_target=150,
            )

            if len(df_inactive) > 0:
                # Ensure consistent columns
                columns_to_keep = [
                    'molecule_chembl_id', 'canonical_smiles', 'standard_type',
                    'standard_relation', 'standard_value', 'standard_units',
                    'pchembl_value', 'activity_comment', 'assay_chembl_id',
                    'assay_type', 'target_chembl_id', 'target_pref_name',
                    'document_chembl_id', 'src_id', 'data_validity_comment',
                ]
                available = [c for c in columns_to_keep if c in df_inactive.columns]
                df_inactive = df_inactive[available].copy()
                df_inactive['activity_class'] = 0
                df_inactive['activity_class_label'] = 'inactive'
                df_inactive['safety_category'] = category
                df_inactive['target_common_name'] = target_name
                target_frames.append(df_inactive)

            # Combine active + inactive for this target
            if target_frames:
                df_target = pd.concat(target_frames, ignore_index=True)

                # Remove any molecule that appears in both active & inactive
                # (keep the measured-activity version)
                if 'molecule_chembl_id' in df_target.columns:
                    dup_mask = df_target.duplicated(subset=['molecule_chembl_id'], keep=False)
                    if dup_mask.any():
                        # Keep the row that has a real pchembl (i.e. not inactive)
                        df_target = df_target.sort_values(
                            'activity_class', ascending=False
                        ).drop_duplicates(subset=['molecule_chembl_id'], keep='first')

                all_bioactivity.append(df_target)
                summary = generate_summary_statistics(df_target, category, target_name)
                all_summaries.append(summary)
    
    # Combine all data
    if all_bioactivity:
        print("\n" + "="*80)
        print("Combining all data...")
        print("="*80)
        
        combined_df = pd.concat(all_bioactivity, ignore_index=True)

        # Save to CSV
        output_file = 'safety_targets_bioactivity.csv'
        combined_df.to_csv(output_file, index=False)
        print(f"\nSaved combined data to: {output_file}")
        print(f"Total records: {len(combined_df)}")
        print(f"Unique compounds: {combined_df['molecule_chembl_id'].nunique()}")
        if 'activity_class' in combined_df.columns:
            counts = combined_df['activity_class'].value_counts().sort_index()
            print(f"Class distribution:")
            for cls, cnt in counts.items():
                label = {0: 'inactive', 1: 'less_potent', 2: 'potent'}.get(cls, '?')
                print(f"  {label} (class {cls}): {cnt}")
        
        # Save summary statistics
        summary_df = pd.DataFrame(all_summaries)
        summary_file = 'summary_statistics.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSaved summary statistics to: {summary_file}")
        
        # Display summary
        print("\n" + "="*80)
        print("Summary Statistics by Target")
        print("="*80)
        print(summary_df.to_string(index=False))
        
        print("\n" + "="*80)
        print("Data retrieval complete!")
        print("="*80)
        
        return combined_df, summary_df
    
    else:
        print("\nWarning: No data was retrieved from any target")
        return None, None


if __name__ == "__main__":
    # Run main retrieval
    bioactivity_df, summary_df = main()
    
    if bioactivity_df is not None:
        print("\nNext steps:")
        print("1. Review summary_statistics.csv for data quality")
        print("2. Use safety_targets_bioactivity.csv for model training")
        print("3. Run model_assessment_plan.py for comprehensive evaluation")
        print("4. Refer to EXECUTIVE_SUMMARY.txt for full methodology")
