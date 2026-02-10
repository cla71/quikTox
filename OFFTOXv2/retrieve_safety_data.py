"""
Script to retrieve bioactivity data for safety targets from ChEMBL database
Focuses on IC50 and Ki measurements for key safety pharmacology endpoints
"""

import pandas as pd
import time
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Define safety targets with their ChEMBL IDs and categories
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

# Activity types to retrieve
ACTIVITY_TYPES = ['IC50', 'Ki']

def get_chembl_bioactivity_data(target_chembl_id: str, 
                                target_name: str,
                                activity_type: str,
                                min_pchembl: float = 4.0,
                                limit: int = 1000) -> pd.DataFrame:
    """
    Retrieve bioactivity data from ChEMBL for a specific target
    
    Parameters:
    -----------
    target_chembl_id : str
        ChEMBL target identifier
    target_name : str
        Human-readable target name
    activity_type : str
        Type of activity measurement (IC50, Ki, etc.)
    min_pchembl : float
        Minimum pChEMBL value to filter (higher = more potent)
    limit : int
        Maximum number of records to retrieve
    
    Returns:
    --------
    pd.DataFrame
        Bioactivity data with standardized columns
    """
    
    # Note: This is a template function. In actual implementation, 
    # you would use the ChEMBL API or web service client
    # For now, creating placeholder structure
    
    print(f"Retrieving {activity_type} data for {target_name} ({target_chembl_id})...")
    
    # Placeholder - in real implementation, this would call ChEMBL API
    # Example structure of what we'd retrieve:
    columns = [
        'molecule_chembl_id',
        'canonical_smiles',
        'standard_value',
        'standard_units',
        'standard_relation',
        'pchembl_value',
        'activity_comment',
        'assay_type',
        'assay_chembl_id',
        'document_chembl_id',
        'target_chembl_id',
        'target_name',
        'activity_type',
        'data_validity_comment'
    ]
    
    return pd.DataFrame(columns=columns)


def compile_safety_dataset() -> pd.DataFrame:
    """
    Compile comprehensive safety pharmacology dataset
    
    Returns:
    --------
    pd.DataFrame
        Combined dataset with all safety targets
    """
    
    all_data = []
    
    for category, targets in SAFETY_TARGETS.items():
        for target_name, chembl_id in targets.items():
            for activity_type in ACTIVITY_TYPES:
                print(f"\n{'='*60}")
                print(f"Category: {category}")
                print(f"Target: {target_name} ({chembl_id})")
                print(f"Activity Type: {activity_type}")
                print(f"{'='*60}")
                
                # Retrieve data
                df = get_chembl_bioactivity_data(
                    target_chembl_id=chembl_id,
                    target_name=target_name,
                    activity_type=activity_type,
                    min_pchembl=4.0,
                    limit=1000
                )
                
                if len(df) > 0:
                    df['safety_category'] = category
                    df['target_common_name'] = target_name
                    all_data.append(df)
                    print(f"Retrieved {len(df)} records")
                else:
                    print(f"No data retrieved")
                
                # Be respectful to API
                time.sleep(0.5)
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\n{'='*60}")
        print(f"Total records retrieved: {len(combined_df)}")
        print(f"{'='*60}")
        return combined_df
    else:
        print("\nWarning: No data retrieved from any target")
        return pd.DataFrame()


def clean_and_preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the bioactivity data
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw bioactivity data
    
    Returns:
    --------
    pd.DataFrame
        Cleaned and preprocessed data
    """
    
    if len(df) == 0:
        return df
    
    print("\nCleaning and preprocessing data...")
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['molecule_chembl_id', 'target_chembl_id', 'activity_type'])
    
    # Filter for valid measurements (standard_relation = '=')
    if 'standard_relation' in df.columns:
        df = df[df['standard_relation'] == '='].copy()
    
    # Remove entries with data validity issues
    if 'data_validity_comment' in df.columns:
        df = df[df['data_validity_comment'].isna()].copy()
    
    # Convert activity values to numeric
    if 'standard_value' in df.columns:
        df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')
    
    if 'pchembl_value' in df.columns:
        df['pchembl_value'] = pd.to_numeric(df['pchembl_value'], errors='coerce')
    
    # Remove entries without SMILES
    if 'canonical_smiles' in df.columns:
        df = df[df['canonical_smiles'].notna()].copy()
    
    # Three-class activity labels:
    #   2 = potent      (pChEMBL >= 5.0, i.e. < 10 µM)
    #   1 = less_potent  (4.0 <= pChEMBL < 5.0, i.e. 10-100 µM)
    #   0 = inactive     (confirmed-inactive compounds added separately)
    # Records reaching this point have pChEMBL >= 4.0 (quality filter).
    if 'pchembl_value' in df.columns:
        df['activity_class'] = df['pchembl_value'].apply(
            lambda x: 2 if x >= 5.0 else 1
        )
        df['activity_class_label'] = df['activity_class'].map(
            {2: 'potent', 1: 'less_potent', 0: 'inactive'}
        )
        # Keep is_active for backward compatibility
        df['is_active'] = (df['pchembl_value'] >= 5.0).astype(int)
    
    print(f"Records after cleaning: {len(df)}")
    
    return df


def generate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned bioactivity data
    
    Returns:
    --------
    pd.DataFrame
        Summary statistics by target
    """
    
    if len(df) == 0:
        return pd.DataFrame()
    
    agg_dict = {
        'molecule_chembl_id': 'count',
        'pchembl_value': ['mean', 'std', 'min', 'max'],
    }
    if 'activity_class' in df.columns:
        agg_dict['activity_class'] = [
            lambda x: (x == 2).sum(),  # n_potent
            lambda x: (x == 1).sum(),  # n_less_potent
            lambda x: (x == 0).sum(),  # n_inactive
        ]
    else:
        agg_dict['is_active'] = 'sum'

    summary = df.groupby(['safety_category', 'target_common_name', 'activity_type']).agg(
        agg_dict
    ).round(2)

    summary.columns = ['_'.join(str(c) for c in col).strip() for col in summary.columns.values]
    summary = summary.rename(columns={
        'molecule_chembl_id_count': 'n_compounds',
    })
    
    return summary.reset_index()


if __name__ == "__main__":
    
    print("="*80)
    print("ChEMBL Safety Pharmacology Data Retrieval")
    print("="*80)
    
    # Print target summary
    print("\nTargets to retrieve:")
    for category, targets in SAFETY_TARGETS.items():
        print(f"\n{category}:")
        for name, chembl_id in targets.items():
            print(f"  - {name}: {chembl_id}")
    
    print(f"\nActivity types: {', '.join(ACTIVITY_TYPES)}")
    
    # Note for user
    print("\n" + "="*80)
    print("NOTE: This script provides the framework for data retrieval.")
    print("Due to API connectivity issues, manual data export from ChEMBL")
    print("web interface may be required. See instructions below.")
    print("="*80)
    
    # Instructions for manual data export
    print("\n" + "="*80)
    print("MANUAL DATA EXPORT INSTRUCTIONS:")
    print("="*80)
    print("""
1. Go to https://www.ebi.ac.uk/chembl/
2. For each target ChEMBL ID:
   - Search for the target ID (e.g., CHEMBL240)
   - Navigate to the "Bioactivities" tab
   - Filter by:
     * Standard Type: IC50, Ki
     * Standard Relation: =
     * Standard Units: nM
   - Export results as CSV
   - Save with naming convention: {target_name}_{activity_type}.csv
   
3. Place all CSV files in the current directory
4. Run the consolidation script to merge all data

Alternatively, you can use ChEMBL web services API directly:
https://www.ebi.ac.uk/chembl/api/data/docs
""")
    
    print("\n" + "="*80)
    print("Script execution complete.")
    print("="*80)
