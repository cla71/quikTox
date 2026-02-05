#!/usr/bin/env python3
"""
ChEMBL Safety Data Retrieval Script
====================================

Direct implementation using ChEMBL REST API
Retrieves IC50 and Ki data for all safety targets
Includes robust error handling and data quality checks

Usage:
    python retrieve_chembl_safety_data.py

Output:
    safety_targets_bioactivity.csv - Combined dataset
    summary_statistics.csv - Data quality summary
"""

import requests
import pandas as pd
import time
import json
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

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
    
    # 7. Add binary activity label (pChEMBL >= 6.0 = active, 1 µM or better)
    if 'pchembl_value' in df.columns:
        df['is_active'] = (df['pchembl_value'] >= 6.0).astype(int)
    
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
        'n_active': int(df['is_active'].sum()) if 'is_active' in df.columns else 0,
        'n_inactive': int((1 - df['is_active']).sum()) if 'is_active' in df.columns else 0
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
            # Retrieve data
            df = retrieve_all_bioactivity(
                target_chembl_id=chembl_id,
                target_name=target_name,
                activity_types=['IC50', 'Ki'],
                max_records=10000
            )
            
            if len(df) > 0:
                # Clean data
                df_clean = clean_bioactivity_data(df)
                
                if len(df_clean) > 0:
                    # Add metadata
                    df_clean['safety_category'] = category
                    df_clean['target_common_name'] = target_name
                    
                    # Store
                    all_bioactivity.append(df_clean)
                    
                    # Generate summary
                    summary = generate_summary_statistics(df_clean, category, target_name)
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
