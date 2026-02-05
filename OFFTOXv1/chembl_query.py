#!/usr/bin/env python3
"""
Query ChEMBL for comprehensive NHR and ADMET safety target dataset
Retrieves IC50/Ki bioactivity data for ML model assessment
"""

import requests
import pandas as pd
import time
from typing import List, Dict
import json

# Comprehensive target panel
TARGETS = {
    # Nuclear Hormone Receptors (NHRs)
    'ER_alpha': 'CHEMBL206',
    'ER_beta': 'CHEMBL242',
    'AR': 'CHEMBL1871',
    'GR': 'CHEMBL2034',
    'PR': 'CHEMBL208',
    'MR': 'CHEMBL1994',
    'PPARg': 'CHEMBL239',
    'PXR': 'CHEMBL2498',
    'CAR': 'CHEMBL2248',
    'LXRa': 'CHEMBL5231',
    'LXRb': 'CHEMBL4309',
    'FXR': 'CHEMBL2001',
    'RXRa': 'CHEMBL2260',
    'VDR': 'CHEMBL1977',

    # Cardiac Safety
    'hERG': 'CHEMBL240',
    'Cav1.2': 'CHEMBL1940',
    'Nav1.5': 'CHEMBL1993',

    # Hepatotoxicity (CYPs)
    'CYP3A4': 'CHEMBL340',
    'CYP2D6': 'CHEMBL289',
    'CYP2C9': 'CHEMBL3397',
    'CYP1A2': 'CHEMBL3356',
    'CYP2C19': 'CHEMBL3622',

    # Transporters
    'P-gp': 'CHEMBL4302',
    'BSEP': 'CHEMBL4105',
}

BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"


def query_bioactivity(target_chembl_id: str, target_name: str,
                      activity_types: List[str] = ['IC50', 'Ki'],
                      min_pchembl: float = 5.0,
                      limit: int = 1000) -> pd.DataFrame:
    """
    Query ChEMBL bioactivity for a specific target

    Parameters:
    -----------
    target_chembl_id : str
        ChEMBL target ID
    target_name : str
        Human-readable target name
    activity_types : list
        Activity types to query (IC50, Ki, etc.)
    min_pchembl : float
        Minimum pChEMBL value filter (higher = more potent)
    limit : int
        Max results per query

    Returns:
    --------
    pd.DataFrame with bioactivity data
    """
    all_data = []

    for act_type in activity_types:
        print(f"Querying {target_name} ({target_chembl_id}) for {act_type}...")

        params = {
            'target_chembl_id': target_chembl_id,
            'standard_type': act_type,
            'pchembl_value__gte': min_pchembl,
            'limit': limit,
            'format': 'json'
        }

        try:
            response = requests.get(
                f"{BASE_URL}/activity.json",
                params=params,
                timeout=60
            )
            response.raise_for_status()
            data = response.json()

            if 'activities' in data:
                for activity in data['activities']:
                    row = {
                        'target_name': target_name,
                        'target_chembl_id': target_chembl_id,
                        'molecule_chembl_id': activity.get('molecule_chembl_id'),
                        'canonical_smiles': activity.get('canonical_smiles'),
                        'standard_type': activity.get('standard_type'),
                        'standard_relation': activity.get('standard_relation'),
                        'standard_value': activity.get('standard_value'),
                        'standard_units': activity.get('standard_units'),
                        'pchembl_value': activity.get('pchembl_value'),
                        'activity_comment': activity.get('activity_comment'),
                        'assay_chembl_id': activity.get('assay_chembl_id'),
                        'assay_type': activity.get('assay_type'),
                        'assay_description': activity.get('assay_description'),
                        'document_chembl_id': activity.get('document_chembl_id'),
                        'src_id': activity.get('src_id'),
                        'data_validity_comment': activity.get('data_validity_comment'),
                    }
                    all_data.append(row)

                print(f"  Retrieved {len(data['activities'])} records")
            else:
                print(f"  No activities found")

            time.sleep(0.5)  # Rate limiting

        except Exception as e:
            print(f"  Error querying {target_name} - {act_type}: {e}")
            continue

    if all_data:
        return pd.DataFrame(all_data)
    else:
        return pd.DataFrame()


def get_molecule_properties(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich dataset with molecular properties from ChEMBL

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with molecule_chembl_id column

    Returns:
    --------
    pd.DataFrame with added molecular properties
    """
    unique_mols = df['molecule_chembl_id'].dropna().unique()
    print(f"\nRetrieving properties for {len(unique_mols)} unique molecules...")

    mol_props = []
    batch_size = 50

    for i in range(0, len(unique_mols), batch_size):
        batch = unique_mols[i:i + batch_size]
        chembl_ids = ','.join(batch)

        try:
            response = requests.get(
                f"{BASE_URL}/molecule.json",
                params={'molecule_chembl_id__in': chembl_ids, 'limit': batch_size},
                timeout=60
            )
            response.raise_for_status()
            data = response.json()

            if 'molecules' in data:
                for mol in data['molecules']:
                    props = mol.get('molecule_properties', {})
                    mol_props.append({
                        'molecule_chembl_id': mol.get('molecule_chembl_id'),
                        'pref_name': mol.get('pref_name'),
                        'max_phase': mol.get('max_phase'),
                        'molecular_weight': props.get('full_mwt'),
                        'alogp': props.get('alogp'),
                        'psa': props.get('psa'),
                        'hba': props.get('hba'),
                        'hbd': props.get('hbd'),
                        'rtb': props.get('rtb'),
                        'aromatic_rings': props.get('aromatic_rings'),
                        'qed_weighted': props.get('qed_weighted'),
                        'num_ro5_violations': props.get('num_ro5_violations'),
                    })

            time.sleep(0.3)
            if (i // batch_size) % 10 == 0:
                print(f"  Processed {i + len(batch)}/{len(unique_mols)} molecules")

        except Exception as e:
            print(f"  Error retrieving properties for batch {i}: {e}")
            continue

    if mol_props:
        props_df = pd.DataFrame(mol_props)
        return df.merge(props_df, on='molecule_chembl_id', how='left')
    else:
        return df


def main():
    """Main execution function"""
    print("=" * 80)
    print("ChEMBL NHR & ADMET Safety Target Dataset Builder")
    print("=" * 80)
    print(f"\nQuerying {len(TARGETS)} targets for IC50/Ki data...")
    print(f"Minimum pChEMBL threshold: 5.0 (10 µM)")
    print()

    all_bioactivity = []

    # Query each target
    for target_name, target_id in TARGETS.items():
        df = query_bioactivity(
            target_chembl_id=target_id,
            target_name=target_name,
            activity_types=['IC50', 'Ki'],
            min_pchembl=5.0,
            limit=1000
        )

        if not df.empty:
            all_bioactivity.append(df)
            print(f"✓ {target_name}: {len(df)} records\n")
        else:
            print(f"✗ {target_name}: No data\n")

    if not all_bioactivity:
        print("\nNo bioactivity data retrieved!")
        return

    # Combine all data
    combined_df = pd.concat(all_bioactivity, ignore_index=True)
    print(f"\n{'=' * 80}")
    print(f"Total bioactivity records: {len(combined_df)}")
    print(f"Unique compounds: {combined_df['molecule_chembl_id'].nunique()}")
    print(f"Unique targets: {combined_df['target_chembl_id'].nunique()}")

    # Enrich with molecular properties
    enriched_df = get_molecule_properties(combined_df)

    # Save to CSV
    output_file = '/mnt/user-data/outputs/chembl_nhr_admet_dataset.csv'
    enriched_df.to_csv(output_file, index=False)
    print(f"\n✓ Dataset saved to: {output_file}")

    # Generate summary statistics
    print(f"\n{'=' * 80}")
    print("DATASET SUMMARY")
    print(f"{'=' * 80}")
    print(f"\nRecords per target:")
    print(enriched_df.groupby('target_name').size().sort_values(ascending=False))

    print(f"\nActivity type distribution:")
    print(enriched_df['standard_type'].value_counts())

    print(f"\npChEMBL value statistics:")
    print(enriched_df['pchembl_value'].describe())

    print(f"\nAssay type distribution:")
    print(enriched_df['assay_type'].value_counts())

    # Create metadata file
    metadata = {
        'targets_queried': TARGETS,
        'total_records': len(enriched_df),
        'unique_compounds': int(enriched_df['molecule_chembl_id'].nunique()),
        'unique_targets': int(enriched_df['target_chembl_id'].nunique()),
        'activity_types': enriched_df['standard_type'].value_counts().to_dict(),
        'pchembl_stats': enriched_df['pchembl_value'].describe().to_dict(),
    }

    with open('/mnt/user-data/outputs/dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Metadata saved to: /mnt/user-data/outputs/dataset_metadata.json")
    print(f"\n{'=' * 80}")
    print("COMPLETE!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()