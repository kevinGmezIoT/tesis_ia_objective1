#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
16_calculate_gower_similarity.py

Calcula la similitud de Gower visual-semántica entre identidades.
Utiliza atributos categóricos:
- upper_garment_type
- lower_garment_type
- upper_color_label
- lower_color_label
- backpack_flag

Ignora (no penaliza ni bonifica) atributos marcados como 'unknown' o vacíos.
Identifica pares de identidades "hard" basados en alta similitud semántica.
"""

import os
import argparse
import pandas as pd
import numpy as np
from itertools import combinations

def canonical_person_id(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.lower().startswith("id"):
        text = text[2:]
    try:
        return str(int(float(text)))
    except ValueError:
        return text.lower()

def calculate_gower_similarity(row_i, row_j, attributes):
    """
    Calcula similitud de Gower (para atributos nominales) entre dos filas,
    ignorando los valores 'unknown'.
    """
    matches = 0
    valid_comparisons = 0
    
    for attr in attributes:
        val_i = str(row_i[attr]).strip().lower()
        val_j = str(row_j[attr]).strip().lower()
        
        is_unknown_i = pd.isna(row_i[attr]) or val_i in ['unknown', 'nan', 'none', '']
        is_unknown_j = pd.isna(row_j[attr]) or val_j in ['unknown', 'nan', 'none', '']
        
        if not is_unknown_i and not is_unknown_j:
            valid_comparisons += 1
            if val_i == val_j:
                matches += 1
                
    if valid_comparisons == 0:
        return np.nan, 0
        
    return matches / valid_comparisons, valid_comparisons

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='dataset/output/metadata/identity_visual_attributes_pair_ready.csv')
    parser.add_argument('--out-dir', type=str, default='dataset/output/metadata')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)

    # Ensure necessary columns are present
    attributes = ['upper_garment_type', 'lower_garment_type', 'upper_color_label', 'lower_color_label', 'backpack_flag']
    id_col = 'person_global_id'
    
    # Normalize identifiers
    df[id_col] = df[id_col].apply(canonical_person_id)

    # Validate uniqueness
    if df[id_col].duplicated().any():
        duplicated = df.loc[df[id_col].duplicated(), id_col].tolist()
        raise ValueError(
            "Existen identidades repetidas en la tabla consolidada: "
            f"{duplicated[:10]}"
        )

    # We will compute a pairwise matrix
    n_ids = len(df)
    print(f"Total identities: {n_ids}")
    
    similarity_matrix = np.full((n_ids, n_ids), np.nan)
    distance_matrix = np.full((n_ids, n_ids), np.nan)
    
    pairs_data = []
    
    print("Calculating Gower pairwise similarities...")
    for i in range(n_ids):
        row_i = df.iloc[i]
        id_i = f"ID{int(row_i[id_col]):04d}" if str(row_i[id_col]).isdigit() else row_i[id_col]
        # Self-similarity is 1.0 (if there are valid attributes)
        sim_self, val_self = calculate_gower_similarity(row_i, row_i, attributes)
        similarity_matrix[i, i] = sim_self
        distance_matrix[i, i] = 0.0 if not np.isnan(sim_self) else np.nan
        
        for j in range(i + 1, n_ids):
            row_j = df.iloc[j]
            id_j = f"ID{int(row_j[id_col]):04d}" if str(row_j[id_col]).isdigit() else row_j[id_col]
            
            sim, val_comps = calculate_gower_similarity(row_i, row_j, attributes)
            
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim
            if not np.isnan(sim):
                distance_matrix[i, j] = 1.0 - sim
                distance_matrix[j, i] = 1.0 - sim
                
            # Helper logic for interpretable boolean columns
            def get_val(row, col):
                v = str(row[col]).strip().lower()
                if pd.isna(row[col]) or v in ['unknown', 'nan', 'none', '']:
                    return None
                return v

            u_garment_i = get_val(row_i, 'upper_garment_type')
            u_garment_j = get_val(row_j, 'upper_garment_type')
            same_u_garment = (u_garment_i == u_garment_j) if (u_garment_i is not None and u_garment_j is not None) else np.nan

            l_garment_i = get_val(row_i, 'lower_garment_type')
            l_garment_j = get_val(row_j, 'lower_garment_type')
            same_l_garment = (l_garment_i == l_garment_j) if (l_garment_i is not None and l_garment_j is not None) else np.nan

            u_color_i = get_val(row_i, 'upper_color_label')
            u_color_j = get_val(row_j, 'upper_color_label')
            same_u_color = (u_color_i == u_color_j) if (u_color_i is not None and u_color_j is not None) else np.nan

            l_color_i = get_val(row_i, 'lower_color_label')
            l_color_j = get_val(row_j, 'lower_color_label')
            same_l_color = (l_color_i == l_color_j) if (l_color_i is not None and l_color_j is not None) else np.nan

            bp_i = get_val(row_i, 'backpack_flag')
            bp_j = get_val(row_j, 'backpack_flag')
            
            both_have_bp = (bp_i == 'yes' and bp_j == 'yes') if (bp_i is not None and bp_j is not None) else np.nan
            both_no_bp = (bp_i == 'no' and bp_j == 'no') if (bp_i is not None and bp_j is not None) else np.nan
            bp_mismatch = (bp_i != bp_j) if (bp_i is not None and bp_j is not None) else np.nan

            pairs_data.append({
                'identity_x': id_i,
                'identity_y': id_j,
                'person_global_id_x': row_i[id_col],
                'person_global_id_y': row_j[id_col],
                'same_upper_garment_type': 1 if same_u_garment is True else (0 if same_u_garment is False else np.nan),
                'same_lower_garment_type': 1 if same_l_garment is True else (0 if same_l_garment is False else np.nan),
                'same_upper_color_label': 1 if same_u_color is True else (0 if same_u_color is False else np.nan),
                'same_lower_color_label': 1 if same_l_color is True else (0 if same_l_color is False else np.nan),
                'both_have_backpack': 1 if both_have_bp is True else (0 if both_have_bp is False else np.nan),
                'both_no_backpack': 1 if both_no_bp is True else (0 if both_no_bp is False else np.nan),
                'backpack_mismatch': 1 if bp_mismatch is True else (0 if bp_mismatch is False else np.nan),
                'n_comparable_attributes': val_comps,
                'gower_similarity': sim,
                'gower_distance': 1.0 - sim if not np.isnan(sim) else np.nan
            })

    # Save matrices
    ids_labels = [f"ID{int(x):04d}" if str(x).isdigit() else x for x in df[id_col]]
    sim_df = pd.DataFrame(similarity_matrix, index=ids_labels, columns=ids_labels)
    dist_df = pd.DataFrame(distance_matrix, index=ids_labels, columns=ids_labels)
    
    sim_out = os.path.join(args.out_dir, 'gower_identity_similarity_matrix.csv')
    dist_out = os.path.join(args.out_dir, 'gower_identity_distance_matrix.csv')
    sim_df.to_csv(sim_out)
    dist_df.to_csv(dist_out)
    print(f"Matrices saved to {sim_out} and {dist_out}")
    
    # Save pairs
    pairs_df = pd.DataFrame(pairs_data)
    pairs_out = os.path.join(args.out_dir, 'gower_identity_pairs.csv')
    pairs_df.to_csv(pairs_out, index=False)
    print(f"Pairs saved to {pairs_out} ({len(pairs_df)} pairs)")
    
    # Select Hard Pairs
    valid_sims = pairs_df['gower_similarity'].dropna()
    p90_sim = np.percentile(valid_sims, 90) if len(valid_sims) > 0 else 0
    print(f"P90 of Gower similarity among valid pairs: {p90_sim:.4f}")
    
    pairs_df['hard_gower_p90_flag'] = (
        (pairs_df['gower_similarity'] >= p90_sim)
        & (pairs_df['n_comparable_attributes'] >= 4)
    ).astype(int)

    pairs_df['hard_gower_fixed_080_flag'] = (
        (pairs_df['gower_similarity'] >= 0.80)
        & (pairs_df['n_comparable_attributes'] >= 4)
    ).astype(int)

    pairs_df['hard_gower_confirmed_flag'] = (
        (pairs_df['gower_similarity'] >= max(p90_sim, 0.80))
        & (pairs_df['n_comparable_attributes'] >= 4)
    ).astype(int)
    
    # Update pairs csv with the new flags
    pairs_df.to_csv(pairs_out, index=False)

    hard_pairs = pairs_df[pairs_df['hard_gower_confirmed_flag'] == 1].copy()
    hard_pairs_out = os.path.join(args.out_dir, 'gower_hard_identity_pairs.csv')
    hard_pairs.to_csv(hard_pairs_out, index=False)
    print(f"Hard pairs saved to {hard_pairs_out} ({len(hard_pairs)} pairs)")
    
    # Summary
    summary = {
        'total_identities': n_ids,
        'total_pairs': len(pairs_df),
        'valid_pairs': len(valid_sims),
        'p90_similarity': p90_sim,
        'median_similarity': np.median(valid_sims) if len(valid_sims) > 0 else np.nan,
        'hard_pairs_count': len(hard_pairs)
    }
    summary_df = pd.DataFrame([summary])
    summary_out = os.path.join(args.out_dir, 'gower_identity_summary.csv')
    summary_df.to_csv(summary_out, index=False)
    print(f"Summary saved to {summary_out}")

if __name__ == '__main__':
    main()
