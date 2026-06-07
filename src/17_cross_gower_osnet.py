#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
17_cross_gower_osnet.py

1. Carga los pares de similitud semántica (Gower).
2. Calcula los prototipos de identidad (y cámara-identidad) 
   usando embeddings de OSNet y CLIP.
3. Calcula la similitud visual (OSNet, CLIP) para cada par.
4. Calcula similitud cruzada entre cámaras.
5. Agrupa en los 4 cuadrantes de ambigüedad.
6. Integra factores de contexto (iluminación, oclusión, calidad).
7. Exporta el CSV final para análisis y correlaciones.
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

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

def normalize_l2(x):
    norm = np.linalg.norm(x)
    if norm == 0:
        return x
    return x / norm

def compute_identity_camera_prototypes(embeddings_df):
    embeddings_df = embeddings_df.copy()
    embeddings_df['person_global_id'] = embeddings_df['person_global_id'].apply(canonical_person_id)
    embeddings_df['cam_id'] = embeddings_df['cam_id'].astype(str).str.strip().str.lower()
    
    # Parse json strings
    print("  Parsing embeddings...")
    embeddings_df['vector'] = embeddings_df['embedding_json'].apply(json.loads).apply(np.array)
    
    print("  Computing camera-identity prototypes...")
    prototypes = {}
    for (pid, cam_id), group in embeddings_df.groupby(['person_global_id', 'cam_id']):
        vectors = np.vstack(group['vector'].values)
        vectors = np.vstack([normalize_l2(vector) for vector in vectors])
        mean_vec = normalize_l2(np.mean(vectors, axis=0))
        prototypes[(pid, cam_id)] = mean_vec
        
    return prototypes

def compute_camera_balanced_identity_prototypes(embeddings_df):
    cam_prototypes = compute_identity_camera_prototypes(embeddings_df)
    
    grouped = {}
    for (pid, cam_id), vector in cam_prototypes.items():
        grouped.setdefault(pid, []).append(vector)
        
    identity_prototypes = {}
    for pid, vectors in grouped.items():
        identity_prototypes[pid] = normalize_l2(np.mean(np.vstack(vectors), axis=0))
        
    return identity_prototypes, cam_prototypes

def compute_context_by_identity(illum_path, occ_path, qual_path):
    print("  Computing context variables by identity...")
    
    # Illumination
    illum_df = pd.read_csv(illum_path, usecols=['person_global_id', 'illum_y_mean', 'illum_y_cv', 'illum_label'])
    illum_df['person_global_id'] = illum_df['person_global_id'].apply(canonical_person_id)
    illum_df['is_low_illum'] = illum_df['illum_label'].isin(['baja', 'media_baja', 'muy_baja']).astype(int)
    
    illum_agg = illum_df.groupby('person_global_id').agg(
        mean_illum_y_mean=('illum_y_mean', 'mean'),
        mean_illum_y_cv=('illum_y_cv', 'mean'),
        low_illum_ratio=('is_low_illum', 'mean')
    ).reset_index()
    
    # Occlusion
    occ_df = pd.read_csv(occ_path, usecols=['person_global_id', 'occlusion_pct_mean', 'occlusion_label'])
    occ_df['person_global_id'] = occ_df['person_global_id'].apply(canonical_person_id)
    occ_df['is_heavy_occ'] = occ_df['occlusion_label'].isin(['parcial', 'alta', 'muy_alta']).astype(int)
    
    occ_agg = occ_df.groupby('person_global_id').agg(
        mean_occlusion_pct=('occlusion_pct_mean', 'mean'),
        heavy_occlusion_ratio=('is_heavy_occ', 'mean')
    ).reset_index()
    
    # Quality
    qual_df = pd.read_csv(qual_path, usecols=['person_global_id', 'sil_quality_mean', 'sil_quality_label'])
    qual_df['person_global_id'] = qual_df['person_global_id'].apply(canonical_person_id)
    qual_df['is_degraded'] = qual_df['sil_quality_label'].isin(['no_confiable', 'variable', 'degradada']).astype(int)
    
    qual_agg = qual_df.groupby('person_global_id').agg(
        mean_sil_quality=('sil_quality_mean', 'mean'),
        degraded_sil_ratio=('is_degraded', 'mean')
    ).reset_index()
    
    # Merge all
    ctx_df = illum_agg.merge(occ_agg, on='person_global_id', how='outer')
    ctx_df = ctx_df.merge(qual_agg, on='person_global_id', how='outer')
    
    return ctx_df.set_index('person_global_id')

def calculate_crosscam_similarity(px, py, cam_prototypes):
    # Encontrar cámaras donde aparecen px y py
    cams_x = {c for (p, c) in cam_prototypes.keys() if p == px}
    cams_y = {c for (p, c) in cam_prototypes.keys() if p == py}
    
    sims = []
    for cx in cams_x:
        for cy in cams_y:
            if cx != cy:
                vec_x = cam_prototypes[(px, cx)]
                vec_y = cam_prototypes[(py, cy)]
                sim = np.dot(vec_x, vec_y)
                sims.append(sim)
                
    if len(sims) == 0:
        return np.nan, np.nan, np.nan, 0
    return np.mean(sims), np.max(sims), 1.0 - np.max(sims), len(sims)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', type=str, default='dataset/output/metadata/gower_identity_pairs.csv')
    parser.add_argument('--osnet', type=str, default='dataset/output/embeddings/sequence_embeddings_osnet.csv')
    parser.add_argument('--clip', type=str, default='dataset/output/embeddings/sequence_embeddings_clip.csv')
    parser.add_argument('--illum', type=str, default='dataset/output/metadata/sequences_illumination.csv')
    parser.add_argument('--occ', type=str, default='dataset/output/metadata/gait_cycles_occlusion.csv')
    parser.add_argument('--qual', type=str, default='dataset/output/metadata/gait_cycles_quality.csv')
    parser.add_argument('--out-dir', type=str, default='dataset/output/metadata')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load pairs
    print(f"Loading Gower pairs from {args.pairs}...")
    pairs_df = pd.read_csv(args.pairs)
    pairs_df['person_global_id_x'] = pairs_df['person_global_id_x'].apply(canonical_person_id)
    pairs_df['person_global_id_y'] = pairs_df['person_global_id_y'].apply(canonical_person_id)
    
    # 2. Compute prototypes
    print(f"Loading OSNet embeddings from {args.osnet}...")
    osnet_df = pd.read_csv(args.osnet, usecols=['person_global_id', 'cam_id', 'embedding_json'])
    osnet_prototypes, osnet_cam_prototypes = compute_camera_balanced_identity_prototypes(osnet_df)
    
    print(f"Loading CLIP embeddings from {args.clip}...")
    clip_df = pd.read_csv(args.clip, usecols=['person_global_id', 'cam_id', 'embedding_json'])
    clip_prototypes, _ = compute_camera_balanced_identity_prototypes(clip_df)
    
    # 3. Compute Context
    ctx_df = compute_context_by_identity(args.illum, args.occ, args.qual)
    
    # 4. Integrate into pairs_df
    print("Calculating similarities and integrating context...")
    osnet_sims = []
    clip_sims = []
    
    cross_means, cross_maxs, cross_min_dists, cross_counts = [], [], [], []
    
    for idx, row in pairs_df.iterrows():
        px = row['person_global_id_x']
        py = row['person_global_id_y']
        
        # OSNet Sim
        if px in osnet_prototypes and py in osnet_prototypes:
            sim_o = np.dot(osnet_prototypes[px], osnet_prototypes[py])
            osnet_sims.append(sim_o)
            
            c_mean, c_max, c_min_dist, c_count = calculate_crosscam_similarity(px, py, osnet_cam_prototypes)
            cross_means.append(c_mean)
            cross_maxs.append(c_max)
            cross_min_dists.append(c_min_dist)
            cross_counts.append(c_count)
        else:
            osnet_sims.append(np.nan)
            cross_means.append(np.nan)
            cross_maxs.append(np.nan)
            cross_min_dists.append(np.nan)
            cross_counts.append(0)
            
        # CLIP Sim
        if px in clip_prototypes and py in clip_prototypes:
            sim_c = np.dot(clip_prototypes[px], clip_prototypes[py])
            clip_sims.append(sim_c)
        else:
            clip_sims.append(np.nan)

    pairs_df['osnet_similarity'] = osnet_sims
    pairs_df['clip_similarity'] = clip_sims
    pairs_df['mean_osnet_crosscam_similarity'] = cross_means
    pairs_df['max_osnet_crosscam_similarity'] = cross_maxs
    pairs_df['min_osnet_crosscam_distance'] = cross_min_dists
    pairs_df['n_crosscam_comparisons'] = cross_counts
    
    # Join Context for X
    ctx_x = ctx_df.add_prefix('x_')
    pairs_df = pairs_df.join(ctx_x, on='person_global_id_x')
    
    # Join Context for Y
    ctx_y = ctx_df.add_prefix('y_')
    pairs_df = pairs_df.join(ctx_y, on='person_global_id_y')
    
    # Add combined context diffs/means/maxs
    pairs_df['pair_low_illum_ratio'] = pairs_df[['x_low_illum_ratio', 'y_low_illum_ratio']].mean(axis=1)
    pairs_df['pair_heavy_occlusion_ratio'] = pairs_df[['x_heavy_occlusion_ratio', 'y_heavy_occlusion_ratio']].mean(axis=1)
    pairs_df['pair_degraded_sil_ratio'] = pairs_df[['x_degraded_sil_ratio', 'y_degraded_sil_ratio']].mean(axis=1)
    
    pairs_df['pair_max_low_illum_ratio'] = pairs_df[['x_low_illum_ratio', 'y_low_illum_ratio']].max(axis=1)
    pairs_df['pair_max_heavy_occlusion_ratio'] = pairs_df[['x_heavy_occlusion_ratio', 'y_heavy_occlusion_ratio']].max(axis=1)
    pairs_df['pair_max_degraded_sil_ratio'] = pairs_df[['x_degraded_sil_ratio', 'y_degraded_sil_ratio']].max(axis=1)

    # 5. Determine Thresholds and Quadrants
    print("Classifying into ambiguity quadrants...")
    
    reliable_gower_mask = (pairs_df['gower_similarity'].notna() & (pairs_df['n_comparable_attributes'] >= 4))
    valid_gower = pairs_df.loc[reliable_gower_mask, 'gower_similarity']
    p90_gower = np.percentile(valid_gower, 90) if len(valid_gower) > 0 else 0.80
    
    valid_osnet = pairs_df['osnet_similarity'].dropna()
    p90_osnet = np.percentile(valid_osnet, 90) if len(valid_osnet) > 0 else 0.80
    
    print(f"  P90 Gower (reliable pairs): {p90_gower:.4f}")
    print(f"  P90 OSNet (all pairs): {p90_osnet:.4f}")
    
    def classify_quadrant(row):
        g_sim = row['gower_similarity']
        o_sim = row['osnet_similarity']
        n_valid = row['n_comparable_attributes']
        
        if pd.isna(g_sim) or pd.isna(o_sim) or n_valid < 4:
            return "incompleto"
            
        high_sem = g_sim >= max(p90_gower, 0.80)
        high_vis = o_sim >= p90_osnet
        
        if high_sem and high_vis:
            return "ambiguedad_confirmada"
        elif not high_sem and high_vis:
            return "ambiguedad_visual_no_explicada"
        elif high_sem and not high_vis:
            return "semejanza_semantica_controlable"
        else:
            return "baja_ambiguedad"

    pairs_df['ambiguity_quadrant'] = pairs_df.apply(classify_quadrant, axis=1)
    
    # 6. Correlation
    valid_corr = pairs_df[
        pairs_df['gower_similarity'].notna()
        & pairs_df['osnet_similarity'].notna()
        & (pairs_df['n_comparable_attributes'] >= 4)
    ]
    if len(valid_corr) > 0:
        rho, p_value = spearmanr(valid_corr['gower_similarity'], valid_corr['osnet_similarity'])
    else:
        rho, p_value = np.nan, np.nan
        
    print(f"Spearman correlation between Gower and OSNet (reliable pairs): {rho:.4f} (p-value: {p_value:.4e})")

    # Guardar matriz enriquecida
    out_file = os.path.join(args.out_dir, 'gower_osnet_cross_analysis.csv')
    pairs_df.to_csv(out_file, index=False)
    print(f"Cross analysis saved to {out_file} ({len(pairs_df)} pairs)")
    
    # 7. Subsets
    confirmed = pairs_df[pairs_df['ambiguity_quadrant'] == 'ambiguedad_confirmada'].copy()
    confirmed = confirmed.sort_values(['gower_similarity', 'osnet_similarity'], ascending=False)
    conf_out = os.path.join(args.out_dir, 'confirmed_ambiguous_identity_pairs.csv')
    confirmed.to_csv(conf_out, index=False)
    
    unexplained = pairs_df[pairs_df['ambiguity_quadrant'] == 'ambiguedad_visual_no_explicada'].copy()
    unexplained = unexplained.sort_values(['osnet_similarity'], ascending=False)
    unexp_out = os.path.join(args.out_dir, 'unexplained_visual_ambiguity_pairs.csv')
    unexplained.to_csv(unexp_out, index=False)
    
    # Resumen
    counts = pairs_df['ambiguity_quadrant'].value_counts()
    reliable_pairs = int((pairs_df['ambiguity_quadrant'] != 'incompleto').sum())
    confirmed_count = int(counts.get('ambiguedad_confirmada', 0))

    summary = {
        'total_pairs': len(pairs_df),
        'reliable_pairs': reliable_pairs,
        'incomplete_pairs': int(counts.get('incompleto', 0)),
        'confirmed_ambiguity_pairs': confirmed_count,
        'unexplained_visual_ambiguity_pairs': int(counts.get('ambiguedad_visual_no_explicada', 0)),
        'controllable_semantic_similarity_pairs': int(counts.get('semejanza_semantica_controlable', 0)),
        'low_ambiguity_pairs': int(counts.get('baja_ambiguedad', 0)),
        'confirmed_ambiguity_ratio': confirmed_count / reliable_pairs if reliable_pairs > 0 else np.nan,
        'p90_gower': p90_gower,
        'p90_osnet': p90_osnet,
        'spearman_gower_osnet_rho': rho,
        'spearman_gower_osnet_p_value': p_value
    }
    
    summary_df = pd.DataFrame([summary])
    sum_file = os.path.join(args.out_dir, 'gower_osnet_ambiguity_summary.csv')
    summary_df.to_csv(sum_file, index=False)
    print(f"Summary saved to {sum_file}")

if __name__ == '__main__':
    main()
