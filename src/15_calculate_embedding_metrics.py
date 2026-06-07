#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
15_calculate_embedding_metrics.py

Calcula métricas de ambigüedad visual interclase a partir de embeddings RGB
(OSNet o CLIP) a nivel de secuencia y a nivel de identidad (prototipos).
Compara internamente cada dataset listado en el archivo de entrada.

Ejemplos de ejecución:
----------------------
# Métricas Globales OSNet (Carga conjunta para balancear el P90 de Hard Negatives)
python src/15_calculate_embedding_metrics.py ^
  --embeddings dataset/output/embeddings/sequence_embeddings_osnet.csv ^
               D:/open_datasets/Mars/embeddings/sequence_embeddings_osnet.csv ^
               D:/open_datasets/i-LIDS-VID/embeddings/sequence_embeddings_osnet.csv ^
  --output-report dataset/output/embeddings/dataset_embedding_ambiguity_summary_osnet.csv

# Métricas Globales CLIP
python src/15_calculate_embedding_metrics.py ^
  --embeddings dataset/output/embeddings/sequence_embeddings_clip.csv ^
               D:/open_datasets/Mars/embeddings/sequence_embeddings_clip.csv ^
               D:/open_datasets/i-LIDS-VID/embeddings/sequence_embeddings_clip.csv ^
  --output-report dataset/output/embeddings/dataset_embedding_ambiguity_summary_clip.csv
"""

import os
import argparse
import pandas as pd
import numpy as np
import json
from scipy.spatial.distance import cdist

def normalize_l2(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def calculate_dataset_metrics(df_dataset):
    """
    Calcula las métricas de embeddings para un dataset individual a nivel secuencia y cross-camera.
    """
    if len(df_dataset) < 2:
        return None
        
    embeddings = []
    identities = []
    seq_ids = []
    cams = []
    
    for _, row in df_dataset.iterrows():
        try:
            emb = np.array(json.loads(row['embedding_json']))
            embeddings.append(emb)
            identities.append(str(row['person_global_id']))
            seq_ids.append(str(row['sequence_id']))
            cams.append(str(row.get('cam_id', 'unknown')))
        except Exception:
            continue
            
    if len(embeddings) < 2:
        return None
        
    embeddings = np.array(embeddings)
    identities = np.array(identities)
    cams = np.array(cams)
    
    # Nivel 1: Métricas a nivel de secuencia
    dist_matrix = cdist(embeddings, embeddings, metric='cosine')
    sim_matrix = 1.0 - dist_matrix
    
    n = len(identities)
    id_matrix = (identities[:, None] == identities[None, :])
    cam_matrix = (cams[:, None] == cams[None, :])
    
    upper_triangle = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    intra_mask = id_matrix & upper_triangle
    inter_mask = (~id_matrix) & upper_triangle
    
    intra_crosscam_mask = id_matrix & (~cam_matrix) & upper_triangle
    inter_crosscam_mask = (~id_matrix) & (~cam_matrix) & upper_triangle
    
    intra_similarities = sim_matrix[intra_mask]
    inter_similarities = sim_matrix[inter_mask]
    intra_distances = dist_matrix[intra_mask]
    inter_distances = dist_matrix[inter_mask]
    
    intra_crosscam_dist = dist_matrix[intra_crosscam_mask]
    inter_crosscam_dist = dist_matrix[inter_crosscam_mask]
    
    mean_intra_dist = np.mean(intra_distances) if len(intra_distances) > 0 else np.nan
    mean_inter_dist = np.mean(inter_distances) if len(inter_distances) > 0 else np.nan
    
    mean_intra_crosscam_dist = np.mean(intra_crosscam_dist) if len(intra_crosscam_dist) > 0 else np.nan
    mean_inter_crosscam_dist = np.mean(inter_crosscam_dist) if len(inter_crosscam_dist) > 0 else np.nan
    
    inter_intra_distance_ratio = np.nan
    if np.isfinite(mean_intra_dist) and mean_intra_dist > 0:
        inter_intra_distance_ratio = mean_inter_dist / mean_intra_dist
        
    crosscam_separability_ratio = np.nan
    if np.isfinite(mean_intra_crosscam_dist) and mean_intra_crosscam_dist > 0:
        crosscam_separability_ratio = mean_inter_crosscam_dist / mean_intra_crosscam_dist
    
    # Top-k Confusability
    sim_matrix_no_self = sim_matrix.copy()
    np.fill_diagonal(sim_matrix_no_self, -np.inf)
    top_indices = np.argsort(sim_matrix_no_self, axis=1)[:, ::-1]
    
    top_k_confusability = {}
    for k in [1, 5, 10]:
        if k >= n:
            top_k_confusability[f'top_{k}_confusability'] = 0.0
            continue
        top_k_idx = top_indices[:, :k]
        is_negative = (identities[top_k_idx] != identities[:, None])
        top_k_confusability[f'top_{k}_confusability'] = np.mean(np.mean(is_negative, axis=1))
        
    # Nearest negative
    dist_matrix_no_self = dist_matrix.copy()
    np.fill_diagonal(dist_matrix_no_self, np.inf)
    dist_matrix_negatives_only = np.where(~id_matrix, dist_matrix_no_self, np.inf)
    min_neg_distances = np.min(dist_matrix_negatives_only, axis=1)
    min_neg_distances = min_neg_distances[min_neg_distances != np.inf]
    
    if len(min_neg_distances) > 0:
        mean_nearest_negative_distance = np.mean(min_neg_distances)
        median_nearest_negative_distance = np.median(min_neg_distances)
        p10_nearest_negative_distance = np.percentile(min_neg_distances, 10)
        mean_nearest_negative_similarity = np.mean(1.0 - min_neg_distances)
        normalized_embedding_ambiguity = np.mean((2.0 - min_neg_distances) / 2.0)
    else:
        mean_nearest_negative_distance = np.nan
        median_nearest_negative_distance = np.nan
        p10_nearest_negative_distance = np.nan
        mean_nearest_negative_similarity = np.nan
        normalized_embedding_ambiguity = np.nan
        
    # Nivel 2: Métricas a nivel de Identidad (Prototipos)
    unique_ids = np.unique(identities)
    prototypes = []
    for pid in unique_ids:
        mask = (identities == pid)
        proto = np.mean(embeddings[mask], axis=0)
        prototypes.append(normalize_l2(proto))
        
    prototypes = np.array(prototypes)
    
    if len(prototypes) > 1:
        proto_dist_matrix = cdist(prototypes, prototypes, metric='cosine')
        proto_upper = np.triu(np.ones((len(unique_ids), len(unique_ids)), dtype=bool), k=1)
        inter_proto_dist = proto_dist_matrix[proto_upper]
        
        mean_inter_id_proto_dist = np.mean(inter_proto_dist)
        mean_inter_id_proto_sim = np.mean(1.0 - inter_proto_dist)
        p10_inter_id_proto_dist = np.percentile(inter_proto_dist, 10)
    else:
        mean_inter_id_proto_dist = np.nan
        mean_inter_id_proto_sim = np.nan
        p10_inter_id_proto_dist = np.nan

    return {
        'n_sequences': n,
        'n_identities': len(unique_ids),
        'intra_id_similarity': np.mean(intra_similarities) if len(intra_similarities) > 0 else np.nan,
        'inter_id_similarity': np.mean(inter_similarities) if len(inter_similarities) > 0 else np.nan,
        'inter_intra_distance_ratio': inter_intra_distance_ratio,
        'intra_crosscam_distance': mean_intra_crosscam_dist,
        'inter_crosscam_distance': mean_inter_crosscam_dist,
        'crosscam_separability_ratio': crosscam_separability_ratio,
        'mean_nearest_negative_distance': mean_nearest_negative_distance,
        'median_nearest_negative_distance': median_nearest_negative_distance,
        'p10_nearest_negative_distance': p10_nearest_negative_distance,
        'mean_nearest_negative_similarity': mean_nearest_negative_similarity,
        'normalized_embedding_ambiguity': normalized_embedding_ambiguity,
        'mean_inter_id_proto_dist': mean_inter_id_proto_dist,
        'mean_inter_id_proto_sim': mean_inter_id_proto_sim,
        'p10_inter_id_proto_dist': p10_inter_id_proto_dist,
        **top_k_confusability
    }

def main(args):
    dfs = []
    for emb_path in args.embeddings:
        if not os.path.exists(emb_path):
            print(f"Error: no existe {emb_path}")
            continue
        df_tmp = pd.read_csv(emb_path)
        if 'dataset_name' in df_tmp.columns and 'embedding_json' in df_tmp.columns:
            dfs.append(df_tmp)
        else:
            print(f"El CSV {emb_path} no tiene el formato esperado.")
            
    if not dfs:
        print("No se encontraron CSVs válidos.")
        return
        
    df = pd.concat(dfs, ignore_index=True)
    
    models = df['embedding_model'].dropna().unique()
    if len(models) != 1:
        raise ValueError(f"El archivo de métricas debe contener embeddings de un único modelo. Se encontraron: {models}")
        
    datasets = df['dataset_name'].unique()
    print(f"Datasets encontrados: {datasets}")
    
    results = []
    
    # Calcular umbral común
    dataset_inter_sims = {}
    for dname in datasets:
        df_d = df[df['dataset_name'] == dname]
        embs = [np.array(json.loads(x)) for x in df_d['embedding_json']]
        ids = np.array(df_d['person_global_id'].astype(str).tolist())
        if len(embs) < 2: continue
        embs = np.array(embs)
        sim_m = 1.0 - cdist(embs, embs, metric='cosine')
        id_m = (ids[:, None] == ids[None, :])
        upper_triangle = np.triu(np.ones((len(ids), len(ids)), dtype=bool), k=1)
        inter_m = (~id_m) & upper_triangle
        
        sims = sim_m[inter_m]
        if len(sims) > 0:
            dataset_inter_sims[dname] = sims

    global_hard_sim_threshold = 0.8
    if dataset_inter_sims:
        min_pairs = min(args.max_calibration_pairs_per_dataset, min(len(sims) for sims in dataset_inter_sims.values()))
        balanced_sims = []
        np.random.seed(42)
        for sims in dataset_inter_sims.values():
            sampled = np.random.choice(sims, size=min_pairs, replace=False)
            balanced_sims.extend(sampled.tolist())
        
        global_hard_sim_threshold = np.percentile(balanced_sims, 90)
        print(f"Umbral común calibrado con muestra balanceada ({min_pairs} pares/dataset) para Hard Negatives (P90): {global_hard_sim_threshold:.4f}")
    
    all_hard_pairs = []
    
    for dname in datasets:
        print(f"Procesando dataset: {dname}")
        df_d = df[df['dataset_name'] == dname]
        model_name = df_d.iloc[0]['embedding_model'] if 'embedding_model' in df_d.columns else 'unknown'
        
        metrics = calculate_dataset_metrics(df_d)
        if metrics is not None:
            embs = np.array([np.array(json.loads(x)) for x in df_d['embedding_json']])
            ids = np.array(df_d['person_global_id'].astype(str).tolist())
            cams = np.array(df_d['cam_id'].astype(str).tolist()) if 'cam_id' in df_d.columns else np.array(['unknown']*len(df_d))
            seq_ids = np.array(df_d['sequence_id'].astype(str).tolist())
            
            sim_m = 1.0 - cdist(embs, embs, metric='cosine')
            dist_m = 1.0 - sim_m
            id_m = (ids[:, None] == ids[None, :])
            upper_triangle = np.triu(np.ones((len(ids), len(ids)), dtype=bool), k=1)
            inter_m = (~id_m)
            
            # Exportar top-k hard pairs por secuencia
            # Para cada secuencia i, buscar los j (j != i) que son de otra identidad, y que superan el umbral.
            # Tomar los top max_hard_pairs.
            np.fill_diagonal(sim_m, -np.inf)
            sim_m_inter_only = np.where(inter_m, sim_m, -np.inf)
            
            hard_negative_count = 0
            total_inter_pairs = np.sum(inter_m & upper_triangle)
            
            for i in range(len(ids)):
                # Todos los negativos para i
                neg_sims = sim_m_inter_only[i]
                hard_indices = np.where(neg_sims > global_hard_sim_threshold)[0]
                if len(hard_indices) > 0:
                    # Contar para el hard_negative_rate global de forma upper triangle
                    hard_negative_count += np.sum((neg_sims > global_hard_sim_threshold) & (np.arange(len(ids)) > i))
                    
                    # Ordenar por similitud y tomar los peores args.max_hard_pairs_per_seq
                    sorted_hard = hard_indices[np.argsort(neg_sims[hard_indices])[::-1]]
                    top_hard = sorted_hard[:args.max_hard_pairs_per_seq]
                    
                    # Solo añadir si (i, j) i < j para no duplicar si consideramos todo simétrico
                    # O mejor, lo guardamos direccional si nos interesa la secuencia "query". 
                    # Lo guardamos direccional (query -> hard_negative).
                    for j in top_hard:
                        all_hard_pairs.append({
                            'dataset_name': dname,
                            'embedding_model': model_name,
                            'sequence_id_query': seq_ids[i],
                            'sequence_id_neg': seq_ids[j],
                            'person_global_id_query': ids[i],
                            'person_global_id_neg': ids[j],
                            'cam_id_query': cams[i],
                            'cam_id_neg': cams[j],
                            'cosine_similarity': sim_m[i, j],
                            'cosine_distance': dist_m[i, j],
                            'hard_negative_flag': 1
                        })
            
            hard_negative_rate = hard_negative_count / total_inter_pairs if total_inter_pairs > 0 else 0.0
            
            row = {
                'dataset_name': dname,
                'embedding_model': model_name
            }
            row.update(metrics)
            row['hard_negative_rate'] = hard_negative_rate
            row['hard_negative_sim_threshold_used'] = global_hard_sim_threshold
            results.append(row)
            
    if not results:
        print("No se generaron métricas.")
        return
        
    out_df = pd.DataFrame(results)
    output_parent = os.path.dirname(args.output_report)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    out_df.to_csv(args.output_report, index=False)
    print(f"Resumen guardado en {args.output_report}")
    
    cols_to_print = ['dataset_name', 'embedding_model', 'mean_inter_id_proto_dist', 'hard_negative_rate', 'inter_intra_distance_ratio', 'mean_nearest_negative_similarity']
    cols_exist = [c for c in cols_to_print if c in out_df.columns]
    print(out_df[cols_exist])
    
    if all_hard_pairs:
        hard_df = pd.DataFrame(all_hard_pairs)
        model_name = results[0]['embedding_model'].replace('/', '_')
        if 'osnet' in args.output_report.lower():
            model_suffix = 'osnet'
        elif 'clip' in args.output_report.lower():
            model_suffix = 'clip'
        else:
            model_suffix = model_name
            
        if output_parent:
            hard_pairs_path = os.path.join(output_parent, f"dataset_embedding_hard_pairs_{model_suffix}.csv")
        else:
            hard_pairs_path = f"dataset_embedding_hard_pairs_{model_suffix}.csv"
            
        hard_df.to_csv(hard_pairs_path, index=False)
        print(f"Guardados {len(hard_df)} pares difíciles direccionales en {hard_pairs_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calcula métricas de embeddings.")
    parser.add_argument('--embeddings', type=str, nargs='+', required=True, help="CSVs generados por 15_extract_rgb_embeddings.py")
    parser.add_argument('--output-report', type=str, required=True, help="Ruta al CSV de salida (resumen)")
    parser.add_argument('--max-calibration-pairs-per-dataset', type=int, default=100000, help="Máximo número de pares a usar por dataset para calibrar el umbral (default 100k)")
    parser.add_argument('--max-hard-pairs-per-seq', type=int, default=5, help="Máximo número de hard negatives a exportar por cada secuencia (query)")
    
    args = parser.parse_args()
    main(args)
