#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

TEXTS = {
    'en': {
        'illum_dist': 'Illumination Level Distribution',
        'lum_dist': 'Luminance Distribution',
        'pct': 'Percentage (%)',
        'occ_dist': 'Occlusion Level Distribution',
        'occ_ylabel': 'Percentage (%)',
        'sq_dist': 'Silhouette Quality Distribution (SQQ)',
        'sq_ylabel': 'Foreground-Sum Quality Score',
        'proposed': 'Proposed',
        'levels': {'Low': 'Low', 'Medium-Low': 'Medium-Low', 'Medium-High': 'Medium-High', 'High': 'High'},
        'occ_levels': {'baja': 'Low', 'parcial': 'Partial', 'alta': 'Heavy'}
    },
    'es': {
        'illum_dist': 'Distribución de Nivel de Iluminación',
        'lum_dist': 'Distribución de Luminancia',
        'pct': 'Porcentaje (%)',
        'occ_dist': 'Distribución de Nivel de Oclusión',
        'occ_ylabel': 'Porcentaje (%)',
        'sq_dist': 'Distribución de Calidad de Silueta (SQQ)',
        'sq_ylabel': 'Puntaje de Calidad (Foreground-Sum)',
        'proposed': 'Propuesto',
        'levels': {'Low': 'Bajo', 'Medium-Low': 'Medio-Bajo', 'Medium-High': 'Medio-Alto', 'High': 'Alto'},
        'occ_levels': {'baja': 'Baja', 'parcial': 'Parcial', 'alta': 'Alta'}
    }
}

def load_data(path, dataset_name):
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['dataset'] = dataset_name
        return df
    return pd.DataFrame()

def main():
    out_dirs = {
        'en': "D:/tesis_dev/result1/src/paper/english/figures",
        'es': "D:/tesis_dev/result1/src/paper/spanish/figures"
    }

    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)
    
    # 1. Illumination
    illum_paths = {
        "Proposed": "D:/tesis_dev/result1/dataset/output/metadata/02_illumination/sequences_illumination.csv",
        "MARS": "D:/open_datasets/MARS/metadata/sequences_illumination.csv",
        "iLIDS-VID": "D:/open_datasets/i-LIDS-VID/metadata/sequences_illumination.csv"
    }
    df_illum = pd.concat([load_data(p, name) for name, p in illum_paths.items() if os.path.exists(p)], ignore_index=True)
    
    # 2. Occlusion
    occ_paths = {
        "Proposed": "D:/tesis_dev/result1/dataset/output/metadata/03_occlusion/gait_cycles_occlusion.csv",
        "MARS": "D:/open_datasets/MARS/metadata/sequences_occlusion.csv",
        "iLIDS-VID": "D:/open_datasets/i-LIDS-VID/metadata/sequences_occlusion.csv"
    }
    df_occ = pd.concat([load_data(p, name) for name, p in occ_paths.items() if os.path.exists(p)], ignore_index=True)
    
    # 3. Silhouette Quality
    sq_paths = {
        "Proposed": "D:/tesis_dev/result1/dataset/output/metadata/05_silhouette_quality/gait_cycles_quality.csv",
        "CASIA-B": "D:/open_datasets/CASIA-B/metadata/gait_cycles_quality.csv",
        "Gait3D": "D:/open_datasets/Gait3D/metadata/gait_cycles_quality.csv"
    }
    df_sq = pd.concat([load_data(p, name) for name, p in sq_paths.items() if os.path.exists(p)], ignore_index=True)
    
    for lang in ['en', 'es']:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Translate "Proposed" dataset name
        if not df_illum.empty:
            df_illum_plot = df_illum.copy()
            df_illum_plot['dataset'] = df_illum_plot['dataset'].replace({'Proposed': TEXTS[lang]['proposed']})
        if not df_occ.empty:
            df_occ_plot = df_occ.copy()
            df_occ_plot['dataset'] = df_occ_plot['dataset'].replace({'Proposed': TEXTS[lang]['proposed']})
        if not df_sq.empty:
            df_sq_plot = df_sq.copy()
            df_sq_plot['dataset'] = df_sq_plot['dataset'].replace({'Proposed': TEXTS[lang]['proposed']})
        
        # Panel 1: Illumination
        if not df_illum.empty and 'illum_level' in df_illum.columns:
            illum_counts = df_illum_plot.groupby(['dataset', 'illum_level']).size().unstack(fill_value=0)
            illum_pct = illum_counts.div(illum_counts.sum(axis=1), axis=0) * 100
            
            levels_order = ['Low', 'Medium-Low', 'Medium-High', 'High']
            levels_present = [l for l in levels_order if l in illum_pct.columns]
            if levels_present:
                illum_pct = illum_pct[levels_present]
            
            # Translate columns
            illum_pct.rename(columns=TEXTS[lang]['levels'], inplace=True)
                
            illum_pct.plot(kind='bar', stacked=True, ax=axes[0], colormap='viridis')
            axes[0].set_title(TEXTS[lang]['illum_dist'])
            axes[0].set_ylabel(TEXTS[lang]['pct'])
            axes[0].set_xlabel("")
            axes[0].tick_params(axis='x', rotation=0)
            
        # Panel 2: Occlusion
        if not df_occ.empty and 'occlusion_label' in df_occ.columns:
            occ_counts = df_occ_plot.groupby(['dataset', 'occlusion_label']).size().unstack(fill_value=0)
            occ_pct = occ_counts.div(occ_counts.sum(axis=1), axis=0) * 100
            
            occ_levels_order = ['baja', 'parcial', 'alta']
            occ_levels_present = [l for l in occ_levels_order if l in occ_pct.columns]
            if occ_levels_present:
                occ_pct = occ_pct[occ_levels_present]
            
            # Translate columns
            occ_pct.rename(columns=TEXTS[lang]['occ_levels'], inplace=True)
            
            occ_pct.plot(kind='bar', stacked=True, ax=axes[1], colormap='autumn_r')
            axes[1].set_title(TEXTS[lang]['occ_dist'])
            axes[1].set_ylabel(TEXTS[lang]['occ_ylabel'])
            axes[1].set_xlabel("")
            axes[1].tick_params(axis='x', rotation=0)
            
        # Panel 3: Silhouette Quality
        if not df_sq.empty and 'sil_quality_mean' in df_sq.columns:
            sns.boxplot(data=df_sq_plot, x='dataset', y='sil_quality_mean', ax=axes[2], palette="Set3", hue='dataset', legend=False)
            axes[2].set_title(TEXTS[lang]['sq_dist'])
            axes[2].set_ylabel(TEXTS[lang]['sq_ylabel'])
            axes[2].set_xlabel("")
            
        plt.tight_layout()
        out_path = os.path.join(out_dirs[lang], "cross_dataset_context.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[{lang}] Saved to {out_path}")

if __name__ == "__main__":
    main()
