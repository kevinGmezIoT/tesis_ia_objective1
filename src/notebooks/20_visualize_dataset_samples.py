#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    metadata_path = "D:/tesis_dev/result1/dataset/output/metadata/gait_cycle_frames.csv"
    base_dir = "D:/tesis_dev/result1/dataset/output"
    out_dir_en = "D:/tesis_dev/result1/src/paper/english/figures"
    out_dir_es = "D:/tesis_dev/result1/src/paper/spanish/figures"
    
    os.makedirs(out_dir_en, exist_ok=True)
    os.makedirs(out_dir_es, exist_ok=True)
    
    print(f"Loading {metadata_path}...")
    df = pd.read_csv(metadata_path)
    
    # We want to pick a few unique identities. Let's say 6.
    # To get good looking ones, let's just sample randomly, but maybe exclude any without silhouettes.
    df = df[df['silhouette_source_path'].notna() & (df['silhouette_source_path'] != "")]
    
    # Let's get the middle frame of each sequence
    unique_ids = df['person_global_id'].unique()
    np.random.seed(42)
    selected_ids = np.random.choice(unique_ids, size=6, replace=False)
    
    rgb_images = []
    sil_images = []
    titles = []
    
    for pid in selected_ids:
        # Get one sequence for this person
        person_df = df[df['person_global_id'] == pid]
        seq_id = person_df['sequence_id'].iloc[0]
        seq_df = person_df[person_df['sequence_id'] == seq_id]
        
        # Take the middle frame
        mid_idx = len(seq_df) // 2
        row = seq_df.iloc[mid_idx]
        
        rgb_path = os.path.join(base_dir, row['rgb_source_path'])
        sil_path = os.path.join(base_dir, row['silhouette_source_path'])
        
        if os.path.exists(rgb_path) and os.path.exists(sil_path):
            rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
            sil = cv2.imread(sil_path, cv2.IMREAD_GRAYSCALE)
            
            rgb_images.append(rgb)
            sil_images.append(sil)
            titles.append(f"ID {int(pid):04d}")
        else:
            print(f"File missing for {pid}")
            
    fig, axes = plt.subplots(2, len(rgb_images), figsize=(2 * len(rgb_images), 5))
    
    for i in range(len(rgb_images)):
        axes[0, i].imshow(rgb_images[i])
        axes[0, i].set_title(titles[i], fontsize=12, pad=10)
        axes[0, i].axis('off')
        
        axes[1, i].imshow(sil_images[i], cmap='gray')
        axes[1, i].axis('off')
        
    plt.tight_layout()
    
    out_path_en = os.path.join(out_dir_en, "dataset_samples_rgb_sil.png")
    out_path_es = os.path.join(out_dir_es, "dataset_samples_rgb_sil.png")
    
    plt.savefig(out_path_en, dpi=300, bbox_inches='tight')
    plt.savefig(out_path_es, dpi=300, bbox_inches='tight')
    
    print(f"Saved to {out_path_en} and {out_path_es}")

if __name__ == "__main__":
    main()
