#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import os
import re

def patch_embeddings(embeddings_path, sequences_path=None):
    if not os.path.exists(embeddings_path):
        print(f"Skipping {embeddings_path}, file not found.")
        return

    seq_metadata = {}
    if sequences_path and os.path.exists(sequences_path):
        print(f"Loading metadata from {sequences_path}...")
        with open(sequences_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                seq_id = row.get('sequence_id')
                if seq_id:
                    seq_metadata[seq_id] = {
                        'person_global_id': row.get('person_global_id', 'unknown'),
                        'cam_id': row.get('cam_id', 'unknown')
                    }
    
    temp_path = embeddings_path + '.tmp'
    patched_count = 0
    
    print(f"Patching {embeddings_path}...")
    with open(embeddings_path, 'r', encoding='utf-8') as fin, \
         open(temp_path, 'w', encoding='utf-8', newline='') as fout:
         
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        
        for row in reader:
            dataset = row.get('dataset_name', '')
            seq_id = row.get('sequence_id', '')
            
            # Patch based on dataset
            if dataset == 'mi_corpus' or dataset == '':
                if seq_id in seq_metadata:
                    row['person_global_id'] = seq_metadata[seq_id]['person_global_id']
                    row['cam_id'] = seq_metadata[seq_id]['cam_id']
                    patched_count += 1
            elif dataset in ['mars', 'ilidsvid', 'ilids']:
                # mars: mars_bbox_test_ID0002_c001_seq0001
                # ilidsvid: ilids_ID0001_c001
                m_id = re.search(r'ID(\d+)', seq_id)
                m_cam = re.search(r'c(\d+)', seq_id)
                
                if m_id:
                    row['person_global_id'] = str(int(m_id.group(1)))
                if m_cam:
                    row['cam_id'] = str(int(m_cam.group(1)))
                patched_count += 1
                
            writer.writerow(row)
            
    os.replace(temp_path, embeddings_path)
    print(f"Patched {patched_count} rows in {embeddings_path}\n")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        # Pass a specific embeddings csv
        emb_csv = sys.argv[1]
        patch_embeddings(emb_csv, None)
    else:
        # Default behavior for mi_corpus
        base_dir = r"d:\tesis_dev\result1\dataset\output"
        seq_csv = os.path.join(base_dir, "metadata", "sequences.csv")
        
        osnet_csv = os.path.join(base_dir, "embeddings", "sequence_embeddings_osnet.csv")
        clip_csv = os.path.join(base_dir, "embeddings", "sequence_embeddings_clip.csv")
        
        patch_embeddings(osnet_csv, seq_csv)
        patch_embeddings(clip_csv, seq_csv)
