#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
14a_extract_rgb_embeddings.py

Extrae embeddings RGB (OSNet o CLIP) a nivel de secuencia (tracklet).
Selecciona hasta 12 frames (o los disponibles) por secuencia, calcula los embeddings
individuales y los promedia normalizados en L2 para obtener el embedding final de la secuencia.

Requisitos:
    - OSNet: pip install torchreid
    - CLIP: pip install git+https://github.com/openai/CLIP.git

Ejemplos de ejecución:
----------------------
# Mi Corpus (OSNet Re-ID)
python src/14a_extract_rgb_embeddings.py --dataset-name mi_corpus --base-dir D:\tesis_dev\result1 --frames-csv D:\tesis_dev\result1\dataset\output\metadata\sequence_frames.csv --sequences-csv D:\tesis_dev\result1\dataset\output\metadata\sequences.csv --model osnet --osnet-model-name osnet_ain_x1_0 --osnet-checkpoint "D:\tesis_dev\result1\src\models\osnet_ain_x1_0_msmt17.pth" --output-csv dataset/output/embeddings/sequence_embeddings_osnet.csv

# Mi Corpus (CLIP)
python src/14a_extract_rgb_embeddings.py --dataset-name mi_corpus --base-dir D:\tesis_dev\result1 --frames-csv D:\tesis_dev\result1\dataset\output\metadata\sequence_frames.csv --sequences-csv D:\tesis_dev\result1\dataset\output\metadata\sequences.csv --model clip --output-csv dataset/output/embeddings/sequence_embeddings_clip.csv

# MARS (OSNet Re-ID)
python src/14a_extract_rgb_embeddings.py --dataset-name mars --base-dir D:\open_datasets\Mars\raw --frames-csv D:\open_datasets\Mars\metadata\sequence_frames.csv --model osnet --osnet-model-name osnet_ain_x1_0 --osnet-checkpoint "D:\tesis_dev\result1\src\models\osnet_ain_x1_0_msmt17.pth" --output-csv D:\open_datasets\Mars\embeddings\sequence_embeddings_osnet.csv

# MARS (CLIP)
python src/14a_extract_rgb_embeddings.py --dataset-name mars --base-dir D:\open_datasets\Mars\raw --frames-csv D:\open_datasets\Mars\metadata\sequence_frames.csv --model clip --output-csv D:\open_datasets\Mars\embeddings\sequence_embeddings_clip.csv

# iLIDS-VID (OSNet Re-ID)
python src/14a_extract_rgb_embeddings.py --dataset-name ilidsvid --base-dir D:\open_datasets\i-LIDS-VID --frames-csv D:\open_datasets\i-LIDS-VID\metadata\sequence_frames.csv --model osnet --osnet-model-name osnet_ain_x1_0 --osnet-checkpoint "D:\tesis_dev\result1\src\models\osnet_ain_x1_0_msmt17.pth" --output-csv D:\open_datasets\i-LIDS-VID\embeddings\sequence_embeddings_osnet.csv

# iLIDS-VID (CLIP)
python src/14a_extract_rgb_embeddings.py --dataset-name ilidsvid --base-dir D:\open_datasets\i-LIDS-VID --frames-csv D:\open_datasets\i-LIDS-VID\metadata\sequence_frames.csv --model clip --output-csv D:\open_datasets\i-LIDS-VID\embeddings\sequence_embeddings_clip.csv
"""

import os
import argparse
import pandas as pd
import numpy as np
import json
import torch
import cv2
import re
from pathlib import Path
from tqdm import tqdm
from PIL import Image

def get_model(model_name, device, osnet_model_name='osnet_ain_x1_0', osnet_checkpoint=None):
    if model_name == 'osnet':
        try:
            import torchreid
        except ImportError:
            raise ImportError("Falta torchreid. Instala deep-person-reid según su documentación.")
        
        if not osnet_checkpoint:
            raise ValueError(
                "Para resultados definitivos debes indicar --osnet-checkpoint "
                "con un checkpoint Re-ID congelado."
            )
            
        print(f"Cargando modelo OSNet: {osnet_model_name}")
        print(f"Checkpoint Re-ID: {osnet_checkpoint}")
        
        model = torchreid.models.build_model(
            name=osnet_model_name,
            num_classes=1000,
            loss='softmax',
            pretrained=False
        )
        
        torchreid.utils.load_pretrained_weights(
            model,
            osnet_checkpoint
        )
        
        model.eval()
        model.to(device)
        return model, osnet_model_name, 512, None
        
    elif model_name == 'clip':
        try:
            import clip
        except ImportError:
            raise ImportError("Falta clip. Instala con: pip install git+https://github.com/openai/CLIP.git")
        
        print("Cargando modelo CLIP (ViT-B/32)...")
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()
        return model, "ViT-B/32", 512, preprocess
    else:
        raise ValueError(f"Modelo desconocido: {model_name}")

def extract_features(model, model_name, image_path, device, preprocess=None):
    if not os.path.exists(image_path):
        return None
        
    if model_name == 'osnet':
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 256))
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img = img.transpose(2, 0, 1)
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = model(img_tensor)
        return features.cpu().numpy().flatten()
        
    elif model_name == 'clip':
        try:
            pil_img = Image.open(image_path).convert("RGB")
        except Exception:
            return None
            
        img_tensor = preprocess(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.encode_image(img_tensor)
        return features.cpu().numpy().flatten()

def normalize_l2(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")
    
    model, exact_model_name, emb_dim, preprocess = get_model(
        args.model, device, args.osnet_model_name, args.osnet_checkpoint
    )
    
    print(f"Cargando metadata de secuencias y frames desde {args.frames_csv}...")
    frames_df = pd.read_csv(args.frames_csv)
    
    if 'sequence_id' not in frames_df.columns:
        raise ValueError("El CSV de frames debe contener la columna 'sequence_id'")
    
    path_col = None
    for col in ['rgb_source_path', 'rgb_output_path', 'rgb_path', 'frame_path']:
        if col in frames_df.columns:
            path_col = col
            break
    if not path_col:
        raise ValueError("No se encontró columna con la ruta de la imagen RGB.")
        
    base_dir = Path(args.base_dir).resolve() if args.base_dir else Path.cwd()
    
    sequences_df = None
    if args.sequences_csv and os.path.exists(args.sequences_csv):
        sequences_df = pd.read_csv(args.sequences_csv)
    
    grouped = frames_df.groupby("sequence_id")
    output_rows = []
    
    for seq_id, group in tqdm(grouped, desc="Extrayendo embeddings"):
        candidates = group.copy()
        
        # Ordenar cronológicamente
        if 'frame_number' in candidates.columns:
            candidates = candidates.sort_values('frame_number').reset_index(drop=True)
        
        # Modos de filtro
        if args.filter_mode == 'controlled_occlusion':
            if 'occlusion_pct' not in candidates.columns:
                raise ValueError("Falta occlusion_pct para usar controlled_occlusion.")
            candidates = candidates[candidates['occlusion_pct'].isna() | (candidates['occlusion_pct'] <= 80)]
            
            if 'frame_valid' in candidates.columns:
                valid = candidates[candidates['frame_valid'] == True]
                if len(valid) > 0:
                    candidates = valid
        
        n_available = len(candidates)
        if n_available == 0:
            continue
            
        if n_available <= args.max_frames:
            sampled = candidates
        else:
            idx = np.linspace(0, n_available - 1, args.max_frames, dtype=int)
            sampled = candidates.iloc[idx]
            
        successful_frame_numbers = []
        sequence_embeddings = []
        
        for _, row in sampled.iterrows():
            raw_path = row[path_col]
            if pd.isna(raw_path) or not str(raw_path).strip():
                continue
            
            p = Path(str(raw_path))
            img_path = p if p.is_absolute() else base_dir / p
            
            emb = extract_features(model, args.model, img_path, device, preprocess)
            if emb is not None:
                sequence_embeddings.append(normalize_l2(emb))
                if 'frame_number' in row and not pd.isna(row['frame_number']):
                    successful_frame_numbers.append(int(row['frame_number']))
                
        if not sequence_embeddings:
            continue
            
        mean_emb = np.mean(sequence_embeddings, axis=0)
        norm_emb = normalize_l2(mean_emb)
        
        person_id = None
        cam_id = None
        
        # 1. Prioridad: metadata por frame
        if 'person_global_id' in group.columns:
            value = group.iloc[0]['person_global_id']
            if not pd.isna(value):
                person_id = str(value)
        elif 'person_id' in group.columns:
            value = group.iloc[0]['person_id']
            if not pd.isna(value):
                person_id = str(value)
                
        if 'cam_id' in group.columns:
            value = group.iloc[0]['cam_id']
            if not pd.isna(value):
                cam_id = str(value)
        elif 'camera_id' in group.columns:
            value = group.iloc[0]['camera_id']
            if not pd.isna(value):
                cam_id = str(value)
                
        # 2. Segunda prioridad: metadata de secuencia
        if sequences_df is not None:
            seq_info = sequences_df[sequences_df['sequence_id'].astype(str) == str(seq_id)]
            if not seq_info.empty:
                source = seq_info.iloc[0]
                if person_id is None:
                    for col in ['person_global_id', 'person_id']:
                        if col in source.index and not pd.isna(source[col]):
                            person_id = str(source[col])
                            break
                if cam_id is None:
                    for col in ['cam_id', 'camera_id']:
                        if col in source.index and not pd.isna(source[col]):
                            cam_id = str(source[col])
                            break
                            
        # 3. Último recurso: regex
        if person_id is None:
            match = re.search(r'ID(\d+)', str(seq_id), flags=re.IGNORECASE)
            if match:
                person_id = str(int(match.group(1)))
                
        if cam_id is None:
            match = re.search(r'c(?:am)?(\d+)', str(seq_id), flags=re.IGNORECASE)
            if match:
                cam_id = str(int(match.group(1)))
                
        # 4. No aceptar IDs desconocidos
        if person_id is None:
            raise ValueError(f"No se pudo recuperar person_global_id para sequence_id={seq_id}")
            
        if cam_id is None:
            cam_id = "unknown"
        
        output_rows.append({
            'dataset_name': args.dataset_name,
            'sequence_id': seq_id,
            'person_global_id': person_id,
            'cam_id': cam_id,
            'n_frames_used': len(sequence_embeddings),
            'selected_frame_numbers_json': json.dumps(successful_frame_numbers),
            'embedding_model': exact_model_name,
            'embedding_dim': emb_dim,
            'embedding_json': json.dumps(norm_emb.tolist())
        })
        
    if not output_rows:
        print("No se extrajeron embeddings. Revisa las rutas de imágenes.")
        return
        
    # Validación obligatoria de IDs unknown
    if any(str(row['person_global_id']).lower() == 'unknown' for row in output_rows):
        raise ValueError("Existen secuencias sin person_global_id. Corrige el manifest antes de calcular métricas.")
        
    out_df = pd.DataFrame(output_rows)
    
    output_parent = os.path.dirname(args.output_csv)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    
    if os.path.exists(args.output_csv) and args.append:
        existing = pd.read_csv(args.output_csv)
        out_df = pd.concat([existing, out_df], ignore_index=True)
        out_df.drop_duplicates(subset=['dataset_name', 'sequence_id', 'embedding_model'], keep='last', inplace=True)

    out_df.to_csv(args.output_csv, index=False)
    print(f"Guardados {len(output_rows)} embeddings en {args.output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extrae embeddings RGB por secuencia.")
    parser.add_argument('--model', type=str, choices=['osnet', 'clip'], required=True)
    parser.add_argument('--osnet-model-name', type=str, default='osnet_ain_x1_0', help='Arquitectura OSNet de torchreid')
    parser.add_argument('--osnet-checkpoint', type=str, default=None, help='Checkpoint Re-ID congelado para OSNet')
    parser.add_argument('--filter-mode', choices=['raw_uniform', 'controlled_occlusion'], default='raw_uniform')
    parser.add_argument('--dataset-name', type=str, required=True, help="Nombre del dataset")
    parser.add_argument('--frames-csv', type=str, required=True, help="CSV con los frames")
    parser.add_argument('--sequences-csv', type=str, help="CSV con metadata de secuencias")
    parser.add_argument('--base-dir', type=str, help="Directorio base")
    parser.add_argument('--output-csv', type=str, required=True, help="CSV de salida")
    parser.add_argument('--max-frames', type=int, default=12, help="Máximo frames por secuencia")
    parser.add_argument('--append', action='store_true', help="Actualiza archivo existente")
    
    args = parser.parse_args()
    main(args)
