#!/usr/bin/env python3
"""
Genera los archivos metadata sequences.csv y sequence_frames.csv para el dataset iLIDS-VID
a partir de su estructura de directorios de secuencias.

Estructura esperada de iLIDS-VID:
  ilids_root/
    ├── sequences/ (o directamente cam1 y cam2)
    │   ├── cam1/
    │   │   ├── person_001/
    │   │   │   ├── 0001.png
    │   │   │   └── ...
    │   │   └── ...
    │   └── cam2/
    │       ├── person_001/
    │       │   ├── 0001.png
    │       │   └── ...
    │       └── ...

Formato de salida compatible con los scripts de etiquetado posterior.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def extract_person_id(dir_name: str) -> Optional[int]:
    """Extrae el ID numérico de la carpeta del personaje (ej. person_001 -> 1)."""
    nums = re.findall(r"\d+", dir_name)
    if nums:
        return int(nums[-1])
    return None


def extract_frame_number(filename: str) -> Optional[int]:
    """Extrae el número de frame del nombre del archivo (ej. 0001.png -> 1)."""
    nums = re.findall(r"\d+", Path(filename).stem)
    if nums:
        return int(nums[-1])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera metadatos compatibles de secuencias para el dataset iLIDS-VID."
    )
    parser.add_argument(
        "--ilids-root",
        required=True,
        help="Ruta raíz de iLIDS-VID (contiene 'sequences/cam1' o 'cam1' y 'cam2' directamente)."
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directorio donde se guardarán los metadatos (sequences.csv, sequence_frames.csv)."
    )
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="Guardar rutas RGB relativas a ilids-root en lugar de absolutas."
    )
    
    args = parser.parse_args()
    
    ilids_root = Path(args.ilids_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    
    if not ilids_root.is_dir():
        raise FileNotFoundError(f"No existe el directorio de iLIDS-VID: {ilids_root}")
        
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Detectar dónde están cam1 y cam2
    candidates = [
        ilids_root / "sequences",
        ilids_root / "ILIDS-VIDsequences",
        ilids_root
    ]
    
    seq_base = None
    for cand in candidates:
        if (cand / "cam1").is_dir() and (cand / "cam2").is_dir():
            seq_base = cand
            break
            
    if seq_base is None:
        raise FileNotFoundError(
            f"No se encontraron las carpetas 'cam1' y 'cam2' bajo las rutas sugeridas en: {ilids_root}"
        )
        
    print(f"[INFO] Carpeta de secuencias identificada en: {seq_base}")
    
    cams = ["cam1", "cam2"]
    
    cycle_rows: List[Dict[str, object]] = []
    frame_rows: List[Dict[str, object]] = []
    
    for cam_name in cams:
        cam_dir = seq_base / cam_name
        cam_num = 1 if cam_name == "cam1" else 2
        cam_label = f"c{cam_num:03d}"
        
        # Encontrar carpetas de personas (person_001, person_002, etc.)
        person_dirs = sorted([
            p for p in cam_dir.iterdir()
            if p.is_dir() and extract_person_id(p.name) is not None
        ])
        
        for person_dir in tqdm(person_dirs, desc=f"Procesando {cam_name}"):
            person_id = extract_person_id(person_dir.name)
            if person_id is None:
                continue
                
            # Encontrar y ordenar frames de este personaje en esta cámara
            frame_files = sorted([
                f for f in person_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS
            ])
            
            if not frame_files:
                continue
                
            # Parsear número de frame para cada uno o usar su índice
            frames_list: List[Tuple[int, Path]] = []
            for file_path in frame_files:
                frame_num = extract_frame_number(file_path.name)
                if frame_num is None:
                    # Fallback al índice si no hay número
                    frame_num = len(frames_list) + 1
                frames_list.append((frame_num, file_path))
                
            # Ordenar por número de frame
            frames_list.sort(key=lambda x: x[0])
            
            # Identificador único de secuencia
            # ej. ilids_ID0001_c001
            seq_label = f"ilids_ID{person_id:04d}_{cam_label}"
            
            frame_numbers = [item[0] for item in frames_list]
            frame_start = frame_numbers[0]
            frame_end = frame_numbers[-1]
            num_frames = len(frames_list)
            
            # Guardar metadatos de secuencia
            cycle_rows.append({
                "sequence_id": seq_label,
                "cycle_id": seq_label,
                "person_global_id": person_id,
                "scene_id": "ilids",
                "cam_id": cam_label,
                "tracklet_index": 1,  # Solo hay un tracklet por persona/cámara en iLIDS-VID
                "cycle_index": 1,
                "num_frames": num_frames,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "frame_numbers_json": json.dumps(frame_numbers),
                "n_samples_rgb": num_frames,
                "n_samples_sil": 0,  # iLIDS-VID usualmente no trae siluetas binarias por defecto
            })
            
            # Guardar metadatos de cada frame
            for pos, (frame_num, path) in enumerate(frames_list, start=1):
                if args.relative_paths:
                    rgb_path_str = str(path.relative_to(ilids_root))
                else:
                    rgb_path_str = str(path)
                    
                frame_rows.append({
                    "sequence_id": seq_label,
                    "cycle_id": seq_label,
                    "person_global_id": person_id,
                    "scene_id": "ilids",
                    "cam_id": cam_label,
                    "tracklet_index": 1,
                    "cycle_index": 1,
                    "position_in_sequence": pos,
                    "position_in_cycle": pos,
                    "frame_number": frame_num,
                    "rgb_source_path": rgb_path_str,
                    "silhouette_source_path": "",
                    "rgb_output_path": rgb_path_str,
                    "silhouette_output_path": "",
                    "bbox_x": None,
                    "bbox_y": None,
                    "bbox_w": None,
                    "bbox_h": None,
                    "bbox_area": None,
                    "foreground_sum": None,
                    "foreground_sum_smooth": None,
                })
                
    # Escribir CSVs
    df_seq = pd.DataFrame(cycle_rows)
    df_frames = pd.DataFrame(frame_rows)
    
    # Nombres principales de salida
    seq_path = out_dir / "sequences.csv"
    frames_path = out_dir / "sequence_frames.csv"
    
    df_seq.to_csv(seq_path, index=False)
    df_frames.to_csv(frames_path, index=False)
    
    # Compatibilidad con nombres de ciclo
    df_seq.to_csv(out_dir / "gait_cycles.csv", index=False)
    df_frames.to_csv(out_dir / "gait_cycle_frames.csv", index=False)
    
    print("\n[INFO] Completado:")
    print(f"  - Secuencias totales: {len(df_seq)}")
    print(f"  - Frames totales: {len(df_frames)}")
    print(f"  - Metadatos guardados en: {out_dir}")


if __name__ == "__main__":
    main()
