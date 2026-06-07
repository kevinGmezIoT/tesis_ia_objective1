#!/usr/bin/env python3
"""
Genera los archivos metadata sequences.csv y sequence_frames.csv para el dataset Gait3D
a partir de su estructura de directorios de siluetas 2D.

Estructura esperada de Gait3D:
  gait3d_root/
    ├── 2D_Silhouettes/ (o directamente las carpetas de IDs)
    │   ├── 0000/
    │   │   ├── camid0_videoid2/
    │   │   │   ├── seq0/
    │   │   │   │   ├── human_crop_f11627.png
    │   │   │   │   └── ...

Formato de salida compatible con los scripts de evaluación (SQQ, etc).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def extract_frame_number(filename: str) -> Optional[int]:
    """Extrae el número de frame del nombre del archivo (ej. human_crop_f11627.png -> 11627)."""
    nums = re.findall(r"\d+", Path(filename).stem)
    if nums:
        return int(nums[-1])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera metadatos compatibles de secuencias para el dataset Gait3D."
    )
    parser.add_argument(
        "--gait3d-root",
        required=True,
        help="Ruta raíz de Gait3D (ej. D:\\open_datasets\\Gait3D)."
    )
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="Guardar rutas relativas a gait3d-root en lugar de absolutas."
    )
    
    args = parser.parse_args()
    
    gait3d_root = Path(args.gait3d_root).expanduser().resolve()
    
    if not gait3d_root.is_dir():
        raise FileNotFoundError(f"No existe el directorio de Gait3D: {gait3d_root}")
        
    out_dir = gait3d_root / "metadata"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Encontrar la carpeta base que contiene los IDs (ej. 0000, 0001)
    seq_base = gait3d_root
    candidates = [
        gait3d_root / "2D_Silhouettes" / "2D_Silhouettes",
        gait3d_root / "2D_Silhouettes",
        gait3d_root
    ]
    for cand in candidates:
        if cand.is_dir() and any(p.is_dir() and p.name.isdigit() for p in cand.iterdir()):
            seq_base = cand
            break
            
    print(f"[INFO] Buscando siluetas de Gait3D en: {seq_base}")
    
    person_dirs = sorted([p for p in seq_base.iterdir() if p.is_dir() and p.name.isdigit()])
    
    cycle_rows: List[Dict[str, object]] = []
    frame_rows: List[Dict[str, object]] = []
    
    for person_dir in tqdm(person_dirs, desc="Procesando identidades Gait3D"):
        person_id_str = person_dir.name
        person_id = int(person_id_str)
        
        # Combinación de cámara y video (ej. camid0_videoid2)
        for cam_video_dir in person_dir.iterdir():
            if not cam_video_dir.is_dir():
                continue
            
            cam_vid_str = cam_video_dir.name
            # Separar camid y videoid si es posible
            parts = cam_vid_str.split("_")
            cam_id = parts[0] if len(parts) > 0 else "unknown_cam"
            scene_id = parts[1] if len(parts) > 1 else "unknown_video"
            
            # Secuencias (ej. seq0, seq1)
            for seq_dir in cam_video_dir.iterdir():
                if not seq_dir.is_dir():
                    continue
                seq_name = seq_dir.name
                
                # Frames
                frame_files = sorted([
                    f for f in seq_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTS
                ])
                
                if not frame_files:
                    continue
                    
                frames_list: List[Tuple[int, Path]] = []
                for file_path in frame_files:
                    frame_num = extract_frame_number(file_path.name)
                    if frame_num is None:
                        frame_num = len(frames_list) + 1
                    frames_list.append((frame_num, file_path))
                    
                frames_list.sort(key=lambda x: x[0])
                
                # Identificador único de secuencia
                seq_label = f"gait3d_{person_id_str}_{cam_vid_str}_{seq_name}"
                
                frame_numbers = [item[0] for item in frames_list]
                frame_start = frame_numbers[0]
                frame_end = frame_numbers[-1]
                num_frames = len(frames_list)
                
                # Metadatos de secuencia
                cycle_rows.append({
                    "sequence_id": seq_label,
                    "cycle_id": seq_label,
                    "person_global_id": person_id,
                    "scene_id": scene_id,
                    "cam_id": cam_id,
                    "tracklet_index": int(seq_name.replace("seq", "")) if "seq" in seq_name else 1,
                    "cycle_index": 1,
                    "num_frames": num_frames,
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                    "frame_numbers_json": json.dumps(frame_numbers),
                    "n_samples_rgb": 0,  # Asumimos siluetas 2D
                    "n_samples_sil": num_frames,
                })
                
                # Metadatos de frame
                for pos, (frame_num, path) in enumerate(frames_list, start=1):
                    if args.relative_paths:
                        path_str = str(path.relative_to(gait3d_root))
                    else:
                        path_str = str(path)
                        
                    frame_rows.append({
                        "sequence_id": seq_label,
                        "cycle_id": seq_label,
                        "person_global_id": person_id,
                        "scene_id": scene_id,
                        "cam_id": cam_id,
                        "tracklet_index": int(seq_name.replace("seq", "")) if "seq" in seq_name else 1,
                        "cycle_index": 1,
                        "position_in_sequence": pos,
                        "position_in_cycle": pos,
                        "frame_number": frame_num,
                        "rgb_source_path": "",
                        "silhouette_source_path": path_str,
                        "rgb_output_path": "",
                        "silhouette_output_path": path_str,
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
    
    df_seq.to_csv(out_dir / "sequences.csv", index=False)
    df_frames.to_csv(out_dir / "sequence_frames.csv", index=False)
    
    df_seq.to_csv(out_dir / "gait_cycles.csv", index=False)
    df_frames.to_csv(out_dir / "gait_cycle_frames.csv", index=False)
    
    print("\n[INFO] Completado Gait3D:")
    print(f"  - Secuencias totales: {len(df_seq)}")
    print(f"  - Frames totales: {len(df_frames)}")
    print(f"  - Metadatos guardados en: {out_dir}")


if __name__ == "__main__":
    main()
