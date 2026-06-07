#!/usr/bin/env python3
"""
Genera los archivos metadata sequences.csv y sequence_frames.csv para el dataset CASIA-B
a partir de su estructura de directorios de siluetas.

Estructura esperada de CASIA-B:
  casiab_root/
    ├── output/ (o directamente las carpetas de IDs)
    │   ├── 001/
    │   │   ├── bg-01/
    │   │   │   ├── 000/
    │   │   │   │   ├── 001-bg-01-000-001.png
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
    """Extrae el número de frame del nombre del archivo (ej. 001-bg-01-000-001.png -> 1)."""
    # Usualmente el formato es: {id}-{cond}-{angle}-{frame}.png
    stem = Path(filename).stem
    parts = stem.split("-")
    if len(parts) >= 4 and parts[-1].isdigit():
        return int(parts[-1])
    
    # Fallback genérico
    nums = re.findall(r"\d+", stem)
    if nums:
        return int(nums[-1])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera metadatos compatibles de secuencias para el dataset CASIA-B."
    )
    parser.add_argument(
        "--casiab-root",
        required=True,
        help="Ruta raíz de CASIA-B (ej. D:\\open_datasets\\CASIA-B)."
    )
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="Guardar rutas relativas a casiab-root en lugar de absolutas."
    )
    
    args = parser.parse_args()
    
    casiab_root = Path(args.casiab_root).expanduser().resolve()
    
    if not casiab_root.is_dir():
        raise FileNotFoundError(f"No existe el directorio de CASIA-B: {casiab_root}")
        
    out_dir = casiab_root / "metadata"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Encontrar la carpeta base que contiene los IDs (ej. 001, 002)
    seq_base = casiab_root
    if (casiab_root / "output").is_dir():
        seq_base = casiab_root / "output"
        
    print(f"[INFO] Buscando siluetas de CASIA-B en: {seq_base}")
    
    person_dirs = sorted([p for p in seq_base.iterdir() if p.is_dir() and p.name.isdigit()])
    
    cycle_rows: List[Dict[str, object]] = []
    frame_rows: List[Dict[str, object]] = []
    
    for person_dir in tqdm(person_dirs, desc="Procesando identidades CASIA-B"):
        person_id_str = person_dir.name
        person_id = int(person_id_str)
        
        # Condiciones (ej. bg-01, bg-02, cl-01, nm-01)
        for cond_dir in person_dir.iterdir():
            if not cond_dir.is_dir():
                continue
            scene_id = cond_dir.name
            
            # Ángulos (ej. 000, 018, 036, etc.)
            for angle_dir in cond_dir.iterdir():
                if not angle_dir.is_dir():
                    continue
                cam_id = angle_dir.name
                
                # Frames
                frame_files = sorted([
                    f for f in angle_dir.iterdir()
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
                seq_label = f"casiab_{person_id_str}_{scene_id}_{cam_id}"
                
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
                    "tracklet_index": 1,
                    "cycle_index": 1,
                    "num_frames": num_frames,
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                    "frame_numbers_json": json.dumps(frame_numbers),
                    "n_samples_rgb": 0,  # Asumimos que son siluetas binarias
                    "n_samples_sil": num_frames,
                })
                
                # Metadatos de frame
                for pos, (frame_num, path) in enumerate(frames_list, start=1):
                    if args.relative_paths:
                        path_str = str(path.relative_to(casiab_root))
                    else:
                        path_str = str(path)
                        
                    frame_rows.append({
                        "sequence_id": seq_label,
                        "cycle_id": seq_label,
                        "person_global_id": person_id,
                        "scene_id": scene_id,
                        "cam_id": cam_id,
                        "tracklet_index": 1,
                        "cycle_index": 1,
                        "position_in_sequence": pos,
                        "position_in_cycle": pos,
                        "frame_number": frame_num,
                        "rgb_source_path": "", # CASIA-B comúnmente distribuye las máscaras, no RGB
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
    
    print("\n[INFO] Completado CASIA-B:")
    print(f"  - Secuencias totales: {len(df_seq)}")
    print(f"  - Frames totales: {len(df_frames)}")
    print(f"  - Metadatos guardados en: {out_dir}")


if __name__ == "__main__":
    main()
