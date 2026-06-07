#!/usr/bin/env python3
"""
Genera los archivos metadata sequences.csv y sequence_frames.csv para el dataset MARS
a partir de su estructura de directorios y nombres de archivo.

Estructura esperada de MARS:
  mars_root/
    ├── bbox_train/
    │   ├── 0001/
    │   │   ├── 0001C1T0001F002.jpg  (o 0001_c1s1_000151_01.jpg)
    │   │   └── ...
    │   └── ...
    └── bbox_test/
        ├── 0001/
        │   ├── 0001C1T0001F002.jpg
        │   └── ...
        └── ...

Formato de salida compatible con los scripts de etiquetado posterior.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Patrones para extraer info del nombre del archivo:
# 1. Estándar: XXXX_cXsY_ZZZZZZ_ZZ.jpg (ID_c[cam]s[seq]_[frame]_[idx])
PAT_STANDARD = re.compile(r"^(\d+)_c(\d+)s(\d+)_(\d+)", re.IGNORECASE)
# 2. Alternativo: XXXXCXTXXXXFXXX.jpg (ID_C[cam]T[seq]F[frame])
PAT_ALT = re.compile(r"^(\d+)c(\d+)t(\d+)f(\d+)", re.IGNORECASE)


def parse_mars_filename(filename: str) -> Optional[Tuple[int, int, int, int]]:
    """
    Parsea el nombre de un archivo del dataset MARS.
    Retorna: (person_id, cam_id, sequence_id, frame_number) o None.
    """
    stem = Path(filename).stem
    
    # Intentar patrón estándar
    m = PAT_STANDARD.match(stem)
    if m:
        return (
            int(m.group(1)),
            int(m.group(2)),
            int(m.group(3)),
            int(m.group(4))
        )
    
    # Intentar patrón alternativo
    m = PAT_ALT.match(stem)
    if m:
        return (
            int(m.group(1)),
            int(m.group(2)),
            int(m.group(3)),
            int(m.group(4))
        )
        
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera metadatos compatibles de secuencias para el dataset MARS."
    )
    parser.add_argument(
        "--mars-root",
        required=True,
        help="Ruta raíz del dataset MARS (contiene bbox_train y/o bbox_test)."
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directorio donde se guardarán los metadatos (sequences.csv, sequence_frames.csv)."
    )
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="Guardar rutas RGB relativas a mars-root en lugar de absolutas."
    )
    
    args = parser.parse_args()
    
    mars_root = Path(args.mars_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    
    if not mars_root.is_dir():
        raise FileNotFoundError(f"No existe el directorio de MARS: {mars_root}")
        
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Encontrar carpetas de conjuntos (bbox_train, bbox_test)
    subsets = ["bbox_train", "bbox_test"]
    active_subsets = [s for s in subsets if (mars_root / s).is_dir()]
    
    if not active_subsets:
        raise FileNotFoundError(
            f"No se encontraron directorios bbox_train o bbox_test en: {mars_root}"
        )
        
    print(f"[INFO] Procesando subsets: {active_subsets}")
    
    # Estructura temporal para agrupar frames por secuencia
    # Llave: (subset, person_id, cam_id, seq_id) -> Lista de tuplas (frame_num, file_path)
    sequences_data: Dict[Tuple[str, int, int, int], List[Tuple[int, Path]]] = {}
    
    for subset in active_subsets:
        subset_path = mars_root / subset
        # Listar directorios de identidades (0001, 0002, etc.)
        person_dirs = sorted([
            p for p in subset_path.iterdir() 
            if p.is_dir() and p.name.isdigit()
        ])
        
        for person_dir in tqdm(person_dirs, desc=f"Escaneando identidades en {subset}"):
            person_id_dir = int(person_dir.name)
            
            # Buscar todas las imágenes de este personaje
            for file_path in person_dir.iterdir():
                if not file_path.is_file() or file_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                
                parsed = parse_mars_filename(file_path.name)
                if parsed is None:
                    # Fallback si no parsea: agrupar por cámara '1' y secuencia '1', frame por orden alfabético
                    person_id = person_id_dir
                    cam_id = 1
                    seq_id = 1
                    # Extraer cualquier número del nombre como frame o usar hash
                    nums = re.findall(r"\d+", file_path.stem)
                    frame_num = int(nums[-1]) if nums else 0
                else:
                    person_id, cam_id, seq_id, frame_num = parsed
                    
                # Validar coherencia del ID del directorio con el del archivo
                if person_id != person_id_dir:
                    # Usamos el del archivo pero avisamos
                    pass
                
                key = (subset, person_id, cam_id, seq_id)
                if key not in sequences_data:
                    sequences_data[key] = []
                sequences_data[key].append((frame_num, file_path))
                
    # Generar filas de salida
    cycle_rows: List[Dict[str, object]] = []
    frame_rows: List[Dict[str, object]] = []
    
    # Procesar y ordenar secuencias reconstruidas
    sorted_keys = sorted(sequences_data.keys())
    
    for key in tqdm(sorted_keys, desc="Generando metadatos de secuencias"):
        subset, person_id, cam_id, seq_id = key
        frames_list = sorted(sequences_data[key], key=lambda x: x[0])
        
        # Identificador único de secuencia
        # ej. mars_bbox_train_ID0001_c001_seq0001
        seq_label = f"mars_{subset}_ID{person_id:04d}_c{cam_id:03d}_seq{seq_id:04d}"
        
        frame_numbers = [item[0] for item in frames_list]
        frame_start = frame_numbers[0]
        frame_end = frame_numbers[-1]
        num_frames = len(frames_list)
        
        # Guardar metadatos de secuencia
        cycle_rows.append({
            "sequence_id": seq_label,
            "cycle_id": seq_label,
            "person_global_id": person_id,
            "scene_id": f"mars_{subset}",
            "cam_id": f"c{cam_id:03d}",
            "tracklet_index": seq_id,
            "cycle_index": seq_id,
            "num_frames": num_frames,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "frame_numbers_json": json.dumps(frame_numbers),
            "n_samples_rgb": num_frames,
            "n_samples_sil": 0,  # MARS usualmente no trae siluetas binarias por defecto
        })
        
        # Guardar metadatos de cada frame
        for pos, (frame_num, path) in enumerate(frames_list, start=1):
            if args.relative_paths:
                rgb_path_str = str(path.relative_to(mars_root))
            else:
                rgb_path_str = str(path)
                
            frame_rows.append({
                "sequence_id": seq_label,
                "cycle_id": seq_label,
                "person_global_id": person_id,
                "scene_id": f"mars_{subset}",
                "cam_id": f"c{cam_id:03d}",
                "tracklet_index": seq_id,
                "cycle_index": seq_id,
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
