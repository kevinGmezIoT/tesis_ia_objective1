#!/usr/bin/env python3
"""
Construye umbrales de iluminación comunes y balanceados a partir de los archivos CSV 
de secuencias de múltiples datasets (ej. Propio, iLIDS-VID, MARS).

Toma el mismo número de secuencias (el mínimo disponible entre los datasets) de cada uno
mediante un muestreo reproducible (seed fija) y calcula los percentiles P25, P50 y P75.

Genera el archivo JSON 'illum_thresholds_common.json'.

python src/10b_build_common_illumination_thresholds.py --input propio D:\tesis_dev\result1\dataset\output\metadata\sequences_labeled_raw.csv --input iLIDS D:\open_datasets\i-LIDS-VID\metadata\sequences_labeled_raw.csv --input MARS D:\open_datasets\Mars\metadata\sequences_labeled_raw.csv --out-json D:\tesis_dev\result1\dataset\output\metadata\illum_thresholds_common.json

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calcula umbrales comunes de iluminación a partir de una muestra balanceada de múltiples datasets."
    )
    parser.add_argument(
        "--input",
        nargs=2,
        action="append",
        required=True,
        metavar=("DATASET_NAME", "CSV_PATH"),
        help="Nombre del dataset y ruta al CSV de secuencias. Ej: --input propio metadata/sequences_propio.csv"
    )
    parser.add_argument(
        "--out-json",
        required=True,
        help="Ruta de salida para el JSON de umbrales comunes."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria para garantizar la reproducibilidad del muestreo."
    )
    parser.add_argument(
        "--luminance-col",
        default="illum_y_mean",
        help="Nombre de la columna que contiene la luminancia media en los CSV."
    )

    args = parser.parse_args()

    inputs: List[Tuple[str, Path]] = []
    for name, path_str in args.input:
        path = Path(path_str).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"No se encontró el archivo CSV para '{name}': {path}")
        inputs.append((name, path))

    out_json_path = Path(args.out_json).expanduser().resolve()
    out_json_path.parent.mkdir(parents=True, exist_ok=True)

    dfs: Dict[str, pd.DataFrame] = {}
    valid_counts: Dict[str, int] = {}

    print("[INFO] Cargando y validando archivos CSV...")
    for name, path in inputs:
        df = pd.read_csv(path)
        if args.luminance_col not in df.columns:
            raise KeyError(
                f"La columna '{args.luminance_col}' no se encuentra en el CSV de '{name}' ({path}).\n"
                f"Columnas disponibles: {df.columns.tolist()}"
            )
        
        # Filtrar solo secuencias con luminancia válida
        df_valid = df.dropna(subset=[args.luminance_col])
        dfs[name] = df_valid
        valid_counts[name] = len(df_valid)
        print(f"  - Dataset '{name}': {len(df)} secuencias cargadas ({len(df_valid)} con luminancia válida).")

    if not valid_counts:
        raise ValueError("No se cargaron datasets válidos.")

    # Calcular el tamaño balanceado (el mínimo de secuencias válidas entre todos)
    n_cal = min(valid_counts.values())
    if n_cal == 0:
        zero_datasets = [k for k, v in valid_counts.items() if v == 0]
        raise ValueError(
            f"Uno o más datasets tienen 0 secuencias válidas de luminancia: {zero_datasets}. "
            "Asegúrate de ejecutar primero la estimación de luminancia con 10b_label_sequence_illumination.py."
        )

    print(f"\n[INFO] Balanceando muestras: seleccionando n_cal = {n_cal} secuencias por dataset (Semilla: {args.seed}).")

    sampled_luminances: List[float] = []
    datasets_info: Dict[str, Dict[str, int]] = {}

    for name, df in dfs.items():
        # Muestreo reproducible
        df_sampled = df.sample(n=n_cal, random_state=args.seed)
        sampled_luminances.extend(df_sampled[args.luminance_col].astype(float).tolist())
        datasets_info[name] = {
            "total_sequences": valid_counts[name],
            "sampled_sequences": n_cal
        }

    # Calcular percentiles conjuntos sobre la muestra consolidada y balanceada
    arr_luminances = np.asarray(sampled_luminances, dtype=float)
    p25 = float(np.percentile(arr_luminances, 25.0))
    p50 = float(np.percentile(arr_luminances, 50.0))
    p75 = float(np.percentile(arr_luminances, 75.0))

    # Construir JSON final
    thresholds_payload = {
        "method": "balanced_common_percentiles_over_sequence_mean_luminance",
        "datasets": list(dfs.keys()),
        "sequences_per_dataset": n_cal,
        "total_calibration_sequences": len(arr_luminances),
        "seed": args.seed,
        "p25": round(p25, 4),
        "p50": round(p50, 4),
        "p75": round(p75, 4),
        "datasets_detail": datasets_info
    }

    # Guardar archivo JSON
    with out_json_path.open("w", encoding="utf-8") as f:
        json.dump(thresholds_payload, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Umbrales comunes calculados con éxito:")
    print(f"  - P25: {p25:.4f}")
    print(f"  - P50: {p50:.4f}")
    print(f"  - P75: {p75:.4f}")
    print(f"  - Total secuencias usadas en calibración: {len(arr_luminances)}")
    print(f"  - JSON guardado en: {out_json_path}")


if __name__ == "__main__":
    main()
