"""
Clasifica la iluminación de secuencias RGB a partir de la luminancia media de
los crops de persona.

Base metodológica
-----------------
El descriptor fotométrico principal es la luminancia media por imagen, siguiendo
el criterio utilizado en "Illumination-Adaptive Person Re-identification"
(IA-ReID) para representar y analizar variaciones de iluminación.

Procedimiento
-------------
1. Muestrea hasta K frames RGB equidistantes por secuencia.
2. Calcula la luminancia media Y de cada crop RGB:
       Y = 0.299 R + 0.587 G + 0.114 B
3. Agrega por secuencia:
       illum_y_mean = media temporal de Y
       illum_y_std  = desviación estándar temporal de Y
       illum_y_cv   = illum_y_std / (illum_y_mean + eps)
4. Clasifica cada secuencia en 4 niveles ordinales mediante umbrales comunes:
       Q1: baja
       Q2: media_baja
       Q3: media_alta
       Q4: alta

Para comparar datasets RGB, los umbrales deben ser los mismos. Genere los
umbrales una sola vez sobre una distribución de referencia común y reutilícelos
con --illum-thresholds-json.

Requisitos
----------
pip install pandas numpy opencv-python tqdm

Ejemplo: ajustar umbrales en un corpus de calibración
-----------------------------------------------------
python 10a_label_sequence_illumination.py ^
  --base-dir dataset_root ^
  --sequences-csv metadata/sequences.csv ^
  --frames-csv metadata/sequence_frames.csv ^
  --out-csv metadata/sequences_illumination.csv ^
  --max-frames-per-seq 10 ^
  --save-illum-thresholds-json metadata/illum_thresholds_common.json

Ejemplo: aplicar umbrales comunes a otro dataset
------------------------------------------------
python 10a_label_sequence_illumination.py ^
  --base-dir mars_root ^
  --sequences-csv metadata/sequences.csv ^
  --frames-csv metadata/sequence_frames.csv ^
  --out-csv metadata/sequences_illumination.csv ^
  --illum-thresholds-json metadata/illum_thresholds_common.json


python src/10a_label_sequence_illumination.py --base-dir D:\tesis_dev\result1 --sequences-csv D:\tesis_dev\result1\dataset\output\metadata\sequences.csv --frames-csv D:\tesis_dev\result1\dataset\output\metadata\sequence_frames.csv --out-csv D:\tesis_dev\result1\dataset\output\metadata\sequences_illumination.csv --illum-thresholds-json D:\tesis_dev\result1\dataset\output\metadata\illum_thresholds_common.json
python src/10a_label_sequence_illumination.py --base-dir D:\open_datasets\i-LIDS-VID --sequences-csv D:\open_datasets\i-LIDS-VID\metadata\sequences.csv --frames-csv D:\open_datasets\i-LIDS-VID\metadata\sequence_frames.csv --out-csv D:\open_datasets\i-LIDS-VID\metadata\sequences_illumination.csv --illum-thresholds-json D:\tesis_dev\result1\dataset\output\metadata\illum_thresholds_common.json
python src/10a_label_sequence_illumination.py --base-dir D:\open_datasets\Mars\raw --sequences-csv D:\open_datasets\Mars\metadata\sequences.csv --frames-csv D:\open_datasets\Mars\metadata\sequence_frames.csv --out-csv D:\open_datasets\Mars\metadata\sequences_illumination.csv --illum-thresholds-json D:\tesis_dev\result1\dataset\output\metadata\illum_thresholds_common.json
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


LABELS = {
    0: "baja",
    1: "media_baja",
    2: "media_alta",
    3: "alta",
}

PERCENTILE_BINS = {
    0: "Q1",
    1: "Q2",
    2: "Q3",
    3: "Q4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clasifica iluminación por secuencia usando luminancia media de crops RGB "
            "y cuatro niveles ordinales definidos por umbrales percentilares comunes."
        )
    )
    parser.add_argument(
        "--base-dir",
        "--base_dir",
        dest="base_dir",
        type=str,
        default=".",
        help="Raíz para resolver rutas relativas de frames RGB.",
    )
    parser.add_argument("--sequences-csv", "--sequences_csv", dest="sequences_csv", required=True)
    parser.add_argument("--frames-csv", "--frames_csv", dest="frames_csv", required=True)
    parser.add_argument("--out-csv", "--out_csv", dest="out_csv", required=True)
    parser.add_argument(
        "--max-frames-per-seq",
        "--max_frames_per_seq",
        dest="max_frames_per_seq",
        type=int,
        default=10,
        help="Máximo de crops RGB equidistantes muestreados por secuencia.",
    )
    parser.add_argument(
        "--rgb-path-col",
        dest="rgb_path_col",
        default=None,
        help=(
            "Columna que contiene la ruta RGB. Si se omite, se detecta automáticamente "
            "entre rgb_path, rgb_source_path y rgb_output_path."
        ),
    )
    parser.add_argument(
        "--illum-thresholds-json",
        dest="illum_thresholds_json",
        default=None,
        help=(
            "JSON con umbrales comunes p25, p50 y p75. Para comparación entre datasets, "
            "use el mismo archivo en todos."
        ),
    )
    parser.add_argument(
        "--save-illum-thresholds-json",
        dest="save_illum_thresholds_json",
        default=None,
        help=(
            "Ruta donde guardar los umbrales estimados. Solo se usa cuando no se pasa "
            "--illum-thresholds-json."
        ),
    )
    parser.add_argument(
        "--min-valid-sequences-for-fit",
        type=int,
        default=10,
        help="Mínimo de secuencias válidas para estimar percentiles.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Constante pequeña para evitar división por cero al calcular CV.",
    )
    return parser.parse_args()


def choose_existing_column(df: pd.DataFrame, explicit: Optional[str], candidates: Sequence[str]) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(
                f"La columna solicitada '{explicit}' no existe. "
                f"Columnas disponibles: {df.columns.tolist()}"
            )
        return explicit

    for name in candidates:
        if name in df.columns:
            return name

    raise ValueError(
        "No se encontró una columna de ruta RGB. "
        f"Se buscó: {list(candidates)}. "
        f"Columnas disponibles: {df.columns.tolist()}"
    )


def resolve_path(base_dir: str, value: object) -> str:
    if pd.isna(value):
        raise FileNotFoundError("Ruta RGB vacía.")
    raw = str(value)
    if os.path.isabs(raw):
        return raw
    return os.path.join(base_dir, raw)


def sample_group_rows(group: pd.DataFrame, k: int) -> pd.DataFrame:
    """Selecciona hasta k frames equidistantes conservando el orden temporal."""
    if k <= 0:
        raise ValueError("--max-frames-per-seq debe ser mayor que cero.")

    sort_candidates = ["position_in_sequence", "position_in_cycle", "frame_number"]
    sort_cols = [c for c in sort_candidates if c in group.columns]
    ordered = group.sort_values(sort_cols) if sort_cols else group

    n = len(ordered)
    if n <= k:
        return ordered

    idx = np.linspace(0, n - 1, num=k, dtype=int)
    return ordered.iloc[idx]


def read_bgr(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def mean_luminance_bt601(rgb_path: str) -> float:
    """
    Calcula luminancia media BT.601/JPEG-YCbCr sobre el crop RGB de persona.

    OpenCV lee BGR, por eso:
        Y = 0.114 B + 0.587 G + 0.299 R
    equivalente a:
        Y = 0.299 R + 0.587 G + 0.114 B
    """
    image = read_bgr(rgb_path).astype(np.float32)
    blue = image[:, :, 0]
    green = image[:, :, 1]
    red = image[:, :, 2]
    luminance = 0.114 * blue + 0.587 * green + 0.299 * red
    return float(np.mean(luminance))


def compute_sequence_metrics(
    frames: pd.DataFrame,
    base_dir: str,
    rgb_path_col: str,
    max_frames_per_seq: int,
    eps: float,
) -> Tuple[pd.DataFrame, int]:
    rows: List[Dict[str, object]] = []
    missing_rgb = 0

    grouped = frames.groupby("sequence_id", sort=False)
    for sequence_id, group in tqdm(grouped, desc="Calculando iluminación por secuencia"):
        sampled = sample_group_rows(group, max_frames_per_seq)
        luminance_values: List[float] = []
        sampled_frame_numbers: List[int] = []

        for _, row in sampled.iterrows():
            try:
                rgb_path = resolve_path(base_dir, row[rgb_path_col])
                luminance_values.append(mean_luminance_bt601(rgb_path))
                if "frame_number" in row and not pd.isna(row["frame_number"]):
                    sampled_frame_numbers.append(int(row["frame_number"]))
            except FileNotFoundError:
                missing_rgb += 1

        n = len(luminance_values)
        if n == 0:
            illum_y_mean = np.nan
            illum_y_std = np.nan
            illum_y_cv = np.nan
        else:
            values = np.asarray(luminance_values, dtype=np.float64)
            illum_y_mean = float(np.mean(values))
            # ddof=1 cuando es posible: desviación estándar muestral temporal.
            illum_y_std = float(np.std(values, ddof=1)) if n > 1 else 0.0
            illum_y_cv = float(illum_y_std / (illum_y_mean + eps))

        rows.append(
            {
                "sequence_id": sequence_id,
                "illum_y_mean": illum_y_mean,
                "illum_y_std": illum_y_std,
                "illum_y_cv": illum_y_cv,
                "n_samples_rgb": n,
                "sampled_frame_numbers_json": json.dumps(sampled_frame_numbers),
            }
        )

    return pd.DataFrame(rows), missing_rgb


def fit_thresholds(values: pd.Series, min_valid: int) -> Dict[str, float]:
    valid = values.dropna().astype(float)
    if len(valid) < min_valid:
        raise ValueError(
            f"Solo hay {len(valid)} secuencias válidas para iluminación; "
            f"se requieren al menos {min_valid} para estimar percentiles."
        )

    p25, p50, p75 = np.percentile(valid.to_numpy(), [25.0, 50.0, 75.0])
    return {
        "method": "common_percentiles_over_sequence_mean_luminance",
        "luminance_formula": "Y = 0.299 R + 0.587 G + 0.114 B",
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "n_reference_sequences": int(len(valid)),
    }


def load_thresholds(path: str) -> Dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"p25", "p50", "p75"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"El JSON de umbrales no contiene: {sorted(missing)}")

    thresholds = {**payload}
    thresholds["p25"] = float(payload["p25"])
    thresholds["p50"] = float(payload["p50"])
    thresholds["p75"] = float(payload["p75"])

    if not (thresholds["p25"] <= thresholds["p50"] <= thresholds["p75"]):
        raise ValueError("Los umbrales deben cumplir p25 <= p50 <= p75.")

    return thresholds


def save_thresholds(path: str, thresholds: Dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(thresholds, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def classify_luminance(value: float, thresholds: Dict[str, float]) -> Tuple[Optional[int], str, str]:
    if pd.isna(value):
        return None, "desconocida", "NA"

    p25 = thresholds["p25"]
    p50 = thresholds["p50"]
    p75 = thresholds["p75"]

    if value <= p25:
        level = 0
    elif value <= p50:
        level = 1
    elif value <= p75:
        level = 2
    else:
        level = 3

    return level, LABELS[level], PERCENTILE_BINS[level]


def main() -> None:
    args = parse_args()

    sequences = pd.read_csv(args.sequences_csv)
    frames = pd.read_csv(args.frames_csv)

    if "sequence_id" not in sequences.columns:
        raise ValueError("Falta 'sequence_id' en sequences_csv.")
    if "sequence_id" not in frames.columns:
        raise ValueError("Falta 'sequence_id' en frames_csv.")

    rgb_path_col = choose_existing_column(
        frames,
        args.rgb_path_col,
        candidates=("rgb_path", "rgb_source_path", "rgb_output_path"),
    )

    metrics, missing_rgb = compute_sequence_metrics(
        frames=frames,
        base_dir=args.base_dir,
        rgb_path_col=rgb_path_col,
        max_frames_per_seq=args.max_frames_per_seq,
        eps=args.eps,
    )

    if args.illum_thresholds_json:
        thresholds = load_thresholds(args.illum_thresholds_json)
        threshold_source = f"cargados desde {args.illum_thresholds_json}"
    else:
        thresholds = fit_thresholds(
            metrics["illum_y_mean"],
            min_valid=args.min_valid_sequences_for_fit,
        )
        threshold_source = "estimados sobre las secuencias procesadas"
        if args.save_illum_thresholds_json:
            save_thresholds(args.save_illum_thresholds_json, thresholds)

    classifications = metrics["illum_y_mean"].apply(
        lambda value: classify_luminance(value, thresholds)
    )
    metrics["illum_level"] = classifications.apply(lambda item: item[0]).astype("Int64")
    metrics["illum_label"] = classifications.apply(lambda item: item[1])
    metrics["illum_percentile_bin"] = classifications.apply(lambda item: item[2])

    # Evita duplicar columnas si sequences.csv ya tenía una versión previa.
    derived_cols = [
        "illum_y_mean",
        "illum_y_std",
        "illum_y_cv",
        "n_samples_rgb",
        "sampled_frame_numbers_json",
        "illum_level",
        "illum_label",
        "illum_percentile_bin",
    ]
    columns_to_drop = [col for col in derived_cols if col in sequences.columns]
    if columns_to_drop:
        sequences = sequences.drop(columns=columns_to_drop)

    output = sequences.merge(metrics, on="sequence_id", how="left", validate="one_to_one")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)

    print("[INFO] Método: luminancia media por crop RGB y agregación por secuencia.")
    print("[INFO] Fórmula: Y = 0.299 R + 0.587 G + 0.114 B")
    print(f"[INFO] Columna RGB utilizada: {rgb_path_col}")
    print(f"[INFO] Umbrales {threshold_source}:")
    print(
        "[INFO] "
        f"p25={thresholds['p25']:.4f}, "
        f"p50={thresholds['p50']:.4f}, "
        f"p75={thresholds['p75']:.4f}"
    )
    print(f"[WARN] Frames RGB faltantes entre los muestreados: {missing_rgb}")
    print(f"[INFO] CSV generado: {out_path} (filas={len(output)})")

    print("[INFO] Distribución por nivel:")
    distribution = (
        output["illum_label"]
        .fillna("desconocida")
        .value_counts(dropna=False)
        .reindex(["baja", "media_baja", "media_alta", "alta", "desconocida"], fill_value=0)
    )
    for label, count in distribution.items():
        print(f"        {label}: {int(count)}")


if __name__ == "__main__":
    main()
