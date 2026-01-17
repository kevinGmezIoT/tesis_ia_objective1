"""
Etiqueta cada ciclo (sequence_id) con:
- Iluminación (baja/media/alta) usando percentiles de brillo promedio por secuencia.
- Oclusión con MENOS falsos positivos usando un proxy más invariante a escala:
    * fill_ratio = area_silueta / area_bbox(silueta)
  y normalización por cámara (cam_id) o (cam_id, scene) para evitar sesgos por distancia/recorte.

Requisitos:
  pip install pandas numpy opencv-python tqdm

Uso:
  python 10_label_ilumination_occlusion_v2.py \
    --base_dir /ruta/a/tu/dataset_root \
    --sequences_csv /ruta/a/sequences.csv \
    --frames_csv /ruta/a/sequence_frames.csv \
    --out_csv /ruta/a/sequences_labeled.csv \
    --max_frames_per_seq 10 \
    --occl_group_cols cam_id \
    --occl_expected_percentile 90 \
    --occl_thr_no 0.90 \
    --occl_thr_partial 0.75
"""

import os
import argparse
import warnings
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm


def read_gray(path: str) -> np.ndarray:
    """Lee imagen y devuelve en escala de grises (uint8)."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def frame_stats_rgb(rgb_path: str) -> Tuple[float, float]:
    """Brillo y contraste por frame (media y std de intensidades en gris)."""
    g = read_gray(rgb_path)
    return float(g.mean()), float(g.std())


def silhouette_metrics(
    sil_path: str,
    thr: int = 0,
) -> Tuple[int, float, int, int, int]:
    """
    Métricas por frame de silueta:
      - area: pixeles foreground
      - fill_ratio: area / area_bbox(fg)  (más invariante a escala/distancia)
      - border_touch: 1 si fg toca borde (indicador de truncamiento por encuadre/crop)
      - bbox_h, bbox_w: dimensiones del bbox del fg
    """
    s = read_gray(sil_path)
    if thr > 0:
        fg = (s > thr)
    else:
        fg = (s > 0)

    area = int(fg.sum())
    if area == 0:
        return 0, np.nan, 0, 0, 0

    ys, xs = np.where(fg)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    bbox_h = (y1 - y0 + 1)
    bbox_w = (x1 - x0 + 1)
    bbox_area = int(bbox_h * bbox_w)

    fill = float(area / bbox_area) if bbox_area > 0 else np.nan

    h, w = fg.shape[:2]
    # toca borde si hay fg en cualquiera de los bordes
    border = int(
        fg[0, :].any() or fg[h - 1, :].any() or fg[:, 0].any() or fg[:, w - 1].any()
    )
    return area, fill, border, bbox_h, bbox_w


def sample_group_rows(g: pd.DataFrame, k: int) -> pd.DataFrame:
    """Toma hasta k frames por secuencia, espaciados uniformemente."""
    n = len(g)
    if n <= k:
        return g
    idx = np.linspace(0, n - 1, num=k, dtype=int)
    return g.iloc[idx]


def parse_group_cols(s: str) -> List[str]:
    cols = [c.strip() for c in s.split(",") if c.strip()]
    return cols


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--base_dir", type=str, default=".", help="Raíz del dataset para resolver rgb_path/silhouette_path")
    ap.add_argument("--sequences_csv", type=str, required=True)
    ap.add_argument("--frames_csv", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)

    ap.add_argument("--max_frames_per_seq", type=int, default=10, help="Frames a muestrear por secuencia (reduce costo)")
    ap.add_argument("--sil_thr", type=int, default=0, help="Umbral foreground en silueta (0 si ya es binaria)")

    # Iluminación
    ap.add_argument("--p_low", type=float, default=25.0, help="Percentil bajo para iluminación")
    ap.add_argument("--p_high", type=float, default=75.0, help="Percentil alto para iluminación")

    # Oclusión (mejorado)
    ap.add_argument("--occl_group_cols", type=str, default="cam_id",
                    help="Columnas para normalizar expected_fill (ej: cam_id o cam_id,scene)")
    ap.add_argument("--occl_expected_percentile", type=float, default=90.0,
                    help="Percentil para estimar expected_fill por grupo")
    ap.add_argument("--occl_border_max_expected", type=float, default=0.10,
                    help="Máx fracción de border_touch permitida para estimar expected_fill (filtra truncamientos)")
    ap.add_argument("--occl_thr_no", type=float, default=0.90,
                    help="Umbral normalized_fill para sin_occlusion (>=)")
    ap.add_argument("--occl_thr_partial", type=float, default=0.75,
                    help="Umbral normalized_fill para parcial (>=)")

    args = ap.parse_args()

    base_dir = args.base_dir

    seq = pd.read_csv(args.sequences_csv)
    frames = pd.read_csv(args.frames_csv)

    required_cols = {"sequence_id", "rgb_path", "silhouette_path", "cam_id", "scene"}
    missing = required_cols - set(frames.columns)
    if missing:
        raise ValueError(f"Faltan columnas en frames_csv: {missing}")

    # Estructuras por secuencia
    rows_out: List[Dict] = []
    missing_rgb = 0
    missing_sil = 0

    # Procesa por secuencia, muestreando k frames por secuencia
    for seq_id, g in tqdm(frames.groupby("sequence_id", sort=False), desc="Procesando secuencias"):
        g_s = sample_group_rows(g, args.max_frames_per_seq)

        # (cam_id, scene) deberían ser constantes por secuencia; tomamos el primero.
        cam_id = g_s["cam_id"].iloc[0] if "cam_id" in g_s.columns else None
        scene = g_s["scene"].iloc[0] if "scene" in g_s.columns else None

        brightness_list: List[float] = []
        contrast_list: List[float] = []

        area_list: List[int] = []
        fill_list: List[float] = []
        border_list: List[int] = []
        bbox_h_list: List[int] = []
        bbox_w_list: List[int] = []

        for _, r in g_s.iterrows():
            rgb_path = os.path.join(base_dir, str(r["rgb_path"]))
            sil_path = os.path.join(base_dir, str(r["silhouette_path"]))

            # RGB stats
            try:
                b, c = frame_stats_rgb(rgb_path)
                brightness_list.append(b)
                contrast_list.append(c)
            except FileNotFoundError:
                missing_rgb += 1

            # Silhouette metrics
            try:
                a, fill, border, bh, bw = silhouette_metrics(sil_path, thr=args.sil_thr)
                area_list.append(a)
                fill_list.append(fill)
                border_list.append(border)
                bbox_h_list.append(bh)
                bbox_w_list.append(bw)
            except FileNotFoundError:
                missing_sil += 1

        if len(brightness_list) == 0 and len(area_list) == 0:
            # Sin datos, registrar para auditoría
            rows_out.append(
                {
                    "sequence_id": seq_id,
                    "cam_id": cam_id,
                    "scene": scene,
                    "illum_mean": np.nan,
                    "illum_std": np.nan,
                    "contrast_mean": np.nan,
                    "area_mean": np.nan,
                    "fill_mean": np.nan,
                    "fill_median": np.nan,
                    "border_frac": np.nan,
                    "bbox_h_median": np.nan,
                    "bbox_w_median": np.nan,
                    "n_samples_rgb": 0,
                    "n_samples_sil": 0,
                }
            )
            continue

        # Agregación robusta (medianas ayudan a reducir outliers por un frame malo)
        rows_out.append(
            {
                "sequence_id": seq_id,
                "cam_id": cam_id,
                "scene": scene,
                "illum_mean": float(np.mean(brightness_list)) if brightness_list else np.nan,
                "illum_std": float(np.std(brightness_list)) if brightness_list else np.nan,
                "contrast_mean": float(np.mean(contrast_list)) if contrast_list else np.nan,
                "area_mean": float(np.mean(area_list)) if area_list else np.nan,
                "fill_mean": float(np.nanmean(fill_list)) if fill_list else np.nan,
                "fill_median": float(np.nanmedian(fill_list)) if fill_list else np.nan,
                "border_frac": float(np.mean(border_list)) if border_list else np.nan,
                "bbox_h_median": float(np.median(bbox_h_list)) if bbox_h_list else np.nan,
                "bbox_w_median": float(np.median(bbox_w_list)) if bbox_w_list else np.nan,
                "n_samples_rgb": len(brightness_list),
                "n_samples_sil": len(area_list),
            }
        )

    metrics = pd.DataFrame(rows_out)

    # -------------------------
    # ILUMINACIÓN (igual que antes)
    # -------------------------
    valid_illum = metrics["illum_mean"].dropna()
    if len(valid_illum) < 10:
        warnings.warn("Muy pocos datos válidos para iluminación; revisa rutas/base_dir.")
        p25 = p75 = np.nan
    else:
        p25 = np.percentile(valid_illum, args.p_low)
        p75 = np.percentile(valid_illum, args.p_high)

    def label_illum(x: float) -> str:
        if np.isnan(x) or np.isnan(p25) or np.isnan(p75):
            return "desconocida"
        if x < p25:
            return "baja"
        if x > p75:
            return "alta"
        return "media"

    metrics["illum_label"] = metrics["illum_mean"].apply(label_illum)

    # Bandera de iluminación "variable"
    valid_var = metrics["illum_std"].dropna()
    if len(valid_var) >= 10:
        var_thr = np.percentile(valid_var, 75)
        metrics["illum_variability"] = np.where(metrics["illum_std"] >= var_thr, "variable", "estable")
    else:
        metrics["illum_variability"] = "desconocida"

    # -------------------------
    # OCLUSIÓN (MEJORADO): fill_ratio normalizado por grupo para reducir falsos positivos por escala
    # -------------------------
    group_cols = parse_group_cols(args.occl_group_cols)
    for c in group_cols:
        if c not in metrics.columns:
            raise ValueError(f"No existe columna '{c}' en métricas. Columnas disponibles: {metrics.columns.tolist()}")

    # Estimar expected_fill por grupo usando secuencias SIN truncamiento (border_frac bajo)
    # Esto evita que "cortes por borde" inflen la noción de oclusión típica.
    expected_fill_map: Dict[Tuple, float] = {}

    # Fallback global si un grupo no tiene suficientes datos
    global_pool = metrics[(metrics["fill_median"].notna()) & (metrics["border_frac"].notna())]
    global_pool = global_pool[global_pool["border_frac"] <= args.occl_border_max_expected]
    if len(global_pool) < 50:
        # si no hay suficientes, usa todos los fill_median disponibles
        global_pool = metrics[metrics["fill_median"].notna()]
    global_expected_fill = float(np.percentile(global_pool["fill_median"].values, args.occl_expected_percentile)) if len(global_pool) else np.nan

    # Por grupo
    for key, g in metrics.groupby(group_cols, dropna=False):
        gg = g[(g["fill_median"].notna()) & (g["border_frac"].notna())]
        gg = gg[gg["border_frac"] <= args.occl_border_max_expected]
        if len(gg) >= 30:
            expected = float(np.percentile(gg["fill_median"].values, args.occl_expected_percentile))
        else:
            expected = global_expected_fill
        expected_fill_map[key if isinstance(key, tuple) else (key,)] = expected

    def lookup_expected(row: pd.Series) -> float:
        k = tuple(row[c] for c in group_cols)
        if k in expected_fill_map:
            return expected_fill_map[k]
        # por seguridad
        return global_expected_fill

    metrics["expected_fill"] = metrics.apply(lookup_expected, axis=1)

    def normalized_fill(fill_med: float, expected: float) -> float:
        if np.isnan(fill_med) or np.isnan(expected) or expected <= 0:
            return np.nan
        return float(np.clip(fill_med / expected, 0.0, 1.5))

    metrics["fill_norm"] = [
        normalized_fill(fm, ef) for fm, ef in zip(metrics["fill_median"].values, metrics["expected_fill"].values)
    ]

    # Etiquetado final con umbrales sobre fill_norm
    def label_occlusion(fill_norm: float) -> str:
        if np.isnan(fill_norm):
            return "desconocida"
        if fill_norm >= args.occl_thr_no:
            return "sin_occlusion"
        if fill_norm >= args.occl_thr_partial:
            return "parcial"
        return "alta"

    metrics["occlusion_label"] = metrics["fill_norm"].apply(label_occlusion)

    # Etiqueta auxiliar: truncamiento por encuadre (no reemplaza occlusion_label, pero ayuda a interpretar falsos positivos)
    metrics["truncation_flag"] = np.where(metrics["border_frac"].notna() & (metrics["border_frac"] >= 0.30), 1, 0)

    # -------------------------
    # Unir con sequences.csv (una fila por ciclo) y exportar
    # -------------------------
    out = seq.merge(metrics, on="sequence_id", how="left")

    # Auditoría mínima
    print(f"[INFO] Iluminación: p25={p25:.2f} p75={p75:.2f}")
    print(f"[INFO] Oclusión: expected_fill global (P{args.occl_expected_percentile:.0f})={global_expected_fill:.4f}")
    print(f"[INFO] Oclusión: group_cols={group_cols} border_max_expected={args.occl_border_max_expected}")
    print(f"[INFO] Oclusión: thr_no={args.occl_thr_no} thr_partial={args.occl_thr_partial}")
    print(f"[WARN] frames RGB faltantes (muestreados): {missing_rgb}")
    print(f"[WARN] frames SIL faltantes (muestreados): {missing_sil}")
    print(f"[INFO] Guardando: {args.out_csv}  (filas={len(out)})")

    out.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
