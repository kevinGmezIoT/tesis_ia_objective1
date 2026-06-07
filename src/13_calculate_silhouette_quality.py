#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
13_label_silhouette_quality_liu_wang.py

Calcula variables de calidad de silueta para secuencias/ciclos de marcha,
separando este módulo de iluminación, oclusión RGB, ropa, color, Gower e
intra/inter-clase.

Fundamento metodológico:
1) Liu, Zheng & Xiong (2009): Silhouette Quality Quantification (SQQ)
   - señal foreground-sum x(t)
   - normalización por ciclo
   - ventana deslizante W ~= 2T
   - matriz de autocorrelación
   - autovalores para energía estructurada y energía de ruido
   - Q(i) = sigma_s^2 - sigma_z^2

2) Wang et al. (2024), QAGait:
   - Maximal Connected Area para remover ruido de fondo y filtrar siluetas no confiables
   - Template Match con Hu moments / cv2.matchShapes para formas no humanas o no identificables
   - minAreaRect para ángulo de inclinación/alineamiento como variable de auditoría

Este script NO calcula:
- iluminación
- oclusión por keypoints
- atributos de ropa/color
- Gower
- distancia intra/inter-clase

Entradas esperadas:
- sequences_csv: una fila por secuencia/ciclo
- frames_csv: una fila por frame con rutas a siluetas, recomendado

Columnas de ruta aceptadas en frames_csv:
- silhouette_path
- silhouette_source_path
- silhouette_output_path
- sil_path
- sil_source_path
- sil_output_path

Columnas de ruta aceptadas en sequences_csv si no se da frames_csv:
- silhouette_dir

Salidas:
- CSV por secuencia con métricas agregadas
- CSV por frame con métricas morfológicas
- CSV por ventana SQQ con Q(i)
- Reporte TXT opcional

python src/13_calculate_silhouette_quality.py --base-dir dataset/output --sequences-csv dataset/output/metadata/gait_cycles.csv --frames-csv dataset/output/metadata/gait_cycle_frames.csv --out-csv dataset/output/metadata/gait_cycles_quality.csv --out-frame-csv dataset/output/metadata/gait_cycle_frames_quality.csv --out-sqq-csv dataset/output/metadata/gait_cycles_sqq_windows.csv --report dataset/output/metadata/silhouette_quality_report.txt

CASIA-B:
python src/13_calculate_silhouette_quality.py --base-dir /open_datasets/CASIA-B --sequences-csv /open_datasets/CASIA-B/metadata/gait_cycles.csv --frames-csv /open_datasets/CASIA-B/metadata/gait_cycle_frames.csv --out-csv /open_datasets/CASIA-B/metadata/gait_cycles_quality.csv --out-frame-csv /open_datasets/CASIA-B/metadata/gait_cycle_frames_quality.csv --out-sqq-csv /open_datasets/CASIA-B/metadata/gait_cycles_sqq_windows.csv --report /open_datasets/CASIA-B/metadata/silhouette_quality_report.txt

Gait3D:
python src/13_calculate_silhouette_quality.py --base-dir /open_datasets/Gait3D --sequences-csv /open_datasets/Gait3D/metadata/gait_cycles.csv --frames-csv /open_datasets/Gait3D/metadata/gait_cycle_frames.csv --out-csv /open_datasets/Gait3D/metadata/gait_cycles_quality.csv --out-frame-csv /open_datasets/Gait3D/metadata/gait_cycle_frames_quality.csv --out-sqq-csv /open_datasets/Gait3D/metadata/gait_cycles_sqq_windows.csv --report /open_datasets/Gait3D/metadata/silhouette_quality_report.txt

"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


SIL_PATH_COLUMNS = [
    "silhouette_path",
    "silhouette_source_path",
    "silhouette_output_path",
    "sil_path",
    "sil_source_path",
    "sil_output_path",
]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_json_list(value) -> List[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    nums = re.findall(r"-?\d+", text)
    return [int(x) for x in nums]


def resolve_path(path_value, base_dir: Path) -> Optional[Path]:
    if path_value is None or (isinstance(path_value, float) and np.isnan(path_value)):
        return None
    p = Path(str(path_value))
    if p.is_absolute():
        return p
    return base_dir / p


def find_path_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def numeric_frame_key(path: Path) -> Tuple[int, str]:
    nums = re.findall(r"\d+", path.stem)
    if nums:
        return (int(nums[-1]), path.name)
    return (10**12, path.name)


def load_binary_silhouette(path: Path, threshold: int = 127) -> Optional[np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    _, binary = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY)
    return binary.astype(np.uint8)


def keep_largest_connected_component(binary: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    QAGait-Maximal Connect Area:
    - Se calcula la componente conectada principal.
    - Se retorna una máscara limpiada conservando solo el cuerpo principal.
    - mca_ratio = largest_component_area / total_foreground_area.
    """
    h, w = binary.shape[:2]
    total_area = int(binary.sum())
    stats = {
        "total_area": float(total_area),
        "main_area": 0.0,
        "small_area": 0.0,
        "mca_ratio": 0.0,
        "n_components": 0.0,
    }
    if total_area <= 0:
        return np.zeros_like(binary), stats

    n_labels, labels, cc_stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    # label 0 is background
    if n_labels <= 1:
        return np.zeros_like(binary), stats

    areas = cc_stats[1:, cv2.CC_STAT_AREA]
    largest_idx = int(np.argmax(areas)) + 1
    main_area = int(areas[largest_idx - 1])
    cleaned = (labels == largest_idx).astype(np.uint8)

    stats.update({
        "main_area": float(main_area),
        "small_area": float(total_area - main_area),
        "mca_ratio": float(main_area / max(total_area, 1)),
        "n_components": float(n_labels - 1),
    })
    return cleaned, stats


def silhouette_basic_metrics(binary: np.ndarray) -> Dict[str, float]:
    h, w = binary.shape[:2]
    area = int(binary.sum())
    fill = area / float(max(h * w, 1))

    if area == 0:
        return {
            "sil_area": 0.0,
            "fill": 0.0,
            "border_fg_frac": 0.0,
            "border_contact_sides": 0.0,
            "bbox_h": 0.0,
            "bbox_w": 0.0,
            "bbox_area": 0.0,
            "contour_area": 0.0,
            "hull_fill": 0.0,
            "perimeter_irregularity": 1.0,
        }

    border_pixels = (
        int(binary[0, :].sum())
        + int(binary[-1, :].sum())
        + int(binary[:, 0].sum())
        + int(binary[:, -1].sum())
    )
    border_fg_frac = border_pixels / float(max(area, 1))
    sides = int(binary[0, :].any()) + int(binary[-1, :].any()) + int(binary[:, 0].any()) + int(binary[:, -1].any())

    contours, _ = cv2.findContours((binary * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "sil_area": float(area),
            "fill": float(fill),
            "border_fg_frac": float(border_fg_frac),
            "border_contact_sides": float(sides),
            "bbox_h": 0.0,
            "bbox_w": 0.0,
            "bbox_area": 0.0,
            "contour_area": 0.0,
            "hull_fill": 0.0,
            "perimeter_irregularity": 1.0,
        }

    cnt = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(cnt)
    contour_area = float(cv2.contourArea(cnt))
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    hull_fill = float(contour_area / hull_area) if hull_area > 0 else 0.0

    perimeter = float(cv2.arcLength(cnt, True))
    if perimeter > 0:
        approx = cv2.approxPolyDP(cnt, 0.005 * perimeter, True)
        approx_perimeter = float(cv2.arcLength(approx, True))
        perimeter_irregularity = float(np.clip((perimeter - approx_perimeter) / perimeter, 0, 1))
    else:
        perimeter_irregularity = 1.0

    return {
        "sil_area": float(area),
        "fill": float(fill),
        "border_fg_frac": float(border_fg_frac),
        "border_contact_sides": float(sides),
        "bbox_h": float(bh),
        "bbox_w": float(bw),
        "bbox_area": float(bh * bw),
        "contour_area": float(contour_area),
        "hull_fill": float(hull_fill),
        "perimeter_irregularity": float(perimeter_irregularity),
    }


def min_area_rect_angle(binary: np.ndarray) -> float:
    pts = cv2.findNonZero((binary * 255).astype(np.uint8))
    if pts is None or len(pts) < 5:
        return float("nan")
    rect = cv2.minAreaRect(pts)
    (w, h) = rect[1]
    angle = float(rect[2])
    # OpenCV angle convention. Convert to approximate lean angle from vertical.
    if w < h:
        lean = angle
    else:
        lean = angle + 90.0
    # Normalize to [-90, 90]
    while lean > 90:
        lean -= 180
    while lean < -90:
        lean += 180
    return float(lean)


class TemplateMatcher:
    def __init__(self, template_dir: Optional[Path], threshold: float = 0.001) -> None:
        self.threshold = threshold
        self.templates: List[np.ndarray] = []
        if template_dir:
            self.templates = self._load_templates(template_dir)

    def _load_templates(self, template_dir: Path) -> List[np.ndarray]:
        paths = []
        if template_dir.exists():
            for ext in IMAGE_EXTS:
                paths.extend(template_dir.rglob(f"*{ext}"))
        templates = []
        for p in sorted(paths):
            b = load_binary_silhouette(p)
            if b is None or b.sum() == 0:
                continue
            cleaned, _ = keep_largest_connected_component(b)
            templates.append((cleaned * 255).astype(np.uint8))
        return templates

    @property
    def enabled(self) -> bool:
        return len(self.templates) > 0

    def score(self, binary: np.ndarray) -> Tuple[float, bool]:
        if not self.enabled:
            return float("nan"), True
        img = (binary * 255).astype(np.uint8)
        if img.sum() == 0:
            return float("inf"), False
        best = float("inf")
        for tmpl in self.templates:
            try:
                d = cv2.matchShapes(img, tmpl, cv2.CONTOURS_MATCH_I1, 0.0)
                if d < best:
                    best = float(d)
            except Exception:
                continue
        return best, bool(best <= self.threshold)


def estimate_step_period_from_signal(x: np.ndarray, min_step: int, max_step: int) -> Tuple[float, float]:
    """
    Estima T mediante autocorrelación normalizada de x(t).
    Retorna (T, peak_score). Si no puede estimar, retorna (nan, nan).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < max(min_step * 2, 5):
        return float("nan"), float("nan")

    y = x - np.nanmean(x)
    denom = float(np.dot(y, y))
    if denom <= 1e-12:
        return float("nan"), float("nan")

    max_lag = min(max_step, n - 2)
    min_lag = max(1, min_step)
    if max_lag < min_lag:
        return float("nan"), float("nan")

    scores = []
    for lag in range(min_lag, max_lag + 1):
        a = y[:-lag]
        b = y[lag:]
        d = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
        scores.append(float(np.dot(a, b) / d) if d > 1e-12 else -np.inf)

    idx = int(np.argmax(scores))
    best_lag = min_lag + idx
    best_score = scores[idx]
    if not np.isfinite(best_score):
        return float("nan"), float("nan")
    return float(best_lag), float(best_score)


def normalize_foreground_sum_piecewise(x: np.ndarray, cycle_len: int) -> np.ndarray:
    """
    Liu Eq. (3)-(4), adaptado:
    f(t) = (x(t) - a0) / a2,
    con a0=mean(x) y a2=(max(x)-min(x))/2 dentro de cada pieza de longitud 2T.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return x.copy()

    W = max(2, int(cycle_len))
    f = np.zeros_like(x, dtype=float)

    for start in range(0, n, W):
        end = min(start + W, n)
        seg = x[start:end]
        a0 = float(np.nanmean(seg))
        a2 = float((np.nanmax(seg) - np.nanmin(seg)) / 2.0)
        if not np.isfinite(a2) or a2 <= 1e-12:
            f[start:end] = 0.0
        else:
            f[start:end] = (seg - a0) / a2
    return f


def window_function(name: str, W: int) -> np.ndarray:
    name = name.lower()
    if name in ("rect", "rectangle", "rectangular", "none"):
        return np.ones(W, dtype=float)
    if name == "hamming":
        return np.hamming(W)
    if name == "bartlett":
        return np.bartlett(W)
    if name == "blackman":
        return np.blackman(W)
    raise ValueError(f"Unsupported window function: {name}")


def autocorr_vector(seg: np.ndarray, M: int) -> np.ndarray:
    """
    Liu Eq. (6), autocorrelación de fi(t) para tau=0..M.
    Implementación robusta con denominador W-tau.
    """
    W = len(seg)
    r = np.zeros(M + 1, dtype=float)
    for tau in range(M + 1):
        n = W - tau
        if n <= 0:
            r[tau] = 0.0
        else:
            r[tau] = float(np.dot(seg[:n], seg[tau:tau+n]) / n)
    return r


def toeplitz_from_r(r: np.ndarray) -> np.ndarray:
    idx = np.arange(len(r))
    return r[np.abs(idx[:, None] - idx[None, :])]


def sqq_windows(
    x: np.ndarray,
    step_period: Optional[float],
    cycle_length: Optional[float],
    window_mode: str,
    window_function_name: str,
    stride: int,
    min_window: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Calcula SQQ de Liu:
    Q(i) = sigma_s^2 - sigma_z^2,
    sigma_s^2 = sum(lambda_1..lambda_2),
    sigma_z^2 = sum(lambda_5..lambda_{M+1}).

    También calcula quality_ratio = sigma_s^2 / (sigma_s^2 + sigma_z^2), [0,1],
    útil como variable normalizada para metadata.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < max(4, min_window):
        return pd.DataFrame(), {"sqq_status": "too_short"}

    if cycle_length is not None and np.isfinite(cycle_length) and cycle_length >= min_window:
        W = int(round(cycle_length))
    elif step_period is not None and np.isfinite(step_period):
        W = int(round(2 * step_period))
    else:
        W = n

    W = max(min_window, min(W, n))
    if W < 6:
        return pd.DataFrame(), {"sqq_status": "window_too_short"}

    if window_mode == "full_if_short" and n < W:
        W = n

    f = normalize_foreground_sum_piecewise(x, W)
    win = window_function(window_function_name, W)
    stride = max(1, int(stride))

    rows = []
    starts = list(range(0, max(n - W + 1, 1), stride))
    if not starts:
        starts = [0]
    if starts[-1] != n - W and n - W > 0:
        starts.append(n - W)

    M = max(2, W // 2 - 1)
    for start in starts:
        end = start + W
        if end > n:
            continue
        seg = f[start:end].copy() * win
        if len(seg) < min_window:
            continue

        r = autocorr_vector(seg, M)
        R = toeplitz_from_r(r)
        try:
            eigvals = np.linalg.eigvalsh(R)
            eigvals = np.sort(np.maximum(eigvals, 0.0))[::-1]
        except np.linalg.LinAlgError:
            continue

        # Liu Eq. (9)
        sigma_s2 = float(np.sum(eigvals[:2])) if len(eigvals) >= 2 else float(np.sum(eigvals))
        sigma_z2 = float(np.sum(eigvals[4:])) if len(eigvals) >= 5 else 0.0
        q_raw = float(sigma_s2 - sigma_z2)
        q_ratio = float(sigma_s2 / (sigma_s2 + sigma_z2 + 1e-12))

        rows.append({
            "window_start_index": int(start),
            "window_end_index": int(end - 1),
            "sqq_window_size": int(W),
            "sigma_signal2": sigma_s2,
            "sigma_noise2": sigma_z2,
            "sqq_raw": q_raw,
            "sqq_quality_ratio": q_ratio,
        })

    status = "ok" if rows else "no_valid_windows"
    return pd.DataFrame(rows), {
        "sqq_status": status,
        "sqq_window_size": float(W),
        "sqq_num_windows": float(len(rows)),
    }


def aggregate_numeric(values: Sequence[float], prefix: str) -> Dict[str, float]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
    }


def collect_sequence_frames(
    sequences_df: pd.DataFrame,
    frames_df: Optional[pd.DataFrame],
    base_dir: Path,
    silhouette_base_dir: Optional[Path],
) -> Dict[str, List[Tuple[Optional[int], Path, Optional[pd.Series]]]]:
    """
    Retorna dict sequence_id -> [(frame_number, silhouette_path, frame_row), ...].
    """
    out: Dict[str, List[Tuple[Optional[int], Path, Optional[pd.Series]]]] = {}

    if frames_df is not None and len(frames_df) > 0:
        path_col = find_path_column(frames_df, SIL_PATH_COLUMNS)
        if path_col is None:
            raise ValueError(f"frames_csv no contiene ninguna columna de ruta válida: {SIL_PATH_COLUMNS}")
        if "sequence_id" not in frames_df.columns:
            raise ValueError("frames_csv debe contener la columna sequence_id.")

        for _, fr in frames_df.iterrows():
            sid = str(fr["sequence_id"])
            p = resolve_path(fr[path_col], base_dir)
            if p is None:
                continue
            frame_number = None
            for c in ("frame_number", "frame", "frame_id"):
                if c in fr and pd.notna(fr[c]):
                    try:
                        frame_number = int(fr[c])
                        break
                    except Exception:
                        pass
            out.setdefault(sid, []).append((frame_number, p, fr))

        for sid in out:
            out[sid].sort(key=lambda item: (item[0] if item[0] is not None else numeric_frame_key(item[1])[0], item[1].name))
        return out

    # Fallback desde sequences_csv + silhouette_dir
    if "silhouette_dir" not in sequences_df.columns:
        raise ValueError("Sin frames_csv, sequences_csv debe contener silhouette_dir.")
    base = silhouette_base_dir if silhouette_base_dir is not None else base_dir

    for _, row in sequences_df.iterrows():
        sid = str(row["sequence_id"]) if "sequence_id" in row else str(_)
        seq_dir = resolve_path(row["silhouette_dir"], base)
        if seq_dir is None:
            out[sid] = []
            continue
        paths = []
        for ext in IMAGE_EXTS:
            paths.extend(seq_dir.glob(f"*{ext}"))
        paths = sorted(paths, key=numeric_frame_key)
        out[sid] = [(numeric_frame_key(p)[0], p, None) for p in paths]
    return out


def classify_sequence_quality(
    sil_quality_mean: float,
    sil_valid_ratio: float,
    sil_degraded_ratio: float,
    valid_ratio_threshold: float,
    degraded_ratio_threshold: float,
    quality_ratio_threshold: float,
) -> Tuple[int, str]:
    """
    Etiqueta descriptiva para caracterización del corpus.
    No reemplaza SQQ; resume su interpretación.
    """
    if not np.isfinite(sil_quality_mean) or sil_valid_ratio < valid_ratio_threshold:
        return 0, "no_confiable"
    if sil_degraded_ratio >= degraded_ratio_threshold:
        return 1, "degradada"
    if sil_quality_mean < quality_ratio_threshold:
        return 2, "variable"
    return 3, "confiable"


def process_sequences(args: argparse.Namespace) -> None:
    base_dir = Path(args.base_dir)
    sequences_csv = Path(args.sequences_csv)
    frames_csv = Path(args.frames_csv) if args.frames_csv else None
    silhouette_base_dir = Path(args.silhouettes_dir) if args.silhouettes_dir else None

    seq_df = pd.read_csv(sequences_csv)
    if "sequence_id" not in seq_df.columns:
        raise ValueError("sequences_csv debe contener sequence_id.")

    if args.limit and args.limit > 0:
        seq_df = seq_df.head(args.limit).copy()

    fr_df = pd.read_csv(frames_csv) if frames_csv else None
    seq_frames = collect_sequence_frames(seq_df, fr_df, base_dir, silhouette_base_dir)

    matcher = TemplateMatcher(Path(args.template_dir) if args.template_dir else None, threshold=args.template_match_threshold)

    seq_rows = []
    frame_rows = []
    sqq_rows = []

    for _, row in tqdm(seq_df.iterrows(), total=len(seq_df), desc="Silhouette quality"):
        sid = str(row["sequence_id"])
        items = seq_frames.get(sid, [])
        if args.max_frames_per_seq and args.max_frames_per_seq > 0 and len(items) > args.max_frames_per_seq:
            idxs = np.linspace(0, len(items) - 1, args.max_frames_per_seq, dtype=int)
            items_proc = [items[i] for i in idxs]
        else:
            items_proc = items

        frame_metrics = []
        foreground_sum = []
        valid_cleaned_count = 0
        sampled_frame_numbers = []

        for pos, (frame_num, sil_path, frame_row) in enumerate(items_proc):
            binary = load_binary_silhouette(sil_path, threshold=args.binary_threshold)
            base_frame = {
                "sequence_id": sid,
                "position_in_sequence": pos,
                "frame_number": frame_num if frame_num is not None else np.nan,
                "silhouette_path": str(sil_path),
                "silhouette_status": "ok",
            }
            if binary is None:
                base_frame["silhouette_status"] = "read_error"
                frame_rows.append(base_frame)
                continue

            cleaned, mca = keep_largest_connected_component(binary)
            metrics_raw = silhouette_basic_metrics(binary)
            metrics_clean = silhouette_basic_metrics(cleaned)

            mca_valid = bool(
                mca["total_area"] >= args.min_foreground_area
                and mca["main_area"] >= args.min_main_area
                and mca["mca_ratio"] >= args.max_connected_ratio_threshold
            )

            if mca_valid:
                tm_dist, tm_valid = matcher.score(cleaned)
            else:
                tm_dist, tm_valid = float("nan"), False

            # Si no hay template_dir, tm_valid=True por diseño para no bloquear.
            frame_valid = bool(mca_valid and tm_valid)
            if frame_valid:
                valid_cleaned_count += 1

            fs = float(metrics_clean["sil_area"]) if mca_valid else float(metrics_raw["sil_area"])
            foreground_sum.append(fs)

            lean_angle = min_area_rect_angle(cleaned if mca_valid else binary)

            if frame_num is not None:
                sampled_frame_numbers.append(int(frame_num))

            fr_out = {
                **base_frame,
                "silhouette_status": "valid" if frame_valid else "invalid",
                "foreground_sum": fs,
                "raw_area": metrics_raw["sil_area"],
                "main_area": mca["main_area"],
                "small_area": mca["small_area"],
                "mca_ratio": mca["mca_ratio"],
                "n_components": mca["n_components"],
                "mca_valid": int(mca_valid),
                "template_match_distance": tm_dist,
                "template_match_valid": int(tm_valid) if np.isfinite(tm_dist) or matcher.enabled else int(tm_valid),
                "frame_valid": int(frame_valid),
                "fill": metrics_clean["fill"],
                "border_fg_frac": metrics_clean["border_fg_frac"],
                "border_contact_sides": metrics_clean["border_contact_sides"],
                "bbox_h": metrics_clean["bbox_h"],
                "bbox_w": metrics_clean["bbox_w"],
                "bbox_area": metrics_clean["bbox_area"],
                "hull_fill": metrics_clean["hull_fill"],
                "perimeter_irregularity": metrics_clean["perimeter_irregularity"],
                "lean_angle_deg": lean_angle,
            }
            frame_rows.append(fr_out)
            frame_metrics.append(fr_out)

        n_frames = len(items_proc)
        if n_frames == 0 or len(foreground_sum) == 0:
            seq_out = row.to_dict()
            seq_out.update({
                "sil_quality_method": "liu_sqq_plus_qagait_mca_template",
                "silhouette_processing_status": "no_silhouettes",
                "n_samples_sil": 0,
                "sil_valid_ratio": 0.0,
                "sil_quality_mean": np.nan,
                "sil_quality_std": np.nan,
                "sil_degraded_ratio": np.nan,
            })
            seq_rows.append(seq_out)
            continue

        x = np.asarray(foreground_sum, dtype=float)

        # Si ya existe estimated_step_period o estimated_cycle_length desde el script de ciclos, se reutiliza.
        step_period = None
        cycle_length = None
        if "estimated_step_period" in row and pd.notna(row["estimated_step_period"]):
            try:
                step_period = float(row["estimated_step_period"])
            except Exception:
                step_period = None
        if "estimated_cycle_length" in row and pd.notna(row["estimated_cycle_length"]):
            try:
                cycle_length = float(row["estimated_cycle_length"])
            except Exception:
                cycle_length = None

        estimated_step_period = step_period
        period_score = np.nan
        if estimated_step_period is None or not np.isfinite(estimated_step_period):
            estimated_step_period, period_score = estimate_step_period_from_signal(
                x, args.min_step_frames, args.max_step_frames
            )

        if cycle_length is None or not np.isfinite(cycle_length):
            if estimated_step_period is not None and np.isfinite(estimated_step_period):
                cycle_length = 2.0 * float(estimated_step_period)
            else:
                cycle_length = float(len(x))

        q_df, q_status = sqq_windows(
            x=x,
            step_period=estimated_step_period,
            cycle_length=cycle_length,
            window_mode=args.window_mode,
            window_function_name=args.window_function,
            stride=args.sqq_stride,
            min_window=args.min_sqq_window,
        )

        q_raw_values = q_df["sqq_raw"].to_numpy(dtype=float) if len(q_df) else np.asarray([], dtype=float)
        q_ratio_values = q_df["sqq_quality_ratio"].to_numpy(dtype=float) if len(q_df) else np.asarray([], dtype=float)

        sil_quality_mean = float(np.nanmean(q_ratio_values)) if q_ratio_values.size else np.nan
        sil_quality_std = float(np.nanstd(q_ratio_values, ddof=1)) if q_ratio_values.size > 1 else (0.0 if q_ratio_values.size == 1 else np.nan)
        sil_degraded_ratio = float(np.mean(q_ratio_values < args.degraded_quality_threshold)) if q_ratio_values.size else np.nan

        valid_ratio = float(valid_cleaned_count / max(n_frames, 1))
        quality_level, quality_label = classify_sequence_quality(
            sil_quality_mean=sil_quality_mean,
            sil_valid_ratio=valid_ratio,
            sil_degraded_ratio=sil_degraded_ratio if np.isfinite(sil_degraded_ratio) else 1.0,
            valid_ratio_threshold=args.sequence_valid_ratio_threshold,
            degraded_ratio_threshold=args.sequence_degraded_ratio_threshold,
            quality_ratio_threshold=args.sequence_quality_ratio_threshold,
        )

        fm = pd.DataFrame(frame_metrics)
        seq_out = row.to_dict()
        seq_out.update({
            "sil_quality_method": "liu_sqq_plus_qagait_mca_template",
            "silhouette_processing_status": "ok",
            "n_samples_sil": int(n_frames),
            "sampled_frame_numbers_json": json.dumps(sampled_frame_numbers, ensure_ascii=False),
            "sil_area_mean": float(fm["raw_area"].mean()) if "raw_area" in fm else np.nan,
            "sil_area_std": float(fm["raw_area"].std(ddof=1)) if len(fm) > 1 else 0.0,
            "fill_mean": float(fm["fill"].mean()) if "fill" in fm else np.nan,
            "fill_median": float(fm["fill"].median()) if "fill" in fm else np.nan,
            "border_frac": float(fm["border_fg_frac"].mean()) if "border_fg_frac" in fm else np.nan,
            "border_contact_sides_mean": float(fm["border_contact_sides"].mean()) if "border_contact_sides" in fm else np.nan,
            "bbox_h_median": float(fm["bbox_h"].median()) if "bbox_h" in fm else np.nan,
            "bbox_w_median": float(fm["bbox_w"].median()) if "bbox_w" in fm else np.nan,
            "bbox_area_mean": float(fm["bbox_area"].mean()) if "bbox_area" in fm else np.nan,
            "mca_ratio_mean": float(fm["mca_ratio"].mean()) if "mca_ratio" in fm else np.nan,
            "mca_valid_ratio": float(fm["mca_valid"].mean()) if "mca_valid" in fm else np.nan,
            "template_match_enabled": int(matcher.enabled),
            "template_match_distance_mean": float(fm["template_match_distance"].replace([np.inf, -np.inf], np.nan).mean()) if "template_match_distance" in fm else np.nan,
            "template_valid_ratio": float(fm["template_match_valid"].mean()) if "template_match_valid" in fm else np.nan,
            "sil_valid_ratio": valid_ratio,
            "lean_angle_mean": float(fm["lean_angle_deg"].replace([np.inf, -np.inf], np.nan).mean()) if "lean_angle_deg" in fm else np.nan,
            "lean_angle_std": float(fm["lean_angle_deg"].replace([np.inf, -np.inf], np.nan).std(ddof=1)) if len(fm) > 1 else 0.0,
            "foreground_sum_mean": float(np.mean(x)),
            "foreground_sum_std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
            "estimated_step_period_sil": float(estimated_step_period) if estimated_step_period is not None and np.isfinite(estimated_step_period) else np.nan,
            "estimated_cycle_length_sil": float(cycle_length) if cycle_length is not None and np.isfinite(cycle_length) else np.nan,
            "periodicity_score_sil": float(period_score) if np.isfinite(period_score) else np.nan,
            "sqq_status": q_status.get("sqq_status", "unknown"),
            "sqq_window_size": q_status.get("sqq_window_size", np.nan),
            "n_sqq_windows": int(q_status.get("sqq_num_windows", 0)),
            "sqq_raw_mean": float(np.nanmean(q_raw_values)) if q_raw_values.size else np.nan,
            "sqq_raw_std": float(np.nanstd(q_raw_values, ddof=1)) if q_raw_values.size > 1 else (0.0 if q_raw_values.size == 1 else np.nan),
            "sqq_raw_min": float(np.nanmin(q_raw_values)) if q_raw_values.size else np.nan,
            "sqq_raw_max": float(np.nanmax(q_raw_values)) if q_raw_values.size else np.nan,
            # Variable normalizada recomendada para metadata/Gower:
            "sil_quality_mean": sil_quality_mean,
            "sil_quality_std": sil_quality_std,
            "sil_degraded_ratio": sil_degraded_ratio,
            "sil_quality_level": int(quality_level),
            "sil_quality_label": quality_label,
            "max_connected_ratio_threshold": float(args.max_connected_ratio_threshold),
            "template_match_threshold": float(args.template_match_threshold),
            "degraded_quality_threshold": float(args.degraded_quality_threshold),
        })
        seq_rows.append(seq_out)

        if len(q_df):
            q_df.insert(0, "sequence_id", sid)
            sqq_rows.extend(q_df.to_dict("records"))

    out_csv = Path(args.out_csv)
    out_frame_csv = Path(args.out_frame_csv)
    out_sqq_csv = Path(args.out_sqq_csv)

    ensure_parent(out_csv)
    pd.DataFrame(seq_rows).to_csv(out_csv, index=False)

    ensure_parent(out_frame_csv)
    pd.DataFrame(frame_rows).to_csv(out_frame_csv, index=False)

    ensure_parent(out_sqq_csv)
    pd.DataFrame(sqq_rows).to_csv(out_sqq_csv, index=False)

    if args.report:
        report = Path(args.report)
        ensure_parent(report)
        sdf = pd.DataFrame(seq_rows)
        with report.open("w", encoding="utf-8") as f:
            f.write("REPORTE DE CALIDAD DE SILUETA\n")
            f.write("=============================\n")
            f.write("Metodo: Liu SQQ + controles morfologicos QAGait\n")
            f.write(f"Secuencias procesadas: {len(sdf)}\n")
            if "sil_quality_mean" in sdf:
                f.write(f"sil_quality_mean global: {sdf['sil_quality_mean'].mean():.6f}\n")
            if "sil_valid_ratio" in sdf:
                f.write(f"sil_valid_ratio global: {sdf['sil_valid_ratio'].mean():.6f}\n")
            if "sil_degraded_ratio" in sdf:
                f.write(f"sil_degraded_ratio global: {sdf['sil_degraded_ratio'].mean():.6f}\n")
            f.write("\nParametros:\n")
            for k, v in vars(args).items():
                f.write(f"{k}: {v}\n")

    print(f"OK: metadata por secuencia -> {out_csv}")
    print(f"OK: metadata por frame -> {out_frame_csv}")
    print(f"OK: ventanas SQQ -> {out_sqq_csv}")
    if args.report:
        print(f"OK: reporte -> {args.report}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Calidad de silueta basada en Liu SQQ y QAGait.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base-dir", "--base_dir", dest="base_dir", required=True,
                   help="Directorio base para resolver rutas relativas.")
    p.add_argument("--sequences-csv", "--sequences_csv", dest="sequences_csv", required=True,
                   help="CSV con una fila por secuencia/ciclo.")
    p.add_argument("--frames-csv", "--frames_csv", dest="frames_csv", default=None,
                   help="CSV con una fila por frame. Recomendado.")
    p.add_argument("--silhouettes-dir", "--silhouettes_dir", dest="silhouettes_dir", default=None,
                   help="Directorio base alternativo para silhouette_dir si no se usa frames_csv.")
    p.add_argument("--out-csv", "--out_csv", dest="out_csv", required=True,
                   help="CSV de salida por secuencia.")
    p.add_argument("--out-frame-csv", "--out_frame_csv", dest="out_frame_csv", required=True,
                   help="CSV de salida por frame.")
    p.add_argument("--out-sqq-csv", "--out_sqq_csv", dest="out_sqq_csv", required=True,
                   help="CSV de salida por ventana SQQ.")
    p.add_argument("--report", default=None, help="Reporte TXT opcional.")
    p.add_argument("--limit", type=int, default=None, help="Procesar solo las primeras N secuencias.")

    # Binarización / QAGait
    p.add_argument("--binary-threshold", "--binary_threshold", dest="binary_threshold", type=int, default=127)
    p.add_argument("--max-connected-ratio-threshold", "--max_connected_ratio_threshold",
                   dest="max_connected_ratio_threshold", type=float, default=0.95,
                   help="Umbral QAGait tipo epsilon: componente principal / foreground total.")
    p.add_argument("--min-foreground-area", "--min_foreground_area", dest="min_foreground_area",
                   type=int, default=10, help="Área mínima de foreground para aceptar una silueta.")
    p.add_argument("--min-main-area", "--min_main_area", dest="min_main_area",
                   type=int, default=10, help="Área mínima de la componente principal.")
    p.add_argument("--template-dir", "--template_dir", dest="template_dir", default=None,
                   help="Directorio opcional con plantillas binarias humanas para Template Match.")
    p.add_argument("--template-match-threshold", "--template_match_threshold",
                   dest="template_match_threshold", type=float, default=0.001,
                   help="Umbral tau de QAGait para cv2.matchShapes. Solo se aplica si hay template_dir.")

    # Liu SQQ
    p.add_argument("--max-frames-per-seq", "--max_frames_per_seq", dest="max_frames_per_seq",
                   type=int, default=0, help="0 = usar todos los frames; >0 = muestreo equidistante.")
    p.add_argument("--min-step-frames", "--min_step_frames", dest="min_step_frames",
                   type=int, default=4, help="Lag mínimo para estimar periodo de paso si no viene en metadata.")
    p.add_argument("--max-step-frames", "--max_step_frames", dest="max_step_frames",
                   type=int, default=40, help="Lag máximo para estimar periodo de paso si no viene en metadata.")
    p.add_argument("--window-function", "--window_function", dest="window_function",
                   default="hamming", choices=["rect", "rectangle", "rectangular", "hamming", "bartlett", "blackman"])
    p.add_argument("--window-mode", "--window_mode", dest="window_mode",
                   default="full_if_short", choices=["full_if_short"],
                   help="Modo para secuencias cortas.")
    p.add_argument("--sqq-stride", "--sqq_stride", dest="sqq_stride",
                   type=int, default=1, help="Desplazamiento de la ventana SQQ.")
    p.add_argument("--min-sqq-window", "--min_sqq_window", dest="min_sqq_window",
                   type=int, default=6, help="Longitud mínima de ventana para calcular SQQ.")

    # Etiquetado descriptivo
    p.add_argument("--degraded-quality-threshold", "--degraded_quality_threshold",
                   dest="degraded_quality_threshold", type=float, default=0.50,
                   help="Ventana degradada si sqq_quality_ratio cae debajo de este umbral.")
    p.add_argument("--sequence-valid-ratio-threshold", "--sequence_valid_ratio_threshold",
                   dest="sequence_valid_ratio_threshold", type=float, default=0.80,
                   help="Secuencia no confiable si sil_valid_ratio cae debajo de este umbral.")
    p.add_argument("--sequence-degraded-ratio-threshold", "--sequence_degraded_ratio_threshold",
                   dest="sequence_degraded_ratio_threshold", type=float, default=0.30,
                   help="Secuencia degradada si la proporción de ventanas degradadas supera este umbral.")
    p.add_argument("--sequence-quality-ratio-threshold", "--sequence_quality_ratio_threshold",
                   dest="sequence_quality_ratio_threshold", type=float, default=0.65,
                   help="Secuencia variable si sil_quality_mean cae debajo de este umbral.")

    return p


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    process_sequences(args)
