#!/usr/bin/env python3
# python src\9a_generate_dataset.py --crops-root transforms\gt_crops\crops --silhouettes-root transforms\silhouettes\masks --gt-root transforms\gt_crops\gt --out-root dataset\output_test --scene S01 --cam c001 --person-id 0 --limit-cycles 20 --materialize copy --overwrite --max-frame-gap 6 --min-periodicity-score 0.22 --min-cycle-continuity 0.65 --min-cycle-frames 8 --max-cycle-frames 100 --allow-period-window-fallback
"""
Genera ciclos de marcha NO solapados a partir de crops RGB identificados por
persona global, escena y cámara, usando siluetas binarias para detectar la
periodicidad de la marcha.

Entrada esperada de crops RGB:
  crops_root/crops/IDxxxx/S0x/c00x/*.jpg
  o bien:
  crops_root/IDxxxx/S0x/c00x/*.jpg

Entrada requerida de siluetas:
  silhouettes_root/IDxxxx/S0x/c00x/*.(png|jpg|...)
  o:
  silhouettes_root/masks/IDxxxx/S0x/c00x/*.(png|jpg|...)

Entrada opcional de GT generado por el script de crops:
  gt_root/S0x/c00x/gt/gt.txt
con filas:
  frame,global_id,x,y,w,h,...

Salida:
  out_root/rgb/IDxxxx/S0x/c00x/cycle0001/*
  out_root/silhouette/IDxxxx/S0x/c00x/cycle0001/*
  out_root/metadata/gait_cycles.csv
  out_root/metadata/gait_cycle_frames.csv
  out_root/metadata/skipped_tracklets.csv

También se escriben aliases de compatibilidad:
  out_root/metadata/sequences.csv
  out_root/metadata/sequence_frames.csv

Este script NO calcula iluminación, ropa, color, SQQ, Gower ni métricas
intra-clase/inter-clase. Esos cálculos deben ejecutarse aparte.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class FrameItem:
    frame_number: int
    rgb_path: Path
    silhouette_path: Path
    bbox: Optional[Tuple[int, int, int, int]]  # x, y, w, h


# =========================================================
# Normalización y parsing
# =========================================================

def normalize_scene(scene_raw: str) -> str:
    s = str(scene_raw).strip()
    m = re.search(r"(\d+)", s)
    if not m:
        raise ValueError(f"No pude extraer número de escena desde: {scene_raw!r}")
    return f"S{int(m.group(1)):02d}"


def normalize_cam(cam_raw: str) -> str:
    s = str(cam_raw).strip()
    m = re.search(r"(\d+)", s)
    if not m:
        raise ValueError(f"No pude extraer número de cámara desde: {cam_raw!r}")
    return f"c{int(m.group(1)):03d}"


def parse_person_id(person_dir_name: str) -> int:
    m = re.fullmatch(r"ID(\d+)", person_dir_name, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Nombre de carpeta de persona inválido: {person_dir_name!r}")
    return int(m.group(1))


def extract_frame_number_from_name(filename: str) -> Optional[int]:
    """
    Extrae el frame desde nombres como:
      c001_000036_True.jpg
      000036.png
      frame_36_mask.png

    Si hay varios grupos numéricos, selecciona el más largo; en empate,
    selecciona el último. Esto evita confundir cámara con frame.
    """
    matches = list(re.finditer(r"\d+", Path(filename).stem))
    if not matches:
        return None
    _, best = max(enumerate(matches), key=lambda item: (len(item[1].group()), item[0]))
    return int(best.group())


def list_image_files(directory: Path) -> List[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def index_files_by_frame(files: Iterable[Path], context: str) -> Dict[int, Path]:
    indexed: Dict[int, Path] = {}
    for path in files:
        frame_number = extract_frame_number_from_name(path.name)
        if frame_number is None:
            print(f"[WARN] No pude extraer frame desde {path}; se omite.", file=sys.stderr)
            continue
        if frame_number in indexed:
            raise ValueError(
                f"Frames duplicados en {context}: frame={frame_number}, "
                f"archivos={indexed[frame_number]} y {path}"
            )
        indexed[frame_number] = path
    return indexed


# =========================================================
# GT / Bounding boxes
# =========================================================

def load_gt_lookup(
    gt_root: Optional[Path],
    scene: str,
    cam: str,
) -> Dict[Tuple[int, int], Tuple[int, int, int, int]]:
    """
    Carga GT con bounding boxes ya usados para recortar crops:
      frame,gid,x,y,w,h,...
    """
    if gt_root is None:
        return {}

    gt_path = gt_root / scene / cam / "gt" / "gt.txt"
    if not gt_path.is_file():
        print(f"[WARN] No existe GT para {scene}/{cam}: {gt_path}", file=sys.stderr)
        return {}

    lookup: Dict[Tuple[int, int], Tuple[int, int, int, int]] = {}
    with gt_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for line_number, row in enumerate(reader, start=1):
            if not row or len(row) < 6:
                continue
            try:
                frame = int(float(row[0]))
                gid = int(float(row[1]))
                x = int(float(row[2]))
                y = int(float(row[3]))
                w = int(float(row[4]))
                h = int(float(row[5]))
            except ValueError as exc:
                raise ValueError(f"GT inválido en {gt_path}, línea {line_number}: {row}") from exc

            key = (gid, frame)
            if key in lookup:
                raise ValueError(f"GT duplicado en {gt_path}: gid={gid}, frame={frame}")
            lookup[key] = (x, y, w, h)
    return lookup


# =========================================================
# Descubrimiento de entradas
# =========================================================

def resolve_crops_base(crops_root: Path) -> Path:
    nested = crops_root / "crops"
    return nested if nested.is_dir() else crops_root


def iter_person_scene_cam(crops_base: Path) -> Iterable[Tuple[int, str, str, Path]]:
    for person_dir in sorted(
        p for p in crops_base.iterdir()
        if p.is_dir() and re.fullmatch(r"ID\d+", p.name, re.I)
    ):
        gid = parse_person_id(person_dir.name)
        for scene_dir in sorted(p for p in person_dir.iterdir() if p.is_dir()):
            scene = normalize_scene(scene_dir.name)
            for cam_dir in sorted(p for p in scene_dir.iterdir() if p.is_dir()):
                cam = normalize_cam(cam_dir.name)
                yield gid, scene, cam, cam_dir


def resolve_silhouette_dir(silhouettes_root: Path, gid: int, scene: str, cam: str) -> Path:
    candidates = [
        silhouettes_root / f"ID{gid:04d}" / scene / cam,
        silhouettes_root / "masks" / f"ID{gid:04d}" / scene / cam,
        silhouettes_root / "silhouette" / f"ID{gid:04d}" / scene / cam,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def build_frame_items(
    gid: int,
    scene: str,
    cam: str,
    rgb_dir: Path,
    silhouettes_root: Path,
    gt_lookup: Dict[Tuple[int, int], Tuple[int, int, int, int]],
) -> List[FrameItem]:
    rgb_by_frame = index_files_by_frame(list_image_files(rgb_dir), context=f"RGB {rgb_dir}")
    if not rgb_by_frame:
        return []

    sil_dir = resolve_silhouette_dir(silhouettes_root, gid, scene, cam)
    sil_by_frame = index_files_by_frame(list_image_files(sil_dir), context=f"siluetas {sil_dir}")
    if not sil_by_frame:
        print(f"[WARN] No hay siluetas para ID{gid:04d}/{scene}/{cam}; se omite.", file=sys.stderr)
        return []

    common_frames = sorted(set(rgb_by_frame).intersection(sil_by_frame))
    if not common_frames:
        return []

    return [
        FrameItem(
            frame_number=frame,
            rgb_path=rgb_by_frame[frame],
            silhouette_path=sil_by_frame[frame],
            bbox=gt_lookup.get((gid, frame)),
        )
        for frame in common_frames
    ]


# =========================================================
# Tracklets y detección de ciclos
# =========================================================

def split_into_tracklets(items: Sequence[FrameItem], max_frame_gap: int) -> List[List[FrameItem]]:
    if max_frame_gap < 1:
        raise ValueError("max_frame_gap debe ser >= 1")
    if not items:
        return []

    ordered = sorted(items, key=lambda item: item.frame_number)
    tracklets: List[List[FrameItem]] = [[ordered[0]]]
    for item in ordered[1:]:
        previous = tracklets[-1][-1]
        if (item.frame_number - previous.frame_number) > max_frame_gap:
            tracklets.append([item])
        else:
            tracklets[-1].append(item)
    return tracklets


def read_foreground_sum(mask_path: Path) -> float:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"No pude leer silueta: {mask_path}")
    return float(np.count_nonzero(mask > 0))


def moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return signal.astype(float, copy=True)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(signal.astype(float), (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def normalized_autocorrelation(signal: np.ndarray) -> np.ndarray:
    x = signal.astype(float)
    x = x - np.mean(x)
    denom = float(np.dot(x, x))
    if denom <= 1e-12:
        return np.zeros(len(x), dtype=float)
    corr = np.correlate(x, x, mode="full")[len(x)-1:]
    return corr / denom


def estimate_step_period(
    signal: np.ndarray,
    min_step_frames: int,
    max_step_frames: int,
) -> Tuple[Optional[int], float, np.ndarray]:
    ac = normalized_autocorrelation(signal)
    if len(ac) <= min_step_frames:
        return None, 0.0, ac

    lo = max(1, min_step_frames)
    hi = min(max_step_frames, len(ac) - 1)
    if lo > hi:
        return None, 0.0, ac

    window = ac[lo:hi + 1]
    rel = int(np.argmax(window))
    period = lo + rel
    score = float(ac[period])
    return period, score, ac


def local_maxima(signal: np.ndarray) -> List[int]:
    if len(signal) < 3:
        return []
    return [
        i for i in range(1, len(signal) - 1)
        if signal[i] >= signal[i - 1] and signal[i] > signal[i + 1]
    ]


def best_phase_peak(
    signal: np.ndarray,
    maxima: Sequence[int],
    target: int,
    radius: int,
    lower: int,
    upper: int,
) -> Optional[int]:
    candidates = [
        idx for idx in maxima
        if lower <= idx <= upper and abs(idx - target) <= radius
    ]
    if not candidates:
        left = max(lower, target - radius)
        right = min(upper, target + radius)
        if left > right:
            return None
        return int(left + np.argmax(signal[left:right + 1]))
    return max(candidates, key=lambda idx: signal[idx])


def phase_boundaries(
    signal: np.ndarray,
    expected_cycle_length: int,
    phase_search_ratio: float,
) -> List[int]:
    """
    Busca máximos de fase equivalente separados aproximadamente por la longitud
    esperada del ciclo. Los intervalos consecutivos [b_i, b_{i+1}) no se solapan.
    """
    if expected_cycle_length < 2 or len(signal) < expected_cycle_length + 1:
        return []

    maxima = local_maxima(signal)
    if not maxima:
        maxima = list(range(len(signal)))

    radius = max(1, int(round(expected_cycle_length * phase_search_ratio)))
    seed_upper = min(len(signal) - 1, expected_cycle_length)
    seed_candidates = [idx for idx in maxima if 0 <= idx <= seed_upper]
    seed = (
        max(seed_candidates, key=lambda idx: signal[idx])
        if seed_candidates
        else int(np.argmax(signal[:seed_upper + 1]))
    )

    forward = [seed]
    while True:
        target = forward[-1] + expected_cycle_length
        if target - radius >= len(signal):
            break
        peak = best_phase_peak(
            signal, maxima, target, radius,
            lower=forward[-1] + 1,
            upper=len(signal) - 1,
        )
        if peak is None or peak <= forward[-1]:
            break
        forward.append(peak)

    backward: List[int] = []
    current = seed
    while True:
        target = current - expected_cycle_length
        if target + radius < 0:
            break
        peak = best_phase_peak(
            signal, maxima, target, radius,
            lower=0,
            upper=current - 1,
        )
        if peak is None or peak >= current:
            break
        backward.append(peak)
        current = peak

    return sorted(set(reversed(backward)) | set(forward))


def temporal_stats(frame_numbers: Sequence[int]) -> Dict[str, float | int]:
    if not frame_numbers:
        raise ValueError("No se pueden calcular estadísticas temporales sin frames")
    ordered = sorted(frame_numbers)
    gaps = [b - a for a, b in zip(ordered, ordered[1:])]
    frame_start = ordered[0]
    frame_end = ordered[-1]
    expected_count = frame_end - frame_start + 1
    missing_count = max(0, expected_count - len(ordered))
    continuity_ratio = len(ordered) / expected_count if expected_count > 0 else 0.0
    return {
        "frame_start": frame_start,
        "frame_end": frame_end,
        "frame_gap_max": max(gaps) if gaps else 0,
        "frame_gap_mean": float(mean(gaps)) if gaps else 0.0,
        "missing_frame_count": missing_count,
        "temporal_continuity_ratio": float(continuity_ratio),
    }


def bbox_stats(items: Sequence[FrameItem]) -> Dict[str, Optional[float]]:
    bboxes = [item.bbox for item in items if item.bbox is not None]
    if not bboxes:
        return {
            "bbox_h_median": None,
            "bbox_w_median": None,
            "bbox_area_mean": None,
            "bbox_available_ratio": 0.0,
        }
    heights = [bbox[3] for bbox in bboxes]
    widths = [bbox[2] for bbox in bboxes]
    areas = [bbox[2] * bbox[3] for bbox in bboxes]
    return {
        "bbox_h_median": float(median(heights)),
        "bbox_w_median": float(median(widths)),
        "bbox_area_mean": float(mean(areas)),
        "bbox_available_ratio": len(bboxes) / len(items),
    }


def cycles_from_tracklet(
    tracklet: Sequence[FrameItem],
    *,
    min_step_frames: int,
    max_step_frames: int,
    cycle_multiplier: float,
    smooth_window: int,
    phase_search_ratio: float,
    min_periodicity_score: float,
    min_cycle_frames: int,
    max_cycle_frames: int,
    min_cycle_continuity: float,
    allow_period_window_fallback: bool,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """
    Devuelve ciclos válidos y diagnóstico del tracklet.
    Cada ciclo contiene items y señales foreground-sum.
    """
    raw_signal = np.array([read_foreground_sum(item.silhouette_path) for item in tracklet], dtype=float)
    smooth_signal = moving_average(raw_signal, smooth_window)

    step_period, step_score, ac = estimate_step_period(
        smooth_signal,
        min_step_frames=min_step_frames,
        max_step_frames=max_step_frames,
    )

    diagnostic: Dict[str, object] = {
        "tracklet_num_frames": len(tracklet),
        "estimated_step_period": step_period,
        "estimated_cycle_length": None,
        "autocorrelation_step_peak": step_score,
        "autocorrelation_cycle_peak": None,
        "periodicity_score": step_score,
        "foreground_sum_mean": float(np.mean(raw_signal)) if len(raw_signal) else None,
        "foreground_sum_std": float(np.std(raw_signal, ddof=1)) if len(raw_signal) > 1 else 0.0,
        "skip_reason": "",
    }

    if step_period is None:
        diagnostic["skip_reason"] = "period_not_detected"
        return [], diagnostic

    expected_cycle = int(round(step_period * cycle_multiplier))
    diagnostic["estimated_cycle_length"] = expected_cycle
    if expected_cycle < min_cycle_frames or expected_cycle > max_cycle_frames:
        diagnostic["skip_reason"] = "estimated_cycle_length_out_of_range"
        return [], diagnostic

    if expected_cycle < len(ac):
        diagnostic["autocorrelation_cycle_peak"] = float(ac[expected_cycle])

    if step_score < min_periodicity_score:
        diagnostic["skip_reason"] = "low_periodicity"
        return [], diagnostic

    boundaries = phase_boundaries(
        smooth_signal,
        expected_cycle_length=expected_cycle,
        phase_search_ratio=phase_search_ratio,
    )

    cycles: List[Dict[str, object]] = []
    method = "foreground_sum_autocorrelation_phase_peaks"

    for start_idx, end_idx in zip(boundaries, boundaries[1:]):
        if end_idx <= start_idx:
            continue
        items = list(tracklet[start_idx:end_idx])  # fin exclusivo: sin overlap
        if not (min_cycle_frames <= len(items) <= max_cycle_frames):
            continue

        tstats = temporal_stats([item.frame_number for item in items])
        if float(tstats["temporal_continuity_ratio"]) < min_cycle_continuity:
            continue

        cycles.append({
            "items": items,
            "raw_signal": raw_signal[start_idx:end_idx],
            "smooth_signal": smooth_signal[start_idx:end_idx],
            "cycle_detection_method": method,
            "cycle_quality_flag": "valid_phase_peaks",
            "cycle_start_index": start_idx,
            "cycle_end_index_exclusive": end_idx,
            "cycle_length_error": len(items) - expected_cycle,
        })

    if cycles:
        return cycles, diagnostic

    if not allow_period_window_fallback:
        diagnostic["skip_reason"] = "no_valid_cycles_after_phase_validation"
        return [], diagnostic

    # Fallback: ventanas no solapadas alineadas con la mejor fase inicial.
    # Solo se usa cuando el usuario lo habilita explícitamente.
    seed_boundaries = phase_boundaries(
        smooth_signal,
        expected_cycle_length=expected_cycle,
        phase_search_ratio=phase_search_ratio,
    )
    start = seed_boundaries[0] if seed_boundaries else 0
    while start + expected_cycle <= len(tracklet):
        end = start + expected_cycle
        items = list(tracklet[start:end])
        tstats = temporal_stats([item.frame_number for item in items])
        if float(tstats["temporal_continuity_ratio"]) >= min_cycle_continuity:
            cycles.append({
                "items": items,
                "raw_signal": raw_signal[start:end],
                "smooth_signal": smooth_signal[start:end],
                "cycle_detection_method": "foreground_sum_autocorrelation_period_window_fallback",
                "cycle_quality_flag": "valid_period_window_fallback",
                "cycle_start_index": start,
                "cycle_end_index_exclusive": end,
                "cycle_length_error": len(items) - expected_cycle,
            })
        start = end

    if not cycles:
        diagnostic["skip_reason"] = "no_valid_cycles_after_fallback"
    return cycles, diagnostic


# =========================================================
# Materialización
# =========================================================

def materialize_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "none":
        return
    else:
        raise ValueError(f"Modo desconocido: {mode}")


def reset_output_root(out_root: Path, overwrite: bool) -> None:
    generated_dirs = [out_root / "rgb", out_root / "silhouette", out_root / "metadata"]
    existing = [p for p in generated_dirs if p.exists()]
    if existing and not overwrite:
        joined = "\n  - ".join(str(p) for p in existing)
        raise FileExistsError(
            "Ya existen salidas generadas. Use --overwrite para regenerar:\n"
            f"  - {joined}"
        )
    if overwrite:
        for path in generated_dirs:
            if path.exists():
                shutil.rmtree(path)


# =========================================================
# Main
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera ciclos reales de marcha NO solapados usando foreground-sum y autocorrelación."
    )
    parser.add_argument("--crops-root", required=True, help="Raíz con crops/IDxxxx/S0x/c00x o directamente IDxxxx/S0x/c00x")
    parser.add_argument("--silhouettes-root", required=True, help="Raíz de siluetas por ID/escena/cámara")
    parser.add_argument("--gt-root", default=None, help="Raíz opcional GT: S0x/c00x/gt/gt.txt")
    parser.add_argument("--out-root", required=True, help="Raíz de salida")

    parser.add_argument("--max-frame-gap", type=int, default=2, help="Salto máximo antes de cortar un tracklet")
    parser.add_argument("--min-step-frames", type=int, default=6, help="Mínimo período de paso esperado")
    parser.add_argument("--max-step-frames", type=int, default=30, help="Máximo período de paso esperado")
    parser.add_argument("--cycle-multiplier", type=float, default=2.0, help="Longitud ciclo completo ~ período de paso x multiplicador")
    parser.add_argument("--smooth-window", type=int, default=5, help="Ventana impar de suavizado de foreground-sum")
    parser.add_argument("--phase-search-ratio", type=float, default=0.25, help="Radio de búsqueda de máximos equivalentes respecto al ciclo esperado")
    parser.add_argument("--min-periodicity-score", type=float, default=0.35, help="Autocorrelación mínima para aceptar periodicidad")
    parser.add_argument("--min-cycle-frames", type=int, default=12, help="Mínimo de frames por ciclo")
    parser.add_argument("--max-cycle-frames", type=int, default=80, help="Máximo de frames por ciclo")
    parser.add_argument("--min-cycle-continuity", type=float, default=0.85, help="Continuidad temporal mínima del ciclo")
    parser.add_argument("--allow-period-window-fallback", action="store_true", help="Permitir ventanas no solapadas alineadas al período cuando fallen picos equivalentes")

    parser.add_argument(
        "--materialize",
        choices=("copy", "hardlink", "symlink", "none"),
        default="copy",
        help="Cómo crear archivos. 'none' genera solo manifests CSV.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Eliminar salidas previas y regenerar")

    parser.add_argument("--scene", default=None, help="Filtrar escena")
    parser.add_argument("--cam", default=None, help="Filtrar cámara")
    parser.add_argument("--person-id", type=int, default=None, help="Filtrar identidad global")
    parser.add_argument("--limit-cycles", type=int, default=None, help="Detenerse tras generar N ciclos")
    # Alias temporal para comandos previos. No aparece en --help.
    parser.add_argument("--limit-sequences", type=int, dest="limit_cycles", help=argparse.SUPPRESS)

    args = parser.parse_args()

    crops_root = Path(args.crops_root).expanduser().resolve()
    silhouettes_root = Path(args.silhouettes_root).expanduser().resolve()
    gt_root = Path(args.gt_root).expanduser().resolve() if args.gt_root else None
    out_root = Path(args.out_root).expanduser().resolve()

    if not crops_root.is_dir():
        raise FileNotFoundError(f"No existe crops-root: {crops_root}")
    if not silhouettes_root.is_dir():
        raise FileNotFoundError(f"No existe silhouettes-root: {silhouettes_root}")
    if args.min_step_frames < 1 or args.max_step_frames < args.min_step_frames:
        raise ValueError("Rango de período de paso inválido")
    if args.cycle_multiplier <= 0:
        raise ValueError("--cycle-multiplier debe ser > 0")
    if not (0 <= args.phase_search_ratio <= 0.5):
        raise ValueError("--phase-search-ratio debe estar entre 0 y 0.5")
    if not (0 <= args.min_cycle_continuity <= 1):
        raise ValueError("--min-cycle-continuity debe estar entre 0 y 1")

    reset_output_root(out_root, overwrite=args.overwrite)
    metadata_dir = out_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    crops_base = resolve_crops_base(crops_root)
    scene_filter = normalize_scene(args.scene) if args.scene else None
    cam_filter = normalize_cam(args.cam) if args.cam else None

    cycle_rows: List[Dict[str, object]] = []
    frame_rows: List[Dict[str, object]] = []
    skipped_rows: List[Dict[str, object]] = []

    gt_cache: Dict[Tuple[str, str], Dict[Tuple[int, int], Tuple[int, int, int, int]]] = {}
    cycle_counter_by_group: Dict[Tuple[int, str, str], int] = {}
    generated_count = 0
    stop = False

    groups = []
    for gid, scene, cam, rgb_input_dir in iter_person_scene_cam(crops_base):
        if args.person_id is not None and gid != args.person_id:
            continue
        if scene_filter is not None and scene != scene_filter:
            continue
        if cam_filter is not None and cam != cam_filter:
            continue
        groups.append((gid, scene, cam, rgb_input_dir))

    for gid, scene, cam, rgb_input_dir in tqdm(groups, desc="Procesando videos"):

        gt_key = (scene, cam)
        if gt_key not in gt_cache:
            gt_cache[gt_key] = load_gt_lookup(gt_root, scene, cam)

        items = build_frame_items(
            gid=gid,
            scene=scene,
            cam=cam,
            rgb_dir=rgb_input_dir,
            silhouettes_root=silhouettes_root,
            gt_lookup=gt_cache[gt_key],
        )
        if not items:
            continue

        tracklets = split_into_tracklets(items, max_frame_gap=args.max_frame_gap)
        for tracklet_index, tracklet in enumerate(tracklets, start=1):
            if len(tracklet) < max(args.min_cycle_frames + 1, args.min_step_frames + 2):
                skipped_rows.append({
                    "person_global_id": gid,
                    "scene_id": scene,
                    "cam_id": cam,
                    "tracklet_index": tracklet_index,
                    "tracklet_num_frames": len(tracklet),
                    "skip_reason": "tracklet_too_short",
                })
                continue

            cycles, diagnostic = cycles_from_tracklet(
                tracklet,
                min_step_frames=args.min_step_frames,
                max_step_frames=args.max_step_frames,
                cycle_multiplier=args.cycle_multiplier,
                smooth_window=args.smooth_window,
                phase_search_ratio=args.phase_search_ratio,
                min_periodicity_score=args.min_periodicity_score,
                min_cycle_frames=args.min_cycle_frames,
                max_cycle_frames=args.max_cycle_frames,
                min_cycle_continuity=args.min_cycle_continuity,
                allow_period_window_fallback=args.allow_period_window_fallback,
            )

            if not cycles:
                skipped_rows.append({
                    "person_global_id": gid,
                    "scene_id": scene,
                    "cam_id": cam,
                    "tracklet_index": tracklet_index,
                    **diagnostic,
                })
                continue

            for detected in cycles:
                group_key = (gid, scene, cam)
                cycle_counter_by_group[group_key] = cycle_counter_by_group.get(group_key, 0) + 1
                cycle_index = cycle_counter_by_group[group_key]
                cycle_id = f"{scene}_{cam}_ID{gid:04d}_cyc{cycle_index:04d}"
                cycle_dir_name = f"cycle{cycle_index:04d}"

                rgb_output_dir = out_root / "rgb" / f"ID{gid:04d}" / scene / cam / cycle_dir_name
                sil_output_dir = out_root / "silhouette" / f"ID{gid:04d}" / scene / cam / cycle_dir_name

                cycle_items: List[FrameItem] = detected["items"]  # type: ignore[assignment]
                raw_signal: np.ndarray = detected["raw_signal"]  # type: ignore[assignment]
                smooth_signal: np.ndarray = detected["smooth_signal"]  # type: ignore[assignment]

                frame_numbers = [item.frame_number for item in cycle_items]
                tstats = temporal_stats(frame_numbers)
                bstats = bbox_stats(cycle_items)

                for position, (item, fg_sum, fg_sum_smooth) in enumerate(
                    zip(cycle_items, raw_signal, smooth_signal),
                    start=1,
                ):
                    rgb_dst = rgb_output_dir / item.rgb_path.name
                    sil_dst = sil_output_dir / item.silhouette_path.name
                    if args.materialize != "none":
                        materialize_file(item.rgb_path, rgb_dst, args.materialize)
                        materialize_file(item.silhouette_path, sil_dst, args.materialize)

                    bbox_x = bbox_y = bbox_w = bbox_h = bbox_area = None
                    if item.bbox is not None:
                        bbox_x, bbox_y, bbox_w, bbox_h = item.bbox
                        bbox_area = bbox_w * bbox_h

                    frame_rows.append({
                        "sequence_id": cycle_id,
                        "cycle_id": cycle_id,
                        "person_global_id": gid,
                        "scene_id": scene,
                        "cam_id": cam,
                        "tracklet_index": tracklet_index,
                        "cycle_index": cycle_index,
                        "position_in_cycle": position,
                        "frame_number": item.frame_number,
                        "rgb_source_path": str(item.rgb_path),
                        "silhouette_source_path": str(item.silhouette_path),
                        "rgb_output_path": str(rgb_dst.relative_to(out_root)) if args.materialize != "none" else "",
                        "silhouette_output_path": str(sil_dst.relative_to(out_root)) if args.materialize != "none" else "",
                        "bbox_x": bbox_x,
                        "bbox_y": bbox_y,
                        "bbox_w": bbox_w,
                        "bbox_h": bbox_h,
                        "bbox_area": bbox_area,
                        "foreground_sum": float(fg_sum),
                        "foreground_sum_smooth": float(fg_sum_smooth),
                    })

                cycle_rows.append({
                    "sequence_id": cycle_id,
                    "cycle_id": cycle_id,
                    "person_global_id": gid,
                    "scene_id": scene,
                    "cam_id": cam,
                    "tracklet_index": tracklet_index,
                    "cycle_index": cycle_index,
                    "rgb_dir": str(rgb_output_dir.relative_to(out_root)) if args.materialize != "none" else "",
                    "silhouette_dir": str(sil_output_dir.relative_to(out_root)) if args.materialize != "none" else "",
                    "num_frames": len(cycle_items),
                    "frame_start": tstats["frame_start"],
                    "frame_end": tstats["frame_end"],
                    "frame_numbers_json": json.dumps(frame_numbers),
                    "n_samples_rgb": len(cycle_items),
                    "n_samples_sil": len(cycle_items),
                    "frame_gap_max": tstats["frame_gap_max"],
                    "frame_gap_mean": tstats["frame_gap_mean"],
                    "missing_frame_count": tstats["missing_frame_count"],
                    "temporal_continuity_ratio": tstats["temporal_continuity_ratio"],
                    "bbox_h_median": bstats["bbox_h_median"],
                    "bbox_w_median": bstats["bbox_w_median"],
                    "bbox_area_mean": bstats["bbox_area_mean"],
                    "bbox_available_ratio": bstats["bbox_available_ratio"],
                    "cycle_detection_method": detected["cycle_detection_method"],
                    "estimated_step_period": diagnostic["estimated_step_period"],
                    "estimated_cycle_length": diagnostic["estimated_cycle_length"],
                    "cycle_start_frame": frame_numbers[0],
                    "cycle_end_frame": frame_numbers[-1],
                    "cycle_length_error": detected["cycle_length_error"],
                    "autocorrelation_step_peak": diagnostic["autocorrelation_step_peak"],
                    "autocorrelation_cycle_peak": diagnostic["autocorrelation_cycle_peak"],
                    "periodicity_score": diagnostic["periodicity_score"],
                    "cycle_quality_flag": detected["cycle_quality_flag"],
                    "foreground_sum_mean": float(np.mean(raw_signal)),
                    "foreground_sum_std": float(np.std(raw_signal, ddof=1)) if len(raw_signal) > 1 else 0.0,
                })

                generated_count += 1
                if args.limit_cycles is not None and generated_count >= args.limit_cycles:
                    stop = True
                    break
            if stop:
                break
        if stop:
            break

    if not cycle_rows:
        raise RuntimeError(
            "No se generaron ciclos. Revise rutas, calidad de siluetas y umbrales. "
            "Para diagnóstico, pruebe --allow-period-window-fallback o reduzca "
            "--min-periodicity-score."
        )

    gait_cycles_csv = metadata_dir / "gait_cycles.csv"
    gait_frames_csv = metadata_dir / "gait_cycle_frames.csv"
    skipped_csv = metadata_dir / "skipped_tracklets.csv"

    cycles_df = pd.DataFrame(cycle_rows)
    frames_df = pd.DataFrame(frame_rows)
    skipped_df = pd.DataFrame(skipped_rows)

    cycles_df.to_csv(gait_cycles_csv, index=False)
    frames_df.to_csv(gait_frames_csv, index=False)
    skipped_df.to_csv(skipped_csv, index=False)

    # Aliases de compatibilidad con scripts previos
    cycles_df.to_csv(metadata_dir / "sequences.csv", index=False)
    frames_df.to_csv(metadata_dir / "sequence_frames.csv", index=False)

    print("\nProceso completado")
    print(f"- Ciclos generados: {len(cycle_rows)}")
    print(f"- Registros frame-ciclo: {len(frame_rows)}")
    print(f"- Tracklets omitidos: {len(skipped_rows)}")
    print(f"- Metadata ciclos: {gait_cycles_csv}")
    print(f"- Metadata frames: {gait_frames_csv}")
    print(f"- Diagnóstico omitidos: {skipped_csv}")
    print("- Overlap: deshabilitado; los intervalos de ciclo son [inicio, fin)")
    print("- Iluminación, ropa, color, SQQ, Gower e intra/inter: scripts separados")


if __name__ == "__main__":
    main()
