#!/usr/bin/env python3
"""
Clasifica el nivel de oclusión peatonal por secuencia siguiendo Gilroy et al. (2021):
"Pedestrian Occlusion Level Classification using Keypoint Detection and 2D Body
Surface Area Estimation".

Método base (a nivel de frame):
1. Detectar keypoints COCO de la persona con Keypoint R-CNN de Detectron2.
2. Marcar un keypoint como visible si su score >= --keypoint-visibility-thr
   (Gilroy et al. usan 0.15).
3. Agrupar keypoints visibles en partes semánticas del cuerpo.
4. Sumar la superficie corporal 2D visible según la adaptación de la Rule of Nines.
5. Calcular oclusión porcentual = 100 * (1 - BSA_visible / 99).

Adaptación para este corpus (a nivel de secuencia):
- Muestrear hasta K frames equidistantes por secuencia.
- Calcular oclusión por frame con Gilroy.
- Agregar por secuencia usando media (configurable: mean/median/max).
- Categorizar la secuencia usando umbrales configurables. Por defecto se usan
  baja < 40%, parcial 40-80%, alta > 80%, coherentes con EuroCity Persons,
  uno de los benchmarks resumidos por Gilroy et al. en su Tabla 1.

Este script NO calcula iluminación, color, ropa, calidad de silueta, Gower,
ni separabilidad intra/inter-clase.

Dependencias:
  pip install pandas numpy opencv-python tqdm
  # Detectron2 debe instalarse según la versión de PyTorch/CUDA del entorno:
  # https://detectron2.readthedocs.io/en/latest/tutorials/install.html

Ejemplo:
  python src/11a_label_sequence_occlusion_gilroy.py --base-dir dataset/output --sequences-csv dataset/output/metadata/gait_cycles.csv --frames-csv dataset/output/metadata/gait_cycle_frames.csv --out-csv dataset/output/metadata/gait_cycles_occlusion.csv --out-frame-csv dataset/output/metadata/gait_cycle_frames_occlusion.csv --max-frames-per-seq 0 --device cuda

Por shards mi dataset:
python src/11a_label_sequence_occlusion_gilroy.py --base-dir dataset/output --sequences-csv dataset/output/metadata/gait_cycles.csv --frames-csv dataset/output/metadata/gait_cycle_frames.csv --out-csv dataset/output/metadata/gait_cycles_occlusion.csv --out-frame-csv dataset/output/metadata/gait_cycle_frames_occlusion.csv --max-frames-per-seq 0 --device cuda --num-shards 3 --shard-id 0
python src/11a_label_sequence_occlusion_gilroy.py --base-dir dataset/output --sequences-csv dataset/output/metadata/gait_cycles.csv --frames-csv dataset/output/metadata/gait_cycle_frames.csv --out-csv dataset/output/metadata/gait_cycles_occlusion.csv --out-frame-csv dataset/output/metadata/gait_cycle_frames_occlusion.csv --max-frames-per-seq 0 --device cuda --num-shards 3 --shard-id 1
python src/11a_label_sequence_occlusion_gilroy.py --base-dir dataset/output --sequences-csv dataset/output/metadata/gait_cycles.csv --frames-csv dataset/output/metadata/gait_cycle_frames.csv --out-csv dataset/output/metadata/gait_cycles_occlusion.csv --out-frame-csv dataset/output/metadata/gait_cycle_frames_occlusion.csv --max-frames-per-seq 0 --device cuda --num-shards 3 --shard-id 2

Dataset iLIDS VID:
python src/11a_label_sequence_occlusion_gilroy.py --base-dir D:\open_datasets\i-LIDS-VID --sequences-csv D:\open_datasets\i-LIDS-VID\metadata\sequences.csv --frames-csv D:\open_datasets\i-LIDS-VID\metadata\sequence_frames.csv --out-csv D:\open_datasets\i-LIDS-VID\metadata\sequences_occlusion.csv --out-frame-csv D:\open_datasets\i-LIDS-VID\metadata\sequence_frames_occlusion.csv --max-frames-per-seq 0 --device cuda

Dataset MARS:
python src/11a_label_sequence_occlusion_gilroy.py --base-dir D:\open_datasets\Mars\raw --sequences-csv D:\open_datasets\Mars\metadata\sequences.csv --frames-csv D:\open_datasets\Mars\metadata\sequence_frames.csv --out-csv D:\open_datasets\Mars\metadata\sequences_occlusion.csv --out-frame-csv D:\open_datasets\Mars\metadata\sequence_frames_occlusion.csv --max-frames-per-seq 10 --device cuda --num-shards 3 --shard-id 0
python src/11a_label_sequence_occlusion_gilroy.py --base-dir D:\open_datasets\Mars\raw --sequences-csv D:\open_datasets\Mars\metadata\sequences.csv --frames-csv D:\open_datasets\Mars\metadata\sequence_frames.csv --out-csv D:\open_datasets\Mars\metadata\sequences_occlusion.csv --out-frame-csv D:\open_datasets\Mars\metadata\sequence_frames_occlusion.csv --max-frames-per-seq 10 --device cuda --num-shards 3 --shard-id 1
python src/11a_label_sequence_occlusion_gilroy.py --base-dir D:\open_datasets\Mars\raw --sequences-csv D:\open_datasets\Mars\metadata\sequences.csv --frames-csv D:\open_datasets\Mars\metadata\sequence_frames.csv --out-csv D:\open_datasets\Mars\metadata\sequences_occlusion.csv --out-frame-csv D:\open_datasets\Mars\metadata\sequence_frames_occlusion.csv --max-frames-per-seq 10 --device cuda --num-shards 3 --shard-id 2

"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# COCO keypoint ordering used by Detectron2 Keypoint R-CNN.
COCO_KEYPOINT_NAMES: Tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

# Gilroy et al., Table 2: 2D body surface area (%) and related keypoints.
# Total = 99%; the paper's adapted representation omits the remaining 1% groin area.
BODY_PARTS: Tuple[Tuple[str, float, str, Tuple[str, ...]], ...] = (
    ("head", 9.0, "any", ("nose", "left_eye", "right_eye", "left_ear", "right_ear")),
    ("upper_torso", 18.0, "all", ("left_shoulder", "right_shoulder")),
    ("upper_left_arm", 4.5, "all", ("left_shoulder", "left_elbow")),
    ("lower_left_arm", 4.5, "all", ("left_elbow", "left_wrist")),
    ("upper_right_arm", 4.5, "all", ("right_shoulder", "right_elbow")),
    ("lower_right_arm", 4.5, "all", ("right_elbow", "right_wrist")),
    ("lower_torso", 18.0, "all", ("left_hip", "right_hip")),
    ("upper_left_leg", 9.0, "all", ("left_hip", "left_knee")),
    ("lower_left_leg", 9.0, "all", ("left_knee", "left_ankle")),
    ("upper_right_leg", 9.0, "all", ("right_hip", "right_knee")),
    ("lower_right_leg", 9.0, "all", ("right_knee", "right_ankle")),
)
TOTAL_2D_BSA = sum(weight for _, weight, _, _ in BODY_PARTS)  # 99.0


@dataclass(frozen=True)
class FrameOcclusionResult:
    status: str
    detection_score: float
    visible_keypoint_count: int
    total_keypoint_count: int
    visible_keypoint_ratio: float
    visible_bsa_pct: float
    occlusion_pct: float
    visible_keypoints: Tuple[str, ...]
    visible_parts: Tuple[str, ...]
    occluded_parts: Tuple[str, ...]


class GilroyOcclusionEstimator:
    """Detectron2 Keypoint R-CNN wrapper implementing Gilroy et al.'s BSA logic."""

    def __init__(
        self,
        *,
        device: str,
        person_score_thr: float,
        keypoint_visibility_thr: float,
        config_name: str,
    ) -> None:
        try:
            from detectron2 import model_zoo
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Detectron2 no está instalado. Instálalo según tu versión de "
                "PyTorch/CUDA antes de ejecutar este script."
            ) from exc

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(config_name))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_name)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = person_score_thr
        cfg.MODEL.DEVICE = device
        self.predictor = DefaultPredictor(cfg)
        self.person_score_thr = float(person_score_thr)
        self.keypoint_visibility_thr = float(keypoint_visibility_thr)

    def estimate(self, image_bgr: np.ndarray) -> FrameOcclusionResult:
        outputs = self.predictor(image_bgr)
        instances = outputs["instances"].to("cpu")

        if len(instances) == 0:
            return self._empty("no_person_detection")

        classes = instances.pred_classes.numpy()
        scores = instances.scores.numpy()
        person_idx = np.where(classes == 0)[0]  # COCO class 0 = person
        if person_idx.size == 0:
            return self._empty("no_person_detection")

        # Crops are expected to contain one target pedestrian. If multiple people are
        # detected, retain the most confident person instance.
        best = int(person_idx[np.argmax(scores[person_idx])])
        detection_score = float(scores[best])
        if detection_score < self.person_score_thr:
            return self._empty("person_score_below_threshold", detection_score)

        keypoints = instances.pred_keypoints[best].numpy()  # (17, 3): x, y, score
        kp_scores = keypoints[:, 2]
        visible_mask = kp_scores >= self.keypoint_visibility_thr
        visible_names = tuple(
            name for name, is_visible in zip(COCO_KEYPOINT_NAMES, visible_mask) if bool(is_visible)
        )
        visible_set = set(visible_names)

        visible_parts: List[str] = []
        occluded_parts: List[str] = []
        visible_bsa = 0.0

        for part_name, weight, rule, related_keypoints in BODY_PARTS:
            if rule == "any":
                part_visible = any(kp in visible_set for kp in related_keypoints)
            elif rule == "all":
                part_visible = all(kp in visible_set for kp in related_keypoints)
            else:  # defensive programming
                raise ValueError(f"Regla semántica no soportada: {rule}")

            if part_visible:
                visible_parts.append(part_name)
                visible_bsa += weight
            else:
                occluded_parts.append(part_name)

        visible_bsa_pct = 100.0 * visible_bsa / TOTAL_2D_BSA
        occlusion_pct = 100.0 - visible_bsa_pct
        visible_count = int(visible_mask.sum())

        return FrameOcclusionResult(
            status="ok",
            detection_score=detection_score,
            visible_keypoint_count=visible_count,
            total_keypoint_count=len(COCO_KEYPOINT_NAMES),
            visible_keypoint_ratio=float(visible_count / len(COCO_KEYPOINT_NAMES)),
            visible_bsa_pct=float(visible_bsa_pct),
            occlusion_pct=float(occlusion_pct),
            visible_keypoints=visible_names,
            visible_parts=tuple(visible_parts),
            occluded_parts=tuple(occluded_parts),
        )

    @staticmethod
    def _empty(status: str, detection_score: float = float("nan")) -> FrameOcclusionResult:
        return FrameOcclusionResult(
            status=status,
            detection_score=float(detection_score),
            visible_keypoint_count=0,
            total_keypoint_count=len(COCO_KEYPOINT_NAMES),
            visible_keypoint_ratio=float("nan"),
            visible_bsa_pct=float("nan"),
            occlusion_pct=float("nan"),
            visible_keypoints=tuple(),
            visible_parts=tuple(),
            occluded_parts=tuple(),
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Clasifica oclusión por secuencia siguiendo Gilroy et al. (2021)."
    )
    ap.add_argument("--base-dir", "--base_dir", dest="base_dir", default=".")
    ap.add_argument("--sequences-csv", "--sequences_csv", dest="sequences_csv", required=True)
    ap.add_argument("--frames-csv", "--frames_csv", dest="frames_csv", required=True)
    ap.add_argument("--out-csv", "--out_csv", dest="out_csv", required=True)
    ap.add_argument(
        "--out-frame-csv",
        "--out_frame_csv",
        dest="out_frame_csv",
        required=True,
        help="Salida de auditoría con resultados de oclusión por frame muestreado.",
    )
    ap.add_argument(
        "--max-frames-per-seq",
        "--max_frames_per_seq",
        dest="max_frames_per_seq",
        type=int,
        default=10,
        help="Máximo de frames equidistantes evaluados por secuencia. Use 0 para todos.",
    )
    ap.add_argument(
        "--rgb-path-col",
        default=None,
        help="Columna de ruta RGB en frames_csv. Si se omite, se detecta automáticamente.",
    )
    ap.add_argument(
        "--frame-number-col",
        default=None,
        help="Columna de número de frame. Si se omite, se detecta automáticamente.",
    )
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument(
        "--detectron2-config",
        default="COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml",
        help="Configuración Detectron2. Gilroy et al. usan Faster/Keypoint R-CNN con ResNet-50-FPN y COCO keypoints.",
    )
    ap.add_argument(
        "--person-score-thr",
        type=float,
        default=0.50,
        help="Confianza mínima de detección de persona. Parámetro operativo del script.",
    )
    ap.add_argument(
        "--keypoint-visibility-thr",
        type=float,
        default=0.15,
        help="Umbral de visibilidad de keypoints. Gilroy et al. usan 0.15.",
    )
    ap.add_argument(
        "--sequence-aggregation",
        choices=("mean", "median", "max"),
        default="mean",
        help="Cómo agregar oclusión por frame a nivel de secuencia. mean es el valor recomendado para caracterización global.",
    )
    ap.add_argument(
        "--low-thr",
        type=float,
        default=40.0,
        help="Umbral porcentual: baja si oclusión < low_thr.",
    )
    ap.add_argument(
        "--heavy-thr",
        type=float,
        default=80.0,
        help="Umbral porcentual: alta si oclusión > heavy_thr; parcial en el intervalo restante.",
    )
    ap.add_argument(
        "--min-valid-pose-ratio",
        type=float,
        default=0.50,
        help="Cobertura mínima de frames con pose válida para etiquetar una secuencia.",
    )
    ap.add_argument(
        "--merge-how",
        choices=("left", "inner"),
        default="left",
        help="Tipo de merge con sequences.csv.",
    )
    ap.add_argument(
        "--num-shards",
        "--num_shards",
        dest="num_shards",
        type=int,
        default=1,
        help="Número total de shards para paralelización.",
    )
    ap.add_argument(
        "--shard-id",
        "--shard_id",
        dest="shard_id",
        type=int,
        default=0,
        help="ID de shard actual (0-indexed, desde 0 hasta num-shards - 1).",
    )
    return ap.parse_args()


def detect_first_existing_column(df: pd.DataFrame, candidates: Sequence[str], explicit: Optional[str]) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"No existe la columna solicitada '{explicit}' en frames_csv.")
        return explicit
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"No se encontró ninguna columna compatible. Opciones esperadas: {list(candidates)}")


def resolve_path(base_dir: Path, raw_path: object, frames_csv: Path) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path

    candidate = base_dir / path
    if candidate.exists():
        return candidate

    candidate_csv = frames_csv.parent / path
    if candidate_csv.exists():
        return candidate_csv

    # Return the base-dir candidate to produce an actionable error path.
    return base_dir / path


def sample_rows_uniformly(group: pd.DataFrame, k: int) -> pd.DataFrame:
    if k <= 0 or len(group) <= k:
        return group
    indices = np.linspace(0, len(group) - 1, num=k, dtype=int)
    return group.iloc[np.unique(indices)]


def classify_occlusion(occlusion_pct: float, valid_pose_ratio: float, low_thr: float, heavy_thr: float, min_valid_pose_ratio: float) -> Tuple[str, Optional[int]]:
    if not np.isfinite(occlusion_pct) or valid_pose_ratio < min_valid_pose_ratio:
        return "desconocida", None
    if occlusion_pct < low_thr:
        return "baja", 0
    if occlusion_pct <= heavy_thr:
        return "parcial", 1
    return "alta", 2


def aggregate(values: Sequence[float], method: str) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan")
    if method == "mean":
        return float(np.mean(arr))
    if method == "median":
        return float(np.median(arr))
    if method == "max":
        return float(np.max(arr))
    raise ValueError(method)


def safe_mean(values: Sequence[float]) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    return float(np.mean(arr)) if arr.size else float("nan")


def safe_std(values: Sequence[float]) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    return float(np.std(arr, ddof=1)) if arr.size >= 2 else (0.0 if arr.size == 1 else float("nan"))


def most_frequent_parts(part_lists: Iterable[Sequence[str]], min_fraction: float = 0.50) -> List[str]:
    counter: Counter[str] = Counter()
    n = 0
    for parts in part_lists:
        n += 1
        counter.update(set(parts))
    if n == 0:
        return []
    return sorted([part for part, count in counter.items() if count / n >= min_fraction])


def main() -> int:
    args = parse_args()

    if args.low_thr < 0 or args.heavy_thr > 100 or args.low_thr >= args.heavy_thr:
        raise ValueError("Se requiere 0 <= low_thr < heavy_thr <= 100.")
    if not 0 <= args.keypoint_visibility_thr <= 1:
        raise ValueError("--keypoint-visibility-thr debe estar entre 0 y 1.")
    if not 0 <= args.min_valid_pose_ratio <= 1:
        raise ValueError("--min-valid-pose-ratio debe estar entre 0 y 1.")

    base_dir = Path(args.base_dir)
    sequences_csv = Path(args.sequences_csv)
    frames_csv = Path(args.frames_csv)
    out_csv = Path(args.out_csv)
    out_frame_csv = Path(args.out_frame_csv)

    seq_df = pd.read_csv(sequences_csv)
    frames_df = pd.read_csv(frames_csv)

    if "sequence_id" not in seq_df.columns:
        raise ValueError("sequences_csv debe contener 'sequence_id'.")
    if "sequence_id" not in frames_df.columns:
        raise ValueError("frames_csv debe contener 'sequence_id'.")

    if args.num_shards < 1:
        raise ValueError("--num-shards debe ser >= 1.")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError(f"--shard-id debe estar entre 0 y {args.num_shards - 1}.")

    if args.num_shards > 1:
        out_csv = out_csv.parent / f"{out_csv.stem}_shard_{args.shard_id}_{args.num_shards}{out_csv.suffix}"
        out_frame_csv = out_frame_csv.parent / f"{out_frame_csv.stem}_shard_{args.shard_id}_{args.num_shards}{out_frame_csv.suffix}"

        seq_df = seq_df.sort_values("sequence_id").reset_index(drop=True)
        indices = [i for i in range(len(seq_df)) if i % args.num_shards == args.shard_id]
        seq_df = seq_df.iloc[indices].reset_index(drop=True)

        selected_seqs = set(seq_df["sequence_id"])
        frames_df = frames_df[frames_df["sequence_id"].isin(selected_seqs)].reset_index(drop=True)
        print(f"[INFO] Sharding activo. Shard {args.shard_id + 1}/{args.num_shards} ({len(seq_df)} secuencias, {len(frames_df)} frames).")

    rgb_path_col = detect_first_existing_column(
        frames_df,
        ("rgb_source_path", "rgb_output_path", "rgb_path", "image_path", "path"),
        args.rgb_path_col,
    )
    frame_number_col = detect_first_existing_column(
        frames_df,
        ("frame_number", "frame", "frame_id", "position_in_cycle", "position_in_sequence"),
        args.frame_number_col,
    )

    estimator = GilroyOcclusionEstimator(
        device=args.device,
        person_score_thr=args.person_score_thr,
        keypoint_visibility_thr=args.keypoint_visibility_thr,
        config_name=args.detectron2_config,
    )

    sequence_rows: List[Dict[str, object]] = []
    frame_rows: List[Dict[str, object]] = []
    missing_rgb = 0

    grouped = frames_df.groupby("sequence_id", sort=False)
    for sequence_id, group in tqdm(grouped, desc="Clasificando oclusión por secuencia"):
        group = group.sort_values(frame_number_col)
        sampled = sample_rows_uniformly(group, args.max_frames_per_seq)

        occlusion_values: List[float] = []
        visible_bsa_values: List[float] = []
        visible_kp_ratios: List[float] = []
        valid_occluded_part_lists: List[Sequence[str]] = []
        sampled_frame_numbers: List[object] = []
        valid_pose_frames = 0

        for _, row in sampled.iterrows():
            frame_number = row[frame_number_col]
            sampled_frame_numbers.append(frame_number)
            rgb_path = resolve_path(base_dir, row[rgb_path_col], frames_csv)

            if not rgb_path.exists():
                missing_rgb += 1
                result = GilroyOcclusionEstimator._empty("rgb_missing")
            else:
                image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
                if image is None:
                    missing_rgb += 1
                    result = GilroyOcclusionEstimator._empty("rgb_unreadable")
                else:
                    result = estimator.estimate(image)

            if result.status == "ok":
                valid_pose_frames += 1
                occlusion_values.append(result.occlusion_pct)
                visible_bsa_values.append(result.visible_bsa_pct)
                visible_kp_ratios.append(result.visible_keypoint_ratio)
                valid_occluded_part_lists.append(result.occluded_parts)

            frame_rows.append(
                {
                    "sequence_id": sequence_id,
                    "frame_number": frame_number,
                    "rgb_path": str(row[rgb_path_col]),
                    "gilroy_status": result.status,
                    "person_detection_score": result.detection_score,
                    "visible_keypoint_count": result.visible_keypoint_count,
                    "total_keypoint_count": result.total_keypoint_count,
                    "visible_keypoint_ratio": result.visible_keypoint_ratio,
                    "visible_bsa_pct": result.visible_bsa_pct,
                    "occlusion_pct": result.occlusion_pct,
                    "visible_keypoints_json": json.dumps(result.visible_keypoints, ensure_ascii=False),
                    "visible_parts_json": json.dumps(result.visible_parts, ensure_ascii=False),
                    "occluded_parts_json": json.dumps(result.occluded_parts, ensure_ascii=False),
                }
            )

        n_sampled = len(sampled)
        valid_pose_ratio = float(valid_pose_frames / n_sampled) if n_sampled else 0.0
        occlusion_score_pct = aggregate(occlusion_values, args.sequence_aggregation)
        occlusion_label, occlusion_level = classify_occlusion(
            occlusion_score_pct,
            valid_pose_ratio,
            args.low_thr,
            args.heavy_thr,
            args.min_valid_pose_ratio,
        )

        sequence_rows.append(
            {
                "sequence_id": sequence_id,
                "occlusion_method": "gilroy_keypoints_2d_bsa",
                "occlusion_aggregation": args.sequence_aggregation,
                "occlusion_score": (occlusion_score_pct / 100.0) if np.isfinite(occlusion_score_pct) else np.nan,
                "occlusion_pct": occlusion_score_pct,
                "occlusion_label": occlusion_label,
                "occlusion_level": occlusion_level,
                "occlusion_pct_mean": safe_mean(occlusion_values),
                "occlusion_pct_median": float(np.median(occlusion_values)) if occlusion_values else np.nan,
                "occlusion_pct_std": safe_std(occlusion_values),
                "occlusion_pct_max": float(np.max(occlusion_values)) if occlusion_values else np.nan,
                "visible_body_surface_pct_mean": safe_mean(visible_bsa_values),
                "visible_keypoint_ratio_mean": safe_mean(visible_kp_ratios),
                "n_samples_rgb": n_sampled,
                "n_valid_pose_frames": valid_pose_frames,
                "valid_pose_ratio": valid_pose_ratio,
                "sampled_frame_numbers_json": json.dumps(sampled_frame_numbers, ensure_ascii=False),
                "frequently_occluded_parts_json": json.dumps(
                    most_frequent_parts(valid_occluded_part_lists), ensure_ascii=False
                ),
                "keypoint_visibility_threshold": args.keypoint_visibility_thr,
                "occlusion_low_threshold_pct": args.low_thr,
                "occlusion_heavy_threshold_pct": args.heavy_thr,
            }
        )

    sequence_metrics_df = pd.DataFrame(sequence_rows)
    frame_metrics_df = pd.DataFrame(frame_rows)

    out_df = seq_df.merge(sequence_metrics_df, on="sequence_id", how=args.merge_how)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_frame_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    frame_metrics_df.to_csv(out_frame_csv, index=False)

    print(f"[INFO] Método: Gilroy keypoints + 2D BSA")
    print(f"[INFO] Umbral visibilidad keypoints: {args.keypoint_visibility_thr:.3f}")
    print(f"[INFO] Categorías: baja < {args.low_thr:.1f}%, parcial <= {args.heavy_thr:.1f}%, alta > {args.heavy_thr:.1f}%")
    print(f"[INFO] Secuencias procesadas: {len(sequence_metrics_df)}")
    print(f"[INFO] Frames muestreados: {len(frame_metrics_df)}")
    print(f"[WARN] RGB faltantes/no legibles: {missing_rgb}")
    print(f"[INFO] Guardado: {out_csv}")
    print(f"[INFO] Auditoría por frame: {out_frame_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
