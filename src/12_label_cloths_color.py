#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12_label_cloths_color.py

Prepara metadatos visuales semiautomáticos para un análisis posterior de pares
entre identidades. El script NO enumera pares todavía. Genera una tabla por
secuencia y otra tabla consolidada por identidad (o identidad-sesión) con los
campos necesarios para derivar posteriormente:

- mismo tipo de prenda superior;
- mismo tipo de prenda inferior;
- mismo color de prenda superior;
- mismo color de prenda inferior;
- ambas identidades tienen mochila;
- ninguna de las dos identidades tiene mochila.

Diseño metodológico:
- La ropa y la presencia de mochila se preanotan con un modelo multimodal sobre
  varios frames representativos de la secuencia. La salida se considera
  semiautomática y auditable, no ground truth definitivo.
- Los colores se estiman por separado para la región superior e inferior sobre
  la máscara binaria de la persona. Se guardan etiqueta, familia cromática y
  confianza. Los umbrales HSV son reglas operativas del pipeline y deben
  validarse sobre una muestra revisada manualmente.
- Se evita asignar valores por defecto sesgados. Cuando la evidencia es
  insuficiente, se usa "unknown".
- Puede aplicarse un CSV de correcciones manuales por secuencia.

Requisitos:
    pip install pandas numpy opencv-python tqdm
Opcional para --annotation-mode openai:
    pip install openai
    set OPENAI_API_KEY=...

Ejemplo:
    python src\\12_label_cloths_color.py ^
      --base-dir dataset\\output_v1 ^
      --sequences-csv dataset\\output_v1\\metadata\\gait_cycles.csv ^
      --frames-csv dataset\\output_v1\\metadata\\gait_cycle_frames.csv ^
      --sil-quality-frames-csv dataset\\output_v1\\metadata\\gait_cycle_frames_silhouette_quality.csv ^
      --occlusion-frames-csv dataset\\output_v1\\metadata\\gait_cycle_frames_occlusion.csv ^
      --out-sequence-csv dataset\\output_v1\\metadata\\sequence_visual_attributes.csv ^
      --out-identity-csv dataset\\output_v1\\metadata\\identity_visual_attributes_pair_ready.csv ^
      --out-frame-csv dataset\\output_v1\\metadata\\visual_attribute_sampled_frames.csv ^
      --review-csv dataset\\output_v1\\metadata\\visual_attributes_manual_review.csv ^
      --pair-schema-json dataset\\output_v1\\metadata\\pair_feature_schema.json ^
      --annotation-mode openai ^
      --model gpt-4.1-mini ^
      --sample-frames 5

python src\12_label_cloths_color.py --base-dir dataset\output --sequences-csv dataset\output\metadata\gait_cycles.csv --frames-csv dataset\output\metadata\gait_cycle_frames.csv --sil-quality-frames-csv dataset\output\metadata\gait_cycle_frames_silhouette_quality.csv --occlusion-frames-csv dataset\output\metadata\gait_cycle_frames_occlusion.csv --out-sequence-csv dataset\output\metadata\sequence_visual_attributes.csv --out-identity-csv dataset\output\metadata\identity_visual_attributes_pair_ready.csv --out-frame-csv dataset\output\metadata\visual_attribute_sampled_frames.csv --review-csv dataset\output\metadata\visual_attributes_manual_review.csv --pair-schema-json dataset\output\metadata\pair_feature_schema.json --annotation-mode openai --model gpt-4.1-mini --cache-jsonl dataset\output\metadata\visual_attributes_cache.jsonl --sample-frames 5

"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


UNKNOWN = "unknown"

UPPER_GARMENT_VOCAB = [
    "tshirt",
    "shirt",
    "blouse",
    "sweater",
    "hoodie",
    "jacket",
    "coat",
    "dress",
    "other",
    UNKNOWN,
]

LOWER_GARMENT_VOCAB = [
    "pants",
    "shorts",
    "skirt",
    "dress",
    "other",
    UNKNOWN,
]

BACKPACK_VOCAB = ["yes", "no", UNKNOWN]

COLOR_VOCAB = [
    "black",
    "white",
    "gray",
    "red",
    "orange",
    "yellow",
    "green",
    "cyan",
    "blue",
    "purple",
    "brown",
    "mixed",
    UNKNOWN,
]

COLOR_FAMILY = {
    "black": "neutral",
    "white": "neutral",
    "gray": "neutral",
    "red": "warm",
    "orange": "warm",
    "yellow": "warm",
    "brown": "warm",
    "green": "cool",
    "cyan": "cool",
    "blue": "cool",
    "purple": "cool",
    "mixed": "mixed",
    UNKNOWN: UNKNOWN,
}


@dataclass
class FrameColorEvidence:
    sequence_id: str
    frame_number: Optional[int]
    rgb_path: str
    silhouette_path: str
    selected_rank: int
    quality_frame_valid: Optional[bool]
    occlusion_pct: Optional[float]
    upper_color_label: str
    upper_color_family: str
    upper_color_confidence: float
    upper_color_pixels: int
    lower_color_label: str
    lower_color_family: str
    lower_color_confidence: float
    lower_color_pixels: int


@dataclass
class SemanticAttributes:
    upper_garment_type: str = UNKNOWN
    lower_garment_type: str = UNKNOWN
    backpack_flag: str = UNKNOWN
    semantic_confidence: float = 0.0
    semantic_notes: str = ""
    annotation_source: str = "not_run"


# -----------------------------------------------------------------------------
# Utilidades generales
# -----------------------------------------------------------------------------

def ensure_parent(path: str | Path) -> None:
    parent = Path(path).expanduser().resolve().parent
    parent.mkdir(parents=True, exist_ok=True)


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def normalize_token(value: Any, allowed: Sequence[str]) -> str:
    token = clean_text(value).lower().replace(" ", "_")
    aliases = {
        "t-shirt": "tshirt",
        "polo": "tshirt",
        "camiseta": "tshirt",
        "camisa": "shirt",
        "blusa": "blouse",
        "sueter": "sweater",
        "sweater": "sweater",
        "chompa": "sweater",
        "buzo": "hoodie",
        "hoodie": "hoodie",
        "chaqueta": "jacket",
        "abrigo": "coat",
        "vestido": "dress",
        "pantalon": "pants",
        "pantalón": "pants",
        "short": "shorts",
        "falda": "skirt",
        "si": "yes",
        "sí": "yes",
        "true": "yes",
        "1": "yes",
        "no": "no",
        "false": "no",
        "0": "no",
        "na": UNKNOWN,
        "n/a": UNKNOWN,
        "none": UNKNOWN,
        "null": UNKNOWN,
        "desconocido": UNKNOWN,
        "unknown": UNKNOWN,
    }
    token = aliases.get(token, token)
    return token if token in set(allowed) else UNKNOWN


def resolve_path(base_dir: Path, raw_path: Any) -> Optional[Path]:
    text = clean_text(raw_path)
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def first_existing_column(df: pd.DataFrame, candidates: Sequence[str], *, required: bool = True) -> Optional[str]:
    for name in candidates:
        if name in df.columns:
            return name
    if required:
        raise ValueError(f"No se encontró ninguna columna entre: {', '.join(candidates)}")
    return None


def to_bool_or_none(value: Any) -> Optional[bool]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "si", "sí", "valid", "ok"}:
        return True
    if token in {"0", "false", "no", "invalid", "bad"}:
        return False
    return None


def to_float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def equidistant_indices(n: int, k: int) -> List[int]:
    if n <= 0:
        return []
    if k <= 0 or k >= n:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, k, dtype=int).tolist()))


def weighted_mode(items: Iterable[Tuple[str, float]]) -> Tuple[str, float, float]:
    """Retorna etiqueta, confianza agregada y razón de consenso."""
    weights: Dict[str, float] = defaultdict(float)
    n = 0
    for label, weight in items:
        label = clean_text(label).lower() or UNKNOWN
        weight = float(max(weight, 0.0))
        if label == UNKNOWN:
            continue
        weights[label] += weight
        n += 1
    if not weights:
        return UNKNOWN, 0.0, 0.0
    ordered = sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))
    label, winning_weight = ordered[0]
    total = sum(weights.values())
    consensus = winning_weight / total if total > 0 else 0.0
    return label, consensus, consensus


# -----------------------------------------------------------------------------
# Máscaras y color por región corporal
# -----------------------------------------------------------------------------

def read_binary_mask(path: Path) -> Optional[np.ndarray]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return (image > 127).astype(np.uint8)


def body_region_masks(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Divide el cuerpo visible en dos regiones aproximadas dentro del bounding box
    de la silueta: superior 20%-55% e inferior 55%-95% de la altura.
    Se excluyen cabeza y pies para reducir contaminación cromática.
    """
    ys, xs = np.where(mask > 0)
    upper = np.zeros_like(mask, dtype=np.uint8)
    lower = np.zeros_like(mask, dtype=np.uint8)
    if len(xs) == 0:
        return upper, lower

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    h = max(1, y1 - y0)

    upper_y0 = y0 + int(round(0.20 * h))
    upper_y1 = y0 + int(round(0.55 * h))
    lower_y0 = y0 + int(round(0.55 * h))
    lower_y1 = y0 + int(round(0.95 * h))

    upper[upper_y0:upper_y1, x0:x1] = mask[upper_y0:upper_y1, x0:x1]
    lower[lower_y0:lower_y1, x0:x1] = mask[lower_y0:lower_y1, x0:x1]
    return upper, lower


def hsv_color_histogram(rgb_bgr: np.ndarray, region_mask: np.ndarray) -> Tuple[Dict[str, float], int]:
    """Clasificador HSV operativo. Devuelve distribución normalizada y número de píxeles."""
    if rgb_bgr is None or region_mask is None:
        return {}, 0
    valid = region_mask.astype(bool)
    if int(valid.sum()) == 0:
        return {}, 0

    hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0][valid].astype(np.float32)   # OpenCV: 0..179
    s = hsv[:, :, 1][valid].astype(np.float32)   # 0..255
    v = hsv[:, :, 2][valid].astype(np.float32)   # 0..255

    labels = np.full(h.shape, "", dtype=object)

    # Acromáticos primero.
    labels[v < 45] = "black"
    labels[(labels == "") & (v > 205) & (s < 38)] = "white"
    labels[(labels == "") & (s < 42)] = "gray"

    # Marrón: tono cálido con brillo medio/bajo.
    labels[(labels == "") & (h < 15) & (s >= 42) & (v < 175)] = "brown"

    # Cromáticos restantes.
    labels[(labels == "") & ((h < 8) | (h >= 172))] = "red"
    labels[(labels == "") & (h >= 8) & (h < 18)] = "orange"
    labels[(labels == "") & (h >= 18) & (h < 35)] = "yellow"
    labels[(labels == "") & (h >= 35) & (h < 82)] = "green"
    labels[(labels == "") & (h >= 82) & (h < 100)] = "cyan"
    labels[(labels == "") & (h >= 100) & (h < 138)] = "blue"
    labels[(labels == "") & (h >= 138) & (h < 172)] = "purple"
    labels[labels == ""] = UNKNOWN

    counts = Counter(labels.tolist())
    total = int(sum(counts.values()))
    dist = {label: count / total for label, count in counts.items() if total > 0}
    return dist, total


def choose_color_label(dist: Mapping[str, float], min_pixels: int, pixel_count: int, dominance_threshold: float) -> Tuple[str, str, float]:
    if pixel_count < min_pixels or not dist:
        return UNKNOWN, UNKNOWN, 0.0
    filtered = {k: float(v) for k, v in dist.items() if k != UNKNOWN}
    if not filtered:
        return UNKNOWN, UNKNOWN, 0.0
    label, confidence = max(filtered.items(), key=lambda kv: kv[1])
    if confidence < dominance_threshold:
        return "mixed", "mixed", confidence
    return label, COLOR_FAMILY.get(label, UNKNOWN), confidence


def analyze_frame_colors(
    sequence_id: str,
    frame_number: Optional[int],
    rgb_path: Path,
    silhouette_path: Path,
    selected_rank: int,
    quality_frame_valid: Optional[bool],
    occlusion_pct: Optional[float],
    min_color_pixels: int,
    color_dominance_threshold: float,
) -> Optional[FrameColorEvidence]:
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    mask = read_binary_mask(silhouette_path)
    if rgb is None or mask is None:
        return None
    if rgb.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    upper_mask, lower_mask = body_region_masks(mask)
    upper_dist, upper_pixels = hsv_color_histogram(rgb, upper_mask)
    lower_dist, lower_pixels = hsv_color_histogram(rgb, lower_mask)
    upper_label, upper_family, upper_conf = choose_color_label(
        upper_dist, min_color_pixels, upper_pixels, color_dominance_threshold
    )
    lower_label, lower_family, lower_conf = choose_color_label(
        lower_dist, min_color_pixels, lower_pixels, color_dominance_threshold
    )

    return FrameColorEvidence(
        sequence_id=sequence_id,
        frame_number=frame_number,
        rgb_path=str(rgb_path),
        silhouette_path=str(silhouette_path),
        selected_rank=selected_rank,
        quality_frame_valid=quality_frame_valid,
        occlusion_pct=occlusion_pct,
        upper_color_label=upper_label,
        upper_color_family=upper_family,
        upper_color_confidence=float(upper_conf),
        upper_color_pixels=int(upper_pixels),
        lower_color_label=lower_label,
        lower_color_family=lower_family,
        lower_color_confidence=float(lower_conf),
        lower_color_pixels=int(lower_pixels),
    )


# -----------------------------------------------------------------------------
# Anotación semántica semiautomática con imágenes múltiples
# -----------------------------------------------------------------------------

def encode_image_data_url(image_bgr: np.ndarray, quality: int = 88) -> str:
    ok, data = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("No se pudo codificar la imagen JPG")
    payload = base64.b64encode(data.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def make_montage(rgb_paths: Sequence[Path], tile_width: int = 180, label_height: int = 24) -> np.ndarray:
    tiles: List[np.ndarray] = []
    for idx, path in enumerate(rgb_paths, start=1):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            continue
        scale = tile_width / float(w)
        resized = cv2.resize(image, (tile_width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
        label = np.zeros((label_height, tile_width, 3), dtype=np.uint8)
        cv2.putText(label, f"frame {idx}", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.vstack([label, resized]))
    if not tiles:
        raise ValueError("No se pudo construir el montaje: no hay imágenes RGB legibles")
    max_h = max(tile.shape[0] for tile in tiles)
    padded = []
    for tile in tiles:
        if tile.shape[0] < max_h:
            pad = np.zeros((max_h - tile.shape[0], tile.shape[1], 3), dtype=np.uint8)
            tile = np.vstack([tile, pad])
        padded.append(tile)
    return np.hstack(padded)


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"No se encontró un objeto JSON en la respuesta: {text[:300]}")
        return json.loads(match.group(0))


def load_jsonl_cache(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    if path is None or not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if "sequence_id" in row:
                    cache[str(row["sequence_id"])] = row
            except json.JSONDecodeError:
                continue
    return cache


def append_jsonl(path: Optional[Path], row: Dict[str, Any]) -> None:
    if path is None:
        return
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def annotate_semantic_openai(
    sequence_id: str,
    rgb_paths: Sequence[Path],
    model: str,
    detail: str,
) -> SemanticAttributes:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Falta instalar el paquete openai: pip install openai") from exc

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("No se encontró OPENAI_API_KEY en el entorno")

    montage = make_montage(rgb_paths)
    data_url = encode_image_data_url(montage)
    prompt = f"""
Analiza el montaje de varios frames de la MISMA secuencia peatonal ({sequence_id}).
Devuelve exclusivamente un objeto JSON válido con estas claves:
- upper_garment_type: uno de {UPPER_GARMENT_VOCAB}
- lower_garment_type: uno de {LOWER_GARMENT_VOCAB}
- backpack_flag: uno de {BACKPACK_VOCAB}
- semantic_confidence: número entre 0 y 1
- semantic_notes: texto breve; explica dudas visibles sin inventar detalles

Reglas:
1. Usa unknown cuando la prenda o mochila no pueda observarse con suficiente certeza.
2. No infieras mochila cuando solo se observe un bolso de mano u otro accesorio.
3. Si los frames no son consistentes, usa unknown y explica la inconsistencia.
4. Clasifica solo la apariencia visible. No identifiques a la persona.
""".strip()

    client = OpenAI()
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url, "detail": detail},
                ],
            }
        ],
    )
    payload = extract_json_object(response.output_text)
    return SemanticAttributes(
        upper_garment_type=normalize_token(payload.get("upper_garment_type"), UPPER_GARMENT_VOCAB),
        lower_garment_type=normalize_token(payload.get("lower_garment_type"), LOWER_GARMENT_VOCAB),
        backpack_flag=normalize_token(payload.get("backpack_flag"), BACKPACK_VOCAB),
        semantic_confidence=float(np.clip(to_float_or_none(payload.get("semantic_confidence")) or 0.0, 0.0, 1.0)),
        semantic_notes=clean_text(payload.get("semantic_notes")),
        annotation_source=f"openai_multiframe:{model}",
    )


# -----------------------------------------------------------------------------
# Consolidación y overrides
# -----------------------------------------------------------------------------

def aggregate_sequence_colors(frame_rows: Sequence[FrameColorEvidence]) -> Dict[str, Any]:
    upper_label, upper_conf, upper_consensus = weighted_mode(
        (row.upper_color_label, max(row.upper_color_confidence, 1e-6) * max(row.upper_color_pixels, 1))
        for row in frame_rows
    )
    lower_label, lower_conf, lower_consensus = weighted_mode(
        (row.lower_color_label, max(row.lower_color_confidence, 1e-6) * max(row.lower_color_pixels, 1))
        for row in frame_rows
    )
    return {
        "upper_color_label": upper_label,
        "lower_color_label": lower_label,
        "upper_color_family": COLOR_FAMILY.get(upper_label, UNKNOWN),
        "lower_color_family": COLOR_FAMILY.get(lower_label, UNKNOWN),
        "upper_color_confidence": float(upper_conf),
        "lower_color_confidence": float(lower_conf),
        "upper_color_consensus_ratio": float(upper_consensus),
        "lower_color_consensus_ratio": float(lower_consensus),
    }


def apply_sequence_overrides(seq_df: pd.DataFrame, overrides_csv: Optional[str]) -> pd.DataFrame:
    if not overrides_csv:
        return seq_df
    overrides_path = Path(overrides_csv)
    if not overrides_path.exists():
        raise FileNotFoundError(f"No existe el CSV de overrides: {overrides_path}")
    overrides = pd.read_csv(overrides_path)
    if "sequence_id" not in overrides.columns:
        raise ValueError("El CSV de overrides debe incluir sequence_id")

    output = seq_df.copy()
    override_fields = [
        "upper_garment_type",
        "lower_garment_type",
        "backpack_flag",
        "upper_color_label",
        "lower_color_label",
    ]
    overrides = overrides.drop_duplicates("sequence_id", keep="last").set_index("sequence_id")
    output = output.set_index("sequence_id")
    for field in override_fields:
        if field not in overrides.columns:
            continue
        for sequence_id, value in overrides[field].items():
            if sequence_id not in output.index or pd.isna(value) or not clean_text(value):
                continue
            if field == "upper_garment_type":
                normalized = normalize_token(value, UPPER_GARMENT_VOCAB)
            elif field == "lower_garment_type":
                normalized = normalize_token(value, LOWER_GARMENT_VOCAB)
            elif field == "backpack_flag":
                normalized = normalize_token(value, BACKPACK_VOCAB)
            else:
                normalized = normalize_token(value, COLOR_VOCAB)
            output.loc[sequence_id, field] = normalized
            output.loc[sequence_id, "manual_override_flag"] = 1
            output.loc[sequence_id, "annotation_source"] = "manual_override"
    return output.reset_index()


def consensus_from_series(series: pd.Series, weights: Optional[pd.Series] = None) -> Tuple[str, float]:
    rows: List[Tuple[str, float]] = []
    for idx, value in series.items():
        weight = 1.0 if weights is None else float(weights.loc[idx])
        rows.append((clean_text(value).lower() or UNKNOWN, weight))
    label, _, consensus = weighted_mode(rows)
    return label, consensus


def consolidate_identity_rows(seq_df: pd.DataFrame, group_cols: Sequence[str], min_consensus: float) -> pd.DataFrame:
    missing = [col for col in group_cols if col not in seq_df.columns]
    if missing:
        raise ValueError(f"No existen columnas de agrupación requeridas: {missing}")

    identity_rows: List[Dict[str, Any]] = []
    grouped = seq_df.groupby(list(group_cols), dropna=False, sort=True)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        row["identity_key"] = "|".join(str(row[col]) for col in group_cols)
        row["n_sequences"] = int(len(group))

        semantic_weights = group["semantic_confidence"].fillna(0.0).clip(lower=0.05)
        color_upper_weights = group["upper_color_confidence"].fillna(0.0).clip(lower=0.05)
        color_lower_weights = group["lower_color_confidence"].fillna(0.0).clip(lower=0.05)

        row["upper_garment_type"], row["upper_garment_consensus_ratio"] = consensus_from_series(
            group["upper_garment_type"], semantic_weights
        )
        row["lower_garment_type"], row["lower_garment_consensus_ratio"] = consensus_from_series(
            group["lower_garment_type"], semantic_weights
        )
        row["backpack_flag"], row["backpack_consensus_ratio"] = consensus_from_series(
            group["backpack_flag"], semantic_weights
        )
        row["upper_color_label"], row["upper_color_consensus_ratio"] = consensus_from_series(
            group["upper_color_label"], color_upper_weights
        )
        row["lower_color_label"], row["lower_color_consensus_ratio"] = consensus_from_series(
            group["lower_color_label"], color_lower_weights
        )

        row["upper_color_family"] = COLOR_FAMILY.get(row["upper_color_label"], UNKNOWN)
        row["lower_color_family"] = COLOR_FAMILY.get(row["lower_color_label"], UNKNOWN)
        row["has_backpack"] = 1 if row["backpack_flag"] == "yes" else 0
        row["no_backpack"] = 1 if row["backpack_flag"] == "no" else 0
        row["backpack_known"] = 1 if row["backpack_flag"] in {"yes", "no"} else 0

        consensus_fields = [
            "upper_garment_consensus_ratio",
            "lower_garment_consensus_ratio",
            "backpack_consensus_ratio",
            "upper_color_consensus_ratio",
            "lower_color_consensus_ratio",
        ]
        has_unknown = any(row[field] == UNKNOWN for field in [
            "upper_garment_type",
            "lower_garment_type",
            "backpack_flag",
            "upper_color_label",
            "lower_color_label",
        ])
        low_consensus = any(float(row[field]) < min_consensus for field in consensus_fields)
        row["manual_review_flag"] = int(has_unknown or low_consensus or bool(group["manual_review_flag"].max()))
        row["source_sequence_ids_json"] = json.dumps(group["sequence_id"].astype(str).tolist(), ensure_ascii=False)
        identity_rows.append(row)
    return pd.DataFrame(identity_rows)


def write_pair_schema(path: str | Path, identity_group_cols: Sequence[str]) -> None:
    payload = {
        "purpose": "Derived pair features to be computed later from identity_visual_attributes_pair_ready.csv",
        "identity_group_cols": list(identity_group_cols),
        "source_identity_fields": [
            "upper_garment_type",
            "lower_garment_type",
            "upper_color_label",
            "lower_color_label",
            "backpack_flag",
            "has_backpack",
            "no_backpack",
            "backpack_known",
        ],
        "future_pair_features": {
            "same_upper_garment_type": "1 if x.upper_garment_type == y.upper_garment_type and neither value is unknown; else 0 or missing when unavailable",
            "same_lower_garment_type": "1 if x.lower_garment_type == y.lower_garment_type and neither value is unknown; else 0 or missing when unavailable",
            "same_upper_color_label": "1 if x.upper_color_label == y.upper_color_label and neither value is unknown; else 0 or missing when unavailable",
            "same_lower_color_label": "1 if x.lower_color_label == y.lower_color_label and neither value is unknown; else 0 or missing when unavailable",
            "both_have_backpack": "1 if x.backpack_flag == 'yes' and y.backpack_flag == 'yes'; else 0; missing if backpack is unknown for either identity",
            "both_no_backpack": "1 if x.backpack_flag == 'no' and y.backpack_flag == 'no'; else 0; missing if backpack is unknown for either identity",
        },
        "note": "Do not treat unknown values as equality. Preserve missingness when a pair cannot be compared.",
    }
    ensure_parent(path)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


# -----------------------------------------------------------------------------
# Pipeline principal
# -----------------------------------------------------------------------------

def merge_optional_frame_metadata(
    frames_df: pd.DataFrame,
    quality_csv: Optional[str],
    occlusion_csv: Optional[str],
) -> pd.DataFrame:
    output = frames_df.copy()
    join_cols = [col for col in ["sequence_id", "frame_number"] if col in output.columns]
    if len(join_cols) < 2:
        raise ValueError("frames-csv debe incluir sequence_id y frame_number")

    if quality_csv:
        quality = pd.read_csv(quality_csv)
        keep = [col for col in ["sequence_id", "frame_number", "frame_valid", "mca_valid"] if col in quality.columns]
        if {"sequence_id", "frame_number"}.issubset(keep):
            quality = quality[keep].drop_duplicates(["sequence_id", "frame_number"], keep="last")
            output = output.merge(quality, on=["sequence_id", "frame_number"], how="left")

    if occlusion_csv:
        occ = pd.read_csv(occlusion_csv)
        occ_col = first_existing_column(occ, ["occlusion_pct", "frame_occlusion_pct"], required=False)
        keep = [col for col in ["sequence_id", "frame_number", occ_col] if col and col in occ.columns]
        if occ_col and {"sequence_id", "frame_number"}.issubset(keep):
            occ = occ[keep].drop_duplicates(["sequence_id", "frame_number"], keep="last")
            if occ_col != "occlusion_pct":
                occ = occ.rename(columns={occ_col: "occlusion_pct"})
            output = output.merge(occ, on=["sequence_id", "frame_number"], how="left")
    return output


def select_representative_frames(
    group: pd.DataFrame,
    sample_frames: int,
    max_occlusion_pct: float,
) -> pd.DataFrame:
    candidates = group.copy().sort_values("frame_number")

    if "frame_valid" in candidates.columns:
        valid_mask = candidates["frame_valid"].map(to_bool_or_none)
        if valid_mask.notna().any():
            preferred = candidates[valid_mask.fillna(False)]
            if len(preferred) > 0:
                candidates = preferred
    elif "mca_valid" in candidates.columns:
        valid_mask = candidates["mca_valid"].map(to_bool_or_none)
        if valid_mask.notna().any():
            preferred = candidates[valid_mask.fillna(False)]
            if len(preferred) > 0:
                candidates = preferred

    if "occlusion_pct" in candidates.columns:
        occ = pd.to_numeric(candidates["occlusion_pct"], errors="coerce")
        preferred = candidates[(occ.isna()) | (occ <= max_occlusion_pct)]
        if len(preferred) > 0:
            candidates = preferred

    idx = equidistant_indices(len(candidates), sample_frames)
    return candidates.iloc[idx].copy() if idx else candidates.head(0).copy()


def main(args: argparse.Namespace) -> None:
    base_dir = Path(args.base_dir).expanduser().resolve()
    sequences = pd.read_csv(args.sequences_csv)
    frames = pd.read_csv(args.frames_csv)

    if "sequence_id" not in sequences.columns or "sequence_id" not in frames.columns:
        raise ValueError("Ambos CSV deben incluir sequence_id")

    rgb_col = first_existing_column(frames, ["rgb_source_path", "rgb_output_path", "rgb_path"])
    sil_col = first_existing_column(frames, ["silhouette_source_path", "silhouette_output_path", "silhouette_path"])
    frames = merge_optional_frame_metadata(frames, args.sil_quality_frames_csv, args.occlusion_frames_csv)

    overrides_cache = load_jsonl_cache(Path(args.cache_jsonl).resolve() if args.cache_jsonl else None)
    seq_metadata_lookup = sequences.drop_duplicates("sequence_id", keep="last").set_index("sequence_id")

    output_sequence_rows: List[Dict[str, Any]] = []
    output_frame_rows: List[Dict[str, Any]] = []

    grouped = frames.groupby("sequence_id", sort=True)
    iterator = tqdm(grouped, total=grouped.ngroups, desc="Preparando atributos visuales")
    for sequence_id, group in iterator:
        sequence_id = str(sequence_id)
        selected = select_representative_frames(group, args.sample_frames, args.max_occlusion_pct)
        frame_evidence: List[FrameColorEvidence] = []
        selected_rgb_paths: List[Path] = []

        for selected_rank, (_, frame_row) in enumerate(selected.iterrows(), start=1):
            rgb_path = resolve_path(base_dir, frame_row[rgb_col])
            sil_path = resolve_path(base_dir, frame_row[sil_col])
            if rgb_path is None or sil_path is None or not rgb_path.exists() or not sil_path.exists():
                continue
            evidence = analyze_frame_colors(
                sequence_id=sequence_id,
                frame_number=int(frame_row["frame_number"]) if not pd.isna(frame_row["frame_number"]) else None,
                rgb_path=rgb_path,
                silhouette_path=sil_path,
                selected_rank=selected_rank,
                quality_frame_valid=to_bool_or_none(frame_row.get("frame_valid", frame_row.get("mca_valid"))),
                occlusion_pct=to_float_or_none(frame_row.get("occlusion_pct")),
                min_color_pixels=args.min_color_pixels,
                color_dominance_threshold=args.color_dominance_threshold,
            )
            if evidence is not None:
                frame_evidence.append(evidence)
                output_frame_rows.append(asdict(evidence))
                selected_rgb_paths.append(rgb_path)

        color_values = aggregate_sequence_colors(frame_evidence)

        if sequence_id in overrides_cache:
            cached = overrides_cache[sequence_id]
            semantic = SemanticAttributes(
                upper_garment_type=normalize_token(cached.get("upper_garment_type"), UPPER_GARMENT_VOCAB),
                lower_garment_type=normalize_token(cached.get("lower_garment_type"), LOWER_GARMENT_VOCAB),
                backpack_flag=normalize_token(cached.get("backpack_flag"), BACKPACK_VOCAB),
                semantic_confidence=float(np.clip(to_float_or_none(cached.get("semantic_confidence")) or 0.0, 0.0, 1.0)),
                semantic_notes=clean_text(cached.get("semantic_notes")),
                annotation_source=clean_text(cached.get("annotation_source")) or "cache",
            )
        elif args.annotation_mode == "openai" and selected_rgb_paths:
            try:
                semantic = annotate_semantic_openai(sequence_id, selected_rgb_paths, args.model, args.image_detail)
                append_jsonl(Path(args.cache_jsonl).resolve() if args.cache_jsonl else None, {
                    "sequence_id": sequence_id,
                    **asdict(semantic),
                })
            except Exception as exc:  # pipeline robusto: registrar error y continuar
                semantic = SemanticAttributes(
                    semantic_notes=f"annotation_error: {type(exc).__name__}: {exc}",
                    annotation_source="openai_error",
                )
        else:
            semantic = SemanticAttributes(annotation_source="manual_required")

        base: Dict[str, Any] = {"sequence_id": sequence_id}
        if sequence_id in seq_metadata_lookup.index:
            source_row = seq_metadata_lookup.loc[sequence_id]
            if isinstance(source_row, pd.DataFrame):
                source_row = source_row.iloc[-1]
            for col in ["person_global_id", "scene_id", "cam_id", "cycle_index", "tracklet_index", "sequence_index"]:
                if col in source_row.index:
                    base[col] = source_row[col]

        has_unknown = any(value == UNKNOWN for value in [
            semantic.upper_garment_type,
            semantic.lower_garment_type,
            semantic.backpack_flag,
            color_values["upper_color_label"],
            color_values["lower_color_label"],
        ])
        low_conf = (
            semantic.semantic_confidence < args.min_semantic_confidence
            or color_values["upper_color_consensus_ratio"] < args.min_color_consensus
            or color_values["lower_color_consensus_ratio"] < args.min_color_consensus
        )
        row = {
            **base,
            **asdict(semantic),
            **color_values,
            "n_selected_frames": int(len(frame_evidence)),
            "selected_frame_numbers_json": json.dumps([r.frame_number for r in frame_evidence], ensure_ascii=False),
            "has_backpack": 1 if semantic.backpack_flag == "yes" else 0,
            "no_backpack": 1 if semantic.backpack_flag == "no" else 0,
            "backpack_known": 1 if semantic.backpack_flag in {"yes", "no"} else 0,
            "manual_override_flag": 0,
            "manual_review_flag": int(has_unknown or low_conf),
        }
        output_sequence_rows.append(row)

        if args.limit and len(output_sequence_rows) >= args.limit:
            break

    sequence_output = pd.DataFrame(output_sequence_rows)
    sequence_output = apply_sequence_overrides(sequence_output, args.manual_overrides_csv)

    group_cols = [item.strip() for item in args.identity_group_cols.split(",") if item.strip()]
    identity_output = consolidate_identity_rows(sequence_output, group_cols, args.min_identity_consensus)

    ensure_parent(args.out_sequence_csv)
    ensure_parent(args.out_identity_csv)
    ensure_parent(args.out_frame_csv)
    ensure_parent(args.review_csv)
    sequence_output.to_csv(args.out_sequence_csv, index=False)
    identity_output.to_csv(args.out_identity_csv, index=False)
    pd.DataFrame(output_frame_rows).to_csv(args.out_frame_csv, index=False)

    review_columns = [col for col in [
        "sequence_id",
        "person_global_id",
        "scene_id",
        "cam_id",
        "upper_garment_type",
        "lower_garment_type",
        "backpack_flag",
        "upper_color_label",
        "lower_color_label",
        "semantic_confidence",
        "upper_color_consensus_ratio",
        "lower_color_consensus_ratio",
        "manual_review_flag",
        "semantic_notes",
        "selected_frame_numbers_json",
    ] if col in sequence_output.columns]
    sequence_output[sequence_output["manual_review_flag"] == 1][review_columns].to_csv(args.review_csv, index=False)

    write_pair_schema(args.pair_schema_json, group_cols)

    print("\nProceso finalizado")
    print(f"Secuencias procesadas: {len(sequence_output)}")
    print(f"Identidades/grupos consolidados: {len(identity_output)}")
    print(f"Secuencias para revisión manual: {int(sequence_output['manual_review_flag'].sum())}")
    print(f"Metadata por secuencia: {args.out_sequence_csv}")
    print(f"Metadata por identidad: {args.out_identity_csv}")
    print(f"Frames muestreados: {args.out_frame_csv}")
    print(f"Cola de revisión: {args.review_csv}")
    print(f"Esquema para cálculo futuro de pares: {args.pair_schema_json}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepara atributos visuales por secuencia e identidad para análisis posterior de pares."
    )
    parser.add_argument("--base-dir", required=True, help="Directorio base para resolver rutas relativas")
    parser.add_argument("--sequences-csv", required=True, help="Metadata por secuencia/ciclo")
    parser.add_argument("--frames-csv", required=True, help="Metadata por frame con rutas RGB y silueta")
    parser.add_argument("--sil-quality-frames-csv", default=None, help="CSV opcional con frame_valid o mca_valid")
    parser.add_argument("--occlusion-frames-csv", default=None, help="CSV opcional con occlusion_pct por frame")

    parser.add_argument("--out-sequence-csv", required=True, help="Salida de atributos observados por secuencia")
    parser.add_argument("--out-identity-csv", required=True, help="Salida consolidada pair-ready por identidad o identidad-sesión")
    parser.add_argument("--out-frame-csv", required=True, help="Salida de evidencia cromática por frame muestreado")
    parser.add_argument("--review-csv", required=True, help="Salida de secuencias que requieren revisión manual")
    parser.add_argument("--pair-schema-json", required=True, help="JSON con definición de variables derivadas para futuros pares")

    parser.add_argument("--annotation-mode", choices=["openai", "manual_only"], default="manual_only",
                        help="openai: preanotación semiautomática; manual_only: deja prendas/mochila como unknown")
    parser.add_argument("--model", default="gpt-4.1-mini", help="Modelo multimodal configurable para preanotación")
    parser.add_argument("--image-detail", choices=["low", "high", "auto"], default="auto")
    parser.add_argument("--cache-jsonl", default=None, help="Cache JSONL opcional para no repetir inferencias")
    parser.add_argument("--manual-overrides-csv", default=None,
                        help="CSV opcional de correcciones manuales por sequence_id")

    parser.add_argument("--sample-frames", type=int, default=5, help="Frames representativos por secuencia")
    parser.add_argument("--max-occlusion-pct", type=float, default=60.0,
                        help="Prioriza frames con oclusión menor o igual a este valor")
    parser.add_argument("--min-color-pixels", type=int, default=80,
                        help="Mínimo de píxeles en una región para asignar color")
    parser.add_argument("--color-dominance-threshold", type=float, default=0.45,
                        help="Dominancia mínima para una etiqueta cromática; por debajo se asigna mixed")
    parser.add_argument("--min-semantic-confidence", type=float, default=0.60)
    parser.add_argument("--min-color-consensus", type=float, default=0.55)
    parser.add_argument("--identity-group-cols", default="person_global_id",
                        help="Columnas para consolidar atributos. Use person_global_id,scene_id si la ropa cambia por sesión")
    parser.add_argument("--min-identity-consensus", type=float, default=0.60)
    parser.add_argument("--limit", type=int, default=None, help="Procesar solo N secuencias para prueba")
    return parser


if __name__ == "__main__":
    try:
        main(build_parser().parse_args())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
