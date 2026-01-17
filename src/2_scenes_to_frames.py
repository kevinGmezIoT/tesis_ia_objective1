#!/usr/bin/env python3
import argparse
import math
import re
import sys
from pathlib import Path

import cv2
import pandas as pd


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def normalize_scene(scene_raw: str) -> str:
    s = str(scene_raw).strip()
    m = re.search(r"(\d+)", s)
    if not m:
        raise ValueError(f"No pude extraer número de escena desde: {scene_raw!r}")
    n = int(m.group(1))
    return f"S{n:02d}"


def normalize_cam(cam_raw: str) -> str:
    s = str(cam_raw).strip()
    m = re.search(r"(\d+)", s)
    if not m:
        raise ValueError(f"No pude extraer número de cámara desde: {cam_raw!r}")
    n = int(m.group(1))
    return f"c{n:03d}"


def _compute_total_to_save(total_frames: int, stride: int, max_frames: int | None) -> int | None:
    """
    Estima cuántos frames se van a guardar.
    Si total_frames no es válido, retorna None.
    """
    if total_frames is None or total_frames <= 0:
        return None

    stride = max(1, stride)

    # Guardamos frames en índices 1, 1+stride, 1+2*stride... => ceil(total/stride)
    expected = math.ceil(total_frames / stride)

    if max_frames is not None:
        expected = min(expected, max_frames)

    return expected


def extract_frames(
    video_path: Path,
    out_dir: Path,
    stride: int = 1,
    max_frames: int | None = None,
    progress_every: int = 100,
) -> int:
    """
    Extrae frames y muestra progreso por video.
    Guarda como 000001.jpg, 000002.jpg...
    """
    ensure_dir(out_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No pude abrir el video: {video_path}")

    total_frames_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_to_save = _compute_total_to_save(total_frames_raw, stride, max_frames)

    saved = 0
    read_idx = 0  # frames leídos del stream (no necesariamente guardados)

    # Encabezado inicial
    if total_to_save is not None:
        print(f"  Progreso: 0/{total_to_save} guardados (faltan {total_to_save})", end="\r", flush=True)
    else:
        print(f"  Progreso: 0 guardados (total desconocido)", end="\r", flush=True)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        read_idx += 1

        if stride > 1 and (read_idx - 1) % stride != 0:
            continue

        saved += 1
        out_path = out_dir / f"{saved:06d}.jpg"
        cv2.imwrite(str(out_path), frame)

        # Corte por debug
        if max_frames is not None and saved >= max_frames:
            break

        # Progreso cada N frames guardados
        if progress_every > 0 and (saved % progress_every == 0):
            if total_to_save is not None:
                remaining = max(total_to_save - saved, 0)
                pct = (saved / total_to_save) * 100 if total_to_save else 0.0
                print(
                    f"  Progreso: {saved}/{total_to_save} guardados "
                    f"(faltan {remaining}) [{pct:6.2f}%]",
                    end="\r",
                    flush=True,
                )
            else:
                print(f"  Progreso: {saved} guardados (total desconocido)", end="\r", flush=True)

    cap.release()

    # Línea final (con salto de línea para no “pisar” el siguiente log)
    if total_to_save is not None:
        remaining = max(total_to_save - saved, 0)
        pct = (saved / total_to_save) * 100 if total_to_save else 100.0
        print(
            f"  Progreso: {saved}/{total_to_save} guardados "
            f"(faltan {remaining}) [{pct:6.2f}%]            "
        )
    else:
        print(f"  Progreso: {saved} guardados (total desconocido)            ")

    return saved


def filter_for_test(df: pd.DataFrame, row_index: int | None, scene: str | None, cam: str | None) -> pd.DataFrame:
    if row_index is not None:
        if row_index < 0 or row_index >= len(df):
            raise IndexError(f"--row-index fuera de rango (0..{len(df)-1}). Recibido: {row_index}")
        return df.iloc[[row_index]].copy()

    mask = pd.Series([True] * len(df))
    if scene is not None:
        ns = normalize_scene(scene)
        mask &= df["scene"].astype(str).apply(lambda x: normalize_scene(x) == ns)
    if cam is not None:
        nc = normalize_cam(cam)
        mask &= df["cam number"].astype(str).apply(lambda x: normalize_cam(x) == nc)

    out = df[mask].copy()
    if len(out) == 0:
        raise ValueError("Modo test: no encontré filas que coincidan con los filtros.")
    return out.iloc[[0]].copy()


def main():
    ap = argparse.ArgumentParser(description="Extrae frames por escena/cámara leyendo un CSV (con progreso por video).")
    ap.add_argument("--csv", required=True, help="Ruta al CSV (con columnas 'scene' y 'cam number').")
    ap.add_argument("--scenes-root", required=True, help="Ruta raíz donde están los videos por escena/cámara.")
    ap.add_argument("--output-root", required=True, help="Ruta raíz de salida para las imágenes.")
    ap.add_argument("--video-ext", default=".mp4", help="Extensión de video (default: .mp4).")
    ap.add_argument("--stride", type=int, default=1, help="Guarda 1 de cada N frames (default: 1 = todos).")
    ap.add_argument("--max-frames", type=int, default=None, help="Límite máximo de frames a guardar (debug).")
    ap.add_argument("--progress-every", type=int, default=100, help="Actualiza el progreso cada N frames guardados.")

    # Modo test
    ap.add_argument("--test", action="store_true", help="Procesa solo una escena/cámara.")
    ap.add_argument("--row-index", type=int, default=None, help="Índice (0-based) de la fila a usar en modo test.")
    ap.add_argument("--scene", default=None, help="Filtro escena para test (ej: S01 o 1).")
    ap.add_argument("--cam", default=None, help="Filtro cámara para test (ej: c001 o 1).")

    args = ap.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    scenes_root = Path(args.scenes_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    df = pd.read_csv(csv_path)

    for col in ("scene", "cam number"):
        if col not in df.columns:
            raise KeyError(f"Falta columna requerida en el CSV: {col!r}. Columnas disponibles: {list(df.columns)}")

    if args.test:
        df = filter_for_test(df, args.row_index, args.scene, args.cam)
        print(f"Modo TEST: usando fila índice {df.index[0]}")

    # Procesar cada (scene, cam) una sola vez
    pairs = sorted({(normalize_scene(r["scene"]), normalize_cam(r["cam number"])) for _, r in df.iterrows()})

    total = len(pairs)
    ok = 0
    skipped = 0

    for i, (scene, cam) in enumerate(pairs, start=1):
        video_path = scenes_root / scene / cam / f"{scene}_{cam}{args.video_ext}"
        out_dir = output_root / scene / cam

        if not video_path.exists():
            skipped += 1
            print(f"[SKIP {i}/{total}] No existe: {video_path}")
            continue

        try:
            print(f"[{i}/{total}] Extrayendo frames: {video_path} -> {out_dir} (stride={args.stride})")
            n = extract_frames(
                video_path,
                out_dir,
                stride=max(1, args.stride),
                max_frames=args.max_frames,
                progress_every=max(1, args.progress_every),
            )
            print(f"  Guardados: {n} frames en {out_dir}\n")
            ok += 1
        except Exception as e:
            print(f"[ERROR {i}/{total}] {scene}/{cam}: {e}\n", file=sys.stderr)

        if args.test:
            break

    print(f"Resumen: OK={ok}, SKIP(no video)={skipped}, TOTAL={total}")


if __name__ == "__main__":
    main()
