#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
from pathlib import Path

import pandas as pd


# ===== Utilidades de parsing/normalización =====

TIME_RE = re.compile(r"^\s*(\d+):(\d+):(\d+(?:\.\d+)?)\s*$")

def hms_to_seconds(hms: str) -> float:
    """
    Convierte 'HH:MM:SS' o 'HH:MM:SS.mmm' a segundos (float).
    """
    m = TIME_RE.match(str(hms))
    if not m:
        raise ValueError(f"Formato de tiempo inválido: {hms!r}. Esperado HH:MM:SS o HH:MM:SS.mmm")
    hh = int(m.group(1))
    mm = int(m.group(2))
    ss = float(m.group(3))
    return hh * 3600 + mm * 60 + ss

def seconds_to_hms(seconds: float) -> str:
    """
    Convierte segundos a string HH:MM:SS.mmm (ffmpeg acepta).
    """
    if seconds < 0:
        seconds = 0
    hh = int(seconds // 3600)
    rem = seconds - hh * 3600
    mm = int(rem // 60)
    ss = rem - mm * 60
    # Mantener milisegundos si existieran
    return f"{hh:02d}:{mm:02d}:{ss:06.3f}".rstrip("0").rstrip(".")

def normalize_scene(scene_raw: str) -> str:
    """
    Acepta 'S01', 's1', 'Scene 1', etc. Devuelve 'S01' (2 dígitos).
    """
    s = str(scene_raw).strip()
    m = re.search(r"(\d+)", s)
    if not m:
        raise ValueError(f"No pude extraer número de escena desde: {scene_raw!r}")
    n = int(m.group(1))
    return f"S{n:02d}"

def normalize_cam(cam_raw: str) -> str:
    """
    Acepta 'c001', 'C1', 'cam 1', etc. Devuelve 'c001' (3 dígitos).
    """
    s = str(cam_raw).strip()
    m = re.search(r"(\d+)", s)
    if not m:
        raise ValueError(f"No pude extraer número de cámara desde: {cam_raw!r}")
    n = int(m.group(1))
    return f"c{n:03d}"

def unique_output_path(base_path: Path, overwrite: bool) -> Path:
    """
    Si overwrite=False y el archivo existe, agrega sufijos _02, _03, ...
    """
    if overwrite or not base_path.exists():
        return base_path

    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent

    i = 2
    while True:
        candidate = parent / f"{stem}_{i:02d}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


# ===== Corte con ffmpeg =====

def run_ffmpeg_cut(
    input_path: Path,
    output_path: Path,
    start_sec: float,
    end_sec: float,
    reencode: bool,
    dry_run: bool,
) -> None:
    if end_sec <= start_sec:
        raise ValueError(f"to_time debe ser > from_time. Recibido: start={start_sec}, end={end_sec}")

    duration_sec = end_sec - start_sec

    # Para robustez: usar -ss (inicio) y -t (duración).
    # -c copy es rápido, pero puede no ser frame-accurate si el inicio no cae en keyframe.
    # --reencode fuerza precisión (más lento).
    cmd = ["ffmpeg", "-y"]  # -y para que ffmpeg no pregunte; nosotros controlamos overwrite por nombre/colisión
    cmd += ["-hide_banner", "-loglevel", "error"]

    # Seek rápido: -ss antes de -i
    cmd += ["-ss", seconds_to_hms(start_sec)]
    cmd += ["-i", str(input_path)]
    cmd += ["-t", seconds_to_hms(duration_sec)]

    if reencode:
        # Re-encode para cortes precisos
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac", "-b:a", "192k"]
    else:
        # Stream copy (rápido)
        cmd += ["-c", "copy"]

    # Evita timestamps negativos ocasionales
    cmd += ["-avoid_negative_ts", "make_zero", str(output_path)]

    print("FFmpeg:", " ".join(cmd))
    if dry_run:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)


# ===== Pipeline principal =====

def build_input_path(input_root: Path, folder: str, filename: str) -> Path:
    return input_root / str(folder) / str(filename)

def process_rows(
    df: pd.DataFrame,
    input_root: Path,
    output_root: Path,
    overwrite: bool,
    reencode: bool,
    dry_run: bool,
) -> None:
    required_cols = [
        "folder",
        "filename",
        "scene",
        "cam number",
        "from_time (HH:MM:SS)",
        "to_time (HH:MM:SS)",
    ]
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"Falta columna requerida en el CSV: {c!r}. Columnas disponibles: {list(df.columns)}")

    total = len(df)
    ok = 0
    skipped = 0

    for idx, row in df.iterrows():
        scene = normalize_scene(row["scene"])
        cam = normalize_cam(row["cam number"])

        in_path = build_input_path(input_root, row["folder"], row["filename"])
        if not in_path.exists():
            skipped += 1
            print(f"[SKIP idx={idx}] No existe el archivo de entrada: {in_path}")
            continue

        start_sec = hms_to_seconds(row["from_time (HH:MM:SS)"])
        end_sec = hms_to_seconds(row["to_time (HH:MM:SS)"])

        out_dir = output_root / scene / cam
        out_base = out_dir / f"{scene}_{cam}.mp4"
        out_path = unique_output_path(out_base, overwrite=overwrite)

        try:
            print(f"[{idx+1}/{total}] {in_path.name} -> {out_path.relative_to(output_root)} ({row['from_time (HH:MM:SS)']}–{row['to_time (HH:MM:SS)']})")
            run_ffmpeg_cut(
                input_path=in_path,
                output_path=out_path,
                start_sec=start_sec,
                end_sec=end_sec,
                reencode=reencode,
                dry_run=dry_run,
            )
            ok += 1
        except Exception as e:
            print(f"[ERROR idx={idx}] {e}")
            # continúa con el siguiente

    print(f"\nResumen: OK={ok}, SKIP(no input)={skipped}, TOTAL={total}")

def filter_for_test(
    df: pd.DataFrame,
    row_index: int | None,
    scene: str | None,
    cam: str | None,
    filename: str | None,
) -> pd.DataFrame:
    if row_index is not None:
        if row_index < 0 or row_index >= len(df):
            raise IndexError(f"--row-index fuera de rango. Recibido {row_index}, pero el CSV tiene {len(df)} filas.")
        return df.iloc[[row_index]].copy()

    mask = pd.Series([True] * len(df))

    if scene is not None:
        ns = normalize_scene(scene)
        mask &= df["scene"].astype(str).apply(lambda x: normalize_scene(x) == ns)

    if cam is not None:
        nc = normalize_cam(cam)
        mask &= df["cam number"].astype(str).apply(lambda x: normalize_cam(x) == nc)

    if filename is not None:
        mask &= df["filename"].astype(str) == str(filename)

    filtered = df[mask].copy()
    if len(filtered) == 0:
        raise ValueError("Modo test: no encontré filas que coincidan con los filtros provistos.")
    if len(filtered) > 1:
        # En test, tomamos la primera por defecto; puedes apuntar con --row-index si quieres una exacta.
        filtered = filtered.iloc[[0]].copy()
    return filtered


def main():
    ap = argparse.ArgumentParser(
        description="Corta videos según CSV (from_time/to_time) y guarda como S0x/c00x/S0x_c00x.mp4"
    )
    ap.add_argument("--csv", required=True, help="Ruta al CSV")
    ap.add_argument("--input-root", required=True, help="Directorio raíz donde están las carpetas de 'folder' del CSV")
    ap.add_argument("--output-root", required=True, help="Directorio raíz de salida")

    ap.add_argument("--reencode", action="store_true", help="Re-encode para cortes más precisos (más lento). Default: stream copy.")
    ap.add_argument("--overwrite", action="store_true", help="Sobrescribe el nombre base si existe. Default: crea sufijo _02, _03, ...")
    ap.add_argument("--dry-run", action="store_true", help="No ejecuta ffmpeg; solo imprime lo que haría.")

    # Modo test (procesa 1 fila)
    ap.add_argument("--test", action="store_true", help="Procesa solo 1 video (según filtros o primera coincidencia).")
    ap.add_argument("--row-index", type=int, default=None, help="Índice (0-based) de la fila a procesar en modo test.")
    ap.add_argument("--scene", default=None, help="Filtro escena para test (ej: S01 o 1).")
    ap.add_argument("--cam", default=None, help="Filtro cámara para test (ej: c001 o 1).")
    ap.add_argument("--filename", default=None, help="Filtro filename exacto para test (ej: cam1_....mp4).")

    args = ap.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    df = pd.read_csv(csv_path)

    if args.test:
        df = filter_for_test(df, args.row_index, args.scene, args.cam, args.filename)
        print(f"Modo TEST: procesando {len(df)} fila(s).")

    process_rows(
        df=df,
        input_root=input_root,
        output_root=output_root,
        overwrite=args.overwrite,
        reencode=args.reencode,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
