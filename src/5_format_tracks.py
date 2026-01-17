###
# Full
# python ./src/5_format_tracks.py --csv ./inputs/raw_videos/input_data.csv --tracking-root ./transforms/tracking --out-root ./transforms/format_tracks
# Test
# python ./src/5_format_tracks.py --csv ./inputs/raw_videos/input_data.csv --tracking-root ./transforms/tracking --out-root ./transforms/format_tracks --test --scene S05 --cam c001
###
import os
import csv
import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd


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


def filter_for_test(df: pd.DataFrame, row_index: Optional[int], scene: Optional[str], cam: Optional[str]) -> pd.DataFrame:
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


def to_int(x: str) -> int:
    return int(float(x))


def default_input_track_path(tracking_root: Path, scene: str, cam: str, track_subdir: str) -> Path:
    # tracking_root/S0x/c00x/output/track/c00x.txt
    return tracking_root / scene / cam / "output" / track_subdir / f"{cam}.txt"


def default_output_track_path(out_root: Path, scene: str, cam: str) -> Path:
    # out_root/format_track/S0x/c00x/c00x.txt
    return out_root / scene / cam / f"{cam}.txt"


def filter_track_file(in_path: Path, out_path: Path, min_area: float) -> int:
    """
    Lee MOT tracking:
      frame,id,x,y,w,h,conf,-1,-1,-1
    Filtra por area w*h >= min_area
    Guarda:
      frame,id,x,y,w,h,1,-1,-1,-1  (x,y,w,h enteros)
    Retorna #filas escritas.
    """
    if not in_path.is_file():
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)

    by_frame = {}  # frm -> {gid: (x,y,w,h)}
    kept = 0

    with in_path.open("r", newline="") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            if not row or len(row) < 6:
                continue

            frm = int(float(row[0]))
            gid = int(float(row[1]))
            x = float(row[2]); y = float(row[3])
            w = float(row[4]); h = float(row[5])

            if (w * h) < min_area:
                continue

            if frm not in by_frame:
                by_frame[frm] = {}
            by_frame[frm][gid] = (to_int(row[2]), to_int(row[3]), to_int(row[4]), to_int(row[5]))
            kept += 1

    with out_path.open("w", newline="") as f_out:
        for frm in sorted(by_frame.keys()):
            for gid in sorted(by_frame[frm].keys()):
                x, y, w, h = by_frame[frm][gid]
                f_out.write(f"{frm},{gid},{x},{y},{w},{h},1,-1,-1,-1\n")

    return kept


def main():
    parser = argparse.ArgumentParser(description="Filtra tracks (MOT .txt) por area mínima, leyendo secuencias desde CSV.")
    parser.add_argument("--csv", required=True, help="Ruta al CSV (columnas: 'scene' y 'cam number').")

    parser.add_argument("--tracking-root", required=True,
                        help="Raíz donde están los resultados del tracking por secuencia (S0x/c00x).")
    parser.add_argument("--out-root", required=True,
                        help="Raíz de salida. Se guardará en out-root/format_track/S0x/c00x/c00x.txt")

    parser.add_argument("--track-subdir", default="track",
                        help="Subcarpeta dentro de output/ que contiene los .txt (default: track).")

    parser.add_argument("--min-area", type=float, default=1000.0,
                        help="Área mínima (w*h) para conservar bbox (default: 1000).")

    # Test
    parser.add_argument("--test", action="store_true", help="Procesa solo 1 secuencia.")
    parser.add_argument("--row-index", type=int, default=None, help="Fila CSV (0-based) para test.")
    parser.add_argument("--scene", default=None, help="Filtro escena para test (ej S01 o 1).")
    parser.add_argument("--cam", default=None, help="Filtro cam para test (ej c001 o 1).")

    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    tracking_root = Path(args.tracking_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()

    df = pd.read_csv(csv_path)
    for col in ("scene", "cam number"):
        if col not in df.columns:
            raise KeyError(f"Falta columna requerida en CSV: {col!r}. Columnas: {list(df.columns)}")

    if args.test:
        df = filter_for_test(df, args.row_index, args.scene, args.cam)
        print(f"Modo TEST: usando fila índice {df.index[0]}")

    pairs = sorted({(normalize_scene(r["scene"]), normalize_cam(r["cam number"])) for _, r in df.iterrows()})

    total = len(pairs)
    ok = 0
    skipped = 0

    for i, (scene, cam) in enumerate(pairs, start=1):
        in_path = default_input_track_path(tracking_root, scene, cam, args.track_subdir)
        out_path = default_output_track_path(out_root, scene, cam)

        print(f"\n[{i}/{total}] {scene}/{cam}")
        print(f"  in : {in_path}")
        print(f"  out: {out_path}")

        if not in_path.exists():
            print("  [SKIP] No existe archivo de track de entrada.")
            skipped += 1
            if args.test:
                break
            continue

        kept = filter_track_file(in_path, out_path, args.min_area)
        print(f"  - Filtradas y guardadas {kept} filas (area >= {args.min_area}).")
        ok += 1

        if args.test:
            break

    print("\nResumen:")
    print(f"- OK: {ok}")
    print(f"- SKIP: {skipped}")
    print(f"- TOTAL: {total}")


if __name__ == "__main__":
    main()
