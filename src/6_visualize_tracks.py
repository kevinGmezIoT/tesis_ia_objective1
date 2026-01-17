###
# TEST
# python ./src/6_visualize_tracks.py --csv ./inputs/raw_videos/input_data.csv --frames-root ./transforms/frames --tracks-root ./transforms/format_tracks --out-vis-root ./transforms/vis_tracks --test --scene S05 --cam c001 --make-video --fps 20
###

import os
import re
import csv
import cv2
import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# -------------------------
# Normalización escena/cam
# -------------------------

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


# -------------------------
# Colores
# -------------------------

def get_spaced_colors(n: int) -> List[Tuple[int, int, int]]:
    max_value = 16581375  # 255**3
    interval = int(max_value / max(1, n))
    colors = [hex(i)[2:].zfill(6) for i in range(0, max_value, interval)]
    out = [(int(c[:2], 16), int(c[2:4], 16), int(c[4:], 16)) for c in colors]
    return out[:n]


# -------------------------
# I/O Frames
# -------------------------

def infer_frames_location(frames_root: Path, scene: str, cam: str) -> Path:
    """
    Soporta:
      frames_root/S0x/c00x/img1/*.jpg
      frames_root/S0x/c00x/*.jpg
    """
    base = frames_root / scene / cam
    cand = base / "img1"
    if cand.is_dir():
        return cand
    return base


def list_frame_paths(img_dir: Path) -> List[Path]:
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    frames = [p for p in sorted(img_dir.iterdir()) if p.is_file() and p.suffix.lower() in exts]
    return frames


def frame_index_from_name(p: Path) -> Optional[int]:
    # intenta stem numérico
    try:
        return int(p.stem)
    except Exception:
        # busca un número en el nombre
        m = re.search(r"(\d+)", p.name)
        return int(m.group(1)) if m else None


# -------------------------
# Lectura tracking (MOT)
# -------------------------

def load_tracks_mot(track_path: Path) -> Tuple[np.ndarray, int]:
    """
    Lee un .txt tipo MOT:
      frame,id,x,y,w,h,conf,-1,-1,-1
    Retorna:
      res_raw Nx7 float32 con columnas [frame,id,x,y,w,h,conf]
      max_frame
    """
    rows = []
    with track_path.open("r", newline="") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            if not row or len(row) < 6:
                continue
            frm = float(row[0])
            tid = float(row[1])
            x = float(row[2]); y = float(row[3])
            w = float(row[4]); h = float(row[5])
            conf = float(row[6]) if len(row) > 6 else 1.0
            rows.append([frm, tid, x, y, w, h, conf])

    if not rows:
        return np.zeros((0, 7), dtype=np.float32), 0

    res_raw = np.array(rows, dtype=np.float32)
    res_raw[:, 0:6] = res_raw[:, 0:6].astype(np.int32)  # frame,id,x,y,w,h
    max_frame = int(res_raw[:, 0].max())
    return res_raw, max_frame


# -------------------------
# Visualización
# -------------------------

def draw_tracks_on_frame(
    img: np.ndarray,
    t: int,
    res_raw: np.ndarray,
    color_list: List[Tuple[int, int, int]],
    trail: int,
    trail_step: int,
    alpha_decay: float,
) -> np.ndarray:
    """
    Dibuja:
      - trayectoria (puntos) para últimos `trail` frames (cada trail_step)
      - bbox + id para frame t
    """
    overlay = img.copy()

    # Filas de este frame
    row_ind = np.where(res_raw[:, 0] == t)[0]

    # ---- trayectorias ----
    for i in range(row_ind.shape[0]):
        tid = int(max(res_raw[row_ind[i], 1], 0))
        color = color_list[tid % len(color_list)]

        # frames en ventana [t-trail, t]
        line_ind = np.where(
            (res_raw[:, 0] >= max(1, t - trail)) &
            (res_raw[:, 0] <= t) &
            (res_raw[:, 1] == tid)
        )[0]

        for j in range(0, line_ind.shape[0], trail_step):
            x = int(res_raw[line_ind[j], 2])
            y = int(res_raw[line_ind[j], 3])
            w = int(res_raw[line_ind[j], 4])
            h = int(res_raw[line_ind[j], 5])

            cx = int(x + 0.5 * w)
            cy = int(y + h)

            bb_w = 5
            x1 = cx - bb_w
            y1 = cy - bb_w
            x2 = cx + bb_w
            y2 = cy + bb_w

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

            t_past = int(res_raw[line_ind[j], 0])
            # alpha mayor para puntos recientes
            # factor ~1.0 para t_past=t, menor si más antiguo
            alpha = max(0.0, 1.0 - (t - t_past) * alpha_decay)
            img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
            overlay = img.copy()

    # ---- bboxes del frame t ----
    for i in range(row_ind.shape[0]):
        tid = int(res_raw[row_ind[i], 1])
        x = int(res_raw[row_ind[i], 2])
        y = int(res_raw[row_ind[i], 3])
        w = int(res_raw[row_ind[i], 4])
        h = int(res_raw[row_ind[i], 5])
        conf = float(res_raw[row_ind[i], 6])

        x2 = x + w
        y2 = y + h
        color = color_list[tid % len(color_list)]

        cv2.rectangle(overlay, (x, y), (x2, y2), color, 2)
        cv2.putText(
            overlay, f"{tid}",
            (x, max(0, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA
        )

    cv2.putText(
        overlay, f"Frame Index: {t:06d}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA
    )

    return overlay


def process_sequence(
    scene: str,
    cam: str,
    frames_root: Path,
    tracks_root: Path,
    out_vis_root: Path,
    track_subdir: str,
    track_filename: Optional[str],
    vis_dirname: str,
    n_colors: int,
    trail: int,
    trail_step: int,
    alpha_decay: float,
    make_video: bool,
    fps: int,
    overwrite: bool
) -> None:
    img_dir = infer_frames_location(frames_root, scene, cam)
    frames = list_frame_paths(img_dir)
    if not frames:
        print(f"[SKIP] Sin frames: {img_dir}")
        return

    # ruta track
    if track_filename:
        track_path = tracks_root / scene / cam / track_subdir / track_filename
    else:
        track_path = tracks_root / scene / cam / track_subdir / f"{cam}.txt"

    if not track_path.exists():
        print(f"[SKIP] No existe track: {track_path}")
        return

    res_raw, max_frame = load_tracks_mot(track_path)
    if res_raw.shape[0] == 0:
        print(f"[WARN] Track vacío: {track_path}")
        return

    # salida
    save_dir = out_vis_root / scene / cam / vis_dirname
    save_dir.mkdir(parents=True, exist_ok=True)

    # colores
    color_list = get_spaced_colors(n_colors)
    random.shuffle(color_list)

    # video writer opcional
    vw = None
    if make_video:
        # tamaño de primer frame
        first_img = cv2.imread(str(frames[0]))
        if first_img is None:
            print(f"[SKIP] No pude leer primer frame: {frames[0]}")
            return
        H, W = first_img.shape[:2]
        video_path = save_dir.parent / f"{scene}_{cam}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(video_path), fourcc, fps, (W, H))

    # index frames por id (para no depender de continuidad perfecta)
    frame_map = {}
    for p in frames:
        idx = frame_index_from_name(p)
        if idx is not None:
            frame_map[idx] = p

    # iterar por frames en tracks (1..max_frame)
    pbar = tqdm(range(1, max_frame + 1), desc=f"{scene}/{cam}", unit="frame")
    for t in pbar:
        out_img_path = save_dir / f"{t:06d}.jpg"
        if out_img_path.exists() and not overwrite:
            continue

        img_path = frame_map.get(t, None)
        if img_path is None:
            # si no hay exact match, se omite (o podrías buscar cercano)
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        vis = draw_tracks_on_frame(
            img=img,
            t=t,
            res_raw=res_raw,
            color_list=color_list,
            trail=trail,
            trail_step=trail_step,
            alpha_decay=alpha_decay
        )

        cv2.imwrite(str(out_img_path), vis)
        if vw is not None:
            vw.write(vis)

    if vw is not None:
        vw.release()
        print(f"  - Video guardado en: {save_dir.parent / f'{scene}_{cam}.mp4'}")


def main():
    ap = argparse.ArgumentParser(description="Visualiza tracking sobre frames usando CSV (S0x/c00x).")
    ap.add_argument("--csv", required=True, help="CSV con columnas: 'scene' y 'cam number'.")
    ap.add_argument("--frames-root", required=True, help="Raíz frames: frames/S0x/c00x/(img1/)?*.jpg")
    ap.add_argument("--tracks-root", required=True,
                    help="Raíz tracks. Se espera: tracks-root/S0x/c00x/<track-subdir>/c00x.txt")
    ap.add_argument("--out-vis-root", required=True,
                    help="Raíz de salida para visualizaciones. Se guardará en out-vis-root/S0x/c00x/<vis-dirname>/")

    ap.add_argument("--track-subdir", default="", help="Subruta dentro de S0x/c00x donde está el .txt (ej: '' o 'track').")
    ap.add_argument("--track-filename", default=None,
                    help="Nombre exacto del archivo de track (si no, usa c00x.txt).")
    ap.add_argument("--vis-dirname", default="vis", help="Nombre carpeta para guardar imágenes (default: vis).")

    ap.add_argument("--n-colors", type=int, default=200, help="Cantidad de colores (default: 200).")
    ap.add_argument("--trail", type=int, default=50, help="Longitud de trayectoria en frames (default: 50).")
    ap.add_argument("--trail-step", type=int, default=5, help="Salto para puntos en la trayectoria (default: 5).")
    ap.add_argument("--alpha-decay", type=float, default=1.0/80.0,
                    help="Decaimiento alpha por frame (default: 1/80).")

    ap.add_argument("--make-video", action="store_true", help="Genera mp4 además de imágenes.")
    ap.add_argument("--fps", type=int, default=25, help="FPS del video si make-video (default: 25).")
    ap.add_argument("--overwrite", action="store_true", help="Sobrescribir imágenes existentes.")

    # Test
    ap.add_argument("--test", action="store_true", help="Procesa solo 1 secuencia.")
    ap.add_argument("--row-index", type=int, default=None, help="Fila CSV (0-based) para test.")
    ap.add_argument("--scene", default=None, help="Filtro escena para test.")
    ap.add_argument("--cam", default=None, help="Filtro cam para test.")

    args = ap.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    frames_root = Path(args.frames_root).expanduser().resolve()
    tracks_root = Path(args.tracks_root).expanduser().resolve()
    out_vis_root = Path(args.out_vis_root).expanduser().resolve()

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
        print(f"\n[{i}/{total}] Visualizando {scene}/{cam}")

        try:
            process_sequence(
                scene=scene,
                cam=cam,
                frames_root=frames_root,
                tracks_root=tracks_root,
                out_vis_root=out_vis_root,
                track_subdir=args.track_subdir,
                track_filename=args.track_filename,
                vis_dirname=args.vis_dirname,
                n_colors=args.n_colors,
                trail=args.trail,
                trail_step=args.trail_step,
                alpha_decay=args.alpha_decay,
                make_video=args.make_video,
                fps=args.fps,
                overwrite=args.overwrite,
            )
            ok += 1
        except Exception as e:
            print(f"  [ERROR] {scene}/{cam}: {e}")
            skipped += 1

        if args.test:
            break

    print("\nResumen:")
    print(f"- OK: {ok}")
    print(f"- SKIP/ERROR: {skipped}")
    print(f"- TOTAL: {total}")


if __name__ == "__main__":
    main()
