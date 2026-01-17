## Example
# python src\7_extract_image_crops.py --csv inputs\raw_videos\input_data_S10_to_S20.csv --frames-root transforms\frames --tracks-root transforms\format_tracks --annotations-root transforms\annotations --out-root transforms\gt_crops

import os
import sys
import re
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import pandas as pd
from tqdm import tqdm


# =========================================================
# Normalización
# =========================================================

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


def to_int(x) -> int:
    return int(float(x))


def clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))


# =========================================================
# Frames
# =========================================================

def infer_frames_dir(frames_root: Path, scene: str, cam: str) -> Path:
    """
    Soporta:
      frames_root/S0x/c00x/img1/*.jpg
      frames_root/S0x/c00x/*.jpg
    """
    base = frames_root / scene / cam
    cand = base / "img1"
    return cand if cand.is_dir() else base


def list_frames(img_dir: Path) -> List[Path]:
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    return [p for p in sorted(img_dir.iterdir()) if p.is_file() and p.suffix.lower() in exts]


def frame_idx_from_name(p: Path) -> Optional[int]:
    try:
        return int(p.stem)
    except Exception:
        m = re.search(r"(\d+)", p.name)
        return int(m.group(1)) if m else None


def load_first_frame_shape(img_dir: Path) -> Tuple[int, int]:
    frames = list_frames(img_dir)
    if not frames:
        raise FileNotFoundError(f"Sin frames en {img_dir}")
    img = cv2.imread(str(frames[0]))
    if img is None:
        raise RuntimeError(f"No pude leer {frames[0]}")
    h, w = img.shape[:2]
    return h, w


# =========================================================
# Tracks + annotation.txt -> GT global
# =========================================================

def load_local_tracks(track_path: Path, bbox_area_limit: float) -> Dict[int, Dict[int, Tuple[int, int, int, int]]]:
    """
    local_dict[local_id][frame] = (x,y,w,h) (int)
    Filtra por area.
    """
    local_dict: Dict[int, Dict[int, Tuple[int, int, int, int]]] = {}
    with track_path.open("r", newline="") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            if not row or len(row) < 6:
                continue
            frm = to_int(row[0])
            lid = to_int(row[1])
            x = to_int(row[2]); y = to_int(row[3])
            w = to_int(row[4]); h = to_int(row[5])

            if (w * h) <= bbox_area_limit:
                continue

            if lid not in local_dict:
                local_dict[lid] = {}
            local_dict[lid][frm] = (x, y, w, h)
    return local_dict


def apply_remove_ops(local_dict: Dict[int, Dict[int, Tuple[int, int, int, int]]], annotation_path: Path) -> None:
    if not annotation_path.is_file():
        return

    with annotation_path.open("r", newline="") as f:
        reader = csv.reader(f, delimiter=",")
        line_idx = 1
        for row in reader:
            if not row or row[0].startswith("#"):
                line_idx += 1
                continue

            op = row[0].strip()
            if op != "remove":
                line_idx += 1
                continue

            if len(row) != 3 or not row[1]:
                print(f"[WARN] remove inválido en línea {line_idx}: {row}")
                line_idx += 1
                continue

            r = row[1].strip()
            lid = int(row[2])
            if lid not in local_dict:
                print(f"[WARN] remove refiere local_id inexistente {lid} (línea {line_idx})")
                line_idx += 1
                continue

            frm_min = 0
            frm_max = sys.maxsize

            dash = r.find("-")
            if dash == -1:
                frm_min = int(r); frm_max = int(r)
            elif dash == 0:
                frm_max = int(r[1:])
            elif dash == len(r) - 1:
                frm_min = int(r[:-1])
            else:
                toks = r.split("-")
                if len(toks) != 2:
                    print(f"[WARN] rango remove inválido en línea {line_idx}: {r}")
                    line_idx += 1
                    continue
                frm_min = int(toks[0]); frm_max = int(toks[1])

            to_delete = [fidx for fidx in local_dict[lid].keys() if frm_min <= fidx <= frm_max]
            for fidx in to_delete:
                del local_dict[lid][fidx]

            line_idx += 1


def build_assign_and_insert(
    local_dict: Dict[int, Dict[int, Tuple[int, int, int, int]]],
    annotation_path: Path,
    bbox_area_limit: float
) -> Dict[int, Dict[int, Tuple[int, int, int, int]]]:
    """
    Retorna annotation_dict:
      annotation_dict[global_id][frame] = (x,y,w,h)
    """
    annotation_dict: Dict[int, Dict[int, Tuple[int, int, int, int]]] = {}
    match_dict: Dict[int, int] = {}

    if not annotation_path.is_file():
        return annotation_dict

    # assign
    with annotation_path.open("r", newline="") as f:
        reader = csv.reader(f, delimiter=",")
        line_idx = 1
        for row in reader:
            if not row or row[0].startswith("#"):
                line_idx += 1
                continue
            if row[0].strip() != "assign":
                line_idx += 1
                continue
            if len(row) != 3:
                print(f"[WARN] assign inválido en línea {line_idx}: {row}")
                line_idx += 1
                continue

            lid = int(row[1]); gid = int(row[2])
            if lid not in local_dict:
                print(f"[WARN] assign refiere local_id inexistente {lid} (línea {line_idx})")
                line_idx += 1
                continue
            if lid in match_dict:
                print(f"[WARN] assign duplicado local_id {lid} (línea {line_idx})")
                line_idx += 1
                continue

            match_dict[lid] = gid
            if gid not in annotation_dict:
                annotation_dict[gid] = {}
            for frm, bbox in local_dict[lid].items():
                annotation_dict[gid][frm] = bbox

            line_idx += 1

    # insert
    with annotation_path.open("r", newline="") as f:
        reader = csv.reader(f, delimiter=",")
        line_idx = 1
        for row in reader:
            if not row or row[0].startswith("#"):
                line_idx += 1
                continue
            if row[0].strip() != "insert":
                line_idx += 1
                continue
            if len(row) != 7:
                print(f"[WARN] insert inválido en línea {line_idx}: {row}")
                line_idx += 1
                continue

            frm = int(row[1])
            lid = int(row[2])
            if lid not in match_dict:
                print(f"[WARN] insert con local_id no asignado {lid} (línea {line_idx})")
                line_idx += 1
                continue

            gid = match_dict[lid]
            if gid not in annotation_dict:
                annotation_dict[gid] = {}

            x = to_int(row[3]); y = to_int(row[4])
            w = to_int(row[5]); h = to_int(row[6])
            if (w * h) > bbox_area_limit:
                annotation_dict[gid][frm] = (x, y, w, h)

            line_idx += 1

    return annotation_dict


def interpolate_linear(
    annotation_dict: Dict[int, Dict[int, Tuple[int, int, int, int]]],
    bbox_area_limit: float,
    interp_frame_limit: int
) -> None:
    for gid in list(annotation_dict.keys()):
        frs = sorted(annotation_dict[gid].keys())
        for i in range(1, len(frs)):
            f0, f1 = frs[i - 1], frs[i]
            diff = f1 - f0
            if 1 < diff <= interp_frame_limit:
                x0, y0, w0, h0 = annotation_dict[gid][f0]
                x1, y1, w1, h1 = annotation_dict[gid][f1]
                dx = (x1 - x0) / float(diff)
                dy = (y1 - y0) / float(diff)
                dw = (w1 - w0) / float(diff)
                dh = (h1 - h0) / float(diff)
                for j in range(1, diff):
                    frm = f0 + j
                    xx = int(round(x0 + dx * j))
                    yy = int(round(y0 + dy * j))
                    ww = int(round(w0 + dw * j))
                    hh = int(round(h0 + dh * j))
                    if (ww * hh) > bbox_area_limit:
                        annotation_dict[gid][frm] = (xx, yy, ww, hh)


def reorganize_to_frame_dict(
    annotation_dict: Dict[int, Dict[int, Tuple[int, int, int, int]]]
) -> Dict[int, Dict[int, Tuple[int, int, int, int]]]:
    gt_dict: Dict[int, Dict[int, Tuple[int, int, int, int]]] = {}
    for gid, frames in annotation_dict.items():
        for frm, bbox in frames.items():
            gt_dict.setdefault(frm, {})
            if gid in gt_dict[frm]:
                print(f"[WARN] Duplicado global_id {gid} en frame {frm}")
                continue
            gt_dict[frm][gid] = bbox
    return gt_dict


def remove_occlusions(
    gt_dict: Dict[int, Dict[int, Tuple[int, int, int, int]]],
    occlude_iou_limit: float
) -> None:
    to_remove: List[Tuple[int, int]] = []
    for frm, gdict in gt_dict.items():
        gids = list(gdict.keys())
        for i in range(len(gids) - 1):
            for j in range(i + 1, len(gids)):
                gid_a, gid_b = gids[i], gids[j]
                xa1, ya1, wa, ha = gdict[gid_a]
                xb1, yb1, wb, hb = gdict[gid_b]
                xa2, ya2 = xa1 + wa, ya1 + ha
                xb2, yb2 = xb1 + wb, yb1 + hb

                x_int_min = max(xa1, xb1)
                y_int_min = max(ya1, yb1)
                x_int_max = min(xa2, xb2)
                y_int_max = min(ya2, yb2)
                inter_area = max(0, x_int_max - x_int_min) * max(0, y_int_max - y_int_min)

                area_a = max(1, wa * ha)
                area_b = max(1, wb * hb)

                if inter_area / float(area_a) > occlude_iou_limit:
                    if ya2 < yb2:
                        to_remove.append((frm, gid_a))
                if inter_area / float(area_b) > occlude_iou_limit:
                    if yb2 < ya2:
                        to_remove.append((frm, gid_b))

    for frm, gid in to_remove:
        if gid in gt_dict.get(frm, {}):
            del gt_dict[frm][gid]


def expand_and_clamp_bbox(
    x: int, y: int, w: int, h: int,
    img_w: int, img_h: int,
    expand_factor: float
) -> Tuple[int, int, int, int]:
    new_w = int(round(w * expand_factor))
    new_h = int(round(h * expand_factor))

    cx = x + w // 2
    cy = y + h // 2

    new_x = max(0, cx - new_w // 2)
    new_y = max(0, cy - new_h // 2)

    new_w = min(new_w, img_w - new_x)
    new_h = min(new_h, img_h - new_y)

    new_w = max(1, new_w)
    new_h = max(1, new_h)

    return new_x, new_y, new_w, new_h


def generate_global_gt_for_sequence(
    scene: str,
    cam: str,
    frames_root: Path,
    tracks_root: Path,
    annotations_root: Path,
    out_root: Path,
    bbox_area_limit: float,
    occlude_iou_limit: float,
    interp_frame_limit: int,
    expand_factor: float,
) -> Optional[Path]:
    """
    Salida:
      out_root/gt/S0x/c00x/gt/gt.txt
    """
    img_dir = infer_frames_dir(frames_root, scene, cam)
    if not img_dir.exists():
        print(f"  [SKIP] No existe frames dir: {img_dir}")
        return None

    img_h, img_w = load_first_frame_shape(img_dir)

    track_path = tracks_root / scene / cam / f"{cam}.txt"
    if not track_path.is_file():
        print(f"  [SKIP] No existe track: {track_path}")
        return None

    annotation_path = annotations_root / scene / cam / "annotation.txt"
    if not annotation_path.is_file():
        print(f"  [SKIP] No existe annotation.txt: {annotation_path}")
        return None

    local_dict = load_local_tracks(track_path, bbox_area_limit=bbox_area_limit)
    apply_remove_ops(local_dict, annotation_path)
    annotation_dict = build_assign_and_insert(local_dict, annotation_path, bbox_area_limit=bbox_area_limit)

    if not annotation_dict:
        print(f"  [SKIP] annotation.txt no produjo asignaciones para {scene}/{cam}")
        return None

    interpolate_linear(annotation_dict, bbox_area_limit=bbox_area_limit, interp_frame_limit=interp_frame_limit)
    gt_dict = reorganize_to_frame_dict(annotation_dict)
    remove_occlusions(gt_dict, occlude_iou_limit=occlude_iou_limit)

    out_gt_path = out_root / "gt" / scene / cam / "gt" / "gt.txt"
    out_gt_path.parent.mkdir(parents=True, exist_ok=True)

    with out_gt_path.open("w", newline="") as f_out:
        for frm in sorted(gt_dict.keys()):
            for gid in sorted(gt_dict[frm].keys()):
                x, y, w, h = gt_dict[frm][gid]
                new_x, new_y, new_w, new_h = expand_and_clamp_bbox(
                    x, y, w, h, img_w=img_w, img_h=img_h, expand_factor=expand_factor
                )
                f_out.write(f"{frm},{gid},{new_x},{new_y},{new_w},{new_h},1,-1,-1,-1\n")

    return out_gt_path


# =========================================================
# GT -> Crops (NUEVO LAYOUT): crops/IDxxxx/S0x/c00x/...
# =========================================================

def load_gt_by_frame(gt_path: Path) -> Dict[int, List[Tuple[int, int, int, int, int]]]:
    by_frame: Dict[int, List[Tuple[int, int, int, int, int]]] = {}
    with gt_path.open("r", newline="") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            if not row or len(row) < 6:
                continue
            frm = to_int(row[0])
            gid = to_int(row[1])
            x = to_int(row[2]); y = to_int(row[3])
            w = to_int(row[4]); h = to_int(row[5])
            by_frame.setdefault(frm, []).append((gid, x, y, w, h))
    return by_frame


def crop_from_gt_sequence(
    scene: str,
    cam: str,
    frames_root: Path,
    gt_path: Path,
    out_root: Path,
    crop_area_min: int
) -> int:
    img_dir = infer_frames_dir(frames_root, scene, cam)
    frames = list_frames(img_dir)
    if not frames:
        return 0

    frame_map: Dict[int, Path] = {}
    for p in frames:
        idx = frame_idx_from_name(p)
        if idx is not None:
            frame_map[idx] = p

    gt_by_frame = load_gt_by_frame(gt_path)
    if not gt_by_frame:
        return 0

    written = 0

    for frm in tqdm(sorted(gt_by_frame.keys()), desc=f"Cropping {scene}/{cam}", unit="frame"):
        img_path = frame_map.get(frm)
        if img_path is None:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        H, W = img.shape[:2]
        for gid, x, y, w, h in gt_by_frame[frm]:
            x1 = clamp(x, 0, W - 1)
            y1 = clamp(y, 0, H - 1)
            x2 = clamp(x + w, 1, W)
            y2 = clamp(y + h, 1, H)

            ww = max(0, x2 - x1)
            hh = max(0, y2 - y1)
            if ww == 0 or hh == 0:
                continue

            crop = img[y1:y1 + hh, x1:x1 + ww]
            flag = (ww * hh) >= crop_area_min

            # NUEVO: crops/IDxxxx/S0x/c00x/
            person_dir = out_root / "crops" / f"ID{gid:04d}" / scene / cam
            person_dir.mkdir(parents=True, exist_ok=True)

            out_name = f"{cam}_{frm:06d}_{flag}.jpg"
            out_path = person_dir / out_name

            if cv2.imwrite(str(out_path), crop):
                written += 1

    return written


# =========================================================
# Main
# =========================================================

def main():
    ap = argparse.ArgumentParser(description="Genera GT global + crops (layout por ID) desde tracks locales y annotation.txt, leyendo CSV.")
    ap.add_argument("--csv", required=True, help="CSV con columnas: 'scene' y 'cam number'.")
    ap.add_argument("--frames-root", required=True, help="Raíz frames: transforms/frames/S0x/c00x/(img1/)?*.jpg")
    ap.add_argument("--tracks-root", required=True, help="Raíz tracks: transforms/format_tracks/S0x/c00x/c00x.txt")
    ap.add_argument("--annotations-root", required=True, help="Raíz annotations: transforms/annotations/S0x/c00x/annotation.txt")
    ap.add_argument("--out-root", required=True, help="Carpeta donde guardar: gt/ y crops/")

    # Parámetros
    ap.add_argument("--bbox-area-limit", type=float, default=1000.0)
    ap.add_argument("--occlude-iou-limit", type=float, default=0.50)
    ap.add_argument("--interp-frame-limit", type=int, default=10)
    ap.add_argument("--expand-factor", type=float, default=1.2)
    ap.add_argument("--crop-area-min", type=int, default=1000)

    # Test / selección
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--row-index", type=int, default=None)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--cam", default=None)

    args = ap.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    frames_root = Path(args.frames_root).expanduser().resolve()
    tracks_root = Path(args.tracks_root).expanduser().resolve()
    annotations_root = Path(args.annotations_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()

    df = pd.read_csv(csv_path)
    for col in ("scene", "cam number"):
        if col not in df.columns:
            raise KeyError(f"Falta columna requerida en CSV: {col!r}. Columnas: {list(df.columns)}")

    # Test: si solo pasas --scene, mantiene TODAS las cams de esa escena.
    if args.test:
        if args.row_index is not None:
            df = df.iloc[[args.row_index]].copy()
            print(f"Modo TEST: usando fila índice {args.row_index}")
        else:
            if args.scene is not None:
                ns = normalize_scene(args.scene)
                df = df[df["scene"].astype(str).apply(lambda x: normalize_scene(x) == ns)].copy()
                if df.empty:
                    raise ValueError(f"Modo TEST: no hay filas para escena {ns}")
            if args.cam is not None:
                nc = normalize_cam(args.cam)
                df = df[df["cam number"].astype(str).apply(lambda x: normalize_cam(x) == nc)].copy()
                if df.empty:
                    raise ValueError(f"Modo TEST: no hay filas para cam {nc}")
            print(f"Modo TEST: filas seleccionadas: {len(df)}")

    pairs = sorted({(normalize_scene(r["scene"]), normalize_cam(r["cam number"])) for _, r in df.iterrows()})

    total = len(pairs)
    ok = 0
    skipped = 0

    for i, (scene, cam) in enumerate(pairs, start=1):
        print(f"\n[{i}/{total}] {scene}/{cam}")

        gt_path = generate_global_gt_for_sequence(
            scene=scene,
            cam=cam,
            frames_root=frames_root,
            tracks_root=tracks_root,
            annotations_root=annotations_root,
            out_root=out_root,
            bbox_area_limit=args.bbox_area_limit,
            occlude_iou_limit=args.occlude_iou_limit,
            interp_frame_limit=args.interp_frame_limit,
            expand_factor=args.expand_factor,
        )

        if gt_path is None or not gt_path.exists():
            skipped += 1
            continue

        n = crop_from_gt_sequence(
            scene=scene,
            cam=cam,
            frames_root=frames_root,
            gt_path=gt_path,
            out_root=out_root,
            crop_area_min=args.crop_area_min,
        )

        print(f"  - GT: {gt_path}")
        print(f"  - Crops escritos: {n}")
        ok += 1

    print("\nResumen:")
    print(f"- OK: {ok}")
    print(f"- SKIP: {skipped}")
    print(f"- TOTAL: {total}")


if __name__ == "__main__":
    main()
