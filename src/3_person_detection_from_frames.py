#!/usr/bin/env python3
# 3_person_detection_from_frames.py
#
# Uso:
#   python src/3_person_detection_from_frames.py --csv ./inputs/raw_videos/input_data_S05.csv --frames-root ./transforms/frames --output-root ./transforms/crops --weights ./src/models/yolo11m.pt --imgsz 1280 --conf 0.25 --device 0 --save-det-txt 
#
# Test:
#   python src/3_person_detection_from_frames.py --csv ./inputs/raw_videos/input_data.csv --frames-root ./transforms/frames --output-root ./transforms/crops --weights ./src/models/yolo11m.pt --test --scene S01 --cam c001 --save-det-txt --save-crops

import argparse
import math
import os
import re
import sys
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO


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


def frame_index_from_name(name: str):
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else None


def xyxy_to_xywh(xyxy):
    x1, y1, x2, y2 = xyxy
    return x1, y1, x2 - x1, y2 - y1


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def find_img_dir(seq_dir: Path) -> Path:
    cand = seq_dir / "img1"
    if cand.is_dir():
        return cand

    imgs = list(seq_dir.glob("*.jpg")) + list(seq_dir.glob("*.png")) + list(seq_dir.glob("*.jpeg"))
    if imgs:
        return seq_dir

    imgs = list(seq_dir.rglob("*.jpg")) + list(seq_dir.rglob("*.png")) + list(seq_dir.rglob("*.jpeg"))
    if not imgs:
        raise FileNotFoundError(f"No se encontraron imágenes bajo {seq_dir}. ¿Generaste los frames?")
    return imgs[0].parent


def count_frames(img_dir: Path) -> int:
    exts = ("*.jpg", "*.png", "*.jpeg")
    n = 0
    for ext in exts:
        n += len(list(img_dir.glob(ext)))
    return n


def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(math.floor(x1)), w - 1))
    y1 = max(0, min(int(math.floor(y1)), h - 1))
    x2 = max(0, min(int(math.ceil(x2)), w))
    y2 = max(0, min(int(math.ceil(y2)), h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def save_yolo_labels(res, yolo_labels_dir: Path, img_path: Path, boxes_xyxy, confs, clss):
    ensure_dir(yolo_labels_dir)
    label_file = yolo_labels_dir / f"{img_path.stem}.txt"
    img_h, img_w = res.orig_shape[:2]

    with open(label_file, "w") as f:
        for box, conf, cls in zip(boxes_xyxy, confs, clss):
            x1, y1, x2, y2 = box
            x_center = ((x1 + x2) / 2) / img_w
            y_center = ((y1 + y2) / 2) / img_h
            width = (x2 - x1) / img_w
            height = (y2 - y1) / img_h
            f.write(f"{int(cls)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.6f}\n")


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


def process_sequence(scene: str, cam: str, frames_root: Path, output_root: Path, model: YOLO, args) -> tuple[int, int, int]:
    seq_dir = frames_root / scene / cam
    img_dir = find_img_dir(seq_dir)
    total_frames = count_frames(img_dir)
    if total_frames == 0:
        print(f"[!] No se encontraron imágenes en {img_dir}, omitiendo...")
        return 0, 0, 0

    out_seq_dir = output_root / scene / cam

    # Salidas opcionales
    persons_dir = None
    if args.save_crops:
        persons_dir = out_seq_dir / args.crops_dirname
        ensure_dir(persons_dir)

    det_txt = None
    if args.save_det_txt:
        det_dir = out_seq_dir / "det"
        ensure_dir(det_dir)
        det_txt = det_dir / "det.txt"

    yolo_labels_dir = None
    if args.save_yolo_labels:
        yolo_labels_dir = out_seq_dir / "labels_yolo11"
        ensure_dir(yolo_labels_dir)

    # Clase "person"
    names = model.names
    person_ids = {i for i, n in names.items() if str(n).lower() == "person"} or {0}

    print(f"\nProcesando: {scene}/{cam}")
    print(f"  - Frames: {total_frames}")
    print(f"  - Img dir: {img_dir}")
    if persons_dir is not None:
        print(f"  - Crops -> {persons_dir}")
    else:
        print(f"  - Crops -> (deshabilitado)")
    if det_txt is not None:
        print(f"  - det.txt -> {det_txt}")
    if yolo_labels_dir is not None:
        print(f"  - YOLO labels -> {yolo_labels_dir}")

    results = model.predict(
        source=str(img_dir),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        stream=True,
        workers=0,
        verbose=False
    )

    frames_proc = 0
    dets_total = 0
    crops_total = 0

    fdet = open(det_txt, "w", buffering=1) if det_txt is not None else None

    try:
        pbar = tqdm(total=total_frames, desc=f"{scene}/{cam}", unit="frame")

        for res in results:
            frames_proc += 1

            img_path = Path(res.path)
            frame_idx = frame_index_from_name(img_path.name) or frames_proc

            orig = res.orig_img
            if orig is None:
                orig = cv2.imread(str(img_path))
            if orig is None:
                pbar.update(1)
                continue

            H, W = orig.shape[:2]
            n_crops_frame = 0

            if res.boxes is not None and len(res.boxes) > 0:
                boxes_xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy().astype(int)

                keep = [i for i, c in enumerate(clss) if c in person_ids]
                boxes_xyxy = boxes_xyxy[keep]
                confs = confs[keep]
                clss = clss[keep]

                if args.save_yolo_labels and yolo_labels_dir is not None and len(boxes_xyxy) > 0:
                    save_yolo_labels(res, yolo_labels_dir, img_path, boxes_xyxy, confs, clss)

                # det.txt y/o crops
                for k, (box, conf) in enumerate(zip(boxes_xyxy, confs), start=1):
                    x1, y1, x2, y2 = box
                    clipped = clamp_box(x1, y1, x2, y2, W, H)
                    if clipped is None:
                        continue
                    x1i, y1i, x2i, y2i = clipped

                    if fdet is not None:
                        x1m, y1m, w, h = xyxy_to_xywh((x1i, y1i, x2i, y2i))
                        fdet.write(f"{frame_idx},-1,{x1m:.2f},{y1m:.2f},{w:.2f},{h:.2f},{conf:.6f},-1,-1,-1\n")
                        dets_total += 1

                    if persons_dir is not None:
                        crop = orig[y1i:y2i, x1i:x2i]
                        if crop.size == 0:
                            continue
                        crop_name = f"{frame_idx:06d}_{k:03d}.jpg"
                        cv2.imwrite(str(persons_dir / crop_name), crop)
                        crops_total += 1
                        n_crops_frame += 1

            pbar.update(1)
            pbar.set_postfix({"dets": dets_total, "crops": crops_total, "this_frame": n_crops_frame})

            if fdet is not None and frames_proc % args.flush_every == 0:
                fdet.flush()
                os.fsync(fdet.fileno())

        pbar.close()
        return frames_proc, dets_total, crops_total

    finally:
        if fdet is not None:
            fdet.close()


def main():
    ap = argparse.ArgumentParser(description="YOLO persons desde frames por escena/cámara (CSV) con crops opcionales.")
    ap.add_argument("--csv", required=True, type=str, help="Ruta al CSV (con columnas 'scene' y 'cam number').")
    ap.add_argument("--frames-root", required=True, type=str, help="Raíz de frames: frames-root/S0x/c00x/000001.jpg (o img1/).")
    ap.add_argument("--output-root", required=True, type=str, help="Raíz salida: output-root/S0x/c00x/...")

    ap.add_argument("--weights", type=str, default="yolo11n.pt", help="Pesos YOLO.")
    ap.add_argument("--imgsz", type=int, default=1280, help="Tamaño para inferencia.")
    ap.add_argument("--conf", type=float, default=0.25, help="Umbral de confianza.")
    ap.add_argument("--device", type=str, default=None, help="Dispositivo: '0', 'cpu', etc.")

    ap.add_argument("--save-det-txt", action="store_true", help="Guardar det.txt (MOT).")
    ap.add_argument("--save-yolo-labels", action="store_true", help="Guardar labels YOLO.")
    ap.add_argument("--save-crops", action="store_true", help="Guardar crops (deshabilitado por defecto).")
    ap.add_argument("--crops-dirname", default="persons", help="Nombre carpeta de crops dentro de S0x/c00x/ (default: persons).")
    ap.add_argument("--flush-every", type=int, default=50, help="Flush a disco cada N frames (si save-det-txt).")

    # Test
    ap.add_argument("--test", action="store_true", help="Procesa solo una escena/cámara.")
    ap.add_argument("--row-index", type=int, default=None, help="Índice (0-based) de fila CSV para test.")
    ap.add_argument("--scene", default=None, help="Filtro escena para test.")
    ap.add_argument("--cam", default=None, help="Filtro cámara para test.")

    args = ap.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    frames_root = Path(args.frames_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    df = pd.read_csv(csv_path)
    for col in ("scene", "cam number"):
        if col not in df.columns:
            raise KeyError(f"Falta columna requerida en el CSV: {col!r}. Columnas: {list(df.columns)}")

    if args.test:
        df = filter_for_test(df, args.row_index, args.scene, args.cam)
        print(f"Modo TEST: usando fila índice {df.index[0]}")

    pairs = sorted({(normalize_scene(r["scene"]), normalize_cam(r["cam number"])) for _, r in df.iterrows()})

    model = YOLO(args.weights)

    total_pairs = len(pairs)
    total_frames = 0
    total_dets = 0
    total_crops = 0
    ok = 0
    skipped = 0

    for i, (scene, cam) in enumerate(pairs, start=1):
        seq_dir = frames_root / scene / cam
        if not seq_dir.exists():
            skipped += 1
            print(f"[SKIP {i}/{total_pairs}] No existe el directorio: {seq_dir}")
            continue

        print(f"\n[{i}/{total_pairs}] {scene}/{cam}")
        try:
            frames_proc, dets, crops = process_sequence(scene, cam, frames_root, output_root, model, args)
            total_frames += frames_proc
            total_dets += dets
            total_crops += crops
            ok += 1
        except Exception as e:
            print(f"[ERROR] {scene}/{cam}: {e}", file=sys.stderr)

        if args.test:
            break

    print("\nResumen:")
    print(f"- Secuencias OK: {ok}")
    print(f"- Secuencias SKIP: {skipped}")
    print(f"- Frames procesados: {total_frames}")
    print(f"- Detecciones (person): {total_dets}")
    if args.save_crops:
        print(f"- Crops guardados: {total_crops}")
        if total_frames > 0:
            print(f"- Promedio crops/frame: {total_crops / total_frames:.3f}")
    else:
        print("- Crops guardados: (deshabilitado)")

if __name__ == "__main__":
    main()