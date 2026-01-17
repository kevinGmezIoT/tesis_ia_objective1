#!/usr/bin/env python3
# run_tracking_from_csv_separate_roots.py
#
# Pipeline unificado para tu estructura:
#   frames-root/S0x/c00x/(img1/)?*.jpg
#   det-root/S0x/c00x/det/det.txt
#
# Produce:
#   out-root/S0x/c00x/annotations/train_cocoformat.json
#   out-root/S0x/c00x/annotations/train_detections.pkl
#   out-root/S0x/c00x/output/  (tracking results)
#
# Ejemplo:
#   python run_tracking_from_csv_separate_roots.py ^
#       --csv "G:/.../input_data.csv" ^
#       --frames-root "G:/.../frames" ^
#       --det-root "G:/.../crops" ^
#       --out-root "G:/.../tracking_runs" ^
#       --config "configs/mot/deepsort/sort_faster-rcnn_fpn_4e_mot17-public.py" ^
#       --default-fps 25 --gpu-id 0 --skip-convert-if-exists
#
# Test:
#   python run_tracking_from_csv_separate_roots.py --csv ... --frames-root ... --det-root ... --out-root ... --config ... --test --scene S01 --cam c001

import argparse
import os
import os.path as osp
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import mmcv
import torch
from mmcv import Config
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import get_dist_info, load_checkpoint
from mmdet.apis import set_random_seed

from mmtrack.core import setup_multi_processes
from mmtrack.datasets import build_dataset
from mmtrack.utils import build_dp, get_device


# -------------------------
# CSV helpers
# -------------------------

def normalize_scene(scene_raw: str) -> str:
    s = str(scene_raw).strip()
    m = re.search(r"(\d+)", s)
    if not m:
        raise ValueError(f"No pude extraer número de escena desde: {scene_raw!r}")
    n = int(m.group(1))
    return f"S{n:02d}"


def save_tracking_results(dataset, outputs, resfile_path: str) -> None:
    """Intenta guardar el resultado de tracking en formato MOT .txt dentro de resfile_path."""
    mmcv.mkdir_or_exist(osp.abspath(resfile_path))

    # 1) Ruta típica en mmtrack
    if hasattr(dataset, "format_results"):
        try:
            dataset.format_results(outputs, resfile_path=resfile_path)
            return
        except TypeError:
            # Algunas versiones no aceptan keyword, o cambian la firma
            try:
                dataset.format_results(outputs, resfile_path)
                return
            except Exception:
                pass

    # 2) Algunas versiones/datasets exponen un método específico
    if hasattr(dataset, "format_track_results"):
        try:
            dataset.format_track_results(outputs, resfile_path=resfile_path)
            return
        except TypeError:
            try:
                dataset.format_track_results(outputs, resfile_path)
                return
            except Exception:
                pass

    # 3) Fallback: al menos guardar el pickle para no perder el resultado
    mmcv.dump(outputs, osp.join(resfile_path, "tracking_outputs.pkl"))

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
# Detections parser (MOT det.txt)
# -------------------------

def parse_dets(det_lines: List[str]) -> Dict[int, List[List[float]]]:
    """
    MOT det.txt:
      frame, -1, x, y, w, h, score, -1, -1, -1
    Retorna dict[frame_id] -> list[[x1,y1,x2,y2,score]]
    """
    outputs = defaultdict(list)
    for det in det_lines:
        det = det.strip().split(',')
        frame_id, ins_id = map(int, det[:2])
        if ins_id != -1:
            continue
        x, y, w, h, score = map(float, det[2:7])
        outputs[frame_id].append([x, y, x + w, y + h, score])
    return outputs


def list_images(img_dir: Path) -> List[Path]:
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    imgs = [p for p in sorted(img_dir.iterdir()) if p.is_file() and p.suffix.lower() in exts]
    return imgs


def infer_frames_location(frames_root: Path, scene: str, cam: str) -> Tuple[Path, str]:
    """
    Devuelve:
      (img_dir, img_subdir_name)
    donde img_dir contiene los frames reales y img_subdir_name se usa en file_name COCO.
    Soporta:
      frames_root/S01/c001/img1/*.jpg  -> img_subdir_name = "img1"
      frames_root/S01/c001/*.jpg       -> img_subdir_name = "" (vacío)
    """
    base = frames_root / scene / cam
    cand_img1 = base / "img1"
    if cand_img1.is_dir():
        return cand_img1, "img1"
    return base, ""


def infer_seqinfo_from_images(img_dir: Path, default_fps: int) -> Tuple[List[Path], int, int, int, int]:
    """
    Retorna (img_paths_sorted, fps, num_imgs, width, height)
    """
    img_paths = list_images(img_dir)
    if not img_paths:
        raise FileNotFoundError(f"Sin imágenes en {img_dir}")

    first = mmcv.imread(str(img_paths[0]))
    h, w = first.shape[:2]
    fps = int(default_fps)
    return img_paths, fps, len(img_paths), w, h


# -------------------------
# Convert: frames + det_root -> COCO json + det.pkl
# -------------------------

def convert_to_coco_from_separate_roots(
    scene: str,
    cam: str,
    frames_root: Path,
    det_root: Path,
    out_ann_dir: Path,
    default_fps: int,
) -> Tuple[Path, Path, str]:
    """
    Genera:
      out_ann_dir/train_cocoformat.json
      out_ann_dir/train_detections.pkl

    Retorna:
      (coco_json_path, det_pkl_path, img_prefix_for_mmtrack)
    """
    out_ann_dir.mkdir(parents=True, exist_ok=True)

    coco_json = out_ann_dir / "train_cocoformat.json"
    det_pkl = out_ann_dir / "train_detections.pkl"

    img_dir, img_subdir = infer_frames_location(frames_root, scene, cam)
    img_paths, fps, num_imgs, width, height = infer_seqinfo_from_images(img_dir, default_fps)

    # img_prefix: frames_root/S01
    img_prefix = str((frames_root / scene).resolve())

    # det file: det_root/S01/c001/det/det.txt
    det_txt = det_root / scene / cam / "det" / "det.txt"
    if det_txt.is_file():
        det_lines = mmcv.list_from_file(str(det_txt))
        img2dets = parse_dets(det_lines)
    else:
        img2dets = defaultdict(list)

    outputs = defaultdict(list)
    outputs["categories"] = [dict(id=1, name="pedestrian")]

    detections = dict(det_bboxes=dict())

    # COCO ids
    vid_id, img_id = 1, 1

    # En COCO video, "videos" suele tener 1 video por secuencia
    video_name = cam  # estable y consistente
    outputs["videos"].append(dict(id=vid_id, name=video_name, fps=fps, width=width, height=height))

    # Construcción de file_name:
    # - si img_subdir="img1": file_name="c001/img1/000001.jpg"
    # - si img_subdir="":    file_name="c001/000001.jpg"
    def mk_file_name(img_name: str) -> str:
        if img_subdir:
            return f"{cam}/{img_subdir}/{img_name}".replace("\\", "/")
        return f"{cam}/{img_name}".replace("\\", "/")

    for frame_id, img_path in enumerate(img_paths):
        name = img_path.name
        # frame number MOT: suele ser el stem numérico
        try:
            mot_frame_id = int(Path(name).stem)
        except Exception:
            mot_frame_id = frame_id + 1

        file_name = mk_file_name(name)

        outputs["images"].append(dict(
            id=img_id,
            video_id=vid_id,
            file_name=file_name,
            height=height,
            width=width,
            frame_id=frame_id,
            mot_frame_id=mot_frame_id
        ))

        # detections.pkl usa key = file_name (igual que el JSON)
        dets_this = np.array(img2dets[mot_frame_id], dtype=np.float32)
        if dets_this.ndim == 1:
            dets_this = np.zeros((0, 5), dtype=np.float32)
        detections["det_bboxes"][file_name] = [dets_this]

        img_id += 1

    outputs["num_instances"] = 0  # no hay GT aquí, sólo detecciones

    mmcv.dump(outputs, str(coco_json))
    mmcv.dump(detections, str(det_pkl))

    return coco_json, det_pkl, img_prefix


# -------------------------
# MMTracking test integrado
# -------------------------

def _set_cfg_by_dotted_key(cfg: Config, dotted_key: str, value) -> None:
    parts = dotted_key.split(".")
    cur = cfg
    for p in parts[:-1]:
        if isinstance(cur, Config):
            if p not in cur:
                cur[p] = {}
            cur = cur[p]
        else:
            if p not in cur:
                cur[p] = {}
            cur = cur[p]
    cur[parts[-1]] = value


def run_mmtrack_tracking(
    config_path: Path,
    cfg_updates: Dict[str, object],
    eval_metric: List[str],
    eval_options: Dict[str, object],
    work_dir: Path,
    checkpoint: Optional[str] = None,
    gpu_id: int = 0,
    fuse_conv_bn_flag: bool = False,
):
    cfg = Config.fromfile(str(config_path))

    # aplicar overrides
    for k, v in cfg_updates.items():
        _set_cfg_by_dotted_key(cfg, k, v)

    setup_multi_processes(cfg)

    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    if cfg.get("seed", None) is not None:
        set_random_seed(cfg.seed, deterministic=cfg.get("deterministic", False))

    cfg.data.test.test_mode = True
    cfg.gpu_ids = [gpu_id]

    rank, _ = get_dist_info()
    json_file = None
    if rank == 0 and work_dir is not None:
        mmcv.mkdir_or_exist(osp.abspath(str(work_dir)))
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        json_file = osp.join(str(work_dir), f"eval_{timestamp}.log.json")

    dataset = build_dataset(cfg.data.test)
    from mmtrack.datasets import build_dataloader
    from mmtrack.models import build_model
    from mmtrack.apis import single_gpu_test

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    cfg.device = get_device() if cfg.get("device", None) is None else cfg.device

    if cfg.get("test_cfg", False):
        model = build_model(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    else:
        model = build_model(cfg.model)

    model.init_weights()

    if checkpoint is not None:
        ckpt = load_checkpoint(model, checkpoint, map_location="cpu")
        if "meta" in ckpt and "CLASSES" in ckpt["meta"]:
            model.CLASSES = ckpt["meta"]["CLASSES"]

    if not hasattr(model, "CLASSES"):
        model.CLASSES = dataset.CLASSES

    if fuse_conv_bn_flag:
        model = fuse_conv_bn(model)

    model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)

    # Compatibilidad de firma con diferentes versiones
    try:
        outputs = single_gpu_test(model, data_loader, show=False)
    except TypeError:
        outputs = single_gpu_test(model, data_loader)
    
    # Guardar SIEMPRE el tracking en resfile_path si fue proporcionado
    if eval_options and "resfile_path" in eval_options and eval_options["resfile_path"]:
        save_tracking_results(dataset, outputs, str(eval_options["resfile_path"]))

    # -----------------------
    # Evitar evaluación si no hay GT
    # -----------------------
    metric = None
    do_eval = bool(eval_metric)

    if do_eval:
        try:
            img_prefix = str(cfg.data.test.img_prefix)  # p.ej. frames/S05
            # Inferir cam desde el ann_file que estamos usando:
            ann_path = str(cfg.data.test.ann_file)      # .../S05/c001/annotations/train_cocoformat.json
            cam_guess = Path(ann_path).parent.parent.name  # c001
            gt_guess = osp.join(img_prefix, cam_guess, "gt", "gt.txt")

            if not osp.isfile(gt_guess):
                print(f"[WARN] No existe GT en {gt_guess}. Omitiendo evaluación (--eval).")
                do_eval = False
        except Exception as e:
            # Si por alguna razón no podemos inferir, más seguro NO evaluar
            print(f"[WARN] No pude verificar GT. Omitiendo evaluación (--eval). Detalle: {e}")
            do_eval = False

    if do_eval:
        eval_kwargs = cfg.get("evaluation", {}).copy()
        for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule", "by_epoch"]:
            eval_kwargs.pop(key, None)

        eval_kwargs.update(dict(metric=eval_metric, **(eval_options or {})))
        metric = dataset.evaluate(outputs, **eval_kwargs)

        if json_file is not None:
            metric_dict = dict(config=str(config_path), mode="test", epoch=cfg.total_epochs)
            metric_dict.update(metric)
            mmcv.dump(metric_dict, json_file)

    return metric


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="MOT det -> COCO + tracking (MMTracking), leyendo CSV y usando roots separados.")
    ap.add_argument("--csv", required=True, help="CSV con columnas: 'scene' y 'cam number'")
    ap.add_argument("--frames-root", required=True, help="Raíz de frames: frames/S0x/c00x/(img1/)?*.jpg")
    ap.add_argument("--det-root", required=True, help="Raíz de detecciones: crops/S0x/c00x/det/det.txt")
    ap.add_argument("--out-root", required=True, help="Raíz de salida: out/S0x/c00x/(annotations,output)")
    ap.add_argument("--config", required=True, help="Config mmtrack (ej: configs/mot/deepsort/...).")
    ap.add_argument("--checkpoint", default=None, help="Checkpoint opcional.")
    ap.add_argument("--default-fps", type=int, default=25)

    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--device", default=None, help="Opcional: fuerza cfg.device (ej 'cuda:0' o 'cpu').")
    ap.add_argument("--fuse-conv-bn", action="store_true")

    ap.add_argument("--skip-convert-if-exists", action="store_true",
                    help="Si ya existen cocoformat.json y detections.pkl, no reconvierte.")
    ap.add_argument("--eval-metric", nargs="+", default=None)
    ap.add_argument("--annotations-dirname", default="annotations")
    ap.add_argument("--output-dirname", default="output")

    # Test
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--row-index", type=int, default=None)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--cam", default=None)

    args = ap.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    frames_root = Path(args.frames_root).expanduser().resolve()
    det_root = Path(args.det_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()

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
        print(f"\n[{i}/{total}] Procesando {scene}/{cam}")

        # Validación mínima de inputs
        # Frames: frames_root/S01/c001 o frames_root/S01/c001/img1
        try:
            img_dir, _ = infer_frames_location(frames_root, scene, cam)
            if not img_dir.exists():
                raise FileNotFoundError
        except Exception:
            skipped += 1
            print(f"  [SKIP] No encuentro frames en: {frames_root/scene/cam} (ni img1/).")
            if args.test:
                break
            continue

        # det.txt: det_root/S01/c001/det/det.txt (puede faltar; se permite, pero no habrá tracking útil)
        det_txt = det_root / scene / cam / "det" / "det.txt"
        if not det_txt.exists():
            print(f"  [WARN] No existe det.txt en: {det_txt} (se generará detections.pkl vacío)")

        seq_out_dir = out_root / scene / cam
        ann_dir = seq_out_dir / args.annotations_dirname
        work_dir = seq_out_dir / args.output_dirname

        coco_json = ann_dir / "train_cocoformat.json"
        det_pkl = ann_dir / "train_detections.pkl"

        if args.skip_convert_if_exists and coco_json.exists() and det_pkl.exists():
            # typo safe guard (si escribes mal el flag)
            pass

        if args.skip_convert_if_exists and coco_json.exists() and det_pkl.exists():
            # necesitamos img_prefix igual para cfg
            img_prefix = str((frames_root / scene).resolve())
            print("  - Conversión: omitida (ya existen annotations)")
        else:
            print("  - Convirtiendo (frames + det_root) -> COCO + detections.pkl ...")
            coco_json, det_pkl, img_prefix = convert_to_coco_from_separate_roots(
                scene=scene,
                cam=cam,
                frames_root=frames_root,
                det_root=det_root,
                out_ann_dir=ann_dir,
                default_fps=args.default_fps
            )
            print(f"    COCO: {coco_json}")
            print(f"    DET : {det_pkl}")
            print(f"    img_prefix: {img_prefix}")

        if not coco_json.exists() or not det_pkl.exists():
            print("  [ERROR] No se generaron los archivos de annotations esperados.", file=sys.stderr)
            if args.test:
                break
            continue

        # Tracking
        print("  - Ejecutando tracking (MMTracking) ...")
        cfg_updates = {
            # data_root puede ser cualquier cosa, pero mantenemos consistente por compatibilidad
            "data_root": str((out_root / scene / cam).resolve()),
            "data.test.detection_file": str(det_pkl.resolve()),
            "data.test.ann_file": str(coco_json.resolve()),
            "data.test.img_prefix": img_prefix,  # IMPORTANTÍSIMO: apunta a frames_root/S0x
        }
        if args.device is not None:
            cfg_updates["device"] = args.device

        eval_options = {
            "resfile_path": str(work_dir.resolve())
        }

        metric = run_mmtrack_tracking(
            config_path=config_path,
            cfg_updates=cfg_updates,
            eval_metric=args.eval_metric,
            eval_options=eval_options,
            work_dir=work_dir,
            checkpoint=args.checkpoint,
            gpu_id=args.gpu_id,
            fuse_conv_bn_flag=args.fuse_conv_bn,
        )

        print(f"  - Tracking finalizado. Output dir: {work_dir}")
        if metric is not None:
            print(f"  - Métricas: {metric}")

        ok += 1

        if args.test:
            break

    print("\nResumen:")
    print(f"- OK: {ok}")
    print(f"- SKIP: {skipped}")
    print(f"- TOTAL: {total}")


if __name__ == "__main__":
    import sys
    main()
