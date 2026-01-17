## Example
# python src\8_crops_to_silhouettes.py --csv inputs\raw_videos\input_data_S06_to_S09.csv --crops-root transforms\gt_crops\crops --out-root transforms\silhouettes --batch-size 8 --rank-mode mul --area-pow 1.0 --score-pow 1.0 --amp fp16 --tf32 --skip-if-exists

import sys
from pathlib import Path

# Ajusta esto al path real donde está el paquete "sam3"
SAM3_REPO_ROOT = Path(__file__).resolve().parent / "sam3"  # src/sam3
sys.path.insert(0, str(SAM3_REPO_ROOT))

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch

import sam3
from sam3 import build_sam3_image_model
from sam3.train.data.collator import collate_fn_api as collate
from sam3.model.utils.misc import copy_data_to_device

from sam3.train.data.sam3_image_dataset import (
    InferenceMetadata,
    FindQueryLoaded,
    Image as SAMImage,
    Datapoint,
)

from sam3.train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
from sam3.eval.postprocessors import PostProcessImage


# =========================================================
# CSV + normalización
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


def load_pairs_from_csv(csv_path: Path, test: bool, row_index: Optional[int], scene: Optional[str], cam: Optional[str]) -> List[Tuple[str, str]]:
    df = pd.read_csv(csv_path)
    for col in ("scene", "cam number"):
        if col not in df.columns:
            raise KeyError(f"Falta columna requerida en CSV: {col!r}. Columnas: {list(df.columns)}")

    if test:
        if row_index is not None:
            if row_index < 0 or row_index >= len(df):
                raise IndexError(f"--row-index fuera de rango (0..{len(df)-1}). Recibido: {row_index}")
            df = df.iloc[[row_index]].copy()
            print(f"Modo TEST: usando fila índice {row_index}")
        else:
            if scene is not None:
                ns = normalize_scene(scene)
                df = df[df["scene"].astype(str).apply(lambda x: normalize_scene(x) == ns)].copy()
                if df.empty:
                    raise ValueError(f"Modo TEST: no hay filas para escena {ns}")
            if cam is not None:
                nc = normalize_cam(cam)
                df = df[df["cam number"].astype(str).apply(lambda x: normalize_cam(x) == nc)].copy()
                if df.empty:
                    raise ValueError(f"Modo TEST: no hay filas para cam {nc}")
            print(f"Modo TEST: filas seleccionadas: {len(df)}")

    return sorted({(normalize_scene(r["scene"]), normalize_cam(r["cam number"])) for _, r in df.iterrows()})


def find_crops_for_pair(crops_root: Path, scene: str, cam: str) -> List[Path]:
    patt = f"ID*/{scene}/{cam}/*.jpg"
    return sorted(crops_root.glob(patt))


def parse_global_id_from_path(crop_path: Path) -> int:
    for p in crop_path.parents:
        if p.name.startswith("ID"):
            m = re.search(r"(\d+)", p.name)
            if m:
                return int(m.group(1))
    raise ValueError(f"No pude extraer global_id desde {crop_path}")


def parse_frame_from_crop_name(crop_path: Path) -> Optional[int]:
    m = re.search(r"_(\d+)_", crop_path.name)
    if m:
        return int(m.group(1))
    m2 = re.search(r"(\d+)", crop_path.stem)
    return int(m2.group(1)) if m2 else None


# =========================================================
# RLE COCO-like
# =========================================================

def mask_to_rle_coco(binary_mask01: np.ndarray) -> Dict:
    m = (binary_mask01 > 0).astype(np.uint8)
    h, w = m.shape
    flat = m.flatten(order="F")

    counts: List[int] = []
    prev = 0
    run_len = 0
    for v in flat:
        if v == prev:
            run_len += 1
        else:
            counts.append(run_len)
            run_len = 1
            prev = v
    counts.append(run_len)

    return {"size": [h, w], "counts": counts}


# =========================================================
# SAM3 datapoints (batch)
# =========================================================

GLOBAL_COUNTER = 1

def create_empty_datapoint() -> Datapoint:
    return Datapoint(find_queries=[], images=[])


def set_image(datapoint: Datapoint, pil_image: Image.Image) -> None:
    w, h = pil_image.size
    datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]


def add_text_prompt(datapoint: Datapoint, text_query: str) -> int:
    global GLOBAL_COUNTER
    assert len(datapoint.images) == 1, "Set image first"
    w, h = datapoint.images[0].size

    qid = GLOBAL_COUNTER
    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=text_query,
            image_id=0,
            object_ids_output=[],
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=qid,
                original_image_id=qid,
                original_category_id=1,
                original_size=[w, h],
                object_id=0,
                frame_index=0,
            )
        )
    )
    GLOBAL_COUNTER += 1
    return qid


def chunked(lst: List[Path], n: int) -> List[List[Path]]:
    return [lst[i:i+n] for i in range(0, len(lst), n)]


def to_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.array(x)


# =========================================================
# NUEVO: selección por (área máscara) + score
# =========================================================

def pick_idx_by_mask_area_and_score(
    masks_np: np.ndarray,
    scores_np: np.ndarray,
    mode: str,
    area_pow: float,
    score_pow: float,
    w_area: float,
    w_score: float,
) -> Tuple[int, np.ndarray]:
    """
    masks_np: (N,H,W) o (N,1,H,W)
    scores_np: (N,)

    Retorna (idx, ranks_vector)
    """
    if masks_np.ndim == 4:
        masks_np = masks_np[:, 0, :, :]

    # área = píxeles positivos
    # (si la máscara es float, >0 funciona)
    areas = (masks_np > 0.0).sum(axis=(1, 2)).astype(np.float64)

    # normalizar por tamaño de imagen
    H, W = masks_np.shape[1], masks_np.shape[2]
    denom = float(max(1, H * W))
    area_norm = areas / denom

    # asegurar scores en float64
    s = scores_np.astype(np.float64)

    if mode == "mul":
        # rank = (area_norm^a) * (score^b)
        # eps para evitar 0^pow
        eps = 1e-12
        ranks = np.power(np.maximum(area_norm, eps), area_pow) * np.power(np.maximum(s, eps), score_pow)
    elif mode == "sum":
        # rank = w_area * area_norm + w_score * score
        ranks = w_area * area_norm + w_score * s
    else:
        raise ValueError(f"Modo de ranking inválido: {mode}. Use 'mul' o 'sum'.")

    idx = int(np.argmax(ranks))
    return idx, ranks


# =========================================================
# Main
# =========================================================

def main():
    ap = argparse.ArgumentParser(description="Batch SAM3: crops -> silhouettes + index (selección por área máscara + score).")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--crops-root", required=True)
    ap.add_argument("--out-root", required=True)

    ap.add_argument("--prompt", default="Person")
    ap.add_argument("--device", default=None, help="cuda | cuda:0 | cpu (default auto)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=1008)
    ap.add_argument("--det-thr", type=float, default=0.5)

    ap.add_argument("--amp", choices=["none", "bf16", "fp16"], default="bf16")
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--skip-if-exists", action="store_true")

    # NUEVOS FLAGS de ranking
    ap.add_argument("--rank-mode", choices=["mul", "sum"], default="mul",
                    help="mul: (area_norm^area_pow)*(score^score_pow). sum: w_area*area_norm + w_score*score")
    ap.add_argument("--area-pow", type=float, default=1.0)
    ap.add_argument("--score-pow", type=float, default=1.0)
    ap.add_argument("--w-area", type=float, default=0.7)
    ap.add_argument("--w-score", type=float, default=0.3)

    ap.add_argument("--index-format", choices=["jsonl", "csv"], default="jsonl")
    ap.add_argument("--index-name", default="sam3_masks_index")

    # Test
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--row-index", type=int, default=None)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--cam", default=None)
    ap.add_argument("--limit", type=int, default=None)

    args = ap.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    crops_root = Path(args.crops_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()

    if args.device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device
    device = torch.device(device_str)

    if args.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- model ---
    model = build_sam3_image_model()
    model.to(device)
    model.eval()

    # --- transforms ---
    transform = ComposeAPI(
        transforms=[
            RandomResizeAPI(sizes=args.img_size, max_size=args.img_size, square=True, consistent_transform=False),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # --- postprocessor ---
    postprocessor = PostProcessImage(
        max_dets_per_img=-1,
        iou_type="segm",
        use_original_sizes_box=True,
        use_original_sizes_mask=True,
        convert_mask_to_rle=False,
        detection_threshold=args.det_thr,
        to_cpu=False,
    )

    pairs = load_pairs_from_csv(csv_path, args.test, args.row_index, args.scene, args.cam)

    index_ext = "jsonl" if args.index_format == "jsonl" else "csv"
    index_path = out_root / f"{args.index_name}.{index_ext}"
    index_path.parent.mkdir(parents=True, exist_ok=True)

    if args.index_format == "csv" and not index_path.exists():
        with index_path.open("w", newline="", encoding="utf-8") as f:
            f.write("scene,cam,global_id,frame,crop_path,mask_png_path,score,scores_json,rank_json,selected_box_json,rle_json\n")

    # recolectar crops desde el CSV
    all_crops: List[Path] = []
    for (scene, cam) in pairs:
        all_crops.extend(find_crops_for_pair(crops_root, scene, cam))

    if not all_crops:
        print("[WARN] No encontré crops para lo indicado por el CSV.")
        return

    if args.limit is not None:
        all_crops = all_crops[: max(0, int(args.limit))]

    pbar = tqdm(total=len(all_crops), desc="SAM3 batched", unit="crop")

    use_amp = (device.type == "cuda" and args.amp != "none")
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16

    total_saved = 0

    for batch_paths in chunked(all_crops, args.batch_size):
        datapoints: List[Datapoint] = []
        qid_to_info: Dict[int, Dict] = {}

        for crop_path in batch_paths:
            gid = parse_global_id_from_path(crop_path)

            cam = crop_path.parent.name
            scene = crop_path.parent.parent.name
            frame = parse_frame_from_crop_name(crop_path)

            mask_png_dir = out_root / "masks" / f"ID{gid:04d}" / scene / cam
            mask_png_dir.mkdir(parents=True, exist_ok=True)
            mask_png_path = mask_png_dir / (crop_path.stem + ".png")

            if args.skip_if_exists and mask_png_path.exists():
                pbar.update(1)
                continue

            try:
                img = Image.open(str(crop_path)).convert("RGB")
            except Exception:
                pbar.update(1)
                continue

            dp = create_empty_datapoint()
            set_image(dp, img)
            qid = add_text_prompt(dp, args.prompt)
            dp = transform(dp)

            datapoints.append(dp)

            try:
                crop_rel = str(crop_path.resolve().relative_to(crops_root.resolve())).replace("\\", "/")
            except Exception:
                crop_rel = str(crop_path.resolve()).replace("\\", "/")
            try:
                mask_rel = str(mask_png_path.resolve().relative_to(out_root.resolve())).replace("\\", "/")
            except Exception:
                mask_rel = str(mask_png_path.resolve()).replace("\\", "/")

            qid_to_info[qid] = {
                "scene": scene,
                "cam": cam,
                "global_id": gid,
                "frame": frame,
                "crop_rel": crop_rel,
                "mask_png_abs": str(mask_png_path.resolve()),
                "mask_rel": mask_rel,
            }

        if not datapoints:
            continue

        batch = collate(datapoints, dict_key="dummy")["dummy"]
        batch = copy_data_to_device(batch, device, non_blocking=True)

        with torch.inference_mode():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    out = model(batch)
            else:
                out = model(batch)

        processed = postprocessor.process_results(out, batch.find_metadatas)

        for qid, info in qid_to_info.items():
            res = processed.get(qid, None)
            if res is None:
                pbar.update(1)
                continue

            boxes = res.get("boxes", None)
            masks = res.get("masks", None)
            scores = res.get("scores", None)

            if masks is None or scores is None:
                pbar.update(1)
                continue

            masks_np = to_numpy(masks)
            scores_np = to_numpy(scores).reshape(-1)

            if scores_np.size == 0:
                pbar.update(1)
                continue

            # ranking por área de máscara + score
            pick, ranks = pick_idx_by_mask_area_and_score(
                masks_np=masks_np,
                scores_np=scores_np,
                mode=args.rank_mode,
                area_pow=args.area_pow,
                score_pow=args.score_pow,
                w_area=args.w_area,
                w_score=args.w_score,
            )

            sel_score = float(scores_np[pick])

            # bbox solo para auditoría si existe
            sel_box = None
            if boxes is not None:
                boxes_np = to_numpy(boxes).reshape(-1, 4)
                if boxes_np.shape[0] > pick:
                    sel_box = [float(x) for x in boxes_np[pick].tolist()]

            m = masks_np[pick]
            if m.ndim == 3:
                m = m.squeeze(0)

            m01 = (m > 0.0).astype(np.uint8)
            m255 = (m01 * 255).astype(np.uint8)

            try:
                Image.fromarray(m255, mode="L").save(info["mask_png_abs"])
            except Exception:
                pbar.update(1)
                continue

            rle = mask_to_rle_coco(m01)

            record = {
                "scene": info["scene"],
                "cam": info["cam"],
                "global_id": int(info["global_id"]),
                "frame": int(info["frame"]) if info["frame"] is not None else None,
                "crop_path": info["crop_rel"],
                "mask_png_path": info["mask_rel"],
                "prompt": args.prompt,
                "score": sel_score,
                "scores": [float(x) for x in scores_np.tolist()],
                "rank": [float(x) for x in np.array(ranks, dtype=np.float64).tolist()],
                "selected_box": sel_box,
                "rle": rle,
                "rank_cfg": {
                    "mode": args.rank_mode,
                    "area_pow": args.area_pow,
                    "score_pow": args.score_pow,
                    "w_area": args.w_area,
                    "w_score": args.w_score,
                },
            }

            if args.index_format == "jsonl":
                with index_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                scores_json = json.dumps(record["scores"], ensure_ascii=False)
                rank_json = json.dumps(record["rank"], ensure_ascii=False)
                box_json = json.dumps(record["selected_box"], ensure_ascii=False)
                rle_json = json.dumps(record["rle"], ensure_ascii=False)
                with index_path.open("a", newline="", encoding="utf-8") as f:
                    f.write(
                        f"{record['scene']},{record['cam']},{record['global_id']},{record['frame'] if record['frame'] is not None else ''},"
                        f"{record['crop_path']},{record['mask_png_path']},{record['score']},"
                        f"\"{scores_json.replace('\"','\"\"')}\","
                        f"\"{rank_json.replace('\"','\"\"')}\","
                        f"\"{box_json.replace('\"','\"\"')}\","
                        f"\"{rle_json.replace('\"','\"\"')}\"\n"
                        )

            total_saved += 1
            pbar.update(1)

    pbar.close()

    print("\nResumen:")
    print(f"- Crops procesados: {len(all_crops)}")
    print(f"- Máscaras guardadas: {total_saved}")
    print(f"- Índice: {index_path}")


if __name__ == "__main__":
    main()
