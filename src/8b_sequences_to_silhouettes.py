## Example
# python src\8b_sequences_to_silhouettes.py --csv outputs\metadata\sequences_mars.csv --rgb-root transforms\frames --out-root transforms\silhouettes --batch-size 8 --rank-mode mul --area-pow 1.0 --score-pow 1.0 --amp fp16 --tf32 --skip-if-exists

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


def chunked(lst, n: int):
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
    areas = (masks_np > 0.0).sum(axis=(1, 2)).astype(np.float64)

    # normalizar por tamaño de imagen
    H, W = masks_np.shape[1], masks_np.shape[2]
    denom = float(max(1, H * W))
    area_norm = areas / denom

    # asegurar scores en float64
    s = scores_np.astype(np.float64)

    if mode == "mul":
        # rank = (area_norm^a) * (score^b)
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
    ap = argparse.ArgumentParser(description="Batch SAM3: sequences -> silhouettes (mirroring rgb_dir).")
    ap.add_argument("--csv", required=True, help="Path to sequences CSV (e.g. sequences_mars.csv)")
    ap.add_argument("--rgb-root", required=True, help="Root directory for frames (e.g. transforms/frames)")
    ap.add_argument("--out-root", required=True, help="Output root for silhouettes")

    ap.add_argument("--prompt", default="Person")
    ap.add_argument("--device", default=None, help="cuda | cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=1008)
    ap.add_argument("--det-thr", type=float, default=0.5)

    ap.add_argument("--amp", choices=["none", "bf16", "fp16"], default="bf16")
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--skip-if-exists", action="store_true")

    # Ranking
    ap.add_argument("--rank-mode", choices=["mul", "sum"], default="mul")
    ap.add_argument("--area-pow", type=float, default=1.0)
    ap.add_argument("--score-pow", type=float, default=1.0)
    ap.add_argument("--w-area", type=float, default=0.7)
    ap.add_argument("--w-score", type=float, default=0.3)

    ap.add_argument("--index-format", choices=["jsonl", "csv"], default="jsonl")
    ap.add_argument("--index-name", default="silhouettes_index")

    # Limit for testing
    ap.add_argument("--limit", type=int, default=None)

    args = ap.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    rgb_root = Path(args.rgb_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()

    if args.device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device
    device = torch.device(device_str)

    if args.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[INFO] TF32 enabled.")

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

    df = pd.read_csv(csv_path)
    if args.limit:
        df = df.head(args.limit)

    index_ext = "jsonl" if args.index_format == "jsonl" else "csv"
    index_path = out_root / f"{args.index_name}.{index_ext}"
    index_path.parent.mkdir(parents=True, exist_ok=True)

    if args.index_format == "csv" and not index_path.exists():
        with index_path.open("w", newline="", encoding="utf-8") as f:
            f.write("sequence_id,person_global_id,scene,cam_id,frame,rgb_path,mask_png_path,score,rle_json\n")

    # Collect all image tasks from sequences
    all_image_tasks = []
    print("Collecting images from sequences...")
    for _, row in df.iterrows():
        seq_id = row["sequence_id"]
        rgb_dir_rel_str = str(row["rgb_dir"]).replace("\\", "/")
        rgb_dir_rel = Path(rgb_dir_rel_str)
        rgb_dir_abs = rgb_root / rgb_dir_rel
        
        if not rgb_dir_abs.exists():
            print(f"[WARN] Sequence directory not found: {rgb_dir_abs}")
            continue
            
        # find all jpgs in that directory (these are already bbox crops)
        images_in_seq = sorted(list(rgb_dir_abs.glob("*.jpg")))
        for img_p in images_in_seq:
            # Construct mirroring path in silhouettes root
            # rgb_dir is e.g. "rgb/ID0001/S0002/c001/cycle0001"
            # We want output to mirror this structure under out_root.
            # If the user wants a folder 'silhouettes', we typically use it as out_root.
            
            parts = list(rgb_dir_rel.parts)
            if parts and parts[0] == "rgb":
                # Remove 'rgb' prefix to mirror structure inside out_root
                rel_to_id = Path(*parts[1:])
            else:
                rel_to_id = rgb_dir_rel
                
            mask_png_dir = out_root / rel_to_id
            mask_png_path = mask_png_dir / (img_p.stem + ".png")
            
            all_image_tasks.append({
                "sequence_id": seq_id,
                "person_global_id": row["person_global_id"],
                "scene": row["scene"],
                "cam_id": row["cam_id"],
                "rgb_path_abs": img_p,
                "mask_path_abs": mask_png_path,
                "rgb_rel": str(rgb_dir_rel / img_p.name).replace("\\", "/"),
                "mask_rel": str(rel_to_id / (img_p.stem + ".png")).replace("\\", "/")
            })

    if not all_image_tasks:
        print("[ERROR] No images found to process.")
        return

    pbar = tqdm(total=len(all_image_tasks), desc="Processing Frames", unit="frame")

    use_amp = (device.type == "cuda" and args.amp != "none")
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16

    total_saved = 0

    for batch_tasks in chunked(all_image_tasks, args.batch_size):
        datapoints: List[Datapoint] = []
        qid_to_info: Dict[int, Dict] = {}

        for task in batch_tasks:
            mask_png_path = task["mask_path_abs"]
            if args.skip_if_exists and mask_png_path.exists():
                pbar.update(1)
                continue

            try:
                img = Image.open(str(task["rgb_path_abs"])).convert("RGB")
            except Exception:
                pbar.update(1)
                continue

            mask_png_path.parent.mkdir(parents=True, exist_ok=True)

            dp = create_empty_datapoint()
            set_image(dp, img)
            qid = add_text_prompt(dp, args.prompt)
            dp = transform(dp)

            datapoints.append(dp)
            qid_to_info[qid] = task

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

        for qid, task in qid_to_info.items():
            res = processed.get(qid, None)
            if res is None:
                pbar.update(1)
                continue

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

            pick, ranks = pick_idx_by_mask_area_and_score(
                masks_np=masks_np,
                scores_np=scores_np,
                mode=args.rank_mode,
                area_pow=args.area_pow,
                score_pow=args.score_pow,
                w_area=args.w_area,
                w_score=args.w_score,
            )

            m = masks_np[pick]
            if m.ndim == 3:
                m = m.squeeze(0)

            m01 = (m > 0.0).astype(np.uint8)
            m255 = (m01 * 255).astype(np.uint8)

            try:
                Image.fromarray(m255, mode="L").save(task["mask_path_abs"])
            except Exception:
                pbar.update(1)
                continue

            rle = mask_to_rle_coco(m01)

            record = {
                "sequence_id": task["sequence_id"],
                "person_global_id": int(task["person_global_id"]),
                "scene": task["scene"],
                "cam_id": task["cam_id"],
                "rgb_path": task["rgb_rel"],
                "mask_png_path": task["mask_rel"],
                "score": float(scores_np[pick]),
                "rle": rle
            }

            if args.index_format == "jsonl":
                with index_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                rle_escaped = rle_json.replace('"', '""')
                with index_path.open("a", newline="", encoding="utf-8") as f:
                    f.write(
                        f"{record['sequence_id']},{record['person_global_id']},{record['scene']},{record['cam_id']},"
                        f"{record['rgb_path']},{record['mask_png_path']},{record['score']},"
                        f"\"{rle_escaped}\"\n"
                    )

            total_saved += 1
            pbar.update(1)

    pbar.close()
    print(f"\nFinished! Processed frames: {total_saved}")
    print(f"Index saved to: {index_path}")

if __name__ == "__main__":
    main()
