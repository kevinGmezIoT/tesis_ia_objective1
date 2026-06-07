## Example
# python src\9_generate_dataset.py --csv inputs\raw_videos\input_data_S06_to_S09.csv --crops-root transforms\gt_crops\crops --silhouettes-root transforms\silhouettes\masks --out-root dataset\output_v1 --seq-length 25 --overlap 10 --require-silhouettes

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm


# -------------------------
# Normalización Sxx / cxxx
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


def natural_sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", str(s))]


# -------------------------
# Parsing IDs / frames
# -------------------------

def parse_person_global_id(id_dir_name: str) -> int:
    m = re.search(r"(\d+)", id_dir_name)
    if not m:
        raise ValueError(f"No pude extraer global_id desde carpeta: {id_dir_name}")
    return int(m.group(1))


def extract_frame_number_from_name(filename: str) -> Optional[int]:
    """
    Soporta:
      - c003_000364_True.jpg -> 364
      - 000364.jpg -> 364
    Heurística: toma el grupo numérico más largo; si empate, el último.
    """
    nums = re.findall(r"(\d+)", filename)
    if not nums:
        return None
    nums_sorted = sorted(nums, key=lambda x: (len(x), nums.index(x)))
    return int(nums_sorted[-1])


def rel_to_out_root(out_root: Path, path: Optional[Path]) -> str:
    """
    Devuelve path relativo a out_root en formato POSIX.
    Si path es None o vacío, devuelve "".
    Si path no cuelga de out_root, devuelve un path POSIX absoluto (fallback).
    """
    if path is None:
        return ""
    s = str(path).strip()
    if s == "":
        return ""

    out_root_abs = out_root.resolve()
    path_abs = Path(s).resolve()

    try:
        return path_abs.relative_to(out_root_abs).as_posix()
    except ValueError:
        # Fallback: no está bajo out_root (o mezcla de relativo/absoluto)
        return path_abs.as_posix()



# -------------------------
# Carga de pares desde CSV
# -------------------------

def load_pairs_from_csv(csv_path: Path) -> List[Tuple[str, str]]:
    df = pd.read_csv(csv_path)
    for col in ("scene", "cam number"):
        if col not in df.columns:
            raise KeyError(f"Falta columna requerida en CSV: {col!r}. Columnas: {list(df.columns)}")
    pairs = sorted({(normalize_scene(r["scene"]), normalize_cam(r["cam number"])) for _, r in df.iterrows()})
    return pairs


def filter_pairs_for_test(pairs: List[Tuple[str, str]], scene: Optional[str], cam: Optional[str]) -> List[Tuple[str, str]]:
    """
    Si --scene está definido y --cam NO, devuelve TODAS las cámaras de esa escena.
    """
    if scene is None and cam is None:
        return pairs[:1]

    out = pairs
    if scene is not None:
        ns = normalize_scene(scene)
        out = [p for p in out if p[0] == ns]

    if cam is not None:
        nc = normalize_cam(cam)
        out = [p for p in out if p[1] == nc]

    if not out:
        raise ValueError("Modo TEST: no hay (scene, cam) que coincidan con los filtros.")
    return out


# -------------------------
# Ciclos (secuencias) por frames
# -------------------------

def build_cycles_from_frames(
    frames: List[Tuple[int, Path]],
    seq_length: int,
    overlap: int,
    max_frame_gap: int,
) -> List[List[Tuple[int, Path]]]:
    if not frames or len(frames) < seq_length:
        return []

    frames = sorted(frames, key=lambda x: x[0])

    continuous: List[List[Tuple[int, Path]]] = []
    cur: List[Tuple[int, Path]] = []

    for fn, p in frames:
        if not cur:
            cur.append((fn, p))
            continue
        if fn - cur[-1][0] <= max_frame_gap:
            cur.append((fn, p))
        else:
            if len(cur) >= seq_length:
                continuous.append(cur)
            cur = [(fn, p)]

    if len(cur) >= seq_length:
        continuous.append(cur)

    step = seq_length - overlap
    if step <= 0:
        step = 1

    cycles: List[List[Tuple[int, Path]]] = []
    for seq in continuous:
        for i in range(0, len(seq) - seq_length + 1, step):
            chunk = seq[i:i + seq_length]
            if len(chunk) == seq_length:
                cycles.append(chunk)

    return cycles


# -------------------------
# Escaneo de archivos por par (scene/cam)
# -------------------------

def scan_rgb_and_masks_for_pair(
    crops_root: Path,
    silhouettes_root: Path,
    scene: str,
    cam: str,
) -> Dict[int, List[Tuple[int, Path, Optional[Path]]]]:
    """
    Retorna dict[person_global_id] -> lista de (frame_num, rgb_path, mask_path|None)
    """
    out: Dict[int, List[Tuple[int, Path, Optional[Path]]]] = {}

    for id_dir in sorted(crops_root.glob("ID*/"), key=lambda p: natural_sort_key(p.name)):
        if not id_dir.is_dir():
            continue

        pid = parse_person_global_id(id_dir.name)

        rgb_dir = id_dir / scene / cam
        if not rgb_dir.is_dir():
            continue

        mask_dir = silhouettes_root / id_dir.name / scene / cam

        rgb_files = sorted([p for p in rgb_dir.glob("*.jpg") if p.is_file()], key=lambda p: natural_sort_key(p.name))
        if not rgb_files:
            continue

        tuples: List[Tuple[int, Path, Optional[Path]]] = []
        for rgb_path in rgb_files:
            fn = extract_frame_number_from_name(rgb_path.name)
            if fn is None:
                continue

            mask_path = None
            if mask_dir.is_dir():
                cand = mask_dir / (rgb_path.stem + ".png")
                if cand.is_file():
                    mask_path = cand

            tuples.append((fn, rgb_path, mask_path))

        if tuples:
            out[pid] = tuples

    return out


# -------------------------
# Copiado de un ciclo (con nombres = frame_number)
# -------------------------

def copy_cycle_files(
    cycle: List[Tuple[int, Path]],
    fn_to_mask: Dict[int, Optional[Path]],
    dst_rgb_cycle_dir: Path,
    dst_sil_cycle_dir: Path,
    require_silhouettes: bool,
    drop_cycle_if_missing_mask: bool,
    overwrite: bool,
) -> Tuple[bool, List[int], List[Path], List[Path]]:
    """
    Copia N frames del ciclo a:
      dst_rgb_cycle_dir/<frame_number>.jpg     (ej: 000364.jpg si frame_number=364)
      dst_sil_cycle_dir/<frame_number>.png

    Retorna:
      (ok, frame_nums, rgb_paths_dst, sil_paths_dst)
    """
    frame_nums = [fn for fn, _ in cycle]

    missing = any(fn_to_mask.get(fn) is None for fn in frame_nums)
    if drop_cycle_if_missing_mask and missing:
        return False, [], [], []
    if require_silhouettes and missing:
        return False, [], [], []

    dst_rgb_cycle_dir.mkdir(parents=True, exist_ok=True)
    dst_sil_cycle_dir.mkdir(parents=True, exist_ok=True)

    rgb_dst_list: List[Path] = []
    sil_dst_list: List[Path] = []

    for fn, rgb_src in cycle:
        rgb_dst = dst_rgb_cycle_dir / f"{fn:06d}.jpg"
        sil_dst = dst_sil_cycle_dir / f"{fn:06d}.png"

        if overwrite or (not rgb_dst.exists()):
            shutil.copy2(rgb_src, rgb_dst)

        mask_src = fn_to_mask.get(fn)
        if mask_src is not None:
            if overwrite or (not sil_dst.exists()):
                shutil.copy2(mask_src, sil_dst)

        rgb_dst_list.append(rgb_dst)
        sil_dst_list.append(sil_dst if sil_dst.exists() else None)


    return True, frame_nums, rgb_dst_list, sil_dst_list


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Crea dataset físico (copiando) y genera CSVs de ciclos RGB+Silueta.")
    ap.add_argument("--csv", required=True, help="CSV referencia con columnas: 'scene' y 'cam number'")

    ap.add_argument("--crops-root", required=True,
                    help=r"Raíz crops: transforms\gt_crops\crops (ID000x/S0x/c00x/*.jpg)")
    ap.add_argument("--silhouettes-root", required=True,
                    help=r"Raíz siluetas: transforms\silhouettes\masks (ID000x/S0x/c00x/*.png)")

    ap.add_argument("--out-root", required=True,
                    help=r"Salida: crea out_root\rgb\... y out_root\silhouette\... + out_root\metadata\csvs")

    ap.add_argument("--seq-length", type=int, default=25)
    ap.add_argument("--overlap", type=int, default=10)
    ap.add_argument("--max-frame-gap", type=int, default=10)

    ap.add_argument("--require-silhouettes", action="store_true",
                    help="Si activo: descarta ciclos con al menos 1 máscara faltante.")
    ap.add_argument("--drop-cycle-if-missing-mask", action="store_true",
                    help="Descarta ciclo completo si falta al menos una máscara (más estricto).")


    ap.add_argument("--overwrite", action="store_true", help="Sobrescribe archivos ya copiados.")
    ap.add_argument("--skip-existing-cycles", action="store_true",
                    help="Si el folder del ciclo ya existe, omite el ciclo (rápido para re-runs).")

    # Test
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--scene", default=None, help="Ej: S05")
    ap.add_argument("--cam", default=None, help="Ej: c003")
    ap.add_argument("--limit-cycles", type=int, default=None, help="En test: limita # de ciclos globales.")

    args = ap.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    crops_root = Path(args.crops_root).expanduser().resolve()
    sil_root = Path(args.silhouettes_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()

    pairs = load_pairs_from_csv(csv_path)
    if args.test:
        pairs = filter_pairs_for_test(pairs, args.scene, args.cam)
        print(f"[TEST] Pares a procesar: {pairs}")

    rgb_out_root = out_root / "rgb"
    sil_out_root = out_root / "silhouette"
    meta_out_root = out_root / "metadata"
    meta_out_root.mkdir(parents=True, exist_ok=True)

    sequences_rows: List[Dict] = []
    frames_rows: List[Dict] = []

    global_cycle_counter = 0

    for scene, cam in pairs:
        print(f"\nProcesando {scene}/{cam} ...")

        pid_to_triples = scan_rgb_and_masks_for_pair(
            crops_root=crops_root,
            silhouettes_root=sil_root,
            scene=scene,
            cam=cam,
        )

        if not pid_to_triples:
            print(f"  [WARN] No encontré crops para {scene}/{cam}.")
            continue

        for pid, triples in tqdm(pid_to_triples.items(), desc=f"{scene}/{cam} IDs", unit="id"):
            triples_sorted = sorted(triples, key=lambda x: x[0])

            frames_for_cycles = [(fn, rgb_path) for fn, rgb_path, _ in triples_sorted]
            cycles = build_cycles_from_frames(
                frames=frames_for_cycles,
                seq_length=args.seq_length,
                overlap=args.overlap,
                max_frame_gap=args.max_frame_gap,
            )
            if not cycles:
                continue

            fn_to_mask: Dict[int, Optional[Path]] = {fn: m for fn, _, m in triples_sorted}

            cycle_idx = 0
            for cycle in cycles:
                cycle_idx += 1
                global_cycle_counter += 1

                sequence_id = f"{scene}_{cam}_ID{pid:04d}_cyc{cycle_idx:04d}"
                cycle_dir_name = f"cycle{cycle_idx:04d}"

                dst_rgb_cycle_dir = rgb_out_root / f"ID{pid:04d}" / scene / cam / cycle_dir_name
                dst_sil_cycle_dir = sil_out_root / f"ID{pid:04d}" / scene / cam / cycle_dir_name

                if args.skip_existing_cycles and dst_rgb_cycle_dir.exists() and dst_sil_cycle_dir.exists():
                    if args.test and args.limit_cycles is not None and global_cycle_counter >= int(args.limit_cycles):
                        break
                    continue

                ok, frame_nums, rgb_dst_list, sil_dst_list = copy_cycle_files(
                    cycle=cycle,
                    fn_to_mask=fn_to_mask,
                    dst_rgb_cycle_dir=dst_rgb_cycle_dir,
                    dst_sil_cycle_dir=dst_sil_cycle_dir,
                    require_silhouettes=args.require_silhouettes,
                    drop_cycle_if_missing_mask=args.drop_cycle_if_missing_mask,
                    overwrite=args.overwrite,
                )
                if not ok:
                    continue

                sequences_rows.append({
                    "sequence_id": sequence_id,
                    "person_global_id": pid,
                    "scene": scene,
                    "cam_id": cam,
                    "cycle_index": cycle_idx,
                    # rutas RELATIVAS al out-root (como pediste)
                    "rgb_dir": rel_to_out_root(out_root, dst_rgb_cycle_dir),
                    "silhouette_dir": rel_to_out_root(out_root, dst_sil_cycle_dir),
                    "num_frames": len(frame_nums),
                    "frame_start": int(frame_nums[0]),
                    "frame_end": int(frame_nums[-1]),
                    "frame_numbers_json": json.dumps(frame_nums, ensure_ascii=False),
                })

                for i in range(len(frame_nums)):
                    frames_rows.append({
                        "sequence_id": sequence_id,
                        "person_global_id": pid,
                        "scene": scene,
                        "cam_id": cam,
                        "cycle_index": cycle_idx,
                        "frame_number": int(frame_nums[i]),
                        # paths RELATIVOS al out-root
                        "rgb_path": rel_to_out_root(out_root, rgb_dst_list[i]),
                        "silhouette_path": rel_to_out_root(out_root, sil_dst_list[i]),
                    })

                if args.test and args.limit_cycles is not None and global_cycle_counter >= int(args.limit_cycles):
                    break

            if args.test and args.limit_cycles is not None and global_cycle_counter >= int(args.limit_cycles):
                break

        if args.test and args.limit_cycles is not None and global_cycle_counter >= int(args.limit_cycles):
            break

    if not sequences_rows:
        print("\n[WARN] No se generaron secuencias. Revisa rutas y parámetros.")
        return

    seq_df = pd.DataFrame(sequences_rows)
    frm_df = pd.DataFrame(frames_rows)

    seq_csv = meta_out_root / "sequences.csv"
    frm_csv = meta_out_root / "sequence_frames.csv"

    seq_df.to_csv(seq_csv, index=False, encoding="utf-8")
    frm_df.to_csv(frm_csv, index=False, encoding="utf-8")

    print("\nOK:")
    print(f"- Dataset RGB:        {rgb_out_root}")
    print(f"- Dataset Silhouette: {sil_out_root}")
    print(f"- sequences.csv:      {seq_csv} (secuencias: {len(seq_df)})")
    print(f"- sequence_frames.csv:{frm_csv} (frames: {len(frm_df)})")


if __name__ == "__main__":
    main()
