#!/usr/bin/env python3
"""
Visualize LIDC synthetic data downloaded into hf_synthetic_data
- Scans the HF snapshot root for LIDC subsets
- Pairs imagesTr and labelsTr (robust filename mapping for LIDC)
- Saves 2D overlays per case (same axial slice chosen by largest mask area)

Usage example:
  python visualize_hf_lidc.py \
    --hf-root /Users/skb/Documents/LeFusion/evaluation_training/synthetic_data/hf_synthetic_data \
    --out /Users/skb/Documents/LeFusion/evaluation_training/outputs/hf_lidc_vis \
    --per-method 24
"""

import os
import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm

# Heuristic method keys (search in path, case-insensitive)
METHOD_KEYS = [
    "lefusion_h_diffmask",
    "lefusion_h",
    "lefusion",
    "baseline",
]

METHOD_DISPLAY = {
    "baseline": "Baseline (P)",
    "lefusion": "LeFusion (P+P')",
    "lefusion_h": "LeFusion-H (P+P'/N')",
    "lefusion_h_diffmask": "LeFusion-H+DiffMask (P+N'/N'')",
}


def find_method_from_path(path_like: str) -> str:
    pl = path_like.lower()
    for key in METHOD_KEYS:
        if key in pl:
            return key
    # try to guess by common tokens
    tokens = Path(pl).parts
    for t in tokens:
        if t in METHOD_KEYS:
            return t
    return "unknown"


def map_label_to_image_candidates(label_name: str) -> List[str]:
    """Return candidate image filenames for a given LIDC label filename.
    Supports common patterns: _Mask_ ↔ _CVol_/_Vol_/same-name and GenMask_.
    """
    cands: List[str] = [
        label_name.replace("_Mask_", "_CVol_"),
        label_name.replace("_Mask_", "_Vol_"),
        label_name,  # sometimes same basename
    ]
    if "GenMask_" in label_name:
        cands += [
            label_name.replace("GenMask_", "CVol_"),
            label_name.replace("GenMask_", "Vol_"),
            label_name.replace("GenMask_", ""),
        ]
    # deduplicate while keeping order
    return list(dict.fromkeys(cands))


def robust_pair_image_for_label(images_dir: Path, label_path: Path) -> Optional[Path]:
    name = label_path.name
    for cand in map_label_to_image_candidates(name):
        p = images_dir / cand
        if p.exists():
            return p
    # fallback: prefix search before _Mask_
    if "_Mask_" in name:
        prefix = name.split("_Mask_")[0]
        hits = [x for x in images_dir.glob("*.nii.gz") if x.name.startswith(prefix)]
        if hits:
            return hits[0]
    return None


def to_lesion_mask(arr: np.ndarray) -> np.ndarray:
    """Convert label to binary lesion mask (supports 0/1 or 0/1/2 where 2 is lesion)."""
    if arr.ndim == 4 and arr.shape[0] >= 3:
        # one-hot (C,H,W,D)
        return (arr[2] > 0.5).astype(np.uint8)
    if arr.max() > 1.5:
        return (arr == 2).astype(np.uint8)
    return (arr > 0.5).astype(np.uint8)


def window_ct(x: np.ndarray, a_min: float = -175, a_max: float = 250) -> np.ndarray:
    x = np.clip(x, a_min, a_max)
    return (x - a_min) / max(a_max - a_min, 1e-6)


def pick_slice_index(mask3d: np.ndarray) -> int:
    m = (mask3d > 0).astype(np.uint8)
    if m.sum() == 0:
        return mask3d.shape[-1] // 2
    areas = m.sum(axis=(0, 1))
    return int(np.argmax(areas))


def overlay_and_save(img3d: np.ndarray, mask3d: np.ndarray, out_path: Path, title: str) -> None:
    z = pick_slice_index(mask3d)
    img = window_ct(img3d[..., z])
    msk = (mask3d[..., z] > 0).astype(np.uint8)

    plt.figure(figsize=(5.2, 5.2))
    plt.imshow(img, cmap="gray")
    # red overlay
    overlay = np.zeros((*msk.shape, 4), dtype=np.float32)
    overlay[..., 0] = 1.0
    overlay[..., 3] = 0.35 * msk
    plt.imshow(overlay)
    plt.axis("off")
    plt.title(title, fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def collect_pairs_by_method(lidc_root: Path) -> Dict[str, List[Tuple[Path, Path]]]:
    """Walk under lidc_root and collect (image,label) pairs grouped by method.
    Unknown layouts are auto-grouped instead of skipped.
    """
    grouped: Dict[str, List[Tuple[Path, Path]]] = {}

    for dirpath, _, filenames in os.walk(lidc_root):
        # We expect leaf like .../labelsTr
        if "labelsTr" not in dirpath:
            continue
        labels_dir = Path(dirpath)
        images_dir = Path(dirpath.replace("labelsTr", "imagesTr"))
        if not images_dir.exists():
            continue

        method = find_method_from_path(dirpath)
        if method == "unknown":
            # try to use a nearby directory name as pseudo-method (e.g., the folder right above labelsTr)
            try:
                method = Path(dirpath).parent.name.lower() or "unknown"
            except Exception:
                method = "unknown"
        if method not in grouped:
            grouped[method] = []

        for fn in filenames:
            if not fn.endswith(".nii.gz"):
                continue
            lbl_path = labels_dir / fn
            img_path = robust_pair_image_for_label(images_dir, lbl_path)
            if img_path is None or not img_path.exists():
                continue
            grouped[method].append((img_path, lbl_path))

    return grouped


def load_nifti(path: Path) -> np.ndarray:
    return nib.load(str(path)).get_fdata()


def main():
    parser = argparse.ArgumentParser(description="Visualize LIDC synthetic data from hf_synthetic_data")
    parser.add_argument(
        "--hf-root",
        default="/Users/skb/Documents/LeFusion/evaluation_training/synthetic_data/hf_synthetic_data",
        help="Path to HF snapshot root (contains lidc/ and emidec/)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for generated PNGs",
    )
    parser.add_argument(
        "--per-method",
        type=int,
        default=24,
        help="Number of random samples to export per method",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    hf_root = Path(args.hf_root).resolve()
    lidc_root = hf_root / "lidc"
    if not lidc_root.exists():
        raise FileNotFoundError(f"LIDC root not found: {lidc_root}")

    print(f"Scanning: {lidc_root}")
    grouped = collect_pairs_by_method(lidc_root)
    all_keys = sorted(grouped.keys())
    for k in all_keys:
        print(f"- {k:<20}: {len(grouped.get(k, []))} pairs")

    out_root = Path(args.out).resolve()
    total_exported = 0

    for method in sorted(grouped.keys()):
        pairs = grouped.get(method, [])
        if not pairs:
            continue
        # sample
        if len(pairs) > args.per_method:
            pairs = random.sample(pairs, args.per_method)

        display_name = METHOD_DISPLAY.get(method, method)
        print(f"Exporting {len(pairs)} overlays for {display_name} ...")
        for img_path, lbl_path in tqdm(pairs, desc=method, ncols=80):
            try:
                img = load_nifti(img_path)
                lbl = load_nifti(lbl_path)
                lesion = to_lesion_mask(lbl)
                case_name = lbl_path.stem
                title = f"{display_name}\n{case_name}"
                out_png = out_root / method / f"{case_name}.png"
                overlay_and_save(img, lesion, out_png, title)
                total_exported += 1
            except Exception:
                # best-effort: skip unreadable cases
                continue

    print(f"Done. Saved {total_exported} PNGs under: {out_root}")


if __name__ == "__main__":
    main() 