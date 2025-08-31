#!/usr/bin/env python3
"""
Compare one LIDC case across methods in hf_synthetic_data and save a single .jpg
- Auto-discovers methods under <hf_root>/lidc/**
- For each method, prepares a 5-column row: Normal, Mask, Image_1, Image_2, Image_3
- Robust LIDC filename mapping (_Mask_ ↔ _CVol_/_Vol_/same-name)

Example:
  python compare_hf_lidc_case.py \
    --hf-root /Users/skb/Documents/LeFusion/evaluation_training/synthetic_data/hf_synthetic_data \
    --case LIDC-IDRI-0008_Mask_000 \
    --out /Users/skb/Documents/LeFusion/evaluation_training/outputs/compare/LIDC-IDRI-0008_Mask_000.jpg
"""
import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

METHOD_ORDER = [
    "baseline",
    "lefusion",
    "lefusion_h",
    "lefusion_h_diffmask",
]
METHOD_DISPLAY = {
    "baseline": "Baseline",
    "lefusion": "LeFusion",
    "lefusion_h": "LeFusion-H",
    "lefusion_h_diffmask": "LeFusion-H+DiffMask",
}
VARIANTS = ["Image_1", "Image_2", "Image_3"]


def map_label_to_images(label_name: str) -> List[str]:
    cands = [
        label_name.replace("_Mask_", "_CVol_"),
        label_name.replace("_Mask_", "_Vol_"),
        label_name,
    ]
    if "GenMask_" in label_name:
        cands += [
            label_name.replace("GenMask_", "CVol_"),
            label_name.replace("GenMask_", "Vol_"),
            label_name.replace("GenMask_", ""),
        ]
    # dedup keep order
    seen = set()
    out = []
    for x in cands:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def to_binary_lesion(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 4 and arr.shape[0] >= 3:
        return (arr[2] > 0.5).astype(np.uint8)
    if arr.max() > 1.5:
        return (arr == 2).astype(np.uint8)
    return (arr > 0.5).astype(np.uint8)


def window_ct(x: np.ndarray, a_min=-175, a_max=250) -> np.ndarray:
    x = np.clip(x, a_min, a_max)
    return (x - a_min) / max(a_max - a_min, 1e-6)


def pick_slice(mask3d: np.ndarray) -> int:
    m = (mask3d > 0).astype(np.uint8)
    if m.sum() == 0:
        return mask3d.shape[-1] // 2
    areas = m.sum(axis=(0, 1))
    return int(np.argmax(areas))


def find_case_paths_for_method(method_root: Path, case: str) -> Tuple[Optional[Path], Optional[Path], List[Optional[Path]]]:
    """Return (normal_img, mask, [var1,var2,var3]) for a method."""
    # Find a labelsTr that contains the case
    label_path: Optional[Path] = None
    for p in method_root.rglob("labelsTr"):
        cand = p / f"{case}.nii.gz"
        if cand.exists():
            label_path = cand
            break
        # try nested folders
        hits = list(p.rglob(f"{case}.nii.gz"))
        if hits:
            label_path = hits[0]
            break
    if label_path is None:
        return None, None, [None, None, None]

    # Map to imagesTr
    images_dir = Path(str(label_path).replace("labelsTr", "imagesTr"))
    images_dir = images_dir if images_dir.name != "imagesTr" else images_dir
    if images_dir.name != "imagesTr":
        # If label was nested, climb to imagesTr root
        tmp = label_path
        found = None
        for _ in range(5):
            tmp = tmp.parent
            candidate = tmp / "imagesTr"
            if candidate.exists():
                found = candidate
                break
        images_dir = found or images_dir

    # Normal image (base image)
    normal: Optional[Path] = None
    if images_dir and Path(images_dir).exists():
        for nm in map_label_to_images(f"{case}.nii.gz"):
            p = Path(images_dir) / nm
            if p.exists():
                normal = p
                break
        if normal is None:
            # fallback by prefix
            prefix = case.split("_Mask_")[0] if "_Mask_" in case else case
            hits = [x for x in Path(images_dir).glob("*.nii.gz") if x.name.startswith(prefix)]
            normal = hits[0] if hits else None

    # Variants
    variant_paths: List[Optional[Path]] = []
    for v in VARIANTS:
        p = None
        if images_dir and Path(images_dir).exists():
            cand = Path(images_dir) / v / f"{case}.nii.gz"
            if cand.exists():
                p = cand
            else:
                # try nested search
                hits = list(Path(images_dir).rglob(f"{v}/{case}.nii.gz"))
                if hits:
                    p = hits[0]
        variant_paths.append(p)

    return normal, label_path, variant_paths


def discover_method_roots(lidc_root: Path) -> Dict[str, Path]:
    roots: Dict[str, Path] = {}
    for dirpath, dirnames, _ in os.walk(lidc_root):
        low = dirpath.lower()
        for m in METHOD_ORDER:
            if m in low and m not in roots:
                roots[m] = Path(dirpath)
        # early stop if all found
        if all(k in roots for k in METHOD_ORDER):
            break
    return roots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-root", required=True)
    ap.add_argument("--case", required=True, help="e.g., LIDC-IDRI-0008_Mask_000")
    ap.add_argument("--out", required=True, help="output .jpg path")
    args = ap.parse_args()

    lidc_root = Path(args.hf_root).resolve() / "lidc"
    if not lidc_root.exists():
        raise FileNotFoundError(f"LIDC root not found: {lidc_root}")

    method_roots = discover_method_roots(lidc_root)
    if not method_roots:
        raise RuntimeError("No method folders found under hf_root/lidc")

    # Collect rows
    rows: List[Tuple[str, Optional[Path], Optional[Path], List[Optional[Path]]]] = []
    for m in METHOD_ORDER:
        root = method_roots.get(m)
        if root is None:
            continue
        normal, mask, variants = find_case_paths_for_method(root, args.case)
        rows.append((m, normal, mask, variants))

    # Figure layout: one row per method, 5 columns
    nrows = len(rows)
    ncols = 5
    fig = plt.figure(figsize=(5*ncols, 3*nrows))

    for i, (m, normal, mask, variants) in enumerate(rows):
        title_row = METHOD_DISPLAY.get(m, m)
        # Load volumes
        img3d = nib.load(str(normal)).get_fdata() if normal and normal.exists() else None
        msk3d = nib.load(str(mask)).get_fdata() if mask and mask.exists() else None
        msk_bin = to_binary_lesion(msk3d) if msk3d is not None else None
        z = pick_slice(msk_bin) if msk_bin is not None else (img3d.shape[-1]//2 if img3d is not None else 0)

        # Col 1: Normal
        ax = fig.add_subplot(nrows, ncols, i*ncols + 1)
        if img3d is not None:
            ax.imshow(window_ct(img3d[..., z]), cmap="gray")
        ax.axis("off"); ax.set_title(f"{title_row}\nNormal", fontsize=12)

        # Col 2: Mask overlay
        ax = fig.add_subplot(nrows, ncols, i*ncols + 2)
        if img3d is not None and msk_bin is not None:
            base = window_ct(img3d[..., z])
            ax.imshow(base, cmap="gray")
            overlay = np.zeros((*base.shape, 4), dtype=np.float32)
            overlay[..., 0] = 1.0
            overlay[..., 3] = 0.35 * (msk_bin[..., z] > 0)
            ax.imshow(overlay)
        ax.axis("off"); ax.set_title("Mask", fontsize=12)

        # Col 3..5: Image_1..3
        for j, vp in enumerate(variants):
            ax = fig.add_subplot(nrows, ncols, i*ncols + 3 + j)
            if vp and vp.exists():
                v = nib.load(str(vp)).get_fdata()
                ax.imshow(window_ct(v[..., z]), cmap="gray")
            ax.axis("off"); ax.set_title(VARIANTS[j], fontsize=12)

    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main() 