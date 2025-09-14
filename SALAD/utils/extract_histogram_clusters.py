#!/usr/bin/env python3
"""
Extract histogram clusters from pathological training data (like LeFusion)
This creates the histogram conditioning needed for inference
"""

import os
import sys
import json
import numpy as np
import nibabel as nib
from pathlib import Path
from sklearn.cluster import KMeans
from tqdm import tqdm

def extract_histograms(data_dir: str, max_samples: int = 500):
    """Extract histograms from pathological images"""
    data_dir = Path(data_dir)
    image_dir = data_dir / "Image"
    mask_dir = data_dir / "Mask"

    histograms = []

    # Find all mask files
    mask_files = sorted(mask_dir.glob("*.nii.gz"))[:max_samples]

    print(f"Extracting histograms from {len(mask_files)} samples...")
    for mask_file in tqdm(mask_files):
        # Find corresponding image
        image_name = mask_file.name.replace("Mask_", "Vol_")
        image_file = image_dir / image_name

        if not image_file.exists():
            continue

        # Load image and mask
        img = nib.load(image_file).get_fdata()
        mask = nib.load(mask_file).get_fdata()

        # Take middle slice if 3D
        if img.ndim == 3:
            img = img[:, :, img.shape[2]//2]
            mask = mask[:, :, mask.shape[2]//2]

        # Normalize image to [-1, 1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 2 - 1
        mask = (mask > 0).astype(np.float32)

        # Extract histogram from lesion region
        lesion_pixels = img[mask > 0]
        if len(lesion_pixels) > 100:  # Only use if enough pixels
            hist, _ = np.histogram(lesion_pixels, bins=16, range=(-1, 1))
            hist = hist.astype(np.float32) / (hist.sum() + 1e-8)
            histograms.append(hist)

    return np.array(histograms)

def cluster_histograms(histograms: np.ndarray, n_clusters: int = 3):
    """Cluster histograms using K-means"""
    print(f"Clustering {len(histograms)} histograms into {n_clusters} clusters...")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(histograms)

    cluster_centers = kmeans.cluster_centers_

    # Sort clusters by average intensity (dark to bright)
    avg_intensities = []
    for center in cluster_centers:
        # Weighted average of bin positions
        bins = np.linspace(-1, 1, 16)
        avg_intensity = np.sum(center * bins)
        avg_intensities.append(avg_intensity)

    sorted_indices = np.argsort(avg_intensities)
    cluster_centers = cluster_centers[sorted_indices]

    return cluster_centers

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract histogram clusters from pathological data")
    parser.add_argument("--data_dir", type=str, default="../data/LIDC/Pathological",
                       help="Path to pathological data directory")
    parser.add_argument("--output_dir", type=str, default="inference/hist_clusters",
                       help="Output directory for cluster files")
    parser.add_argument("--dataset", type=str, default="lidc", choices=["lidc", "emidec"],
                       help="Dataset name")
    parser.add_argument("--n_clusters", type=int, default=3,
                       help="Number of histogram clusters")
    parser.add_argument("--max_samples", type=int, default=500,
                       help="Maximum samples to use")

    args = parser.parse_args()

    # Extract histograms
    histograms = extract_histograms(args.data_dir, args.max_samples)
    print(f"Extracted {len(histograms)} valid histograms")

    if len(histograms) < args.n_clusters:
        print(f"Error: Not enough histograms ({len(histograms)}) for {args.n_clusters} clusters")
        sys.exit(1)

    # Cluster histograms
    cluster_centers = cluster_histograms(histograms, args.n_clusters)

    # Save results (LeFusion format)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{args.dataset}_clusters.json"

    # Format like LeFusion
    result = [{
        "n_class": args.n_clusters,
        "centers": cluster_centers.tolist()
    }]

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved cluster centers to {output_file}")

    # Print cluster info
    print(f"\nCluster Statistics:")
    for i, center in enumerate(cluster_centers):
        avg_intensity = np.sum(center * np.linspace(-1, 1, 16))
        max_bin = np.argmax(center)
        print(f"  Cluster {i}: avg_intensity={avg_intensity:.3f}, peak_bin={max_bin}")

if __name__ == "__main__":
    main()