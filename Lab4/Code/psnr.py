#!/usr/bin/env python3
"""
pgm_psnr.py — Compute PSNR (and optionally MSE) between two .pgm images.

Usage:
  python pgm_psnr.py reference.pgm test.pgm
  python pgm_psnr.py reference.pgm test.pgm --mse
"""

import argparse
import numpy as np
import imageio.v3 as iio
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # disable decompression bomb check

def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> tuple[float, float]:
    """Compute MSE and PSNR between two images."""
    if img1.shape != img2.shape:
        raise ValueError(f"Image dimensions differ: {img1.shape} vs {img2.shape}")
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        psnr = float("inf")
    else:
        max_val = np.max([img1.max(), img2.max(), 1.0])
        psnr = 10 * np.log10((max_val ** 2) / mse)
    return mse, psnr

def main():
    parser = argparse.ArgumentParser(description="Compute PSNR between two PGM images.")
    parser.add_argument("ref", help="Reference .pgm image")
    parser.add_argument("test", help="Test .pgm image")
    parser.add_argument("--mse", action="store_true", help="Also display MSE")
    args = parser.parse_args()

    img1 = iio.imread(args.ref)
    img2 = iio.imread(args.test)

    mse, psnr = compute_psnr(img1, img2)
    if args.mse:
        print(f"PSNR: {psnr:.4f} dB | MSE: {mse:.8f}")
    else:
        print(f"PSNR: {psnr:.4f} dB")

if __name__ == "__main__":
    main()
