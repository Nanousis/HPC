#!/usr/bin/env python3
import argparse
import numpy as np
import sys
from pathlib import Path

def load_matrix(path: Path) -> np.ndarray:
    try:
        arr = np.loadtxt(path, dtype=float)
    except Exception as e:
        print(f"Error reading {path}: {e}", file=sys.stderr)
        sys.exit(2)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    # Auto-drop index column if it's [0,1,2,...] (within tiny tolerance)
    first_col = arr[:, 0]
    if np.allclose(first_col, np.arange(arr.shape[0]), rtol=0, atol=1e-9):
        arr = arr[:, 1:]
    return arr

def main():
    p = argparse.ArgumentParser(
        description="Compare two numeric tables and report difference sums."
    )
    p.add_argument("file1", type=Path)
    p.add_argument("file2", type=Path)
    p.add_argument(
        "--metric",
        choices=["abs", "squared", "raw"],
        default="abs",
        help="Difference metric to sum (default: abs = sum(|A-B|)).",
    )
    p.add_argument(
        "--per-row",
        action="store_true",
        help="Also print per-row difference sums.",
    )
    args = p.parse_args()

    A = load_matrix(args.file1)
    B = load_matrix(args.file2)

    if A.shape != B.shape:
        print(
            f"Shape mismatch: {args.file1} has {A.shape}, {args.file2} has {B.shape}",
            file=sys.stderr,
        )
        sys.exit(1)

    D = B - A
    if args.metric == "abs":
        diff_vals = np.abs(D)
        total = diff_vals.sum()
        mean = diff_vals.mean()
        max_val = diff_vals.max()
        max_pos = np.unravel_index(diff_vals.argmax(), diff_vals.shape)
        metric_name = "sum(|B - A|)"
    elif args.metric == "squared":
        diff_vals = D**2
        total = diff_vals.sum()
        mean = diff_vals.mean()
        max_val = diff_vals.max()
        max_pos = np.unravel_index(diff_vals.argmax(), diff_vals.shape)
        metric_name = "sum((B - A)^2)"
    else:  # raw
        diff_vals = D
        total = diff_vals.sum()
        mean = diff_vals.mean()
        # For raw, report max absolute with its position for usefulness
        absD = np.abs(D)
        max_val = absD.max()
        max_pos = np.unravel_index(absD.argmax(), absD.shape)
        metric_name = "sum(B - A)"

    n_elems = A.size
    print(f"Compared: {args.file1} vs {args.file2}")
    print(f"Shape: {A.shape}, elements: {n_elems}")
    print(f"Metric: {metric_name}")
    print(f"Total: {total:.9g}")
    print(f"Mean per element: {mean:.9g}")
    print(f"Max element diff (by magnitude): {max_val:.9g} at row {max_pos[0]}, col {max_pos[1]}")

    if args.per_row:
        row_sums = diff_vals.sum(axis=1)
        print("\nPer-row sums:")
        for i, v in enumerate(row_sums):
            print(f"row {i}: {v:.9g}")

if __name__ == "__main__":
    main()
