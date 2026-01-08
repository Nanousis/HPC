#!/usr/bin/env python3
"""
Run `make run_gpu` 12 times, parse:
- Total GPU Time (seconds)
- Average GPU Throughput (Billion Interactions / second)

Discard the highest and lowest throughput runs (trimmed 10/12),
then report mean + stddev (sample) for both metrics.

Usage:
  python3 bench_gpu.py
  python3 bench_gpu.py --runs 12 --cmd "make run_gpu" --keep-logs
"""

from __future__ import annotations
import argparse
import re
import statistics as stats
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple


THROUGHPUT_RE = re.compile(
    r"Average\s+GPU\s+Throughput:\s*([0-9]*\.?[0-9]+)\s*Billion\s+Interactions\s*/\s*second",
    re.IGNORECASE,
)
TOTAL_TIME_RE = re.compile(
    r"Total\s+GPU\s+Time:\s*([0-9]*\.?[0-9]+)\s*seconds",
    re.IGNORECASE,
)


@dataclass
class RunResult:
    idx: int
    total_time_s: float
    throughput_bips: float  # billion interactions per second
    stdout: str


def run_once(cmd: str, idx: int) -> RunResult:
    proc = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out = proc.stdout

    if proc.returncode != 0:
        raise RuntimeError(f"Run {idx}: command failed (exit {proc.returncode}). Output:\n{out}")

    m_thr = THROUGHPUT_RE.search(out)
    m_t = TOTAL_TIME_RE.search(out)

    if not m_thr or not m_t:
        raise ValueError(
            f"Run {idx}: failed to parse required fields.\n"
            f"Need lines like:\n"
            f"  Total GPU Time: <num> seconds\n"
            f"  Average  GPU Throughput: <num> Billion Interactions / second\n\n"
            f"Output was:\n{out}"
        )

    thr = float(m_thr.group(1))
    t = float(m_t.group(1))
    return RunResult(idx=idx, total_time_s=t, throughput_bips=thr, stdout=out)


def trimmed_stats(values: List[float]) -> Tuple[float, float, List[int]]:
    """
    Discard highest and lowest, compute mean and sample stddev.
    Returns (mean, stddev, kept_indices_sorted_by_value_removed?)
    """
    if len(values) < 3:
        raise ValueError("Need at least 3 values to discard highest/lowest.")

    # Get indices of min/max (first occurrence)
    min_i = min(range(len(values)), key=values.__getitem__)
    max_i = max(range(len(values)), key=values.__getitem__)

    kept = [v for i, v in enumerate(values) if i not in (min_i, max_i)]

    mean = stats.mean(kept)
    stdev = stats.stdev(kept) if len(kept) >= 2 else 0.0
    return mean, stdev, [min_i, max_i]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=12, help="Number of runs (default: 12)")
    ap.add_argument("--cmd", type=str, default="make run_gpu", help='Command to run (default: "make run_gpu")')
    ap.add_argument("--keep-logs", action="store_true", help="Print full output for each run")
    args = ap.parse_args()

    runs: List[RunResult] = []
    print(f"Running: {args.cmd!r} x {args.runs}\n")

    for i in range(1, args.runs + 1):
        try:
            r = run_once(args.cmd, i)
            runs.append(r)
            print(f"[{i:02d}/{args.runs:02d}] Total={r.total_time_s:.6f}s  Throughput={r.throughput_bips:.3f} BIPS")
            if args.keep_logs:
                print(r.stdout.rstrip())
                print("-" * 80)
        except Exception as e:
            print(f"\nERROR: {e}", file=sys.stderr)
            return 2

    throughputs = [r.throughput_bips for r in runs]
    times = [r.total_time_s for r in runs]

    thr_mean, thr_std, (thr_min_i, thr_max_i) = trimmed_stats(throughputs)

    # Apply the SAME trimming (min/max by throughput) to times so they're aligned to kept runs
    kept_times = [t for k, t in enumerate(times) if k not in (thr_min_i, thr_max_i)]
    time_mean = stats.mean(kept_times)
    time_std = stats.stdev(kept_times) if len(kept_times) >= 2 else 0.0

    print("\n=== Trimmed results (discard lowest & highest throughput run) ===")
    print(f"Discarded (lowest throughput): run #{runs[thr_min_i].idx:02d} -> {runs[thr_min_i].throughput_bips:.3f} BIPS, Total={runs[thr_min_i].total_time_s:.6f}s")
    print(f"Discarded (highest throughput): run #{runs[thr_max_i].idx:02d} -> {runs[thr_max_i].throughput_bips:.3f} BIPS, Total={runs[thr_max_i].total_time_s:.6f}s")
    print(f"\nKept runs: {args.runs - 2} / {args.runs}")

    print("\nThroughput (Billion Interactions / second):")
    print(f"  mean = {thr_mean:.3f} BIPS")
    print(f"  std  = {thr_std:.3f} BIPS  (sample)")

    print("\nTotal GPU Time (seconds) on kept runs:")
    print(f"  mean = {time_mean:.6f} s")
    print(f"  std  = {time_std:.6f} s  (sample)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
