#!/usr/bin/env python3
import subprocess, re, statistics, sys, argparse, os, csv
from datetime import datetime
from shutil import which

# Regex to capture "Computation timing = ... sec"
TIMING_RE = re.compile(r"Computation timing\s*=\s*([0-9]*\.?[0-9]+)\s*sec", re.IGNORECASE)

def run_once(cmd, env=None):
    """Run one instance and extract computation timing."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    m = TIMING_RE.search(p.stdout)
    if not m:
        print(f"[WARN] No timing found in output:\n{p.stdout}", file=sys.stderr)
        return None
    return float(m.group(1))

def run_many(label, exe, args, runs, env=None):
    """Run binary multiple times and collect timings."""
    times = []
    for i in range(runs):
        t = run_once([exe] + args, env)
        if t is not None:
            times.append(t)
            print(f"[{label}] Run {i+1}/{runs}: {t:.6f} sec")
    if not times:
        print(f"[{label}] All runs failed.")
        return None, None
    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    print(f"[{label}] Mean = {mean:.6f} sec, Std = {std:.6f}\n")
    return mean, std

def main():
    parser = argparse.ArgumentParser(description="Benchmark seq_main and par_main across OMP threads.")
    parser.add_argument("--seq", default="./seq_main", help="Sequential binary path")
    parser.add_argument("--par", default="./par_main", help="Parallel binary path")
    parser.add_argument("--runs", type=int, default=12, help="Number of runs per configuration (default 12)")
    parser.add_argument("--threads", nargs="*", type=int, default=[1, 4, 8, 12, 14, 28, 56],
                        help="Thread counts to test (default: 1 4 8 12 14 28 56)")
    parser.add_argument("--csv", default="timings_summary.csv", help="CSV file to store summarized results")
    parser.add_argument("program_args", nargs=argparse.REMAINDER,
                        help="Arguments passed to both programs (use '--' before them)")
    args = parser.parse_args()

    if args.program_args and args.program_args[0] == "--":
        args.program_args = args.program_args[1:]

    seq_exe = args.seq if which(args.seq) or os.path.exists(args.seq) else None
    par_exe = args.par if which(args.par) or os.path.exists(args.par) else None

    if not seq_exe or not par_exe:
        print("Missing seq_main or par_main executable.", file=sys.stderr)
        sys.exit(1)

    results = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Sequential baseline
    print("=== Sequential baseline ===")
    seq_mean, seq_std = run_many("SEQ", seq_exe, args.program_args, args.runs)
    if seq_mean is not None:
        results.append(["seq", 0, seq_mean, seq_std, timestamp])

    # Parallel runs
    for n_threads in args.threads:
        print(f"=== Parallel: OMP_NUM_THREADS = {n_threads} ===")
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(n_threads)
        par_mean, par_std = run_many(f"PAR-{n_threads}", par_exe, args.program_args, args.runs, env=env)
        if par_mean is not None:
            results.append(["par", n_threads, par_mean, par_std, timestamp])

    # Write summarized results to CSV
    with open(args.csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["program", "threads", "mean_time_sec", "std_dev_sec", "timestamp"])
        writer.writerows(results)
    print(f"\nSummarized results saved to {args.csv}")

    # Print summary table
    print("\n=== Summary ===")
    for prog, threads, mean, std, _ in results:
        print(f"{prog.upper():<5} threads={threads:<2} → mean={mean:.6f} s, std={std:.6f}")

if __name__ == "__main__":
    main()
