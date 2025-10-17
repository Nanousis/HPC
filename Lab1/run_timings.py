# Script to build and ran the sobel_orig.c with all the possible combinations of optimizations and measure timings.
import subprocess
import re
import statistics
import sys
import math
import argparse
import itertools
import matplotlib.pyplot as plt
import numpy as np

EXECUTABLE = "sobel_orig"
RUNS = 3  # total runs per optimization level (use >=5 so trimming makes sense)

TIME_RE = re.compile(r"Total time\s*=\s*([\d.]+)\s*seconds")
PSNR_RE = re.compile(r"PSNR[^:]*:\s*([^\s]+)", re.IGNORECASE)

# Build the program with given CFLAGS
def build(opt_flag):
    print(f"\n=== Building with {opt_flag} ===")
    subprocess.run(["make", "clean"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cflags = f"-Wall {opt_flag}"
    print(f"make CFLAGS={cflags}")
    result = subprocess.run(["make", f"CFLAGS={cflags}"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Build failed:")
        print(result.stderr)
        sys.exit(1)

# Parse time and PSNR from program output
def parse_time_and_psnr(output: str):
    # Time
    m_time = TIME_RE.search(output)
    t = float(m_time.group(1)) if m_time else None

    # PSNR (could be 'inf')
    m_psnr = PSNR_RE.search(output)
    psnr_val = None
    if m_psnr:
        s = m_psnr.group(1)
        if s.lower() in ("inf", "infinity"):
            psnr_val = math.inf
        else:
            try:
                psnr_val = float(s)
            except ValueError:
                psnr_val = None
    return t, psnr_val

# Runs the sobel_orig executable and captures its output
def run_program():
    run = subprocess.run([f"./{EXECUTABLE}"], capture_output=True, text=True)
    # print(run.stdout)
    print(run.stderr, file=sys.stderr)
    return parse_time_and_psnr(run.stdout.strip())

def trimmed_stats(values):
    """Return (mean, stdev, trimmed_values) after removing one min and one max.
       For PSNR: if any value is math.inf in the trimmed set, return (math.inf, None, trimmed)."""
    if len(values) < 3:
        return (None, None, values)
    vals_sorted = sorted(values, key=lambda x: (math.isfinite(x), x))
    trimmed = vals_sorted[1:-1]
    if not trimmed:
        return (None, None, trimmed)

    # If any inf in trimmed -> mean=inf, stddev=None
    if any(v is math.inf for v in trimmed):
        return (math.inf, None, trimmed)

    # Only compute on finite numbers
    mean = statistics.mean(trimmed)
    std = statistics.stdev(trimmed) if len(trimmed) > 1 else 0.0
    return (mean, std, trimmed)

def measure(opt_flag,runs):
    build(opt_flag)
    times = []
    psnrs = []

    print(f"Running {EXECUTABLE} ({opt_flag}) {runs} times...")
    for i in range(runs):
        t, p = run_program()
        if t is not None:
            times.append(t)
        if p is not None:
            psnrs.append(p)
        # Per-run print (show PSNR as 'inf' if needed)
        p_str = "inf" if p is math.inf else (f"{p:.4f}" if p is not None else "N/A")
        t_str = f"{t:.6f} s" if t is not None else "N/A"
        print(f"Run {i+1:02d}: time={t_str}, PSNR={p_str}")

    if len(times) < 3:
        print("Not enough valid runs.")
        return None

    t_mean, t_std, t_trim = trimmed_stats(times)
    p_mean, p_std, p_trim = trimmed_stats(psnrs) if len(psnrs) >= 3 else (None, None, psnrs)

    print(f"\nResults for {opt_flag}:")
    print(f"All times: {[f'{v:.4f}' for v in times]}")
    print(f"All PSNRs: {[('inf' if v is math.inf else f'{v:.4f}') for v in psnrs]}")

    print(f"Trimmed time avg: {t_mean:.6f} s")
    print(f"Time stddev:      {t_std:.6f} s")

    if p_mean is math.inf:
        print("Trimmed PSNR avg: inf")
        print("PSNR stddev:      N/A (contains inf)")
    elif p_mean is None:
        print("Trimmed PSNR avg: N/A")
        print("PSNR stddev:      N/A")
    else:
        print(f"Trimmed PSNR avg: {p_mean:.6f}")
        print(f"PSNR stddev:      {p_std:.6f}")

    return (t_mean, t_std, p_mean, p_std)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sobel timings with various flags.")
    parser.add_argument("--runs", type=int, default=RUNS, help="Number of runs per test (default: 3)")
    parser.add_argument(
        "--combos",
        action="store_true",
        help="Exhaustively test combinations of -D flags in addition to singles and ALLDEFS"
    )
    parser.add_argument(
        "--min-combo-size", type=int, default=2,
        help="Minimum number of -D flags in a combo when --combos is set (default: 2)"
    )
    parser.add_argument(
        "--max-combo-size", type=int, default=None,
        help="Maximum number of -D flags in a combo when --combos is set (default: all)"
    )
    args = parser.parse_args()
    runs = args.runs if args.runs >= 3 else 3

    opt_variants = [
        ("O0", "-O0"),
        ("FAST", "-fast"),
        ("O3_ZNVER4", "-O3 -march=znver4 -mtune=znver4"),
    ]

    def_flags = ["", "-DLOOP_SWAP", "-DLOOP_UNROLL", "-DLOOP_UNROLL2", "-DCOMPILER_ASSIST", "-DSTRENGTH_REDUCTION"]

    tests = []
    for label, flags in opt_variants:
        for d in def_flags:
            tests.append((label, f"{(flags + ' ' + d).strip()}"))

    if args.combos:
        only_defs = [d for d in def_flags if d]  # skip the empty baseline token
        max_sz = args.max_combo_size or len(only_defs)
        min_sz = max(1, args.min_combo_size)
        for label, flags in opt_variants:
            for r in range(min_sz, max_sz + 1):
                for combo in itertools.combinations(only_defs, r):
                    combo_flags = (flags + " " + " ".join(combo)).strip()
                    short = "+".join(x.replace("-D", "") for x in combo)
                    tests.append((f"{label}_C{r}[{short}]", combo_flags))

    # ALLDEFS (keep an explicit all-on test too)
    all_defs = " ".join([d for d in def_flags if d])
    for label, flags in opt_variants:
        tests.append((f"{label}_ALLDEFS", f"{(flags + ' ' + all_defs).strip()}"))


    results = {}
    # set baselines: O0 with no -D as primary baseline; O3_ZNVER4 with no -D as O3 baseline
    baseline_key = next(k for k, v in tests if k == "O0" and " -D" not in (" " + v))
    #o3_baseline_key = next(k for k, v in tests if k == "O3_ZNVER4" and " -D" not in (" " + v))
    o3_baseline_key = next(k for k, v in tests if k=="FAST" and "-D" not in ("" + v))

    for key, buildflags in tests:
        print(f"\n=== Testing: {buildflags} ===")
        res = measure(buildflags,   runs)
        results[key + " :: " + buildflags] = res  # keep both the label and the exact flags

    base_time = results.get(baseline_key + " :: " + next(v for k, v in tests if k == baseline_key))
    o3_time  = results.get(o3_baseline_key + " :: " + next(v for k, v in tests if k == o3_baseline_key))
    base_time = base_time[0] if base_time else None
    o3_time   = o3_time[0] if o3_time else None

    print("\n=== Summary ===")
    for test, res in results.items():
        if not res:
            print(f"\033[1m{test}\033[0m:\n\t (insufficient data)\n")
            continue
        avg_t, std_t, avg_p, std_p = res

        speedup_vs_o0 = (base_time / avg_t) if (base_time and avg_t) else float('inf')
        speedup_vs_o3 = (o3_time / avg_t) if (o3_time and avg_t) else None  # None => N/A

        avg_p_str = "inf" if avg_p is math.inf else (f"{avg_p:.6f}" if avg_p is not None else "N/A")
        std_p_str = "N/A" if (avg_p is math.inf or std_p is None) else f"{std_p:.6f}"
        speedup_o3_str = f"{speedup_vs_o3:.2f}x" if speedup_vs_o3 is not None else "N/A"

        print(
            f"\033[1m{test}\033[0m:\n"
            f"\t Average time = \033[1;32m{avg_t:.6f}\033[0m s, "
            f"Stddev = \033[1;31m{std_t:.6f}\033[0m s, "
            f"PSNR(avg±std) = {avg_p_str} ± {std_p_str}, "
            f"speedup vs O0: {speedup_vs_o0:.2f}x, "
            f"speedup vs O3: {speedup_o3_str}\n"
        )
    if args.combos and base_time:
        ranked = []
        for test, res in results.items():
            if not res: 
                continue
            avg_t, _, _, _ = res
            if avg_t:
                ranked.append((base_time / avg_t, test, avg_t))
        ranked.sort(reverse=True, key=lambda x: x[0])
        print("\n=== Top combinations by speedup vs O0 (descending) ===")
        for spd, name, t in ranked[:10]:
            print(f"{spd:6.2f}x  {name}   (avg {t:.6f} s)")
