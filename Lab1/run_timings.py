import subprocess
import re
import statistics
import sys
import math

EXECUTABLE = "sobel_orig"
RUNS = 3  # total runs per optimization level (use >=5 so trimming makes sense)

TIME_RE = re.compile(r"Total time\s*=\s*([\d.]+)\s*seconds")
PSNR_RE = re.compile(r"PSNR[^:]*:\s*([^\s]+)", re.IGNORECASE)

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

def measure(opt_flag):
    build(opt_flag)
    times = []
    psnrs = []

    print(f"Running {EXECUTABLE} ({opt_flag}) {RUNS} times...")
    for i in range(RUNS):
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
    opt_flags = {
        "-O0": "-O0",
        "-O3": "-fast",
    }
    def_flags = ["", "-DLOOP_SWAP", "-DLOOP_UNROLL", "-DLOOP_UNROLL2", "-DFUNC_INLINE", "-DCOMPILER_ASSIST"]
    tests = []

    all_defs = " ".join(def_flags)
    for opt in opt_flags.keys():
        for def_flag in def_flags:
            tests.append((opt_flags[opt] + " " + def_flag).strip())
        tests.append((opt_flags[opt] + " " + all_defs).strip())

    results = {}
    baseline_key = tests[0]

    for test in tests:
        print(f"\n=== Testing: {test} ===")
        res = measure(test)
        results[test] = res  # may be None if not enough runs

    # Summary
    base = results.get(baseline_key)
    base_time = base[0] if base else None

    print("\n=== Summary ===")
    for test, res in results.items():
        if not res:
            print(f"\033[1m{test}\033[0m:\n\t (insufficient data)\n")
            continue
        avg_t, std_t, avg_p, std_p = res
        speedup = (base_time / avg_t) if (base_time and avg_t) else float('inf')

        avg_p_str = "inf" if avg_p is math.inf else (f"{avg_p:.6f}" if avg_p is not None else "N/A")
        std_p_str = "N/A" if (avg_p is math.inf or std_p is None) else f"{std_p:.6f}"

        print(
            f"\033[1m{test}\033[0m:\n"
            f"\t Average time = \033[1;32m{avg_t:.6f}\033[0m s, "
            f"Stddev = \033[1;31m{std_t:.6f}\033[0m s, "
            f"PSNR(avg±std) = {avg_p_str} ± {std_p_str}, "
            f"speedup vs baseline: {speedup:.2f}x\n"
        )
