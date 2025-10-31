#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_kmeans_results.py timings_summary.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print(f"Error: file '{csv_path}' not found.")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # Extract sequential baseline
    seq_row = df[df["program"] == "seq"]
    if seq_row.empty:
        print("No sequential (seq) data found in CSV.")
        sys.exit(1)
    seq_time = float(seq_row["mean_time_sec"].values[0])
    print(f"Sequential baseline: {seq_time:.6f} sec")

    # Parallel data
    par_df = df[df["program"] == "par"].copy()
    par_df.sort_values("threads", inplace=True)
    par_df["speedup"] = seq_time / par_df["mean_time_sec"]
    par_df["efficiency"] = par_df["speedup"] / par_df["threads"]

    # Plot computation time
    plt.figure(figsize=(7,5))
    plt.errorbar(par_df["threads"], par_df["mean_time_sec"], yerr=par_df["std_dev_sec"],
                 fmt='-o', capsize=4, label='Parallel (mean ± std)')
    plt.axhline(seq_time, color='r', linestyle='--', label='Sequential')
    plt.title("K-Means Computation Time vs Threads")
    plt.xlabel("OMP Threads")
    plt.ylabel("Computation Time (sec)")
    # plt.xscale("log")
    plt.xticks(par_df["threads"])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("computation_time_vs_threads.png", dpi=150)
    print("Saved: computation_time_vs_threads.png")

    # Plot speedup
    plt.figure(figsize=(7,5))
    plt.plot(par_df["threads"], par_df["speedup"], '-o', label='Speedup')
    plt.plot(par_df["threads"], par_df["threads"], '--', color='gray', label='Ideal Speedup (y=x)')
    plt.title("Speedup vs Threads")
    plt.xlabel("OMP Threads")
    plt.ylabel("Speedup (×)")
    # plt.xscale("log")
    plt.xticks(par_df["threads"])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("speedup_vs_threads.png", dpi=150)
    print("Saved: speedup_vs_threads.png")

    plt.show()

if __name__ == "__main__":
    main()
