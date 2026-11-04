"""Time the Rust benchmark examples of two builds, interleaved.

Each case runs alternately from two directories of example binaries
(`cargo build --release --examples`, then copy `target/release/examples/*`
before and after a change), so both builds see the same machine load.
The time is the one the example prints (its own timer, not process
start-up). Writes one row per case and build, plus the best-of ratio and
the median of the per-round ratios (new / old).

    python benchmarks/compare_examples.py --before /tmp/old --after /tmp/new \
        --stim-dir /tmp/stim --rounds 9 --out benchmarks/results/examples_before_after.csv

`--stim-dir` holds `d21.stim` and `d51.stim`: Stim's
`surface_code:rotated_memory_z` at that distance and as many rounds, all
four noise parameters at 0.001 (written by this script if missing and
Stim is installed).
"""

import argparse
import csv
import os
import re
import statistics
import subprocess


def cases(stim_dir):
    return [
        ("statevector random 26 f32", ["bench", "random", "26", "20", "f32", "4", "14"], r"total (\S+) s"),
        ("statevector qft 26 f32", ["bench", "qft", "26", "0", "f32", "4", "14"], r"total (\S+) s"),
        ("statevector random 25 f64", ["bench", "random", "25", "20", "f64", "4", "14"], r"total (\S+) s"),
        ("shots 24 qubits 1M shots", ["shots", "24", "1000000"], r"total (\S+) s"),
        ("mps random 100 depth 16 chi 128", ["mps", "random", "100", "16", "128", "1000"], r"total (\S+) s"),
        ("mps qft 100", ["mps", "qft", "100", "0", "256", "1000"], r"total (\S+) s"),
        ("nearclifford 100 qubits 24 T", ["nearclifford", "100", "30", "24", "1000"], r"total (\S+) s"),
        ("pauliprop 20 steps 0.2", ["pauliprop", "20", "0.2", "Z62", "1e-5"], r"(\S+) s on"),
        ("pauliprop 5 steps 1.2", ["pauliprop", "5", "1.2", "Z62", "1e-6"], r"(\S+) s on"),
        ("stabilizer d21", ["stabilizer", "surface", "21", "21", "10000"], r": (\S+) s \("),
        ("stabilizer d51", ["stabilizer", "surface", "51", "51", "10000"], r": (\S+) s \("),
        ("stabilizer d21 noise 0.001", ["stabilizer", "surface", "21", "21", "10000", "--noise", "0.001"], r": (\S+) s \("),
        ("stabilizer d51 noise 0.001", ["stabilizer", "surface", "51", "51", "10000", "--noise", "0.001"], r": (\S+) s \("),
        ("stim detectors d21 (sample)", ["stabilizer", "detectors", os.path.join(stim_dir, "d21.stim"), "10000"], r"sample (\S+) s"),
        ("stim detectors d51 (sample)", ["stabilizer", "detectors", os.path.join(stim_dir, "d51.stim"), "10000"], r"sample (\S+) s"),
    ]


def write_stim_files(stim_dir):
    os.makedirs(stim_dir, exist_ok=True)
    for d in (21, 51):
        path = os.path.join(stim_dir, f"d{d}.stim")
        if not os.path.exists(path):
            import stim

            c = stim.Circuit.generated(
                "surface_code:rotated_memory_z", distance=d, rounds=d,
                after_clifford_depolarization=0.001, before_round_data_depolarization=0.001,
                before_measure_flip_probability=0.001, after_reset_flip_probability=0.001)
            with open(path, "w") as f:
                f.write(str(c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--stim-dir", required=True)
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--only", default=None, help="run only cases containing one of these comma-separated texts")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    write_stim_files(args.stim_dir)
    rows = []
    for name, cmd, pattern in cases(args.stim_dir):
        if args.only and not any(o in name for o in args.only.split(",")):
            continue
        times = {"before": [], "after": []}
        for r in range(args.rounds):
            order = ("before", "after") if r % 2 == 0 else ("after", "before")
            for build in order:
                exe = os.path.join(getattr(args, build), cmd[0])
                out = subprocess.run([exe] + cmd[1:], capture_output=True, text=True, check=True).stdout
                times[build].append(float(re.search(pattern, out).group(1)))
        best = {k: min(v) for k, v in times.items()}
        ratio = best["after"] / best["before"]
        median = statistics.median(a / b for a, b in zip(times["after"], times["before"]))
        print(f"{name:34s} before {best['before']:.4f} s  after {best['after']:.4f} s  "
              f"best-of {ratio:.3f}  median paired {median:.3f}", flush=True)
        rows.append([name, args.rounds, best["before"], best["after"], f"{ratio:.3f}", f"{median:.3f}"])
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["case", "rounds", "before_seconds", "after_seconds", "best_of_ratio", "median_paired_ratio"])
            w.writerows(rows)


if __name__ == "__main__":
    main()
