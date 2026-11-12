"""Time the noisy near-Clifford cases of two builds, interleaved.

The circuits and timing are those of `bench_noisy_nearclifford.py` (10,000
shots, best of 3 in one process). Each round runs every case once in each
of two Python environments, with qevedo-simulator installed from the two
builds, in alternating order, so that both see the same machine load.
Writes one row per case with the best time of each build over all rounds,
their ratio (after / before), and the median of the per-round ratios.

    python benchmarks/compare_noisy_builds.py --before /venv-old/bin/python \
        --after /venv-new/bin/python --rounds 5 \
        --out benchmarks/results/noisy_nearclifford_before_after.csv

`--qubits 24` keeps only the cases of that many qubits (to recheck a short
case with many more rounds). `--shots N` times N shots instead of 10,000
(best of 15 below 1000 shots), to see a fixed cost per run:

    python benchmarks/compare_noisy_builds.py ... --shots 1 \
        --out benchmarks/results/noisy_nearclifford_before_after_1shot.csv
"""

import argparse
import csv
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

# One process per (build, case, threads): import the benchmark's circuit
# builder and print the best of 3.
CHILD = """
import sys, time
sys.path.insert(0, {here!r})
from bench_noisy_nearclifford import noisy_clifford_t
from qevedo.simulator import run
n, depth, t_count, p, threads = {case!r}
circuit, _ = noisy_clifford_t(n, depth, t_count, p)
if p == 0.0:
    circuit.x_error(0, 0.0)
best = float("inf")
for _ in range({repeats}):
    start = time.perf_counter()
    run(circuit, {shots}, backend="near_clifford", seed=1, threads=threads)
    best = min(best, time.perf_counter() - start)
print(best)
"""

CASES = [
    # (qubits, depth, T count, p, threads)
    (24, 20, 10, 0.001, 4),
    (24, 20, 10, 0.001, 1),
    (60, 20, 12, 0.001, 4),
    (60, 20, 12, 0.001, 1),
    (100, 30, 14, 0.001, 4),
    (100, 30, 14, 0.001, 1),
    (200, 40, 12, 0.001, 4),
    (200, 40, 12, 0.001, 1),
    (60, 20, 12, 0.01, 4),
    (60, 20, 12, 0.01, 1),
    (100, 30, 14, 0.0, 1),
    (100, 30, 14, 1e-5, 1),
    (100, 30, 14, 1e-4, 1),
]


def time_case(python, case, shots):
    # Best of 3 at 10,000 shots; runs of a few milliseconds need more.
    repeats = 3 if shots >= 1000 else 15
    code = CHILD.format(here=str(HERE), case=case, shots=shots, repeats=repeats)
    out = subprocess.run([python, "-c", code], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True, help="Python with the old build")
    ap.add_argument("--after", required=True, help="Python with the new build")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--qubits", type=int, default=None, help="only cases of this many qubits")
    ap.add_argument("--shots", type=int, default=10_000)
    args = ap.parse_args()
    rows = []
    for case in CASES:
        if args.qubits is not None and case[0] != args.qubits:
            continue
        times = {"before": [], "after": []}
        for r in range(args.rounds):
            order = ("before", "after") if r % 2 == 0 else ("after", "before")
            for build in order:
                times[build].append(time_case(getattr(args, build), case, args.shots))
        best = {k: min(v) for k, v in times.items()}
        ratios = [a / b for a, b in zip(times["after"], times["before"])]
        n, depth, t_count, p, threads = case
        row = dict(qubits=n, depth=depth, t_count=t_count, p=p, shots=args.shots, threads=threads,
                   rounds=args.rounds, before_seconds=f"{best['before']:.4f}",
                   after_seconds=f"{best['after']:.4f}",
                   best_of_ratio=f"{best['after'] / best['before']:.3f}",
                   median_paired_ratio=f"{statistics.median(ratios):.3f}")
        rows.append(row)
        print(row, flush=True)
    if args.out:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
