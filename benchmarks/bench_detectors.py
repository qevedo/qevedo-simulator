"""Detection events of Stim's own generated circuits: qvd's Stim reader and
detector sampling against Stim's detector sampler, for speed, for the
sampled statistics and for decoded logical error rates.

    cd python && maturin develop --release && cd ..
    pip install stim pymatching
    python benchmarks/bench_detectors.py [--distances 11,21,51,101] [--shots 10000]

Every circuit is `stim.Circuit.generated("surface_code:rotated_memory_z",
distance=d, rounds=d, ...)` with all four of Stim's noise parameters set to
p (depolarization after Clifford gates and before each round on the data,
flips after resets and before measurements). Both simulators get the same
text.

Timing, in one process, both returning shot-major NumPy bool arrays
(detectors, observables): qvd's `sample_detectors` on 4 threads and on 1
(this includes reading the text and unrolling its REPEAT block), and
Stim's `compile_detector_sampler().sample(shots, separate_observables=True)`
(single-threaded; compiling includes its reference sample), and the same
with `bit_packed=True` (eight detectors per byte, Stim's fastest output; qvd
has no bit-packed shot-major output yet). Best of 3, one run at d = 101.

Agreement: the rate at which each detector fires, 200,000 shots, seed 1 in
both, with a two-sample z-score per detector; and the logical error rate
after decoding both simulators' samples with the same PyMatching decoder
(built from Stim's detector error model).

With --long, only a long run of the smallest code (d = 3, p = 0.001,
24 million shots over six seeds), comparing every detector's and the
observable's rate: results/detectors_long.csv.

Writes results/detectors.csv and results/detectors_agreement.csv.
"""

import argparse
import csv
import math
import re
import time
from pathlib import Path

import numpy as np
import pymatching
import stim

from qevedo.simulator import sample_detectors

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks" / "results"
VERSION = re.search(r'^version = "(.*)"', (ROOT / "python" / "Cargo.toml").read_text(), re.M).group(1)


def generated(d, p, rounds=None):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d,
        rounds=rounds or d,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
    )


def best_of(repeats, run):
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        best = min(best, time.perf_counter() - start)
    return best


def timing(distances, shots, p):
    rows = []
    for d in distances:
        circuit = generated(d, p)
        text = str(circuit)
        repeats = 1 if d >= 101 else 3
        print(f"d = {d}: {circuit.num_qubits} qubits, {circuit.num_detectors} detectors", flush=True)
        common = dict(
            distance=d, rounds=d, qubits=circuit.num_qubits, detectors=circuit.num_detectors,
            measurements=circuit.num_measurements, p=p, shots=shots,
        )
        for threads in (4, 1):
            seconds = best_of(repeats, lambda: sample_detectors(text, shots, seed=1, threads=threads))
            rows.append(dict(common, simulator="qvd", method="sample_detectors", version=VERSION,
                             threads=threads, seconds=f"{seconds:.3f}"))
            print(f"  qvd, {threads} threads: {seconds:.3f} s", flush=True)
        seconds = best_of(repeats, lambda: stim.Circuit(text).compile_detector_sampler(seed=1)
                          .sample(shots, separate_observables=True))
        rows.append(dict(common, simulator="stim", method="compile_detector_sampler().sample",
                         version=stim.__version__, threads=1, seconds=f"{seconds:.3f}"))
        print(f"  stim: {seconds:.3f} s", flush=True)
        # Stim's fastest output: shot-major, eight detectors per byte.
        seconds = best_of(repeats, lambda: stim.Circuit(text).compile_detector_sampler(seed=1)
                          .sample(shots, separate_observables=True, bit_packed=True))
        rows.append(dict(common, simulator="stim", method="compile_detector_sampler().sample(bit_packed=True)",
                         version=stim.__version__, threads=1, seconds=f"{seconds:.3f}"))
        print(f"  stim, bit-packed: {seconds:.3f} s", flush=True)
    return rows


def agreement(cases, shots):
    rows = []
    for d, p in cases:
        circuit = generated(d, p)
        ours_d, ours_o = sample_detectors(circuit, shots, seed=1)
        theirs_d, theirs_o = circuit.compile_detector_sampler(seed=1).sample(shots, separate_observables=True)
        a, b = ours_d.mean(axis=0), theirs_d.mean(axis=0)
        z = np.abs(a - b) / np.sqrt(np.maximum(a * (1 - a) + b * (1 - b), 1e-12) / shots)
        matching = pymatching.Matching.from_detector_error_model(circuit.detector_error_model(decompose_errors=True))
        logical = []
        for dets, obs in ((ours_d, ours_o), (theirs_d, theirs_o)):
            predicted = matching.decode_batch(dets)
            logical.append(np.any(predicted != obs, axis=1).mean())
        sigma = math.sqrt((logical[0] * (1 - logical[0]) + logical[1] * (1 - logical[1])) / shots)
        row = dict(
            distance=d, rounds=d, p=p, shots=shots, detectors=circuit.num_detectors,
            qvd_detection_rate=f"{a.mean():.6f}", stim_detection_rate=f"{b.mean():.6f}",
            max_abs_z=f"{z.max():.2f}", qvd_logical_error_rate=f"{logical[0]:.6f}",
            stim_logical_error_rate=f"{logical[1]:.6f}",
            logical_z=f"{abs(logical[0] - logical[1]) / max(sigma, 1e-12):.2f}",
        )
        print(row, flush=True)
        rows.append(row)
    return rows


def long_run(d, p, shots, seeds):
    """Many more shots of one small code, summed over seeds: per-detector
    and observable rates, to see whether a difference in the agreement
    table is chance or a bias."""
    circuit = generated(d, p)
    ours = np.zeros(circuit.num_detectors + circuit.num_observables)
    theirs = np.zeros_like(ours)
    for seed in seeds:
        dets, obs = sample_detectors(circuit, shots, seed=seed)
        ours += np.concatenate([dets.sum(axis=0), obs.sum(axis=0)])
        dets, obs = circuit.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
        theirs += np.concatenate([dets.sum(axis=0), obs.sum(axis=0)])
    total = shots * len(seeds)
    a, b = ours / total, theirs / total
    z = (a - b) / np.sqrt((a * (1 - a) + b * (1 - b)) / total)
    rows = [dict(distance=d, rounds=d, p=p, shots=total,
                 target=f"detector {i}" if i < circuit.num_detectors else "observable 0",
                 qvd_rate=f"{a[i]:.7f}", stim_rate=f"{b[i]:.7f}", z=f"{z[i]:.2f}")
            for i in range(len(z))]
    print(f"d = {d}, p = {p}, {total} shots: largest |z| {np.abs(z).max():.2f}, mean z^2 {np.mean(z**2):.2f}")
    return rows


def write(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distances", default="11,21,51,101")
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--p", type=float, default=0.001)
    parser.add_argument("--agreement-shots", type=int, default=200_000)
    parser.add_argument("--long", action="store_true",
                        help="only the long run: d = 3, p = 0.001, 6 seeds x 4,000,000 shots")
    args = parser.parse_args()
    if args.long:
        write(RESULTS / "detectors_long.csv", long_run(3, 0.001, 4_000_000, range(20, 26)))
        return
    distances = [int(d) for d in args.distances.split(",")]
    write(RESULTS / "detectors.csv", timing(distances, args.shots, args.p))
    cases = [(3, 0.001), (5, 0.001), (5, 0.005), (7, 0.003), (9, 0.005)]
    write(RESULTS / "detectors_agreement.csv", agreement(cases, args.agreement_shots))


if __name__ == "__main__":
    main()
