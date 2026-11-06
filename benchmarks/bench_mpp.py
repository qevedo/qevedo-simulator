"""Pauli product measurements (`MPP`), `PAULI_CHANNEL_2` and correlated
errors (`E`, `ELSE_CORRELATED_ERROR`): a toric-code memory in Stim's
format, sampled by Qevedo and by Stim.

    cd python && maturin develop --release && cd ..
    pip install stim pymatching
    python benchmarks/bench_mpp.py [--sizes 10,20,30] [--shots 10000]

The circuit (`toric_mpp`): the L x L toric code, 2 L^2 data qubits and no
ancillas, every stabilizer measured directly with `MPP`, L rounds. Before
each round every data qubit gets `DEPOLARIZE1(p)`, every horizontal /
vertical edge pair at one vertex a biased `PAULI_CHANNEL_2` (ZZ at p / 3,
XX at p / 6, each other Pauli at p / 30), and L random pairs of qubits a
correlated chain `E(p) X X; ELSE_CORRELATED_ERROR(p) Z Z`, as a crosstalk
model would add. Measurements flip with probability p (`MPP(p)`, `M(p)`).
Detectors compare each stabilizer with the round before (Z plaquettes
also with the initial |0...0> and the final data measurement); the
observable is a Z logical.

Timing (results/mpp.csv): 10,000 shots of detection events, shot-major
bool NumPy arrays, best of 3 in one process; Qevedo on 4 threads and on 1
(its time includes reading the text), Stim's `compile_detector_sampler()`
on its one thread (including compiling).

Agreement (results/mpp_agreement.csv): 200,000 shots each, per-detector
rates with a two-sample z-score, and logical error rates after the same
PyMatching decoder, built from Stim's detector error model. `--long`
repeats such cases with 4 to 12 million shots per simulator
(results/mpp_long.csv).
"""

import argparse
import csv
import math
import random
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


def toric_mpp(L, rounds, p, seed=7):
    """The toric-code memory described above, as Stim text."""
    rng = random.Random(seed)
    h = lambda i, j: (i % L) * L + (j % L)  # horizontal edges
    v = lambda i, j: L * L + (i % L) * L + (j % L)  # vertical edges
    n = 2 * L * L
    stars = [[h(i, j), h(i, j - 1), v(i, j), v(i - 1, j)] for i in range(L) for j in range(L)]
    plaquettes = [[h(i, j), h(i + 1, j), v(i, j), v(i, j + 1)] for i in range(L) for j in range(L)]
    pairs = [q for i in range(L) for j in range(L) for q in (h(i, j), v(i, j))]
    channel = [p / 30] * 15
    channel[14] = p / 3  # ZZ
    channel[4] = p / 6  # XX
    lines = ["R " + " ".join(map(str, range(n)))]
    m = 0
    last_z = last_x = None
    for _ in range(rounds):
        lines.append(f"DEPOLARIZE1({p}) " + " ".join(map(str, range(n))))
        lines.append(f"PAULI_CHANNEL_2({', '.join(map(str, channel))}) " + " ".join(map(str, pairs)))
        for _ in range(L):
            a, b = rng.sample(range(n), 2)
            lines.append(f"E({p}) X{a} X{b}")
            lines.append(f"ELSE_CORRELATED_ERROR({p}) Z{a} Z{b}")
        lines.append(f"MPP({p}) " + " ".join("*".join(f"Z{q}" for q in s) for s in plaquettes))
        z = list(range(m, m + len(plaquettes)))
        m += len(plaquettes)
        lines.append(f"MPP({p}) " + " ".join("*".join(f"X{q}" for q in s) for s in stars))
        x = list(range(m, m + len(stars)))
        m += len(stars)
        for k, mz in enumerate(z):
            refs = [mz] + ([last_z[k]] if last_z else [])
            lines.append("DETECTOR " + " ".join(f"rec[{a - m}]" for a in refs))
        if last_x:
            for k, mx in enumerate(x):
                lines.append(f"DETECTOR rec[{mx - m}] rec[{last_x[k] - m}]")
        last_z, last_x = z, x
    lines.append(f"M({p}) " + " ".join(map(str, range(n))))
    data = list(range(m, m + n))
    m += n
    for k, s in enumerate(plaquettes):
        lines.append("DETECTOR " + " ".join(f"rec[{data[q] - m}]" for q in s) + f" rec[{last_z[k] - m}]")
    lines.append("OBSERVABLE_INCLUDE(0) " + " ".join(f"rec[{data[v(i, 0)] - m}]" for i in range(L)))
    return "\n".join(lines) + "\n"


def best_of(repeats, run):
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        best = min(best, time.perf_counter() - start)
    return best


def timing(sizes, shots, p):
    rows = []
    for L in sizes:
        text = toric_mpp(L, L, p)
        circuit = stim.Circuit(text)
        print(f"L = {L}: {circuit.num_qubits} qubits, {circuit.num_measurements} measurements, "
              f"{circuit.num_detectors} detectors", flush=True)
        common = dict(size=L, rounds=L, qubits=circuit.num_qubits, measurements=circuit.num_measurements,
                      detectors=circuit.num_detectors, p=p, shots=shots)
        for threads in (4, 1):
            seconds = best_of(3, lambda: sample_detectors(text, shots, seed=1, threads=threads))
            rows.append(dict(common, simulator="qvd", method="sample_detectors", version=VERSION,
                             threads=threads, seconds=f"{seconds:.3f}"))
            print(f"  Qevedo, {threads} threads: {seconds:.3f} s", flush=True)
        seconds = best_of(3, lambda: stim.Circuit(text).compile_detector_sampler(seed=1)
                          .sample(shots, separate_observables=True))
        rows.append(dict(common, simulator="stim", method="compile_detector_sampler().sample",
                         version=stim.__version__, threads=1, seconds=f"{seconds:.3f}"))
        print(f"  stim: {seconds:.3f} s", flush=True)
    return rows


def agreement(cases, shots):
    rows = []
    for L, p in cases:
        text = toric_mpp(L, L, p)
        circuit = stim.Circuit(text)
        ours_d, ours_o = sample_detectors(text, shots, seed=1)
        theirs_d, theirs_o = circuit.compile_detector_sampler(seed=1).sample(shots, separate_observables=True)
        quiet, _ = sample_detectors(toric_mpp(L, L, 0.0), 10_000, seed=2)
        a, b = ours_d.mean(axis=0), theirs_d.mean(axis=0)
        z = np.abs(a - b) / np.sqrt(np.maximum(a * (1 - a) + b * (1 - b), 1e-12) / shots)
        dem = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
        matching = pymatching.Matching.from_detector_error_model(dem)
        logical = [np.any(matching.decode_batch(d) != o, axis=1).mean()
                   for d, o in ((ours_d, ours_o), (theirs_d, theirs_o))]
        sigma = math.sqrt((logical[0] * (1 - logical[0]) + logical[1] * (1 - logical[1])) / shots)
        row = dict(
            size=L, rounds=L, p=p, shots=shots, detectors=circuit.num_detectors,
            noiseless_detections=int(quiet.sum()),
            qvd_detection_rate=f"{a.mean():.6f}", stim_detection_rate=f"{b.mean():.6f}",
            max_abs_z=f"{z.max():.2f}", qvd_logical_error_rate=f"{logical[0]:.6f}",
            stim_logical_error_rate=f"{logical[1]:.6f}",
            logical_z=f"{abs(logical[0] - logical[1]) / max(sigma, 1e-12):.2f}",
        )
        print(row, flush=True)
        rows.append(row)
    return rows


def long_run(cases, shots):
    """Many more shots, summed over seeds: per-detector and observable
    rates, and the logical error rate after decoding, to tell a chance
    difference in the agreement table from a bias."""
    rows = []
    for L, p, seeds in cases:
        text = toric_mpp(L, L, p)
        circuit = stim.Circuit(text)
        dem = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
        matching = pymatching.Matching.from_detector_error_model(dem)
        n = circuit.num_detectors
        ours, theirs = np.zeros(n + 1), np.zeros(n + 1)
        failures = [0, 0]
        for seed in seeds:
            results = (sample_detectors(text, shots, seed=seed),
                       circuit.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True))
            for k, ((dets, obs), acc) in enumerate(zip(results, (ours, theirs))):
                acc += np.concatenate([dets.sum(axis=0), obs.sum(axis=0)])
                failures[k] += int(np.any(matching.decode_batch(dets) != obs, axis=1).sum())
        total = shots * len(seeds)
        a, b = ours / total, theirs / total
        z = (a - b) / np.sqrt(np.maximum(a * (1 - a) + b * (1 - b), 1e-30) / total)
        la, lb = failures[0] / total, failures[1] / total
        lz = (la - lb) / math.sqrt(max((la * (1 - la) + lb * (1 - lb)) / total, 1e-30))
        row = dict(size=L, rounds=L, p=p, shots=total, detectors=n,
                   max_abs_detector_z=f"{np.abs(z[:n]).max():.2f}", mean_detector_z2=f"{np.mean(z[:n] ** 2):.2f}",
                   observable_z=f"{z[n]:.2f}", qvd_logical_error_rate=f"{la:.7f}",
                   stim_logical_error_rate=f"{lb:.7f}", logical_z=f"{lz:.2f}")
        print(row, flush=True)
        rows.append(row)
    return rows


def write(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="10,20,30")
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--p", type=float, default=0.001)
    parser.add_argument("--agreement-shots", type=int, default=200_000)
    parser.add_argument("--long", action="store_true",
                        help="only the long run: 400,000 shots per seed, 30 seeds at L = 5, p = 0.002 "
                             "and 0.005, 10 seeds at p = 0.01 (L = 3 and 5)")
    args = parser.parse_args()
    if args.long:
        cases = [(5, 0.002, range(10, 40)), (5, 0.005, range(10, 40)),
                 (3, 0.01, range(10, 20)), (5, 0.01, range(10, 20))]
        write(RESULTS / "mpp_long.csv", long_run(cases, 400_000))
        return
    write(RESULTS / "mpp.csv", timing([int(s) for s in args.sizes.split(",")], args.shots, args.p))
    cases = [(3, 0.003), (5, 0.002), (5, 0.005), (7, 0.003)]
    write(RESULTS / "mpp_agreement.csv", agreement(cases, args.agreement_shots))


if __name__ == "__main__":
    main()
