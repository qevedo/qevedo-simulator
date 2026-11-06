"""Noisy Clifford+T circuits on the near-Clifford backend, against Clifft.

    cd python && maturin develop --release && cd ..
    pip install clifft numpy
    python benchmarks/bench_noisy_nearclifford.py [--shots 10000] [--agreement-shots 200000]

The circuits (`noisy_clifford_t`): n qubits, `depth` layers. Each layer
applies a random one-qubit Clifford (H, S, S†, √X or nothing) to every
qubit, then CX on a random perfect matching of the qubits. T or T† gates
go on random qubits after random layers, `t_count` in all. The noise is
uniform circuit-level depolarizing noise of strength p, as in
`library::with_depolarizing_noise`: `DEPOLARIZE1(p)` after every one-qubit
gate (T included), `DEPOLARIZE2(p)` after every CX, and `X_ERROR(p)`
before the final measurement of every qubit. Both simulators get the same
circuit: Qevedo through the Python `Circuit` methods, Clifft as Stim-format
text with `T` and `T_DAG`.

Timing (results/noisy_nearclifford.csv): shots of every qubit's final
measurement, best of 3 in one process. Qevedo: `run(..., backend=
"near_clifford")`, which returns counts (a dict of bit strings); Clifft:
`compile` and `sample` (a shot-major uint8 array). Both on 4 threads and
on 1 (Clifft's `threads` argument). The time of building Qevedo's circuit
in Python is not included, nor that of writing Clifft's text.

Sweep (`--sweep`, results/noisy_nearclifford_sweep.csv): the 100-qubit
circuit at p = 0, 1e-5, 1e-4 and 1e-3, on one thread, to show how the time
depends on how many shots errors split off. At p = 0 the circuit keeps a
`X_ERROR(0)` channel, so Qevedo still takes its noisy path.

Agreement (results/noisy_nearclifford_agreement.csv): per-qubit rates of
measuring 1 with a two-sample z-score, and, on circuits of at most 8
qubits, the whole distribution: a chi-square test of homogeneity between
the two samples (outcomes seen fewer than 5 times in both pooled into one
bin) and its p-value.
"""

import argparse
import csv
import math
import random
import time
from pathlib import Path

import numpy as np

import clifft
from qevedo.simulator import Circuit, run

try:
    from importlib.metadata import version

    VERSION = version("qevedo-simulator")
    CLIFFT_VERSION = version("clifft")
except Exception:  # noqa: BLE001
    VERSION = CLIFFT_VERSION = "?"

RESULTS = Path(__file__).parent / "results"


def noisy_clifford_t(n, depth, t_count, p, seed=1):
    """The circuit as (Qevedo Circuit, Stim-format text for Clifft)."""
    rng = random.Random(seed)
    circuit = Circuit(n, n)
    lines = []
    t_layers = sorted(rng.randrange(depth) for _ in range(t_count))

    def one(name, q):
        method = {"H": circuit.h, "S": circuit.s, "S_DAG": circuit.sdg,
                  "SQRT_X": circuit.sx, "T": circuit.t, "T_DAG": circuit.tdg}[name]
        method(q)
        lines.append(f"{name} {q}")
        if p > 0:
            circuit.depolarize1(q, p)
            lines.append(f"DEPOLARIZE1({p}) {q}")

    for layer in range(depth):
        for q in range(n):
            gate = rng.choice(["H", "S", "S_DAG", "SQRT_X", None])
            if gate:
                one(gate, q)
        order = list(range(n))
        rng.shuffle(order)
        for a, b in zip(order[::2], order[1::2]):
            circuit.cx(a, b)
            lines.append(f"CX {a} {b}")
            if p > 0:
                circuit.depolarize2(a, b, p)
                lines.append(f"DEPOLARIZE2({p}) {a} {b}")
        while t_layers and t_layers[0] == layer:
            t_layers.pop(0)
            one(rng.choice(["T", "T_DAG"]), rng.randrange(n))
    for q in range(n):
        if p > 0:
            circuit.x_error(q, p)
            lines.append(f"X_ERROR({p}) {q}")
        circuit.measure(q, q)
        lines.append(f"M {q}")
    return circuit, "\n".join(lines) + "\n"


def qvd_shots(circuit, n, shots, seed, threads=None):
    """Shot-major uint8 array (shots, n) from Qevedo's counts."""
    result = run(circuit, shots, backend="near_clifford", seed=seed, threads=threads)
    rows = []
    for key, count in result.counts.items():
        # Classical bit 0 is the last character.
        bits = np.frombuffer(key[::-1].encode(), dtype=np.uint8) - ord("0")
        rows.append(np.broadcast_to(bits, (count, n)))
    return np.concatenate(rows), result


def clifft_shots(text, shots, seed, threads=1):
    program = clifft.compile(text)
    return np.asarray(clifft.sample(program, shots, seed=seed, threads=threads).measurements)


def best_of(repeats, call):
    best = math.inf
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        best = min(best, time.perf_counter() - start)
    return best


CASES = [
    # (qubits, depth, T count, p)
    (24, 20, 10, 0.001),
    (60, 20, 12, 0.001),
    (100, 30, 14, 0.001),
    (200, 40, 12, 0.001),
    (60, 20, 12, 0.01),
]


def timing(shots):
    rows = []
    for n, depth, t_count, p in CASES:
        circuit, text = noisy_clifford_t(n, depth, t_count, p)
        peak = run(circuit, 1, backend="near_clifford", seed=1).peak_active_qubits
        common = dict(qubits=n, depth=depth, t_count=t_count, p=p, shots=shots,
                      peak_active_qubits=peak)
        print(f"n = {n}, depth {depth}, {t_count} T, p = {p}, peak {peak}", flush=True)
        for threads in (4, 1):
            seconds = best_of(3, lambda: run(circuit, shots, backend="near_clifford",
                                             seed=1, threads=threads))
            rows.append(dict(common, simulator="qvd", version=VERSION, threads=threads,
                             seconds=f"{seconds:.4f}"))
            print(f"  Qevedo, {threads} threads: {seconds:.4f} s", flush=True)
            seconds = best_of(3, lambda: clifft_shots(text, shots, 1, threads))
            rows.append(dict(common, simulator="clifft", version=CLIFFT_VERSION,
                             threads=threads, seconds=f"{seconds:.4f}"))
            print(f"  clifft, {threads} threads: {seconds:.4f} s", flush=True)
    return rows


def sweep(shots):
    rows = []
    for p in (0.0, 1e-5, 1e-4, 1e-3):
        circuit, text = noisy_clifford_t(100, 30, 14, p)
        if p == 0.0:
            circuit.x_error(0, 0.0)
        for name, version, call in (
            ("qvd", VERSION, lambda: run(circuit, shots, backend="near_clifford", seed=1, threads=1)),
            ("clifft", CLIFFT_VERSION, lambda: clifft_shots(text, shots, 1, 1)),
        ):
            seconds = best_of(3, call)
            rows.append(dict(qubits=100, depth=30, t_count=14, p=p, shots=shots, simulator=name,
                             version=version, threads=1, seconds=f"{seconds:.4f}"))
            print(f"p = {p}: {'Qevedo' if name == 'qvd' else name} {seconds:.4f} s", flush=True)
    return rows


def chi_square(ours, theirs):
    """Chi-square test of homogeneity between two samples of bit rows; rare
    outcomes (fewer than 5 in both pooled) share one bin. Returns (chi2, dof,
    p-value)."""
    def counts(sample):
        keys, k = np.unique(np.packbits(sample, axis=1), axis=0, return_counts=True)
        return {bytes(key): c for key, c in zip(keys, k)}

    a, b = counts(ours), counts(theirs)
    na, nb = len(ours), len(theirs)
    bins, rare = [], [0, 0]
    for key in set(a) | set(b):
        x, y = a.get(key, 0), b.get(key, 0)
        if x + y < 5:
            rare[0] += x
            rare[1] += y
        else:
            bins.append((x, y))
    if sum(rare):
        bins.append(tuple(rare))
    chi2 = 0.0
    for x, y in bins:
        total = x + y
        ex, ey = total * na / (na + nb), total * nb / (na + nb)
        chi2 += (x - ex) ** 2 / ex + (y - ey) ** 2 / ey
    dof = len(bins) - 1
    # Survival function of chi-square through the regularized gamma
    # function, by its series (fine for these sizes).
    return chi2, dof, chi2_sf(chi2, dof)


def chi2_sf(x, k):
    if k <= 0:
        return 1.0
    a, z = k / 2, x / 2
    if z < a + 1:
        term = total = 1 / a
        for i in range(1, 10_000):
            term *= z / (a + i)
            total += term
            if term < total * 1e-15:
                break
        lower = total * math.exp(-z + a * math.log(z) - math.lgamma(a)) if z > 0 else 0.0
        return 1 - lower
    # Continued fraction for the upper tail.
    b, c, d = z + 1 - a, 1e300, 1 / (z + 1 - a)
    h = d
    for i in range(1, 10_000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = b + an / c
        c = c if abs(c) > 1e-300 else 1e-300
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return math.exp(-z + a * math.log(z) - math.lgamma(a)) * h


AGREEMENT = [
    # (qubits, depth, T count, p)
    (6, 12, 6, 0.01),
    (8, 16, 8, 0.02),
    (8, 16, 8, 0.05),
    (24, 20, 10, 0.003),
    (60, 20, 12, 0.003),
]


def agreement(shots):
    rows = []
    for n, depth, t_count, p in AGREEMENT:
        circuit, text = noisy_clifford_t(n, depth, t_count, p, seed=2)
        ours, _ = qvd_shots(circuit, n, shots, seed=1)
        theirs = clifft_shots(text, shots, seed=1, threads=4)
        ra, rb = ours.mean(axis=0), theirs.mean(axis=0)
        pooled = (ra + rb) / 2
        se = np.sqrt(np.maximum(pooled * (1 - pooled), 1e-300) * 2 / shots)
        z = np.where(pooled > 0, (ra - rb) / se, 0.0)
        row = dict(qubits=n, depth=depth, t_count=t_count, p=p, shots=shots,
                   mean_rate_qvd=f"{ra.mean():.6f}", mean_rate_clifft=f"{rb.mean():.6f}",
                   max_abs_z=f"{np.abs(z).max():.2f}", mean_z2=f"{(z ** 2).mean():.3f}")
        if n <= 8:
            chi2, dof, pvalue = chi_square(ours, theirs)
            row.update(chi2=f"{chi2:.1f}", dof=dof, p_value=f"{pvalue:.3f}")
        rows.append(row)
        print(row, flush=True)
    return rows


def write(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--agreement-shots", type=int, default=200_000)
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument("--skip-agreement", action="store_true")
    parser.add_argument("--sweep", action="store_true", help="only the noise-rate sweep")
    args = parser.parse_args()
    if args.sweep:
        write(RESULTS / "noisy_nearclifford_sweep.csv", sweep(args.shots))
        return
    if not args.skip_agreement:
        write(RESULTS / "noisy_nearclifford_agreement.csv", agreement(args.agreement_shots))
    if not args.skip_timing:
        write(RESULTS / "noisy_nearclifford.csv", timing(args.shots))


if __name__ == "__main__":
    main()
