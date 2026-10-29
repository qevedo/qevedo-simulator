"""Noisy surface codes: qvd's stabilizer backend against Stim, for speed and
for agreement of the sampled statistics.

    cargo build --release --example stabilizer
    cd python && maturin develop --release && cd ..   # for the agreement check
    python benchmarks/bench_noise.py [--distances 11,21,51,101] [--shots 10000] [--p 0,0.001]

Every case is `library::surface_code(d, d)` with uniform circuit-level
depolarizing noise of strength p (`library::with_depolarizing_noise`),
written by the `stabilizer` example in Stim's format, so both simulators run
exactly the same noisy circuit (p = 0 is the noiseless circuit). The file
has one instruction per layer and channel, as Stim's generated circuits do. Timing: qvd on all cores and on one thread
(best of 3, the example's printed time); Stim 1 thread, `reference_sample()`
plus a `FlipSimulator` (measurement-major results, as qvd's) and
`compile_sampler().sample()` (shot-major), best of 3 (1 run at d = 101).

Agreement: a detector is the XOR of two consecutive measurements of the same
qubit (an ancilla's stabilizer in consecutive rounds), which never fires
without noise. For smaller codes, the fraction of shots in which each
detector fires is compared between qvd (through the Python package) and
Stim's sampler, with a two-sample z-score per detector.

Writes results/noise.csv and results/noise_agreement.csv.
"""

import argparse
import csv
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import stim

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "target" / "release" / "examples" / "stabilizer"
RESULTS = ROOT / "benchmarks" / "results"
VERSION = re.search(r'^version = "(.*)"', (ROOT / "python" / "Cargo.toml").read_text(), re.M).group(1)


def noise_args(p):
    # p = 0 is the noiseless circuit, without zero-probability channels.
    return ["--noise", repr(p)] if p > 0 else []


def write_circuit(d, p, path, shots=1):
    out = subprocess.run(
        [str(EXAMPLE), "surface", str(d), str(d), str(shots), *noise_args(p), "--stim", path],
        capture_output=True, text=True, check=True,
    ).stdout
    qubits, gates, measurements = map(int, re.search(r"(\d+) qubits, (\d+) gates, (\d+) measurements", out).groups())
    return qubits, gates, measurements


def time_qvd(d, p, shots, threads, path, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        out = subprocess.run(
            [str(EXAMPLE), "surface", str(d), str(d), str(shots), *noise_args(p), "--stim", path],
            capture_output=True, text=True, check=True, env={**os.environ, "QVD_THREADS": str(threads)},
        ).stdout
        best = min(best, float(re.search(r": ([\d.]+) s", out).group(1)))
    return best


def time_stim(circuit, shots, repeats):
    flip, sample = float("inf"), float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        circuit.reference_sample()
        simulator = stim.FlipSimulator(batch_size=shots)
        simulator.do(circuit)
        simulator.get_measurement_flips(bit_packed=True)
        flip = min(flip, time.perf_counter() - start)
        start = time.perf_counter()
        circuit.compile_sampler().sample(shots, bit_packed=True)
        sample = min(sample, time.perf_counter() - start)
    return flip, sample


def from_stim_text(text):
    """The qvd circuit of a Stim file written by qvd's `to_stim`."""
    from qevedo.simulator import Circuit

    lines = [line.split() for line in text.splitlines() if line.strip()]
    n = 1 + max(int(q) for line in lines for q in line[1:])
    circuit = Circuit(n)
    names = {"H": "h", "S": "s", "S_DAG": "sdg", "X": "x", "Y": "y", "Z": "z", "CX": "cx", "CZ": "cz", "SWAP": "swap"}
    clbit = 0
    for name, *targets in lines:
        qubits = [int(q) for q in targets]
        if name == "M":
            for q in qubits:
                circuit.measure(q, clbit)
                clbit += 1
        elif name == "R":
            for q in qubits:
                circuit.reset(q)
        elif name in ("CX", "CZ", "SWAP"):
            for a, b in zip(qubits[::2], qubits[1::2]):
                circuit.gate(names[name], [a, b])
        elif name in names:
            for q in qubits:
                circuit.gate(names[name], [q])
        else:
            channel, args = re.fullmatch(r"(\w+)\((.*)\)", name).groups()
            args = [float(a) for a in args.split(",")]
            if channel == "DEPOLARIZE2":
                for a, b in zip(qubits[::2], qubits[1::2]):
                    circuit.depolarize2(a, b, args[0])
            else:
                for q in qubits:
                    {
                        "X_ERROR": lambda: circuit.x_error(q, args[0]),
                        "Y_ERROR": lambda: circuit.y_error(q, args[0]),
                        "Z_ERROR": lambda: circuit.z_error(q, args[0]),
                        "DEPOLARIZE1": lambda: circuit.depolarize1(q, args[0]),
                        "PAULI_CHANNEL_1": lambda: circuit.pauli_channel_1(q, *args),
                    }[channel]()
    return circuit


def detector_pairs(text):
    """Pairs of measurement indices (consecutive measurements of one qubit)."""
    last, pairs, index = {}, [], 0
    for line in text.splitlines():
        if line.startswith("M "):
            for q in map(int, line.split()[1:]):
                if q in last:
                    pairs.append((last[q], index))
                last[q] = index
                index += 1
    return np.array(pairs)


def agreement(d, p, shots):
    from qevedo.simulator import sample

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "c.stim")
        write_circuit(d, p, path)
        text = open(path).read()
    pairs = detector_pairs(text)
    ours = sample(from_stim_text(text), shots, seed=1)  # (measurements, shots)
    ours = ours[pairs[:, 0]] ^ ours[pairs[:, 1]]
    theirs = stim.Circuit(text).compile_sampler(seed=1).sample(shots).T
    theirs = theirs[pairs[:, 0]] ^ theirs[pairs[:, 1]]
    a, b = ours.mean(axis=1), theirs.mean(axis=1)
    pooled = (a + b) / 2
    sigma = np.sqrt(np.maximum(pooled * (1 - pooled), 1e-300) * 2 / shots)
    z = np.where(pooled > 0, (a - b) / sigma, 0.0)
    return len(pairs), a.mean(), b.mean(), float(np.abs(z).max()), float(np.mean(np.abs(z) > 3))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distances", default="11,21,51,101")
    parser.add_argument("--shots", type=int, default=10000)
    parser.add_argument("--p", default="0,0.001")
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    parser.add_argument("--agreement", default="3:0.001,5:0.001,5:0.005,7:0.003")
    parser.add_argument("--agreement-shots", type=int, default=200_000)
    parser.add_argument("--qvd-version", default=VERSION, help="for the CSV (default: python/Cargo.toml)")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "c.stim")
        for p in map(float, args.p.split(",")):
            for d in map(int, args.distances.split(",")):
                qubits, gates, measurements = write_circuit(d, p, path)
                circuit = stim.Circuit.from_file(path)
                base = dict(distance=d, qubits=qubits, rounds=d, gates=gates, measurements=measurements,
                            p=p, shots=args.shots)
                for threads in (args.threads, 1):
                    seconds = time_qvd(d, p, args.shots, threads, path)
                    rows.append(dict(base, simulator="qvd", method="stabilizer::sample", version=args.qvd_version,
                                     threads=threads, seconds=f"{seconds:.3f}"))
                    print(rows[-1], flush=True)
                flip, sampled = time_stim(circuit, args.shots, 1 if d > 51 else 3)
                for method, seconds in (("reference_sample + FlipSimulator", flip),
                                        ("compile_sampler().sample", sampled)):
                    rows.append(dict(base, simulator="stim", method=method, version=stim.__version__,
                                     threads=1, seconds=f"{seconds:.3f}"))
                    print(rows[-1], flush=True)
    with open(RESULTS / "noise.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    agree = []
    for case in args.agreement.split(","):
        d, p = case.split(":")
        detectors, ours, theirs, max_z, over3 = agreement(int(d), float(p), args.agreement_shots)
        agree.append(dict(distance=d, rounds=d, p=p, shots=args.agreement_shots, detectors=detectors,
                          qvd_detection_rate=f"{ours:.6f}", stim_detection_rate=f"{theirs:.6f}",
                          max_abs_z=f"{max_z:.2f}", fraction_abs_z_over_3=f"{over3:.4f}"))
        print(agree[-1], flush=True)
    with open(RESULTS / "noise_agreement.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(agree[0]))
        writer.writeheader()
        writer.writerows(agree)


if __name__ == "__main__":
    main()
