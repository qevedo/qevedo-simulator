"""The cost of handing results to Python: Qevedo's `sample` (raw measurement
records) and `sample_detectors` (detection events) on Stim's generated
surface codes, as NumPy bool arrays and bit-packed, against Stim's
equivalents.

    cd python && maturin develop --release && cd ..
    pip install stim
    python benchmarks/bench_output.py [--distances 11,21,51,101] [--shots 10000]

Every circuit is `stim.Circuit.generated("surface_code:rotated_memory_z",
distance=d, rounds=d, ...)` with all four noise parameters at p = 0.001,
the same text in both simulators, 10,000 shots, best of 3, in one Python
process. Qevedo runs on 4 threads and on 1, Stim on its one.

Measurement records (`sample`):
- Qevedo `sample(text, shots)`: measurement-major, shape (measurements, shots),
  bool; with `bit_packed=True`, uint8 with eight shots per byte.
- Stim `compile_sampler().sample(shots)`: shot-major, shape (shots,
  measurements), bool; and with `bit_packed=True`. Stim has no
  measurement-major NumPy output from its sampler; its `FlipSimulator`
  gives the flips measurement-major but not the records, so the sampler is
  the like-for-like interface.

Detection events (`sample_detectors`): both shot-major, bool and
bit-packed (eight detectors per byte), observables separate.

At d = 101 only the bit-packed outputs run (each 1.3 GB; a bool array
would be 10 GB, on a 15 GB machine), once each.

Writes results/output.csv; with --label, appends to it rows marked with
that label instead (used for before/after timings of one change).
"""

import argparse
import csv
import re
import time
from pathlib import Path

import stim

from qevedo.simulator import sample, sample_detectors

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks" / "results"
VERSION = re.search(r'^version = "(.*)"', (ROOT / "python" / "Cargo.toml").read_text(), re.M).group(1)


def generated(d, p):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d,
        rounds=d,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
    )


def best_of(repeats, run):
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        result = run()
        best = min(best, time.perf_counter() - start)
        del result
    return best


def cases(text, shots, bit_packed_detectors):
    """(output, simulator, method, threads, run) for one circuit."""
    for threads in (4, 1):
        yield ("records", "qvd", "sample", threads,
               lambda t=threads: sample(text, shots, seed=1, threads=t))
        yield ("records", "qvd", "sample(bit_packed=True)", threads,
               lambda t=threads: sample(text, shots, seed=1, threads=t, bit_packed=True))
    yield ("records", "stim", "compile_sampler().sample", 1,
           lambda: stim.Circuit(text).compile_sampler(seed=1).sample(shots))
    yield ("records", "stim", "compile_sampler().sample(bit_packed=True)", 1,
           lambda: stim.Circuit(text).compile_sampler(seed=1).sample(shots, bit_packed=True))
    for threads in (4, 1):
        yield ("detectors", "qvd", "sample_detectors", threads,
               lambda t=threads: sample_detectors(text, shots, seed=1, threads=t))
        if bit_packed_detectors:
            yield ("detectors", "qvd", "sample_detectors(bit_packed=True)", threads,
                   lambda t=threads: sample_detectors(text, shots, seed=1, threads=t, bit_packed=True))
    yield ("detectors", "stim", "compile_detector_sampler().sample", 1,
           lambda: stim.Circuit(text).compile_detector_sampler(seed=1)
           .sample(shots, separate_observables=True))
    yield ("detectors", "stim", "compile_detector_sampler().sample(bit_packed=True)", 1,
           lambda: stim.Circuit(text).compile_detector_sampler(seed=1)
           .sample(shots, separate_observables=True, bit_packed=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distances", default="11,21,51,101")
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--p", type=float, default=0.001)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--only", choices=["qvd", "stim"], help="time one simulator only")
    parser.add_argument("--label", help="append rows with this label instead of rewriting")
    args = parser.parse_args()

    try:
        sample_detectors(str(generated(3, args.p)), 8, bit_packed=True)
        bit_packed_detectors = True
    except TypeError:  # versions before 0.1.4
        bit_packed_detectors = False

    rows = []
    for d in map(int, args.distances.split(",")):
        circuit = generated(d, args.p)
        text = str(circuit)
        print(f"d = {d}: {circuit.num_measurements} measurements, "
              f"{circuit.num_detectors} detectors", flush=True)
        for output, simulator, method, threads, run in cases(text, args.shots, bit_packed_detectors):
            if args.only and simulator != args.only:
                continue
            if d >= 101 and "bit_packed" not in method:
                continue
            seconds = best_of(1 if d >= 101 else args.repeats, run)
            rows.append(dict(
                label=args.label or "", distance=d, rounds=d, qubits=circuit.num_qubits,
                measurements=circuit.num_measurements, detectors=circuit.num_detectors,
                p=args.p, shots=args.shots, output=output, simulator=simulator, method=method,
                version=VERSION if simulator == "qvd" else stim.__version__,
                threads=threads, seconds=f"{seconds:.3f}",
            ))
            print(f"  {simulator} {method}, {threads} thread(s): {seconds:.3f} s", flush=True)

    path = RESULTS / "output.csv"
    append = bool(args.label) and path.exists()
    with open(path, "a" if append else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        if not append:
            writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
