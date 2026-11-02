"""Two builds of the Python package against each other, interleaved, on the
output benchmarks of `bench_output.py`: how one change moved the time of
`sample` and `sample_detectors`.

    python benchmarks/compare_builds.py --before /venv-a/bin/python --after /venv-b/bin/python

Each interpreter has a different build of qevedo-simulator installed (for
example the previous release from PyPI and a `maturin build` of the
change). For every case the two run in turn, each in a fresh process timing
the best of `--repeats` calls, for `--rounds` rounds; the best time of each
build is kept. Interleaving matters on a shared virtual machine, where the
speed drifts by several percent within minutes.

Cases: Stim's generated `surface_code:rotated_memory_z`, d rounds, all
four noise parameters at 0.001, 10,000 shots; `sample` (bool, bit-packed),
`sample_detectors` (bool, observables separate or appended), on 4 threads
and on 1. Writes results/output_before_after.csv, or the file given with
--out (the small codes take a few milliseconds per call, so they are also
run with --repeats 15 into output_before_after_small.csv).
"""

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
CALLS = {
    "sample": "sample(text, 10000, seed=1, threads=threads)",
    "sample(bit_packed=True)": "sample(text, 10000, seed=1, threads=threads, bit_packed=True)",
    "sample_detectors": "sample_detectors(text, 10000, seed=1, threads=threads)",
    "sample_detectors(separate_observables=False)":
        "sample_detectors(text, 10000, seed=1, threads=threads, separate_observables=False)",
}


def one(d, method, threads, repeats):
    """Run in the child process: print the best time of one case."""
    import stim
    from qevedo.simulator import sample, sample_detectors  # noqa: F401 (used by eval)

    text = str(stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=d,
        after_clifford_depolarization=0.001, before_round_data_depolarization=0.001,
        after_reset_flip_probability=0.001, before_measure_flip_probability=0.001))
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        result = eval(CALLS[method])
        best = min(best, time.perf_counter() - start)
        del result
    print(best)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before")
    parser.add_argument("--after")
    parser.add_argument("--distances", default="11,21,51")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default="output_before_after.csv")
    parser.add_argument("--one", nargs=4, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.one:
        d, method, threads, repeats = args.one
        return one(int(d), method, int(threads), int(repeats))

    rows = []
    for d in map(int, args.distances.split(",")):
        for method in CALLS:
            for threads in (4, 1):
                best = {"before": float("inf"), "after": float("inf")}
                for _ in range(args.rounds):
                    for build in ("before", "after"):
                        python = getattr(args, build)
                        out = subprocess.run(
                            [python, "-I", __file__, "--one", str(d), method, str(threads),
                             str(args.repeats)],
                            check=True, capture_output=True, text=True).stdout
                        best[build] = min(best[build], float(out))
                ratio = best["after"] / best["before"]
                rows.append(dict(distance=d, method=method, threads=threads,
                                 before=f"{best['before']:.4f}", after=f"{best['after']:.4f}",
                                 ratio=f"{ratio:.3f}"))
                print(rows[-1], flush=True)
    with open(RESULTS / args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    sys.exit(main())
