"""Time qvd's Pauli propagation against IBM's pauli-prop and write results/pauliprop.csv.

    cargo build --release --example pauliprop
    python benchmarks/bench_pauliprop.py [--repeats 3]

Each case is IBM's 127-qubit kicked Ising circuit (library::kicked_ising on
library::eagle_heavy_hex) and the observable Z62. qvd runs on all threads and
on one (best of --repeats each); pauli-prop runs once, in its own process,
through benchmarks/compare_pauliprop.py with --max-terms set well above the
terms the case needs, so that only the coefficient threshold truncates.
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXAMPLE = ROOT / "target" / "release" / "examples" / "pauliprop"

# (steps, θ_h, threshold, pauli-prop max_terms)
CASES = [
    (20, 0.2, 1e-5, 200_000),
    (20, 0.4, 1e-5, 4_000_000),
    (20, 0.6, 1e-4, 2_500_000),
    (5, 1.2, 1e-6, 6_000_000),
    (5, 1.4, 1e-6, 2_000_000),
]


def qvd(steps, theta, atol, threads, qasm=None):
    env = dict(os.environ, QVD_THREADS=str(threads))
    command = [str(EXAMPLE), str(steps), str(theta), "Z62", str(atol)]
    if qasm:
        command += ["--qasm", qasm]
    out = subprocess.run(command, env=env, capture_output=True, text=True, check=True).stdout
    m = re.search(
        r": (-?[\d.]+) \(dropped one-norm ([\d.e+-]+)\), \d+ rotations, peak (\d+) terms, final (\d+), ([\d.]+) s",
        out,
    )
    value, norm, peak, final, seconds = m.groups()
    return float(value), float(norm), int(peak), int(final), float(seconds)


def pauli_prop(qasm, atol, max_terms):
    out = subprocess.run(
        [sys.executable, str(HERE / "compare_pauliprop.py"), qasm, "Z62", "--atol", str(atol),
         "--max-terms", str(max_terms)],
        capture_output=True, text=True, check=True,
    ).stdout
    if "hit --max-terms" in out:
        raise SystemExit(f"pauli-prop hit max_terms={max_terms}; raise it")
    m = re.search(
        r"= (-?[\d.]+) \(truncated one-norm ([\d.e+-]+)\), (\d+) terms, wall ([\d.]+) s "
        r"\(Clifford preprocessing ([\d.]+) s, propagation ([\d.]+) s\)",
        out,
    )
    value, norm, terms, wall, _, propagation = m.groups()
    return float(value), float(norm), int(terms), float(wall), float(propagation)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    from importlib.metadata import version

    threads = os.cpu_count()
    qvd_version = re.search(r'^version = "(.*)"', (ROOT / "Cargo.toml").read_text(), re.M).group(1)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for steps, theta, atol, max_terms in CASES:
            qasm = os.path.join(tmp, f"ki_{steps}_{theta}.qasm")
            case = dict(qubits=127, steps=steps, theta_h=theta, observable="Z62", min_abs_coefficient=atol)
            for t in (threads, 1):
                runs = [qvd(steps, theta, atol, t, qasm) for _ in range(args.repeats)]
                value, norm, peak, final, _ = runs[0]
                seconds = min(r[4] for r in runs)
                rows.append(dict(case, simulator="qvd", version=qvd_version, threads=t, part="total",
                                 seconds=seconds, value=value, truncated_one_norm=norm, terms=final))
                print(rows[-1], flush=True)
            value, norm, terms, wall, propagation = pauli_prop(qasm, atol, max_terms)
            for part, seconds in (("total", wall), ("propagation only", propagation)):
                rows.append(dict(case, simulator="pauli-prop", version=version("pauli-prop"), threads="",
                                 part=part, seconds=seconds, value=value, truncated_one_norm=norm, terms=terms))
            print(rows[-2], flush=True)

    with open(HERE / "results" / "pauliprop.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
