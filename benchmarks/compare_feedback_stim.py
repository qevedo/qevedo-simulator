"""Random Stim circuits with classical feedback: Qevedo against Stim.

    python benchmarks/compare_feedback_stim.py 600 2026

Draws `cases` random circuits (seeded) of 1-5 qubits with one- and
two-qubit Clifford gates, measurements in every basis (some inverted,
some with flip probabilities), MPP products, MPAD, classically
controlled Paulis (CX/CY/CZ/XCZ/YCZ with rec[-k], sometimes sweep[k])
and, in every other circuit, Pauli noise and detectors. Each runs 20,000
shots in both simulators. Noiseless circuits: the sets of distinct
outcomes must be equal and the per-bit rates within 5 sigma. Noisy
circuits: per-bit rates, the rates of every pair's XOR and the detector
rates within 5 sigma. Prints the failures and a summary line; exits 1
if any circuit differs. Needs stim and numpy.
"""

import random
import sys

import numpy as np
import stim

import qevedo.simulator as qs

ONE = ["H", "S", "S_DAG", "SQRT_X", "X", "Y", "Z", "H_YZ", "C_XYZ"]
TWO = ["CX", "CY", "CZ", "SWAP", "XCZ", "ISWAP", "SQRT_XX"]
CONTROLLED = {"CX": 0, "CY": 0, "CNOT": 0, "ZCX": 0, "CZ": 2, "XCZ": 1, "YCZ": 1}


def random_circuit(rng, n, noisy):
    lines = []
    measured = 0
    for _ in range(rng.randint(8, 20)):
        kind = rng.random()
        if kind < 0.25:
            lines.append(f"{rng.choice(ONE)} {rng.randrange(n)}")
        elif kind < 0.5 and n > 1:
            a, b = rng.sample(range(n), 2)
            lines.append(f"{rng.choice(TWO)} {a} {b}")
        elif kind < 0.68:
            name = rng.choice(["M", "MX", "MY", "MR", "MRX"])
            p = f"({rng.choice([0.1, 0.25])})" if noisy and rng.random() < 0.5 else ""
            bang = "!" if rng.random() < 0.25 else ""
            lines.append(f"{name}{p} {bang}{rng.randrange(n)}")
            measured += 1
        elif kind < 0.73 and n > 1:
            a, b = rng.sample(range(n), 2)
            p = f"({rng.choice([0.1, 0.25])})" if noisy and rng.random() < 0.5 else ""
            lines.append(f"MPP{p} {rng.choice('XYZ')}{a}*{rng.choice('XYZ')}{b}")
            measured += 1
        elif kind < 0.76:
            lines.append(f"MPAD {rng.randrange(2)}")
            measured += 1
        elif kind < 0.92 and measured > 0:
            name = rng.choice(list(CONTROLLED))
            side = CONTROLLED[name]
            rec = f"rec[-{rng.randint(1, min(measured, 4))}]"
            q = str(rng.randrange(n))
            if side == 2:
                side = rng.randrange(2)
            pair = (rec, q) if side == 0 else (q, rec)
            if rng.random() < 0.15:
                pair = tuple(f"sweep[{rng.randrange(3)}]" if t == rec else t for t in pair)
            lines.append(f"{name} {pair[0]} {pair[1]}")
        elif noisy:
            ch = rng.choice(["X_ERROR(0.1)", "Z_ERROR(0.2)", "DEPOLARIZE1(0.15)"])
            lines.append(f"{ch} {rng.randrange(n)}")
    lines.append("M " + " ".join(str(q) for q in range(n)))
    measured += n
    if noisy and measured >= 2:
        for _ in range(3):
            k = sorted(rng.sample(range(1, measured + 1), 2))
            lines.append(f"DETECTOR rec[-{k[0]}] rec[-{k[1]}]")
    return "\n".join(lines) + "\n"


def close(a, b, shots, what):
    # Two independent estimates of the same rate.
    p = (a + b) / 2
    sigma = np.sqrt(2 * p * (1 - p) / shots) + 1e-12
    return np.all(np.abs(a - b) <= 5 * sigma), what


def main(cases, seed):
    rng = random.Random(seed)
    shots = 20_000
    failures = 0
    feedback_lines = 0
    for case in range(cases):
        n = rng.randint(1, 5)
        noisy = case % 2 == 1
        text = random_circuit(rng, n, noisy)
        feedback_lines += sum(("rec[" in l or "sweep[" in l) and not l.startswith("DETECTOR") for l in text.splitlines())
        circuit = stim.Circuit(text)
        ours = qs.sample(text, shots, seed=case).astype(np.uint8)  # (m, shots)
        theirs = circuit.compile_sampler(seed=case).sample(shots).T.astype(np.uint8)
        checks = [close(ours.mean(1), theirs.mean(1), shots, "rates")]
        if noisy:
            m = ours.shape[0]
            xo = np.array([(ours[i] ^ ours[j]).mean() for i in range(m) for j in range(i)])
            xt = np.array([(theirs[i] ^ theirs[j]).mean() for i in range(m) for j in range(i)])
            checks.append(close(xo, xt, shots, "pair xors"))
            if circuit.num_detectors:
                do = np.asarray(qs.sample_detectors(text, shots, seed=case)[0]).T.astype(np.uint8)
                dt = circuit.compile_detector_sampler(seed=case).sample(shots).T.astype(np.uint8)
                checks.append(close(do.mean(1), dt.mean(1), shots, "detectors"))
        else:
            so = {r.tobytes() for r in ours.T}
            st = {r.tobytes() for r in theirs.T}
            checks.append((so == st, f"outcome sets {len(so)} vs {len(st)}"))
        bad = [what for ok, what in checks if not ok]
        if bad:
            failures += 1
            print(f"case {case} ({'noisy' if noisy else 'noiseless'}): {bad}\n{text}")
    print(f"{cases} circuits, {feedback_lines} classically controlled lines, {failures} failures")
    return failures


if __name__ == "__main__":
    sys.exit(1 if main(int(sys.argv[1]), int(sys.argv[2])) else 0)
