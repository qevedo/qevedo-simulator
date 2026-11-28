"""Dynamic circuits (mid-circuit measurements and `if_test` feed-forward)
through Qiskit: the Qevedo simulator's backend against Qiskit Aer's.

Both run the same `QuantumCircuit` through `backend.run`, with no transpiling
(Aer reads `if_else` natively; Qevedo turns `if_test` blocks of Pauli gates
into classical feedback and samples them on its stabilizer backend). Each
simulator uses its defaults: Aer's "automatic" method (it picks the
stabilizer method for these Clifford circuits and runs every shot again),
Qevedo's "auto". Times are the best of three runs and include converting the
circuit. Every result is checked against what the circuit must give:

- teleport: a state teleported along a line of qubits by one-bit
  teleportation (CZ, measure in X, X correction), then its preparation
  undone: the last qubit reads 0 in every shot;
- ghz: the constant-depth GHZ preparation of IBM's dynamic-circuit
  tutorials (parities of neighbouring data qubits measured on ancillas, then
  X on each data qubit for every odd parity before it, one `if_test` per
  bit): every data qubit reads the same;
- repetition: rounds of ZZ checks on data qubits in |+>, the ancillas reset
  by measurement and `if_test` X: every round reads the first round's
  syndrome.

    pip install "qevedo-simulator[qiskit]" qiskit-aer
    python benchmarks/bench_dynamic.py            # -> results/dynamic.csv
"""

import argparse
import csv
import os
import time

import qiskit
import qiskit_aer
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator

import qevedo.simulator
from qevedo.simulator.qiskit import QevedoSimulator


def teleport(n):
    qc = QuantumCircuit(n, n)
    qc.h(0)
    qc.s(0)
    for a in range(n - 1):
        b = a + 1
        qc.h(b)
        qc.cz(a, b)
        qc.h(a)
        qc.measure(a, a)
        # The wire holds X^m H |psi>: undo the X.
        with qc.if_test((qc.clbits[a], 1)):
            qc.x(b)
    # The last qubit holds H^(n-1) S H |0>.
    if (n - 1) % 2:
        qc.h(n - 1)
    qc.sdg(n - 1)
    qc.h(n - 1)
    qc.measure(n - 1, n - 1)

    def check(key):
        return key[0] == "0"

    return qc, check


def ghz(k):
    """k data qubits (even positions) and k - 1 ancillas between them."""
    qc = QuantumCircuit(QuantumRegister(2 * k - 1, "q"), ClassicalRegister(k - 1, "m"), ClassicalRegister(k, "d"))
    m, d = qc.cregs
    for j in range(k):
        qc.h(2 * j)
    for i in range(k - 1):
        qc.cx(2 * i, 2 * i + 1)
        qc.cx(2 * i + 2, 2 * i + 1)
    for i in range(k - 1):
        qc.measure(2 * i + 1, m[i])
    # Data qubit j is flipped by the XOR of the parities before it.
    for j in range(1, k):
        for i in range(j):
            with qc.if_test((m[i], 1)):
                qc.x(2 * j)
    for j in range(k):
        qc.measure(2 * j, d[j])

    def check(key):
        data = key.split(" ")[0]
        return len(set(data)) == 1

    return qc, check


def repetition(d, rounds):
    data = QuantumRegister(d, "data")
    anc = QuantumRegister(d - 1, "anc")
    regs = [ClassicalRegister(d - 1, f"r{r}") for r in range(rounds)]
    qc = QuantumCircuit(data, anc, *regs)
    qc.h(data)
    for r in range(rounds):
        for i in range(d - 1):
            qc.cx(data[i], anc[i])
            qc.cx(data[i + 1], anc[i])
        for i in range(d - 1):
            qc.measure(anc[i], regs[r][i])
            with qc.if_test((regs[r][i], 1)):
                qc.x(anc[i])

    def check(key):
        return len(set(key.split(" "))) == 1

    return qc, check


CASES = {
    "teleport 50": lambda: teleport(50),
    "teleport 200": lambda: teleport(200),
    "ghz 25 (49 qubits)": lambda: ghz(25),
    "ghz 100 (199 qubits)": lambda: ghz(100),
    "repetition d=25, 10 rounds": lambda: repetition(25, 10),
    "repetition d=50, 20 rounds": lambda: repetition(50, 20),
}


def best(f, repeats):
    times, out = [], None
    for _ in range(repeats):
        t = time.perf_counter()
        out = f()
        times.append(time.perf_counter() - t)
    return min(times), out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shots", type=int, nargs="+", default=[1000, 10_000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--only", help="run the cases whose name contains this")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results", "dynamic.csv"))
    args = parser.parse_args()
    simulators = {"qvd": QevedoSimulator(), "qiskit-aer": AerSimulator()}
    rows = []
    for name, build in CASES.items():
        if args.only and args.only not in name:
            continue
        qc, check = build()
        ifs = sum(1 for i in qc.data if i.operation.name == "if_else")
        for shots in args.shots:
            times = {}
            for sim, backend in simulators.items():
                seconds, result = best(lambda: backend.run(qc, shots=shots, seed_simulator=1).result(), args.repeats)
                counts = result.get_counts()
                ok = sum(counts.values()) == shots and all(check(k) for k in counts)
                method = result.results[0].metadata.get("method", "")
                times[sim] = seconds
                rows.append(
                    {
                        "simulator": sim,
                        "case": name,
                        "qubits": qc.num_qubits,
                        "if_tests": ifs,
                        "shots": shots,
                        "seconds": f"{seconds:.4f}",
                        "method": method,
                        "check": "pass" if ok else "FAIL",
                        "version": qevedo.simulator.__version__ if sim == "qvd" else qiskit_aer.__version__,
                    }
                )
                print(f"{name:<28} {shots:>6} shots  {sim:<10} {seconds:9.4f} s  {method:<11} {rows[-1]['check']}", flush=True)
            print(f"{'':<28} Aer / Qevedo: {times['qiskit-aer'] / times['qvd']:.1f}x", flush=True)
    if not args.only:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.out} (qiskit {qiskit.__version__}, qiskit-aer {qiskit_aer.__version__})")


if __name__ == "__main__":
    main()
