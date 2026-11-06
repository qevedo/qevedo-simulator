"""The Qevedo simulator through Qiskit (qevedo.simulator.qiskit) against Qiskit Aer.

Both run the same transpiled circuits through Qiskit's own interfaces
(backend.run and EstimatorV2), so the times include converting the circuit.

    pip install "qevedo-simulator[qiskit]" qiskit-aer
    python benchmarks/bench_qiskit.py
"""

import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT, quantum_volume
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as AerEstimator

from qevedo.simulator.qiskit import QevedoSimulator, QevedoEstimator, from_qiskit


def best(f, k=3):
    times = []
    for _ in range(k):
        t = time.perf_counter()
        f()
        times.append(time.perf_counter() - t)
    return min(times)


def row(name, tq, ta, extra=""):
    print(f"{name:<34} Qevedo {tq:8.3f} s   Aer {ta:8.3f} s   {ta / tq:7.2f}x {extra}")


qevedo, aer = QevedoSimulator(), AerSimulator()
for n in (20, 24, 26, 28):
    qc = transpile(quantum_volume(n, seed=1), basis_gates=["u", "cx"], optimization_level=0)
    qc.measure_all()
    convert = best(lambda: from_qiskit(qc))
    tq = best(lambda: qevedo.run(qc, shots=1000, seed_simulator=1).result())
    ta = best(lambda: aer.run(qc, shots=1000, seed_simulator=1).result())
    row(f"Quantum Volume {n}, 1000 shots", tq, ta, f"(conversion {convert * 1e3:.1f} ms)")
for n in (24, 28):
    qc = transpile(QFT(n), basis_gates=["u", "cx"], optimization_level=0)
    qc.h(range(n))
    qc.measure_all()
    row(f"QFT {n}, 1000 shots", best(lambda: qevedo.run(qc, shots=1000).result()), best(lambda: aer.run(qc, shots=1000).result()))

# Aer's stabilizer method simulates every shot again when there are
# measurements; Qevedo samples Pauli frames off one reference shot.
n = 500
qc = QuantumCircuit(n)
qc.h(0)
for q in range(n - 1):
    qc.cx(q, q + 1)
for q in range(0, n, 3):
    qc.s(q)
    qc.h(q)
qc.measure_all()
row(
    f"Clifford {n} qubits, 10k shots",
    best(lambda: qevedo.run(qc, shots=10_000).result(), 2),
    best(lambda: AerSimulator(method="stabilizer").run(qc, shots=10_000).result(), 1),
)

for n in (16, 20, 22):
    qc = transpile(quantum_volume(n, seed=2), basis_gates=["u", "cx"], optimization_level=0)
    obs = [SparsePauliOp("".join(np.random.default_rng(i).choice(list("IXYZ"), n))) for i in range(50)]
    estimator = AerEstimator(options={"backend_options": {"method": "statevector"}})
    a = QevedoEstimator().run([(qc, obs)]).result()[0].data.evs
    b = estimator.run([(qc, obs)]).result()[0].data.evs
    row(
        f"Estimator QV {n}, 50 observables",
        best(lambda: QevedoEstimator().run([(qc, obs)]).result()),
        best(lambda: estimator.run([(qc, obs)]).result()),
        f"(max |difference| {np.abs(a - b).max():.0e})",
    )
