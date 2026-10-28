"""Compute <0|U† P U|0> for an OpenQASM 2 circuit with IBM's pauli-prop, for timing.

    python benchmarks/compare_pauliprop.py circuit.qasm Z62 [--atol 1e-5] [--max-terms 10000000]

pauli-prop (pip install pauli-prop) is IBM's Rust-accelerated Pauli
propagation package. `propagate_through_circuit` moves the Clifford gates to
the front of the circuit (`evolve_through_cliffords`), propagates the
observable through the remaining Pauli rotations in the Heisenberg picture,
dropping coefficients below `atol` and keeping at most `max_terms` terms
(it preallocates buffers for `max_terms`), and evolves the result through
the Clifford. The expectation on |0...0> is then the sum of the
coefficients of the terms with no X or Y factor.

The observable is a sparse Pauli string, `Z62` or `"X37 Y75 Z38"`.
"""

import argparse
import re
import time


def sparse_label(text, n):
    label = ["I"] * n
    for pauli, qubit in re.findall(r"([IXYZ])\s*(\d+)", text.upper()):
        label[int(qubit)] = pauli
    return "".join(reversed(label))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("qasm")
    parser.add_argument("observable")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--max-terms", type=int, default=10_000_000)
    args = parser.parse_args()

    import numpy as np
    from pauli_prop import (
        circuit_to_rotation_gates,
        evolve_through_cliffords,
        propagate_through_rotation_gates,
    )
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp

    circuit = QuantumCircuit.from_qasm_file(args.qasm)
    observable = SparsePauliOp(sparse_label(args.observable, circuit.num_qubits))
    # The steps of pauli-prop's propagate_through_circuit, timed separately.
    start = time.perf_counter()
    clifford, rotations = evolve_through_cliffords(circuit)
    rotation_gates = circuit_to_rotation_gates(rotations)
    prepared = time.perf_counter()
    evolved, truncated = propagate_through_rotation_gates(
        observable, rotation_gates, args.max_terms, args.atol, "h"
    )
    propagated = time.perf_counter()
    evolved = SparsePauliOp(
        evolved.paulis.evolve(clifford, frame="h"),
        evolved.coeffs.copy(),
        copy=False,
        ignore_pauli_phase=False,
    )
    diagonal = ~np.any(evolved.paulis.x, axis=1)
    # Coefficients carry the Paulis' signs (phases are folded in).
    phases = (-1j) ** evolved.paulis.phase
    value = float(np.real(np.sum(evolved.coeffs[diagonal] * phases[diagonal])))
    wall = time.perf_counter() - start
    print(
        f"pauli-prop: <{args.observable}> = {value:.6f} (truncated one-norm {truncated:.3e}), "
        f"{len(evolved)} terms, wall {wall:.3f} s (Clifford preprocessing {prepared - start:.3f} s, "
        f"propagation {propagated - prepared:.3f} s)"
    )
    if len(evolved) >= args.max_terms:
        print("pauli-prop: hit --max-terms; raise it for an untruncated comparison")


if __name__ == "__main__":
    main()
