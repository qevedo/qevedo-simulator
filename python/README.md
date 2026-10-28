# qevedo-simulator

Python bindings of [qvd](https://github.com/qevedo/qevedo-simulator), a fast
quantum circuit simulator written in Rust. It installs as
`qevedo-simulator` and imports as `qevedo.simulator`.

```bash
pip install qevedo-simulator          # add [numpy] for NumPy state vectors
```

```python
from qevedo.simulator import Circuit, expectation, run, statevector

bell = Circuit(2).h(0).cx(0, 1).measure_all()
result = run(bell, shots=1000, seed=1)
print(result.counts, result.backend)  # {'00': ..., '11': ...} stabilizer

# OpenQASM 2 or 3 text works anywhere a circuit does.
result = run("""
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
bit[3] c;
h q[0];
cx q[0], q[1];
ccx q[0], q[1], q[2];
c = measure q;
""", shots=1000)

state = statevector(Circuit(3).h(0).cx(0, 1).t(2))  # NumPy complex128 array

# Expectation values by Pauli propagation, far beyond a state vector:
# a Pauli string, a {string: coefficient} dict or a Qiskit SparsePauliOp.
e = expectation(Circuit(2).h(0).cx(0, 1), {"Z0 Z1": 1.0, "X0 X1": 0.5})
print(e.value, e.truncated_one_norm)  # 1.5, and a bound on its error (0)
```

`run` chooses the simulation method automatically:

| Backend | For | Scale on one workstation |
|---|---|---|
| `stabilizer` | Clifford circuits | tens of thousands of qubits |
| `near_clifford` | Clifford circuits with few non-Clifford gates | hundreds of qubits, ~30 active |
| `statevector` | anything that fits in memory (AVX2 kernels, gate fusion) | ~32 qubits |
| `mps` | low-entanglement circuits | 100+ qubits, approximate beyond χ |

Pass `backend=` to force one. The result reports the method that ran, its
timings, and the MPS fidelity estimate or the near-Clifford peak active
qubits. Simulations release the GIL.

`expectation` propagates the observable backwards through the circuit
(Cliffords for free, small coefficients dropped), for up to 1024 qubits:
`<Z62>` of IBM's 127-qubit, 20-step kicked Ising circuit takes 0.25–9.6 s
at the thresholds in the main README's benchmark.

x86-64 wheels are built for AVX2 + FMA processors (2013–2015 or newer);
other x86-64 machines can build from source with a Rust toolchain.

License: MIT.
