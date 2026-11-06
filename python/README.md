# qevedo-simulator

Python bindings of [Qevedo](https://github.com/qevedo/qevedo-simulator), a fast
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

Clifford circuits can carry Pauli noise channels (`x_error`, `y_error`,
`z_error`, `pauli_channel_1`, `depolarize1`, `depolarize2`,
`pauli_channel_2`, Stim's channels), sampled independently in every shot by the stabilizer backend.
`sample` returns the raw measurement records, measurement-major like Stim's
`FlipSimulator`, and `to_stim` writes the same circuit for Stim:

```python
from qevedo.simulator import Circuit, sample

c = Circuit(3)
for _ in range(10):                      # a small repetition code
    c.reset(1).x_error(1, 0.001)
    c.cx(0, 1).depolarize2(0, 1, 0.001).cx(2, 1).depolarize2(2, 1, 0.001)
    c.x_error(1, 0.001).measure(1, c.num_clbits)
bits = sample(c, 100_000, seed=1)        # NumPy bool array, (10, 100000)
print(c.to_stim().splitlines()[:3])      # ['R 1', 'X_ERROR(0.001) 1', 'CX 0 1']
```

Error-correction circuits written in Stim's format (text or a
`stim.Circuit`, for example from `stim.Circuit.generated`) run directly.
`sample_detectors` returns detection events and observable flips,
shot-major as Stim's detector sampler returns them and as decoders take
them; `from_stim` gives the `Circuit` with its detectors and observables
as lists of measurement indices, and `sample` takes Stim text too:

```python
import pymatching, stim
from qevedo.simulator import sample_detectors

circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=5, rounds=5,
                                 after_clifford_depolarization=0.003)
detectors, observables = sample_detectors(circuit, 100_000, seed=1)  # (100000, 120), (100000, 1)
matching = pymatching.Matching.from_detector_error_model(circuit.detector_error_model(decompose_errors=True))
logical_error_rate = (matching.decode_batch(detectors) != observables).any(axis=1).mean()
```

`bit_packed=True` returns uint8 arrays with eight detectors per byte
(detector `d` in bit `d % 8` of byte `d // 8`), Stim's packing and an
eighth of the memory, as decoders that take packed input expect;
`sample(..., bit_packed=True)` packs eight shots per byte. All of these
are written by Rust into the array's memory, without a copy in Python.

The reader takes Stim's Clifford gates, `M`/`MX`/`MY`/`MR`/`MRX`/`MRY`
with measurement-flip probabilities (`M(0.01)`) and inverted targets
(`M !3`), `R`/`RX`/`RY`, `MPAD`, Pauli product measurements (`MPP
X0*Y1*Z2`, `MPP(0.01) !Z3*Z4`), the Pauli channels including
`PAULI_CHANNEL_2`, correlated errors (`E`, `ELSE_CORRELATED_ERROR`),
`DETECTOR`, `OBSERVABLE_INCLUDE` and nested `REPEAT` blocks; anything else
(heralded errors, `MXX`/`MZZ`, classical control, ...) is an error naming
the line. A chain of correlated errors read from Stim appears in
`instructions()` as one `correlated_error` with each product's
unconditional probability; `to_stim()` writes it back as the chain.

`run` chooses the simulation method automatically:

| Backend | For | Scale on one workstation |
|---|---|---|
| `stabilizer` | Clifford circuits, with Pauli noise | tens of thousands of qubits |
| `near_clifford` | Clifford circuits with few non-Clifford gates, with Pauli noise | hundreds of qubits, ~30 active |
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

## From Qiskit

`pip install qevedo-simulator[qiskit]` (Qiskit 2.x). `qevedo.simulator.qiskit`
has a backend and the V2 primitives:

```python
from qiskit import QuantumCircuit, transpile
from qevedo.simulator.qiskit import QevedoEstimator, QevedoSampler, QevedoSimulator

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

backend = QevedoSimulator()              # BackendV2: any standard gate, no coupling map
measured = qc.measure_all(inplace=False)
counts = backend.run(transpile(measured, backend), shots=1000).result().get_counts()

bits = QevedoSampler(seed=1).run([measured], shots=1000).result()[0].data.meas    # BitArray
evs = QevedoEstimator().run([(qc, ["ZZ", "XX"])]).result()[0].data.evs            # [1. 1.]
```

- `QevedoSimulator` runs circuits with `run`, so it picks the method the same
  way and reports it in `result.results[i].metadata["method"]`. Options:
  `shots`, `seed_simulator`, `method` (Qevedo's or Aer's names), `precision`,
  `max_bond_dimension`, `truncation_threshold`, `threads`, `memory`.
  Qiskit's `BackendSamplerV2` and `BackendEstimatorV2` work on it too.
- `QevedoSampler` returns one `BitArray` per classical register. Measurements
  may be anywhere in the circuit, and resets are allowed.
- `QevedoEstimator` evaluates every observable exactly on the state vector up
  to `statevector_max_qubits` (24) qubits, and by Pauli propagation beyond
  that, up to 1024 qubits. Propagation drops small Pauli coefficients
  (`min_abs_coefficient`), and `data.truncated_one_norm` bounds the error.
- Every standard Qiskit gate runs. Gates without a kernel of their own (`ecr`,
  `iswap`, `ryy`, ...) run as their exact matrices, and others (QFT, MCX,
  `UnitaryGate`, `PauliEvolutionGate`, ...) through their definitions.
  Control flow and unbound parameters are rejected. Aer noise models are not
  read.
- `run`, `statevector` and `expectation` also take a `QuantumCircuit`
  directly. `statevector` keeps its global phase.

License: MIT.
