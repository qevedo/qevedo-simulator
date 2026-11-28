"""The Qevedo simulator inside Qiskit: a backend and the V2 primitives.

With ``qevedo-simulator`` and Qiskit installed::

    from qiskit import QuantumCircuit, transpile
    from qevedo.simulator.qiskit import QevedoSimulator, QevedoEstimator, QevedoSampler

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    backend = QevedoSimulator()
    counts = backend.run(transpile(qc, backend), shots=1000).result().get_counts()

    sampler = QevedoSampler()
    bits = sampler.run([qc], shots=1000).result()[0].data.meas

    estimator = QevedoEstimator()
    ev = estimator.run([(qc.remove_final_measurements(inplace=False), "ZZ")]).result()[0].data.evs

The backend and the sampler run circuits with ``qevedo.simulator.run``, so
they choose the method the same way: the stabilizer backend for Clifford
circuits, the near-Clifford backend, the state vector, or a matrix product
state, and they keep mid-circuit measurements and resets. The estimator
computes exact expectation values from the state vector up to
``statevector_max_qubits`` qubits and by Pauli propagation beyond.

Every standard Qiskit gate is accepted; those the simulator has no kernel for (``ecr``,
``iswap``, ``ryy``, ...) run as their exact matrices, and other gates through
their definitions. Dynamic circuits run when their classical control is
feedback: an ``if_test`` (with or without ``else``) on one classical bit, or
on a one-bit register, whose bodies apply only Pauli gates (the corrections of
teleportation, measurement-based resets and IBM's feed-forward tutorials)
becomes ``Circuit.feedback``, so a Clifford dynamic circuit stays on the
stabilizer backend::

    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 1)):
        qc.x(1)

Other control flow (loops, ``switch``, conditions on several bits, bodies
with other gates) and unbound parameters are rejected. Qiskit Aer's noise
models are not read: noisy Clifford circuits can be built with the
simulator's own ``Circuit`` (``depolarize1``, ...).
"""

from __future__ import annotations

import cmath
import uuid
from collections.abc import Iterable

import numpy as np
from qiskit.circuit import ClassicalRegister, Clbit, ControlFlowOp, IfElseOp, QuantumCircuit
from qiskit.circuit.classical import expr
from qiskit.circuit.library import get_standard_gate_name_mapping
from qiskit.primitives import BaseEstimatorV2, BaseSamplerV2
from qiskit.primitives.containers import (
    BitArray,
    DataBin,
    PrimitiveResult,
    PubResult,
    SamplerPubResult,
)
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.primitives.primitive_job import PrimitiveJob
from qiskit.providers import BackendV2, JobStatus, JobV1, Options
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit.transpiler import Target

from qevedo.simulator import BACKENDS, Circuit, __version__
from qevedo.simulator import expectation as _expectation
from qevedo.simulator import run as _run
from qevedo.simulator import statevector as _qvd_statevector

__all__ = ["QevedoSimulator", "QevedoJob", "QevedoSampler", "QevedoEstimator", "from_qiskit"]

# Qiskit's gate names that Qevedo's `Circuit.gate` reads, with the same
# matrices (including the phases of u2, u3, cu and cu3), by Qevedo's name.
_NATIVE = {
    name: name
    for name in (
        "id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "sxdg",
        "rx", "ry", "rz", "p", "u", "u1", "u2", "u3",
        "cx", "cy", "cz", "ch", "csx", "swap", "cp", "cu1", "crx", "cry", "crz",
        "cu", "cu3", "rxx", "rzz", "ccx", "cswap",
    )
}  # fmt: skip
_NATIVE["c3sx"] = "c3sqrtx"

# Gates that run as their matrix rather than through their definitions.
_MAX_MATRIX_QUBITS = 4

# Pauli propagation takes Qevedo's gates except arbitrary unitaries on two or
# more qubits, which Qevedo uses for these: they go through their definitions.
_NOT_PROPAGATED = {"csx", "cu", "cu3", "c3sx"}

# Aer's method names, for code written against AerSimulator.
_METHOD_ALIASES = {"automatic": "auto", "matrix_product_state": "mps", "extended_stabilizer": "near_clifford"}


def _method(name: str) -> str:
    method = _METHOD_ALIASES.get(name, name)
    if method not in BACKENDS:
        raise ValueError(f"unknown method {name!r}; expected one of {', '.join(BACKENDS)}")
    return method


def from_qiskit(circuit: QuantumCircuit) -> tuple[Circuit, float]:
    """A Qiskit circuit as a ``qevedo.simulator.Circuit``, and its global phase.

    Qubit ``q`` and classical bit ``c`` keep their indices in the Qiskit
    circuit, so counts read the same way (classical bit 0 on the right).
    The phase is Qiskit's ``global_phase`` plus those of the definitions
    that were expanded; these circuits carry none, and only state vectors
    need it.
    """
    return _convert(circuit, propagation=False)


def _convert(circuit: QuantumCircuit, propagation: bool) -> tuple[Circuit, float]:
    """``from_qiskit``; with ``propagation``, only gates Pauli propagation takes
    (no matrices on two or more qubits)."""
    if circuit.parameters:
        names = ", ".join(sorted(p.name for p in circuit.parameters))
        raise ValueError(f"the circuit has unbound parameters ({names}); bind them first")
    out = Circuit(circuit.num_qubits, circuit.num_clbits)
    qubits = [circuit.find_bit(q).index for q in circuit.qubits]
    clbits = [circuit.find_bit(c).index for c in circuit.clbits]
    phase = _append(out, circuit, qubits, clbits, propagation)
    return out, phase


def _append(out: Circuit, circuit: QuantumCircuit, qubits: list[int], clbits: list[int], propagation: bool) -> float:
    phase = float(circuit.global_phase)
    qindex = {q: qubits[i] for i, q in enumerate(circuit.qubits)}
    cindex = {c: clbits[i] for i, c in enumerate(circuit.clbits)}
    for instruction in circuit.data:
        op = instruction.operation
        qs = [qindex[q] for q in instruction.qubits]
        cs = [cindex[c] for c in instruction.clbits]
        name = op.name
        if name in ("barrier", "delay"):
            continue
        if name == "measure":
            out.measure(qs[0], cs[0])
        elif name == "reset":
            out.reset(qs[0])
        elif name == "global_phase":
            phase += float(op.params[0])
        elif name == "box":
            phase += _append(out, op.blocks[0], qs, cs, propagation)
        elif name == "if_else" and not propagation:
            _append_if(out, op, qs, cindex)
        elif isinstance(op, ControlFlowOp):
            raise ValueError(f"control flow (`{name}`) is not supported by the Qevedo simulator")
        elif name in _NATIVE and not (propagation and name in _NOT_PROPAGATED):
            out.gate(_NATIVE[name], qs, [float(p) for p in op.params])
        elif (matrix := _matrix(op, 1 if propagation else _MAX_MATRIX_QUBITS)) is not None:
            out.unitary(matrix.tolist(), qs)
        elif op.definition is not None:
            phase += _append(out, op.definition, qs, cs, propagation)
        else:
            raise ValueError(f"the Qevedo simulator cannot run `{name}`: no matrix and no definition")
    return phase


def _append_if(out: Circuit, op, qubits: list[int], cindex: dict) -> None:
    """An ``if_test`` on one bit with Pauli bodies, as classical feedback.

    ``Circuit.feedback`` applies a Pauli when a bit reads 1, so the body
    taken on 0 is applied unconditionally and, when the bit reads 1, the
    product of both bodies on each qubit (taking the 0-body back and putting
    the 1-body on). Paulis multiply qubit by qubit up to a phase; that phase
    is conditioned on a classical bit, which only a measurement (an
    incoherent branch) or nothing (a fixed 0) wrote, so it is unobservable.
    The OpenQASM reader (``qasm.rs``, ``Parser::conditional``) does the same.
    """
    clbit, wanted = _condition_bit(op.condition, cindex)
    bodies = [_pauli_body(block, qubits) for block in op.blocks]
    then, otherwise = bodies[0], bodies[1] if len(bodies) > 1 else {}
    on_one, on_zero = (then, otherwise) if wanted else (otherwise, then)
    for qubit, bits in on_zero.items():
        if bits:
            out.gate(_PAULI_NAMES[bits].lower(), [qubit], [])
    for qubit in sorted(set(on_one) | set(on_zero)):
        bits = on_one.get(qubit, 0) ^ on_zero.get(qubit, 0)
        if bits:
            out.feedback(_PAULI_NAMES[bits], qubit, clbit)


# A Pauli by its bits (1: X, 2: Z), and the Paulis' matrices.
_PAULI_NAMES = {1: "X", 2: "Z", 3: "Y"}
_PAULI_MATRICES = {
    0: np.eye(2),
    1: np.array([[0, 1], [1, 0]]),
    2: np.array([[1, 0], [0, -1]]),
    3: np.array([[0, -1j], [1j, 0]]),
}


def _condition_bit(condition, cindex: dict) -> tuple[int, bool]:
    """The classical bit an ``if_test`` reads, and the value of it that
    selects the ``if`` body: ``(clbit, value)``, ``(one-bit register,
    value)``, or an expression over one bit (``expr.lift(clbit)``,
    ``expr.logic_not``, ``expr.bit_not``, ``expr.equal``,
    ``expr.not_equal`` against 0, 1, true or false)."""
    unsupported = ValueError(
        f"the Qevedo simulator runs `if_test` on one classical bit (as feedback); not on {condition}"
    )

    def bit(target) -> int:
        if isinstance(target, Clbit):
            return cindex[target]
        if isinstance(target, ClassicalRegister) and len(target) == 1:
            return cindex[target[0]]
        raise unsupported

    def value(v) -> bool:
        if isinstance(v, expr.Value):
            v = v.value
        if v is True or v is False or v in (0, 1):
            return bool(v)
        raise unsupported

    def read(e) -> tuple[int, bool]:
        """The bit, and the value of it for which `e` is true."""
        if isinstance(e, expr.Var):
            return bit(e.var), True
        if isinstance(e, expr.Cast):
            return read(e.operand)
        if isinstance(e, expr.Unary) and e.op in (expr.Unary.Op.LOGIC_NOT, expr.Unary.Op.BIT_NOT):
            clbit, wanted = read(e.operand)
            return clbit, not wanted
        if isinstance(e, expr.Binary) and e.op in (expr.Binary.Op.EQUAL, expr.Binary.Op.NOT_EQUAL):
            left, right = (e.left, e.right) if isinstance(e.right, expr.Value) else (e.right, e.left)
            clbit, wanted = read(left)
            # `left == 1` is true where `left` is; `left == 0` where it isn't.
            equal = e.op == expr.Binary.Op.EQUAL
            return clbit, wanted if value(right) == equal else not wanted
        raise unsupported

    if isinstance(condition, tuple):
        target, v = condition
        return bit(target), value(v)
    return read(condition)


def _pauli_body(block: QuantumCircuit, qubits: list[int]) -> dict[int, int]:
    """The Pauli (bits 1: X, 2: Z) each qubit gets from an ``if_test``
    body that applies only Pauli gates, up to a phase. Any one-qubit gate
    whose matrix is a Pauli times a phase counts (a transpiler may write
    ``x`` as ``rx(pi)`` or ``u(pi, 0, pi)``)."""
    index = {q: qubits[i] for i, q in enumerate(block.qubits)}
    body: dict[int, int] = {}
    for instruction in block.data:
        op = instruction.operation
        if op.name in ("barrier", "delay", "global_phase"):
            continue
        bits = None
        if op.num_qubits == 1 and not isinstance(op, ControlFlowOp):
            matrix = _matrix(op, 1)
            if matrix is not None:
                for k, pauli in _PAULI_MATRICES.items():
                    # matrix = e^{i phi} pauli: |tr(pauli† matrix)| = 2.
                    if abs(abs(np.trace(pauli.conj().T @ matrix)) - 2) < 1e-9:
                        bits = k
                        break
        if bits is None:
            raise ValueError(
                f"`{op.name}` inside `if_test` is not supported: the Qevedo simulator runs `if_test` "
                "bodies of Pauli gates (x, y, z, id) as classical feedback"
            )
        qubit = index[instruction.qubits[0]]
        body[qubit] = body.get(qubit, 0) ^ bits
    return body


def _matrix(op, max_qubits: int) -> np.ndarray | None:
    if op.num_clbits or not 0 < op.num_qubits <= max_qubits or not hasattr(op, "to_matrix"):
        return None
    try:
        return np.asarray(op.to_matrix(), dtype=complex)
    except Exception:  # no matrix (e.g. an opaque gate): try the definition
        return None


def _statevector(circuit: QuantumCircuit, threads) -> np.ndarray:
    converted, phase = from_qiskit(circuit)
    state = np.asarray(_qvd_statevector(converted, threads=threads))
    return state * cmath.exp(1j * phase) if phase else state


def _seed(rng: np.random.Generator) -> int:
    return int(rng.integers(2**63))


def _shots(counts: dict[str, int], num_clbits: int, rng: np.random.Generator) -> np.ndarray:
    """Counts as a ``(shots, num_clbits)`` bool array, column ``c`` being
    classical bit ``c``, in random order.

    The simulator returns counts, not the order of the shots; the shots were drawn
    independently, so a uniformly random order of them has the same
    distribution as the order they were drawn in.
    """
    keys = list(counts)
    if num_clbits == 0:
        return np.zeros((sum(counts.values()), 0), dtype=bool)
    rows = np.array([[bit == "1" for bit in reversed(key)] for key in keys], dtype=bool)
    shots = np.repeat(rows, [counts[k] for k in keys], axis=0)
    return shots[rng.permutation(len(shots))]


class QevedoJob(JobV1):
    """A finished run: :class:`QevedoSimulator` simulates when ``run`` is called."""

    def __init__(self, backend, job_id: str, result: Result):
        super().__init__(backend, job_id)
        self._result = result

    def submit(self):
        raise RuntimeError("QevedoSimulator jobs run when they are created")

    def result(self, timeout: float | None = None) -> Result:
        return self._result

    def status(self) -> JobStatus:
        return JobStatus.DONE


class QevedoSimulator(BackendV2):
    """The Qevedo simulator as a Qiskit backend: an ideal simulator of any standard gate on
    up to ``num_qubits`` qubits, without coupling constraints.

    Options (``backend.set_options(...)`` or ``run(..., **options)``):
    ``shots``, ``seed_simulator``, ``method`` (``"auto"``,
    ``"statevector"``, ``"stabilizer"``, ``"mps"``, ``"near_clifford"``, or
    Aer's names), ``precision`` (``"single"`` or ``"double"``, for the
    state vector), ``max_bond_dimension`` and ``truncation_threshold`` (MPS),
    ``threads`` and ``memory`` (per-shot results in ``get_memory()``).
    How many qubits fit depends on the method: about 32 for the state
    vector, far more for Clifford and low-entanglement circuits.
    """

    def __init__(self, num_qubits: int = 1024, **options):
        super().__init__(
            name="qevedo_simulator",
            description="Qevedo, a fast quantum circuit simulator (qevedo-simulator)",
            backend_version=__version__,
        )
        names = [n for n in get_standard_gate_name_mapping() if n != "global_phase"]
        self._target = Target.from_configuration(names, num_qubits=num_qubits)
        # `transpile` keeps `if_test` blocks only if the target has them; the
        # ones that are feedback run, the others are rejected by `run`.
        self._target.add_instruction(IfElseOp, name="if_else")
        self._target.description = "Qevedo ideal simulator"
        if options:
            self.set_options(**options)

    @classmethod
    def _default_options(cls) -> Options:
        return Options(
            shots=1024,
            seed_simulator=None,
            method="auto",
            precision="single",
            max_bond_dimension=256,
            truncation_threshold=1e-16,
            threads=None,
            memory=False,
        )

    @property
    def target(self) -> Target:
        return self._target

    @property
    def max_circuits(self) -> None:
        return None

    def run(self, run_input, **options) -> QevedoJob:
        circuits = [run_input] if isinstance(run_input, QuantumCircuit) else list(run_input)
        unknown = set(options) - set(self.options.__dict__)
        if unknown:
            raise ValueError(f"unknown options: {', '.join(sorted(unknown))}")
        opts = {**self.options.__dict__, **options}
        method = _method(opts["method"])
        rng = np.random.default_rng(opts["seed_simulator"])
        results = [self._run_one(c, opts, method, rng) for c in circuits]
        job_id = str(uuid.uuid4())
        result = Result(
            backend_name=self.name,
            backend_version=self.backend_version,
            job_id=job_id,
            success=True,
            results=results,
        )
        return QevedoJob(self, job_id, result)

    def _run_one(self, circuit: QuantumCircuit, opts: dict, method: str, rng) -> ExperimentResult:
        converted, _ = from_qiskit(circuit)
        seed = _seed(rng)
        shots = int(opts["shots"])
        out = _run(
            converted,
            shots,
            backend=method,
            seed=seed,
            precision=opts["precision"],
            max_bond_dimension=opts["max_bond_dimension"],
            truncation_threshold=opts["truncation_threshold"],
            threads=opts["threads"],
        )
        # No classical bits: no counts, as Qiskit's BasicSimulator.
        counts = {hex(int(k, 2)): v for k, v in out.counts.items() if k}
        data = {"counts": counts}
        if opts["memory"]:
            bits = _shots(out.counts, circuit.num_clbits, rng)
            weights = 1 << np.arange(circuit.num_clbits, dtype=object)
            data["memory"] = [hex(int(row.astype(object) @ weights)) if len(row) else "0x0" for row in bits]
        header = {
            "name": circuit.name,
            "memory_slots": circuit.num_clbits,
            "n_qubits": circuit.num_qubits,
            "metadata": circuit.metadata,
        }
        # Spaces between registers in get_counts(), when the registers are
        # the classical bits in order (no loose or shared bits).
        if circuit.cregs and [c for r in circuit.cregs for c in r] == list(circuit.clbits):
            header["creg_sizes"] = [[r.name, r.size] for r in circuit.cregs]
        metadata = {"method": out.backend, "gate_seconds": out.gate_seconds, "measure_seconds": out.measure_seconds}
        if out.fidelity is not None:
            metadata["fidelity"] = out.fidelity
            metadata["max_bond_dimension"] = out.max_bond_dimension
        if out.peak_active_qubits is not None:
            metadata["peak_active_qubits"] = out.peak_active_qubits
        return ExperimentResult(
            shots=shots,
            success=True,
            data=ExperimentResultData(**data),
            header=header,
            seed=seed,
            status="DONE",
            metadata=metadata,
            time_taken=out.gate_seconds + out.measure_seconds,
        )


class QevedoSampler(BaseSamplerV2):
    """Qiskit's ``SamplerV2`` on the Qevedo simulator.

    Each classical register becomes a ``BitArray`` in the result's data, as
    with Qiskit's reference sampler; unlike it, measurements may be anywhere
    in the circuit, and resets are allowed. ``options`` go to
    ``qevedo.simulator.run`` (``method``, ``precision``,
    ``max_bond_dimension``, ``truncation_threshold``, ``threads``).
    """

    def __init__(self, *, default_shots: int = 1024, seed=None, options: dict | None = None):
        self._default_shots = default_shots
        self._seed = seed
        self._options = dict(options or {})
        if "method" in self._options:
            self._options["backend"] = _method(self._options.pop("method"))

    @property
    def default_shots(self) -> int:
        return self._default_shots

    @property
    def seed(self):
        return self._seed

    def run(self, pubs: Iterable, *, shots: int | None = None) -> PrimitiveJob[PrimitiveResult[SamplerPubResult]]:
        if shots is None:
            shots = self._default_shots
        coerced = [SamplerPub.coerce(pub, shots) for pub in pubs]
        job = PrimitiveJob(self._run, coerced)
        job._submit()
        return job

    def _run(self, pubs: list[SamplerPub]) -> PrimitiveResult[SamplerPubResult]:
        rng = np.random.default_rng(self._seed)
        return PrimitiveResult([self._run_pub(pub, rng) for pub in pubs], metadata={"version": 2})

    def _run_pub(self, pub: SamplerPub, rng) -> SamplerPubResult:
        circuit = pub.circuit
        bound = pub.parameter_values.bind_all(circuit)
        registers = [(r.name, [circuit.find_bit(c).index for c in r]) for r in circuit.cregs]
        arrays = {name: np.zeros(pub.shape + (pub.shots, (len(bits) + 7) // 8), dtype=np.uint8) for name, bits in registers}
        methods = set()
        for index in np.ndindex(*pub.shape):
            converted, _ = from_qiskit(bound[index])
            out = _run(converted, pub.shots, seed=_seed(rng), **self._options)
            methods.add(out.backend)
            shots = _shots(out.counts, circuit.num_clbits, rng)
            for name, bits in registers:
                arrays[name][index] = BitArray.from_bool_array(shots[:, bits], order="little").array
        data = {name: BitArray(arrays[name], len(bits)) for name, bits in registers}
        metadata = {"shots": pub.shots, "circuit_metadata": circuit.metadata, "methods": sorted(methods)}
        return SamplerPubResult(DataBin(**data, shape=pub.shape), metadata=metadata)


class QevedoEstimator(BaseEstimatorV2):
    """Qiskit's ``EstimatorV2`` on the Qevedo simulator.

    Circuits of up to ``statevector_max_qubits`` qubits are simulated once
    per parameter set, and every observable is evaluated exactly on the
    state vector. Larger circuits use Pauli propagation
    (``qevedo.simulator.expectation``), for up to 1024 qubits: Clifford
    gates are exact, and Pauli strings with coefficients below
    ``min_abs_coefficient`` are dropped (at most ``max_terms`` kept); the
    sum of what was dropped, a bound on the error, is in the result's
    ``data.truncated_one_norm`` (zero for exact values).

    ``method`` forces one: ``"statevector"`` or ``"pauli_propagation"``.
    With a nonzero ``precision``, Gaussian noise of that standard deviation
    is added to each value, as Qiskit's reference estimator does.
    Measurements at the end of a circuit are ignored.
    """

    def __init__(
        self,
        *,
        default_precision: float = 0.0,
        seed=None,
        method: str = "auto",
        statevector_max_qubits: int = 24,
        min_abs_coefficient: float = 1e-5,
        max_terms: int | None = None,
        threads=None,
    ):
        if method not in ("auto", "statevector", "pauli_propagation"):
            raise ValueError("method must be 'auto', 'statevector' or 'pauli_propagation'")
        self._default_precision = default_precision
        self._seed = seed
        self._method = method
        self._statevector_max_qubits = statevector_max_qubits
        self._min_abs_coefficient = min_abs_coefficient
        self._max_terms = max_terms
        self._threads = threads

    @property
    def default_precision(self) -> float:
        return self._default_precision

    @property
    def seed(self):
        return self._seed

    def run(self, pubs: Iterable, *, precision: float | None = None) -> PrimitiveJob[PrimitiveResult[PubResult]]:
        if precision is None:
            precision = self._default_precision
        coerced = [EstimatorPub.coerce(pub, precision) for pub in pubs]
        job = PrimitiveJob(self._run, coerced)
        job._submit()
        return job

    def _run(self, pubs: list[EstimatorPub]) -> PrimitiveResult[PubResult]:
        rng = np.random.default_rng(self._seed)
        return PrimitiveResult([self._run_pub(pub, rng) for pub in pubs], metadata={"version": 2})

    def _pick(self, num_qubits: int) -> str:
        if self._method != "auto":
            return self._method
        return "statevector" if num_qubits <= self._statevector_max_qubits else "pauli_propagation"

    def _run_pub(self, pub: EstimatorPub, rng) -> PubResult:
        circuit = pub.circuit.remove_final_measurements(inplace=False)
        method = self._pick(circuit.num_qubits)
        bound = pub.parameter_values.bind_all(circuit)
        bc_circuits, bc_obs = np.broadcast_arrays(bound, pub.observables)
        evs = np.zeros(bc_circuits.shape, dtype=np.float64)
        bounds = np.zeros(bc_circuits.shape, dtype=np.float64)
        states: dict[int, Statevector] = {}
        converted: dict[int, Circuit] = {}
        for index in np.ndindex(*bc_circuits.shape):
            qc = bc_circuits[index]
            terms = bc_obs[index]
            if method == "statevector":
                # Broadcasting repeats the same bound circuit object.
                if id(qc) not in states:
                    states[id(qc)] = Statevector(_statevector(qc, self._threads))
                op = SparsePauliOp(list(terms), list(terms.values()))
                value = float(np.real(states[id(qc)].expectation_value(op)))
            else:
                if id(qc) not in converted:
                    converted[id(qc)] = _convert(qc, propagation=True)[0]
                out = _expectation(
                    converted[id(qc)],
                    terms,
                    min_abs_coefficient=self._min_abs_coefficient,
                    max_terms=self._max_terms,
                    threads=self._threads,
                )
                value, bounds[index] = out.value, out.truncated_one_norm
            if pub.precision:
                value = rng.normal(value, pub.precision)
            evs[index] = value
        data = DataBin(evs=evs, stds=np.zeros_like(evs), truncated_one_norm=bounds, shape=evs.shape)
        metadata = {"target_precision": pub.precision, "circuit_metadata": pub.circuit.metadata, "method": method}
        return PubResult(data, metadata=metadata)

