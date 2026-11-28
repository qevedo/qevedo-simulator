"""The Qiskit backend and primitives (qevedo.simulator.qiskit)."""

import math

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile  # noqa: E402
from qiskit import qasm3  # noqa: E402
from qiskit.circuit import Clbit, Parameter  # noqa: E402
from qiskit.circuit.classical import expr  # noqa: E402
from qiskit.circuit.library import (  # noqa: E402
    MCXGate,
    PauliEvolutionGate,
    QFTGate,
    UnitaryGate,
    get_standard_gate_name_mapping,
)
from qiskit.primitives import (  # noqa: E402
    BackendEstimatorV2,
    BackendSamplerV2,
    StatevectorEstimator,
)
from qiskit.quantum_info import SparsePauliOp, Statevector, random_unitary  # noqa: E402

from qevedo.simulator import expectation, run, statevector  # noqa: E402
from qevedo.simulator.qiskit import QevedoSimulator, QevedoEstimator, QevedoSampler, from_qiskit  # noqa: E402

GATES = {n: g for n, g in get_standard_gate_name_mapping().items() if n not in ("measure", "reset", "delay")}


def random_circuit(num_qubits, depth, rng, names=None):
    names = sorted(names or GATES)
    qc = QuantumCircuit(num_qubits)
    for _ in range(depth):
        gate = GATES[str(rng.choice(names))]
        if gate.num_qubits > num_qubits:
            continue
        params = list(rng.uniform(-math.pi, math.pi, len(gate.params)))
        op = gate.base_class(*params) if params else gate.base_class()
        qubits = [int(q) for q in rng.choice(num_qubits, gate.num_qubits, replace=False)]
        qc.append(op, qubits)
    return qc


@pytest.mark.parametrize("name", sorted(GATES))
def test_every_standard_gate_matches_qiskit_with_phase(name):
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    gate = GATES[name]
    n = max(gate.num_qubits, 1) + 1
    for _ in range(3):
        qc = QuantumCircuit(n)
        for q in range(n):  # a generic input state
            qc.u(*rng.uniform(-math.pi, math.pi, 3), q)
        params = list(rng.uniform(-math.pi, math.pi, len(gate.params)))
        op = gate.base_class(*params) if params else gate.base_class()
        qubits = [int(q) for q in rng.permutation(n)[: gate.num_qubits]]
        qc.append(op, qubits)
        expected = Statevector(qc).data
        np.testing.assert_allclose(statevector(qc), expected, atol=1e-10)



@pytest.mark.parametrize("name", sorted(GATES))
def test_every_standard_gate_by_pauli_propagation(name):
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    gate = GATES[name]
    n = max(gate.num_qubits, 1) + 1
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.u(*rng.uniform(-math.pi, math.pi, 3), q)
    params = list(rng.uniform(-math.pi, math.pi, len(gate.params)))
    qc.append(gate.base_class(*params) if params else gate.base_class(), list(range(gate.num_qubits)))
    labels = ["".join(rng.choice(list("IXYZ"), n)) for _ in range(6)]
    observable = SparsePauliOp(labels, rng.uniform(-1, 1, 6))
    exact = float(np.real(Statevector(qc).expectation_value(observable)))
    assert expectation(qc, observable, min_abs_coefficient=0).value == pytest.approx(exact, abs=1e-9)

def test_composite_gates_through_definitions():
    rng = np.random.default_rng(7)
    qc = QuantumCircuit(6, global_phase=0.3)
    qc.h(range(6))
    qc.append(QFTGate(5), [5, 0, 2, 1, 3])
    qc.append(MCXGate(5), [0, 1, 2, 3, 4, 5])
    qc.append(UnitaryGate(random_unitary(8, seed=1)), [4, 1, 3])
    qc.append(PauliEvolutionGate(SparsePauliOp(["XXYZI", "ZZIIX"], [0.4, -0.7]), time=0.9), range(5))
    sub = random_circuit(3, 12, rng)
    qc.append(sub.to_gate(label="block"), [2, 5, 0])
    np.testing.assert_allclose(statevector(qc), Statevector(qc).data, atol=1e-9)


def test_random_circuits_match_qiskit():
    rng = np.random.default_rng(11)
    for n in (3, 5, 7):
        qc = random_circuit(n, 60, rng)
        np.testing.assert_allclose(statevector(qc), Statevector(qc).data, atol=1e-9)


def test_rejects_unbound_parameters_and_control_flow():
    qc = QuantumCircuit(1)
    qc.rx(Parameter("theta"), 0)
    with pytest.raises(ValueError, match="unbound parameters"):
        from_qiskit(qc)
    # `if_test` with Pauli bodies is feedback (below); other bodies, other
    # conditions and other control flow are rejected.
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 1)):
        qc.h(0)
    with pytest.raises(ValueError, match="`h` inside `if_test`"):
        from_qiskit(qc)
    qc = QuantumCircuit(1, 2)
    with qc.if_test((qc.cregs[0], 3)):
        qc.x(0)
    with pytest.raises(ValueError, match="one classical bit"):
        from_qiskit(qc)
    qc = QuantumCircuit(1, 1)
    with qc.for_loop(range(2)):
        qc.x(0)
    with pytest.raises(ValueError, match="control flow"):
        from_qiskit(qc)


def test_box_is_inlined():
    qc = QuantumCircuit(2)
    with qc.box():
        qc.h(0)
        qc.cx(0, 1)
    expected = QuantumCircuit(2)
    expected.h(0)
    expected.cx(0, 1)
    np.testing.assert_allclose(statevector(qc), Statevector(expected).data, atol=1e-12)


def total_variation(counts, probabilities):
    shots = sum(counts.values())
    keys = set(counts) | set(probabilities)
    return 0.5 * sum(abs(counts.get(k, 0) / shots - probabilities.get(k, 0.0)) for k in keys)


def test_backend_counts_follow_the_distribution():
    rng = np.random.default_rng(3)
    qc = random_circuit(5, 40, rng)
    exact = Statevector(qc).probabilities_dict()
    qc.measure_all()
    backend = QevedoSimulator()
    compiled = transpile(qc, backend, optimization_level=0)
    assert compiled.num_qubits == 5
    counts = backend.run(compiled, shots=40_000, seed_simulator=5).result().get_counts()
    assert total_variation(counts, exact) < 0.02


def test_backend_registers_memory_seeds_and_metadata():
    q = QuantumRegister(3)
    a = ClassicalRegister(1, "a")
    c = ClassicalRegister(2, "c")
    qc = QuantumCircuit(q, a, c, name="mid")
    qc.x(0)
    qc.measure(0, a[0])
    qc.reset(0)  # mid-circuit measurement and reset
    qc.h(1)
    qc.measure([0, 1], c)
    backend = QevedoSimulator(seed_simulator=9)
    result = backend.run(qc, shots=500, memory=True).result()
    counts = result.get_counts("mid")
    assert set(counts) <= {"00 1", "10 1"}
    assert sum(counts.values()) == 500
    memory = result.get_memory()
    assert len(memory) == 500
    assert {k: memory.count(k) for k in set(memory)} == counts
    assert result.results[0].metadata["method"] == "stabilizer"
    again = backend.run(qc, shots=500, memory=True).result()
    assert again.get_memory() == memory  # the same seed
    assert backend.run(qc, shots=500, seed_simulator=10).result().get_counts() != counts


def test_backend_several_circuits_methods_and_options():
    bell = QuantumCircuit(2)
    bell.h(0)
    bell.cx(0, 1)
    bell.measure_all()
    t = QuantumCircuit(2)
    t.h(0)
    t.t(0)
    t.h(0)
    t.measure_all()
    backend = QevedoSimulator()
    result = backend.run([bell, t], shots=2000, seed_simulator=1, method="matrix_product_state").result()
    assert set(result.get_counts(0)) == {"00", "11"}
    assert abs(result.get_counts(1).get("00", 0) / 2000 - math.cos(math.pi / 8) ** 2) < 0.04
    assert all(r.metadata["method"] == "mps" for r in result.results)
    with pytest.raises(ValueError, match="unknown options"):
        backend.run(bell, shot=10)
    with pytest.raises(ValueError, match="unknown method"):
        backend.run(bell, method="density_matrix")


def test_backend_no_measurements():
    qc = QuantumCircuit(2)
    qc.h(0)
    assert QevedoSimulator().run(qc, shots=10).result().get_counts() == {}  # as BasicSimulator


def test_backend_large_clifford_circuit():
    n = 300
    qc = QuantumCircuit(n)
    qc.h(0)
    for q in range(n - 1):
        qc.cx(q, q + 1)
    qc.measure_all()
    counts = QevedoSimulator().run(qc, shots=200, seed_simulator=2).result().get_counts()
    assert set(counts) <= {"0" * n, "1" * n} and len(counts) == 2


def test_qiskit_backend_primitives_run_on_the_backend():
    backend = QevedoSimulator(seed_simulator=4)
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    measured = qc.measure_all(inplace=False)
    bits = BackendSamplerV2(backend=backend).run([measured], shots=1000).result()[0].data.meas
    assert set(bits.get_counts()) == {"00", "11"}
    ev = BackendEstimatorV2(backend=backend).run([(qc, "ZZ")]).result()[0].data.evs
    assert abs(float(ev) - 1.0) < 1e-12


def test_sampler_parameters_registers_and_seed():
    theta = Parameter("theta")
    qc = QuantumCircuit(QuantumRegister(2), ClassicalRegister(1, "x"), ClassicalRegister(1, "y"))
    qc.ry(theta, 0)
    qc.measure(0, 0)
    qc.reset(0)
    qc.x(1)
    qc.measure(1, 1)
    values = np.array([[0.0], [math.pi], [math.pi / 2]])
    sampler = QevedoSampler(seed=1)
    result = sampler.run([(qc, values)], shots=4000).result()[0]
    assert result.data.x.shape == (3,) and result.data.x.num_shots == 4000
    ones = [result.data.x[i].get_counts().get("1", 0) / 4000 for i in range(3)]
    assert ones[0] == 0 and ones[1] == 1 and abs(ones[2] - 0.5) < 0.04
    assert all(result.data.y[i].get_counts() == {"1": 4000} for i in range(3))
    again = QevedoSampler(seed=1).run([(qc, values)], shots=4000).result()[0]
    assert np.array_equal(again.data.x.array, result.data.x.array)


def test_sampler_matches_distribution_and_shuffles_shots():
    rng = np.random.default_rng(8)
    qc = random_circuit(4, 30, rng)
    exact = Statevector(qc).probabilities_dict()
    qc.measure_all()
    bits = QevedoSampler(seed=2).run([qc], shots=20_000).result()[0].data.meas
    assert total_variation(bits.get_counts(), exact) < 0.03
    strings = bits.get_bitstrings()
    assert strings[:50] != sorted(strings[:50])  # not grouped by outcome


def test_estimator_matches_qiskit_reference():
    a, b = Parameter("a"), Parameter("b")
    qc = QuantumCircuit(4)
    qc.h(range(4))
    qc.rzz(a, 0, 1)
    qc.rx(b, 2)
    qc.cx(2, 3)
    qc.ryy(a + b, 1, 3)
    qc.append(QFTGate(3), [0, 2, 3])
    observables = [[SparsePauliOp(["ZZII", "XIXI"], [0.5, -1.5])], [SparsePauliOp("YIZX")]]
    values = np.random.default_rng(1).uniform(-2, 2, (3, 2))
    expected = StatevectorEstimator().run([(qc, observables, values)]).result()[0].data.evs
    for method in ("statevector", "pauli_propagation"):
        estimator = QevedoEstimator(method=method, min_abs_coefficient=0)
        result = estimator.run([(qc, observables, values)]).result()[0]
        assert result.data.evs.shape == (2, 3)
        np.testing.assert_allclose(result.data.evs, expected, atol=1e-9)
        assert result.metadata["method"] == method
        assert not result.data.truncated_one_norm.any()


def test_estimator_large_circuit_uses_pauli_propagation():
    n = 80
    qc = QuantumCircuit(n)
    qc.h(0)
    for q in range(n - 1):
        qc.cx(q, q + 1)
    qc.ry(0.3, n - 1)
    result = QevedoEstimator().run([(qc, ["Z" * n, "X" * n])]).result()[0]
    assert result.metadata["method"] == "pauli_propagation"
    np.testing.assert_allclose(result.data.evs, [math.cos(0.3)] * 2, atol=1e-12)


def test_estimator_precision_and_final_measurements():
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.measure_all()
    exact = QevedoEstimator().run([(qc, "X")]).result()[0].data.evs
    assert float(exact) == pytest.approx(1.0)
    noisy = QevedoEstimator(seed=3).run([(qc, ["X"] * 2000)], precision=0.1).result()[0].data.evs
    assert abs(noisy.mean() - 1.0) < 0.01 and abs(noisy.std() - 0.1) < 0.01


def test_core_functions_accept_qiskit_circuits():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    assert expectation(qc, "ZZI").value == pytest.approx(1.0)
    measured = qc.measure_all(inplace=False)
    assert set(run(measured, shots=200, seed=1).counts) == {"000", "111"}


def literal_distribution(qc):
    """The exact distribution of a dynamic circuit's classical bits,
    branching on every measurement and evaluating each ``if_test`` as
    written: its condition on the bits so far, then the chosen body."""
    out = {}

    def evaluate(condition, bits):
        def bit_value(target):
            if isinstance(target, Clbit):
                return int(bits[qc.find_bit(target).index])
            return sum(int(bits[qc.find_bit(c).index]) << i for i, c in enumerate(target))

        def ev(e):
            if isinstance(e, expr.Var):
                return bit_value(e.var)
            if isinstance(e, expr.Value):
                return int(e.value)
            if isinstance(e, expr.Cast):
                return ev(e.operand)
            if isinstance(e, expr.Unary):
                return int(not ev(e.operand))
            assert isinstance(e, expr.Binary)
            equal = ev(e.left) == ev(e.right)
            return int(equal if e.op == expr.Binary.Op.EQUAL else not equal)

        if isinstance(condition, tuple):
            return bit_value(condition[0]) == int(condition[1])
        return bool(ev(condition))

    def walk(data, state, bits, weight):
        for i, instruction in enumerate(data):
            op = instruction.operation
            qargs = [qc.find_bit(q).index for q in instruction.qubits]
            if op.name == "measure":
                c = qc.find_bit(instruction.clbits[0]).index
                for outcome in (0, 1):
                    p = state.probabilities([qargs[0]])[outcome]
                    if p < 1e-12:
                        continue
                    projector = np.diag([1 - outcome, outcome])
                    branch = state.evolve(projector, [qargs[0]]) / math.sqrt(p)
                    walk(data[i + 1 :], branch, bits[:c] + (bool(outcome),) + bits[c + 1 :], weight * p)
                return
            if op.name == "if_else":
                taken = evaluate(op.condition, bits)
                block = op.blocks[0] if taken else (op.blocks[1] if len(op.blocks) > 1 else None)
                if block is not None:
                    state = state.evolve(block, qargs)
            elif op.name != "barrier":
                state = state.evolve(op, qargs)
        key = "".join("1" if b else "0" for b in reversed(bits))
        out[key] = out.get(key, 0.0) + weight

    walk(list(qc.data), Statevector.from_label("0" * qc.num_qubits), (False,) * qc.num_clbits, 1.0)
    return out


def random_dynamic_circuit(n, rng):
    """Clifford gates, mid-circuit measurements and ``if_test``s with Pauli
    bodies (some with ``else``), conditions in every form Qiskit writes."""
    qr = QuantumRegister(n, "q")
    flags = ClassicalRegister(1, "flag")
    c = ClassicalRegister(2, "c")
    qc = QuantumCircuit(qr, c, flags)
    clbits = list(c) + list(flags)
    paulis = ["x", "y", "z", "id"]
    for _ in range(int(rng.integers(8, 18))):
        kind = int(rng.integers(0, 10))
        if kind <= 3:
            getattr(qc, str(rng.choice(["h", "s", "sdg", "sx"])))(int(rng.integers(n)))
        elif kind <= 5 and n > 1:
            a, b = (int(x) for x in rng.choice(n, 2, replace=False))
            getattr(qc, str(rng.choice(["cx", "cz"])))(a, b)
        elif kind <= 7:
            qc.measure(int(rng.integers(n)), clbits[int(rng.integers(3))])
        else:
            bit = clbits[int(rng.integers(3))]
            form = int(rng.integers(0, 6))
            if form == 0:
                condition = (bit, int(rng.integers(2)))
            elif form == 1:
                condition = (flags, int(rng.integers(2)))
            elif form == 2:
                condition = expr.lift(bit)
            elif form == 3:
                condition = expr.logic_not(bit)
            elif form == 4:
                condition = expr.equal(flags, int(rng.integers(2)))
            else:
                condition = expr.not_equal(bit, bool(rng.integers(2)))

            def body():
                for _ in range(int(rng.integers(1, 4))):
                    getattr(qc, str(rng.choice(paulis)))(int(rng.integers(n)))

            if rng.random() < 0.5:
                with qc.if_test(condition) as else_:
                    body()
                with else_:
                    body()
            else:
                with qc.if_test(condition):
                    body()
    for q in range(n):
        qc.measure(q, clbits[q % 3])
    return qc


def test_dynamic_circuits_with_pauli_feedback_match_their_meaning():
    rng = np.random.default_rng(11)
    backend = QevedoSimulator()
    for case in range(30):
        qc = random_dynamic_circuit(1 + case % 4, rng)
        exact = literal_distribution(qc)
        result = backend.run(qc, shots=20_000, seed_simulator=case).result()
        assert result.results[0].metadata["method"] == "stabilizer"
        counts = {k.replace(" ", ""): v for k, v in result.get_counts().items()}
        assert set(counts) <= {k for k, p in exact.items() if p > 1e-12}, case
        assert total_variation(counts, exact) < 0.02, case
        # The same through the state vector, shot by shot.
        counts = backend.run(qc, shots=4000, seed_simulator=case, method="statevector").result().get_counts()
        counts = {k.replace(" ", ""): v for k, v in counts.items()}
        assert total_variation(counts, exact) < 0.05, case


def test_teleportation_and_reset_by_feedback():
    # Teleport |+i>, undo its preparation: the target reads 0 every time.
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.s(0)
    qc.h(1)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.h(0)
    qc.measure(0, 0)
    qc.measure(1, 1)
    with qc.if_test((qc.clbits[1], 1)):
        qc.x(2)
    with qc.if_test((qc.clbits[0], 1)):
        qc.z(2)
    qc.sdg(2)
    qc.h(2)
    qc.measure(2, 2)
    backend = QevedoSimulator()
    counts = backend.run(transpile(qc, backend), shots=2000, seed_simulator=1).result().get_counts()
    assert len(counts) == 4 and all(k[0] == "0" for k in counts)
    # A reset by measurement and feedback, on 200 qubits of a GHZ state.
    n = 200
    qc = QuantumCircuit(n, n + 1)
    qc.h(0)
    for q in range(n - 1):
        qc.cx(q, q + 1)
    for q in range(n):
        qc.measure(q, q)
        with qc.if_test((qc.clbits[q], 1)):
            qc.x(q)
    qc.measure(n - 1, n)
    out = backend.run(qc, shots=1000, seed_simulator=2).result()
    assert out.results[0].metadata["method"] == "stabilizer"
    counts = out.get_counts()
    assert set(counts) == {"0" + "0" * n, "0" + "1" * n}


def test_feedback_through_transpile_sampler_and_openqasm_3():
    rng = np.random.default_rng(5)
    backend = QevedoSimulator()
    for case in range(6):
        qc = random_dynamic_circuit(3, rng)
        exact = literal_distribution(qc)
        compiled = transpile(qc, backend, optimization_level=3, seed_transpiler=case)
        counts = backend.run(compiled, shots=20_000, seed_simulator=case).result().get_counts()
        assert total_variation({k.replace(" ", ""): v for k, v in counts.items()}, exact) < 0.02, case
        # Qiskit's OpenQASM 3 export, read by the simulator's own parser.
        text = qasm3.dumps(qc)
        counts = run(text, 20_000, seed=case).counts
        assert total_variation(counts, exact) < 0.02, (case, text)
        # The sampler, register by register.
        result = QevedoSampler(seed=case).run([qc], shots=20_000).result()[0]
        joined = result.join_data().get_counts()
        assert total_variation(joined, exact) < 0.02, case
