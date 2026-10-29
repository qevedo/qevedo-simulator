import cmath
import math
import random

import pytest

from qevedo.simulator import Circuit, choose_backend, expectation, run, sample, statevector


def probabilities(circuit):
    """Exact outcome probabilities of a circuit measured at the end."""
    unitary = Circuit(circuit.num_qubits)
    measures = []
    for name, qubits, params in circuit.instructions():
        if name == "measure":
            measures.append((qubits[0], int(params[0])))
        elif name not in ("barrier",):
            unitary.gate(name, qubits, params)
    state = statevector(unitary)
    out = {}
    width = circuit.num_clbits
    for index, amplitude in enumerate(state):
        p = abs(amplitude) ** 2
        if p < 1e-15:
            continue
        bits = ["0"] * width
        for qubit, clbit in measures:
            bits[width - 1 - clbit] = str((index >> qubit) & 1)
        key = "".join(bits)
        out[key] = out.get(key, 0.0) + p
    return out


def assert_matches(counts, exact, shots):
    for key in counts:
        assert key in exact, f"impossible outcome {key}"
    for key, p in exact.items():
        observed = counts.get(key, 0) / shots
        sigma = math.sqrt(p * (1 - p) / shots)
        assert abs(observed - p) <= 5 * sigma + 3 / shots, (key, observed, p)


def test_bell_state_on_the_stabilizer_backend():
    bell = Circuit(2).h(0).cx(0, 1).measure_all()
    result = run(bell, shots=4000, seed=1)
    assert set(result.counts) == {"00", "11"}
    assert sum(result.counts.values()) == 4000
    assert result.backend == "stabilizer"
    assert result.gates == 2


def test_qasm_text_runs_directly():
    program = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[3] q;
        bit[3] c;
        h q[0];
        cx q[0], q[1];
        ccx q[0], q[1], q[2];
        c = measure q;
    """
    result = run(program, shots=2000, seed=3)
    assert set(result.counts) == {"000", "111"}
    assert result.backend in ("near_clifford", "statevector")


def test_statevector_matches_known_states():
    ghz = statevector(Circuit(3).h(0).cx(0, 1).cx(1, 2))
    r = 1 / math.sqrt(2)
    assert abs(ghz[0] - r) < 1e-12 and abs(ghz[7] - r) < 1e-12
    assert sum(abs(a) ** 2 for a in ghz) == pytest.approx(1.0)
    phased = statevector(Circuit(1).h(0).t(0))
    assert abs(phased[1] - r * cmath.exp(1j * math.pi / 4)) < 1e-12


@pytest.mark.parametrize("backend", ["statevector", "mps", "near_clifford"])
def test_every_backend_samples_the_exact_distribution(backend):
    rng = random.Random(7)
    circuit = Circuit(5)
    for _ in range(40):
        a, b = rng.sample(range(5), 2)
        choice = rng.randrange(6)
        if choice == 0:
            circuit.h(a)
        elif choice == 1:
            circuit.t(a)
        elif choice == 2:
            circuit.rx(rng.uniform(0, 6.28), a)
        elif choice == 3:
            circuit.cx(a, b)
        elif choice == 4:
            circuit.cp(rng.uniform(0, 6.28), a, b)
        else:
            circuit.ccx(a, b, rng.choice([q for q in range(5) if q not in (a, b)]))
    circuit.measure_all()
    shots = 20000
    result = run(circuit, shots=shots, backend=backend, seed=5, max_bond_dimension=64)
    assert result.backend == backend
    assert_matches(result.counts, probabilities(circuit), shots)
    if backend == "mps":
        assert result.fidelity == pytest.approx(1.0)
    if backend == "near_clifford":
        assert result.peak_active_qubits is not None


def test_stabilizer_backend_rejects_non_clifford_gates():
    with pytest.raises(RuntimeError, match="not a Clifford"):
        run(Circuit(1).t(0).measure_all(), shots=10, backend="stabilizer")


def test_automatic_choice():
    big_clifford = Circuit(500).h(0)
    for q in range(1, 500):
        big_clifford.cx(q - 1, q)
    assert choose_backend(big_clifford) == "stabilizer"
    assert choose_backend(Circuit(4).h(0).t(0).cx(0, 1)) in ("statevector", "near_clifford")
    many_t = Circuit(60)
    for q in range(60):
        many_t.h(q).t(q).h(q)
    assert choose_backend(many_t) == "mps"


def test_qasm_round_trip_and_instructions():
    circuit = Circuit(3, 3)
    circuit.h(0).cx(0, 1).rz(0.25, 2).gate("cu3", [2, 0], [0.1, 0.2, 0.3]).measure(1, 2)
    with pytest.raises(ValueError, match="unitaries"):
        circuit.to_qasm()  # cu3 is not native, so it is stored as a unitary
    plain = Circuit(3, 3).h(0).cx(0, 1).rz(0.25, 2).measure(1, 2)
    again = Circuit.from_qasm(plain.to_qasm())
    assert again.instructions() == plain.instructions()
    assert plain.instructions()[2] == ("rz", [2], [0.25])
    assert plain.instructions()[3] == ("measure", [1], [2.0])


def test_unitary_from_nested_lists():
    sx = [[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]]
    a = statevector(Circuit(1).unitary(sx, [0]))
    b = statevector(Circuit(1).sx(0))
    assert all(abs(x - y) < 1e-12 for x, y in zip(a, b))
    with pytest.raises(ValueError, match="not unitary"):
        Circuit(1).unitary([[1, 1], [0, 1]], [0])


def test_errors():
    with pytest.raises(ValueError, match="out of range"):
        Circuit(2).h(2)
    with pytest.raises(ValueError, match="same qubit"):
        Circuit(2).cx(1, 1)
    with pytest.raises(ValueError, match="line 3"):
        Circuit.from_qasm('OPENQASM 2.0;\nqreg q[1];\nfoo q[0];')
    with pytest.raises(ValueError, match="unknown backend"):
        run(Circuit(1).measure_all(), backend="quantum")
    with pytest.raises(ValueError, match="unknown gate"):
        Circuit(1).gate("nope", [0])


def test_threads_and_reproducibility():
    circuit = Circuit(6)
    for q in range(6):
        circuit.h(q).t(q)
    for q in range(5):
        circuit.cx(q, q + 1)
    circuit.measure_all()
    a = run(circuit, shots=3000, seed=9, backend="statevector", threads=1)
    b = run(circuit, shots=3000, seed=9, backend="statevector", threads=4)
    assert a.counts == b.counts


def test_against_qiskit():
    qiskit = pytest.importorskip("qiskit")
    from qiskit import qasm2, qasm3
    from qiskit.circuit.random import random_circuit
    from qiskit.quantum_info import Statevector

    for seed in range(40):
        qc = random_circuit(5, 6, max_operands=3, seed=seed)
        expected = Statevector(qc).data
        for text in (qasm2.dumps(qc), qasm3.dumps(qc)):
            got = statevector(text)
            # Qiskit's exported definitions of some gates (ecr, for one) drop
            # their global phase, so compare up to a global phase.
            overlap = abs(sum(x.conjugate() * y for x, y in zip(got, expected)))
            assert overlap == pytest.approx(1.0, abs=1e-9), (seed, text[:200])


def dense_expectation(state, label):
    """<ψ|P|ψ> for a dense label (qubit 0 on the right) by direct sums."""
    n = len(label)
    total = 0.0
    for index, amplitude in enumerate(state):
        flipped, phase = index, 1.0 + 0j
        for q, p in enumerate(reversed(label)):
            bit = (index >> q) & 1
            if p in "XY":
                flipped ^= 1 << q
            if p == "Z" and bit:
                phase = -phase
            if p == "Y":
                phase *= 1j if bit == 0 else -1j
        # P|index> = phase |flipped>, so <ψ|P|ψ> sums conj(ψ[flipped]) phase ψ[index].
        total += (state[flipped].conjugate() * phase * amplitude).real
    return total


def test_expectation_matches_the_state_vector():
    rng = random.Random(7)
    gates = [("h", 1, 0), ("s", 1, 0), ("t", 1, 0), ("rx", 1, 1), ("ry", 1, 1), ("rz", 1, 1),
             ("cx", 2, 0), ("cz", 2, 0), ("cp", 2, 1), ("rzz", 2, 1), ("ccx", 3, 0)]
    for _ in range(10):
        n = rng.randint(3, 6)
        circuit = Circuit(n)
        for _ in range(25):
            name, arity, params = rng.choice(gates)
            circuit.gate(name, rng.sample(range(n), arity), [rng.uniform(-3, 3) for _ in range(params)])
        state = statevector(circuit)
        labels = ["".join(rng.choice("IXYZ") for _ in range(n)) for _ in range(3)]
        observable = {label: rng.uniform(-1, 1) for label in labels}
        result = expectation(circuit, observable, min_abs_coefficient=0)
        want = sum(c * dense_expectation(state, label) for label, c in observable.items())
        assert result.value == pytest.approx(want, abs=1e-9)
        assert result.truncated_one_norm == 0


def test_expectation_at_utility_scale():
    # Two steps of the 127-qubit kicked Ising circuit is beyond a state
    # vector; at θ_h = 0 every <Z_q> is 1, and the sparse label form works.
    edges = [(q, q + 1) for q in range(126)]
    circuit = Circuit(127)
    for _ in range(2):
        for q in range(127):
            circuit.gate("rx", [q], [0.3])
        for a, b in edges:
            circuit.gate("rzz", [a, b], [-math.pi / 2])
    result = expectation(circuit, "Z62", min_abs_coefficient=1e-8)
    assert -1 <= result.value <= 1 and result.peak_terms > 1
    assert expectation(circuit, [("Z62", 0.5), ("Z62", 0.5)]).value == pytest.approx(
        expectation(circuit, "Z62").value
    )
    with pytest.raises(ValueError):
        expectation(Circuit(2).h(0).measure_all(), "Z0")
    with pytest.raises(ValueError):
        expectation(Circuit(2), {"Z0": 1j})


def test_expectation_against_qiskit():
    pytest.importorskip("qiskit")
    from qiskit.circuit.random import random_circuit
    from qiskit.quantum_info import SparsePauliOp, Statevector
    from qiskit import qasm2, transpile

    basis = ["h", "s", "sdg", "t", "tdg", "sx", "rx", "ry", "rz", "u", "cx", "cz", "cp", "rzz", "swap"]
    for seed in range(5):
        qc = transpile(random_circuit(5, 6, seed=seed), basis_gates=basis, optimization_level=0)
        operator = SparsePauliOp(["XXZIY", "ZZIII", "IYXZI"], [0.5, -1.0, 0.25])
        want = Statevector(qc).expectation_value(operator).real
        got = expectation(qasm2.dumps(qc), operator, min_abs_coefficient=0)
        assert got.value == pytest.approx(want, abs=1e-9)


def test_noise_channels_flip_at_their_rates():
    pytest.importorskip("numpy")
    n, shots = 200, 20_000
    circuit = Circuit(n)
    for q in range(0, n, 2):
        circuit.x_error(q, 0.01).depolarize1(q + 1, 0.3)
    circuit.measure_all()
    assert circuit.has_noise() and choose_backend(circuit) == "stabilizer"
    bits = sample(circuit, shots, seed=3)
    assert bits.shape == (n, shots) and bits.dtype == bool
    # A bit flip shows in Z with probability p; depolarizing in 2p/3.
    for rows, p in ((bits[0::2], 0.01), (bits[1::2], 0.2)):
        sigma = math.sqrt(p * (1 - p) / rows.size)
        assert abs(rows.mean() - p) < 5 * sigma
    packed = sample(circuit, shots, seed=3, bit_packed=True)
    assert packed.shape == (n, (shots + 7) // 8)
    import numpy as np

    assert (np.unpackbits(packed, axis=1, count=shots, bitorder="little").astype(bool) == bits).all()
    # Two-qubit depolarizing: each qubit of the pair flips in 8 of 15 cases.
    pair = Circuit(2).depolarize2(0, 1, 0.75).measure_all()
    flips = sample(pair, 60_000, seed=1).mean(axis=1)
    assert abs(flips - 0.4).max() < 0.01
    counts = run(pair, 10_000, seed=1).counts
    assert set(counts) == {"00", "01", "10", "11"}


def test_noise_errors_and_stim_text():
    with pytest.raises(ValueError):
        Circuit(1).x_error(0, 1.5)
    with pytest.raises(ValueError):
        Circuit(1).pauli_channel_1(0, 0.5, 0.4, 0.3)
    with pytest.raises(ValueError):
        Circuit(2).depolarize2(0, 0, 0.1)
    noisy = Circuit(2).h(0).cx(0, 1).depolarize2(0, 1, 0.01).z_error(1, 0.02).measure_all()
    for backend in ("statevector", "mps", "near_clifford"):
        with pytest.raises(RuntimeError):
            run(noisy, 10, backend=backend)
    with pytest.raises(ValueError):
        noisy.to_qasm()
    assert noisy.instructions()[2:4] == [
        ("depolarize2", [0, 1], [0.01]),
        ("z_error", [1], [0.02]),
    ]
    assert noisy.to_stim() == "H 0\nCX 0 1\nDEPOLARIZE2(0.01) 0 1\nZ_ERROR(0.02) 1\nM 0 1\n"
    with pytest.raises(ValueError):
        sample(Circuit(1).t(0).measure_all(), 10)


def test_noise_against_stim():
    stim = pytest.importorskip("stim")
    pytest.importorskip("numpy")
    # A repetition code with noise everywhere: the same circuit in both
    # simulators must give the same rate for every measurement parity.
    circuit = Circuit(5)
    clbit = 0
    for _ in range(3):
        for a in (1, 3):
            circuit.reset(a).x_error(a, 0.02)
        for a, b in ((0, 1), (2, 1), (2, 3), (4, 3)):
            circuit.cx(a, b).depolarize2(a, b, 0.03)
        for a in (1, 3):
            circuit.x_error(a, 0.01).measure(a, clbit)
            clbit += 1
    shots = 100_000
    measured = list(range(clbit))
    ours = sample(circuit, shots, seed=2)
    theirs = stim.Circuit(circuit.to_stim()).compile_sampler(seed=2).sample(shots).T
    for i in range(2, len(measured)):
        a = (ours[i] ^ ours[i - 2]).mean()
        b = (theirs[i] ^ theirs[i - 2]).mean()
        sigma = math.sqrt(max(a * (1 - a), 1e-9) * 2 / shots)
        assert abs(a - b) < 5 * sigma, (i, a, b)
