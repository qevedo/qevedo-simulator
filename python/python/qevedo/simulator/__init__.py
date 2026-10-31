"""qvd: a fast quantum circuit simulator, as ``qevedo.simulator``.

Build circuits with :class:`Circuit` (or read OpenQASM 2/3 with
:meth:`Circuit.from_qasm`), then sample them with :func:`run` or get the
final state with :func:`statevector`, draw raw measurement records of
noisy Clifford circuits with :func:`sample`, or compute expectation values of
circuits far too large for a state vector with :func:`expectation`. ``run`` chooses the method
automatically: the stabilizer backend for Clifford circuits, the
near-Clifford backend when few non-Clifford gates are ever active, the
state vector when it fits in memory, and a matrix product state otherwise.

    >>> from qevedo.simulator import Circuit, run
    >>> bell = Circuit(2).h(0).cx(0, 1).measure_all()
    >>> result = run(bell, shots=1000, seed=1)
    >>> sorted(result.counts)
    ['00', '11']
    >>> result.backend
    'stabilizer'
"""

from __future__ import annotations

import platform as _platform

def _check_cpu() -> None:
    # x86-64 wheels use AVX2 + FMA kernels; fail clearly instead of crashing
    # on an older processor.
    if _platform.machine().lower() not in ("x86_64", "amd64"):
        return
    try:
        with open("/proc/cpuinfo") as f:
            flags = next((line for line in f if line.startswith("flags")), "")
    except OSError:
        return
    if flags and not {"avx2", "fma"} <= set(flags.split()):
        raise ImportError(
            "qevedo.simulator's x86-64 wheels need a processor with AVX2 and FMA "
            "(Intel Haswell or AMD Excavator, 2013-2015, or newer); "
            "build from source with `pip install --no-binary qevedo-simulator qevedo-simulator`"
        )


_check_cpu()

from qevedo.simulator._native import (  # noqa: E402
    Circuit,
    ExpectationResult,
    RunResult,
    __version__,
    choose_backend as _choose_backend,
    circuit_from_stim as _circuit_from_stim,
    expectation as _expectation,
    run as _run,
    sample_bytes as _sample_bytes,
    statevector_bytes as _statevector_bytes,
    stim_detector_bytes as _stim_detector_bytes,
    stim_sample_bytes as _stim_sample_bytes,
    uses_avx2,
)

__all__ = [
    "Circuit",
    "ExpectationResult",
    "RunResult",
    "run",
    "sample",
    "sample_detectors",
    "from_stim",
    "statevector",
    "expectation",
    "choose_backend",
    "uses_avx2",
    "__version__",
]

BACKENDS = ("auto", "statevector", "stabilizer", "mps", "near_clifford")


def _circuit(circuit: Circuit | str) -> Circuit:
    if isinstance(circuit, str):
        return Circuit.from_qasm(circuit)
    if not isinstance(circuit, Circuit):
        raise TypeError("expected a qevedo.simulator.Circuit or OpenQASM text")
    return circuit


def run(
    circuit: Circuit | str,
    shots: int = 1024,
    *,
    backend: str = "auto",
    seed: int | None = None,
    precision: str = "single",
    max_bond_dimension: int = 256,
    truncation_threshold: float = 1e-16,
    threads: int | str | None = None,
) -> RunResult:
    """Sample ``shots`` shots of a circuit (or OpenQASM 2/3 text).

    ``backend`` is one of ``"auto"``, ``"statevector"``, ``"stabilizer"``,
    ``"mps"`` and ``"near_clifford"``. ``precision`` (``"single"`` or
    ``"double"``) applies to the state vector. ``max_bond_dimension`` and
    ``truncation_threshold`` control the MPS: at most that many singular
    values per bond, and the smallest are dropped while the sum of their
    squares stays below the threshold. ``threads`` is a thread count or a
    placement (``"pcores"``, ``"pthreads"``, ``"cores"``, ``"all"``);
    ``None`` uses all cores. The simulation runs without holding the GIL.

    The result has ``counts`` (bitstrings with classical bit 0 on the
    right, as in Qiskit), ``backend`` (the method that ran), timings, and
    method-specific statistics: ``fidelity`` and ``max_bond_dimension`` for
    the MPS, ``peak_active_qubits`` for the near-Clifford backend.
    """
    return _run(
        _circuit(circuit),
        shots,
        backend=backend,
        seed=seed,
        precision=precision,
        max_bond_dimension=max_bond_dimension,
        truncation_threshold=truncation_threshold,
        threads=threads,
    )


def statevector(circuit: Circuit | str, *, threads: int | str | None = None):
    """The final state of a circuit without measurements, in double
    precision: a NumPy array of complex128 if NumPy is installed, a list of
    complex numbers otherwise. Index bit ``q`` is qubit ``q`` (Qiskit's
    order)."""
    raw = _statevector_bytes(_circuit(circuit), threads=threads)
    try:
        import numpy as np
    except ImportError:
        import struct

        values = struct.unpack(f"<{len(raw) // 8}d", raw)
        return [complex(values[i], values[i + 1]) for i in range(0, len(values), 2)]
    return np.frombuffer(raw, dtype="<c16").copy()


def sample(
    circuit: Circuit | str,
    shots: int,
    *,
    seed: int | None = None,
    threads: int | str | None = None,
    bit_packed: bool = False,
):
    """Shots of a Clifford circuit on the stabilizer backend, as raw
    measurement results rather than counts: the input decoders and
    error-correction studies need.

    Pauli noise channels (``x_error``, ``depolarize1``, ``depolarize2``,
    ...) are sampled independently in every shot. The result is
    measurement-major, as Stim's ``FlipSimulator``: a NumPy bool array of
    shape ``(num_clbits, shots)``, or with ``bit_packed=True`` a uint8
    array of shape ``(num_clbits, ceil(shots / 8))`` with shot ``s`` in
    bit ``s % 8`` of byte ``s // 8``. Needs NumPy.

    ``circuit`` may also be Stim's text format or a ``stim.Circuit``:
    measurement ``m`` is then row ``m``, and measurement flips (``M(p)``)
    and inverted results (``M !q``) are applied. For detection events, see
    ``sample_detectors``.
    """
    import numpy as np

    if _is_stim(circuit):
        raw, rows = _stim_sample_bytes(str(circuit), shots, seed=seed, threads=threads)
    else:
        circuit = _circuit(circuit)
        raw = _sample_bytes(circuit, shots, seed=seed, threads=threads)
        rows = circuit.num_clbits
    packed = np.frombuffer(raw, dtype=np.uint8).reshape(rows, (shots + 7) // 8)
    if bit_packed:
        return packed.copy()
    return np.unpackbits(packed, axis=1, count=shots, bitorder="little").astype(bool)


def _is_stim(circuit) -> bool:
    # A stim.Circuit, or text in Stim's format: OpenQASM ends every
    # statement with ';', Stim never uses one.
    if type(circuit).__module__.split(".")[0] == "stim":
        return True
    return isinstance(circuit, str) and ";" not in circuit


def _stim_text(circuit) -> str:
    if isinstance(circuit, Circuit):
        return circuit.to_stim()
    return str(circuit)


def from_stim(text) -> tuple[Circuit, list[list[int]], list[list[int]]]:
    """Read a circuit in Stim's format (text, or a ``stim.Circuit``).

    Returns the ``Circuit`` (measurement ``m``, in Stim's order, is
    classical bit ``m``; ``REPEAT`` blocks are unrolled), the detectors and
    the observables, each a list of measurement indices. Fails on
    instructions qvd cannot run, naming the line.
    """
    return _circuit_from_stim(str(text))


def sample_detectors(
    circuit,
    shots: int,
    *,
    seed: int | None = None,
    threads: int | str | None = None,
    separate_observables: bool = True,
):
    """Detection events and observable flips of a noisy error-correction
    circuit, as Stim's detector sampler returns them.

    ``circuit`` is Stim's text format, a ``stim.Circuit`` (for example from
    ``stim.Circuit.generated``), or a qvd ``Circuit`` (exported with
    ``to_stim``, so without detectors). A detector fires when the parity of
    its measurements differs from the noiseless circuit's.

    The arrays are shot-major, as Stim's and as decoders take them
    (``pymatching.Matching.decode_batch``): with ``separate_observables``
    a pair ``(detectors, observables)`` of NumPy bool arrays of shapes
    ``(shots, num_detectors)`` and ``(shots, num_observables)``; without,
    one array with the observables appended after the detectors.
    """
    import numpy as np

    raw_d, raw_o, nd, no = _stim_detector_bytes(
        _stim_text(circuit), shots, seed=seed, threads=threads
    )
    # Shot-major bytes of 0 and 1 from Rust: a bool view, no copy.
    detectors = np.frombuffer(raw_d, dtype=np.bool_).reshape(shots, nd)
    observables = np.frombuffer(raw_o, dtype=np.bool_).reshape(shots, no)
    if separate_observables:
        return detectors, observables
    return np.concatenate([detectors, observables], axis=1)


def _terms(observable) -> list[tuple[str, float]]:
    if isinstance(observable, str):
        items = [(observable, 1.0)]
    elif isinstance(observable, dict):
        items = list(observable.items())
    elif hasattr(observable, "to_list"):  # Qiskit's SparsePauliOp
        items = observable.to_list()
    else:
        items = list(observable)
    terms = []
    for label, coefficient in items:
        coefficient = complex(coefficient)
        if abs(coefficient.imag) > 1e-12 * max(1.0, abs(coefficient)):
            raise ValueError("observables must be Hermitian: real coefficients only")
        terms.append((str(label), coefficient.real))
    return terms


def expectation(
    circuit: Circuit | str,
    observable,
    *,
    min_abs_coefficient: float = 1e-5,
    max_terms: int | None = None,
    threads: int | str | None = None,
) -> ExpectationResult:
    """``<0...0| U† O U |0...0>`` for a unitary circuit ``U`` by Pauli
    propagation, for any number of qubits up to 1024.

    The observable ``O`` is a Pauli string, a ``{string: coefficient}``
    dict, a list of ``(string, coefficient)`` pairs, or a Qiskit
    ``SparsePauliOp``. Strings are sparse (``"Z62"``, ``"X0 Y3 Z7"``) or
    dense labels with qubit 0 on the right (``"IXZ"``, Qiskit's order).

    The observable is propagated backwards through the circuit; Clifford
    gates are absorbed into a frame for free, and every other rotation can
    double the number of Pauli strings. Strings whose coefficient falls
    below ``min_abs_coefficient`` are dropped (0 keeps all: exact, and
    exponential in the number of rotations), and at most ``max_terms`` of
    the largest are kept. The result's ``truncated_one_norm``, the sum of
    the dropped magnitudes, bounds the error of ``value``.
    """
    return _expectation(
        _circuit(circuit),
        _terms(observable),
        min_abs_coefficient=min_abs_coefficient,
        max_terms=max_terms,
        threads=threads,
    )


def choose_backend(
    circuit: Circuit | str,
    *,
    backend: str = "auto",
    precision: str = "single",
    max_bond_dimension: int = 256,
    truncation_threshold: float = 1e-16,
) -> str:
    """The method :func:`run` would use for this circuit."""
    return _choose_backend(
        _circuit(circuit),
        backend=backend,
        precision=precision,
        max_bond_dimension=max_bond_dimension,
        truncation_threshold=truncation_threshold,
    )
