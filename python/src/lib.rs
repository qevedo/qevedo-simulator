//! Python bindings of qvd: the native module `qevedo.simulator._native`.
//! The Python package `qevedo.simulator` wraps it with documentation and
//! NumPy conversions.

use std::collections::BTreeMap;

use num_complex::Complex64;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes};

use qvd::threads::{Placement, pool};
use qvd::{Backend, Circuit, Gate, Instruction, Matrix, Noise, Options};

/// A quantum circuit: gates, measurements, resets, barriers and Pauli noise
/// channels on numbered qubits and classical bits (qubit 0 is the least significant bit, as in
/// Qiskit).
#[pyclass(name = "Circuit", module = "qevedo.simulator", skip_from_py_object)]
#[derive(Clone)]
struct PyCircuit {
    inner: Circuit,
}

fn value_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

impl PyCircuit {
    fn check(&self, qubits: &[usize]) -> PyResult<()> {
        if let Some(&q) = qubits.iter().find(|&&q| q >= self.inner.num_qubits) {
            return Err(value_error(format!(
                "qubit {q} is out of range for {} qubits",
                self.inner.num_qubits
            )));
        }
        let mut sorted = qubits.to_vec();
        sorted.sort();
        sorted.dedup();
        if sorted.len() != qubits.len() {
            return Err(value_error("a gate cannot act on the same qubit twice"));
        }
        Ok(())
    }

    fn push(&mut self, gate: Gate, qubits: &[usize]) -> PyResult<()> {
        self.check(qubits)?;
        if gate.arity() != qubits.len() {
            return Err(value_error(format!(
                "the gate acts on {} qubits, given {}",
                gate.arity(),
                qubits.len()
            )));
        }
        self.inner.gate(gate, qubits);
        Ok(())
    }

    fn push_noise(&mut self, channel: Noise, qubits: &[usize]) -> PyResult<()> {
        self.check(qubits)?;
        let probabilities = match channel {
            Noise::Pauli1 { px, py, pz } => vec![px, py, pz],
            Noise::Depolarize2(p) => vec![p],
        };
        if !probabilities.iter().all(|p| (0.0..=1.0).contains(p)) || channel.probability() > 1.0 {
            return Err(value_error(
                "noise probabilities must lie in [0, 1] and sum to at most 1",
            ));
        }
        self.inner.noise(channel, qubits);
        Ok(())
    }
}

/// Chainable methods for fixed gates: `circuit.h(0).cx(0, 1)`.
macro_rules! gates {
    ($( $method:ident => $gate:expr, ($($q:ident),+) $(, ($($p:ident),+))? );* $(;)?) => {
        #[pymethods]
        impl PyCircuit {
            $(
                fn $method<'py>(mut slf: PyRefMut<'py, Self>, $($($p: f64,)+)? $($q: usize),+) -> PyResult<PyRefMut<'py, Self>> {
                    let gate = $gate;
                    slf.push(gate, &[$($q),+])?;
                    Ok(slf)
                }
            )*
        }
    };
}

gates! {
    id => Gate::I, (q);
    h => Gate::H, (q);
    x => Gate::X, (q);
    y => Gate::Y, (q);
    z => Gate::Z, (q);
    s => Gate::S, (q);
    sdg => Gate::Sdg, (q);
    t => Gate::T, (q);
    tdg => Gate::Tdg, (q);
    sx => Gate::SX, (q);
    sxdg => Gate::SXdg, (q);
    rx => Gate::RX(theta), (q), (theta);
    ry => Gate::RY(theta), (q), (theta);
    rz => Gate::RZ(theta), (q), (theta);
    p => Gate::P(lam), (q), (lam);
    u => Gate::U(theta, phi, lam), (q), (theta, phi, lam);
    cx => Gate::CX, (control, target);
    cy => Gate::CY, (control, target);
    cz => Gate::CZ, (a, b);
    ch => Gate::CH, (control, target);
    swap => Gate::SWAP, (a, b);
    cp => Gate::CP(lam), (control, target), (lam);
    crx => Gate::CRX(theta), (control, target), (theta);
    cry => Gate::CRY(theta), (control, target), (theta);
    crz => Gate::CRZ(theta), (control, target), (theta);
    rzz => Gate::RZZ(theta), (a, b), (theta);
    rxx => Gate::RXX(theta), (a, b), (theta);
    ccx => Gate::CCX, (a, b, target);
    cswap => Gate::CSWAP, (control, a, b);
}

#[pymethods]
impl PyCircuit {
    #[new]
    #[pyo3(signature = (num_qubits, num_clbits = 0))]
    fn new(num_qubits: usize, num_clbits: usize) -> Self {
        let mut inner = Circuit::new(num_qubits);
        inner.num_clbits = num_clbits;
        PyCircuit { inner }
    }

    /// Parse an OpenQASM 2 or 3 program.
    #[staticmethod]
    fn from_qasm(source: &str) -> PyResult<Self> {
        qvd::qasm::parse(source)
            .map(|inner| PyCircuit { inner })
            .map_err(|e| value_error(e.to_string()))
    }

    /// The circuit as OpenQASM 2 (`qelib1.inc` gate names).
    fn to_qasm(&self) -> PyResult<String> {
        self.inner.to_qasm().map_err(value_error)
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits
    }

    #[getter]
    fn num_clbits(&self) -> usize {
        self.inner.num_clbits
    }

    /// Number of gates (not counting measurements, resets and barriers).
    #[getter]
    fn gate_count(&self) -> usize {
        self.inner.gate_count()
    }

    /// Apply a gate by its OpenQASM name (`qelib1.inc` or `stdgates.inc`),
    /// e.g. `circuit.gate("cu3", [0, 1], [0.1, 0.2, 0.3])`.
    #[pyo3(signature = (name, qubits, params = Vec::new()))]
    fn gate<'py>(
        mut slf: PyRefMut<'py, Self>,
        name: &str,
        qubits: Vec<usize>,
        params: Vec<f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let gate = qvd::qasm::gate(name, &params)
            .ok_or_else(|| value_error(format!("unknown gate `{name}`")))?
            .map_err(value_error)?;
        slf.push(gate, &qubits)?;
        Ok(slf)
    }

    /// Apply a unitary matrix (a `2^k x 2^k` nested list or array) to `k`
    /// qubits; bit `j` of the matrix index is `qubits[j]`.
    fn unitary<'py>(
        mut slf: PyRefMut<'py, Self>,
        matrix: Vec<Vec<Complex64>>,
        qubits: Vec<usize>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let dim = 1usize << qubits.len();
        if matrix.len() != dim || matrix.iter().any(|row| row.len() != dim) {
            return Err(value_error(format!(
                "a gate on {} qubits needs a {dim}x{dim} matrix",
                qubits.len()
            )));
        }
        let m = Matrix::new(matrix.into_iter().flatten().collect());
        if m.mul(&m.adjoint())
            .distance(&Matrix::identity(qubits.len()))
            > 1e-8
        {
            return Err(value_error("the matrix is not unitary"));
        }
        slf.push(Gate::Unitary(m), &qubits)?;
        Ok(slf)
    }

    /// Measure `qubit` into classical bit `clbit` (the circuit grows its
    /// classical bits as needed).
    fn measure<'py>(
        mut slf: PyRefMut<'py, Self>,
        qubit: usize,
        clbit: usize,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.check(&[qubit])?;
        slf.inner.measure(qubit, clbit);
        Ok(slf)
    }

    /// Measure every qubit `q` into classical bit `q`.
    fn measure_all(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.inner.measure_all();
        slf
    }

    fn reset(mut slf: PyRefMut<'_, Self>, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.check(&[qubit])?;
        slf.inner.reset(qubit);
        Ok(slf)
    }

    fn barrier(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.inner.barrier();
        slf
    }

    /// A bit flip (X) on `qubit` with probability `p`.
    fn x_error(mut slf: PyRefMut<'_, Self>, qubit: usize, p: f64) -> PyResult<PyRefMut<'_, Self>> {
        slf.push_noise(
            Noise::Pauli1 {
                px: p,
                py: 0.0,
                pz: 0.0,
            },
            &[qubit],
        )?;
        Ok(slf)
    }

    /// A Y error on `qubit` with probability `p`.
    fn y_error(mut slf: PyRefMut<'_, Self>, qubit: usize, p: f64) -> PyResult<PyRefMut<'_, Self>> {
        slf.push_noise(
            Noise::Pauli1 {
                px: 0.0,
                py: p,
                pz: 0.0,
            },
            &[qubit],
        )?;
        Ok(slf)
    }

    /// A phase flip (Z) on `qubit` with probability `p`.
    fn z_error(mut slf: PyRefMut<'_, Self>, qubit: usize, p: f64) -> PyResult<PyRefMut<'_, Self>> {
        slf.push_noise(
            Noise::Pauli1 {
                px: 0.0,
                py: 0.0,
                pz: p,
            },
            &[qubit],
        )?;
        Ok(slf)
    }

    /// X, Y or Z on `qubit` with probabilities `px`, `py`, `pz`.
    fn pauli_channel_1(
        mut slf: PyRefMut<'_, Self>,
        qubit: usize,
        px: f64,
        py: f64,
        pz: f64,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.push_noise(Noise::Pauli1 { px, py, pz }, &[qubit])?;
        Ok(slf)
    }

    /// X, Y or Z on `qubit`, each with probability `p / 3`.
    fn depolarize1(
        mut slf: PyRefMut<'_, Self>,
        qubit: usize,
        p: f64,
    ) -> PyResult<PyRefMut<'_, Self>> {
        let q = p / 3.0;
        slf.push_noise(
            Noise::Pauli1 {
                px: q,
                py: q,
                pz: q,
            },
            &[qubit],
        )?;
        Ok(slf)
    }

    /// Each of the 15 non-identity Paulis on qubits `a` and `b` with
    /// probability `p / 15`.
    fn depolarize2(
        mut slf: PyRefMut<'_, Self>,
        a: usize,
        b: usize,
        p: f64,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.push_noise(Noise::Depolarize2(p), &[a, b])?;
        Ok(slf)
    }

    /// Whether the circuit has noise channels (only the stabilizer backend
    /// runs those).
    fn has_noise(&self) -> bool {
        self.inner.has_noise()
    }

    /// The circuit in Stim's text format (Clifford circuits only), so that
    /// `stim.Circuit(circuit.to_stim())` samples the same noisy circuit.
    fn to_stim(&self) -> PyResult<String> {
        qvd::stabilizer::to_stim(&self.inner).map_err(value_error)
    }

    /// The instructions as `(name, qubits, params)` tuples; `params` holds
    /// the classical bit for measurements, is empty for unitaries, and holds
    /// the probabilities of noise channels (named as in Stim, lower case).
    fn instructions(&self) -> Vec<(String, Vec<usize>, Vec<f64>)> {
        self.inner
            .instructions
            .iter()
            .map(|instruction| match instruction {
                Instruction::Gate { gate, qubits } => match gate.qasm_name() {
                    Some((name, params)) => (name.to_string(), qubits.clone(), params),
                    None => ("unitary".to_string(), qubits.clone(), Vec::new()),
                },
                Instruction::Measure { qubit, clbit } => {
                    ("measure".into(), vec![*qubit], vec![*clbit as f64])
                }
                Instruction::Reset { qubit } => ("reset".into(), vec![*qubit], Vec::new()),
                Instruction::Barrier { qubits } => ("barrier".into(), qubits.clone(), Vec::new()),
                Instruction::Noise { channel, qubits } => {
                    let (name, params) = channel.stim_name();
                    (name.to_lowercase(), qubits.clone(), params)
                }
            })
            .collect()
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    fn __len__(&self) -> usize {
        self.inner.instructions.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "Circuit({} qubits, {} classical bits, {} gates)",
            self.inner.num_qubits,
            self.inner.num_clbits,
            self.inner.gate_count()
        )
    }
}

/// The outcome counts and statistics of a run.
#[pyclass(name = "RunResult", module = "qevedo.simulator", frozen)]
struct PyRunResult {
    counts: BTreeMap<String, usize>,
    backend: String,
    gates: usize,
    gate_seconds: f64,
    measure_seconds: f64,
    fidelity: Option<f64>,
    max_bond_dimension: Option<usize>,
    peak_active_qubits: Option<usize>,
}

#[pymethods]
impl PyRunResult {
    /// Counts per outcome bitstring (classical bit 0 on the right).
    #[getter]
    fn counts(&self) -> BTreeMap<String, usize> {
        self.counts.clone()
    }

    /// The method that ran: "statevector", "stabilizer", "mps" or
    /// "near_clifford".
    #[getter]
    fn backend(&self) -> String {
        self.backend.clone()
    }

    #[getter]
    fn gates(&self) -> usize {
        self.gates
    }

    #[getter]
    fn gate_seconds(&self) -> f64 {
        self.gate_seconds
    }

    #[getter]
    fn measure_seconds(&self) -> f64 {
        self.measure_seconds
    }

    /// MPS: the estimated fidelity with the exact state (product of the
    /// weights kept at each truncation); `None` for exact methods.
    #[getter]
    fn fidelity(&self) -> Option<f64> {
        self.fidelity
    }

    /// MPS: the largest bond dimension reached.
    #[getter]
    fn max_bond_dimension(&self) -> Option<usize> {
        self.max_bond_dimension
    }

    /// Near-Clifford: the largest number of active qubits.
    #[getter]
    fn peak_active_qubits(&self) -> Option<usize> {
        self.peak_active_qubits
    }

    fn __repr__(&self) -> String {
        format!(
            "RunResult(backend={:?}, {} outcomes)",
            self.backend,
            self.counts.len()
        )
    }
}

/// An expectation value from Pauli propagation, with its error bound.
#[pyclass(
    name = "ExpectationResult",
    module = "qevedo.simulator",
    frozen,
    get_all
)]
struct PyExpectationResult {
    /// The (approximate) expectation value.
    value: f64,
    /// Sum of the magnitudes of the dropped coefficients: a rigorous bound
    /// on the error of `value`; 0 when nothing was dropped.
    truncated_one_norm: f64,
    /// Non-Clifford Pauli rotations propagated through.
    rotations: usize,
    /// The most Pauli strings held at once.
    peak_terms: usize,
    /// Pauli strings left at the end.
    final_terms: usize,
    seconds: f64,
}

#[pymethods]
impl PyExpectationResult {
    fn __repr__(&self) -> String {
        format!(
            "ExpectationResult(value={}, truncated_one_norm={:e}, peak_terms={})",
            self.value, self.truncated_one_norm, self.peak_terms
        )
    }
}

/// `<0|U† O U|0>` by Pauli propagation; see `qevedo.simulator.expectation`.
/// `terms` are `(Pauli string, coefficient)` pairs.
#[pyfunction]
#[pyo3(signature = (circuit, terms, min_abs_coefficient = 1e-5, max_terms = None, threads = None))]
fn expectation(
    py: Python<'_>,
    circuit: &PyCircuit,
    terms: Vec<(String, f64)>,
    min_abs_coefficient: f64,
    max_terms: Option<usize>,
    threads: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyExpectationResult> {
    use qvd::pauliprop::{Observable, PropagationOptions, parse_pauli};
    let observable: Observable = terms
        .iter()
        .map(|(label, c)| Ok((*c, parse_pauli(label).map_err(value_error)?)))
        .collect::<PyResult<_>>()?;
    let options = PropagationOptions {
        min_abs_coefficient,
        max_terms,
    };
    let placement = parse_threads(threads)?;
    let circuit = circuit.inner.clone();
    let result = compute(py, placement, move || {
        qvd::pauliprop::expectation(&circuit, &observable, &options)
    })
    .map_err(value_error)?;
    Ok(PyExpectationResult {
        value: result.value,
        truncated_one_norm: result.truncated_one_norm,
        rotations: result.rotations,
        peak_terms: result.peak_terms,
        final_terms: result.final_terms,
        seconds: result.seconds,
    })
}

fn backend_name(backend: Backend) -> &'static str {
    match backend {
        Backend::Auto => "auto",
        Backend::StateVector => "statevector",
        Backend::Stabilizer => "stabilizer",
        Backend::Mps => "mps",
        Backend::NearClifford => "near_clifford",
    }
}

fn parse_backend(name: &str) -> PyResult<Backend> {
    Ok(match name {
        "auto" => Backend::Auto,
        "statevector" | "state_vector" => Backend::StateVector,
        "stabilizer" => Backend::Stabilizer,
        "mps" | "matrix_product_state" => Backend::Mps,
        "near_clifford" | "nearclifford" => Backend::NearClifford,
        _ => {
            return Err(value_error(format!(
                "unknown backend `{name}` (auto, statevector, stabilizer, mps, near_clifford)"
            )));
        }
    })
}

fn parse_threads(threads: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Placement>> {
    let Some(threads) = threads else {
        return Ok(None);
    };
    if let Ok(n) = threads.extract::<usize>() {
        return Ok(Some(Placement::Unpinned { threads: n.max(1) }));
    }
    let name: String = threads
        .extract()
        .map_err(|_| value_error("threads must be a number or a placement name"))?;
    Ok(Some(match name.as_str() {
        "pcores" => Placement::PerformanceCores,
        "pthreads" => Placement::PerformanceThreads,
        "cores" | "allcores" => Placement::AllCores,
        "all" => Placement::AllThreads,
        _ => {
            return Err(value_error(format!(
                "unknown thread placement `{name}` (pcores, pthreads, cores, all)"
            )));
        }
    }))
}

/// Run `f` without the GIL, on the requested threads.
fn compute<T: Send>(
    py: Python<'_>,
    placement: Option<Placement>,
    f: impl FnOnce() -> T + Send,
) -> T {
    py.detach(move || match placement {
        Some(placement) => pool(&placement).install(f),
        None => f(),
    })
}

#[allow(clippy::too_many_arguments)]
fn options(
    backend: &str,
    seed: Option<u64>,
    max_bond_dimension: usize,
    truncation_threshold: f64,
) -> PyResult<Options> {
    Ok(Options {
        backend: parse_backend(backend)?,
        seed,
        max_bond_dimension,
        truncation_threshold,
        ..Options::default()
    })
}

/// Run a circuit for `shots` shots; see `qevedo.simulator.run`.
#[pyfunction]
#[pyo3(signature = (circuit, shots, backend = "auto", seed = None, precision = "single", max_bond_dimension = 256, truncation_threshold = 1e-16, threads = None))]
#[allow(clippy::too_many_arguments)]
fn run(
    py: Python<'_>,
    circuit: &PyCircuit,
    shots: usize,
    backend: &str,
    seed: Option<u64>,
    precision: &str,
    max_bond_dimension: usize,
    truncation_threshold: f64,
    threads: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyRunResult> {
    let options = options(backend, seed, max_bond_dimension, truncation_threshold)?;
    let placement = parse_threads(threads)?;
    let circuit = circuit.inner.clone();
    let result = match precision {
        "single" => compute(py, placement, move || {
            qvd::run::<f32>(&circuit, shots, &options)
        }),
        "double" => compute(py, placement, move || {
            qvd::run::<f64>(&circuit, shots, &options)
        }),
        _ => return Err(value_error("precision must be \"single\" or \"double\"")),
    }
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let stats = result.stats;
    Ok(PyRunResult {
        counts: result.counts,
        backend: backend_name(stats.backend.unwrap_or(Backend::Auto)).to_string(),
        gates: stats.gates,
        gate_seconds: stats.gate_seconds,
        measure_seconds: stats.measure_seconds,
        fidelity: stats.fidelity,
        max_bond_dimension: stats.max_bond_dimension,
        peak_active_qubits: stats.peak_active_qubits,
    })
}

/// The final state of a circuit without measurements or resets, as raw
/// little-endian complex128 bytes (index bit `q` = qubit `q`); see
/// `qevedo.simulator.statevector`.
#[pyfunction]
#[pyo3(signature = (circuit, threads = None))]
fn statevector_bytes<'py>(
    py: Python<'py>,
    circuit: &PyCircuit,
    threads: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyBytes>> {
    if circuit.inner.num_qubits > 34 {
        return Err(value_error(
            "a state vector of more than 34 qubits does not fit in memory",
        ));
    }
    let placement = parse_threads(threads)?;
    let inner = circuit.inner.clone();
    let options = Options {
        backend: Backend::StateVector,
        ..Options::default()
    };
    let state = compute(py, placement, move || {
        qvd::statevector::<f64>(&inner, &options)
    })
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
    .0;
    let mut bytes = Vec::with_capacity(16 << circuit.inner.num_qubits);
    for amplitude in state.to_vec() {
        bytes.extend_from_slice(&amplitude.re.to_le_bytes());
        bytes.extend_from_slice(&amplitude.im.to_le_bytes());
    }
    Ok(PyBytes::new(py, &bytes))
}

/// Shots of a Clifford circuit (noisy or not) on the stabilizer backend,
/// measurement-major and bit-packed: classical bit `c` of shot `s` is bit
/// `s % 8` of byte `c * ceil(shots / 8) + s / 8`; see
/// `qevedo.simulator.sample`.
#[pyfunction]
#[pyo3(signature = (circuit, shots, seed = None, threads = None))]
fn sample_bytes<'py>(
    py: Python<'py>,
    circuit: &PyCircuit,
    shots: usize,
    seed: Option<u64>,
    threads: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyBytes>> {
    let placement = parse_threads(threads)?;
    let inner = circuit.inner.clone();
    let samples = compute(py, placement, move || {
        qvd::stabilizer::sample(&inner, shots, seed)
    })
    .map_err(value_error)?;
    let bytes = rows_to_bytes(
        (0..samples.num_clbits).map(|c| samples.clbit_words(c)),
        shots,
    );
    Ok(PyBytes::new(py, &bytes))
}

/// Rows of bit-packed words (one per classical bit, detector or
/// observable) as bytes, `ceil(shots / 8)` per row, with the bits past the
/// last shot cleared.
fn rows_to_bytes<'a>(rows: impl Iterator<Item = &'a [u64]>, shots: usize) -> Vec<u8> {
    let width = shots.div_ceil(8);
    let mut bytes = Vec::new();
    for row in rows {
        let start = bytes.len();
        for word in row {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes.truncate(start + width);
        if !shots.is_multiple_of(8) {
            bytes[start + width - 1] &= (1u8 << (shots % 8)) - 1;
        }
    }
    bytes
}

/// Detection events and observable flips of a circuit in Stim's format,
/// shot-major with one byte (0 or 1) per detector or observable, as two
/// bytearrays, with the number of detectors and of observables; see
/// `qevedo.simulator.sample_detectors`.
#[pyfunction]
#[pyo3(signature = (text, shots, seed = None, threads = None))]
#[allow(clippy::type_complexity)]
fn stim_detector_bytes<'py>(
    py: Python<'py>,
    text: &str,
    shots: usize,
    seed: Option<u64>,
    threads: Option<&Bound<'py, PyAny>>,
) -> PyResult<(
    Bound<'py, PyByteArray>,
    Bound<'py, PyByteArray>,
    usize,
    usize,
)> {
    let placement = parse_threads(threads)?;
    let circuit = qvd::stim::parse(text).map_err(value_error)?;
    let samples = compute(py, placement.clone(), move || {
        qvd::stabilizer::sample_detectors(&circuit, shots, seed)
    })
    .map_err(value_error)?;
    // Written straight into the bytearrays NumPy will view, without the GIL:
    // nothing else can see them yet. A result of a gigabyte or more is
    // common at large distances, and a second copy would cost as much as
    // the sampling.
    let fill = |len: usize, write: &(dyn Fn(&mut [u8]) + Sync)| {
        PyByteArray::new_with(py, len, |buffer| {
            compute(py, placement.clone(), || write(buffer));
            Ok(())
        })
    };
    let detectors = fill(shots * samples.num_detectors, &|b| {
        samples.write_detectors_shot_major(b)
    })?;
    let observables = fill(shots * samples.num_observables, &|b| {
        samples.write_observables_shot_major(b)
    })?;
    Ok((
        detectors,
        observables,
        samples.num_detectors,
        samples.num_observables,
    ))
}

/// Measurement results of a circuit in Stim's format (with `M(p)` flips
/// and inverted results applied), laid out as `sample_bytes`, with the
/// number of measurements.
#[pyfunction]
#[pyo3(signature = (text, shots, seed = None, threads = None))]
fn stim_sample_bytes<'py>(
    py: Python<'py>,
    text: &str,
    shots: usize,
    seed: Option<u64>,
    threads: Option<&Bound<'py, PyAny>>,
) -> PyResult<(Bound<'py, PyBytes>, usize)> {
    let placement = parse_threads(threads)?;
    let circuit = qvd::stim::parse(text).map_err(value_error)?;
    let samples = compute(py, placement, move || {
        qvd::stabilizer::sample_stim(&circuit, shots, seed)
    })
    .map_err(value_error)?;
    let bytes = rows_to_bytes(
        (0..samples.num_clbits).map(|c| samples.clbit_words(c)),
        shots,
    );
    Ok((PyBytes::new(py, &bytes), samples.num_clbits))
}

/// A circuit in Stim's format as a `Circuit` (one classical bit per
/// measurement, in Stim's order), with its detectors and observables as
/// lists of measurement indices. Measurement flips (`M(p)`) and inverted
/// results are not part of a `Circuit` and are rejected here.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn circuit_from_stim(text: &str) -> PyResult<(PyCircuit, Vec<Vec<usize>>, Vec<Vec<usize>>)> {
    let circuit = qvd::stim::parse(text).map_err(value_error)?;
    if !circuit.flips.is_empty() || !circuit.inverted.is_empty() {
        return Err(value_error(
            "measurement flips (M(p)) and inverted results (M !q, MPAD 1) have no Circuit \
             form; use sample_detectors or sample on the Stim text",
        ));
    }
    Ok((
        PyCircuit {
            inner: circuit.circuit,
        },
        circuit.detectors,
        circuit.observables,
    ))
}

/// The backend `run` would use for this circuit and these options.
#[pyfunction]
#[pyo3(signature = (circuit, backend = "auto", precision = "single", max_bond_dimension = 256, truncation_threshold = 1e-16))]
fn choose_backend(
    circuit: &PyCircuit,
    backend: &str,
    precision: &str,
    max_bond_dimension: usize,
    truncation_threshold: f64,
) -> PyResult<String> {
    let options = options(backend, None, max_bond_dimension, truncation_threshold)?;
    let chosen = match precision {
        "single" => qvd::choose_backend::<f32>(&circuit.inner, &options),
        "double" => qvd::choose_backend::<f64>(&circuit.inner, &options),
        _ => return Err(value_error("precision must be \"single\" or \"double\"")),
    };
    Ok(backend_name(chosen).to_string())
}

/// Whether qvd was compiled for AVX2 + FMA (x86-64-v3) kernels.
#[pyfunction]
fn uses_avx2() -> bool {
    cfg!(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCircuit>()?;
    m.add_class::<PyRunResult>()?;
    m.add_class::<PyExpectationResult>()?;
    m.add_function(wrap_pyfunction!(expectation, m)?)?;
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(statevector_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(sample_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(stim_detector_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(stim_sample_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(circuit_from_stim, m)?)?;
    m.add_function(wrap_pyfunction!(choose_backend, m)?)?;
    m.add_function(wrap_pyfunction!(uses_avx2, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
