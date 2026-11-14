# Benchmarks

Recorded results, the scripts that produced them, and the charts drawn from
them. Every comparison runs the same OpenQASM file, written by Qevedo, through
each simulator.

```
benchmarks/
├── results/
│   ├── machine.json            hardware, versions, measurement notes
│   ├── statevector.csv         dense simulation: Qevedo vs qsim vs Qiskit Aer
│   ├── stabilizer.csv          Clifford circuits: Qevedo vs Stim vs Qiskit Aer
│   ├── noise.csv               noisy (and noiseless) surface codes: Qevedo vs Stim
│   ├── noise_agreement.csv     detection rates of noisy codes: Qevedo vs Stim
│   ├── detectors*.csv          detection events of Stim's circuits: Qevedo vs Stim
│   ├── output.csv              records and detection events into NumPy: Qevedo vs Stim
│   ├── output_before_after.csv the same, two Qevedo builds interleaved (0.1.3 vs 0.1.4)
│   ├── mpp*.csv                MPP toric-code memory with correlated errors: Qevedo vs Stim
│   ├── examples_before_after*.csv the Rust examples, two builds interleaved (0.1.10 vs 0.1.11; the recheck file 0.1.11 vs 0.1.12 on the binaries that changed)
│   ├── mps.csv                 matrix product states: Qevedo vs Qiskit Aer vs quimb
│   ├── nearclifford.csv        Clifford+T circuits: Qevedo vs Qiskit Aer
│   ├── noisy_nearclifford*.csv noisy Clifford+T circuits: Qevedo vs Clifft, and two Qevedo builds
│   ├── pauliprop.csv           expectation values: Qevedo vs IBM's pauli-prop
│   └── stabilizer_steps.csv    how each optimisation changed the d=101 time
├── charts/                     SVGs drawn from results/ by plot.py
├── compare.py                  times Qiskit Aer and qsim on a QASM file
├── compare_stim.py             times Stim and Qiskit Aer's stabilizer method
├── bench_noise.py              noisy surface codes against Stim -> noise*.csv
├── bench_detectors.py          Stim's circuits, detection events -> detectors*.csv
├── bench_output.py             sample / sample_detectors, bool and bit-packed -> output.csv
├── compare_builds.py           two builds interleaved -> output_before_after.csv
├── bench_mpp.py                MPP, PAULI_CHANNEL_2 and correlated errors -> mpp*.csv
├── compare_examples.py         the Rust examples of two builds interleaved -> examples_before_after.csv
├── compare_mps.py              times Qiskit Aer's and quimb's MPS simulators
├── compare_nearclifford.py     times Qiskit Aer's methods on Clifford+T circuits
├── bench_noisy_nearclifford.py noisy Clifford+T circuits against Clifft -> noisy_nearclifford*.csv
├── compare_noisy_builds.py     the same circuits, two builds interleaved -> noisy_nearclifford_before_after.csv
├── compare_pauliprop.py        times IBM's pauli-prop on an expectation value
├── bench_pauliprop.py          runs Qevedo and pauli-prop on every case -> pauliprop.csv
└── plot.py                     results/*.csv -> charts/*.svg
```

The CSV files are in long format (one row per simulator and case), with
times in seconds, so they can be loaded directly by other tools or pages.

## Machine

Intel Core i9-14900K (8 P-cores with hyper-threading + 16 E-cores, 32
threads, AVX2 + FMA, no AVX-512), 62 GB DDR5, Linux 7.0, Rust 1.96. Python
packages: qiskit 2.5.2, qiskit-aer 0.17.2, qsimcirq 0.22.1, cirq-core 1.7.0,
stim 1.16.0, quimb 1.15.0. Measured on 2026-10-09.

## State-vector simulation

![State-vector benchmark](charts/statevector.svg)

Random circuits are depth-20 Google-style circuits (√X, √Y, √W on every
qubit and CZ on a brick pattern); QFT is the quantum Fourier transform.
Each simulator gets its best of 16 or 32 threads. Qevedo's times include
allocating the state; the others include sampling one shot, and qsim's
include converting the circuit in Python. qsim's CPU backend is single
precision only. Amplitudes agree with Qiskit Aer to 7×10⁻¹³ in double
precision.

## Clifford circuits

![Stabilizer benchmark](charts/stabilizer.svg)

Rotated surface-code memory experiments, d rounds, 10,000 shots. Qevedo ran
on a pool of 8 P-cores and on one thread; Stim is single-threaded. Stim is
timed two ways:
- `reference_sample()` plus a `FlipSimulator`, whose results are
  measurement-major like Qevedo's (the fair comparison);
- `compile_sampler().sample(bit_packed=True)`, which returns shot-major
  arrays and is much slower at these sizes.

Qiskit Aer's stabilizer method was not run past d = 21 (it took 636 s
there). A virtual machine shared the machine: Stim, Aer and Qevedo on one
thread ran while it used about six cores, and Qevedo on 8 P-cores was
re-measured after the parallel reference shot while it used about 14. So
absolute times are a little high, and pessimistic for Qevedo's 8-core column.

![Stabilizer optimisation steps](charts/stabilizer_steps.svg)

The steps that took the distance-101 code (20,401 qubits, 101 rounds) from
37 s to 1.5 s. Steps 1–7 were measured on a quiet machine. Step 8 was
measured under the heavier load; under that same load the build before it
took 2.57 s. [docs/DESIGN.md](../docs/DESIGN.md)
explains each one.

Stim's times here come from circuits with one instruction per line, as
`compare_stim.py` wrote them then; that slows Stim by 1.7× at d = 21 and
1.07× at d = 51 (measured on the 4-vCPU VM below). The script now groups
consecutive instructions, and the next section has the fair comparison.

## Noisy Clifford circuits

![Noisy stabilizer benchmark](charts/noise.svg)

The surface codes above with uniform circuit-level depolarizing noise,
p = 0.001 (`library::with_depolarizing_noise`), and without it (p = 0 rows
in `noise.csv`), 10,000 shots. Measured on the 4-vCPU Xeon virtual machine
(`pauliprop_machine` in `machine.json`), not the i9. Both simulators read
the same Stim file written by the `stabilizer` example (`--stim`), with one
instruction per layer and channel as in Stim's generated circuits. Qevedo ran
on 4 threads and on 1 (best of 3); Stim on one thread, best of 3 (one run
at d = 101). `noise_agreement.csv` compares the fraction of shots in which
each detector fires (XOR of consecutive measurements of one ancilla),
Qevedo through the Python package against Stim's sampler, 200,000 shots each,
with a two-sample z-score per detector.

## Stim circuits and detection events

Stim's own generated circuits (`surface_code:rotated_memory_z`, d rounds,
all four noise parameters at p = 0.001), read by Qevedo's Stim reader, against
Stim's detector sampler, 10,000 shots (`detectors.csv`). Measured on a
4-vCPU Xeon virtual machine (`detectors_machine` in `machine.json`). Both
run in one Python process and return shot-major NumPy bool arrays; Qevedo on
4 threads and on 1, its time including reading the text; Stim on one
thread, with and without `bit_packed=True`; best of 3. Distance 101 is
left out: a shot-major result of a million detectors × 10,000 shots is
10 GB in either simulator. `detectors_agreement.csv` compares detection
rates (200,000 shots, a two-sample z-score per detector) and the logical
error rates after the same PyMatching decoder; `detectors_long.csv` repeats
the smallest code with 24 million shots per simulator, per detector.

## Output into NumPy

`bench_output.py` (`output.csv`) times what a Python user gets back from
the same generated codes (p = 0.001, 10,000 shots, best of 3; once at
d = 101): raw measurement records from Qevedo's `sample` (measurement-major,
bool or eight shots per byte) against Stim's `compile_sampler().sample`
(shot-major, bool or bit-packed; Stim has no measurement-major output
from its sampler), and detection events from both simulators, shot-major,
bool and bit-packed. At d = 101 only the bit-packed outputs run, since a
bool array would be 10 GB. Same 4-vCPU machine type as `detectors.csv`,
measured on 2026-11-02 with numpy 2.5.4 and stim 1.16.0.

`compare_builds.py` (`output_before_after.csv`) runs two builds of the
package (0.1.3 from before this output work, and 0.1.4) in turn, each case
in a fresh process, best of 3 calls over 3 rounds, to show what the change
did on the same machine at the same time.

## Pauli product measurements and correlated errors

`bench_mpp.py` writes a toric-code memory in Stim's format (`toric_mpp`
in the script): L×L toric code, 2L² data qubits and no ancillas, every
stabilizer measured directly with `MPP(p)` for L rounds. Before each
round: `DEPOLARIZE1(p)` on every qubit, a biased `PAULI_CHANNEL_2` (ZZ at
p/3, XX at p/6, the others at p/30) on the horizontal and vertical edge
of every vertex, and L correlated chains `E(p) X X;
ELSE_CORRELATED_ERROR(p) Z Z` on random pairs. `mpp.csv`: 10,000 shots of
detection events, shot-major bool, best of 3; Qevedo on 4 threads and 1,
Stim's detector sampler on one. `mpp_agreement.csv`: detection rates and
logical error rates after PyMatching (from Stim's detector error model,
`approximate_disjoint_errors=True` for the chains), 200,000 shots, plus
the number of detection events in 10,000 noiseless shots.
`mpp_long.csv` (`--long`): 4–12 million shots per simulator and case.
Same 4-vCPU machine type as `detectors.csv`, measured on 2026-11-04 with
stim 1.16.0 and pymatching 2.4.0.

## Before and after a change (Rust examples)

`compare_examples.py` runs the benchmark examples (`bench`, `shots`,
`mps`, `nearclifford`, `pauliprop`, `stabilizer`, including its Stim
`detectors` mode) from two directories of binaries in alternation, 9
rounds, and records each build's best time and the median of the
per-round ratios. `examples_before_after.csv` compares 0.1.10 with the
build released as 0.1.11 on the 4-vCPU VM, which changed only the noisy
near-Clifford path (sign registers) and added one tableau function: best-of
ratios 0.903-1.152, medians 0.920-1.054. The `bench`, `stabilizer` and
`pauliprop` binaries have byte-identical machine code (`.text`) to 0.1.10's
sources built with the version number bumped (the bump alone changes
symbol hashes and so the layout), so their cases, including readings up to
1.069 best-of, are the VM's noise. `examples_before_after_mps_recheck.csv`
reran the two `mps` cases of that comparison for 60 rounds: 0.987 and
1.000 best-of, 1.001 and 1.000 median. `examples_before_after_recheck.csv`
compares 0.1.11 with the build released as 0.1.12 (no split where a
rotation brings a new qubit into the dense part; again only the noisy
near-Clifford path and one tableau function changed), both with the
version 0.1.12: `bench`, `stabilizer`, `pauliprop`, `kernels`, `l2` and
`bandwidth` have byte-identical `.text`, and the four binaries that differ
(`shots`, `mps`, `nearclifford`, and `quickstart`, which is not timed) ran
15 rounds: best-of 0.919-0.992, medians 0.958-1.028.

## Noisy near-Clifford, before and after

`compare_noisy_builds.py` times the circuits of `bench_noisy_nearclifford.py`
(and three noise rates of its sweep) from two Python environments with the
two builds installed, in alternation, 5 rounds, best of 3 per process.
`noisy_nearclifford_before_after.csv` compares 0.1.11 with the build
released as 0.1.12 (a shot whose error flips the sign of a rotation that
brings a new qubit into the dense part takes a Z on that qubit into its
sign register instead of splitting off), both built for x86-64-v3 as the
wheels are, on the 4-vCPU VM: the after / before ratio is 0.285-0.704
best-of (0.292-0.700 median paired) on the ten noisy cases of the main
table, 0.285 and 0.293 on the 100-qubit circuit at p = 10⁻³ (4 threads,
one) and 0.363 and 0.324 on 60 qubits at p = 10⁻²; on the sweep, 0.873 at
10⁻⁵, 0.581 at 10⁻⁴, and 0.969 without errors. The random
draws differ (fewer splits), so the counts for a seed differ too; the
distributions are checked against exact ones in the tests and against
Clifft below. `noisy_nearclifford_before_after_1shot.csv` (`--shots 1`,
best of 15 per process) shows the fixed cost of a run: 0.921-1.046
best-of, noise around 1; `noisy_nearclifford_before_after_1shot_recheck.csv`
reruns the 200-qubit case, whose median read 1.048, for 20 rounds: 1.025
and 1.015 best-of, 1.011 and 1.012 median. `noisy_nearclifford_before_after_recheck.csv`
is from an older comparison (0.1.9 with 0.1.10, the two 24-qubit cases
with 25 rounds: 0.87 and 0.90 best-of). (0.1.11 against 0.1.10 was
0.24-0.77; 0.1.10 against 0.1.9 0.60-0.91; 0.1.9 against 0.1.8
0.25-0.59.) `--qubits N` keeps only the
cases of N qubits.

## Matrix product states

![MPS benchmark](charts/mps.svg)

1000 shots, χ capped as labelled and the smallest singular values dropped
while their squares sum below 10⁻¹⁶, in all three simulators. Qevedo ran on 8
P-cores (best of three), Qiskit Aer with 8 threads, quimb's
`CircuitPermMPS` with 8 BLAS threads after a warm-up. Aer reports only a
total; `mps.csv` has gate and sampling times for the other two, and the
fidelity estimates. A busy virtual machine shared the machine during these
runs, for all three alike.

## Clifford+T circuits

![Clifford+T benchmark](charts/nearclifford.svg)

Random Clifford+T circuits from `library::random_clifford_t` (layers of
random one-qubit Cliffords and CX on random pairs, T gates on random
qubits), 1000 shots, 8 threads. Qevedo's near-Clifford time is the best of
three; every Aer method ran in its own process, stopped at 600 s. Aer's
`extended_stabilizer` is approximate and rejects circuits over 63 qubits;
there is no state vector past 30 qubits. A virtual machine shared the
machine during these runs.

## Noisy Clifford+T circuits

`bench_noisy_nearclifford.py` builds random Clifford+T circuits with
uniform depolarizing noise (`DEPOLARIZE1(p)` after each one-qubit gate,
`DEPOLARIZE2(p)` after each CX, `X_ERROR(p)` before each measurement) and
runs the same circuit on Qevedo's near-Clifford backend (Python `Circuit`) and
on Clifft (Stim-format text with `T` and `T_DAG`). `noisy_nearclifford.csv`:
10,000 shots, best of 3 in one process, 4 threads and 1. Qevedo's time
includes building its counts dict; Clifft's includes compiling.
`noisy_nearclifford_sweep.csv` (`--sweep`): the 100-qubit circuit at p = 0
to 10⁻³ on one thread. `noisy_nearclifford_agreement.csv`: 200,000 shots
each; per-qubit rates with two-sample z-scores, and on up to 8 qubits a
chi-square test of homogeneity of the whole distribution.

## Expectation values (Pauli propagation)

![Pauli propagation benchmark](charts/pauliprop.svg)

`<Z62>` of IBM's 127-qubit kicked Ising circuit (`library::kicked_ising`
on `library::eagle_heavy_hex`), the same coefficient threshold in both.
Measured on a different machine from the rest of this page: a cloud
virtual machine with 4 vCPUs of an Intel Xeon at 2.80 GHz and 15 GB
(`pauliprop_machine` in `machine.json`). Qevedo ran on 4 threads and on 1
(best of 3); pauli-prop 0.2.1 ran once per case in its own process, on
one thread (its CPU time matched its wall time), with `max_terms` well
above what each case needs so that only the threshold truncates. Its
"propagation only" time is its Rust propagation through the rotations;
its total adds moving the Cliffords out in Python and evolving the final
strings through them in Qiskit. Both return the same value, the same
number of strings and the same dropped one-norm in every case.

## Reproducing

```sh
python3 -m venv .bench-venv
.bench-venv/bin/pip install qiskit qiskit-aer qsimcirq cirq-core ply stim matplotlib quimb

# State vector: Qevedo's time is printed; the QASM file feeds the others.
cargo run --release --example bench -- random 28 20 f32 4 14 --qasm /tmp/c.qasm
.bench-venv/bin/python benchmarks/compare.py /tmp/c.qasm --precision single --threads 32

# Clifford: surface code of distance 51, 51 rounds, 10,000 shots.
cargo run --release --example stabilizer -- surface 51 51 10000 --qasm /tmp/s.qasm
.bench-venv/bin/python benchmarks/compare_stim.py /tmp/s.qasm --shots 10000

# Noisy surface codes against Stim (and the Python package, for agreement).
cargo build --release --example stabilizer
(cd python && maturin develop --release)
.bench-venv/bin/python benchmarks/bench_noise.py

# Detection events of Stim's generated circuits against Stim
# (pip install pymatching; the Python package built as above).
.bench-venv/bin/python benchmarks/bench_detectors.py --distances 11,21,51
.bench-venv/bin/python benchmarks/bench_detectors.py --long

# MPP, PAULI_CHANNEL_2 and correlated errors: a toric-code memory against Stim.
.bench-venv/bin/python benchmarks/bench_mpp.py
.bench-venv/bin/python benchmarks/bench_mpp.py --long

# Records and detection events into NumPy, bool and bit-packed, against Stim.
.bench-venv/bin/python benchmarks/bench_output.py
# Two builds of the package, interleaved (each venv has one build installed).
.bench-venv/bin/python benchmarks/compare_builds.py --before /venv-old/bin/python --after /venv-new/bin/python

# MPS: 64 qubits, depth 20, χ ≤ 256, 1000 shots.
cargo run --release --example mps -- random 64 20 256 1000 --qasm /tmp/m.qasm
.bench-venv/bin/python benchmarks/compare_mps.py /tmp/m.qasm --max-bond 256 --shots 1000

# Clifford+T: 100 qubits, depth 30, 24 T gates, 1000 shots.
cargo run --release --example nearclifford -- 100 30 24 1000 --qasm /tmp/n.qasm
.bench-venv/bin/python benchmarks/compare_nearclifford.py /tmp/n.qasm --shots 1000 \
    --methods matrix_product_state,extended_stabilizer

# Pauli propagation: every case, Qevedo and pauli-prop (pip install pauli-prop).
cargo build --release --example pauliprop
.bench-venv/bin/python benchmarks/bench_pauliprop.py

# After editing results/*.csv:
.bench-venv/bin/python benchmarks/plot.py
```

`bench` takes the circuit (`random`, `qft`, `ghz`), qubits, depth, precision
(`f32`, `f64`), maximum fused qubits and region bits (0 turns cache blocking
off); `QVD_PLACEMENT=pcores|pthreads|allcores|all` selects the threads. The
`stabilizer` example takes `surface <distance> <rounds> <shots>` or
`random <qubits> <depth> <shots>`, with `--noise P` for depolarizing noise
and `--stim FILE` to write the circuit in Stim's format, or
`detectors <Stim file> <shots>` to time reading a Stim file, sampling its
detectors and making them shot-major;
`QVD_THREADS=pcores|all|<n>` selects the threads (P-cores by default). The `mps` example takes
`<random|grid|qft|ghz> <qubits | ROWSxCOLS> <depth> <max bond> [shots]`, and
the `nearclifford` example `<qubits> <depth> <T count> [shots] [--backend nc|sv|mps]`,
and the `pauliprop` example `<steps> <θ_h> <observable> [threshold] [max terms]`
(`QVD_THREADS=<n>` sets its threads).
