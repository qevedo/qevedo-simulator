# Improvement log

One entry per improvement run: what changed, why, what it measured, and
what to try next. Newest first.

## 2026-11-06 — Pauli noise on the near-Clifford backend (0.1.7)

**What.**
- The near-Clifford backend samples every channel of `qvd::Noise`, so
  noisy circuits that are not Clifford run (before, every backend refused
  them). `Backend::Auto` sends them there when at most 28 qubits are ever
  active. New module `src/nearclifford/noisy.rs`.
- Each shot is in the state `C P_s (φ_g ⊗ |0...0>)`, after Clifft (Chase &
  Labib, arXiv:2604.27058): the Clifford frame `C` shared by a batch of
  shots, the dense part `φ_g` by a group, a Pauli frame `P_s` per shot.
  Errors multiply `C† E C` into `P_s` (geometric gaps between errors, as in
  the stabilizer sampler). A rotation acts on `φ_g` as `exp(+iθ/2 P')` in
  the shots whose frame anticommutes with `P'`, so a group splits there;
  a dense measurement with both outcomes splits it too. The X that
  retiring a qubit would prepend to `C` for outcome 1 goes into the shots'
  frames instead, so all groups of a batch share `C` and advance in
  lockstep: each Clifford, gathering and retirement is done once per batch.
  Mid-circuit measurements and resets work on this path.
- The noiseless final-measurement sampler now uses the same `Group` type
  (frames, records, an alive mask, `split_off`). Its counts are identical
  to 0.1.5's and 0.1.6's for the same seed (checked against the 0.1.5
  wheel; 0.1.6 changed only the Python layer).
  `retire_plan` and `virtual_pair` were pulled out of `NearClifford`.
- `Noise::error` picks a channel's error for a draw (the readable form of
  the stabilizer sampler's inlined loop).
- Tests: noisy non-Clifford circuits with every channel, mid-circuit
  measurement and reset, against the exact distribution summed over every
  error; 300 small circuits on the noisy path with measurements and resets
  against exact branching; a 150-qubit circuit with known flip rates;
  reproducibility across thread counts; automatic selection. The test
  helper `random_noise` moved to `tests/common`. Two tests that asserted
  the old refusal (`noise_needs_the_stabilizer_backend`, and the Python
  `test_noise_errors_and_stim_text`) now assert that the near-Clifford
  backend runs noise and the state vector and MPS still refuse it.
- `benchmarks/bench_noisy_nearclifford.py` against Clifft 0.13.

**Why.** Every previous log named this first (noisy magic-state circuits:
cultivation, distillation, injection, whose study is what Clifft, SOFT and
Tsim were built for), and the simulator's frames were already per shot for the final
measurements. It is exact and could be verified end to end today. The
other candidates (`MXX`/`MYY`/`MZZ`, `SPP`, heralded errors; detectors
without keeping every record) widen formats or save memory on what
already runs; this runs a class of circuits nothing in the simulator could.

**Measured** (4-vCPU Xeon VM, 2.1 GHz).
- Against Clifft, 10,000 shots of random Clifford+T circuits with
  depolarizing noise (`results/noisy_nearclifford.csv`): **Clifft is faster
  on every case**, 3.4-17.8×; at 200 qubits 1.5× on 4 threads (Qevedo 1.25 s,
  Clifft 0.82 s) and 5.1× on one. Without errors the two are close (100
  qubits, 14 T: 0.33 s against 0.23 s); Qevedo's time grows with the shots
  errors split off (6.73 s at p = 10⁻³), Clifft's does not (0.33-0.35 s)
  (`results/noisy_nearclifford_sweep.csv`).
- Agreement with Clifft, 200,000 shots: chi-square p-values 0.385, 0.893,
  0.861 on the whole distributions of 6- and 8-qubit circuits; per-qubit
  rates on 24 and 60 qubits, largest |z| 1.79 and 3.33, mean z² 0.84 and
  0.93.
- No regression: `compare_examples.py`, 0.1.5 (the Rust core of 0.1.6)
  against this build, 9
  rounds, then 25 for the six cases with a ratio above 1.03
  (`results/examples_before_after*.csv`). Every median paired ratio in the
  rerun is at most 1.026; best-of 1.032 (near-Clifford, 100 qubits) and
  1.030 (stabilizer d = 51).

**Learned.**
- The first build of this made the noiseless 100-qubit near-Clifford
  example 5-9% slower with identical counts and identical instruction
  counts in every hot function (callgrind); built for AVX2 it was 5%
  faster. Fat LTO with one codegen unit places code globally, and the new
  noise code moved the dense kernels. Marking the noise path `#[cold]` and
  `#[inline(never)]` brought it back, as `Frames::noise` is kept out of
  line in the stabilizer sampler. Comparisons of this kind need paired,
  randomly ordered runs: this VM drifted by ±10% between batches.
- A first version gave every group its own copy of `C`. Sharing `C` made
  the 200-qubit case 2.8× faster on 4 threads but the 60- and 100-qubit
  ones 14-40% slower (every operation sweeps all groups' frames); summed,
  15.2 s against 22.7 s on one thread. Neither is near Clifft once errors
  split the shots: the cost is per split-off shot (a copy of `φ`, frame
  words, a sweep per operation).
- A bug in the first lockstep version (each shot's flips for a 50/50
  measurement taken after the collapse had conjugated the frames) passed
  33 of the 40 exact-distribution cases; the targeted test with 300 small
  circuits catches it at once.

**Next.**
1. A per-shot path for the noisy near-Clifford backend once errors have
   split a shot off: run such shots one at a time through the operations
   precompiled once for the batch (the pulled-back Paulis, retire plans
   and error images), with only `φ` and `P_s` per shot, as Clifft does.
   Sampling the final measurements of each group in one pass over its `φ`
   (diagonalize the commuting measured Paulis, then draw basis states)
   would cut the per-shot dense work further.
2. Read Stim's format (with `T`, `T_DAG` and the rotation gates Clifft
   adds) and return detection events on the near-Clifford backend, so
   magic-state cultivation circuits run directly.
3. `MXX`/`MYY`/`MZZ`, `SPP` and heralded errors in the Stim reader, and
   `to_stim` writing detectors and observables.

## 2026-11-06 — Qiskit backend and primitives (0.1.6)

**What.** `python/python/qevedo/simulator/qiskit.py`, a new module:
- `from_qiskit(QuantumCircuit)`: every standard Qiskit gate. The 37 that the
  simulator knows by name are read directly, the others (`ecr`, `iswap`, `ryy`,
  `rzx`, `xx_plus_yy`, ...) as their exact matrices (up to 4 qubits), and
  anything else (QFT, MCX, `UnitaryGate`, `PauliEvolutionGate`, custom
  gates) through its definitions. `box` is inlined. Control flow and
  unbound parameters are rejected. The global phase is tracked so state
  vectors equal Qiskit's exactly.
- `QevedoSimulator` (`BackendV2`): a target of all standard gates with no
  coupling map (1024 qubits by default), Aer-like options (`method` takes
  Aer's names too), `memory`, register-formatted counts, and the method
  that ran in each result's metadata. Qiskit's `BackendSamplerV2` and
  `BackendEstimatorV2` work on it.
- `QevedoSampler` (`SamplerV2`): one `BitArray` per register, with
  mid-circuit measurements and resets (Qiskit's reference sampler rejects
  both). The simulator returns counts, so the shots are put in a seeded random order:
  for independent shots this has the same distribution as the order they
  were drawn in.
- `QevedoEstimator` (`EstimatorV2`): exact on the state vector up to 24
  qubits (one simulation per parameter set for all observables), Pauli
  propagation beyond that, with the dropped one-norm in
  `data.truncated_one_norm`. Gates the simulator stores as multi-qubit matrices
  go through their definitions for propagation.
- `run`, `statevector` and `expectation` accept a `QuantumCircuit`.
- New `[qiskit]` extra; Qiskit added to `[test]` so CI runs
  `tests/test_qiskit.py`.

**Why.** Roadmap item 5 in the workspace plan (the simulator as a Qiskit backend),
which is the way most users would try a simulator.

**Measured** (i9-14900K, 32 threads, shared with other jobs;
`benchmarks/bench_qiskit.py`, best of 3, through Qiskit's interfaces on
both sides). With `backend.run`, 1000 shots: Quantum Volume 20 / 24 / 26 /
28 took 0.041 / 0.264 / 1.23 / 5.91 s against Aer's 1.31 / 3.36 / 10.4 /
39.0 s (32x / 13x / 8.5x / 6.6x). QFT 24 / 28 was 20x / 8.0x faster.
Converting the circuit takes 3–12 ms. A 500-qubit Clifford circuit with 10k
shots took 0.023 s against 80 s for Aer's stabilizer method, which
simulates every shot again. With the estimator on QV 16 / 20 / 22 and 50
Pauli observables: 30x / 16x / 5.5x faster than Aer's `EstimatorV2`, with
values within 6e-17. Tests: every standard gate against `Statevector`
(phase included) on both the state-vector and propagation paths, plus
composite gates, distributions, registers, memory, seeds and broadcasting.
They pass on Qiskit 2.5.2 and 2.0.3.

**Next.**
1. Aer noise models (at least Pauli errors on Clifford circuits, into
   the simulator's channels).
2. Classical control flow (`if_test`) on the backends that already
   measure mid-circuit.
3. Cirq and PennyLane devices.

## 2026-11-04 — Pauli product measurements, PAULI_CHANNEL_2 and correlated errors (0.1.5)

**What.**
- The Stim reader takes `MPP` (products such as `X0*!Y1*Z2`, several per
  line, with a flip probability `MPP(p)`). Each product becomes the
  rotations of `MX`/`MY`, a CX ladder onto its last qubit, a Z
  measurement and the same gates undone (`Builder::mpp`). The tableau and
  the frame loop are unchanged.
- Two new channels in `qvd::Noise`: `Pauli2` (`PAULI_CHANNEL_2`, boxed)
  and `Correlated` (Pauli products with probabilities). The reader merges
  `E` and the `ELSE_CORRELATED_ERROR` lines after it (annotations may
  come between) into one `Correlated` over the union of their qubits,
  with unconditional probabilities `p_i ∏_{j<i}(1 − p_j)`. `to_stim`
  writes it back as the chain, and `PAULI_CHANNEL_2` under its own name.
  `Frames::weighted_error` (out of line) picks a Pauli for both; the
  common channels' loop did not change. `Noise::check` and
  `Noise::probabilities` replace the copies of that validation in the
  circuit builder and the bindings. `Noise` is no longer `Copy`.
- `Circuit::pauli_channel_2` in Rust and Python.
- Tests: `MPP` against the reference state vector on 200 random Clifford
  circuits, a Bell pair, flips and inversions; `PAULI_CHANNEL_2`'s order;
  ELSE chain rates; round trips through `to_stim`; reader errors. The
  exact-distribution tests (summed over every error) now draw `Pauli2`
  and `Correlated` channels too, and the rate test has a case for each.
  Python: 300 random `MPP` circuits against Stim (same affine space of
  outcomes, same per-bit rates); a toric code with all three features,
  detector by detector against Stim, and silent without noise.
- `benchmarks/bench_mpp.py` (`results/mpp*.csv`) and
  `benchmarks/compare_examples.py` (`results/examples_before_after*.csv`).

**Why.** Item 2 of the previous entry's list. These were the
instructions most likely to stop a Stim file in Qevedo's reader. `MPP` is
how Floquet codes (the honeycomb code), lattice surgery and
measurement-only surface and toric codes are written. `PAULI_CHANNEL_2`
and correlated errors are how calibrated, Pauli-twirled hardware noise
models arrive. A search of recent stabilizer-simulation work (QuaSARQ,
arXiv:2603.14641; Sdim, arXiv:2511.12777; STABSim, arXiv:2507.03092)
found no new sampling method that beats reference-plus-frames on CPUs.
So making more real circuits run, each checked exactly against Stim and
a state vector, was the most valuable change that could be verified end
to end today. Item 1 (noise on near-Clifford frames) serves fewer users.

**Measured** (4-vCPU Xeon VM at 2.1 GHz, complete runs of both scripts).
Toric-code memory with `MPP(p)`, `PAULI_CHANNEL_2` and E/ELSE chains,
p = 0.001, 10,000 shots of detection events: L = 10 / 20 / 30, Qevedo on 4
vCPUs 0.021 / 0.162 / 0.787 s, on 1 thread 0.019 / 0.173 / 0.898 s;
Stim's detector sampler 0.083 / 1.116 / 4.310 s (4.0–6.9× on 4 vCPUs,
4.4–6.5× on one thread). Agreement (200,000 shots): detection rates
within sampling error, no detection event without noise; logical error
rates |z| 0.41–0.88 except L = 5, p = 0.002 (15 against 2 failures,
z = 3.15). The long runs (`mpp_long.csv`, 4–12 million shots per simulator): logical
z = −1.90, 2.03, 0.88 and 0.91; largest detector |z| 2.81–3.39; mean
detector z² 0.99–1.05, and 1.44 over the 54 detectors of L = 3,
p = 0.01.

No regression. `compare_examples.py`, 0.1.4 against this build,
interleaved, 9 rounds: best-of ratios 0.885–1.046, every stabilizer and
detector case 0.975–1.000. Three cases came out above 1.03 (pauliprop 20
steps, 1.038; pauliprop 5 steps, 1.046 with median 1.079; mps QFT-100,
1.046, 68 against 65 ms). Rerun with 25 rounds
(`examples_before_after_recheck.csv`): 1.014, 0.977 and 0.970 (medians
1.012, 1.018, 0.986). In an earlier, unrecorded
pass, the same pauliprop cases came out 0.951 and 0.950. Their code does
not touch noise, so these are VM noise.

**Learned.**
- Two slowdowns came from data size and inlining, not from the new code
  paths. Storing the 15 probabilities inline tripled `Noise`'s size, and
  so every `Instruction`'s (to 150 bytes): noiseless d = 51 sampling came
  out 4–6% slower over 15–21 interleaved pairs. Boxing the array brought
  `Noise` back to 32 bytes and `Instruction` to 64; a test now pins both.
  Separately, `PendingNoise::add`, with the channel's clone, was inlined
  into the reference shot's loop and grew it by 1 KB; `#[inline(never)]`
  restored it.
- In a statistics check with few events, |z| ≈ 3 happens. The first long
  run (10 seeds, since replaced by the 30-seed run) gave logical z =
  −2.94 and +3.46 at L = 5 (opposite signs);
  unrecorded reruns with other seeds gave 0.83, −0.18, −1.89 and −1.65,
  and at p = 0.01 (50,000 failures) −1.8 and +1.35. Pairwise detector
  coincidences (ad hoc, all 1,485 pairs at L = 3, p = 0.01, 2 million
  shots, with and without the new channels) showed mean z² 0.91–1.03.
- My first random-circuit check compared sets of distinct outcomes,
  which fails when there are more outcomes than shots; the affine span is
  the right invariant.

**Next.**
1. Noise on the near-Clifford backend's per-shot frames (noisy
   magic-state circuits).
2. `MXX`/`MYY`/`MZZ`, `SPP` and heralded errors in the Stim reader, and
   `to_stim` writing detectors and observables.
3. Detection events without keeping every measurement record (memory at
   d = 101 is dominated by the 1.3 GB record).

## 2026-11-02 — Results into NumPy without the Python-side work, and bit-packed detection events (0.1.4)

**What.**
- `sample_detectors(..., bit_packed=True)`: shot-major uint8 arrays with
  eight detectors (or observables) per byte, detector `d` in bit `d % 8`
  of byte `d // 8`, as Stim's `sample(bit_packed=True)`; without
  `separate_observables` the detectors and observables are packed
  together, as in Stim. In Rust: `DetectorSamples::
  write_detectors_shot_major_packed`, `write_observables_shot_major_packed`
  and `write_shot_major(packed, out)` (observables appended, bool or
  packed), all from one const-generic transpose.
- `Samples::write_bytes` and `write_packed`: measurement-major records as
  one byte per shot (a 256-entry table expands each byte into eight) or
  eight shots per byte, in parallel by rows. `sample` no longer unpacks
  with NumPy.
- The binding asks Python for `numpy.empty` arrays of the final size
  before sampling (sizes follow from the circuit), checks each one's
  `__array_interface__`, and samples and writes into them in one visit to
  a thread pool built once per call. No bytearray, no copy, no
  `np.unpackbits`, no `np.concatenate`.
- `benchmarks/bench_output.py` (`results/output.csv`) and
  `benchmarks/compare_builds.py` (`results/output_before_after*.csv`).

**Why.** Item 1 of the previous entry's list. The previous entry measured
12 s for `sample` on the d = 51 code. On this machine (numpy 2.5.4) the
0.1.3 build took 1.6–2.5 s for that call, so the measured weakness was
smaller than recorded: about half the time went to unpacking in NumPy
rather than sampling. The bit-packed detector output adds a capability: at
d = 101, 10,000 shots of detection events are 10 GB as bools and do not
fit this 15 GB machine, but they are 1.3 GB packed, the layout decoders
and Stim's users already handle. A search of recent work (stabilizer
simulators on GPUs, Quantum 2026; STABSim, arXiv:2507.03092; SymPhase)
found no detector-sampling method that would beat Qevedo's current approach
on CPUs. Making the Python results as fast as the Rust ones was the most
useful step that could be verified end to end today. The other candidates
(noise on near-Clifford frames, `MPP` in the reader) are features fewer
people use.

**Measured** (4-vCPU Xeon VM at 2.1 GHz; complete runs of both scripts).
Against 0.1.3, interleaved, each case in a fresh process
(`output_before_after.csv`, best of 3 × 3 rounds), 4 threads / 1 thread:

| d = 51 | 0.1.3 | 0.1.4 | ratio |
|---|---|---|---|
| `sample` (bool) | 1.70 / 2.52 s | 0.70 / 1.39 s | 0.41 / 0.55 |
| `sample(bit_packed=True)` | 0.91 / 1.46 s | 0.68 / 1.29 s | 0.75 / 0.88 |
| `sample_detectors` (bool) | 1.76 / 2.49 s | 1.18 / 2.16 s | 0.67 / 0.87 |
| same, observables appended | 3.25 / 5.24 s | 1.13 / 1.93 s | 0.35 / 0.37 |

At d = 21 the ratios are 0.48–0.97; at d = 11 (calls of 6–25 ms, also
timed with 15 repeats in `output_before_after_small.csv`) 0.40–0.93,
except bit-packed records on one thread, 1.03 (8.1 against 8.4 ms).
Longer development checks of that case (hundreds of calls per build,
not recorded) put it at about 2%. In the 3-repeat file the same case on
4 threads reads 1.70 (6.9 against 11.7 ms). The 15-repeat run gave 0.93,
and so did the longer checks, so it was a spike on the VM, not a
regression.

Against Stim 1.16 (`output.csv`), detection events, shot-major: bool 9.0×
faster at d = 51 on 4 vCPUs (1.13 s against 10.1 s; 5.1× on one thread);
bit-packed, 13.6 s against 303 s at d = 101 (34.0 s on one thread). Raw
records at d = 51: 0.72 s against 11.8 s for Stim's shot-major sampler.

The Rust benchmark binaries: `bench`, `mps`, `nearclifford`, `pauliprop`,
`shots` and `kernels` have byte-identical machine code before and after
(`.text` compared). `stabilizer` differs only in the detector writers;
timed with the example itself, interleaved, best of 5, its cases came
out 0.94–1.02, and noisy d = 51 on one thread 1.055. Its code there is unchanged, only moved by 224 bytes,
so 20 more interleaved pairs were run on that case: median ratio 0.98,
mean paired ratio 1.017, best-of ratio 1.09. The spread (0.80–1.09 s) is
far wider than any difference. The Stim-format `detectors` mode came out
0.95–0.98.

**Learned.**
- Where the output memory comes from mattered more than the code that
  writes it. Writing into a Python bytearray made bool records at d = 21
  slower on one thread than the old NumPy unpack: the bytearray is zeroed
  by one thread in 4 KB pages, while NumPy's allocator asks for huge pages
  and does not zero. Switching to `numpy.empty` arrays took that case
  from 1.04 (a complete `compare_builds.py` run of the intermediate build)
  to 0.64–0.75.
- Building a thread pool per step (sample, then write) and visiting a pool
  twice both showed in the millisecond calls at d = 11; one pool and one
  visit per call fixed both.
- A table of per-row slices for the transpose (2 MB at d = 51, read by
  every 64-shot block) was measurably slower than computing row offsets.
- PyO3's buffer protocol is not available under `abi3-py310` (it needs the
  3.11 limited API), hence `__array_interface__`.
- Timings of a few milliseconds on this VM need hundreds of calls per
  build: several cases moved by ±15% between 3-repeat passes.

**Next.**
1. Noise on the near-Clifford backend's per-shot frames (noisy
   magic-state circuits).
2. `MPP` and `PAULI_CHANNEL_2` in the Stim reader, and `to_stim` writing
   detectors and observables.
3. Detection events without keeping every measurement record (memory at
   d = 101 is dominated by the 1.3 GB record, as large as the output).


## 2026-10-31 — Stim's circuit format and detection events (0.1.3)

**What.**
- A new module, `src/stim.rs`: `qvd::stim::parse` reads Stim's circuit
  format into a `StimCircuit`: the `Circuit` (one classical bit per
  measurement, in Stim's `rec` order, `REPEAT` blocks unrolled), the
  detectors and observables as lists of measurements, measurement flips
  (`M(p)`, `MX(p)`, `MR(p)`, ...) and inverted results (`M !q`, `MPAD 1`).
  It takes Stim's Clifford gates (including `SQRT_Y`, `H_XY`, `H_YZ`,
  `C_XYZ`, `C_ZYX`), measurements and resets in every basis, the Pauli
  channels, `DETECTOR`, `OBSERVABLE_INCLUDE`, and ignores coordinates and
  `TICK`. Anything else is an error naming the line, never skipped.
- `stabilizer::sample_detectors`: detection events and observable flips,
  relative to the reference (noiseless) shot as in Stim, bit-packed, with
  shot-major byte output (`detectors_shot_major`, written in parallel by
  walking set bits). `stabilizer::sample_stim` returns the raw records with
  flips and inversions. `M(p)` flips are drawn after the run with
  geometric gaps; the frame loop did not change.
- Python: `sample_detectors(circuit, shots)` takes Stim text, a
  `stim.Circuit` or a Qevedo `Circuit`, and returns shot-major bool arrays as
  Stim's detector sampler does (written straight into the buffer NumPy
  views); `from_stim`; `sample` accepts Stim text.
- The `stabilizer` example's `detectors <file> <shots>` mode, and
  `benchmarks/bench_detectors.py`.

**Why.** Item 1 of the previous entry's list, and what kept the noisy
stabilizer backend from being used: error-correction work starts from
Stim circuit files (Stim's generators and the tools built on them) and
ends in a decoder that takes detection events. Before this run, Qevedo could
sample a noisy circuit only after it had been rebuilt by hand in Qevedo's
API, and the user then had to compute detectors from raw records. The
method is Stim's (Gidney, *Quantum* 5, 497, 2021, arXiv:2103.02202). It
could be checked end to end today: against Stim's own generated circuits,
detector by detector and through the same decoder. A search of recent
work turned up no published detector-sampling method that beats Stim's
reference-plus-frames approach in practice, so matching Stim's interface
at Qevedo's speed was the most useful step. The other candidates (noise on
the near-Clifford frames, a compact Pauli-propagation store) improve
features that fewer people use.

**Measured** (4-vCPU Xeon VM at 2.1 GHz; `benchmarks/results/detectors*.csv`).
Stim's generated rotated surface code, p = 0.001 on all four of its noise
parameters, 10,000 shots, shot-major bool arrays from both:

| d | Qevedo, 4 threads | Qevedo, 1 thread | Stim `sample` | Stim `sample(bit_packed=True)` |
|---|---|---|---|---|
| 11 | 0.015 s | 0.013 s | 0.049 s | 0.037 s |
| 21 | 0.112 s | 0.136 s | 0.560 s | 0.359 s |
| 51 | 1.63 s | 2.32 s | 9.92 s | 12.95 s |

Detection rates agree (largest |z| 2.29–3.90 over 24–720 detectors,
200,000 shots). The decoded logical error rates differed by 2.1–2.3
standard errors in four of five codes, in both directions. A long run of
the d = 3 code (24 million shots each, `--long`) found every detector and
the observable in agreement (largest |z| 1.64). Calibrating by hand with
one million shots at d = 5, p = 0.005 gave the same: Qevedo against Stim
looked like Stim against Stim with another seed, pairwise detector
correlations included (mean z² 0.96–0.99 for both). Existing benchmarks
before and after, interleaved best of 5 (11 for the closest cases):
ratios 0.96–1.03. The binaries changed only in the reference shot, which
now also keeps each classical bit's reference value.

**Learned.**
- The first Python version unpacked bits with NumPy and transposed them.
  That took 0.69 s at d = 21, slower than Stim's 0.52 s, although the
  sampling itself took 0.09 s. Writing the shot-major bytes in Rust, then
  straight into the bytearray NumPy views without a second copy, gave
  0.11 s. At d = 51 the result is 1.3 GB, so one copy costs about as much
  as the sampling.
- `qevedo.simulator.sample` has the same problem in measurement-major form:
  12 s at d = 51 for Stim text, where the sampling takes under a second;
  the rest goes to unpacking and converting 1.4 billion bits with NumPy.
  It is not changed here.
- Stim's generated circuits put measurement noise in an `X_ERROR` before
  `M`, not `M(p)`. `M(p)` is tested separately (rates by hand in Rust; a
  d = 5 code rewritten with `MR(0.01)` and `MX(0.02)`, 2 million shots,
  against Stim).

**Next.**
1. Fast output for `sample` (measurement-major) and bit-packed shot-major
   output for `sample_detectors`, both written by Rust into the NumPy
   buffer.
2. Noise on the near-Clifford backend's per-shot frames (noisy
   magic-state circuits).
3. `to_stim` writing detectors and observables, and `MPP` and
   `PAULI_CHANNEL_2` in the reader.

## 2026-10-29 — Pauli noise in the stabilizer backend (0.1.2)

**What.** Pauli noise channels, sampled independently in every shot:
- `Instruction::Noise` with `qvd::Noise::Pauli1 { px, py, pz }` (Stim's
  `X_ERROR`, `Y_ERROR`, `Z_ERROR`, `DEPOLARIZE1`, `PAULI_CHANNEL_1`) and
  `Noise::Depolarize2(p)`; `Circuit::x_error`, `depolarize1`, ... .
- The stabilizer sampler adds the errors to each shot's Pauli frame,
  drawing geometric gaps between errors rather than a random bit per qubit
  and shot. Channels are held back until a gate touches their qubits, so a
  layer's per-gate channels become one step.
- `library::with_depolarizing_noise` (Stim's uniform circuit-level model),
  `stabilizer::to_stim`, the `--noise` and `--stim` options of the
  `stabilizer` example, and `benchmarks/bench_noise.py`.
- In Python: the channels on `Circuit`, `Circuit.to_stim`, and
  `sample(circuit, shots)`, which returns the raw measurement records as a
  NumPy array (measurement-major, like Stim's `FlipSimulator`).
- Other backends reject noisy circuits instead of ignoring the noise.

**Why.** Roadmap item 6, and the gap that kept the stabilizer backend from
its main use: error-correction studies sample noisy circuits, and before
this run Qevedo could only sample noiseless ones, so its lead over Stim held
only on a workload nobody runs. It is Stim's own method (Gidney, *Quantum*
5, 497, 2021, arXiv:2103.02202). It can be checked end to end today:
exactly against summed error combinations, and statistically against Stim
on the same file. The other candidates (a compact Pauli-propagation store,
a calibrated backend selector) improve features that already work.

**Measured** (4-vCPU Xeon VM; `benchmarks/results/noise*.csv`). Surface
codes with p = 0.001, 10,000 shots, the same Stim file in both:

| d | Qevedo, 4 threads | Qevedo, 1 thread | Stim reference + `FlipSimulator` |
|---|---|---|---|
| 11 | 0.009 s | 0.007 s | 0.008 s |
| 21 | 0.040 s | 0.050 s | 0.060 s |
| 51 | 0.54 s | 0.90 s | 1.54 s |
| 101 | 6.0 s | 14.9 s | 37.5 s |

Detection rates agree with Stim within sampling error (largest |z| over
16–288 detectors: 1.57–2.42 at d = 3 and 5. At d = 7 it was 4.19 on one
detector; four further seeds gave 2.4–3.3, each time on a different
detector). Noiseless timings, best of 5–9, interleaved against the
previous build: ratios 0.998–1.008 on the stabilizer cases. The other
benchmark binaries came out 0.96–1.03× the previous build in one
interleaved pass (best of 3–5). Their code changed only by a scan for noise
channels, and the 1.033× on the 24-qubit state vector is within the
pass-to-pass spread on this VM.

**Learned.**
- The first version put the channel inline in the step enum. That doubled
  every step, and noiseless circuits ran 5–8% slower, because the steps are
  walked in cache-sized segments. An index into a side table, with the
  noise routine kept out of line, removed the slowdown.
- A circuit written one target per line costs Stim a lot: 14 GB and a kill
  at d = 51 with noise, and 1.7× its time at d = 21 without. The recorded
  noiseless Stim times (`stabilizer.csv`, the i9) were measured that way,
  so they flatter Qevedo at d = 11 and 21. `compare_stim.py` now groups
  instructions. Re-timed fairly on the VM without noise, Qevedo and Stim are
  level at d = 11 and 21 (0.005 / 0.006 s, 0.030 / 0.039 s on one thread);
  at d = 51 and 101 Qevedo stays 1.4× and 3.0× faster on one thread. The README
  says this next to the i9 table.

**Next.**
1. A Stim-format reader with `DETECTOR`, `OBSERVABLE_INCLUDE` and `M(p)`,
   so Qevedo runs Stim's generated circuits directly and returns detection
   events (what decoders such as PyMatching take), and a logical-error-rate
   example.
2. Noise on the near-Clifford backend's per-shot frames (noisy
   magic-state circuits, as Clifft and SymFT target).
3. The compact Pauli-propagation store from the previous entry.

## 2026-10-28 — Pauli propagation for expectation values (0.1.1)

**What.** A new module, `pauliprop/`: `<0|U† O U|0>` for unitary circuits
of up to 1024 qubits by propagating the observable backwards as a sparse
sum of Pauli strings (sparse Pauli dynamics, Begušić, Gray and Chan,
arXiv:2308.05077). Cliffords collect in a frame (images of `X_q`, `Z_q`)
and never touch the sum; non-Clifford rotations split anticommuting
strings in two parallel passes over 64 hash-map shards; strings below a
coefficient threshold (and optionally beyond a term cap) are dropped, and
the dropped one-norm is reported as an error bound. Also
`library::eagle_heavy_hex` and `library::kicked_ising` (IBM's utility
circuit), the `pauliprop` example, `expectation` in Python (Pauli strings,
dicts or Qiskit `SparsePauliOp`), and benchmarks against IBM's pauli-prop.

**Why.** It was item 4 of the roadmap and the one family of questions the
simulator could not answer: expectation values of large, entangling,
non-Clifford circuits (127 qubits and up), which the state vector cannot
hold, the MPS cannot follow and the near-Clifford backend cannot afford.
It is the method of the classical reproductions of IBM's utility
experiment, can be checked end to end today (exactly at small sizes, at
Clifford points at 127 qubits, and term for term against IBM's pauli-prop),
and touches no existing hot path.

**Measured** (4-vCPU Xeon VM; `benchmarks/results/pauliprop.csv`). `<Z62>`
of the 127-qubit kicked Ising circuit at five settings: the same value,
strings and dropped one-norm as pauli-prop 0.2.1 in every case;
14.6–117× faster end to end on 4 threads, and 2.4–5.5× faster on one
thread than pauli-prop's Rust propagation alone (e.g. 20 steps, θ_h = 0.4,
threshold 10⁻⁵: 9.6 s against 191 s). All existing benchmark binaries
(`bench`, `stabilizer`, `mps`, `nearclifford`) are byte-identical before
and after the change, so no existing result can have moved.

**Learned.** Splitting a rotation into `θ - π/2` and a Clifford (Clifford
perturbation theory, arXiv:2306.04797) was implemented and removed: with
a coefficient threshold it keeps exactly the same strings (cosine and sine
swap). It only helps with truncation by perturbative order. The
dropped-one-norm bound is rigorous but far above 1 at 20 steps; the
20-step θ_h = 0.6 value is not converged at 10⁻⁵, and finer thresholds
run out of 15 GB.

**Release.** Version 0.1.1 is committed (`Release 0.1.1`, CI green), but
pushing the tag `v0.1.1` was refused by the session's git proxy (HTTP
403), so it is not on PyPI yet, and the website was left unchanged until
it is. To finish: push the tag (`git tag v0.1.1 4ba5149 && git push
origin v0.1.1`), then add the Pauli propagation result to qevedo.com.
(The first CI run of the feature commit failed on macOS and Windows only
because pip preferred the published 0.1.0 wheel over the local build of
the same version; the version bump fixed it.)

**Next.**
1. A more compact string store (sorted arrays or packed open addressing
   instead of hash maps of full strings) to reach converged 20-step
   utility values in less memory; then weight truncation and Pauli noise
   channels (Angrisani et al., arXiv:2501.13101).
2. Noise in the stabilizer and near-Clifford samplers (roadmap item 6).
3. A backend selector calibrated on the running machine (roadmap item 4).
