# Improvement log

One entry per improvement run: what changed, why, what it measured, and
what to try next. Newest first.

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
found no detector-sampling method that would beat qvd's current approach
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
  `stim.Circuit` or a qvd `Circuit`, and returns shot-major bool arrays as
  Stim's detector sampler does (written straight into the buffer NumPy
  views); `from_stim`; `sample` accepts Stim text.
- The `stabilizer` example's `detectors <file> <shots>` mode, and
  `benchmarks/bench_detectors.py`.

**Why.** Item 1 of the previous entry's list, and what kept the noisy
stabilizer backend from being used: error-correction work starts from
Stim circuit files (Stim's generators and the tools built on them) and
ends in a decoder that takes detection events. Before this run, qvd could
sample a noisy circuit only after it had been rebuilt by hand in qvd's
API, and the user then had to compute detectors from raw records. The
method is Stim's (Gidney, *Quantum* 5, 497, 2021, arXiv:2103.02202). It
could be checked end to end today: against Stim's own generated circuits,
detector by detector and through the same decoder. A search of recent
work turned up no published detector-sampling method that beats Stim's
reference-plus-frames approach in practice, so matching Stim's interface
at qvd's speed was the most useful step. The other candidates (noise on
the near-Clifford frames, a compact Pauli-propagation store) improve
features that fewer people use.

**Measured** (4-vCPU Xeon VM at 2.1 GHz; `benchmarks/results/detectors*.csv`).
Stim's generated rotated surface code, p = 0.001 on all four of its noise
parameters, 10,000 shots, shot-major bool arrays from both:

| d | qvd, 4 threads | qvd, 1 thread | Stim `sample` | Stim `sample(bit_packed=True)` |
|---|---|---|---|---|
| 11 | 0.015 s | 0.013 s | 0.049 s | 0.037 s |
| 21 | 0.112 s | 0.136 s | 0.560 s | 0.359 s |
| 51 | 1.63 s | 2.32 s | 9.92 s | 12.95 s |

Detection rates agree (largest |z| 2.29–3.90 over 24–720 detectors,
200,000 shots). The decoded logical error rates differed by 2.1–2.3
standard errors in four of five codes, in both directions. A long run of
the d = 3 code (24 million shots each, `--long`) found every detector and
the observable in agreement (largest |z| 1.64). Calibrating by hand with
one million shots at d = 5, p = 0.005 gave the same: qvd against Stim
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
this run qvd could only sample noiseless ones, so its lead over Stim held
only on a workload nobody runs. It is Stim's own method (Gidney, *Quantum*
5, 497, 2021, arXiv:2103.02202). It can be checked end to end today:
exactly against summed error combinations, and statistically against Stim
on the same file. The other candidates (a compact Pauli-propagation store,
a calibrated backend selector) improve features that already work.

**Measured** (4-vCPU Xeon VM; `benchmarks/results/noise*.csv`). Surface
codes with p = 0.001, 10,000 shots, the same Stim file in both:

| d | qvd, 4 threads | qvd, 1 thread | Stim reference + `FlipSimulator` |
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
  so they flatter qvd at d = 11 and 21. `compare_stim.py` now groups
  instructions. Re-timed fairly on the VM without noise, qvd and Stim are
  level at d = 11 and 21 (0.005 / 0.006 s, 0.030 / 0.039 s on one thread);
  at d = 51 and 101 qvd stays 1.4× and 3.0× faster on one thread. The README
  says this next to the i9 table.

**Next.**
1. A Stim-format reader with `DETECTOR`, `OBSERVABLE_INCLUDE` and `M(p)`,
   so qvd runs Stim's generated circuits directly and returns detection
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
