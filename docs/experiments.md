# Experiments: what worked, what didn't, what's next

This file is the simulator's memory across improvement runs. Before choosing
what to work on, read **Next** and search **Didn't work** and **Considered**
for the idea, its file and its source. Don't retry something listed there
unless you say what is different this time. At the end of every run, kept or
not, update all four sections in the same commit as the change; if nothing
ships, commit this file alone.

`docs/improvement-log.md` has the full story of each shipped change; this
file is the short index, including everything that was tried and dropped.
Times are from the run's own machine (a 4-vCPU Xeon VM unless stated) and
only comparable within that run. "d" is the rotated surface code distance;
noisy runs use p = 0.001 and 10,000 shots unless stated.

## Next

In priority order. Replace this list at the end of each run.

1. **Noisy near-Clifford: sign registers instead of frame words.** After 0.1.9 Clifft is still 1.8–4.9× faster on 9 of 10 cases. A split-off shot conjugates frame words over all virtual qubits at every gathering and collapse (frame measurements + conjugation ≈ 25% of the time at 100 qubits, p = 1e-3). Precompute, per noise site, which later rotations and measurements each error flips (Clifft's expression registers), so per shot the frame work is O(errors). The outcome-driven X insertions (frame and dense measurements, resets) become precomputed vectors too.
2. **SIMD dense kernels** (`dense.rs`: `rotate`, `expectation`, `retired`; they are scalar, Clifft's are AVX-512), and keep the outcome probability in the `Task` so a group split at a dense measurement does not recompute the expectation.
3. **Stim's format on the near-Clifford backend.** Read `T`, `T_DAG` and Clifft's rotation gates and return detection events, so magic-state cultivation circuits run directly.
4. **Rerun `compare_nearclifford.py`** (noiseless near-Clifford against Aer): the recorded `nearclifford.csv` predates 0.1.9, which made the 100-qubit 24-T example 4.6× faster.
5. **Stim reader and writer gaps.** `MXX`/`MYY`/`MZZ`, `SPP` and heralded errors in the reader; `to_stim` writing detectors and observables.
6. **Detection events without keeping every measurement record.** At d = 101 the 1.3 GB record is as large as the output.
7. **Qiskit layer**: Aer noise models (at least Pauli errors on Clifford circuits, into the simulator's channels); classical control flow (`if_test`) on the backends that already measure mid-circuit; Cirq and PennyLane devices.
8. **Pauli propagation.** A compact string store (sorted arrays or packed open addressing instead of hash maps of full strings), to reach converged 20-step utility values in less memory; then weight truncation and noise (Angrisani et al., arXiv:2501.13101).
9. **A backend selector calibrated on the running machine.**
10. **Skip straight from one error to the next across blocks of frame words** (stabilizer noise). At d = 101 the countdown walks 1.1e9 steps for 7.2e7 errors (2026-10-28).

## Worked

Newest first. Details are in `improvement-log.md` under the same date.

| Date | Version | Change | Effect |
|---|---|---|---|
| 2026-11-08 | 0.1.9 (on master; not on PyPI yet: release job hit an HTTP 504) | Noisy near-Clifford compiled once into a program for the groups, run one group at a time depth first (`noisy.rs`); consecutive noise channels of one probability as one run of sites; dense measurement as one gather into the half-size state (`Dense::retired`, all dense measurements) | Noisy cases 1.7–4.1× faster than 0.1.8; against Clifft 1.8–4.9× slower (was 3.4–17.8×), 1.23× faster at 200 qubits on 4 threads; noiseless near-Clifford example 4.6× faster, same counts |
| 2026-11-06 | 0.1.8 | Pauli noise on the near-Clifford backend (`nearclifford/noisy.rs`): one shared Clifford frame per batch, a Pauli frame per shot, a dense state per group (as Clifft, Chase & Labib, arXiv:2604.27058). 0.1.7 never published: CI's newer clippy rejected `drain(..).collect()`; fixed in 0.1.8 | Noisy Clifford+T circuits run, with exact statistics (χ² p = 0.385 / 0.893 / 0.861 at 200k shots). Within 1.3–1.4× of Clifft without errors, 3.4–17.8× slower with them; noiseless path unchanged |
| 2026-11-06 | 0.1.6 | Qiskit backend and primitives (`qevedo.simulator.qiskit`: QevedoSimulator, QevedoSampler, QevedoEstimator), from an interactive session | 6.6–32× faster than Aer through `backend.run` on QV 20–28; estimator 5.5–30× faster |
| 2026-11-04 | 0.1.5 | Stim reader: `MPP` (basis rotations plus a CX ladder), `PAULI_CHANNEL_2`, correlated `E`/`ELSE_CORRELATED_ERROR` | Toric code L = 10/20/30: 0.021/0.162/0.787 s against Stim's 0.083/1.116/4.310 s (4.0–6.9×) |
| 2026-11-02 | 0.1.4 | Results written by Rust straight into `numpy.empty` arrays, one pool and one visit per call; bit-packed `sample_detectors` | d = 51 records 0.41× the time (1 thread 0.55×); detectors with observables 0.35×; d = 101 bit-packed detectors 13.6 s against Stim's 303 s |
| 2026-10-31 | 0.1.3 | Stim's circuit format and detection events (`stim.rs`, `sample_detectors`) | d = 51 detection events 1.63 s against Stim's 9.92 s; rates agree with Stim over 24 M shots |
| 2026-10-29 | 0.1.2 | Pauli noise in the stabilizer frames, with geometric gaps and channels deferred to the next touching gate; other backends reject noise | Noisy d = 51 0.54 s, d = 101 6.0 s, against Stim's 1.54 s and 37.5 s |
| 2026-10-28 | 0.1.1 | Pauli propagation for expectation values (`pauliprop`, Begušić, Gray & Chan arXiv:2308.05077) | Same values as IBM pauli-prop; 14.6–117× faster on 4 threads |

## Didn't work

Don't retry these as they were tried. Each line says why. "Retry if" says
what new fact would justify another attempt.

**Data layout and inlining (applies to every new channel or step type)**
- **Inline noise payloads** (2026-10-28, 2026-10-29, 2026-11-04). Each time they grew the hot structs and slowed noiseless circuits:
  - `Instruction::Noise` inline: 56 → 64 bytes, near-Clifford 14–15% slower.
  - `Step::Noise` inline: doubled each step, 21% slower at d = 51 at first, 5–8% after tuning.
  - Inline 15 `Pauli2` probabilities: `Instruction` grew to 150 bytes, noiseless d = 51 4–6% slower.

  Box the payload or keep a side-table index. A test pins `Noise` at 32 bytes and `Instruction` at 64.
- **Letting cold noise code inline into hot loops** (2026-11-04, 2026-11-06).
  - `PendingNoise::add` inlined into the reference shot made it about 5% slower.
  - The near-Clifford noise code made the noiseless 100-qubit example 5–9% slower, with identical callgrind instruction counts: fat LTO with one codegen unit placed the code differently.

  Mark noise paths `#[cold]` + `#[inline(never)]`.
- **Dead ends chasing the 2026-11-06 noiseless regression** (the cause was code placement, above):
  - compacting groups once a quarter of their slots are dead (+1.3% median, noise);
  - `rayon::join` for measurement splits (still 3–7% slower);
  - transparent huge pages (THP is madvise-only here);
  - swapping back the old terminal sampler (equally slow).

**Near-Clifford noise (`nearclifford/noisy.rs`)**
- **All groups in lockstep, one operation at a time** (0.1.8, replaced 2026-11-08). Once errors split most shots off, every operation streamed all groups' dense parts through memory (439 groups × 2^14 amplitudes, 112 MB per 640-shot batch): 4–10 ns per amplitude. Run groups depth first through a compiled program instead.
- **Branch-free dense kernels alone** (2026-11-08): visiting only written indices, coefficient as phase × parity sign. On the lockstep path, 4.3 → 3.8 s (within noise) because the loops waited on memory. Kept, but only worth it once states stay in cache.
- **`&dyn Fn` element callbacks to share the serial and parallel kernel loops** (2026-11-08): slower than the original (p = 1e-4 case 0.76 → 1.2 s). Use generic helpers.
- **A fresh `Vec` per dense measurement** (2026-11-08): above 128 KB glibc maps fresh pages, page faults every time. Reuse one buffer per batch.
- **Dead end: sampling all final measurements of a group from one transformed state** (2026-11-08, Next of 0.1.8). Considered in detail: a measurement tree costs ~k/2 shots' worth of dense work, and applying the whole Clifford at full size costs more passes than the sequential halving. With 1–2 shots per group at p = 1e-3 neither helps; the one-pass retirement does. Retry if groups stay large (low noise, many shots) and the final dense measurements dominate.
- **A copy of the Clifford frame per group** (2026-11-06). 2.8× slower at 200 qubits, though 14–40% faster at 60–100; summed on one thread 22.7 s against 15.2 s for the shared frame.
- **Pre-trajectory sampling for state-vector, MPS and near-Clifford** (2026-10-28, Patti et al. arXiv:2504.16297): simulate each distinct error pattern once for all shots that drew it. Fully implemented, and it passed exact-distribution tests on every backend, but it was **lost**: that run couldn't push (HTTP 403). It is not rejected; it was replaced by frames on near-Clifford (0.1.8), and the other backends reject noise. Retry if noise is wanted on the state vector or MPS.

**Python output (`python/src/lib.rs`)**
- **Unpacking bits with NumPy in Python** (2026-10-31): d = 21 took 0.69 s, slower than Stim, while sampling took 0.09 s.
- **Writing into a Python `bytearray`** (2026-11-02): it is zeroed by one thread in 4 KB pages, giving 1.04× the old time at d = 21. Use `numpy.empty`, which gets huge pages and no zeroing: 0.64–0.75×.
- **A thread pool per step, or two visits to one pool** (2026-11-02): showed in millisecond calls at d = 11.
- **A table of per-row slices for the transpose** (2026-11-02): 2 MB at d = 51, measurably slower than computing row offsets.
- **PyO3's buffer protocol** (2026-11-02): unavailable under `abi3-py310` (it needs the 3.11 limited API). Use `__array_interface__`.

**Pauli propagation (`pauliprop`)**
- **Clifford perturbation theory's angle split, θ − π/2 plus a Clifford** (2026-10-28, arXiv:2306.04797). With a coefficient threshold it keeps exactly the same strings. Retry only for truncation by perturbative order.
- **Known limits, measured, not fixed** (2026-10-28):
  - 20 steps near θ_h = π/2 grows to tens of millions of strings;
  - θ_h = 0.6 at 20 steps is not converged at 1e-5;
  - θ_h = 1.0 at 1e-5 runs out of 15 GB;
  - the dropped-one-norm bound is far above 1 at 20 steps.

**Benchmarks and checks**
- **Stim circuits with one target per line** (2026-10-29). Stim needed 268 MB at d = 21, and at d = 51 used 14 GB and was killed. Ungrouped circuits also flattered Qevedo's noiseless speed-up: 1.7× at d = 21. Always group instructions (as `compare_stim.py` does) and warm Stim up first.
- **Distance-101 bool detection events** (2026-10-31): about 10 GB, too big for the 15 GB VM. Use bit-packed output.
- **Comparing sets of distinct outcomes in random-circuit checks** (2026-11-04): fails when there are more outcomes than shots. Compare the affine span.
- **Treating |z| ≈ 2–3.5 in agreement checks as a bug** (2026-10-31, 2026-11-04). The deviations vanished at 1–24 M shots and changed sign between seeds. Rerun with more shots and seeds before suspecting anything.
- **An exact noisy test over 4ⁿ branches** (2026-10-29): it took 68 s. Restricted to 2 qubits it takes 4.9 s.
- **An apparent MPS QFT-100 regression** (2026-11-04): a 1.088 median over 25 pairs was noise (60 alternating pairs gave 0.986). Valgrind is unusable on binaries built for the native CPU.

## Considered, not tried

- **Clifft's MEASURE_ACTIVE pivot kernels and expression registers** (read from `clifft.compile(...).inspect()` on the 100-qubit benchmark circuit, 2026-11-08: 113 actions, peak width 13, noise kept apart as `noise_site_probabilities`, every rotation and record carrying a sign register `e_i`). The registers are Next item 1; the one-pass retirement here covers the pivot kernel's job.

- **Faster detector sampling than reference-plus-frames.** Searched three times (2026-10-31, 2026-11-02, 2026-11-04): GPU stabilizer simulators, STABSim (arXiv:2507.03092), SymPhase, QuaSARQ (arXiv:2603.14641), Sdim (arXiv:2511.12777). Nothing beats the current approach on CPUs.
- **Propagating each error straight to the detectors it flips** (2026-10-31). Described in Stim's paper but not implemented there, and not tried here. A possible lead for many-shot, low-noise runs.
- **PauliPropagation.jl comparison** (2026-10-28): not installed.
- **Repeated passes over:** the compact Pauli-propagation store and the calibrated selector (in **Next**); `MPP` and near-Clifford noise (both since done). They were passed over because they "improve features that fewer people use".

## Method lessons

- **Release workflow flakes** (2026-11-08): 0.1.9's Release run failed only because the macOS arm64 job got HTTP 504 installing maturin; nothing was published. Check which job failed before suspecting the code; a rerun of the failed jobs is the owner's call.
- **Profiling without perf** (2026-11-08): no `perf` on the VM. Temporary `Instant` timers per operation kind in the hot loop (and a count of amplitudes touched) found the bottleneck within minutes; per-amplitude ns from those counts told memory-bound from compute-bound. Remove them before committing.
- **Python environments** (2026-11-08): `/usr/local/bin/pip` belongs to Python 3.13 while `python3` is 3.11, and Clifft needs ≥ 3.12. Make one `python3.13 -m venv` per build (before / after) and install each wheel there; `compare_noisy_builds.py` takes the two interpreters.
- **Comparing counts across builds:** Python's `hash()` of a string is salted per process; use `hashlib`.

- **Timing on the VM.**
  - Builds drift ±8–10% between batches and 3-repeat runs move ±15%.
  - Compare before and after with paired, randomly ordered runs (15–60 pairs).
  - Millisecond calls need hundreds of repetitions.
  - Compare `.text` of unchanged benchmark binaries: identical code can't have regressed.
- **CI.**
  - While master's version equals PyPI's, `pip install --find-links dist` in `ci.yml` installs the published wheel, so feature commits fail on macOS and Windows until the version bump. This is known and harmless; don't chase it.
  - CI uses the latest stable clippy, which can be newer than the run's: 1.99 rejected code that 1.97 passed and stopped 0.1.7. Update the toolchain before releasing.
- **Lost work.** The 2026-10-28 morning run lost two finished features because it couldn't push and left no record. If a push fails, commit this file's entry and report under "Needs the owner"; don't end with only local commits.
