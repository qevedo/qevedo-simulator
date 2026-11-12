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

1. **What a split-off shot still costs: its own dense state.** After 0.1.11 Clifft is still 1.15–1.94× faster on 8 of 10 noisy cases, most at p = 1e-2 (1.94× on one thread) and on the 100-qubit circuit (0.57 s against 0.37 s on one thread). Profile the 0.1.11 run (temporary `Instant` timers, as on 2026-11-08) before choosing between: one pass fewer per dense measurement (item 2), AVX-512 kernels (item 3), and copying a dense state only when an error really changes it (an error whose rows touch no later rotation needs no split).
2. **One pass fewer per dense measurement.** Accumulate the expectation of the next dense measurement's Pauli (known at compile time when no rotation lies between) inside the retirement that precedes it, reading the partner from the freshly written half of the buffer; and keep the outcome probability in the `Task` so a group split at a dense measurement does not recompute it.
3. **AVX-512 dense kernels behind runtime detection** (`is_x86_feature_detected!`), since wheels are built for x86-64-v3 and Clifft uses AVX-512; and NEON kernels for the ARM wheels, which still run the scalar loops.
4. **Stim's format on the near-Clifford backend.** Read `T`, `T_DAG` and Clifft's rotation gates and return detection events, so magic-state cultivation circuits run directly.
5. **Rerun `compare_nearclifford.py`** (noiseless near-Clifford against Aer): the recorded `nearclifford.csv` predates 0.1.9.
6. **Stim reader and writer gaps.** `MXX`/`MYY`/`MZZ`, `SPP` and heralded errors in the reader; `to_stim` writing detectors and observables.
7. **Detection events without keeping every measurement record.** At d = 101 the 1.3 GB record is as large as the output.
8. **Qiskit layer**: Aer noise models (at least Pauli errors on Clifford circuits, into the simulator's channels); classical control flow (`if_test`) on the backends that already measure mid-circuit; Cirq and PennyLane devices.
9. **Pauli propagation.** A compact string store (sorted arrays or packed open addressing instead of hash maps of full strings), to reach converged 20-step utility values in less memory; then weight truncation and noise (Angrisani et al., arXiv:2501.13101).
10. **A backend selector calibrated on the running machine.**
11. **Skip straight from one error to the next across blocks of frame words** (stabilizer noise). At d = 101 the countdown walks 1.1e9 steps for 7.2e7 errors (2026-10-28).

## Worked

Newest first. Details are in `improvement-log.md` under the same date.

| Date | Version | Change | Effect |
|---|---|---|---|
| 2026-11-12 | 0.1.11 | Sign registers instead of Pauli frames on the noisy near-Clifford path (`link`, `rows`, `Shots` in `nearclifford/noisy.rs`, `Tableau::push_forward_x`): which checks (rotation and measurement Paulis) each insertion (error, frame-measurement X, retired X, reset fix) flips is computed once per run by carrying the checks' physical Paulis backwards through the physical gates; a shot keeps one bit per check, an error XORs its row; no frame conjugations per shot | Noisy cases 0.24–0.77 of 0.1.10's time (200 qubits 1 thread 2.03 → 0.49 s; 100 qubits 1.06 → 0.54 s), same counts for the same seed; against Clifft 1.15–1.94× slower on 8 of 10 (was 1.6–3.4× on 9), 4.1× faster at 200 qubits on 4 threads and 1.76× on one; one shot 0.64–0.78 of 0.1.10's time, since `compile` no longer pulls back every error's image |
| 2026-11-10 | 0.1.10 (also ships 0.1.9's work, which never reached PyPI) | AVX2 dense kernels for the near-Clifford backend (`nearclifford/dense/avx2.rs`): expectation, rotation and X-type retirement, two amplitudes per register, partners as one register at `i ^ xh` (swapped for X at position 0), signs from a block table; scalar serial sums with four accumulators | Kernels 2–4× faster in cache (e.g. retirement 1.3 → 0.32–0.43 ns per amplitude); noisy cases 0.60–0.91 of 0.1.9's time (100 qubits p = 1e-3: 1.71 → 1.21 s on 1 thread, 0.52 → 0.31 s on 4); against Clifft 1.6–3.4× slower on 9 of 10 (was 1.8–4.9×), 1.10× faster at 200 qubits on 4 threads; same counts for the same seed |
| 2026-11-08 | 0.1.9 (never on PyPI: release job hit an HTTP 504; shipped in 0.1.10) | Noisy near-Clifford compiled once into a program for the groups, run one group at a time depth first (`noisy.rs`); consecutive noise channels of one probability as one run of sites; dense measurement as one gather into the half-size state (`Dense::retired`, all dense measurements) | Noisy cases 1.7–4.1× faster than 0.1.8; against Clifft 1.8–4.9× slower (was 3.4–17.8×), 1.23× faster at 200 qubits on 4 threads; noiseless near-Clifford example 4.6× faster, same counts |
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
- **Carrying the virtual images of the insertions forwards to find the rows** (2026-11-12, first version of 0.1.11's `link`): one slot per insertion (11,000 at 100 qubits, 29,000 at 200) conjugated by every virtual Clifford (5,700 / 22,000). Same rows, but 18–30 ms at 200 qubits, and `compile` still pulled back two dense images per noisy qubit: one shot took 1.04–1.56× 0.1.10's time (`compare_noisy_builds.py --shots 1`), about level at 100 shots. Neither skipping the words of slots not yet put in nor a 64×64 block transpose helped (1 shot at 200 qubits 81 → 78.5 ms), nor did chunks in parallel (1.31× on 4 threads), nor carrying the *virtual* checks backwards (still 30 ms: the virtual images are dense). Replaced by carrying the *physical* checks backwards through the physical gates, where errors are one-qubit Paulis: one shot 0.64–0.78 of 0.1.10's time.
- **All groups in lockstep, one operation at a time** (0.1.8, replaced 2026-11-08). Once errors split most shots off, every operation streamed all groups' dense parts through memory (439 groups × 2^14 amplitudes, 112 MB per 640-shot batch): 4–10 ns per amplitude. Run groups depth first through a compiled program instead.
- **Branch-free dense kernels alone** (2026-11-08): visiting only written indices, coefficient as phase × parity sign. On the lockstep path, 4.3 → 3.8 s (within noise) because the loops waited on memory. Kept, but only worth it once states stay in cache.
- **`&dyn Fn` element callbacks to share the serial and parallel kernel loops** (2026-11-08): slower than the original (p = 1e-4 case 0.76 → 1.2 s). Use generic helpers.
- **A fresh `Vec` per dense measurement** (2026-11-08): above 128 KB glibc maps fresh pages, page faults every time. Reuse one buffer per batch.
- **Dead end: sampling all final measurements of a group from one transformed state** (2026-11-08, Next of 0.1.8). Considered in detail: a measurement tree costs ~k/2 shots' worth of dense work, and applying the whole Clifford at full size costs more passes than the sequential halving. With 1–2 shots per group at p = 1e-3 neither helps; the one-pass retirement does. Retry if groups stay large (low noise, many shots) and the final dense measurements dominate.
- **A copy of the Clifford frame per group** (2026-11-06). 2.8× slower at 200 qubits, though 14–40% faster at 60–100; summed on one thread 22.7 s against 15.2 s for the shared frame.
- **Pre-trajectory sampling for state-vector, MPS and near-Clifford** (2026-10-28, Patti et al. arXiv:2504.16297): simulate each distinct error pattern once for all shots that drew it. Fully implemented, and it passed exact-distribution tests on every backend, but it was **lost**: that run couldn't push (HTTP 403). It is not rejected; it was replaced by frames on near-Clifford (0.1.8), and the other backends reject noise. Retry if noise is wanted on the state vector or MPS.

**Dense kernels (`nearclifford/dense.rs`, `dense/avx2.rs`, 2026-11-10)**
- **Scalar rewrite alone**: blocked signs (table of the low bits, one population count per 16), unchecked indexing, four accumulators. Expectation 1.7×, retirement 1.6×, rotation 1.3×; the Z-type rotation got slower (0.85 → 1.5 ns) with per-element complex factors. Contiguous (lowest X bit ≥ 4) and XOR-scrambled partner loads ran equally fast: the loops were at scalar throughput, not memory- or gather-bound. Replaced by AVX2; only the multi-accumulator serial sum kept.
- **Blocking contiguous below the lowest X bit** (planned, not built): 6 of 14 measured and 4 of 14 rotation Paulis of the 100-qubit benchmark have X at position 0, where the blocks are one amplitude. The register-pair formulation (`i ^ xh`, swap for bit 0) covers every Pauli.
- **A population count per register for the signs** (first AVX2 version): retirement 0.64–0.66 ns per input amplitude; a block table brought it to 0.32–0.48.
- **Full-sum expectation with both `Re` and `Im` accumulators**: 0.42–0.47 ns per amplitude; pairs once with the one product the phase needs: 0.22–0.35.
- **Dead end: nested parallelism.** Forcing the dense kernels serial inside each batch (`PARALLEL` = 2^24) changed nothing (2.41 s against 2.28 s, noise).

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
- **Comparing `.text` across a version bump** (2026-11-12). Every example binary differed from 0.1.10's even where no code changed: the version enters the symbol hashes, which move the layout. Build the "before" sources with the version bumped too; then `bench`, `pauliprop`, `stabilizer` were byte-identical.
- **Treating 1.03–1.04 medians on `shots` and `mps` as regressions** (2026-11-12). Their hot functions had the same sizes in both builds (only noisy-path symbols differed). `shots`: 60 rounds read 1.033, the next 60 1.004; `mps qft 100`: 1.036 over 60, then 1.000 over the next 60. Rerun 60 rounds before believing a 3–4% reading.
- **Stim circuits with one target per line** (2026-10-29). Stim needed 268 MB at d = 21, and at d = 51 used 14 GB and was killed. Ungrouped circuits also flattered Qevedo's noiseless speed-up: 1.7× at d = 21. Always group instructions (as `compare_stim.py` does) and warm Stim up first.
- **Distance-101 bool detection events** (2026-10-31): about 10 GB, too big for the 15 GB VM. Use bit-packed output.
- **Comparing sets of distinct outcomes in random-circuit checks** (2026-11-04): fails when there are more outcomes than shots. Compare the affine span.
- **Treating |z| ≈ 2–3.5 in agreement checks as a bug** (2026-10-31, 2026-11-04). The deviations vanished at 1–24 M shots and changed sign between seeds. Rerun with more shots and seeds before suspecting anything.
- **An exact noisy test over 4ⁿ branches** (2026-10-29): it took 68 s. Restricted to 2 qubits it takes 4.9 s.
- **An apparent MPS QFT-100 regression** (2026-11-04): a 1.088 median over 25 pairs was noise (60 alternating pairs gave 0.986). Valgrind is unusable on binaries built for the native CPU.

## Considered, not tried

- **Check-major sign columns with sparse rows** (2026-11-12): keep, per check, a column over the group's shots and flip one bit per (error, check in its row). Cheaper per check for large groups, but an error costs one scattered write per check it flips; shot-major registers cost a few words per error and one bit read per shot per check, and groups are small once errors split them. Built shot-major only.
- **SymFT** (arXiv:2607.28600, "Universal Fault-Tolerant Quantum Circuit Simulation via Symbolic Clifford–Pauli Frames and Stabilizer Coordinates"), 2026-11-12 search: symbolic frames with presampled expression values, compared against Clifft. Not read in detail or timed; read before the next near-Clifford change.
- **Tsim** (Haenel, Luo & Zhao, arXiv:2604.01059, QuEra; PyPI `bloqade-tsim`), checked 2026-11-12: ZX diagrams with parameterized Pauli-channel vertices, stabilizer-rank decomposition compiled to JAX, aimed at GPUs and huge shot counts. A different regime from per-shot dense states; not timed (would need JAX here).
- **Clifft's MEASURE_ACTIVE pivot kernels and expression registers** (read from `clifft.compile(...).inspect()` on the 100-qubit benchmark circuit, 2026-11-08: 113 actions, peak width 13, noise kept apart as `noise_site_probabilities`, every rotation and record carrying a sign register `e_i`). The registers shipped in 0.1.11 (computed by `link`); the one-pass retirement here covers the pivot kernel's job.

- **Split real/imaginary layout for `Dense`** (blocks of 4 re + 4 im, as the state vector's `simd.rs`), 2026-11-10: no shuffles for complex products and the existing portable `Vector` trait (ARM too), but the retirement into a half-size state needs lane compaction when the target is position 0 or 1, and every `Dense` user changes. The interleaved register-pair kernels got 2–4× without it. Retry if NEON or AVX-512 kernels are written anyway.
- **Tsim and SyQMA** (2026-11-10 search, named in Clifft's paper arXiv:2604.27058 v2 as concurrent work): Tsim compiles once and samples universal noisy QEC circuits with ZX reductions and stabilizer-rank methods; SyQMA is an exact symbolic simulator. Not read or timed this run (arXiv ids not checked); a search for 2026 near-Clifford samplers faster than Clifft found none. Retry: read both before choosing the next near-Clifford change.
- **AVX-512 now** (2026-11-10): the VM has it, but the x86-64 wheels target x86-64-v3, so it needs runtime dispatch; in Next.

- **Faster detector sampling than reference-plus-frames.** Searched three times (2026-10-31, 2026-11-02, 2026-11-04): GPU stabilizer simulators, STABSim (arXiv:2507.03092), SymPhase, QuaSARQ (arXiv:2603.14641), Sdim (arXiv:2511.12777). Nothing beats the current approach on CPUs.
- **Propagating each error straight to the detectors it flips** (2026-10-31). Described in Stim's paper but not implemented there, and not tried here. A possible lead for many-shot, low-noise runs.
- **PauliPropagation.jl comparison** (2026-10-28): not installed.
- **Repeated passes over:** the compact Pauli-propagation store and the calibrated selector (in **Next**); `MPP` and near-Clifford noise (both since done). They were passed over because they "improve features that fewer people use".

## Method lessons

- **Build the "before" wheel from `git archive HEAD`** (2026-11-12): the first baseline build ran from the working tree while the change was being written and failed on half-edited code. Export HEAD to the scratch directory and build there; bump its version too if `.text` is to be compared (above).
- **Check counts digests first** (2026-11-12): a change that keeps the random draws should give identical counts for a seed; `hashlib` digests of `RunResult.counts` for five circuits in both venvs settled correctness in seconds, before any timing.
- **Few-shot runs show fixed costs** (2026-11-12): timings at 10,000 shots hide per-run compile work; `compare_noisy_builds.py --shots 1` showed the first version of the change 1.04–1.56× slower there while 10,000 shots were 0.25–0.93.
- **Count both sides of a meet-in-the-middle** (2026-11-12): propagating whichever side has fewer Paulis, in the picture where they are sparse, took a 30 ms pass off the run (one shot at 200 qubits 0.66 of 0.1.10's time instead of 1.46); temporary `Instant` timers around compile / link / run found where the time went.
- **Floating-point sums are latency-bound** (2026-11-10): a sum over 2^14 cached terms with one accumulator ran at 2.1 ns per term, with eight at 0.3. Check any `.sum()` in a hot loop.
- **Microbenchmarks on the VM** (2026-11-10): single timings of the same kernel moved ±50% between runs; best of 30 interleaved repetitions (old, new, old, ...) of 100 calls was stable to a few percent.
- **Benchmark the shipped instruction set** (2026-11-10): `.cargo/config.toml` builds for the native CPU (AVX-512 here); wheels are x86-64-v3. Set `RUSTFLAGS="-C target-cpu=x86-64-v3"` for both before/after wheels, and run the tests once with `-C target-cpu=x86-64` (a separate `--target-dir`) to cover the scalar fallback.
- **Benchmark rows name the version** (2026-11-10): `bench_noisy_nearclifford.py` reads it from the installed wheel, so bump the version locally before the timing runs (commit the bump separately as the release).
- **`.text` comparison settles example outliers** (2026-11-10): `objcopy -O binary --only-section=.text` on both builds; `bench`, `stabilizer`, `pauliprop` were identical, so their 1.02–1.06 medians needed no rerun.
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
