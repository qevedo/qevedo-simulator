# Improvement log

One entry per improvement run: what changed, why, what it measured, and
what to try next. Newest first.

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
