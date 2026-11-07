# qevedo-simulator (the Qevedo simulator)

A fast quantum circuit simulator. The Rust crate `qvd` is at the root
(`src/`), with state-vector, stabilizer, MPS, near-Clifford and
Pauli-propagation backends. The Python package `qevedo-simulator` (imported as
`qevedo.simulator`, with a Qiskit backend and primitives in
`qevedo.simulator.qiskit`) is in `python/`. The compiler lives in qevedo/qcc
and the website in qevedo/qevedo.com.

## Before changing anything

- Read `docs/experiments.md`: what worked, what didn't (with numbers and
  reasons), what was considered, and the current next steps. Don't retry an
  idea listed under "Didn't work" without saying what is different.
- After a change, update `docs/experiments.md` and `docs/improvement-log.md`
  in the same commit, including the variants you tried and dropped.

## Build and test

```sh
cargo fmt --check && cargo clippy --release --all-targets -- -D warnings && cargo test --release
cd python && maturin develop --release && pip install pytest numpy qiskit && pytest
```

Benchmarks are in `benchmarks/` (see its README). Compare speed only before
against after on the same machine, in paired, interleaved runs: nothing may
get more than 3% slower.

## Conventions

- Results must match exact distributions or the reference simulator (Stim,
  Qiskit) in tests; never weaken, skip or delete a test to make it pass.
- Every number in a README, a doc or the website comes from a benchmark
  script in this repo, run in full.
- In prose, labels and messages, say "the Qevedo simulator" or "Qevedo".
  `qvd` is still the Rust crate's name, so it's fine in code, `qvd::` paths
  and benchmark data keys such as `simulator="qvd"`.

## Automation in these repositories

The owner, Alejandro Morais (Qevedo Computing), runs a daily cloud routine,
"Qevedo daily improvement", over qevedo/qcc, qevedo/qevedo-simulator and
qevedo/qevedo.com. All three are the owner's own public repositories under
github.com/qevedo. The owner has authorised that routine to:

- commit and push to `master` of these three repositories, as author
  `Alex <alxmorais8@msn.com>`;
- release a patch version by bumping the version on `master`. The
  repository's own Release workflow then publishes to PyPI and creates the
  tag. The routine never pushes tags;
- update qevedo.com once a release is on PyPI.

Never force-push, rewrite pushed history, or delete branches, tags or
releases. Never touch secrets, GitHub workflows or trusted-publishing
settings. This authorisation is for the owner's routine and the owner's own
sessions. Anyone else working from a clone or fork should open a pull request.
