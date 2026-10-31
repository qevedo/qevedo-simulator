//! Stabilizer simulation of Clifford circuits: thousands of qubits,
//! millions of shots.
//!
//! A circuit made only of Clifford gates (H, S, CX, CZ, Paulis, rotations
//! by multiples of π/2, ...) plus measurements and resets keeps the state a
//! stabilizer state, which [`Tableau`] stores in `O(n^2)` bits instead of
//! `2^n` amplitudes. Shots are drawn with Pauli frames, as in Stim: one
//! reference run on the tableau, then every other shot is the reference
//! plus a Pauli "frame" propagated through the circuit, 64 shots per
//! machine word.
//!
//! Pauli noise channels ([`Noise`]) cost nothing in the reference run: a
//! Pauli error only multiplies a shot's frame, so each shot draws its
//! errors independently and adds them to its frame. Errors are rare, so
//! the sampler draws the gaps between them (geometrically distributed)
//! instead of one random number per qubit and shot, as Stim does.

mod tableau;

use std::f64::consts::FRAC_PI_2;
use std::time::Instant;

use rand::{Rng, RngExt, SeedableRng};
use rand_pcg::Pcg64;
use rayon::prelude::*;

use crate::circuit::{Circuit, Gate, Instruction, Noise};
use crate::simulator::{RunResult, counts_from_keys};

pub use tableau::{Layout, PauliString, Tableau};

/// A primitive Clifford operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Clifford {
    H(usize),
    S(usize),
    Sdg(usize),
    X(usize),
    Y(usize),
    Z(usize),
    CX(usize, usize),
    CZ(usize, usize),
    Swap(usize, usize),
}

impl Clifford {
    /// The qubits the gate acts on (one or two).
    pub fn qubits(self) -> impl Iterator<Item = usize> {
        let (qubits, len) = match self {
            Clifford::H(q)
            | Clifford::S(q)
            | Clifford::Sdg(q)
            | Clifford::X(q)
            | Clifford::Y(q)
            | Clifford::Z(q) => ([q, q], 1),
            Clifford::CX(a, b) | Clifford::CZ(a, b) | Clifford::Swap(a, b) => ([a, b], 2),
        };
        qubits.into_iter().take(len)
    }
}

/// `angle / (π/2)` if it is (within 1e-9) an integer, reduced mod 4.
fn quarter_turns(angle: f64) -> Option<u8> {
    let turns = angle / FRAC_PI_2;
    let rounded = turns.round();
    ((turns - rounded).abs() < 1e-9).then(|| rounded.rem_euclid(4.0) as u8)
}

/// `RZ` by `k` quarter turns, up to global phase.
fn rz(k: u8, q: usize, out: &mut Vec<Clifford>) {
    match k {
        1 => out.push(Clifford::S(q)),
        2 => out.push(Clifford::Z(q)),
        3 => out.push(Clifford::Sdg(q)),
        _ => {}
    }
}

/// Lower a gate to primitive Clifford operations, in application order, or
/// `None` if the gate is not Clifford. Global phases are dropped.
pub fn lower(gate: &Gate, qubits: &[usize]) -> Option<Vec<Clifford>> {
    use Clifford as C;
    let q = qubits;
    let mut out = Vec::new();
    match gate {
        Gate::I => {}
        Gate::H => out.push(C::H(q[0])),
        Gate::X => out.push(C::X(q[0])),
        Gate::Y => out.push(C::Y(q[0])),
        Gate::Z => out.push(C::Z(q[0])),
        Gate::S => out.push(C::S(q[0])),
        Gate::Sdg => out.push(C::Sdg(q[0])),
        Gate::SX => out.extend([C::H(q[0]), C::S(q[0]), C::H(q[0])]),
        Gate::SXdg => out.extend([C::H(q[0]), C::Sdg(q[0]), C::H(q[0])]),
        Gate::RZ(t) | Gate::P(t) => rz(quarter_turns(*t)?, q[0], &mut out),
        Gate::RX(t) => {
            // RX = H RZ H.
            let k = quarter_turns(*t)?;
            out.push(C::H(q[0]));
            rz(k, q[0], &mut out);
            out.push(C::H(q[0]));
        }
        Gate::RY(t) => {
            // RY = S RX S†.
            let k = quarter_turns(*t)?;
            out.extend([C::Sdg(q[0]), C::H(q[0])]);
            rz(k, q[0], &mut out);
            out.extend([C::H(q[0]), C::S(q[0])]);
        }
        Gate::U(theta, phi, lambda) => {
            // U(θ, φ, λ) = RZ(φ) RY(θ) RZ(λ) up to global phase.
            let (kt, kp, kl) = (
                quarter_turns(*theta)?,
                quarter_turns(*phi)?,
                quarter_turns(*lambda)?,
            );
            rz(kl, q[0], &mut out);
            out.extend([C::Sdg(q[0]), C::H(q[0])]);
            rz(kt, q[0], &mut out);
            out.extend([C::H(q[0]), C::S(q[0])]);
            rz(kp, q[0], &mut out);
        }
        Gate::CX => out.push(C::CX(q[0], q[1])),
        Gate::CZ => out.push(C::CZ(q[0], q[1])),
        Gate::CY => out.extend([C::Sdg(q[1]), C::CX(q[0], q[1]), C::S(q[1])]),
        Gate::SWAP => out.push(C::Swap(q[0], q[1])),
        Gate::CP(t) => match quarter_turns(*t)? {
            0 => {}
            2 => out.push(C::CZ(q[0], q[1])),
            _ => return None,
        },
        Gate::RZZ(t) => {
            // RZZ(θ) = CX RZ_b(θ) CX.
            let k = quarter_turns(*t)?;
            out.push(C::CX(q[0], q[1]));
            rz(k, q[1], &mut out);
            out.push(C::CX(q[0], q[1]));
        }
        Gate::RXX(t) => {
            let k = quarter_turns(*t)?;
            out.extend([C::H(q[0]), C::H(q[1]), C::CX(q[0], q[1])]);
            rz(k, q[1], &mut out);
            out.extend([C::CX(q[0], q[1]), C::H(q[0]), C::H(q[1])]);
        }
        Gate::T
        | Gate::Tdg
        | Gate::CH
        | Gate::CRX(_)
        | Gate::CRY(_)
        | Gate::CRZ(_)
        | Gate::CCX
        | Gate::CSWAP
        | Gate::Unitary(_) => return None,
    }
    Some(out)
}

/// Whether every gate of `circuit` is Clifford (so the stabilizer backend
/// can run it).
pub fn is_clifford(circuit: &Circuit) -> bool {
    circuit.instructions.iter().all(|i| match i {
        Instruction::Gate { gate, qubits } => lower(gate, qubits).is_some(),
        _ => true,
    })
}

impl Tableau {
    pub fn apply(&mut self, op: Clifford) {
        self.apply_layer(&[op]);
    }

    /// The tableau after running the gates of a circuit (measurements and
    /// resets are applied too, with outcomes drawn from `rng`).
    pub fn from_circuit(circuit: &Circuit, rng: &mut impl Rng) -> Result<Tableau, String> {
        let mut tableau = Tableau::new(circuit.num_qubits);
        let mut layer = Layer::new(circuit.num_qubits);
        for instruction in &circuit.instructions {
            match instruction {
                Instruction::Gate { gate, qubits } => {
                    for op in lower(gate, qubits)
                        .ok_or_else(|| format!("{gate:?} is not a Clifford gate"))?
                    {
                        layer.push(op, &mut tableau);
                    }
                }
                Instruction::Measure { qubit, .. } => {
                    layer.flush(&mut tableau);
                    tableau.measure(*qubit, rng);
                }
                Instruction::Reset { qubit } => {
                    layer.flush(&mut tableau);
                    tableau.reset(*qubit, rng);
                }
                Instruction::Barrier { .. } => {}
                Instruction::Noise { .. } => {
                    return Err("a single tableau cannot follow noise; use sample".into());
                }
            }
        }
        layer.flush(&mut tableau);
        Ok(tableau)
    }
}

/// Pauli frames for a batch of shots: bit `s` of `x[q]` (`z[q]`) is the X
/// (Z) component of shot `s`'s frame on qubit `q`.
struct Frames {
    words: usize,
    x: Vec<u64>,
    z: Vec<u64>,
    /// One random stream per word of 64 shots, so results depend on the
    /// seed but not on how the shots are split between threads.
    rngs: Vec<Pcg64>,
}

impl Frames {
    fn new(qubits: usize, rngs: Vec<Pcg64>) -> Frames {
        let words = rngs.len();
        let mut frames = Frames {
            words,
            x: vec![0; qubits * words],
            z: vec![0; qubits * words],
            rngs,
        };
        // Z errors on |0> change nothing; random ones make the measurements
        // that should be random come out random once gates turn them into
        // X components (Stim's trick).
        for q in 0..qubits {
            frames.randomize_z(q);
        }
        frames
    }

    fn qubit(&self, q: usize) -> std::ops::Range<usize> {
        q * self.words..(q + 1) * self.words
    }

    fn randomize_z(&mut self, q: usize) {
        let r = self.qubit(q);
        for (z, rng) in self.z[r].iter_mut().zip(&mut self.rngs) {
            *z = rng.next_u64();
        }
    }

    /// `a ^= b` between rows of the same table.
    fn xor(table: &mut [u64], words: usize, dst: usize, src: usize) {
        let (d, s) = (dst * words, src * words);
        if d < s {
            let (lo, hi) = table.split_at_mut(s);
            for (a, b) in lo[d..d + words].iter_mut().zip(&hi[..words]) {
                *a ^= b;
            }
        } else {
            let (lo, hi) = table.split_at_mut(d);
            for (a, b) in hi[..words].iter_mut().zip(&lo[s..s + words]) {
                *a ^= b;
            }
        }
    }

    fn swap_tables(&mut self, q: usize) {
        let r = self.qubit(q);
        self.x[r.clone()].swap_with_slice(&mut self.z[r]);
    }

    /// Add the errors of a noise channel on `targets` (qubits, or pairs of
    /// qubits for `Depolarize2`) to every shot's frame.
    ///
    /// In each word, the 64 shots of every target form one run of
    /// `64 * targets` Bernoulli trials ("does this shot get an error on
    /// this target?"). The gap before the next error is geometric, so
    /// with error rate `p` a word costs about `1 + 64 * targets * p` random
    /// draws rather than `64 * targets`. Which error it is comes from a
    /// second draw.
    // Kept out of line: inlined into the step loop it slowed noiseless
    // circuits down by a few percent.
    #[inline(never)]
    fn noise(&mut self, channel: Noise, targets: &[usize]) {
        let p = channel.probability();
        if p <= 0.0 {
            return;
        }
        let arity = channel.arity();
        let trials = targets.len() / arity * 64;
        // ln(1 - p); -inf when p = 1, where every trial is an error.
        let log_miss = (-p).ln_1p();
        let words = self.words;
        for w in 0..words {
            let rng = &mut self.rngs[w];
            let mut next = 0usize;
            loop {
                if log_miss > f64::NEG_INFINITY {
                    // P(gap >= k) = (1 - p)^k for u uniform in (0, 1].
                    let u = 1.0 - rng.random::<f64>();
                    next = next.saturating_add((u.ln() / log_miss) as usize);
                }
                if next >= trials {
                    break;
                }
                let (target, bit) = (next / 64, 1u64 << (next % 64));
                match channel {
                    Noise::Pauli1 { px, py, .. } => {
                        // X, Y or Z in proportion to px, py, pz.
                        let r = rng.random::<f64>() * p;
                        let (x, z) = if r < px {
                            (true, false)
                        } else if r < px + py {
                            (true, true)
                        } else {
                            (false, true)
                        };
                        let i = targets[target] * words + w;
                        if x {
                            self.x[i] ^= bit;
                        }
                        if z {
                            self.z[i] ^= bit;
                        }
                    }
                    Noise::Depolarize2(_) => {
                        // One of the 15 non-identity Paulis on the pair:
                        // the bits of k are (x_a, z_a, x_b, z_b).
                        let k: u32 = rng.random_range(1..16);
                        for (half, q) in targets[2 * target..2 * target + 2].iter().enumerate() {
                            let i = q * words + w;
                            let bits = k >> (2 * half);
                            if bits & 1 == 1 {
                                self.x[i] ^= bit;
                            }
                            if bits & 2 == 2 {
                                self.z[i] ^= bit;
                            }
                        }
                    }
                }
                next += 1;
            }
        }
    }

    fn apply(&mut self, op: Clifford) {
        let w = self.words;
        match op {
            Clifford::H(q) => self.swap_tables(q),
            Clifford::S(q) | Clifford::Sdg(q) => {
                let r = self.qubit(q);
                for i in r {
                    self.z[i] ^= self.x[i];
                }
            }
            Clifford::X(_) | Clifford::Y(_) | Clifford::Z(_) => {}
            Clifford::CX(c, t) => {
                Self::xor(&mut self.x, w, t, c);
                Self::xor(&mut self.z, w, c, t);
            }
            Clifford::CZ(a, b) => {
                // z_a ^= x_b; z_b ^= x_a (x is unchanged).
                for (i, j) in self.qubit(a).zip(self.qubit(b)) {
                    self.z[i] ^= self.x[j];
                    self.z[j] ^= self.x[i];
                }
            }
            Clifford::Swap(a, b) => {
                for (i, j) in self.qubit(a).zip(self.qubit(b)) {
                    self.x.swap(i, j);
                    self.z.swap(i, j);
                }
            }
        }
    }
}

/// Measurement results of many shots, bit-packed: one bit per shot per
/// classical bit, 64 shots per word.
#[derive(Clone, Debug)]
pub struct Samples {
    pub shots: usize,
    pub num_clbits: usize,
    words: usize,
    /// Classical bit `c` of shot `s` is bit `s % 64` of `bits[c * words + s / 64]`.
    bits: Vec<u64>,
}

impl Samples {
    /// Classical bit `clbit` of shot `shot`.
    pub fn get(&self, shot: usize, clbit: usize) -> bool {
        assert!(shot < self.shots && clbit < self.num_clbits);
        (self.bits[clbit * self.words + shot / 64] >> (shot % 64)) & 1 == 1
    }

    /// The shots of classical bit `clbit`, 64 per word (bits past `shots`
    /// in the last word are unspecified).
    pub fn clbit_words(&self, clbit: usize) -> &[u64] {
        &self.bits[clbit * self.words..(clbit + 1) * self.words]
    }

    /// Count the outcomes as bitstrings (classical bit 0 on the right).
    pub fn counts(&self) -> std::collections::BTreeMap<String, usize> {
        if self.num_clbits <= 128 {
            let mut keys = vec![0u128; self.shots];
            for c in 0..self.num_clbits {
                let words = self.clbit_words(c);
                for (s, key) in keys.iter_mut().enumerate() {
                    *key |= (((words[s / 64] >> (s % 64)) & 1) as u128) << c;
                }
            }
            return counts_from_keys(keys, self.num_clbits);
        }
        let mut counts = std::collections::BTreeMap::new();
        for s in 0..self.shots {
            let bits: String = (0..self.num_clbits)
                .rev()
                .map(|c| if self.get(s, c) { '1' } else { '0' })
                .collect();
            *counts.entry(bits).or_default() += 1;
        }
        counts
    }
}

/// Gates waiting to be applied as one layer: consecutive gates on pairwise
/// disjoint qubits, which commute.
struct Layer {
    ops: Vec<Clifford>,
    /// `stamp[q] == current` iff qubit `q` is used by a gate in `ops`.
    stamp: Vec<u32>,
    current: u32,
}

impl Layer {
    fn new(qubits: usize) -> Layer {
        Layer {
            ops: Vec::new(),
            stamp: vec![0; qubits],
            current: 1,
        }
    }

    /// Add a gate, first applying the layer if the gate shares a qubit
    /// with it.
    fn push(&mut self, op: Clifford, tableau: &mut Tableau) {
        if op.qubits().any(|q| self.stamp[q] == self.current) {
            self.flush(tableau);
        }
        for q in op.qubits() {
            self.stamp[q] = self.current;
        }
        self.ops.push(op);
    }

    fn flush(&mut self, tableau: &mut Tableau) {
        if !self.ops.is_empty() {
            tableau.apply_layer(&self.ops);
            self.ops.clear();
            self.current += 1;
        }
    }
}

/// A circuit step with the reference outcome of measurements filled in.
#[derive(Clone, Copy)]
enum Step {
    Gate(Clifford),
    Measure {
        qubit: usize,
        clbit: usize,
        reference: bool,
    },
    Reset {
        qubit: usize,
    },
    /// The noise channel `noises[index]`. Kept out of line so that every
    /// step stays as small as before (the steps are walked in cache-sized
    /// segments, and doubling their size slowed noiseless circuits down).
    Noise {
        index: usize,
    },
}

/// A noise channel step: the channel and its targets,
/// `noise_targets[start..end]`.
struct NoiseStep {
    channel: Noise,
    start: usize,
    end: usize,
}

/// Noise channels waiting to be added to the steps. A Pauli channel
/// commutes with every gate on other qubits and with every other Pauli
/// channel, so channels are held back until a gate, measurement or reset
/// touches one of their qubits. That gathers the per-gate channels of a
/// layer (`CX a b; DEPOLARIZE2 a b; CX c d; DEPOLARIZE2 c d; ...`) into one
/// step, whose gaps between errors then span the whole layer.
struct PendingNoise {
    groups: Vec<(Noise, Vec<usize>)>,
    noisy: Vec<bool>,
}

impl PendingNoise {
    fn add(&mut self, channel: Noise, qubits: &[usize]) {
        match self.groups.iter_mut().find(|(c, _)| *c == channel) {
            Some((_, targets)) => targets.extend_from_slice(qubits),
            None => self.groups.push((channel, qubits.to_vec())),
        }
        for &q in qubits {
            self.noisy[q] = true;
        }
    }

    /// Emit the held channels if any of `qubits` has one.
    fn before(
        &mut self,
        qubits: impl IntoIterator<Item = usize>,
        steps: &mut Vec<Step>,
        noises: &mut Vec<NoiseStep>,
        targets: &mut Vec<usize>,
    ) {
        if !self.groups.is_empty() && qubits.into_iter().any(|q| self.noisy[q]) {
            self.flush(steps, noises, targets);
        }
    }

    fn flush(
        &mut self,
        steps: &mut Vec<Step>,
        noises: &mut Vec<NoiseStep>,
        targets: &mut Vec<usize>,
    ) {
        for (channel, qubits) in self.groups.drain(..) {
            for &q in &qubits {
                self.noisy[q] = false;
            }
            let start = targets.len();
            targets.extend(qubits);
            steps.push(Step::Noise {
                index: noises.len(),
            });
            noises.push(NoiseStep {
                channel,
                start,
                end: targets.len(),
            });
        }
    }
}

/// A pointer that may be shared between threads writing disjoint ranges.
#[derive(Clone, Copy)]
struct Shared(*mut u64);
// SAFETY: only used to write disjoint words from different threads.
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

impl Shared {
    fn get(self) -> *mut u64 {
        self.0
    }
}

/// Sample `shots` shots of a Clifford circuit.
///
/// One reference shot runs on the tableau; the others are Pauli frames
/// relative to it, 64 shots per word. The frames are split into blocks of
/// a few words that run through the whole circuit independently, in
/// parallel and in cache. Noise channels are sampled independently in every
/// shot. Fails if the circuit contains a non-Clifford gate.
pub fn sample(circuit: &Circuit, shots: usize, seed: Option<u64>) -> Result<Samples, String> {
    sample_with_reference(circuit, shots, &mut seeded(seed)).map(|(samples, _)| samples)
}

fn seeded(seed: Option<u64>) -> Pcg64 {
    match seed {
        Some(seed) => Pcg64::seed_from_u64(seed),
        None => Pcg64::from_rng(&mut rand::rng()),
    }
}

/// [`sample`], also returning the reference shot: the last result written
/// to each classical bit in the noiseless run on the tableau.
fn sample_with_reference(
    circuit: &Circuit,
    shots: usize,
    rng: &mut Pcg64,
) -> Result<(Samples, Vec<bool>), String> {
    let mut reference_bits = vec![false; circuit.num_clbits];
    let n = circuit.num_qubits;
    // Reference shot, with gates on disjoint qubits applied in parallel.
    let mut tableau = Tableau::new(n);
    let mut layer = Layer::new(n);
    let mut steps = Vec::with_capacity(circuit.instructions.len());
    let mut noises = Vec::new();
    let mut noise_targets = Vec::new();
    let mut pending = PendingNoise {
        groups: Vec::new(),
        noisy: vec![false; n],
    };
    for instruction in &circuit.instructions {
        match instruction {
            Instruction::Gate { gate, qubits } => {
                for op in
                    lower(gate, qubits).ok_or_else(|| format!("{gate:?} is not a Clifford gate"))?
                {
                    pending.before(op.qubits(), &mut steps, &mut noises, &mut noise_targets);
                    layer.push(op, &mut tableau);
                    steps.push(Step::Gate(op));
                }
            }
            Instruction::Measure { qubit, clbit } => {
                pending.before([*qubit], &mut steps, &mut noises, &mut noise_targets);
                layer.flush(&mut tableau);
                let reference = tableau.measure(*qubit, rng);
                reference_bits[*clbit] = reference;
                steps.push(Step::Measure {
                    qubit: *qubit,
                    clbit: *clbit,
                    reference,
                });
            }
            Instruction::Reset { qubit } => {
                pending.before([*qubit], &mut steps, &mut noises, &mut noise_targets);
                layer.flush(&mut tableau);
                tableau.reset(*qubit, rng);
                steps.push(Step::Reset { qubit: *qubit });
            }
            Instruction::Barrier { .. } => {}
            // The reference shot is noiseless; errors live in the frames.
            Instruction::Noise { channel, qubits } => pending.add(*channel, qubits),
        }
    }
    pending.flush(&mut steps, &mut noises, &mut noise_targets);
    drop((tableau, layer, pending));
    // Frames, one block per thread, at most 2048 shots each so a block's
    // frames stay in cache. Run inside a pool of performance cores: the
    // blocks synchronise after every segment of steps, so a slow core holds
    // up all the others.
    let words = shots.div_ceil(64);
    let block = words.div_ceil(rayon::current_num_threads()).clamp(1, 32);
    let mut rngs: Vec<Pcg64> = (0..words)
        .map(|_| Pcg64::seed_from_u64(rng.next_u64()))
        .collect();
    // Latest measurement record of each classical bit, one bit per shot.
    let mut bits = vec![0u64; circuit.num_clbits * words];
    let out = Shared(bits.as_mut_ptr());
    let mut blocks: Vec<(usize, Frames)> = Vec::new();
    while !rngs.is_empty() {
        let rest = rngs.split_off(block.min(rngs.len()));
        blocks.push((
            words - rngs.len() - rest.len(),
            Frames::new(n, std::mem::replace(&mut rngs, rest)),
        ));
    }
    // All blocks walk the same cache-sized segment of steps at a time.
    for segment in steps.chunks(1 << 15) {
        blocks.par_iter_mut().for_each(|(first, frames)| {
            for step in segment {
                match *step {
                    Step::Gate(op) => frames.apply(op),
                    Step::Measure {
                        qubit,
                        clbit,
                        reference,
                    } => {
                        let fill = if reference { u64::MAX } else { 0 };
                        for (i, &x) in frames.x[frames.qubit(qubit)].iter().enumerate() {
                            // SAFETY: this block alone writes words
                            // `first..first + width` of each classical bit.
                            unsafe { *out.get().add(clbit * words + *first + i) = x ^ fill };
                        }
                        frames.randomize_z(qubit);
                    }
                    Step::Reset { qubit } => {
                        let r = frames.qubit(qubit);
                        frames.x[r].fill(0);
                        frames.randomize_z(qubit);
                    }
                    Step::Noise { index } => {
                        let step = &noises[index];
                        frames.noise(step.channel, &noise_targets[step.start..step.end]);
                    }
                }
            }
        });
    }
    let samples = Samples {
        shots,
        num_clbits: circuit.num_clbits,
        words,
        bits,
    };
    Ok((samples, reference_bits))
}

/// Detection events and observable flips of many shots, bit-packed as
/// [`Samples`] is: one row of `ceil(shots / 64)` words per detector and
/// per observable.
#[derive(Clone, Debug)]
pub struct DetectorSamples {
    pub shots: usize,
    pub num_detectors: usize,
    pub num_observables: usize,
    words: usize,
    /// Detector `d` of shot `s` is bit `s % 64` of `detectors[d * words + s / 64]`.
    detectors: Vec<u64>,
    observables: Vec<u64>,
}

impl DetectorSamples {
    /// Whether detector `detector` fired in shot `shot`.
    pub fn detector(&self, shot: usize, detector: usize) -> bool {
        assert!(shot < self.shots && detector < self.num_detectors);
        (self.detectors[detector * self.words + shot / 64] >> (shot % 64)) & 1 == 1
    }

    /// Whether observable `observable` flipped in shot `shot`.
    pub fn observable(&self, shot: usize, observable: usize) -> bool {
        assert!(shot < self.shots && observable < self.num_observables);
        (self.observables[observable * self.words + shot / 64] >> (shot % 64)) & 1 == 1
    }

    /// The shots of detector `detector`, 64 per word (bits past `shots` in
    /// the last word are zero).
    pub fn detector_words(&self, detector: usize) -> &[u64] {
        &self.detectors[detector * self.words..(detector + 1) * self.words]
    }

    /// The shots of observable `observable`, 64 per word (bits past
    /// `shots` in the last word are zero).
    pub fn observable_words(&self, observable: usize) -> &[u64] {
        &self.observables[observable * self.words..(observable + 1) * self.words]
    }

    /// The detection events shot-major, one byte (0 or 1) per detector:
    /// byte `s * num_detectors + d` is detector `d` of shot `s`. This is
    /// the layout of Stim's detector sampler and of decoders' batch input.
    pub fn detectors_shot_major(&self) -> Vec<u8> {
        let mut out = vec![0; self.shots * self.num_detectors];
        self.write_detectors_shot_major(&mut out);
        out
    }

    /// The observable flips shot-major, as [`Self::detectors_shot_major`].
    pub fn observables_shot_major(&self) -> Vec<u8> {
        let mut out = vec![0; self.shots * self.num_observables];
        self.write_observables_shot_major(&mut out);
        out
    }

    /// [`Self::detectors_shot_major`] into a buffer of
    /// `shots * num_detectors` bytes (a NumPy array's, say).
    pub fn write_detectors_shot_major(&self, out: &mut [u8]) {
        shot_major(
            &self.detectors,
            self.num_detectors,
            self.words,
            self.shots,
            out,
        );
    }

    /// [`Self::observables_shot_major`] into a buffer of
    /// `shots * num_observables` bytes.
    pub fn write_observables_shot_major(&self, out: &mut [u8]) {
        shot_major(
            &self.observables,
            self.num_observables,
            self.words,
            self.shots,
            out,
        );
    }
}

/// Transpose `count` bit-packed rows of `words` words into `shots` rows of
/// `count` bytes in `out`.
///
/// Detection events are rare, so this walks the set bits of each word
/// (`trailing_zeros`) rather than every bit. Each block of 64 shots is one
/// task, which zeroes its own `64 * count` bytes and then sets its ones
/// while they are in cache (and, for a fresh allocation, touches its pages
/// first, in parallel).
fn shot_major(rows: &[u64], count: usize, words: usize, shots: usize, out: &mut [u8]) {
    assert_eq!(out.len(), shots * count);
    if count == 0 {
        return;
    }
    out.par_chunks_mut(64 * count)
        .enumerate()
        .for_each(|(w, block)| {
            block.fill(0);
            for r in 0..count {
                let mut word = rows[r * words + w];
                while word != 0 {
                    let bit = word.trailing_zeros() as usize;
                    block[bit * count + r] = 1;
                    word &= word - 1;
                }
            }
        });
}

/// Measurement results of a circuit read from Stim's format, with its
/// measurement flips (`M(p)`) and inverted results (`M !q`) applied.
/// Measurement `m` is classical bit `m`.
pub fn sample_stim(
    circuit: &crate::stim::StimCircuit,
    shots: usize,
    seed: Option<u64>,
) -> Result<Samples, String> {
    let mut rng = seeded(seed);
    let (mut samples, _) = sample_with_reference(&circuit.circuit, shots, &mut rng)?;
    flip_records(&mut samples, circuit, &mut rng);
    for &m in &circuit.inverted {
        for word in &mut samples.bits[m * samples.words..(m + 1) * samples.words] {
            *word = !*word;
        }
    }
    Ok(samples)
}

/// Sample detection events and observable flips of a circuit read from
/// Stim's format, as Stim's detector sampler does.
///
/// A detector fires when the parity of its measurements differs from the
/// parity in the noiseless circuit; an observable flips likewise. The
/// noiseless parities come from the reference shot, so detectors must be
/// deterministic without noise (as in every error-correction circuit;
/// a detector that is not is random, as in Stim). Inverted results
/// (`M !q`) change a measurement and its reference alike and so cancel.
pub fn sample_detectors(
    circuit: &crate::stim::StimCircuit,
    shots: usize,
    seed: Option<u64>,
) -> Result<DetectorSamples, String> {
    let mut rng = seeded(seed);
    let (mut samples, reference) = sample_with_reference(&circuit.circuit, shots, &mut rng)?;
    flip_records(&mut samples, circuit, &mut rng);
    let words = samples.words;
    // Parities relative to the reference, one row of words each.
    let parities = |groups: &[Vec<usize>]| {
        let mut out = vec![0u64; groups.len() * words];
        out.par_chunks_mut(words.max(1))
            .zip(groups)
            .for_each(|(row, group)| {
                let mut flipped = false;
                for &m in group {
                    flipped ^= reference[m];
                    for (a, b) in row.iter_mut().zip(samples.clbit_words(m)) {
                        *a ^= b;
                    }
                }
                if flipped {
                    for a in row.iter_mut() {
                        *a = !*a;
                    }
                }
                // Zero the bits past the last shot.
                if !shots.is_multiple_of(64) {
                    if let Some(last) = row.last_mut() {
                        *last &= (1u64 << (shots % 64)) - 1;
                    }
                }
            });
        out
    };
    Ok(DetectorSamples {
        shots,
        num_detectors: circuit.detectors.len(),
        num_observables: circuit.observables.len(),
        words,
        detectors: parities(&circuit.detectors),
        observables: parities(&circuit.observables),
    })
}

/// Flip each recorded result of `circuit.flips` with its probability,
/// independently in every shot. A flip of the record leaves the state
/// alone, so it can be applied after the run: each measurement of a Stim
/// circuit has its own classical bit.
///
/// The measurements with the same probability form one run of
/// `measurements * shots` Bernoulli trials, and the gaps between flips are
/// drawn (geometrically distributed), as the noise channels do: about
/// `1 + measurements * shots * p` draws instead of one per shot.
fn flip_records(samples: &mut Samples, circuit: &crate::stim::StimCircuit, rng: &mut Pcg64) {
    let mut groups: Vec<(f64, Vec<usize>)> = Vec::new();
    for &(m, p) in &circuit.flips {
        match groups.iter_mut().find(|(q, _)| *q == p) {
            Some((_, ms)) => ms.push(m),
            None => groups.push((p, vec![m])),
        }
    }
    let (shots, words) = (samples.shots, samples.words);
    for (p, measurements) in groups {
        let trials = measurements.len() * shots;
        let log_miss = (-p).ln_1p();
        let mut next = 0usize;
        loop {
            if log_miss > f64::NEG_INFINITY {
                let u = 1.0 - rng.random::<f64>();
                next = next.saturating_add((u.ln() / log_miss) as usize);
            }
            if next >= trials {
                break;
            }
            let (m, shot) = (measurements[next / shots], next % shots);
            samples.bits[m * words + shot / 64] ^= 1u64 << (shot % 64);
            next += 1;
        }
    }
}

/// The circuit in Stim's format, so Stim can sample exactly the same noisy
/// circuit: gates are lowered to Stim's `H`, `S`, `S_DAG`, Paulis, `CX`,
/// `CZ` and `SWAP` (dropping global phases), noise channels keep their
/// Stim names. Stim numbers measurement results in the order they happen,
/// which are qvd's classical bits when each bit is written once, in
/// increasing order (as in [`crate::library::surface_code`]). Fails on
/// non-Clifford gates.
pub fn to_stim(circuit: &Circuit) -> Result<String, String> {
    // Consecutive instructions with the same name and arguments share a
    // line, as in Stim's own circuits; Stim applies a line's targets in
    // order, so this is the same circuit.
    let mut out = String::new();
    let mut line = String::new();
    let mut emit = |head: String, targets: &mut dyn Iterator<Item = usize>| {
        if !line.starts_with(&head) || line.as_bytes().get(head.len()) != Some(&b' ') {
            if !line.is_empty() {
                out.push_str(&line);
                out.push('\n');
            }
            line = head;
        }
        for q in targets {
            line.push(' ');
            line.push_str(&q.to_string());
        }
    };
    for instruction in &circuit.instructions {
        match instruction {
            Instruction::Gate { gate, qubits } => {
                for op in
                    lower(gate, qubits).ok_or_else(|| format!("{gate:?} is not a Clifford gate"))?
                {
                    let name = match op {
                        Clifford::H(_) => "H",
                        Clifford::S(_) => "S",
                        Clifford::Sdg(_) => "S_DAG",
                        Clifford::X(_) => "X",
                        Clifford::Y(_) => "Y",
                        Clifford::Z(_) => "Z",
                        Clifford::CX(..) => "CX",
                        Clifford::CZ(..) => "CZ",
                        Clifford::Swap(..) => "SWAP",
                    };
                    emit(name.to_string(), &mut op.qubits());
                }
            }
            Instruction::Measure { qubit, .. } => emit("M".into(), &mut std::iter::once(*qubit)),
            Instruction::Reset { qubit } => emit("R".into(), &mut std::iter::once(*qubit)),
            Instruction::Barrier { .. } => {}
            Instruction::Noise { channel, qubits } => {
                let (name, args) = channel.stim_name();
                let args: Vec<String> = args.iter().map(|p| format!("{p:?}")).collect();
                emit(
                    format!("{name}({})", args.join(", ")),
                    &mut qubits.iter().copied(),
                );
            }
        }
    }
    if !line.is_empty() {
        out.push_str(&line);
        out.push('\n');
    }
    Ok(out)
}

/// Run a Clifford circuit for `shots` shots and count the outcomes.
///
/// Fails if the circuit contains a non-Clifford gate. For circuits with
/// many classical bits, [`sample`] avoids building one string per shot.
pub fn run(circuit: &Circuit, shots: usize, seed: Option<u64>) -> Result<RunResult, String> {
    let start = Instant::now();
    let samples = sample(circuit, shots, seed)?;
    let gate_seconds = start.elapsed().as_secs_f64();
    let count_start = Instant::now();
    let mut result = RunResult {
        counts: samples.counts(),
        ..RunResult::default()
    };
    result.stats.gates = circuit.gate_count();
    result.stats.gate_seconds = gate_seconds;
    result.stats.measure_seconds = count_start.elapsed().as_secs_f64();
    Ok(result)
}
