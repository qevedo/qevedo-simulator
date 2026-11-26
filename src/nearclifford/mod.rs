//! Near-Clifford simulation: a Clifford frame around a small dense state.
//!
//! The state is `|ψ> = C (|φ>_A ⊗ |0...0>_D)`, after the design of Clifft
//! (Chase & Labib, arXiv:2604.27058). `C` is a Clifford on `n` qubits, kept
//! as an inverse stabilizer tableau, and the virtual qubits are split into
//! a few *active* ones, `A`, whose joint state `φ` is a dense vector, and
//! *dormant* ones, `D`, all in `|0>`.
//!
//! - A Clifford gate `G` only updates the frame: `C -> G C`.
//! - Every other gate is lowered to Cliffords and Pauli rotations
//!   `exp(-iθ/2 P)`. Pulled back through the frame, `P` becomes a virtual
//!   Pauli `P'`. On dormant qubits its Z parts act as `+1`. If it has no X
//!   part there, the rotation acts on `φ` alone. Otherwise the dormant X
//!   parts are gathered onto one dormant qubit by CX gates prepended to the
//!   frame (`C -> C W`, and `W` fixes `|0...0>`, so `φ` does not change),
//!   and that qubit becomes active: the dense state doubles.
//! - A Z measurement pulls `Z_q` back the same way. With an X part on a
//!   dormant qubit `d`, the outcome is uniformly random and the collapsed
//!   state is `C V (φ ⊗ |0>)` for a Clifford `V` (a Pauli on `A` controlled
//!   by `d`, then H and maybe X on `d`): only the frame changes. Otherwise
//!   the measurement acts on `φ`. After it, a Clifford on the active qubits
//!   turns the measured Pauli into `Z` on one of them, which then factors
//!   out as dormant: the dense state halves.
//!
//! The cost is exponential only in the number of active qubits, which
//! grows by at most one per non-Clifford rotation and shrinks at
//! measurements. With no non-Clifford gates nothing is ever active and
//! this is a (slower) stabilizer simulator; with many it is a state vector.
//!
//! Pauli noise (the channels of [`crate::Noise`]) runs as Clifft does it: each shot
//! carries a Pauli frame `P_s` on the virtual qubits, and is in the state
//! `C P_s (φ ⊗ |0...0>)`. An error `E` on physical qubits is the virtual
//! Pauli `C† E C`, multiplied into the frames of the shots it hits.
//! Clifford gates leave the frames alone (`G C P_s = (G C) P_s`); Cliffords
//! prepended to `C` conjugate them. A measurement's outcome flips in the
//! shots whose frame anticommutes with the measured Pauli. A rotation
//! `exp(-iθ/2 P')` moves through `P_s` as `exp(+iθ/2 P')` where they
//! anticommute, so those shots need their own dense state: a group of
//! shots that share `C` and `φ` splits in two there, and only there, unless
//! the rotation brings a qubit in `|0>` into the dense part, where a Z on
//! that qubit in the frame makes up for the sign (see `noisy::Op::Rotate`).
//! With few errors, most shots stay in one group and the dense work is
//! done once for all of them; with many errors and rotations on qubits
//! already in the dense part, the groups approach one per shot, which is
//! Clifft's per-shot execution.

mod dense;
mod noisy;

use std::f64::consts::FRAC_PI_4;
use std::sync::Arc;
use std::time::Instant;

use rand::{Rng, RngExt, SeedableRng};
use rand_pcg::Pcg64;

use crate::circuit::{Circuit, Gate, Instruction};
use crate::matrix::Matrix;
use crate::simulator::{RunResult, bitstring};
use crate::stabilizer::{Clifford, Tableau, lower};

use dense::{Dense, Pauli};

/// A step of a lowered gate.
#[derive(Clone, Debug, PartialEq)]
pub enum Step {
    Clifford(Clifford),
    /// `exp(-i θ/2 P)` for the Pauli with these `(qubit, 'X' | 'Y' | 'Z')`
    /// factors.
    Rotation(Vec<(usize, char)>, f64),
}

fn rotation(factors: &[(usize, char)], theta: f64) -> Step {
    Step::Rotation(factors.to_vec(), theta)
}

/// The ZYZ Euler angles `(θ, φ, λ)` of a one-qubit unitary, up to global
/// phase: `U ∝ RZ(φ) RY(θ) RZ(λ)`.
fn euler_angles(m: &Matrix) -> (f64, f64, f64) {
    let (a, b, c, d) = (m[(0, 0)], m[(0, 1)], m[(1, 0)], m[(1, 1)]);
    let theta = 2.0 * c.norm().atan2(a.norm());
    // Remove the global phase so that the determinant is 1.
    let det = a * d - b * c;
    let phase = det.sqrt();
    let (a, c) = (a / phase, c / phase);
    // a = cos(θ/2) e^{-i(φ+λ)/2}, c = sin(θ/2) e^{i(φ-λ)/2}.
    let sum = -2.0 * a.arg();
    let difference = 2.0 * c.arg();
    if c.norm() < 1e-12 {
        (theta, sum, 0.0)
    } else if a.norm() < 1e-12 {
        (theta, difference, 0.0)
    } else {
        (theta, (sum + difference) / 2.0, (sum - difference) / 2.0)
    }
}

/// Lower a gate to Clifford steps and Pauli rotations (exactly, up to
/// global phase), or `None` for gates this backend does not support
/// (arbitrary unitaries on two or more qubits).
pub fn decompose(gate: &Gate, qubits: &[usize]) -> Option<Vec<Step>> {
    if let Some(ops) = lower(gate, qubits) {
        return Some(ops.into_iter().map(Step::Clifford).collect());
    }
    let q = qubits[0];
    let steps = match *gate {
        Gate::T => vec![rotation(&[(q, 'Z')], FRAC_PI_4)],
        Gate::Tdg => vec![rotation(&[(q, 'Z')], -FRAC_PI_4)],
        Gate::RZ(t) | Gate::P(t) => vec![rotation(&[(q, 'Z')], t)],
        Gate::RX(t) => vec![rotation(&[(q, 'X')], t)],
        Gate::RY(t) => vec![rotation(&[(q, 'Y')], t)],
        Gate::U(theta, phi, lambda) => vec![
            rotation(&[(q, 'Z')], lambda),
            rotation(&[(q, 'Y')], theta),
            rotation(&[(q, 'Z')], phi),
        ],
        Gate::Unitary(ref m) if qubits.len() == 1 => {
            let (theta, phi, lambda) = euler_angles(m);
            vec![
                rotation(&[(q, 'Z')], lambda),
                rotation(&[(q, 'Y')], theta),
                rotation(&[(q, 'Z')], phi),
            ]
        }
        // CP(λ) = exp(iλ|11><11|) ∝ RZ_c(λ/2) RZ_t(λ/2) R_ZZ(-λ/2).
        Gate::CP(l) => {
            let (c, t) = (qubits[0], qubits[1]);
            vec![
                rotation(&[(c, 'Z')], l / 2.0),
                rotation(&[(t, 'Z')], l / 2.0),
                rotation(&[(c, 'Z'), (t, 'Z')], -l / 2.0),
            ]
        }
        // CR_P(θ) = exp(-iθ/2 P_t |1><1|_c) = R_{P_t}(θ/2) R_{Z_c P_t}(-θ/2).
        Gate::CRX(t) | Gate::CRY(t) | Gate::CRZ(t) => {
            let p = match gate {
                Gate::CRX(_) => 'X',
                Gate::CRY(_) => 'Y',
                _ => 'Z',
            };
            let (c, target) = (qubits[0], qubits[1]);
            vec![
                rotation(&[(target, p)], t / 2.0),
                rotation(&[(c, 'Z'), (target, p)], -t / 2.0),
            ]
        }
        Gate::RZZ(t) => vec![rotation(&[(qubits[0], 'Z'), (qubits[1], 'Z')], t)],
        Gate::RXX(t) => vec![rotation(&[(qubits[0], 'X'), (qubits[1], 'X')], t)],
        // Qiskit's definition: S H T CX T† H S† on the target.
        Gate::CH => {
            let (c, t) = (qubits[0], qubits[1]);
            vec![
                Step::Clifford(Clifford::S(t)),
                Step::Clifford(Clifford::H(t)),
                rotation(&[(t, 'Z')], FRAC_PI_4),
                Step::Clifford(Clifford::CX(c, t)),
                rotation(&[(t, 'Z')], -FRAC_PI_4),
                Step::Clifford(Clifford::H(t)),
                Step::Clifford(Clifford::Sdg(t)),
            ]
        }
        Gate::CCX => toffoli(qubits[0], qubits[1], qubits[2]),
        Gate::CSWAP => {
            let (c, a, b) = (qubits[0], qubits[1], qubits[2]);
            let mut steps = vec![Step::Clifford(Clifford::CX(b, a))];
            steps.extend(toffoli(c, a, b));
            steps.push(Step::Clifford(Clifford::CX(b, a)));
            steps
        }
        _ => return None,
    };
    Some(steps)
}

/// The standard seven-T Toffoli.
fn toffoli(a: usize, b: usize, t: usize) -> Vec<Step> {
    let (h, cx) = (Clifford::H, Clifford::CX);
    let tg = |q| rotation(&[(q, 'Z')], FRAC_PI_4);
    let tdg = |q| rotation(&[(q, 'Z')], -FRAC_PI_4);
    vec![
        Step::Clifford(h(t)),
        Step::Clifford(cx(b, t)),
        tdg(t),
        Step::Clifford(cx(a, t)),
        tg(t),
        Step::Clifford(cx(b, t)),
        tdg(t),
        Step::Clifford(cx(a, t)),
        tg(b),
        tg(t),
        Step::Clifford(h(t)),
        Step::Clifford(cx(a, b)),
        tg(a),
        tdg(b),
        Step::Clifford(cx(a, b)),
    ]
}

/// Whether every gate of the circuit can be lowered. Noise channels are
/// all supported (see the module documentation).
pub fn is_supported(circuit: &Circuit) -> bool {
    circuit.instructions.iter().all(|i| match i {
        Instruction::Gate { gate, qubits } => decompose(gate, qubits).is_some(),
        // The shots' branches share one record per group of shots, not
        // one per shot, so a Pauli that depends on a result has no place.
        Instruction::Feedback { .. } => false,
        _ => true,
    })
}

/// How many non-Clifford rotations the circuit has: an upper bound on the
/// number of active qubits, since each activates at most one.
pub fn rotation_count(circuit: &Circuit) -> usize {
    circuit
        .instructions
        .iter()
        .filter_map(|i| match i {
            Instruction::Gate { gate, qubits } => decompose(gate, qubits),
            _ => None,
        })
        .flatten()
        .filter(|s| matches!(s, Step::Rotation(..)))
        .count()
}

/// The largest number of active qubits the circuit reaches, found by
/// running only its Clifford frame (activations depend neither on
/// measurement outcomes nor on errors, which only touch the shots' Pauli
/// frames), or `None` if it has unsupported gates. The dense
/// part needs `16 * 2^peak` bytes.
pub fn peak_active_qubits(circuit: &Circuit) -> Option<usize> {
    if !is_supported(circuit) {
        return None;
    }
    let mut state = NearClifford::new(circuit.num_qubits);
    state.symbolic = true;
    let mut rng = Pcg64::seed_from_u64(0);
    for instruction in &circuit.instructions {
        match instruction {
            Instruction::Gate { gate, qubits } => state.apply(gate, qubits).ok()?,
            Instruction::Measure { qubit, .. } => {
                state.measure(*qubit, &mut rng);
            }
            Instruction::Reset { qubit } => state.reset(*qubit, &mut rng),
            Instruction::Barrier { .. } | Instruction::Noise { .. } => {}
            Instruction::Feedback { .. } => unreachable!("not supported"),
        }
    }
    Some(state.peak_active_qubits())
}

/// A near-Clifford state; see the module documentation.
#[derive(Clone, Debug)]
pub struct NearClifford {
    n: usize,
    /// The inverse tableau of `C`; its columns are the virtual qubits.
    frame: Tableau,
    /// Shared copy-on-write between branches of a measurement tree.
    dense: Arc<Dense>,
    /// The probability last computed by [`NearClifford::probability_one`].
    cached: Option<(usize, f64)>,
    /// The virtual qubit at each dense position.
    active: Vec<usize>,
    peak: usize,
    /// Track only which qubits are active, without amplitudes (see
    /// [`peak_active_qubits`]).
    symbolic: bool,
    /// When set, every Clifford prepended to the frame is also recorded
    /// here, for Pauli frames that must follow the frame (see
    /// [`sample_frames`]).
    log: Option<Vec<Clifford>>,
}

/// A virtual Pauli: sign and x/z bit words over the virtual qubits.
type Virtual = (bool, Vec<u64>, Vec<u64>);

fn bit(words: &[u64], q: usize) -> bool {
    words[q / 64] >> (q % 64) & 1 == 1
}

fn set_bits(words: &[u64]) -> impl Iterator<Item = usize> + '_ {
    words.iter().enumerate().flat_map(|(w, &word)| {
        std::iter::successors((word != 0).then_some(word), |&rest| {
            Some(rest & (rest - 1)).filter(|&r| r != 0)
        })
        .map(move |rest| w * 64 + rest.trailing_zeros() as usize)
    })
}

impl NearClifford {
    /// `|0...0>` on `n` qubits: nothing active.
    pub fn new(n: usize) -> NearClifford {
        NearClifford {
            n,
            frame: Tableau::new(n),
            dense: Arc::new(Dense::new()),
            cached: None,
            active: Vec::new(),
            peak: 0,
            symbolic: false,
            log: None,
        }
    }

    pub fn num_qubits(&self) -> usize {
        self.n
    }

    /// `C -> C V` for a Clifford `V` on virtual qubits that the caller
    /// compensates for (it fixes the state, or `φ` was changed by `V†`).
    fn prepend(&mut self, op: Clifford) {
        self.frame.prepend(op);
        if let Some(log) = &mut self.log {
            log.push(op);
        }
    }

    /// The number of active qubits now.
    pub fn active_qubits(&self) -> usize {
        self.active.len()
    }

    /// The largest number of active qubits so far.
    pub fn peak_active_qubits(&self) -> usize {
        self.peak
    }

    /// Apply a gate; fails for unsupported gates.
    pub fn apply(&mut self, gate: &Gate, qubits: &[usize]) -> Result<(), String> {
        let steps = decompose(gate, qubits)
            .ok_or_else(|| format!("{gate:?} is not supported by the near-Clifford backend"))?;
        for step in steps {
            self.step(step);
        }
        Ok(())
    }

    pub fn step(&mut self, step: Step) {
        self.cached = None;
        match step {
            Step::Clifford(op) => self.frame.apply(op),
            Step::Rotation(factors, theta) => self.rotate(&factors, theta),
        }
    }

    fn position(&self, virtual_qubit: usize) -> Option<usize> {
        self.active.iter().position(|&v| v == virtual_qubit)
    }

    /// The active part of a virtual Pauli, as a Pauli on positions (its Z
    /// parts on dormant qubits act as `+1` and are dropped).
    fn on_positions(&self, (negative, x, z): &Virtual) -> Pauli {
        let mut p = Pauli {
            x: 0,
            z: 0,
            negative: *negative,
        };
        for (position, &v) in self.active.iter().enumerate() {
            p.x |= (bit(x, v) as u64) << position;
            p.z |= (bit(z, v) as u64) << position;
        }
        p
    }

    /// Dormant qubits where the virtual Pauli has an X or Y.
    fn dormant_x(&self, (_, x, _): &Virtual) -> Vec<usize> {
        set_bits(x)
            .filter(|&v| self.position(v).is_none())
            .collect()
    }

    /// Gather the dormant X parts of the Pauli pulled back from `factors`
    /// onto one dormant qubit with CX gates prepended to the frame, and
    /// return that qubit with the new pulled-back Pauli.
    fn gather_dormant(&mut self, factors: &[(usize, char)]) -> (Option<usize>, Virtual) {
        let pulled = self.frame.pull_back(factors);
        let dormant = self.dormant_x(&pulled);
        let Some((&first, rest)) = dormant.split_first() else {
            return (None, pulled);
        };
        // CX(first, d) maps X_first X_d to X_first; with `first` in |0> it
        // leaves the state alone.
        for &d in rest {
            self.prepend(Clifford::CX(first, d));
        }
        (Some(first), self.frame.pull_back(factors))
    }

    /// `exp(-i θ/2 P)` for a physical Pauli.
    pub fn rotate(&mut self, factors: &[(usize, char)], theta: f64) {
        let (dormant, pulled) = self.gather_dormant(factors);
        self.rotate_gathered(dormant, &pulled, theta);
    }

    /// The rotation `exp(-i θ/2 P')` for the virtual Pauli `pulled`, after
    /// [`NearClifford::gather_dormant`] left its dormant X parts on
    /// `dormant` (if any), which becomes active.
    fn rotate_gathered(&mut self, dormant: Option<usize>, pulled: &Virtual, theta: f64) {
        self.cached = None;
        if let Some(d) = dormant {
            self.active.push(d);
            self.peak = self.peak.max(self.active.len());
            if self.symbolic {
                return;
            }
            self.dense_mut().grow();
        }
        let p = self.on_positions(pulled);
        if !p.is_identity() && !self.symbolic {
            self.dense_mut().rotate(p, theta);
        }
        debug_assert!(self.symbolic || self.dense.qubits() == self.active.len());
    }

    /// Measure qubit `q` in the Z basis.
    pub fn measure(&mut self, q: usize, rng: &mut impl Rng) -> bool {
        let outcome = match self.probability_one(q) {
            p if p == 0.0 || p == 1.0 => p == 1.0,
            0.5 => rng.random::<bool>(),
            p => rng.random::<f64>() < p,
        };
        self.collapse(q, outcome);
        outcome
    }

    /// The probability that measuring qubit `q` gives 1: exactly 0 or 1 when
    /// determined, exactly 1/2 when the measured Pauli has an X part on a
    /// dormant qubit. May rewrite the frame without changing the state.
    pub fn probability_one(&mut self, q: usize) -> f64 {
        let (dormant, pulled) = self.gather_dormant(&[(q, 'Z')]);
        if dormant.is_some() {
            return 0.5;
        }
        let p = self.on_positions(&pulled);
        if p.is_identity() {
            return if p.negative { 1.0 } else { 0.0 };
        }
        if self.symbolic {
            return 0.5;
        }
        let one = ((1.0 - self.dense.expectation(p)) / 2.0).clamp(0.0, 1.0);
        self.cached = Some((q, one));
        one
    }

    /// Collapse the state as if measuring qubit `q` gave `outcome`, which
    /// must have nonzero probability.
    pub fn collapse(&mut self, q: usize, outcome: bool) {
        let factors = [(q, 'Z')];
        let (dormant, pulled) = self.gather_dormant(&factors);
        if let Some(d) = dormant {
            self.cached = None;
            return self.collapse_dormant(&factors, d, pulled, outcome);
        }
        let p = self.on_positions(&pulled);
        if p.is_identity() {
            debug_assert_eq!(p.negative, outcome, "collapsing onto an impossible outcome");
            return;
        }
        let eigenvalue = if outcome { -1.0 } else { 1.0 };
        let probability = (!self.symbolic).then(|| {
            let one = match self.cached.take() {
                Some((cached, one)) if cached == q => one,
                _ => ((1.0 - self.dense.expectation(p)) / 2.0).clamp(0.0, 1.0),
            };
            let probability = if outcome { one } else { 1.0 - one };
            assert!(probability > 0.0, "collapsing onto an impossible outcome");
            probability
        });
        self.retire(p, eigenvalue, probability);
    }

    /// [`NearClifford::probability_one`] for a measurement that acts on the
    /// dense part, from its pulled-back Pauli (given by
    /// [`NearClifford::gather_dormant`] with no dormant X part, the frame
    /// unchanged since): the same value, without pulling the Pauli back
    /// again. Each pull-back gathers a row of the column-major frame, one
    /// bit per qubit, and a measurement in a group of shots did up to five.
    fn dense_probability_one(&self, pulled: &Virtual) -> f64 {
        debug_assert!(!self.symbolic);
        let p = self.on_positions(pulled);
        ((1.0 - self.dense.expectation(p)) / 2.0).clamp(0.0, 1.0)
    }

    /// [`NearClifford::collapse`] onto `outcome` for a measurement that acts
    /// on the dense part, from its pulled-back Pauli (as in
    /// [`NearClifford::dense_probability_one`]) and its probability `one`
    /// of outcome 1.
    fn collapse_dense(&mut self, pulled: &Virtual, outcome: bool, one: f64) {
        let p = self.on_positions(pulled);
        debug_assert!(!p.is_identity() && !self.symbolic);
        let probability = if outcome { one } else { 1.0 - one };
        assert!(probability > 0.0, "collapsing onto an impossible outcome");
        self.cached = None;
        self.retire(p, if outcome { -1.0 } else { 1.0 }, Some(probability));
    }

    /// The dense part, copied first if another state shares it.
    fn dense_mut(&mut self) -> &mut Dense {
        Arc::make_mut(&mut self.dense)
    }

    /// The measured Pauli has an X or Y on dormant qubit `d` (and on no
    /// other dormant qubit): each outcome has probability 1/2, and the
    /// collapse is a Clifford on the frame.
    fn collapse_dormant(
        &mut self,
        factors: &[(usize, char)],
        d: usize,
        mut pulled: Virtual,
        outcome: bool,
    ) {
        if bit(&pulled.2, d) {
            // Y on d: S† Y S = X, and S fixes |0>.
            self.prepend(Clifford::S(d));
            pulled = self.frame.pull_back(factors);
        }
        // P' = ± P_A ⊗ X_d (⊗ Z on dormant qubits), so the collapsed state is
        // (φ ⊗ |0> + c P_A φ ⊗ |1>)/√2 with c = ±1: controlled-P_A (control
        // d) after H on d, and X before the H if c = -1.
        for v in self.active.clone() {
            match (bit(&pulled.1, v), bit(&pulled.2, v)) {
                (true, false) => self.prepend(Clifford::CX(d, v)),
                (false, true) => self.prepend(Clifford::CZ(d, v)),
                (true, true) => {
                    // CY = S CX S† on the target.
                    self.prepend(Clifford::S(v));
                    self.prepend(Clifford::CX(d, v));
                    self.prepend(Clifford::Sdg(v));
                }
                (false, false) => {}
            }
        }
        self.prepend(Clifford::H(d));
        if outcome ^ pulled.0 {
            self.prepend(Clifford::X(d));
        }
    }

    /// Project `φ` onto the `eigenvalue` eigenspace of `p` (of nonzero
    /// `probability`, `None` when symbolic), turn `p` into Z on one position
    /// with Cliffords `G` on the active qubits (`φ -> G† φ`, `C -> C G`, so
    /// the state is unchanged), and retire that qubit. The frame takes the
    /// Cliffords one by one; the dense part takes the projection, the
    /// Cliffords and the removal of the qubit in one pass
    /// ([`Dense::retired`]).
    fn retire(&mut self, p: Pauli, eigenvalue: f64, probability: Option<f64>) {
        let (gates, target, negative) = retire_plan(p);
        // φ is then an eigenstate of ±Z_target; X makes it +1.
        let one = (eigenvalue < 0.0) ^ negative;
        if let Some(probability) = probability {
            self.dense = Arc::new(
                self.dense
                    .retired(p, eigenvalue, probability, one, Vec::new()),
            );
        }
        self.retire_frame(gates, target, one);
    }

    /// The frame's part of [`NearClifford::retire`], once the dense part is
    /// the new one: the plan's `gates` and, when `one`, an X on `target`
    /// are prepended, and the qubit at `target` leaves the active ones.
    fn retire_frame(&mut self, gates: Vec<Clifford>, target: usize, one: bool) {
        for g in gates.into_iter().chain(one.then_some(Clifford::X(target))) {
            self.prepend(self.virtual_gate(g));
        }
        self.active.remove(target);
        debug_assert!(self.symbolic || self.dense.qubits() == self.active.len());
    }

    /// Both outcomes of a measurement that acts on the dense part, from its
    /// pulled-back Pauli and probability `one` of outcome 1 (as in
    /// [`NearClifford::collapse_dense`], both outcomes possible): this state
    /// becomes the one after outcome 0, and the one after outcome 1 is
    /// returned. The two new dense parts come from one pass over the
    /// current one ([`Dense::retired_both`]).
    fn split_dense(&mut self, pulled: &Virtual, one: f64) -> NearClifford {
        let p = self.on_positions(pulled);
        debug_assert!(!p.is_identity() && !self.symbolic);
        assert!(
            one > 0.0 && one < 1.0,
            "collapsing onto an impossible outcome"
        );
        let (gates, target, negative) = retire_plan(p);
        let (zero_dense, one_dense) = self.dense.retired_both(p, one, negative);
        self.cached = None;
        let mut other = self.clone();
        self.dense = Arc::new(zero_dense);
        other.dense = Arc::new(one_dense);
        self.retire_frame(gates.clone(), target, negative);
        other.retire_frame(gates, target, !negative);
        other
    }

    /// A Clifford on positions as the same Clifford on the virtual qubits at
    /// those positions.
    fn virtual_gate(&self, g: Clifford) -> Clifford {
        let virt = |p: usize| self.active[p];
        match g {
            Clifford::H(p) => Clifford::H(virt(p)),
            Clifford::S(p) => Clifford::S(virt(p)),
            Clifford::Sdg(p) => Clifford::Sdg(virt(p)),
            Clifford::X(p) => Clifford::X(virt(p)),
            Clifford::CX(c, t) => Clifford::CX(virt(c), virt(t)),
            Clifford::CZ(a, b) => Clifford::CZ(virt(a), virt(b)),
            other => unreachable!("{other:?} is not used on positions"),
        }
    }

    /// Reset qubit `q` to |0>.
    pub fn reset(&mut self, q: usize, rng: &mut impl Rng) {
        if self.measure(q, rng) {
            self.frame.apply(Clifford::X(q));
        }
    }

    /// `<ψ|P|ψ>` for a physical Pauli.
    pub fn expectation(&self, factors: &[(usize, char)]) -> f64 {
        let pulled = self.frame.pull_back(factors);
        if !self.dormant_x(&pulled).is_empty() {
            return 0.0;
        }
        let p = self.on_positions(&pulled);
        if p.is_identity() {
            return if p.negative { -1.0 } else { 1.0 };
        }
        self.dense.expectation(p)
    }
}

/// The Cliffords on positions that turn the Pauli `p` (not the identity)
/// into `±Z` on one position, that position, and whether the sign is then
/// negative. Depends only on `p`, so groups of shots that share the frame
/// share the plan.
fn retire_plan(mut p: Pauli) -> (Vec<Clifford>, usize, bool) {
    // Room for both stages in the usual case (one gate per position the
    // Pauli touches in each, plus S and H), so the list is allocated once
    // rather than grown a few times per measurement.
    let mut gates = Vec::with_capacity(2 * (p.x | p.z).count_ones() as usize + 2);
    let target = if p.x != 0 {
        let j = p.x.trailing_zeros() as usize;
        if p.z >> j & 1 == 1 {
            gates.push(Clifford::S(j));
        }
        for i in set_positions(p.x & !(1 << j)) {
            gates.push(Clifford::CX(j, i));
        }
        j
    } else {
        let j = p.z.trailing_zeros() as usize;
        for i in set_positions(p.z & !(1 << j)) {
            gates.push(Clifford::CX(i, j));
        }
        j
    };
    for &g in &gates {
        conjugate(&mut p, g);
    }
    if p.x != 0 {
        // Clear the Z parts left next to the X on `target`.
        let first = gates.len();
        for i in set_positions(p.z & !(1 << target)) {
            gates.push(Clifford::CZ(target, i));
        }
        if p.z >> target & 1 == 1 {
            gates.push(Clifford::S(target));
        }
        gates.push(Clifford::H(target));
        for &g in &gates[first..] {
            conjugate(&mut p, g);
        }
    }
    debug_assert_eq!(
        (p.x, p.z),
        (0, 1 << target),
        "the measured Pauli should be Z on one position"
    );
    (gates, target, p.negative)
}

/// The set bits of `mask`, lowest first, one `trailing_zeros` each rather
/// than a test of all 64 positions.
fn set_positions(mut mask: u64) -> impl Iterator<Item = usize> {
    std::iter::from_fn(move || {
        let i = mask.trailing_zeros() as usize;
        mask &= mask.wrapping_sub(1);
        (i < 64).then_some(i)
    })
}

/// `P -> G† P G` on positions.
fn conjugate(p: &mut Pauli, g: Clifford) {
    let get = |m: u64, i: usize| m >> i & 1 == 1;
    let flip = |m: &mut u64, i: usize, on: bool| *m ^= (on as u64) << i;
    match g {
        Clifford::H(j) => {
            let (x, z) = (get(p.x, j), get(p.z, j));
            p.negative ^= x && z;
            flip(&mut p.x, j, x != z);
            flip(&mut p.z, j, x != z);
        }
        Clifford::S(j) => {
            let (x, z) = (get(p.x, j), get(p.z, j));
            p.negative ^= x && !z;
            flip(&mut p.z, j, x);
        }
        Clifford::CX(c, t) => {
            let (xc, zc, xt, zt) = (get(p.x, c), get(p.z, c), get(p.x, t), get(p.z, t));
            p.negative ^= xc && zt && (xt == zc);
            flip(&mut p.x, t, xc);
            flip(&mut p.z, c, zt);
        }
        Clifford::CZ(a, b) => {
            let (xa, za, xb, zb) = (get(p.x, a), get(p.z, a), get(p.x, b), get(p.z, b));
            p.negative ^= xa && xb && (za != zb);
            flip(&mut p.z, a, xb);
            flip(&mut p.z, b, xa);
        }
        other => unreachable!("{other:?} is not used on positions"),
    }
}

/// Draws from `Binomial(n, p)`, as `n` Bernoulli trials.
fn binomial(rng: &mut impl Rng, n: usize, p: f64) -> usize {
    (0..n).filter(|_| rng.random::<f64>() < p).count()
}

/// The end of one branch of a measurement tree: a classical record, how many
/// shots ended with it, and the peak number of active qubits on the way.
type Leaf = (Vec<bool>, usize, usize);

/// What a Z measurement does to a state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// The outcome is fixed.
    Fixed,
    /// 50/50, changing only the frame.
    Frame,
    /// Acts on the dense part, which then loses a qubit.
    Dense,
}

impl NearClifford {
    fn kind(&self, q: usize) -> Kind {
        let pulled = self.frame.pull_back(&[(q, 'Z')]);
        if !self.dormant_x(&pulled).is_empty() {
            Kind::Frame
        } else if self.on_positions(&pulled).is_identity() {
            Kind::Fixed
        } else {
            Kind::Dense
        }
    }

    /// Which of the `pending` measurements (`(qubit, clbit)`) to do next,
    /// and its kind: the first that acts on the dense part (each halves it
    /// for everything after it), else the first with a fixed outcome (no
    /// branching), else the first. The scan stops at the first dense one:
    /// each [`NearClifford::kind`] pulls a Pauli back through the frame,
    /// and classifying every pending measurement before each one made the
    /// final measurements quadratic in their number (a third of the
    /// noiseless 100-qubit example's sampling time at 10,000 shots, where
    /// nearly every shot ends in a group of its own).
    ///
    /// `classifier` is any [`Classifier`] for this many qubits: it is set
    /// to this state's active qubits here, so one per measurement tree
    /// branch serves every call.
    fn next_measurement(
        &self,
        pending: &[(usize, usize)],
        classifier: &mut Classifier,
    ) -> (usize, Kind) {
        classifier.set_active(self);
        let mut first: Option<(usize, Kind)> = None;
        let mut fixed = None;
        for (i, &(q, _)) in pending.iter().enumerate() {
            let kind = classifier.kind(self, q);
            match kind {
                Kind::Dense => return (i, kind),
                Kind::Fixed if fixed.is_none() => fixed = Some((i, kind)),
                _ => {}
            }
            first.get_or_insert((i, kind));
        }
        fixed.or(first).expect("a pending measurement")
    }
}

/// Classifies Z measurements of one state as [`NearClifford::kind`] does,
/// from the bits of the pulled-back Pauli alone, with the active qubits as
/// a mask and buffers reused between measurements. `kind` (a pull-back with
/// its sign and four allocations, and a search of the active list per set
/// bit) took about 0.9 µs a call on the 100-qubit example; with this, the
/// scan for the next measurement took 87 ms instead of 278 at 10,000
/// shots on one thread (temporary timers, 2026-11-18). One classifier
/// serves a whole branch of [`sample_frames`] (made once, its mask reset
/// per call) rather than three allocations per measurement.
struct Classifier {
    /// Bit `v` set when virtual qubit `v` is active.
    active: Vec<u64>,
    x: Vec<u64>,
    z: Vec<u64>,
}

impl Classifier {
    /// A classifier for states of `n` qubits.
    fn new(n: usize) -> Classifier {
        let words = n.div_ceil(64);
        Classifier {
            active: vec![0; words],
            x: vec![0; words],
            z: vec![0; words],
        }
    }

    /// Take `state`'s active qubits as the mask.
    fn set_active(&mut self, state: &NearClifford) {
        self.active.fill(0);
        for &v in &state.active {
            self.active[v / 64] |= 1 << (v % 64);
        }
    }

    /// The kind of a Z measurement of qubit `q` in `state` (the state the
    /// classifier was made for): 50/50 if `C† Z_q C` has an X or Y on a
    /// dormant qubit, else dense if it acts on an active qubit, else fixed.
    fn kind(&mut self, state: &NearClifford, q: usize) -> Kind {
        state.frame.pull_back_z_bits(q, &mut self.x, &mut self.z);
        let mut dense = false;
        for ((&x, &z), &a) in self.x.iter().zip(&self.z).zip(&self.active) {
            if x & !a != 0 {
                return Kind::Frame;
            }
            dense |= (x | z) & a != 0;
        }
        if dense { Kind::Dense } else { Kind::Fixed }
    }
}

/// Run `shots` shots of `pending` measurements (`(qubit, clbit)`, distinct
/// qubits, so in any order) and then `tail` from `state`, as one
/// measurement tree. Each measurement is done once per branch: its outcome
/// probability splits the branch's shots binomially, and each outcome with
/// shots continues as its own branch (sharing the dense part until one of
/// them changes it). Measurements that act on the dense part go first, while
/// the shots are still together: each halves the dense part for everything
/// after it. Branches with many shots run in parallel; each draws its own
/// seed, so the result depends only on `seed`.
fn branch(
    mut state: NearClifford,
    mut pending: Vec<(usize, usize)>,
    mut tail: &[Instruction],
    shots: usize,
    seed: u64,
    mut clbits: Vec<bool>,
) -> Vec<Leaf> {
    let mut rng = Pcg64::seed_from_u64(seed);
    loop {
        if !pending.is_empty()
            && tail
                .iter()
                .all(|i| matches!(i, Instruction::Barrier { .. }))
        {
            // The final measurements: Pauli frames instead of branches.
            let group = Group::new(state.n, shots, &clbits);
            return sample_frames(state, pending, group, rng.random());
        }
        // The next measurement or reset: (qubit, clbit or None for reset).
        let (qubit, clbit) = if !pending.is_empty() {
            // Dense measurements first (each halves the dense part), then
            // fixed ones (no branching), then 50/50 ones.
            let index = if state.active_qubits() > 0 {
                state
                    .next_measurement(&pending, &mut Classifier::new(state.n))
                    .0
            } else {
                0
            };
            let (q, c) = pending.swap_remove(index);
            (q, Some(c))
        } else {
            let Some((instruction, rest)) = tail.split_first() else {
                break;
            };
            tail = rest;
            match instruction {
                Instruction::Gate { gate, qubits } => {
                    state.apply(gate, qubits).expect("checked by run");
                    continue;
                }
                Instruction::Barrier { .. } => continue,
                Instruction::Noise { .. } | Instruction::Feedback { .. } => {
                    unreachable!("rejected by run")
                }
                Instruction::Reset { qubit } => (*qubit, None),
                Instruction::Measure { qubit, clbit } => {
                    // Gather the run of measurements on distinct qubits.
                    pending.push((*qubit, *clbit));
                    while let Some((Instruction::Measure { qubit, clbit }, rest)) =
                        tail.split_first()
                    {
                        if pending.iter().any(|&(q, _)| q == *qubit) {
                            break;
                        }
                        pending.push((*qubit, *clbit));
                        tail = rest;
                    }
                    continue;
                }
            }
        };
        let finish = |state: &mut NearClifford, clbits: &mut Vec<bool>, outcome: bool| {
            state.collapse(qubit, outcome);
            match clbit {
                Some(c) => clbits[c] = outcome,
                None if outcome => state.frame.apply(Clifford::X(qubit)),
                None => {}
            }
        };
        let p = state.probability_one(qubit);
        let ones = match p {
            0.0 => 0,
            1.0 => shots,
            p => binomial(&mut rng, shots, p),
        };
        if ones == 0 || ones == shots {
            finish(&mut state, &mut clbits, ones == shots);
            continue;
        }
        let (seed0, seed1) = (rng.random::<u64>(), rng.random::<u64>());
        let (mut other, mut other_clbits) = (state.clone(), clbits.clone());
        finish(&mut state, &mut clbits, false);
        finish(&mut other, &mut other_clbits, true);
        let other_pending = pending.clone();
        let (mut zeros, ones_leaves) = if shots >= 64 {
            rayon::join(
                || branch(state, pending, tail, shots - ones, seed0, clbits),
                || branch(other, other_pending, tail, ones, seed1, other_clbits),
            )
        } else {
            (
                branch(state, pending, tail, shots - ones, seed0, clbits),
                branch(other, other_pending, tail, ones, seed1, other_clbits),
            )
        };
        zeros.extend(ones_leaves);
        return zeros;
    }
    let peak = state.peak_active_qubits();
    vec![(clbits, shots, peak)]
}

/// A group of shots that share the frame `C` and the dense part `φ`, each
/// with its own Pauli frame `P_s` on the virtual qubits (shot `s` is in the
/// state `C P_s (φ ⊗ |0...0>)`) and its own classical bits, 64 shots a
/// word. Signs of `P_s` are dropped, as they only change a shot's global
/// phase.
///
/// Shots leave a group when it splits: at a measurement of the dense part
/// with both outcomes drawn, or at a rotation that anticommutes with some
/// frames but not all. The leaving side is copied out. After a
/// measurement, the staying side is compacted too, so that noiseless
/// circuits do exactly the work (and draw exactly the random numbers) of
/// the sampler before noise was added. After a rotation, where errors
/// usually split off a few shots from many, it keeps its layout and only
/// clears the leavers in `alive`, so the split costs time in proportion to
/// those few; it is compacted once a quarter of its slots are dead.
#[derive(Clone, Debug)]
struct Group {
    words: usize,
    /// The shots still in the group.
    alive: Vec<u64>,
    x: Vec<u64>,
    z: Vec<u64>,
    /// One column of shots per classical bit, all in one allocation
    /// (column `c` at [`Group::column`]`(c)`): a split copies a group's
    /// records with one allocation rather than one per classical bit.
    records: Vec<u64>,
}

impl Group {
    /// `shots` (at least one) error-free shots with classical bits `clbits`.
    fn new(qubits: usize, shots: usize, clbits: &[bool]) -> Group {
        let words = shots.div_ceil(64);
        let alive = valid(shots);
        let mut records = Vec::with_capacity(clbits.len() * words);
        for &b in clbits {
            records.extend(alive.iter().map(|&a| if b { a } else { 0 }));
        }
        Group {
            words,
            alive,
            x: vec![0; qubits * words],
            z: vec![0; qubits * words],
            records,
        }
    }

    fn qubits(&self) -> usize {
        self.x.len() / self.words
    }

    fn count(&self) -> usize {
        self.alive.iter().map(|w| w.count_ones() as usize).sum()
    }

    fn column(&self, q: usize) -> std::ops::Range<usize> {
        q * self.words..(q + 1) * self.words
    }

    fn xor(table: &mut [u64], words: usize, dst: usize, src: usize) {
        for w in 0..words {
            table[dst * words + w] ^= table[src * words + w];
        }
    }

    /// `P_s -> V† P_s V` after `V` was prepended to the shared frame (`C P_s
    /// = (C V)(V† P_s V)`).
    fn conjugate(&mut self, op: Clifford) {
        let w = self.words;
        match op {
            Clifford::H(q) => {
                let r = self.column(q);
                self.x[r.clone()].swap_with_slice(&mut self.z[r]);
            }
            Clifford::S(q) | Clifford::Sdg(q) => {
                for i in self.column(q) {
                    self.z[i] ^= self.x[i];
                }
            }
            Clifford::X(_) | Clifford::Y(_) | Clifford::Z(_) => {}
            Clifford::CX(c, t) => {
                Self::xor(&mut self.x, w, t, c);
                Self::xor(&mut self.z, w, c, t);
            }
            Clifford::CZ(a, b) => {
                for i in 0..w {
                    let (xa, xb) = (self.x[a * w + i], self.x[b * w + i]);
                    self.z[a * w + i] ^= xb;
                    self.z[b * w + i] ^= xa;
                }
            }
            Clifford::Swap(a, b) => {
                for i in 0..w {
                    self.x.swap(a * w + i, b * w + i);
                    self.z.swap(a * w + i, b * w + i);
                }
            }
        }
    }

    /// Per live shot, whether `P_s` anticommutes with the virtual Pauli
    /// `(x, z)`.
    fn anticommutes(&self, x: &[u64], z: &[u64]) -> Vec<u64> {
        let mut out = Vec::new();
        self.anticommutes_into(x, z, &mut out);
        out
    }

    /// [`Group::anticommutes`] written into `out` (resized to the group's
    /// words), so a caller can reuse one buffer.
    fn anticommutes_into(&self, x: &[u64], z: &[u64], out: &mut Vec<u64>) {
        out.clear();
        out.resize(self.words, 0);
        for v in set_bits(x) {
            for (o, &f) in out.iter_mut().zip(&self.z[self.column(v)]) {
                *o ^= f;
            }
        }
        for v in set_bits(z) {
            for (o, &f) in out.iter_mut().zip(&self.x[self.column(v)]) {
                *o ^= f;
            }
        }
        for (o, a) in out.iter_mut().zip(&self.alive) {
            *o &= a;
        }
    }

    /// Multiply the frame of shot `s` alone by the virtual Pauli `(x, z)`.
    fn multiply_shot(&mut self, s: usize, x: &[u64], z: &[u64]) {
        let (w, bit) = (s / 64, 1u64 << (s % 64));
        for v in set_bits(x) {
            self.x[v * self.words + w] ^= bit;
        }
        for v in set_bits(z) {
            self.z[v * self.words + w] ^= bit;
        }
    }

    /// A compact copy of the shots in `mask` (which must be alive), in
    /// order.
    fn extract(&self, mask: &[u64]) -> Group {
        let plan = CompressPlan::new(mask);
        let words = plan.count.div_ceil(64);
        // Columns that are zero over the group stay zero, with no gather:
        // without noise, frames only pick up bits at 50/50 measurements,
        // after the dense part is gone, and record columns at their own
        // measurement, so most columns are zero while the measurement tree
        // splits groups.
        let gather = |from: &[u64], to: &mut [u64]| {
            for (column, out) in from
                .chunks_exact(self.words)
                .zip(to.chunks_exact_mut(words))
            {
                if column.iter().all(|&w| w == 0) {
                    continue;
                }
                plan.gather(column, out);
            }
        };
        let mut x = vec![0; self.qubits() * words];
        let mut z = vec![0; self.qubits() * words];
        let mut records = vec![0; self.records.len() / self.words * words];
        gather(&self.x, &mut x);
        gather(&self.z, &mut z);
        gather(&self.records, &mut records);
        Group {
            words,
            alive: valid(plan.count),
            x,
            z,
            records,
        }
    }

    /// One leaf per live shot: its classical bits.
    fn leaves(&self, peak: usize) -> Vec<Leaf> {
        set_bits(&self.alive)
            .map(|s| {
                let clbits = self
                    .records
                    .chunks_exact(self.words)
                    .map(|column| column[s / 64] >> (s % 64) & 1 == 1)
                    .collect();
                (clbits, 1, peak)
            })
            .collect()
    }

    /// Move the shots in `mask` (alive, and not all of them) into a group
    /// of their own and return it. With `compact`, the shots that stay are
    /// compacted as well; otherwise only once a quarter of the slots are
    /// dead.
    fn split_off(&mut self, mask: &[u64], compact: bool) -> Group {
        let other = self.extract(mask);
        for (a, m) in self.alive.iter_mut().zip(mask) {
            *a &= !m;
        }
        if compact || (self.words > 1 && self.count() * 4 < self.words * 64 * 3) {
            *self = self.extract(&self.alive.clone());
        }
        other
    }
}

/// How to gather the bits of one mask out of any column of a group's
/// words, a word at a time: [`Group::extract`] used to move one bit per
/// chosen shot per non-zero column (0.62e9 of 7.7e9 instructions of the
/// noiseless 100-qubit example at 20,000 shots, nearly all at the large
/// splits near the root of the measurement tree). Each mask word is
/// planned once and its plan reused over all ~300 columns: a full word is
/// copied, a sparse one (at most [`CompressPlan::SPARSE`] bits) moves its
/// bits one by one, and a denser one is compressed with the masks of
/// Hacker's Delight's `compress` (Warren, 2nd ed., §7-4), six shift-and-mask
/// steps whatever its bit count. BMI2's `pext` does the same in one
/// instruction, but is microcoded (~250 cycles) on AMD Zen 1 and 2, which
/// run the x86-64-v3 wheels.
struct CompressPlan {
    words: Vec<PlannedWord>,
    /// Bits in the whole mask: the size of the gathered column.
    count: usize,
}

struct PlannedWord {
    /// Index of the word in the column.
    word: usize,
    mask: u64,
    /// Bit position of this word's first chosen bit in the output.
    offset: usize,
    /// The six move masks of `compress`, for dense words.
    moves: Option<[u64; 6]>,
}

impl CompressPlan {
    /// Mask words with at most this many bits move them one at a time
    /// (about four operations per bit, against about 24 for `compress`).
    const SPARSE: u32 = 6;

    fn new(mask: &[u64]) -> CompressPlan {
        let mut words = Vec::new();
        let mut count = 0;
        for (word, &m) in mask.iter().enumerate() {
            if m == 0 {
                continue;
            }
            let ones = m.count_ones();
            let moves = (ones > Self::SPARSE && m != u64::MAX).then(|| compress_moves(m));
            words.push(PlannedWord {
                word,
                mask: m,
                offset: count,
                moves,
            });
            count += ones as usize;
        }
        CompressPlan { words, count }
    }

    /// OR the chosen bits of `column` into `out` (zero, `count` bits),
    /// in order.
    fn gather(&self, column: &[u64], out: &mut [u64]) {
        for w in &self.words {
            let v = column[w.word];
            let bits = match (w.mask, &w.moves) {
                (u64::MAX, _) => v,
                (m, Some(moves)) => compress(v, m, moves),
                (m, None) => {
                    let (mut bits, mut rest, mut i) = (0u64, m, 0);
                    while rest != 0 {
                        let s = rest.trailing_zeros();
                        bits |= (v >> s & 1) << i;
                        rest &= rest - 1;
                        i += 1;
                    }
                    bits
                }
            };
            let (o, shift) = (w.offset / 64, w.offset % 64);
            out[o] |= bits << shift;
            if shift > 0 && shift + w.mask.count_ones() as usize > 64 {
                out[o + 1] |= bits >> (64 - shift);
            }
        }
    }
}

/// The move masks of Hacker's Delight's `compress` for mask `m` (Warren,
/// *Hacker's Delight*, 2nd ed., Figure 7-10): step `i` moves the bits in
/// `moves[i]` right by `2^i`.
fn compress_moves(mut m: u64) -> [u64; 6] {
    let mut moves = [0u64; 6];
    // Bits with a zero of the mask to their right, to be counted.
    let mut mk = !m << 1;
    for (i, mv) in moves.iter_mut().enumerate() {
        // Parallel suffix: bit j of `mp` is the parity of the bits of `mk`
        // at or below j, the parity of bit i of each mask bit's distance.
        let mut mp = mk ^ (mk << 1);
        mp ^= mp << 2;
        mp ^= mp << 4;
        mp ^= mp << 8;
        mp ^= mp << 16;
        mp ^= mp << 32;
        let v = mp & m;
        *mv = v;
        m = (m ^ v) | (v >> (1 << i));
        mk &= !mp;
    }
    moves
}

/// The bits of `x` under `m`, packed into the low bits in order, with
/// `moves` from [`compress_moves`]`(m)`.
#[inline]
fn compress(x: u64, m: u64, moves: &[u64; 6]) -> u64 {
    let mut x = x & m;
    for (i, &mv) in moves.iter().enumerate() {
        let t = x & mv;
        x = (x ^ t) | (t >> (1 << i));
    }
    x
}

/// Bits over `shots`: word-wise mask of the valid shots.
fn valid(shots: usize) -> Vec<u64> {
    (0..shots.div_ceil(64))
        .map(|w| {
            if (w + 1) * 64 <= shots {
                u64::MAX
            } else {
                (1u64 << (shots % 64)) - 1
            }
        })
        .collect()
}

/// A group split off by [`measure_shots`]: its state, its shots, the seed
/// of its random stream, and the seed of a new stream for the group it left.
type Split = (NearClifford, Group, u64, u64);

/// Conjugate the group's frames by the Cliffords prepended to the shared
/// frame since the last call.
fn sync(state: &mut NearClifford, group: &mut Group) {
    for op in state.log.as_mut().expect("logging").drain(..) {
        group.conjugate(op);
    }
}

/// Measure qubit `q`, of kind `kind` in `state`, in every shot of the
/// group, writing the outcomes to record column `c`. Measurements with a fixed outcome, or a 50/50 one
/// that only changes the frame, are done once for all shots: per shot the
/// outcome flips where its frame anticommutes with the measured Pauli, and
/// a random 50/50 outcome goes into its frame. A measurement on the dense
/// part draws each shot's outcome in the shared state (its recorded
/// outcome is that, flipped where its frame anticommutes); if both
/// outcomes occur, the group splits: the shots of outcome 1 are returned as
/// a group of their own with their collapsed state, with seeds for their
/// random stream and for that of the shots of outcome 0, which stay.
fn measure_shots(
    state: &mut NearClifford,
    group: &mut Group,
    q: usize,
    c: usize,
    kind: Kind,
    rng: &mut Pcg64,
    scratch: &mut Scratch,
) -> Option<Split> {
    let (dormant, pulled) = state.gather_dormant(&[(q, 'Z')]);
    sync(state, group);
    let Scratch {
        flips, effective, ..
    } = scratch;
    group.anticommutes_into(&pulled.1, &pulled.2, flips);
    match kind {
        Kind::Fixed => {
            let fixed = if state.on_positions(&pulled).negative {
                u64::MAX
            } else {
                0
            };
            let column = group.column(c);
            for ((r, f), a) in group.records[column]
                .iter_mut()
                .zip(flips.iter())
                .zip(&group.alive)
            {
                *r = (f ^ fixed) & a;
            }
            None
        }
        Kind::Frame => {
            let d = dormant.expect("a dormant X part");
            // Collapse the shared state onto effective outcome 0; a shot
            // with effective outcome 1 carries X_d in its frame.
            state.collapse(q, false);
            sync(state, group);
            effective.clear();
            effective.extend(group.alive.iter().map(|a| rng.random::<u64>() & a));
            let column = group.column(d);
            for (x, e) in group.x[column].iter_mut().zip(effective.iter()) {
                *x ^= e;
            }
            let column = group.column(c);
            for ((r, e), f) in group.records[column]
                .iter_mut()
                .zip(effective.iter())
                .zip(flips.iter())
            {
                *r = e ^ f;
            }
            None
        }
        Kind::Dense => {
            // No dormant X part, so `gather_dormant` left the frame alone
            // and `pulled` stays valid through both collapses.
            let one = state.dense_probability_one(&pulled);
            effective.clear();
            effective.extend(group.alive.iter().map(|a| {
                (0..64).fold(0u64, |acc, b| {
                    acc | ((rng.random::<f64>() < one) as u64) << b
                }) & a
            }));
            let column = group.column(c);
            for ((r, e), f) in group.records[column]
                .iter_mut()
                .zip(effective.iter())
                .zip(flips.iter())
            {
                *r = e ^ f;
            }
            let ones: usize = effective.iter().map(|w| w.count_ones() as usize).sum();
            let count = group.count();
            if ones == 0 || ones == count {
                state.collapse_dense(&pulled, ones == count, one);
                sync(state, group);
                return None;
            }
            let mut other_group = group.split_off(effective, true);
            let mut other = state.split_dense(&pulled, one);
            sync(state, group);
            sync(&mut other, &mut other_group);
            let (seed0, seed1) = (rng.random::<u64>(), rng.random::<u64>());
            Some((other, other_group, seed1, seed0))
        }
    }
}

/// Buffers that one branch of [`sample_frames`] reuses for all its
/// measurements: with nearly every shot in a group of its own at 10,000
/// shots, the three classifier buffers and two word buffers allocated per
/// measurement were most of the sampler's `malloc` and `free` calls.
struct Scratch {
    classifier: Classifier,
    /// Per live shot, whether its frame flips the measured outcome.
    flips: Vec<u64>,
    /// Per live shot, the outcome drawn in the shared state.
    effective: Vec<u64>,
}

impl Scratch {
    fn new(n: usize) -> Scratch {
        Scratch {
            classifier: Classifier::new(n),
            flips: Vec::new(),
            effective: Vec::new(),
        }
    }
}

/// Sample the final measurements `pending` (on distinct qubits, so in any
/// order) of the shots of `group` from `state`, and return one leaf per
/// shot. Measurements on the dense part go first, while the shots are still
/// together (each halves the dense part for everything after it), then
/// fixed ones, then 50/50 ones. The two sides of a split run depth-first
/// under `rayon::join`, each with a seed drawn from its parent's stream, so
/// the result depends only on `seed`.
fn sample_frames(
    mut state: NearClifford,
    mut pending: Vec<(usize, usize)>,
    mut group: Group,
    seed: u64,
) -> Vec<Leaf> {
    let mut rng = Pcg64::seed_from_u64(seed);
    // Frames start without errors; from here on every Clifford prepended
    // to the shared frame must reach them.
    state.log.get_or_insert_with(Vec::new);
    let mut scratch = Scratch::new(state.n);
    while !pending.is_empty() {
        let (index, kind) = state.next_measurement(&pending, &mut scratch.classifier);
        let (q, c) = pending.swap_remove(index);
        if let Some((other, other_group, seed, own_seed)) =
            measure_shots(&mut state, &mut group, q, c, kind, &mut rng, &mut scratch)
        {
            let other_pending = pending.clone();
            let (mut leaves, more) = rayon::join(
                || sample_frames(state, pending, group, own_seed),
                || sample_frames(other, other_pending, other_group, seed),
            );
            leaves.extend(more);
            return leaves;
        }
    }
    group.leaves(state.peak_active_qubits())
}

/// Run `circuit` for `shots` shots and count the classical outcomes. The
/// gates before the first measurement or reset (or noise channel) run once.
/// Without noise, the rest runs as a measurement tree, so each measurement
/// is done once per distinct history rather than once per shot. With noise,
/// it runs on groups of shots with their own Pauli frames, which split
/// where errors make their dense states differ (see the module
/// documentation).
pub fn run(circuit: &Circuit, shots: usize, seed: Option<u64>) -> Result<RunResult, String> {
    let mut rng = match seed {
        Some(seed) => Pcg64::seed_from_u64(seed),
        None => Pcg64::from_rng(&mut rand::rng()),
    };
    if circuit.has_feedback() {
        return Err(crate::simulator::FEEDBACK_BACKENDS.into());
    }
    if !is_supported(circuit) {
        return Err("the circuit has gates the near-Clifford backend does not support".into());
    }
    let start = Instant::now();
    let instructions = &circuit.instructions;
    let first = instructions
        .iter()
        .position(|i| {
            matches!(
                i,
                Instruction::Measure { .. } | Instruction::Reset { .. } | Instruction::Noise { .. }
            )
        })
        .unwrap_or(instructions.len());
    let mut base = NearClifford::new(circuit.num_qubits);
    for instruction in &instructions[..first] {
        if let Instruction::Gate { gate, qubits } = instruction {
            base.apply(gate, qubits)?;
        }
    }
    let mut result = RunResult::default();
    result.stats.gates = circuit.gate_count();
    result.stats.gate_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let leaves = if shots == 0 {
        Vec::new()
    } else if circuit.has_noise() {
        noisy::sample(
            &base,
            &instructions[first..],
            shots,
            circuit.num_clbits,
            rng.random(),
        )?
    } else {
        branch(
            base,
            Vec::new(),
            &instructions[first..],
            shots,
            rng.random(),
            vec![false; circuit.num_clbits],
        )
    };
    let mut counts = std::collections::BTreeMap::new();
    let mut peak = 0;
    for (clbits, count, leaf_peak) in leaves {
        *counts.entry(bitstring(&clbits)).or_default() += count;
        peak = peak.max(leaf_peak);
    }
    result.counts = counts;
    result.stats.measure_seconds = start.elapsed().as_secs_f64();
    result.stats.peak_active_qubits = Some(peak);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The chosen bits of `column`, one at a time, as `extract` gathered
    /// them before [`CompressPlan`].
    fn gather_bitwise(column: &[u64], mask: &[u64]) -> Vec<u64> {
        let chosen: Vec<usize> = set_bits(mask).collect();
        let mut out = vec![0u64; chosen.len().div_ceil(64)];
        for (i, &s) in chosen.iter().enumerate() {
            out[i / 64] |= (column[s / 64] >> (s % 64) & 1) << (i % 64);
        }
        out
    }

    #[test]
    fn compress_matches_bit_by_bit_gather() {
        let mut rng = Pcg64::seed_from_u64(7);
        for trial in 0..2000 {
            let words = 1 + trial % 5;
            // Masks from empty to full, sparse to dense, including whole
            // words of zeros and ones.
            let mask: Vec<u64> = (0..words)
                .map(|_| match rng.random_range(0..6) {
                    0 => 0,
                    1 => u64::MAX,
                    2 => 1u64 << rng.random_range(0..64),
                    3 => rng.random::<u64>() & rng.random::<u64>() & rng.random::<u64>(),
                    4 => rng.random::<u64>() | rng.random::<u64>(),
                    _ => rng.random::<u64>(),
                })
                .collect();
            let column: Vec<u64> = (0..words).map(|_| rng.random()).collect();
            let plan = CompressPlan::new(&mask);
            let expected = gather_bitwise(&column, &mask);
            assert_eq!(plan.count, set_bits(&mask).count());
            let mut out = vec![0u64; plan.count.div_ceil(64)];
            plan.gather(&column, &mut out);
            assert_eq!(out, expected, "mask {mask:x?}");
            for &m in &mask {
                let x = rng.random::<u64>();
                assert_eq!(
                    compress(x, m, &compress_moves(m)),
                    gather_bitwise(&[x], &[m]).first().copied().unwrap_or(0)
                );
            }
        }
    }

    #[test]
    fn extract_keeps_each_chosen_shot_in_order() {
        let mut rng = Pcg64::seed_from_u64(11);
        for shots in [1, 5, 64, 65, 200, 300] {
            let mut group = Group::new(7, shots, &[true, false, true]);
            for v in group
                .x
                .iter_mut()
                .chain(group.z.iter_mut())
                .chain(group.records.iter_mut())
            {
                *v = rng.random();
            }
            // Columns of zeros must stay zero.
            let w = group.words;
            group.x[2 * w..3 * w].fill(0);
            let mut mask: Vec<u64> = group
                .alive
                .iter()
                .map(|&a| rng.random::<u64>() & a)
                .collect();
            // A split always moves at least one shot.
            mask[0] |= 1;
            let other = group.extract(&mask);
            let count = set_bits(&mask).count();
            assert_eq!(other.count(), count);
            for (table, from) in [(&other.x, &group.x), (&other.z, &group.z)]
                .into_iter()
                .chain([(&other.records, &group.records)])
            {
                for (column, out) in from.chunks_exact(w).zip(table.chunks_exact(other.words)) {
                    assert_eq!(out, gather_bitwise(column, &mask).as_slice());
                }
            }
        }
    }
}
