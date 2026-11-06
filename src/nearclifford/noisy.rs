//! Noisy circuits on the near-Clifford backend: groups of shots that share
//! one Clifford frame.
//!
//! Each shot is in the state `C P_s (φ_g ⊗ |0...0>)`: the frame `C` and the
//! active qubits are the same for every shot of a batch, the dense part
//! `φ_g` belongs to the shot's group `g`, and the Pauli frame `P_s` (errors,
//! and outcomes) to the shot alone (see the parent module). Every operation
//! is applied to all groups of a batch in lockstep:
//!
//! - Clifford gates, the CX gates that gather a rotation's dormant X parts,
//!   and the Cliffords that retire a measured qubit change `C` once for the
//!   whole batch, and conjugate every shot's `P_s`.
//! - A rotation acts on each `φ_g`, as `exp(+iθ/2 P')` in the shots whose
//!   frame anticommutes with its Pauli `P'`; a group with shots of both
//!   kinds splits in two there.
//! - A measurement of the dense part draws each shot's outcome from its
//!   group's `φ_g`; a group with both outcomes splits. Retiring the measured
//!   qubit would prepend an X to `C` for outcome 1 only; that X goes into the
//!   frames of those shots instead, so `C` stays shared.
//!
//! This is Clifft's division of the state (Chase & Labib, arXiv:2604.27058):
//! a Clifford frame computed once, a Pauli frame per shot, and dense states
//! only for what differs. The Clifford work is done once per batch rather
//! than once per group: an earlier version, where every group carried its
//! own copy of `C`, was 3-16x slower than Clifft on noisy random Clifford+T
//! circuits, because errors split most shots into groups of their own.
//!
//! Shots run in batches whose number depends only on the shot count and
//! the peak number of active qubits (bounding the memory of all groups'
//! dense parts), each batch with its own random stream, in parallel. The
//! counts depend on the seed, not on the number of threads.
//!
//! Everything here is cold and out of line: placed among the dense kernels,
//! this code moved those kernels' machine code, and noiseless sampling of a
//! 23-active-qubit circuit ran 5-9% slower with the same instruction counts.

use std::sync::Arc;

use rand::{RngExt, SeedableRng};
use rand_pcg::Pcg64;
use rayon::prelude::*;

use super::dense::{Dense, Gate as DenseGate};
use super::{Group, Kind, Leaf, NearClifford, Step, Virtual, decompose, retire_plan};
use crate::circuit::{Instruction, Noise};
use crate::stabilizer::Clifford;

/// A circuit lowered for [`sample`]: the gates as Clifford steps and
/// rotations, the other instructions as they are (barriers dropped).
#[derive(Clone, Debug)]
enum Op<'a> {
    Step(Step),
    Measure { qubit: usize, clbit: usize },
    Reset(usize),
    Noise(&'a Noise, &'a [usize]),
}

fn lower(instructions: &[Instruction]) -> Result<Vec<Op<'_>>, String> {
    let mut ops = Vec::new();
    for instruction in instructions {
        match instruction {
            Instruction::Gate { gate, qubits } => ops.extend(
                decompose(gate, qubits)
                    .ok_or_else(|| {
                        format!("{gate:?} is not supported by the near-Clifford backend")
                    })?
                    .into_iter()
                    .map(Op::Step),
            ),
            &Instruction::Measure { qubit, clbit } => ops.push(Op::Measure { qubit, clbit }),
            &Instruction::Reset { qubit } => ops.push(Op::Reset(qubit)),
            Instruction::Noise { channel, qubits } => ops.push(Op::Noise(channel, qubits)),
            Instruction::Barrier { .. } => {}
        }
    }
    Ok(ops)
}

/// The memory all groups of a batch may use for their dense parts, in
/// amplitudes (2 GiB).
const BATCH_AMPLITUDES: usize = 1 << 27;

/// Run `instructions` (the rest of a circuit with noise channels) for
/// `shots` shots from `base`, whose dense part is shared by all of them at
/// the start, and return one leaf per shot. `peak` is the circuit's peak
/// number of active qubits.
#[cold]
#[inline(never)]
pub(super) fn sample(
    base: &NearClifford,
    instructions: &[Instruction],
    shots: usize,
    clbits: usize,
    peak: usize,
    seed: u64,
) -> Result<Vec<Leaf>, String> {
    let ops = lower(instructions)?;
    // A batch of at most this many shots keeps every shot's own dense part
    // (the worst case: errors split every shot off) within the budget. At
    // least 64 shots (one word), and at most a sixteenth of the shots in
    // larger runs, so that batches spread over the threads.
    let fit = (BATCH_AMPLITUDES >> peak.min(27)).max(64);
    let batch = fit.min(shots.div_ceil(16).next_multiple_of(64).max(256));
    let batches: Vec<(usize, u64)> = {
        let mut rng = Pcg64::seed_from_u64(seed);
        (0..shots.div_ceil(batch))
            .map(|i| (batch.min(shots - i * batch), rng.random()))
            .collect()
    };
    let leaves = batches
        .into_par_iter()
        .map(|(shots, seed)| Batch::new(base, shots, clbits, seed).run(&ops))
        .flatten_iter()
        .collect();
    Ok(leaves)
}

/// The groups of a batch of shots, all on one frame.
struct Batch {
    /// The shared frame `C` and the active qubits. Its own dense part is
    /// never used (it is symbolic).
    frame: NearClifford,
    groups: Vec<(Arc<Dense>, Group)>,
    rng: Pcg64,
}

/// Run `f` on every group, in parallel when there are many.
fn each_group(groups: &mut [(Arc<Dense>, Group)], f: impl Fn(&mut Arc<Dense>, &mut Group) + Sync) {
    groups
        .par_iter_mut()
        .with_min_len(8)
        .for_each(|(dense, group)| f(dense, group));
}

impl Batch {
    fn new(base: &NearClifford, shots: usize, clbits: usize, seed: u64) -> Batch {
        let mut frame = base.clone();
        let dense = std::mem::replace(&mut frame.dense, Arc::new(Dense::new()));
        frame.symbolic = true;
        frame.log = Some(Vec::new());
        let group = Group::new(frame.n, shots, &vec![false; clbits]);
        Batch {
            frame,
            groups: vec![(dense, group)],
            rng: Pcg64::seed_from_u64(seed),
        }
    }

    fn run(mut self, ops: &[Op]) -> Vec<Leaf> {
        for op in ops {
            match op {
                Op::Step(Step::Clifford(g)) => self.frame.frame.apply(*g),
                Op::Step(Step::Rotation(factors, theta)) => self.rotate(factors, *theta),
                &Op::Measure { qubit, clbit } => self.measure(qubit, clbit),
                &Op::Reset(q) => self.reset(q),
                Op::Noise(channel, targets) => self.noise(channel, targets),
            }
        }
        let peak = self.frame.peak_active_qubits();
        self.groups
            .iter()
            .flat_map(|(_, group)| group.leaves(peak))
            .collect()
    }

    /// Conjugate every shot's frame by the Cliffords prepended to `C` since
    /// the last call.
    fn sync(&mut self) {
        let ops: Vec<Clifford> = self
            .frame
            .log
            .as_mut()
            .expect("logging")
            .drain(..)
            .collect();
        if !ops.is_empty() {
            each_group(&mut self.groups, |_, group| {
                for &op in &ops {
                    group.conjugate(op);
                }
            });
        }
    }

    /// `exp(-iθ/2 P)` for a physical Pauli, in every shot.
    fn rotate(&mut self, factors: &[(usize, char)], theta: f64) {
        let (dormant, pulled) = self.frame.gather_dormant(factors);
        self.sync();
        if let Some(d) = dormant {
            self.frame.active.push(d);
            self.frame.peak = self.frame.peak.max(self.frame.active.len());
            each_group(&mut self.groups, |dense, _| Arc::make_mut(dense).grow());
        }
        let p = self.frame.on_positions(&pulled);
        if p.is_identity() {
            // A global phase in every shot, whatever its frame.
            return;
        }
        // Split the groups whose shots disagree on the sign of the angle;
        // the smaller side leaves, sharing the dense part until it rotates.
        let mut signs = Vec::with_capacity(self.groups.len());
        let mut leaving = Vec::new();
        for (dense, group) in &mut self.groups {
            let flipped = group.anticommutes(&pulled.1, &pulled.2);
            let (count, n_flipped) = (group.count(), popcount(&flipped));
            if n_flipped == 0 || n_flipped == count {
                signs.push(if n_flipped == 0 { 1.0 } else { -1.0 });
                continue;
            }
            let (mask, sign) = if 2 * n_flipped <= count {
                (flipped, -1.0)
            } else {
                (and_not(&group.alive, &flipped), 1.0)
            };
            leaving.push(((dense.clone(), group.split_off(&mask, false)), sign));
            signs.push(-sign);
        }
        for (group, sign) in leaving {
            self.groups.push(group);
            signs.push(sign);
        }
        self.groups
            .par_iter_mut()
            .with_min_len(8)
            .zip(signs)
            .for_each(|((dense, _), sign)| Arc::make_mut(dense).rotate(p, sign * theta));
    }

    /// Measure qubit `q` in every shot into record column `c`.
    fn measure(&mut self, q: usize, c: usize) {
        let kind = self.frame.kind(q);
        let (dormant, pulled) = self.frame.gather_dormant(&[(q, 'Z')]);
        self.sync();
        match kind {
            Kind::Fixed => {
                let fixed = if self.frame.on_positions(&pulled).negative {
                    u64::MAX
                } else {
                    0
                };
                each_group(&mut self.groups, |_, group| {
                    let flips = group.anticommutes(&pulled.1, &pulled.2);
                    group.records[c] = flips
                        .iter()
                        .zip(&group.alive)
                        .map(|(f, a)| (f ^ fixed) & a)
                        .collect();
                });
            }
            Kind::Frame => {
                // The shared frame collapses onto effective outcome 0; a
                // shot with effective outcome 1 carries X_d in its frame.
                // Each shot's flips are against the measured Pauli as it
                // was pulled back, so they come before the collapse
                // conjugates the frames.
                let d = dormant.expect("a dormant X part");
                let flips: Vec<Vec<u64>> = self
                    .groups
                    .iter()
                    .map(|(_, group)| group.anticommutes(&pulled.1, &pulled.2))
                    .collect();
                self.frame.collapse(q, false);
                self.sync();
                for ((_, group), flips) in self.groups.iter_mut().zip(flips) {
                    let effective: Vec<u64> = group
                        .alive
                        .iter()
                        .map(|a| self.rng.random::<u64>() & a)
                        .collect();
                    let column = group.column(d);
                    for (x, e) in group.x[column].iter_mut().zip(&effective) {
                        *x ^= e;
                    }
                    group.records[c] = effective.iter().zip(&flips).map(|(e, f)| e ^ f).collect();
                }
            }
            Kind::Dense => self.measure_dense(&pulled, c),
        }
    }

    /// A measurement of the dense part, whose measured Pauli pulls back to
    /// `pulled` (with no X part on a dormant qubit).
    fn measure_dense(&mut self, pulled: &Virtual, c: usize) {
        let p = self.frame.on_positions(pulled);
        // P(effective outcome 1) in each group's dense part.
        let ones: Vec<f64> = self
            .groups
            .par_iter()
            .with_min_len(8)
            .map(|(dense, _)| ((1.0 - dense.expectation(p)) / 2.0).clamp(0.0, 1.0))
            .collect();
        // Draw each shot's effective outcome; a group with both splits, the
        // shots of outcome 1 leaving.
        let mut outcomes = Vec::with_capacity(self.groups.len());
        let mut leaving = Vec::new();
        for ((dense, group), &one) in self.groups.iter_mut().zip(&ones) {
            let flips = group.anticommutes(&pulled.1, &pulled.2);
            let effective: Vec<u64> = group
                .alive
                .iter()
                .map(|a| {
                    (0..64).fold(0u64, |acc, b| {
                        acc | ((self.rng.random::<f64>() < one) as u64) << b
                    }) & a
                })
                .collect();
            group.records[c] = effective.iter().zip(&flips).map(|(e, f)| e ^ f).collect();
            let (count, n_ones) = (group.count(), popcount(&effective));
            if n_ones == 0 || n_ones == count {
                outcomes.push((n_ones == count, one));
                continue;
            }
            leaving.push(((dense.clone(), group.split_off(&effective, true)), one));
            outcomes.push((false, one));
        }
        for (group, one) in leaving {
            self.groups.push(group);
            outcomes.push((true, one));
        }
        // Collapse, then turn the measured Pauli into Z on one position with
        // Cliffords applied to every dense part and prepended to `C` once.
        let (gates, target, negative) = retire_plan(p);
        let pairs: Vec<(DenseGate, Clifford)> =
            gates.iter().map(|&g| self.frame.virtual_pair(g)).collect();
        for &(_, virtual_gate) in &pairs {
            self.frame.prepend(virtual_gate);
        }
        self.sync();
        // X on the retired virtual qubit, as x bits over the virtual qubits.
        let retired = self.frame.active[target];
        let mut x_retired = vec![0u64; self.frame.n.div_ceil(64)];
        x_retired[retired / 64] |= 1 << (retired % 64);
        self.groups
            .par_iter_mut()
            .with_min_len(8)
            .zip(outcomes)
            .for_each(|((dense, group), (outcome, one))| {
                let dense = Arc::make_mut(dense);
                let (eigenvalue, probability) = if outcome {
                    (-1.0, one)
                } else {
                    (1.0, 1.0 - one)
                };
                dense.project(p, eigenvalue, probability);
                for &(dense_gate, _) in &pairs {
                    dense.apply(dense_gate);
                }
                // φ is now an eigenstate of ±Z_target. For -1, X on the
                // target in φ and in every shot's frame (rather than in `C`)
                // leaves the state alone and the target in |0>.
                if outcome ^ negative {
                    dense.apply(DenseGate::X(target));
                    let alive = group.alive.clone();
                    group.multiply(&alive, &x_retired, &[]);
                }
                dense.shrink(target);
            });
        self.frame.active.remove(target);
    }

    /// Reset qubit `q`: measure it into a scratch column, then multiply the
    /// frames of the shots that read 1 by `C† X_q C`.
    fn reset(&mut self, q: usize) {
        for (_, group) in &mut self.groups {
            group.records.push(vec![0; group.words]);
        }
        let column = self.groups[0].1.records.len() - 1;
        self.measure(q, column);
        let (_, x, z) = self.frame.frame.pull_back(&[(q, 'X')]);
        each_group(&mut self.groups, |_, group| {
            let ones = group.records.pop().expect("the scratch column");
            group.multiply(&ones, &x, &z);
        });
    }

    /// A noise channel on `targets` (qubits, or groups of [`Noise::arity`]
    /// qubits) in every live shot. As in the stabilizer sampler, the gap to
    /// the next error is drawn (geometrically distributed) rather than a
    /// draw per shot and target. An error `E` multiplies the shot's frame by
    /// `C† E C`: the images of `X_q` and `Z_q` are pulled back through the
    /// frame the first time an error on `q` needs them.
    fn noise(&mut self, channel: &Noise, targets: &[usize]) {
        let p = channel.probability();
        if p <= 0.0 {
            return;
        }
        let arity = channel.arity();
        // ln(1 - p); -inf when p = 1, where every trial is an error.
        let log_miss = (-p).ln_1p();
        let depolarize2 = matches!(channel, Noise::Depolarize2(_));
        let rng = &mut self.rng;
        let tableau = &self.frame.frame;
        for target in targets.chunks(arity) {
            let mut images: Vec<Option<(Virtual, Virtual)>> = vec![None; arity];
            for (_, group) in &mut self.groups {
                let trials = group.words * 64;
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
                    let s = next;
                    next += 1;
                    if group.alive[s / 64] >> (s % 64) & 1 == 0 {
                        continue;
                    }
                    let k = if depolarize2 {
                        rng.random_range(1..16)
                    } else {
                        0
                    };
                    let error = channel.error(rng.random::<f64>() * p, k);
                    for ((&q, &(ex, ez)), image) in target.iter().zip(&error).zip(&mut images) {
                        if !(ex || ez) {
                            continue;
                        }
                        let (x, z) = image.get_or_insert_with(|| {
                            (
                                tableau.pull_back(&[(q, 'X')]),
                                tableau.pull_back(&[(q, 'Z')]),
                            )
                        });
                        if ex {
                            group.multiply_shot(s, &x.1, &x.2);
                        }
                        if ez {
                            group.multiply_shot(s, &z.1, &z.2);
                        }
                    }
                }
            }
        }
    }
}

fn popcount(words: &[u64]) -> usize {
    words.iter().map(|w| w.count_ones() as usize).sum()
}

fn and_not(a: &[u64], b: &[u64]) -> Vec<u64> {
    a.iter().zip(b).map(|(a, b)| a & !b).collect()
}
