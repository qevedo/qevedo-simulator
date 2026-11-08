//! Noisy circuits on the near-Clifford backend: groups of shots that share
//! one Clifford frame.
//!
//! Each shot is in the state `C P_s (φ_g ⊗ |0...0>)`: the frame `C` and the
//! active qubits are the same for every shot, the dense part `φ_g` belongs
//! to the shot's group `g`, and the Pauli frame `P_s` (errors, and
//! outcomes) to the shot alone (see the parent module). This is Clifft's
//! division of the state (Chase & Labib, arXiv:2604.27058): a Clifford
//! frame computed once, a Pauli frame per shot, and dense states only for
//! what differs.
//!
//! Nothing that happens to `C` depends on a shot: Clifford gates, the CX
//! gates that gather a rotation's dormant X parts, and the Cliffords that
//! collapse or retire a measured qubit are the same whatever the outcomes
//! and errors (an outcome-1 X goes into the shots' frames instead of `C`).
//! So [`compile`] walks the circuit once with a symbolic state and writes a
//! program for the groups, of steps that touch only `P_s` and `φ_g`:
//!
//! - the Cliffords prepended to `C`, which conjugate every `P_s`;
//! - a rotation `exp(-iθ/2 P')` on `φ_g`, which is `exp(+iθ/2 P')` in the
//!   shots whose frame anticommutes with `P'`: a group with shots of both
//!   kinds splits in two there;
//! - a measurement: fixed, 50/50 in the frame, or on the dense part, where
//!   each shot's outcome is drawn from its group's `φ_g` and a group with
//!   both outcomes splits;
//! - a noise channel, with the images `C† X_q C` and `C† Z_q C` of its
//!   qubits' Paulis, which its errors multiply into `P_s`.
//!
//! [`run`] then takes one group at a time through the whole program, depth
//! first: when it splits, one part goes on and the other waits on a stack.
//! An earlier version moved all groups through each operation in lockstep;
//! once errors had split most shots off, every operation then streamed all
//! groups' dense parts through memory (112 MB per batch at 14 active
//! qubits), at 4-10 ns per amplitude. One group's dense part (256 KB at 14
//! active qubits) stays in the cache from one step to the next, as in
//! Clifft's per-shot execution.
//!
//! Shots run in batches whose number depends only on the shot count, each
//! batch with its own random stream, in parallel. The counts depend on the
//! seed, not on the number of threads.
//!
//! Everything here is cold and out of line: placed among the dense kernels,
//! this code moved those kernels' machine code, and noiseless sampling of a
//! 23-active-qubit circuit ran 5-9% slower with the same instruction counts.

use std::sync::Arc;

use rand::{RngExt, SeedableRng};
use rand_pcg::Pcg64;
use rayon::prelude::*;

use super::dense::{Dense, Pauli};
use super::{Group, Kind, Leaf, NearClifford, Step, decompose, retire_plan};
use crate::circuit::{Instruction, Noise};
use crate::stabilizer::Clifford;

/// A virtual Pauli without its sign: x and z bit words over the virtual
/// qubits. A shot's frame anticommutes with it where [`Group::anticommutes`]
/// says so.
type Check = (Vec<u64>, Vec<u64>);

/// A step of the program that every group of shots runs (see the module
/// documentation).
#[derive(Debug)]
enum Op<'a> {
    /// Conjugate every shot's frame by these Cliffords, which were
    /// prepended to `C` in this order.
    Conjugate(Vec<Clifford>),
    /// `exp(-iθ/2 p)` on the dense part, which first grows by a qubit in
    /// `|0>` if `grow`; `-θ` in the shots whose frame anticommutes with
    /// `check`.
    Rotate {
        grow: bool,
        p: Pauli,
        theta: f64,
        check: Check,
    },
    /// A measurement whose outcome is `fixed`, flipped in the shots whose
    /// frame anticommutes with `check`.
    MeasureFixed {
        clbit: usize,
        check: Check,
        fixed: bool,
    },
    /// A 50/50 measurement that collapses `C` (with the Cliffords
    /// `collapse`) onto effective outcome 0; shots with effective outcome 1
    /// carry `X_d` in their frame.
    MeasureFrame {
        clbit: usize,
        check: Check,
        collapse: Vec<Clifford>,
        d: usize,
    },
    /// A measurement of `p` on the dense part, then the Cliffords `retire`
    /// that make it Z on one position, whose qubit (`x_retired` on the
    /// virtual qubits) is removed; it reads 1 where the effective outcome
    /// differs from `negative`, which an X in the shots' frames fixes.
    MeasureDense {
        clbit: usize,
        check: Check,
        p: Pauli,
        retire: Vec<Clifford>,
        negative: bool,
        x_retired: Vec<u64>,
    },
    /// A scratch record column for the outcome of a reset.
    Scratch,
    /// The end of a reset: the shots whose scratch column reads 1 multiply
    /// their frame by `C† X_q C`.
    ResetFix(Check),
    /// Noise channels of one probability `p`, with nothing between them.
    Noise { p: f64, sites: Vec<Site<'a>> },
}

/// Where a noise channel can strike: one target of the channel (one qubit,
/// or a group of [`Noise::arity`] qubits), with `C† X_q C` and `C† Z_q C`
/// for each of its qubits.
#[derive(Debug)]
struct Site<'a> {
    channel: &'a Noise,
    images: Vec<(Check, Check)>,
}

/// The program for `instructions` (the rest of a circuit with noise
/// channels) from `base`, and the peak number of active qubits.
fn compile<'a>(
    base: &NearClifford,
    instructions: &'a [Instruction],
) -> Result<(Vec<Op<'a>>, usize), String> {
    let mut frame = base.clone();
    frame.dense = Arc::new(Dense::new());
    frame.symbolic = true;
    frame.log = Some(Vec::new());
    let mut ops = Vec::new();
    for instruction in instructions {
        match instruction {
            Instruction::Gate { gate, qubits } => {
                let steps = decompose(gate, qubits).ok_or_else(|| {
                    format!("{gate:?} is not supported by the near-Clifford backend")
                })?;
                for step in steps {
                    match step {
                        Step::Clifford(g) => frame.frame.apply(g),
                        Step::Rotation(factors, theta) => {
                            let (dormant, pulled) = frame.gather_dormant(&factors);
                            conjugate(&mut frame, &mut ops);
                            if let Some(d) = dormant {
                                frame.active.push(d);
                                frame.peak = frame.peak.max(frame.active.len());
                            }
                            let p = frame.on_positions(&pulled);
                            // An identity is a global phase in every shot.
                            if !p.is_identity() || dormant.is_some() {
                                ops.push(Op::Rotate {
                                    grow: dormant.is_some(),
                                    p,
                                    theta,
                                    check: (pulled.1, pulled.2),
                                });
                            }
                        }
                    }
                }
            }
            &Instruction::Measure { qubit, clbit } => measure(&mut frame, &mut ops, qubit, clbit),
            &Instruction::Reset { qubit } => {
                ops.push(Op::Scratch);
                measure(&mut frame, &mut ops, qubit, usize::MAX);
                let (_, x, z) = frame.frame.pull_back(&[(qubit, 'X')]);
                ops.push(Op::ResetFix((x, z)));
            }
            Instruction::Noise { channel, qubits } => {
                let p = channel.probability();
                if p <= 0.0 {
                    continue;
                }
                let image = |q: usize| {
                    let (_, xx, xz) = frame.frame.pull_back(&[(q, 'X')]);
                    let (_, zx, zz) = frame.frame.pull_back(&[(q, 'Z')]);
                    ((xx, xz), (zx, zz))
                };
                let new = qubits.chunks(channel.arity()).map(|target| Site {
                    channel,
                    images: target.iter().map(|&q| image(q)).collect(),
                });
                // Consecutive channels of the same probability make one run
                // of sites, so a shot draws one gap per error rather than
                // one per channel (a circuit with noise after every gate has
                // thousands of them).
                match ops.last_mut() {
                    Some(Op::Noise { p: last, sites }) if *last == p => sites.extend(new),
                    _ => ops.push(Op::Noise {
                        p,
                        sites: new.collect(),
                    }),
                }
            }
            Instruction::Barrier { .. } => {}
        }
    }
    Ok((ops, frame.peak_active_qubits()))
}

/// Move the Cliffords prepended to the frame since the last call into the
/// program.
fn conjugate(frame: &mut NearClifford, ops: &mut Vec<Op>) {
    let log = std::mem::take(frame.log.as_mut().expect("logging"));
    if !log.is_empty() {
        ops.push(Op::Conjugate(log));
    }
}

/// Compile a measurement of qubit `q` into record column `clbit`
/// (`usize::MAX`: the newest scratch column).
fn measure(frame: &mut NearClifford, ops: &mut Vec<Op>, q: usize, clbit: usize) {
    let kind = frame.kind(q);
    let (dormant, pulled) = frame.gather_dormant(&[(q, 'Z')]);
    conjugate(frame, ops);
    let p = frame.on_positions(&pulled);
    let check = (pulled.1, pulled.2);
    match kind {
        Kind::Fixed => ops.push(Op::MeasureFixed {
            clbit,
            check,
            fixed: p.negative,
        }),
        Kind::Frame => {
            // Each shot's flips are against the measured Pauli as it was
            // pulled back, so they come before the collapse conjugates the
            // frames.
            frame.collapse(q, false);
            let collapse = std::mem::take(frame.log.as_mut().expect("logging"));
            ops.push(Op::MeasureFrame {
                clbit,
                check,
                collapse,
                d: dormant.expect("a dormant X part"),
            });
        }
        Kind::Dense => {
            let (gates, target, negative) = retire_plan(p);
            for g in gates {
                frame.prepend(frame.virtual_gate(g));
            }
            let retire = std::mem::take(frame.log.as_mut().expect("logging"));
            let retired = frame.active.remove(target);
            let mut x_retired = vec![0u64; frame.n.div_ceil(64)];
            x_retired[retired / 64] |= 1 << (retired % 64);
            ops.push(Op::MeasureDense {
                clbit,
                check,
                p,
                retire,
                negative,
                x_retired,
            });
        }
    }
}

/// Run `instructions` (the rest of a circuit with noise channels) for
/// `shots` shots from `base`, whose dense part is shared by all of them at
/// the start, and return one leaf per shot.
#[cold]
#[inline(never)]
pub(super) fn sample(
    base: &NearClifford,
    instructions: &[Instruction],
    shots: usize,
    clbits: usize,
    seed: u64,
) -> Result<Vec<Leaf>, String> {
    let (ops, peak) = compile(base, instructions)?;
    // At most a sixteenth of the shots in larger runs, so that batches
    // spread over the threads, and at least 256.
    let batch = shots.div_ceil(16).next_multiple_of(64).max(256);
    let batches: Vec<(usize, u64)> = {
        let mut rng = Pcg64::seed_from_u64(seed);
        (0..shots.div_ceil(batch))
            .map(|i| (batch.min(shots - i * batch), rng.random()))
            .collect()
    };
    let leaves = batches
        .into_par_iter()
        .map(|(shots, seed)| {
            let group = Group::new(base.n, shots, &vec![false; clbits]);
            run(&ops, base.dense.clone(), group, peak, seed)
        })
        .flatten_iter()
        .collect();
    Ok(leaves)
}

/// A group waiting to go on from step `pc`, where it split off: with the
/// sign of its rotation, or the effective outcome of its measurement, at
/// `pc` already decided (and the step's work before the split done).
struct Task {
    pc: usize,
    dense: Arc<Dense>,
    group: Group,
    decided: Option<bool>,
}

/// Take the shots of `group`, in state `dense`, through the program, one
/// group at a time (depth first), and return one leaf per shot.
fn run(ops: &[Op], dense: Arc<Dense>, group: Group, peak: usize, seed: u64) -> Vec<Leaf> {
    let mut rng = Pcg64::seed_from_u64(seed);
    let mut leaves = Vec::new();
    // The allocation that the next dense measurement writes into: the one
    // its state leaves behind, when no other group shares it.
    let mut spare = Vec::new();
    let mut stack = vec![Task {
        pc: 0,
        dense,
        group,
        decided: None,
    }];
    while let Some(Task {
        mut pc,
        mut dense,
        mut group,
        mut decided,
    }) = stack.pop()
    {
        while pc < ops.len() {
            let mut split = |dense: &Arc<Dense>, group: Group, decided: bool| {
                stack.push(Task {
                    pc,
                    dense: dense.clone(),
                    group,
                    decided: Some(decided),
                })
            };
            match &ops[pc] {
                Op::Conjugate(cliffords) => {
                    for &op in cliffords {
                        group.conjugate(op);
                    }
                }
                Op::Rotate {
                    grow,
                    p,
                    theta,
                    check,
                } => {
                    // Whether this group's shots see -θ; a group whose shots
                    // disagree splits, the smaller side leaving (sharing the
                    // dense part until it rotates).
                    let flipped = decided.take().unwrap_or_else(|| {
                        if *grow {
                            Arc::make_mut(&mut dense).grow();
                        }
                        let flips = group.anticommutes(&check.0, &check.1);
                        let (count, n_flipped) = (group.count(), popcount(&flips));
                        if n_flipped == 0 || n_flipped == count {
                            return n_flipped != 0;
                        }
                        let leaving = 2 * n_flipped <= count;
                        let mask = if leaving {
                            flips
                        } else {
                            and_not(&group.alive, &flips)
                        };
                        split(&dense, group.split_off(&mask, false), leaving);
                        !leaving
                    });
                    if !p.is_identity() {
                        let sign = if flipped { -1.0 } else { 1.0 };
                        Arc::make_mut(&mut dense).rotate(*p, sign * theta);
                    }
                }
                &Op::MeasureFixed {
                    clbit,
                    ref check,
                    fixed,
                } => {
                    let fixed = if fixed { u64::MAX } else { 0 };
                    let flips = group.anticommutes(&check.0, &check.1);
                    *record(&mut group, clbit) = flips
                        .iter()
                        .zip(&group.alive)
                        .map(|(f, a)| (f ^ fixed) & a)
                        .collect();
                }
                Op::MeasureFrame {
                    clbit,
                    check,
                    collapse,
                    d,
                } => {
                    let flips = group.anticommutes(&check.0, &check.1);
                    for &op in collapse {
                        group.conjugate(op);
                    }
                    let effective: Vec<u64> = group
                        .alive
                        .iter()
                        .map(|a| rng.random::<u64>() & a)
                        .collect();
                    let column = group.column(*d);
                    for (x, e) in group.x[column].iter_mut().zip(&effective) {
                        *x ^= e;
                    }
                    *record(&mut group, *clbit) =
                        effective.iter().zip(&flips).map(|(e, f)| e ^ f).collect();
                }
                Op::MeasureDense {
                    clbit,
                    check,
                    p,
                    retire,
                    negative,
                    x_retired,
                } => {
                    // P(effective outcome 1) in this group's dense part.
                    let one = ((1.0 - dense.expectation(*p)) / 2.0).clamp(0.0, 1.0);
                    let outcome = decided.take().unwrap_or_else(|| {
                        // Draw each shot's effective outcome; a group with
                        // both splits, the shots of outcome 1 leaving.
                        let flips = group.anticommutes(&check.0, &check.1);
                        let mut effective = vec![0u64; group.words];
                        for s in super::set_bits(&group.alive) {
                            effective[s / 64] |= ((rng.random::<f64>() < one) as u64) << (s % 64);
                        }
                        *record(&mut group, *clbit) =
                            effective.iter().zip(&flips).map(|(e, f)| e ^ f).collect();
                        let (count, n_ones) = (group.count(), popcount(&effective));
                        if n_ones == 0 || n_ones == count {
                            return n_ones != 0;
                        }
                        split(&dense, group.split_off(&effective, true), true);
                        false
                    });
                    for &op in retire {
                        group.conjugate(op);
                    }
                    let (eigenvalue, probability) = if outcome {
                        (-1.0, one)
                    } else {
                        (1.0, 1.0 - one)
                    };
                    // φ is then an eigenstate of ±Z on the retired position.
                    // For -1, X there in φ and in every shot's frame (rather
                    // than in `C`) leaves the state alone and the qubit in
                    // |0>.
                    let flip = outcome ^ negative;
                    let buffer = std::mem::take(&mut spare);
                    let new = Arc::new(dense.retired(*p, eigenvalue, probability, flip, buffer));
                    if let Ok(old) = Arc::try_unwrap(std::mem::replace(&mut dense, new)) {
                        spare = old.amps;
                    }
                    if flip {
                        let alive = group.alive.clone();
                        group.multiply(&alive, x_retired, &[]);
                    }
                }
                Op::Scratch => group.records.push(vec![0; group.words]),
                Op::ResetFix((x, z)) => {
                    let ones = group.records.pop().expect("the scratch column");
                    group.multiply(&ones, x, z);
                }
                Op::Noise { p, sites } => noise(&mut group, &mut rng, *p, sites),
            }
            pc += 1;
        }
        leaves.extend(group.leaves(peak));
    }
    leaves
}

/// The record column `clbit`, or the newest scratch column for
/// `usize::MAX`.
fn record(group: &mut Group, clbit: usize) -> &mut Vec<u64> {
    if clbit == usize::MAX {
        group.records.last_mut().expect("a scratch column")
    } else {
        &mut group.records[clbit]
    }
}

/// A run of noise channels of probability `p` on every live shot of
/// `group`. As in the stabilizer sampler, the gap to the next error is
/// drawn (geometrically distributed), here over the trials of every site of
/// the run and every shot slot (site-major), rather than a draw per shot
/// and site. An error `E` multiplies the shot's frame by `C† E C`, from the
/// images of `X_q` and `Z_q`.
fn noise(group: &mut Group, rng: &mut Pcg64, p: f64, sites: &[Site]) {
    // ln(1 - p); -inf when p = 1, where every trial is an error.
    let log_miss = (-p).ln_1p();
    let slots = group.words * 64;
    let trials = sites.len() * slots;
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
        let (site, s) = (&sites[next / slots], next % slots);
        next += 1;
        if group.alive[s / 64] >> (s % 64) & 1 == 0 {
            continue;
        }
        let k = if matches!(site.channel, Noise::Depolarize2(_)) {
            rng.random_range(1..16)
        } else {
            0
        };
        let error = site.channel.error(rng.random::<f64>() * p, k);
        for (&(ex, ez), (x, z)) in error.iter().zip(&site.images) {
            if ex {
                group.multiply_shot(s, &x.0, &x.1);
            }
            if ez {
                group.multiply_shot(s, &z.0, &z.1);
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
