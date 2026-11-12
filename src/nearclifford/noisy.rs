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
//! The groups do not carry the frames `P_s` themselves. Each frame is
//! only ever asked whether it anticommutes with a rotation's or a
//! measurement's virtual Pauli (a check), and the answer is linear in the
//! Paulis that went into it, so [`link`] works out once which checks each
//! possible insertion flips, and a shot keeps one sign bit per check (its
//! register): an error XORs a row of bits into it, as Clifft keeps sign
//! bits of its expressions, and the conjugations leave the shots' work.
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

/// A physical Pauli without its sign, and where it acts: after how many of
/// the program's physical Clifford gates (see [`compile`]), and x and z bit
/// words over the physical qubits. [`rows`] works with these.
type Physical = (usize, Vec<u64>, Vec<u64>);

/// A step of the program, as compiled against the shots' Pauli frames (see
/// the module documentation); [`link`] turns it into [`Act`]s on sign
/// registers. The virtual Paulis (`check`, the Cliffords, `d`,
/// `x_retired`) are only read by the tests, whose reference carries frames
/// the old way; they cost one pull-back per check, which `compile` needs
/// anyway, unlike the virtual images of every error (`Site`).
#[derive(Debug)]
#[cfg_attr(not(test), allow(dead_code))]
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
        physical: Physical,
    },
    /// A measurement whose outcome is `fixed`, flipped in the shots whose
    /// frame anticommutes with `check`.
    MeasureFixed {
        clbit: usize,
        check: Check,
        physical: Physical,
        fixed: bool,
    },
    /// A 50/50 measurement that collapses `C` (with the Cliffords
    /// `collapse`) onto effective outcome 0; shots with effective outcome 1
    /// carry `X_d` in their frame (`x_d`: `C X_d C†` after the collapse).
    MeasureFrame {
        clbit: usize,
        check: Check,
        physical: Physical,
        collapse: Vec<Clifford>,
        d: usize,
        x_d: Physical,
    },
    /// A measurement of `p` on the dense part, then the Cliffords `retire`
    /// that make it Z on one position, whose qubit (`x_retired` on the
    /// virtual qubits) is removed; it reads 1 where the effective outcome
    /// differs from `negative`, which an X in the shots' frames fixes
    /// (`x_physical`: that X as `C X C†`).
    MeasureDense {
        clbit: usize,
        check: Check,
        physical: Physical,
        p: Pauli,
        retire: Vec<Clifford>,
        negative: bool,
        x_retired: Vec<u64>,
        x_physical: Physical,
    },
    /// A scratch record column for the outcome of a reset.
    Scratch,
    /// The end of a reset: the shots whose scratch column reads 1 multiply
    /// their frame by `C† X_q C` (physically `X_q`).
    ResetFix(Check, Physical),
    /// Noise channels of one probability `p`, with nothing between them.
    Noise { p: f64, sites: Vec<Site<'a>> },
}

/// Where a noise channel can strike: one target of the channel (one qubit,
/// or a group of [`Noise::arity`] qubits), after `at` physical Clifford
/// gates. Its errors are `X_q` and `Z_q` on those physical qubits, which is
/// all [`rows`] needs; the tests also keep their virtual images `C† X_q C`
/// and `C† Z_q C`, which are dense and cost a pull-back each.
#[derive(Debug)]
struct Site<'a> {
    channel: &'a Noise,
    qubits: &'a [usize],
    at: usize,
    #[cfg(test)]
    images: Vec<(Check, Check)>,
}

/// The program for `instructions` (the rest of a circuit with noise
/// channels) from `base`, the peak number of active qubits, and the
/// physical Clifford gates in order (which the program itself does not
/// need: they never touch a shot's frame).
fn compile<'a>(
    base: &NearClifford,
    instructions: &'a [Instruction],
) -> Result<(Vec<Op<'a>>, usize, Vec<Clifford>), String> {
    let mut frame = base.clone();
    frame.dense = Arc::new(Dense::new());
    frame.symbolic = true;
    frame.log = Some(Vec::new());
    let mut ops = Vec::new();
    let mut gates = Vec::new();
    let words = base.n.div_ceil(64);
    let physical = |factors: &[(usize, char)], at: usize| {
        let (mut x, mut z) = (vec![0u64; words], vec![0u64; words]);
        for &(q, c) in factors {
            if c != 'Z' {
                x[q / 64] |= 1 << (q % 64);
            }
            if c != 'X' {
                z[q / 64] |= 1 << (q % 64);
            }
        }
        (at, x, z)
    };
    for instruction in instructions {
        match instruction {
            Instruction::Gate { gate, qubits } => {
                let steps = decompose(gate, qubits).ok_or_else(|| {
                    format!("{gate:?} is not supported by the near-Clifford backend")
                })?;
                for step in steps {
                    match step {
                        Step::Clifford(g) => {
                            frame.frame.apply(g);
                            gates.push(g);
                        }
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
                                    physical: physical(&factors, gates.len()),
                                });
                            }
                        }
                    }
                }
            }
            &Instruction::Measure { qubit, clbit } => {
                let z = physical(&[(qubit, 'Z')], gates.len());
                measure(&mut frame, &mut ops, qubit, clbit, z);
            }
            &Instruction::Reset { qubit } => {
                ops.push(Op::Scratch);
                let z = physical(&[(qubit, 'Z')], gates.len());
                measure(&mut frame, &mut ops, qubit, usize::MAX, z);
                let (_, x, z) = frame.frame.pull_back(&[(qubit, 'X')]);
                let fix = physical(&[(qubit, 'X')], gates.len());
                ops.push(Op::ResetFix((x, z), fix));
            }
            Instruction::Noise { channel, qubits } => {
                let p = channel.probability();
                if p <= 0.0 {
                    continue;
                }
                #[cfg(test)]
                let image = |q: usize| {
                    let (_, xx, xz) = frame.frame.pull_back(&[(q, 'X')]);
                    let (_, zx, zz) = frame.frame.pull_back(&[(q, 'Z')]);
                    ((xx, xz), (zx, zz))
                };
                let at = gates.len();
                let new = qubits.chunks(channel.arity()).map(|target| Site {
                    channel,
                    qubits: target,
                    at,
                    #[cfg(test)]
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
    Ok((ops, frame.peak_active_qubits(), gates))
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
/// (`usize::MAX`: the newest scratch column); `physical` is `Z_q`.
fn measure(
    frame: &mut NearClifford,
    ops: &mut Vec<Op>,
    q: usize,
    clbit: usize,
    physical: Physical,
) {
    let at = physical.0;
    let kind = frame.kind(q);
    let (dormant, pulled) = frame.gather_dormant(&[(q, 'Z')]);
    conjugate(frame, ops);
    let p = frame.on_positions(&pulled);
    let check = (pulled.1, pulled.2);
    match kind {
        Kind::Fixed => ops.push(Op::MeasureFixed {
            clbit,
            check,
            physical,
            fixed: p.negative,
        }),
        Kind::Frame => {
            // Each shot's flips are against the measured Pauli as it was
            // pulled back, so they come before the collapse conjugates the
            // frames.
            frame.collapse(q, false);
            let collapse = std::mem::take(frame.log.as_mut().expect("logging"));
            let d = dormant.expect("a dormant X part");
            let (x, z) = frame.frame.push_forward_x(d);
            ops.push(Op::MeasureFrame {
                clbit,
                check,
                physical,
                collapse,
                d,
                x_d: (at, x, z),
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
            let (x, z) = frame.frame.push_forward_x(retired);
            ops.push(Op::MeasureDense {
                clbit,
                check,
                physical,
                p,
                retire,
                negative,
                x_retired,
                x_physical: (at, x, z),
            });
        }
    }
}

/// The words of a shot's sign register that an insertion flips: pairs of
/// a word index and the bits to flip in it, in increasing order of word.
type Row = Vec<(usize, u64)>;

/// A step of the program in terms of sign registers (see [`link`]): every
/// `check` is now the index of a bit in each shot's register, and every
/// Pauli that went into the shots' frames is now the row of register bits
/// it flips.
#[derive(Debug)]
enum Act<'a> {
    /// As [`Op::Rotate`]: `-θ` in the shots whose register bit `check` is
    /// set.
    Rotate {
        grow: bool,
        p: Pauli,
        theta: f64,
        check: usize,
    },
    /// As [`Op::MeasureFixed`].
    MeasureFixed {
        clbit: usize,
        check: usize,
        fixed: bool,
    },
    /// As [`Op::MeasureFrame`]: shots with effective outcome 1 flip `row`
    /// (the bits that `X_d` after the collapse flips).
    MeasureFrame {
        clbit: usize,
        check: usize,
        row: Row,
    },
    /// As [`Op::MeasureDense`]: where the retired qubit reads 1, every shot
    /// of the group flips `row` (the bits that the X on the retired qubit
    /// flips).
    MeasureDense {
        clbit: usize,
        check: usize,
        p: Pauli,
        negative: bool,
        row: Row,
    },
    /// As [`Op::Scratch`].
    Scratch,
    /// As [`Op::ResetFix`]: the shots whose scratch column reads 1 flip
    /// `row`.
    ResetFix(Row),
    /// As [`Op::Noise`], with the rows of `C† X_q C` and `C† Z_q C` for each
    /// qubit of each site.
    Noise { p: f64, sites: Vec<Hit<'a>> },
}

/// A noise site in terms of sign registers: for each of its qubits, the
/// rows that an X and a Z error there flip.
#[derive(Debug)]
struct Hit<'a> {
    channel: &'a Noise,
    rows: Vec<(Row, Row)>,
}

/// The program `ops` in terms of sign registers, and the number of checks
/// (bits per register).
///
/// A shot's Pauli frame matters only through whether it anticommutes with
/// each check: the virtual Pauli of a rotation, which picks the sign of
/// its angle, or of a measurement, which flips its outcome. And the frame
/// is a product of Paulis put in at known steps: the errors at noise sites,
/// the `X_d` of a 50/50 measurement with effective outcome 1, the X on a
/// retired qubit reading 1, the fix of a reset. Only whether each one is
/// put in depends on the shot. Everything that happens to a Pauli once it
/// is in the frame, the conjugations by the Cliffords prepended to `C`, is
/// the same in every shot and linear, and so is the symplectic product with
/// a check. So each insertion flips a fixed set of checks, its row, and a
/// shot's frame can be replaced by the XOR of the rows of what was put in:
/// one bit per check, its sign register. This is how Clifft (Chase & Labib,
/// arXiv:2604.27058) keeps a shot's frame as sign bits of its expressions.
/// Per shot the frame work is then a few words per error and one bit per
/// check: no conjugations, no products with the checks' supports.
///
/// The rows come from [`rows`].
fn link<'a>(ops: &[Op<'a>], gates: &[Clifford], n: usize) -> (Vec<Act<'a>>, usize) {
    // Number the checks and the insertions in program order; the rows,
    // which need every later check, replace the insertion numbers below.
    enum Pending<'a> {
        Act(Act<'a>),
        Frame(usize, usize, usize),
        Dense(usize, usize, Pauli, bool, usize),
        Reset(usize),
        Noise(f64, Vec<(&'a Noise, Vec<(usize, usize)>)>),
    }
    let (mut checks, mut insertions) = (0usize, 0usize);
    let count = |counter: &mut usize| {
        *counter += 1;
        *counter - 1
    };
    let mut pending = Vec::with_capacity(ops.len());
    for op in ops {
        pending.push(match op {
            Op::Conjugate(_) => continue,
            &Op::Rotate { grow, p, theta, .. } => Pending::Act(Act::Rotate {
                grow,
                p,
                theta,
                check: count(&mut checks),
            }),
            &Op::MeasureFixed { clbit, fixed, .. } => Pending::Act(Act::MeasureFixed {
                clbit,
                check: count(&mut checks),
                fixed,
            }),
            &Op::MeasureFrame { clbit, .. } => {
                let k = count(&mut checks);
                Pending::Frame(clbit, k, count(&mut insertions))
            }
            &Op::MeasureDense {
                clbit, p, negative, ..
            } => {
                let k = count(&mut checks);
                Pending::Dense(clbit, k, p, negative, count(&mut insertions))
            }
            Op::Scratch => Pending::Act(Act::Scratch),
            Op::ResetFix(..) => Pending::Reset(count(&mut insertions)),
            Op::Noise { p, sites } => Pending::Noise(
                *p,
                sites
                    .iter()
                    .map(|site| {
                        let pairs = site
                            .qubits
                            .iter()
                            .map(|_| {
                                let ix = count(&mut insertions);
                                (ix, count(&mut insertions))
                            })
                            .collect();
                        (site.channel, pairs)
                    })
                    .collect(),
            ),
        });
    }
    let mut rows = rows(ops, gates, n, checks, insertions);
    let mut take = |i: usize| std::mem::take(&mut rows[i]);
    let acts = pending
        .into_iter()
        .map(|p| match p {
            Pending::Act(act) => act,
            Pending::Frame(clbit, check, i) => Act::MeasureFrame {
                clbit,
                check,
                row: take(i),
            },
            Pending::Dense(clbit, check, p, negative, i) => Act::MeasureDense {
                clbit,
                check,
                p,
                negative,
                row: take(i),
            },
            Pending::Reset(i) => Act::ResetFix(take(i)),
            Pending::Noise(p, sites) => Act::Noise {
                p,
                sites: sites
                    .into_iter()
                    .map(|(channel, pairs)| Hit {
                        channel,
                        rows: pairs.into_iter().map(|(a, b)| (take(a), take(b))).collect(),
                    })
                    .collect(),
            },
        })
        .collect();
    (acts, checks)
}

/// The rows of the `insertions` of `ops` (numbered in program order, as
/// in [`link`]), over its `checks`; `gates` are the physical Clifford gates
/// of the circuit, in order.
///
/// Insertion `i` flips check `k` if `k` comes later and the two Paulis
/// anticommute once `i` is carried to `k`. That holds in either picture:
/// conjugating both by `C` turns the virtual frame Paulis into physical
/// ones, and in the physical picture a Pauli put in the frame changes only
/// with the physical Clifford gates (`G C P = (G C) P`), not with the
/// Cliffords prepended to `C`, which `C P C†` absorbs. Physically an error
/// is a one-qubit Pauli and a measurement's check is `Z_q`, while their
/// virtual images span most virtual qubits. And there are far fewer checks
/// than insertions (212 against 29,000 on the 200-qubit benchmark
/// circuit), so this carries the checks backwards rather than the
/// insertions forwards. It walks the program from the end with one slot
/// per check of a [`Group`]: slot `k` takes the check's physical Pauli
/// where the check is and is conjugated back through each earlier physical
/// gate (on the symplectic part, all that anticommutation sees, each gate
/// here is its own inverse); at an insertion, the slots' columns give its
/// row: for `X_q`, which slots have Z on `q`. Carrying the virtual images
/// forwards, one slot per insertion, gave the same rows but made one-shot
/// runs 1.04-1.56× slower than the frames.
fn rows(ops: &[Op], gates: &[Clifford], n: usize, checks: usize, insertions: usize) -> Vec<Row> {
    /// The walk's state: the checks' slots, conjugated back to just after
    /// the first `cursor` gates, the next check and insertion (counting
    /// down), and the rows so far.
    struct Back<'g> {
        gates: &'g [Clifford],
        slots: Group,
        cursor: usize,
        k: usize,
        i: usize,
        rows: Vec<Row>,
    }
    impl Back<'_> {
        fn rewind(&mut self, at: usize) {
            for &g in self.gates[at..self.cursor].iter().rev() {
                self.slots.conjugate(g);
            }
            self.cursor = at;
        }
        fn check(&mut self, (at, x, z): &Physical) {
            self.rewind(*at);
            self.k -= 1;
            self.slots.multiply_shot(self.k, x, z);
        }
        fn insertion(&mut self, (at, x, z): &Physical) {
            self.rewind(*at);
            self.i -= 1;
            self.rows[self.i] = sparse(&self.slots.anticommutes(x, z));
        }
    }
    fn sparse(bits: &[u64]) -> Row {
        bits.iter()
            .enumerate()
            .filter(|&(_, &b)| b != 0)
            .map(|(w, &b)| (w, b))
            .collect()
    }
    let mut back = Back {
        gates,
        slots: Group::new(n, checks.max(1), &[]),
        cursor: gates.len(),
        k: checks,
        i: insertions,
        rows: vec![Vec::new(); insertions],
    };
    for op in ops.iter().rev() {
        match op {
            Op::Conjugate(_) | Op::Scratch => {}
            Op::Rotate { physical, .. } | Op::MeasureFixed { physical, .. } => back.check(physical),
            Op::MeasureFrame { physical, x_d, .. } => {
                back.insertion(x_d);
                back.check(physical);
            }
            Op::MeasureDense {
                physical,
                x_physical,
                ..
            } => {
                back.insertion(x_physical);
                back.check(physical);
            }
            Op::ResetFix(_, fix) => back.insertion(fix),
            Op::Noise { sites, .. } => {
                for site in sites.iter().rev() {
                    back.rewind(site.at);
                    for &q in site.qubits.iter().rev() {
                        // Z_q flips the checks with X on q, X_q those with Z.
                        let column = back.slots.column(q);
                        let i = back.i;
                        back.rows[i - 1] = sparse(&back.slots.x[column.clone()]);
                        back.rows[i - 2] = sparse(&back.slots.z[column]);
                        back.i -= 2;
                    }
                }
            }
        }
    }
    debug_assert_eq!((back.k, back.i), (0, 0));
    back.rows
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
    let (ops, peak, gates) = compile(base, instructions)?;
    let (acts, checks) = link(&ops, &gates, base.n);
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
            let group = Shots::new(checks, shots, clbits);
            run(&acts, base.dense.clone(), group, peak, seed)
        })
        .flatten_iter()
        .collect();
    Ok(leaves)
}

/// The shots of a group: which are still in it, each one's sign register
/// (one bit per check, [`link`]), and the record columns.
struct Shots {
    words: usize,
    alive: Vec<u64>,
    /// Words per register.
    width: usize,
    /// The registers, one after another, one per slot.
    signs: Vec<u64>,
    /// One column of shots per classical bit, then scratch columns for the
    /// outcomes of resets in progress.
    records: Vec<Vec<u64>>,
}

impl Shots {
    /// `shots` (at least one) error-free shots with `clbits` classical bits
    /// at 0.
    fn new(checks: usize, shots: usize, clbits: usize) -> Shots {
        let words = shots.div_ceil(64);
        let width = checks.div_ceil(64).max(1);
        Shots {
            words,
            alive: super::valid(shots),
            width,
            signs: vec![0; words * 64 * width],
            records: vec![vec![0; words]; clbits],
        }
    }

    fn count(&self) -> usize {
        popcount(&self.alive)
    }

    /// Per live shot, its register bit `k`: whether its frame anticommutes
    /// with check `k`.
    fn flips(&self, k: usize) -> Vec<u64> {
        let (w, b) = (k / 64, k % 64);
        let mut out = vec![0u64; self.words];
        for s in super::set_bits(&self.alive) {
            out[s / 64] |= (self.signs[s * self.width + w] >> b & 1) << (s % 64);
        }
        out
    }

    /// Flip `row` in the register of shot `s`.
    fn flip(&mut self, s: usize, row: &Row) {
        let register = &mut self.signs[s * self.width..(s + 1) * self.width];
        for &(w, bits) in row {
            register[w] ^= bits;
        }
    }

    /// Flip `row` in the registers of the shots in `mask`.
    fn flip_all(&mut self, mask: &[u64], row: &Row) {
        if row.is_empty() {
            return;
        }
        for s in super::set_bits(mask) {
            self.flip(s, row);
        }
    }

    /// A compact copy of the shots in `mask` (which must be alive), in
    /// order.
    fn extract(&self, mask: &[u64]) -> Shots {
        let chosen: Vec<usize> = super::set_bits(mask).collect();
        let words = chosen.len().div_ceil(64);
        let mut signs = vec![0u64; words * 64 * self.width];
        for (i, &s) in chosen.iter().enumerate() {
            signs[i * self.width..(i + 1) * self.width]
                .copy_from_slice(&self.signs[s * self.width..(s + 1) * self.width]);
        }
        let records = self
            .records
            .iter()
            .map(|column| {
                let mut out = vec![0u64; words];
                for (i, &s) in chosen.iter().enumerate() {
                    out[i / 64] |= (column[s / 64] >> (s % 64) & 1) << (i % 64);
                }
                out
            })
            .collect();
        Shots {
            words,
            alive: super::valid(chosen.len()),
            width: self.width,
            signs,
            records,
        }
    }

    /// As [`Group::split_off`].
    fn split_off(&mut self, mask: &[u64], compact: bool) -> Shots {
        let other = self.extract(mask);
        for (a, m) in self.alive.iter_mut().zip(mask) {
            *a &= !m;
        }
        if compact || (self.words > 1 && self.count() * 4 < self.words * 64 * 3) {
            *self = self.extract(&self.alive.clone());
        }
        other
    }

    /// One leaf per live shot: its classical bits.
    fn leaves(&self, peak: usize) -> Vec<Leaf> {
        super::set_bits(&self.alive)
            .map(|s| {
                let clbits = self
                    .records
                    .iter()
                    .map(|column| column[s / 64] >> (s % 64) & 1 == 1)
                    .collect();
                (clbits, 1, peak)
            })
            .collect()
    }
}

/// A group waiting to go on from step `pc`, where it split off: with the
/// sign of its rotation, or the effective outcome of its measurement, at
/// `pc` already decided (and the step's work before the split done).
struct Task {
    pc: usize,
    dense: Arc<Dense>,
    group: Shots,
    decided: Option<bool>,
}

/// Take the shots of `group`, in state `dense`, through the program, one
/// group at a time (depth first), and return one leaf per shot.
fn run(acts: &[Act], dense: Arc<Dense>, group: Shots, peak: usize, seed: u64) -> Vec<Leaf> {
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
        while pc < acts.len() {
            let mut split = |dense: &Arc<Dense>, group: Shots, decided: bool| {
                stack.push(Task {
                    pc,
                    dense: dense.clone(),
                    group,
                    decided: Some(decided),
                })
            };
            match &acts[pc] {
                &Act::Rotate {
                    grow,
                    ref p,
                    theta,
                    check,
                } => {
                    // Whether this group's shots see -θ; a group whose shots
                    // disagree splits, the smaller side leaving (sharing the
                    // dense part until it rotates).
                    let flipped = decided.take().unwrap_or_else(|| {
                        if grow {
                            Arc::make_mut(&mut dense).grow();
                        }
                        let flips = group.flips(check);
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
                &Act::MeasureFixed {
                    clbit,
                    check,
                    fixed,
                } => {
                    let fixed = if fixed { u64::MAX } else { 0 };
                    let flips = group.flips(check);
                    *record(&mut group, clbit) = flips
                        .iter()
                        .zip(&group.alive)
                        .map(|(f, a)| (f ^ fixed) & a)
                        .collect();
                }
                &Act::MeasureFrame {
                    clbit,
                    check,
                    ref row,
                } => {
                    let flips = group.flips(check);
                    let effective: Vec<u64> = group
                        .alive
                        .iter()
                        .map(|a| rng.random::<u64>() & a)
                        .collect();
                    group.flip_all(&effective, row);
                    *record(&mut group, clbit) =
                        effective.iter().zip(&flips).map(|(e, f)| e ^ f).collect();
                }
                &Act::MeasureDense {
                    clbit,
                    check,
                    ref p,
                    negative,
                    ref row,
                } => {
                    // P(effective outcome 1) in this group's dense part.
                    let one = ((1.0 - dense.expectation(*p)) / 2.0).clamp(0.0, 1.0);
                    let outcome = decided.take().unwrap_or_else(|| {
                        // Draw each shot's effective outcome; a group with
                        // both splits, the shots of outcome 1 leaving.
                        let flips = group.flips(check);
                        let mut effective = vec![0u64; group.words];
                        for s in super::set_bits(&group.alive) {
                            effective[s / 64] |= ((rng.random::<f64>() < one) as u64) << (s % 64);
                        }
                        *record(&mut group, clbit) =
                            effective.iter().zip(&flips).map(|(e, f)| e ^ f).collect();
                        let (count, n_ones) = (group.count(), popcount(&effective));
                        if n_ones == 0 || n_ones == count {
                            return n_ones != 0;
                        }
                        split(&dense, group.split_off(&effective, true), true);
                        false
                    });
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
                        group.flip_all(&alive, row);
                    }
                }
                Act::Scratch => group.records.push(vec![0; group.words]),
                Act::ResetFix(row) => {
                    let ones = group.records.pop().expect("the scratch column");
                    group.flip_all(&ones, row);
                }
                Act::Noise { p, sites } => noise(&mut group, &mut rng, *p, sites),
            }
            pc += 1;
        }
        leaves.extend(group.leaves(peak));
    }
    leaves
}

/// The record column `clbit`, or the newest scratch column for
/// `usize::MAX`.
fn record(group: &mut Shots, clbit: usize) -> &mut Vec<u64> {
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
/// and site. An error `E` flips, in the shot's register, the rows of the X
/// and Z parts of `C† E C`.
fn noise(group: &mut Shots, rng: &mut Pcg64, p: f64, sites: &[Hit]) {
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
        for (&(ex, ez), (x, z)) in error.iter().zip(&site.rows) {
            if ex {
                group.flip(s, x);
            }
            if ez {
                group.flip(s, z);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::{Circuit, Gate};
    use rand::RngExt;

    /// For each insertion of `ops`, in the order [`link`] numbers them, the
    /// checks it flips, found the slow way: a frame of its own holding only
    /// that Pauli, conjugated step by step, multiplied with every later
    /// check.
    fn rows_one_by_one(ops: &[Op], n: usize) -> Vec<Vec<usize>> {
        let mut rows = Vec::new();
        for target in 0.. {
            let mut frame = Group::new(n, 1, &[]);
            let (mut next, mut k, mut row) = (0usize, 0usize, Vec::new());
            let mut check = |frame: &Group, (x, z): &Check, k: &mut usize| {
                if frame.anticommutes(x, z)[0] & 1 == 1 {
                    row.push(*k);
                }
                *k += 1;
            };
            let mut insert = |frame: &mut Group, x: &[u64], z: &[u64]| {
                if next == target {
                    frame.multiply_shot(0, x, z);
                }
                next += 1;
            };
            let x_on = |d: usize| {
                let mut x = vec![0u64; n.div_ceil(64)];
                x[d / 64] |= 1 << (d % 64);
                x
            };
            for op in ops {
                match op {
                    Op::Conjugate(cliffords) => cliffords.iter().for_each(|&g| frame.conjugate(g)),
                    Op::Rotate { check: c, .. } | Op::MeasureFixed { check: c, .. } => {
                        check(&frame, c, &mut k)
                    }
                    Op::MeasureFrame {
                        check: c,
                        collapse,
                        d,
                        ..
                    } => {
                        check(&frame, c, &mut k);
                        collapse.iter().for_each(|&g| frame.conjugate(g));
                        insert(&mut frame, &x_on(*d), &[]);
                    }
                    Op::MeasureDense {
                        check: c,
                        retire,
                        x_retired,
                        ..
                    } => {
                        check(&frame, c, &mut k);
                        retire.iter().for_each(|&g| frame.conjugate(g));
                        insert(&mut frame, x_retired, &[]);
                    }
                    Op::Scratch => {}
                    Op::ResetFix((x, z), _) => insert(&mut frame, x, z),
                    Op::Noise { sites, .. } => {
                        for site in sites {
                            for (x, z) in &site.images {
                                insert(&mut frame, &x.0, &x.1);
                                insert(&mut frame, &z.0, &z.1);
                            }
                        }
                    }
                }
            }
            if target >= next {
                break;
            }
            rows.push(row);
        }
        rows
    }

    /// The rows of `acts`, in insertion order.
    fn rows_of(acts: &[Act]) -> Vec<Vec<usize>> {
        let bits = |row: &Row| {
            row.iter()
                .flat_map(|&(w, b)| {
                    super::super::set_bits(&[b])
                        .map(move |i| w * 64 + i)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        let mut rows = Vec::new();
        for act in acts {
            match act {
                Act::MeasureFrame { row, .. }
                | Act::MeasureDense { row, .. }
                | Act::ResetFix(row) => rows.push(bits(row)),
                Act::Noise { sites, .. } => {
                    for site in sites {
                        for (x, z) in &site.rows {
                            rows.push(bits(x));
                            rows.push(bits(z));
                        }
                    }
                }
                _ => {}
            }
        }
        rows
    }

    #[test]
    fn sign_registers_match_frames_one_insertion_at_a_time() {
        // Random Clifford+T circuits with every kind of step: noise of one
        // and two qubits between gates, mid-circuit measurements that are
        // fixed, 50/50 or on the dense part, resets, and final measurements.
        // Over 70 qubits, so that the frames and registers take two words.
        let mut rng = Pcg64::seed_from_u64(5);
        let mut measured = 0;
        for (case, n) in [5usize, 9, 70].into_iter().enumerate() {
            let mut circuit = Circuit::new(n);
            circuit.num_clbits = n + 8;
            for layer in 0..12 {
                for q in 0..n {
                    let gate = [Gate::H, Gate::S, Gate::T, Gate::SX, Gate::I]
                        [rng.random_range(0..5)]
                    .clone();
                    circuit.gate(gate, &[q]);
                    if rng.random::<f64>() < 0.5 {
                        circuit.depolarize1(q, 0.01);
                    }
                }
                for q in (layer % 2..n - 1).step_by(2) {
                    circuit.cx(q, q + 1);
                    circuit.depolarize2(q, q + 1, 0.02);
                }
                if layer % 4 == 3 {
                    let q = rng.random_range(0..n);
                    circuit.measure(q, n + layer / 4);
                    circuit.reset(rng.random_range(0..n));
                    circuit.x_error(q, 0.1);
                }
            }
            circuit.measure_all();
            let (ops, _, gates) = compile(&NearClifford::new(n), &circuit.instructions).unwrap();
            let expected = rows_one_by_one(&ops, n);
            assert!(expected.iter().any(|r| !r.is_empty()), "case {case}");
            let (acts, checks) = link(&ops, &gates, n);
            assert_eq!(rows_of(&acts), expected, "case {case}");
            // The measurements' and resets' own insertions (physical images
            // of virtual Paulis, from the tableau) flip later checks too.
            measured += acts
                .iter()
                .filter(|act| match act {
                    Act::MeasureFrame { row, .. }
                    | Act::MeasureDense { row, .. }
                    | Act::ResetFix(row) => !row.is_empty(),
                    _ => false,
                })
                .count();
            // Over 64 checks, so that rows take two words.
            assert!(n < 64 || checks > 64, "case {case}");
        }
        assert!(measured > 0);
    }
}
