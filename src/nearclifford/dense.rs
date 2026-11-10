//! The dense state of the active virtual qubits.
//!
//! Bit `p` of an amplitude's index is the value of the qubit at position
//! `p`. Paulis on positions are `(x, z, negative)` bit masks meaning
//! `(-1)^negative i^{|x & z|} X^x Z^z` (so `(1, 1)` on a qubit is Y).

use std::ops::Range;

use rayon::prelude::*;

use crate::matrix::C64;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
mod avx2;

/// Indices at or above which loops run in parallel.
const PARALLEL: usize = 1 << 14;

/// Whether the AVX2 kernels take states of `len` amplitudes (at least two
/// blocks); smaller states, and builds without AVX2 and FMA, use the scalar
/// loops.
#[inline(always)]
fn vectorized(len: usize) -> bool {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    {
        len >= 2 * avx2::BLOCK
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    {
        let _ = len;
        false
    }
}

/// Run `f` on consecutive ranges covering `0..units`, in parallel for
/// large counts (the ranges are multiples of the AVX2 block).
#[cfg_attr(
    not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )),
    allow(dead_code)
)]
fn ranges(units: usize, f: impl Fn(Range<usize>) + Sync + Send) {
    let size = PARALLEL / 8;
    if units >= PARALLEL / 2 {
        (0..units.div_ceil(size))
            .into_par_iter()
            .for_each(|c| f(c * size..units.min((c + 1) * size)));
    } else {
        f(0..units);
    }
}

/// `f` summed over consecutive ranges covering `0..units`, as [`ranges`].
#[cfg_attr(
    not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )),
    allow(dead_code)
)]
fn range_sum(units: usize, f: impl Fn(Range<usize>) -> f64 + Sync + Send) -> f64 {
    let size = PARALLEL / 8;
    if units >= PARALLEL / 2 {
        (0..units.div_ceil(size))
            .into_par_iter()
            .map(|c| f(c * size..units.min((c + 1) * size)))
            .sum()
    } else {
        f(0..units)
    }
}

/// A Pauli on positions; see the module documentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Pauli {
    pub x: u64,
    pub z: u64,
    pub negative: bool,
}

impl Pauli {
    /// `P|i> = c |i ^ x>`: the coefficient `c`, which is
    /// `phase() * sign(i)`.
    #[cfg(test)]
    fn coefficient(&self, i: usize) -> C64 {
        self.phase() * self.sign(i)
    }

    /// `(-1)^negative i^{|x & z|}`: the part of the coefficient that does
    /// not depend on the index.
    fn phase(&self) -> C64 {
        let c = match (self.x & self.z).count_ones() % 4 {
            0 => C64::new(1.0, 0.0),
            1 => C64::new(0.0, 1.0),
            2 => C64::new(-1.0, 0.0),
            _ => C64::new(0.0, -1.0),
        };
        if self.negative { -c } else { c }
    }

    /// `(-1)^{|z & i|}`, as `±1`.
    #[inline(always)]
    fn sign(&self, i: usize) -> f64 {
        1.0 - 2.0 * ((self.z & i as u64).count_ones() & 1) as f64
    }

    pub fn is_identity(&self) -> bool {
        self.x == 0 && self.z == 0
    }
}

/// A raw pointer that disjoint index pairs may write through in parallel.
#[derive(Clone, Copy)]
struct Shared(*mut C64);
// SAFETY: only used for writes to disjoint indices.
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

impl Shared {
    fn get(self) -> *mut C64 {
        self.0
    }
}

/// `f(0) + ... + f(count - 1)`, in parallel for large counts.
fn sum(count: usize, f: impl Fn(usize) -> f64 + Sync + Send) -> f64 {
    if count >= PARALLEL / 2 {
        (0..count)
            .into_par_iter()
            .with_min_len(PARALLEL / 8)
            .map(f)
            .sum()
    } else {
        // Four running sums: one waits on the adder's latency at every
        // term (2 ns per term, against 0.3 with several).
        let mut acc = [0.0; 4];
        for k in (0..count & !3).step_by(4) {
            for (e, a) in acc.iter_mut().enumerate() {
                *a += f(k + e);
            }
        }
        (count & !3..count).map(f).sum::<f64>() + acc.iter().sum::<f64>()
    }
}

/// Append `f(0), ..., f(count - 1)` to `out`, in parallel for large counts.
fn gather(out: &mut Vec<C64>, count: usize, f: impl Fn(usize) -> C64 + Sync + Send) {
    if count >= PARALLEL {
        out.par_extend((0..count).into_par_iter().with_min_len(PARALLEL / 4).map(f));
    } else {
        out.extend((0..count).map(f));
    }
}

/// The amplitudes of the active qubits.
#[derive(Clone, Debug)]
pub(super) struct Dense {
    pub amps: Vec<C64>,
}

impl Dense {
    /// The one-amplitude state of no qubits.
    pub fn new() -> Dense {
        Dense {
            amps: vec![C64::new(1.0, 0.0)],
        }
    }

    pub fn qubits(&self) -> usize {
        self.amps.len().trailing_zeros() as usize
    }

    /// Add a qubit in |0> at the next position.
    pub fn grow(&mut self) {
        let len = self.amps.len();
        self.amps.resize(2 * len, C64::new(0.0, 0.0));
    }

    /// Remove the qubit at `position`, which must be |0>. (The reference
    /// for [`retired`](Self::retired), with `project` and `apply`.)
    #[cfg(test)]
    pub fn shrink(&mut self, position: usize) {
        let half = self.amps.len() / 2;
        let low = (1usize << position) - 1;
        let old = std::mem::take(&mut self.amps);
        self.amps = (0..half)
            .into_par_iter()
            .with_min_len(PARALLEL)
            .map(|i| old[(i & low) | ((i & !low) << 1)])
            .collect();
    }

    /// Run `f(i, j)` for every pair `i < j = i ^ x` (every index once when
    /// `x == 0`, with `j == i`), in parallel for large states.
    #[cfg(test)]
    fn for_pairs(&mut self, x: u64, f: impl Fn(usize, usize, *mut C64) + Sync) {
        let len = self.amps.len();
        let ptr = Shared(self.amps.as_mut_ptr());
        let low = if x == 0 {
            0
        } else {
            1usize << x.trailing_zeros()
        };
        let run = |i: usize| {
            if i & low == 0 {
                f(i, i ^ x as usize, ptr.get());
            }
        };
        if len >= PARALLEL {
            (0..len)
                .into_par_iter()
                .with_min_len(PARALLEL / 4)
                .for_each(run);
        } else {
            (0..len).for_each(run);
        }
    }

    /// Run `f(i)` for every index `i` with bit `j` clear (every index when
    /// `j` is `None`), in parallel for large states. The loops below visit
    /// only the indices they write, rather than testing every index, and
    /// write the coefficient of `P` as a phase times a sign
    /// ([`Pauli::phase`], [`Pauli::sign`]), without branches: the near-Clifford
    /// backend spends most of its time on noisy circuits in them.
    fn for_half(&mut self, j: Option<u32>, f: impl Fn(usize, *mut C64) + Sync) {
        let ptr = Shared(self.amps.as_mut_ptr());
        let (count, low) = match j {
            Some(j) => (self.amps.len() / 2, (1usize << j) - 1),
            None => (self.amps.len(), usize::MAX),
        };
        let run = |k: usize| f((k & low) | ((k & !low) << 1), ptr.get());
        if count >= PARALLEL / 2 {
            (0..count)
                .into_par_iter()
                .with_min_len(PARALLEL / 8)
                .for_each(run);
        } else {
            (0..count).for_each(run);
        }
    }

    /// `exp(-i θ/2 P)`.
    pub fn rotate(&mut self, p: Pauli, theta: f64) {
        let (c, s) = ((theta / 2.0).cos(), (theta / 2.0).sin());
        // exp(-iθ/2 P)|i> = c |i> - i s phase sign(i) |i ^ x>.
        let m = C64::new(0.0, -s) * p.phase();
        let len = self.amps.len();
        if vectorized(len) {
            #[cfg(all(
                target_arch = "x86_64",
                target_feature = "avx2",
                target_feature = "fma"
            ))]
            {
                let units = if p.x & !1 == 0 { len } else { len / 2 };
                let ptr = Shared(self.amps.as_mut_ptr());
                // SAFETY: the ranges touch disjoint amplitudes.
                ranges(units, |r| unsafe { avx2::rotate(ptr.get(), p, c, m, r) });
            }
            return;
        }
        self.rotate_scalar(p, c, m);
    }

    /// [`rotate`](Self::rotate) one amplitude at a time.
    fn rotate_scalar(&mut self, p: Pauli, c: f64, m: C64) {
        if p.x == 0 {
            let (plus, minus) = (C64::new(c, 0.0) + m, C64::new(c, 0.0) - m);
            self.for_half(None, |i, amps| {
                let factor = if p.sign(i) > 0.0 { plus } else { minus };
                // SAFETY: each index is visited once.
                unsafe { *amps.add(i) *= factor };
            });
            return;
        }
        let x = p.x as usize;
        // sign(i ^ x) = sign(i) sign(x).
        let mx = m * p.sign(x);
        self.for_half(Some(p.x.trailing_zeros()), |i, amps| {
            let j = i ^ x;
            let s = p.sign(i);
            // SAFETY: the pairs (i, i ^ x) are disjoint.
            unsafe {
                let (a, b) = (*amps.add(i), *amps.add(j));
                *amps.add(i) = a * c + mx * s * b;
                *amps.add(j) = b * c + m * s * a;
            }
        });
    }

    /// `<φ|P|φ>` (real for a Hermitian `P`): the phase times the sum of
    /// `sign(i) conj(φ(i ^ x)) φ(i)`. Over a pair `i, i ^ x` with
    /// `sign(i ^ x) = σ sign(i)` (`σ = sign(x)`), the two terms add up to
    /// `sign(i) (conj(b) a + σ conj(a) b)` with `a = φ(i)`, `b = φ(i ^ x)`:
    /// `2 sign(i) Re(conj(b) a)` for `σ = 1`, and `2i sign(i) Im(conj(b) a)`
    /// for `σ = -1`, so the sum is of reals.
    pub fn expectation(&self, p: Pauli) -> f64 {
        let len = self.amps.len();
        if vectorized(len) {
            #[cfg(all(
                target_arch = "x86_64",
                target_feature = "avx2",
                target_feature = "fma"
            ))]
            {
                let units = if p.x & !1 == 0 { len } else { len / 2 };
                let ptr = Shared(self.amps.as_ptr() as *mut C64);
                // SAFETY: only reads, within the state.
                return range_sum(units, |r| unsafe { avx2::expectation(ptr.get(), p, r) });
            }
        }
        self.expectation_scalar(p)
    }

    /// [`expectation`](Self::expectation) one amplitude at a time.
    fn expectation_scalar(&self, p: Pauli) -> f64 {
        let amps = &self.amps;
        if p.x == 0 {
            let sum = sum(amps.len(), |i| p.sign(i) * amps[i].norm_sqr());
            return p.phase().re * sum;
        }
        let x = p.x as usize;
        let low = (1usize << p.x.trailing_zeros()) - 1;
        let pair = |k: usize| {
            let i = (k & low) | ((k & !low) << 1);
            (i, amps[i], amps[i ^ x])
        };
        if p.sign(x) > 0.0 {
            let sum = sum(amps.len() / 2, |k| {
                let (i, a, b) = pair(k);
                p.sign(i) * (a.re * b.re + a.im * b.im)
            });
            2.0 * p.phase().re * sum
        } else {
            let sum = sum(amps.len() / 2, |k| {
                let (i, a, b) = pair(k);
                p.sign(i) * (a.im * b.re - a.re * b.im)
            });
            // Re(phase * 2i sum) = -2 Im(phase) sum.
            -2.0 * p.phase().im * sum
        }
    }

    /// Project onto the `eigenvalue` (`±1`) eigenspace of `P` and
    /// renormalise, given that eigenspace's probability.
    #[cfg(test)]
    pub fn project(&mut self, p: Pauli, eigenvalue: f64, probability: f64) {
        let scale = 0.5 / probability.sqrt();
        self.for_pairs(p.x, |i, j, amps| {
            // SAFETY: the pairs (i, j) are disjoint.
            unsafe {
                if i == j {
                    let a = *amps.add(i);
                    *amps.add(i) = (a + eigenvalue * p.coefficient(i) * a) * scale;
                } else {
                    let (a, b) = (*amps.add(i), *amps.add(j));
                    *amps.add(i) = (a + eigenvalue * p.coefficient(j) * b) * scale;
                    *amps.add(j) = (b + eigenvalue * p.coefficient(i) * a) * scale;
                }
            }
        });
    }

    /// The state after measuring `p` with `eigenvalue` (`±1`, of nonzero
    /// `probability`) and retiring the measured qubit, in one pass: what
    /// [`project`](Self::project), the gates of the parent module's
    /// `retire_plan(p)` (applied as `G†`), an X on its target when `one`
    /// (the target then reads 1) and [`shrink`](Self::shrink) of the target
    /// give, without writing the full-size state in between.
    ///
    /// Why one gather is enough. Let `j` be the plan's target, and `a0` an
    /// index with bit `j` clear.
    /// - If `p` has an X part, `j` is its lowest X position. Every gate of
    ///   the plan before its last (an H on `j`) is S or S† on `j`, or a CX
    ///   or CZ controlled by `j`, so it fixes `|a0>`, and they take the
    ///   projected state `v` to an X eigenstate on `j` with `v(a0)`
    ///   unchanged. The H then puts `√2 v(a0)` at `a0` (eigenvalue +1) or
    ///   at `a0 | 1 << j` (-1, which the X moves back). With
    ///   `v(i) = (φ(i) + λ (Pφ)(i)) / (2 √prob)` and `(Pφ)(i) = c(i ^ x)
    ///   φ(i ^ x)`, the new amplitude at `a` (`a0` without bit `j`) is
    ///   `(φ(a0) + λ c(a0 ^ x) φ(a0 ^ x)) / √(2 prob)`.
    /// - If `p` is Z-type, `j` is its lowest Z position, and the plan is CX
    ///   gates from the other Z positions onto `j`, which leave bit `j`
    ///   equal to the parity of `z & i`, which is `one` on the projected
    ///   state. So the new amplitude is `φ(b) / √prob`, where `b` is `a0`
    ///   with bit `j` set to `one ^ parity(z & a0)`.
    ///
    /// The plan's other gates and the X are still prepended to the frame;
    /// this replaces only their work on the dense state, which was up to
    /// two passes per gate.
    ///
    /// The new state is written into `buffer` (cleared first), so that a
    /// caller can reuse one allocation: a fresh one per measurement costs
    /// page faults once the states are larger than the allocator's
    /// threshold for mapping memory (128 KB, 13 active qubits).
    pub fn retired(
        &self,
        p: Pauli,
        eigenvalue: f64,
        probability: f64,
        one: bool,
        mut buffer: Vec<C64>,
    ) -> Dense {
        let half = self.amps.len() / 2;
        let amps = &self.amps;
        buffer.clear();
        if p.x != 0 && vectorized(self.amps.len()) {
            #[cfg(all(
                target_arch = "x86_64",
                target_feature = "avx2",
                target_feature = "fma"
            ))]
            {
                let scale = (0.5 / probability).sqrt();
                let m = p.phase() * eigenvalue * scale;
                buffer.reserve(half);
                let (from, to) = (
                    Shared(amps.as_ptr() as *mut C64),
                    Shared(buffer.as_mut_ptr()),
                );
                // SAFETY: the ranges write disjoint parts of the reserved
                // buffer, all of it, before its length is set.
                unsafe {
                    ranges(half, |r| {
                        avx2::retired_x(from.get(), p, m, scale, to.get(), r)
                    });
                    buffer.set_len(half);
                }
            }
        } else if p.x != 0 {
            let j = p.x.trailing_zeros();
            let low = (1usize << j) - 1;
            let scale = (0.5 / probability).sqrt();
            let m = p.phase() * eigenvalue;
            let x = p.x as usize;
            gather(&mut buffer, half, |a: usize| {
                let a0 = (a & low) | ((a & !low) << 1);
                (amps[a0] + m * p.sign(a0 ^ x) * amps[a0 ^ x]) * scale
            })
        } else {
            let j = p.z.trailing_zeros();
            let low = (1usize << j) - 1;
            let scale = 1.0 / probability.sqrt();
            let z = p.z as usize;
            gather(&mut buffer, half, |a: usize| {
                let a0 = (a & low) | ((a & !low) << 1);
                let bit = one as usize ^ ((z & a0).count_ones() as usize & 1);
                amps[a0 | bit << j] * scale
            })
        }
        Dense { amps: buffer }
    }

    /// Apply a one- or two-position Clifford.
    #[cfg(test)]
    pub fn apply(&mut self, gate: Gate) {
        let bit = |p: usize| 1usize << p;
        match gate {
            Gate::H(p) => {
                let r = std::f64::consts::FRAC_1_SQRT_2;
                self.for_pairs(bit(p) as u64, |i, j, amps| unsafe {
                    // SAFETY: the pairs (i, j) are disjoint.
                    let (a, b) = (*amps.add(i), *amps.add(j));
                    *amps.add(i) = (a + b) * r;
                    *amps.add(j) = (a - b) * r;
                });
            }
            Gate::Phase(p, factor) => self.for_pairs(0, |i, _, amps| {
                if i & bit(p) != 0 {
                    // SAFETY: each index is visited once.
                    unsafe { *amps.add(i) *= factor };
                }
            }),
            Gate::X(p) => self.for_pairs(bit(p) as u64, |i, j, amps| unsafe {
                // SAFETY: the pairs (i, j) are disjoint.
                std::ptr::swap(amps.add(i), amps.add(j));
            }),
            Gate::CX(c, t) => self.for_pairs(bit(t) as u64, |i, j, amps| {
                if i & bit(c) != 0 {
                    // SAFETY: the pairs (i, j) are disjoint.
                    unsafe { std::ptr::swap(amps.add(i), amps.add(j)) };
                }
            }),
            Gate::CZ(a, b) => self.for_pairs(0, |i, _, amps| {
                if i & bit(a) != 0 && i & bit(b) != 0 {
                    // SAFETY: each index is visited once.
                    unsafe { *amps.add(i) = -*amps.add(i) };
                }
            }),
        }
    }
}

/// Cliffords applied to the dense state, on positions.
#[cfg(test)]
#[derive(Clone, Copy, Debug)]
pub(super) enum Gate {
    H(usize),
    /// Multiply the amplitudes where the position is 1.
    Phase(usize, C64),
    X(usize),
    CX(usize, usize),
    CZ(usize, usize),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_state(qubits: usize, seed: u64) -> Dense {
        let mut x = seed;
        let mut next = || {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        let amps: Vec<C64> = (0..1 << qubits).map(|_| C64::new(next(), next())).collect();
        let norm = amps.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
        Dense {
            amps: amps.into_iter().map(|a| a / norm).collect(),
        }
    }

    #[test]
    fn rotation_by_pi_is_minus_i_times_the_pauli() {
        let p = Pauli {
            x: 0b101,
            z: 0b110,
            negative: true,
        };
        let state = random_state(3, 1);
        let mut rotated = state.clone();
        rotated.rotate(p, std::f64::consts::PI);
        for i in 0..8 {
            // exp(-i π/2 P) = -i P, and P|i> = c(i) |i ^ x>.
            let expected = C64::new(0.0, -1.0) * p.coefficient(i) * state.amps[i];
            assert!((rotated.amps[i ^ p.x as usize] - expected).norm() < 1e-12);
        }
    }

    #[test]
    fn projection_makes_the_expectation_the_eigenvalue() {
        let p = Pauli {
            x: 0b011,
            z: 0b001,
            negative: false,
        };
        let mut state = random_state(3, 2);
        let e = state.expectation(p);
        assert!(e.abs() < 1.0);
        state.project(p, -1.0, (1.0 - e) / 2.0);
        assert!((state.expectation(p) + 1.0).abs() < 1e-12);
        let norm: f64 = state.amps.iter().map(|a| a.norm_sqr()).sum();
        assert!((norm - 1.0).abs() < 1e-12);
    }

    /// Every Pauli on `qubits` positions (with both signs), and a few on
    /// larger states where the kernels run in parallel.
    fn paulis(qubits: usize) -> impl Iterator<Item = Pauli> {
        let all = 1u64 << qubits;
        (0..all).flat_map(move |x| {
            (0..all).flat_map(move |z| [false, true].map(|negative| Pauli { x, z, negative }))
        })
    }

    /// The AVX2 kernels (states of at least 32 amplitudes, in builds with
    /// AVX2 and FMA) against the scalar ones: every Pauli on 5 and 6
    /// positions, then Paulis with X parts from every low position on 15,
    /// where the loops run over parallel ranges.
    #[test]
    fn vectorized_kernels_match_the_scalar_ones() {
        let check = |state: &Dense, p: Pauli| {
            let e = state.expectation(p);
            assert!((e - state.expectation_scalar(p)).abs() < 1e-12, "{p:?}");
            let theta = 0.3 + p.x as f64 * 0.01;
            let (c, m) = (
                (theta / 2.0).cos(),
                C64::new(0.0, -(theta / 2.0).sin()) * p.phase(),
            );
            let mut fast = state.clone();
            fast.rotate(p, theta);
            let mut slow = state.clone();
            slow.rotate_scalar(p, c, m);
            for (a, b) in fast.amps.iter().zip(&slow.amps) {
                assert!((a - b).norm() < 1e-12, "{p:?}");
            }
        };
        for qubits in [5, 6] {
            let state = random_state(qubits, qubits as u64);
            for p in paulis(qubits) {
                check(&state, p);
            }
        }
        let state = random_state(15, 7);
        for low in 0..4 {
            for (x, z) in [(0b1011_0110_0101_0000, 0b0110_1100_1011_0111), (0b100, 0b1)] {
                let negative = low % 2 == 1;
                check(
                    &state,
                    Pauli {
                        x: (x << low) & 0x7fff | 1 << low,
                        z,
                        negative,
                    },
                );
                check(
                    &state,
                    Pauli {
                        x: 0,
                        z: z | 1 << low,
                        negative,
                    },
                );
            }
        }
    }

    /// `retired` on a large state (parallel ranges), against the same
    /// gather done one amplitude at a time.
    #[test]
    fn retiring_a_large_state_matches_the_gather() {
        let state = random_state(15, 9);
        for low in 0..4 {
            let p = Pauli {
                x: (0b0110_0101_1100_1000 << low) & 0x7fff | 1 << low,
                z: 0b0101_1011_0110_0011,
                negative: low == 2,
            };
            let e = state.expectation(p);
            let probability = (1.0 - e) / 2.0;
            let fused = state.retired(p, -1.0, probability, false, Vec::new());
            let j = p.x.trailing_zeros();
            let low_mask = (1usize << j) - 1;
            let m = p.phase() * -1.0;
            for (a, new) in fused.amps.iter().enumerate() {
                let a0 = (a & low_mask) | ((a & !low_mask) << 1);
                let x = p.x as usize;
                let expected = (state.amps[a0] + m * p.sign(a0 ^ x) * state.amps[a0 ^ x])
                    * (0.5 / probability).sqrt();
                assert!((new - expected).norm() < 1e-12, "{p:?}");
            }
        }
    }

    /// `retired` against what it replaces: the projection, the plan's gates
    /// one by one (as `G†`), the X for a target reading 1, and the shrink;
    /// for every Pauli on 4, 5 and 6 positions (the last two through the
    /// AVX2 kernels where they are built), both eigenvalues, on random
    /// states.
    #[test]
    fn retiring_in_one_pass_matches_the_gates_one_by_one() {
        use crate::stabilizer::Clifford;
        let dagger = |g: Clifford| match g {
            Clifford::H(p) => Gate::H(p),
            Clifford::S(p) => Gate::Phase(p, C64::new(0.0, -1.0)),
            Clifford::Sdg(p) => Gate::Phase(p, C64::new(0.0, 1.0)),
            Clifford::X(p) => Gate::X(p),
            Clifford::CX(c, t) => Gate::CX(c, t),
            Clifford::CZ(a, b) => Gate::CZ(a, b),
            other => unreachable!("{other:?}"),
        };
        let mut checked = 0;
        for qubits in [4, 5, 6] {
            for p in paulis(qubits) {
                let Pauli { x, z, negative } = p;
                {
                    if p.is_identity() {
                        continue;
                    }
                    let state = random_state(qubits, x * 37 + z * 3 + negative as u64);
                    let e = state.expectation(p);
                    for eigenvalue in [1.0, -1.0] {
                        let probability = (1.0 + eigenvalue * e) / 2.0;
                        let (gates, target, plan_negative) = super::super::retire_plan(p);
                        let one = (eigenvalue < 0.0) ^ plan_negative;
                        let mut reference = state.clone();
                        reference.project(p, eigenvalue, probability);
                        for &g in &gates {
                            reference.apply(dagger(g));
                        }
                        if one {
                            reference.apply(Gate::X(target));
                        }
                        // The target reads 0 before it is removed.
                        for (i, a) in reference.amps.iter().enumerate() {
                            assert!(i >> target & 1 == 0 || a.norm() < 1e-12);
                        }
                        reference.shrink(target);
                        let fused = state.retired(p, eigenvalue, probability, one, Vec::new());
                        for (a, b) in fused.amps.iter().zip(&reference.amps) {
                            assert!((a - b).norm() < 1e-12, "{p:?} {eigenvalue}");
                        }
                        checked += 1;
                    }
                }
            }
        }
        assert_eq!(checked, 2 * 2 * (255 + 1023 + 4095));
    }

    #[test]
    fn shrink_removes_a_zero_qubit() {
        let mut state = Dense {
            amps: (0..8)
                .map(|i| C64::new(if i & 2 == 0 { i as f64 } else { 0.0 }, 0.0))
                .collect(),
        };
        state.shrink(1);
        assert_eq!(state.amps, [0.0, 1.0, 4.0, 5.0].map(|r| C64::new(r, 0.0)));
    }
}
