//! Pauli propagation: expectation values `<0|U† O U|0>` of circuits far
//! beyond a state vector, by evolving the observable instead of the state.
//!
//! The observable is a sum of Pauli strings `O = Σ c_P P`, propagated
//! backwards through the circuit (the Heisenberg picture, `O -> G† O G`).
//! A Clifford gate maps each string to one string; a Pauli rotation
//! `R = exp(-iφ/2 Q)` leaves the strings that commute with `Q` alone and
//! splits each one that anticommutes in two:
//!
//! ```text
//! R† P R = cos φ P + sin φ (iQP)
//! ```
//!
//! so the number of strings can double at every rotation. Strings whose
//! coefficient falls below a threshold are dropped, the "sparse Pauli
//! dynamics" of Begušić, Gray and Chan (arXiv:2308.05077), which reproduced
//! IBM's 127-qubit kicked Ising experiment on a laptop; the method and its
//! variants are reviewed by Rudolph et al. (arXiv:2505.21606). For random
//! circuits the mean squared error of truncated propagation is provably
//! small (Angrisani et al., arXiv:2409.01706).
//!
//! Three choices shape this implementation:
//!
//! - **Clifford frame.** Cliffords never touch the observable. Walking the
//!   circuit from the end, the Cliffords seen so far are collected into a
//!   frame `K` ([`Frame`]); a rotation `R_Q(θ)` met behind them is moved
//!   past them as `K R_Q(θ) = R_{KQK†}(θ) K`, and only the rotation, with
//!   its Pauli rewritten, is applied to the observable. At the end
//!   `<0|K† P K|0>` is read off the frame for each surviving string. In
//!   the kicked Ising circuits, two thirds of the gates are Clifford ZZ
//!   rotations and cost nothing.
//! - **Quarter turns join the frame.** A rotation by a multiple of π/2
//!   (as lowering `CP(π)` or `U(π/2, 0, π)` produces) is Clifford and goes
//!   to the frame instead of splitting strings. Splitting off the π/2 part
//!   of other angles, `R_Q(θ) = R_Q(θ - π/2) R_Q(π/2)`, would not help:
//!   it yields the same strings with the cosine and sine swapped, so a
//!   coefficient threshold drops exactly the same ones.
//! - **Sharded sum.** The strings live in 64 hash maps chosen by hash. A
//!   rotation first scales every anticommuting string in place and emits
//!   its partner `iQP` to the partner's shard (64 shards in parallel), then
//!   every shard adds what it received (in parallel again). Partners are
//!   unique (`P -> QP` is a bijection), so each string receives at most one
//!   contribution per rotation and no locks are needed.
//!
//! Dropping strings is the only approximation, and its cost is bounded:
//! the dropped part of the observable keeps its operator norm under the
//! rest of the circuit, so the error of the result is at most the sum of
//! the dropped coefficients' magnitudes, reported as
//! [`Expectation::truncated_one_norm`]. That bound is rigorous but loose;
//! errors in practice are far smaller.

mod pauli;

use std::collections::HashMap;
use std::f64::consts::FRAC_PI_2;
use std::hash::{BuildHasherDefault, Hasher};
use std::time::Instant;

use rayon::prelude::*;

use crate::circuit::{Circuit, Instruction};
use crate::nearclifford::{Step, decompose};

pub use pauli::{Frame, Pauli, Signed};

/// An observable: a real linear combination of Pauli strings, each given
/// as `(qubit, 'X' | 'Y' | 'Z')` factors.
pub type Observable = Vec<(f64, Vec<(usize, char)>)>;

/// Parse a Pauli string written as factors, `"Z62"` or `"X0 Y3 Z7"`
/// (spaces optional, `"X0Y3Z7"`), or as a dense label of `I`, `X`, `Y`, `Z`
/// with qubit 0 on the right (Qiskit's order, `"IXZ"` is `Z0 X1`).
pub fn parse_pauli(text: &str) -> Result<Vec<(usize, char)>, String> {
    let text = text.trim();
    let is_pauli = |c: char| matches!(c.to_ascii_uppercase(), 'I' | 'X' | 'Y' | 'Z');
    if !text.is_empty() && text.chars().all(is_pauli) {
        return Ok(text
            .chars()
            .rev()
            .enumerate()
            .filter(|&(_, c)| !c.eq_ignore_ascii_case(&'I'))
            .map(|(q, c)| (q, c.to_ascii_uppercase()))
            .collect());
    }
    let mut factors = Vec::new();
    let mut chars = text.chars().filter(|c| !c.is_whitespace()).peekable();
    while let Some(c) = chars.next() {
        if !is_pauli(c) {
            return Err(format!("cannot parse Pauli string {text:?}"));
        }
        let mut digits = String::new();
        while let Some(&d) = chars.peek().filter(|d| d.is_ascii_digit()) {
            digits.push(d);
            chars.next();
        }
        let qubit = digits
            .parse()
            .map_err(|_| format!("cannot parse Pauli string {text:?}: {c} needs a qubit"))?;
        if !c.eq_ignore_ascii_case(&'I') {
            factors.push((qubit, c.to_ascii_uppercase()));
        }
    }
    Ok(factors)
}

/// Truncation settings.
#[derive(Clone, Debug)]
pub struct PropagationOptions {
    /// Drop strings whose coefficient is smaller than this in magnitude
    /// (relative to an observable whose coefficients sum to 1 in
    /// magnitude, the typical case being a single string). 0 keeps
    /// everything: exact, and exponential in the number of rotations.
    pub min_abs_coefficient: f64,
    /// Keep at most this many strings, the largest, after each rotation.
    pub max_terms: Option<usize>,
}

impl Default for PropagationOptions {
    fn default() -> PropagationOptions {
        PropagationOptions {
            min_abs_coefficient: 1e-5,
            max_terms: None,
        }
    }
}

/// The result of [`expectation`].
#[derive(Clone, Debug, Default)]
pub struct Expectation {
    pub value: f64,
    /// Sum of the magnitudes of every dropped coefficient: a rigorous bound
    /// on `|value - exact value|`. 0 means the value is exact (to rounding).
    pub truncated_one_norm: f64,
    /// Non-Clifford rotations applied to the observable (rotations by
    /// multiples of π/2 join the frame instead).
    pub rotations: usize,
    /// The most strings held at once.
    pub peak_terms: usize,
    /// Strings left at the end.
    pub final_terms: usize,
    pub seconds: f64,
}

/// `<0...0| U† O U |0...0>` for the unitary circuit `U` by Pauli
/// propagation. Fails on measurements, resets and arbitrary unitaries on
/// two or more qubits, and above 1024 qubits.
pub fn expectation(
    circuit: &Circuit,
    observable: &Observable,
    options: &PropagationOptions,
) -> Result<Expectation, String> {
    // Strings are fixed-size bit words; round the word count up to one of a
    // few sizes so only those are compiled.
    match circuit.num_qubits.max(1).div_ceil(64) {
        1 => run::<1>(circuit, observable, options),
        2 => run::<2>(circuit, observable, options),
        3 | 4 => run::<4>(circuit, observable, options),
        5..=8 => run::<8>(circuit, observable, options),
        9..=16 => run::<16>(circuit, observable, options),
        _ => Err(format!(
            "Pauli propagation supports up to 1024 qubits, not {}",
            circuit.num_qubits
        )),
    }
}

/// The circuit's steps in time order: Cliffords and Pauli rotations.
fn steps(circuit: &Circuit) -> Result<Vec<Step>, String> {
    let mut steps = Vec::new();
    for instruction in &circuit.instructions {
        match instruction {
            Instruction::Gate { gate, qubits } => steps.extend(
                decompose(gate, qubits)
                    .ok_or_else(|| format!("{gate:?} is not supported by Pauli propagation"))?,
            ),
            Instruction::Barrier { .. } => {}
            other => {
                return Err(format!(
                    "expectation values need a unitary circuit, found {other:?}"
                ));
            }
        }
    }
    Ok(steps)
}

fn run<const W: usize>(
    circuit: &Circuit,
    observable: &Observable,
    options: &PropagationOptions,
) -> Result<Expectation, String> {
    let start = Instant::now();
    let n = circuit.num_qubits;
    let mut sum = Sum::<W>::new();
    for (coefficient, factors) in observable {
        if let Some(&(q, _)) = factors.iter().find(|&&(q, _)| q >= n) {
            return Err(format!("qubit {q} is out of range"));
        }
        sum.add(Pauli::from_factors(factors)?, *coefficient);
    }
    let steps = steps(circuit)?;
    let mut frame = Frame::<W>::identity(n);
    let mut result = Expectation {
        peak_terms: sum.len(),
        ..Expectation::default()
    };
    let atol = options.min_abs_coefficient.max(0.0);
    for step in steps.iter().rev() {
        match step {
            Step::Clifford(g) => frame.apply_before(*g),
            Step::Rotation(factors, theta) => {
                let rotated = frame.conjugate(&Pauli::from_factors(factors)?);
                let angle = if rotated.negative { -theta } else { *theta };
                let turns = (angle / FRAC_PI_2).round();
                if (angle - turns * FRAC_PI_2).abs() < 1e-12 {
                    frame.rotate_after(&rotated.pauli, turns.rem_euclid(4.0) as u32);
                } else {
                    result.truncated_one_norm += sum.rotate(&rotated.pauli, angle, atol);
                    if let Some(max) = options.max_terms {
                        result.truncated_one_norm += sum.keep_largest(max);
                    }
                    result.rotations += 1;
                    result.peak_terms = result.peak_terms.max(sum.len());
                }
            }
        }
    }
    result.final_terms = sum.len();
    result.value = sum.expectation(&frame);
    result.seconds = start.elapsed().as_secs_f64();
    Ok(result)
}

/// FxHash's mixing step with a final avalanche, so the low bits (the
/// bucket) and the top seven (hashbrown's tag) are both well mixed.
#[derive(Default)]
struct WordHasher(u64);

impl Hasher for WordHasher {
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.write_u64(u64::from_le_bytes(word));
        }
    }

    fn write_u64(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }

    fn write_usize(&mut self, word: usize) {
        self.write_u64(word as u64);
    }

    fn finish(&self) -> u64 {
        let mut h = self.0;
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^ (h >> 33)
    }
}

type Map<const W: usize> = HashMap<Pauli<W>, f64, BuildHasherDefault<WordHasher>>;

const SHARD_BITS: u32 = 6;
const SHARDS: usize = 1 << SHARD_BITS;

/// Below this many strings, shards are processed on one thread: a parallel
/// pass costs a few microseconds, and circuits have thousands of rotations.
const PARALLEL_TERMS: usize = 1 << 14;

/// The shard of a string: the top bits of a multiplicative hash that is
/// independent of the map's own.
fn shard<const W: usize>(p: &Pauli<W>) -> usize {
    let mut h = 0u64;
    for w in 0..W {
        h = (h ^ p.x[w]).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        h = (h ^ p.z[w]).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
    (h >> (64 - SHARD_BITS)) as usize
}

/// A sparse sum of Pauli strings with real coefficients.
struct Sum<const W: usize> {
    shards: Vec<Map<W>>,
}

/// Run `f` on every shard (with its index), in parallel for large sums.
fn each_shard<T: Send, M: Send>(
    shards: &mut [M],
    parallel: bool,
    f: impl Fn(usize, &mut M) -> T + Sync,
) -> Vec<T> {
    if parallel {
        shards
            .par_iter_mut()
            .enumerate()
            .map(|(i, m)| f(i, m))
            .collect()
    } else {
        shards
            .iter_mut()
            .enumerate()
            .map(|(i, m)| f(i, m))
            .collect()
    }
}

impl<const W: usize> Sum<W> {
    fn new() -> Sum<W> {
        Sum {
            shards: (0..SHARDS).map(|_| Map::default()).collect(),
        }
    }

    fn len(&self) -> usize {
        self.shards.iter().map(|m| m.len()).sum()
    }

    fn add(&mut self, p: Pauli<W>, c: f64) {
        *self.shards[shard(&p)].entry(p).or_insert(0.0) += c;
    }

    /// `O -> R† O R` for `R = exp(-iφ/2 Q)`, dropping coefficients below
    /// `atol`; returns the magnitude dropped.
    ///
    /// Strings below `atol` are dropped one rotation late, when they are
    /// next visited, so that a string scaled by `cos φ` can still receive
    /// its partner's contribution first.
    fn rotate(&mut self, q: &Pauli<W>, phi: f64, atol: f64) -> f64 {
        let (sin, cos) = phi.sin_cos();
        let parallel = self.len() >= PARALLEL_TERMS;
        // Pass 1: scale anticommuting strings and emit their partners.
        let emitted = each_shard(&mut self.shards, parallel, |_, map| {
            let mut out: Vec<Vec<(Pauli<W>, f64)>> = vec![Vec::new(); SHARDS];
            let mut dropped = 0.0;
            map.retain(|p, c| {
                if c.abs() < atol || *c == 0.0 {
                    dropped += c.abs();
                    return false;
                }
                if p.anticommutes(q) {
                    // iQP = i^{k+1} R, and k + 1 is even.
                    let (k, partner) = q.mul(p);
                    let value = if k == 3 { sin * *c } else { -sin * *c };
                    if value.abs() < atol {
                        dropped += value.abs();
                    } else {
                        out[shard(&partner)].push((partner, value));
                    }
                    *c *= cos;
                }
                true
            });
            (out, dropped)
        });
        let mut dropped = 0.0;
        let mut inboxes: Vec<Vec<Vec<(Pauli<W>, f64)>>> = (0..SHARDS).map(|_| Vec::new()).collect();
        for (out, d) in emitted {
            dropped += d;
            for (inbox, part) in inboxes.iter_mut().zip(out) {
                if !part.is_empty() {
                    inbox.push(part);
                }
            }
        }
        // Pass 2: every shard adds what it received. Each string receives
        // at most one contribution, so a cancellation is final.
        let parallel =
            parallel || inboxes.iter().flatten().map(Vec::len).sum::<usize>() >= PARALLEL_TERMS;
        let dropped_on_merge = each_shard(&mut self.shards, parallel, |i, map| {
            let mut dropped = 0.0;
            for &(p, value) in inboxes[i].iter().flatten() {
                let c = map.entry(p).or_insert(0.0);
                *c += value;
                if c.abs() < atol || *c == 0.0 {
                    dropped += c.abs();
                    map.remove(&p);
                }
            }
            dropped
        });
        dropped + dropped_on_merge.iter().sum::<f64>()
    }

    /// Keep the `max` strings with the largest coefficients; returns the
    /// magnitude dropped.
    fn keep_largest(&mut self, max: usize) -> f64 {
        let len = self.len();
        if len <= max {
            return 0.0;
        }
        let mut magnitudes: Vec<f64> = self
            .shards
            .par_iter()
            .flat_map_iter(|m| m.values().map(|c| c.abs()))
            .collect();
        let cut = len - max;
        // Everything strictly below the cut's magnitude goes, and then
        // enough of the strings equal to it.
        let (_, &mut threshold, _) = magnitudes.select_nth_unstable_by(cut, f64::total_cmp);
        let below = magnitudes.iter().filter(|&&m| m < threshold).count();
        let mut ties = cut - below;
        let mut dropped = 0.0;
        for map in &mut self.shards {
            map.retain(|_, c| {
                let m = c.abs();
                if m < threshold || (m == threshold && ties > 0) {
                    if m == threshold {
                        ties -= 1;
                    }
                    dropped += m;
                    false
                } else {
                    true
                }
            });
        }
        dropped
    }

    /// `Σ c_P <0|K† P K|0>`. Strings with an X part in the frame's output
    /// basis contribute nothing; the rest are ±1.
    fn expectation(&mut self, frame: &Frame<W>) -> f64 {
        let parallel = self.len() >= PARALLEL_TERMS;
        each_shard(&mut self.shards, parallel, |_, map| {
            map.iter()
                .map(|(p, c)| c * frame.expectation(p))
                .sum::<f64>()
        })
        .iter()
        .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sparse_and_dense_labels() {
        assert_eq!(parse_pauli("Z62").unwrap(), vec![(62, 'Z')]);
        assert_eq!(
            parse_pauli("X0 y3Z7").unwrap(),
            vec![(0, 'X'), (3, 'Y'), (7, 'Z')]
        );
        assert_eq!(parse_pauli("IXZ").unwrap(), vec![(0, 'Z'), (1, 'X')]);
        assert!(parse_pauli("X").is_ok());
        assert!(parse_pauli("Q3").is_err());
        assert!(parse_pauli("X3 Z").is_err());
    }

    #[test]
    fn products_follow_the_pauli_algebra() {
        let p = |s: &str| Pauli::<1>::from_factors(&parse_pauli(s).unwrap()).unwrap();
        // XY = iZ, YX = -iZ, ZX = iY, XX = I.
        assert_eq!(p("X0").mul(&p("Y0")), (1, p("Z0")));
        assert_eq!(p("Y0").mul(&p("X0")), (3, p("Z0")));
        assert_eq!(p("Z0").mul(&p("X0")), (1, p("Y0")));
        assert_eq!(p("X0").mul(&p("X0")), (0, Pauli::IDENTITY));
        // Per qubit XZ = -iY and ZX = iY: the phases cancel.
        assert_eq!(p("X0 Z1").mul(&p("Z0 X1")), (0, p("Y0 Y1")));
        assert!(p("X0").anticommutes(&p("Z0")));
        assert!(!p("X0 X1").anticommutes(&p("Z0 Z1")));
        assert_eq!(p("X0 Y5 Z9").weight(), 3);
    }
}
