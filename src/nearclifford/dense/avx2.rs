//! AVX2 and FMA versions of the dense kernels that noisy near-Clifford
//! sampling spends most of its time in: [`expectation`], [`rotate`] and
//! [`retired_x`] (the X-type case of [`Dense::retired`](super::Dense::retired)).
//!
//! The scalar kernels ran at 0.9-1.4 ns per amplitude on states that sit in
//! the cache (2^14 amplitudes, the 100-qubit benchmark's peak): one
//! complex amplitude at a time, a population count for every sign, and
//! the expectation's sum waiting on one floating-point accumulator. Here a
//! 256-bit register holds two amplitudes, `[re, im, re, im]` at indices
//! `i, i + 1`, as they lie in memory. A complex constant times a register
//! is one shuffle, one multiply and one `fmaddsub`.
//!
//! **Pairs as registers.** Every kernel combines `φ(i)` with `φ(i ^ x)`.
//! Split `x` into its bit 0 and the rest, `xh`. The partners of the
//! register at an even `i` are the register at `i ^ xh`, with its two
//! amplitudes swapped when `x` has bit 0. So the pairs of one register
//! come from one other register, whatever the Pauli: no gathers, and no
//! case for Paulis whose X part starts at position 0 or 1, which are
//! common (positions are numbered in the order qubits became active).
//!
//! **Signs from a table.** `(-1)^{|z & i|}` is linear in `i` under XOR. The
//! loops run over blocks of [`BLOCK`] index units whose low bits come from
//! the table of [`table`] (computed once per call) and whose high bits give
//! one population count per block. A sign is applied as an XOR of the sign
//! bits.
//!
//! The functions take a range of their loop index (amplitudes for loops over
//! all of them, pairs for loops over pairs; multiples of [`BLOCK`]), so that
//! the caller can run large states in parallel; that needs at least
//! `2 * BLOCK` amplitudes.

use std::arch::x86_64::*;
use std::ops::Range;

use super::Pauli;
use crate::matrix::C64;

/// Index units per block: 8 registers.
pub(super) const BLOCK: usize = 16;

/// The two amplitudes at `i` and `i + 1`.
///
/// # Safety
/// `i + 1` must be in bounds.
#[inline(always)]
unsafe fn load(a: *const C64, i: usize) -> __m256d {
    unsafe { _mm256_loadu_pd(a.add(i) as *const f64) }
}

/// # Safety
/// `i + 1` must be in bounds.
#[inline(always)]
unsafe fn store(a: *mut C64, i: usize, v: __m256d) {
    unsafe { _mm256_storeu_pd(a.add(i) as *mut f64, v) }
}

/// The two amplitudes swapped.
#[inline(always)]
fn swap(v: __m256d) -> __m256d {
    unsafe { _mm256_permute4x64_pd(v, 0b01_00_11_10) }
}

/// Real and imaginary parts swapped in each amplitude.
#[inline(always)]
fn flip(v: __m256d) -> __m256d {
    unsafe { _mm256_permute_pd(v, 0b0101) }
}

/// `m v` for the complex `m = re + i im`, given as splats: `(re vr - im vi,
/// re vi + im vr)`.
#[inline(always)]
fn cmul(v: __m256d, re: __m256d, im: __m256d) -> __m256d {
    unsafe { _mm256_fmaddsub_pd(v, re, _mm256_mul_pd(flip(v), im)) }
}

#[inline(always)]
fn splat(x: f64) -> __m256d {
    unsafe { _mm256_set1_pd(x) }
}

#[inline(always)]
fn xor(a: __m256d, b: __m256d) -> __m256d {
    unsafe { _mm256_xor_pd(a, b) }
}

#[inline(always)]
fn fma(a: __m256d, b: __m256d, c: __m256d) -> __m256d {
    unsafe { _mm256_fmadd_pd(a, b, c) }
}

/// `|z & i|` mod 2.
#[inline(always)]
fn parity(z: u64, i: usize) -> i64 {
    ((z & i as u64).count_ones() & 1) as i64
}

/// The mask that negates both amplitudes when `parity` is 1.
#[inline(always)]
fn both(parity: i64) -> __m256d {
    unsafe { _mm256_castsi256_pd(_mm256_set1_epi64x(parity << 63)) }
}

/// The mask that negates the first amplitude when `p0` is 1 and the second
/// when `p1` is.
#[inline(always)]
fn pair(p0: i64, p1: i64) -> __m256d {
    unsafe { _mm256_castsi256_pd(_mm256_set_epi64x(p1 << 63, p1 << 63, p0 << 63, p0 << 63)) }
}

/// `k` with a 0 inserted at bit `j` (`k` itself for `j >= 63`).
#[inline(always)]
fn insert(k: usize, j: u32) -> usize {
    let low = (1usize << j.min(63)) - 1;
    (k & low) | ((k & !low) << 1)
}

/// The sign masks of the registers of a block: entry `h` for the indices
/// `o, o + 1` with `o = insert(2h, j)`. A block of index units from `kb`
/// (a multiple of [`BLOCK`]) covers `insert(kb, j) | o`, so its signs are
/// the entry times the sign of `insert(kb, j)`.
fn table(z: u64, j: u32) -> [__m256d; BLOCK / 2] {
    std::array::from_fn(|h| {
        let o = insert(2 * h, j);
        pair(parity(z, o), parity(z, o ^ 1))
    })
}

/// The part of `<φ|P|φ>` from the loop units in `range` (see
/// [`Dense::expectation`](super::Dense::expectation)).
///
/// `<φ|P|φ> = phase Σ_i sign(i) conj(φ(i ^ x)) φ(i)`. Its imaginary part
/// cancels, and with `σ = sign(x)` only `Re(conj(b) a)` counts when `σ = 1`
/// (the phase is real) and only `Im(conj(b) a)` when `σ = -1`, so one
/// product per register. With `xh = 0` the loop is over every amplitude
/// (units are amplitudes); otherwise over the pairs `i, i ^ x` with bit `J`
/// (the lowest of `xh`) clear (units are pairs), whose two terms are equal,
/// so each counts twice.
///
/// # Safety
/// `amps` must hold `2^q >= 2 BLOCK` amplitudes, and `range` lie within
/// the loop (`2^q` units with `xh = 0`, `2^(q-1)` otherwise).
#[inline(always)]
pub(super) unsafe fn expectation(amps: *const C64, p: Pauli, range: Range<usize>) -> f64 {
    let xh = (p.x & !1) as usize;
    let swapped = p.x & 1 == 1;
    let real = p.sign(p.x as usize) > 0.0;
    let j = if xh == 0 { 63 } else { xh.trailing_zeros() };
    let signs = table(p.z, j);
    // Two accumulators, so consecutive registers don't wait on each other.
    let mut acc = [splat(0.0); 2];
    for kb in range.step_by(BLOCK) {
        let base = insert(kb, j);
        let high = both(parity(p.z, base));
        for h in 0..BLOCK / 2 {
            let i = base | insert(2 * h, j);
            // SAFETY: i + 1 and (i ^ xh) + 1 are indices of the state.
            let (a, b) = unsafe { (load(amps, i), load(amps, i ^ xh)) };
            let a = xor(a, xor(signs[h], high));
            let b = if swapped { swap(b) } else { b };
            let b = if real { b } else { flip(b) };
            acc[h & 1] = fma(a, b, acc[h & 1]);
        }
    }
    let mut lanes = [0.0f64; 4];
    // SAFETY: four f64.
    unsafe { _mm256_storeu_pd(lanes.as_mut_ptr(), _mm256_add_pd(acc[0], acc[1])) };
    let twice = if xh == 0 { 1.0 } else { 2.0 };
    let phase = p.phase();
    if real {
        // Re(conj(b) a) = ar br + ai bi.
        twice * phase.re * lanes.iter().sum::<f64>()
    } else {
        // Im(conj(b) a) = ai br - ar bi, from the lanes (ar bi, ai br).
        -twice * phase.im * (lanes[1] + lanes[3] - lanes[0] - lanes[2])
    }
}

/// `exp(-iθ/2 P)` on the loop units in `range`, with `c = cos(θ/2)` and
/// `m = -i sin(θ/2) phase` (see [`Dense::rotate`](super::Dense::rotate)):
/// the new amplitude at `j` is `c φ(j) + m sign(j ^ x) φ(j ^ x)`, where
/// `sign(j ^ x) = sign(j) sign(x)`. With `xh = 0` the partners are in the
/// same register (units are amplitudes); otherwise each pair of registers
/// at `i` (bit `J` clear) and `i ^ xh` is read and written once (units are
/// pairs of amplitudes).
///
/// # Safety
/// As for [`expectation`], and no other thread may touch the amplitudes
/// that `range` covers.
#[inline(always)]
pub(super) unsafe fn rotate(amps: *mut C64, p: Pauli, c: f64, m: C64, range: Range<usize>) {
    let mx = m * p.sign(p.x as usize);
    let (c, m_re, m_im, mx_re, mx_im) = (
        splat(c),
        splat(m.re),
        splat(m.im),
        splat(mx.re),
        splat(mx.im),
    );
    let xh = (p.x & !1) as usize;
    let swapped = p.x & 1 == 1;
    let j = if xh == 0 { 63 } else { xh.trailing_zeros() };
    let signs = table(p.z, j);
    for kb in range.step_by(BLOCK) {
        let base = insert(kb, j);
        let high = both(parity(p.z, base));
        for (h, &sign) in signs.iter().enumerate() {
            let i = base | insert(2 * h, j);
            let s = xor(sign, high);
            // SAFETY: i + 1 and (i ^ xh) + 1 are indices of the state, and
            // only this call writes them.
            unsafe {
                let a = load(amps, i);
                if xh == 0 {
                    let b = if swapped { swap(a) } else { a };
                    store(amps, i, fma(a, c, cmul(xor(b, s), mx_re, mx_im)));
                } else {
                    let stored = load(amps, i ^ xh);
                    // Aligned with a: lane l holds φ((i + l) ^ x).
                    let b = if swapped { swap(stored) } else { stored };
                    let new_a = fma(a, c, cmul(xor(b, s), mx_re, mx_im));
                    let new_b = fma(b, c, cmul(xor(a, s), m_re, m_im));
                    store(amps, i, new_a);
                    store(amps, i ^ xh, if swapped { swap(new_b) } else { new_b });
                }
            }
        }
    }
}

/// [`Dense::retired`](super::Dense::retired) for a `p` with an X part: the
/// new amplitudes `k` in `range` (a range of the half-size state), `(φ(a0)
/// + m sign(a0 ^ x) φ(a0 ^ x)) scale` with `a0` = `k` with a 0 inserted at
/// `j`, the lowest X position, and `m` = phase times eigenvalue times
/// `scale`.
///
/// For `j >= 1` the outputs `k, k + 1` read the registers at `a0` and
/// `a0 ^ x`. For `j = 0` they read the first amplitudes of the registers
/// at `2k` and `2k + 2`, and their partners are the second amplitudes of
/// those at `q = 2k ^ xh` and `q ^ 2`.
///
/// # Safety
/// `amps` must hold `2^q >= 2 BLOCK` amplitudes and `out` have room for
/// the `2^(q-1)` of the new state, of which no other thread may write those
/// in `range`.
#[inline(always)]
pub(super) unsafe fn retired_x(
    amps: *const C64,
    p: Pauli,
    m: C64,
    scale: f64,
    out: *mut C64,
    range: Range<usize>,
) {
    let x = p.x as usize;
    let xh = x & !1;
    let j = p.x.trailing_zeros();
    let (scale, m_re, m_im) = (splat(scale), splat(m.re), splat(m.im));
    if j >= 1 {
        let signs = table(p.z, j);
        for kb in range.step_by(BLOCK) {
            let base = insert(kb, j);
            let high = both(parity(p.z, base ^ x));
            for (h, &sign) in signs.iter().enumerate() {
                let a0 = base | insert(2 * h, j);
                // SAFETY: a0 + 1 and (a0 ^ x) + 1 are indices of the state,
                // and kb + 2h + 1 of the new one.
                unsafe {
                    let b = xor(load(amps, a0 ^ x), xor(sign, high));
                    store(
                        out,
                        kb + 2 * h,
                        fma(load(amps, a0), scale, cmul(b, m_re, m_im)),
                    );
                }
            }
        }
    } else {
        // The partners' indices are q ^ 1 and q ^ 3 with q = 2kb ^ xh ^ 4h.
        let signs: [__m256d; BLOCK / 2] =
            std::array::from_fn(|h| pair(parity(p.z, (4 * h) ^ 1), parity(p.z, (4 * h) ^ 3)));
        for kb in range.step_by(BLOCK) {
            let q_base = (2 * kb) ^ xh;
            let high = both(parity(p.z, q_base));
            for (h, &sign) in signs.iter().enumerate() {
                let (i, q) = (2 * kb + 4 * h, q_base ^ (4 * h));
                // SAFETY: i + 3, q + 1 and (q ^ 2) + 1 (q is even) are
                // indices of the full state, and kb + 2h + 1 of the new one.
                unsafe {
                    let a = _mm256_permute2f128_pd(load(amps, i), load(amps, i + 2), 0x20);
                    let b = _mm256_permute2f128_pd(load(amps, q), load(amps, q ^ 2), 0x31);
                    let b = xor(b, xor(sign, high));
                    store(out, kb + 2 * h, fma(a, scale, cmul(b, m_re, m_im)));
                }
            }
        }
    }
}
