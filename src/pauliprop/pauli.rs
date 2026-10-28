//! Pauli strings as bit words, and a Clifford frame stored as the images of
//! the single-qubit Paulis.

use std::hash::{Hash, Hasher};

use crate::stabilizer::Clifford;

/// A Hermitian Pauli string on up to `64 W` qubits: `P(x, z) = i^{x·z}
/// X^x Z^z`, so a qubit with both bits set holds `Y = iXZ`. Hermitian
/// strings keep the coefficients of a propagated observable real.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pauli<const W: usize> {
    pub x: [u64; W],
    pub z: [u64; W],
}

// Word by word: the derived impl would hash the arrays as byte slices with
// a length prefix.
impl<const W: usize> Hash for Pauli<W> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for w in self.x.iter().chain(&self.z) {
            state.write_u64(*w);
        }
    }
}

/// A Pauli string with a sign: `-P` when `negative`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Signed<const W: usize> {
    pub negative: bool,
    pub pauli: Pauli<W>,
}

fn popcount<const W: usize>(a: &[u64; W], b: &[u64; W]) -> u32 {
    a.iter().zip(b).map(|(a, b)| (a & b).count_ones()).sum()
}

impl<const W: usize> Pauli<W> {
    pub const IDENTITY: Pauli<W> = Pauli {
        x: [0; W],
        z: [0; W],
    };

    /// The string with these `(qubit, 'X' | 'Y' | 'Z')` factors (`'I'` is
    /// skipped). Qubits must fit in `64 W` bits.
    pub fn from_factors(factors: &[(usize, char)]) -> Result<Pauli<W>, String> {
        let mut p = Pauli::IDENTITY;
        for &(q, c) in factors {
            if q >= 64 * W {
                return Err(format!("qubit {q} is out of range"));
            }
            let (w, b) = (q / 64, 1u64 << (q % 64));
            if (p.x[w] | p.z[w]) & b != 0 {
                return Err(format!("qubit {q} appears twice in a Pauli string"));
            }
            match c.to_ascii_uppercase() {
                'I' => {}
                'X' => p.x[w] |= b,
                'Z' => p.z[w] |= b,
                'Y' => {
                    p.x[w] |= b;
                    p.z[w] |= b;
                }
                other => return Err(format!("{other:?} is not a Pauli")),
            }
        }
        Ok(p)
    }

    pub fn single(q: usize, x: bool, z: bool) -> Pauli<W> {
        let mut p = Pauli::IDENTITY;
        p.x[q / 64] |= (x as u64) << (q % 64);
        p.z[q / 64] |= (z as u64) << (q % 64);
        p
    }

    pub fn anticommutes(&self, other: &Pauli<W>) -> bool {
        let mut parity = 0u64;
        for w in 0..W {
            parity ^= (self.x[w] & other.z[w]) ^ (self.z[w] & other.x[w]);
        }
        parity.count_ones() & 1 == 1
    }

    /// Whether the string has an X or Y factor anywhere (its expectation on
    /// a computational basis state is then zero).
    pub fn has_x(&self) -> bool {
        self.x.iter().any(|&w| w != 0)
    }

    /// The number of non-identity factors.
    pub fn weight(&self) -> u32 {
        self.x
            .iter()
            .zip(&self.z)
            .map(|(x, z)| (x | z).count_ones())
            .sum()
    }

    /// `self · other = i^k P(x ⊕ x', z ⊕ z')`, returned as `(k mod 4, P)`.
    ///
    /// With `P(x, z) = i^{x·z} X^x Z^z` and `Z^z X^x = (-1)^{z·x} X^x Z^z`:
    /// `P(a) P(b) = i^{a_x·a_z + b_x·b_z + 2 a_z·b_x - c_x·c_z} P(c)`, the
    /// dot products counted as integers. Four popcounts per word.
    pub fn mul(&self, other: &Pauli<W>) -> (u32, Pauli<W>) {
        let mut c = Pauli::IDENTITY;
        for w in 0..W {
            c.x[w] = self.x[w] ^ other.x[w];
            c.z[w] = self.z[w] ^ other.z[w];
        }
        let k = popcount(&self.x, &self.z)
            + popcount(&other.x, &other.z)
            + 2 * popcount(&self.z, &other.x)
            + 3 * popcount(&c.x, &c.z);
        (k % 4, c)
    }
}

impl<const W: usize> Signed<W> {
    fn new(negative: bool, pauli: Pauli<W>) -> Signed<W> {
        Signed { negative, pauli }
    }

    /// `i^extra · self · other`, which must be Hermitian (so `extra` plus
    /// the product's phase is even): for commuting factors with `extra =
    /// 0`, or anticommuting ones with `extra = ±1`.
    fn times(&self, other: &Signed<W>, extra: u32) -> Signed<W> {
        let (k, pauli) = self.pauli.mul(&other.pauli);
        let k = (k + extra + 2 * (self.negative ^ other.negative) as u32) % 4;
        debug_assert!(k.is_multiple_of(2), "product of Paulis is not Hermitian");
        Signed::new(k == 2, pauli)
    }
}

/// A Clifford unitary `K`, stored as the images `K X_q K†` and `K Z_q K†`
/// of the single-qubit Paulis: `2n` signed strings, `O(n^2)` bits.
#[derive(Clone, Debug)]
pub struct Frame<const W: usize> {
    xs: Vec<Signed<W>>,
    zs: Vec<Signed<W>>,
}

impl<const W: usize> Frame<W> {
    pub fn identity(n: usize) -> Frame<W> {
        Frame {
            xs: (0..n)
                .map(|q| Signed::new(false, Pauli::single(q, true, false)))
                .collect(),
            zs: (0..n)
                .map(|q| Signed::new(false, Pauli::single(q, false, true)))
                .collect(),
        }
    }

    /// `K P K†`: the product of the images of `P`'s factors, with
    /// `P = i^{x·z} X^x Z^z`.
    pub fn conjugate(&self, p: &Pauli<W>) -> Signed<W> {
        let mut phase = popcount(&p.x, &p.z);
        let mut acc = Pauli::IDENTITY;
        let mut factor = |row: &Signed<W>| {
            let (k, product) = acc.mul(&row.pauli);
            phase += k + 2 * row.negative as u32;
            acc = product;
        };
        for (w, &word) in p.x.iter().enumerate() {
            for_each_bit(word, |b| factor(&self.xs[64 * w + b]));
        }
        for (w, &word) in p.z.iter().enumerate() {
            for_each_bit(word, |b| factor(&self.zs[64 * w + b]));
        }
        debug_assert!(phase.is_multiple_of(2), "conjugated Pauli is not Hermitian");
        Signed::new(phase % 4 == 2, acc)
    }

    /// `K -> K g`: the frame of a gate applied before `K`. The new images
    /// are `K (g P g†) K†`, products of the old images.
    pub fn apply_before(&mut self, g: Clifford) {
        let (xs, zs) = (&mut self.xs, &mut self.zs);
        match g {
            // H X H = Z.
            Clifford::H(q) => std::mem::swap(&mut xs[q], &mut zs[q]),
            // S X S† = Y = i X Z; S† X S = -Y.
            Clifford::S(q) => xs[q] = xs[q].times(&zs[q], 1),
            Clifford::Sdg(q) => xs[q] = xs[q].times(&zs[q], 3),
            // A Pauli flips the signs of the Paulis it anticommutes with.
            Clifford::X(q) => zs[q].negative ^= true,
            Clifford::Z(q) => xs[q].negative ^= true,
            Clifford::Y(q) => {
                xs[q].negative ^= true;
                zs[q].negative ^= true;
            }
            // CX: X_c -> X_c X_t, Z_t -> Z_c Z_t.
            Clifford::CX(c, t) => {
                xs[c] = xs[c].times(&xs[t], 0);
                zs[t] = zs[c].times(&zs[t], 0);
            }
            // CZ: X_a -> X_a Z_b, X_b -> Z_a X_b.
            Clifford::CZ(a, b) => {
                xs[a] = xs[a].times(&zs[b], 0);
                xs[b] = zs[a].times(&xs[b], 0);
            }
            Clifford::Swap(a, b) => {
                xs.swap(a, b);
                zs.swap(a, b);
            }
        }
    }

    /// `K -> V K` for `V = exp(-i k π/4 Q)`, a rotation by `k` quarter
    /// turns applied after `K`. Images that commute with `Q` stay; the
    /// others become `V² R = exp(-i k π/2 Q) R`: `-iQR`, `-R` or `iQR`.
    pub fn rotate_after(&mut self, q: &Pauli<W>, quarter_turns: u32) {
        let q = Signed::new(false, *q);
        for row in self.xs.iter_mut().chain(self.zs.iter_mut()) {
            if !row.pauli.anticommutes(&q.pauli) {
                continue;
            }
            match quarter_turns % 4 {
                1 => *row = q.times(row, 3),
                2 => row.negative ^= true,
                3 => *row = q.times(row, 1),
                _ => {}
            }
        }
    }

    /// `<0...0| K† P K |0...0>`, which is 0 or ±1.
    ///
    /// `K† P K` has an X part on qubit `q` exactly when `P` anticommutes
    /// with the stabilizer `K Z_q K†`; with none it is `±Z^z`, and `P` is
    /// that sign times the product of the stabilizers in `z`.
    pub fn expectation(&self, p: &Pauli<W>) -> f64 {
        if self.zs.iter().any(|s| s.pauli.anticommutes(p)) {
            return 0.0;
        }
        let mut product = Signed::new(false, Pauli::IDENTITY);
        for (q, x) in self.xs.iter().enumerate() {
            // Z_q appears in K† P K when P anticommutes with K X_q K†.
            if x.pauli.anticommutes(p) {
                product = product.times(&self.zs[q], 0);
            }
        }
        debug_assert_eq!(product.pauli, *p);
        if product.negative { -1.0 } else { 1.0 }
    }
}

fn for_each_bit(mut word: u64, mut f: impl FnMut(usize)) {
    while word != 0 {
        f(word.trailing_zeros() as usize);
        word &= word - 1;
    }
}
