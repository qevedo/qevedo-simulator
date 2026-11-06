#![allow(dead_code)]

use qvd::{C64, Circuit, Gate, Instruction, Matrix, Noise, reference};
use rand::{Rng, RngExt, SeedableRng};
use rand_pcg::Pcg64;

pub fn rng(seed: u64) -> Pcg64 {
    Pcg64::seed_from_u64(seed)
}

/// A Haar-ish random unitary: Gram-Schmidt on a random complex matrix.
pub fn random_unitary(rng: &mut impl Rng, qubits: usize) -> Matrix {
    let dim = 1 << qubits;
    let mut columns: Vec<Vec<C64>> = Vec::with_capacity(dim);
    while columns.len() < dim {
        let mut v: Vec<C64> = (0..dim)
            .map(|_| C64::new(rng.random::<f64>() - 0.5, rng.random::<f64>() - 0.5))
            .collect();
        for u in &columns {
            let dot: C64 = u.iter().zip(&v).map(|(a, b)| a.conj() * b).sum();
            for (x, y) in v.iter_mut().zip(u) {
                *x -= dot * y;
            }
        }
        let norm = v.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        if norm > 1e-6 {
            columns.push(v.into_iter().map(|x| x / norm).collect());
        }
    }
    let mut entries = vec![C64::new(0.0, 0.0); dim * dim];
    for (c, column) in columns.iter().enumerate() {
        for (r, &x) in column.iter().enumerate() {
            entries[r * dim + c] = x;
        }
    }
    Matrix::new(entries)
}

pub fn random_diagonal(rng: &mut impl Rng, qubits: usize) -> Matrix {
    let entries: Vec<C64> = (0..1 << qubits)
        .map(|_| C64::from_polar(1.0, rng.random::<f64>() * std::f64::consts::TAU))
        .collect();
    Matrix::diagonal(&entries)
}

pub fn random_state(rng: &mut impl Rng, qubits: usize) -> Vec<C64> {
    let v: Vec<C64> = (0..1 << qubits)
        .map(|_| C64::new(rng.random::<f64>() - 0.5, rng.random::<f64>() - 0.5))
        .collect();
    let norm = v.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
    v.into_iter().map(|x| x / norm).collect()
}

/// `k` distinct random qubits out of `n`, in random order.
pub fn random_qubits(rng: &mut impl Rng, n: usize, k: usize) -> Vec<usize> {
    let mut all: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = rng.random_range(i..n);
        all.swap(i, j);
    }
    all.truncate(k);
    all
}

pub fn max_distance(a: &[C64], b: &[C64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).norm())
        .fold(0.0, f64::max)
}

/// Project `state` onto `qubit = outcome` and renormalise; returns the
/// probability of the outcome.
pub fn project(state: &mut [C64], qubit: usize, outcome: bool) -> f64 {
    let p: f64 = state
        .iter()
        .enumerate()
        .filter(|(i, _)| ((i >> qubit) & 1 == 1) == outcome)
        .map(|(_, a)| a.norm_sqr())
        .sum();
    for (i, a) in state.iter_mut().enumerate() {
        *a = if ((i >> qubit) & 1 == 1) == outcome {
            *a / p.sqrt()
        } else {
            C64::new(0.0, 0.0)
        };
    }
    p
}

/// The Pauli errors a noise channel applies to one target, with their
/// probabilities (the identity first).
fn noise_branches(channel: Noise, target: &[usize]) -> Vec<(f64, Vec<(Gate, usize)>)> {
    let paulis = [None, Some(Gate::X), Some(Gate::Z), Some(Gate::Y)];
    match channel {
        Noise::Pauli1 { px, py, pz } => vec![
            (1.0 - px - py - pz, vec![]),
            (px, vec![(Gate::X, target[0])]),
            (py, vec![(Gate::Y, target[0])]),
            (pz, vec![(Gate::Z, target[0])]),
        ],
        Noise::Depolarize2(p) => (0..16)
            .map(|k| {
                let errors: Vec<(Gate, usize)> = [(k & 3, target[0]), (k >> 2, target[1])]
                    .into_iter()
                    .filter_map(|(i, q)| paulis[i].clone().map(|g| (g, q)))
                    .collect();
                (if k == 0 { 1.0 - p } else { p / 15.0 }, errors)
            })
            .collect(),
        Noise::Pauli2(p) => {
            // Stim's order: index k + 1 has the first qubit's letter in
            // its high base-4 digit (I, X, Y, Z = 0..3).
            let letters = [None, Some(Gate::X), Some(Gate::Y), Some(Gate::Z)];
            let mut branches = vec![(1.0 - p.iter().sum::<f64>(), vec![])];
            for (k, &q) in p.iter().enumerate() {
                let errors = [((k + 1) / 4, target[0]), ((k + 1) % 4, target[1])]
                    .into_iter()
                    .filter_map(|(i, t)| letters[i].clone().map(|g| (g, t)))
                    .collect();
                branches.push((q, errors));
            }
            branches
        }
        Noise::Correlated(products) => {
            let none = 1.0 - products.iter().map(|(p, _)| p).sum::<f64>();
            let mut branches = vec![(none, vec![])];
            for (p, paulis) in products {
                let errors = target
                    .iter()
                    .zip(paulis.chars())
                    .filter_map(|(&q, c)| match c {
                        'X' => Some((Gate::X, q)),
                        'Y' => Some((Gate::Y, q)),
                        'Z' => Some((Gate::Z, q)),
                        _ => None,
                    })
                    .collect();
                branches.push((p, errors));
            }
            branches
        }
    }
}

/// The exact distribution of the classical bits, branching on every
/// measurement and reset of a dense simulation, and on every error of
/// every noise channel.
pub fn exact_distribution(circuit: &Circuit) -> std::collections::BTreeMap<String, f64> {
    fn walk(
        circuit: &Circuit,
        from: usize,
        mut state: Vec<C64>,
        clbits: Vec<bool>,
        weight: f64,
        out: &mut std::collections::BTreeMap<String, f64>,
    ) {
        for (i, instruction) in circuit.instructions.iter().enumerate().skip(from) {
            let (qubit, clbit) = match instruction {
                Instruction::Gate { gate, qubits } => {
                    reference::apply(&mut state, &gate.matrix(), qubits);
                    continue;
                }
                Instruction::Barrier { .. } => continue,
                Instruction::Noise { channel, qubits } => {
                    // One target per instruction (split below).
                    for (p, errors) in noise_branches(channel.clone(), qubits) {
                        if p <= 0.0 {
                            continue;
                        }
                        let mut branch = state.clone();
                        for (gate, q) in errors {
                            reference::apply(&mut branch, &gate.matrix(), &[q]);
                        }
                        walk(circuit, i + 1, branch, clbits.clone(), weight * p, out);
                    }
                    return;
                }
                Instruction::Measure { qubit, clbit } => (*qubit, Some(*clbit)),
                Instruction::Reset { qubit } => (*qubit, None),
            };
            // Branch on the outcome.
            for outcome in [false, true] {
                let mut branch = state.clone();
                let p = project(&mut branch, qubit, outcome);
                if p < 1e-12 {
                    continue;
                }
                let mut bits = clbits.clone();
                match clbit {
                    Some(c) => bits[c] = outcome,
                    None if outcome => reference::apply(&mut branch, &Gate::X.matrix(), &[qubit]),
                    None => {}
                }
                walk(circuit, i + 1, branch, bits, weight * p, out);
            }
            return;
        }
        let key: String = clbits
            .iter()
            .rev()
            .map(|&b| if b { '1' } else { '0' })
            .collect();
        *out.entry(key).or_default() += weight;
    }
    // Noise channels with several targets act on each independently.
    let mut split = circuit.clone();
    split.instructions = circuit
        .instructions
        .iter()
        .flat_map(|instruction| match instruction {
            Instruction::Noise { channel, qubits } => qubits
                .chunks(channel.arity())
                .map(|target| Instruction::Noise {
                    channel: channel.clone(),
                    qubits: target.to_vec(),
                })
                .collect(),
            other => vec![other.clone()],
        })
        .collect();
    let circuit = &split;
    let mut state = vec![C64::new(0.0, 0.0); 1 << circuit.num_qubits];
    state[0] = C64::new(1.0, 0.0);
    let mut out = std::collections::BTreeMap::new();
    walk(
        circuit,
        0,
        state,
        vec![false; circuit.num_clbits],
        1.0,
        &mut out,
    );
    out
}

/// A random noise channel: one of the Pauli channels, `Depolarize2`,
/// `Pauli2` or a correlated error, with probabilities up to 0.4 and
/// sometimes exactly 0 or 1.
pub fn random_noise(rng: &mut impl Rng, n: usize) -> (Noise, Vec<usize>) {
    let mut p = || match rng.random_range(0..8) {
        0 => 0.0,
        1 => 1.0,
        _ => rng.random::<f64>() * 0.4,
    };
    let (a, b, c) = (p(), p(), p());
    let channel = match rng.random_range(0..8) {
        0 => Noise::Pauli1 {
            px: a,
            py: 0.0,
            pz: 0.0,
        },
        1 => Noise::Pauli1 {
            px: 0.0,
            py: 0.0,
            pz: a,
        },
        2 => Noise::Pauli1 {
            px: a / 3.0,
            py: a / 3.0,
            pz: a / 3.0,
        },
        3 => {
            let total = (a + b + c).max(1.0);
            Noise::Pauli1 {
                px: a / total,
                py: b / total,
                pz: c / total,
            }
        }
        4 | 5 if n >= 2 => Noise::Depolarize2(a),
        6 if n >= 2 => {
            let mut weights = [0.0; 15];
            for w in weights.iter_mut() {
                *w = rng.random::<f64>();
            }
            // Total probability a, spread unevenly over the 15 Paulis.
            let sum: f64 = weights.iter().sum();
            Noise::Pauli2(Box::new(weights.map(|w| w / sum * a)))
        }
        7 => {
            // One to three products on up to three qubits, scaled so
            // their probabilities sum to at most 1.
            let k = rng.random_range(1..=n.min(3));
            let mut products = Vec::new();
            for _ in 0..rng.random_range(1..4) {
                let mut paulis = String::new();
                for _ in 0..k {
                    paulis.push(['I', 'X', 'Y', 'Z'][rng.random_range(0..4)]);
                }
                products.push((rng.random::<f64>() * 0.3, paulis));
            }
            Noise::Correlated(products)
        }
        _ => Noise::Pauli1 {
            px: 0.0,
            py: a,
            pz: 0.0,
        },
    };
    let qubits = random_qubits(rng, n, channel.arity());
    (channel, qubits)
}
