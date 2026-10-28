//! Pauli propagation against the dense reference simulator, and against the
//! near-Clifford backend's exact expectation values at 127 qubits.

mod common;

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

use qvd::library::{eagle_heavy_hex, kicked_ising, random_clifford, random_gate_circuit};
use qvd::nearclifford::NearClifford;
use qvd::pauliprop::{Observable, PropagationOptions, expectation, parse_pauli};
use qvd::{C64, Circuit, Gate, Instruction, reference};
use rand::{Rng, RngExt};

/// A random Pauli string as `(qubit, 'X' | 'Y' | 'Z')` factors.
fn random_pauli(rng: &mut impl Rng, n: usize) -> Vec<(usize, char)> {
    (0..n)
        .filter_map(|q| match rng.random_range(0..4) {
            0 => None,
            1 => Some((q, 'X')),
            2 => Some((q, 'Y')),
            _ => Some((q, 'Z')),
        })
        .collect()
}

fn random_observable(rng: &mut impl Rng, n: usize, terms: usize) -> Observable {
    (0..terms)
        .map(|_| (rng.random::<f64>() - 0.5, random_pauli(rng, n)))
        .collect()
}

/// `<ψ|O|ψ>` on a dense state.
fn dense_expectation(state: &[C64], observable: &Observable) -> f64 {
    let mut total = 0.0;
    for (coefficient, pauli) in observable {
        let mut out = state.to_vec();
        for &(q, p) in pauli {
            let gate = match p {
                'X' => Gate::X,
                'Y' => Gate::Y,
                _ => Gate::Z,
            };
            reference::apply(&mut out, &gate.matrix(), &[q]);
        }
        total += coefficient
            * state
                .iter()
                .zip(&out)
                .map(|(a, b)| (a.conj() * b).re)
                .sum::<f64>();
    }
    total
}

fn exact() -> PropagationOptions {
    PropagationOptions {
        min_abs_coefficient: 0.0,
        max_terms: None,
    }
}

/// Every gate this backend supports, on random qubits, with angles that
/// include exact and near multiples of π/2 (the angle reduction's cases).
fn every_gate_circuit(rng: &mut impl Rng, n: usize, gates: usize) -> Circuit {
    let mut c = Circuit::new(n);
    let angles = [
        FRAC_PI_2,
        PI,
        -FRAC_PI_2,
        3.0 * FRAC_PI_2,
        FRAC_PI_4,
        1.0,
        -2.5,
        0.0,
    ];
    for _ in 0..gates {
        let mut qubits: Vec<usize> = (0..n).collect();
        for i in 0..3 {
            let j = rng.random_range(i..n);
            qubits.swap(i, j);
        }
        let a = if rng.random::<bool>() {
            angles[rng.random_range(0..angles.len())] + rng.random_range(-1..=1) as f64 * 1e-3
        } else {
            rng.random::<f64>() * 8.0 - 4.0
        };
        let gate = match rng.random_range(0..29) {
            0 => Gate::I,
            1 => Gate::H,
            2 => Gate::X,
            3 => Gate::Y,
            4 => Gate::Z,
            5 => Gate::S,
            6 => Gate::Sdg,
            7 => Gate::T,
            8 => Gate::Tdg,
            9 => Gate::SX,
            10 => Gate::SXdg,
            11 => Gate::RX(a),
            12 => Gate::RY(a),
            13 => Gate::RZ(a),
            14 => Gate::P(a),
            15 => Gate::U(a, 0.7 * a, -1.3 * a),
            16 => Gate::CX,
            17 => Gate::CY,
            18 => Gate::CZ,
            19 => Gate::CH,
            20 => Gate::CP(a),
            21 => Gate::CRX(a),
            22 => Gate::CRY(a),
            23 => Gate::CRZ(a),
            24 => Gate::SWAP,
            25 => Gate::RZZ(a),
            26 => Gate::RXX(a),
            27 => Gate::CCX,
            _ => Gate::CSWAP,
        };
        let arity = gate.arity();
        c.gate(gate, &qubits[..arity]);
    }
    c
}

#[test]
fn exact_propagation_matches_dense_state() {
    let mut rng = common::rng(11);
    for seed in 0..60u64 {
        let n = 3 + (seed as usize % 6);
        let circuit = if seed % 2 == 0 {
            every_gate_circuit(&mut rng, n, 30)
        } else {
            random_gate_circuit(n, 30, seed)
        };
        let dense = reference::statevector(&circuit);
        for terms in [1, 3] {
            let observable = random_observable(&mut rng, n, terms);
            let got = expectation(&circuit, &observable, &exact()).unwrap();
            let want = dense_expectation(&dense, &observable);
            assert!(
                (got.value - want).abs() < 1e-9,
                "seed {seed}, {observable:?}: {} vs {want}",
                got.value
            );
            assert_eq!(got.truncated_one_norm, 0.0);
        }
    }
}

#[test]
fn clifford_circuits_need_no_rotations() {
    let mut rng = common::rng(5);
    for seed in 0..30u64 {
        let n = 2 + (seed as usize % 7);
        let circuit = random_clifford(n, 12, seed);
        let dense = reference::statevector(&circuit);
        for _ in 0..10 {
            let observable = random_observable(&mut rng, n, 1);
            let got = expectation(&circuit, &observable, &exact()).unwrap();
            assert!((got.value - dense_expectation(&dense, &observable)).abs() < 1e-9);
            assert_eq!(got.rotations, 0);
            assert_eq!(got.final_terms, 1);
        }
    }
}

/// Strings past the first 64 qubits use more words; spread a small
/// circuit's qubits over 300 and compare with the compact one.
#[test]
fn wide_strings_match_dense_state() {
    let mut rng = common::rng(17);
    for (seed, n) in [(0u64, 70usize), (1, 130), (2, 300), (3, 1024)] {
        let compact = every_gate_circuit(&mut rng, 6, 40);
        let mut map: Vec<usize> = (0..n).collect();
        for i in 0..6 {
            let j = rng.random_range(i..n);
            map.swap(i, j);
        }
        let mut wide = Circuit::new(n);
        for instruction in &compact.instructions {
            if let Instruction::Gate { gate, qubits } = instruction {
                let qubits: Vec<usize> = qubits.iter().map(|&q| map[q]).collect();
                wide.gate(gate.clone(), &qubits);
            }
        }
        let dense = reference::statevector(&compact);
        for _ in 0..5 {
            let observable = random_observable(&mut rng, 6, 2);
            let wide_observable: Observable = observable
                .iter()
                .map(|(c, p)| (*c, p.iter().map(|&(q, f)| (map[q], f)).collect()))
                .collect();
            let got = expectation(&wide, &wide_observable, &exact()).unwrap();
            let want = dense_expectation(&dense, &observable);
            assert!(
                (got.value - want).abs() < 1e-9,
                "seed {seed}, n {n}: {} vs {want}",
                got.value
            );
        }
    }
}

/// The reported one-norm bounds the error of a truncated run.
#[test]
fn truncation_error_is_bounded_by_the_dropped_one_norm() {
    let edges: Vec<(usize, usize)> = (0..11).map(|q| (q, q + 1)).collect();
    for theta_h in [0.3, 0.7, 1.0] {
        let circuit = kicked_ising(12, &edges, 6, theta_h);
        let dense = reference::statevector(&circuit);
        let observable: Observable = vec![(1.0, vec![(5, 'Z')]), (0.5, vec![(2, 'X'), (3, 'X')])];
        let want = dense_expectation(&dense, &observable);
        for options in [
            PropagationOptions {
                min_abs_coefficient: 1e-2,
                max_terms: None,
            },
            PropagationOptions {
                min_abs_coefficient: 1e-4,
                max_terms: None,
            },
            PropagationOptions {
                min_abs_coefficient: 0.0,
                max_terms: Some(50),
            },
        ] {
            let got = expectation(&circuit, &observable, &options).unwrap();
            let error = (got.value - want).abs();
            assert!(
                error <= got.truncated_one_norm + 1e-12,
                "θ_h {theta_h}, {options:?}: error {error} > bound {}",
                got.truncated_one_norm
            );
            if let Some(max) = options.max_terms {
                assert!(got.peak_terms <= max.max(observable.len()));
            }
        }
        // A finer threshold converges.
        let fine = PropagationOptions {
            min_abs_coefficient: 1e-7,
            max_terms: None,
        };
        assert!((expectation(&circuit, &observable, &fine).unwrap().value - want).abs() < 1e-4);
    }
}

#[test]
fn eagle_lattice_is_heavy_hex() {
    let edges = eagle_heavy_hex();
    assert_eq!(edges.len(), 144);
    let mut degree = [0usize; 127];
    let mut neighbours = vec![Vec::new(); 127];
    for &(a, b) in &edges {
        assert!(a < 127 && b < 127 && a != b);
        degree[a] += 1;
        degree[b] += 1;
        neighbours[a].push(b);
        neighbours[b].push(a);
    }
    assert!(degree.iter().all(|&d| (1..=3).contains(&d)));
    let mut sorted: Vec<(usize, usize)> =
        edges.iter().map(|&(a, b)| (a.min(b), a.max(b))).collect();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), 144);
    // Known couplers of ibm_kyiv.
    for edge in [
        (0, 14),
        (14, 18),
        (4, 15),
        (15, 22),
        (62, 72),
        (72, 81),
        (96, 109),
        (109, 114),
        (112, 126),
    ] {
        assert!(sorted.contains(&edge), "{edge:?}");
    }
    neighbours[62].sort();
    assert_eq!(neighbours[62], vec![61, 63, 72]);
    // Connected.
    let mut seen = [false; 127];
    let mut stack = vec![0];
    while let Some(q) = stack.pop() {
        if !std::mem::replace(&mut seen[q], true) {
            stack.extend(&neighbours[q]);
        }
    }
    assert!(seen.iter().all(|&s| s));
}

/// At the Clifford points of the 127-qubit kicked Ising circuit the answer
/// is known exactly from the near-Clifford backend (no active qubits).
#[test]
fn utility_circuit_at_clifford_points() {
    let edges = eagle_heavy_hex();
    let mut rng = common::rng(23);
    for theta_h in [0.0, FRAC_PI_2, PI] {
        let circuit = kicked_ising(127, &edges, 20, theta_h);
        let mut state = NearClifford::new(127);
        for instruction in &circuit.instructions {
            if let Instruction::Gate { gate, qubits } = instruction {
                state.apply(gate, qubits).unwrap();
            }
        }
        for _ in 0..20 {
            let pauli = random_pauli(&mut rng, 127);
            let got = expectation(&circuit, &vec![(1.0, pauli.clone())], &exact()).unwrap();
            assert_eq!(got.rotations, 0);
            assert!((got.value - state.expectation(&pauli)).abs() < 1e-12);
        }
        let z62 = expectation(
            &circuit,
            &vec![(1.0, parse_pauli("Z62").unwrap())],
            &exact(),
        )
        .unwrap();
        assert!((z62.value - state.expectation(&[(62, 'Z')])).abs() < 1e-12);
    }
}

#[test]
fn rejects_what_it_cannot_propagate() {
    let mut c = Circuit::new(2);
    c.h(0).measure(0, 0);
    assert!(expectation(&c, &vec![(1.0, vec![(0, 'Z')])], &exact()).is_err());
    let mut c = Circuit::new(2);
    c.unitary(qvd::Matrix::identity(2), &[0, 1]);
    assert!(expectation(&c, &vec![(1.0, vec![(0, 'Z')])], &exact()).is_err());
    let c = Circuit::new(2);
    assert!(expectation(&c, &vec![(1.0, vec![(2, 'Z')])], &exact()).is_err());
    assert!(expectation(&Circuit::new(1100), &vec![(1.0, vec![(0, 'Z')])], &exact()).is_err());
}

/// Rotations by multiples of π/2 that only appear after lowering (here the
/// halves of `CP(π)` and `CRX(π)`) are Clifford and never split strings.
#[test]
fn quarter_turn_rotations_join_the_frame() {
    let mut c = Circuit::new(3);
    c.h(0)
        .h(1)
        .cp(PI, 0, 1)
        .gate(Gate::CRX(PI), &[1, 2])
        .rx(-FRAC_PI_2, 2);
    let dense = reference::statevector(&c);
    let mut rng = common::rng(29);
    for _ in 0..20 {
        let observable = random_observable(&mut rng, 3, 1);
        let got = expectation(&c, &observable, &exact()).unwrap();
        assert_eq!(got.rotations, 0);
        assert!((got.value - dense_expectation(&dense, &observable)).abs() < 1e-12);
    }
}
