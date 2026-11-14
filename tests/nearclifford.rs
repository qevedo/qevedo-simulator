//! The near-Clifford backend against the dense reference simulator.

mod common;

use common::{exact_distribution, project};
use qvd::library::{ghz, random_gate_circuit};
use qvd::nearclifford::{self, NearClifford};
use qvd::{Backend, C64, Circuit, Gate, Instruction, Options, reference};
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

/// `<ψ|P|ψ>` on a dense state.
fn dense_expectation(state: &[C64], pauli: &[(usize, char)]) -> f64 {
    let mut out = state.to_vec();
    for &(q, p) in pauli {
        let gate = match p {
            'X' => Gate::X,
            'Y' => Gate::Y,
            _ => Gate::Z,
        };
        reference::apply(&mut out, &gate.matrix(), &[q]);
    }
    state.iter().zip(&out).map(|(a, b)| (a.conj() * b).re).sum()
}

fn simulate(circuit: &Circuit) -> NearClifford {
    let mut state = NearClifford::new(circuit.num_qubits);
    for instruction in &circuit.instructions {
        if let Instruction::Gate { gate, qubits } = instruction {
            state.apply(gate, qubits).unwrap();
        }
    }
    state
}

#[test]
fn expectations_match_dense_state() {
    let mut rng = common::rng(3);
    for seed in 0..80u64 {
        let n = 1 + (seed as usize % 8);
        let circuit = random_gate_circuit(n, 40, seed);
        let state = simulate(&circuit);
        let dense = reference::statevector(&circuit);
        for _ in 0..20 {
            let pauli = random_pauli(&mut rng, n);
            let (got, expected) = (state.expectation(&pauli), dense_expectation(&dense, &pauli));
            assert!(
                (got - expected).abs() < 1e-9,
                "seed {seed}, {pauli:?}: {got} vs {expected}"
            );
        }
        assert!(state.peak_active_qubits() <= n);
    }
}

#[test]
fn single_qubit_unitaries_decompose() {
    let mut rng = common::rng(4);
    for seed in 0..20 {
        let mut circuit = Circuit::new(2);
        circuit.gate(Gate::H, &[0]).cx(0, 1);
        circuit.unitary(common::random_unitary(&mut rng, 1), &[seed % 2]);
        circuit.cx(1, 0);
        let state = simulate(&circuit);
        let dense = reference::statevector(&circuit);
        for _ in 0..10 {
            let pauli = random_pauli(&mut rng, 2);
            assert!((state.expectation(&pauli) - dense_expectation(&dense, &pauli)).abs() < 1e-9);
        }
    }
}

#[test]
fn measurement_matches_dense_projection() {
    let mut rng = common::rng(5);
    for seed in 0..60u64 {
        let n = 2 + (seed as usize % 7);
        let circuit = random_gate_circuit(n, 40, seed);
        let mut state = simulate(&circuit);
        let mut dense = reference::statevector(&circuit);
        for round in 0..n {
            let q = rng.random_range(0..n);
            let outcome = state.measure(q, &mut rng);
            let p = project(&mut dense, q, outcome);
            assert!(
                p > 1e-9,
                "seed {seed} round {round}: outcome {outcome} of q{q} has probability {p}"
            );
            for _ in 0..10 {
                let pauli = random_pauli(&mut rng, n);
                let (got, expected) =
                    (state.expectation(&pauli), dense_expectation(&dense, &pauli));
                assert!(
                    (got - expected).abs() < 1e-9,
                    "seed {seed} round {round}, {pauli:?}: {got} vs {expected}"
                );
            }
        }
        // Measuring every qubit leaves nothing active.
        for q in 0..n {
            state.measure(q, &mut rng);
        }
        assert_eq!(state.active_qubits(), 0, "seed {seed}");
    }
}

#[test]
fn sampling_matches_exact_distribution() {
    for case in 0..30u64 {
        let n = 2 + (case as usize % 5);
        let mut circuit = random_gate_circuit(n, 25, case);
        if case % 2 == 1 {
            circuit.measure(0, 0);
            circuit.reset(1);
            circuit
                .instructions
                .extend(random_gate_circuit(n, 10, case + 100).instructions);
        }
        for q in 0..n {
            circuit.measure(q, q + 1);
        }
        let shots = 40_000;
        let options = Options {
            backend: Backend::NearClifford,
            seed: Some(case),
            ..Options::default()
        };
        let result = qvd::run::<f64>(&circuit, shots, &options).unwrap();
        let exact = exact_distribution(&circuit);
        for key in result.counts.keys() {
            assert!(
                exact.contains_key(key),
                "case {case}: impossible outcome {key}"
            );
        }
        for (key, &p) in &exact {
            let observed = *result.counts.get(key).unwrap_or(&0) as f64 / shots as f64;
            let sigma = (p * (1.0 - p) / shots as f64).sqrt();
            // A few counts of slack: an outcome of probability 1e-6 seen once
            // in 40,000 shots is 6 sigma off but not evidence of an error.
            let slack = 3.0 / shots as f64;
            assert!(
                (observed - p).abs() <= 5.0 * sigma + slack,
                "case {case} {key}: observed {observed}, exact {p}"
            );
        }
    }
}

#[test]
fn clifford_circuits_stay_inactive() {
    let mut circuit = ghz(50);
    circuit
        .gate(Gate::S, &[3])
        .gate(Gate::SX, &[7])
        .gate(Gate::RZ(std::f64::consts::FRAC_PI_2), &[9]);
    let state = simulate(&circuit);
    assert_eq!(state.peak_active_qubits(), 0);
    assert_eq!(nearclifford::rotation_count(&circuit), 0);
}

#[test]
fn large_ghz_with_a_t_gate() {
    // (|0...0> + e^{iπ/4} |1...1>)/√2 on 200 qubits: <X...X> = cos(π/4),
    // <Y X...X> = sin(π/4), and only one qubit is ever active.
    let n = 200;
    let mut circuit = ghz(n);
    circuit.gate(Gate::T, &[0]);
    let state = simulate(&circuit);
    assert_eq!(state.peak_active_qubits(), 1);
    let xs: Vec<(usize, char)> = (0..n).map(|q| (q, 'X')).collect();
    let mut yxs = xs.clone();
    yxs[0] = (0, 'Y');
    let r = std::f64::consts::FRAC_1_SQRT_2;
    assert!((state.expectation(&xs) - r).abs() < 1e-12);
    assert!((state.expectation(&yxs) - r).abs() < 1e-12);
    // Sampling: all zeros or all ones, half and half.
    circuit.measure_all();
    let options = Options {
        backend: Backend::NearClifford,
        seed: Some(1),
        ..Options::default()
    };
    let result = qvd::run::<f64>(&circuit, 2000, &options).unwrap();
    assert_eq!(result.counts.len(), 2);
    let zeros = result.counts.get(&"0".repeat(n)).copied().unwrap_or(0);
    assert!((850..1150).contains(&zeros), "{zeros}");
    assert_eq!(result.stats.peak_active_qubits, Some(1));
}

#[test]
fn sampling_is_reproducible_across_thread_counts() {
    let mut circuit = random_gate_circuit(10, 60, 9);
    circuit.measure_all();
    let options = Options {
        backend: Backend::NearClifford,
        seed: Some(2),
        ..Options::default()
    };
    let sample = |threads| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| qvd::run::<f64>(&circuit, 2000, &options))
            .unwrap()
            .counts
    };
    assert_eq!(sample(1), sample(4));
}

#[test]
fn symbolic_peak_matches_and_auto_selects() {
    for seed in 0..10 {
        let mut circuit = qvd::library::random_clifford_t(12, 8, 6, seed);
        circuit.measure(3, 0);
        circuit
            .instructions
            .extend(qvd::library::random_clifford_t(12, 4, 3, seed + 50).instructions);
        let predicted = nearclifford::peak_active_qubits(&circuit).unwrap();
        let mut state = NearClifford::new(12);
        let mut rng = common::rng(seed);
        for instruction in &circuit.instructions {
            match instruction {
                Instruction::Gate { gate, qubits } => state.apply(gate, qubits).unwrap(),
                Instruction::Measure { qubit, .. } => {
                    state.measure(*qubit, &mut rng);
                }
                _ => {}
            }
        }
        assert_eq!(state.peak_active_qubits(), predicted, "seed {seed}");
        assert!(predicted <= 9);
    }
    // 80 qubits do not fit a state vector; 10 T gates keep at most 10 active.
    let mut circuit = qvd::library::random_clifford_t(80, 10, 10, 1);
    circuit.measure_all();
    let result = qvd::run::<f64>(
        &circuit,
        100,
        &Options {
            seed: Some(1),
            ..Options::default()
        },
    )
    .unwrap();
    let peak = result
        .stats
        .peak_active_qubits
        .expect("Auto should choose the near-Clifford backend");
    assert!(peak <= 10);
}

#[test]
fn auto_prefers_near_clifford_when_much_cheaper() {
    // 24 qubits fit a state vector, but with 6 T gates at most 6 qubits are
    // ever active: far less work.
    let mut circuit = qvd::library::random_clifford_t(24, 10, 6, 3);
    circuit.measure_all();
    let result = qvd::run::<f64>(
        &circuit,
        200,
        &Options {
            seed: Some(1),
            ..Options::default()
        },
    )
    .unwrap();
    assert!(
        result.stats.peak_active_qubits.is_some(),
        "Auto should choose the near-Clifford backend"
    );
    // A dense random circuit stays on the state vector.
    let mut dense = random_gate_circuit(10, 80, 4);
    dense.measure_all();
    let result = qvd::run::<f64>(
        &dense,
        200,
        &Options {
            seed: Some(1),
            ..Options::default()
        },
    )
    .unwrap();
    assert_eq!(result.stats.peak_active_qubits, None);
}

/// A noisy non-Clifford circuit on `n` qubits: random gates (rotations by
/// any angle, T, controlled phases, ...) with random noise channels between
/// them, optionally a mid-circuit measurement and a reset, then every qubit
/// measured.
fn noisy_circuit(rng: &mut impl Rng, n: usize, case: u64, mid_circuit: bool) -> Circuit {
    let mut circuit = Circuit::new(n);
    for part in 0..3 {
        let gates = random_gate_circuit(n, 8, case * 10 + part);
        circuit.instructions.extend(gates.instructions);
        let (channel, qubits) = common::random_noise(rng, n);
        circuit.noise(channel, &qubits);
    }
    if mid_circuit {
        circuit.measure(0, 0);
        circuit.reset(n - 1);
        circuit.x_error(n - 1, 0.25);
        let more = random_gate_circuit(n, 8, case * 10 + 7);
        circuit.instructions.extend(more.instructions);
        let (channel, qubits) = common::random_noise(rng, n);
        circuit.noise(channel, &qubits);
    }
    for q in 0..n {
        circuit.measure(q, q + 1);
    }
    circuit
}

#[test]
fn noisy_sampling_matches_exact_distribution() {
    // Noise between non-Clifford gates, against the exact distribution
    // summed over every combination of errors and every measurement
    // outcome of a dense simulation. Errors before a rotation make some
    // shots' frames anticommute with it, which splits the group of shots.
    let mut rng = common::rng(41);
    for case in 0..40u64 {
        let n = 1 + (case as usize % 5);
        let circuit = noisy_circuit(&mut rng, n, case, case % 2 == 1);
        assert!(circuit.has_noise() && !qvd::stabilizer::is_clifford(&circuit));
        let shots = 50_000;
        let options = Options {
            backend: Backend::NearClifford,
            seed: Some(200 + case),
            ..Options::default()
        };
        let result = qvd::run::<f64>(&circuit, shots, &options).unwrap();
        assert_eq!(result.counts.values().sum::<usize>(), shots, "case {case}");
        let exact = exact_distribution(&circuit);
        let total: f64 = exact.values().sum();
        assert!((total - 1.0).abs() < 1e-9, "case {case}: total {total}");
        for key in result.counts.keys() {
            assert!(
                exact.get(key).is_some_and(|&p| p > 1e-12),
                "case {case}: impossible outcome {key}"
            );
        }
        for (key, &p) in &exact {
            let observed = *result.counts.get(key).unwrap_or(&0) as f64 / shots as f64;
            let sigma = (p * (1.0 - p) / shots as f64).sqrt();
            let slack = 3.0 / shots as f64;
            assert!(
                (observed - p).abs() <= 5.0 * sigma + slack,
                "case {case} {key}: observed {observed}, exact {p}"
            );
        }
    }
}

#[test]
fn errors_before_a_qubit_joins_the_dense_part_match_exact_distribution() {
    // Every T here brings a new qubit into the dense part (the qubits start
    // in X or Y eigenstates), and the errors just before it flip its sign in
    // some shots: those shots take a Z on the new qubit into their frames
    // instead of a dense part of their own, and that Z then flips later
    // rotations and measurements. Against the exact distribution, at
    // several error rates, including one where most shots are struck.
    for (case, (px, py, pz)) in [(0.1, 0.0, 0.0), (0.2, 0.1, 0.05), (0.4, 0.3, 0.2)]
        .into_iter()
        .enumerate()
    {
        let mut circuit = Circuit::new(3);
        circuit.gate(Gate::H, &[0]);
        circuit.gate(Gate::H, &[1]).gate(Gate::S, &[1]);
        circuit.gate(Gate::SX, &[2]);
        circuit.noise(qvd::Noise::Pauli1 { px, py, pz }, &[0, 1, 2]);
        for q in 0..3 {
            circuit.gate(Gate::T, &[q]);
        }
        circuit.cx(0, 1).cx(1, 2);
        circuit.noise(qvd::Noise::Pauli1 { px, py, pz }, &[1]);
        circuit
            .gate(Gate::H, &[0])
            .gate(Gate::T, &[0])
            .gate(Gate::H, &[0]);
        circuit
            .gate(Gate::H, &[2])
            .gate(Gate::Tdg, &[2])
            .gate(Gate::SX, &[1]);
        circuit.measure_all();
        let shots = 100_000;
        let options = Options {
            backend: Backend::NearClifford,
            seed: Some(31 + case as u64),
            ..Options::default()
        };
        let result = qvd::run::<f64>(&circuit, shots, &options).unwrap();
        assert_eq!(result.stats.peak_active_qubits, Some(3), "case {case}");
        let exact = exact_distribution(&circuit);
        for key in result.counts.keys() {
            assert!(
                exact.get(key).is_some_and(|&p| p > 1e-12),
                "case {case}: impossible outcome {key}"
            );
        }
        for (key, &p) in &exact {
            let observed = *result.counts.get(key).unwrap_or(&0) as f64 / shots as f64;
            let sigma = (p * (1.0 - p) / shots as f64).sqrt();
            assert!(
                (observed - p).abs() <= 5.0 * sigma + 3.0 / shots as f64,
                "case {case} {key}: observed {observed}, exact {p}"
            );
        }
    }
}

#[test]
fn noise_on_a_wide_circuit_flips_at_the_right_rates() {
    // 150 qubits (frames of several words per qubit, many shots per word),
    // one after the other: H, T, a noise channel, T†, H, measure. Without
    // errors every qubit reads 0, with one active qubit at a time. A Z error
    // between T and T† commutes with both and makes the qubit read 1. An X
    // error anticommutes with them, so those shots split off and see T T =
    // S instead of the identity, after which H S H|0> reads 1 with
    // probability 1/2. Each qubit therefore reads 1 at rate pz + px / 2,
    // independently.
    let n = 150;
    let shots = 20_000;
    for (pz, px) in [(0.01, 0.0), (0.2, 0.0), (0.0, 0.1), (0.05, 0.3), (1.0, 0.0)] {
        let mut circuit = Circuit::new(n);
        for q in 0..n {
            circuit.gate(Gate::H, &[q]).gate(Gate::T, &[q]);
            circuit.noise(qvd::Noise::Pauli1 { px, py: 0.0, pz }, &[q]);
            circuit.gate(Gate::Tdg, &[q]).gate(Gate::H, &[q]);
            circuit.measure(q, q);
        }
        let options = Options {
            backend: Backend::NearClifford,
            seed: Some(5),
            ..Options::default()
        };
        let result = qvd::run::<f64>(&circuit, shots, &options).unwrap();
        assert_eq!(result.stats.peak_active_qubits, Some(1));
        let expected = pz + px / 2.0;
        let mut ones = vec![0usize; n];
        for (key, &count) in &result.counts {
            // Bit strings put classical bit 0 last.
            for (q, c) in key.chars().rev().enumerate() {
                if c == '1' {
                    ones[q] += count;
                }
            }
        }
        let sigma = (expected * (1.0 - expected) / shots as f64).sqrt();
        for (q, &k) in ones.iter().enumerate() {
            let rate = k as f64 / shots as f64;
            assert!(
                (rate - expected).abs() <= 5.0 * sigma + 1e-12,
                "pz {pz} px {px}, qubit {q}: rate {rate}, expected {expected}"
            );
        }
    }
}

#[test]
fn noisy_sampling_is_reproducible_across_thread_counts() {
    let mut rng = common::rng(43);
    let circuit = noisy_circuit(&mut rng, 6, 3, true);
    let options = Options {
        backend: Backend::NearClifford,
        seed: Some(4),
        ..Options::default()
    };
    let sample = |threads| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| qvd::run::<f64>(&circuit, 5000, &options))
            .unwrap()
            .counts
    };
    assert_eq!(sample(1), sample(4));
}

#[test]
fn auto_sends_noisy_non_clifford_circuits_to_near_clifford() {
    let mut circuit = Circuit::new(3);
    circuit.gate(Gate::H, &[0]).gate(Gate::T, &[0]).cx(0, 1);
    circuit.depolarize1(1, 0.1);
    circuit.measure_all();
    let result = qvd::run::<f64>(&circuit, 100, &Options::default()).unwrap();
    assert_eq!(result.stats.backend, Some(Backend::NearClifford));
    // Noisy Clifford circuits still go to the stabilizer backend, and the
    // state vector still refuses noise.
    let mut clifford = Circuit::new(2);
    clifford
        .gate(Gate::H, &[0])
        .cx(0, 1)
        .depolarize1(0, 0.1)
        .measure_all();
    let result = qvd::run::<f64>(&clifford, 100, &Options::default()).unwrap();
    assert_eq!(result.stats.backend, Some(Backend::Stabilizer));
    let options = Options {
        backend: Backend::StateVector,
        ..Options::default()
    };
    assert!(qvd::run::<f64>(&circuit, 100, &options).is_err());
}

#[test]
fn noisy_path_measures_and_resets_like_the_exact_distribution() {
    // Small circuits with a mid-circuit measurement and a reset, run on the
    // noisy path (a channel of probability 0 sends them there) without any
    // error, against the exact distribution. Every kind of measurement
    // occurs: of the dense part (splitting groups and retiring a qubit, with
    // the X for outcome 1 put into the shots' frames), 50/50 ones that
    // only change the frame, and fixed ones; each shot's outcome flips must
    // be taken before a collapse conjugates the frames.
    let shots = 4000;
    for n in 2..=3usize {
        for seed in 0..150u64 {
            let mut circuit = random_gate_circuit(n, 5, seed);
            circuit.x_error(0, 0.0);
            circuit.measure(0, 0);
            circuit.reset(n - 1);
            let more = random_gate_circuit(n, 5, seed + 1000);
            circuit.instructions.extend(more.instructions);
            for q in 0..n {
                circuit.measure(q, q + 1);
            }
            let options = Options {
                backend: Backend::NearClifford,
                seed: Some(seed),
                ..Options::default()
            };
            let result = qvd::run::<f64>(&circuit, shots, &options).unwrap();
            let exact = exact_distribution(&circuit);
            for (key, &count) in &result.counts {
                assert!(
                    exact.get(key).is_some_and(|&p| p > 1e-12),
                    "n {n} seed {seed}: impossible outcome {key} ({count} shots)"
                );
            }
            for (key, &p) in &exact {
                // Sums of branch weights can exceed 1 by a rounding error.
                let p = p.clamp(0.0, 1.0);
                let observed = *result.counts.get(key).unwrap_or(&0) as f64 / shots as f64;
                let sigma = (p * (1.0 - p) / shots as f64).sqrt();
                assert!(
                    (observed - p).abs() <= 6.0 * sigma + 3.0 / shots as f64,
                    "n {n} seed {seed} {key}: observed {observed}, exact {p}"
                );
            }
        }
    }
}

#[test]
fn noise_after_every_gate_matches_the_exact_distribution() {
    // A channel after every gate, all of probability 1/8 (exactly, so that
    // consecutive channels make one run of sites, through which each shot
    // draws one gap per error), of different kinds: X, Y or Z flips, a
    // mixed flip, two-qubit depolarizing noise. Against the exact
    // distribution summed over every error.
    let p = 0.125;
    let flips = [
        qvd::Noise::Pauli1 {
            px: p,
            py: 0.0,
            pz: 0.0,
        },
        qvd::Noise::Pauli1 {
            px: 0.0,
            py: p,
            pz: 0.0,
        },
        qvd::Noise::Pauli1 {
            px: 0.0,
            py: 0.0,
            pz: p,
        },
        qvd::Noise::Pauli1 {
            px: p / 2.0,
            py: 0.0,
            pz: p / 2.0,
        },
    ];
    let shots = 40_000;
    for seed in 0..12u64 {
        let n = 2 + seed as usize % 2;
        let gates = random_gate_circuit(n, 5, 300 + seed);
        let mut circuit = Circuit::new(n);
        for (i, instruction) in gates.instructions.into_iter().enumerate() {
            let qubits = match &instruction {
                Instruction::Gate { qubits, .. } => qubits.clone(),
                _ => Vec::new(),
            };
            circuit.instructions.push(instruction);
            match qubits.len() {
                1 => circuit.noise(flips[(i + seed as usize) % 4].clone(), &qubits),
                2 => circuit.noise(qvd::Noise::Depolarize2(p), &qubits),
                _ => &mut circuit,
            };
        }
        circuit.measure_all();
        assert!(circuit.has_noise() && !qvd::stabilizer::is_clifford(&circuit));
        let options = Options {
            backend: Backend::NearClifford,
            seed: Some(seed),
            ..Options::default()
        };
        let result = qvd::run::<f64>(&circuit, shots, &options).unwrap();
        let exact = exact_distribution(&circuit);
        for key in result.counts.keys() {
            assert!(
                exact.get(key).is_some_and(|&p| p > 1e-12),
                "seed {seed}: impossible outcome {key}"
            );
        }
        for (key, &p) in &exact {
            let p = p.clamp(0.0, 1.0);
            let observed = *result.counts.get(key).unwrap_or(&0) as f64 / shots as f64;
            let sigma = (p * (1.0 - p) / shots as f64).sqrt();
            assert!(
                (observed - p).abs() <= 5.0 * sigma + 3.0 / shots as f64,
                "seed {seed} {key}: observed {observed}, exact {p}"
            );
        }
    }
}
