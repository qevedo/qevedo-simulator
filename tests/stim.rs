//! The Stim-format reader and detector sampling: parsing (blocks,
//! records, errors), the gates' conventions against Stim's Pauli tables,
//! detection events against rates computed by hand, Pauli product
//! measurements (`MPP`) against the reference state vector, and
//! `PAULI_CHANNEL_2` and correlated errors against Stim's definitions.

mod common;

use qvd::stabilizer::{self, sample_detectors, sample_stim};
use qvd::stim::parse;
use qvd::{C64, Gate, Instruction, Matrix, Noise, reference};
use rand::RngExt;

#[test]
fn parses_blocks_records_and_annotations() {
    let text = "\
QUBIT_COORDS(0, 0) 0  # coordinates are ignored
R 0 1 2
TICK
REPEAT 2 {
    CX 0 1 2 1
    MR(0.01) 1
    DETECTOR(1, 0) rec[-1]
}
M !0 2
mpad 1
DETECTOR rec[-3] rec[-4]
OBSERVABLE_INCLUDE(1) rec[-2] rec[-3]
";
    let c = parse(text).unwrap();
    assert_eq!(c.circuit.num_qubits, 3);
    assert_eq!(c.num_measurements(), 5);
    // Measurements 0 and 1 are the two MRs of the loop, 2 and 3 the final M
    // of qubits 0 and 2, 4 the MPAD.
    assert_eq!(c.detectors, vec![vec![0], vec![1], vec![2, 1]]);
    assert_eq!(c.observables, vec![vec![], vec![3, 2]]);
    assert_eq!(c.flips, vec![(0, 0.01), (1, 0.01)]);
    assert_eq!(c.inverted, vec![2, 4]);
    let measured: Vec<(usize, usize)> = c
        .circuit
        .instructions
        .iter()
        .filter_map(|i| match i {
            Instruction::Measure { qubit, clbit } => Some((*qubit, *clbit)),
            _ => None,
        })
        .collect();
    assert_eq!(measured, vec![(1, 0), (1, 1), (0, 2), (2, 3)]);
}

#[test]
fn errors_name_the_line() {
    for (text, message) in [
        (
            "H 0\nMXX 0 1\n",
            "line 2: Stim instruction MXX is not supported",
        ),
        ("MPP X0*Z0\n", "line 1: MPP: qubit 0 appears twice"),
        ("MPP 0 1\n", "line 1: MPP takes Pauli products"),
        ("H X0\n", "line 1: H takes qubit targets, not Paulis"),
        ("E(0.1) 0 1\n", "line 1: E takes Pauli targets"),
        ("E(0.1) X0 Z0\n", "line 1: E names qubit 0 twice"),
        (
            "H 0\nELSE_CORRELATED_ERROR(0.1) X0\n",
            "line 2: ELSE_CORRELATED_ERROR must follow E",
        ),
        (
            "E(0.1) X0\nH 0\nELSE_CORRELATED_ERROR(0.1) X0\n",
            "line 3: ELSE_CORRELATED_ERROR must follow E",
        ),
        (
            "PAULI_CHANNEL_2(0.1) 0 1\n",
            "line 1: PAULI_CHANNEL_2 takes 15 probability",
        ),
        (
            "M 0\nCX rec[-1] 1\n",
            "line 2: CX with measurement record targets",
        ),
        (
            "M 0\nDETECTOR rec[-2]\n",
            "line 2: rec[-2] refers to a measurement before",
        ),
        ("REPEAT 3 {\nH 0\n", "line 1: REPEAT block is never closed"),
        ("H 0\n}\n", "line 2: '}' without a REPEAT block"),
        (
            "X_ERROR(1.5) 0\n",
            "line 1: X_ERROR: probabilities must lie in [0, 1]",
        ),
        (
            "DEPOLARIZE2(0.1) 0 0\n",
            "line 1: DEPOLARIZE2 needs pairs of distinct qubits",
        ),
        ("CZ 0 1 2\n", "line 1: CZ needs pairs of targets"),
        ("H sweep[0]\n", "line 1: target 'sweep[0]' is not supported"),
        (
            "REPEAT 2 {\n  REPEAT 2 {\n    FOO 0\n  }\n}\n",
            "line 3: Stim instruction FOO",
        ),
    ] {
        let error = parse(text).unwrap_err();
        assert!(error.starts_with(message), "{text:?}: {error}");
    }
}

/// `U P U†` for the single-qubit gates of `circuit` and a Pauli matrix.
fn conjugate(circuit: &qvd::Circuit, pauli: &Matrix) -> Matrix {
    let mut u = Matrix::identity(1);
    for instruction in &circuit.instructions {
        if let Instruction::Gate { gate, .. } = instruction {
            u = gate.matrix().mul(&u);
        }
    }
    u.mul(pauli).mul(&u.adjoint())
}

#[test]
fn gates_follow_stims_pauli_tables() {
    let x = Gate::X.matrix();
    let y = Gate::Y.matrix();
    let z = Gate::Z.matrix();
    let neg = |m: &Matrix| Matrix::new(m.entries().iter().map(|e| -e).collect());
    // Where each gate sends X and Z, from Stim's gate reference.
    let table: [(&str, Matrix, Matrix); 18] = [
        ("I", x.clone(), z.clone()),
        ("X", x.clone(), neg(&z)),
        ("Y", neg(&x), neg(&z)),
        ("Z", neg(&x), z.clone()),
        ("H", z.clone(), x.clone()),
        ("H_XZ", z.clone(), x.clone()),
        ("S", y.clone(), z.clone()),
        ("SQRT_Z", y.clone(), z.clone()),
        ("S_DAG", neg(&y), z.clone()),
        ("SQRT_Z_DAG", neg(&y), z.clone()),
        ("SQRT_X", x.clone(), neg(&y)),
        ("SQRT_X_DAG", x.clone(), y.clone()),
        ("SQRT_Y", neg(&z), x.clone()),
        ("SQRT_Y_DAG", z.clone(), neg(&x)),
        ("H_XY", y.clone(), neg(&z)),
        ("H_YZ", neg(&x), y.clone()),
        ("C_XYZ", y.clone(), x.clone()),
        ("C_ZYX", z.clone(), y.clone()),
    ];
    for (name, to_x, to_z) in table {
        let c = parse(&format!("{name} 0\n")).unwrap().circuit;
        assert!(conjugate(&c, &x).distance(&to_x) < 1e-12, "{name} on X");
        assert!(conjugate(&c, &z).distance(&to_z) < 1e-12, "{name} on Z");
    }
}

#[test]
fn measurement_bases_and_resets() {
    // Each basis' reset prepares its +1 eigenstate, which its measurement
    // finds every time; inverted results and MPAD 1 read 1.
    let text = "RX 0\nRY 1\nR 2\nMX 0\nMY 1\nM 2\nMRX 0\nMRY !1\nMPAD 0 1\n";
    let c = parse(text).unwrap();
    let samples = sample_stim(&c, 200, Some(1)).unwrap();
    let expected = [false, false, false, false, true, false, true];
    for (m, &bit) in expected.iter().enumerate() {
        assert!(
            (0..200).all(|s| samples.get(s, m) == bit),
            "measurement {m}"
        );
    }
    // A Z measurement of |+> is random, and RY leaves |+i> (Y = +1).
    let c = parse("RX 0\nM 0\n").unwrap();
    let samples = sample_stim(&c, 4000, Some(2)).unwrap();
    let ones = (0..4000).filter(|&s| samples.get(s, 0)).count();
    assert!((1800..2200).contains(&ones), "{ones}");
}

/// A distance-`d` repetition code, `rounds` rounds of `CX`s onto ancillas
/// measured with `MR(p)`, in Stim's format with its detectors and the
/// observable (data qubit 0).
fn repetition_code(d: usize, rounds: usize, flip: f64, data_error: f64) -> String {
    let ancillas = d - 1;
    let mut s = format!(
        "R {}\n",
        (0..2 * d - 1)
            .map(|q| q.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    );
    let round = |s: &mut String| {
        for a in 0..ancillas {
            s.push_str(&format!("CX {} {}\n", 2 * a, 2 * a + 1));
            s.push_str(&format!("CX {} {}\n", 2 * a + 2, 2 * a + 1));
        }
        let ancilla_list: Vec<String> = (0..ancillas).map(|a| (2 * a + 1).to_string()).collect();
        s.push_str(&format!("MR({flip}) {}\n", ancilla_list.join(" ")));
    };
    round(&mut s);
    for a in 0..ancillas {
        s.push_str(&format!("DETECTOR rec[-{}]\n", ancillas - a));
    }
    s.push_str(&format!("REPEAT {} {{\n", rounds - 1));
    if data_error > 0.0 {
        s.push_str(&format!("X_ERROR({data_error}) 0\n"));
    }
    round(&mut s);
    for a in 0..ancillas {
        let k = ancillas - a;
        s.push_str(&format!("DETECTOR rec[-{k}] rec[-{}]\n", k + ancillas));
    }
    s.push_str("}\n");
    let data: Vec<String> = (0..d).map(|i| (2 * i).to_string()).collect();
    s.push_str(&format!("M {}\n", data.join(" ")));
    s.push_str(&format!("OBSERVABLE_INCLUDE(0) rec[-{d}]\n"));
    s
}

#[test]
fn noiseless_detectors_never_fire() {
    let c = parse(&repetition_code(7, 6, 0.0, 0.0)).unwrap();
    assert_eq!(c.detectors.len(), 6 * 6);
    let samples = sample_detectors(&c, 1000, Some(3)).unwrap();
    for d in 0..samples.num_detectors {
        assert!(
            samples.detector_words(d).iter().all(|&w| w == 0),
            "detector {d}"
        );
    }
    assert!(samples.observable_words(0).iter().all(|&w| w == 0));
}

#[test]
fn certain_errors_fire_their_detectors() {
    // An X on data qubit 0 before every round after the first: ancilla 0
    // (the parity of data qubits 0 and 1) toggles every round, so its
    // detector fires in every round after the first; ancilla 1 never
    // changes. The observable flips with the parity of the number of X's.
    let (d, rounds) = (3, 4);
    let c = parse(&repetition_code(d, rounds, 0.0, 1.0)).unwrap();
    let samples = sample_detectors(&c, 100, Some(4)).unwrap();
    for shot in 0..100 {
        for r in 0..rounds {
            for a in 0..d - 1 {
                let fires = r > 0 && a == 0;
                assert_eq!(samples.detector(shot, r * (d - 1) + a), fires, "r{r} a{a}");
            }
        }
        // Three X errors: data qubit 0 ends flipped.
        assert!(samples.observable(shot, 0));
    }
}

#[test]
fn measurement_flips_have_the_right_detector_rates() {
    // With only measurement flips of probability p, a detector comparing
    // two noisy rounds fires with probability 2p(1 - p), one in the first
    // round with probability p; the observable never flips.
    let (d, rounds, p, shots) = (5, 5, 0.05, 40_000);
    let c = parse(&repetition_code(d, rounds, p, 0.0)).unwrap();
    let samples = sample_detectors(&c, shots, Some(5)).unwrap();
    for detector in 0..samples.num_detectors {
        let fired = (0..shots)
            .filter(|&s| samples.detector(s, detector))
            .count() as f64;
        let expected = if detector < d - 1 {
            p
        } else {
            2.0 * p * (1.0 - p)
        };
        let sigma = (expected * (1.0 - expected) / shots as f64).sqrt();
        let rate = fired / shots as f64;
        assert!(
            (rate - expected).abs() < 5.0 * sigma,
            "detector {detector}: {rate} vs {expected}"
        );
    }
    assert!(samples.observable_words(0).iter().all(|&w| w == 0));
    // The shot-major bytes hold the same bits.
    let bytes = samples.detectors_shot_major();
    assert_eq!(bytes.len(), shots * samples.num_detectors);
    for shot in (0..shots).step_by(97) {
        for detector in 0..samples.num_detectors {
            assert_eq!(
                bytes[shot * samples.num_detectors + detector] == 1,
                samples.detector(shot, detector)
            );
        }
    }
    assert!(samples.observables_shot_major().iter().all(|&b| b == 0));
    // Bits past the last shot are zero.
    let odd = sample_detectors(&c, 100, Some(5)).unwrap();
    for detector in 0..odd.num_detectors {
        assert_eq!(odd.detector_words(detector)[1] >> 36, 0);
    }
}

#[test]
fn detectors_match_the_measurement_records() {
    // Detection events are the parities of sample_stim's records against
    // the noiseless parities, shot by shot.
    let text =
        repetition_code(5, 4, 0.02, 0.03).replace("CX 0 1\n", "CX 0 1\nDEPOLARIZE2(0.02) 0 1\n");
    let c = parse(&text).unwrap();
    let records = sample_stim(&c, 3000, Some(6)).unwrap();
    let noiseless = parse(&repetition_code(5, 4, 0.0, 0.0)).unwrap();
    let reference = sample_stim(&noiseless, 1, Some(6)).unwrap();
    let detectors = sample_detectors(&c, 3000, Some(6)).unwrap();
    for (i, detector) in c.detectors.iter().enumerate() {
        for shot in 0..3000 {
            let parity = detector.iter().fold(false, |acc, &m| {
                acc ^ records.get(shot, m) ^ reference.get(0, m)
            });
            assert_eq!(
                detectors.detector(shot, i),
                parity,
                "detector {i}, shot {shot}"
            );
        }
    }
}

#[test]
fn detector_sampling_is_reproducible_across_thread_counts() {
    let text = repetition_code(9, 9, 0.01, 0.02);
    let c = parse(&text).unwrap();
    let run = |threads| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| sample_detectors(&c, 5000, Some(8)))
            .unwrap()
    };
    let (one, three) = (run(1), run(3));
    for d in 0..one.num_detectors {
        assert_eq!(
            one.detector_words(d),
            three.detector_words(d),
            "detector {d}"
        );
    }
    assert_eq!(one.observable_words(0), three.observable_words(0));
}

#[test]
fn round_trip_through_to_stim() {
    // Qevedo's own export reads back as the same samples.
    let circuit = qvd::library::with_depolarizing_noise(&qvd::library::surface_code(3, 3), 0.01);
    let text = stabilizer::to_stim(&circuit).unwrap();
    let parsed = parse(&text).unwrap();
    let a = stabilizer::sample(&circuit, 2000, Some(9)).unwrap();
    let b = sample_stim(&parsed, 2000, Some(9)).unwrap();
    assert_eq!(a.num_clbits, b.num_clbits);
    for c in 0..a.num_clbits {
        assert_eq!(a.clbit_words(c), b.clbit_words(c), "classical bit {c}");
    }
}

#[test]
fn every_output_layout_holds_the_same_bits() {
    // The byte layouts handed to NumPy (measurement-major records, bool and
    // bit-packed; shot-major detection events, bool, bit-packed and with the
    // observables appended) against the samples read bit by bit. Shot
    // counts that are not multiples of 8 or 64, and a detector count that
    // is not a multiple of 8, exercise the padding.
    let text =
        repetition_code(6, 5, 0.05, 0.05).replace("CX 0 1\n", "CX 0 1\nDEPOLARIZE2(0.05) 0 1\n");
    let c = parse(&text).unwrap();
    for shots in [1, 7, 64, 100, 1000] {
        let records = sample_stim(&c, shots, Some(11)).unwrap();
        let (m, width) = (records.num_clbits, shots.div_ceil(8));
        let mut bytes = vec![7; m * shots];
        records.write_bytes(&mut bytes);
        let mut packed = vec![7; m * width];
        records.write_packed(&mut packed);
        for clbit in 0..m {
            for shot in 0..shots {
                let bit = records.get(shot, clbit);
                assert_eq!(bytes[clbit * shots + shot], bit as u8, "{shots} shots");
                assert_eq!(
                    (packed[clbit * width + shot / 8] >> (shot % 8)) & 1,
                    bit as u8
                );
            }
            if shots % 8 != 0 {
                assert_eq!(packed[clbit * width + width - 1] >> (shots % 8), 0);
            }
        }

        let samples = sample_detectors(&c, shots, Some(11)).unwrap();
        let (nd, no) = (samples.num_detectors, samples.num_observables);
        assert_ne!(nd % 8, 0, "the test needs a partly filled byte");
        let (wd, wo, wj) = (nd.div_ceil(8), no.div_ceil(8), (nd + no).div_ceil(8));
        let mut d_packed = vec![7; shots * wd];
        samples.write_detectors_shot_major_packed(&mut d_packed);
        let mut o_packed = vec![7; shots * wo];
        samples.write_observables_shot_major_packed(&mut o_packed);
        let mut joined = vec![7; shots * (nd + no)];
        samples.write_shot_major(false, &mut joined);
        let mut joined_packed = vec![7; shots * wj];
        samples.write_shot_major(true, &mut joined_packed);
        let bytes = samples.detectors_shot_major();
        let packed_bit = |buffer: &[u8], width: usize, shot: usize, r: usize| {
            (buffer[shot * width + r / 8] >> (r % 8)) & 1 == 1
        };
        for shot in 0..shots {
            for d in 0..nd {
                let bit = samples.detector(shot, d);
                assert_eq!(bytes[shot * nd + d] == 1, bit);
                assert_eq!(packed_bit(&d_packed, wd, shot, d), bit, "detector {d}");
                assert_eq!(joined[shot * (nd + no) + d] == 1, bit);
                assert_eq!(packed_bit(&joined_packed, wj, shot, d), bit);
            }
            for o in 0..no {
                let bit = samples.observable(shot, o);
                assert_eq!(packed_bit(&o_packed, wo, shot, o), bit);
                assert_eq!(joined[shot * (nd + no) + nd + o] == 1, bit);
                assert_eq!(packed_bit(&joined_packed, wj, shot, nd + o), bit);
            }
            // Padding bits are zero.
            assert_eq!(d_packed[shot * wd + wd - 1] >> (nd % 8), 0);
            if (nd + no) % 8 != 0 {
                assert_eq!(joined_packed[shot * wj + wj - 1] >> ((nd + no) % 8), 0);
            }
        }
        // Some detectors fired, or the comparison would prove little.
        assert!(bytes.contains(&1) || shots < 64);
    }
}

/// `<ψ|P|ψ>` for the state of the unitary `prefix` and the Pauli product
/// `factors` (letter, qubit), from the reference state vector.
fn pauli_expectation(prefix: &qvd::Circuit, factors: &[(u8, usize)]) -> f64 {
    let state = reference::statevector(prefix);
    let mut moved = state.clone();
    for &(pauli, q) in factors {
        let gate = match pauli {
            b'X' => Gate::X,
            b'Y' => Gate::Y,
            _ => Gate::Z,
        };
        reference::apply(&mut moved, &gate.matrix(), &[q]);
    }
    let overlap: C64 = state.iter().zip(&moved).map(|(a, b)| a.conj() * b).sum();
    overlap.re
}

#[test]
fn mpp_measures_pauli_products_as_the_state_vector_predicts() {
    // Random Clifford circuits, then one MPP of a random product: when the
    // state is an eigenstate of the product (expectation ±1), every shot
    // reads that sign (1 for -1, flipped again by `!`); otherwise the
    // expectation is 0 and the result is a fair coin.
    let mut rng = common::rng(11);
    let one = ["H", "S", "S_DAG", "SQRT_X", "SQRT_Y", "H_YZ", "C_XYZ"];
    let two = ["CX", "CZ", "CY", "SWAP"];
    let (mut deterministic, mut random) = (0, 0);
    for _ in 0..200 {
        let n = rng.random_range(2..6usize);
        let mut text = String::new();
        for _ in 0..rng.random_range(0..12) {
            if rng.random::<bool>() {
                let g = one[rng.random_range(0..one.len())];
                text += &format!("{g} {}\n", rng.random_range(0..n));
            } else {
                let q = common::random_qubits(&mut rng, n, 2);
                let g = two[rng.random_range(0..two.len())];
                text += &format!("{g} {} {}\n", q[0], q[1]);
            }
        }
        let prefix = parse(&text).unwrap().circuit;
        let k = rng.random_range(1..=n);
        let qubits = common::random_qubits(&mut rng, n, k);
        let factors: Vec<(u8, usize)> = qubits
            .iter()
            .map(|&q| (b"XYZ"[rng.random_range(0..3)], q))
            .collect();
        let inverted = rng.random::<bool>();
        let product: Vec<String> = factors
            .iter()
            .map(|&(p, q)| format!("{}{q}", p as char))
            .collect();
        let bang = if inverted { "!" } else { "" };
        text += &format!("MPP {bang}{}\n", product.join("*"));
        let mut c = parse(&text).unwrap();
        // Stim's qubit count is the largest named qubit + 1; pad the
        // prefix to the same width.
        let mut prefix = prefix;
        prefix.num_qubits = c.circuit.num_qubits.max(prefix.num_qubits);
        c.circuit.num_qubits = prefix.num_qubits;
        let expectation = pauli_expectation(&prefix, &factors);
        let samples = sample_stim(&c, 2000, Some(rng.random())).unwrap();
        let ones = (0..2000).filter(|&s| samples.get(s, 0)).count();
        if (expectation.abs() - 1.0).abs() < 1e-9 {
            deterministic += 1;
            let expected = (expectation < 0.0) != inverted;
            assert_eq!(ones, if expected { 2000 } else { 0 }, "{text}");
        } else {
            random += 1;
            assert!(expectation.abs() < 1e-9, "{text}: {expectation}");
            assert!((850..1150).contains(&ones), "{text}: {ones}");
        }
    }
    assert!(
        deterministic > 15 && random > 100,
        "{deterministic} {random}"
    );
}

#[test]
fn mpp_on_a_bell_pair_and_its_flips() {
    // |Φ+> is stabilized by XX, -YY and ZZ; Z0 Z1 measured twice agrees
    // with itself, and X0 alone is random. MPP(p) flips the record only.
    let text = "H 0\nCX 0 1\nMPP X0*X1 Y0*Y1 Z0*Z1 !Z1*Z0\nMPP(0.1) Z0*Z1\nMPP X0\nM 0 1\n";
    let c = parse(text).unwrap();
    assert_eq!(c.num_measurements(), 8);
    assert_eq!(c.flips, vec![(4, 0.1)]);
    let shots = 40_000;
    let samples = sample_stim(&c, shots, Some(3)).unwrap();
    let count = |m: usize| (0..shots).filter(|&s| samples.get(s, m)).count();
    assert_eq!(
        [count(0), count(1), count(2), count(3)],
        [0, shots, 0, shots]
    );
    let flipped = count(4) as f64 / shots as f64;
    assert!((flipped - 0.1).abs() < 0.006, "{flipped}");
    let x0 = count(5) as f64 / shots as f64;
    assert!((x0 - 0.5).abs() < 0.02, "{x0}");
    // Measuring X0 leaves a state stabilized by X0 and X1 (XX times X0),
    // so the final Z measurements are fair coins.
    let (m0, m1) = (
        count(6) as f64 / shots as f64,
        count(7) as f64 / shots as f64,
    );
    assert!(
        (m0 - 0.5).abs() < 0.02 && (m1 - 0.5).abs() < 0.02,
        "{m0} {m1}"
    );
}

#[test]
fn pauli_channel_2_follows_stims_order() {
    // With one of the 15 probabilities at 1: qubits 0, 1 start in |0>
    // and show X components in Z measurements; qubits 2, 3 start in |+>
    // and show Z components in X measurements. Stim's order is IX, IY,
    // IZ, XI, ..., ZZ with the first letter on the first qubit of a pair.
    let letters = ['I', 'X', 'Y', 'Z'];
    for k in 0..15 {
        let mut p = vec!["0"; 15];
        p[k] = "1";
        let text = format!(
            "R 0 1\nRX 2 3\nPAULI_CHANNEL_2({}) 0 1 2 3\nM 0 1\nMX 2 3\n",
            p.join(",")
        );
        let c = parse(&text).unwrap();
        let samples = sample_stim(&c, 64, Some(k as u64)).unwrap();
        let (a, b) = (letters[(k + 1) / 4], letters[(k + 1) % 4]);
        let has_x = |l: char| l == 'X' || l == 'Y';
        let has_z = |l: char| l == 'Z' || l == 'Y';
        let expected = [has_x(a), has_x(b), has_z(a), has_z(b)];
        for (m, &bit) in expected.iter().enumerate() {
            assert!(
                (0..64).all(|s| samples.get(s, m) == bit),
                "{a}{b}: measurement {m}"
            );
        }
    }
}

#[test]
fn correlated_error_chains_have_stims_rates() {
    // E(0.2) X0 X1, then ELSE 0.25 on X1 X2, ELSE 0.5 on X0, a TICK, and
    // ELSE 0.5 on X2: exactly one product or none, with probabilities
    // 0.2, 0.8 * 0.25 = 0.2, 0.6 * 0.5 = 0.3, 0.3 * 0.5 = 0.15, and 0.15.
    let text = "R 0 1 2\nE(0.2) X0 X1\nELSE_CORRELATED_ERROR(0.25) X1 X2\n\
                ELSE_CORRELATED_ERROR(0.5) X0\nTICK\nELSE_CORRELATED_ERROR(0.5) X2\nM 0 1 2\n";
    let c = parse(text).unwrap();
    let noise: Vec<&Instruction> = c
        .circuit
        .instructions
        .iter()
        .filter(|i| matches!(i, Instruction::Noise { .. }))
        .collect();
    assert_eq!(noise.len(), 1, "one chain, one channel");
    let shots = 200_000;
    let samples = sample_stim(&c, shots, Some(8)).unwrap();
    let mut counts = [0usize; 8];
    for s in 0..shots {
        let key = (0..3).fold(0, |k, m| k * 2 + samples.get(s, m) as usize);
        counts[key] += 1;
    }
    // Keys read qubit 0 as the high bit: XX on 0, 1 is 0b110.
    let expected = [
        (0b110, 0.2),
        (0b011, 0.2),
        (0b100, 0.3),
        (0b001, 0.15),
        (0, 0.15),
    ];
    for (key, p) in expected {
        let rate = counts[key] as f64 / shots as f64;
        let sigma = (p * (1.0 - p) / shots as f64).sqrt();
        assert!((rate - p).abs() < 5.0 * sigma, "{key:03b}: {rate} vs {p}");
    }
    let others: usize = [0b010, 0b101, 0b111].iter().map(|&k| counts[k]).sum();
    assert_eq!(others, 0);
    // Two separate chains are independent: XX on (0, 1), and Y on 2
    // (which a Z measurement sees).
    let c = parse("R 0 1 2\nE(0.5) X0 X1\nE(0.5) Y2\nM 0 1 2\n").unwrap();
    let samples = sample_stim(&c, shots, Some(9)).unwrap();
    let both = (0..shots)
        .filter(|&s| samples.get(s, 0) && samples.get(s, 1) && samples.get(s, 2))
        .count() as f64
        / shots as f64;
    assert!((both - 0.25).abs() < 0.005, "{both}");
}

#[test]
fn new_channels_round_trip_through_to_stim() {
    // PAULI_CHANNEL_2 keeps its name and arguments; a correlated chain is
    // written back as E and ELSE_CORRELATED_ERROR with conditional
    // probabilities, and reads back as the same channel.
    let mut p = [0.0; 15];
    for (k, v) in p.iter_mut().enumerate() {
        *v = 0.001 * (k + 1) as f64;
    }
    let text = format!(
        "PAULI_CHANNEL_2({}) 0 1\nE(0.1) X0 Z2\nELSE_CORRELATED_ERROR(0.2) Y1\nE(0.3) Z1\nM 0 1 2\n",
        p.map(|v| v.to_string()).join(",")
    );
    let first = parse(&text).unwrap();
    let written = stabilizer::to_stim(&first.circuit).unwrap();
    assert!(
        written.contains("E(0.1) X0 Z2\nELSE_CORRELATED_ERROR(0.2"),
        "{written}"
    );
    assert!(written.contains("\nE(0.3) Z1\n"), "{written}");
    let second = parse(&written).unwrap();
    let channels = |c: &qvd::Circuit| -> Vec<(Noise, Vec<usize>)> {
        c.instructions
            .iter()
            .filter_map(|i| match i {
                Instruction::Noise { channel, qubits } => Some((channel.clone(), qubits.clone())),
                _ => None,
            })
            .collect()
    };
    let (a, b) = (channels(&first.circuit), channels(&second.circuit));
    assert_eq!(a.len(), 3);
    assert_eq!(a[0], b[0]);
    for ((ca, qa), (cb, qb)) in a[1..].iter().zip(&b[1..]) {
        assert_eq!(qa, qb);
        let (Noise::Correlated(pa), Noise::Correlated(pb)) = (ca, cb) else {
            panic!("{ca:?} {cb:?}");
        };
        for ((x, sa), (y, sb)) in pa.iter().zip(pb) {
            assert_eq!(sa, sb);
            assert!((x - y).abs() < 1e-15);
        }
    }
    // The second product's unconditional probability: 0.9 * 0.2.
    let Noise::Correlated(products) = &a[1].0 else {
        unreachable!()
    };
    assert_eq!(products[0], (0.1, "XZI".to_string()));
    assert!((products[1].0 - 0.18).abs() < 1e-15 && products[1].1 == "IIY");
}
