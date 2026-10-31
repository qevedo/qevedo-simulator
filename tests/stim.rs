//! The Stim-format reader and detector sampling: parsing (blocks,
//! records, errors), the gates' conventions against Stim's Pauli tables,
//! and detection events against rates computed by hand.

mod common;

use qvd::stabilizer::{self, sample_detectors, sample_stim};
use qvd::stim::parse;
use qvd::{Gate, Instruction, Matrix};

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
            "H 0\nMPP X0*X1\n",
            "line 2: Stim instruction MPP is not supported",
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
    // qvd's own export reads back as the same samples.
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
