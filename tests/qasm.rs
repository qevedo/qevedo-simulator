//! The OpenQASM reader against circuits built directly.

mod common;

use common::{exact_distribution, max_distance, project};
use qvd::library::random_gate_circuit;
use qvd::qasm::parse;
use qvd::{Backend, C64, Circuit, Gate, Instruction, Options, Pauli, reference};
use rand::{Rng, RngExt};
use std::collections::BTreeMap;

fn same_state(a: &Circuit, b: &Circuit) -> f64 {
    assert_eq!(a.num_qubits, b.num_qubits);
    max_distance(&reference::statevector(a), &reference::statevector(b))
}

#[test]
fn round_trips_through_qvds_export() {
    for seed in 0..30 {
        let n = 1 + seed as usize % 6;
        let circuit = random_gate_circuit(n, 50, seed);
        let parsed = parse(&circuit.to_qasm().unwrap()).unwrap();
        assert_eq!(parsed.gate_count(), circuit.gate_count());
        assert!(same_state(&circuit, &parsed) < 1e-12, "seed {seed}");
    }
}

#[test]
fn builtin_gates_match_qvds() {
    let cases: Vec<(&str, Gate, usize)> = vec![
        ("id", Gate::I, 1),
        ("h", Gate::H, 1),
        ("x", Gate::X, 1),
        ("y", Gate::Y, 1),
        ("z", Gate::Z, 1),
        ("s", Gate::S, 1),
        ("sdg", Gate::Sdg, 1),
        ("t", Gate::T, 1),
        ("tdg", Gate::Tdg, 1),
        ("sx", Gate::SX, 1),
        ("sxdg", Gate::SXdg, 1),
        ("rx(0.3)", Gate::RX(0.3), 1),
        ("ry(-1.2)", Gate::RY(-1.2), 1),
        ("rz(pi/3)", Gate::RZ(std::f64::consts::PI / 3.0), 1),
        ("p(0.7)", Gate::P(0.7), 1),
        ("u1(0.7)", Gate::P(0.7), 1),
        (
            "u2(0.1, 0.2)",
            Gate::U(std::f64::consts::FRAC_PI_2, 0.1, 0.2),
            1,
        ),
        ("u3(0.1, 0.2, 0.3)", Gate::U(0.1, 0.2, 0.3), 1),
        ("U(0.1, 0.2, 0.3)", Gate::U(0.1, 0.2, 0.3), 1),
        ("cx", Gate::CX, 2),
        ("CX", Gate::CX, 2),
        ("cy", Gate::CY, 2),
        ("cz", Gate::CZ, 2),
        ("ch", Gate::CH, 2),
        ("cp(0.4)", Gate::CP(0.4), 2),
        ("cu1(0.4)", Gate::CP(0.4), 2),
        ("crx(0.5)", Gate::CRX(0.5), 2),
        ("cry(0.5)", Gate::CRY(0.5), 2),
        ("crz(0.5)", Gate::CRZ(0.5), 2),
        ("swap", Gate::SWAP, 2),
        ("rzz(0.6)", Gate::RZZ(0.6), 2),
        ("rxx(0.6)", Gate::RXX(0.6), 2),
        ("ccx", Gate::CCX, 3),
        ("cswap", Gate::CSWAP, 3),
        ("csx", Gate::Unitary(Gate::SX.matrix().controlled(1, 1)), 2),
        (
            "cu3(0.1, 0.2, 0.3)",
            Gate::Unitary(Gate::U(0.1, 0.2, 0.3).matrix().controlled(1, 1)),
            2,
        ),
        ("c3x", Gate::Unitary(Gate::X.matrix().controlled(3, 7)), 4),
    ];
    let mut rng = common::rng(1);
    for (call, gate, arity) in cases {
        // Prepare a random state, apply the gate both ways, compare.
        let mut prefix = Circuit::new(4);
        prefix.unitary(common::random_unitary(&mut rng, 4), &[0, 1, 2, 3]);
        let qubits = common::random_qubits(&mut rng, 4, arity);
        let mut direct = prefix.clone();
        direct.gate(gate, &qubits);
        let operands: Vec<String> = qubits.iter().map(|q| format!("q[{q}]")).collect();
        let program = format!(
            "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[4];\n{call} {};",
            operands.join(", ")
        );
        let mut parsed = prefix.clone();
        parsed
            .instructions
            .extend(parse(&program).unwrap().instructions);
        assert!(same_state(&direct, &parsed) < 1e-12, "{call}");
    }
}

#[test]
fn openqasm_3_matches_openqasm_2() {
    let v2 = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg a[2];
        qreg b[1];
        creg c[3];
        h a[0];
        cx a[0], a[1];
        u3(0.1, 0.2, 0.3) b[0];
        cu1(pi/4) a[1], b[0];
        measure a -> c[0:1];
        measure b[0] -> c[2];
    "#;
    let v3 = r#"
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] a;
        qubit b;
        bit[3] c;
        h a[0];
        cx a[0], a[1];
        U(0.1, 0.2, 0.3) b;
        cp(π/4) a[1], b;
        c[0:1] = measure a;
        c[2] = measure b;
    "#;
    let (c2, c3) = (parse(v2).unwrap(), parse(v3).unwrap());
    assert_eq!((c2.num_qubits, c2.num_clbits), (3, 3));
    assert_eq!(c2.instructions.len(), c3.instructions.len());
    let unitary = |c: &Circuit| {
        let mut u = c.clone();
        u.instructions
            .retain(|i| matches!(i, Instruction::Gate { .. }));
        u
    };
    assert!(same_state(&unitary(&c2), &unitary(&c3)) < 1e-12);
    let measures: Vec<_> = c2
        .instructions
        .iter()
        .filter(|i| matches!(i, Instruction::Measure { .. }))
        .collect();
    assert_eq!(
        measures,
        [
            &Instruction::Measure { qubit: 0, clbit: 0 },
            &Instruction::Measure { qubit: 1, clbit: 1 },
            &Instruction::Measure { qubit: 2, clbit: 2 },
        ]
    );
}

#[test]
fn gate_definitions_and_expressions() {
    let program = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        gate rot(theta, phi) q { rz(phi) q; ry(theta / 2 + 0.1) q; }
        gate pair(a) x, y { rot(a, -a * 2) x; cx x, y; rot(sin(a)^2, cos(a)) y; }
        qreg q[2];
        pair(0.4) q[0], q[1];
        pair(-1.0e-1) q[1], q[0];
    "#;
    let parsed = parse(program).unwrap();
    let mut direct = Circuit::new(2);
    for (a, x, y) in [(0.4f64, 0, 1), (-0.1, 1, 0)] {
        direct.rz(-a * 2.0, x).ry(a / 2.0 + 0.1, x).cx(x, y);
        direct.rz(a.cos(), y).ry(a.sin().powi(2) / 2.0 + 0.1, y);
    }
    assert!(same_state(&direct, &parsed) < 1e-12);
}

#[test]
fn modifiers() {
    let pairs = [
        ("ctrl @ x q[0], q[1];", "cx q[0], q[1];"),
        ("inv @ s q[1];", "sdg q[1];"),
        ("pow(2) @ s q[0];", "z q[0];"),
        ("pow(-1) @ t q[0];", "tdg q[0];"),
        ("ctrl(2) @ x q[0], q[1], q[2];", "ccx q[0], q[1], q[2];"),
        ("negctrl @ x q[2], q[0];", "x q[2]; cx q[2], q[0]; x q[2];"),
        ("ctrl @ inv @ rz(0.3) q[1], q[2];", "crz(-0.3) q[1], q[2];"),
        ("ctrl @ swap q[0], q[1], q[2];", "cswap q[0], q[1], q[2];"),
    ];
    let mut rng = common::rng(2);
    for (with, without) in pairs {
        let head = "OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[3] q;\n";
        let mut prefix = Circuit::new(3);
        prefix.unitary(common::random_unitary(&mut rng, 3), &[0, 1, 2]);
        let build = |body: &str| {
            let mut c = prefix.clone();
            c.instructions
                .extend(parse(&format!("{head}{body}")).unwrap().instructions);
            c
        };
        assert!(same_state(&build(with), &build(without)) < 1e-12, "{with}");
    }
    // Modifiers on a defined gate.
    let defined = parse(
        "OPENQASM 3.0;\ninclude \"stdgates.inc\";\ngate g(a) x, y { rx(a) x; cz x, y; }\nqubit[3] q;\nctrl @ inv @ g(0.7) q[2], q[0], q[1];",
    )
    .unwrap();
    let Instruction::Gate {
        gate: Gate::Unitary(m),
        qubits,
    } = &defined.instructions[0]
    else {
        panic!("expected a unitary")
    };
    assert_eq!(qubits, &[2, 0, 1]);
    let mut inner = Circuit::new(2);
    inner.rx(0.7, 0).cz(0, 1);
    let g = unitary(&inner);
    let expected = g.adjoint().controlled(1, 1);
    assert!(m.distance(&expected) < 1e-12);
}

fn unitary(circuit: &Circuit) -> qvd::Matrix {
    let dim = 1 << circuit.num_qubits;
    let mut entries = vec![C64::new(0.0, 0.0); dim * dim];
    for column in 0..dim {
        let mut state = vec![C64::new(0.0, 0.0); dim];
        state[column] = C64::new(1.0, 0.0);
        for instruction in &circuit.instructions {
            if let Instruction::Gate { gate, qubits } = instruction {
                reference::apply(&mut state, &gate.matrix(), qubits);
            }
        }
        for (row, x) in state.into_iter().enumerate() {
            entries[row * dim + column] = x;
        }
    }
    qvd::Matrix::new(entries)
}

#[test]
fn broadcasting_ranges_resets_and_barriers() {
    let parsed = parse(
        "OPENQASM 3;\nqubit[4] q;\nbit[4] c;\nh q;\ncx q[0:1], q[2:3];\nbarrier;\nreset q[0:2:2];\nc = measure q;",
    )
    .unwrap();
    let kinds: Vec<String> = parsed
        .instructions
        .iter()
        .map(|i| match i {
            Instruction::Gate { gate, qubits } => format!("{gate:?}{qubits:?}"),
            Instruction::Measure { qubit, clbit } => format!("m{qubit}{clbit}"),
            Instruction::Reset { qubit } => format!("r{qubit}"),
            Instruction::Barrier { qubits } => format!("b{qubits:?}"),
            Instruction::Noise { .. } | Instruction::Feedback { .. } => {
                unreachable!("not in this program")
            }
        })
        .collect();
    assert_eq!(
        kinds,
        [
            "H[0]",
            "H[1]",
            "H[2]",
            "H[3]",
            "CX[0, 2]",
            "CX[1, 3]",
            "b[0, 1, 2, 3]",
            "r0",
            "r2",
            "m00",
            "m11",
            "m22",
            "m33"
        ]
    );
}

#[test]
fn errors_point_at_the_problem() {
    let cases = [
        (
            "OPENQASM 2.0;\nqreg q[2];\nfoo q[0];",
            3,
            "unknown gate `foo`",
        ),
        ("OPENQASM 2.0;\nqreg q[2];\nh q[5];", 3, "out of range"),
        ("OPENQASM 2.0;\nqreg q[2];\ncx q[0];", 3, "acts on 2 qubits"),
        (
            "OPENQASM 2.0;\nqreg q[2];\nrx q[0];",
            3,
            "takes 1 parameters",
        ),
        (
            "OPENQASM 2.0;\nqreg q[2];\ncreg c[2];\nif (c == 1) x q[0];",
            4,
            "a register of 2 bits, is not supported",
        ),
        (
            "OPENQASM 3.0;\nqubit[2] q;\nbit[1] c;\nif (c[0]) h q[0];",
            4,
            "`h` inside `if` is not supported",
        ),
        (
            "OPENQASM 3.0;\nqubit[2] q;\nbit[1] c;\nif (c[0]) { x q[0];\n measure q[1] -> c[0]; }",
            5,
            "`measure` inside `if` is not supported",
        ),
        (
            "OPENQASM 3.0;\nqubit[2] q;\nbit[1] c;\nif (c[0] == 2) x q[0];",
            4,
            "only be compared with 0, 1, true or false",
        ),
        (
            "OPENQASM 3.0;\nqubit[2] q;\nbit[1] c;\nelse x q[0];",
            4,
            "`else` is not supported",
        ),
        (
            "OPENQASM 3.0;\nqubit[2] q;\nfor int i in [0:1] { h q[i]; }",
            3,
            "`for` is not supported",
        ),
        (
            "OPENQASM 2.0;\nqreg q[2];\ncx q[0], q[0];",
            3,
            "same qubit twice",
        ),
        (
            "OPENQASM 2.0;\ninclude \"mylib.inc\";",
            2,
            "only qelib1.inc and stdgates.inc",
        ),
        (
            "OPENQASM 2.0;\nqreg q[2];\nh r[0];",
            3,
            "unknown qubit register `r`",
        ),
        (
            "OPENQASM 2.0;\nqreg q[1];\nrz(theta) q[0];",
            3,
            "unknown parameter `theta`",
        ),
    ];
    for (program, line, message) in cases {
        let error = parse(program).unwrap_err();
        assert_eq!(error.line, line, "{program:?}: {error}");
        assert!(error.message.contains(message), "{program:?}: {error}");
    }
}

#[test]
fn relative_phase_toffolis_follow_qelib1_and_can_be_redefined() {
    let parsed =
        parse("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[3];\nrccx q[0], q[1], q[2];")
            .unwrap();
    let mut direct = Circuit::new(3);
    let quarter = std::f64::consts::FRAC_PI_4;
    let h = Gate::U(std::f64::consts::FRAC_PI_2, 0.0, std::f64::consts::PI);
    direct
        .gate(h.clone(), &[2])
        .gate(Gate::P(quarter), &[2])
        .cx(1, 2)
        .gate(Gate::P(-quarter), &[2])
        .cx(0, 2);
    direct
        .gate(Gate::P(quarter), &[2])
        .cx(1, 2)
        .gate(Gate::P(-quarter), &[2])
        .gate(h, &[2]);
    assert!(same_state(&direct, &parsed) < 1e-12);
    // A program's own definition replaces the built-in one.
    let redefined =
        parse("OPENQASM 2.0;\ngate rccx a,b,c { ccx a,b,c; }\nqreg q[3];\nrccx q[0], q[1], q[2];")
            .unwrap();
    assert_eq!(redefined.instructions.len(), 1);
}

/// What a test `if` does when its bit reads 1 (`then`) or 0 (`otherwise`),
/// as Pauli gates applied literally.
#[derive(Clone, Debug)]
enum Step {
    Gate(Gate, Vec<usize>),
    Measure(usize, usize),
    If {
        clbit: usize,
        on_one: Vec<(Gate, usize)>,
        on_zero: Vec<(Gate, usize)>,
    },
}

/// The exact distribution of the classical bits of `steps`, applying each
/// `if`'s body as written (not as feedback), branching on measurements.
fn literal_distribution(n: usize, bits: usize, steps: &[Step]) -> BTreeMap<String, f64> {
    fn walk(
        steps: &[Step],
        mut state: Vec<C64>,
        mut clbits: Vec<bool>,
        weight: f64,
        out: &mut BTreeMap<String, f64>,
    ) {
        for (i, step) in steps.iter().enumerate() {
            match step {
                Step::Gate(gate, qubits) => reference::apply(&mut state, &gate.matrix(), qubits),
                Step::If {
                    clbit,
                    on_one,
                    on_zero,
                } => {
                    let body = if clbits[*clbit] { on_one } else { on_zero };
                    for (gate, q) in body {
                        reference::apply(&mut state, &gate.matrix(), &[*q]);
                    }
                }
                &Step::Measure(qubit, clbit) => {
                    for outcome in [false, true] {
                        let mut branch = state.clone();
                        let p = project(&mut branch, qubit, outcome);
                        if p < 1e-12 {
                            continue;
                        }
                        clbits[clbit] = outcome;
                        walk(&steps[i + 1..], branch, clbits.clone(), weight * p, out);
                    }
                    return;
                }
            }
        }
        let key: String = clbits
            .iter()
            .rev()
            .map(|&b| if b { '1' } else { '0' })
            .collect();
        *out.entry(key).or_default() += weight;
    }
    let mut state = vec![C64::new(0.0, 0.0); 1 << n];
    state[0] = C64::new(1.0, 0.0);
    let mut out = BTreeMap::new();
    walk(steps, state, vec![false; bits], 1.0, &mut out);
    out
}

/// A random OpenQASM 3 program of Clifford gates, mid-circuit measurements
/// and `if`s with Pauli bodies (with and without `else`, braces, `!`,
/// `==`, `!=`, `true`/`false`, a broadcast `x q;`), and its steps.
fn random_conditional_program(rng: &mut impl Rng, n: usize, bits: usize) -> (String, Vec<Step>) {
    let mut text =
        format!("OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[{n}] q;\nbit[{bits}] c;\n");
    let mut steps = Vec::new();
    let one = [
        ("h", Gate::H),
        ("s", Gate::S),
        ("sdg", Gate::Sdg),
        ("sx", Gate::SX),
        ("x", Gate::X),
    ];
    let paulis = [
        ("x", Gate::X),
        ("y", Gate::Y),
        ("z", Gate::Z),
        ("id", Gate::I),
    ];
    let body = |rng: &mut dyn rand::Rng, text: &mut String| -> Vec<(Gate, usize)> {
        let mut out = Vec::new();
        let count = rng.random_range(1..4usize);
        if count == 1 && rng.random_bool(0.5) {
            let (name, gate) = paulis[rng.random_range(0..4usize)].clone();
            let q = rng.random_range(0..n);
            text.push_str(&format!("{name} q[{q}];\n"));
            return vec![(gate, q)];
        }
        text.push_str("{\n");
        for _ in 0..count {
            if rng.random_bool(0.15) {
                // Broadcast over the register.
                text.push_str("  x q;\n");
                out.extend((0..n).map(|q| (Gate::X, q)));
            } else {
                let (name, gate) = paulis[rng.random_range(0..4usize)].clone();
                let q = rng.random_range(0..n);
                text.push_str(&format!("  {name} q[{q}];\n"));
                out.push((gate, q));
            }
        }
        text.push_str("}\n");
        out
    };
    for _ in 0..rng.random_range(6..16usize) {
        match rng.random_range(0..10) {
            0..=3 => {
                let (name, gate) = one[rng.random_range(0..one.len())].clone();
                let q = rng.random_range(0..n);
                text.push_str(&format!("{name} q[{q}];\n"));
                steps.push(Step::Gate(gate, vec![q]));
            }
            4 | 5 if n > 1 => {
                let a = rng.random_range(0..n);
                let b = (a + rng.random_range(1..n)) % n;
                let (name, gate) = if rng.random_bool(0.5) {
                    ("cx", Gate::CX)
                } else {
                    ("cz", Gate::CZ)
                };
                text.push_str(&format!("{name} q[{a}], q[{b}];\n"));
                steps.push(Step::Gate(gate, vec![a, b]));
            }
            6 | 7 => {
                let (q, k) = (rng.random_range(0..n), rng.random_range(0..bits));
                if rng.random_bool(0.5) {
                    text.push_str(&format!("c[{k}] = measure q[{q}];\n"));
                } else {
                    text.push_str(&format!("measure q[{q}] -> c[{k}];\n"));
                }
                steps.push(Step::Measure(q, k));
            }
            _ => {
                let k = rng.random_range(0..bits);
                // The condition and whether it selects the first body on 1.
                let (condition, on_one_first) = match rng.random_range(0..6) {
                    0 => (format!("c[{k}]"), true),
                    1 => (format!("!c[{k}]"), false),
                    2 => (format!("c[{k}] == 1"), true),
                    3 => (format!("c[{k}] == 0"), false),
                    4 => (format!("c[{k}] != true"), false),
                    _ => (format!("c[{k}] == false"), false),
                };
                text.push_str(&format!("if ({condition}) "));
                let first = body(rng, &mut text);
                let second = if rng.random_bool(0.5) {
                    text.push_str("else ");
                    body(rng, &mut text)
                } else {
                    Vec::new()
                };
                let (on_one, on_zero) = if on_one_first {
                    (first, second)
                } else {
                    (second, first)
                };
                steps.push(Step::If {
                    clbit: k,
                    on_one,
                    on_zero,
                });
            }
        }
    }
    // Read every qubit at the end.
    for q in 0..n {
        let k = q % bits;
        text.push_str(&format!("measure q[{q}] -> c[{k}];\n"));
        steps.push(Step::Measure(q, k));
    }
    (text, steps)
}

#[test]
fn conditional_paulis_become_feedback() {
    let program = "OPENQASM 3.0;\nqubit[3] q;\nbit[2] c;\n\
        c[1] = measure q[0];\n\
        if (c[1]) x q[1];\n\
        if (!c[0]) { z q[0]; y q[2]; }\n\
        if (c[1] == 1) z q[2]; else x q[2];\n\
        if (c[0]) { x q[0]; id q[1]; x q[0]; }\n";
    let parsed = parse(program).unwrap();
    let feedback = |pauli, qubit, clbit| Instruction::Feedback {
        pauli,
        qubit,
        clbit,
    };
    let gate = |gate, qubit| Instruction::Gate {
        gate,
        qubits: vec![qubit],
    };
    assert_eq!(
        parsed.instructions,
        vec![
            Instruction::Measure { qubit: 0, clbit: 1 },
            feedback(Pauli::X, 1, 1),
            // On 0 the block, on 1 nothing: the block always, then again on 1.
            gate(Gate::Z, 0),
            gate(Gate::Y, 2),
            feedback(Pauli::Z, 0, 0),
            feedback(Pauli::Y, 2, 0),
            // The else body always, then Z·X = Y on 1.
            gate(Gate::X, 2),
            feedback(Pauli::Y, 2, 1),
            // X twice and the identity: nothing at all.
        ]
    );
    assert!(parsed.has_feedback());
    // OpenQASM 2 on a one-bit register, as Qiskit's exporter writes it.
    let parsed = parse(
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncreg a[1];\ncreg b[1];\n\
         measure q[0] -> b[0];\nif(b==1) x q[1];\nif(b==0) z q;\n",
    )
    .unwrap();
    assert_eq!(
        parsed.instructions[1..],
        [
            feedback(Pauli::X, 1, 1),
            gate(Gate::Z, 0),
            gate(Gate::Z, 1),
            feedback(Pauli::Z, 0, 1),
            feedback(Pauli::Z, 1, 1),
        ]
    );
    // A gate defined as Paulis is a Pauli body too.
    let parsed = parse(
        "OPENQASM 3.0;\nqubit[2] q;\nbit c;\ngate flip a, b { x a; z b; }\n\
         c = measure q[0];\nif (c) flip q[0], q[1];\n",
    )
    .unwrap();
    assert_eq!(
        parsed.instructions[1..],
        [feedback(Pauli::X, 0, 0), feedback(Pauli::Z, 1, 0)]
    );
}

#[test]
fn conditional_programs_match_their_literal_meaning() {
    // The feedback the reader writes (the 0-body always, then the product
    // of both bodies on 1) against the `if`s applied as written, on the
    // exact distribution of every classical bit, then sampled on the
    // stabilizer backend and the state vector.
    let mut rng = common::rng(77);
    for case in 0..60 {
        let n = 1 + case % 4;
        let bits = 1 + case % 3;
        let (program, steps) = random_conditional_program(&mut rng, n, bits);
        let parsed = parse(&program).unwrap_or_else(|e| panic!("case {case}: {e}\n{program}"));
        let literal = literal_distribution(n, bits, &steps);
        let exact = exact_distribution(&parsed);
        for key in literal.keys().chain(exact.keys()) {
            let (a, b) = (
                literal.get(key).copied().unwrap_or(0.0),
                exact.get(key).copied().unwrap_or(0.0),
            );
            assert!(
                (a - b).abs() < 1e-9,
                "case {case} {key}: literal {a}, read {b}\n{program}"
            );
        }
        if case % 4 != 0 {
            continue;
        }
        for backend in [Backend::Stabilizer, Backend::StateVector] {
            let shots = 20_000;
            let options = Options {
                backend,
                seed: Some(500 + case as u64),
                ..Options::default()
            };
            let result = qvd::run::<f64>(&parsed, shots, &options).unwrap();
            for (key, &count) in &result.counts {
                assert!(
                    literal.get(key).is_some_and(|&p| p > 1e-12),
                    "case {case} {backend:?}: impossible {key}"
                );
                let p = literal[key].min(1.0);
                let sigma = (p * (1.0 - p) / shots as f64).sqrt();
                let observed = count as f64 / shots as f64;
                assert!(
                    (observed - p).abs() <= 5.0 * sigma + 1e-9,
                    "case {case} {backend:?} {key}: {observed} against {p}"
                );
            }
        }
    }
}

#[test]
fn teleportation_with_if_statements() {
    // Teleport |+i> (S H |0>) from q[0] to q[2] with OpenQASM 3 `if`s, then
    // undo the preparation: q[2] reads 0 in every shot, on the backend that
    // `run` chooses (the stabilizer, since every gate is Clifford).
    let program = "OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[3] q;\nbit[3] c;\n\
        h q[0]; s q[0];\n\
        h q[1]; cx q[1], q[2];\n\
        cx q[0], q[1]; h q[0];\n\
        c[0] = measure q[0];\nc[1] = measure q[1];\n\
        if (c[1]) x q[2];\nif (c[0]) z q[2];\n\
        sdg q[2]; h q[2];\nc[2] = measure q[2];\n";
    let parsed = parse(program).unwrap();
    assert!(qvd::stabilizer::is_clifford(&parsed));
    let result = qvd::run::<f64>(
        &parsed,
        4000,
        &Options {
            seed: Some(3),
            ..Options::default()
        },
    )
    .unwrap();
    assert_eq!(result.stats.backend, Some(Backend::Stabilizer));
    assert_eq!(result.counts.len(), 4, "{:?}", result.counts);
    assert!(
        result.counts.keys().all(|k| k.starts_with('0')),
        "{:?}",
        result.counts
    );
}
