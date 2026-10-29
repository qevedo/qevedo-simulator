//! Circuits and the standard gate set.
//!
//! Gate definitions follow Qiskit exactly (including the sign conventions of
//! the rotations and the phase of `U`), so results can be compared with
//! Qiskit Aer amplitude by amplitude.

use std::f64::consts::FRAC_1_SQRT_2;

use crate::matrix::{C64, Matrix};

/// A gate. Multi-qubit gates list their qubits as Qiskit does: for `CX`,
/// `[control, target]`; for `CCX`, `[control, control, target]`.
#[derive(Clone, Debug, PartialEq)]
pub enum Gate {
    I,
    H,
    X,
    Y,
    Z,
    S,
    Sdg,
    T,
    Tdg,
    SX,
    SXdg,
    RX(f64),
    RY(f64),
    RZ(f64),
    /// Phase gate `diag(1, e^{iλ})`.
    P(f64),
    /// `U(θ, φ, λ)`, Qiskit's general single-qubit gate.
    U(f64, f64, f64),
    CX,
    CY,
    CZ,
    CH,
    CP(f64),
    CRX(f64),
    CRY(f64),
    CRZ(f64),
    SWAP,
    /// `exp(-i θ/2 Z⊗Z)`.
    RZZ(f64),
    /// `exp(-i θ/2 X⊗X)`.
    RXX(f64),
    CCX,
    CSWAP,
    /// An arbitrary unitary on the gate's qubits (index bit j = qubit j).
    Unitary(Matrix),
}

impl Gate {
    /// The gate's name in OpenQASM 2's `qelib1.inc` and its parameters, or
    /// `None` for [`Gate::Unitary`].
    pub fn qasm_name(&self) -> Option<(&'static str, Vec<f64>)> {
        use Gate::*;
        Some(match *self {
            I => ("id", vec![]),
            H => ("h", vec![]),
            X => ("x", vec![]),
            Y => ("y", vec![]),
            Z => ("z", vec![]),
            S => ("s", vec![]),
            Sdg => ("sdg", vec![]),
            T => ("t", vec![]),
            Tdg => ("tdg", vec![]),
            SX => ("sx", vec![]),
            SXdg => ("sxdg", vec![]),
            RX(t) => ("rx", vec![t]),
            RY(t) => ("ry", vec![t]),
            RZ(t) => ("rz", vec![t]),
            P(l) => ("u1", vec![l]),
            U(a, b, c) => ("u3", vec![a, b, c]),
            CX => ("cx", vec![]),
            CY => ("cy", vec![]),
            CZ => ("cz", vec![]),
            CH => ("ch", vec![]),
            CP(l) => ("cu1", vec![l]),
            CRX(t) => ("crx", vec![t]),
            CRY(t) => ("cry", vec![t]),
            CRZ(t) => ("crz", vec![t]),
            SWAP => ("swap", vec![]),
            RZZ(t) => ("rzz", vec![t]),
            RXX(t) => ("rxx", vec![t]),
            CCX => ("ccx", vec![]),
            CSWAP => ("cswap", vec![]),
            Unitary(_) => return None,
        })
    }

    /// Number of qubits the gate acts on.
    pub fn arity(&self) -> usize {
        use Gate::*;
        match self {
            I | H | X | Y | Z | S | Sdg | T | Tdg | SX | SXdg | RX(_) | RY(_) | RZ(_) | P(_)
            | U(..) => 1,
            CX | CY | CZ | CH | CP(_) | CRX(_) | CRY(_) | CRZ(_) | SWAP | RZZ(_) | RXX(_) => 2,
            CCX | CSWAP => 3,
            Unitary(m) => m.qubits(),
        }
    }

    /// Whether the gate's matrix is diagonal in the computational basis.
    pub fn is_diagonal(&self) -> bool {
        use Gate::*;
        match self {
            I | Z | S | Sdg | T | Tdg | RZ(_) | P(_) | CZ | CP(_) | CRZ(_) | RZZ(_) => true,
            Unitary(m) => m.is_diagonal(),
            _ => false,
        }
    }

    /// For controlled gates, the number of leading control qubits and the
    /// gate applied to the remaining qubits.
    pub fn controlled_form(&self) -> Option<(usize, Gate)> {
        use Gate::*;
        Some(match self {
            CX => (1, X),
            CY => (1, Y),
            CZ => (1, Z),
            CH => (1, H),
            CP(l) => (1, P(*l)),
            CRX(t) => (1, RX(*t)),
            CRY(t) => (1, RY(*t)),
            CRZ(t) => (1, RZ(*t)),
            CCX => (2, X),
            CSWAP => (1, SWAP),
            _ => return None,
        })
    }

    /// The gate's unitary matrix (index bit j = qubit j of the gate).
    pub fn matrix(&self) -> Matrix {
        use Gate::*;
        let c = |re: f64, im: f64| C64::new(re, im);
        let z = c(0.0, 0.0);
        let one = c(1.0, 0.0);
        let phase = |angle: f64| C64::from_polar(1.0, angle);
        if let Some((controls, target)) = self.controlled_form() {
            return target.matrix().controlled(controls, (1 << controls) - 1);
        }
        match self {
            I => Matrix::identity(1),
            H => Matrix::new(vec![
                c(FRAC_1_SQRT_2, 0.),
                c(FRAC_1_SQRT_2, 0.),
                c(FRAC_1_SQRT_2, 0.),
                c(-FRAC_1_SQRT_2, 0.),
            ]),
            X => Matrix::new(vec![z, one, one, z]),
            Y => Matrix::new(vec![z, c(0., -1.), c(0., 1.), z]),
            Z => Matrix::diagonal(&[one, c(-1., 0.)]),
            S => Matrix::diagonal(&[one, c(0., 1.)]),
            Sdg => Matrix::diagonal(&[one, c(0., -1.)]),
            T => Matrix::diagonal(&[one, phase(std::f64::consts::FRAC_PI_4)]),
            Tdg => Matrix::diagonal(&[one, phase(-std::f64::consts::FRAC_PI_4)]),
            SX => Matrix::new(vec![c(0.5, 0.5), c(0.5, -0.5), c(0.5, -0.5), c(0.5, 0.5)]),
            SXdg => Matrix::new(vec![c(0.5, -0.5), c(0.5, 0.5), c(0.5, 0.5), c(0.5, -0.5)]),
            RX(t) => {
                let (s, co) = (t / 2.0).sin_cos();
                Matrix::new(vec![c(co, 0.), c(0., -s), c(0., -s), c(co, 0.)])
            }
            RY(t) => {
                let (s, co) = (t / 2.0).sin_cos();
                Matrix::new(vec![c(co, 0.), c(-s, 0.), c(s, 0.), c(co, 0.)])
            }
            RZ(t) => Matrix::diagonal(&[phase(-t / 2.0), phase(t / 2.0)]),
            P(l) => Matrix::diagonal(&[one, phase(*l)]),
            U(theta, phi, lambda) => {
                let (s, co) = (theta / 2.0).sin_cos();
                Matrix::new(vec![
                    c(co, 0.),
                    -phase(*lambda) * s,
                    phase(*phi) * s,
                    phase(phi + lambda) * co,
                ])
            }
            SWAP => Matrix::new(vec![
                one, z, z, z, //
                z, z, one, z, //
                z, one, z, z, //
                z, z, z, one,
            ]),
            RZZ(t) => {
                let a = phase(-t / 2.0);
                let b = phase(t / 2.0);
                Matrix::diagonal(&[a, b, b, a])
            }
            RXX(t) => {
                let (s, co) = (t / 2.0).sin_cos();
                let (cc, ms) = (c(co, 0.), c(0., -s));
                Matrix::new(vec![
                    cc, z, z, ms, //
                    z, cc, ms, z, //
                    z, ms, cc, z, //
                    ms, z, z, cc,
                ])
            }
            Unitary(m) => m.clone(),
            CX | CY | CZ | CH | CP(_) | CRX(_) | CRY(_) | CRZ(_) | CCX | CSWAP => unreachable!(),
        }
    }
}

/// One step of a circuit.
#[derive(Clone, Debug, PartialEq)]
pub enum Instruction {
    Gate {
        gate: Gate,
        qubits: Vec<usize>,
    },
    /// Measure `qubit` into classical bit `clbit`.
    Measure {
        qubit: usize,
        clbit: usize,
    },
    Reset {
        qubit: usize,
    },
    Barrier {
        qubits: Vec<usize>,
    },
    /// A Pauli noise channel on `qubits` (for [`Noise::Depolarize2`], on
    /// consecutive pairs). Only the stabilizer backend runs noisy circuits.
    Noise {
        channel: Noise,
        qubits: Vec<usize>,
    },
}

/// A Pauli noise channel: with some probability a random Pauli error,
/// drawn independently in every shot. These are the channels of Stim's
/// circuit-level noise models, and the stabilizer backend samples them
/// exactly by adding the errors to each shot's Pauli frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Noise {
    /// On each qubit independently: X with probability `px`, Y with `py`,
    /// Z with `pz` (`X_ERROR`, `Y_ERROR`, `Z_ERROR`, `DEPOLARIZE1` and
    /// `PAULI_CHANNEL_1` in Stim).
    Pauli1 { px: f64, py: f64, pz: f64 },
    /// On each pair of qubits: each of the 15 non-identity two-qubit Paulis
    /// with probability `p / 15` (`DEPOLARIZE2` in Stim).
    Depolarize2(f64),
}

impl Noise {
    /// The probability that the channel applies any error (to one qubit,
    /// or to one pair for `Depolarize2`).
    pub fn probability(&self) -> f64 {
        match *self {
            Noise::Pauli1 { px, py, pz } => px + py + pz,
            Noise::Depolarize2(p) => p,
        }
    }

    /// Qubits per target: 1, or 2 for `Depolarize2`.
    pub fn arity(&self) -> usize {
        match self {
            Noise::Pauli1 { .. } => 1,
            Noise::Depolarize2(_) => 2,
        }
    }

    /// Stim's name and arguments for the channel.
    pub fn stim_name(&self) -> (&'static str, Vec<f64>) {
        match *self {
            Noise::Pauli1 { px, py, pz } => match (px, py, pz) {
                (p, 0.0, 0.0) => ("X_ERROR", vec![p]),
                (0.0, p, 0.0) => ("Y_ERROR", vec![p]),
                (0.0, 0.0, p) => ("Z_ERROR", vec![p]),
                _ if px == py && py == pz => ("DEPOLARIZE1", vec![3.0 * px]),
                _ => ("PAULI_CHANNEL_1", vec![px, py, pz]),
            },
            Noise::Depolarize2(p) => ("DEPOLARIZE2", vec![p]),
        }
    }
}

/// A quantum circuit on `num_qubits` qubits and `num_clbits` classical bits.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Circuit {
    pub num_qubits: usize,
    pub num_clbits: usize,
    pub instructions: Vec<Instruction>,
}

macro_rules! single_qubit_methods {
    ($($name:ident => $gate:ident),* $(,)?) => {
        $(
            pub fn $name(&mut self, qubit: usize) -> &mut Self {
                self.gate(Gate::$gate, &[qubit])
            }
        )*
    };
}

impl Circuit {
    pub fn new(num_qubits: usize) -> Circuit {
        Circuit {
            num_qubits,
            num_clbits: 0,
            instructions: Vec::new(),
        }
    }

    /// Append a gate. Panics if a qubit is out of range or repeated.
    pub fn gate(&mut self, gate: Gate, qubits: &[usize]) -> &mut Self {
        assert_eq!(
            gate.arity(),
            qubits.len(),
            "{gate:?} acts on {} qubits",
            gate.arity()
        );
        for (i, &q) in qubits.iter().enumerate() {
            assert!(q < self.num_qubits, "qubit {q} out of range");
            assert!(!qubits[..i].contains(&q), "qubit {q} used twice");
        }
        self.instructions.push(Instruction::Gate {
            gate,
            qubits: qubits.to_vec(),
        });
        self
    }

    single_qubit_methods! {
        h => H, x => X, y => Y, z => Z, s => S, sdg => Sdg, t => T, tdg => Tdg, sx => SX,
    }

    pub fn rx(&mut self, theta: f64, qubit: usize) -> &mut Self {
        self.gate(Gate::RX(theta), &[qubit])
    }

    pub fn ry(&mut self, theta: f64, qubit: usize) -> &mut Self {
        self.gate(Gate::RY(theta), &[qubit])
    }

    pub fn rz(&mut self, theta: f64, qubit: usize) -> &mut Self {
        self.gate(Gate::RZ(theta), &[qubit])
    }

    pub fn p(&mut self, lambda: f64, qubit: usize) -> &mut Self {
        self.gate(Gate::P(lambda), &[qubit])
    }

    pub fn u(&mut self, theta: f64, phi: f64, lambda: f64, qubit: usize) -> &mut Self {
        self.gate(Gate::U(theta, phi, lambda), &[qubit])
    }

    pub fn cx(&mut self, control: usize, target: usize) -> &mut Self {
        self.gate(Gate::CX, &[control, target])
    }

    pub fn cz(&mut self, a: usize, b: usize) -> &mut Self {
        self.gate(Gate::CZ, &[a, b])
    }

    pub fn cp(&mut self, lambda: f64, control: usize, target: usize) -> &mut Self {
        self.gate(Gate::CP(lambda), &[control, target])
    }

    pub fn swap(&mut self, a: usize, b: usize) -> &mut Self {
        self.gate(Gate::SWAP, &[a, b])
    }

    pub fn ccx(&mut self, a: usize, b: usize, target: usize) -> &mut Self {
        self.gate(Gate::CCX, &[a, b, target])
    }

    pub fn unitary(&mut self, matrix: Matrix, qubits: &[usize]) -> &mut Self {
        self.gate(Gate::Unitary(matrix), qubits)
    }

    pub fn measure(&mut self, qubit: usize, clbit: usize) -> &mut Self {
        assert!(qubit < self.num_qubits);
        self.num_clbits = self.num_clbits.max(clbit + 1);
        self.instructions
            .push(Instruction::Measure { qubit, clbit });
        self
    }

    pub fn measure_all(&mut self) -> &mut Self {
        for q in 0..self.num_qubits {
            self.measure(q, q);
        }
        self
    }

    pub fn reset(&mut self, qubit: usize) -> &mut Self {
        assert!(qubit < self.num_qubits);
        self.instructions.push(Instruction::Reset { qubit });
        self
    }

    /// Add a noise channel on `qubits` (pairs of qubits for
    /// [`Noise::Depolarize2`]).
    pub fn noise(&mut self, channel: Noise, qubits: &[usize]) -> &mut Self {
        let probabilities = match channel {
            Noise::Pauli1 { px, py, pz } => vec![px, py, pz],
            Noise::Depolarize2(p) => vec![p],
        };
        assert!(
            probabilities.iter().all(|p| (0.0..=1.0).contains(p)) && channel.probability() <= 1.0,
            "noise probabilities must lie in [0, 1] and sum to at most 1: {channel:?}"
        );
        assert!(qubits.iter().all(|&q| q < self.num_qubits));
        assert!(
            qubits.len().is_multiple_of(channel.arity()),
            "{channel:?} acts on pairs of qubits"
        );
        if channel.arity() == 2 {
            assert!(
                qubits.chunks(2).all(|p| p[0] != p[1]),
                "the qubits of a pair must differ"
            );
        }
        self.instructions.push(Instruction::Noise {
            channel,
            qubits: qubits.to_vec(),
        });
        self
    }

    /// A bit flip (X) with probability `p`.
    pub fn x_error(&mut self, qubit: usize, p: f64) -> &mut Self {
        self.noise(
            Noise::Pauli1 {
                px: p,
                py: 0.0,
                pz: 0.0,
            },
            &[qubit],
        )
    }

    /// A Y error with probability `p`.
    pub fn y_error(&mut self, qubit: usize, p: f64) -> &mut Self {
        self.noise(
            Noise::Pauli1 {
                px: 0.0,
                py: p,
                pz: 0.0,
            },
            &[qubit],
        )
    }

    /// A phase flip (Z) with probability `p`.
    pub fn z_error(&mut self, qubit: usize, p: f64) -> &mut Self {
        self.noise(
            Noise::Pauli1 {
                px: 0.0,
                py: 0.0,
                pz: p,
            },
            &[qubit],
        )
    }

    /// X, Y or Z, each with probability `p / 3`.
    pub fn depolarize1(&mut self, qubit: usize, p: f64) -> &mut Self {
        let q = p / 3.0;
        self.noise(
            Noise::Pauli1 {
                px: q,
                py: q,
                pz: q,
            },
            &[qubit],
        )
    }

    /// Each non-identity two-qubit Pauli with probability `p / 15`.
    pub fn depolarize2(&mut self, a: usize, b: usize, p: f64) -> &mut Self {
        self.noise(Noise::Depolarize2(p), &[a, b])
    }

    /// Whether the circuit contains noise channels.
    pub fn has_noise(&self) -> bool {
        self.instructions
            .iter()
            .any(|i| matches!(i, Instruction::Noise { .. }))
    }

    pub fn barrier(&mut self) -> &mut Self {
        self.instructions.push(Instruction::Barrier {
            qubits: (0..self.num_qubits).collect(),
        });
        self
    }

    /// The circuit as OpenQASM 2 using `qelib1.inc` gate names, so other
    /// simulators can run exactly the same circuit. Fails on `Unitary` gates
    /// and noise channels.
    pub fn to_qasm(&self) -> Result<String, String> {
        use std::fmt::Write;
        let mut out = String::from("OPENQASM 2.0;\ninclude \"qelib1.inc\";\n");
        let _ = writeln!(out, "qreg q[{}];", self.num_qubits);
        if self.num_clbits > 0 {
            let _ = writeln!(out, "creg c[{}];", self.num_clbits);
        }
        let angle = |x: f64| format!("{x:.17}");
        for instruction in &self.instructions {
            match instruction {
                Instruction::Gate { gate, qubits } => {
                    let (name, params) = gate
                        .qasm_name()
                        .ok_or("arbitrary unitaries cannot be written as OpenQASM 2")?;
                    let params = if params.is_empty() {
                        String::new()
                    } else {
                        format!(
                            "({})",
                            params
                                .iter()
                                .map(|&x| angle(x))
                                .collect::<Vec<_>>()
                                .join(",")
                        )
                    };
                    let operands: Vec<String> = qubits.iter().map(|q| format!("q[{q}]")).collect();
                    let _ = writeln!(out, "{name}{params} {};", operands.join(","));
                }
                Instruction::Measure { qubit, clbit } => {
                    let _ = writeln!(out, "measure q[{qubit}] -> c[{clbit}];");
                }
                Instruction::Reset { qubit } => {
                    let _ = writeln!(out, "reset q[{qubit}];");
                }
                Instruction::Barrier { .. } => out.push_str("barrier q;\n"),
                Instruction::Noise { .. } => {
                    return Err("noise channels cannot be written as OpenQASM 2".into());
                }
            }
        }
        Ok(out)
    }

    /// Number of gates (excluding measurements, resets and barriers).
    pub fn gate_count(&self) -> usize {
        self.instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Gate { .. }))
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_unitary(m: &Matrix) -> bool {
        m.mul(&m.adjoint()).distance(&Matrix::identity(m.qubits())) < 1e-12
    }

    #[test]
    fn every_gate_is_unitary() {
        use Gate::*;
        let gates = [
            I,
            H,
            X,
            Y,
            Z,
            S,
            Sdg,
            T,
            Tdg,
            SX,
            SXdg,
            RX(0.3),
            RY(0.4),
            RZ(0.5),
            P(0.6),
            U(0.1, 0.2, 0.3),
            CX,
            CY,
            CZ,
            CH,
            CP(0.7),
            CRX(0.8),
            CRY(0.9),
            CRZ(1.0),
            SWAP,
            RZZ(1.1),
            RXX(1.2),
            CCX,
            CSWAP,
        ];
        for gate in gates {
            let m = gate.matrix();
            assert_eq!(m.qubits(), gate.arity(), "{gate:?}");
            assert!(is_unitary(&m), "{gate:?} is not unitary");
            assert_eq!(
                m.is_diagonal(),
                gate.is_diagonal(),
                "{gate:?} diagonal flag"
            );
        }
    }

    #[test]
    fn sx_squared_is_x_and_u_matches_qiskit() {
        let sx = Gate::SX.matrix();
        assert!(sx.mul(&sx).distance(&Gate::X.matrix()) < 1e-12);
        // Qiskit: U(pi, 0, pi) = X and U(pi/2, 0, pi) = H.
        let pi = std::f64::consts::PI;
        assert!(Gate::U(pi, 0.0, pi).matrix().distance(&Gate::X.matrix()) < 1e-12);
        assert!(
            Gate::U(pi / 2.0, 0.0, pi)
                .matrix()
                .distance(&Gate::H.matrix())
                < 1e-12
        );
    }
}
