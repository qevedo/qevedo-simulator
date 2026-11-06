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
    /// A Pauli noise channel on `qubits` (for the two-qubit channels, on
    /// consecutive pairs; for [`Noise::Correlated`], on consecutive groups
    /// as long as its products). The stabilizer and near-Clifford backends run
    /// noisy circuits.
    Noise {
        channel: Noise,
        qubits: Vec<usize>,
    },
}

/// A Pauli noise channel: with some probability a random Pauli error,
/// drawn independently in every shot. These are the channels of Stim's
/// circuit-level noise models. The stabilizer and near-Clifford backends
/// sample them exactly by adding the errors to each shot's Pauli frame.
#[derive(Clone, Debug, PartialEq)]
pub enum Noise {
    /// On each qubit independently: X with probability `px`, Y with `py`,
    /// Z with `pz` (`X_ERROR`, `Y_ERROR`, `Z_ERROR`, `DEPOLARIZE1` and
    /// `PAULI_CHANNEL_1` in Stim).
    Pauli1 { px: f64, py: f64, pz: f64 },
    /// On each pair of qubits: each of the 15 non-identity two-qubit Paulis
    /// with probability `p / 15` (`DEPOLARIZE2` in Stim).
    Depolarize2(f64),
    /// On each pair of qubits: the 15 non-identity two-qubit Paulis with
    /// their own probabilities, in Stim's order `IX, IY, IZ, XI, XX, XY,
    /// XZ, YI, ..., ZZ`, the first letter on the first qubit of the pair
    /// (`PAULI_CHANNEL_2` in Stim). Hardware noise models are often given
    /// in this form, from Pauli twirling of a calibrated two-qubit gate.
    /// Boxed, so that `Noise` (and every [`Instruction`]) stays as small
    /// as before: a larger instruction made noiseless sampling ~4% slower.
    Pauli2(Box<[f64; 15]>),
    /// On each group of qubits (as many as the strings are long): at most
    /// one of several Pauli products, product `i` with probability `p_i`,
    /// the products written as strings of `I`, `X`, `Y` and `Z`, one letter
    /// per qubit of the group. These are Stim's correlated errors: `E(p) X0
    /// Z3` is the single product `XZ` on qubits 0 and 3, and a chain
    /// `E(p1) P1; ELSE_CORRELATED_ERROR(p2) P2; ...` is the products
    /// `P1, P2, ...` with the unconditional probabilities `p1`,
    /// `(1 - p1) p2`, ... (so the `p_i` here sum to at most 1).
    Correlated(Vec<(f64, String)>),
}

impl Noise {
    /// The probability that the channel applies any error (to one target:
    /// one qubit, or one group of [`arity`](Self::arity) qubits).
    pub fn probability(&self) -> f64 {
        match self {
            Noise::Pauli1 { px, py, pz } => px + py + pz,
            Noise::Depolarize2(p) => *p,
            Noise::Pauli2(p) => p.iter().sum(),
            Noise::Correlated(products) => products.iter().map(|(p, _)| p).sum(),
        }
    }

    /// The probabilities the channel is given by: each must lie in [0, 1].
    pub fn probabilities(&self) -> Vec<f64> {
        match self {
            Noise::Pauli1 { px, py, pz } => vec![*px, *py, *pz],
            Noise::Depolarize2(p) => vec![*p],
            Noise::Pauli2(p) => p.to_vec(),
            Noise::Correlated(products) => products.iter().map(|(p, _)| *p).collect(),
        }
    }

    /// Qubits per target: 1, 2 for the two-qubit channels, and the length
    /// of the products for `Correlated`.
    pub fn arity(&self) -> usize {
        match self {
            Noise::Pauli1 { .. } => 1,
            Noise::Depolarize2(_) | Noise::Pauli2(_) => 2,
            Noise::Correlated(products) => products.first().map_or(1, |(_, s)| s.len()),
        }
    }

    /// Stim's name and arguments for the channel. A `Correlated` channel
    /// is named `CORRELATED_ERROR` with its products' probabilities here;
    /// [`crate::stabilizer::to_stim`] writes it as a chain of `E` and
    /// `ELSE_CORRELATED_ERROR` lines.
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
            Noise::Pauli2(ref p) => ("PAULI_CHANNEL_2", p.to_vec()),
            Noise::Correlated(_) => ("CORRELATED_ERROR", self.probabilities()),
        }
    }

    /// The error on one target given that the channel applies one, for a
    /// draw `r` uniform in `[0, probability)` and `k` uniform in `1..16`
    /// (used by `Depolarize2` only): one `(x, z)` pair per qubit of the
    /// target, `(true, true)` for Y. This is the readable form of the
    /// stabilizer sampler's inner loop, which inlines the common channels
    /// for speed; the near-Clifford backend, which draws far fewer errors,
    /// uses it as it is.
    pub fn error(&self, r: f64, k: u32) -> Vec<(bool, bool)> {
        // I, X, Y, Z coded 0..3, as in Stim's PAULI_CHANNEL_2 order.
        let letter = |code: usize| (code == 1 || code == 2, code == 2 || code == 3);
        // The first weight whose cumulative sum exceeds r; rounding can leave
        // r just above the last sum, which then picks the last nonzero one.
        let pick = |weights: &mut dyn Iterator<Item = f64>| {
            let (mut acc, mut last) = (0.0, 0);
            for (i, w) in weights.enumerate() {
                acc += w;
                if w > 0.0 {
                    last = i;
                }
                if r < acc {
                    return i;
                }
            }
            last
        };
        match self {
            &Noise::Pauli1 { px, py, .. } => {
                let code = if r < px {
                    1
                } else if r < px + py {
                    2
                } else {
                    3
                };
                vec![letter(code)]
            }
            // The bits of k are (x_a, z_a, x_b, z_b), as in the sampler.
            Noise::Depolarize2(_) => vec![(k & 1 == 1, k & 2 == 2), (k & 4 == 4, k & 8 == 8)],
            Noise::Pauli2(p) => {
                let code = pick(&mut p.iter().copied()) + 1;
                vec![letter(code / 4), letter(code % 4)]
            }
            Noise::Correlated(products) => {
                let chosen = pick(&mut products.iter().map(|(p, _)| *p));
                products[chosen]
                    .1
                    .bytes()
                    .map(|c| (c == b'X' || c == b'Y', c == b'Z' || c == b'Y'))
                    .collect()
            }
        }
    }

    /// Why the channel is malformed, if it is: probabilities outside
    /// [0, 1] or summing past 1, or products that are empty, of different
    /// lengths, or not made of `I`, `X`, `Y` and `Z`.
    pub fn check(&self) -> Result<(), String> {
        let probabilities = self.probabilities();
        // A little slack for sums computed in floating point (the
        // unconditional probabilities of a long ELSE chain).
        if !probabilities.iter().all(|p| (0.0..=1.0).contains(p))
            || self.probability() > 1.0 + 1e-12
        {
            return Err(format!(
                "noise probabilities must lie in [0, 1] and sum to at most 1: {self:?}"
            ));
        }
        if let Noise::Correlated(products) = self {
            let k = self.arity();
            if products.is_empty() || k == 0 {
                return Err("a correlated error needs at least one non-empty Pauli product".into());
            }
            for (_, s) in products {
                if s.len() != k || !s.bytes().all(|c| matches!(c, b'I' | b'X' | b'Y' | b'Z')) {
                    return Err(format!(
                        "the Pauli products of a correlated error must be strings of I, X, Y and Z \
                         of the same length: {self:?}"
                    ));
                }
            }
        }
        Ok(())
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

    /// Add a noise channel on `qubits` (pairs of qubits for the two-qubit
    /// channels, groups as long as the products for [`Noise::Correlated`]).
    pub fn noise(&mut self, channel: Noise, qubits: &[usize]) -> &mut Self {
        if let Err(e) = channel.check() {
            panic!("{e}");
        }
        assert!(qubits.iter().all(|&q| q < self.num_qubits));
        let arity = channel.arity();
        assert!(
            qubits.len().is_multiple_of(arity),
            "{channel:?} acts on groups of {arity} qubits"
        );
        if arity > 1 {
            assert!(
                qubits
                    .chunks(arity)
                    .all(|g| (1..g.len()).all(|i| !g[..i].contains(&g[i]))),
                "the qubits of a group must differ"
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

    /// The 15 non-identity Paulis on qubits `a` and `b` with probabilities
    /// `p`, in Stim's order (`IX, IY, IZ, XI, ..., ZZ`, the first letter on
    /// `a`), as `PAULI_CHANNEL_2`.
    pub fn pauli_channel_2(&mut self, a: usize, b: usize, p: [f64; 15]) -> &mut Self {
        self.noise(Noise::Pauli2(Box::new(p)), &[a, b])
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

    #[test]
    fn noise_keeps_instructions_small() {
        // The stabilizer backend walks hundreds of thousands of
        // instructions per circuit; a 15-probability channel stored inline
        // tripled their size and slowed noiseless sampling down.
        assert!(std::mem::size_of::<Noise>() <= 32);
        assert!(std::mem::size_of::<Instruction>() <= 64);
    }

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
