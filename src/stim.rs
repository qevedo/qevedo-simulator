//! A reader for Stim's circuit format, with its detector annotations.
//!
//! Error-correction circuits are usually written in Stim's format (Gidney,
//! *Quantum* 5, 497, 2021): Stim's own generators, and the tools around it,
//! emit noisy Clifford circuits whose `DETECTOR` and `OBSERVABLE_INCLUDE`
//! lines say which measurement results a decoder looks at. This module
//! reads such a file into a [`StimCircuit`]: a Qevedo [`Circuit`] with one
//! classical bit per measurement, in the order Stim numbers them, plus the
//! detectors and observables as lists of those measurements.
//! [`crate::stabilizer::sample_detectors`] then samples detection events
//! and observable flips, the input of decoders such as PyMatching.
//!
//! What is read:
//! - **Gates:** `I`, `X`, `Y`, `Z`, `H` (`H_XZ`), `S` (`SQRT_Z`), `S_DAG`
//!   (`SQRT_Z_DAG`), `SQRT_X`, `SQRT_X_DAG`, `SQRT_Y`, `SQRT_Y_DAG`,
//!   `H_XY`, `H_YZ`, `C_XYZ`, `C_ZYX`, `CX`
//!   (`CNOT`, `ZCX`), `CY` (`ZCY`), `CZ` (`ZCZ`) and `SWAP`.
//! - **Measurements and resets:** `M` (`MZ`), `MX`, `MY`, `MR` (`MRZ`),
//!   `MRX`, `MRY`, `R` (`RZ`), `RX`, `RY` and `MPAD`. A measurement's
//!   argument, as in `M(0.01)`, flips the recorded result with that
//!   probability without disturbing the state; `M !q` records it inverted.
//! - **Pauli product measurements:** `MPP X0*Y1*Z2 !Z3*Z4 ...`, with a
//!   flip probability as `MPP(0.01)`. Each product becomes basis changes
//!   to Z, a ladder of CX gates onto its last qubit, a Z measurement of
//!   that qubit and the same gates undone: the measured qubit's Z is then
//!   the product, so the reference shot measures it on the tableau and the
//!   frames see its parity, as for any other measurement.
//! - **Noise:** `X_ERROR`, `Y_ERROR`, `Z_ERROR`, `DEPOLARIZE1`,
//!   `DEPOLARIZE2`, `PAULI_CHANNEL_1`, `PAULI_CHANNEL_2`, and correlated
//!   errors: `E` (`CORRELATED_ERROR`) and `ELSE_CORRELATED_ERROR`. A chain
//!   of `E` and the `ELSE_CORRELATED_ERROR` lines right after it is one
//!   [`Noise::Correlated`] channel (at most one of its products happens),
//!   with the unconditional probability of each product.
//! - **Annotations:** `DETECTOR` and `OBSERVABLE_INCLUDE` with `rec[-k]`
//!   targets; `TICK`, `QUBIT_COORDS` and `SHIFT_COORDS` are ignored
//!   (coordinates do not change what is sampled).
//! - **Blocks:** `REPEAT n { ... }`, nested, unrolled.
//!
//! Everything else (heralded errors, `MXX`/`MYY`/`MZZ` and `SPP`,
//! classically controlled gates such as `CX rec[-1] 0`, sweep bits, ...) is rejected
//! with the line it is on, rather than skipped: skipping an instruction
//! would silently sample a different circuit.

use std::f64::consts::FRAC_PI_2;

use crate::circuit::{Circuit, Gate, Instruction, Noise};

/// A circuit read from Stim's format.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StimCircuit {
    /// The circuit, with measurement `m` (counting from 0 in the order
    /// they happen, as Stim's `rec` does) written to classical bit `m`.
    pub circuit: Circuit,
    /// Measurements whose recorded result flips with a probability (from
    /// `M(p)`, `MX(p)`, ...): `(measurement, p)`.
    pub flips: Vec<(usize, f64)>,
    /// Measurements recorded inverted (`M !q`, `MPAD 1`).
    pub inverted: Vec<usize>,
    /// Each detector: the measurements whose parity it compares with the
    /// noiseless circuit's.
    pub detectors: Vec<Vec<usize>>,
    /// Each logical observable: the measurements whose parity it is. A
    /// measurement included twice cancels, as in Stim.
    pub observables: Vec<Vec<usize>>,
}

impl StimCircuit {
    /// Number of measurement results (the circuit's classical bits).
    pub fn num_measurements(&self) -> usize {
        self.circuit.num_clbits
    }
}

/// One target of an instruction.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Target {
    Qubit(usize),
    /// `!q`: a qubit whose measurement result is recorded inverted.
    Inverted(usize),
    /// `rec[-k]`: the `k`-th most recent measurement result.
    Record(usize),
    /// A Pauli on a qubit, as in `E(p) X0 Z1` and `MPP X0*Y1`: the letter
    /// (`b'X'`, `b'Y'` or `b'Z'`), the qubit, whether it was written `!X0`
    /// (an inverted result), and whether `*` joins it to the next target
    /// (one product of `MPP`).
    Pauli {
        pauli: u8,
        qubit: usize,
        inverted: bool,
        joined: bool,
    },
}

/// A parsed line, or a `REPEAT` block.
#[derive(Debug)]
enum Item {
    Op {
        line: usize,
        name: String,
        args: Vec<f64>,
        targets: Vec<Target>,
    },
    Repeat {
        count: u64,
        body: Vec<Item>,
    },
}

/// The instructions other than gates that [`Builder::op`] runs.
const OTHER_INSTRUCTIONS: &[&str] = &[
    "TICK",
    "QUBIT_COORDS",
    "SHIFT_COORDS",
    "M",
    "MZ",
    "MX",
    "MY",
    "MR",
    "MRZ",
    "MRX",
    "MRY",
    "R",
    "RZ",
    "RX",
    "RY",
    "MPAD",
    "X_ERROR",
    "Y_ERROR",
    "Z_ERROR",
    "DEPOLARIZE1",
    "PAULI_CHANNEL_1",
    "DEPOLARIZE2",
    "PAULI_CHANNEL_2",
    "E",
    "CORRELATED_ERROR",
    "ELSE_CORRELATED_ERROR",
    "MPP",
    "DETECTOR",
    "OBSERVABLE_INCLUDE",
];

/// Read a circuit in Stim's format.
pub fn parse(text: &str) -> Result<StimCircuit, String> {
    let lines: Vec<&str> = text.lines().collect();
    let mut next = 0;
    let mut max_qubit = None;
    let items = parse_block(&lines, &mut next, &mut max_qubit, None)?;
    let mut builder = Builder {
        out: StimCircuit {
            circuit: Circuit::new(max_qubit.map_or(0, |q| q + 1)),
            ..StimCircuit::default()
        },
        measurements: 0,
        chain: None,
    };
    builder.run(&items)?;
    builder.out.circuit.num_clbits = builder.measurements;
    Ok(builder.out)
}

/// Parse lines until the end of the text, or until the `}` closing the
/// block opened on line `open` (1-based).
fn parse_block(
    lines: &[&str],
    next: &mut usize,
    max_qubit: &mut Option<usize>,
    open: Option<usize>,
) -> Result<Vec<Item>, String> {
    let mut items = Vec::new();
    while *next < lines.len() {
        let number = *next + 1;
        let line = lines[*next];
        *next += 1;
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let error = |message: String| format!("line {number}: {message}");
        if line == "}" {
            return match open {
                Some(_) => Ok(items),
                None => Err(error("'}' without a REPEAT block".into())),
            };
        }
        let (name, args, rest) = split_instruction(line).map_err(error)?;
        if name == "REPEAT" {
            // `REPEAT n {`
            let mut words = rest.split_whitespace();
            let count = words
                .next()
                .and_then(|w| w.parse::<u64>().ok())
                .ok_or_else(|| error("REPEAT needs a repetition count".into()))?;
            if words.next() != Some("{") || words.next().is_some() || !args.is_empty() {
                return Err(error("expected 'REPEAT n {'".into()));
            }
            let body = parse_block(lines, next, max_qubit, Some(number))?;
            items.push(Item::Repeat { count, body });
            continue;
        }
        // Name an unsupported instruction before complaining about its
        // targets (`MXX`'s, say).
        if gate_sequence(&name).is_none() && !OTHER_INSTRUCTIONS.contains(&name.as_str()) {
            return Err(error(format!("Stim instruction {name} is not supported")));
        }
        let mut targets = Vec::new();
        for word in rest.split_whitespace() {
            parse_targets(word, &mut targets).map_err(error)?;
        }
        for target in &targets {
            if let Target::Qubit(q) | Target::Inverted(q) | Target::Pauli { qubit: q, .. } = *target
            {
                *max_qubit = Some(max_qubit.map_or(q, |m: usize| m.max(q)));
            }
        }
        items.push(Item::Op {
            line: number,
            name,
            args,
            targets,
        });
    }
    match open {
        Some(line) => Err(format!("line {line}: REPEAT block is never closed")),
        None => Ok(items),
    }
}

/// Split `NAME(arg, arg) targets` into its name (upper case, as Stim's
/// names are case-insensitive), arguments and the rest of the line.
fn split_instruction(line: &str) -> Result<(String, Vec<f64>, &str), String> {
    let end = line
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .unwrap_or(line.len());
    let name = line[..end].to_ascii_uppercase();
    if name.is_empty() {
        return Err(format!("expected an instruction, found '{line}'"));
    }
    let rest = &line[end..];
    if let Some(inner) = rest.strip_prefix('(') {
        let close = inner
            .find(')')
            .ok_or_else(|| format!("unclosed '(' in '{line}'"))?;
        let args = inner[..close]
            .split(',')
            .map(str::trim)
            .filter(|a| !a.is_empty())
            .map(|a| {
                a.parse::<f64>()
                    .map_err(|_| format!("'{a}' is not a number"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        return Ok((name, args, &inner[close + 1..]));
    }
    Ok((name, Vec::new(), rest))
}

/// Parse one whitespace-separated word of targets: a qubit, `!q`,
/// `rec[-k]`, a Pauli target such as `X3` or `!Y3`, or a product of Pauli
/// targets joined by `*` (`X0*!Y1*Z2`).
fn parse_targets(word: &str, out: &mut Vec<Target>) -> Result<(), String> {
    if word.contains('*') {
        let factors: Vec<&str> = word.split('*').collect();
        for (i, factor) in factors.iter().enumerate() {
            match parse_pauli(factor) {
                Some((pauli, qubit, inverted)) => out.push(Target::Pauli {
                    pauli,
                    qubit,
                    inverted,
                    joined: i + 1 < factors.len(),
                }),
                None => {
                    return Err(format!(
                        "'{factor}' in '{word}' is not a Pauli target such as X3"
                    ));
                }
            }
        }
        return Ok(());
    }
    if let Some((pauli, qubit, inverted)) = parse_pauli(word) {
        out.push(Target::Pauli {
            pauli,
            qubit,
            inverted,
            joined: false,
        });
        return Ok(());
    }
    out.push(parse_target(word)?);
    Ok(())
}

/// `X3`, `y3` or `!Z3` as (letter, qubit, inverted).
fn parse_pauli(word: &str) -> Option<(u8, usize, bool)> {
    let (inverted, rest) = match word.strip_prefix('!') {
        Some(rest) => (true, rest),
        None => (false, word),
    };
    let pauli = rest.bytes().next()?.to_ascii_uppercase();
    if !matches!(pauli, b'X' | b'Y' | b'Z') {
        return None;
    }
    rest[1..].parse().ok().map(|q| (pauli, q, inverted))
}

fn parse_target(word: &str) -> Result<Target, String> {
    if let Some(k) = word.strip_prefix("rec[-").and_then(|w| w.strip_suffix(']')) {
        return match k.parse::<usize>() {
            Ok(k) if k > 0 => Ok(Target::Record(k)),
            _ => Err(format!("bad measurement record target '{word}'")),
        };
    }
    if let Some(q) = word.strip_prefix('!') {
        if let Ok(q) = q.parse() {
            return Ok(Target::Inverted(q));
        }
    }
    word.parse()
        .map(Target::Qubit)
        .map_err(|_| format!("target '{word}' is not supported (qubits, !qubits and rec[-k] are)"))
}

/// The gates of a Stim gate name, applied in order to each target (one
/// qubit) or each pair of targets (two qubits).
fn gate_sequence(name: &str) -> Option<(usize, Vec<Gate>)> {
    let one = |gates: Vec<Gate>| Some((1, gates));
    match name {
        "I" => one(vec![]),
        "X" => one(vec![Gate::X]),
        "Y" => one(vec![Gate::Y]),
        "Z" => one(vec![Gate::Z]),
        "H" | "H_XZ" => one(vec![Gate::H]),
        "S" | "SQRT_Z" => one(vec![Gate::S]),
        "S_DAG" | "SQRT_Z_DAG" => one(vec![Gate::Sdg]),
        // Stim's SQRT_X maps Z to -Y, as SX = H S H does.
        "SQRT_X" => one(vec![Gate::SX]),
        "SQRT_X_DAG" => one(vec![Gate::SXdg]),
        // Stim's SQRT_Y maps X to -Z and Z to X: RY(π/2) up to phase.
        "SQRT_Y" => one(vec![Gate::RY(FRAC_PI_2)]),
        "SQRT_Y_DAG" => one(vec![Gate::RY(-FRAC_PI_2)]),
        // H_XY swaps X and Y (Z to -Z): X then S. H_YZ swaps Y and Z
        // (X to -X): SX then Z.
        "H_XY" => one(vec![Gate::X, Gate::S]),
        "H_YZ" => one(vec![Gate::SX, Gate::Z]),
        // C_XYZ maps X to Y, Y to Z and Z to X: S† then H. C_ZYX undoes it.
        "C_XYZ" => one(vec![Gate::Sdg, Gate::H]),
        "C_ZYX" => one(vec![Gate::H, Gate::S]),
        "CX" | "CNOT" | "ZCX" => Some((2, vec![Gate::CX])),
        "CY" | "ZCY" => Some((2, vec![Gate::CY])),
        "CZ" | "ZCZ" => Some((2, vec![Gate::CZ])),
        "SWAP" => Some((2, vec![Gate::SWAP])),
        _ => None,
    }
}

/// The basis of a measurement or reset: the gates that take it to Z
/// (applied before measuring) and back (after).
fn basis(name: &str) -> (&'static [Gate], &'static [Gate]) {
    // Y to Z: S† then H (S† maps Y to X, H maps X to Z); back: H then S.
    match name.chars().last() {
        Some('X') => (&[Gate::H], &[Gate::H]),
        Some('Y') => (&[Gate::Sdg, Gate::H], &[Gate::H, Gate::S]),
        _ => (&[], &[]),
    }
}

/// Unrolls the parsed items into the circuit.
struct Builder {
    out: StimCircuit,
    measurements: usize,
    /// While the last instruction (annotations aside) was `E` or
    /// `ELSE_CORRELATED_ERROR`: the index of that chain's
    /// [`Noise::Correlated`] instruction, which a following
    /// `ELSE_CORRELATED_ERROR` extends.
    chain: Option<usize>,
}

impl Builder {
    fn run(&mut self, items: &[Item]) -> Result<(), String> {
        for item in items {
            match item {
                Item::Repeat { count, body } => {
                    for _ in 0..*count {
                        self.run(body)?;
                    }
                }
                Item::Op {
                    line,
                    name,
                    args,
                    targets,
                } => {
                    // Annotations do not touch the frames, so they do not
                    // break a chain of correlated errors; anything else
                    // ends it.
                    let keeps_chain = matches!(
                        name.as_str(),
                        "E" | "CORRELATED_ERROR"
                            | "ELSE_CORRELATED_ERROR"
                            | "TICK"
                            | "QUBIT_COORDS"
                            | "SHIFT_COORDS"
                            | "DETECTOR"
                            | "OBSERVABLE_INCLUDE"
                    );
                    if !keeps_chain {
                        self.chain = None;
                    }
                    self.op(name, args, targets)
                        .map_err(|e| format!("line {line}: {e}"))?
                }
            }
        }
        Ok(())
    }

    fn qubits(name: &str, targets: &[Target], invertible: bool) -> Result<Vec<usize>, String> {
        targets
            .iter()
            .map(|t| match *t {
                Target::Qubit(q) => Ok(q),
                Target::Inverted(q) if invertible => Ok(q),
                Target::Inverted(_) => Err(format!("{name} takes no inverted targets")),
                Target::Record(_) => Err(format!(
                    "{name} with measurement record targets (classical control) is not supported"
                )),
                Target::Pauli { .. } => Err(format!("{name} takes qubit targets, not Paulis")),
            })
            .collect()
    }

    /// The measurements of `rec[-k]` targets.
    fn records(&self, name: &str, targets: &[Target]) -> Result<Vec<usize>, String> {
        targets
            .iter()
            .map(|t| match *t {
                Target::Record(k) if k <= self.measurements => Ok(self.measurements - k),
                Target::Record(k) => Err(format!(
                    "rec[-{k}] refers to a measurement before the first ({} so far)",
                    self.measurements
                )),
                _ => Err(format!("{name} takes only rec[-k] targets")),
            })
            .collect()
    }

    fn no_args(name: &str, args: &[f64]) -> Result<(), String> {
        if args.is_empty() {
            Ok(())
        } else {
            Err(format!("{name} takes no arguments"))
        }
    }

    fn probability(name: &str, args: &[f64], count: usize) -> Result<(), String> {
        if args.len() != count {
            return Err(format!("{name} takes {count} probability argument(s)"));
        }
        if args.iter().any(|p| !(0.0..=1.0).contains(p)) || args.iter().sum::<f64>() > 1.0 {
            return Err(format!(
                "{name}: probabilities must lie in [0, 1] and sum to at most 1"
            ));
        }
        Ok(())
    }

    fn op(&mut self, name: &str, args: &[f64], targets: &[Target]) -> Result<(), String> {
        let circuit = &mut self.out.circuit;
        if let Some((arity, gates)) = gate_sequence(name) {
            Self::no_args(name, args)?;
            let qubits = Self::qubits(name, targets, false)?;
            if !qubits.len().is_multiple_of(arity) {
                return Err(format!("{name} needs pairs of targets"));
            }
            for chunk in qubits.chunks(arity) {
                if arity == 2 && chunk[0] == chunk[1] {
                    return Err(format!("{name} on qubit {} twice", chunk[0]));
                }
                for gate in &gates {
                    circuit.gate(gate.clone(), chunk);
                }
            }
            return Ok(());
        }
        match name {
            "TICK" | "QUBIT_COORDS" | "SHIFT_COORDS" => {}
            "M" | "MZ" | "MX" | "MY" | "MR" | "MRZ" | "MRX" | "MRY" => {
                if args.len() > 1 {
                    return Err(format!("{name} takes at most one argument"));
                }
                Self::probability(name, args, args.len())?;
                let flip = args.first().copied().unwrap_or(0.0);
                let reset = name.starts_with("MR");
                let (to_z, back) = basis(name);
                for target in targets {
                    let (q, inverted) = match *target {
                        Target::Qubit(q) => (q, false),
                        Target::Inverted(q) => (q, true),
                        Target::Record(_) | Target::Pauli { .. } => {
                            return Err(format!("{name} takes qubit targets"));
                        }
                    };
                    for gate in to_z {
                        circuit.gate(gate.clone(), &[q]);
                    }
                    let m = self.measurements;
                    circuit.measure(q, m);
                    self.measurements += 1;
                    if reset {
                        circuit.reset(q);
                    }
                    for gate in back {
                        circuit.gate(gate.clone(), &[q]);
                    }
                    if flip > 0.0 {
                        self.out.flips.push((m, flip));
                    }
                    if inverted {
                        self.out.inverted.push(m);
                    }
                }
            }
            "R" | "RZ" | "RX" | "RY" => {
                Self::no_args(name, args)?;
                // Reset to |0>, then rotate into the basis's +1 state.
                let (_, back) = basis(name);
                for q in Self::qubits(name, targets, false)? {
                    circuit.reset(q);
                    for gate in back {
                        circuit.gate(gate.clone(), &[q]);
                    }
                }
            }
            "MPAD" => {
                // A measurement result fixed at 0 or 1, measuring nothing.
                Self::no_args(name, args)?;
                for target in targets {
                    match *target {
                        Target::Qubit(0) => {}
                        Target::Qubit(1) => self.out.inverted.push(self.measurements),
                        _ => return Err("MPAD takes the targets 0 and 1".into()),
                    }
                    self.measurements += 1;
                }
            }
            "X_ERROR" | "Y_ERROR" | "Z_ERROR" | "DEPOLARIZE1" | "PAULI_CHANNEL_1" => {
                let channel = if name == "PAULI_CHANNEL_1" {
                    Self::probability(name, args, 3)?;
                    Noise::Pauli1 {
                        px: args[0],
                        py: args[1],
                        pz: args[2],
                    }
                } else {
                    Self::probability(name, args, 1)?;
                    let p = args[0];
                    match name {
                        "X_ERROR" => Noise::Pauli1 {
                            px: p,
                            py: 0.0,
                            pz: 0.0,
                        },
                        "Y_ERROR" => Noise::Pauli1 {
                            px: 0.0,
                            py: p,
                            pz: 0.0,
                        },
                        "Z_ERROR" => Noise::Pauli1 {
                            px: 0.0,
                            py: 0.0,
                            pz: p,
                        },
                        _ => Noise::Pauli1 {
                            px: p / 3.0,
                            py: p / 3.0,
                            pz: p / 3.0,
                        },
                    }
                };
                let qubits = Self::qubits(name, targets, false)?;
                if !qubits.is_empty() {
                    circuit.noise(channel, &qubits);
                }
            }
            "DEPOLARIZE2" => {
                Self::probability(name, args, 1)?;
                let qubits = Self::qubits(name, targets, false)?;
                if !qubits.len().is_multiple_of(2) || qubits.chunks(2).any(|p| p[0] == p[1]) {
                    return Err("DEPOLARIZE2 needs pairs of distinct qubits".into());
                }
                if !qubits.is_empty() {
                    circuit.noise(Noise::Depolarize2(args[0]), &qubits);
                }
            }
            "PAULI_CHANNEL_2" => {
                Self::probability(name, args, 15)?;
                let qubits = Self::qubits(name, targets, false)?;
                if !qubits.len().is_multiple_of(2) || qubits.chunks(2).any(|p| p[0] == p[1]) {
                    return Err("PAULI_CHANNEL_2 needs pairs of distinct qubits".into());
                }
                if !qubits.is_empty() {
                    let mut p = [0.0; 15];
                    p.copy_from_slice(args);
                    circuit.noise(Noise::Pauli2(Box::new(p)), &qubits);
                }
            }
            "E" | "CORRELATED_ERROR" | "ELSE_CORRELATED_ERROR" => {
                self.correlated(name, args, targets)?;
            }
            "MPP" => self.mpp(args, targets)?,
            "DETECTOR" => {
                // Arguments are coordinates, which do not change sampling.
                let records = self.records(name, targets)?;
                self.out.detectors.push(records);
            }
            "OBSERVABLE_INCLUDE" => {
                let index = match args {
                    [k] if *k >= 0.0 && k.fract() == 0.0 => *k as usize,
                    _ => return Err("OBSERVABLE_INCLUDE takes one observable index".into()),
                };
                let records = self.records(name, targets)?;
                let observables = &mut self.out.observables;
                if observables.len() <= index {
                    observables.resize(index + 1, Vec::new());
                }
                observables[index].extend(records);
            }
            _ => return Err(format!("Stim instruction {name} is not supported")),
        }
        Ok(())
    }

    /// The Pauli targets of a correlated error, as a product: its qubits
    /// and one letter per qubit.
    fn product(name: &str, targets: &[Target]) -> Result<(Vec<usize>, Vec<u8>), String> {
        let mut qubits = Vec::new();
        let mut letters = Vec::new();
        for target in targets {
            match *target {
                Target::Pauli {
                    pauli,
                    qubit,
                    inverted: false,
                    joined: false,
                } => {
                    if qubits.contains(&qubit) {
                        return Err(format!("{name} names qubit {qubit} twice"));
                    }
                    qubits.push(qubit);
                    letters.push(pauli);
                }
                _ => return Err(format!("{name} takes Pauli targets such as X0 Z3")),
            }
        }
        if qubits.is_empty() {
            return Err(format!("{name} needs at least one Pauli target"));
        }
        Ok((qubits, letters))
    }

    /// `E(p) P` starts a chain; `ELSE_CORRELATED_ERROR(p) P` adds `P` to
    /// the chain before it, happening with probability `p` in the shots
    /// where no earlier product of the chain did. The chain is one
    /// [`Noise::Correlated`] instruction over the union of its qubits,
    /// each product padded with `I`, holding unconditional probabilities.
    fn correlated(&mut self, name: &str, args: &[f64], targets: &[Target]) -> Result<(), String> {
        Self::probability(name, args, 1)?;
        let (qubits, letters) = Self::product(name, targets)?;
        let p = args[0];
        let circuit = &mut self.out.circuit;
        if name != "ELSE_CORRELATED_ERROR" {
            let paulis = String::from_utf8(letters).expect("ASCII letters");
            circuit.noise(Noise::Correlated(vec![(p, paulis)]), &qubits);
            self.chain = Some(circuit.instructions.len() - 1);
            return Ok(());
        }
        let index = self.chain.ok_or(
            "ELSE_CORRELATED_ERROR must follow E or ELSE_CORRELATED_ERROR \
             (only annotations may come between)",
        )?;
        let Instruction::Noise {
            channel: Noise::Correlated(products),
            qubits: support,
        } = &mut circuit.instructions[index]
        else {
            unreachable!("the chain points at a correlated error");
        };
        // New qubits join the support; earlier products get I on them.
        for &q in &qubits {
            if !support.contains(&q) {
                support.push(q);
                for (_, paulis) in products.iter_mut() {
                    paulis.push('I');
                }
            }
        }
        let mut paulis = vec![b'I'; support.len()];
        for (&q, &letter) in qubits.iter().zip(&letters) {
            let at = support.iter().position(|&s| s == q).expect("just added");
            paulis[at] = letter;
        }
        let taken: f64 = products.iter().map(|(q, _)| q).sum();
        let unconditional = p * (1.0 - taken).max(0.0);
        products.push((
            unconditional,
            String::from_utf8(paulis).expect("ASCII letters"),
        ));
        Ok(())
    }

    /// `MPP`: measure each Pauli product. For a product `P1 ⊗ ... ⊗ Pk`
    /// on qubits `q1..qk`, each `qi` is rotated so that its `Pi` becomes
    /// Z (`U_i† Z U_i = P_i`, the same rotations as `MX` and `MY`), CX
    /// gates from `q1..q(k-1)` onto `qk` turn `Z_qk` into `Z_q1 ... Z_qk`,
    /// and `qk` is measured in Z. Undoing the CX gates and the rotations
    /// leaves the state as a direct product measurement would: in the
    /// frames, the random Z put on `qk` after the measurement becomes the
    /// measured product, which is what Stim applies after an `MPP`.
    fn mpp(&mut self, args: &[f64], targets: &[Target]) -> Result<(), String> {
        if args.len() > 1 {
            return Err("MPP takes at most one argument".into());
        }
        Self::probability("MPP", args, args.len())?;
        let flip = args.first().copied().unwrap_or(0.0);
        let mut start = 0;
        while start < targets.len() {
            // The product: targets up to the first one not joined by `*`.
            let mut end = start;
            loop {
                match targets[end] {
                    Target::Pauli { joined: true, .. } if end + 1 < targets.len() => end += 1,
                    Target::Pauli { joined: false, .. } => break,
                    _ => return Err("MPP takes Pauli products such as X0*Y1*Z2".into()),
                }
            }
            let mut factors = Vec::new();
            let mut inverted = false;
            for target in &targets[start..=end] {
                if let Target::Pauli {
                    pauli,
                    qubit,
                    inverted: bang,
                    ..
                } = *target
                {
                    if factors.iter().any(|&(_, q)| q == qubit) {
                        return Err(format!("MPP: qubit {qubit} appears twice in one product"));
                    }
                    factors.push((pauli, qubit));
                    inverted ^= bang;
                }
            }
            start = end + 1;
            let circuit = &mut self.out.circuit;
            let rotation = |pauli: u8| match pauli {
                b'X' => basis("MX"),
                b'Y' => basis("MY"),
                _ => basis("MZ"),
            };
            let (_, last) = *factors.last().expect("at least one factor");
            for &(pauli, q) in &factors {
                for gate in rotation(pauli).0 {
                    circuit.gate(gate.clone(), &[q]);
                }
            }
            for &(_, q) in &factors[..factors.len() - 1] {
                circuit.gate(Gate::CX, &[q, last]);
            }
            let m = self.measurements;
            circuit.measure(last, m);
            self.measurements += 1;
            for &(_, q) in factors[..factors.len() - 1].iter().rev() {
                circuit.gate(Gate::CX, &[q, last]);
            }
            for &(pauli, q) in &factors {
                for gate in rotation(pauli).1 {
                    circuit.gate(gate.clone(), &[q]);
                }
            }
            if flip > 0.0 {
                self.out.flips.push((m, flip));
            }
            if inverted {
                self.out.inverted.push(m);
            }
        }
        Ok(())
    }
}
