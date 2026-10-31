//! A reader for Stim's circuit format, with its detector annotations.
//!
//! Error-correction circuits are usually written in Stim's format (Gidney,
//! *Quantum* 5, 497, 2021): Stim's own generators, and the tools around it,
//! emit noisy Clifford circuits whose `DETECTOR` and `OBSERVABLE_INCLUDE`
//! lines say which measurement results a decoder looks at. This module
//! reads such a file into a [`StimCircuit`]: a qvd [`Circuit`] with one
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
//! - **Noise:** `X_ERROR`, `Y_ERROR`, `Z_ERROR`, `DEPOLARIZE1`,
//!   `DEPOLARIZE2` and `PAULI_CHANNEL_1`.
//! - **Annotations:** `DETECTOR` and `OBSERVABLE_INCLUDE` with `rec[-k]`
//!   targets; `TICK`, `QUBIT_COORDS` and `SHIFT_COORDS` are ignored
//!   (coordinates do not change what is sampled).
//! - **Blocks:** `REPEAT n { ... }`, nested, unrolled.
//!
//! Everything else (`MPP`, correlated and heralded errors, classically
//! controlled gates such as `CX rec[-1] 0`, sweep bits, ...) is rejected
//! with the line it is on, rather than skipped: skipping an instruction
//! would silently sample a different circuit.

use std::f64::consts::FRAC_PI_2;

use crate::circuit::{Circuit, Gate, Noise};

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
        // targets (MPP's `X0*X1`, say).
        if gate_sequence(&name).is_none() && !OTHER_INSTRUCTIONS.contains(&name.as_str()) {
            return Err(error(format!("Stim instruction {name} is not supported")));
        }
        let targets = rest
            .split_whitespace()
            .map(parse_target)
            .collect::<Result<Vec<_>, _>>()
            .map_err(error)?;
        for target in &targets {
            if let Target::Qubit(q) | Target::Inverted(q) = *target {
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
                } => self
                    .op(name, args, targets)
                    .map_err(|e| format!("line {line}: {e}"))?,
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
                        Target::Record(_) => {
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
}
