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
//! What is read (every instruction of Stim 1.16):
//! - **Gates:** `I`, `X`, `Y`, `Z`, `H` (`H_XZ`), `S` (`SQRT_Z`), `S_DAG`
//!   (`SQRT_Z_DAG`), `SQRT_X`, `SQRT_X_DAG`, `SQRT_Y`, `SQRT_Y_DAG`,
//!   `H_XY`, `H_YZ`, `H_NXY`, `H_NXZ`, `H_NYZ`, `C_XYZ`, `C_ZYX` and the
//!   negated cycles (`C_NXYZ`, `C_XNYZ`, `C_XYNZ`, `C_NZYX`, `C_ZNYX`,
//!   `C_ZYNX`); `II`, `CX` (`CNOT`, `ZCX`), `CY` (`ZCY`), `CZ` (`ZCZ`),
//!   `SWAP`, `XCX`, `XCY`, `XCZ`, `YCX`, `YCY`, `YCZ`, `CXSWAP`,
//!   `SWAPCX`, `CZSWAP` (`SWAPCZ`), `ISWAP`, `ISWAP_DAG`, `SQRT_XX`,
//!   `SQRT_YY`, `SQRT_ZZ` and their `_DAG`s, each as a short sequence of
//!   the simulator's gates.
//! - **Measurements and resets:** `M` (`MZ`), `MX`, `MY`, `MR` (`MRZ`),
//!   `MRX`, `MRY`, `R` (`RZ`), `RX`, `RY` and `MPAD`. A measurement's
//!   argument, as in `M(0.01)`, flips the recorded result with that
//!   probability without disturbing the state; `M !q` records it inverted.
//! - **Pauli product measurements:** `MPP X0*Y1*Z2 !Z3*Z4 ...`, with a
//!   flip probability as `MPP(0.01)`, and the pair forms `MXX`, `MYY`,
//!   `MZZ`. Each product becomes basis changes to Z, a ladder of CX gates
//!   onto its last qubit, a Z measurement of that qubit and the same
//!   gates undone: the measured qubit's Z is then the product, so the
//!   reference shot measures it on the tableau and the frames see its
//!   parity, as for any other measurement.
//! - **Pauli product rotations:** `SPP` and `SPP_DAG`, an S (or S†) on
//!   the last qubit between the same basis changes and CX ladder.
//! - **Noise:** `X_ERROR`, `Y_ERROR`, `Z_ERROR`, `DEPOLARIZE1`,
//!   `DEPOLARIZE2`, `PAULI_CHANNEL_1`, `PAULI_CHANNEL_2`, and correlated
//!   errors: `E` (`CORRELATED_ERROR`) and `ELSE_CORRELATED_ERROR`. A chain
//!   of `E` and the `ELSE_CORRELATED_ERROR` lines right after it is one
//!   [`Noise::Correlated`] channel (at most one of its products happens),
//!   with the unconditional probability of each product. `I_ERROR` and
//!   `II_ERROR` do nothing.
//! - **Heralded errors:** `HERALDED_ERASE` and `HERALDED_PAULI_CHANNEL_1`
//!   record a herald per target. They run on one extra qubit after
//!   Stim's (see `Builder::heralded`), so such a circuit has one qubit
//!   more than Stim counts.
//! - **Annotations:** `DETECTOR` and `OBSERVABLE_INCLUDE` with `rec[-k]`
//!   targets; `TICK`, `QUBIT_COORDS` and `SHIFT_COORDS` are ignored
//!   (coordinates do not change what is sampled), as are tags
//!   (`X_ERROR[leak](0.1) 0`).
//! - **Blocks:** `REPEAT n { ... }`, nested, unrolled.
//! - **Classical control:** `CX rec[-1] 5`, `CY`, `CZ` (and their `Z`
//!   aliases) with the record first, `CZ` and `XCZ`/`YCZ` with it second,
//!   each an [`Instruction::Feedback`] reading the record as Stim does,
//!   flips and inversions included (see `Builder::controlled`). Sweep
//!   bits (`sweep[k]`) read 0, as in Stim's samplers given no sweep data.
//!
//! Everything else (a record as the target, `rec[-k]` on other gates,
//! unknown instructions) is rejected with the line it is on, rather than
//! skipped: skipping an instruction would silently sample a different
//! circuit.

use std::f64::consts::FRAC_PI_2;

use crate::circuit::{Circuit, Gate, Instruction, Noise, Pauli};

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
    /// `sweep[k]`: bit `k` of a shot's sweep configuration.
    Sweep(usize),
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
    "MXX",
    "MYY",
    "MZZ",
    "SPP",
    "SPP_DAG",
    "HERALDED_ERASE",
    "HERALDED_PAULI_CHANNEL_1",
    "I_ERROR",
    "II_ERROR",
    "DETECTOR",
    "OBSERVABLE_INCLUDE",
];

/// Read a circuit in Stim's format.
pub fn parse(text: &str) -> Result<StimCircuit, String> {
    let lines: Vec<&str> = text.lines().collect();
    let mut next = 0;
    let mut max_qubit = None;
    let items = parse_block(&lines, &mut next, &mut max_qubit, None)?;
    let qubits = max_qubit.map_or(0, |q| q + 1);
    // Heralded errors record their herald on one extra qubit past Stim's;
    // so does a noisy measurement whose record feedback may read.
    let copy_flips = feedback(&items) && noisy_measurement(&items);
    let scratch = (heralded(&items) || copy_flips).then_some(qubits);
    let mut builder = Builder {
        out: StimCircuit {
            circuit: Circuit::new(qubits + usize::from(scratch.is_some())),
            ..StimCircuit::default()
        },
        measurements: 0,
        chain: None,
        scratch,
        copy_flips,
    };
    builder.run(&items)?;
    builder.out.circuit.num_clbits = builder.measurements;
    Ok(builder.out)
}

/// Whether any instruction, in any block, is a heralded error.
fn heralded(items: &[Item]) -> bool {
    items.iter().any(|item| match item {
        Item::Op { name, .. } => name.starts_with("HERALDED_"),
        Item::Repeat { body, .. } => heralded(body),
    })
}

/// Whether any gate, in any block, is classically controlled (has a
/// `rec[-k]` target; the annotations' records are not control).
fn feedback(items: &[Item]) -> bool {
    items.iter().any(|item| match item {
        Item::Op { name, targets, .. } => {
            !matches!(name.as_str(), "DETECTOR" | "OBSERVABLE_INCLUDE")
                && targets.iter().any(|t| matches!(t, Target::Record(_)))
        }
        Item::Repeat { body, .. } => feedback(body),
    })
}

/// Whether any measurement, in any block, flips its result with a
/// probability (`M(p)`, `MPP(p)`, ...).
fn noisy_measurement(items: &[Item]) -> bool {
    items.iter().any(|item| match item {
        Item::Op { name, args, .. } => {
            name.starts_with('M') && name != "MPAD" && args.first().is_some_and(|&p| p > 0.0)
        }
        Item::Repeat { body, .. } => noisy_measurement(body),
    })
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
    let mut rest = &line[end..];
    // A tag, as in `X_ERROR[leakage](0.1) 0`, is for tools reading the
    // circuit and does not change what it does (Stim 1.15).
    if let Some(inner) = rest.strip_prefix('[') {
        let close = inner
            .find(']')
            .ok_or_else(|| format!("unclosed '[' in '{line}'"))?;
        rest = &inner[close + 1..];
    }
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
    if let Some(k) = word
        .strip_prefix("sweep[")
        .and_then(|w| w.strip_suffix(']'))
    {
        return k
            .parse::<usize>()
            .map(Target::Sweep)
            .map_err(|_| format!("bad sweep bit target '{word}'"));
    }
    if let Some(q) = word.strip_prefix('!') {
        if let Ok(q) = q.parse() {
            return Ok(Target::Inverted(q));
        }
    }
    word.parse().map(Target::Qubit).map_err(|_| {
        format!("target '{word}' is not supported (qubits, !qubits, rec[-k] and sweep[k] are)")
    })
}

/// One gate of a Stim gate's decomposition, on the qubits of one target
/// (or one pair of targets) by position: `&[0]` is the first, `&[1, 0]`
/// the pair reversed.
type Step = (Gate, &'static [usize]);

/// The gates of a Stim gate name, applied in order to each target (one
/// qubit) or each pair of targets (two qubits).
///
/// Every decomposition was checked against `stim.Tableau.from_named_gate`
/// (Stim 1.16) and is pinned by the Pauli tables in `tests/stim.rs`.
fn gate_sequence(name: &str) -> Option<(usize, &'static [Step])> {
    const A: &[usize] = &[0];
    const B: &[usize] = &[1];
    const AB: &[usize] = &[0, 1];
    const BA: &[usize] = &[1, 0];
    // Constant tables, so that no line of an unrolled block allocates.
    macro_rules! steps {
        ($arity:expr, [$($step:expr),*]) => {{
            const STEPS: &[Step] = &[$($step),*];
            Some(($arity, STEPS))
        }};
    }
    match name {
        "I" => steps!(1, []),
        "X" => steps!(1, [(Gate::X, A)]),
        "Y" => steps!(1, [(Gate::Y, A)]),
        "Z" => steps!(1, [(Gate::Z, A)]),
        "H" | "H_XZ" => steps!(1, [(Gate::H, A)]),
        "S" | "SQRT_Z" => steps!(1, [(Gate::S, A)]),
        "S_DAG" | "SQRT_Z_DAG" => steps!(1, [(Gate::Sdg, A)]),
        // Stim's SQRT_X maps Z to -Y, as SX = H S H does.
        "SQRT_X" => steps!(1, [(Gate::SX, A)]),
        "SQRT_X_DAG" => steps!(1, [(Gate::SXdg, A)]),
        // Stim's SQRT_Y maps X to -Z and Z to X: RY(π/2) up to phase.
        "SQRT_Y" => steps!(1, [(Gate::RY(FRAC_PI_2), A)]),
        "SQRT_Y_DAG" => steps!(1, [(Gate::RY(-FRAC_PI_2), A)]),
        // H_XY swaps X and Y (Z to -Z): X then S. H_YZ swaps Y and Z
        // (X to -X): SX then Z.
        "H_XY" => steps!(1, [(Gate::X, A), (Gate::S, A)]),
        "H_YZ" => steps!(1, [(Gate::SX, A), (Gate::Z, A)]),
        // The negated Hadamards: H_NXY maps X to -Y and Z to -Z, H_NXZ X
        // to -Z, H_NYZ Z to -Y (and Y to -Z, X to -X).
        "H_NXY" => steps!(1, [(Gate::S, A), (Gate::X, A)]),
        "H_NXZ" => steps!(1, [(Gate::H, A), (Gate::Y, A)]),
        "H_NYZ" => steps!(1, [(Gate::Y, A), (Gate::SXdg, A)]),
        // C_XYZ maps X to Y, Y to Z and Z to X: S† then H. C_ZYX undoes it.
        "C_XYZ" => steps!(1, [(Gate::Sdg, A), (Gate::H, A)]),
        "C_ZYX" => steps!(1, [(Gate::H, A), (Gate::S, A)]),
        // The same cycles with one axis negated (shortest sequences found
        // by search over H, S, S†, SX, SX† and the Paulis).
        "C_NXYZ" => steps!(1, [(Gate::SX, A), (Gate::Sdg, A)]),
        "C_XNYZ" => steps!(1, [(Gate::H, A), (Gate::SX, A)]),
        "C_XYNZ" => steps!(1, [(Gate::SXdg, A), (Gate::S, A)]),
        "C_NZYX" => steps!(1, [(Gate::Sdg, A), (Gate::SX, A)]),
        "C_ZNYX" => steps!(1, [(Gate::H, A), (Gate::Sdg, A)]),
        "C_ZYNX" => steps!(1, [(Gate::S, A), (Gate::SXdg, A)]),
        "II" => steps!(2, []),
        "CX" | "CNOT" | "ZCX" => steps!(2, [(Gate::CX, AB)]),
        "CY" | "ZCY" => steps!(2, [(Gate::CY, AB)]),
        "CZ" | "ZCZ" => steps!(2, [(Gate::CZ, AB)]),
        "SWAP" => steps!(2, [(Gate::SWAP, AB)]),
        // Controls in the X or Y basis: the control rotated into Z around a
        // CX or CY. XCZ and YCZ are CX and CY with the roles swapped.
        "XCX" => steps!(2, [(Gate::H, A), (Gate::CX, AB), (Gate::H, A)]),
        "XCY" => steps!(2, [(Gate::H, A), (Gate::CY, AB), (Gate::H, A)]),
        "XCZ" => steps!(2, [(Gate::CX, BA)]),
        "YCX" => steps!(
            2,
            [
                (Gate::Sdg, A),
                (Gate::H, A),
                (Gate::CX, AB),
                (Gate::H, A),
                (Gate::S, A)
            ]
        ),
        "YCY" => steps!(
            2,
            [
                (Gate::Sdg, A),
                (Gate::H, A),
                (Gate::CY, AB),
                (Gate::H, A),
                (Gate::S, A)
            ]
        ),
        "YCZ" => steps!(2, [(Gate::CY, BA)]),
        // CXSWAP is CX then SWAP; SWAPCX the other order.
        "CXSWAP" => steps!(2, [(Gate::CX, AB), (Gate::SWAP, AB)]),
        "SWAPCX" => steps!(2, [(Gate::SWAP, AB), (Gate::CX, AB)]),
        "CZSWAP" | "SWAPCZ" => steps!(2, [(Gate::CZ, AB), (Gate::SWAP, AB)]),
        // ISWAP = (S ⊗ S) SWAP CZ as a circuit: CZ first, then the swap,
        // then S on both; ISWAP_DAG is its inverse.
        "ISWAP" => steps!(2, [(Gate::CZ, AB), (Gate::SWAP, AB), (Gate::S, AB)]),
        "ISWAP_DAG" => steps!(2, [(Gate::Sdg, AB), (Gate::SWAP, AB), (Gate::CZ, AB)]),
        // exp(∓iπ/4 P⊗P) up to phase: both qubits rotated so that P becomes
        // Z, then exp(∓iπ/4 Z⊗Z) = CX (I ⊗ S^±1) CX, then rotated back.
        "SQRT_ZZ" => steps!(2, [(Gate::CX, AB), (Gate::S, B), (Gate::CX, AB)]),
        "SQRT_ZZ_DAG" => steps!(2, [(Gate::CX, AB), (Gate::Sdg, B), (Gate::CX, AB)]),
        "SQRT_XX" => steps!(
            2,
            [
                (Gate::H, AB),
                (Gate::CX, AB),
                (Gate::S, B),
                (Gate::CX, AB),
                (Gate::H, AB)
            ]
        ),
        "SQRT_XX_DAG" => steps!(
            2,
            [
                (Gate::H, AB),
                (Gate::CX, AB),
                (Gate::Sdg, B),
                (Gate::CX, AB),
                (Gate::H, AB)
            ]
        ),
        "SQRT_YY" => steps!(
            2,
            [
                (Gate::Sdg, AB),
                (Gate::H, AB),
                (Gate::CX, AB),
                (Gate::S, B),
                (Gate::CX, AB),
                (Gate::H, AB),
                (Gate::S, AB)
            ]
        ),
        "SQRT_YY_DAG" => steps!(
            2,
            [
                (Gate::Sdg, AB),
                (Gate::H, AB),
                (Gate::CX, AB),
                (Gate::Sdg, B),
                (Gate::CX, AB),
                (Gate::H, AB),
                (Gate::S, AB)
            ]
        ),
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
    /// One extra qubit past Stim's, in `|0>` between uses: it records
    /// heralds (see [`Self::heralded`]) and the noisy measurements of a
    /// circuit with feedback (see [`Self::measure_into`]).
    scratch: Option<usize>,
    /// Whether noisy measurements copy their result onto the scratch
    /// qubit and flip it there, instead of listing the flip in
    /// [`StimCircuit::flips`]: feedback reads the flipped record, so the
    /// flip must happen in the shot, not after the run.
    copy_flips: bool,
}

/// Apply a gate's decomposition (see [`gate_sequence`]) to one target or
/// one pair of targets.
fn apply_sequence(circuit: &mut Circuit, gates: &[Step], chunk: &[usize]) {
    for (gate, at) in gates {
        if gate.arity() == at.len() {
            // On the stack: this runs for every target pair of every
            // unrolled line.
            let mut on = [0usize; 2];
            for (o, &i) in on.iter_mut().zip(at.iter()) {
                *o = chunk[i];
            }
            circuit.gate(gate.clone(), &on[..at.len()]);
        } else {
            // A one-qubit gate on each listed qubit.
            for &i in at.iter() {
                circuit.gate(gate.clone(), &[chunk[i]]);
            }
        }
    }
}

/// Measure Z of `q` into classical bit `m`, or, with `copy` = `Some(s)`,
/// copy Z of `q` onto the scratch qubit `s` (in `|0>`), flip the copy with
/// probability `flip`, measure `s` into `m` and reset it.
///
/// With `s` in `|0>`, CX q→s maps `a|0>|0> + b|1>|0>` to
/// `a|0>|0> + b|1>|1>`, so measuring `s` collapses `q` as measuring `q`
/// would; an X error on `s` before its measurement flips only the record.
/// The flipped record is then part of the shot, where feedback can read
/// it, as Stim's does.
fn measure_z(circuit: &mut Circuit, q: usize, m: usize, copy: Option<usize>, flip: f64) {
    match copy {
        Some(s) => {
            circuit.cx(q, s);
            circuit.x_error(s, flip);
            circuit.measure(s, m);
            circuit.reset(s);
        }
        None => {
            circuit.measure(q, m);
        }
    }
}

impl Builder {
    /// The scratch qubit a measurement with flip probability `flip` copies
    /// its result onto (see [`measure_z`]), if it does.
    fn copy_onto(&self, flip: f64) -> Option<usize> {
        if flip > 0.0 && self.copy_flips {
            Some(self.scratch.expect("parse adds the scratch qubit"))
        } else {
            None
        }
    }

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
                Target::Record(_) | Target::Sweep(_) => Err(format!(
                    "{name} takes no measurement record or sweep bit targets \
                     (classical control is CX, CY, CZ, XCZ and YCZ)"
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
            if targets
                .iter()
                .any(|t| matches!(t, Target::Record(_) | Target::Sweep(_)))
            {
                return self.controlled(name, gates, targets);
            }
            let qubits = Self::qubits(name, targets, false)?;
            if !qubits.len().is_multiple_of(arity) {
                return Err(format!("{name} needs pairs of targets"));
            }
            for chunk in qubits.chunks(arity) {
                if arity == 2 && chunk[0] == chunk[1] {
                    return Err(format!("{name} on qubit {} twice", chunk[0]));
                }
                apply_sequence(circuit, gates, chunk);
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
                        Target::Record(_) | Target::Sweep(_) | Target::Pauli { .. } => {
                            return Err(format!("{name} takes qubit targets"));
                        }
                    };
                    let m = self.measurements;
                    let copy = self.copy_onto(flip);
                    let circuit = &mut self.out.circuit;
                    for gate in to_z {
                        circuit.gate(gate.clone(), &[q]);
                    }
                    measure_z(circuit, q, m, copy, flip);
                    self.measurements += 1;
                    if reset {
                        circuit.reset(q);
                    }
                    for gate in back {
                        circuit.gate(gate.clone(), &[q]);
                    }
                    if flip > 0.0 && copy.is_none() {
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
            "MPP" => {
                let flip = Self::flip(name, args)?;
                for (factors, inverted) in Self::products(name, targets)? {
                    self.measure_product(&factors, inverted, flip);
                }
            }
            "MXX" | "MYY" | "MZZ" => {
                // Two-qubit parity measurements: MXX a b is MPP Xa*Xb.
                let flip = Self::flip(name, args)?;
                let pauli = name.as_bytes()[1];
                if !targets.len().is_multiple_of(2) {
                    return Err(format!("{name} needs pairs of targets"));
                }
                for pair in targets.chunks(2) {
                    let mut factors = Vec::new();
                    let mut inverted = false;
                    for target in pair {
                        match *target {
                            Target::Qubit(q) => factors.push((pauli, q)),
                            Target::Inverted(q) => {
                                factors.push((pauli, q));
                                inverted = !inverted;
                            }
                            _ => return Err(format!("{name} takes qubit targets")),
                        }
                    }
                    if factors[0].1 == factors[1].1 {
                        return Err(format!("{name} on qubit {} twice", factors[0].1));
                    }
                    self.measure_product(&factors, inverted, flip);
                }
            }
            "SPP" | "SPP_DAG" => {
                Self::no_args(name, args)?;
                for (factors, inverted) in Self::products(name, targets)? {
                    // S phases the -1 eigenspace of Z by i; SPP does the
                    // same for the product. On `-P` (an inverted product)
                    // that is the +1 eigenspace of P: S† up to phase.
                    let s = if (name == "SPP") != inverted {
                        Gate::S
                    } else {
                        Gate::Sdg
                    };
                    self.on_product(&factors, |circuit, last| {
                        circuit.gate(s.clone(), &[last]);
                    });
                }
            }
            "HERALDED_ERASE" | "HERALDED_PAULI_CHANNEL_1" => self.heralded(name, args, targets)?,
            "I_ERROR" | "II_ERROR" => {
                // Identity channels (placeholders for tools): any
                // arguments, no effect.
                let qubits = Self::qubits(name, targets, false)?;
                if name == "II_ERROR" && !qubits.len().is_multiple_of(2) {
                    return Err("II_ERROR needs pairs of targets".into());
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

    /// A controlled gate with classical controls among its pairs of
    /// targets, as in `CX rec[-1] 5` or `CZ 2 sweep[0]`.
    ///
    /// A classical control applies the Pauli the gate would put on the
    /// other qubit when the control is 1: `CX`, `CY`, `CZ` (and `ZCX`,
    /// `ZCY`, `ZCZ`, `CNOT`) take it first, putting X, Y or Z on the
    /// second; `XCZ` and `YCZ` take it second, putting X or Y on the first;
    /// `CZ` takes it on either side, and two classical bits do nothing.
    /// A classical target would edit the record, which Stim rejects too,
    /// as it rejects classical bits on every other gate.
    ///
    /// `rec[-k]` reads the record as Stim's does, inversions and flips
    /// included: an inverted record (`M !q`, `MPAD 1`) is the Pauli always
    /// times the Pauli when the measured result is 1; flips happen in the
    /// shot (see [`measure_z`]). Sweep bits read 0, as in Stim's samplers
    /// when they are given no sweep data, so they apply nothing.
    fn controlled(&mut self, name: &str, gates: &[Step], targets: &[Target]) -> Result<(), String> {
        let (pauli, sides) = match name {
            "CX" | "CNOT" | "ZCX" => (Pauli::X, [true, false]),
            "CY" | "ZCY" => (Pauli::Y, [true, false]),
            "CZ" | "ZCZ" => (Pauli::Z, [true, true]),
            "XCZ" => (Pauli::X, [false, true]),
            "YCZ" => (Pauli::Y, [false, true]),
            _ => {
                return Err(format!(
                    "{name} takes no measurement record or sweep bit targets \
                     (classical control is CX, CY, CZ, XCZ and YCZ)"
                ));
            }
        };
        if !targets.len().is_multiple_of(2) {
            return Err(format!("{name} needs pairs of targets"));
        }
        for pair in targets.chunks(2) {
            let classical = |t: &Target| matches!(t, Target::Record(_) | Target::Sweep(_));
            let (control, target) = match (classical(&pair[0]), classical(&pair[1])) {
                (false, false) => {
                    let qubits = Self::qubits(name, pair, false)?;
                    if qubits[0] == qubits[1] {
                        return Err(format!("{name} on qubit {} twice", qubits[0]));
                    }
                    apply_sequence(&mut self.out.circuit, gates, &qubits);
                    continue;
                }
                (true, true) if sides == [true, true] => {
                    // Validate the records; a CZ between bits is nothing.
                    self.records(
                        name,
                        &pair
                            .iter()
                            .filter(|t| matches!(t, Target::Record(_)))
                            .copied()
                            .collect::<Vec<_>>(),
                    )?;
                    continue;
                }
                (true, false) if sides[0] => (pair[0], pair[1]),
                (false, true) if sides[1] => (pair[1], pair[0]),
                _ => {
                    return Err(format!(
                        "{name} with a classical target would edit the measurement record, \
                         which is not supported"
                    ));
                }
            };
            let q = Self::qubits(name, &[target], false)?[0];
            if let Target::Record(_) = control {
                let m = self.records(name, &[control])?[0];
                let circuit = &mut self.out.circuit;
                if self.out.inverted.binary_search(&m).is_ok() {
                    circuit.gate(pauli.gate(), &[q]);
                }
                circuit.feedback(pauli, q, m);
            }
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

    /// The flip probability of a measurement, `M(p)` or `MPP(p)`: at most
    /// one argument.
    fn flip(name: &str, args: &[f64]) -> Result<f64, String> {
        if args.len() > 1 {
            return Err(format!("{name} takes at most one argument"));
        }
        Self::probability(name, args, args.len())?;
        Ok(args.first().copied().unwrap_or(0.0))
    }

    /// The Pauli products of `MPP` or `SPP` targets (`X0*Y1*Z2 !Z3`):
    /// each one's factors (letter, qubit) and whether an odd number of
    /// them was written inverted.
    #[allow(clippy::type_complexity)]
    fn products(name: &str, targets: &[Target]) -> Result<Vec<(Vec<(u8, usize)>, bool)>, String> {
        let mut out = Vec::new();
        let mut start = 0;
        while start < targets.len() {
            // The product: targets up to the first one not joined by `*`.
            let mut end = start;
            loop {
                match targets[end] {
                    Target::Pauli { joined: true, .. } if end + 1 < targets.len() => end += 1,
                    Target::Pauli { joined: false, .. } => break,
                    _ => return Err(format!("{name} takes Pauli products such as X0*Y1*Z2")),
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
                        return Err(format!(
                            "{name}: qubit {qubit} appears twice in one product"
                        ));
                    }
                    factors.push((pauli, qubit));
                    inverted ^= bang;
                }
            }
            start = end + 1;
            out.push((factors, inverted));
        }
        Ok(out)
    }

    /// Run `inner` on the last qubit `qk` of the product `P1 ⊗ ... ⊗ Pk`
    /// (on `q1..qk`) after mapping the product onto `Z_qk`, then map it
    /// back. Each `qi` is rotated so that its `Pi` becomes Z
    /// (`U_i† Z U_i = P_i`, the same rotations as `MX` and `MY`), and CX
    /// gates from `q1..q(k-1)` onto `qk` turn `Z_qk` into
    /// `Z_q1 ... Z_qk`. A Z measurement of `qk` in between is then a
    /// measurement of the product, and an S on `qk` is `SPP` of it.
    fn on_product(&mut self, factors: &[(u8, usize)], inner: impl FnOnce(&mut Circuit, usize)) {
        let circuit = &mut self.out.circuit;
        let rotation = |pauli: u8| match pauli {
            b'X' => basis("MX"),
            b'Y' => basis("MY"),
            _ => basis("MZ"),
        };
        let (_, last) = *factors.last().expect("at least one factor");
        for &(pauli, q) in factors {
            for gate in rotation(pauli).0 {
                circuit.gate(gate.clone(), &[q]);
            }
        }
        for &(_, q) in &factors[..factors.len() - 1] {
            circuit.gate(Gate::CX, &[q, last]);
        }
        inner(circuit, last);
        for &(_, q) in factors[..factors.len() - 1].iter().rev() {
            circuit.gate(Gate::CX, &[q, last]);
        }
        for &(pauli, q) in factors {
            for gate in rotation(pauli).1 {
                circuit.gate(gate.clone(), &[q]);
            }
        }
    }

    /// Measure one Pauli product (`MPP`, `MXX`, ...) into the next
    /// record. Undoing the CX gates and the rotations of
    /// [`Self::on_product`] leaves the state as a direct product
    /// measurement would: in the frames, the random Z put on `qk` after
    /// the measurement becomes the measured product, which is what Stim
    /// applies after an `MPP`.
    fn measure_product(&mut self, factors: &[(u8, usize)], inverted: bool, flip: f64) {
        let m = self.measurements;
        let copy = self.copy_onto(flip);
        self.on_product(factors, |circuit, last| {
            measure_z(circuit, last, m, copy, flip);
        });
        self.measurements += 1;
        if flip > 0.0 && copy.is_none() {
            self.out.flips.push((m, flip));
        }
        if inverted {
            self.out.inverted.push(m);
        }
    }

    /// `HERALDED_ERASE(p)` and `HERALDED_PAULI_CHANNEL_1(pI, pX, pY, pZ)`:
    /// on each target, with probability `p` (or `pI + pX + pY + pZ`) a
    /// herald records 1 and the qubit takes a Pauli (uniform for an
    /// erasure, `pI : pX : pY : pZ` for the channel); otherwise the
    /// herald records 0 and nothing happens.
    ///
    /// The herald is a qubit of its own, `h`, in `|0>`: a correlated
    /// error puts `X_h` together with the Pauli on the target (`X_h I`,
    /// `X_h X`, `X_h Y`, `X_h Z`, at most one of them), `h` is measured
    /// into the record and reset. The event and the herald are then
    /// exactly as correlated as in Stim, and `h` touches nothing else.
    fn heralded(&mut self, name: &str, args: &[f64], targets: &[Target]) -> Result<(), String> {
        let probabilities = if name == "HERALDED_ERASE" {
            Self::probability(name, args, 1)?;
            [args[0] / 4.0; 4]
        } else {
            Self::probability(name, args, 4)?;
            [args[0], args[1], args[2], args[3]]
        };
        let herald = self.scratch.expect("parse adds the scratch qubit");
        for q in Self::qubits(name, targets, false)? {
            let circuit = &mut self.out.circuit;
            let products: Vec<(f64, String)> = probabilities
                .iter()
                .zip(["XI", "XX", "XY", "XZ"])
                .filter(|(p, _)| **p > 0.0)
                .map(|(&p, paulis)| (p, paulis.to_string()))
                .collect();
            if !products.is_empty() {
                circuit.noise(Noise::Correlated(products), &[herald, q]);
            }
            circuit.measure(herald, self.measurements);
            circuit.reset(herald);
            self.measurements += 1;
        }
        Ok(())
    }
}
