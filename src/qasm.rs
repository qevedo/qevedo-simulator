//! Reading OpenQASM 2 and 3 programs.
//!
//! The gates of `qelib1.inc` (OpenQASM 2) and `stdgates.inc` (OpenQASM 3) map
//! to Qevedo's gates, natively where Qevedo has the gate and as exact unitaries
//! otherwise (`csx`, `cu3`, `cu`, `c3x`, `c3sqrtx`, `c4x`), and through their
//! `qelib1.inc` definitions for the relative-phase Toffolis (`rccx`,
//! `rc3x`). Programs can
//! declare registers (`qreg`/`creg`, or OpenQASM 3's `qubit`/`bit`), define
//! gates (`gate name(params) qubits { ... }`, expanded where they are
//! called), use parameter expressions, broadcast over whole registers,
//! measure, reset and use barriers. OpenQASM 3's `inv @`, `ctrl @`,
//! `negctrl @` and integer `pow @` modifiers are supported.
//!
//! An `if` on one classical bit whose body (and `else` body) holds only
//! Pauli gates is read as classical feedback ([`Instruction::Feedback`]):
//! OpenQASM 3's `if (c[k]) x q[j];`, `if (!c[k]) { z q[0]; x q[1]; }`,
//! `if (c[k] == 1) ... else ...`, and OpenQASM 2's `if (c == 1) x q[0];` on
//! a one-bit register. This covers the corrections of teleportation,
//! measurement-based resets and the feed-forward of IBM's dynamic circuits,
//! and keeps them on the stabilizer backend.
//!
//! Anything else that needs classical state (`if` on several bits or with
//! other gates, loops, variables, subroutines, measurement results used in
//! expressions) is rejected with the line and column of the statement.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::f64::consts::{E, PI, TAU};
use std::fmt;

use crate::circuit::{Circuit, Gate, Instruction, Pauli};
use crate::matrix::{C64, Matrix};

/// A syntax or semantic error, with its position (1-based).
#[derive(Clone, Debug, PartialEq)]
pub struct QasmError {
    pub line: usize,
    pub column: usize,
    pub message: String,
}

impl fmt::Display for QasmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "line {}, column {}: {}",
            self.line, self.column, self.message
        )
    }
}

impl std::error::Error for QasmError {}

/// Parse an OpenQASM 2 or 3 program into a circuit. Qubits and classical
/// bits are numbered in declaration order across registers.
pub fn parse(source: &str) -> Result<Circuit, QasmError> {
    // The prelude's definitions come first, so a program may redefine them.
    let mut prelude = Parser {
        tokens: lex(PRELUDE).expect("valid prelude"),
        at: 0,
        program: Program::default(),
    };
    prelude.program().expect("valid prelude");
    let tokens = lex(source)?;
    let mut parser = Parser {
        tokens,
        at: 0,
        program: Program {
            gates: prelude.program.gates,
            ..Program::default()
        },
    };
    parser.program()?;
    let program = parser.program;
    let mut circuit = Circuit::new(program.qubits);
    circuit.num_clbits = program.clbits;
    circuit.instructions = program.instructions;
    Ok(circuit)
}

/// Gates of `qelib1.inc` defined by their decomposition rather than mapped
/// to a Qevedo gate: the relative-phase Toffolis, whose phases are what their
/// definitions make them (from `qelib1.inc`, the OpenQASM 2 standard
/// library).
const PRELUDE: &str = "
gate rccx a,b,c
{
  u2(0,pi) c; u1(pi/4) c; cx b, c; u1(-pi/4) c; cx a, c;
  u1(pi/4) c; cx b, c; u1(-pi/4) c; u2(0,pi) c;
}
gate rc3x a,b,c,d
{
  u2(0,pi) d; u1(pi/4) d; cx c,d; u1(-pi/4) d; u2(0,pi) d;
  cx a,d; u1(pi/4) d; cx b,d; u1(-pi/4) d; cx a,d;
  u1(pi/4) d; cx b,d; u1(-pi/4) d; u2(0,pi) d;
  u1(pi/4) d; cx c,d; u1(-pi/4) d; u2(0,pi) d;
}
";

// -- lexer -----------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum Kind {
    Ident(String),
    Number(f64),
    Str(String),
    /// Punctuation and operators: `; , ( ) [ ] { } -> = + - * / ^ ** @ :`.
    Symbol(&'static str),
    End,
}

#[derive(Clone, Debug)]
struct Token {
    kind: Kind,
    line: usize,
    column: usize,
}

fn lex(source: &str) -> Result<Vec<Token>, QasmError> {
    let chars: Vec<char> = source.chars().collect();
    let (mut i, mut line, mut column) = (0, 1, 1);
    let mut tokens = Vec::new();
    let error = |line, column, message: String| QasmError {
        line,
        column,
        message,
    };
    while i < chars.len() {
        let c = chars[i];
        let (start_line, start_column) = (line, column);
        let advance = |i: &mut usize, line: &mut usize, column: &mut usize, n: usize| {
            for k in 0..n {
                if chars[*i + k] == '\n' {
                    *line += 1;
                    *column = 1;
                } else {
                    *column += 1;
                }
            }
            *i += n;
        };
        if c.is_whitespace() {
            advance(&mut i, &mut line, &mut column, 1);
        } else if c == '/' && chars.get(i + 1) == Some(&'/') {
            while i < chars.len() && chars[i] != '\n' {
                advance(&mut i, &mut line, &mut column, 1);
            }
        } else if c == '/' && chars.get(i + 1) == Some(&'*') {
            advance(&mut i, &mut line, &mut column, 2);
            while i < chars.len() && !(chars[i] == '*' && chars.get(i + 1) == Some(&'/')) {
                advance(&mut i, &mut line, &mut column, 1);
            }
            if i >= chars.len() {
                return Err(error(
                    start_line,
                    start_column,
                    "unterminated comment".into(),
                ));
            }
            advance(&mut i, &mut line, &mut column, 2);
        } else if c.is_alphabetic() || c == '_' {
            let mut j = i;
            while j < chars.len() && (chars[j].is_alphanumeric() || chars[j] == '_') {
                j += 1;
            }
            let word: String = chars[i..j].iter().collect();
            tokens.push(Token {
                kind: Kind::Ident(word),
                line,
                column,
            });
            let n = j - i;
            advance(&mut i, &mut line, &mut column, n);
        } else if c.is_ascii_digit()
            || (c == '.' && chars.get(i + 1).is_some_and(|d| d.is_ascii_digit()))
        {
            let mut j = i;
            while j < chars.len()
                && (chars[j].is_ascii_digit() || chars[j] == '.' || chars[j] == '_')
            {
                j += 1;
            }
            if j < chars.len() && (chars[j] == 'e' || chars[j] == 'E') {
                let mut k = j + 1;
                if k < chars.len() && (chars[k] == '+' || chars[k] == '-') {
                    k += 1;
                }
                if k < chars.len() && chars[k].is_ascii_digit() {
                    j = k;
                    while j < chars.len() && chars[j].is_ascii_digit() {
                        j += 1;
                    }
                }
            }
            let text: String = chars[i..j].iter().filter(|&&d| d != '_').collect();
            let value = text
                .parse::<f64>()
                .map_err(|_| error(line, column, format!("invalid number `{text}`")))?;
            tokens.push(Token {
                kind: Kind::Number(value),
                line,
                column,
            });
            let n = j - i;
            advance(&mut i, &mut line, &mut column, n);
        } else if c == '"' || c == '\'' {
            let mut j = i + 1;
            while j < chars.len() && chars[j] != c && chars[j] != '\n' {
                j += 1;
            }
            if j >= chars.len() || chars[j] != c {
                return Err(error(line, column, "unterminated string".into()));
            }
            let text: String = chars[i + 1..j].iter().collect();
            tokens.push(Token {
                kind: Kind::Str(text),
                line,
                column,
            });
            let n = j + 1 - i;
            advance(&mut i, &mut line, &mut column, n);
        } else {
            let two: String = chars[i..(i + 2).min(chars.len())].iter().collect();
            let symbol = match two.as_str() {
                "->" => Some("->"),
                "**" => Some("**"),
                "==" => Some("=="),
                "!=" => Some("!="),
                _ => None,
            };
            let (symbol, width) = match symbol {
                Some(s) => (s, 2),
                None => {
                    let s = match c {
                        ';' => ";",
                        ',' => ",",
                        '(' => "(",
                        ')' => ")",
                        '[' => "[",
                        ']' => "]",
                        '{' => "{",
                        '}' => "}",
                        '=' => "=",
                        '+' => "+",
                        '-' => "-",
                        '*' => "*",
                        '/' => "/",
                        '^' => "^",
                        '@' => "@",
                        ':' => ":",
                        '<' => "<",
                        '>' => ">",
                        '!' => "!",
                        '&' => "&",
                        '|' => "|",
                        '%' => "%",
                        '~' => "~",
                        '$' => "$",
                        _ => {
                            return Err(error(line, column, format!("unexpected character `{c}`")));
                        }
                    };
                    (s, 1)
                }
            };
            tokens.push(Token {
                kind: Kind::Symbol(symbol),
                line,
                column,
            });
            advance(&mut i, &mut line, &mut column, width);
        }
    }
    tokens.push(Token {
        kind: Kind::End,
        line,
        column,
    });
    Ok(tokens)
}

// -- expressions ----------------------------------------------------------

#[derive(Clone, Debug)]
enum Expr {
    Number(f64),
    Name(String, usize, usize),
    Negate(Box<Expr>),
    Binary(char, Box<Expr>, Box<Expr>),
    Call(String, Box<Expr>, usize, usize),
}

impl Expr {
    fn eval(&self, env: &HashMap<String, f64>) -> Result<f64, QasmError> {
        Ok(match self {
            Expr::Number(x) => *x,
            Expr::Name(name, line, column) => match name.as_str() {
                "pi" | "π" => PI,
                "tau" | "τ" => TAU,
                "euler" | "ℇ" => E,
                _ => *env.get(name).ok_or_else(|| QasmError {
                    line: *line,
                    column: *column,
                    message: format!("unknown parameter `{name}`"),
                })?,
            },
            Expr::Negate(e) => -e.eval(env)?,
            Expr::Binary(op, a, b) => {
                let (a, b) = (a.eval(env)?, b.eval(env)?);
                match op {
                    '+' => a + b,
                    '-' => a - b,
                    '*' => a * b,
                    '/' => a / b,
                    _ => a.powf(b),
                }
            }
            Expr::Call(name, arg, line, column) => {
                let x = arg.eval(env)?;
                match name.as_str() {
                    "sin" => x.sin(),
                    "cos" => x.cos(),
                    "tan" => x.tan(),
                    "exp" => x.exp(),
                    "ln" => x.ln(),
                    "sqrt" => x.sqrt(),
                    "arcsin" | "asin" => x.asin(),
                    "arccos" | "acos" => x.acos(),
                    "arctan" | "atan" => x.atan(),
                    _ => {
                        return Err(QasmError {
                            line: *line,
                            column: *column,
                            message: format!("unknown function `{name}`"),
                        });
                    }
                }
            }
        })
    }
}

// -- program --------------------------------------------------------------

/// A gate call inside a gate definition, with its arguments as names.
#[derive(Clone, Debug)]
struct Call {
    modifiers: Vec<Modifier>,
    name: String,
    params: Vec<Expr>,
    args: Vec<String>,
    line: usize,
    column: usize,
}

#[derive(Clone, Debug)]
enum Modifier {
    Inv,
    Ctrl(usize),
    NegCtrl(usize),
    Pow(Expr),
}

#[derive(Clone, Debug)]
struct Definition {
    params: Vec<String>,
    qubits: Vec<String>,
    body: Vec<Call>,
}

#[derive(Default)]
struct Program {
    qubits: usize,
    clbits: usize,
    /// Register name -> (first index, size).
    qregs: HashMap<String, (usize, usize)>,
    cregs: HashMap<String, (usize, usize)>,
    gates: HashMap<String, Definition>,
    instructions: Vec<Instruction>,
}

struct Parser {
    tokens: Vec<Token>,
    at: usize,
    program: Program,
}

/// The statement keywords of OpenQASM 3 that need classical state.
/// (`if` is read by [`Parser::conditional`] when its bodies are Paulis; a
/// stray `else` is an error.)
const UNSUPPORTED: &[&str] = &[
    "else",
    "for",
    "while",
    "def",
    "defcal",
    "defcalgrammar",
    "cal",
    "let",
    "const",
    "input",
    "output",
    "int",
    "uint",
    "float",
    "angle",
    "bool",
    "complex",
    "duration",
    "stretch",
    "delay",
    "box",
    "return",
    "break",
    "continue",
    "end",
    "extern",
    "array",
    "opaque",
    "pragma",
];

impl Parser {
    fn peek(&self) -> &Token {
        &self.tokens[self.at]
    }

    fn next(&mut self) -> Token {
        let token = self.tokens[self.at].clone();
        if self.at + 1 < self.tokens.len() {
            self.at += 1;
        }
        token
    }

    fn error<T>(&self, token: &Token, message: impl Into<String>) -> Result<T, QasmError> {
        Err(QasmError {
            line: token.line,
            column: token.column,
            message: message.into(),
        })
    }

    fn is_symbol(&self, symbol: &str) -> bool {
        matches!(&self.peek().kind, Kind::Symbol(s) if *s == symbol)
    }

    fn expect(&mut self, symbol: &str) -> Result<Token, QasmError> {
        let token = self.next();
        match &token.kind {
            Kind::Symbol(s) if *s == symbol => Ok(token),
            _ => self.error(
                &token,
                format!("expected `{symbol}`, found {}", describe(&token.kind)),
            ),
        }
    }

    fn ident(&mut self) -> Result<(String, Token), QasmError> {
        let token = self.next();
        match &token.kind {
            Kind::Ident(name) => Ok((name.clone(), token)),
            _ => self.error(
                &token,
                format!("expected a name, found {}", describe(&token.kind)),
            ),
        }
    }

    fn integer(&mut self) -> Result<usize, QasmError> {
        let token = self.next();
        match token.kind {
            Kind::Number(x) if x >= 0.0 && x.fract() == 0.0 => Ok(x as usize),
            _ => self.error(
                &token,
                format!(
                    "expected a non-negative integer, found {}",
                    describe(&token.kind)
                ),
            ),
        }
    }

    fn program(&mut self) -> Result<(), QasmError> {
        while self.peek().kind != Kind::End {
            self.statement()?;
        }
        Ok(())
    }

    fn statement(&mut self) -> Result<(), QasmError> {
        let token = self.peek().clone();
        let word = match &token.kind {
            Kind::Ident(word) => word.clone(),
            other => {
                return self.error(
                    &token,
                    format!("expected a statement, found {}", describe(other)),
                );
            }
        };
        match word.as_str() {
            "OPENQASM" => {
                self.next();
                let version = self.next();
                match version.kind {
                    Kind::Number(v) if (2.0..4.0).contains(&v) => {}
                    _ => return self.error(&version, "only OpenQASM 2 and 3 are supported"),
                }
                self.expect(";")?;
            }
            "include" => {
                self.next();
                let file = self.next();
                match &file.kind {
                    Kind::Str(name) if name == "qelib1.inc" || name == "stdgates.inc" => {}
                    Kind::Str(name) => {
                        return self.error(&file, format!("cannot include `{name}`: only qelib1.inc and stdgates.inc are built in"));
                    }
                    _ => return self.error(&file, "expected a file name"),
                }
                self.expect(";")?;
            }
            "qreg" | "creg" => {
                self.next();
                let (name, at) = self.ident()?;
                self.expect("[")?;
                let size = self.integer()?;
                self.expect("]")?;
                self.expect(";")?;
                self.declare(word == "qreg", name, size, &at)?;
            }
            "qubit" | "bit" => {
                self.next();
                let size = if self.is_symbol("[") {
                    self.next();
                    let size = self.integer()?;
                    self.expect("]")?;
                    size
                } else {
                    1
                };
                let (name, at) = self.ident()?;
                if self.is_symbol("=") {
                    return self.error(&at, "initialising classical bits is not supported");
                }
                self.expect(";")?;
                self.declare(word == "qubit", name, size, &at)?;
            }
            "gate" => self.definition()?,
            "if" => self.conditional()?,
            "measure" => {
                self.next();
                let qubits = self.operand(true)?;
                if self.is_symbol("->") {
                    self.next();
                    let clbits = self.operand(false)?;
                    self.expect(";")?;
                    self.measure(qubits, clbits, &token)?;
                } else {
                    return self.error(&token, "a measurement must store its result (`measure q -> c;` or `c = measure q;`)");
                }
            }
            "reset" => {
                self.next();
                let qubits = self.operand(true)?;
                self.expect(";")?;
                for qubit in qubits {
                    self.program.instructions.push(Instruction::Reset { qubit });
                }
            }
            "barrier" => {
                self.next();
                let mut qubits = Vec::new();
                while !self.is_symbol(";") {
                    qubits.extend(self.operand(true)?);
                    if self.is_symbol(",") {
                        self.next();
                    }
                }
                self.expect(";")?;
                if qubits.is_empty() {
                    qubits = (0..self.program.qubits).collect();
                }
                self.program
                    .instructions
                    .push(Instruction::Barrier { qubits });
            }
            "gphase" => {
                // A global phase changes nothing measurable.
                while !self.is_symbol(";") && self.peek().kind != Kind::End {
                    self.next();
                }
                self.expect(";")?;
            }
            w if UNSUPPORTED.contains(&w) => {
                return self.error(
                    &token,
                    format!(
                        "`{w}` is not supported (the Qevedo simulator runs gates, measurements and resets only)"
                    ),
                );
            }
            _ => {
                // `c = measure q;`, `c[0] = measure q[0];` or a gate call.
                let save = self.at;
                self.next();
                if self.is_symbol("[") || self.is_symbol("=") {
                    self.at = save;
                    let clbits = self.operand(false)?;
                    self.expect("=")?;
                    let (keyword, at) = self.ident()?;
                    if keyword != "measure" {
                        return self.error(
                            &at,
                            "assigning classical values other than measurements is not supported",
                        );
                    }
                    let qubits = self.operand(true)?;
                    self.expect(";")?;
                    self.measure(qubits, clbits, &token)?;
                } else {
                    self.at = save;
                    let call = self.call()?;
                    self.expect(";")?;
                    self.top_level_call(call)?;
                }
            }
        }
        Ok(())
    }

    /// `if (condition) body` or `if (condition) body else body`, where the
    /// condition tests one classical bit and each body is a gate call or a
    /// `{ ... }` block of gate calls that expand to Pauli gates only.
    ///
    /// Classical feedback ([`Instruction::Feedback`]) applies a Pauli when a
    /// bit reads 1, so the two bodies are rewritten as: the body for 0
    /// unconditionally, then, when the bit reads 1, the product of both
    /// bodies on each qubit (taking the 0-body back and the 1-body on).
    /// Paulis multiply qubit by qubit up to a phase, and that phase is
    /// conditioned on a classical bit, which only a measurement (an
    /// incoherent branch) or nothing (a fixed 0) wrote: it is unobservable.
    fn conditional(&mut self) -> Result<(), QasmError> {
        let at = self.next();
        self.expect("(")?;
        let (clbit, wanted) = self.condition()?;
        self.expect(")")?;
        let then = self.pauli_body(&at)?;
        let otherwise = if matches!(&self.peek().kind, Kind::Ident(w) if w == "else") {
            self.next();
            self.pauli_body(&at)?
        } else {
            BTreeMap::new()
        };
        let (on_one, on_zero) = if wanted {
            (then, otherwise)
        } else {
            (otherwise, then)
        };
        for (&qubit, &pauli) in &on_zero {
            if let Some(pauli) = pauli_of(pauli) {
                self.program.instructions.push(Instruction::Gate {
                    gate: pauli.gate(),
                    qubits: vec![qubit],
                });
            }
        }
        let qubits: BTreeSet<usize> = on_one.keys().chain(on_zero.keys()).copied().collect();
        for qubit in qubits {
            let flip = on_one.get(&qubit).copied().unwrap_or(0)
                ^ on_zero.get(&qubit).copied().unwrap_or(0);
            if let Some(pauli) = pauli_of(flip) {
                self.program.instructions.push(Instruction::Feedback {
                    pauli,
                    qubit,
                    clbit,
                });
            }
        }
        Ok(())
    }

    /// The tested bit and the value that selects the first body:
    /// `c[k]`, `!c[k]`, `c[k] == 1`, `c[k] != false`, or a one-bit
    /// register's name in place of `c[k]` (OpenQASM 2's `if (c == 1)`).
    fn condition(&mut self) -> Result<(usize, bool), QasmError> {
        let mut negated = false;
        while self.is_symbol("!") {
            self.next();
            negated = !negated;
        }
        let (name, at) = self.ident()?;
        let Some(&(first, size)) = self.program.cregs.get(&name) else {
            return self.error(
                &at,
                format!("unknown classical bit register `{name}` in a condition"),
            );
        };
        let clbit = if self.is_symbol("[") {
            self.next();
            let index = self.integer()?;
            self.expect("]")?;
            if index >= size {
                return self.error(
                    &at,
                    format!("index {index} is out of range for `{name}` of size {size}"),
                );
            }
            first + index
        } else if size == 1 {
            first
        } else {
            return self.error(
                &at,
                format!("a condition on `{name}`, a register of {size} bits, is not supported: only one bit can be tested (`if ({name}[k])`)"),
            );
        };
        let mut wanted = true;
        if self.is_symbol("==") || self.is_symbol("!=") {
            let equal = self.next().kind == Kind::Symbol("==");
            let token = self.next();
            let value = match &token.kind {
                Kind::Number(v) if *v == 0.0 || *v == 1.0 => *v == 1.0,
                Kind::Ident(w) if w == "true" || w == "false" => w == "true",
                other => {
                    return self.error(
                        &token,
                        format!(
                            "a bit can only be compared with 0, 1, true or false, not {}",
                            describe(other)
                        ),
                    );
                }
            };
            wanted = value == equal;
        }
        Ok((clbit, wanted != negated))
    }

    /// One gate call, or a `{ ... }` block of them, as the Pauli each
    /// qubit gets (bit 0: X, bit 1: Z, up to a phase).
    fn pauli_body(&mut self, at: &Token) -> Result<BTreeMap<usize, u8>, QasmError> {
        let block = self.is_symbol("{");
        if block {
            self.next();
        }
        let start = self.program.instructions.len();
        loop {
            if block && self.is_symbol("}") {
                self.next();
                break;
            }
            let token = self.peek().clone();
            match &token.kind {
                Kind::Ident(w) if w == "barrier" => {
                    while !self.is_symbol(";") && self.peek().kind != Kind::End {
                        self.next();
                    }
                    self.expect(";")?;
                }
                Kind::Ident(w)
                    if matches!(w.as_str(), "measure" | "reset" | "if" | "gate")
                        || UNSUPPORTED.contains(&w.as_str()) =>
                {
                    return self.error(&token, format!("`{w}` inside `if` is not supported: an `if` body can only apply Pauli gates (x, y, z, id)"));
                }
                Kind::Ident(_) => {
                    let call = self.call()?;
                    self.expect(";")?;
                    let before = self.program.instructions.len();
                    let name = call.name.clone();
                    self.top_level_call(call)?;
                    let paulis = self.program.instructions[before..].iter().all(|i| {
                        matches!(
                            i,
                            Instruction::Gate {
                                gate: Gate::I | Gate::X | Gate::Y | Gate::Z,
                                ..
                            }
                        )
                    });
                    if !paulis {
                        return self.error(&token, format!("`{name}` inside `if` is not supported: an `if` body can only apply Pauli gates (x, y, z, id), which run as classical feedback"));
                    }
                }
                Kind::End => return self.error(at, "unterminated `if` block"),
                other => {
                    return self.error(
                        &token,
                        format!("expected a gate call, found {}", describe(other)),
                    );
                }
            }
            if !block {
                break;
            }
        }
        let mut body = BTreeMap::new();
        for instruction in self.program.instructions.drain(start..) {
            if let Instruction::Gate { gate, qubits } = instruction {
                let bits = match gate {
                    Gate::X => 1,
                    Gate::Z => 2,
                    Gate::Y => 3,
                    _ => 0,
                };
                *body.entry(qubits[0]).or_insert(0) ^= bits;
            }
        }
        Ok(body)
    }

    fn declare(
        &mut self,
        quantum: bool,
        name: String,
        size: usize,
        at: &Token,
    ) -> Result<(), QasmError> {
        if self.program.qregs.contains_key(&name) || self.program.cregs.contains_key(&name) {
            return self.error(at, format!("`{name}` is already declared"));
        }
        if quantum {
            self.program.qregs.insert(name, (self.program.qubits, size));
            self.program.qubits += size;
        } else {
            self.program.cregs.insert(name, (self.program.clbits, size));
            self.program.clbits += size;
        }
        Ok(())
    }

    fn measure(
        &mut self,
        qubits: Vec<usize>,
        clbits: Vec<usize>,
        at: &Token,
    ) -> Result<(), QasmError> {
        if qubits.len() != clbits.len() {
            return self.error(
                at,
                format!(
                    "measuring {} qubits into {} bits",
                    qubits.len(),
                    clbits.len()
                ),
            );
        }
        for (qubit, clbit) in qubits.into_iter().zip(clbits) {
            self.program
                .instructions
                .push(Instruction::Measure { qubit, clbit });
        }
        Ok(())
    }

    /// `name`, `name[i]`, or (OpenQASM 3) `name[a:b]` / `name[a:step:b]`,
    /// as global qubit (or bit) indices.
    fn operand(&mut self, quantum: bool) -> Result<Vec<usize>, QasmError> {
        let (name, at) = self.ident()?;
        let registers = if quantum {
            &self.program.qregs
        } else {
            &self.program.cregs
        };
        let Some(&(first, size)) = registers.get(&name) else {
            let kind = if quantum { "qubit" } else { "classical bit" };
            return self.error(&at, format!("unknown {kind} register `{name}`"));
        };
        if !self.is_symbol("[") {
            return Ok((first..first + size).collect());
        }
        self.next();
        let start = self.integer()?;
        let mut indices = vec![start];
        if self.is_symbol(":") {
            self.next();
            let second = self.integer()?;
            let (step, end) = if self.is_symbol(":") {
                self.next();
                (second, self.integer()?)
            } else {
                (1, second)
            };
            if step == 0 {
                return self.error(&at, "a range step must be positive");
            }
            indices = (start..=end).step_by(step).collect();
        }
        self.expect("]")?;
        if let Some(&bad) = indices.iter().find(|&&i| i >= size) {
            return self.error(
                &at,
                format!("index {bad} is out of range for `{name}` of size {size}"),
            );
        }
        Ok(indices.into_iter().map(|i| first + i).collect())
    }

    /// `modifiers @ name(params) args` (arguments as names, unresolved).
    fn call(&mut self) -> Result<Call, QasmError> {
        let mut modifiers = Vec::new();
        loop {
            let token = self.peek().clone();
            let Kind::Ident(word) = &token.kind else {
                break;
            };
            let modifier = match word.as_str() {
                "inv" | "ctrl" | "negctrl" | "pow" => word.clone(),
                _ => break,
            };
            // A modifier is followed by `@` (possibly after an argument).
            let save = self.at;
            self.next();
            let argument = if self.is_symbol("(") {
                self.next();
                let e = self.expression()?;
                self.expect(")")?;
                Some(e)
            } else {
                None
            };
            if !self.is_symbol("@") {
                self.at = save;
                break;
            }
            self.next();
            let count = |e: &Option<Expr>| -> Result<usize, QasmError> {
                match e {
                    None => Ok(1),
                    Some(e) => {
                        let x = e.eval(&HashMap::new())?;
                        if x >= 1.0 && x.fract() == 0.0 {
                            Ok(x as usize)
                        } else {
                            Err(QasmError {
                                line: token.line,
                                column: token.column,
                                message: "a control count must be a positive integer".into(),
                            })
                        }
                    }
                }
            };
            modifiers.push(match modifier.as_str() {
                "inv" => Modifier::Inv,
                "ctrl" => Modifier::Ctrl(count(&argument)?),
                "negctrl" => Modifier::NegCtrl(count(&argument)?),
                _ => match argument {
                    Some(e) => Modifier::Pow(e),
                    None => return self.error(&token, "`pow` needs an exponent"),
                },
            });
        }
        let (name, at) = self.ident()?;
        let mut params = Vec::new();
        if self.is_symbol("(") {
            self.next();
            if !self.is_symbol(")") {
                loop {
                    params.push(self.expression()?);
                    if self.is_symbol(",") {
                        self.next();
                    } else {
                        break;
                    }
                }
            }
            self.expect(")")?;
        }
        // Arguments: names with optional index or range, kept as text for
        // gate bodies and resolved at the top level.
        let mut args = Vec::new();
        while !self.is_symbol(";") && !self.is_symbol("}") && self.peek().kind != Kind::End {
            let start = self.at;
            let (arg, _) = self.ident()?;
            let mut text = arg;
            if self.is_symbol("[") {
                while !self.is_symbol("]") && self.peek().kind != Kind::End {
                    self.next();
                }
                self.expect("]")?;
                // Store the index tokens' positions: re-parsed by `resolve`.
                text = format!("#{start}");
            }
            args.push(text);
            if self.is_symbol(",") {
                self.next();
            } else {
                break;
            }
        }
        Ok(Call {
            modifiers,
            name,
            params,
            args,
            line: at.line,
            column: at.column,
        })
    }

    /// Expand a top-level gate call: resolve arguments to qubits and
    /// broadcast over registers.
    fn top_level_call(&mut self, call: Call) -> Result<(), QasmError> {
        let at = Token {
            kind: Kind::End,
            line: call.line,
            column: call.column,
        };
        let mut operands = Vec::new();
        for arg in &call.args {
            let qubits = if let Some(position) = arg.strip_prefix('#') {
                let save = self.at;
                self.at = position.parse().expect("token position");
                let qubits = self.operand(true)?;
                self.at = save;
                qubits
            } else {
                match self.program.qregs.get(arg) {
                    Some(&(first, size)) => (first..first + size).collect(),
                    None => return self.error(&at, format!("unknown qubit register `{arg}`")),
                }
            };
            operands.push(qubits);
        }
        let width = operands
            .iter()
            .map(Vec::len)
            .filter(|&n| n != 1)
            .max()
            .unwrap_or(1);
        if operands.iter().any(|o| o.len() != 1 && o.len() != width) {
            return self.error(
                &at,
                "registers of different sizes cannot be broadcast together",
            );
        }
        let params: Vec<f64> = call
            .params
            .iter()
            .map(|e| e.eval(&HashMap::new()))
            .collect::<Result<_, _>>()?;
        for i in 0..width {
            let qubits: Vec<usize> = operands
                .iter()
                .map(|o| if o.len() == 1 { o[0] } else { o[i] })
                .collect();
            let mut distinct = qubits.clone();
            distinct.sort();
            distinct.dedup();
            if distinct.len() != qubits.len() {
                return self.error(
                    &at,
                    format!("`{}` is applied to the same qubit twice", call.name),
                );
            }
            let mut out = Vec::new();
            self.expand(
                &call.modifiers,
                &call.name,
                &params,
                &qubits,
                &at,
                &mut out,
                0,
            )?;
            self.program.instructions.extend(out);
        }
        Ok(())
    }

    /// The instructions of `modifiers @ name(params)` on `qubits`.
    #[allow(clippy::too_many_arguments)]
    fn expand(
        &self,
        modifiers: &[Modifier],
        name: &str,
        params: &[f64],
        qubits: &[usize],
        at: &Token,
        out: &mut Vec<Instruction>,
        depth: usize,
    ) -> Result<(), QasmError> {
        if depth > 64 {
            return self.error(at, format!("gate `{name}` is defined recursively"));
        }
        if !modifiers.is_empty() {
            // Build the unmodified gate's matrix, then apply the modifiers.
            let controls: usize = modifiers
                .iter()
                .map(|m| match m {
                    Modifier::Ctrl(n) | Modifier::NegCtrl(n) => *n,
                    _ => 0,
                })
                .sum();
            if qubits.len() <= controls {
                return self.error(
                    at,
                    format!("`{name}` with {controls} controls needs more qubits"),
                );
            }
            let targets = &qubits[controls..];
            let local: Vec<usize> = (0..targets.len()).collect();
            let mut body = Vec::new();
            self.expand(&[], name, params, &local, at, &mut body, depth + 1)?;
            let mut matrix = unitary_of(&body, targets.len());
            // Modifiers apply right to left: the one nearest the gate first.
            let mut used = controls;
            for modifier in modifiers.iter().rev() {
                match modifier {
                    Modifier::Inv => matrix = matrix.adjoint(),
                    Modifier::Pow(e) => {
                        let k = e.eval(&HashMap::new())?;
                        if k.fract() != 0.0 {
                            return self.error(at, "only integer powers are supported");
                        }
                        let base = if k < 0.0 {
                            matrix.adjoint()
                        } else {
                            matrix.clone()
                        };
                        let mut result = Matrix::identity(base.qubits());
                        for _ in 0..k.abs() as usize {
                            result = base.mul(&result);
                        }
                        matrix = result;
                    }
                    Modifier::Ctrl(n) | Modifier::NegCtrl(n) => {
                        let state = if matches!(modifier, Modifier::Ctrl(_)) {
                            (1 << n) - 1
                        } else {
                            0
                        };
                        matrix = matrix.controlled(*n, state);
                        used -= n;
                    }
                }
            }
            debug_assert_eq!(used, 0);
            // Controls nearest the gate come last in the operand list.
            out.push(Instruction::Gate {
                gate: Gate::Unitary(matrix),
                qubits: qubits.to_vec(),
            });
            return Ok(());
        }
        if let Some(definition) = self.program.gates.get(name) {
            if definition.params.len() != params.len() || definition.qubits.len() != qubits.len() {
                return self.error(
                    at,
                    format!(
                        "`{name}` takes {} parameters and {} qubits, given {} and {}",
                        definition.params.len(),
                        definition.qubits.len(),
                        params.len(),
                        qubits.len()
                    ),
                );
            }
            let env: HashMap<String, f64> = definition
                .params
                .iter()
                .cloned()
                .zip(params.iter().copied())
                .collect();
            let map: HashMap<&str, usize> = definition
                .qubits
                .iter()
                .map(String::as_str)
                .zip(qubits.iter().copied())
                .collect();
            for call in &definition.body {
                let inner_at = Token {
                    kind: Kind::End,
                    line: call.line,
                    column: call.column,
                };
                let values: Vec<f64> = call
                    .params
                    .iter()
                    .map(|e| e.eval(&env))
                    .collect::<Result<_, _>>()?;
                let args: Vec<usize> = call
                    .args
                    .iter()
                    .map(|a| {
                        map.get(a.as_str()).copied().ok_or_else(|| QasmError {
                            line: call.line,
                            column: call.column,
                            message: format!("unknown qubit `{a}` in the definition of `{name}`"),
                        })
                    })
                    .collect::<Result<_, _>>()?;
                self.expand(
                    &call.modifiers,
                    &call.name,
                    &values,
                    &args,
                    &inner_at,
                    out,
                    depth + 1,
                )?;
            }
            return Ok(());
        }
        let gate = gate(name, params)
            .ok_or(())
            .or_else(|_| self.error(at, format!("unknown gate `{name}`")))?;
        let gate = gate.map_err(|message| QasmError {
            line: at.line,
            column: at.column,
            message,
        })?;
        if gate.arity() != qubits.len() {
            return self.error(
                at,
                format!(
                    "`{name}` acts on {} qubits, given {}",
                    gate.arity(),
                    qubits.len()
                ),
            );
        }
        out.push(Instruction::Gate {
            gate,
            qubits: qubits.to_vec(),
        });
        Ok(())
    }

    fn definition(&mut self) -> Result<(), QasmError> {
        self.next();
        let (name, at) = self.ident()?;
        let mut params = Vec::new();
        if self.is_symbol("(") {
            self.next();
            while !self.is_symbol(")") {
                params.push(self.ident()?.0);
                if self.is_symbol(",") {
                    self.next();
                }
            }
            self.expect(")")?;
        }
        let mut qubits = Vec::new();
        while !self.is_symbol("{") {
            qubits.push(self.ident()?.0);
            if self.is_symbol(",") {
                self.next();
            }
        }
        self.expect("{")?;
        let mut body = Vec::new();
        while !self.is_symbol("}") {
            if self.peek().kind == Kind::End {
                return self.error(&at, format!("unterminated definition of `{name}`"));
            }
            if matches!(&self.peek().kind, Kind::Ident(w) if w == "barrier") {
                while !self.is_symbol(";") {
                    self.next();
                }
                self.expect(";")?;
                continue;
            }
            let call = self.call()?;
            if call.args.iter().any(|a| a.starts_with('#')) {
                return self.error(
                    &at,
                    format!("the body of `{name}` can only use its own qubit names"),
                );
            }
            body.push(call);
            self.expect(";")?;
        }
        self.expect("}")?;
        self.program.gates.insert(
            name,
            Definition {
                params,
                qubits,
                body,
            },
        );
        Ok(())
    }

    // Expressions: precedence climbing over + -, * /, then ^ / ** (right
    // associative), then unary minus.
    fn expression(&mut self) -> Result<Expr, QasmError> {
        let mut left = self.term()?;
        while self.is_symbol("+") || self.is_symbol("-") {
            let op = if self.is_symbol("+") { '+' } else { '-' };
            self.next();
            left = Expr::Binary(op, Box::new(left), Box::new(self.term()?));
        }
        Ok(left)
    }

    fn term(&mut self) -> Result<Expr, QasmError> {
        let mut left = self.power()?;
        while self.is_symbol("*") || self.is_symbol("/") {
            let op = if self.is_symbol("*") { '*' } else { '/' };
            self.next();
            left = Expr::Binary(op, Box::new(left), Box::new(self.power()?));
        }
        Ok(left)
    }

    fn power(&mut self) -> Result<Expr, QasmError> {
        let base = self.unary()?;
        if self.is_symbol("^") || self.is_symbol("**") {
            self.next();
            return Ok(Expr::Binary('^', Box::new(base), Box::new(self.power()?)));
        }
        Ok(base)
    }

    fn unary(&mut self) -> Result<Expr, QasmError> {
        if self.is_symbol("-") {
            self.next();
            return Ok(Expr::Negate(Box::new(self.unary()?)));
        }
        if self.is_symbol("+") {
            self.next();
            return self.unary();
        }
        let token = self.next();
        match token.kind.clone() {
            Kind::Number(x) => Ok(Expr::Number(x)),
            Kind::Symbol("(") => {
                let e = self.expression()?;
                self.expect(")")?;
                Ok(e)
            }
            Kind::Ident(name) => {
                if self.is_symbol("(") {
                    self.next();
                    let arg = self.expression()?;
                    self.expect(")")?;
                    Ok(Expr::Call(name, Box::new(arg), token.line, token.column))
                } else {
                    Ok(Expr::Name(name, token.line, token.column))
                }
            }
            other => self.error(
                &token,
                format!("expected an expression, found {}", describe(&other)),
            ),
        }
    }
}

/// The Pauli with these bits (bit 0: X, bit 1: Z), `None` for the identity.
fn pauli_of(bits: u8) -> Option<Pauli> {
    match bits {
        1 => Some(Pauli::X),
        2 => Some(Pauli::Z),
        3 => Some(Pauli::Y),
        _ => None,
    }
}

fn describe(kind: &Kind) -> String {
    match kind {
        Kind::Ident(w) => format!("`{w}`"),
        Kind::Number(x) => format!("`{x}`"),
        Kind::Str(s) => format!("\"{s}\""),
        Kind::Symbol(s) => format!("`{s}`"),
        Kind::End => "the end of the program".into(),
    }
}

/// The built-in gates of `qelib1.inc` and `stdgates.inc` (and the `U`/`CX`
/// primitives) by name: `None` if the name is not one of them, `Some(Err)`
/// for a wrong number of parameters.
pub fn gate(name: &str, p: &[f64]) -> Option<Result<Gate, String>> {
    let expect = |n: usize| -> Result<(), String> {
        if p.len() == n {
            Ok(())
        } else {
            Err(format!("`{name}` takes {n} parameters, given {}", p.len()))
        }
    };
    let fixed = |n: usize, gate: Gate| expect(n).map(|_| gate);
    let gate = match name {
        "id" | "i" => fixed(0, Gate::I),
        "u0" => expect(1).map(|_| Gate::I),
        "h" => fixed(0, Gate::H),
        "x" => fixed(0, Gate::X),
        "y" => fixed(0, Gate::Y),
        "z" => fixed(0, Gate::Z),
        "s" => fixed(0, Gate::S),
        "sdg" => fixed(0, Gate::Sdg),
        "t" => fixed(0, Gate::T),
        "tdg" => fixed(0, Gate::Tdg),
        "sx" => fixed(0, Gate::SX),
        "sxdg" => fixed(0, Gate::SXdg),
        "rx" => expect(1).map(|_| Gate::RX(p[0])),
        "ry" => expect(1).map(|_| Gate::RY(p[0])),
        "rz" => expect(1).map(|_| Gate::RZ(p[0])),
        "p" | "u1" | "phase" => expect(1).map(|_| Gate::P(p[0])),
        "u2" => expect(2).map(|_| Gate::U(PI / 2.0, p[0], p[1])),
        "U" | "u" | "u3" => expect(3).map(|_| Gate::U(p[0], p[1], p[2])),
        "CX" | "cx" | "cnot" => fixed(0, Gate::CX),
        "cy" => fixed(0, Gate::CY),
        "cz" => fixed(0, Gate::CZ),
        "ch" => fixed(0, Gate::CH),
        "cp" | "cu1" | "cphase" => expect(1).map(|_| Gate::CP(p[0])),
        "crx" => expect(1).map(|_| Gate::CRX(p[0])),
        "cry" => expect(1).map(|_| Gate::CRY(p[0])),
        "crz" => expect(1).map(|_| Gate::CRZ(p[0])),
        "swap" => fixed(0, Gate::SWAP),
        "rzz" => expect(1).map(|_| Gate::RZZ(p[0])),
        "rxx" => expect(1).map(|_| Gate::RXX(p[0])),
        "ccx" | "toffoli" => fixed(0, Gate::CCX),
        "cswap" | "fredkin" => fixed(0, Gate::CSWAP),
        // Not native in Qevedo: exact unitaries, controls first.
        "csx" => expect(0).map(|_| Gate::Unitary(Gate::SX.matrix().controlled(1, 1))),
        "cu3" => {
            expect(3).map(|_| Gate::Unitary(Gate::U(p[0], p[1], p[2]).matrix().controlled(1, 1)))
        }
        "cu" => expect(4).map(|_| {
            let phase = C64::from_polar(1.0, p[3]);
            let u = Gate::U(p[0], p[1], p[2]).matrix();
            let scaled = Matrix::new(u.entries().iter().map(|&x| x * phase).collect());
            Gate::Unitary(scaled.controlled(1, 1))
        }),
        "c3x" => expect(0).map(|_| Gate::Unitary(Gate::X.matrix().controlled(3, 0b111))),
        "c4x" => expect(0).map(|_| Gate::Unitary(Gate::X.matrix().controlled(4, 0b1111))),
        "c3sqrtx" => expect(0).map(|_| Gate::Unitary(Gate::SX.matrix().controlled(3, 0b111))),
        _ => return None,
    };
    Some(gate)
}

/// The unitary of a sequence of gate instructions on qubits `0..k`.
fn unitary_of(body: &[Instruction], k: usize) -> Matrix {
    let dim = 1 << k;
    let mut entries = vec![C64::new(0.0, 0.0); dim * dim];
    for column in 0..dim {
        let mut state = vec![C64::new(0.0, 0.0); dim];
        state[column] = C64::new(1.0, 0.0);
        for instruction in body {
            if let Instruction::Gate { gate, qubits } = instruction {
                crate::reference::apply(&mut state, &gate.matrix(), qubits);
            }
        }
        for (row, value) in state.into_iter().enumerate() {
            entries[row * dim + column] = value;
        }
    }
    Matrix::new(entries)
}
