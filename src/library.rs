//! Standard circuits for tests and benchmarks.

use rand::{RngExt, SeedableRng};
use rand_pcg::Pcg64;

use crate::circuit::{Circuit, Gate, Instruction};

/// GHZ state preparation: H on qubit 0, then a CX chain.
pub fn ghz(n: usize) -> Circuit {
    let mut c = Circuit::new(n);
    c.h(0);
    for q in 1..n {
        c.cx(q - 1, q);
    }
    c
}

/// Quantum Fourier transform (without the final qubit reversal).
pub fn qft(n: usize) -> Circuit {
    let mut c = Circuit::new(n);
    for target in (0..n).rev() {
        c.h(target);
        for control in (0..target).rev() {
            let angle = std::f64::consts::PI / (1u64 << (target - control)) as f64;
            c.cp(angle, control, target);
        }
    }
    c
}

/// A random circuit in the style of Google's supremacy benchmarks (and
/// qsim's): `depth` layers, each a random single-qubit gate from
/// {sqrt(X), sqrt(Y), sqrt(W)} on every qubit followed by CZ gates on a
/// brick pattern of neighbouring pairs (on a line of qubits).
pub fn random_circuit(n: usize, depth: usize, seed: u64) -> Circuit {
    let mut rng = Pcg64::seed_from_u64(seed);
    let mut c = Circuit::new(n);
    let half = std::f64::consts::FRAC_PI_2;
    for layer in 0..depth {
        for q in 0..n {
            let gate = match rng.random_range(0..3) {
                0 => Gate::RX(half),
                1 => Gate::RY(half),
                // sqrt(W), W = (X + Y) / sqrt(2)
                _ => Gate::U(
                    half,
                    -std::f64::consts::FRAC_PI_4,
                    std::f64::consts::FRAC_PI_4,
                ),
            };
            c.gate(gate, &[q]);
        }
        let mut q = layer % 2;
        while q + 1 < n {
            c.cz(q, q + 1);
            q += 2;
        }
    }
    c
}

/// A Google-style random circuit on a `rows x cols` grid (qubit
/// `r * cols + c`): each layer applies √X, √Y or √W to every qubit, then CZ
/// on one of four coupler patterns in turn (horizontal and vertical,
/// alternating offsets), so every qubit entangles with all its grid
/// neighbours every four layers.
pub fn random_grid_circuit(rows: usize, cols: usize, depth: usize, seed: u64) -> Circuit {
    let mut rng = Pcg64::seed_from_u64(seed);
    let n = rows * cols;
    let mut c = Circuit::new(n);
    let half = std::f64::consts::FRAC_PI_2;
    for layer in 0..depth {
        for q in 0..n {
            let gate = match rng.random_range(0..3) {
                0 => Gate::RX(half),
                1 => Gate::RY(half),
                _ => Gate::U(
                    half,
                    -std::f64::consts::FRAC_PI_4,
                    std::f64::consts::FRAC_PI_4,
                ),
            };
            c.gate(gate, &[q]);
        }
        let offset = (layer / 2) % 2;
        for r in 0..rows {
            for col in 0..cols {
                let q = r * cols + col;
                if layer % 2 == 0 && col % 2 == offset && col + 1 < cols {
                    c.cz(q, q + 1);
                }
                if layer % 2 == 1 && r % 2 == offset && r + 1 < rows {
                    c.cz(q, q + cols);
                }
            }
        }
    }
    c
}

/// A random circuit over the full standard gate set (rotations with random
/// angles, CX/CZ/CP/SWAP/CCX on random qubits). Used for correctness tests.
pub fn random_gate_circuit(n: usize, gates: usize, seed: u64) -> Circuit {
    let mut rng = Pcg64::seed_from_u64(seed);
    let mut c = Circuit::new(n);
    for _ in 0..gates {
        let mut qubits: Vec<usize> = (0..n).collect();
        for i in 0..n.min(3) {
            let j = rng.random_range(i..n);
            qubits.swap(i, j);
        }
        let angle = rng.random::<f64>() * std::f64::consts::TAU;
        let choice = rng.random_range(
            0..if n >= 3 {
                14
            } else if n == 2 {
                12
            } else {
                8
            },
        );
        let (gate, arity) = match choice {
            0 => (Gate::H, 1),
            1 => (Gate::X, 1),
            2 => (Gate::RX(angle), 1),
            3 => (Gate::RY(angle), 1),
            4 => (Gate::RZ(angle), 1),
            5 => (Gate::U(angle, angle * 0.5, angle * 0.25), 1),
            6 => (Gate::T, 1),
            7 => (Gate::SX, 1),
            8 => (Gate::CX, 2),
            9 => (Gate::CZ, 2),
            10 => (Gate::CP(angle), 2),
            11 => (Gate::SWAP, 2),
            12 => (Gate::CCX, 3),
            _ => (Gate::CSWAP, 3),
        };
        c.gate(gate, &qubits[..arity]);
    }
    c
}

/// A rotated surface-code memory experiment of distance `d`: `rounds`
/// rounds of stabilizer measurement (reset ancillas, CNOTs to the
/// neighbouring data qubits, measure ancillas), then every data qubit is
/// measured. Data qubits are `0..d*d`; the `d*d - 1` ancillas follow.
///
/// This is the kind of circuit stabilizer simulators are built for: about
/// `2 d^2` qubits, mostly deterministic measurements.
pub fn surface_code(d: usize, rounds: usize) -> Circuit {
    assert!(d >= 2);
    let data = |r: usize, c: usize| r * d + c;
    // Plaquette (i, j), i, j in 0..=d, touches data (i-1|i, j-1|j). Interior
    // plaquettes alternate X/Z in a checkerboard; on the top and bottom
    // boundaries only X plaquettes are kept, on the left and right only Z.
    let mut plaquettes: Vec<(bool, Vec<Option<usize>>)> = Vec::new();
    for i in 0..=d {
        for j in 0..=d {
            let is_x = (i + j) % 2 == 0;
            let corner = |di: usize, dj: usize| -> Option<usize> {
                let (r, c) = ((i + di).checked_sub(1)?, (j + dj).checked_sub(1)?);
                (r < d && c < d).then(|| data(r, c))
            };
            // Neighbours in hook-safe order: NW, NE, SW, SE for X; NW, SW, NE, SE for Z.
            let order = if is_x {
                [(0, 0), (0, 1), (1, 0), (1, 1)]
            } else {
                [(0, 0), (1, 0), (0, 1), (1, 1)]
            };
            let neighbours: Vec<Option<usize>> = order.iter().map(|&(a, b)| corner(a, b)).collect();
            let weight = neighbours.iter().flatten().count();
            let on_row_boundary = i == 0 || i == d;
            let on_col_boundary = j == 0 || j == d;
            let keep = match weight {
                4 => true,
                2 => (on_row_boundary && is_x) || (on_col_boundary && !is_x),
                _ => false,
            };
            if keep {
                plaquettes.push((is_x, neighbours));
            }
        }
    }
    let ancilla0 = d * d;
    let n = ancilla0 + plaquettes.len();
    let mut c = Circuit::new(n);
    let mut clbit = 0;
    for _ in 0..rounds {
        for a in 0..plaquettes.len() {
            c.reset(ancilla0 + a);
        }
        for (a, (is_x, _)) in plaquettes.iter().enumerate() {
            if *is_x {
                c.h(ancilla0 + a);
            }
        }
        for step in 0..4 {
            for (a, (is_x, neighbours)) in plaquettes.iter().enumerate() {
                if let Some(q) = neighbours[step] {
                    if *is_x {
                        c.cx(ancilla0 + a, q);
                    } else {
                        c.cx(q, ancilla0 + a);
                    }
                }
            }
        }
        for (a, (is_x, _)) in plaquettes.iter().enumerate() {
            if *is_x {
                c.h(ancilla0 + a);
            }
        }
        for a in 0..plaquettes.len() {
            c.measure(ancilla0 + a, clbit);
            clbit += 1;
        }
    }
    for q in 0..d * d {
        c.measure(q, clbit);
        clbit += 1;
    }
    c
}

/// `circuit` with uniform circuit-level depolarizing noise of strength `p`,
/// the usual model of error-correction threshold studies:
/// `DEPOLARIZE1(p)` after every one-qubit gate, `DEPOLARIZE2(p)` after
/// every two-qubit gate, and a bit flip with probability `p` before every
/// measurement and after every reset. These are Stim's generated circuits'
/// `after_clifford_depolarization`, `before_measure_flip_probability` and
/// `after_reset_flip_probability`, all set to `p`; idle qubits get no noise.
///
/// As in Stim's circuits, each layer's noise is one instruction: a channel
/// is held back until a later instruction touches one of its qubits (Pauli
/// channels commute with gates on other qubits), and a run of measurements
/// gets its bit flips first, as one instruction.
pub fn with_depolarizing_noise(circuit: &Circuit, p: f64) -> Circuit {
    use crate::circuit::Noise;
    let mut noisy = Circuit::new(circuit.num_qubits);
    noisy.num_clbits = circuit.num_clbits;
    /// Noise channels held back, and which qubits they touch.
    struct Held {
        pending: Vec<(Noise, Vec<usize>)>,
        held: Vec<bool>,
    }
    impl Held {
        fn hold(&mut self, channel: Noise, qubits: &[usize]) {
            match self.pending.iter_mut().find(|(c, _)| *c == channel) {
                Some((_, targets)) => targets.extend_from_slice(qubits),
                None => self.pending.push((channel, qubits.to_vec())),
            }
            for &q in qubits {
                self.held[q] = true;
            }
        }

        fn flush(&mut self, noisy: &mut Circuit) {
            for (channel, qubits) in self.pending.drain(..) {
                for &q in &qubits {
                    self.held[q] = false;
                }
                noisy.noise(channel, &qubits);
            }
        }
    }
    let mut held = Held {
        pending: Vec::new(),
        held: vec![false; circuit.num_qubits],
    };
    let flip = Noise::Pauli1 {
        px: p,
        py: 0.0,
        pz: 0.0,
    };
    let instructions = &circuit.instructions;
    let mut i = 0;
    while i < instructions.len() {
        let instruction = &instructions[i];
        if let Instruction::Measure { .. } = instruction {
            // The bit flips of the whole run of measurements, then the run.
            let run: Vec<usize> = instructions[i..]
                .iter()
                .map_while(|m| match m {
                    Instruction::Measure { qubit, .. } => Some(*qubit),
                    _ => None,
                })
                .collect();
            held.hold(flip, &run);
            held.flush(&mut noisy);
            noisy
                .instructions
                .extend_from_slice(&instructions[i..i + run.len()]);
            i += run.len();
            continue;
        }
        let touched: Vec<usize> = match instruction {
            Instruction::Gate { qubits, .. } | Instruction::Barrier { qubits } => qubits.clone(),
            Instruction::Reset { qubit } => vec![*qubit],
            Instruction::Noise { qubits, .. } => qubits.clone(),
            Instruction::Measure { .. } => unreachable!(),
        };
        if touched.iter().any(|&q| held.held[q]) {
            held.flush(&mut noisy);
        }
        noisy.instructions.push(instruction.clone());
        match instruction {
            Instruction::Gate { qubits, .. } if qubits.len() == 1 => {
                held.hold(
                    Noise::Pauli1 {
                        px: p / 3.0,
                        py: p / 3.0,
                        pz: p / 3.0,
                    },
                    qubits,
                );
            }
            Instruction::Gate { qubits, .. } if qubits.len() == 2 => {
                held.hold(Noise::Depolarize2(p), qubits);
            }
            Instruction::Gate { gate, .. } => {
                panic!(
                    "no depolarizing channel for the {}-qubit {gate:?}",
                    gate.arity()
                )
            }
            Instruction::Reset { qubit } => held.hold(flip, &[*qubit]),
            _ => {}
        }
        i += 1;
    }
    held.flush(&mut noisy);
    noisy
}

/// A random Clifford circuit: `depth` layers of random single-qubit
/// Cliffords on every qubit and CX on random disjoint pairs.
pub fn random_clifford(n: usize, depth: usize, seed: u64) -> Circuit {
    let mut rng = Pcg64::seed_from_u64(seed);
    let mut c = Circuit::new(n);
    for _ in 0..depth {
        for q in 0..n {
            match rng.random_range(0..3) {
                0 => c.h(q),
                1 => c.s(q),
                _ => c.sx(q),
            };
        }
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = rng.random_range(0..=i);
            order.swap(i, j);
        }
        for pair in order.chunks_exact(2) {
            c.cx(pair[0], pair[1]);
        }
    }
    c
}

/// A random Clifford+T circuit: the layers of [`random_clifford`], with
/// `t_count` T gates on random qubits after random layers. The kind of
/// circuit near-Clifford simulators are built for: entangling Cliffords on
/// many qubits, few non-Clifford gates.
pub fn random_clifford_t(n: usize, depth: usize, t_count: usize, seed: u64) -> Circuit {
    let mut rng = Pcg64::seed_from_u64(seed ^ 0x5eed);
    let mut placements: Vec<(usize, usize)> = (0..t_count)
        .map(|_| (rng.random_range(0..depth.max(1)), rng.random_range(0..n)))
        .collect();
    placements.sort();
    let layers = random_clifford(n, depth, seed);
    // Each layer of `random_clifford` is n one-qubit gates and n/2 CX.
    let per_layer = n + n / 2;
    let mut c = Circuit::new(n);
    let mut next = placements.iter().peekable();
    for (i, instruction) in layers.instructions.into_iter().enumerate() {
        c.instructions.push(instruction);
        if (i + 1) % per_layer == 0 {
            let layer = i / per_layer;
            while let Some(&&(_, q)) = next.peek().filter(|p| p.0 == layer) {
                c.gate(Gate::T, &[q]);
                next.next();
            }
        }
    }
    c
}

/// The couplers of IBM's 127-qubit Eagle processors (heavy-hex lattice,
/// 144 edges, as in ibm_kyiv), with Qiskit's qubit numbering. Seven rows
/// of 14 or 15 qubits, joined by four bridge qubits between neighbouring
/// rows; rows and bridges are numbered in reading order.
pub fn eagle_heavy_hex() -> Vec<(usize, usize)> {
    // (first column, length) of each row; bridges sit at columns 0, 4, 8, 12
    // below even rows and 2, 6, 10, 14 below odd ones.
    let rows = [
        (0, 14),
        (0, 15),
        (0, 15),
        (0, 15),
        (0, 15),
        (0, 15),
        (1, 14),
    ];
    let mut edges = Vec::new();
    let mut starts = Vec::new();
    let mut next = 0;
    for (r, &(first, len)) in rows.iter().enumerate() {
        starts.push(next);
        edges.extend((next..next + len - 1).map(|q| (q, q + 1)));
        next += len;
        if r + 1 < rows.len() {
            // Bridges follow the row; their lower ends are numbered with the
            // next row, whose first column is always 0 or 1.
            let below_start = next + 4;
            let below_first = rows[r + 1].0;
            for (b, col) in (0..4).map(|b| (b, 4 * b + 2 * (r % 2))) {
                let bridge = next + b;
                edges.push((starts[r] + col - first, bridge));
                edges.push((bridge, below_start + col - below_first));
            }
            next += 4;
        }
    }
    edges
}

/// The kicked transverse-field Ising circuit of IBM's utility experiment
/// (Kim et al., Nature 618, 500 (2023)): `steps` Trotter steps, each
/// `RX(θ_h)` on every qubit, then `RZZ(-π/2)` on every edge.
pub fn kicked_ising(
    num_qubits: usize,
    edges: &[(usize, usize)],
    steps: usize,
    theta_h: f64,
) -> Circuit {
    let mut c = Circuit::new(num_qubits);
    for _ in 0..steps {
        for q in 0..num_qubits {
            c.rx(theta_h, q);
        }
        for &(a, b) in edges {
            c.gate(Gate::RZZ(-std::f64::consts::FRAC_PI_2), &[a, b]);
        }
    }
    c
}
