//! Time the stabilizer backend.
//!
//!     cargo run --release --example stabilizer -- surface <distance> <rounds> <shots> [--qasm FILE]
//!     cargo run --release --example stabilizer -- random <qubits> <depth> <shots> [--qasm FILE]
//!     cargo run --release --example stabilizer -- detectors <stim file> <shots>
//!
//! `--noise P` adds uniform circuit-level depolarizing noise of strength P
//! (`library::with_depolarizing_noise`); `--stim FILE` writes the circuit,
//! noise included, in Stim's format for `benchmarks/compare_stim.py`.
//!
//! `detectors` reads a circuit in Stim's format (with `DETECTOR` and
//! `OBSERVABLE_INCLUDE` annotations) and samples its detection events.
//!
//! `QVD_THREADS=all|pcores|<n>` picks the thread pool (default: P-cores).

use std::time::Instant;

use qvd::library::{random_clifford, surface_code, with_depolarizing_noise};
use qvd::stabilizer;
use qvd::threads::{Placement, pool};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let kind = args.first().map_or("surface", |s| s.as_str());
    if kind == "detectors" {
        let shots = args.get(2).map_or(10_000, |s| s.parse().unwrap());
        return detectors(&args[1], shots);
    }
    let a: usize = args.get(1).map_or(15, |s| s.parse().unwrap());
    let b: usize = args.get(2).map_or(15, |s| s.parse().unwrap());
    let shots: usize = args.get(3).map_or(10_000, |s| s.parse().unwrap());
    let mut circuit = match kind {
        "random" => random_clifford(a, b, 1),
        _ => surface_code(a, b),
    };
    if kind == "random" {
        circuit.measure_all();
    }
    if let Some(i) = args.iter().position(|s| s == "--qasm") {
        std::fs::write(&args[i + 1], circuit.to_qasm().unwrap()).unwrap();
    }
    if let Some(i) = args.iter().position(|s| s == "--noise") {
        circuit = with_depolarizing_noise(&circuit, args[i + 1].parse().unwrap());
    }
    if let Some(i) = args.iter().position(|s| s == "--stim") {
        std::fs::write(&args[i + 1], stabilizer::to_stim(&circuit).unwrap()).unwrap();
    }
    let measurements = circuit
        .instructions
        .iter()
        .filter(|i| matches!(i, qvd::Instruction::Measure { .. }))
        .count();
    let pool = pool(&placement());
    let start = Instant::now();
    let samples = pool
        .install(|| stabilizer::sample(&circuit, shots, Some(1)))
        .unwrap();
    let elapsed = start.elapsed().as_secs_f64();
    let ones: usize = (0..samples.num_clbits.min(64))
        .filter(|&c| samples.get(0, c))
        .count();
    println!(
        "qvd stabilizer: {} qubits, {} gates, {} measurements, {} shots: {:.3} s ({ones} ones in shot 0's first 64 bits)",
        circuit.num_qubits,
        circuit.gate_count(),
        measurements,
        shots,
        elapsed,
    );
}

fn placement() -> Placement {
    match std::env::var("QVD_THREADS").as_deref() {
        Ok("all") => Placement::AllThreads,
        Ok("pcores") | Err(_) => Placement::PerformanceCores,
        Ok(n) => Placement::Unpinned {
            threads: n.parse().expect("QVD_THREADS: all, pcores or a number"),
        },
    }
}

/// Read a Stim file and time its detector sampling, step by step.
fn detectors(path: &str, shots: usize) {
    let text = std::fs::read_to_string(path).unwrap();
    let pool = pool(&placement());
    let start = Instant::now();
    let circuit = qvd::stim::parse(&text).unwrap();
    let parsed = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let samples = pool
        .install(|| stabilizer::sample_detectors(&circuit, shots, Some(1)))
        .unwrap();
    let sampled = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let bytes = pool.install(|| samples.detectors_shot_major());
    let transposed = start.elapsed().as_secs_f64();
    let fired = bytes.iter().filter(|&&b| b == 1).count();
    println!(
        "qvd detectors: {} qubits, {} measurements, {} detectors, {} shots: read {parsed:.3} s, \
         sample {sampled:.3} s, shot-major {transposed:.3} s (detection rate {:.6})",
        circuit.circuit.num_qubits,
        circuit.num_measurements(),
        circuit.detectors.len(),
        shots,
        fired as f64 / bytes.len().max(1) as f64,
    );
}
