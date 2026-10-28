//! Expectation values of IBM's 127-qubit kicked Ising circuit by Pauli
//! propagation.
//!
//!     cargo run --release --example pauliprop -- <steps> <θ_h> <observable> [min |coefficient|] [max terms]
//!         [--qasm FILE]
//!
//! The observable is a sparse Pauli string such as `Z62` or `"X37 Y75 Z38"`.
//! `QVD_THREADS=<n>` sets the thread count.

use std::time::Instant;

use qvd::library::{eagle_heavy_hex, kicked_ising};
use qvd::pauliprop::{PropagationOptions, expectation, parse_pauli};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .map(|i| args[i + 1].clone())
    };
    let positional: Vec<&String> = args.iter().take_while(|a| !a.starts_with("--")).collect();
    let steps: usize = positional.first().map_or(20, |s| s.parse().unwrap());
    let theta_h: f64 = positional.get(1).map_or(0.7, |s| s.parse().unwrap());
    let observable = positional.get(2).map_or("Z62", |s| s.as_str());
    let options = PropagationOptions {
        min_abs_coefficient: positional.get(3).map_or(1e-5, |s| s.parse().unwrap()),
        max_terms: positional.get(4).map(|s| s.parse().unwrap()),
    };
    let circuit = kicked_ising(127, &eagle_heavy_hex(), steps, theta_h);
    if let Some(path) = flag("--qasm") {
        std::fs::write(&path, circuit.to_qasm().unwrap()).unwrap();
    }
    let threads = std::env::var("QVD_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(rayon::current_num_threads);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    let start = Instant::now();
    let result = pool
        .install(|| {
            expectation(
                &circuit,
                &vec![(1.0, parse_pauli(observable).unwrap())],
                &options,
            )
        })
        .unwrap();
    let total = start.elapsed().as_secs_f64();
    println!(
        "qvd: <{observable}> after {steps} steps at θ_h = {theta_h}: {:.6} (dropped one-norm {:.3e}), \
         {} rotations, peak {} terms, final {}, {total:.3} s on {threads} threads",
        result.value,
        result.truncated_one_norm,
        result.rotations,
        result.peak_terms,
        result.final_terms,
    );
}
