#!/usr/bin/env python3
"""
Benchmarking script for Patch Rotation circuits.

Sweeps through circuit configurations, computing:
- Graph-like distance for each configuration
- Logical error rates using pymatching decoder

Uses sinter for parallelized sampling and decoding.

Parameters swept:
- k: 1, 2, 3
- basis: 'z', 'x' (logical qubit init/measure basis)
- Physical error rate: np.logspace(-3.5, -2, 7)
"""

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import stim
import sinter

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

from patch_rotation_manual import (
    generate_patch_rotation_circuit,
    generate_crumble_url,
)
from tqec import NoiseModel


# =============================================================================
# Configuration
# =============================================================================

# Parameter ranges
K_VALUES = [1, 2, 3]
BASIS_VALUES = ['z', 'x']  # Logical qubit init/measure basis
PHYSICAL_ERROR_RATES = np.logspace(-3.5, -2, 7)  # ~[0.000316, 0.001, 0.00316, 0.01]

# Sampling configuration
MAX_SHOTS = 100_000_000
MAX_ERRORS = 3000
NUM_WORKERS = 10
RANDOM_SEED = 42

# Output file
OUTPUT_CSV = 'benchmark_data/patch_rotation_benchmark.csv'


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CircuitConfig:
    """Configuration for a single circuit variant."""
    k: int
    basis: str  # 'z' or 'x'
    
    def __str__(self):
        return f"k={self.k}, basis={self.basis}"


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    # Circuit configuration
    k: int
    basis: str
    distance: Optional[int]
    
    # Error rate configuration
    physical_error_rate: float
    
    # Results
    logical_error_rate: float
    errors: int
    shots: int
    error_bar: float
    decode_time: float = 0.0


# =============================================================================
# Circuit Generation and Distance Computation
# =============================================================================

def generate_circuit_for_config(config: CircuitConfig, return_both: bool = False) -> stim.Circuit:
    """Generate a patch rotation circuit for the given configuration (without noise, compacted).
    
    Args:
        config: Circuit configuration
        return_both: If True, return tuple (before_compact, after_compact)
        
    Returns:
        The compacted circuit (without noise), or tuple if return_both=True
    """
    from compact_circuit import compact_and_delay_init
    
    # Generate circuit WITHOUT noise
    circuit_before = generate_patch_rotation_circuit(k=config.k, manhattan_radius=2, basis=config.basis)
    
    # Compact the circuit (ASAP + ALAP scheduling)
    circuit_after = compact_and_delay_init(circuit_before)
    
    if return_both:
        return circuit_before, circuit_after
    return circuit_after


def calculate_graphlike_distance(circuit: stim.Circuit) -> int | None:
    """Calculate the graph-like distance of a circuit."""
    try:
        shortest_error = circuit.shortest_graphlike_error(canonicalize_circuit_errors=True)
        return len(shortest_error)
    except Exception as e:
        print(f"Error calculating graph-like distance: {e}")
        return None


def compute_distances_for_all_configs() -> dict[tuple[int, str], int]:
    """
    Compute graph-like distance for all circuit configurations.
    
    Returns:
        Dictionary mapping (k, basis) to distance
    """
    print("=" * 70)
    print("Computing Graph-like Distances")
    print("=" * 70)
    
    distances = {}
    
    # Use a small noise model for distance calculation
    # (distance calculation requires errors in the circuit)
    distance_noise_model = NoiseModel.uniform_depolarizing(0.001)
    
    for basis in BASIS_VALUES:
        for k in K_VALUES:
            config = CircuitConfig(k, basis)
            print(f"\nGenerating circuit: {config}")
            
            # Generate compacted circuit (without noise)
            circuit = generate_circuit_for_config(config)
            
            # Add noise for distance calculation
            noisy_circuit = distance_noise_model.noisy_circuit(circuit)
            
            print(f"  Qubits: {noisy_circuit.num_qubits}, Detectors: {noisy_circuit.num_detectors}")
            
            # Calculate distance
            distance = calculate_graphlike_distance(noisy_circuit)
            distances[(k, basis)] = distance
            
            print(f"  Graph-like distance: {distance} (expected: {2*k+1})")
    
    print("\n" + "=" * 70)
    print("Distance Summary")
    print("=" * 70)
    print(f"{'k':<5} {'Basis':<8} {'Distance':<10} {'Expected':<10}")
    print("-" * 40)
    for (k, basis), distance in sorted(distances.items()):
        expected = 2 * k + 1
        match = "✓" if distance == expected else "✗"
        print(f"{k:<5} {basis:<8} {distance:<10} {expected:<10} {match}")
    
    return distances


# =============================================================================
# Sinter-based Benchmarking
# =============================================================================

def run_sinter_for_single_task(
    circuit: stim.Circuit,
    metadata: dict,
    max_shots: int,
    max_errors: int,
    num_workers: int,
) -> BenchmarkResult:
    """
    Run sinter benchmark for a single circuit.
    
    Args:
        circuit: The stim circuit to benchmark
        metadata: Dictionary with configuration metadata
        max_shots: Maximum shots per task
        max_errors: Maximum errors for early stopping
        num_workers: Number of parallel workers
        
    Returns:
        BenchmarkResult object
    """
    task = sinter.Task(
        circuit=circuit,
        decoder='pymatching',
        json_metadata=metadata,
    )
    
    # Run sinter
    start_time = time.time()
    stats = sinter.collect(
        tasks=[task],
        max_shots=max_shots,
        max_errors=max_errors,
        num_workers=num_workers,
    )
    decode_time = time.time() - start_time
    
    # Process result
    if stats:
        stat = stats[0]
        
        if stat.shots > 0:
            error_rate = stat.errors / stat.shots
            error_bar = np.sqrt(error_rate * (1 - error_rate) / stat.shots)
        else:
            error_rate = 0.0
            error_bar = 0.0
        
        result = BenchmarkResult(
            k=metadata['k'],
            basis=metadata['basis'],
            distance=metadata.get('distance'),
            physical_error_rate=metadata['physical_error_rate'],
            logical_error_rate=error_rate,
            errors=stat.errors,
            shots=stat.shots,
            error_bar=error_bar,
            decode_time=decode_time,
        )
        
        print(f"    {stat.errors}/{stat.shots} errors, rate={error_rate:.6f}, time={decode_time:.1f}s")
        return result
    
    return None


def append_results_to_csv(results: list[BenchmarkResult], filepath: str, write_header: bool = False) -> None:
    """Append benchmark results to CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    fieldnames = [
        'k',
        'basis',
        'distance',
        'physical_error_rate',
        'logical_error_rate',
        'errors',
        'shots',
        'error_bar',
        'decode_time',
    ]
    
    mode = 'w' if write_header else 'a'
    with open(filepath, mode, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        
        for result in results:
            writer.writerow({
                'k': result.k,
                'basis': result.basis,
                'distance': result.distance,
                'physical_error_rate': result.physical_error_rate,
                'logical_error_rate': result.logical_error_rate,
                'errors': result.errors,
                'shots': result.shots,
                'error_bar': result.error_bar,
                'decode_time': result.decode_time,
            })


# =============================================================================
# Main Benchmarking Loop
# =============================================================================

def run_benchmark(skip_distance: bool = True) -> list[BenchmarkResult]:
    """
    Run the full benchmark sweep using sinter for parallelization.
    One configuration at a time, saving to CSV after each.
    
    Loop order (outer to inner):
    1. basis
    2. k
    3. physical_error_rate
    
    Args:
        skip_distance: If True, skip distance calculation (use None for distance values)
    
    Returns:
        List of BenchmarkResult objects
    """
    all_results = []
    
    # Compute distances (or skip)
    if skip_distance:
        print("Skipping distance calculation (distances already verified)")
        distances = {}  # Will use None for all distances
    else:
        distances = compute_distances_for_all_configs()
    
    # Count total configurations
    total_tasks = len(BASIS_VALUES) * len(K_VALUES) * len(PHYSICAL_ERROR_RATES)
    
    print("\n" + "=" * 70)
    print("Running Logical Error Rate Benchmarks with Sinter")
    print("=" * 70)
    print(f"Total tasks: {total_tasks}")
    print(f"Decoder: pymatching")
    print(f"Max shots per config: {MAX_SHOTS:,}")
    print(f"Max errors for early stopping: {MAX_ERRORS:,}")
    print(f"Number of workers: {NUM_WORKERS}")
    print(f"Output file: {OUTPUT_CSV} (incremental save)")
    print()
    
    # Initialize CSV with header
    append_results_to_csv([], OUTPUT_CSV, write_header=True)
    
    current_task = 0
    
    for basis in BASIS_VALUES:
        print(f"\n{'='*70}")
        print(f"BASIS: {basis}")
        print(f"{'='*70}")
        
        for k in K_VALUES:
            print(f"\n  k={k}")
            
            for noise_level in PHYSICAL_ERROR_RATES:
                current_task += 1
                config = CircuitConfig(k, basis)
                distance = distances.get((k, basis))
                
                print(f"    [{current_task}/{total_tasks}] p={noise_level:.6f}")
                
                # Generate circuit (without noise)
                circuit = generate_circuit_for_config(config)
                
                # Add noise using NoiseModel.noisy_circuit()
                noise_model = NoiseModel.uniform_depolarizing(noise_level)
                noisy_circuit = noise_model.noisy_circuit(circuit)
                
                metadata = {
                    'k': k,
                    'basis': basis,
                    'distance': distance,
                    'physical_error_rate': noise_level,
                }
                
                # Run benchmark
                try:
                    result = run_sinter_for_single_task(
                        circuit=noisy_circuit,
                        metadata=metadata,
                        max_shots=MAX_SHOTS,
                        max_errors=MAX_ERRORS,
                        num_workers=NUM_WORKERS,
                    )
                    
                    if result:
                        # Save result immediately
                        append_results_to_csv([result], OUTPUT_CSV, write_header=False)
                        all_results.append(result)
                    
                except Exception as e:
                    print(f"      ERROR: {e}")
                    import traceback
                    traceback.print_exc()
    
    return all_results


# =============================================================================
# CSV Output
# =============================================================================

def save_results_to_csv(results: list[BenchmarkResult], filepath: str) -> None:
    """Save benchmark results to CSV file."""
    print(f"\nSaving results to {filepath}...")
    
    fieldnames = [
        'k',
        'basis',
        'distance',
        'physical_error_rate',
        'logical_error_rate',
        'errors',
        'shots',
        'error_bar',
        'decode_time',
    ]
    
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            writer.writerow({
                'k': result.k,
                'basis': result.basis,
                'distance': result.distance,
                'physical_error_rate': result.physical_error_rate,
                'logical_error_rate': result.logical_error_rate,
                'errors': result.errors,
                'shots': result.shots,
                'error_bar': result.error_bar,
                'decode_time': result.decode_time,
            })
    
    print(f"Saved {len(results)} results to {filepath}")


def load_results_from_csv(filepath: str) -> list[dict]:
    """Load benchmark results from CSV file."""
    results = []
    with open(filepath, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Convert types
            row['k'] = int(row['k'])
            row['basis'] = row['basis']
            row['distance'] = int(row['distance']) if row['distance'] else None
            row['physical_error_rate'] = float(row['physical_error_rate'])
            row['logical_error_rate'] = float(row['logical_error_rate'])
            row['errors'] = int(row['errors'])
            row['shots'] = int(row['shots'])
            row['error_bar'] = float(row['error_bar'])
            row['decode_time'] = float(row['decode_time'])
            results.append(row)
    return results


# =============================================================================
# Plotting
# =============================================================================

def plot_results(results: list[BenchmarkResult], output_dir: str = 'benchmark_plots') -> None:
    """Generate plots from benchmark results."""
    os.makedirs(output_dir, exist_ok=True)
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(K_VALUES)))
    linestyles = {'z': '-', 'x': '--'}
    markers = {'z': 'o', 'x': 's'}
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title('Patch Rotation Benchmark (pymatching)')
    ax.set_xlabel('Physical Error Rate')
    ax.set_ylabel('Logical Error Rate')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    for basis in BASIS_VALUES:
        for k, color in zip(K_VALUES, colors):
            # Filter results for this k and basis
            filtered = [r for r in results if r.k == k and r.basis == basis]
            
            if not filtered:
                continue
            
            # Sort by physical error rate
            filtered.sort(key=lambda r: r.physical_error_rate)
            
            x = [r.physical_error_rate for r in filtered]
            y = [r.logical_error_rate for r in filtered]
            yerr = [r.error_bar for r in filtered]
            
            distance = filtered[0].distance if filtered[0].distance else 2*k+1
            ax.errorbar(x, y, yerr=yerr, 
                       marker=markers[basis],
                       linestyle=linestyles[basis],
                       label=f'k={k}, basis={basis} (d={distance})',
                       color=color, capsize=3)
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'patch_rotation_benchmark.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")
    plt.close()


# =============================================================================
# Summary
# =============================================================================

def print_summary(results: list[BenchmarkResult]) -> None:
    """Print a summary of benchmark results."""
    print("\n" + "=" * 70)
    print("Benchmark Summary")
    print("=" * 70)
    
    for basis in BASIS_VALUES:
        print(f"\nBasis: {basis}")
        print(f"{'k':<5} {'Distance':<10} {'p_phys':<12} {'p_log':<12} {'Errors':<10} {'Shots':<12}")
        print("-" * 70)
        
        sorted_results = sorted(
            [r for r in results if r.basis == basis],
            key=lambda r: (r.k, r.physical_error_rate)
        )
        
        for r in sorted_results:
            print(f"{r.k:<5} {r.distance or 'N/A':<10} {r.physical_error_rate:<12.6f} "
                  f"{r.logical_error_rate:<12.6f} {r.errors:<10} {r.shots:<12}")


# =============================================================================
# Crumble URL Generation
# =============================================================================

def save_crumble_urls_html(urls: dict, output_dir: str = 'crumble_urls') -> None:
    """Save Crumble URLs to an HTML file for easy access."""
    os.makedirs(output_dir, exist_ok=True)
    
    html_path = os.path.join(output_dir, 'patch_rotation_crumble.html')
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Patch Rotation Circuit Crumble URLs</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        .config { margin: 20px 0; padding: 10px; background: #f5f5f5; border-radius: 5px; }
        .config h3 { margin: 0 0 10px 0; color: #555; }
        a { color: #0066cc; text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>Patch Rotation Circuit Crumble URLs</h1>
"""
    
    for config_str, url in sorted(urls.items()):
        html_content += f"""
    <div class="config">
        <h3>{config_str}</h3>
        <p><a href="{url}" target="_blank">Open in Crumble →</a></p>
    </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"Saved Crumble URLs to {html_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Benchmark Patch Rotation circuits')
    parser.add_argument('--csv', type=str, default=OUTPUT_CSV,
                       help='Output CSV file path')
    parser.add_argument('--plot-only', action='store_true',
                       help='Only plot from existing CSV (skip benchmarking)')
    parser.add_argument('--distance-only', action='store_true',
                       help='Only compute distances (skip benchmarking)')
    parser.add_argument('--crumble-only', action='store_true',
                       help='Only generate Crumble URLs')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Patch Rotation Benchmark")
    print("=" * 70)
    
    # Crumble-only mode
    if args.crumble_only:
        print("\nGenerating Crumble URLs...")
        urls = {}
        for basis in BASIS_VALUES:
            for k in K_VALUES:
                config_str = f"k={k}, basis={basis}"
                try:
                    url = generate_crumble_url(k=k, manhattan_radius=2, basis=basis)
                    urls[config_str] = url
                    print(f"  Generated URL for {config_str}")
                except Exception as e:
                    print(f"  Error generating URL for {config_str}: {e}")
        
        if urls:
            save_crumble_urls_html(urls)
        
        print("\nCrumble URL generation complete!")
        return
    
    # Distance-only mode
    if args.distance_only:
        compute_distances_for_all_configs()
        print("\nDistance computation complete!")
        return
    
    # Plot-only mode
    if args.plot_only:
        print(f"\nLoading results from {args.csv}...")
        results_dicts = load_results_from_csv(args.csv)
        
        # Convert to BenchmarkResult objects
        results = []
        for r in results_dicts:
            results.append(BenchmarkResult(
                k=r['k'],
                basis=r['basis'],
                distance=r['distance'],
                physical_error_rate=r['physical_error_rate'],
                logical_error_rate=r['logical_error_rate'],
                errors=r['errors'],
                shots=r['shots'],
                error_bar=r['error_bar'],
                decode_time=r['decode_time'],
            ))
        
        # Plot
        plot_results(results)
        
        # Print summary
        print_summary(results)
        
        print("\nPlot-only mode complete!")
        return
    
    # Full benchmark mode
    print("Configuration:")
    print(f"  k values: {K_VALUES}")
    print(f"  Basis values: {BASIS_VALUES}")
    print(f"  Physical error rates: {[f'{p:.6f}' for p in PHYSICAL_ERROR_RATES]}")
    print(f"  Decoder: pymatching")
    print(f"  Max shots: {MAX_SHOTS:,}")
    print(f"  Max errors: {MAX_ERRORS:,}")
    print(f"  Num workers: {NUM_WORKERS}")
    print(f"  Output file: {args.csv}")
    print()
    
    start_time = time.time()
    
    # Generate Crumble URLs
    print("=" * 70)
    print("Generating Crumble URLs")
    print("=" * 70)
    
    all_crumble_urls = {}
    for basis in BASIS_VALUES:
        for k in K_VALUES:
            config_str = f"k={k}, basis={basis}"
            try:
                url = generate_crumble_url(k=k, manhattan_radius=2, basis=basis)
                all_crumble_urls[config_str] = url
                print(f"  Generated URL for {config_str}")
            except Exception as e:
                print(f"  Error generating URL for {config_str}: {e}")
    
    # Save Crumble URLs to HTML
    if all_crumble_urls:
        save_crumble_urls_html(all_crumble_urls)
    
    # Run benchmark with distance calculation enabled
    results = run_benchmark(skip_distance=False)
    
    # Plot results
    plot_results(results)
    
    # Print summary
    print_summary(results)
    
    print(f"\nResults saved incrementally to: {args.csv}")
    
    total_time = time.time() - start_time
    print(f"\nTotal benchmark time: {total_time:.2f}s ({total_time/60:.1f} minutes)")


if __name__ == "__main__":
    main()
