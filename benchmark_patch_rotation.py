#!/usr/bin/env python3
"""
Benchmarking script for Patch Rotation circuits.

Sweeps through circuit configurations, computing:
- Graph-like distance for each configuration
- Logical error rates using correlated PyMatching decoder

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
import pymatching


# =============================================================================
# Correlated PyMatching Decoder
# =============================================================================

class CorrelatedPymatchingDecoder(sinter.Decoder):
    """Sinter decoder wrapper for correlated PyMatching."""
    
    def decode_via_files(
        self,
        *,
        num_shots: int,
        num_dets: int,
        num_obs: int,
        dem_path: str,
        dets_b8_in_path: str,
        obs_predictions_b8_out_path: str,
        tmp_dir: str,
    ) -> None:
        """Decode using file-based interface."""
        import pathlib
        
        # Load DEM
        dem = stim.DetectorErrorModel.from_file(dem_path)
        
        # Create matcher with correlations enabled
        matcher = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
        
        # Load detector data
        dets = stim.read_shot_data_file(
            path=dets_b8_in_path,
            format='b8',
            num_detectors=num_dets,
            num_observables=0,
        )
        
        # Decode with correlations enabled
        predictions = matcher.decode_batch(dets, enable_correlations=True)
        
        # Write predictions
        stim.write_shot_data_file(
            data=predictions,
            path=obs_predictions_b8_out_path,
            format='b8',
            num_observables=num_obs,
        )


# Custom decoder registry for sinter
CUSTOM_DECODERS = {
    'pymatching': 'pymatching',  # Built-in sinter decoder
    'correlated_pymatching': CorrelatedPymatchingDecoder(),
}


# =============================================================================
# Configuration
# =============================================================================

# Parameter ranges
K_VALUES = [4]
BASIS_VALUES = ['z']  # Logical qubit init/measure basis
PHYSICAL_ERROR_RATES = [np.logspace(-4, -2, 9)[2:][::-1][-1]]  # ~[0.000316, 0.001, 0.00316, 0.01]

# Sampling configuration
MAX_SHOTS = 1500_000_000
MAX_ERRORS = 3000
NUM_WORKERS = 10
RANDOM_SEED = 42

# Decoder configuration (can be overridden via --decoder argument)
DECODER = 'correlated_pymatching'  # Can be 'pymatching' or 'correlated_pymatching'

# Output file (will be set in main() based on decoder)
OUTPUT_CSV = 'benchmark_data/patch_rotation_benchmark_correlated_pymatching.csv'


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
    decoder: str = 'correlated_pymatching',
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
    # For pymatching and correlated_pymatching, we need a decomposed (graphlike) DEM
    # Pre-compute it with decompose_errors=True as per PyMatching instructions
    detector_error_model = None
    if decoder in ['pymatching', 'correlated_pymatching']:
        detector_error_model = circuit.detector_error_model(
            decompose_errors=True,
            ignore_decomposition_failures=True
        )
    
    task = sinter.Task(
        circuit=circuit,
        decoder=decoder,
        detector_error_model=detector_error_model,  # Use pre-computed DEM for pymatching decoders
        json_metadata=metadata,
    )
    
    # Run sinter
    start_time = time.time()
    # Only pass custom_decoders if using a custom decoder
    collect_kwargs = {
        'tasks': [task],
        'max_shots': max_shots,
        'max_errors': max_errors,
        'num_workers': num_workers,
    }
    if decoder in CUSTOM_DECODERS and isinstance(CUSTOM_DECODERS[decoder], sinter.Decoder):
        collect_kwargs['custom_decoders'] = CUSTOM_DECODERS
    stats = sinter.collect(**collect_kwargs)
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
    print(f"Decoder: {DECODER}")
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
                        decoder=DECODER,
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

def results_to_sinter_stats(results: list[dict], decoder='correlated_pymatching') -> list[sinter.TaskStats]:
    """Convert results to sinter.TaskStats for plotting.
    
    Args:
        results: List of result dictionaries
        decoder: Decoder name to use in TaskStats
        
    Returns:
        List of sinter.TaskStats objects
    """
    stats_list = []
    
    for result in results:
        stats = sinter.TaskStats(
            strong_id=f"k{result['k']}_{result['basis']}_p{result['physical_error_rate']}",
            decoder=decoder,
            json_metadata={
                'p': result['physical_error_rate'],
                'k': result['k'],
                'd': 2 * result['k'] + 1,
                'basis': result['basis'],
            },
            shots=result['shots'],
            errors=result['errors'],
        )
        stats_list.append(stats)
    
    return stats_list


def fit_and_plot_distance(ax, stats_list, group_func, x_func, plot_args_func, min_points=2):
    """Fit p_logical = A * p^((d+1)/2) to data, add fit lines, and create inset d vs k plot.
    
    Args:
        ax: matplotlib axes object
        stats_list: List of sinter.TaskStats objects
        group_func: Function to group stats into curves (same as used in sinter.plot_error_rate)
        x_func: Function to extract x value (physical error rate)
        plot_args_func: Function to get plot styling for each curve
        min_points: Minimum number of points with reasonable error bars needed for fitting
    """
    from collections import defaultdict
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    # Group stats by curve
    curves = defaultdict(list)
    for s in stats_list:
        curve_id = group_func(s)
        curves[curve_id].append(s)
    
    # Collect fitted distances for inset plot: list of (k, d_eff, plot_args, curve_id)
    fitted_distances = []
    
    for idx, (curve_id, stats) in enumerate(curves.items()):
        # Extract data points: (p, p_logical, error_bar)
        points = []
        k_value = None
        for s in stats:
            if k_value is None:
                k_value = s.json_metadata.get('k')
            if s.errors > 0 and s.shots > 0:
                p = x_func(s)
                p_logical = s.errors / s.shots
                # Approximate error bar (binomial standard error)
                error_bar = np.sqrt(p_logical * (1 - p_logical) / s.shots)
                # Only include points with reasonable error bars (< 10% of value)
                if error_bar < 0.1 * p_logical and p_logical > 0:
                    points.append((p, p_logical, error_bar))
        
        # Sort by physical error rate and take the two lowest
        points.sort(key=lambda x: x[0])
        
        if len(points) < min_points:
            continue
        
        # Use the two points with lowest physical error rate
        fit_points = points[:min_points]
        
        # Fit in log-log space: log(p_logical) = log(A) + slope * log(p)
        # where slope = (d+1)/2, so d = 2*slope - 1
        log_p = np.array([np.log(pt[0]) for pt in fit_points])
        log_p_logical = np.array([np.log(pt[1]) for pt in fit_points])
        
        # Linear fit in log-log space
        slope, intercept = np.polyfit(log_p, log_p_logical, 1)
        
        # Calculate effective distance: slope = (d+1)/2 => d = 2*slope - 1
        d_eff = 2 * slope - 1
        
        # Get plot styling
        plot_args = plot_args_func(idx, curve_id)
        color = plot_args.get('color', 'black')
        marker = plot_args.get('marker', 'o')
        linestyle = plot_args.get('linestyle', '-')
        
        # Generate fit line across fixed x range (use different start for k=4)
        p_min = 2.7e-4 if k_value == 4 else 8e-5
        p_range = np.logspace(np.log10(p_min), np.log10(1e-3), 100)
        p_logical_fit = np.exp(intercept) * p_range ** slope
        
        # Plot fit line as very faint dotted line
        ax.plot(p_range, p_logical_fit, color=color, linestyle=':', alpha=0.25, linewidth=1.5)
        
        # Store for inset plot
        if k_value is not None:
            fitted_distances.append((k_value, d_eff, color, marker, linestyle, curve_id))
    
    # Create inset plot for d vs k
    if fitted_distances:
        # Create inset axes in bottom right corner
        inset_ax = inset_axes(ax, width="30%", height="30%", loc='lower right', 
                              bbox_to_anchor=(0, 0.08, 1, 1), bbox_transform=ax.transAxes, borderpad=2)
        
        # Group by curve characteristics (excluding k) to connect points with same style
        from collections import defaultdict
        curve_groups = defaultdict(list)
        for k, d, color, marker, linestyle, curve_id in fitted_distances:
            # Extract non-k part of curve_id for grouping
            group_key = (color, marker, linestyle)
            curve_groups[group_key].append((k, d))
        
        # Plot each group with lines connecting points
        for (color, marker, linestyle), points in curve_groups.items():
            points.sort(key=lambda x: x[0])  # Sort by k
            ks = [p[0] for p in points]
            ds = [p[1] for p in points]
            inset_ax.plot(ks, ds, color=color, marker=marker, linestyle=linestyle, 
                         markersize=6, linewidth=1.5)
        
        # Style the inset
        inset_ax.set_xlabel('k', fontsize=22)
        inset_ax.set_ylabel('$d_{eff}$', fontsize=22)
        inset_ax.tick_params(axis='both', labelsize=22)
        inset_ax.set_xticks([1, 2, 3, 4])
        # Set y-ticks to integers only
        from matplotlib.ticker import MaxNLocator
        inset_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        inset_ax.grid(True, alpha=0.3)


def plot_single_basis(
    stats_list: list,
    basis: str,
    save_path: str,
    k_colors: dict,
):
    """Plot logical error rates for a single basis as a standalone figure.
    
    Args:
        stats_list: List of sinter.TaskStats objects
        basis: 'z' or 'x'
        save_path: Path to save the plot
        k_colors: Dict mapping k values to colors
    """
    # Filter stats for this basis
    basis_stats = [s for s in stats_list if s.json_metadata['basis'] == basis]
    
    if not basis_stats:
        return
    
    def plot_args_func(index, curve_id):
        k = int(curve_id.split('=')[1])
        return {
            'color': k_colors.get(k, 'black'),
            'marker': 'o',
            'linestyle': '-',
        }
    
    group_func = lambda s: f"k={s.json_metadata['k']}"
    x_func = lambda s: s.json_metadata['p']
    
    # Create standalone figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sinter.plot_error_rate(
        ax=ax,
        stats=basis_stats,
        x_func=x_func,
        group_func=group_func,
        plot_args_func=plot_args_func,
    )
    
    # Add fit lines with distance labels
    fit_and_plot_distance(ax, basis_stats, group_func, x_func, plot_args_func)
    
    ax.legend(fontsize=22, loc='upper left')
    ax.loglog()
    ax.set_xlim(left=7e-5)
    ax.set_xlabel("Physical Error Rate", fontsize=22)
    ax.set_ylabel("Logical Error Rate", fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.tick_params(axis='both', which='minor', labelsize=22)
    ax.grid(which='major', alpha=0.5)
    ax.grid(which='minor', alpha=0.2)
    # No title for standalone figures
    
    fig.set_dpi(150)
    fig.tight_layout()
    
    # Save plot
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved standalone {basis}-basis plot to {save_path}")
    
    plt.close()


def plot_results(results: list[BenchmarkResult], output_dir: str = 'benchmark_plots') -> None:
    """Generate plots from benchmark results using sinter.plot_error_rate.
    
    Creates two panels: one for z basis, one for x basis.
    Also saves each panel as a standalone PDF file.
    Groups by k value with consistent styling.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert BenchmarkResult objects to dictionaries for results_to_sinter_stats
    results_dicts = []
    for r in results:
        results_dicts.append({
            'k': r.k,
            'basis': r.basis,
            'distance': r.distance,
            'physical_error_rate': r.physical_error_rate,
            'logical_error_rate': r.logical_error_rate,
            'errors': r.errors,
            'shots': r.shots,
            'error_bar': r.error_bar,
        })
    
    # Use global DECODER variable (set in main())
    stats_list = results_to_sinter_stats(results_dicts, decoder=DECODER)
    
    if not stats_list:
        print("No data to plot")
        return
    
    # Define colors by k value
    k_colors = {1: 'C0', 2: 'C1', 3: 'C2', 4: 'C3', 5: 'C4'}
    
    # Generate standalone plots for each basis
    for basis in ['z', 'x']:
        standalone_path = os.path.join(output_dir, f'patch_rotation_benchmark_{basis}_basis.pdf')
        plot_single_basis(
            stats_list=stats_list,
            basis=basis,
            save_path=standalone_path,
            k_colors=k_colors,
        )
    
    # Create figure with two panels, sharing y-axis
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    
    def plot_args_func(index, curve_id):
        # curve_id format: "k=1" or "k=2"
        k = int(curve_id.split('=')[1])
        return {
            'color': k_colors.get(k, 'black'),
            'marker': 'o',
            'linestyle': '-',
        }
    
    # Plot for each basis
    for i, (ax, basis) in enumerate(zip(axes, ['z', 'x'])):
        # Filter stats for this basis
        basis_stats = [s for s in stats_list if s.json_metadata['basis'] == basis]
        
        if not basis_stats:
            continue
        
        group_func = lambda s: f"k={s.json_metadata['k']}"
        x_func = lambda s: s.json_metadata['p']
        
        sinter.plot_error_rate(
            ax=ax,
            stats=basis_stats,
            x_func=x_func,
            group_func=group_func,
            plot_args_func=plot_args_func,
        )
        
        # Add fit lines with distance labels
        fit_and_plot_distance(ax, basis_stats, group_func, x_func, plot_args_func)
        
        ax.legend(fontsize=22, loc='upper left')
        ax.loglog()
        ax.set_xlim(left=7e-5)
        ax.set_xlabel("Physical Error Rate", fontsize=22)
        if i == 0:  # Only set ylabel on left panel
            ax.set_ylabel("Logical Error Rate", fontsize=22)
        ax.tick_params(axis='both', which='major', labelsize=22)
        ax.tick_params(axis='both', which='minor', labelsize=22)
        ax.grid(which='major', alpha=0.5)
        ax.grid(which='minor', alpha=0.2)
        ax.set_title(f"Basis: {basis.upper()}", fontsize=20)
    
    fig.set_dpi(150)
    fig.tight_layout()
    
    # Save combined plot as PDF
    plot_path = os.path.join(output_dir, 'patch_rotation_benchmark.pdf')
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined plot to {plot_path}")
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

def save_crumble_urls_html(urls_dict: dict, output_dir: str = 'crumble_urls', experiment_name: str = 'patch_rotation') -> None:
    """Save Crumble URLs as HTML files with clickable links.
    
    Args:
        urls_dict: Dict with structure {config_str: {'before': url, 'after': url}}
        output_dir: Directory to save HTML files
        experiment_name: Name of the experiment for the HTML title
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create an index HTML file
    index_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{experiment_name.replace('_', ' ').title()} - Crumble URLs</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #555; margin-top: 30px; }}
        h3 {{ color: #777; }}
        .circuit {{ margin: 10px 0; padding: 10px; background: #f5f5f5; border-radius: 5px; }}
        a {{ color: #0066cc; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        .before {{ color: #cc6600; }}
        .after {{ color: #006600; }}
    </style>
</head>
<body>
    <h1>{experiment_name.replace('_', ' ').title()} - Crumble Circuit Visualizations</h1>
"""
    
    for config_str, urls in sorted(urls_dict.items()):
        index_html += f"    <div class='circuit'>\n"
        index_html += f"        <h3>{config_str}</h3>\n"
        
        if 'before' in urls and urls['before']:
            index_html += f"        <p class='before'>Before compactification: <a href='{urls['before']}' target='_blank'>Open in Crumble</a></p>\n"
        
        if 'after' in urls and urls['after']:
            index_html += f"        <p class='after'>After compactification: <a href='{urls['after']}' target='_blank'>Open in Crumble</a></p>\n"
        
        index_html += f"    </div>\n"
    
    index_html += """</body>
</html>
"""
    
    # Save the index file
    index_path = os.path.join(output_dir, f"{experiment_name}_crumble_urls.html")
    with open(index_path, 'w') as f:
        f.write(index_html)
    
    print(f"Saved Crumble URLs to {index_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Benchmark Patch Rotation circuits')
    parser.add_argument('--csv', type=str, default=None,
                       help='Output CSV file path (default: based on decoder)')
    parser.add_argument('--decoder', type=str, default='correlated_pymatching',
                       choices=['pymatching', 'correlated_pymatching'],
                       help='Decoder to use (default: correlated_pymatching)')
    parser.add_argument('--plot-only', action='store_true',
                       help='Only plot from existing CSV (skip benchmarking)')
    parser.add_argument('--distance-only', action='store_true',
                       help='Only compute distances (skip benchmarking)')
    parser.add_argument('--crumble-only', action='store_true',
                       help='Only generate Crumble URLs')
    args = parser.parse_args()
    
    # Set global decoder and output CSV
    global DECODER, OUTPUT_CSV
    DECODER = args.decoder
    if args.csv is None:
        OUTPUT_CSV = f'benchmark_data/patch_rotation_benchmark_{DECODER}.csv'
    else:
        OUTPUT_CSV = args.csv
    
    print("=" * 70)
    print("Patch Rotation Benchmark")
    print("=" * 70)
    
    # Crumble-only mode
    if args.crumble_only:
        print("\nGenerating Crumble URLs (before and after compactification)...")
        urls = {}
        for basis in BASIS_VALUES:
            for k in K_VALUES:
                config = CircuitConfig(k, basis)
                config_str = f"k={k}, basis={basis}"
                try:
                    circuit_before, circuit_after = generate_circuit_for_config(config, return_both=True)
                    
                    url_before = circuit_before.to_crumble_url()
                    url_after = circuit_after.to_crumble_url()
                    
                    urls[config_str] = {
                        'before': url_before,
                        'after': url_after
                    }
                    print(f"  Generated URLs for {config_str}")
                except Exception as e:
                    print(f"  Error generating URLs for {config_str}: {e}")
        
        if urls:
            save_crumble_urls_html(urls, output_dir='crumble_urls', experiment_name='patch_rotation')
        
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
    print(f"  Decoder: {DECODER}")
    print(f"  Max shots: {MAX_SHOTS:,}")
    print(f"  Max errors: {MAX_ERRORS:,}")
    print(f"  Num workers: {NUM_WORKERS}")
    print(f"  Output file: {args.csv}")
    print()
    
    start_time = time.time()
    
    # Generate Crumble URLs (before and after compactification)
    print("=" * 70)
    print("Generating Crumble URLs (before and after compactification)")
    print("=" * 70)
    
    all_crumble_urls = {}
    for basis in BASIS_VALUES:
        for k in K_VALUES:
            config = CircuitConfig(k, basis)
            config_str = f"k={k}, basis={basis}"
            try:
                circuit_before, circuit_after = generate_circuit_for_config(config, return_both=True)
                
                url_before = circuit_before.to_crumble_url()
                url_after = circuit_after.to_crumble_url()
                
                all_crumble_urls[config_str] = {
                    'before': url_before,
                    'after': url_after
                }
                print(f"  Generated URLs for {config_str}")
            except Exception as e:
                print(f"  Error generating URLs for {config_str}: {e}")
    
    # Save Crumble URLs to HTML
    if all_crumble_urls:
        save_crumble_urls_html(all_crumble_urls, output_dir='crumble_urls', experiment_name='patch_rotation')
    
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
