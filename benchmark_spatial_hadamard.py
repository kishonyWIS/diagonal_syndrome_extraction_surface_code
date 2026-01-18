#!/usr/bin/env python3
"""
Comprehensive benchmarking script for Spatial Hadamard circuits.

Sweeps through circuit configurations and decoder types, computing:
- Graph-like distance for each circuit variant
- Logical error rates for each configuration with multiple decoders

Uses sinter for parallelized sampling and decoding.

Parameters swept:
- Direction: 'x', 'y'
- Flag config: 'all', 'partial', 'none'
- k: 1, 2, 3
- Physical error rate: np.logspace(-3.5, -2, 7)
- Decoder: plain pymatching, correlated pymatching, tesseract
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
import pymatching
import stim
import sinter
import tesseract_decoder.tesseract as tesseract

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

from spatial_hadamard_manual_construction import (
    generate_spatial_hadamard_circuit,
    calculate_graphlike_distance,
)
from tqec import NoiseModel


# =============================================================================
# Configuration
# =============================================================================

# Parameter ranges
DIRECTIONS = ['y']
# Flag configurations:
#   'all': flags measured every round (measure_shared_data_final_only=False)
#   'partial': flags measured only in final round (measure_shared_data_final_only=True)
#   'none': no flags at all (measure_coupling_aux_mz=False, measure_shared_data=False)
FLAG_CONFIGS = ['none', 'partial', 'all']
K_VALUES = [4]
PHYSICAL_ERROR_RATES = np.logspace(-4, -2, 9)[2:][::-1][-2:]  # ~[0.000316, 0.001, 0.00316, 0.01]
DECODERS = ['correlated_pymatching']#, 'tesseract']

# Sampling configuration
MAX_SHOTS = 500_000_000
MAX_ERRORS = 300
NUM_WORKERS = 10
RANDOM_SEED = 42

# Output file
OUTPUT_CSV = 'benchmark_data/spatial_hadamard_benchmark_k4_y_only_correlated_pymatching_only.csv'


# =============================================================================
# Custom Sinter Decoders
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
        
        # Create matcher with correlations enabled (BOTH here AND in decode_batch)
        matcher = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
        
        # Load detector data
        dets = stim.read_shot_data_file(
            path=dets_b8_in_path,
            format='b8',
            num_detectors=num_dets,
            num_observables=0,
        )
        
        # Decode with correlations enabled (BOTH here AND in from_detector_error_model)
        predictions = matcher.decode_batch(dets, enable_correlations=True)
        
        # Write predictions
        stim.write_shot_data_file(
            data=predictions,
            path=obs_predictions_b8_out_path,
            format='b8',
            num_observables=num_obs,
        )


class TesseractDecoder(sinter.Decoder):
    """Sinter decoder wrapper for Tesseract hypergraph decoder."""
    
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
        
        # Load DEM (full, not decomposed)
        dem = stim.DetectorErrorModel.from_file(dem_path)
        
        # Create tesseract decoder
        tesseract_config = tesseract.TesseractConfig(
            dem=dem,
            pqlimit=200_000,
            det_beam=15,
            beam_climbing=True,
            det_orders=[],
            no_revisit_dets=True,
        )
        decoder = tesseract.TesseractDecoder(tesseract_config)
        
        # Load detector data
        dets = stim.read_shot_data_file(
            path=dets_b8_in_path,
            format='b8',
            num_detectors=num_dets,
            num_observables=0,
        )
        
        # Decode
        predictions = decoder.decode_batch(dets)
        
        # Write predictions
        stim.write_shot_data_file(
            data=predictions,
            path=obs_predictions_b8_out_path,
            format='b8',
            num_observables=num_obs,
        )


# Custom decoder registry for sinter
CUSTOM_DECODERS = {
    'correlated_pymatching': CorrelatedPymatchingDecoder(),
    'tesseract': TesseractDecoder(),
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CircuitConfig:
    """Configuration for a single circuit variant."""
    direction: str
    flag_config: str  # 'all', 'partial', or 'none'
    k: int
    
    def __str__(self):
        return f"dir={self.direction}, flags={self.flag_config}, k={self.k}"


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    # Circuit configuration
    direction: str
    flag_config: str  # 'all', 'partial', or 'none'
    k: int
    distance: Optional[int]
    
    # Error rate configuration
    physical_error_rate: float
    decoder: str
    
    # Results
    logical_error_rate: float
    errors: int
    shots: int
    error_bar: float
    decode_time: float = 0.0


# =============================================================================
# Circuit Generation and Distance Computation
# =============================================================================

def generate_circuit_for_config(
    config: CircuitConfig,
    return_both: bool = False,
) -> stim.Circuit:
    """Generate a spatial Hadamard circuit for the given configuration (without noise, compacted).
    
    Args:
        config: Circuit configuration
        return_both: If True, return tuple (before_compact, after_compact)
    """
    from compact_circuit import compact_and_delay_init
    
    # Generate circuit WITHOUT noise
    circuit_before = generate_spatial_hadamard_circuit(
        k=config.k,
        axis=config.direction,
        noise_model=None,
        flag_config=config.flag_config,
    )
    
    # Compact the circuit (ASAP + ALAP scheduling)
    circuit_after = compact_and_delay_init(circuit_before)
    
    if return_both:
        return circuit_before, circuit_after
    return circuit_after


def compute_distances_for_all_configs() -> dict[tuple, int]:
    """
    Compute graph-like distance for all circuit configurations.
    
    Returns:
        Dictionary mapping (direction, flag_config, k) to distance
    """
    print("=" * 70)
    print("Computing Graph-like Distances")
    print("=" * 70)
    
    distances = {}
    
    # Use a small noise model for distance calculation
    # (distance calculation requires errors in the circuit)
    distance_noise_model = NoiseModel.uniform_depolarizing(0.001)
    
    for direction in DIRECTIONS:
        for flag_config in FLAG_CONFIGS:
            for k in K_VALUES:
                config = CircuitConfig(direction, flag_config, k)
                print(f"\nGenerating circuit: {config}")
                
                # Generate compacted circuit (without noise)
                circuit = generate_circuit_for_config(config)
                
                # Add noise for distance calculation
                noisy_circuit = distance_noise_model.noisy_circuit(circuit)
                
                print(f"  Qubits: {noisy_circuit.num_qubits}, Detectors: {noisy_circuit.num_detectors}")
                
                # Calculate distance
                distance = calculate_graphlike_distance(noisy_circuit)
                distances[(direction, flag_config, k)] = distance
                
                print(f"  Graph-like distance: {distance}")
    
    print("\n" + "=" * 70)
    print("Distance Summary")
    print("=" * 70)
    print(f"{'Direction':<10} {'Flags':<10} {'k':<5} {'Distance':<10}")
    print("-" * 40)
    for (direction, flag_config, k), distance in sorted(distances.items()):
        print(f"{direction:<10} {flag_config:<10} {k:<5} {distance}")
    
    return distances


# =============================================================================
# Sinter-based Benchmarking
# =============================================================================

def run_sinter_for_single_task(
    circuit: stim.Circuit,
    metadata: dict,
    decoder_name: str,
    max_shots: int,
    max_errors: int,
    num_workers: int,
) -> BenchmarkResult:
    """
    Run sinter benchmark for a single circuit + decoder combination.
    
    Args:
        circuit: The stim circuit to benchmark
        metadata: Dictionary with configuration metadata
        decoder_name: Name of the decoder to use
        max_shots: Maximum shots per task
        max_errors: Maximum errors for early stopping
        num_workers: Number of parallel workers
        
    Returns:
        BenchmarkResult object
    """
    task = sinter.Task(
        circuit=circuit,
        decoder=decoder_name,
        json_metadata={**metadata, 'decoder': decoder_name},
    )
    
    # Run sinter for this decoder
    start_time = time.time()
    stats = sinter.collect(
        tasks=[task],
        max_shots=max_shots,
        max_errors=max_errors,
        num_workers=num_workers,
        custom_decoders=CUSTOM_DECODERS,
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
            direction=metadata['direction'],
            flag_config=metadata['flag_config'],
            k=metadata['k'],
            distance=metadata.get('distance'),
            physical_error_rate=metadata['physical_error_rate'],
            decoder=decoder_name,
            logical_error_rate=error_rate,
            errors=stat.errors,
            shots=stat.shots,
            error_bar=error_bar,
            decode_time=decode_time,
        )
        
        print(f"        {stat.errors}/{stat.shots} errors, rate={error_rate:.6f}, time={decode_time:.1f}s")
        return result
    
    return None


def append_results_to_csv(results: list[BenchmarkResult], filepath: str, write_header: bool = False) -> None:
    """Append benchmark results to CSV file."""
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    fieldnames = [
        'direction',
        'flag_config',
        'k',
        'distance',
        'physical_error_rate',
        'decoder',
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
                'direction': result.direction,
                'flag_config': result.flag_config,
                'k': result.k,
                'distance': result.distance,
                'physical_error_rate': result.physical_error_rate,
                'decoder': result.decoder,
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
    1. decoder
    2. k
    3. physical_error_rate
    4. flag_config
    5. direction
    
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
    total_tasks = (
        len(DECODERS) *
        len(K_VALUES) * 
        len(PHYSICAL_ERROR_RATES) *
        len(FLAG_CONFIGS) * 
        len(DIRECTIONS)
    )
    
    print("\n" + "=" * 70)
    print("Running Logical Error Rate Benchmarks with Sinter")
    print("=" * 70)
    print(f"Total tasks: {total_tasks}")
    print(f"Loop order: decoder -> k -> noise -> flag_config -> direction")
    print(f"Flag configs: {FLAG_CONFIGS}")
    print(f"Max shots per config: {MAX_SHOTS:,}")
    print(f"Max errors for early stopping: {MAX_ERRORS:,}")
    print(f"Number of workers: {NUM_WORKERS}")
    print(f"Output file: {OUTPUT_CSV} (incremental save)")
    print()
    
    # Initialize CSV with header
    append_results_to_csv([], OUTPUT_CSV, write_header=True)
    
    current_task = 0
    
    # Loop order: decoder -> k -> noise -> flag_config -> direction
    for decoder_name in DECODERS:
        print(f"\n{'='*70}")
        print(f"DECODER: {decoder_name}")
        print(f"{'='*70}")
        
        for k in K_VALUES:
            print(f"\n  k={k}")
            
            for noise_level in PHYSICAL_ERROR_RATES:
                print(f"\n    p={noise_level:.6f}")
                
                for flag_config in FLAG_CONFIGS:
                    for direction in DIRECTIONS:
                        current_task += 1
                        config = CircuitConfig(direction, flag_config, k)
                        distance = distances.get((direction, flag_config, k))
                        
                        print(f"      [{current_task}/{total_tasks}] {config}")
                        
                        # Generate compacted circuit (without noise)
                        circuit = generate_circuit_for_config(config)
                        
                        # Add noise using NoiseModel.noisy_circuit()
                        noise_model = NoiseModel.uniform_depolarizing(noise_level)
                        noisy_circuit = noise_model.noisy_circuit(circuit)
                        
                        metadata = {
                            'direction': direction,
                            'flag_config': flag_config,
                            'k': k,
                            'distance': distance,
                            'physical_error_rate': noise_level,
                        }
                        
                        # Run this decoder for this configuration
                        try:
                            result = run_sinter_for_single_task(
                                circuit=noisy_circuit,
                                metadata=metadata,
                                decoder_name=decoder_name,
                                max_shots=MAX_SHOTS,
                                max_errors=MAX_ERRORS,
                                num_workers=NUM_WORKERS,
                            )
                            
                            if result:
                                # Save result immediately
                                append_results_to_csv([result], OUTPUT_CSV, write_header=False)
                                all_results.append(result)
                            
                        except Exception as e:
                            print(f"        ERROR: {e}")
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
        'direction',
        'flag_config',
        'k',
        'distance',
        'physical_error_rate',
        'decoder',
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
                'direction': result.direction,
                'flag_config': result.flag_config,
                'k': result.k,
                'distance': result.distance,
                'physical_error_rate': result.physical_error_rate,
                'decoder': result.decoder,
                'logical_error_rate': result.logical_error_rate,
                'errors': result.errors,
                'shots': result.shots,
                'error_bar': result.error_bar,
                'decode_time': result.decode_time,
            })
    
    print(f"Saved {len(results)} results to {filepath}")


def save_crumble_urls_html(urls_dict, output_dir="crumble_urls", experiment_name="spatial_hadamard"):
    """Save Crumble URLs as HTML files with clickable links.
    
    Args:
        urls_dict: Dict with structure {config_str: {'before': url, 'after': url}}
        output_dir: Directory to save HTML files
        experiment_name: Name of the experiment for the HTML title
    """
    import os
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
# Plotting Functions
# =============================================================================

def load_results_from_csv(filepath: str = "benchmark_data/spatial_hadamard_benchmark.csv") -> list[dict]:
    """Load benchmark results from CSV file.
    
    Returns:
        List of dictionaries with result data
    """
    results = []
    
    with open(filepath, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            results.append({
                'direction': row['direction'],
                'flag_config': row['flag_config'],
                'k': int(row['k']),
                'distance': int(row['distance']) if row['distance'] else None,
                'physical_error_rate': float(row['physical_error_rate']),
                'decoder': row['decoder'],
                'logical_error_rate': float(row['logical_error_rate']),
                'errors': int(row['errors']),
                'shots': int(row['shots']),
                'error_bar': float(row['error_bar']),
                'decode_time': float(row['decode_time']),
            })
    
    print(f"Loaded {len(results)} results from {filepath}")
    return results


def results_to_sinter_stats(results: list[dict], decoder_filter: str = None) -> list[sinter.TaskStats]:
    """Convert results to sinter.TaskStats for plotting.
    
    Args:
        results: List of result dictionaries
        decoder_filter: If specified, only include results for this decoder
    
    Returns:
        List of sinter.TaskStats objects
    """
    stats_list = []
    
    for result in results:
        if decoder_filter and result['decoder'] != decoder_filter:
            continue
        
        # Create a descriptive label
        # Format: "direction flag_config k=N"
        stats = sinter.TaskStats(
            strong_id=f"{result['direction']}_{result['flag_config']}_k{result['k']}_p{result['physical_error_rate']}_{result['decoder']}",
            decoder=result['decoder'],
            json_metadata={
                'p': result['physical_error_rate'],
                'k': result['k'],
                'd': 2 * result['k'] + 1,
                'direction': result['direction'],
                'flag_config': result['flag_config'],
                'decoder': result['decoder'],
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
        
        # Group by linestyle to draw black connecting lines
        from collections import defaultdict
        linestyle_groups = defaultdict(list)
        for k, d, color, marker, linestyle, curve_id in fitted_distances:
            linestyle_groups[linestyle].append((k, d, color, marker))
        
        # First draw black connecting lines by linestyle
        for linestyle, points in linestyle_groups.items():
            points.sort(key=lambda x: x[0])  # Sort by k
            ks = [p[0] for p in points]
            ds = [p[1] for p in points]
            inset_ax.plot(ks, ds, color='black', linestyle=linestyle, linewidth=1.5, zorder=1)
        
        # Then plot markers on top with their colors
        for k, d, color, marker, linestyle, curve_id in fitted_distances:
            inset_ax.plot(k, d, color=color, marker=marker, linestyle='none', 
                         markersize=6, zorder=2)
        
        # Style the inset with same fonts as main panel
        inset_ax.set_xlabel('k', fontsize=22)
        inset_ax.set_ylabel('$d_{eff}$', fontsize=22)
        inset_ax.tick_params(axis='both', labelsize=22)
        inset_ax.set_xticks([1, 2, 3, 4])
        # Set y-ticks to integers only
        from matplotlib.ticker import MaxNLocator
        inset_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        inset_ax.grid(True, alpha=0.3)


def plot_single_direction(
    stats_list: list,
    direction: str,
    decoder: str,
    save_path: str,
    k_colors: dict,
    flag_markers: dict,
    flag_styles: dict,
    flag_order: dict,
):
    """Plot logical error rates for a single direction as a standalone figure.
    
    Args:
        stats_list: List of sinter.TaskStats objects
        direction: 'x' or 'y'
        decoder: Decoder name (for title)
        save_path: Path to save the plot
        k_colors: Dict mapping k values to colors
        flag_markers: Dict mapping flag_config to markers
        flag_styles: Dict mapping flag_config to linestyles
        flag_order: Dict mapping flag_config to sort order
    """
    # Filter stats for this direction
    direction_stats = [s for s in stats_list if s.json_metadata['direction'] == direction]
    
    if not direction_stats:
        return
    
    def plot_args_func(index, curve_id):
        parts = curve_id.split()
        flag_config = parts[0].split('_')[1] if '_' in parts[0] else parts[0]
        k = int(parts[1].split('=')[1])
        return {
            'color': k_colors.get(k, 'black'),
            'marker': flag_markers.get(flag_config, 'o'),
            'linestyle': flag_styles.get(flag_config, '-'),
        }
    
    # Create standalone figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    group_func = lambda s: f"{flag_order[s.json_metadata['flag_config']]}_{s.json_metadata['flag_config']} k={s.json_metadata['k']}"
    x_func = lambda s: s.json_metadata['p']
    
    sinter.plot_error_rate(
        ax=ax,
        stats=direction_stats,
        x_func=x_func,
        group_func=group_func,
        plot_args_func=plot_args_func,
    )
    
    # Add fit lines with distance labels
    fit_and_plot_distance(ax, direction_stats, group_func, x_func, plot_args_func)
    
    # Fix legend labels by removing the sort prefix
    handles, labels = ax.get_legend_handles_labels()
    new_labels = [label.split('_', 1)[1] if '_' in label else label for label in labels]
    ax.legend(handles, new_labels, fontsize=18, ncol=3, loc='upper left', columnspacing=0.5, handletextpad=0.3)
    
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
    print(f"Saved standalone {direction}-direction plot to {save_path}")
    
    plt.close()


def plot_error_rates(
    results: list[dict],
    decoder: str = 'pymatching',
    save_path: str = "benchmark_plots/spatial_hadamard_error_rates.pdf"
):
    """Plot logical error rates using sinter.plot_error_rate.
    
    Creates two panels: one for x direction, one for y direction.
    Also saves each panel as a standalone PDF file.
    Groups by (flag_config, k) with consistent styling.
    
    Args:
        results: List of result dictionaries
        decoder: Which decoder's results to plot
        save_path: Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Filter results for this decoder
    stats_list = results_to_sinter_stats(results, decoder_filter=decoder)
    
    if not stats_list:
        print(f"No data to plot for decoder: {decoder}")
        return
    
    # Define colors by k value
    k_colors = {1: 'C0', 2: 'C1', 3: 'C2', 4: 'C3', 5: 'C4'}
    
    # Define markers by flag_config
    flag_markers = {'all': 'o', 'partial': 's', 'none': '^'}
    
    # Define linestyles by flag_config (since direction is now in separate panels)
    flag_styles = {'all': '-', 'partial': '--', 'none': ':'}
    
    # Define the desired order for flag_config (used for sorting legend)
    flag_order = {'none': '0', 'partial': '1', 'all': '2'}
    
    # Generate standalone plots for each direction
    base_path = save_path.rsplit('.', 1)[0]  # Remove extension
    for direction in ['x', 'y']:
        standalone_path = f"{base_path}_{direction}_direction.pdf"
        plot_single_direction(
            stats_list=stats_list,
            direction=direction,
            decoder=decoder,
            save_path=standalone_path,
            k_colors=k_colors,
            flag_markers=flag_markers,
            flag_styles=flag_styles,
            flag_order=flag_order,
        )
    
    # Create figure with two panels, sharing y-axis
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
    
    def plot_args_func(index, curve_id):
        # curve_id format: "0_none k=1" or "1_partial k=2" (with sort prefix)
        parts = curve_id.split()
        # Remove the sort prefix (e.g., "0_none" -> "none")
        flag_config = parts[0].split('_')[1] if '_' in parts[0] else parts[0]
        k = int(parts[1].split('=')[1])
        
        return {
            'color': k_colors.get(k, 'black'),
            'marker': flag_markers.get(flag_config, 'o'),
            'linestyle': flag_styles.get(flag_config, '-'),
        }
    
    # Plot for each direction
    for i, (ax, direction) in enumerate(zip(axes, ['x', 'y'])):
        # Filter stats for this direction
        direction_stats = [s for s in stats_list if s.json_metadata['direction'] == direction]
        
        if not direction_stats:
            continue
        
        # Use a sortable prefix in group_func to control legend order
        # Format: "0_none k=1" will sort before "1_partial k=1" which sorts before "2_all k=1"
        group_func = lambda s: f"{flag_order[s.json_metadata['flag_config']]}_{s.json_metadata['flag_config']} k={s.json_metadata['k']}"
        x_func = lambda s: s.json_metadata['p']
        
        sinter.plot_error_rate(
            ax=ax,
            stats=direction_stats,
            x_func=x_func,
            group_func=group_func,
            plot_args_func=plot_args_func,
        )
        
        # Add fit lines with distance labels
        fit_and_plot_distance(ax, direction_stats, group_func, x_func, plot_args_func)
        
        # Fix legend labels by removing the sort prefix
        handles, labels = ax.get_legend_handles_labels()
        new_labels = [label.split('_', 1)[1] if '_' in label else label for label in labels]
        ax.legend(handles, new_labels, fontsize=18, ncol=3, loc='upper left', columnspacing=0.5, handletextpad=0.3)
        
        ax.loglog()
        ax.set_xlim(left=7e-5)
        ax.set_xlabel("Physical Error Rate", fontsize=22)
        if i == 0:  # Only set ylabel on left panel
            ax.set_ylabel("Logical Error Rate", fontsize=22)
        ax.tick_params(axis='both', which='major', labelsize=22)
        ax.tick_params(axis='both', which='minor', labelsize=22)
        ax.grid(which='major', alpha=0.5)
        ax.grid(which='minor', alpha=0.2)
        ax.set_title(f"{direction} direction", fontsize=20)
    
    fig.set_dpi(150)
    fig.tight_layout()
    
    # Save plot
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined error rate plot to {save_path}")
    
    plt.close()
    
    return fig


def plot_all_decoders(
    results: list[dict],
    output_dir: str = "benchmark_plots"
):
    """Plot error rates for all decoders.
    
    Args:
        results: List of result dictionaries
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique decoders
    decoders = set(r['decoder'] for r in results)
    
    for decoder in sorted(decoders):
        save_path = os.path.join(output_dir, f"spatial_hadamard_{decoder}_error_rates.pdf")
        plot_error_rates(results, decoder=decoder, save_path=save_path)
    
    print(f"\nGenerated plots for {len(decoders)} decoders in {output_dir}")


def print_summary(results: list[BenchmarkResult]) -> None:
    """Print a summary of the benchmark results."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    
    # Group by decoder and show average error rates
    for decoder in DECODERS:
        decoder_results = [r for r in results if r.decoder == decoder]
        if decoder_results:
            avg_rate = np.mean([r.logical_error_rate for r in decoder_results])
            total_errors = sum(r.errors for r in decoder_results)
            total_shots = sum(r.shots for r in decoder_results)
            total_time = sum(r.decode_time for r in decoder_results)
            print(f"\n{decoder}:")
            print(f"  Total errors/shots: {total_errors:,}/{total_shots:,}")
            print(f"  Average error rate: {avg_rate:.6f}")
            print(f"  Total decode time: {total_time:.2f}s")
    
    # Show results by k value
    print("\n" + "-" * 70)
    print("Error rates by k value:")
    for k in K_VALUES:
        print(f"\n  k={k}:")
        for decoder in DECODERS:
            k_results = [r for r in results if r.k == k and r.decoder == decoder]
            if k_results:
                avg_rate = np.mean([r.logical_error_rate for r in k_results])
                print(f"    {decoder}: {avg_rate:.6f}")


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark Spatial Hadamard circuits with various decoders.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full benchmark
  python benchmark_spatial_hadamard.py
  
  # Plot only from existing CSV (no benchmark)
  python benchmark_spatial_hadamard.py --plot-only
  
  # Plot from a specific CSV file
  python benchmark_spatial_hadamard.py --plot-only --csv benchmark_data/my_results.csv
"""
    )
    parser.add_argument(
        '--plot-only',
        action='store_true',
        help='Only generate plots from existing CSV data (skip benchmark)'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default=OUTPUT_CSV,
        help=f'Path to CSV file for loading/saving results (default: {OUTPUT_CSV})'
    )
    return parser.parse_args()


def main():
    """Main entry point for the benchmark script."""
    args = parse_args()
    
    print("=" * 70)
    print("Spatial Hadamard Circuit Benchmark (Sinter-based)")
    print("=" * 70)
    print()
    
    if args.plot_only:
        # Plot-only mode: load from CSV and generate plots
        print(f"PLOT-ONLY MODE: Loading results from {args.csv}")
        print("=" * 70)
        
        if not os.path.exists(args.csv):
            print(f"Error: CSV file not found: {args.csv}")
            sys.exit(1)
        
        # Load results from CSV (as dictionaries for plot functions)
        results_dicts = load_results_from_csv(args.csv)
        
        # Generate plots (plot functions expect dictionaries)
        print("\nGenerating plots...")
        plot_all_decoders(results_dicts, output_dir="benchmark_plots")
        
        # Convert to BenchmarkResult objects for print_summary
        results = []
        for r in results_dicts:
            results.append(BenchmarkResult(
                direction=r['direction'],
                flag_config=r['flag_config'],
                k=r['k'],
                distance=r['distance'],
                physical_error_rate=r['physical_error_rate'],
                decoder=r['decoder'],
                logical_error_rate=r['logical_error_rate'],
                errors=r['errors'],
                shots=r['shots'],
                error_bar=r['error_bar'],
                decode_time=r['decode_time'],
            ))
        
        # Print summary
        print_summary(results)
        
        print("\nPlot-only mode complete!")
        return
    
    # Full benchmark mode
    print("Configuration:")
    print(f"  Directions: {DIRECTIONS}")
    print(f"  Flag configs: {FLAG_CONFIGS}")
    print(f"  k values: {K_VALUES}")
    print(f"  Physical error rates: {[f'{p:.6f}' for p in PHYSICAL_ERROR_RATES]}")
    print(f"  Decoders: {DECODERS}")
    print(f"  Max shots: {MAX_SHOTS:,}")
    print(f"  Max errors: {MAX_ERRORS:,}")
    print(f"  Num workers: {NUM_WORKERS}")
    print(f"  Output file: {args.csv}")
    print()
    
    start_time = time.time()
    
    # Generate Crumble URLs for all configurations
    print("=" * 70)
    print("Generating Crumble URLs for all configurations")
    print("=" * 70)
    
    all_crumble_urls = {}
    for direction in DIRECTIONS:
        for flag_config in FLAG_CONFIGS:
            for k in K_VALUES:
                config = CircuitConfig(direction, flag_config, k)
                config_str = f"direction={direction}, flags={flag_config}, k={k}"
                
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
    
    # # Save Crumble URLs to HTML
    # if all_crumble_urls:
    #     save_crumble_urls_html(all_crumble_urls, output_dir="crumble_urls", experiment_name="spatial_hadamard")
    
    # Run benchmark with distance calculation enabled
    # Results are saved incrementally to CSV after each configuration
    results = run_benchmark(skip_distance=False)
    
    # Print summary
    print_summary(results)
    
    print(f"\nResults saved incrementally to: {args.csv}")
    
    total_time = time.time() - start_time
    print(f"\nTotal benchmark time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"\nResults saved to: {args.csv}")


if __name__ == "__main__":
    main()
