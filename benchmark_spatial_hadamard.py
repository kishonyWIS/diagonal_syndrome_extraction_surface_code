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
- Physical error rate: np.logspace(-3.5, -2, 4)
- Decoder: plain pymatching, correlated pymatching, tesseract
"""

import csv
import sys
import time
from dataclasses import dataclass
from typing import Optional
import numpy as np
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
DIRECTIONS = ['x', 'y']
# Flag configurations:
#   'all': flags measured every round (measure_shared_data_final_only=False)
#   'partial': flags measured only in final round (measure_shared_data_final_only=True)
#   'none': no flags at all (measure_coupling_aux_mz=False, measure_shared_data=False)
FLAG_CONFIGS = ['all', 'partial', 'none']
K_VALUES = [1, 2, 3]
PHYSICAL_ERROR_RATES = np.logspace(-3.5, -2, 4)  # ~[0.000316, 0.001, 0.00316, 0.01]
DECODERS = ['pymatching', 'correlated_pymatching', 'tesseract']

# Sampling configuration
MAX_SHOTS = 100_000_000
MAX_ERRORS = 3000
NUM_WORKERS = 10
RANDOM_SEED = 42

# Output file
OUTPUT_CSV = 'spatial_hadamard_benchmark.csv'


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
    noise_model: Optional[NoiseModel] = None,
) -> stim.Circuit:
    """Generate a spatial Hadamard circuit for the given configuration."""
    return generate_spatial_hadamard_circuit(
        k=config.k,
        axis=config.direction,
        noise_model=noise_model,
        flag_config=config.flag_config,
    )


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
                
                # Generate circuit with noise for distance calculation
                circuit = generate_circuit_for_config(config, noise_model=distance_noise_model)
                
                print(f"  Qubits: {circuit.num_qubits}, Detectors: {circuit.num_detectors}")
                
                # Calculate distance
                distance = calculate_graphlike_distance(circuit)
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
                        
                        # Generate circuit
                        noise_model = NoiseModel.uniform_depolarizing(noise_level)
                        circuit = generate_circuit_for_config(config, noise_model)
                        
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
                                circuit=circuit,
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

def main():
    """Main entry point for the benchmark script."""
    print("=" * 70)
    print("Spatial Hadamard Circuit Benchmark (Sinter-based)")
    print("=" * 70)
    print()
    print("Configuration:")
    print(f"  Directions: {DIRECTIONS}")
    print(f"  Flag configs: {FLAG_CONFIGS}")
    print(f"  k values: {K_VALUES}")
    print(f"  Physical error rates: {[f'{p:.6f}' for p in PHYSICAL_ERROR_RATES]}")
    print(f"  Decoders: {DECODERS}")
    print(f"  Max shots: {MAX_SHOTS:,}")
    print(f"  Max errors: {MAX_ERRORS:,}")
    print(f"  Num workers: {NUM_WORKERS}")
    print(f"  Output file: {OUTPUT_CSV}")
    print()
    
    start_time = time.time()
    
    # Run benchmark (skip distance calculation - already verified)
    # Results are saved incrementally to CSV after each configuration
    results = run_benchmark(skip_distance=True)
    
    # Print summary
    print_summary(results)
    
    print(f"\nResults saved incrementally to: {OUTPUT_CSV}")
    
    total_time = time.time() - start_time
    print(f"\nTotal benchmark time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"\nResults saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
