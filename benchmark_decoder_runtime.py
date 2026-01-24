#!/usr/bin/env python3
"""
Dedicated decoder runtime benchmarking script.

Runs fixed number of shots for each (decoder, error rate, k) combination
to get accurate runtime measurements for comparison.

Uses:
- Full noise model (not interface-only)
- Only "partial" flag configuration
- Fixed number of shots per combination (no early stopping)
"""

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import stim
import sinter
import pymatching

# Optional tesseract import
try:
    import tesseract_decoder.tesseract as tesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: tesseract_decoder not available. Tesseract decoder will be skipped.")

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

from spatial_hadamard_manual_construction import generate_spatial_hadamard_circuit
from compact_circuit import compact_and_delay_init
from tqec import NoiseModel


# =============================================================================
# Configuration
# =============================================================================

# Parameters
DIRECTION = 'y'
FLAG_CONFIG = 'partial'
K_VALUES = [1, 2, 3, 4]
PHYSICAL_ERROR_RATES = np.logspace(-4, -2, 9)  # 1e-4 to 1e-2

# Decoders - tesseract only if available
DECODERS = ['pymatching', 'correlated_pymatching']
if TESSERACT_AVAILABLE:
    DECODERS.append('tesseract')

# Shot counts - variable based on decoder and error rate
# Tesseract is slow, so we scale shots inversely with error rate
# pymatching/correlated_pymatching get 10x more shots since they're faster
BASE_SHOTS_LOW_P = 5_000   # Tesseract shots at lowest error rate (p=1e-4)
BASE_SHOTS_HIGH_P = 100     # Tesseract shots at highest error rate (p=1e-2)
MATCHING_MULTIPLIER = 1000    # Multiplier for pymatching/correlated_pymatching
SHOT_SCALE = 1.0            # Global multiplier for all shots (set via --shot-scale)
NUM_WORKERS = 10

# Output file
OUTPUT_CSV = 'benchmark_data/decoder_runtime_benchmark.csv'


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
        dem = stim.DetectorErrorModel.from_file(dem_path)
        matcher = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
        
        dets = stim.read_shot_data_file(
            path=dets_b8_in_path,
            format='b8',
            num_detectors=num_dets,
            num_observables=0,
        )
        
        predictions = matcher.decode_batch(dets, enable_correlations=True)
        
        stim.write_shot_data_file(
            data=predictions,
            path=obs_predictions_b8_out_path,
            format='b8',
            num_observables=num_obs,
        )


if TESSERACT_AVAILABLE:
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
            dem = stim.DetectorErrorModel.from_file(dem_path)
            
            tesseract_config = tesseract.TesseractConfig(
                dem=dem,
                pqlimit=200_000,
                det_beam=15,
                beam_climbing=True,
                det_orders=[],
                no_revisit_dets=True,
            )
            decoder = tesseract.TesseractDecoder(tesseract_config)
            
            dets = stim.read_shot_data_file(
                path=dets_b8_in_path,
                format='b8',
                num_detectors=num_dets,
                num_observables=0,
            )
            
            predictions = decoder.decode_batch(dets)
            
            stim.write_shot_data_file(
                data=predictions,
                path=obs_predictions_b8_out_path,
                format='b8',
                num_observables=num_obs,
            )


CUSTOM_DECODERS = {
    'correlated_pymatching': CorrelatedPymatchingDecoder(),
}

if TESSERACT_AVAILABLE:
    CUSTOM_DECODERS['tesseract'] = TesseractDecoder()


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RuntimeResult:
    """Result of a runtime benchmark run."""
    k: int
    physical_error_rate: float
    decoder: str
    shots: int
    decode_time: float  # Total decode time in seconds
    errors: int
    logical_error_rate: float


# =============================================================================
# Shot Count Calculation
# =============================================================================

def get_num_shots(decoder: str, physical_error_rate: float) -> int:
    """
    Calculate number of shots based on decoder type and error rate.
    
    Tesseract is slow, so shots scale inversely with error rate (log scale).
    pymatching and correlated_pymatching get 10x more shots.
    
    Args:
        decoder: Decoder name
        physical_error_rate: Physical error rate
        
    Returns:
        Number of shots to run
    """
    # Log-linear interpolation between low and high error rates
    p_low = PHYSICAL_ERROR_RATES[0]   # 1e-4
    p_high = PHYSICAL_ERROR_RATES[-1]  # 1e-2
    
    # Compute interpolation factor in log space (0 at p_low, 1 at p_high)
    log_p = np.log10(physical_error_rate)
    log_p_low = np.log10(p_low)
    log_p_high = np.log10(p_high)
    t = (log_p - log_p_low) / (log_p_high - log_p_low)
    t = np.clip(t, 0, 1)
    
    # Interpolate shots in log space
    log_shots_low = np.log10(BASE_SHOTS_LOW_P)
    log_shots_high = np.log10(BASE_SHOTS_HIGH_P)
    log_shots = log_shots_low + t * (log_shots_high - log_shots_low)
    base_shots = int(10 ** log_shots)
    
    # Apply multiplier for faster decoders
    if decoder in ['pymatching', 'correlated_pymatching']:
        shots = base_shots * MATCHING_MULTIPLIER
    else:
        shots = base_shots
    
    # Apply global shot scale
    shots = max(1, int(shots * SHOT_SCALE))
    return shots


# =============================================================================
# Circuit Generation
# =============================================================================

def generate_noisy_circuit(k: int, physical_error_rate: float) -> stim.Circuit:
    """Generate a spatial Hadamard circuit with full noise model.
    
    Args:
        k: Scaling factor (distance = 2k+1)
        physical_error_rate: Physical error rate for noise model
        
    Returns:
        Noisy stim circuit
    """
    # Generate circuit WITHOUT noise
    circuit = generate_spatial_hadamard_circuit(
        k=k,
        axis=DIRECTION,
        noise_model=None,
        flag_config=FLAG_CONFIG,
    )
    
    # Compact the circuit
    circuit = compact_and_delay_init(circuit)
    
    # Apply full noise model
    noise_model = NoiseModel.uniform_depolarizing(physical_error_rate)
    noisy_circuit = noise_model.noisy_circuit(circuit)
    
    return noisy_circuit


# =============================================================================
# Benchmark Execution
# =============================================================================

def run_single_benchmark(
    circuit: stim.Circuit,
    decoder_name: str,
    k: int,
    physical_error_rate: float,
    num_shots: int,
    num_workers: int,
) -> RuntimeResult:
    """Run benchmark for a single (circuit, decoder) combination.
    
    Args:
        circuit: The noisy stim circuit
        decoder_name: Name of the decoder
        k: Scaling factor
        physical_error_rate: Physical error rate
        num_shots: Number of shots to run
        num_workers: Number of parallel workers
        
    Returns:
        RuntimeResult object
    """
    task = sinter.Task(
        circuit=circuit,
        decoder=decoder_name,
        json_metadata={
            'k': k,
            'physical_error_rate': physical_error_rate,
            'decoder': decoder_name,
        },
    )
    
    # Run sinter with fixed shots (no early stopping via max_errors)
    start_time = time.time()
    stats = sinter.collect(
        tasks=[task],
        max_shots=num_shots,
        max_errors=num_shots + 1,  # Effectively disable early stopping
        num_workers=num_workers,
        custom_decoders=CUSTOM_DECODERS,
    )
    decode_time = time.time() - start_time
    
    if stats:
        stat = stats[0]
        error_rate = stat.errors / stat.shots if stat.shots > 0 else 0.0
        
        return RuntimeResult(
            k=k,
            physical_error_rate=physical_error_rate,
            decoder=decoder_name,
            shots=stat.shots,
            decode_time=decode_time,
            errors=stat.errors,
            logical_error_rate=error_rate,
        )
    
    return None


def save_results_to_csv(results: list, filepath: str) -> None:
    """Save benchmark results to CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    fieldnames = [
        'k',
        'physical_error_rate',
        'decoder',
        'shots',
        'decode_time',
        'errors',
        'logical_error_rate',
    ]
    
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            writer.writerow({
                'k': result.k,
                'physical_error_rate': result.physical_error_rate,
                'decoder': result.decoder,
                'shots': result.shots,
                'decode_time': result.decode_time,
                'errors': result.errors,
                'logical_error_rate': result.logical_error_rate,
            })


def run_benchmark() -> list:
    """Run the full runtime benchmark sweep.
    
    Loop order (outer to inner):
    1. k
    2. physical_error_rate
    3. decoder
    
    Saves results incrementally after each task.
    
    Returns:
        List of RuntimeResult objects
    """
    all_results = []
    
    total_tasks = len(K_VALUES) * len(PHYSICAL_ERROR_RATES) * len(DECODERS)
    task_num = 0
    first_write = True
    
    print("=" * 70)
    print("Decoder Runtime Benchmark")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Direction: {DIRECTION}")
    print(f"  Flag config: {FLAG_CONFIG}")
    print(f"  k values: {K_VALUES}")
    print(f"  Error rates: {len(PHYSICAL_ERROR_RATES)} values from {PHYSICAL_ERROR_RATES[0]:.2e} to {PHYSICAL_ERROR_RATES[-1]:.2e}")
    print(f"  Decoders: {DECODERS}")
    print(f"  Tesseract shots: {int(BASE_SHOTS_LOW_P * SHOT_SCALE)} (at p={PHYSICAL_ERROR_RATES[0]:.0e}) to {int(BASE_SHOTS_HIGH_P * SHOT_SCALE)} (at p={PHYSICAL_ERROR_RATES[-1]:.0e})")
    print(f"  Matching decoders: {MATCHING_MULTIPLIER}x more shots")
    print(f"  Shot scale: {SHOT_SCALE}x")
    print(f"  Total tasks: {total_tasks}")
    print(f"  Output file: {OUTPUT_CSV}")
    print("=" * 70)
    
    for k in K_VALUES:
        print(f"\n{'='*70}")
        print(f"k = {k} (distance = {2*k+1})")
        print(f"{'='*70}")
        
        for p in PHYSICAL_ERROR_RATES:
            print(f"\n  Physical error rate: {p:.2e}")
            
            # Generate circuit once per (k, p)
            circuit = generate_noisy_circuit(k, p)
            print(f"    Circuit: {circuit.num_qubits} qubits, {circuit.num_detectors} detectors")
            
            for decoder_name in DECODERS:
                task_num += 1
                num_shots = get_num_shots(decoder_name, p)
                print(f"    [{task_num}/{total_tasks}] Decoder: {decoder_name} ({num_shots} shots) ... ", end='', flush=True)
                
                result = run_single_benchmark(
                    circuit=circuit,
                    decoder_name=decoder_name,
                    k=k,
                    physical_error_rate=p,
                    num_shots=num_shots,
                    num_workers=NUM_WORKERS,
                )
                
                if result:
                    all_results.append(result)
                    print(f"time={result.decode_time:.2f}s, errors={result.errors}/{result.shots}")
                    
                    # Save incrementally
                    save_result_to_csv(result, OUTPUT_CSV, write_header=first_write)
                    first_write = False
                else:
                    print("FAILED")
    
    return all_results


def save_result_to_csv(result: RuntimeResult, filepath: str, write_header: bool = False) -> None:
    """Append a single result to CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    fieldnames = [
        'k',
        'physical_error_rate',
        'decoder',
        'shots',
        'decode_time',
        'errors',
        'logical_error_rate',
    ]
    
    mode = 'w' if write_header else 'a'
    with open(filepath, mode, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        
        writer.writerow({
            'k': result.k,
            'physical_error_rate': result.physical_error_rate,
            'decoder': result.decoder,
            'shots': result.shots,
            'decode_time': result.decode_time,
            'errors': result.errors,
            'logical_error_rate': result.logical_error_rate,
        })


# =============================================================================
# Main
# =============================================================================

def main():
    global NUM_WORKERS, OUTPUT_CSV, K_VALUES, SHOT_SCALE
    
    parser = argparse.ArgumentParser(
        description='Benchmark decoder runtime for spatial Hadamard circuits'
    )
    parser.add_argument(
        '--output', '-o',
        default=OUTPUT_CSV,
        help=f'Output CSV file (default: {OUTPUT_CSV})'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=NUM_WORKERS,
        help=f'Number of parallel workers (default: {NUM_WORKERS})'
    )
    parser.add_argument(
        '--k', '-k',
        type=int,
        nargs='+',
        default=None,
        help='Specific k value(s) to run (default: all [1,2,3,4])'
    )
    parser.add_argument(
        '--shot-scale',
        type=float,
        default=1.0,
        help='Multiplier for all shot counts (e.g., 0.1 for 10x fewer shots)'
    )
    
    args = parser.parse_args()
    
    # Update globals
    NUM_WORKERS = args.workers
    OUTPUT_CSV = args.output
    if args.k is not None:
        K_VALUES = args.k
    SHOT_SCALE = args.shot_scale
    
    # Run benchmark (results are saved incrementally)
    results = run_benchmark()
    
    print(f"\nBenchmark complete! {len(results)} results saved to {OUTPUT_CSV}")
    print(f"To plot results, run: python plot_decoder_runtime.py --csv {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
