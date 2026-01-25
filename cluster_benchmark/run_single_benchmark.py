#!/usr/bin/env python3
"""
Single benchmark task runner for cluster execution.

This script runs a single benchmark configuration and saves the result to a CSV file.
Designed to be called from LSF job scripts with environment variables.
"""

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pymatching
import stim
import sinter
import tesseract_decoder.tesseract as tesseract

# Add project directory to path for imports
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from spatial_hadamard_manual_construction import (
    generate_spatial_hadamard_circuit,
)
from compact_circuit import compact_and_delay_init
from tqec import NoiseModel

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)


# =============================================================================
# Interface-Only Noise Support
# =============================================================================

def get_qubit_coordinates(circuit: stim.Circuit) -> dict[int, tuple[float, float]]:
    """Extract qubit coordinates from QUBIT_COORDS instructions.
    
    Returns:
        Dict mapping qubit index to (x, y) coordinates.
    """
    coords = {}
    for inst in circuit.flattened():
        if inst.name == "QUBIT_COORDS":
            args = inst.gate_args_copy()
            for target in inst.targets_copy():
                coords[target.value] = (args[0], args[1])
    return coords


def identify_interface_qubits(
    circuit: stim.Circuit,
    k: int,
    axis: str,
    margin: float = 1.0,
) -> set[int]:
    """Identify qubits at or near the interface between the two cubes.
    
    The interface is located at coordinates 4*k+2 to 4*k+4 along the interface axis.
    Qubits within `margin` of this range are considered "interface qubits".
    
    Args:
        circuit: The stim circuit with QUBIT_COORDS
        k: Scaling factor
        axis: 'x' or 'y' - the direction of the two-cube layout
        margin: How far from the interface to include qubits (default: 1.0)
        
    Returns:
        Set of qubit indices that are at or near the interface.
    """
    coords = get_qubit_coordinates(circuit)
    
    # Interface is at 4*k+2 to 4*k+4
    interface_min = 4 * k + 2 - margin
    interface_max = 4 * k + 4 + margin
    
    interface_qubits = set()
    for qubit, (x, y) in coords.items():
        # Check the relevant coordinate based on axis
        coord = x if axis == 'x' else y
        if interface_min <= coord <= interface_max:
            interface_qubits.add(qubit)
    
    return interface_qubits


def apply_interface_only_noise(
    circuit: stim.Circuit,
    noise_model: NoiseModel,
    k: int,
    axis: str,
    margin: float = 1.0,
) -> stim.Circuit:
    """Apply noise only to qubits at or near the interface.
    
    All qubits NOT at the interface are marked as "immune" to noise.
    
    Args:
        circuit: The noise-free circuit
        noise_model: The noise model to apply
        k: Scaling factor
        axis: 'x' or 'y' - the direction of the two-cube layout
        margin: How far from the interface to include qubits
        
    Returns:
        Circuit with noise applied only to interface qubits.
    """
    # Identify interface qubits
    interface_qubits = identify_interface_qubits(circuit, k, axis, margin)
    
    # All other qubits are immune
    all_qubits = set(range(circuit.num_qubits))
    immune_qubits = all_qubits - interface_qubits
    
    # Apply noise with immune qubits
    return noise_model.noisy_circuit(circuit, immune_qubits=immune_qubits)


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
# Circuit Generation
# =============================================================================

def generate_circuit_for_config(
    direction: str,
    flag_config: str,
    k: int,
) -> stim.Circuit:
    """Generate a spatial Hadamard circuit for the given configuration (without noise, compacted).
    
    Args:
        direction: 'x' or 'y'
        flag_config: 'all', 'partial', or 'none'
        k: Scaling factor
    """
    # Generate circuit WITHOUT noise
    circuit_before = generate_spatial_hadamard_circuit(
        k=k,
        axis=direction,
        noise_model=None,
        flag_config=flag_config,
    )
    
    # Compact the circuit (ASAP + ALAP scheduling)
    circuit_after = compact_and_delay_init(circuit_before)
    
    return circuit_after


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
    # For pymatching and correlated_pymatching, we need a decomposed (graphlike) DEM
    # Pre-compute it with decompose_errors=True as per PyMatching instructions
    detector_error_model = None
    if decoder_name in ['pymatching', 'correlated_pymatching']:
        detector_error_model = circuit.detector_error_model(
            decompose_errors=True,
            ignore_decomposition_failures=True
        )
    
    task = sinter.Task(
        circuit=circuit,
        decoder=decoder_name,
        detector_error_model=detector_error_model,  # Use pre-computed DEM for pymatching decoders
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
        
        print(f"Result: {stat.errors}/{stat.shots} errors, rate={error_rate:.6f}, time={decode_time:.1f}s")
        return result
    
    return None


def save_result_to_csv(result: BenchmarkResult, filepath: str) -> None:
    """Save a single benchmark result to CSV file."""
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
    
    # Check if file exists to determine if we need a header
    write_header = not os.path.exists(filepath)
    
    with open(filepath, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        
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
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for single benchmark task."""
    parser = argparse.ArgumentParser(
        description="Run a single benchmark configuration for cluster execution."
    )
    parser.add_argument('--decoder', type=str, required=True,
                        help='Decoder name (pymatching, correlated_pymatching, tesseract)')
    parser.add_argument('--k', type=int, required=True,
                        help='Scaling factor k')
    parser.add_argument('--noise', type=float, required=True,
                        help='Physical error rate')
    parser.add_argument('--flag-config', type=str, required=True,
                        choices=['all', 'partial', 'none'],
                        help='Flag configuration')
    parser.add_argument('--direction', type=str, required=True,
                        choices=['x', 'y'],
                        help='Direction (x or y)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output CSV file path')
    parser.add_argument('--noise-mode', type=str, default='interface_only',
                        choices=['full', 'interface_only'],
                        help='Noise mode (default: interface_only)')
    parser.add_argument('--max-shots', type=int, default=1_000_000_000_000,
                        help='Maximum shots (default: 1e12)')
    parser.add_argument('--max-errors', type=int, default=1000,
                        help='Maximum errors for early stopping (default: 1000)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')
    parser.add_argument('--interface-margin', type=float, default=1.0,
                        help='Interface margin for interface_only noise (default: 1.0)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Single Benchmark Task")
    print("=" * 70)
    print(f"Decoder: {args.decoder}")
    print(f"k: {args.k}")
    print(f"Noise: {args.noise:.6f}")
    print(f"Flag config: {args.flag_config}")
    print(f"Direction: {args.direction}")
    print(f"Noise mode: {args.noise_mode}")
    print(f"Max shots: {args.max_shots:,}")
    print(f"Max errors: {args.max_errors:,}")
    print(f"Num workers: {args.num_workers}")
    print(f"Output: {args.output}")
    print()
    
    # Generate compacted circuit (without noise)
    print("Generating circuit...")
    circuit = generate_circuit_for_config(
        direction=args.direction,
        flag_config=args.flag_config,
        k=args.k,
    )
    print(f"  Qubits: {circuit.num_qubits}, Detectors: {circuit.num_detectors}")
    
    # Add noise using NoiseModel.noisy_circuit()
    print("Applying noise...")
    noise_model = NoiseModel.uniform_depolarizing(args.noise)
    if args.noise_mode == 'interface_only':
        noisy_circuit = apply_interface_only_noise(
            circuit, noise_model, args.k, args.direction, args.interface_margin
        )
    else:
        noisy_circuit = noise_model.noisy_circuit(circuit)
    print("  Noise applied")
    
    metadata = {
        'direction': args.direction,
        'flag_config': args.flag_config,
        'k': args.k,
        'distance': None,  # Distance not computed for single runs
        'physical_error_rate': args.noise,
    }
    
    # Run benchmark
    print("Running benchmark...")
    try:
        result = run_sinter_for_single_task(
            circuit=noisy_circuit,
            metadata=metadata,
            decoder_name=args.decoder,
            max_shots=args.max_shots,
            max_errors=args.max_errors,
            num_workers=args.num_workers,
        )
        
        if result:
            # Save result
            save_result_to_csv(result, args.output)
            print(f"\nResult saved to {args.output}")
        else:
            print("ERROR: No result returned from benchmark")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
