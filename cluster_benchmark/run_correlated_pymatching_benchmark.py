#!/usr/bin/env python3
"""
Unified benchmark runner for correlated_pymatching decoder across all experiment types.

This script runs a single benchmark configuration and saves the result to a CSV file
in a unified format that works for all experiment types.
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

# Add project directory to path for imports
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from tqec import NoiseModel
from compact_circuit import compact_and_delay_init

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)


# =============================================================================
# Custom Sinter Decoder
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
        
        # Convert predictions to bool_ dtype if needed (decode_batch may return int or bool)
        import numpy as np
        if predictions.dtype != np.bool_:
            predictions = predictions.astype(np.bool_)
        
        # Write predictions (use bool_ dtype for bit-packed format)
        stim.write_shot_data_file(
            data=predictions,
            path=obs_predictions_b8_out_path,
            format='b8',
            num_observables=num_obs,
        )


# Custom decoder registry for sinter
CUSTOM_DECODERS = {
    'correlated_pymatching': CorrelatedPymatchingDecoder(),
}


# =============================================================================
# Circuit Generation Functions
# =============================================================================

def generate_memory_circuit(k: int, schedule: str) -> stim.Circuit:
    """Generate memory circuit (N/Z or diagonal schedule)."""
    from benchmark_memory import (
        create_original_memory_circuit,
        create_diagonal_memory_circuit,
    )
    
    if schedule.lower() == 'n/z':
        return create_original_memory_circuit(k)
    elif schedule.lower() == 'diagonal':
        return create_diagonal_memory_circuit(k)
    else:
        raise ValueError(f"Unknown schedule for memory: {schedule}")


def generate_x_junction_circuit(k: int, schedule: str) -> stim.Circuit:
    """Generate X junction circuit (N/Z or diagonal schedule)."""
    from benchmark_x_junction import (
        create_x_junction_block_graph,
        compile_and_generate,
    )
    from benchmark_x_junction import FIXED_BULK_CONVENTION
    from benchmark_memory import create_diagonal_convention
    
    graph = create_x_junction_block_graph()
    
    if schedule.lower() == 'n/z':
        result = compile_and_generate(
            graph, "N/Z Fixed-Bulk", FIXED_BULK_CONVENTION, k=k, use_diagonal=False
        )
    elif schedule.lower() == 'diagonal':
        diagonal_convention = create_diagonal_convention()
        result = compile_and_generate(
            graph, "Diagonal Schedule", diagonal_convention, k=k, use_diagonal=False
        )
    else:
        raise ValueError(f"Unknown schedule for x_junction: {schedule}")
    
    if result is None:
        raise RuntimeError(f"Failed to generate X junction circuit for k={k}, schedule={schedule}")
    
    return result['circuit']


def generate_patch_rotation_circuit(k: int, basis: str) -> stim.Circuit:
    """Generate patch rotation circuit."""
    from benchmark_patch_rotation import (
        generate_circuit_for_config,
        CircuitConfig,
    )
    
    config = CircuitConfig(k, basis)
    return generate_circuit_for_config(config)


def generate_spatial_hadamard_circuit(k: int, direction: str, flag_config: str) -> stim.Circuit:
    """Generate spatial hadamard circuit."""
    from benchmark_spatial_hadamard import generate_circuit_for_config, CircuitConfig
    
    config = CircuitConfig(direction, flag_config, k)
    return generate_circuit_for_config(config)


def apply_interface_only_noise(
    circuit: stim.Circuit,
    noise_model: NoiseModel,
    k: int,
    axis: str,
    margin: float = 1.0,
) -> stim.Circuit:
    """Apply noise only to qubits at or near the interface."""
    from benchmark_spatial_hadamard import (
        get_qubit_coordinates,
        identify_interface_qubits,
    )
    
    # Identify interface qubits
    interface_qubits = identify_interface_qubits(circuit, k, axis, margin)
    
    # All other qubits are immune
    all_qubits = set(range(circuit.num_qubits))
    immune_qubits = all_qubits - interface_qubits
    
    # Apply noise with immune qubits
    return noise_model.noisy_circuit(circuit, immune_qubits=immune_qubits)


# =============================================================================
# Benchmark Execution
# =============================================================================

def run_sinter_benchmark(
    circuit: stim.Circuit,
    max_shots: int,
    max_errors: int,
    num_workers: int,
) -> tuple[float, int, int, float, float]:
    """
    Run sinter benchmark for a single circuit.
    
    Returns:
        (logical_error_rate, errors, shots, error_bar, decode_time)
    """
    # For correlated_pymatching, we need a decomposed (graphlike) DEM
    # Pre-compute it with decompose_errors=True as per PyMatching instructions
    detector_error_model = circuit.detector_error_model(
        decompose_errors=True,
        ignore_decomposition_failures=True
    )
    
    task = sinter.Task(
        circuit=circuit,
        decoder='correlated_pymatching',
        detector_error_model=detector_error_model,
        json_metadata={},
    )
    
    # Run sinter
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
        
        return error_rate, stat.errors, stat.shots, error_bar, decode_time
    
    return 0.0, 0, 0, 0.0, decode_time


def save_result_to_csv(
    experiment_type: str,
    schedule: str,
    noise_mode: str,
    k: int,
    physical_error_rate: float,
    logical_error_rate: float,
    errors: int,
    shots: int,
    error_bar: float,
    decode_time: float,
    distance: Optional[int],
    basis: str,
    direction: str,
    flag_config: str,
    filepath: str,
) -> None:
    """Save a single benchmark result to CSV file in unified format."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    fieldnames = [
        'experiment_type',
        'schedule',
        'noise_mode',
        'k',
        'physical_error_rate',
        'decoder',
        'logical_error_rate',
        'errors',
        'shots',
        'error_bar',
        'decode_time',
        'distance',
        'basis',
        'direction',
        'flag_config',
    ]
    
    # Check if file exists to determine if we need a header
    write_header = not os.path.exists(filepath)
    
    with open(filepath, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        
        writer.writerow({
            'experiment_type': experiment_type,
            'schedule': schedule if schedule else '',
            'noise_mode': noise_mode if noise_mode else '',
            'k': k,
            'physical_error_rate': physical_error_rate,
            'decoder': 'correlated_pymatching',
            'logical_error_rate': logical_error_rate,
            'errors': errors,
            'shots': shots,
            'error_bar': error_bar,
            'decode_time': decode_time,
            'distance': distance if distance is not None else '',
            'basis': basis if basis else '',
            'direction': direction if direction else '',
            'flag_config': flag_config if flag_config else '',
        })


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for single benchmark task."""
    parser = argparse.ArgumentParser(
        description="Run a single benchmark configuration for cluster execution."
    )
    parser.add_argument('--experiment-type', type=str, required=True,
                        choices=['memory', 'x_junction', 'patch_rotation', 'spatial_hadamard'],
                        help='Experiment type')
    parser.add_argument('--schedule', type=str, default='',
                        choices=['N/Z', 'diagonal', ''],
                        help='Schedule type (for memory/x_junction)')
    parser.add_argument('--k', type=int, required=True,
                        help='Scaling factor k')
    parser.add_argument('--noise', type=float, required=True,
                        help='Physical error rate')
    parser.add_argument('--noise-mode', type=str, default='',
                        choices=['full', 'interface_only', ''],
                        help='Noise mode (for spatial_hadamard)')
    parser.add_argument('--flag-config', type=str, default='',
                        choices=['none', 'partial', 'all', ''],
                        help='Flag configuration (for spatial_hadamard)')
    parser.add_argument('--direction', type=str, default='',
                        choices=['x', 'y', ''],
                        help='Direction (for spatial_hadamard)')
    parser.add_argument('--basis', type=str, default='',
                        choices=['z', 'x', ''],
                        help='Basis (for patch_rotation)')
    parser.add_argument('--max-shots', type=int, default=1_000_000_000,
                        help='Maximum shots')
    parser.add_argument('--max-errors', type=int, default=3000,
                        help='Maximum errors for early stopping')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--output', type=str, required=True,
                        help='Output CSV file path')
    parser.add_argument('--interface-margin', type=float, default=1.0,
                        help='Interface margin for interface_only noise (default: 1.0)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Correlated PyMatching Benchmark Task")
    print("=" * 70)
    print(f"Experiment type: {args.experiment_type}")
    if args.schedule:
        print(f"Schedule: {args.schedule}")
    print(f"k: {args.k}")
    print(f"Noise: {args.noise:.6f}")
    if args.noise_mode:
        print(f"Noise mode: {args.noise_mode}")
    if args.flag_config:
        print(f"Flag config: {args.flag_config}")
    if args.direction:
        print(f"Direction: {args.direction}")
    if args.basis:
        print(f"Basis: {args.basis}")
    print(f"Max shots: {args.max_shots:,}")
    print(f"Max errors: {args.max_errors:,}")
    print(f"Num workers: {args.num_workers}")
    print(f"Output: {args.output}")
    print()
    
    # Generate circuit (without noise)
    print("Generating circuit...")
    try:
        if args.experiment_type == 'memory':
            if not args.schedule:
                raise ValueError("Schedule required for memory experiment")
            circuit = generate_memory_circuit(args.k, args.schedule)
        elif args.experiment_type == 'x_junction':
            if not args.schedule:
                raise ValueError("Schedule required for x_junction experiment")
            circuit = generate_x_junction_circuit(args.k, args.schedule)
        elif args.experiment_type == 'patch_rotation':
            if not args.basis:
                raise ValueError("Basis required for patch_rotation experiment")
            circuit = generate_patch_rotation_circuit(args.k, args.basis)
        elif args.experiment_type == 'spatial_hadamard':
            if not args.flag_config or not args.direction:
                raise ValueError("Flag config and direction required for spatial_hadamard experiment")
            circuit = generate_spatial_hadamard_circuit(args.k, args.direction, args.flag_config)
        else:
            raise ValueError(f"Unknown experiment type: {args.experiment_type}")
        
        print(f"  Qubits: {circuit.num_qubits}, Detectors: {circuit.num_detectors}")
    except Exception as e:
        print(f"ERROR generating circuit: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Add noise
    print("Applying noise...")
    try:
        noise_model = NoiseModel.uniform_depolarizing(args.noise)
        
        if args.experiment_type == 'spatial_hadamard' and args.noise_mode == 'interface_only':
            noisy_circuit = apply_interface_only_noise(
                circuit, noise_model, args.k, args.direction, args.interface_margin
            )
        else:
            noisy_circuit = noise_model.noisy_circuit(circuit)
        
        print("  Noise applied")
    except Exception as e:
        print(f"ERROR applying noise: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Run benchmark
    print("Running benchmark...")
    try:
        logical_error_rate, errors, shots, error_bar, decode_time = run_sinter_benchmark(
            noisy_circuit,
            max_shots=args.max_shots,
            max_errors=args.max_errors,
            num_workers=args.num_workers,
        )
        
        print(f"Result: {errors}/{shots} errors, rate={logical_error_rate:.6f}, time={decode_time:.1f}s")
        
        # Save result
        save_result_to_csv(
            experiment_type=args.experiment_type,
            schedule=args.schedule,
            noise_mode=args.noise_mode,
            k=args.k,
            physical_error_rate=args.noise,
            logical_error_rate=logical_error_rate,
            errors=errors,
            shots=shots,
            error_bar=error_bar,
            decode_time=decode_time,
            distance=None,  # Distance not computed for single runs
            basis=args.basis,
            direction=args.direction,
            flag_config=args.flag_config,
            filepath=args.output,
        )
        
        print(f"\nResult saved to {args.output}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
