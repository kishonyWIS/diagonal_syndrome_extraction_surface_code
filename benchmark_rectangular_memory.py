#!/usr/bin/env python3
"""Benchmark rectangular memory experiment: two ZXZ cubes connected by a pipe."""

import csv
import os
import sys
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, 'venv/lib/python3.13/site-packages')

from tqec.computation.block_graph import BlockGraph
from tqec.computation.cube import ZXCube
from tqec.utils.position import Position3D
from tqec.compile.compile import compile_block_graph
from tqec.compile.convention import FIXED_BULK_CONVENTION
from tqec.utils.noise_model import NoiseModel
from tqec.utils.enums import Basis
import stim
import sinter

try:
    import pymatching
    PYMATCHING_AVAILABLE = True
except ImportError:
    PYMATCHING_AVAILABLE = False

# Optional tesseract import
try:
    import tesseract_decoder.tesseract as tesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    tesseract = None


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
            # Load DEM (full, not decomposed)
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


# Custom decoder registry for sinter
CUSTOM_DECODERS = {
    'pymatching': 'pymatching',  # Built-in sinter decoder
    'correlated_pymatching': CorrelatedPymatchingDecoder(),
}

if TESSERACT_AVAILABLE:
    CUSTOM_DECODERS['tesseract'] = TesseractDecoder()

# Modify MEASUREMENT_SCHEDULE BEFORE importing any tqec modules
import tqec.plaquette.constants as constants
constants.MEASUREMENT_SCHEDULE = 8  # Allow schedule values up to 7

# Reload the translator module to pick up the new MEASUREMENT_SCHEDULE
import importlib
import tqec.plaquette.rpng.translators.default as default_translator
importlib.reload(default_translator)

# Import diagonal convention
from benchmark_memory import create_diagonal_convention
from compact_circuit import compact_and_delay_init


def create_rectangular_memory_block_graph():
    """Create a block graph for rectangular memory: two ZXZ cubes connected by a pipe.
    
    - ZXZ cube at position (0, 0, 0)
    - ZXZ cube at position (1, 0, 0)
    - Pipe connecting them in X direction
    """
    graph = BlockGraph("Rectangular Memory")
    
    # Define positions
    cube1_pos = Position3D(0, 0, 0)
    cube2_pos = Position3D(1, 0, 0)
    
    # Add two ZXZ cubes
    graph.add_cube(cube1_pos, ZXCube.from_str("ZXZ"), "cube1")
    graph.add_cube(cube2_pos, ZXCube.from_str("ZXZ"), "cube2")
    
    # Add pipe connecting them in X direction
    graph.add_pipe(cube1_pos, cube2_pos)
    
    return graph


def compile_and_generate(graph, convention_name, convention, k=1):
    """Compile the graph and generate a Stim circuit (without noise, compacted)."""
    print(f"\nCompiling with {convention_name} convention (k={k})...")
    
    try:
        compiled_graph = compile_block_graph(
            block_graph=graph,
            convention=convention
        )
        print(f"✓ Successfully compiled block graph")
        
        manhattan_radius = 2
        
        # Generate circuit WITHOUT noise
        circuit_before = compiled_graph.generate_stim_circuit(
            k=k, 
            manhattan_radius=manhattan_radius,
            noise_model=None
        )
        
        # Compact the circuit (ASAP + ALAP scheduling)
        circuit = compact_and_delay_init(circuit_before)
        print(f"✓ Successfully generated and compacted Stim circuit for k={k}")
        print(f"  Number of instructions: {len(circuit)}")
        print(f"  Number of qubits: {circuit.num_qubits}")
        print(f"  Number of detectors: {circuit.num_detectors}")
        print(f"  Number of observables: {circuit.num_observables}")
        
        return {
            'circuit': circuit,
            'circuit_before_compact': circuit_before,
            'num_instructions': len(circuit),
            'num_qubits': circuit.num_qubits,
            'num_detectors': circuit.num_detectors,
            'num_observables': circuit.num_observables,
            'compiled_graph': compiled_graph,
        }
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_logical_error_rate(circuit, shots=100000, noise_levels=[0.001], decoder='correlated_pymatching'):
    """Calculate logical error rate using sinter.
    
    Args:
        circuit: A noise-free stim circuit (noise will be added)
        shots: Number of shots per noise level
        noise_levels: List of physical error rates to test
        decoder: Decoder to use ('pymatching' or 'correlated_pymatching')
    """
    if decoder in ('pymatching', 'correlated_pymatching') and not PYMATCHING_AVAILABLE:
        print("Skipping logical error rate calculation - pymatching not available")
        return {}
    if decoder == 'tesseract' and not TESSERACT_AVAILABLE:
        print("Skipping logical error rate calculation - tesseract_decoder not available")
        return {}
    
    print(f"Calculating logical error rate with {shots} shots...")
    
    results = {}
    
    for noise_level in noise_levels:
        print(f"  Testing noise level: {noise_level}")
        
        # Add noise to the circuit using NoiseModel.noisy_circuit()
        noise_model = NoiseModel.uniform_depolarizing(noise_level)
        noisy_circuit = noise_model.noisy_circuit(circuit)
        
        # For pymatching and correlated_pymatching, we need a decomposed (graphlike) DEM.
        # For tesseract, we pass the full DEM (so leave detector_error_model=None).
        detector_error_model = None
        if decoder in ['pymatching', 'correlated_pymatching']:
            detector_error_model = noisy_circuit.detector_error_model(
                decompose_errors=True,
                ignore_decomposition_failures=True
            )
        
        # Use sinter to collect statistics
        task = sinter.Task(
            circuit=noisy_circuit,
            decoder=decoder,
            detector_error_model=detector_error_model,  # Use pre-computed DEM for pymatching decoders
            json_metadata={'noise_level': noise_level}
        )
        
        # Collect statistics using sinter and measure decode time
        start_time = time.time()
        # Only pass custom_decoders if using a custom decoder
        collect_kwargs = {
            'tasks': [task],
            'max_shots': shots,
            'max_errors': 3000,
            'num_workers': 10,
        }
        if decoder in CUSTOM_DECODERS and isinstance(CUSTOM_DECODERS[decoder], sinter.Decoder):
            collect_kwargs['custom_decoders'] = CUSTOM_DECODERS
        stats = sinter.collect(**collect_kwargs)
        decode_time = time.time() - start_time
        
        # Extract results
        if stats:
            stat = stats[0]
            logical_error_rate = stat.errors / stat.shots
            logical_errors = stat.errors
            
            # Calculate error bars using binomial distribution
            error_bar = np.sqrt(logical_error_rate * (1 - logical_error_rate) / stat.shots)
            
            results[noise_level] = {
                'logical_error_rate': logical_error_rate,
                'logical_errors': logical_errors,
                'shots': stat.shots,
                'error_bar': error_bar,
                'decode_time': decode_time
            }
            
            print(f"    Logical error rate: {logical_error_rate:.6f} ± {error_bar:.6f} ({logical_errors}/{stat.shots}), time={decode_time:.1f}s")
        else:
            print(f"    No statistics collected for noise level {noise_level}")
            results[noise_level] = {
                'logical_error_rate': 0.0,
                'logical_errors': 0,
                'shots': 0,
                'error_bar': 0.0,
                'decode_time': decode_time
            }
    
    return results


def save_error_rates_to_csv(all_error_rates, filepath=None, decoder='correlated_pymatching'):
    """Save logical error rate results to CSV file.
    
    Args:
        all_error_rates: Dict with structure {k: {circuit_type: {noise_level: {logical_error_rate, logical_errors, shots, error_bar}}}}
        filepath: Path to save CSV file (default: based on decoder name)
        decoder: Decoder name for default filename
    """
    if filepath is None:
        if decoder == 'correlated_pymatching':
            decoder_suffix = 'correlated_pymatching'
        elif decoder == 'pymatching':
            decoder_suffix = 'pymatching'
        else:
            decoder_suffix = decoder
        filepath = f"benchmark_data/rectangular_memory_error_rates_{decoder_suffix}.csv"
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    fieldnames = ['k', 'circuit_type', 'physical_error_rate', 'logical_error_rate', 'logical_errors', 'shots', 'error_bar', 'decode_time']
    
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for k in sorted(all_error_rates.keys()):
            for circuit_type in sorted(all_error_rates[k].keys()):
                circuit_results = all_error_rates[k][circuit_type]
                for noise_level in sorted(circuit_results.keys()):
                    result = circuit_results[noise_level]
                    writer.writerow({
                        'k': k,
                        'circuit_type': circuit_type,
                        'physical_error_rate': noise_level,
                        'logical_error_rate': result['logical_error_rate'],
                        'logical_errors': result['logical_errors'],
                        'shots': result['shots'],
                        'error_bar': result['error_bar'],
                        'decode_time': result.get('decode_time', 0.0),
                    })
    
    print(f"\n✓ Saved error rates to {filepath}")


def load_error_rates_from_csv(filepath):
    """Load logical error rate results from CSV file.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        Dict with structure {k: {circuit_type: {noise_level: {logical_error_rate, logical_errors, shots, error_bar}}}}
    """
    if not os.path.exists(filepath):
        return {}
    
    all_error_rates = {}
    
    with open(filepath, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            k = int(row['k'])
            circuit_type = row['circuit_type']
            noise_level = float(row['physical_error_rate'])
            
            if k not in all_error_rates:
                all_error_rates[k] = {}
            if circuit_type not in all_error_rates[k]:
                all_error_rates[k][circuit_type] = {}
            
            all_error_rates[k][circuit_type][noise_level] = {
                'logical_error_rate': float(row['logical_error_rate']),
                'logical_errors': int(row['logical_errors']),
                'shots': int(row['shots']),
                'error_bar': float(row['error_bar']),
                'decode_time': float(row.get('decode_time', 0.0)),
            }
    
    return all_error_rates


def main():
    parser = argparse.ArgumentParser(description='Benchmark rectangular memory experiment')
    parser.add_argument('--k-values', nargs='+', type=int, default=[1, 2, 3, 4],
                       help='k values to test (default: 1 2 3 4)')
    parser.add_argument('--shots', type=int, default=100000000,
                       help='Number of shots per noise level (default: 100000000)')
    parser.add_argument('--noise-levels', nargs='+', type=float,
                       default=[0.001],
                       help='Physical error rates to test (default: 0.001)')
    decoder_choices = ['pymatching', 'correlated_pymatching']
    if TESSERACT_AVAILABLE:
        decoder_choices.append('tesseract')
    parser.add_argument('--decoder', type=str, default='correlated_pymatching',
                       choices=decoder_choices,
                       help='Decoder to use (default: correlated_pymatching)')
    parser.add_argument('--load-error-rates', type=str, default=None,
                       help='Load error rates from CSV file instead of recomputing')
    parser.add_argument('--plot-only', action='store_true',
                       help='Only generate plots from existing CSV data (skip all computations)')
    
    args = parser.parse_args()
    
    k_values = args.k_values
    shots = args.shots
    noise_levels = args.noise_levels
    
    print("=" * 70)
    print("Rectangular Memory Experiment Benchmark")
    print("=" * 70)
    print(f"Testing k values: {k_values}")
    print(f"Physical error rates: {noise_levels}")
    print(f"Shots per configuration: {shots:,}")
    print(f"Decoder: {args.decoder}")
    print()
    
    # Create block graph
    graph = create_rectangular_memory_block_graph()
    print(f"Created block graph: {graph}")
    print()
    
    # Create diagonal convention
    diagonal_convention = create_diagonal_convention()
    
    all_results = {}
    all_error_rates = {}
    
    # Load existing data if requested
    if args.load_error_rates:
        print(f"Loading error rates from {args.load_error_rates}...")
        all_error_rates = load_error_rates_from_csv(args.load_error_rates)
        print(f"Loaded data for k values: {sorted(all_error_rates.keys())}")
        print()
    
    # Process each k value
    for k in k_values:
        print("=" * 70)
        print(f"Processing k={k} (distance={2*k+1})")
        print("=" * 70)
        
        # Compile with standard (N/Z) convention
        print("\n--- N/Z Schedule ---")
        standard_result = compile_and_generate(
            graph, 
            "N/Z (Fixed Bulk)", 
            FIXED_BULK_CONVENTION, 
            k=k
        )
        
        if standard_result is None:
            print("Failed to compile standard circuit, skipping k={k}")
            continue
        
        # Compile with diagonal convention
        print("\n--- Diagonal Schedule ---")
        diagonal_result = compile_and_generate(
            graph,
            "Diagonal",
            diagonal_convention,
            k=k
        )
        
        if diagonal_result is None:
            print("Failed to compile diagonal circuit, skipping k={k}")
            continue
        
        all_results[k] = {
            'standard': standard_result,
            'diagonal': diagonal_result,
        }
        
        # Calculate logical error rates (optional, skip if loaded from CSV)
        if args.load_error_rates:
            if k in all_error_rates:
                print(f"Using loaded error rates for k={k}")
            else:
                print(f"Warning: No loaded error rates for k={k}")
            print()
        elif not args.plot_only and PYMATCHING_AVAILABLE:
            print(f"\nCalculating logical error rates for k={k}...")
            k_error_rates = {}
            
            # Determine CSV filepath for saving
            csv_filepath = f"benchmark_data/rectangular_memory_error_rates_{args.decoder}.csv"
            
            # Standard (N/Z) circuit
            print("\nN/Z Circuit:")
            standard_error_rates = calculate_logical_error_rate(
                standard_result['circuit'],
                shots=shots,
                noise_levels=noise_levels,
                decoder=args.decoder
            )
            k_error_rates['N/Z Circuit'] = standard_error_rates
            
            # Diagonal circuit
            print("\nDiagonal Circuit:")
            diagonal_error_rates = calculate_logical_error_rate(
                diagonal_result['circuit'],
                shots=shots,
                noise_levels=noise_levels,
                decoder=args.decoder
            )
            k_error_rates['Diagonal Circuit'] = diagonal_error_rates
            
            all_error_rates[k] = k_error_rates
            
            # Save incrementally
            save_error_rates_to_csv(all_error_rates, decoder=args.decoder)
            print()
        elif args.plot_only:
            print(f"Skipping logical error rate calculation (plot-only mode)")
            print()
        elif not PYMATCHING_AVAILABLE:
            print(f"Skipping logical error rate calculation (pymatching not available)")
            print()
    
    # Final save
    if all_error_rates and not args.plot_only:
        save_error_rates_to_csv(all_error_rates, decoder=args.decoder)
    
    print("=" * 70)
    print("Benchmark complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
