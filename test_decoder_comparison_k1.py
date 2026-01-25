#!/usr/bin/env python3
"""Quick test to compare pymatching vs correlated_pymatching for k=1."""

import sys
sys.path.insert(0, 'venv/lib/python3.13/site-packages')

import stim
import sinter
import pymatching
import numpy as np
from tqec.gallery import memory
from tqec import compile_block_graph, NoiseModel
from tqec.utils.enums import Basis
from benchmark_memory import create_diagonal_convention, CorrelatedPymatchingDecoder

if __name__ == '__main__':
    # Test parameters
    k = 1
    physical_error_rate = 0.001
    shots = 100000

    print("=" * 70)
    print(f"Decoder Comparison Test: k={k}, p={physical_error_rate}, shots={shots}")
    print("=" * 70)
    print()

    # Create diagonal memory circuit
    print("Creating diagonal memory circuit...")
    from benchmark_memory import create_diagonal_memory_circuit
    circuit = create_diagonal_memory_circuit(k=k)
    print(f"Circuit created: {len(circuit)} instructions, {circuit.num_qubits} qubits, {circuit.num_detectors} detectors")
    print()

    # Add noise
    print(f"Adding noise (p={physical_error_rate})...")
    noise_model = NoiseModel.uniform_depolarizing(physical_error_rate)
    noisy_circuit = noise_model.noisy_circuit(circuit)
    print()

    # Test both decoders
    decoders_to_test = ['pymatching', 'correlated_pymatching']
    results = {}

    CUSTOM_DECODERS = {
        'correlated_pymatching': CorrelatedPymatchingDecoder(),
    }

    for decoder_name in decoders_to_test:
        print(f"Testing {decoder_name}...")
        
        # Create DEM with decompose_errors=True for both decoders
        detector_error_model = noisy_circuit.detector_error_model(
            decompose_errors=True,
            ignore_decomposition_failures=True
        )
        
        # Create task
        task = sinter.Task(
            circuit=noisy_circuit,
            decoder=decoder_name,
            detector_error_model=detector_error_model,
            json_metadata={'decoder': decoder_name}
        )
        
        # Collect statistics
        collect_kwargs = {
            'tasks': [task],
            'max_shots': shots,
            'max_errors': 3000,
            'num_workers': 10,
        }
        if decoder_name in CUSTOM_DECODERS:
            collect_kwargs['custom_decoders'] = CUSTOM_DECODERS
        
        stats = sinter.collect(**collect_kwargs)
        
        if stats:
            stat = stats[0]
            logical_error_rate = stat.errors / stat.shots
            error_bar = np.sqrt(logical_error_rate * (1 - logical_error_rate) / stat.shots)
            
            results[decoder_name] = {
                'errors': stat.errors,
                'shots': stat.shots,
                'logical_error_rate': logical_error_rate,
                'error_bar': error_bar,
            }
            
            print(f"  Errors: {stat.errors}/{stat.shots}")
            print(f"  Logical error rate: {logical_error_rate:.6f} ± {error_bar:.6f}")
        else:
            print(f"  No results!")
        print()

    # Compare results
    print("=" * 70)
    print("Comparison:")
    print("=" * 70)
    if 'pymatching' in results and 'correlated_pymatching' in results:
        pm = results['pymatching']
        cpm = results['correlated_pymatching']
        
        print(f"PyMatching:           {pm['logical_error_rate']:.6f} ± {pm['error_bar']:.6f}")
        print(f"Correlated PyMatching: {cpm['logical_error_rate']:.6f} ± {cpm['error_bar']:.6f}")
        
        ratio = cpm['logical_error_rate'] / pm['logical_error_rate'] if pm['logical_error_rate'] > 0 else float('inf')
        print(f"\nRatio (correlated/regular): {ratio:.3f}")
        
        if ratio > 1.0:
            print(f"⚠️  Correlated matching is WORSE by {(ratio - 1.0) * 100:.1f}%")
        elif ratio < 1.0:
            print(f"✓ Correlated matching is BETTER by {(1.0 - ratio) * 100:.1f}%")
        else:
            print("= Same performance")
    else:
        print("Missing results for comparison")

    print()
    print("=" * 70)
