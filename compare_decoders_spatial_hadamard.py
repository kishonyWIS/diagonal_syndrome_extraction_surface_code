#!/usr/bin/env python3
"""
Compare decoder performance on the Spatial Hadamard circuit.

Compares three decoders:
1. Plain pymatching - standard MWPM decoder with decomposed DEM
2. Tesseract - hypergraph decoder
3. Correlated pymatching - MWPM with full DEM
"""

import time
from typing import Optional
import numpy as np
import pymatching
import stim
import tesseract_decoder.tesseract as tesseract

from spatial_hadamard_manual_construction import (
    generate_two_cube_circuit,
    generate_two_cube_circuit_with_flag_detectors,
    get_flag_detector_indices_from_circuit,
)
from tqec import NoiseModel


def compare_decoders(
    k: int = 2,
    noise_level: float = 0.001,
    shots: int = 10000,
    seed: Optional[int] = None,
    use_flags: bool = True,
    measure_shared_data_final_only: bool = True,
) -> dict:
    """
    Compare decoder performance on the Spatial Hadamard circuit.
    
    Args:
        k: Scaling factor (code distance ≈ 2k+1)
        noise_level: Depolarizing noise level
        shots: Number of Monte Carlo shots
        seed: Random seed for reproducibility
        use_flags: If True, include flag measurements. If False, generate without flags.
        measure_shared_data_final_only: If True, only measure/reset shared data at first/last round.
        
    Returns:
        Dictionary with error rates and timing for each decoder
    """
    print("=" * 70)
    print("Decoder Comparison: Spatial Hadamard Circuit")
    print("=" * 70)
    print(f"k = {k} (code distance ≈ {2*k+1})")
    print(f"Noise level: {noise_level}")
    print(f"Shots: {shots}")
    print(f"Use flags: {use_flags}")
    print()
    
    # =========================================================================
    # Generate circuit
    # =========================================================================
    noise_model = NoiseModel.uniform_depolarizing(noise_level)
    flag_detector_indices = None
    
    if use_flags:
        print("Generating circuit with flag detectors (two-pass approach)...")
        circuit = generate_two_cube_circuit_with_flag_detectors(
            k=k,
            noise_model=noise_model,
            measure_shared_data_final_only=measure_shared_data_final_only,
        )
        # Extract flag detector indices from tagged detectors
        flag_detector_indices = get_flag_detector_indices_from_circuit(circuit)
        print(f"  Flag detectors: {len(flag_detector_indices)} detectors (tagged)")
    else:
        print("Generating circuit without flags...")
        circuit = generate_two_cube_circuit(
            k=k,
            noise_model=noise_model,
            measure_zxz_coupling_aux=False,
            measure_xzx_coupling_aux=True,
            measure_shared_data=False,
            measure_shared_data_final_only=measure_shared_data_final_only,
        )
    
    print(f"  Qubits: {circuit.num_qubits}")
    print(f"  Detectors: {circuit.num_detectors}")
    print(f"  Observables: {circuit.num_observables}")
    print(f"  Measurements: {circuit.num_measurements}")
    print()
    
    # =========================================================================
    # Build detector error models
    # =========================================================================
    print("Building detector error models...")
    
    # Full DEM with hyperedges (for tesseract and correlated matching)
    dem_full = circuit.detector_error_model(decompose_errors=False)
    print(f"  Full DEM: {dem_full.num_errors} errors, {dem_full.num_detectors} detectors")
    
    # Graphlike DEM (decomposed, for plain pymatching)
    dem_graphlike = circuit.detector_error_model(
        decompose_errors=True,
        ignore_decomposition_failures=True,
    )
    print(f"  Graphlike DEM: {dem_graphlike.num_errors} errors")
    print()
    
    # =========================================================================
    # Sample shots
    # =========================================================================
    print("Sampling shots...")
    sampler = circuit.compile_detector_sampler(seed=seed)
    sample_result = sampler.sample(shots, separate_observables=True)
    detector_samples = sample_result[0]  # Shape: (shots, num_detectors)
    observable_samples = sample_result[1]  # Shape: (shots, num_observables)
    print(f"  Detector samples shape: {detector_samples.shape}")
    print(f"  Observable samples shape: {observable_samples.shape}")
    print()
    
    results = {}

    # =========================================================================
    # Decoder 1: Plain pymatching
    # =========================================================================
    print("=" * 70)
    print("Decoder 1: Plain pymatching (graphlike DEM)")
    print("=" * 70)
    
    try:
        start_time = time.time()
        matching_plain = pymatching.Matching.from_detector_error_model(dem_graphlike)
        init_time = time.time() - start_time
        print(f"  Initialization: {init_time:.4f}s")
        
        start_time = time.time()
        predicted = matching_plain.decode_batch(detector_samples)
        decode_time = time.time() - start_time
        
        errors = np.sum(np.any(predicted != observable_samples, axis=1))
        error_rate = errors / shots
        
        print(f"  Decode time: {decode_time:.4f}s ({decode_time/shots*1000:.3f} ms/shot)")
        print(f"  Errors: {errors}/{shots}")
        print(f"  Error rate: {error_rate:.6f}")
        
        results['plain_pymatching'] = {
            'errors': int(errors),
            'error_rate': error_rate,
            'init_time': init_time,
            'decode_time': decode_time,
        }
    except Exception as e:
        print(f"  FAILED: {e}")
        results['plain_pymatching'] = {'error': str(e)}
    print()
    
    # =========================================================================
    # Decoder 2: Correlated pymatching (full DEM with enable_correlations)
    # =========================================================================
    print("=" * 70)
    print("Decoder 2: Correlated pymatching (full DEM)")
    print("=" * 70)
    
    try:
        start_time = time.time()
        matching_corr = pymatching.Matching.from_detector_error_model(dem_graphlike, enable_correlations=True)
        init_time = time.time() - start_time
        print(f"  Initialization: {init_time:.4f}s")
        
        start_time = time.time()
        predicted = matching_corr.decode_batch(detector_samples, enable_correlations=True)
        decode_time = time.time() - start_time
        
        errors = np.sum(np.any(predicted != observable_samples, axis=1))
        error_rate = errors / shots
        
        print(f"  Decode time: {decode_time:.4f}s ({decode_time/shots*1000:.3f} ms/shot)")
        print(f"  Errors: {errors}/{shots}")
        print(f"  Error rate: {error_rate:.6f}")
        
        results['correlated_pymatching'] = {
            'errors': int(errors),
            'error_rate': error_rate,
            'init_time': init_time,
            'decode_time': decode_time,
        }
    except Exception as e:
        print(f"  FAILED: {e}")
        results['correlated_pymatching'] = {'error': str(e)}
    print()
    
    # =========================================================================
    # Decoder 3: Tesseract (hypergraph decoder)
    # =========================================================================
    print("=" * 70)
    print("Decoder 3: Tesseract (hypergraph decoder)")
    print("=" * 70)
    
    try:
        start_time = time.time()
        tesseract_config = tesseract.TesseractConfig(
            dem=dem_full,
            pqlimit=200_000,
            det_beam=15,
            beam_climbing=True,
            det_orders=[],  # Empty list means single fixed ordering
            no_revisit_dets=True,
        )
        tesseract_decoder = tesseract.TesseractDecoder(tesseract_config)
        init_time = time.time() - start_time
        print(f"  Initialization: {init_time:.4f}s")
        
        start_time = time.time()
        predicted = tesseract_decoder.decode_batch(detector_samples)
        decode_time = time.time() - start_time
        
        errors = np.sum(np.any(predicted != observable_samples, axis=1))
        error_rate = errors / shots
        
        print(f"  Decode time: {decode_time:.4f}s ({decode_time/shots*1000:.3f} ms/shot)")
        print(f"  Errors: {errors}/{shots}")
        print(f"  Error rate: {error_rate:.6f}")
        
        results['tesseract'] = {
            'errors': int(errors),
            'error_rate': error_rate,
            'init_time': init_time,
            'decode_time': decode_time,
        }
    except Exception as e:
        print(f"  FAILED: {e}")
        results['tesseract'] = {'error': str(e)}
    print()
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Decoder':<25} {'Errors':>10} {'Error Rate':>15} {'Decode Time':>15}")
    print("-" * 70)
    
    decoder_names = [
        ('plain_pymatching', 'Plain PyMatching'),
        ('tesseract', 'Tesseract'),
        ('correlated_pymatching', 'Correlated PyMatching'),
    ]
    
    for key, name in decoder_names:
        if key in results:
            res = results[key]
            if 'error_rate' in res:
                print(f"{name:<25} {res['errors']:>10} {res['error_rate']:>15.6f} {res['decode_time']:>14.4f}s")
            else:
                print(f"{name:<25} {'FAILED':>10} {'-':>15} {'-':>15}")
    
    print("=" * 70)
    
    return results


def run_noise_sweep(
    k: int = 2,
    noise_levels: list[float] | None = None,
    shots: int = 10000,
    seed: Optional[int] = None,
    use_flags: bool = True,
    measure_shared_data_final_only: bool = True,
) -> dict:
    """
    Run decoder comparison across multiple noise levels.
    
    Args:
        k: Scaling factor
        noise_levels: List of noise levels to test
        shots: Shots per noise level
        seed: Random seed
        use_flags: If True, include flag measurements. If False, generate without flags.
        measure_shared_data_final_only: If True, only measure/reset shared data at first/last round.
        
    Returns:
        Dictionary mapping noise level to results
    """
    if noise_levels is None:
        noise_levels = [0.0001, 0.0005, 0.001, 0.002, 0.005]
    
    all_results = {}
    for noise in noise_levels:
        print(f"\n{'#' * 70}")
        print(f"# Noise level: {noise}")
        print(f"{'#' * 70}\n")
        all_results[noise] = compare_decoders(k=k, noise_level=noise, shots=shots, seed=seed, use_flags=use_flags, measure_shared_data_final_only=measure_shared_data_final_only)
    
    # Print summary table
    print("\n" + "=" * 70)
    print("NOISE SWEEP SUMMARY")
    print("=" * 70)
    print(f"{'Noise':<12} {'Plain':>12} {'Tesseract':>12} {'Correlated':>12}")
    print("-" * 70)
    
    for noise in noise_levels:
        res = all_results[noise]
        plain = res.get('plain_pymatching', {}).get('error_rate', float('nan'))
        tess = res.get('tesseract', {}).get('error_rate', float('nan'))
        corr = res.get('correlated_pymatching', {}).get('error_rate', float('nan'))
        print(f"{noise:<12.5f} {plain:>12.6f} {tess:>12.6f} {corr:>12.6f}")
    
    print("=" * 70)
    
    return all_results


if __name__ == "__main__":
    # Run single comparison at default settings
    # Set use_flags=False to compare circuit without flag measurements
    results = compare_decoders(k=2, noise_level=0.001, shots=10000, seed=42, use_flags=True, measure_shared_data_final_only=True)
