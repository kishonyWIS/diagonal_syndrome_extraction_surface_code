#!/usr/bin/env python3
"""
Debug script for testing the Bayesian flag decoder on the spatial Hadamard circuit.

This script loads a pre-generated circuit from file and tests:
1. Correlated PyMatching decoder
2. Bayesian Flag Decoder

Usage:
    1. First run: python spatial_hadamard_manual_construction.py
       (This generates spatial_hadamard_circuit.stim)
    2. Then run: python debug_bayesian_decoder.py
"""

import time
from collections import defaultdict
from typing import Set

import numpy as np
import pymatching
import stim

from spatial_hadamard_manual_construction import (
    load_circuit_from_file,
    get_flag_detector_indices_from_circuit,
)
from bayesian_flag_decoder import BayesianFlagDecoder


def load_circuit(filepath: str = "spatial_hadamard_circuit.stim") -> stim.Circuit:
    """Load circuit from file."""
    print(f"Loading circuit from {filepath}...")
    circuit = load_circuit_from_file(filepath)
    print(f"  Qubits: {circuit.num_qubits}")
    print(f"  Detectors: {circuit.num_detectors}")
    print(f"  Observables: {circuit.num_observables}")
    print(f"  Measurements: {circuit.num_measurements}")
    return circuit


def test_correlated_pymatching(
    circuit: stim.Circuit,
    shots: int = 10000,
    seed: int = 42,
) -> dict:
    """Test correlated PyMatching decoder."""
    print()
    print("=" * 70)
    print("Correlated PyMatching Decoder")
    print("=" * 70)
    
    # Build DEM
    dem = circuit.detector_error_model(decompose_errors=True, ignore_decomposition_failures=True)
    print(f"DEM: {dem.num_errors} errors, {dem.num_detectors} detectors")
    
    # Sample
    print(f"Sampling {shots} shots...")
    sampler = circuit.compile_detector_sampler(seed=seed)
    detector_samples, observable_samples = sampler.sample(shots, separate_observables=True)
    
    # Decode
    print("Decoding...")
    start_time = time.time()
    matching = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
    predicted = matching.decode_batch(detector_samples, enable_correlations=True)
    decode_time = time.time() - start_time
    
    # Count errors
    errors = np.sum(np.any(predicted != observable_samples, axis=1))
    error_rate = errors / shots
    
    print(f"  Decode time: {decode_time:.4f}s ({decode_time/shots*1000:.3f} ms/shot)")
    print(f"  Errors: {errors}/{shots}")
    print(f"  Error rate: {error_rate:.6f}")
    
    return {
        'errors': errors,
        'error_rate': error_rate,
        'decode_time': decode_time,
        'detector_samples': detector_samples,
        'observable_samples': observable_samples,
        'dem': dem,
        'matching': matching,
    }


def test_bayesian_decoder(
    circuit: stim.Circuit,
    detector_samples: np.ndarray,
    observable_samples: np.ndarray,
    enable_correlations: bool = False,
) -> dict:
    """Test Bayesian flag decoder.
    
    Args:
        circuit: The stim circuit
        detector_samples: Detector samples
        observable_samples: Observable samples
        enable_correlations: If True, use correlated matching internally
    """
    print()
    print("=" * 70)
    corr_str = " (with correlations)" if enable_correlations else ""
    print(f"Bayesian Flag Decoder{corr_str}")
    print("=" * 70)
    
    shots = detector_samples.shape[0]
    
    # Get flag detector indices from tagged detectors
    flag_detector_indices = get_flag_detector_indices_from_circuit(circuit)
    print(f"Flag detectors: {len(flag_detector_indices)} (from tags)")
    
    # Build full DEM (no decomposition) for flag grouping
    dem_full = circuit.detector_error_model(decompose_errors=False)
    print(f"Full DEM: {dem_full.num_errors} errors")
    
    # Build decomposed DEM if using correlations
    decomposed_dem = None
    if enable_correlations:
        decomposed_dem = circuit.detector_error_model(
            decompose_errors=True, ignore_decomposition_failures=True
        )
        print(f"Decomposed DEM: {decomposed_dem.num_errors} errors")
    
    # Create decoder
    print("Initializing Bayesian decoder...")
    start_time = time.time()
    bayesian_decoder = BayesianFlagDecoder(
        dem_full,
        flag_detectors=flag_detector_indices,
        enable_correlations=enable_correlations,
        decomposed_dem=decomposed_dem,
    )
    init_time = time.time() - start_time
    print(f"  Initialization: {init_time:.4f}s")
    print(f"  Correlations: {enable_correlations}")
    
    # Decode
    print(f"Decoding {shots} shots...")
    start_time = time.time()
    predicted = bayesian_decoder.decode_batch(detector_samples)
    decode_time = time.time() - start_time
    
    errors = np.sum(np.any(predicted != observable_samples, axis=1))
    error_rate = errors / shots
    
    print(f"  Decode time: {decode_time:.4f}s ({decode_time/shots*1000:.3f} ms/shot)")
    print(f"  Errors: {errors}/{shots}")
    print(f"  Error rate: {error_rate:.6f}")
    
    return {
        'errors': errors,
        'error_rate': error_rate,
        'init_time': init_time,
        'decode_time': decode_time,
        'bayesian_decoder': bayesian_decoder,
        'flag_detector_indices': flag_detector_indices,
        'dem_full': dem_full,
    }


def analyze_failures(
    circuit: stim.Circuit,
    detector_samples: np.ndarray,
    observable_samples: np.ndarray,
    pymatching_matching: pymatching.Matching,
    bayesian_decoder: BayesianFlagDecoder,
    flag_detector_indices: Set[int],
    max_detailed: int = 5,
) -> dict:
    """
    Analyze shots where Bayesian decoder fails but PyMatching succeeds.
    
    Args:
        circuit: The stim circuit
        detector_samples: Detector samples (shots x num_detectors)
        observable_samples: Observable samples (shots x num_observables)
        pymatching_matching: Pre-built PyMatching decoder
        bayesian_decoder: Pre-built Bayesian decoder
        flag_detector_indices: Set of flag detector indices
        max_detailed: Max number of failures to show in detail
    
    Returns:
        Dictionary with analysis results
    """
    print()
    print("=" * 70)
    print("Failure Analysis")
    print("=" * 70)
    
    shots = detector_samples.shape[0]
    
    # Track different failure categories
    bayesian_only_failures = []  # Bayesian wrong, PyMatching correct
    pymatching_only_failures = []  # PyMatching wrong, Bayesian correct
    both_fail = []  # Both wrong
    both_correct = 0
    
    # Statistics on flag involvement in ALL Bayesian failures
    all_bayesian_failures_with_flags = 0
    all_bayesian_failures_without_flags = 0
    # Statistics on flag involvement in Bayesian-only failures  
    bayesian_only_with_flags = 0
    bayesian_only_without_flags = 0
    flag_involvement_count = defaultdict(int)  # Which flags are involved in failures
    
    print(f"Analyzing {shots} shots...")
    
    for i in range(shots):
        sample = detector_samples[i]
        observable = observable_samples[i]
        
        # Get triggered detectors
        triggered = set(np.where(sample)[0])
        triggered_flags = triggered & flag_detector_indices
        triggered_core = triggered - flag_detector_indices
        
        # Decode with both
        pymatching_pred = pymatching_matching.decode(sample, enable_correlations=True)
        bayesian_pred = bayesian_decoder.decode(sample)
        
        pymatching_correct = np.array_equal(pymatching_pred, observable)
        bayesian_correct = np.array_equal(bayesian_pred, observable)
        
        if bayesian_correct and pymatching_correct:
            both_correct += 1
        elif not bayesian_correct and pymatching_correct:
            # Bayesian failed, PyMatching succeeded
            bayesian_only_failures.append({
                'shot_idx': i,
                'triggered_core': triggered_core,
                'triggered_flags': triggered_flags,
                'observable': observable.copy(),
                'pymatching_pred': pymatching_pred.copy(),
                'bayesian_pred': bayesian_pred.copy(),
            })
            # Track for Bayesian-only
            if triggered_flags:
                bayesian_only_with_flags += 1
            else:
                bayesian_only_without_flags += 1
            # Track for ALL Bayesian failures
            if triggered_flags:
                all_bayesian_failures_with_flags += 1
                for flag in triggered_flags:
                    flag_involvement_count[flag] += 1
            else:
                all_bayesian_failures_without_flags += 1
        elif bayesian_correct and not pymatching_correct:
            # Bayesian succeeded, PyMatching failed
            pymatching_only_failures.append(i)
        else:
            # Both failed - also a Bayesian failure!
            both_fail.append({'shot_idx': i, 'triggered_flags': triggered_flags})
            if triggered_flags:
                all_bayesian_failures_with_flags += 1
                for flag in triggered_flags:
                    flag_involvement_count[flag] += 1
            else:
                all_bayesian_failures_without_flags += 1
    
    # Print summary statistics
    print()
    print("-" * 70)
    print("Summary Statistics")
    print("-" * 70)
    print(f"Total shots: {shots}")
    print(f"Both correct: {both_correct}")
    print(f"Bayesian-only failures: {len(bayesian_only_failures)}")
    print(f"PyMatching-only failures: {len(pymatching_only_failures)}")
    print(f"Both failed: {len(both_fail)}")
    
    total_bayesian_failures = len(bayesian_only_failures) + len(both_fail)
    
    print()
    print("-" * 70)
    print("ALL Bayesian Failures (by flag involvement)")
    print("-" * 70)
    print(f"Total Bayesian failures: {total_bayesian_failures}")
    print(f"  WITH flags triggered: {all_bayesian_failures_with_flags}")
    print(f"  WITHOUT flags triggered: {all_bayesian_failures_without_flags}")
    
    print()
    print("-" * 70)
    print("Bayesian-Only Failures (by flag involvement)")
    print("-" * 70)
    print(f"Bayesian-only failures: {len(bayesian_only_failures)}")
    print(f"  WITH flags triggered: {bayesian_only_with_flags}")
    print(f"  WITHOUT flags triggered: {bayesian_only_without_flags}")
    
    if flag_involvement_count:
        print()
        print("Flag involvement frequency in failures:")
        for flag_idx in sorted(flag_involvement_count.keys()):
            count = flag_involvement_count[flag_idx]
            print(f"  Flag D{flag_idx}: {count} failures")
    
    # Show detailed analysis of first few failures
    if bayesian_only_failures:
        print()
        print("-" * 70)
        print(f"Detailed Analysis of First {min(max_detailed, len(bayesian_only_failures))} Bayesian Failures")
        print("-" * 70)
        
        for j, failure in enumerate(bayesian_only_failures[:max_detailed]):
            print()
            print(f"=== Failure {j+1}: Shot {failure['shot_idx']} ===")
            print(f"Triggered core detectors ({len(failure['triggered_core'])}): {sorted(failure['triggered_core'])}")
            print(f"Triggered flag detectors ({len(failure['triggered_flags'])}): {sorted(failure['triggered_flags'])}")
            print(f"True observable: {failure['observable']}")
            print(f"PyMatching prediction: {failure['pymatching_pred']} (CORRECT)")
            print(f"Bayesian prediction: {failure['bayesian_pred']} (WRONG)")
            
            # Get the reduced DEM used by Bayesian for this shot
            triggered_all = failure['triggered_core'] | failure['triggered_flags']
            reduced_dem = bayesian_decoder._create_matchable_dem_for_shot(triggered_all)
            
            print()
            print(f"Reduced DEM for this shot ({reduced_dem.num_errors} errors):")
            
            # Show errors that flip the observable
            errors_with_observable = []
            errors_without_observable = []
            for inst in reduced_dem:
                if inst.type == 'error':
                    inst_str = str(inst)
                    if 'L0' in inst_str:
                        errors_with_observable.append(inst_str)
                    else:
                        errors_without_observable.append(inst_str)
            
            print(f"  Errors WITH observable flip ({len(errors_with_observable)}):")
            for err in errors_with_observable[:10]:  # Show first 10
                print(f"    {err}")
            if len(errors_with_observable) > 10:
                print(f"    ... and {len(errors_with_observable) - 10} more")
            
            print(f"  Errors WITHOUT observable flip ({len(errors_without_observable)}):")
            # Just show count, too many to list
            
            # Check if there's a path in the reduced DEM that matches the syndrome
            print()
            print("  Checking reduced DEM decoding...")
            reduced_matching = pymatching.Matching.from_detector_error_model(reduced_dem)
            
            # Extract non-flag detector hits for reduced DEM
            non_flag_hits = np.zeros(len(bayesian_decoder.non_flag_detectors), dtype=bool)
            for k, det_idx in enumerate(bayesian_decoder.non_flag_detectors):
                if det_idx in triggered_all:
                    non_flag_hits[k] = True
            
            reduced_pred = reduced_matching.decode(non_flag_hits)
            print(f"  Reduced DEM decode result: {reduced_pred}")
    
    return {
        'bayesian_only_failures': len(bayesian_only_failures),
        'pymatching_only_failures': len(pymatching_only_failures),
        'both_fail': len(both_fail),
        'both_correct': both_correct,
        'total_bayesian_failures': total_bayesian_failures,
        'all_bayesian_with_flags': all_bayesian_failures_with_flags,
        'all_bayesian_without_flags': all_bayesian_failures_without_flags,
        'bayesian_only_with_flags': bayesian_only_with_flags,
        'bayesian_only_without_flags': bayesian_only_without_flags,
        'flag_involvement_count': dict(flag_involvement_count),
        'failure_details': bayesian_only_failures[:max_detailed],
    }


def main():
    """Main entry point."""
    print("=" * 70)
    print("Bayesian Flag Decoder Debug Script")
    print("=" * 70)
    
    # Load circuit
    circuit = load_circuit()
    
    # Test correlated PyMatching
    pymatching_results = test_correlated_pymatching(circuit, shots=10000, seed=42)
    
    # Test Bayesian decoder WITHOUT correlations (reuse samples)
    bayesian_results = test_bayesian_decoder(
        circuit,
        pymatching_results['detector_samples'],
        pymatching_results['observable_samples'],
        enable_correlations=False,
    )
    
    # Test Bayesian decoder WITH correlations (uses decomposed DEM)
    bayesian_corr_results = test_bayesian_decoder(
        circuit,
        pymatching_results['detector_samples'],
        pymatching_results['observable_samples'],
        enable_correlations=True,
    )
    
    # Analyze failures (for correlated Bayesian decoder)
    analysis_results = analyze_failures(
        circuit,
        pymatching_results['detector_samples'],
        pymatching_results['observable_samples'],
        pymatching_results['matching'],
        bayesian_corr_results['bayesian_decoder'],
        bayesian_corr_results['flag_detector_indices'],
        max_detailed=5,
    )
    
    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Correlated PyMatching:        {pymatching_results['error_rate']:.6f} error rate")
    print(f"Bayesian (no correlations):   {bayesian_results['error_rate']:.6f} error rate")
    print(f"Bayesian (with correlations): {bayesian_corr_results['error_rate']:.6f} error rate")
    print()
    print(f"Bayesian vs PyMatching:")
    print(f"  No correlations:   {bayesian_results['error_rate'] - pymatching_results['error_rate']:+.6f}")
    print(f"  With correlations: {bayesian_corr_results['error_rate'] - pymatching_results['error_rate']:+.6f}")
    print()
    print("Failure breakdown (Bayesian with correlations):")
    print(f"  ALL Bayesian failures: {analysis_results['total_bayesian_failures']}")
    print(f"    - With flags: {analysis_results['all_bayesian_with_flags']}")
    print(f"    - Without flags: {analysis_results['all_bayesian_without_flags']}")
    print(f"  Bayesian-only failures: {analysis_results['bayesian_only_failures']}")
    print(f"    - With flags: {analysis_results['bayesian_only_with_flags']}")
    print(f"    - Without flags: {analysis_results['bayesian_only_without_flags']}")
    print(f"  PyMatching-only failures: {analysis_results['pymatching_only_failures']}")


if __name__ == "__main__":
    main()
