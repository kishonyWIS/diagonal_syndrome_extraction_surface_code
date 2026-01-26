#!/usr/bin/env python3
"""
Test if single physical errors cause logical errors in k=1 memory circuits with correlated_pymatching.

Iterates over all single errors from the full DEM and decodes using the decomposed DEM.
"""

import sys
sys.path.insert(0, 'venv/lib/python3.13/site-packages')

import tqec.plaquette.constants as constants
constants.MEASUREMENT_SCHEDULE = 8

from tqec import NoiseModel
import stim
import numpy as np
from benchmark_memory import create_original_memory_circuit, create_diagonal_memory_circuit

try:
    import pymatching
except ImportError:
    print("Error: pymatching not available")
    sys.exit(1)


def get_all_dem_errors(dem: stim.DetectorErrorModel):
    """Extract all error mechanisms from a DEM.
    
    Returns a list of (error_index, detectors, observables) tuples.
    """
    errors = []
    for instruction in dem:
        if instruction.type == 'error':
            detectors = []
            observables = []
            for target in instruction.targets_copy():
                if target.is_relative_detector_id():
                    detectors.append(target.val)
                elif target.is_logical_observable_id():
                    observables.append(target.val)
            errors.append((len(errors), detectors, observables))
    return errors


def get_detector_coordinates(circuit: stim.Circuit) -> dict[int, tuple]:
    """Extract detector coordinates from a circuit.
    
    Returns a dictionary mapping detector index to coordinates tuple.
    """
    detector_coords = {}
    detector_idx = 0
    
    for inst in circuit.flattened():
        if inst.name == 'DETECTOR':
            # Get coordinates from gate arguments
            coords = tuple(inst.gate_args_copy())
            detector_coords[detector_idx] = coords
            detector_idx += 1
    
    return detector_coords


def test_circuit(circuit, circuit_name, k=1, physical_error_rate=0.001):
    """
    Test a circuit by exhaustively checking all single physical errors.
    
    Tests both pymatching and correlated_pymatching decoders.
    For each single error from the full DEM:
    1. Get its detector and observable pattern
    2. Decode using the decomposed DEM with both decoders
    3. Check if it causes a logical error
    """
    print(f"\n{'='*60}")
    print(f"Testing {circuit_name} circuit (k={k})")
    print(f"{'='*60}")
    
    noise_model = NoiseModel.uniform_depolarizing(physical_error_rate)
    noisy_circuit = noise_model.noisy_circuit(circuit)
    
    # Create full DEM (not decomposed) - iterate over this
    full_dem = noisy_circuit.detector_error_model(decompose_errors=False)
    
    # Create decomposed DEM (for decoding)
    decomposed_dem = noisy_circuit.detector_error_model(
        decompose_errors=True,
        ignore_decomposition_failures=True
    )
    
    print(f"Extracting all error mechanisms from full DEM...")
    all_errors = get_all_dem_errors(full_dem)
    print(f"Found {len(all_errors)} error mechanisms in full DEM")
    print(f"Decomposed DEM has {decomposed_dem.num_detectors} detectors and {decomposed_dem.num_observables} observables")
    
    # Get detector coordinates from the circuit
    detector_coords = get_detector_coordinates(noisy_circuit)
    
    # Get qubit coordinates for each error using explain_detector_error_model_errors
    print(f"Getting qubit coordinates for each error...")
    explained_errors = noisy_circuit.explain_detector_error_model_errors(
        dem_filter=full_dem,
        reduce_to_one_representative_error=True
    )
    
    # Map error index to qubit information
    # We'll match by the detector pattern since each error has unique detectors
    error_to_qubits = {}
    error_to_coords = {}
    
    for explained_err in explained_errors:
        # Extract detector pattern from dem_error_terms to match with our error list
        dem_detectors = []
        for dem_target_with_coords in explained_err.dem_error_terms:
            dem_target = dem_target_with_coords.dem_target
            if dem_target.is_relative_detector_id():
                dem_detectors.append(dem_target.val)
        
        # Extract qubit information from circuit_error_locations
        qubits = set()
        coords_list = []
        for circuit_err_loc in explained_err.circuit_error_locations:
            # Get qubits from flipped_pauli_product (it's a list of GateTargetWithCoords)
            flipped_pauli = circuit_err_loc.flipped_pauli_product
            for target_with_coords in flipped_pauli:
                gate_target = target_with_coords.gate_target
                # GateTarget has a value property for qubit index
                if hasattr(gate_target, 'value'):
                    qubit_id = gate_target.value
                    qubits.add(qubit_id)
                    # Get coordinates if available
                    if hasattr(target_with_coords, 'coords') and target_with_coords.coords:
                        coords_list.append((qubit_id, tuple(target_with_coords.coords)))
        
        # Match this explained error to our error list by detector pattern
        for error_idx, (error_id, error_detectors, error_observables) in enumerate(all_errors):
            if set(error_detectors) == set(dem_detectors):
                error_to_qubits[error_id] = sorted(qubits)
                if coords_list:
                    error_to_coords[error_id] = coords_list
                break
    
    print(f"Extracted qubit information for {len(error_to_qubits)} errors")
    
    # Create decoders using the decomposed DEM
    matcher_pymatching = pymatching.Matching.from_detector_error_model(
        decomposed_dem, 
        enable_correlations=False
    )
    matcher_correlated = pymatching.Matching.from_detector_error_model(
        decomposed_dem, 
        enable_correlations=True
    )
    
    # Test each single error with both decoders
    print(f"\nTesting all {len(all_errors)} single errors with both decoders...")
    single_errors_pymatching = []
    single_errors_correlated = []
    
    batch_size = max(1, len(all_errors) // 100)
    
    for error_idx, (error_id, error_detectors, error_observables) in enumerate(all_errors):
        
        # Get detector and observable pattern for this error from full DEM
        detector_pattern = np.zeros(decomposed_dem.num_detectors, dtype=bool)
        actual_obs = np.zeros(circuit.num_observables, dtype=bool)
        
        for det in error_detectors:
            if det < decomposed_dem.num_detectors:
                detector_pattern[det] = True
        for obs in error_observables:
            if obs < circuit.num_observables:
                actual_obs[obs] = True
        
        # Decode using both decoders
        det_samples = detector_pattern.reshape(1, -1)
        
        # PyMatching (no correlations)
        predictions_pymatching = matcher_pymatching.decode_batch(det_samples, enable_correlations=False)
        predicted_obs_pymatching = predictions_pymatching[0]
        
        # Correlated PyMatching
        predictions_correlated = matcher_correlated.decode_batch(det_samples, enable_correlations=True)
        predicted_obs_correlated = predictions_correlated[0]
        
        # Check if logical error occurred with each decoder
        if not np.array_equal(actual_obs, predicted_obs_pymatching):
            qubits = error_to_qubits.get(error_id, [])
            coords = error_to_coords.get(error_id, [])
            single_errors_pymatching.append({
                'error_id': error_id,
                'detectors': error_detectors,
                'observables': error_observables,
                'qubits': qubits,
                'coords': coords,
                'actual_obs': actual_obs.copy(),
                'predicted_obs': predicted_obs_pymatching.copy(),
            })
        
        if not np.array_equal(actual_obs, predicted_obs_correlated):
            qubits = error_to_qubits.get(error_id, [])
            coords = error_to_coords.get(error_id, [])
            single_errors_correlated.append({
                'error_id': error_id,
                'detectors': error_detectors,
                'observables': error_observables,
                'qubits': qubits,
                'coords': coords,
                'actual_obs': actual_obs.copy(),
                'predicted_obs': predicted_obs_correlated.copy(),
            })
    
    # Report results for both decoders
    print(f"\nResults for PyMatching:")
    print(f"  Total single errors tested: {len(all_errors)}")
    print(f"  Single errors causing logical errors: {len(single_errors_pymatching)}")
    
    if len(single_errors_pymatching) > 0:
        pct = 100 * len(single_errors_pymatching) / len(all_errors)
        print(f"  Percentage: {pct:.4f}%")
        print(f"\n  WARNING: Found {len(single_errors_pymatching)} single error(s) that cause logical errors!")
        print(f"  This suggests the distance may be reduced by ignore_decomposition_failures=True")
        
        print(f"\n  First 10 examples:")
        for i, err_info in enumerate(single_errors_pymatching[:10], 1):
            print(f"    {i}. Error ID {err_info['error_id']}:")
            # Print detector coordinates instead of indices
            det_coords = [detector_coords.get(d, None) for d in err_info['detectors']]
            print(f"       Detector coordinates: {det_coords}")
            print(f"       Observables: {err_info['observables']}")
            if err_info['qubits']:
                print(f"       Qubits: {err_info['qubits']}")
            if err_info['coords']:
                print(f"       Qubit coordinates: {err_info['coords']}")
            print(f"       Actual obs: {err_info['actual_obs']}")
            print(f"       Predicted obs: {err_info['predicted_obs']}")
    else:
        print(f"\n  No single errors found that cause logical errors")
        print(f"  This suggests the distance is preserved (at least 2 physical errors needed)")
    
    print(f"\nResults for Correlated PyMatching:")
    print(f"  Total single errors tested: {len(all_errors)}")
    print(f"  Single errors causing logical errors: {len(single_errors_correlated)}")
    
    if len(single_errors_correlated) > 0:
        pct = 100 * len(single_errors_correlated) / len(all_errors)
        print(f"  Percentage: {pct:.4f}%")
        print(f"\n  WARNING: Found {len(single_errors_correlated)} single error(s) that cause logical errors!")
        print(f"  This suggests the distance may be reduced by ignore_decomposition_failures=True")
        
        print(f"\n  First 10 examples:")
        for i, err_info in enumerate(single_errors_correlated[:10], 1):
            print(f"    {i}. Error ID {err_info['error_id']}:")
            # Print detector coordinates instead of indices
            det_coords = [detector_coords.get(d, None) for d in err_info['detectors']]
            print(f"       Detector coordinates: {det_coords}")
            print(f"       Observables: {err_info['observables']}")
            if err_info['qubits']:
                print(f"       Qubits: {err_info['qubits']}")
            if err_info['coords']:
                print(f"       Qubit coordinates: {err_info['coords']}")
            print(f"       Actual obs: {err_info['actual_obs']}")
            print(f"       Predicted obs: {err_info['predicted_obs']}")
    else:
        print(f"\n  No single errors found that cause logical errors")
        print(f"  This suggests the distance is preserved (at least 2 physical errors needed)")


def main():
    k = 1
    physical_error_rate = 0.001
    
    print(f"Testing k={k} memory circuits with pymatching and correlated_pymatching")
    print(f"Exhaustively testing all single physical errors from full DEM")
    print(f"Decoding using decomposed DEM")
    print(f"\nUsing circuits BEFORE compact_and_delay_init (non-compacted)")
    
    print("\nCreating N/Z circuit (non-compacted)...")
    nz_circuit_before, _ = create_original_memory_circuit(k=k, return_both=True)
    test_circuit(nz_circuit_before, "N/Z (non-compacted)", k=k, physical_error_rate=physical_error_rate)
    
    print("\nCreating Diagonal circuit (non-compacted)...")
    diagonal_circuit_before, _ = create_diagonal_memory_circuit(k=k, return_both=True)
    test_circuit(diagonal_circuit_before, "Diagonal (non-compacted)", k=k, physical_error_rate=physical_error_rate)
    
    print(f"\n{'='*60}")
    print("Testing complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
