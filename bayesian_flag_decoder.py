#!/usr/bin/env python3
"""Bayesian flag decoder for detector-error-models with flagged matchable errors.

This module implements a stage-by-stage Bayesian update algorithm to convert
a detector-error-model with flagged errors into a matchable DEM that can be
used with pymatching, conditioning on observed flag detector outcomes.
"""

from typing import Dict, Set, List, Tuple, Optional
from collections import defaultdict
import stim
import numpy as np
import time


def _extract_detectors_from_instruction(inst: stim.DemInstruction) -> Set[int]:
    """Extract detector indices from a DEM instruction.
    
    Args:
        inst: A stim DemInstruction
        
    Returns:
        Set of detector indices
    """
    detectors = set()
    for target in inst.targets_copy():
        s = str(target)
        if s.startswith('D'):
            detectors.add(int(s[1:]))
    return detectors


def _extract_observables_from_instruction(inst: stim.DemInstruction) -> Set[int]:
    """Extract observable indices from a DEM instruction.
    
    Args:
        inst: A stim DemInstruction
        
    Returns:
        Set of observable indices
    """
    observables = set()
    for target in inst.targets_copy():
        s = str(target)
        if s.startswith('L'):
            observables.add(int(s[1:]))
    return observables


def _create_error_instruction(detectors: Set[int], probability: float, observables: Optional[Set[int]] = None) -> stim.DemInstruction:
    """Create a stim DemInstruction for an error.
    
    Args:
        detectors: Set of detector indices
        probability: Error probability
        observables: Optional set of observable indices
        
    Returns:
        A stim DemInstruction
    """
    targets = []
    for d in sorted(detectors):
        targets.append(stim.target_relative_detector_id(d))
    if observables:
        for o in sorted(observables):
            targets.append(stim.target_logical_observable_id(o))
    return stim.DemInstruction('error', [probability], targets)


def group_errors_by_flags(dem: stim.DetectorErrorModel, flag_detectors: Set[int]) -> Dict[Optional[int], List[stim.DemInstruction]]:
    """Stage 1: Group errors by which flag detector they trigger.
    
    Args:
        dem: The original detector-error-model
        flag_detectors: Set of detector indices that are flag detectors
        
    Returns:
        Dictionary mapping flag detector index (or None for no flag) to list of error instructions
    """
    groups: Dict[Optional[int], List[stim.DemInstruction]] = defaultdict(list)
    
    for inst in dem:
        if inst.type != 'error':
            # Skip non-error instructions (like shift_detectors, detector, etc.)
            continue
        
        detectors = _extract_detectors_from_instruction(inst)
        triggered_flags = detectors & flag_detectors
        
        if len(triggered_flags) == 0:
            # Error triggers no flags
            groups[None].append(inst)
        elif len(triggered_flags) == 1:
            # Error triggers exactly one flag
            flag_idx = next(iter(triggered_flags))
            groups[flag_idx].append(inst)
        else:
            # Error triggers multiple flags - this shouldn't happen in our model
            raise ValueError(
                f"Error triggers multiple flags: {triggered_flags}. "
                "Model assumes each error triggers at most one flag."
            )
    
    return dict(groups)


def add_non_flagged_errors(
    groups: Dict[Optional[int], List[stim.DemInstruction]],
    flag_detectors: Set[int]
) -> Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float]:
    """Stage 2: Add non-flagged errors to new DEM.
    
    Args:
        groups: Error groups from Stage 1
        flag_detectors: Set of flag detector indices
        
    Returns:
        Dictionary mapping (detector_tuple, observable_tuple) to probability
    """
    new_dem: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float] = {}
    
    # Process errors that trigger no flags (G_0)
    non_flagged_errors = groups.get(None, [])
    
    for inst in non_flagged_errors:
        detectors = _extract_detectors_from_instruction(inst)
        observables = _extract_observables_from_instruction(inst)
        non_flag_detectors = detectors - flag_detectors
        
        # Only include if it triggers ≤2 detectors (already verified no flags in Stage 1)
        if len(non_flag_detectors) <= 2:
            # Error is matchable and has no flag detectors
            detector_key = tuple(sorted(non_flag_detectors))
            observable_key = tuple(sorted(observables)) if observables else tuple()
            key = (detector_key, observable_key)
            prob = inst.args_copy()[0]
            
            # Increment probability if key already exists
            if key in new_dem:
                new_dem[key] += prob
            else:
                new_dem[key] = prob
    
    return new_dem


def process_triggered_flags(
    groups: Dict[Optional[int], List[stim.DemInstruction]],
    triggered_flags: Set[int],
    flag_detectors: Set[int],
    new_dem: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float]
) -> None:
    """Stage 3: Process triggered flags and add errors with normalized probabilities to new_dem.
    
    Args:
        groups: Error groups from Stage 1
        triggered_flags: Set of flag detector indices that were triggered in the shot
        flag_detectors: Set of all flag detector indices
        new_dem: Dictionary to update (modified in place)
    """
    for flag_idx in triggered_flags:
        if flag_idx not in groups:
            # No errors trigger this flag - skip
            continue
        
        # Get all errors that could have triggered this flag
        candidate_errors = groups[flag_idx]
        
        if len(candidate_errors) == 0:
            continue
        
        # Calculate normalization constant Z_{flag_i}
        z_flag = sum(inst.args_copy()[0] for inst in candidate_errors)
        
        if z_flag == 0:
            # All candidate errors have zero probability - skip
            continue
        
        # Create new errors for each candidate with flag removed
        for inst in candidate_errors:
            detectors = _extract_detectors_from_instruction(inst)
            observables = _extract_observables_from_instruction(inst)
            non_flag_detectors = detectors - flag_detectors
            
            # Skip if removing flag leaves more than 2 detectors
            if len(non_flag_detectors) > 2:
                continue
            
            # Create key from detectors and observables
            detector_key = tuple(sorted(non_flag_detectors))
            observable_key = tuple(sorted(observables)) if observables else tuple()
            key = (detector_key, observable_key)
            normalized_prob = inst.args_copy()[0] / z_flag
            
            # Increment probability in new_dem
            if key in new_dem:
                new_dem[key] += normalized_prob
            else:
                new_dem[key] = normalized_prob


def create_matchable_dem_from_flags(
    original_dem: stim.DetectorErrorModel,
    flag_detectors: Set[int],
    triggered_detectors: Set[int]
) -> stim.DetectorErrorModel:
    """Create a matchable DEM from an original DEM with flagged errors.
    
    This is the main entry point that implements all stages of the algorithm.
    
    Args:
        original_dem: Original detector-error-model with flagged errors
        flag_detectors: Set of detector indices that are flag detectors
        triggered_detectors: Set of all detectors triggered in the shot (including flags)
        
    Returns:
        New matchable stim.DetectorErrorModel (only non-flag detectors, errors trigger ≤2 detectors)
    """
    # Determine which flags were triggered
    triggered_flags = triggered_detectors & flag_detectors
    
    # Stage 1: Group errors by flags
    groups = group_errors_by_flags(original_dem, flag_detectors)
    
    # Stage 2: Add non-flagged errors
    new_dem = add_non_flagged_errors(groups, flag_detectors)
    
    # Stage 3: Process triggered flags (modifies new_dem in place)
    process_triggered_flags(groups, triggered_flags, flag_detectors, new_dem)
    
    # Stage 4: Merging is now done directly in Stages 2 and 3 (errors with same detectors AND observables are merged)
    # Stage 5: Non-triggered flags are automatically handled (not added in Stage 3)
    
    # Convert to stim.DetectorErrorModel
    instruction_lines = []
    for (detector_key, observable_key), prob in new_dem.items():
        if prob > 0 and len(detector_key) > 0:  # Only include errors with non-zero probability and at least one detector
            detectors = set(detector_key)
            observables = set(observable_key) if observable_key else None
            inst = _create_error_instruction(detectors, prob, observables)
            instruction_lines.append(str(inst))
    
    if instruction_lines:
        return stim.DetectorErrorModel('\n'.join(instruction_lines))
    else:
        return stim.DetectorErrorModel()


def example_usage():
    """Demonstrate the algorithm with the example from the plan."""
    # Original DEM from example
    original_dem = stim.DetectorErrorModel('''
        error(0.01) D0 D1
        error(0.005) D0 D1 D10
        error(0.003) D2 D10
        error(0.002) D3 D4 D11
    ''')
    
    flag_detectors = {10, 11}  # flag1=10, flag2=11
    
    # Shot outcome: {A, B, flag1} -> detectors {0, 1, 10}
    triggered_detectors = {0, 1, 10}
    
    # Create matchable DEM
    matchable_dem = create_matchable_dem_from_flags(
        original_dem,
        flag_detectors,
        triggered_detectors
    )
    
    print("Original DEM:")
    print(original_dem)
    
    print(f"\nFlag detectors: {flag_detectors}")
    print(f"Triggered detectors: {triggered_detectors}")
    print(f"Triggered flags: {triggered_detectors & flag_detectors}")
    
    print("\nMatchable DEM (after Bayesian update):")
    print(matchable_dem)
    
    # Verify expected results
    print("\nExpected results:")
    print("  Error with detectors {0, 1}: P ≈ 0.635")
    print("  Error with detectors {2}: P ≈ 0.375")
    
    # Check actual results
    for inst in matchable_dem:
        if inst.type == 'error':
            detectors = _extract_detectors_from_instruction(inst)
            prob = inst.args_copy()[0]
            if detectors == {0, 1}:
                print(f"\n✓ Found {{0, 1}} error with P={prob:.6f} (expected 0.635)")
            elif detectors == {2}:
                print(f"✓ Found {{2}} error with P={prob:.6f} (expected 0.375)")


def compare_decoder_performance(
    original_dem: stim.DetectorErrorModel,
    flag_detectors: Set[int],
    shots: int = 10000,
    seed: Optional[int] = None
) -> Dict[str, float]:
    """Compare logical error rates across multiple decoders.
    
    Compares:
    1. pymatching on original DEM
    2. tesseract decoder on original DEM (if available)
    3. pymatching on reduced DEM (Bayesian flag decoder)
    
    Args:
        original_dem: Original detector-error-model with flagged errors
        flag_detectors: Set of flag detector indices
        shots: Number of shots to test
        seed: Optional random seed
        
    Returns:
        Dictionary with error rates and statistics
    """
    try:
        import pymatching
        import tesseract_decoder.tesseract as tesseract
    except ImportError:
        print("Error: pymatching or tesseract_decoder not available. Cannot perform decoder comparison.")
        return {}
    
    print(f"\n{'='*70}")
    print(f"Comparing Decoder Performance")
    print(f"{'='*70}")
    print(f"Shots: {shots}")
    
    # Build matching objects
    print("\nBuilding decoders...")
    matching_original = pymatching.Matching.from_detector_error_model(original_dem)
    print("  ✓ Pymatching decoder initialized")
    
    tesseract_config = tesseract.TesseractConfig(
        dem=original_dem,
        pqlimit=200_000,
        det_beam=15,
        beam_climbing=True,
        det_orders=[],  # Empty list means single fixed ordering
        no_revisit_dets=True,
    )
    tesseract_decoder_obj = tesseract.TesseractDecoder(tesseract_config)

    # Sample shots from the original DEM
    print("Sampling shots from original DEM...")
    sampler = original_dem.compile_sampler(seed=seed)
    
    # Sample all shots at once
    sample_result = sampler.sample(shots)
    detector_samples = sample_result[0]  # Shape: (shots, num_detectors), dtype bool
    observable_samples = sample_result[1]  # Shape: (shots, num_observables) if num_observables > 0 else (shots, 0)
    
    print("Decoding shots (batch mode for original DEM and tesseract)...")
    
    # Batch decode with original DEM pymatching
    start_time = time.time()
    predicted_observables_original_batch = matching_original.decode_batch(detector_samples)
    time_original = time.time() - start_time
    
    # Batch decode with tesseract
    start_time = time.time()
    predicted_observables_tesseract_batch = tesseract_decoder_obj.decode_batch(detector_samples)
    time_tesseract = time.time() - start_time
    
    # Count errors for original and tesseract decoders
    errors_original = np.sum(np.any(predicted_observables_original_batch != observable_samples, axis=1))
    errors_tesseract = np.sum(np.any(predicted_observables_tesseract_batch != observable_samples, axis=1))
    
    # For reduced DEM, we still need to process shot-by-shot since each shot
    # may have a different reduced DEM (based on which flags were triggered)
    print("Decoding shots with reduced DEM (per-shot processing required)...")
    errors_reduced = 0
    start_time = time.time()
    
    for shot_idx in range(shots):
        if (shot_idx + 1) % 1000 == 0:
            print(f"  Processed {shot_idx + 1}/{shots} shots...")
        
        detector_hits = detector_samples[shot_idx]  # 1D array of shape (num_detectors,)
        observable_flips = observable_samples[shot_idx]
        triggered_detectors = set(np.where(detector_hits)[0].tolist())
        
        # Create reduced DEM for this shot
        reduced_dem = create_matchable_dem_from_flags(
            original_dem,
            flag_detectors,
            triggered_detectors
        )

        matching_reduced = pymatching.Matching.from_detector_error_model(reduced_dem)
        
        # Extract which detectors are actually used in the reduced DEM
        reduced_dem_detector_set = set()
        for inst in reduced_dem:
            if inst.type == 'error':
                reduced_dem_detector_set.update(_extract_detectors_from_instruction(inst))
        
        # Create mapping: original detector index -> reduced DEM index
        # pymatching internally renumbers detectors in order they appear
        reduced_dem_detector_list = sorted(reduced_dem_detector_set)
        detector_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(reduced_dem_detector_list)}
        
        # Create reduced detector hits array
        reduced_detector_hits = np.zeros(reduced_dem.num_detectors, dtype=bool)
        for orig_idx, new_idx in detector_map.items():
            if new_idx < len(reduced_detector_hits):
                reduced_detector_hits[new_idx] = detector_hits[orig_idx]
        
        predicted_observables_reduced = matching_reduced.decode(reduced_detector_hits)
        is_error_reduced = not np.array_equal(predicted_observables_reduced, observable_flips)
        if is_error_reduced:
            errors_reduced += 1
    
    time_reduced = time.time() - start_time
    
    # Calculate error rates
    error_rate_original = errors_original / shots
    error_rate_tesseract = errors_tesseract / shots
    error_rate_reduced = errors_reduced / shots
    
    print(f"\n{'='*70}")
    print("Results:")
    print(f"{'='*70}")
    print(f"1. Original DEM (pymatching):")
    print(f"   Errors: {errors_original}/{shots}")
    print(f"   Error rate: {error_rate_original:.6f}")
    print(f"   Time: {time_original:.4f} seconds ({time_original/shots*1000:.2f} ms/shot)")
    
    print(f"\n2. Original DEM (tesseract decoder):")
    print(f"   Errors: {errors_tesseract}/{shots}")
    print(f"   Error rate: {error_rate_tesseract:.6f}")
    print(f"   Time: {time_tesseract:.4f} seconds ({time_tesseract/shots*1000:.2f} ms/shot)")
    
    print(f"\n3. Reduced DEM (Bayesian flag decoder + pymatching):")
    print(f"   Errors: {errors_reduced}/{shots}")
    print(f"   Error rate: {error_rate_reduced:.6f}")
    print(f"   Time: {time_reduced:.4f} seconds ({time_reduced/shots*1000:.2f} ms/shot)")
    
    return {
        'shots': shots,
        'errors_original': errors_original,
        'errors_tesseract': errors_tesseract,
        'errors_reduced': errors_reduced,
        'error_rate_original': error_rate_original,
        'error_rate_tesseract': error_rate_tesseract,
        'error_rate_reduced': error_rate_reduced,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Example from Plan")
    print("=" * 70)
    print()
    example_usage()
    
    # Compare decoder performance
    print("\n\n")
    # Create the periodic boundary DEM for comparison
    L = 20
    p = 0.2
    flag_detectors = {L + i for i in range(L)}  # Set of flag detectors
    
    instructions = []
    for i in range(L):
        next_detector = (i + 1) % L
        detectors = {i, next_detector}
        observables = {0} if i == 0 else None
        instructions.append(_create_error_instruction(detectors, p, observables))
    
    for i in range(L):
        flag_detector = L + i  # flag_detector index
        observables = {0} if i == 0 or i == L - 1 else None
        instructions.append(_create_error_instruction({i, (i + 2) % L, flag_detector}, p, observables))
        # observables = {0} if i == 0 else None
        # instructions.append(_create_error_instruction({(i) % L, (i + 1) % L, flag_detector}, p, observables))
        # this is the hook error:
        observables = {0} if i == L - 1 else None
        instructions.append(_create_error_instruction({(i + 1) % L, (i + 2) % L, flag_detector}, p, observables))
        instructions.append(_create_error_instruction({flag_detector}, p))
    
    # Convert instructions to strings before joining
    instruction_lines = [str(inst) for inst in instructions]
    original_dem = stim.DetectorErrorModel('\n'.join(instruction_lines))
    
    compare_decoder_performance(original_dem, flag_detectors, shots=10000, seed=42)
