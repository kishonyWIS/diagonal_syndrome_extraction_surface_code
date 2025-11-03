#!/usr/bin/env python3
"""Bayesian flag decoder for detector-error-models with flagged matchable errors.

This module implements a stage-by-stage Bayesian update algorithm to convert
a detector-error-model with flagged errors into a matchable DEM that can be
used with pymatching, conditioning on observed flag detector outcomes.
"""

from typing import Dict, Set, List, Tuple, Optional
from collections import defaultdict
import pymatching
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


def _dem_dict_to_stim_dem(
    dem_dict: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float],
    ensure_detectors: Optional[Set[int]] = None
) -> stim.DetectorErrorModel:
    """Convert a DEM dictionary to a stim.DetectorErrorModel.
    
    Args:
        dem_dict: Dictionary mapping (detector_key, observable_key) to probability
        ensure_detectors: Optional set of detector indices to ensure are included
                        (by adding zero-probability boundary edges)
        
    Returns:
        stim.DetectorErrorModel representation
    """
    instruction_lines = []
    detectors_in_errors = set()
    
    for (detector_key, observable_key), prob in dem_dict.items():
        if prob > 0 and len(detector_key) > 0:
            detectors = set(detector_key)
            detectors_in_errors.update(detectors)
            observables = set(observable_key) if observable_key else None
            inst = _create_error_instruction(detectors, prob, observables)
            instruction_lines.append(str(inst))
    
    # Ensure all required detectors are included (add zero-probability boundary edges)
    if ensure_detectors is not None:
        missing_detectors = ensure_detectors - detectors_in_errors
        for det in sorted(missing_detectors):
            # Add a very low probability boundary edge so the detector is included
            instruction_lines.append(f'error(1e-20) D{det}')
    
    if instruction_lines:
        return stim.DetectorErrorModel('\n'.join(instruction_lines))
    else:
        return stim.DetectorErrorModel()


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


def extract_flag_detectors_from_tags(dem: stim.DetectorErrorModel, tag: str = 'flag') -> Set[int]:
    """Extract flag detector indices from a DEM based on stim tags.
    
    Looks for detector instructions tagged with the specified tag. Only detector
    instructions (not error instructions) are used to identify flag detectors.
    Tagged error instructions indicate that those errors involve flag detectors,
    but the specific flag detector IDs come from tagged detector instructions.
    
    Args:
        dem: The detector-error-model
        tag: Tag name to search for (default: 'flag')
        
    Returns:
        Set of detector indices that are flagged (based on tags)
    """
    flag_detectors = set()
    
    for inst in dem:
        inst_str = str(inst)
        
        # Check if instruction has the tag
        tag_str = f'[{tag}]'
        if tag_str in inst_str:
            if inst.type == 'detector':
                # Detector instruction with tag - extract detector ID(s)
                detectors = _extract_detectors_from_instruction(inst)
                flag_detectors.update(detectors)
            # Note: Tagged error instructions indicate the error involves flags,
            # but we don't extract flag detector IDs from them - only from detector instructions
    
    return flag_detectors


class BayesianFlagDecoder:
    """Fast wrapper for pymatching that performs Bayesian flag decoding.
    
    Precomputes error groups and base DEM during initialization, then efficiently
    creates matchable DEMs per-shot by copying the base DEM and adding conditional errors.
    """
    
    def __init__(
        self, 
        original_dem: stim.DetectorErrorModel, 
        flag_detectors: Optional[Set[int]] = None,
        flag_detector_tag: Optional[str] = None
    ):
        """Initialize the Bayesian flag decoder.
        
        Precomputes:
        - Error groups (Stage 1)
        - Base DEM dictionary (Stage 2)
        - Flag conditional errors and normalization constants
        
        Args:
            original_dem: Original detector-error-model with flagged errors
            flag_detectors: Set of flag detector indices (optional, if flag_detector_tag is provided)
            flag_detector_tag: Tag name to use for extracting flag detectors from DEM (optional)
        
        Raises:
            ValueError: If neither flag_detectors nor flag_detector_tag is provided
        """
        self.original_dem = original_dem
        
        # Determine flag detectors
        if flag_detector_tag is not None:
            # Extract from tags
            self.flag_detectors = extract_flag_detectors_from_tags(original_dem, flag_detector_tag)
            if not self.flag_detectors:
                raise ValueError(f"No flag detectors found with tag '{flag_detector_tag}' in DEM")
        elif flag_detectors is not None:
            # Use provided set
            self.flag_detectors = flag_detectors
        else:
            raise ValueError("Must provide either flag_detectors or flag_detector_tag")
        
        # Stage 1: Precompute error groups
        self.groups = group_errors_by_flags(original_dem, self.flag_detectors)
        
        # Stage 2: Precompute base DEM dictionary (non-flag errors)
        self.base_dem_dict = add_non_flagged_errors(self.groups, self.flag_detectors)
        
        # Stage 3 preparation: Precompute conditional errors for each flag
        # For each flag, store: list of (detector_key, observable_key, normalized_prob) tuples
        # (normalized probabilities are precomputed here)
        self.flag_conditional_errors: Dict[int, List[Tuple[Tuple[int, ...], Tuple[int, ...], float]]] = {}
        
        for flag_idx in self.flag_detectors:
            if flag_idx not in self.groups:
                continue
            
            candidate_errors = self.groups[flag_idx]
            if len(candidate_errors) == 0:
                continue
            
            # Calculate normalization constant
            z_flag = sum(inst.args_copy()[0] for inst in candidate_errors)
            if z_flag == 0:
                continue
            
            # Precompute conditional errors for this flag with normalized probabilities
            conditional_errors = []
            for inst in candidate_errors:
                detectors = _extract_detectors_from_instruction(inst)
                observables = _extract_observables_from_instruction(inst)
                non_flag_detectors = detectors - self.flag_detectors
                
                # Skip if removing flag leaves more than 2 detectors or empty
                if len(non_flag_detectors) == 0 or len(non_flag_detectors) > 2:
                    continue
                
                detector_key = tuple(sorted(non_flag_detectors))
                observable_key = tuple(sorted(observables)) if observables else tuple()
                base_prob = inst.args_copy()[0]
                normalized_prob = base_prob / z_flag  # Precompute normalized probability
                
                conditional_errors.append((detector_key, observable_key, normalized_prob))
            
            self.flag_conditional_errors[flag_idx] = conditional_errors
        
        # Precompute the set of all non-flag detectors that can appear in reduced DEMs
        # This is fixed across all shots since all errors use non-flag detectors
        all_possible_detectors = set()
        for (detector_key, _), _ in self.base_dem_dict.items():
            all_possible_detectors.update(detector_key)
        for flag_idx in self.flag_detectors:
            if flag_idx in self.flag_conditional_errors:
                for detector_key, _, _ in self.flag_conditional_errors[flag_idx]:
                    all_possible_detectors.update(detector_key)
        
        # Precompute sorted list for numpy indexing
        self.non_flag_detectors = sorted(all_possible_detectors)
        self.non_flag_detectors_array = np.array(self.non_flag_detectors, dtype=np.int64)
    
    def _create_matchable_dem_for_shot(
        self, triggered_detectors: Set[int]
    ) -> stim.DetectorErrorModel:
        """Create a matchable DEM for a specific shot.
        
        Args:
            triggered_detectors: Set of all detectors triggered in the shot (including flags)
            
        Returns:
            Matchable stim.DetectorErrorModel for this shot
        """
        # Copy base DEM dictionary
        dem_dict = dict(self.base_dem_dict)
        
        # Identify triggered flags
        triggered_flags = triggered_detectors & self.flag_detectors
        
        # Add conditional errors for each triggered flag
        for flag_idx in triggered_flags:
            if flag_idx not in self.flag_conditional_errors:
                continue
            
            # Add each conditional error with precomputed normalized probability
            for detector_key, observable_key, normalized_prob in self.flag_conditional_errors[flag_idx]:
                key = (detector_key, observable_key)
                
                # Increment probability in dem_dict
                if key in dem_dict:
                    dem_dict[key] += normalized_prob
                else:
                    dem_dict[key] = normalized_prob
        
        # Convert to stim.DetectorErrorModel
        # Ensure all possible non-flag detectors are included so detector indexing is consistent
        # (Even if they don't appear in errors for this shot, add zero-prob boundary edges)
        return _dem_dict_to_stim_dem(dem_dict, ensure_detectors=set(self.non_flag_detectors))
    
    def decode(self, detector_hits: np.ndarray) -> np.ndarray:
        """Decode a single shot.
        
        Args:
            detector_hits: 1D numpy array of booleans representing detector outcomes
            
        Returns:
            1D numpy array of booleans representing predicted observable flips
        """
        
        # Ensure detector_hits is boolean
        if detector_hits.dtype != bool:
            detector_hits = detector_hits.astype(bool)
        
        triggered_detectors = set(np.where(detector_hits)[0].tolist())
        
        # Create matchable DEM for this shot
        reduced_dem = self._create_matchable_dem_for_shot(triggered_detectors)
        
        # If reduced DEM is empty, return empty observable array
        if len(reduced_dem) == 0:
            # Try to infer number of observables from original DEM
            num_observables = self.original_dem.num_observables
            return np.zeros(num_observables, dtype=bool)
        
        # Create decoder
        matching = pymatching.Matching.from_detector_error_model(reduced_dem)
        
        # If no detectors in reduced DEM, return empty observable array
        if len(self.non_flag_detectors) == 0:
            num_observables = self.original_dem.num_observables
            return np.zeros(num_observables, dtype=bool)
        
        # Extract reduced detector hits using simple numpy indexing
        # reduced_detector_hits[i] = detector_hits[non_flag_detectors[i]]
        reduced_detector_hits = detector_hits[self.non_flag_detectors_array]
        
        # Decode
        predicted_observables = matching.decode(reduced_detector_hits)
        
        return predicted_observables
    
    def decode_batch(self, detector_hits_batch: np.ndarray) -> np.ndarray:
        """Decode a batch of shots.
        
        Args:
            detector_hits_batch: 2D numpy array of booleans, shape (num_shots, num_detectors)
            
        Returns:
            2D numpy array of booleans, shape (num_shots, num_observables)
        """
        num_shots = detector_hits_batch.shape[0]
        results = []
        
        for shot_idx in range(num_shots):
            result = self.decode(detector_hits_batch[shot_idx])
            results.append(result)
        
        return np.array(results)


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
    
    # Create decoder and get matchable DEM for this shot
    decoder = BayesianFlagDecoder(original_dem, flag_detectors=flag_detectors)
    matchable_dem = decoder._create_matchable_dem_for_shot(triggered_detectors)
    
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
    flag_detectors: Optional[Set[int]] = None,
    shots: int = 10000,
    seed: Optional[int] = None,
    flag_detector_tag: Optional[str] = None
) -> Dict[str, float]:
    """Compare logical error rates across multiple decoders.
    
    Compares:
    1. pymatching on original DEM
    2. tesseract decoder on original DEM (if available)
    3. pymatching on reduced DEM (Bayesian flag decoder)
    
    Args:
        original_dem: Original detector-error-model with flagged errors
        flag_detectors: Set of flag detector indices (optional, if flag_detector_tag is provided)
        shots: Number of shots to test
        seed: Optional random seed
        flag_detector_tag: Tag name to use for extracting flag detectors from DEM (optional)
        
    Returns:
        Dictionary with error rates and statistics
    
    Raises:
        ValueError: If neither flag_detectors nor flag_detector_tag is provided
    """
    # Determine flag detectors
    if flag_detector_tag is not None:
        flag_detectors_set = extract_flag_detectors_from_tags(original_dem, flag_detector_tag)
        if not flag_detectors_set:
            raise ValueError(f"No flag detectors found with tag '{flag_detector_tag}' in DEM")
    elif flag_detectors is not None:
        flag_detectors_set = flag_detectors
    else:
        raise ValueError("Must provide either flag_detectors or flag_detector_tag")
    
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
    
    # For reduced DEM, use the optimized BayesianFlagDecoder wrapper
    print("Decoding shots with reduced DEM (using optimized wrapper)...")
    print("  Building BayesianFlagDecoder wrapper...")
    start_time = time.time()
    if flag_detector_tag is not None:
        bayesian_decoder = BayesianFlagDecoder(original_dem, flag_detector_tag=flag_detector_tag)
    else:
        bayesian_decoder = BayesianFlagDecoder(original_dem, flag_detectors=flag_detectors_set)
    init_time = time.time() - start_time
    print(f"  Initialization took {init_time:.4f} seconds")
    
    errors_reduced = 0
    start_time = time.time()
    
    # Use batch decode
    predicted_observables_reduced_batch = bayesian_decoder.decode_batch(detector_samples.astype(bool))
    
    # Count errors
    errors_reduced = np.sum(np.any(predicted_observables_reduced_batch.astype(bool) != observable_samples, axis=1))
    
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
    print(f"   (Includes initialization: {init_time:.4f} seconds)")
    
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
        # Tag only the detector instruction for flag detectors
        instructions.append(stim.DemInstruction('detector', [], [stim.target_relative_detector_id(flag_detector)], tag='flag'))
    
    # Convert instructions to strings before joining
    instruction_lines = [str(inst) for inst in instructions]
    original_dem = stim.DetectorErrorModel('\n'.join(instruction_lines))
    
    compare_decoder_performance(original_dem, flag_detector_tag='flag', shots=10000, seed=42)
