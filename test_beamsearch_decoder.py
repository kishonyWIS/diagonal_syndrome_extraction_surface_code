#!/usr/bin/env python3
"""
Quick test to compare beam search decoder vs tesseract on a single circuit.
"""
import sys
import time
import numpy as np
import stim

# Add the BeamSearchDecoder to path
sys.path.insert(0, '/Users/giladkishony/PycharmProjects/BeamSearchDecoder')

from spatial_hadamard_manual_construction import generate_spatial_hadamard_circuit
from tqec import NoiseModel
import tesseract_decoder.tesseract as tesseract

# Try to import beam search decoder
from beamsearch import BeamSearch

def test_decoder_speed(k=2, axis='y', flag_config='partial', noise_rate=0.001, num_shots=1000):
    """Compare decoder speeds on a single circuit configuration."""
    
    print(f"Testing with k={k}, axis={axis}, flag_config={flag_config}")
    print(f"Noise rate: {noise_rate}, Shots: {num_shots}")
    print("=" * 70)
    
    # Generate circuit
    circuit = generate_spatial_hadamard_circuit(
        k=k,
        axis=axis,
        flag_config=flag_config,
    )
    
    # Add noise
    noise_model = NoiseModel.uniform_depolarizing(noise_rate)
    noisy_circuit = noise_model.noisy_circuit(circuit)
    
    print(f"Circuit: {noisy_circuit.num_qubits} qubits, {noisy_circuit.num_detectors} detectors")
    
    # Get DEM
    dem = noisy_circuit.detector_error_model(decompose_errors=True)
    dem_full = noisy_circuit.detector_error_model(decompose_errors=False)
    
    # Sample shots
    sampler = noisy_circuit.compile_detector_sampler()
    shots, obs = sampler.sample(num_shots, separate_observables=True)
    
    print(f"Sampled {num_shots} shots")
    print()
    
    # Test Tesseract
    print("Testing Tesseract decoder...")
    tesseract_config = tesseract.TesseractConfig(
        dem=dem_full,
        pqlimit=200_000,
        det_beam=15,
        beam_climbing=True,
        det_orders=[],
        no_revisit_dets=True,
    )
    tesseract_decoder = tesseract.TesseractDecoder(tesseract_config)
    
    start_time = time.time()
    tesseract_predictions = tesseract_decoder.decode_batch(shots)
    tesseract_time = time.time() - start_time
    tesseract_errors = np.sum(tesseract_predictions != obs)
    
    print(f"  Total time: {tesseract_time:.3f}s")
    print(f"  Time per shot: {tesseract_time/num_shots*1000:.3f}ms")
    print(f"  Errors: {tesseract_errors}/{num_shots} ({tesseract_errors/num_shots*100:.1f}%)")
    print()
    
    # Test Beam Search decoder with different beam widths
    for beam_width in [8, 32]:
        print(f"Testing BeamSearch decoder (beam_width={beam_width})...")
        beamsearch_decoder = BeamSearch(
            model=dem,
            max_rounds=10,
            beam_width=beam_width,
            num_results=1,
            initial_iters=30,
            iters_per_round=20,
        )
        
        start_time = time.time()
        beamsearch_predictions = beamsearch_decoder.decode_batch(shots)
        beamsearch_time = time.time() - start_time
        beamsearch_errors = np.sum(beamsearch_predictions != obs)
        
        print(f"  Total time: {beamsearch_time:.3f}s")
        print(f"  Time per shot: {beamsearch_time/num_shots*1000:.3f}ms")
        print(f"  Errors: {beamsearch_errors}/{num_shots} ({beamsearch_errors/num_shots*100:.1f}%)")
        print(f"  Speedup vs Tesseract: {tesseract_time/beamsearch_time:.2f}x")
        print()

if __name__ == "__main__":
    for k in [2]:
        test_decoder_speed(k=k, num_shots=10000)
        print("\n" + "=" * 70 + "\n")
