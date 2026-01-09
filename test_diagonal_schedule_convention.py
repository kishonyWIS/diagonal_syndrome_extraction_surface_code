#!/usr/bin/env python3
"""Test the diagonal schedule convention."""

if __name__ == "__main__":
    import sys
    sys.path.insert(0, 'venv/lib/python3.13/site-packages')

    # Modify MEASUREMENT_SCHEDULE BEFORE importing any tqec modules
    import tqec.plaquette.constants as constants
    constants.MEASUREMENT_SCHEDULE = 8  # Allow schedule values up to 7

    # Reload the translator module to pick up the new MEASUREMENT_SCHEDULE
    import importlib
    import tqec.plaquette.rpng.translators.default as default_translator
    importlib.reload(default_translator)

    from tqec.gallery import memory
    from tqec import compile_block_graph, NoiseModel
    from tqec.compile.convention import DIAGONAL_SCHEDULE_CONVENTION, ALL_CONVENTIONS
    from tqec.utils.enums import Basis

    print("=" * 80)
    print("DIAGONAL SCHEDULE CONVENTION TEST")
    print("=" * 80)
    print()

    print(f"Available conventions: {list(ALL_CONVENTIONS.keys())}")
    print(f"Using convention: {DIAGONAL_SCHEDULE_CONVENTION}")
    print()

    # Test 1: Create a memory circuit
    print("Test 1: Creating memory circuit with diagonal schedule convention...")
    try:
        mem_graph = memory(Basis.Z)
        compiled = compile_block_graph(mem_graph, convention=DIAGONAL_SCHEDULE_CONVENTION)
        noise_model = NoiseModel.uniform_depolarizing(0.001)
        circuit = compiled.generate_stim_circuit(k=1, noise_model=noise_model)
        print(f"✓ Success! Circuit created with {len(circuit)} instructions")
        print(f"  Qubits: {circuit.num_qubits}")
        print(f"  Detectors: {circuit.num_detectors}")
        print(f"  Observables: {circuit.num_observables}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

    print()
    print("=" * 80)

