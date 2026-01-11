#!/usr/bin/env python3
"""Benchmark script for patch rotation block graph structure.

This creates a 6-block staircase structure that will eventually have
some boundaries with different colors (different Pauli types X/Z).

Structure (z is temporal, going up):
z=3:          [1,0,3]
               |
z=2:  [0,0,2]-[1,0,2]
         |      |
z=1:  [0,0,1]-[1,0,1]
         |
z=0:  [0,0,0]

All blocks start as ZXZ type, but we'll modify boundaries later.
"""

import sys
import importlib

# Modify MEASUREMENT_SCHEDULE BEFORE importing any tqec modules
sys.path.insert(0, 'venv/lib/python3.13/site-packages')

# Change the constant before any imports
import tqec.plaquette.constants as constants
constants.MEASUREMENT_SCHEDULE = 8  # Allow schedule values up to 7

# Now import tqec modules
from tqec.computation.block_graph import BlockGraph
from tqec.computation.cube import ZXCube
from tqec.utils.position import Position3D
from tqec.compile.compile import compile_block_graph
from tqec.compile.convention import FIXED_BULK_CONVENTION
from tqec.utils.enums import Basis, Orientation

# Reload the translator module to pick up the new MEASUREMENT_SCHEDULE
import tqec.plaquette.rpng.translators.default as default_translator
importlib.reload(default_translator)
DefaultRPNGTranslator = default_translator.DefaultRPNGTranslator

# Import diagonal plaquette infrastructure from benchmark_memory
from benchmark_memory import (
    create_diagonal_convention,
    DiagonalCubeBuilder,
    DiagonalPipeBuilder,
    CustomIdentityPlaquetteCompiler,
)
from compact_circuit import compact_and_delay_init


def create_patch_rotation_block_graph():
    """Create a block graph for the patch rotation structure.
    
    Full 6-cube staircase structure.
    
    Structure (z is temporal direction):
    z=3:            [1,0,3]
                       |
    z=2:  [0,0,2]-[1,0,2]
             |
    z=1:  [0,0,1]-[1,0,1]
             |
    z=0:  [0,0,0]
    
    All blocks are ZXZ type.
    """
    graph = BlockGraph("Patch Rotation (6 cubes staircase)")
    
    # Define positions
    pos_000 = Position3D(0, 0, 0)
    pos_001 = Position3D(0, 0, 1)
    pos_101 = Position3D(1, 0, 1)
    pos_002 = Position3D(0, 0, 2)
    pos_102 = Position3D(1, 0, 2)
    pos_103 = Position3D(1, 0, 3)
    
    # Add cubes as ZXZ type
    cube_type = ZXCube.from_str("ZXZ")
    
    graph.add_cube(pos_000, cube_type, "block_000")
    graph.add_cube(pos_001, cube_type, "block_001")
    graph.add_cube(pos_101, cube_type, "block_101")
    graph.add_cube(pos_002, cube_type, "block_002")
    graph.add_cube(pos_102, cube_type, "block_102")
    graph.add_cube(pos_103, cube_type, "block_103")
    
    # Temporal pipes (z-direction)
    graph.add_pipe(pos_000, pos_001)  # (0,0,0) -> (0,0,1)
    graph.add_pipe(pos_001, pos_002)  # (0,0,1) -> (0,0,2)
    graph.add_pipe(pos_101, pos_102)  # (1,0,1) -> (1,0,2)
    graph.add_pipe(pos_102, pos_103)  # (1,0,2) -> (1,0,3)
    
    # Spatial pipes (x-direction)
    graph.add_pipe(pos_001, pos_101)  # (0,0,1) -> (1,0,1)
    graph.add_pipe(pos_002, pos_102)  # (0,0,2) -> (1,0,2)
    
    return graph


def create_full_patch_rotation_block_graph():
    """Create the full 6-cube block graph for the patch rotation structure.
    
    Structure (z is temporal direction):
    z=3:          [1,0,3]
                   |
    z=2:  [0,0,2]-[1,0,2]
             |      |
    z=1:  [0,0,1]-[1,0,1]
             |
    z=0:  [0,0,0]
    
    All blocks are ZXZ type (for now - we'll modify boundaries later).
    """
    graph = BlockGraph("Patch Rotation (Full)")
    
    # Define all 6 positions
    pos_000 = Position3D(0, 0, 0)
    pos_001 = Position3D(0, 0, 1)
    pos_101 = Position3D(1, 0, 1)
    pos_002 = Position3D(0, 0, 2)
    pos_102 = Position3D(1, 0, 2)
    pos_103 = Position3D(1, 0, 3)
    
    # Add all cubes as ZXZ type
    # (ZXX doesn't support x-direction pipes because both y and z faces are X)
    # ZXZ supports x-pipes (y=X, z=Z are different) and z-pipes (x=Z, y=X are different)
    cube_type = ZXCube.from_str("ZXZ")
    
    graph.add_cube(pos_000, cube_type, "block_000")
    graph.add_cube(pos_001, cube_type, "block_001")
    graph.add_cube(pos_101, cube_type, "block_101")
    graph.add_cube(pos_002, cube_type, "block_002")
    graph.add_cube(pos_102, cube_type, "block_102")
    graph.add_cube(pos_103, cube_type, "block_103")
    
    # Add pipes connecting adjacent blocks
    # Temporal pipes (z-direction)
    graph.add_pipe(pos_000, pos_001)  # (0,0,0) -> (0,0,1)
    graph.add_pipe(pos_001, pos_002)  # (0,0,1) -> (0,0,2)
    graph.add_pipe(pos_101, pos_102)  # (1,0,1) -> (1,0,2)
    graph.add_pipe(pos_102, pos_103)  # (1,0,2) -> (1,0,3)
    
    # Spatial pipes (x-direction)
    graph.add_pipe(pos_001, pos_101)  # (0,0,1) -> (1,0,1)
    graph.add_pipe(pos_002, pos_102)  # (0,0,2) -> (1,0,2)
    
    return graph


def compile_and_generate(graph, k=2, use_diagonal=True):
    """Compile the graph and generate a Stim circuit.
    
    Args:
        graph: The block graph to compile
        k: Scale factor
        use_diagonal: If True, use diagonal schedule convention. If False, use standard.
    """
    convention_name = "diagonal" if use_diagonal else "standard"
    print(f"\nCompiling patch rotation graph with {convention_name} convention (k={k})...")
    
    if use_diagonal:
        # Create diagonal convention
        convention = create_diagonal_convention()
    else:
        # Use standard TQEC convention
        from tqec.compile.convention import FIXED_BULK_CONVENTION
        convention = FIXED_BULK_CONVENTION
    
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
        
        # Generate crumble URLs
        print("\nGenerating Crumble URLs...")
        try:
            crumble_url_before = circuit_before.to_crumble_url()
            crumble_url_after = circuit.to_crumble_url()
            
            print(f"\n{'='*80}")
            print("CRUMBLE URL (before compaction):")
            print(f"{'='*80}")
            print(crumble_url_before)
            
            print(f"\n{'='*80}")
            print("CRUMBLE URL (after compaction):")
            print(f"{'='*80}")
            print(crumble_url_after)
            
        except Exception as e:
            print(f"  Warning: Could not generate Crumble URL: {e}")
        
        return circuit, circuit_before
        
    except Exception as e:
        print(f"✗ Failed to compile: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """Main entry point."""
    print("=" * 80)
    print("PATCH ROTATION BLOCK GRAPH")
    print("=" * 80)
    
    # Create the block graph
    print("\nCreating patch rotation block graph...")
    graph = create_patch_rotation_block_graph()
    
    print(f"  Created graph with {len(list(graph.cubes))} cubes")
    print(f"  Cube positions and types:")
    for cube in graph.cubes:
        print(f"    {cube.position}: {cube.kind}")
    
    print(f"\n  Pipes:")
    for pipe in graph.pipes:
        print(f"    {pipe.u.position} <-> {pipe.v.position} ({pipe.kind})")
    
    # Compile and generate circuit - first try standard convention to verify structure
    k = 2
    
    print("\n" + "-" * 40)
    print("Trying STANDARD convention first...")
    print("-" * 40)
    circuit_std, circuit_std_before = compile_and_generate(graph, k=k, use_diagonal=False)
    
    if circuit_std is not None:
        print("\n" + "=" * 80)
        print("SUCCESS with standard convention!")
        print("=" * 80)
    
    print("\n" + "-" * 40)
    print("Now trying DIAGONAL convention...")
    print("-" * 40)
    circuit_diag, circuit_diag_before = compile_and_generate(graph, k=k, use_diagonal=True)
    
    if circuit_diag is not None:
        print("\n" + "=" * 80)
        print("SUCCESS with diagonal convention!")
        print("=" * 80)


if __name__ == "__main__":
    main()
