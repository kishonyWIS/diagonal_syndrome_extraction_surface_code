#!/usr/bin/env python3
"""Compare standard vs diagonal schedule X junction implementations."""

import sys
sys.path.insert(0, 'venv/lib/python3.13/site-packages')

from tqec.computation.block_graph import BlockGraph
from tqec.computation.cube import ZXCube
from tqec.utils.position import Position3D
from tqec.compile.compile import compile_block_graph
from tqec.compile.convention import FIXED_BULK_CONVENTION
from tqec.utils.noise_model import NoiseModel
from tqec.utils.enums import Basis

# Modify MEASUREMENT_SCHEDULE BEFORE importing any tqec modules
import tqec.plaquette.constants as constants
constants.MEASUREMENT_SCHEDULE = 7  # Allow schedule index 6

# Reload the translator module to pick up the new MEASUREMENT_SCHEDULE
import importlib
import tqec.plaquette.rpng.translators.default as default_translator
importlib.reload(default_translator)


def create_x_junction_block_graph():
    """Create a block graph for an X junction with central block and 4 neighbors."""
    
    # Create a new block graph
    graph = BlockGraph("X Junction")
    
    # Define positions
    central_pos = Position3D(1, 1, 0)
    north_pos = Position3D(1, 0, 0)
    south_pos = Position3D(1, 2, 0)
    west_pos = Position3D(0, 1, 0)
    east_pos = Position3D(2, 1, 0)
    
    
    graph.add_cube(central_pos, ZXCube.from_str("XXZ"), "central")
    graph.add_cube(north_pos, ZXCube.from_str("XZZ"), "north")
    graph.add_cube(south_pos, ZXCube.from_str("XZZ"), "south")
    graph.add_cube(west_pos, ZXCube.from_str("ZXZ"), "west")
    graph.add_cube(east_pos, ZXCube.from_str("ZXZ"), "east")
    
    # Add pipes connecting central to each neighbor
    graph.add_pipe(central_pos, north_pos)   # Y direction
    graph.add_pipe(central_pos, south_pos)   # Y direction  
    graph.add_pipe(central_pos, west_pos)    # X direction
    graph.add_pipe(central_pos, east_pos)    # X direction
    
    return graph


def replace_rpng_with_diagonal(rpng_desc):
    """Replace standard RPNG schedule with diagonal schedule."""
    # Convert to string
    desc_str = str(rpng_desc)
    
    # Diagonal schedules: Z=[6,4,3,5], X=[1,4,3,2]
    
    # Handle Z plaquettes: change -z1- to -z6- (keeping others the same)
    if 'z' in desc_str and '-z1-' in desc_str:
        desc_str = desc_str.replace('-z1-', '-z6-')
    
    # Handle X plaquettes: change from [1,2,3,5] to [1,4,3,2]
    # Pattern: "-x1- -x2- -x3- -x5-" -> "-x1- -x4- -x3- -x2-"
    if 'x' in desc_str:
        parts = desc_str.split()
        if len(parts) == 4 and '-x1-' in parts[0] and '-x2-' in parts[1] and '-x3-' in parts[2] and '-x5-' in parts[3]:
            # 4-body X plaquette with [1,2,3,5] schedule
            desc_str = parts[0] + ' ' + parts[1].replace('-x2-', '-x4-') + ' ' + parts[2] + ' ' + parts[3].replace('-x5-', '-x2-')
        # Handle boundary X plaquettes (may have ---- entries)
        if '-x2-' in desc_str and '-x5-' in desc_str:
            # Check if this is a 4-body pattern
            x_count = desc_str.count('x')
            if x_count >= 3:  # Some entries might be ---- 
                # Replace -x2- with -x4-
                desc_str = desc_str.replace('-x2-', '-x4-', 1)
                # Replace last -x5- with -x2-
                desc_str = desc_str.rsplit('-x5-', 1)[0] + '-x2-' + desc_str.rsplit('-x5-', 1)[1] if '-x5-' in desc_str else desc_str
    
    # For other cases (like [1,4,3,5] for Z), just change -z1- to -z6-
    # This handles the case where Z already uses [1,4,3,5] except needs -z6- instead
    
    from tqec.plaquette.rpng import RPNGDescription
    return RPNGDescription.from_string(desc_str)


def replace_plaquettes_in_graph(compiled_graph):
    """Walk through the compiled graph and replace RPNG descriptions with diagonal versions."""
    print("Walking through layer tree to replace RPNG descriptions...")
    
    # For now, since accessing and modifying the internal structure is complex,
    # let's just note that the plaquettes would be replaced here
    # In practice, this would require deeper access to the compiled structure
    
    print("  Note: RPNG replacement would happen here in a full implementation")
    print("  For now, skipping and using diagonal convention directly")
    
    return compiled_graph


def compile_and_generate(graph, convention_name, convention, use_diagonal=False):
    """Compile the graph and generate a Stim circuit."""
    print(f"\nCompiling with {convention_name} convention...")
    
    try:
        compiled_graph = compile_block_graph(
            block_graph=graph,
            convention=convention
        )
        print(f"✓ Successfully compiled block graph")
        
        # If use_diagonal, replace RPNG descriptions
        if use_diagonal:
            compiled_graph = replace_plaquettes_in_graph(compiled_graph)
        
        # Generate the Stim circuit with k=2 and detectors
        k = 1
        manhattan_radius = 2
        
        # Create a noise model (uniform depolarizing noise with p=0.001)
        noise_model = NoiseModel.uniform_depolarizing(0.001)
        
        circuit = compiled_graph.generate_stim_circuit(
            k=k, 
            manhattan_radius=manhattan_radius,
            noise_model=noise_model
        )
        print(f"✓ Successfully generated Stim circuit for k={k}")
        print(f"  Number of instructions: {len(circuit)}")
        print(f"  Number of qubits: {circuit.num_qubits}")
        print(f"  Number of detectors: {circuit.num_detectors}")
        print(f"  Number of observables: {circuit.num_observables}")
        
        # Generate crumble URL
        print("\nGenerating Crumble URL...")
        crumble_url = compiled_graph.generate_crumble_url(k=k)
        print(f"✓ Crumble URL generated")
        
        # Calculate graph-like distance
        print("\nCalculating graph-like distance...")
        try:
            graphlike_errors = circuit.shortest_graphlike_error(canonicalize_circuit_errors=True)
            graphlike_distance = len(graphlike_errors)
            print(f"✓ Graph-like distance: {graphlike_distance}")
        except Exception as e:
            print(f"✗ Error calculating graph-like distance: {e}")
            import traceback
            traceback.print_exc()
            graphlike_distance = None
        
        return {
            'circuit': circuit,
            'crumble_url': crumble_url,
            'graphlike_distance': graphlike_distance,
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


def compare_results(standard, diagonal):
    """Compare and print results side-by-side."""
    print("\n" + "=" * 60)
    print("X-Junction Comparison: Standard vs Diagonal Schedule")
    print("=" * 60)
    print()
    
    if standard and diagonal:
        print(f"{'Metric':<30} {'Standard':<20} {'Diagonal':<20}")
        print("-" * 70)
        print(f"{'Graph-like distance':<30} {standard.get('graphlike_distance', 'N/A'):<20} {diagonal.get('graphlike_distance', 'N/A'):<20}")
        print(f"{'Number of instructions':<30} {standard.get('num_instructions', 'N/A'):<20} {diagonal.get('num_instructions', 'N/A'):<20}")
        print(f"{'Number of qubits':<30} {standard.get('num_qubits', 'N/A'):<20} {diagonal.get('num_qubits', 'N/A'):<20}")
        print(f"{'Number of detectors':<30} {standard.get('num_detectors', 'N/A'):<20} {diagonal.get('num_detectors', 'N/A'):<20}")
        print(f"{'Number of observables':<30} {standard.get('num_observables', 'N/A'):<20} {diagonal.get('num_observables', 'N/A'):<20}")
        print()
        
        print("Standard Crumble URL:")
        print(standard.get('crumble_url', 'N/A'))
        print()
        
        print("Diagonal Crumble URL:")
        print(diagonal.get('crumble_url', 'N/A'))
    else:
        print("✗ Failed to generate one or both circuits")


if __name__ == "__main__":
    print("Creating X junction block graph...")
    print("=" * 60)
    
    # Create the block graph
    graph = create_x_junction_block_graph()
    
    if graph is None:
        sys.exit(1)
    
    # Display graph information
    print("\nBlock Graph Information:")
    print(f"Name: {graph.name}")
    print(f"Number of cubes: {graph.num_cubes}")
    print(f"Number of pipes: {graph.num_pipes}")
    print(f"Occupied positions: {graph.occupied_positions}")
    
    # List cubes
    print("\nCubes:")
    for cube in graph.cubes:
        print(f"  {cube.label}: {cube.kind} at {cube.position}")
    
    # List pipes
    print("\nPipes:")
    for pipe in graph.pipes:
        print(f"  {pipe.u.position} <-> {pipe.v.position} ({pipe.direction})")
    
    # Compile with standard convention
    print("\n" + "=" * 60)
    standard_result = compile_and_generate(graph, "Standard Fixed-Bulk", FIXED_BULK_CONVENTION, use_diagonal=False)
    
    # For diagonal, use the diagonal convention we created
    print("\n" + "=" * 60)
    print("\nCompiling with Diagonal Schedule convention...")
    from complete_comparison import create_diagonal_convention
    diagonal_convention = create_diagonal_convention()
    diagonal_result = compile_and_generate(graph, "Diagonal Schedule", diagonal_convention, use_diagonal=False)
    
    # Compare results
    compare_results(standard_result, diagonal_result)
    
    print("\n" + "=" * 60)
    print("✓ Comparison complete!")
    print("=" * 60)

