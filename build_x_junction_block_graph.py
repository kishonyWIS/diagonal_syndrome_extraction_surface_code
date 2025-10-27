#!/usr/bin/env python3
"""Create and compile an X junction block graph."""

import sys
sys.path.insert(0, 'venv/lib/python3.13/site-packages')

from tqec.computation.block_graph import BlockGraph
from tqec.computation.cube import ZXCube, Port
from tqec.utils.position import Position3D
from tqec.compile.compile import compile_block_graph
from tqec.compile.convention import FIXED_BULK_CONVENTION
from tqec.utils.noise_model import NoiseModel

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
    # The pipes will be automatically determined based on the cubes
    graph.add_pipe(central_pos, north_pos)   # Y direction
    graph.add_pipe(central_pos, south_pos)   # Y direction  
    graph.add_pipe(central_pos, west_pos)    # X direction
    graph.add_pipe(central_pos, east_pos)    # X direction
    
    # Validate the graph
    print("Validating block graph...")
    try:
        graph.validate()
        print("✓ Block graph is valid")
    except Exception as e:
        print(f"✗ Validation error: {e}")
        return None
    
    return graph

def compile_and_generate(graph):
    """Compile the graph and generate a Stim circuit."""
    print("\nCompiling block graph...")
    print(f"Convention: {FIXED_BULK_CONVENTION.name}")
    
    try:
        compiled_graph = compile_block_graph(
            block_graph=graph,
            convention=FIXED_BULK_CONVENTION
        )
        print("✓ Successfully compiled block graph")
        
        # Generate the Stim circuit with k=2 and detectors
        print("\nGenerating Stim circuit...")
        k = 2
        manhattan_radius = 2
        
        # Create a noise model (uniform depolarizing noise with p=0.001)
        noise_model = NoiseModel.uniform_depolarizing(0.001)
        
        circuit = compiled_graph.generate_stim_circuit(
            k=k, 
            manhattan_radius=manhattan_radius,
            noise_model=noise_model
        )
        print(f"✓ Successfully generated Stim circuit for k={k}")
        print(f"Number of instructions: {len(circuit)}")
        print(f"Number of qubits: {circuit.num_qubits}")
        print(f"Number of detectors: {circuit.num_detectors}")
        print(f"Number of observables: {circuit.num_observables}")
        
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
        
        return circuit, crumble_url, graphlike_distance
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

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
    
    # Compile the graph and generate circuit
    print("\n" + "=" * 60)
    result = compile_and_generate(graph)
    
    if result:
        circuit, crumble_url, graphlike_distance = result
        print("\n" + "=" * 60)
        print("\n✓ X Junction successfully compiled and converted to Stim circuit!")
        
        if graphlike_distance is not None:
            print(f"\n📏 Graph-like distance: {graphlike_distance}")
        
        print("\n🔗 Crumble URL (click to visualize the circuit):")
        print(crumble_url)
        print("\nFirst 20 instructions:")
        for i, instruction in enumerate(circuit[:20]):
            print(f"  {i+1}: {instruction}")

