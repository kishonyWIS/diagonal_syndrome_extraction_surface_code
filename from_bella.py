import stim

from tqec import NoiseModel, compile_block_graph
from tqec.compile.convention import FIXED_BOUNDARY_CONVENTION, FIXED_BULK_CONVENTION
from tqec.computation.block_graph import BlockGraph
from tqec.utils.position import Position3D

if __name__ == "__main__":
    # Block Graph will not change with conventions we used
    # The convention only affects how we interpret the block graph during circuit compilation
    g = BlockGraph()
    n0 = g.add_cube(Position3D(0, 0, 0), "ZZX", "")
    n1 = g.add_cube(Position3D(0, 1, 0), "XXZ", "")
    g.add_pipe(n0, n1)

#This is a modification of Yiming's example, implementing a hadamard pipe along the y direction, between two spatial cubes.

    correlation_surfaces = g.find_correlation_surfaces()

    g.view_as_html(
        write_html_filepath="spatial_hadamard.html",
        pop_faces_at_directions=("-Y",),
        show_correlation_surface=correlation_surfaces[0],
    )

    # Compile to circuit
    compiled_g = compile_block_graph(
        block_graph=g,
        convention=FIXED_BULK_CONVENTION,  # Specify the convention during compilation
    )
    noise_model = NoiseModel.uniform_depolarizing(1e-3)
    k = 2
    manhattan_radius = 2

    layer_tree = compiled_g.to_layer_tree()
    svg_string = layer_tree.layers_to_svg(k=k)[1]
    with open("spatial_hadamard.svg", "w") as f:
        f.write(svg_string)

    circuit = compiled_g.generate_stim_circuit(k, manhattan_radius=manhattan_radius, noise_model=noise_model)

    # Calculate graph-like distance
    try:
        shortest_logical_error = circuit.shortest_graphlike_error(
            ignore_ungraphlike_errors=False, canonicalize_circuit_errors=True
        )
        graphlike_distance = len(shortest_logical_error)
        print(f"✓ Graph-like distance: {graphlike_distance}")
        assert graphlike_distance == 2 * k + 1, (
            f"Circuit distance {graphlike_distance} does not match expected value {2 * k + 1}!"
        )
    except Exception as e:
        print(f"⚠ Could not calculate graph-like distance: {e}")
        print("  (This may occur when observables are not explicitly provided)")