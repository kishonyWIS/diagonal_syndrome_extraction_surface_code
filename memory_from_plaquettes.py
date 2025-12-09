#!/usr/bin/env python3
"""
Example: Generate a memory experiment circuit directly from plaquettes.

This bypasses the BlockGraph abstraction and directly defines plaquette layouts
with custom RPNG descriptions to generate a stim circuit.
"""

import stim
from tqec import NoiseModel
from tqec.compile.blocks.block import Block
from tqec.compile.blocks.layers.atomic.layout import LayoutLayer
from tqec.compile.blocks.layers.atomic.plaquettes import PlaquetteLayer
from tqec.compile.blocks.layers.composed.repeated import RepeatedLayer
from tqec.compile.blocks.layers.composed.sequenced import SequencedLayers
from tqec.compile.blocks.positioning import LayoutCubePosition2D
from tqec.compile.generation import generate_circuit
from tqec.compile.observables.abstract_observable import AbstractObservable, CubeWithArms
from tqec.compile.observables.fixed_bulk_builder import FIXED_BULK_OBSERVABLE_BUILDER
from tqec.compile.tree.tree import LayerTree
from tqec.computation.cube import Cube, ZXCube
from tqec.plaquette.plaquette import Plaquette, Plaquettes
from tqec.plaquette.rpng import RPNGDescription
from tqec.plaquette.rpng.translators.default import DefaultRPNGTranslator
from tqec.post_processing.shift import shift_to_only_positive
from tqec.templates.qubit import QubitTemplate
from tqec.utils.frozendefaultdict import FrozenDefaultDict
from tqec.utils.position import Position3D
from tqec.utils.scale import LinearFunction, PhysicalQubitScalable2D


def create_translator():
    """Create the RPNG translator."""
    return DefaultRPNGTranslator()


def rpng_to_plaquette(rpng_desc: RPNGDescription, translator) -> Plaquette:
    """Convert an RPNG description to a Plaquette."""
    # Just use the translator - no need for additional compilation
    return translator.translate(rpng_desc)


def create_zxz_plaquettes(
    translator,
    reset: str | None = None,
    measurement: str | None = None,
) -> Plaquettes:
    """
    Create plaquettes for a ZXZ cube (Z-basis memory).
    
    ZXZ means: Z boundaries on left/right (x-axis), X boundaries on top/bottom (y-axis), stores Z.
    
    The QubitTemplate layout for k=2 looks like:
    
        1  5  6  5  6  2
        7  9 10  9 10 11
        8 10  9 10  9 12
        7  9 10  9 10 11
        8 10  9 10  9 12
        3 13 14 13 14  4
    
    Where:
    - 1,2,3,4: corners (empty)
    - 5,6: top boundary (5=X active, 6=empty)
    - 7,8: left boundary (7=empty, 8=Z active)
    - 9,10: bulk plaquettes (9=Z, 10=X)
    - 11,12: right boundary (11=Z active, 12=empty)
    - 13,14: bottom boundary (13=empty, 14=X active)
    
    Args:
        translator: RPNG translator
        reset: Reset basis ('x' or 'z') to apply to data qubits, or None
        measurement: Measurement basis ('x' or 'z') to apply to data qubits, or None
    """
    r = reset if reset else "-"
    m = measurement if measurement else "-"
    
    # Diagonal schedules for ZXZ cube:
    # X-basis: 1, 4, 3, 2 (positions 0, 1, 2, 3)
    # Z-basis: 6, 4, 3, 5 (positions 0, 1, 2, 3)
    
    rpng_descriptions = {
        # Corners (empty)
        1: RPNGDescription.empty(),
        2: RPNGDescription.empty(),
        3: RPNGDescription.empty(),
        4: RPNGDescription.empty(),
        
        # Top boundary - X-basis 2-body (positions 2,3 → timings 3,2)
        5: RPNGDescription.from_string(f"---- ---- {r}x3{m} {r}x2{m}"),
        6: RPNGDescription.empty(),
        
        # Left boundary - Z-basis 2-body (positions 1,3 → timings 4,5)
        7: RPNGDescription.empty(),
        8: RPNGDescription.from_string(f"---- {r}z4{m} ---- {r}z5{m}"),
        
        # Bulk plaquettes (diagonal schedules)
        9: RPNGDescription.from_string(f"{r}z6{m} {r}z4{m} {r}z3{m} {r}z5{m}"),   # Z-basis: 6,4,3,5
        10: RPNGDescription.from_string(f"{r}x1{m} {r}x4{m} {r}x3{m} {r}x2{m}"),  # X-basis: 1,4,3,2
        
        # Right boundary - Z-basis 2-body (positions 0,2 → timings 6,3)
        # original (before merging)
        # 11: RPNGDescription.from_string(f"{r}z6{m} ---- {r}z3{m} ----"),
        # 12: RPNGDescription.empty(),

        # new (after merging)
        11: RPNGDescription.from_string(f"{r}z6{m} ---- {r}z3{m} ----"),
        12: RPNGDescription.empty(),
        
        # Bottom boundary - X-basis 2-body (positions 0,1 → timings 1,4)
        13: RPNGDescription.empty(),
        14: RPNGDescription.from_string(f"{r}x1{m} {r}x4{m} ---- ----"),
    }
    
    plaquette_collection = {
        idx: rpng_to_plaquette(rpng, translator)
        for idx, rpng in rpng_descriptions.items()
    }
    
    empty_plaquette = rpng_to_plaquette(RPNGDescription.empty(), translator)
    return Plaquettes(FrozenDefaultDict(plaquette_collection, default_value=empty_plaquette))


def create_xzx_plaquettes(
    translator,
    reset: str | None = None,
    measurement: str | None = None,
) -> Plaquettes:
    """
    Create plaquettes for an XZX cube (X-basis memory).
    
    XZX means: X boundaries on left/right (x-axis), Z boundaries on top/bottom (y-axis), stores X.
    
    The QubitTemplate layout for k=2 looks like:
    
        1  5  6  5  6  2
        7  9 10  9 10 11
        8 10  9 10  9 12
        7  9 10  9 10 11
        8 10  9 10  9 12
        3 13 14 13 14  4
    
    Where:
    - 1,2,3,4: corners (empty)
    - 5,6: top boundary (5=empty, 6=Z active) - opposite of ZXZ
    - 7,8: left boundary (7=X active, 8=empty) - opposite of ZXZ
    - 9,10: bulk plaquettes (9=Z, 10=X) - same as ZXZ
    - 11,12: right boundary (11=empty, 12=X active) - opposite of ZXZ
    - 13,14: bottom boundary (13=Z active, 14=empty) - opposite of ZXZ
    
    Args:
        translator: RPNG translator
        reset: Reset basis ('x' or 'z') to apply to data qubits, or None
        measurement: Measurement basis ('x' or 'z') to apply to data qubits, or None
    """
    r = reset if reset else "-"
    m = measurement if measurement else "-"
    
    # Diagonal schedules for XZX cube (opposite of ZXZ):
    # Z-basis: 1, 4, 3, 2 (positions 0, 1, 2, 3)
    # X-basis: 6, 4, 3, 5 (positions 0, 1, 2, 3)
    
    rpng_descriptions = {
        # Corners (empty)
        1: RPNGDescription.empty(),
        2: RPNGDescription.empty(),
        3: RPNGDescription.empty(),
        4: RPNGDescription.empty(),
        
        # Top boundary - Z-basis 2-body (positions 2,3 → timings 3,2)
        5: RPNGDescription.empty(),
        6: RPNGDescription.from_string(f"---- ---- {r}z3{m} {r}z2{m}"),
        
        # Left boundary - X-basis 2-body (positions 1,3 → timings 4,5)
        7: RPNGDescription.from_string(f"---- {r}x4{m} ---- {r}x5{m}"),
        8: RPNGDescription.empty(),
        
        # Bulk plaquettes (diagonal schedules - opposite of ZXZ)
        9: RPNGDescription.from_string(f"{r}z1{m} {r}z4{m} {r}z3{m} {r}z2{m}"),   # Z-basis: 1,4,3,2
        10: RPNGDescription.from_string(f"{r}x6{m} {r}x4{m} {r}x3{m} {r}x5{m}"),  # X-basis: 6,4,3,5
        
        # Right boundary - X-basis 2-body (positions 0,2 → timings 6,3)
        11: RPNGDescription.empty(),
        12: RPNGDescription.from_string(f"{r}x6{m} ---- {r}x3{m} ----"),
        
        # Bottom boundary - Z-basis 2-body (positions 0,1 → timings 1,4)
        13: RPNGDescription.from_string(f"{r}z1{m} {r}z4{m} ---- ----"),
        14: RPNGDescription.empty(),
    }
    
    plaquette_collection = {
        idx: rpng_to_plaquette(rpng, translator)
        for idx, rpng in rpng_descriptions.items()
    }
    
    empty_plaquette = rpng_to_plaquette(RPNGDescription.empty(), translator)
    return Plaquettes(FrozenDefaultDict(plaquette_collection, default_value=empty_plaquette))


# Default scalable qubit shape (matches TQEC's _DEFAULT_SCALABLE_QUBIT_SHAPE)
DEFAULT_SCALABLE_QUBIT_SHAPE = PhysicalQubitScalable2D(
    LinearFunction(4, 5), LinearFunction(4, 5)
)


def plaquette_layers_to_layout_layer(
    plaquette_layers: dict[tuple[int, int], PlaquetteLayer],
    scalable_shape: PhysicalQubitScalable2D = DEFAULT_SCALABLE_QUBIT_SHAPE,
) -> LayoutLayer:
    """
    Wrap multiple PlaquetteLayers in a LayoutLayer at different positions.
    
    This is needed because LayerTree expects LayoutLayer as leaf nodes.
    
    Args:
        plaquette_layers: Dict mapping (x, y) block positions to PlaquetteLayers
        scalable_shape: The scalable qubit shape
        
    Returns:
        LayoutLayer containing the plaquette layers at their positions
    """
    layers_dict = {
        LayoutCubePosition2D(2 * x, 2 * y): layer
        for (x, y), layer in plaquette_layers.items()
    }
    return LayoutLayer(layers_dict, scalable_shape)


def create_multi_cube_layers(
    template: QubitTemplate,
    cube_plaquettes: dict[tuple[int, int], tuple[Plaquettes, Plaquettes, Plaquettes]],
    scalable_shape: PhysicalQubitScalable2D = DEFAULT_SCALABLE_QUBIT_SHAPE,
) -> SequencedLayers:
    """
    Create layers representing multiple cubes in a memory experiment.
    
    The layer structure is:
    - 1 initialization layer (with data qubit reset)
    - 2k+1 bulk layers (syndrome extraction only)
    - 1 measurement layer (with data qubit readout)
    
    Args:
        template: The QubitTemplate for each logical qubit
        cube_plaquettes: Dict mapping (x, y) block positions to 
                        (init_plaquettes, bulk_plaquettes, meas_plaquettes) tuples
        scalable_shape: The scalable qubit shape
        
    Returns:
        A SequencedLayers containing the layer structure with LayoutLayer leaves
    """
    # Create PlaquetteLayers for each cube and each layer type
    init_layers = {}
    bulk_layers = {}
    meas_layers = {}
    
    for pos, (init_plaq, bulk_plaq, meas_plaq) in cube_plaquettes.items():
        init_layers[pos] = PlaquetteLayer(template, init_plaq)
        bulk_layers[pos] = PlaquetteLayer(template, bulk_plaq)
        meas_layers[pos] = PlaquetteLayer(template, meas_plaq)
    
    # Wrap in LayoutLayers (combining all cubes at each time step)
    init_layout = plaquette_layers_to_layout_layer(init_layers, scalable_shape)
    bulk_layout = plaquette_layers_to_layout_layer(bulk_layers, scalable_shape)
    meas_layout = plaquette_layers_to_layout_layer(meas_layers, scalable_shape)
    
    # Create layer structure with RepeatedLayer for bulk
    layers = [
        init_layout,
        RepeatedLayer(
            bulk_layout,
            repetitions=LinearFunction(2, 1),  # 2k+1 repetitions
        ),
        meas_layout,
    ]
    return SequencedLayers(layers)


def create_single_cube_layers(
    template: QubitTemplate,
    init_plaquettes: Plaquettes,
    bulk_plaquettes: Plaquettes,
    meas_plaquettes: Plaquettes,
    scalable_shape: PhysicalQubitScalable2D = DEFAULT_SCALABLE_QUBIT_SHAPE,
) -> SequencedLayers:
    """
    Create layers representing a single cube memory experiment.
    
    Convenience function that wraps create_multi_cube_layers for a single cube at (0, 0).
    """
    return create_multi_cube_layers(
        template,
        {(0, 0): (init_plaquettes, bulk_plaquettes, meas_plaquettes)},
        scalable_shape,
    )


def create_memory_observable(
    cube_type: str = "ZXZ",
    position: tuple[int, int, int] = (0, 0, 0),
) -> AbstractObservable:
    """
    Create an AbstractObservable for a memory experiment.
    
    Args:
        cube_type: Cube type string (e.g., "ZXZ" or "XZX")
        position: (x, y, z) position of the cube
        
    Returns:
        AbstractObservable specifying data qubit readouts at the top of the block
    """
    cube = Cube(Position3D(*position), ZXCube.from_str(cube_type))
    return AbstractObservable(
        top_readout_cubes=frozenset([CubeWithArms(cube)])
    )


def create_multi_cube_observable(
    cubes: list[tuple[str, tuple[int, int, int]]],
) -> list[AbstractObservable]:
    """
    Create AbstractObservables for multiple cubes.
    
    Args:
        cubes: List of (cube_type, (x, y, z)) tuples
        
    Returns:
        List of AbstractObservables, one for each cube
    """
    return [create_memory_observable(cube_type, pos) for cube_type, pos in cubes]


def create_layer_tree(
    layers: SequencedLayers,
    abstract_observables: list[AbstractObservable] | None = None,
) -> LayerTree:
    """
    Create a LayerTree from layers for automatic detector computation.
    
    Args:
        layers: The SequencedLayers containing the layer structure
        abstract_observables: Optional list of observable specifications
        
    Returns:
        LayerTree that can generate circuits with detectors
    """
    # Wrap the layers in another SequencedLayers (represents time blocks)
    root = SequencedLayers([layers])
    
    return LayerTree(
        root=root,
        observable_builder=FIXED_BULK_OBSERVABLE_BUILDER,
        abstract_observables=abstract_observables,
    )


def generate_memory_circuit(
    k: int = 2,
    noise_model: NoiseModel | None = None,
) -> stim.Circuit:
    """
    Generate a complete Z-basis memory experiment circuit using LayerTree.
    
    This uses TQEC's LayerTree infrastructure for automatic detector computation.
    
    Args:
        k: Scaling factor (code distance ≈ 2k+1)
        noise_model: Optional noise model to apply to the circuit
        
    Returns:
        Complete stim circuit with detectors and observables
    """
    translator = create_translator()
    template = QubitTemplate()
    
    # Create plaquettes for ZXZ cube (Z-basis memory)
    init_plaquettes = create_zxz_plaquettes(translator, reset="z")
    bulk_plaquettes = create_zxz_plaquettes(translator)
    meas_plaquettes = create_zxz_plaquettes(translator, measurement="z")
    
    # Create the layer structure with LayoutLayer leaves
    layers = create_single_cube_layers(template, init_plaquettes, bulk_plaquettes, meas_plaquettes)
    
    # Create the observable for the memory experiment
    abstract_observable = create_memory_observable("ZXZ", (0, 0, 0))
    
    # Create the LayerTree for automatic detector computation
    layer_tree = create_layer_tree(layers, [abstract_observable])
    
    # Generate the circuit with detectors
    circuit = layer_tree.generate_circuit(k, manhattan_radius=2)
    
    # Apply noise model if provided
    if noise_model is not None:
        circuit = noise_model.noisy_circuit(circuit)
    
    return circuit


def generate_two_cube_circuit(
    k: int = 2,
    noise_model: NoiseModel | None = None,
) -> stim.Circuit:
    """
    Generate a circuit with two decoupled cubes:
    - ZXZ cube at position (0, 0, 0) - stores Z
    - XZX cube at position (1, 0, 0) - stores X
    
    The cubes are decoupled (no pipe connecting them).
    
    Args:
        k: Scaling factor (code distance ≈ 2k+1)
        noise_model: Optional noise model to apply to the circuit
        
    Returns:
        Complete stim circuit with detectors and observables for both cubes
    """
    translator = create_translator()
    template = QubitTemplate()
    
    # Create plaquettes for ZXZ cube at (0, 0) - stores Z
    zxz_init = create_zxz_plaquettes(translator, reset="z")
    zxz_bulk = create_zxz_plaquettes(translator)
    zxz_meas = create_zxz_plaquettes(translator, measurement="z")
    
    # Create plaquettes for XZX cube at (1, 0) - stores X
    xzx_init = create_xzx_plaquettes(translator, reset="x")
    xzx_bulk = create_xzx_plaquettes(translator)
    xzx_meas = create_xzx_plaquettes(translator, measurement="x")
    
    # Create cube plaquettes dict: position -> (init, bulk, meas)
    cube_plaquettes = {
        (0, 0): (zxz_init, zxz_bulk, zxz_meas),  # ZXZ at origin
        (1, 0): (xzx_init, xzx_bulk, xzx_meas),  # XZX to the right
    }
    
    # Create the layer structure
    layers = create_multi_cube_layers(template, cube_plaquettes)
    
    # Create observables for both cubes
    observables = create_multi_cube_observable([
        ("ZXZ", (0, 0, 0)),  # Observable for ZXZ cube
        ("XZX", (1, 0, 0)),  # Observable for XZX cube
    ])
    
    # Create the LayerTree for automatic detector computation
    layer_tree = create_layer_tree(layers, observables)
    
    # Generate the circuit with detectors
    circuit = layer_tree.generate_circuit(k, manhattan_radius=2)
    
    # Apply noise model if provided
    if noise_model is not None:
        circuit = noise_model.noisy_circuit(circuit)
    
    return circuit


def generate_blockgraph_memory_circuit(
    k: int = 2,
    noise_model: NoiseModel | None = None,
) -> stim.Circuit:
    """Generate a memory circuit using the BlockGraph approach."""
    from tqec.gallery import memory
    from tqec import compile_block_graph
    from tqec.compile.convention import FIXED_BULK_CONVENTION
    from tqec.utils.enums import Basis
    
    mem_graph = memory(Basis.Z)
    compiled = compile_block_graph(mem_graph, convention=FIXED_BULK_CONVENTION)
    return compiled.generate_stim_circuit(k=k, noise_model=noise_model)


def calculate_graphlike_distance(circuit: stim.Circuit) -> int | None:
    """Calculate the graph-like distance of a circuit."""
    try:
        shortest_error = circuit.shortest_graphlike_error(canonicalize_circuit_errors=True)
        return len(shortest_error)
    except Exception as e:
        print(f"Error calculating graph-like distance: {e}")
        return None


def calculate_circuit_distance(circuit: stim.Circuit) -> int | None:
    """Calculate the circuit-level distance (undetectable logical errors)."""
    try:
        logical_errors = circuit.search_for_undetectable_logical_errors(
            dont_explore_detection_event_sets_with_size_above=4,
            dont_explore_edges_with_degree_above=9999,
            dont_explore_edges_increasing_symptom_degree=False,
            canonicalize_circuit_errors=True
        )
        return len(logical_errors)
    except Exception as e:
        print(f"Error calculating circuit distance: {e}")
        return None


def main():
    """Generate and compare circuits with one and two cubes."""
    k = 2
    noise_level = 0.001
    
    print("=" * 70)
    print("Two-Cube Circuit: ZXZ at (0,0,0) + XZX at (1,0,0) (Decoupled)")
    print("=" * 70)
    print(f"k = {k} (surface code distance ≈ {2*k+1})")
    print(f"Noise level: {noise_level}")
    print()
    
    # Create noise model
    noise_model = NoiseModel.uniform_depolarizing(noise_level)
    
    # Generate single-cube circuit for comparison
    print("Generating single-cube circuit (ZXZ only)...")
    single_cube_circuit = generate_memory_circuit(k=k, noise_model=noise_model)
    
    # Generate two-cube circuit
    print("Generating two-cube circuit (ZXZ + XZX)...")
    two_cube_circuit = generate_two_cube_circuit(k=k, noise_model=noise_model)
    
    print()
    print("=" * 70)
    print("Circuit Statistics")
    print("=" * 70)
    print(f"Single-cube circuit: {len(single_cube_circuit)} instructions")
    print(f"Two-cube circuit:    {len(two_cube_circuit)} instructions")
    
    print()
    print("=" * 70)
    print("Distance Calculations")
    print("=" * 70)
    
    # Calculate graph-like distances
    print("Calculating graph-like distances...")
    single_graphlike = calculate_graphlike_distance(single_cube_circuit)
    two_cube_graphlike = calculate_graphlike_distance(two_cube_circuit)
    
    print(f"Single-cube circuit: graph-like distance = {single_graphlike}")
    print(f"Two-cube circuit:    graph-like distance = {two_cube_graphlike}")
    
    # Calculate circuit-level distances
    print()
    print("Calculating circuit-level distances...")
    single_circuit_dist = calculate_circuit_distance(single_cube_circuit)
    two_cube_circuit_dist = calculate_circuit_distance(two_cube_circuit)
    
    print(f"Single-cube circuit: circuit distance = {single_circuit_dist}")
    print(f"Two-cube circuit:    circuit distance = {two_cube_circuit_dist}")
    
    print()
    print("=" * 70)
    print("Crumble URLs")
    print("=" * 70)
    print("Single-cube circuit (ZXZ):")
    print(shift_to_only_positive(single_cube_circuit).to_crumble_url())
    print()
    print("Two-cube circuit (ZXZ + XZX):")
    print(shift_to_only_positive(two_cube_circuit).to_crumble_url())


if __name__ == "__main__":
    main()

