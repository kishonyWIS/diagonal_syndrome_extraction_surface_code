#!/usr/bin/env python3
"""
Manual construction of patch rotation circuit with asymmetric boundaries.

Structure (z is temporal direction):
z=3:          [1,0,3]  <- XZZ-like (±x=X, ±y=Z), measure Z
               |
z=2:  [0,0,2]-[1,0,2]  <- (0,0,2) has +y=Z override, measure X at (0,0,2)
         |      |
z=1:  [0,0,1]-[1,0,1]  <- (1,0,1) has +x=X, -y=Z overrides, reset Z at (1,0,1)
         |
z=0:  [0,0,0]          <- standard ZXZ, reset Z

Each z-level has 2k+1 layers (1 init + 2k-1 bulk + 1 meas).
Total: 4×(2k+1) layers.
"""

# Set MEASUREMENT_SCHEDULE BEFORE importing any tqec modules
import tqec.plaquette.constants as constants
constants.MEASUREMENT_SCHEDULE = 8

import importlib
import tqec.plaquette.rpng.translators.default as default_translator
importlib.reload(default_translator)

import stim
from tqec.compile.blocks.layers.atomic.layout import LayoutLayer
from tqec.compile.blocks.layers.atomic.plaquettes import PlaquetteLayer
from tqec.compile.blocks.layers.composed.repeated import RepeatedLayer
from tqec.compile.blocks.layers.composed.sequenced import SequencedLayers
from tqec.compile.blocks.positioning import LayoutCubePosition2D
from tqec.compile.observables.abstract_observable import AbstractObservable, CubeWithArms
from tqec.compile.observables.fixed_bulk_builder import FIXED_BULK_OBSERVABLE_BUILDER
from tqec.compile.tree.tree import LayerTree
from tqec.computation.cube import Cube, ZXCube
from tqec.plaquette.plaquette import Plaquettes
from tqec.plaquette.rpng import RPNGDescription
from tqec.plaquette.rpng.translators.default import DefaultRPNGTranslator
from tqec.templates.qubit import QubitTemplate
from tqec.utils.frozendefaultdict import FrozenDefaultDict
from tqec.utils.position import Position3D
from tqec.utils.scale import LinearFunction, PhysicalQubitScalable2D


# Default scalable qubit shape
DEFAULT_SCALABLE_QUBIT_SHAPE = PhysicalQubitScalable2D(
    LinearFunction(4, 5), LinearFunction(4, 5)
)


def create_translator():
    """Create the RPNG translator."""
    return DefaultRPNGTranslator()


def rpng_to_plaquette(rpng_desc: RPNGDescription, translator):
    """Convert an RPNG description to a Plaquette."""
    return translator.translate(rpng_desc)


# =============================================================================
# RPNG Description Helpers
# =============================================================================

def z_bulk(r="-", m="-"):
    """Z-basis bulk plaquette (schedule 1,3,4,2)."""
    return RPNGDescription.from_string(f"{r}z1{m} {r}z3{m} {r}z4{m} {r}z2{m}")

def x_bulk(r="-", m="-"):
    """X-basis bulk plaquette (schedule 7,5,4,6)."""
    return RPNGDescription.from_string(f"{r}x7{m} {r}x5{m} {r}x4{m} {r}x6{m}")

def z_2body_up(r="-", m="-"):
    """Z-basis 2-body UP boundary (positions 2,3)."""
    return RPNGDescription.from_string(f"---- ---- {r}z4{m} {r}z2{m}")

def x_2body_up(r="-", m="-"):
    """X-basis 2-body UP boundary (positions 2,3)."""
    return RPNGDescription.from_string(f"---- ---- {r}x4{m} {r}x6{m}")

def z_2body_down(r="-", m="-"):
    """Z-basis 2-body DOWN boundary (positions 0,1)."""
    return RPNGDescription.from_string(f"{r}z1{m} {r}z3{m} ---- ----")

def x_2body_down(r="-", m="-"):
    """X-basis 2-body DOWN boundary (positions 0,1)."""
    return RPNGDescription.from_string(f"{r}x7{m} {r}x5{m} ---- ----")

def z_2body_left(r="-", m="-"):
    """Z-basis 2-body LEFT boundary (positions 1,3)."""
    return RPNGDescription.from_string(f"---- {r}z3{m} ---- {r}z2{m}")

def x_2body_left(r="-", m="-"):
    """X-basis 2-body LEFT boundary (positions 1,3)."""
    return RPNGDescription.from_string(f"---- {r}x5{m} ---- {r}x6{m}")

def z_2body_right(r="-", m="-"):
    """Z-basis 2-body RIGHT boundary (positions 0,2)."""
    return RPNGDescription.from_string(f"{r}z1{m} ---- {r}z4{m} ----")

def x_2body_right(r="-", m="-"):
    """X-basis 2-body RIGHT boundary (positions 0,2)."""
    return RPNGDescription.from_string(f"{r}x7{m} ---- {r}x4{m} ----")

def z_corner(r="-", m="-"):
    """Z-basis corner (weight-2, positions depend on corner location)."""
    # This is a placeholder - actual corner plaquettes depend on which corner
    return RPNGDescription.from_string(f"{r}z1{m} ---- ---- {r}z2{m}")

def x_corner(r="-", m="-"):
    """X-basis corner (weight-2)."""
    return RPNGDescription.from_string(f"{r}x7{m} ---- ---- {r}x6{m}")


# =============================================================================
# Plaquette Sets for Each Cube Configuration
# =============================================================================

def create_empty_plaquettes(translator) -> Plaquettes:
    """Create a plaquette set where all plaquettes are empty."""
    empty = RPNGDescription.empty()
    rpng_dict = {i: empty for i in range(1, 15)}
    plaquette_dict = {i: rpng_to_plaquette(rpng, translator) for i, rpng in rpng_dict.items()}
    return Plaquettes(FrozenDefaultDict(plaquette_dict, default_value=rpng_to_plaquette(empty, translator)))


def create_z0_cube00_plaquettes(
    translator,
    reset: str | None = None,
    measurement: str | None = None,
) -> Plaquettes:
    """
    z=0, Cube (0,0): Standard ZXZ with all open boundaries.
    LEFT=Z, RIGHT=Z, UP=X, DOWN=X
    
    Corners: ALL EMPTY (this is the first block, no continuations)
    """
    r = reset if reset else "-"
    m = measurement if measurement else "-"
    
    rpng_dict = {
        # Corners - all empty for (0,0,0)
        1: RPNGDescription.empty(),
        2: RPNGDescription.empty(),
        3: RPNGDescription.empty(),
        4: RPNGDescription.empty(),
        
        # UP boundary - X-basis
        5: x_2body_up(r, m),
        6: RPNGDescription.empty(),
        
        # LEFT boundary - Z-basis
        7: RPNGDescription.empty(),
        8: z_2body_left(r, m),
        
        # Bulk
        9: z_bulk(r, m),
        10: x_bulk(r, m),
        
        # RIGHT boundary - Z-basis (open edge)
        11: z_2body_right(r, m),
        12: RPNGDescription.empty(),
        
        # DOWN boundary - X-basis (open edge)
        13: RPNGDescription.empty(),
        14: x_2body_down(r, m),
    }
    
    plaquette_dict = {i: rpng_to_plaquette(rpng, translator) for i, rpng in rpng_dict.items()}
    empty = rpng_to_plaquette(RPNGDescription.empty(), translator)
    return Plaquettes(FrozenDefaultDict(plaquette_dict, default_value=empty))


def create_z1_cube00_plaquettes(
    translator,
    reset: str | None = None,
    measurement: str | None = None,
) -> Plaquettes:
    """
    z=1, Cube (0,0): ZXZ with RIGHT boundary as pipe (bulk plaquettes).
    LEFT=Z, RIGHT=bulk(pipe), UP=X, DOWN=X
    
    Corners: At (0,0,z), corners 1,3 must be empty. Corners 2,4 can be filled.
    - Corner 2 (X-type): U=X matches -> FILLED
    - Corner 4: R=pipe -> empty (no boundary to extend)
    """
    r = reset if reset else "-"
    m = measurement if measurement else "-"
    
    # Corner 2 is X-type - top corners use lower qubits (positions 2,3)
    # X schedule is 7,5,4,6 for positions 0,1,2,3
    # Positions 2,3 -> timings 4,6
    x_corner_2 = RPNGDescription.from_string(f"---- ---- {r}x4{m} {r}x6{m}")
    
    rpng_dict = {
        # Corners: 1,3 empty (rule for 0,0,z), 4 empty (pipe has no boundary)
        1: RPNGDescription.empty(),
        2: x_corner_2,  # X-type: U=X matches -> FILLED
        3: RPNGDescription.empty(),
        4: RPNGDescription.empty(),
        
        # UP boundary - X-basis
        5: x_2body_up(r, m),
        6: RPNGDescription.empty(),
        
        # LEFT boundary - Z-basis
        7: RPNGDescription.empty(),
        8: z_2body_left(r, m),
        
        # Bulk
        9: z_bulk(r, m),
        10: x_bulk(r, m),
        
        # RIGHT boundary - becomes BULK (pipe to (1,0))
        11: z_bulk(r, m),  # Z bulk instead of 2-body
        12: x_bulk(r, m),  # X bulk instead of empty
        
        # DOWN boundary - X-basis
        13: RPNGDescription.empty(),
        14: x_2body_down(r, m),
    }
    
    plaquette_dict = {i: rpng_to_plaquette(rpng, translator) for i, rpng in rpng_dict.items()}
    empty = rpng_to_plaquette(RPNGDescription.empty(), translator)
    return Plaquettes(FrozenDefaultDict(plaquette_dict, default_value=empty))


def create_z1_cube10_plaquettes(
    translator,
    reset: str | None = None,
    measurement: str | None = None,
) -> Plaquettes:
    """
    z=1, Cube (1,0): ZXZ with overrides +x=X, +y=Z (swapped from -y=Z).
    LEFT=bulk(pipe), RIGHT=X, UP=Z, DOWN=X
    
    Corners: At (1,0,z), corners 2,4 must be empty. Corners 1,3 can be filled.
    - Corner 1: L=pipe -> empty (no boundary to extend)
    - Corner 3 (X-type): D=X matches -> FILLED
    """
    r = reset if reset else "-"
    m = measurement if measurement else "-"
    
    # Corner 3 is X-type - bottom corners use upper qubits (positions 0,1)
    # X schedule is 7,5,4,6 for positions 0,1,2,3
    # Positions 0,1 -> timings 7,5
    x_corner_3 = RPNGDescription.from_string(f"{r}x7{m} {r}x5{m} ---- ----")
    
    rpng_dict = {
        # Corners: 2,4 empty (rule for 1,0,z), 1 empty (pipe has no boundary)
        1: RPNGDescription.empty(),
        2: RPNGDescription.empty(),
        3: x_corner_3,  # X-type: D=X matches -> FILLED
        4: RPNGDescription.empty(),
        
        # UP boundary - Z-basis (index 6 is Z-type)
        5: RPNGDescription.empty(),
        6: z_2body_up(r, m),
        
        # LEFT boundary - becomes BULK (pipe from (0,0))
        7: x_bulk(r, m),  # X bulk (index 7 is X-type)
        8: z_bulk(r, m),  # Z bulk (index 8 is Z-type)
        
        # Bulk
        9: z_bulk(r, m),
        10: x_bulk(r, m),
        
        # RIGHT boundary - X-basis (index 12 is X-type)
        11: RPNGDescription.empty(),
        12: x_2body_right(r, m),
        
        # DOWN boundary - X-basis (index 14 is X-type)
        13: RPNGDescription.empty(),
        14: x_2body_down(r, m),
    }
    
    plaquette_dict = {i: rpng_to_plaquette(rpng, translator) for i, rpng in rpng_dict.items()}
    empty = rpng_to_plaquette(RPNGDescription.empty(), translator)
    return Plaquettes(FrozenDefaultDict(plaquette_dict, default_value=empty))


def create_z2_cube00_plaquettes(
    translator,
    reset: str | None = None,
    measurement: str | None = None,
) -> Plaquettes:
    """
    z=2, Cube (0,0): ZXZ with override -y=Z (swapped from +y=Z).
    LEFT=Z, RIGHT=bulk(pipe), UP=X, DOWN=Z
    
    Corners: At (0,0,z), corners 1,3 must be empty. Corners 2,4 can be filled.
    - Corner 2 (X-type): U=X matches -> FILLED
    - Corner 4 (Z-type): D=Z matches -> FILLED
    """
    r = reset if reset else "-"
    m = measurement if measurement else "-"
    
    # Corner 2 is X-type - top corners use lower qubits (positions 2,3)
    # X schedule is 7,5,4,6 for positions 0,1,2,3
    # Positions 2,3 -> timings 4,6
    x_corner_2 = RPNGDescription.from_string(f"---- ---- {r}x4{m} {r}x6{m}")
    
    # Corner 4 is Z-type - bottom corners use upper qubits (positions 0,1)
    # Z schedule is 1,3,4,2 for positions 0,1,2,3
    # Positions 0,1 -> timings 1,3
    z_corner_4 = RPNGDescription.from_string(f"{r}z1{m} {r}z3{m} ---- ----")
    
    rpng_dict = {
        # Corners: 1,3 empty (rule for 0,0,z)
        1: RPNGDescription.empty(),
        2: x_corner_2,  # X-type: U=X matches -> FILLED
        3: RPNGDescription.empty(),
        4: z_corner_4,  # Z-type: D=Z matches -> FILLED
        
        # UP boundary - X-basis (index 5 is X-type)
        5: x_2body_up(r, m),
        6: RPNGDescription.empty(),
        
        # LEFT boundary - Z-basis (index 8 is Z-type)
        7: RPNGDescription.empty(),
        8: z_2body_left(r, m),
        
        # Bulk
        9: z_bulk(r, m),
        10: x_bulk(r, m),
        
        # RIGHT boundary - becomes BULK (pipe to (1,0))
        11: z_bulk(r, m),
        12: x_bulk(r, m),
        
        # DOWN boundary - Z-basis (index 13 is Z-type)
        13: z_2body_down(r, m),
        14: RPNGDescription.empty(),
    }
    
    plaquette_dict = {i: rpng_to_plaquette(rpng, translator) for i, rpng in rpng_dict.items()}
    empty = rpng_to_plaquette(RPNGDescription.empty(), translator)
    return Plaquettes(FrozenDefaultDict(plaquette_dict, default_value=empty))


def create_z2_cube10_plaquettes(
    translator,
    reset: str | None = None,
    measurement: str | None = None,
) -> Plaquettes:
    """
    z=2, Cube (1,0): ZXZ with overrides +x=X, +y=Z, -y=Z.
    LEFT=bulk(pipe), RIGHT=X, UP=Z, DOWN=Z
    
    Corners: ALL EMPTY per user specification.
    """
    r = reset if reset else "-"
    m = measurement if measurement else "-"
    
    rpng_dict = {
        # Corners: all empty per user specification
        1: RPNGDescription.empty(),
        2: RPNGDescription.empty(),
        3: RPNGDescription.empty(),
        4: RPNGDescription.empty(),
        
        # UP boundary - Z-basis (index 6 is Z-type)
        5: RPNGDescription.empty(),
        6: z_2body_up(r, m),
        
        # LEFT boundary - becomes BULK (pipe from (0,0))
        7: x_bulk(r, m),  # X bulk (index 7 is X-type)
        8: z_bulk(r, m),  # Z bulk (index 8 is Z-type)
        
        # Bulk
        9: z_bulk(r, m),
        10: x_bulk(r, m),
        
        # RIGHT boundary - X-basis (index 12 is X-type)
        11: RPNGDescription.empty(),
        12: x_2body_right(r, m),
        
        # DOWN boundary - Z-basis (index 13 is Z-type)
        13: z_2body_down(r, m),
        14: RPNGDescription.empty(),
    }
    
    plaquette_dict = {i: rpng_to_plaquette(rpng, translator) for i, rpng in rpng_dict.items()}
    empty = rpng_to_plaquette(RPNGDescription.empty(), translator)
    return Plaquettes(FrozenDefaultDict(plaquette_dict, default_value=empty))


def create_z3_cube10_plaquettes(
    translator,
    reset: str | None = None,
    measurement: str | None = None,
) -> Plaquettes:
    """
    z=3, Cube (1,0): XZZ-like with ±x=X, ±y=Z.
    LEFT=X, RIGHT=X, UP=Z, DOWN=Z
    
    Corners: ALL EMPTY per user specification.
    """
    r = reset if reset else "-"
    m = measurement if measurement else "-"
    
    rpng_dict = {
        # Corners: all empty per user specification
        1: RPNGDescription.empty(),
        2: RPNGDescription.empty(),
        3: RPNGDescription.empty(),
        4: RPNGDescription.empty(),
        
        # UP boundary - Z-basis (index 6 is Z-type)
        5: RPNGDescription.empty(),
        6: z_2body_up(r, m),
        
        # LEFT boundary - X-basis (index 7 is X-type)
        7: x_2body_left(r, m),
        8: RPNGDescription.empty(),
        
        # Bulk
        9: z_bulk(r, m),
        10: x_bulk(r, m),
        
        # RIGHT boundary - X-basis (index 12 is X-type)
        11: RPNGDescription.empty(),
        12: x_2body_right(r, m),
        
        # DOWN boundary - Z-basis (index 13 is Z-type)
        13: z_2body_down(r, m),
        14: RPNGDescription.empty(),
    }
    
    plaquette_dict = {i: rpng_to_plaquette(rpng, translator) for i, rpng in rpng_dict.items()}
    empty = rpng_to_plaquette(RPNGDescription.empty(), translator)
    return Plaquettes(FrozenDefaultDict(plaquette_dict, default_value=empty))


# =============================================================================
# Layer Construction
# =============================================================================

def plaquette_layers_to_layout_layer(
    plaquette_layers: dict[tuple[int, int], PlaquetteLayer],
    scalable_shape: PhysicalQubitScalable2D = DEFAULT_SCALABLE_QUBIT_SHAPE,
) -> LayoutLayer:
    """Wrap multiple PlaquetteLayers in a LayoutLayer at different positions."""
    layers_dict = {
        LayoutCubePosition2D(2 * x, 2 * y): layer
        for (x, y), layer in plaquette_layers.items()
    }
    return LayoutLayer(layers_dict, scalable_shape)


def create_block_layers(
    template: QubitTemplate,
    cube_plaquettes: dict[tuple[int, int], tuple[Plaquettes, Plaquettes, Plaquettes]],
    scalable_shape: PhysicalQubitScalable2D = DEFAULT_SCALABLE_QUBIT_SHAPE,
) -> SequencedLayers:
    """
    Create layers for one temporal block (init + bulk×(2k-1) + meas).
    
    Args:
        template: QubitTemplate for each cube
        cube_plaquettes: Dict mapping (x, y) to (init_plaq, bulk_plaq, meas_plaq)
    """
    init_layers = {}
    bulk_layers = {}
    meas_layers = {}
    
    for pos, (init_plaq, bulk_plaq, meas_plaq) in cube_plaquettes.items():
        init_layers[pos] = PlaquetteLayer(template, init_plaq)
        bulk_layers[pos] = PlaquetteLayer(template, bulk_plaq)
        meas_layers[pos] = PlaquetteLayer(template, meas_plaq)
    
    init_layout = plaquette_layers_to_layout_layer(init_layers, scalable_shape)
    bulk_layout = plaquette_layers_to_layout_layer(bulk_layers, scalable_shape)
    meas_layout = plaquette_layers_to_layout_layer(meas_layers, scalable_shape)
    
    return SequencedLayers([
        init_layout,
        RepeatedLayer(bulk_layout, repetitions=LinearFunction(2, -1)),  # 2k-1 repetitions
        meas_layout,
    ])


def create_patch_rotation_layers(
    scalable_shape: PhysicalQubitScalable2D = DEFAULT_SCALABLE_QUBIT_SHAPE,
) -> SequencedLayers:
    """
    Create all layers for the patch rotation circuit.
    
    4 temporal blocks, each with 2k+1 layers.
    """
    translator = create_translator()
    template = QubitTemplate()
    empty_plaquettes = create_empty_plaquettes(translator)
    
    # =========================================================================
    # z=0 Block: Only cube (0,0), reset Z
    # =========================================================================
    z0_init_00 = create_z0_cube00_plaquettes(translator, reset="z")
    z0_bulk_00 = create_z0_cube00_plaquettes(translator)
    z0_meas_00 = create_z0_cube00_plaquettes(translator)
    
    z0_plaquettes = {
        (0, 0): (z0_init_00, z0_bulk_00, z0_meas_00),
        (1, 0): (empty_plaquettes, empty_plaquettes, empty_plaquettes),
    }
    z0_layers = create_block_layers(template, z0_plaquettes, scalable_shape)
    
    # =========================================================================
    # z=1 Block: Cubes (0,0) and (1,0) with pipe, reset Z at (1,0)
    # =========================================================================
    z1_init_00 = create_z1_cube00_plaquettes(translator)
    z1_bulk_00 = create_z1_cube00_plaquettes(translator)
    z1_meas_00 = create_z1_cube00_plaquettes(translator)
    
    z1_init_10 = create_z1_cube10_plaquettes(translator, reset="z")
    z1_bulk_10 = create_z1_cube10_plaquettes(translator)
    z1_meas_10 = create_z1_cube10_plaquettes(translator)
    
    z1_plaquettes = {
        (0, 0): (z1_init_00, z1_bulk_00, z1_meas_00),
        (1, 0): (z1_init_10, z1_bulk_10, z1_meas_10),
    }
    z1_layers = create_block_layers(template, z1_plaquettes, scalable_shape)
    
    # =========================================================================
    # z=2 Block: Cubes (0,0) and (1,0) with pipe, measure X at (0,0)
    # =========================================================================
    z2_init_00 = create_z2_cube00_plaquettes(translator)
    z2_bulk_00 = create_z2_cube00_plaquettes(translator)
    z2_meas_00 = create_z2_cube00_plaquettes(translator, measurement="x")
    
    z2_init_10 = create_z2_cube10_plaquettes(translator)
    z2_bulk_10 = create_z2_cube10_plaquettes(translator)
    z2_meas_10 = create_z2_cube10_plaquettes(translator)
    
    z2_plaquettes = {
        (0, 0): (z2_init_00, z2_bulk_00, z2_meas_00),
        (1, 0): (z2_init_10, z2_bulk_10, z2_meas_10),
    }
    z2_layers = create_block_layers(template, z2_plaquettes, scalable_shape)
    
    # =========================================================================
    # z=3 Block: Only cube (1,0), measure Z
    # =========================================================================
    z3_init_10 = create_z3_cube10_plaquettes(translator)
    z3_bulk_10 = create_z3_cube10_plaquettes(translator)
    z3_meas_10 = create_z3_cube10_plaquettes(translator, measurement="z")
    
    z3_plaquettes = {
        (0, 0): (empty_plaquettes, empty_plaquettes, empty_plaquettes),
        (1, 0): (z3_init_10, z3_bulk_10, z3_meas_10),
    }
    z3_layers = create_block_layers(template, z3_plaquettes, scalable_shape)
    
    # =========================================================================
    # Combine all blocks
    # =========================================================================
    return SequencedLayers([z0_layers, z1_layers, z2_layers, z3_layers])


def create_patch_rotation_layer_tree() -> LayerTree:
    """Create the LayerTree for the patch rotation circuit.
    
    Note: This creates the layer tree WITHOUT an observable.
    The observable is added separately using add_observable_from_missing_detector()
    because the FIXED_BULK_OBSERVABLE_BUILDER doesn't work for this asymmetric structure.
    """
    layers = create_patch_rotation_layers()
    
    # Wrap in SequencedLayers as expected by LayerTree
    root = SequencedLayers([layers])
    
    # Create LayerTree without observable - we'll add it manually
    return LayerTree(
        root=root,
        observable_builder=FIXED_BULK_OBSERVABLE_BUILDER,
        abstract_observables=[],  # No abstract observables - we add manually
    )


def add_observable_from_missing_detector(circuit: stim.Circuit) -> stim.Circuit:
    """Add the observable based on the missing detector.
    
    For the patch rotation, the observable is a stretched stabilizer that spans
    across block boundaries through the pipe connection. This is detected as a
    "missing detector" when no observable is defined.
    
    The observable tracks the logical qubit as it moves:
    - From block (0,0) bottom boundary (MX measurements at y=10)
    - To block (1,0) final measurements (MZ at the end)
    
    Args:
        circuit: The circuit without observable
        
    Returns:
        Circuit with the correct observable added
    """
    # Remove any existing OBSERVABLE_INCLUDE
    circuit_no_obs = stim.Circuit()
    for instr in circuit.flattened():
        if instr.name != 'OBSERVABLE_INCLUDE':
            circuit_no_obs.append(instr)
    
    # Find the missing detector - this IS the observable
    missing = circuit_no_obs.missing_detectors()
    
    if len(missing) != 1:
        raise ValueError(f"Expected exactly 1 missing detector for observable, found {len(missing)}")
    
    # Get the measurement record targets from the missing detector
    missing_det = list(missing)[0]
    obs_targets = list(missing_det.targets_copy())
    
    # Add as OBSERVABLE_INCLUDE
    circuit_no_obs.append('OBSERVABLE_INCLUDE', obs_targets, [0])
    
    return circuit_no_obs


def generate_patch_rotation_circuit(
    k: int = 2,
    manhattan_radius: int = 2,
) -> stim.Circuit:
    """Generate the patch rotation circuit with the correct observable.
    
    The observable is derived from the "missing detector" which represents
    the stretched stabilizer spanning the block boundaries.
    """
    layer_tree = create_patch_rotation_layer_tree()
    circuit = layer_tree.generate_circuit(k, manhattan_radius=manhattan_radius)
    
    # Add the correct observable
    circuit_with_obs = add_observable_from_missing_detector(circuit)
    
    return circuit_with_obs


def generate_crumble_url(k: int = 2, manhattan_radius: int = 2) -> str:
    """Generate a Crumble URL for visualization."""
    layer_tree = create_patch_rotation_layer_tree()
    return layer_tree.generate_crumble_url(k, manhattan_radius=manhattan_radius)


def compute_graphlike_distance(k: int = 2, noise_rate: float = 0.001) -> int:
    """Compute the graphlike distance of the patch rotation circuit.
    
    Args:
        k: The code distance parameter (distance = 2k+1)
        noise_rate: Depolarizing noise rate for error model
        
    Returns:
        The graphlike distance
    """
    circuit = generate_patch_rotation_circuit(k, manhattan_radius=2)
    
    # Add noise for error model
    noisy = stim.Circuit()
    for instr in circuit.flattened():
        if instr.name in ['CX', 'CZ']:
            targets = [t.value for t in instr.targets_copy() if t.is_qubit_target]
            if targets:
                noisy.append('DEPOLARIZE2', targets, [noise_rate])
        noisy.append(instr)
    
    # Generate detector error model
    dem = noisy.detector_error_model(decompose_errors=True)
    
    # Compute shortest graphlike error
    error = dem.shortest_graphlike_error(ignore_ungraphlike_errors=True)
    
    return len(error)


if __name__ == "__main__":
    print("Generating patch rotation circuit...")
    
    k = 2
    
    # Generate Crumble URL
    print(f"\nGenerating Crumble URL for k={k}...")
    url = generate_crumble_url(k, manhattan_radius=2)
    print(f"\nCrumble URL:\n{url}")
    
    # Generate circuit with correct observable
    print(f"\nGenerating circuit...")
    circuit = generate_patch_rotation_circuit(k, manhattan_radius=2)
    print(f"Circuit has {circuit.num_qubits} qubits")
    print(f"Circuit has {circuit.num_detectors} detectors")
    print(f"Circuit has {circuit.num_observables} observables")
    
    # Compute distance
    print(f"\nComputing graphlike distance...")
    distance = compute_graphlike_distance(k)
    print(f"Graphlike distance: {distance} (expected: {2*k+1})")
