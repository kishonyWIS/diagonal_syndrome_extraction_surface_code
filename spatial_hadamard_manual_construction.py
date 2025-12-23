#!/usr/bin/env python3
"""
Example: Generate a memory experiment circuit directly from plaquettes.

This bypasses the BlockGraph abstraction and directly defines plaquette layouts
with custom RPNG descriptions to generate a stim circuit.
"""

# Set MEASUREMENT_SCHEDULE BEFORE importing any tqec modules
# This controls when bulk plaquette measurements happen
import tqec.plaquette.constants as constants
constants.MEASUREMENT_SCHEDULE = 8  # All measurements at step 8

# Reload the translator module to pick up the new MEASUREMENT_SCHEDULE
import importlib
import tqec.plaquette.rpng.translators.default as default_translator
importlib.reload(default_translator)

import stim
from tqec import NoiseModel
from tqec.compile.blocks.layers.atomic.layout import LayoutLayer
from tqec.compile.blocks.layers.atomic.plaquettes import PlaquetteLayer
from tqec.compile.blocks.layers.composed.repeated import RepeatedLayer
from tqec.compile.blocks.layers.composed.sequenced import SequencedLayers
from tqec.compile.blocks.positioning import LayoutCubePosition2D
from tqec.compile.observables.abstract_observable import AbstractObservable, CubeWithArms
from tqec.compile.observables.fixed_bulk_builder import FIXED_BULK_OBSERVABLE_BUILDER
from tqec.compile.tree.tree import LayerTree
from tqec.computation.cube import Cube, ZXCube
from tqec.circuit.schedule.circuit import ScheduledCircuit
from tqec.plaquette.plaquette import Plaquette, Plaquettes
from tqec.plaquette.qubit import PlaquetteQubits, SquarePlaquetteQubits
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


def create_custom_coupling_plaquette_11(
    reset_data: str | None = None,
    measurement_data: str | None = None,
    measure_aux: bool = True,
    measure_shared_data: bool = False,
    reset_shared_data: bool = True,
) -> Plaquette:
    """
    Create a custom coupling plaquette for index 11 (right boundary of ZXZ cube).
    
    This plaquette interacts with the bottom-right data qubit twice, which cannot
    be expressed in RPNG notation.
    
    RX(aux) reset, MZ(aux) measurement - FLAG (Z-basis aux measurement).
    (Unchanged by role exchange)
    
    Circuit:
        Step 0: RX(aux), RZ(bottom-right)
        Step 1: CX(aux, bottom-right)      # aux is control
        Step 4: CZ(aux, top-left)
        Step 5: CZ(aux, bottom-left)
        Step 6: CX(bottom-right, aux)      # data qubit is control!
        Step 8: MZ(aux), MZ(bottom-right)  # all measurements simultaneously
    
    Qubit layout (SquarePlaquetteQubits):
        - Index 0: top-left (-1, -1)
        - Index 1: top-right (1, -1)  -- NOT USED
        - Index 2: bottom-left (-1, 1)
        - Index 3: bottom-right (1, 1)  -- SHARED with XZX cube
        - Index 4: auxiliary/syndrome (0, 0)
    
    Args:
        reset_data: Reset basis for data qubits ('z' or None for no reset)
        measurement_data: Measurement basis for data qubits ('z' or None for no measurement)
        measure_aux: If True, measure the auxiliary qubit at the end
        measure_shared_data: If True, measure the shared data qubit (bottom-right) in Z basis
        
    Returns:
        Custom Plaquette for coupling at index 11
    """
    # Use standard square plaquette qubit layout
    qubits = SquarePlaquetteQubits()
    
    # Qubit indices:
    # Data: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
    # Syndrome: 4=auxiliary
    AUX = 4
    TOP_LEFT = 0
    BOTTOM_LEFT = 2
    BOTTOM_RIGHT = 3
    
    # Build the circuit
    circuit = stim.Circuit()
    
    # Step 0: Reset auxiliary in X basis, optionally reset shared data qubit in Z basis
    circuit.append("RX", [AUX], [])
    if reset_shared_data:
        circuit.append("RZ", [BOTTOM_RIGHT], [])  # Shared qubit - reset in Z basis
    circuit.append("TICK", [], [])
    
    # Step 1: CX between auxiliary (control) and bottom-right (target) - NOT delayed
    circuit.append("CX", [AUX, BOTTOM_RIGHT], [])
    circuit.append("TICK", [], [])
    
    # Steps 2, 3: (empty)
    circuit.append("TICK", [], [])
    circuit.append("TICK", [], [])
    
    # Step 4: CZ between auxiliary and top-left
    circuit.append("CZ", [AUX, TOP_LEFT], [])
    circuit.append("TICK", [], [])
    
    # Step 5: CZ between auxiliary and bottom-left
    circuit.append("CZ", [AUX, BOTTOM_LEFT], [])
    circuit.append("TICK", [], [])
    
    # Step 6: CX between bottom-right (control) and auxiliary (target) - moved from step 8
    circuit.append("CX", [BOTTOM_RIGHT, AUX], [])
    circuit.append("TICK", [], [])
    
    # Step 7: (empty - wait for other plaquettes, only needed if measurements follow)
    # Step 8: All measurements simultaneously
    if measure_aux or measure_shared_data:
        circuit.append("TICK", [], [])  # Step 7 wait
    if measure_aux:
        circuit.append("MZ", [AUX], [])
    if measure_shared_data:
        circuit.append("MZ", [BOTTOM_RIGHT], [])
    
    # Create scheduled circuit with explicit schedule
    if measure_aux or measure_shared_data:
        schedule = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    else:
        schedule = [0, 1, 2, 3, 4, 5, 6, 7]
    scheduled_circuit = ScheduledCircuit.from_circuit(circuit, schedule, qubits.qubit_map)
    
    # Filter to keep only the used data qubits (0, 2, 3) and syndrome qubit
    used_data_qubits = [qubits.data_qubits[i] for i in [TOP_LEFT, BOTTOM_LEFT, BOTTOM_RIGHT]]
    kept_qubits = used_data_qubits + qubits.syndrome_qubits
    filtered_circuit = scheduled_circuit.filter_by_qubits(kept_qubits)
    
    # Create the plaquette with filtered qubits
    plaquette_qubits = PlaquetteQubits(used_data_qubits, qubits.syndrome_qubits)
    
    # Generate a unique name
    r_str = reset_data if reset_data else "-"
    m_str = measurement_data if measurement_data else "-"
    aux_str = "maux" if measure_aux else "noaux"
    shared_str = "mshared" if measure_shared_data else ""
    name = f"coupling_11_r{r_str}_m{m_str}_{aux_str}{shared_str}"
    
    return Plaquette(
        name=name,
        qubits=plaquette_qubits,
        circuit=filtered_circuit,
        mergeable_instructions=frozenset(["RX", "RZ", "MZ", "MX", "H"]),
    )


def create_custom_coupling_plaquette_2(
    reset_data: str | None = None,
    measurement_data: str | None = None,
    measure_aux: bool = True,
    measure_shared_data: bool = False,
    reset_shared_data: bool = True,
) -> Plaquette:
    """
    Create a custom coupling plaquette for index 2 (corner of ZXZ cube).
    
    Same as plaquette 12 but without the gate towards top-left.
    This plaquette interacts with the bottom-right data qubit twice.
    
    After role exchange: RZ(aux) reset, MX(aux) measurement - NOT a flag.
    (Exchanged roles with XZX plaquette 1)
    
    Circuit:
        Step 0: RZ(aux), RZ(bottom-right)
        Step 2: CX(bottom-right, aux)      # data qubit is control!
        Step 3: CX(aux, bottom-left)
        Step 7: CX(aux, bottom-right)
        Step 8: MX(aux), MZ(bottom-right)  # all measurements simultaneously
    
    Qubit layout (SquarePlaquetteQubits):
        - Index 0: top-left (-1, -1)  -- NOT USED
        - Index 1: top-right (1, -1)  -- NOT USED
        - Index 2: bottom-left (-1, 1)
        - Index 3: bottom-right (1, 1)  -- SHARED with XZX cube
        - Index 4: auxiliary/syndrome (0, 0)
    
    Args:
        reset_data: Reset basis for data qubits ('z' or None for no reset)
        measurement_data: Measurement basis for data qubits ('z' or None for no measurement)
        measure_aux: If True, measure the auxiliary qubit (X-basis) at the end
        measure_shared_data: If True, measure the shared data qubit (bottom-right) in Z basis
        
    Returns:
        Custom Plaquette for coupling at index 2
    """
    # Use standard square plaquette qubit layout
    qubits = SquarePlaquetteQubits()
    
    # Qubit indices:
    # Data: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
    # Syndrome: 4=auxiliary
    AUX = 4
    BOTTOM_LEFT = 2
    BOTTOM_RIGHT = 3
    
    # Build the circuit
    circuit = stim.Circuit()
    
    # Step 0: Reset auxiliary in Z basis, optionally reset shared data qubit in Z basis
    circuit.append("RZ", [AUX], [])
    if reset_shared_data:
        circuit.append("RZ", [BOTTOM_RIGHT], [])  # Shared qubit - reset in Z basis
    circuit.append("TICK", [], [])
    
    # Step 1: (empty)
    circuit.append("TICK", [], [])
    
    # Step 2: CX between bottom-right (control) and auxiliary (target)
    circuit.append("CX", [BOTTOM_RIGHT, AUX], [])
    circuit.append("TICK", [], [])
    
    # Step 3: CX between auxiliary and bottom-left
    circuit.append("CX", [AUX, BOTTOM_LEFT], [])
    circuit.append("TICK", [], [])
    
    # Steps 4, 5, 6: (empty)
    circuit.append("TICK", [], [])
    circuit.append("TICK", [], [])
    circuit.append("TICK", [], [])
    
    # Step 7: CX between auxiliary (control) and bottom-right (target) - moved from step 9
    circuit.append("CX", [AUX, BOTTOM_RIGHT], [])
    
    # Step 8: All measurements simultaneously (only add TICK if measurements follow)
    if measure_aux or measure_shared_data:
        circuit.append("TICK", [], [])
        if measure_aux:
            circuit.append("MX", [AUX], [])
    if measure_shared_data:
        circuit.append("MZ", [BOTTOM_RIGHT], [])
    
    # Create scheduled circuit with explicit schedule
    # With measurements: 8 TICKs = 9 moments (0-8), without: 7 TICKs = 8 moments (0-7)
    schedule = [0, 1, 2, 3, 4, 5, 6, 7, 8] if (measure_aux or measure_shared_data) else [0, 1, 2, 3, 4, 5, 6, 7]
    scheduled_circuit = ScheduledCircuit.from_circuit(circuit, schedule, qubits.qubit_map)
    
    # Filter to keep only the used data qubits (2, 3) and syndrome qubit
    used_data_qubits = [qubits.data_qubits[i] for i in [BOTTOM_LEFT, BOTTOM_RIGHT]]
    kept_qubits = used_data_qubits + qubits.syndrome_qubits
    filtered_circuit = scheduled_circuit.filter_by_qubits(kept_qubits)
    
    # Create the plaquette with filtered qubits
    plaquette_qubits = PlaquetteQubits(used_data_qubits, qubits.syndrome_qubits)
    
    # Generate a unique name
    r_str = reset_data if reset_data else "-"
    m_str = measurement_data if measurement_data else "-"
    aux_str = "maux" if measure_aux else "noaux"
    shared_str = "mshared" if measure_shared_data else ""
    name = f"coupling_2_r{r_str}_m{m_str}_{aux_str}{shared_str}"
    
    return Plaquette(
        name=name,
        qubits=plaquette_qubits,
        circuit=filtered_circuit,
        mergeable_instructions=frozenset(["RX", "RZ", "MZ", "MX", "H"]),
    )


def create_custom_coupling_plaquette_12(
    reset_data: str | None = None,
    measurement_data: str | None = None,
    measure_aux: bool = True,
    measure_shared_data: bool = False,
    reset_shared_data: bool = True,
) -> Plaquette:
    """
    Create a custom coupling plaquette for index 12 (right boundary of ZXZ cube).
    
    This plaquette interacts with the bottom-right data qubit twice, which cannot
    be expressed in RPNG notation.
    
    After role exchange: RZ(aux) reset, MX(aux) measurement - NOT a flag.
    (Exchanged roles with XZX plaquette 8)
    
    Circuit:
        Step 0: RZ(aux), RZ(bottom-right)
        Step 2: CX(bottom-right, aux)      # data qubit is control!
        Step 3: CX(aux, bottom-left)
        Step 4: CX(aux, top-left)
        Step 7: CX(aux, bottom-right)
        Step 8: MX(aux), MZ(bottom-right)  # all measurements simultaneously
    
    Qubit layout (SquarePlaquetteQubits):
        - Index 0: top-left (-1, -1)
        - Index 1: top-right (1, -1)  -- NOT USED
        - Index 2: bottom-left (-1, 1)
        - Index 3: bottom-right (1, 1)  -- SHARED with XZX cube
        - Index 4: auxiliary/syndrome (0, 0)
    
    Args:
        reset_data: Reset basis for data qubits ('z' or None for no reset)
        measurement_data: Measurement basis for data qubits ('z' or None for no measurement)
        measure_aux: If True, measure the auxiliary qubit at the end
        measure_shared_data: If True, measure the shared data qubit (bottom-right) in Z basis
        
    Returns:
        Custom Plaquette for coupling at index 12
    """
    # Use standard square plaquette qubit layout
    qubits = SquarePlaquetteQubits()
    
    # Qubit indices:
    # Data: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
    # Syndrome: 4=auxiliary
    AUX = 4
    TOP_LEFT = 0
    BOTTOM_LEFT = 2
    BOTTOM_RIGHT = 3
    
    # Build the circuit
    circuit = stim.Circuit()
    
    # Step 0: Reset auxiliary in Z basis, optionally reset shared data qubit in Z basis
    circuit.append("RZ", [AUX], [])
    if reset_shared_data:
        circuit.append("RZ", [BOTTOM_RIGHT], [])  # Shared qubit - reset in Z basis
    circuit.append("TICK", [], [])
    
    # Step 1: (empty)
    circuit.append("TICK", [], [])
    
    # Step 2: CX between bottom-right (control) and auxiliary (target)
    circuit.append("CX", [BOTTOM_RIGHT, AUX], [])
    circuit.append("TICK", [], [])
    
    # Step 3: CX between auxiliary and bottom-left
    circuit.append("CX", [AUX, BOTTOM_LEFT], [])
    circuit.append("TICK", [], [])
    
    # Step 4: CX between auxiliary and top-left
    circuit.append("CX", [AUX, TOP_LEFT], [])
    circuit.append("TICK", [], [])
    
    # Steps 5, 6: (empty)
    circuit.append("TICK", [], [])
    circuit.append("TICK", [], [])
    
    # Step 7: CX between auxiliary (control) and bottom-right (target) - moved from step 9
    circuit.append("CX", [AUX, BOTTOM_RIGHT], [])
    
    # Step 8: All measurements simultaneously (only add TICK if measurements follow)
    if measure_aux or measure_shared_data:
        circuit.append("TICK", [], [])
        if measure_aux:
            circuit.append("MX", [AUX], [])
    if measure_shared_data:
        circuit.append("MZ", [BOTTOM_RIGHT], [])
    
    # Create scheduled circuit with explicit schedule
    # With measurements: 8 TICKs = 9 moments (0-8), without: 7 TICKs = 8 moments (0-7)
    schedule = [0, 1, 2, 3, 4, 5, 6, 7, 8] if (measure_aux or measure_shared_data) else [0, 1, 2, 3, 4, 5, 6, 7]
    scheduled_circuit = ScheduledCircuit.from_circuit(circuit, schedule, qubits.qubit_map)
    
    # Filter to keep only the used data qubits (0, 2, 3) and syndrome qubit
    used_data_qubits = [qubits.data_qubits[i] for i in [TOP_LEFT, BOTTOM_LEFT, BOTTOM_RIGHT]]
    kept_qubits = used_data_qubits + qubits.syndrome_qubits
    filtered_circuit = scheduled_circuit.filter_by_qubits(kept_qubits)
    
    # Create the plaquette with filtered qubits
    plaquette_qubits = PlaquetteQubits(used_data_qubits, qubits.syndrome_qubits)
    
    # Generate a unique name
    r_str = reset_data if reset_data else "-"
    m_str = measurement_data if measurement_data else "-"
    aux_str = "maux" if measure_aux else "noaux"
    shared_str = "mshared" if measure_shared_data else ""
    name = f"coupling_12_r{r_str}_m{m_str}_{aux_str}{shared_str}"
    
    return Plaquette(
        name=name,
        qubits=plaquette_qubits,
        circuit=filtered_circuit,
        mergeable_instructions=frozenset(["RX", "RZ", "MZ", "MX", "H"]),
    )


# ============================================================================
# Custom plaquettes for XZX cube (right cube) - left boundary coupling
# ============================================================================

def create_custom_coupling_plaquette_xzx_7(
    reset_data: str | None = None,
    measurement_data: str | None = None,
    measure_aux: bool = True,
    reset_shared_data: bool = True,
) -> Plaquette:
    """
    Create a custom coupling plaquette for index 7 (left boundary of XZX cube).
    
    This plaquette interacts with the bottom-left data qubit twice.
    
    RZ(aux) reset, MX(aux) measurement - NOT a flag (X-basis aux measurement).
    (Unchanged by role exchange)
    
    Circuit:
        Step 0: RZ(aux), RZ(bottom-left)
        Step 2: CX(bottom-left, aux)
        Step 5: CX(aux, top-right)
        Step 6: CX(aux, bottom-right)
        Step 7: CX(aux, bottom-left)       # moved from step 9
        Step 8: MX(aux)                    # all measurements simultaneously
    
    Qubit layout (SquarePlaquetteQubits):
        - Index 0: top-left (-1, -1)  -- NOT USED
        - Index 1: top-right (1, -1)
        - Index 2: bottom-left (-1, 1)  -- SHARED with ZXZ cube
        - Index 3: bottom-right (1, 1)
        - Index 4: auxiliary/syndrome (0, 0)
    
    Args:
        reset_data: Reset basis for data qubits ('x' or None for no reset)
        measurement_data: Measurement basis for data qubits ('x' or None for no measurement)
        measure_aux: If True, measure the auxiliary qubit (X-basis) at the end
        
    Returns:
        Custom Plaquette for coupling at index 7 (XZX cube)
    """
    # Use standard square plaquette qubit layout
    qubits = SquarePlaquetteQubits()
    
    # Qubit indices:
    # Data: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
    # Syndrome: 4=auxiliary
    AUX = 4
    TOP_RIGHT = 1
    BOTTOM_LEFT = 2
    BOTTOM_RIGHT = 3
    
    # Build the circuit
    circuit = stim.Circuit()
    
    # Step 0: Reset auxiliary in Z basis, optionally reset shared data qubit in Z basis
    circuit.append("RZ", [AUX], [])
    if reset_shared_data:
        circuit.append("RZ", [BOTTOM_LEFT], [])  # Shared qubit - reset in Z basis
    circuit.append("TICK", [], [])
    
    # Step 1: (empty)
    circuit.append("TICK", [], [])
    
    # Step 2: CX between bottom-left (control) and auxiliary (target)
    circuit.append("CX", [BOTTOM_LEFT, AUX], [])
    circuit.append("TICK", [], [])
    
    # Steps 3, 4: (empty)
    circuit.append("TICK", [], [])
    circuit.append("TICK", [], [])
    
    # Step 5: CX between auxiliary and top-right
    circuit.append("CX", [AUX, TOP_RIGHT], [])
    circuit.append("TICK", [], [])
    
    # Step 6: CX between auxiliary and bottom-right
    circuit.append("CX", [AUX, BOTTOM_RIGHT], [])
    circuit.append("TICK", [], [])
    
    # Step 7: CX between auxiliary and bottom-left (shared) - moved from step 9
    circuit.append("CX", [AUX, BOTTOM_LEFT], [])
    
    # Step 8: Measure auxiliary in X basis (optional, only add TICK if measurements follow)
    if measure_aux:
        circuit.append("TICK", [], [])
        circuit.append("MX", [AUX], [])
    
    # Create scheduled circuit with explicit schedule
    # With measurements: 8 TICKs = 9 moments (0-8), without: 7 TICKs = 8 moments (0-7)
    schedule = [0, 1, 2, 3, 4, 5, 6, 7, 8] if measure_aux else [0, 1, 2, 3, 4, 5, 6, 7]
    scheduled_circuit = ScheduledCircuit.from_circuit(circuit, schedule, qubits.qubit_map)
    
    # Filter to keep only the used data qubits (1, 2, 3) and syndrome qubit
    used_data_qubits = [qubits.data_qubits[i] for i in [TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT]]
    kept_qubits = used_data_qubits + qubits.syndrome_qubits
    filtered_circuit = scheduled_circuit.filter_by_qubits(kept_qubits)
    
    # Create the plaquette with filtered qubits
    plaquette_qubits = PlaquetteQubits(used_data_qubits, qubits.syndrome_qubits)
    
    # Generate a unique name
    r_str = reset_data if reset_data else "-"
    m_str = measurement_data if measurement_data else "-"
    aux_str = "maux" if measure_aux else "noaux"
    name = f"coupling_xzx_7_r{r_str}_m{m_str}_{aux_str}"
    
    return Plaquette(
        name=name,
        qubits=plaquette_qubits,
        circuit=filtered_circuit,
        mergeable_instructions=frozenset(["RX", "RZ", "MZ", "MX", "H"]),
    )


def create_custom_coupling_plaquette_xzx_8(
    reset_data: str | None = None,
    measurement_data: str | None = None,
    measure_aux: bool = True,
    reset_shared_data: bool = True,
) -> Plaquette:
    """
    Create a custom coupling plaquette for index 8 (left boundary of XZX cube).
    
    This plaquette interacts with the bottom-left data qubit twice.
    
    After role exchange: RX(aux) reset, MZ(aux) measurement - FLAG (Z-basis aux measurement).
    (Exchanged roles with ZXZ plaquette 12)
    
    Circuit:
        Step 0: RX(aux), RZ(bottom-left)
        Step 1: CX(aux, bottom-left)       # aux is control
        Step 2: CZ(aux, bottom-right)
        Step 3: CZ(aux, top-right)
        Step 6: CX(bottom-left, aux)       # data qubit is control! (moved from step 8)
        Step 8: MZ(aux)                    # all measurements simultaneously
    
    Qubit layout (SquarePlaquetteQubits):
        - Index 0: top-left (-1, -1)  -- NOT USED
        - Index 1: top-right (1, -1)
        - Index 2: bottom-left (-1, 1)  -- SHARED with ZXZ cube
        - Index 3: bottom-right (1, 1)
        - Index 4: auxiliary/syndrome (0, 0)
    
    Args:
        reset_data: Reset basis for data qubits ('x' or None for no reset)
        measurement_data: Measurement basis for data qubits ('x' or None for no measurement)
        measure_aux: If True, measure the auxiliary qubit (Z-basis) at the end - FLAG
        
    Returns:
        Custom Plaquette for coupling at index 8 (XZX cube)
    """
    # Use standard square plaquette qubit layout
    qubits = SquarePlaquetteQubits()
    
    # Qubit indices:
    # Data: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
    # Syndrome: 4=auxiliary
    AUX = 4
    TOP_RIGHT = 1
    BOTTOM_LEFT = 2
    BOTTOM_RIGHT = 3
    
    # Build the circuit
    circuit = stim.Circuit()
    
    # Step 0: Reset auxiliary in X basis, optionally reset shared data qubit in Z basis
    circuit.append("RX", [AUX], [])
    if reset_shared_data:
        circuit.append("RZ", [BOTTOM_LEFT], [])  # Shared qubit - reset in Z basis
    circuit.append("TICK", [], [])
    
    # Step 1: CX between auxiliary (control) and bottom-left (target)
    circuit.append("CX", [AUX, BOTTOM_LEFT], [])
    circuit.append("TICK", [], [])
    
    # Step 2: CZ with bottom-right
    circuit.append("CZ", [AUX, BOTTOM_RIGHT], [])
    circuit.append("TICK", [], [])
    
    # Step 3: CZ between auxiliary and top-right
    circuit.append("CZ", [AUX, TOP_RIGHT], [])
    circuit.append("TICK", [], [])
    
    # Steps 4, 5: (empty)
    circuit.append("TICK", [], [])
    circuit.append("TICK", [], [])
    
    # Step 6: CX between bottom-left (control) and auxiliary (target) - moved from step 8
    circuit.append("CX", [BOTTOM_LEFT, AUX], [])
    circuit.append("TICK", [], [])
    
    # Step 7: (empty - wait for other plaquettes, only needed if measurements follow)
    # Step 8: Measure auxiliary in Z basis (optional) - all measurements simultaneously
    if measure_aux:
        circuit.append("TICK", [], [])  # Step 7 wait
        circuit.append("MZ", [AUX], [])
    
    # Create scheduled circuit with explicit schedule
    # With measurements: 8 TICKs = 9 moments (0-8), without: 7 TICKs = 8 moments (0-7)
    schedule = [0, 1, 2, 3, 4, 5, 6, 7, 8] if measure_aux else [0, 1, 2, 3, 4, 5, 6, 7]
    scheduled_circuit = ScheduledCircuit.from_circuit(circuit, schedule, qubits.qubit_map)
    
    # Filter to keep only the used data qubits (1, 2, 3) and syndrome qubit
    used_data_qubits = [qubits.data_qubits[i] for i in [TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT]]
    kept_qubits = used_data_qubits + qubits.syndrome_qubits
    filtered_circuit = scheduled_circuit.filter_by_qubits(kept_qubits)
    
    # Create the plaquette with filtered qubits
    plaquette_qubits = PlaquetteQubits(used_data_qubits, qubits.syndrome_qubits)
    
    # Generate a unique name
    r_str = reset_data if reset_data else "-"
    m_str = measurement_data if measurement_data else "-"
    aux_str = "maux" if measure_aux else "noaux"
    name = f"coupling_xzx_8_r{r_str}_m{m_str}_{aux_str}"
    
    return Plaquette(
        name=name,
        qubits=plaquette_qubits,
        circuit=filtered_circuit,
        mergeable_instructions=frozenset(["RX", "RZ", "MZ", "MX", "H"]),
    )


def create_custom_coupling_plaquette_xzx_1(
    reset_data: str | None = None,
    measurement_data: str | None = None,
    measure_aux: bool = True,
    reset_shared_data: bool = True,
) -> Plaquette:
    """
    Create a custom coupling plaquette for index 1 (corner of XZX cube).
    
    Same as plaquette 8 but without the gate towards top-right.
    This plaquette interacts with the bottom-left data qubit twice.
    
    After role exchange: RX(aux) reset, MZ(aux) measurement - FLAG (Z-basis aux measurement).
    (Exchanged roles with ZXZ plaquette 2)
    
    Circuit:
        Step 0: RX(aux), RZ(bottom-left)
        Step 1: CX(aux, bottom-left)       # aux is control
        Step 2: CZ(aux, bottom-right)
        Step 6: CX(bottom-left, aux)       # data qubit is control! (moved from step 8)
        Step 8: MZ(aux)                    # all measurements simultaneously
    
    Qubit layout (SquarePlaquetteQubits):
        - Index 0: top-left (-1, -1)  -- NOT USED
        - Index 1: top-right (1, -1)  -- NOT USED
        - Index 2: bottom-left (-1, 1)  -- SHARED with ZXZ cube
        - Index 3: bottom-right (1, 1)
        - Index 4: auxiliary/syndrome (0, 0)
    
    Args:
        reset_data: Reset basis for data qubits ('x' or None for no reset)
        measurement_data: Measurement basis for data qubits ('x' or None for no measurement)
        measure_aux: If True, measure the auxiliary qubit (Z-basis) at the end - FLAG
        
    Returns:
        Custom Plaquette for coupling at index 1 (XZX cube)
    """
    # Use standard square plaquette qubit layout
    qubits = SquarePlaquetteQubits()
    
    # Qubit indices:
    # Data: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
    # Syndrome: 4=auxiliary
    AUX = 4
    BOTTOM_LEFT = 2
    BOTTOM_RIGHT = 3
    
    # Build the circuit
    circuit = stim.Circuit()
    
    # Step 0: Reset auxiliary in X basis, optionally reset shared data qubit in Z basis
    circuit.append("RX", [AUX], [])
    if reset_shared_data:
        circuit.append("RZ", [BOTTOM_LEFT], [])  # Shared qubit - reset in Z basis
    circuit.append("TICK", [], [])
    
    # Step 1: CX between auxiliary (control) and bottom-left (target)
    circuit.append("CX", [AUX, BOTTOM_LEFT], [])
    circuit.append("TICK", [], [])
    
    # Step 2: CZ with bottom-right
    circuit.append("CZ", [AUX, BOTTOM_RIGHT], [])
    circuit.append("TICK", [], [])
    
    # Steps 3, 4, 5: (empty - no top-right gate)
    circuit.append("TICK", [], [])
    circuit.append("TICK", [], [])
    circuit.append("TICK", [], [])
    
    # Step 6: CX between bottom-left (control) and auxiliary (target) - moved from step 8
    circuit.append("CX", [BOTTOM_LEFT, AUX], [])
    circuit.append("TICK", [], [])
    
    # Step 7: (empty - wait for other plaquettes, only needed if measurements follow)
    # Step 8: Measure auxiliary in Z basis (optional) - all measurements simultaneously
    if measure_aux:
        circuit.append("TICK", [], [])  # Step 7 wait
        circuit.append("MZ", [AUX], [])
    
    # Create scheduled circuit with explicit schedule
    # With measurements: 8 TICKs = 9 moments (0-8), without: 7 TICKs = 8 moments (0-7)
    schedule = [0, 1, 2, 3, 4, 5, 6, 7, 8] if measure_aux else [0, 1, 2, 3, 4, 5, 6, 7]
    scheduled_circuit = ScheduledCircuit.from_circuit(circuit, schedule, qubits.qubit_map)
    
    # Filter to keep only the used data qubits (2, 3) and syndrome qubit
    used_data_qubits = [qubits.data_qubits[i] for i in [BOTTOM_LEFT, BOTTOM_RIGHT]]
    kept_qubits = used_data_qubits + qubits.syndrome_qubits
    filtered_circuit = scheduled_circuit.filter_by_qubits(kept_qubits)
    
    # Create the plaquette with filtered qubits
    plaquette_qubits = PlaquetteQubits(used_data_qubits, qubits.syndrome_qubits)
    
    # Generate a unique name
    r_str = reset_data if reset_data else "-"
    m_str = measurement_data if measurement_data else "-"
    aux_str = "maux" if measure_aux else "noaux"
    name = f"coupling_xzx_1_r{r_str}_m{m_str}_{aux_str}"
    
    return Plaquette(
        name=name,
        qubits=plaquette_qubits,
        circuit=filtered_circuit,
        mergeable_instructions=frozenset(["RX", "RZ", "MZ", "MX", "H"]),
    )


# ============================================================================
# Custom plaquettes for Y-axis Hadamard: XZZ cube (bottom cube) - bottom boundary coupling
# ============================================================================

def create_custom_coupling_plaquette_xzz_13(
    reset_data: str | None = None,
    measurement_data: str | None = None,
    measure_aux: bool = True,
    measure_shared_data: bool = False,
    reset_shared_data: bool = True,
) -> Plaquette:
    """
    Create a custom coupling plaquette for index 13 (bottom boundary of XZZ cube).
    
    This plaquette interacts with the bottom-right data qubit (shared with ZXX cube).
    
    MZ(aux) measurement - FLAG (Z-basis aux measurement).
    RX reset for aux.
    
    Circuit:
        Step 0: RX(aux), RZ(shared)
        Step 1: CX(aux, shared)           # first shared interaction, aux is control
        Step 2: CZ(aux, top-right)        # non-shared, aux is control (CZ for XZZ)
        Step 3: CZ(aux, top-left)         # non-shared, aux is control (CZ for XZZ)
        Step 6: CX(shared, aux)           # second shared interaction, shared is control
        Step 8: MZ(aux), MZ(shared)       # all measurements simultaneously
    
    Qubit layout (SquarePlaquetteQubits):
        - Index 0: top-left (-1, -1)
        - Index 1: top-right (1, -1)
        - Index 2: bottom-left (-1, 1)  -- NOT USED
        - Index 3: bottom-right (1, 1)  -- SHARED with ZXX cube
        - Index 4: auxiliary/syndrome (0, 0)
    
    Args:
        reset_data: Reset basis for data qubits ('z' or None for no reset)
        measurement_data: Measurement basis for data qubits ('z' or None for no measurement)
        measure_aux: If True, measure the auxiliary qubit (Z-basis) at the end - FLAG
        measure_shared_data: If True, measure the shared data qubit (bottom-right) in Z basis
        reset_shared_data: If True, reset the shared data qubit at the start
        
    Returns:
        Custom Plaquette for coupling at index 13 (XZZ cube)
    """
    qubits = SquarePlaquetteQubits()
    
    AUX = 4
    TOP_LEFT = 0
    TOP_RIGHT = 1
    BOTTOM_RIGHT = 3  # Shared with ZXX
    
    circuit = stim.Circuit()
    
    # Step 0: Reset auxiliary in X basis, optionally reset shared data in Z basis
    circuit.append("RX", [AUX], [])
    if reset_shared_data:
        circuit.append("RZ", [BOTTOM_RIGHT], [])
    circuit.append("TICK", [], [])
    
    # Step 1: CX(aux, shared) - first shared interaction, aux is control
    circuit.append("CX", [AUX, BOTTOM_RIGHT], [])
    circuit.append("TICK", [], [])
    
    # Step 2: CZ(aux, top-right) - non-shared, aux is control
    circuit.append("CZ", [AUX, TOP_RIGHT], [])
    circuit.append("TICK", [], [])
    
    # Step 3: CZ(aux, top-left) - non-shared, aux is control
    circuit.append("CZ", [AUX, TOP_LEFT], [])
    circuit.append("TICK", [], [])
    
    # Steps 4, 5: (empty)
    circuit.append("TICK", [], [])
    circuit.append("TICK", [], [])
    
    # Step 6: CX(shared, aux) - second shared interaction, shared is control
    circuit.append("CX", [BOTTOM_RIGHT, AUX], [])
    circuit.append("TICK", [], [])
    
    # Step 7: (empty - wait for measurements)
    if measure_aux or measure_shared_data:
        circuit.append("TICK", [], [])
    
    # Step 8: All measurements simultaneously
    if measure_aux:
        circuit.append("MZ", [AUX], [])
    if measure_shared_data:
        circuit.append("MZ", [BOTTOM_RIGHT], [])
    
    # Schedule
    schedule = [0, 1, 2, 3, 4, 5, 6, 7, 8] if (measure_aux or measure_shared_data) else [0, 1, 2, 3, 4, 5, 6, 7]
    scheduled_circuit = ScheduledCircuit.from_circuit(circuit, schedule, qubits.qubit_map)
    
    # Filter to keep only used data qubits and syndrome qubit
    used_data_qubits = [qubits.data_qubits[i] for i in [TOP_LEFT, TOP_RIGHT, BOTTOM_RIGHT]]
    kept_qubits = used_data_qubits + qubits.syndrome_qubits
    filtered_circuit = scheduled_circuit.filter_by_qubits(kept_qubits)
    
    plaquette_qubits = PlaquetteQubits(used_data_qubits, qubits.syndrome_qubits)
    
    r_str = reset_data if reset_data else "-"
    m_str = measurement_data if measurement_data else "-"
    aux_str = "maux" if measure_aux else "noaux"
    shared_str = "mshared" if measure_shared_data else ""
    name = f"coupling_xzz_13_r{r_str}_m{m_str}_{aux_str}{shared_str}"
    
    return Plaquette(
        name=name,
        qubits=plaquette_qubits,
        circuit=filtered_circuit,
        mergeable_instructions=frozenset(["RX", "RZ", "MZ", "MX", "H"]),
    )


def create_custom_coupling_plaquette_xzz_14(
    reset_data: str | None = None,
    measurement_data: str | None = None,
    measure_aux: bool = True,
    measure_shared_data: bool = False,
    reset_shared_data: bool = True,
) -> Plaquette:
    """
    Create a custom coupling plaquette for index 14 (bottom boundary of XZZ cube).
    
    This plaquette interacts with the bottom-right data qubit (shared with ZXX cube).
    
    MX(aux) measurement - NOT a flag (X-basis aux measurement).
    RZ reset for aux.
    
    Circuit:
        Step 0: RZ(aux), RZ(shared)
        Step 2: CX(shared, aux)           # first shared interaction, shared is control
        Step 5: CX(aux, top-left)         # non-shared, aux is control (CX for plaquette 14)
        Step 6: CX(aux, top-right)        # non-shared, aux is control (CX for plaquette 14)
        Step 7: CX(aux, shared)           # second shared interaction, aux is control
        Step 8: MX(aux), MZ(shared)       # all measurements simultaneously
    
    Qubit layout (SquarePlaquetteQubits):
        - Index 0: top-left (-1, -1)
        - Index 1: top-right (1, -1)
        - Index 2: bottom-left (-1, 1)  -- NOT USED
        - Index 3: bottom-right (1, 1)  -- SHARED with ZXX cube
        - Index 4: auxiliary/syndrome (0, 0)
    
    Args:
        reset_data: Reset basis for data qubits ('z' or None for no reset)
        measurement_data: Measurement basis for data qubits ('z' or None for no measurement)
        measure_aux: If True, measure the auxiliary qubit (X-basis) at the end
        measure_shared_data: If True, measure the shared data qubit (bottom-right) in Z basis
        reset_shared_data: If True, reset the shared data qubit at the start
        
    Returns:
        Custom Plaquette for coupling at index 14 (XZZ cube)
    """
    qubits = SquarePlaquetteQubits()
    
    AUX = 4
    TOP_LEFT = 0
    TOP_RIGHT = 1
    BOTTOM_RIGHT = 3  # Shared with ZXX
    
    circuit = stim.Circuit()
    
    # Step 0: Reset auxiliary in Z basis, optionally reset shared data in Z basis
    circuit.append("RZ", [AUX], [])
    if reset_shared_data:
        circuit.append("RZ", [BOTTOM_RIGHT], [])
    circuit.append("TICK", [], [])
    
    # Step 1: (empty)
    circuit.append("TICK", [], [])
    
    # Step 2: CX(shared, aux) - first shared interaction, shared is control
    circuit.append("CX", [BOTTOM_RIGHT, AUX], [])
    circuit.append("TICK", [], [])
    
    # Steps 3, 4: (empty)
    circuit.append("TICK", [], [])
    circuit.append("TICK", [], [])
    
    # Step 5: CX(aux, top-left) - non-shared, aux is control (CX for plaquette 14)
    circuit.append("CX", [AUX, TOP_LEFT], [])
    circuit.append("TICK", [], [])
    
    # Step 6: CX(aux, top-right) - non-shared, aux is control (CX for plaquette 14)
    circuit.append("CX", [AUX, TOP_RIGHT], [])
    circuit.append("TICK", [], [])
    
    # Step 7: CX(aux, shared) - second shared interaction, aux is control
    circuit.append("CX", [AUX, BOTTOM_RIGHT], [])
    
    # Step 8: All measurements simultaneously
    if measure_aux or measure_shared_data:
        circuit.append("TICK", [], [])
        if measure_aux:
            circuit.append("MX", [AUX], [])
        if measure_shared_data:
            circuit.append("MZ", [BOTTOM_RIGHT], [])
    
    # Schedule
    schedule = [0, 1, 2, 3, 4, 5, 6, 7, 8] if (measure_aux or measure_shared_data) else [0, 1, 2, 3, 4, 5, 6, 7]
    scheduled_circuit = ScheduledCircuit.from_circuit(circuit, schedule, qubits.qubit_map)
    
    # Filter to keep only used data qubits and syndrome qubit
    used_data_qubits = [qubits.data_qubits[i] for i in [TOP_LEFT, TOP_RIGHT, BOTTOM_RIGHT]]
    kept_qubits = used_data_qubits + qubits.syndrome_qubits
    filtered_circuit = scheduled_circuit.filter_by_qubits(kept_qubits)
    
    plaquette_qubits = PlaquetteQubits(used_data_qubits, qubits.syndrome_qubits)
    
    r_str = reset_data if reset_data else "-"
    m_str = measurement_data if measurement_data else "-"
    aux_str = "maux" if measure_aux else "noaux"
    shared_str = "mshared" if measure_shared_data else ""
    name = f"coupling_xzz_14_r{r_str}_m{m_str}_{aux_str}{shared_str}"
    
    return Plaquette(
        name=name,
        qubits=plaquette_qubits,
        circuit=filtered_circuit,
        mergeable_instructions=frozenset(["RX", "RZ", "MZ", "MX", "H"]),
    )


def create_custom_coupling_plaquette_xzz_3(
    reset_data: str | None = None,
    measurement_data: str | None = None,
    measure_aux: bool = True,
    measure_shared_data: bool = False,
    reset_shared_data: bool = True,
) -> Plaquette:
    """
    Create a custom coupling plaquette for index 3 (corner of XZZ cube).
    
    This is a corner plaquette with only one non-shared data qubit.
    
    MX(aux) measurement - NOT a flag (X-basis aux measurement).
    RZ reset for aux.
    
    Circuit:
        Step 0: RZ(aux), RZ(shared)
        Step 2: CX(shared, aux)           # first shared interaction, shared is control
        Step 6: CX(aux, top-right)        # non-shared, aux is control (CX for plaquette 3)
        Step 7: CX(aux, shared)           # second shared interaction, aux is control
        Step 8: MX(aux), MZ(shared)       # all measurements simultaneously
    
    Qubit layout (SquarePlaquetteQubits):
        - Index 0: top-left (-1, -1)    -- NOT USED
        - Index 1: top-right (1, -1)
        - Index 2: bottom-left (-1, 1)  -- NOT USED
        - Index 3: bottom-right (1, 1)  -- SHARED with ZXX cube
        - Index 4: auxiliary/syndrome (0, 0)
    
    Args:
        reset_data: Reset basis for data qubits ('z' or None for no reset)
        measurement_data: Measurement basis for data qubits ('z' or None for no measurement)
        measure_aux: If True, measure the auxiliary qubit (X-basis) at the end
        measure_shared_data: If True, measure the shared data qubit (bottom-right) in Z basis
        reset_shared_data: If True, reset the shared data qubit at the start
        
    Returns:
        Custom Plaquette for coupling at index 3 (XZZ cube corner)
    """
    qubits = SquarePlaquetteQubits()
    
    AUX = 4
    TOP_RIGHT = 1
    BOTTOM_RIGHT = 3  # Shared with ZXX
    
    circuit = stim.Circuit()
    
    # Step 0: Reset auxiliary in Z basis, optionally reset shared data in Z basis
    circuit.append("RZ", [AUX], [])
    if reset_shared_data:
        circuit.append("RZ", [BOTTOM_RIGHT], [])
    circuit.append("TICK", [], [])
    
    # Step 1: (empty)
    circuit.append("TICK", [], [])
    
    # Step 2: CX(shared, aux) - first shared interaction, shared is control
    circuit.append("CX", [BOTTOM_RIGHT, AUX], [])
    circuit.append("TICK", [], [])
    
    # Steps 3, 4, 5: (empty)
    circuit.append("TICK", [], [])
    circuit.append("TICK", [], [])
    circuit.append("TICK", [], [])
    
    # Step 6: CX(aux, top-right) - non-shared, aux is control (CX for plaquette 3)
    circuit.append("CX", [AUX, TOP_RIGHT], [])
    circuit.append("TICK", [], [])
    
    # Step 7: CX(aux, shared) - second shared interaction, aux is control
    circuit.append("CX", [AUX, BOTTOM_RIGHT], [])
    
    # Step 8: All measurements simultaneously
    if measure_aux or measure_shared_data:
        circuit.append("TICK", [], [])
        if measure_aux:
            circuit.append("MX", [AUX], [])
        if measure_shared_data:
            circuit.append("MZ", [BOTTOM_RIGHT], [])
    
    # Schedule
    schedule = [0, 1, 2, 3, 4, 5, 6, 7, 8] if (measure_aux or measure_shared_data) else [0, 1, 2, 3, 4, 5, 6, 7]
    scheduled_circuit = ScheduledCircuit.from_circuit(circuit, schedule, qubits.qubit_map)
    
    # Filter to keep only used data qubits and syndrome qubit
    used_data_qubits = [qubits.data_qubits[i] for i in [TOP_RIGHT, BOTTOM_RIGHT]]
    kept_qubits = used_data_qubits + qubits.syndrome_qubits
    filtered_circuit = scheduled_circuit.filter_by_qubits(kept_qubits)
    
    plaquette_qubits = PlaquetteQubits(used_data_qubits, qubits.syndrome_qubits)
    
    r_str = reset_data if reset_data else "-"
    m_str = measurement_data if measurement_data else "-"
    aux_str = "maux" if measure_aux else "noaux"
    shared_str = "mshared" if measure_shared_data else ""
    name = f"coupling_xzz_3_r{r_str}_m{m_str}_{aux_str}{shared_str}"
    
    return Plaquette(
        name=name,
        qubits=plaquette_qubits,
        circuit=filtered_circuit,
        mergeable_instructions=frozenset(["RX", "RZ", "MZ", "MX", "H"]),
    )


def create_zxz_plaquettes(
    translator,
    reset: str | None = None,
    measurement: str | None = None,
    use_custom_coupling: bool = False,
    measure_coupling_aux_mz: bool = True,
    measure_coupling_aux_mx: bool = True,
    measure_shared_data: bool = False,
    reset_shared_data: bool = True,
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
    
    Custom coupling plaquettes (after role exchange):
    - Plaquette 11: MZ(aux) - controlled by measure_coupling_aux_mz (FLAG)
    - Plaquettes 2, 12: MX(aux) - controlled by measure_coupling_aux_mx (not flags)
    
    Args:
        translator: RPNG translator
        reset: Reset basis ('x' or 'z') to apply to data qubits, or None
        measurement: Measurement basis ('x' or 'z') to apply to data qubits, or None
        use_custom_coupling: If True, use custom coupling plaquettes for indices 2, 11, 12
        measure_coupling_aux_mz: If True, measure Z-basis aux (plaquette 11) - these are FLAGS
        measure_coupling_aux_mx: If True, measure X-basis aux (plaquettes 2, 12)
    """
    r = reset if reset else "-"
    m = measurement if measurement else "-"
    
    # Diagonal schedules for ZXZ cube (all CX/CZ delayed by 1):
    # X-basis: 2, 5, 4, 3 (positions 0, 1, 2, 3)
    # Z-basis: 7, 5, 4, 6 (positions 0, 1, 2, 3)
    
    rpng_descriptions = {
        # Corners (empty)
        1: RPNGDescription.empty(),
        2: RPNGDescription.empty(),
        3: RPNGDescription.empty(),
        4: RPNGDescription.empty(),
        
        # Top boundary - X-basis 2-body (positions 2,3 → timings 4,6 from X schedule 7,5,4,6)
        5: RPNGDescription.from_string(f"---- ---- {r}x4{m} {r}x6{m}"),
        6: RPNGDescription.empty(),
        
        # Left boundary - Z-basis 2-body (positions 1,3 → timings 3,2 from Z schedule 1,3,4,2)
        7: RPNGDescription.empty(),
        8: RPNGDescription.from_string(f"---- {r}z3{m} ---- {r}z2{m}"),
        
        # Bulk plaquettes (same schedule for both cubes)
        9: RPNGDescription.from_string(f"{r}z1{m} {r}z3{m} {r}z4{m} {r}z2{m}"),   # Z-basis: 1,3,4,2
        10: RPNGDescription.from_string(f"{r}x7{m} {r}x5{m} {r}x4{m} {r}x6{m}"),  # X-basis: 7,5,4,6
        
        # Right boundary - Z-basis 2-body (positions 0,2 → timings 1,4 from Z schedule 1,3,4,2)
        # Index 11 may be replaced with custom coupling plaquette
        11: RPNGDescription.from_string(f"{r}z1{m} ---- {r}z4{m} ----"),
        12: RPNGDescription.empty(),
        
        # Bottom boundary - X-basis 2-body (positions 0,1 → timings 7,5 from X schedule 7,5,4,6)
        13: RPNGDescription.empty(),
        14: RPNGDescription.from_string(f"{r}x7{m} {r}x5{m} ---- ----"),
    }
    
    plaquette_collection = {
        idx: rpng_to_plaquette(rpng, translator)
        for idx, rpng in rpng_descriptions.items()
    }
    
    # Replace indices 2, 11, and 12 with custom coupling plaquettes if requested
    if use_custom_coupling:
        # Plaquette 2: MX(aux) - controlled by measure_coupling_aux_mx
        plaquette_collection[2] = create_custom_coupling_plaquette_2(
            reset_data=reset,
            measurement_data=measurement,
            measure_aux=measure_coupling_aux_mx,
            measure_shared_data=measure_shared_data,
            reset_shared_data=reset_shared_data,
        )
        # Plaquette 11: MZ(aux) - controlled by measure_coupling_aux_mz (FLAG)
        plaquette_collection[11] = create_custom_coupling_plaquette_11(
            reset_data=reset,
            measurement_data=measurement,
            measure_aux=measure_coupling_aux_mz,
            measure_shared_data=measure_shared_data,
            reset_shared_data=reset_shared_data,
        )
        # Plaquette 12: MX(aux) - controlled by measure_coupling_aux_mx
        plaquette_collection[12] = create_custom_coupling_plaquette_12(
            reset_data=reset,
            measurement_data=measurement,
            measure_aux=measure_coupling_aux_mx,
            measure_shared_data=measure_shared_data,
            reset_shared_data=reset_shared_data,
        )
    
    empty_plaquette = rpng_to_plaquette(RPNGDescription.empty(), translator)
    return Plaquettes(FrozenDefaultDict(plaquette_collection, default_value=empty_plaquette))


# ============================================================================
# Custom plaquettes for Y-axis Hadamard: ZXX cube (top cube) - top boundary coupling
# ============================================================================

def create_custom_coupling_plaquette_zxx_5(
    reset_data: str | None = None,
    measurement_data: str | None = None,
    measure_aux: bool = True,
    reset_shared_data: bool = True,
) -> Plaquette:
    """
    Create a custom coupling plaquette for index 5 (top boundary of ZXX cube).
    
    This plaquette interacts with the top-right data qubit (shared with XZZ cube).
    
    MX(aux) measurement - NOT a flag (X-basis aux measurement).
    RZ reset for aux.
    
    Circuit:
        Step 0: RZ(aux), RZ(shared)
        Step 2: CX(shared, aux)           # first shared interaction, shared is control
        Step 3: CX(aux, bottom-left)      # non-shared, aux is control (CX for ZXX)
        Step 4: CX(aux, bottom-right)     # non-shared, aux is control (CX for ZXX)
        Step 7: CX(aux, shared)           # second shared interaction, aux is control
        Step 8: MX(aux)                   # aux only - shared data measured by XZZ
    
    Qubit layout (SquarePlaquetteQubits):
        - Index 0: top-left (-1, -1)    -- NOT USED
        - Index 1: top-right (1, -1)    -- SHARED with XZZ cube
        - Index 2: bottom-left (-1, 1)
        - Index 3: bottom-right (1, 1)
        - Index 4: auxiliary/syndrome (0, 0)
    
    Args:
        reset_data: Reset basis for data qubits ('x' or None for no reset)
        measurement_data: Measurement basis for data qubits ('x' or None for no measurement)
        measure_aux: If True, measure the auxiliary qubit (X-basis) at the end
        reset_shared_data: If True, reset the shared data qubit at the start
        
    Returns:
        Custom Plaquette for coupling at index 5 (ZXX cube)
    """
    qubits = SquarePlaquetteQubits()
    
    AUX = 4
    TOP_RIGHT = 1     # Shared with XZZ
    BOTTOM_LEFT = 2
    BOTTOM_RIGHT = 3
    
    circuit = stim.Circuit()
    
    # Step 0: Reset auxiliary in Z basis, optionally reset shared data in Z basis
    circuit.append("RZ", [AUX], [])
    if reset_shared_data:
        circuit.append("RZ", [TOP_RIGHT], [])
    circuit.append("TICK", [], [])
    
    # Step 1: (empty)
    circuit.append("TICK", [], [])
    
    # Step 2: CX(shared, aux) - first shared interaction, shared is control
    circuit.append("CX", [TOP_RIGHT, AUX], [])
    circuit.append("TICK", [], [])
    
    # Step 3: CX(aux, bottom-left) - non-shared, aux is control
    circuit.append("CX", [AUX, BOTTOM_LEFT], [])
    circuit.append("TICK", [], [])
    
    # Step 4: CX(aux, bottom-right) - non-shared, aux is control
    circuit.append("CX", [AUX, BOTTOM_RIGHT], [])
    circuit.append("TICK", [], [])
    
    # Steps 5, 6: (empty)
    circuit.append("TICK", [], [])
    circuit.append("TICK", [], [])
    
    # Step 7: CX(aux, shared) - second shared interaction, aux is control
    circuit.append("CX", [AUX, TOP_RIGHT], [])
    
    # Step 8: Measurement (aux only - shared data measured by XZZ side)
    if measure_aux:
        circuit.append("TICK", [], [])
        circuit.append("MX", [AUX], [])
    
    # Schedule
    schedule = [0, 1, 2, 3, 4, 5, 6, 7, 8] if measure_aux else [0, 1, 2, 3, 4, 5, 6, 7]
    scheduled_circuit = ScheduledCircuit.from_circuit(circuit, schedule, qubits.qubit_map)
    
    # Filter to keep only used data qubits and syndrome qubit
    used_data_qubits = [qubits.data_qubits[i] for i in [TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT]]
    kept_qubits = used_data_qubits + qubits.syndrome_qubits
    filtered_circuit = scheduled_circuit.filter_by_qubits(kept_qubits)
    
    plaquette_qubits = PlaquetteQubits(used_data_qubits, qubits.syndrome_qubits)
    
    r_str = reset_data if reset_data else "-"
    m_str = measurement_data if measurement_data else "-"
    aux_str = "maux" if measure_aux else "noaux"
    name = f"coupling_zxx_5_r{r_str}_m{m_str}_{aux_str}"
    
    return Plaquette(
        name=name,
        qubits=plaquette_qubits,
        circuit=filtered_circuit,
        mergeable_instructions=frozenset(["RX", "RZ", "MZ", "MX", "H"]),
    )


def create_custom_coupling_plaquette_zxx_6(
    reset_data: str | None = None,
    measurement_data: str | None = None,
    measure_aux: bool = True,
    reset_shared_data: bool = True,
) -> Plaquette:
    """
    Create a custom coupling plaquette for index 6 (top boundary of ZXX cube).
    
    This plaquette interacts with the top-right data qubit (shared with XZZ cube).
    
    MZ(aux) measurement - FLAG (Z-basis aux measurement).
    RX reset for aux.
    
    Circuit:
        Step 0: RX(aux), RZ(shared)
        Step 1: CX(aux, shared)           # first shared interaction, aux is control
        Step 4: CZ(aux, bottom-right)     # non-shared, aux is control (CZ for plaquette 6)
        Step 5: CZ(aux, bottom-left)      # non-shared, aux is control (CZ for plaquette 6)
        Step 6: CX(shared, aux)           # second shared interaction, shared is control
        Step 8: MZ(aux)                   # aux only - shared data measured by XZZ
    
    Qubit layout (SquarePlaquetteQubits):
        - Index 0: top-left (-1, -1)    -- NOT USED
        - Index 1: top-right (1, -1)    -- SHARED with XZZ cube
        - Index 2: bottom-left (-1, 1)
        - Index 3: bottom-right (1, 1)
        - Index 4: auxiliary/syndrome (0, 0)
    
    Args:
        reset_data: Reset basis for data qubits ('x' or None for no reset)
        measurement_data: Measurement basis for data qubits ('x' or None for no measurement)
        measure_aux: If True, measure the auxiliary qubit (Z-basis) at the end - FLAG
        reset_shared_data: If True, reset the shared data qubit at the start
        
    Returns:
        Custom Plaquette for coupling at index 6 (ZXX cube)
    """
    qubits = SquarePlaquetteQubits()
    
    AUX = 4
    TOP_RIGHT = 1     # Shared with XZZ
    BOTTOM_LEFT = 2
    BOTTOM_RIGHT = 3
    
    circuit = stim.Circuit()
    
    # Step 0: Reset auxiliary in X basis, optionally reset shared data in Z basis
    circuit.append("RX", [AUX], [])
    if reset_shared_data:
        circuit.append("RZ", [TOP_RIGHT], [])
    circuit.append("TICK", [], [])
    
    # Step 1: CX(aux, shared) - first shared interaction, aux is control
    circuit.append("CX", [AUX, TOP_RIGHT], [])
    circuit.append("TICK", [], [])
    
    # Steps 2, 3: (empty)
    circuit.append("TICK", [], [])
    circuit.append("TICK", [], [])
    
    # Step 4: CZ(aux, bottom-right) - non-shared, aux is control (CZ for plaquette 6)
    circuit.append("CZ", [AUX, BOTTOM_RIGHT], [])
    circuit.append("TICK", [], [])
    
    # Step 5: CZ(aux, bottom-left) - non-shared, aux is control (CZ for plaquette 6)
    circuit.append("CZ", [AUX, BOTTOM_LEFT], [])
    circuit.append("TICK", [], [])
    
    # Step 6: CX(shared, aux) - second shared interaction, shared is control
    circuit.append("CX", [TOP_RIGHT, AUX], [])
    circuit.append("TICK", [], [])
    
    # Step 7: (empty - wait for measurements)
    if measure_aux:
        circuit.append("TICK", [], [])
    
    # Step 8: Measurement (aux only - shared data measured by XZZ side)
    if measure_aux:
        circuit.append("MZ", [AUX], [])
    
    # Schedule
    schedule = [0, 1, 2, 3, 4, 5, 6, 7, 8] if measure_aux else [0, 1, 2, 3, 4, 5, 6, 7]
    scheduled_circuit = ScheduledCircuit.from_circuit(circuit, schedule, qubits.qubit_map)
    
    # Filter to keep only used data qubits and syndrome qubit
    used_data_qubits = [qubits.data_qubits[i] for i in [TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT]]
    kept_qubits = used_data_qubits + qubits.syndrome_qubits
    filtered_circuit = scheduled_circuit.filter_by_qubits(kept_qubits)
    
    plaquette_qubits = PlaquetteQubits(used_data_qubits, qubits.syndrome_qubits)
    
    r_str = reset_data if reset_data else "-"
    m_str = measurement_data if measurement_data else "-"
    aux_str = "maux" if measure_aux else "noaux"
    name = f"coupling_zxx_6_r{r_str}_m{m_str}_{aux_str}"
    
    return Plaquette(
        name=name,
        qubits=plaquette_qubits,
        circuit=filtered_circuit,
        mergeable_instructions=frozenset(["RX", "RZ", "MZ", "MX", "H"]),
    )


def create_custom_coupling_plaquette_zxx_1(
    reset_data: str | None = None,
    measurement_data: str | None = None,
    measure_aux: bool = True,
    reset_shared_data: bool = True,
) -> Plaquette:
    """
    Create a custom coupling plaquette for index 1 (corner of ZXX cube).
    
    This is a corner plaquette with only one non-shared data qubit.
    
    MZ(aux) measurement - FLAG (Z-basis aux measurement).
    RX reset for aux.
    
    Circuit:
        Step 0: RX(aux), RZ(shared)
        Step 1: CX(aux, shared)           # first shared interaction, aux is control
        Step 4: CZ(aux, bottom-right)     # non-shared, aux is control (CZ for plaquette 1)
        Step 6: CX(shared, aux)           # second shared interaction, shared is control
        Step 8: MZ(aux)                   # aux only - shared data measured by XZZ
    
    Qubit layout (SquarePlaquetteQubits):
        - Index 0: top-left (-1, -1)    -- NOT USED
        - Index 1: top-right (1, -1)    -- SHARED with XZZ cube
        - Index 2: bottom-left (-1, 1)  -- NOT USED
        - Index 3: bottom-right (1, 1)
        - Index 4: auxiliary/syndrome (0, 0)
    
    Args:
        reset_data: Reset basis for data qubits ('x' or None for no reset)
        measurement_data: Measurement basis for data qubits ('x' or None for no measurement)
        measure_aux: If True, measure the auxiliary qubit (Z-basis) at the end - FLAG
        reset_shared_data: If True, reset the shared data qubit at the start
        
    Returns:
        Custom Plaquette for coupling at index 1 (ZXX cube corner)
    """
    qubits = SquarePlaquetteQubits()
    
    AUX = 4
    TOP_RIGHT = 1     # Shared with XZZ
    BOTTOM_RIGHT = 3
    
    circuit = stim.Circuit()
    
    # Step 0: Reset auxiliary in X basis, optionally reset shared data in Z basis
    circuit.append("RX", [AUX], [])
    if reset_shared_data:
        circuit.append("RZ", [TOP_RIGHT], [])
    circuit.append("TICK", [], [])
    
    # Step 1: CX(aux, shared) - first shared interaction, aux is control
    circuit.append("CX", [AUX, TOP_RIGHT], [])
    circuit.append("TICK", [], [])
    
    # Steps 2, 3: (empty)
    circuit.append("TICK", [], [])
    circuit.append("TICK", [], [])
    
    # Step 4: CZ(aux, bottom-right) - non-shared, aux is control (CZ for plaquette 1)
    circuit.append("CZ", [AUX, BOTTOM_RIGHT], [])
    circuit.append("TICK", [], [])
    
    # Step 5: (empty)
    circuit.append("TICK", [], [])
    
    # Step 6: CX(shared, aux) - second shared interaction, shared is control
    circuit.append("CX", [TOP_RIGHT, AUX], [])
    circuit.append("TICK", [], [])
    
    # Step 7: (empty - wait for measurements)
    if measure_aux:
        circuit.append("TICK", [], [])
    
    # Step 8: Measurement (aux only - shared data measured by XZZ side)
    if measure_aux:
        circuit.append("MZ", [AUX], [])
    
    # Schedule
    schedule = [0, 1, 2, 3, 4, 5, 6, 7, 8] if measure_aux else [0, 1, 2, 3, 4, 5, 6, 7]
    scheduled_circuit = ScheduledCircuit.from_circuit(circuit, schedule, qubits.qubit_map)
    
    # Filter to keep only used data qubits and syndrome qubit
    used_data_qubits = [qubits.data_qubits[i] for i in [TOP_RIGHT, BOTTOM_RIGHT]]
    kept_qubits = used_data_qubits + qubits.syndrome_qubits
    filtered_circuit = scheduled_circuit.filter_by_qubits(kept_qubits)
    
    plaquette_qubits = PlaquetteQubits(used_data_qubits, qubits.syndrome_qubits)
    
    r_str = reset_data if reset_data else "-"
    m_str = measurement_data if measurement_data else "-"
    aux_str = "maux" if measure_aux else "noaux"
    name = f"coupling_zxx_1_r{r_str}_m{m_str}_{aux_str}"
    
    return Plaquette(
        name=name,
        qubits=plaquette_qubits,
        circuit=filtered_circuit,
        mergeable_instructions=frozenset(["RX", "RZ", "MZ", "MX", "H"]),
    )


def create_xzx_plaquettes(
    translator,
    reset: str | None = None,
    measurement: str | None = None,
    use_custom_coupling: bool = False,
    measure_coupling_aux_mz: bool = True,
    measure_coupling_aux_mx: bool = True,
    reset_shared_data: bool = True,
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
    
    Custom coupling plaquettes (after role exchange):
    - Plaquettes 1, 8: MZ(aux) - controlled by measure_coupling_aux_mz (FLAGS)
    - Plaquette 7: MX(aux) - controlled by measure_coupling_aux_mx (not a flag)
    
    Args:
        translator: RPNG translator
        reset: Reset basis ('x' or 'z') to apply to data qubits, or None
        measurement: Measurement basis ('x' or 'z') to apply to data qubits, or None
        use_custom_coupling: If True, use custom coupling plaquettes for indices 1, 7, 8
        measure_coupling_aux_mz: If True, measure Z-basis aux (plaquettes 1, 8) - these are FLAGS
        measure_coupling_aux_mx: If True, measure X-basis aux (plaquette 7)
    """
    r = reset if reset else "-"
    m = measurement if measurement else "-"
    
    # Diagonal schedules for XZX cube (opposite of ZXZ, all CX/CZ delayed by 1):
    # Z-basis: 2, 5, 4, 3 (positions 0, 1, 2, 3)
    # X-basis: 7, 5, 4, 6 (positions 0, 1, 2, 3)
    
    rpng_descriptions = {
        # Corners (empty)
        1: RPNGDescription.empty(),
        2: RPNGDescription.empty(),
        3: RPNGDescription.empty(),
        4: RPNGDescription.empty(),
        
        # Top boundary - Z-basis 2-body (positions 2,3 → timings 4,2 from Z schedule 1,3,4,2)
        5: RPNGDescription.empty(),
        6: RPNGDescription.from_string(f"---- ---- {r}z4{m} {r}z2{m}"),
        
        # Left boundary - X-basis 2-body (positions 1,3 → timings 5,6 from X schedule 7,5,4,6)
        7: RPNGDescription.from_string(f"---- {r}x5{m} ---- {r}x6{m}"),
        8: RPNGDescription.empty(),
        
        # Bulk plaquettes (same schedule for both cubes)
        9: RPNGDescription.from_string(f"{r}z1{m} {r}z3{m} {r}z4{m} {r}z2{m}"),   # Z-basis: 1,3,4,2
        10: RPNGDescription.from_string(f"{r}x7{m} {r}x5{m} {r}x4{m} {r}x6{m}"),  # X-basis: 7,5,4,6
        
        # Right boundary - X-basis 2-body (positions 0,2 → timings 7,4 from X schedule 7,5,4,6)
        11: RPNGDescription.empty(),
        12: RPNGDescription.from_string(f"{r}x7{m} ---- {r}x4{m} ----"),
        
        # Bottom boundary - Z-basis 2-body (positions 0,1 → timings 1,3 from Z schedule 1,3,4,2)
        13: RPNGDescription.from_string(f"{r}z1{m} {r}z3{m} ---- ----"),
        14: RPNGDescription.empty(),
    }
    
    plaquette_collection = {
        idx: rpng_to_plaquette(rpng, translator)
        for idx, rpng in rpng_descriptions.items()
    }
    
    # Replace indices 1, 7, and 8 with custom coupling plaquettes if requested
    if use_custom_coupling:
        # Plaquette 1: MZ(aux) - controlled by measure_coupling_aux_mz (FLAG)
        plaquette_collection[1] = create_custom_coupling_plaquette_xzx_1(
            reset_data=reset,
            measurement_data=measurement,
            measure_aux=measure_coupling_aux_mz,
            reset_shared_data=reset_shared_data,
        )
        # Plaquette 7: MX(aux) - controlled by measure_coupling_aux_mx
        plaquette_collection[7] = create_custom_coupling_plaquette_xzx_7(
            reset_data=reset,
            measurement_data=measurement,
            measure_aux=measure_coupling_aux_mx,
            reset_shared_data=reset_shared_data,
        )
        # Plaquette 8: MZ(aux) - controlled by measure_coupling_aux_mz (FLAG)
        plaquette_collection[8] = create_custom_coupling_plaquette_xzx_8(
            reset_data=reset,
            measurement_data=measurement,
            measure_aux=measure_coupling_aux_mz,
            reset_shared_data=reset_shared_data,
        )
    
    empty_plaquette = rpng_to_plaquette(RPNGDescription.empty(), translator)
    return Plaquettes(FrozenDefaultDict(plaquette_collection, default_value=empty_plaquette))


def create_xzz_plaquettes(
    translator,
    reset: str | None = None,
    measurement: str | None = None,
    use_custom_coupling: bool = False,
    measure_coupling_aux_mz: bool = True,
    measure_coupling_aux_mx: bool = True,
    measure_shared_data: bool = False,
    reset_shared_data: bool = True,
) -> Plaquettes:
    """
    Create plaquettes for an XZZ cube (Z-basis memory with XZX-style boundaries).
    
    XZZ means: X boundaries on left/right (x-axis), Z boundaries on top/bottom (y-axis), stores Z.
    
    This has the same boundary structure as XZX but stores Z instead of X.
    
    The QubitTemplate layout for k=2 looks like:
    
        1  5  6  5  6  2
        7  9 10  9 10 11
        8 10  9 10  9 12
        7  9 10  9 10 11
        8 10  9 10  9 12
        3 13 14 13 14  4
    
    Where:
    - 1,2,3,4: corners (empty)
    - 5,6: top boundary (5=empty, 6=Z active)
    - 7,8: left boundary (7=X active, 8=empty)
    - 9,10: bulk plaquettes (9=Z, 10=X)
    - 11,12: right boundary (11=empty, 12=X active)
    - 13,14: bottom boundary (13=Z active, 14=empty) <- interface with ZXX
    
    Interface plaquettes (for Y-axis Hadamard): indices 3, 13, 14
    Custom coupling plaquettes:
    - Plaquette 13: MZ(aux) - controlled by measure_coupling_aux_mz (FLAG)
    - Plaquettes 3, 14: MX(aux) - controlled by measure_coupling_aux_mx (not flags)
    
    Args:
        translator: RPNG translator
        reset: Reset basis ('x' or 'z') to apply to data qubits, or None
        measurement: Measurement basis ('x' or 'z') to apply to data qubits, or None
        use_custom_coupling: If True, use custom coupling plaquettes for indices 3, 13, 14
        measure_coupling_aux_mz: If True, measure Z-basis aux (plaquette 13) - these are FLAGS
        measure_coupling_aux_mx: If True, measure X-basis aux (plaquettes 3, 14)
        measure_shared_data: If True, measure the shared data qubit in Z basis
        reset_shared_data: If True, reset the shared data qubit at the start
    """
    r = reset if reset else "-"
    m = measurement if measurement else "-"
    
    # Same boundary structure as XZX
    rpng_descriptions = {
        # Corners (empty)
        1: RPNGDescription.empty(),
        2: RPNGDescription.empty(),
        3: RPNGDescription.empty(),
        4: RPNGDescription.empty(),
        
        # Top boundary - Z-basis 2-body (positions 2,3 → timings 4,2 from Z schedule 1,3,4,2)
        5: RPNGDescription.empty(),
        6: RPNGDescription.from_string(f"---- ---- {r}z4{m} {r}z2{m}"),
        
        # Left boundary - X-basis 2-body (positions 1,3 → timings 5,6 from X schedule 7,5,4,6)
        7: RPNGDescription.from_string(f"---- {r}x5{m} ---- {r}x6{m}"),
        8: RPNGDescription.empty(),
        
        # Bulk plaquettes (same schedule for all cubes)
        9: RPNGDescription.from_string(f"{r}z1{m} {r}z3{m} {r}z4{m} {r}z2{m}"),   # Z-basis: 1,3,4,2
        10: RPNGDescription.from_string(f"{r}x7{m} {r}x5{m} {r}x4{m} {r}x6{m}"),  # X-basis: 7,5,4,6
        
        # Right boundary - X-basis 2-body (positions 0,2 → timings 7,4 from X schedule 7,5,4,6)
        11: RPNGDescription.empty(),
        12: RPNGDescription.from_string(f"{r}x7{m} ---- {r}x4{m} ----"),
        
        # Bottom boundary - Z-basis 2-body (positions 0,1 → timings 1,3 from Z schedule 1,3,4,2)
        # This is the interface boundary - will be replaced with custom if use_custom_coupling
        13: RPNGDescription.from_string(f"{r}z1{m} {r}z3{m} ---- ----"),
        14: RPNGDescription.empty(),
    }
    
    plaquette_collection = {
        idx: rpng_to_plaquette(rpng, translator)
        for idx, rpng in rpng_descriptions.items()
    }
    
    # Replace interface plaquettes with custom coupling plaquettes if requested
    if use_custom_coupling:
        # Plaquette 3: MX(aux) - corner, controlled by measure_coupling_aux_mx
        plaquette_collection[3] = create_custom_coupling_plaquette_xzz_3(
            reset_data=reset,
            measurement_data=measurement,
            measure_aux=measure_coupling_aux_mx,
            measure_shared_data=measure_shared_data,
            reset_shared_data=reset_shared_data,
        )
        # Plaquette 13: MZ(aux) - controlled by measure_coupling_aux_mz (FLAG)
        plaquette_collection[13] = create_custom_coupling_plaquette_xzz_13(
            reset_data=reset,
            measurement_data=measurement,
            measure_aux=measure_coupling_aux_mz,
            measure_shared_data=measure_shared_data,
            reset_shared_data=reset_shared_data,
        )
        # Plaquette 14: MX(aux) - controlled by measure_coupling_aux_mx
        plaquette_collection[14] = create_custom_coupling_plaquette_xzz_14(
            reset_data=reset,
            measurement_data=measurement,
            measure_aux=measure_coupling_aux_mx,
            measure_shared_data=measure_shared_data,
            reset_shared_data=reset_shared_data,
        )
    
    empty_plaquette = rpng_to_plaquette(RPNGDescription.empty(), translator)
    return Plaquettes(FrozenDefaultDict(plaquette_collection, default_value=empty_plaquette))


def create_zxx_plaquettes(
    translator,
    reset: str | None = None,
    measurement: str | None = None,
    use_custom_coupling: bool = False,
    measure_coupling_aux_mz: bool = True,
    measure_coupling_aux_mx: bool = True,
    reset_shared_data: bool = True,
) -> Plaquettes:
    """
    Create plaquettes for a ZXX cube (X-basis memory with ZXZ-style boundaries).
    
    ZXX means: Z boundaries on left/right (x-axis), X boundaries on top/bottom (y-axis), stores X.
    
    This has the same boundary structure as ZXZ but stores X instead of Z.
    
    The QubitTemplate layout for k=2 looks like:
    
        1  5  6  5  6  2
        7  9 10  9 10 11
        8 10  9 10  9 12
        7  9 10  9 10 11
        8 10  9 10  9 12
        3 13 14 13 14  4
    
    Where:
    - 1,2,3,4: corners (empty)
    - 5,6: top boundary (5=X active, 6=empty) <- interface with XZZ
    - 7,8: left boundary (7=empty, 8=Z active)
    - 9,10: bulk plaquettes (9=Z, 10=X)
    - 11,12: right boundary (11=Z active, 12=empty)
    - 13,14: bottom boundary (13=empty, 14=X active)
    
    Interface plaquettes (for Y-axis Hadamard): indices 1, 5, 6
    Custom coupling plaquettes:
    - Plaquettes 1, 6: MZ(aux) - controlled by measure_coupling_aux_mz (FLAGS)
    - Plaquette 5: MX(aux) - controlled by measure_coupling_aux_mx (not a flag)
    
    Note: ZXX plaquettes do NOT measure the shared data qubit - only XZZ does.
    This mirrors the X-axis structure where XZX plaquettes don't measure shared data.
    
    Args:
        translator: RPNG translator
        reset: Reset basis ('x' or 'z') to apply to data qubits, or None
        measurement: Measurement basis ('x' or 'z') to apply to data qubits, or None
        use_custom_coupling: If True, use custom coupling plaquettes for indices 1, 5, 6
        measure_coupling_aux_mz: If True, measure Z-basis aux (plaquettes 1, 6) - these are FLAGS
        measure_coupling_aux_mx: If True, measure X-basis aux (plaquette 5)
        reset_shared_data: If True, reset the shared data qubit at the start
    """
    r = reset if reset else "-"
    m = measurement if measurement else "-"
    
    # Same boundary structure as ZXZ
    rpng_descriptions = {
        # Corners (empty)
        1: RPNGDescription.empty(),
        2: RPNGDescription.empty(),
        3: RPNGDescription.empty(),
        4: RPNGDescription.empty(),
        
        # Top boundary - X-basis 2-body (positions 2,3 → timings 4,6 from X schedule 7,5,4,6)
        # This is the interface boundary - will be replaced with custom if use_custom_coupling
        5: RPNGDescription.from_string(f"---- ---- {r}x4{m} {r}x6{m}"),
        6: RPNGDescription.empty(),
        
        # Left boundary - Z-basis 2-body (positions 1,3 → timings 3,2 from Z schedule 1,3,4,2)
        7: RPNGDescription.empty(),
        8: RPNGDescription.from_string(f"---- {r}z3{m} ---- {r}z2{m}"),
        
        # Bulk plaquettes (same schedule for all cubes)
        9: RPNGDescription.from_string(f"{r}z1{m} {r}z3{m} {r}z4{m} {r}z2{m}"),   # Z-basis: 1,3,4,2
        10: RPNGDescription.from_string(f"{r}x7{m} {r}x5{m} {r}x4{m} {r}x6{m}"),  # X-basis: 7,5,4,6
        
        # Right boundary - Z-basis 2-body (positions 0,2 → timings 1,4 from Z schedule 1,3,4,2)
        11: RPNGDescription.from_string(f"{r}z1{m} ---- {r}z4{m} ----"),
        12: RPNGDescription.empty(),
        
        # Bottom boundary - X-basis 2-body (positions 0,1 → timings 7,5 from X schedule 7,5,4,6)
        13: RPNGDescription.empty(),
        14: RPNGDescription.from_string(f"{r}x7{m} {r}x5{m} ---- ----"),
    }
    
    plaquette_collection = {
        idx: rpng_to_plaquette(rpng, translator)
        for idx, rpng in rpng_descriptions.items()
    }
    
    # Replace interface plaquettes with custom coupling plaquettes if requested
    # Note: ZXX plaquettes don't measure shared data - only XZZ does
    if use_custom_coupling:
        # Plaquette 1: MZ(aux) - corner, controlled by measure_coupling_aux_mz (FLAG)
        plaquette_collection[1] = create_custom_coupling_plaquette_zxx_1(
            reset_data=reset,
            measurement_data=measurement,
            measure_aux=measure_coupling_aux_mz,
            reset_shared_data=reset_shared_data,
        )
        # Plaquette 5: MX(aux) - controlled by measure_coupling_aux_mx
        plaquette_collection[5] = create_custom_coupling_plaquette_zxx_5(
            reset_data=reset,
            measurement_data=measurement,
            measure_aux=measure_coupling_aux_mx,
            reset_shared_data=reset_shared_data,
        )
        # Plaquette 6: MZ(aux) - controlled by measure_coupling_aux_mz (FLAG)
        plaquette_collection[6] = create_custom_coupling_plaquette_zxx_6(
            reset_data=reset,
            measurement_data=measurement,
            measure_aux=measure_coupling_aux_mz,
            reset_shared_data=reset_shared_data,
        )
    
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
    # Total cycles = 1 (init) + (2k-1) (bulk) + 1 (meas) = 2k+1
    layers = [
        init_layout,
        RepeatedLayer(
            bulk_layout,
            repetitions=LinearFunction(2, -1),  # 2k-1 repetitions
        ),
        meas_layout,
    ]
    return SequencedLayers(layers)


def create_combined_observable(
    cubes: list[tuple[str, tuple[int, int, int]]],
) -> AbstractObservable:
    """
    Create a single AbstractObservable that combines measurements from multiple cubes.
    
    This is used when blocks are coupled and should have a single logical observable
    that is the product of measurements from all cubes.
    
    Args:
        cubes: List of (cube_type, (x, y, z)) tuples
        
    Returns:
        A single AbstractObservable containing all cubes' readouts
    """
    all_cubes = frozenset(
        CubeWithArms(Cube(Position3D(*pos), ZXCube.from_str(cube_type)))
        for cube_type, pos in cubes
    )
    return AbstractObservable(top_readout_cubes=all_cubes)


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


def generate_two_cube_circuit(
    k: int = 2,
    noise_model: NoiseModel | None = None,
    measure_coupling_aux_mz: bool = True,
    measure_coupling_aux_mx: bool = True,
    measure_shared_data: bool = False,
    measure_shared_data_final_only: bool = False,
    manhattan_radius: int = 2,
) -> stim.Circuit:
    """
    Generate a circuit with two decoupled cubes:
    - ZXZ cube at position (0, 0, 0) - stores Z
    - XZX cube at position (1, 0, 0) - stores X
    
    The cubes are decoupled (no pipe connecting them).
    
    Custom coupling plaquettes (after role exchange):
    - Z-basis aux measurements (FLAGS): ZXZ plaquette 11, XZX plaquettes 1, 8
    - X-basis aux measurements: ZXZ plaquettes 2, 12, XZX plaquette 7
    
    Args:
        k: Scaling factor (code distance ≈ 2k+1)
        noise_model: Optional noise model to apply to the circuit
        measure_coupling_aux_mz: If True, measure Z-basis aux (plaquettes 11, 1, 8) - these are FLAGS
        measure_coupling_aux_mx: If True, measure X-basis aux (plaquettes 2, 12, 7)
        measure_shared_data: If True, measure the shared data qubits at the boundary in Z basis
        measure_shared_data_final_only: If True, only measure shared data in the final (meas) layer
        manhattan_radius: Parameter for automatic detector computation. Set to 0 to disable.
        
    Returns:
        Complete stim circuit with detectors and observables for both cubes
    """
    translator = create_translator()
    template = QubitTemplate()
    
    # Create plaquettes for ZXZ cube at (0, 0) - stores Z
    # If measure_shared_data_final_only:
    #   - Only measure shared data in the meas layer
    #   - Only reset shared data in the init layer (not in bulk/meas)
    measure_shared_init_bulk = measure_shared_data and not measure_shared_data_final_only
    measure_shared_meas = measure_shared_data
    
    # Reset shared data: always reset in init, optionally reset in bulk/meas (if not final_only)
    reset_shared_init = True  # Always reset in init to initialize qubits
    reset_shared_bulk = not measure_shared_data_final_only  # Don't reset in bulk if final_only
    reset_shared_meas = not measure_shared_data_final_only  # Don't reset in meas if final_only
    
    zxz_init = create_zxz_plaquettes(
        translator, reset="z", use_custom_coupling=True,
        measure_coupling_aux_mz=measure_coupling_aux_mz,
        measure_coupling_aux_mx=measure_coupling_aux_mx,
        measure_shared_data=measure_shared_init_bulk,
        reset_shared_data=reset_shared_init,
    )
    zxz_bulk = create_zxz_plaquettes(
        translator, use_custom_coupling=True,
        measure_coupling_aux_mz=measure_coupling_aux_mz,
        measure_coupling_aux_mx=measure_coupling_aux_mx,
        measure_shared_data=measure_shared_init_bulk,
        reset_shared_data=reset_shared_bulk,
    )
    zxz_meas = create_zxz_plaquettes(
        translator, measurement="z", use_custom_coupling=True,
        measure_coupling_aux_mz=measure_coupling_aux_mz,
        measure_coupling_aux_mx=measure_coupling_aux_mx,
        measure_shared_data=measure_shared_meas,
        reset_shared_data=reset_shared_meas,
    )
    
    # Create plaquettes for XZX cube at (1, 0) - stores X
    # Note: XZX plaquettes do NOT measure shared data (only ZXZ does)
    xzx_init = create_xzx_plaquettes(
        translator, reset="x", use_custom_coupling=True,
        measure_coupling_aux_mz=measure_coupling_aux_mz,
        measure_coupling_aux_mx=measure_coupling_aux_mx,
        reset_shared_data=reset_shared_init,
    )
    xzx_bulk = create_xzx_plaquettes(
        translator, use_custom_coupling=True,
        measure_coupling_aux_mz=measure_coupling_aux_mz,
        measure_coupling_aux_mx=measure_coupling_aux_mx,
        reset_shared_data=reset_shared_bulk,
    )
    xzx_meas = create_xzx_plaquettes(
        translator, measurement="x", use_custom_coupling=True,
        measure_coupling_aux_mz=measure_coupling_aux_mz,
        measure_coupling_aux_mx=measure_coupling_aux_mx,
        reset_shared_data=reset_shared_meas,
    )
    
    # Create cube plaquettes dict: position -> (init, bulk, meas)
    cube_plaquettes = {
        (0, 0): (zxz_init, zxz_bulk, zxz_meas),  # ZXZ at origin
        (1, 0): (xzx_init, xzx_bulk, xzx_meas),  # XZX to the right
    }
    
    # Create the layer structure
    layers = create_multi_cube_layers(template, cube_plaquettes)
    
    # Create a single combined observable for the merged blocks
    # The logical observable is the product of measurements from both cubes
    combined_observable = create_combined_observable([
        ("ZXZ", (0, 0, 0)),  # ZXZ cube readouts
        ("XZX", (1, 0, 0)),  # XZX cube readouts
    ])
    
    # Create the LayerTree for automatic detector computation
    layer_tree = create_layer_tree(layers, [combined_observable])
    
    # Generate the circuit with detectors (manhattan_radius=0 disables automatic detector computation)
    circuit = layer_tree.generate_circuit(k, manhattan_radius=manhattan_radius)
    
    # Apply noise model if provided
    if noise_model is not None:
        circuit = noise_model.noisy_circuit(circuit)
    
    return circuit


def generate_two_cube_circuit_y_axis(
    k: int = 2,
    noise_model: NoiseModel | None = None,
    measure_coupling_aux_mz: bool = True,
    measure_coupling_aux_mx: bool = True,
    measure_shared_data: bool = False,
    measure_shared_data_final_only: bool = False,
    manhattan_radius: int = 2,
) -> stim.Circuit:
    """
    Generate a Y-axis oriented spatial Hadamard circuit with two cubes:
    - XZZ cube at position (0, 0, 0) - stores Z, XZX-style boundaries
    - ZXX cube at position (0, 1, 0) - stores X, ZXZ-style boundaries
    
    The cubes are neighbors along the y-axis. Interface is between:
    - XZZ bottom boundary (indices 3, 13, 14)
    - ZXX top boundary (indices 1, 5, 6)
    
    Custom coupling plaquettes:
    - Z-basis aux measurements (FLAGS): XZZ plaquette 13, ZXX plaquettes 1, 6
    - X-basis aux measurements: XZZ plaquettes 3, 14, ZXX plaquette 5
    
    Args:
        k: Scaling factor (code distance ≈ 2k+1)
        noise_model: Optional noise model to apply to the circuit
        measure_coupling_aux_mz: If True, measure Z-basis aux (plaquettes 13, 1, 6) - these are FLAGS
        measure_coupling_aux_mx: If True, measure X-basis aux (plaquettes 3, 14, 5)
        measure_shared_data: If True, measure the shared data qubits at the boundary in Z basis
        measure_shared_data_final_only: If True, only measure shared data in the final (meas) layer
        manhattan_radius: Parameter for automatic detector computation. Set to 0 to disable.
        
    Returns:
        Complete stim circuit with detectors and observables for both cubes
    """
    translator = create_translator()
    template = QubitTemplate()
    
    # Determine when to measure and reset shared data
    measure_shared_init_bulk = measure_shared_data and not measure_shared_data_final_only
    measure_shared_meas = measure_shared_data
    
    # Reset shared data: always reset in init, optionally reset in bulk/meas (if not final_only)
    reset_shared_init = True  # Always reset in init to initialize qubits
    reset_shared_bulk = not measure_shared_data_final_only  # Don't reset in bulk if final_only
    reset_shared_meas = not measure_shared_data_final_only  # Don't reset in meas if final_only
    
    # Create plaquettes for XZZ cube at (0, 0) - stores Z, XZX-style boundaries
    xzz_init = create_xzz_plaquettes(
        translator, reset="z", use_custom_coupling=True,
        measure_coupling_aux_mz=measure_coupling_aux_mz,
        measure_coupling_aux_mx=measure_coupling_aux_mx,
        measure_shared_data=measure_shared_init_bulk,
        reset_shared_data=reset_shared_init,
    )
    xzz_bulk = create_xzz_plaquettes(
        translator, use_custom_coupling=True,
        measure_coupling_aux_mz=measure_coupling_aux_mz,
        measure_coupling_aux_mx=measure_coupling_aux_mx,
        measure_shared_data=measure_shared_init_bulk,
        reset_shared_data=reset_shared_bulk,
    )
    xzz_meas = create_xzz_plaquettes(
        translator, measurement="z", use_custom_coupling=True,
        measure_coupling_aux_mz=measure_coupling_aux_mz,
        measure_coupling_aux_mx=measure_coupling_aux_mx,
        measure_shared_data=measure_shared_meas,
        reset_shared_data=reset_shared_meas,
    )
    
    # Create plaquettes for ZXX cube at (0, 1) - stores X, ZXZ-style boundaries
    # Note: ZXX plaquettes do NOT measure shared data (only XZZ does)
    zxx_init = create_zxx_plaquettes(
        translator, reset="x", use_custom_coupling=True,
        measure_coupling_aux_mz=measure_coupling_aux_mz,
        measure_coupling_aux_mx=measure_coupling_aux_mx,
        reset_shared_data=reset_shared_init,
    )
    zxx_bulk = create_zxx_plaquettes(
        translator, use_custom_coupling=True,
        measure_coupling_aux_mz=measure_coupling_aux_mz,
        measure_coupling_aux_mx=measure_coupling_aux_mx,
        reset_shared_data=reset_shared_bulk,
    )
    zxx_meas = create_zxx_plaquettes(
        translator, measurement="x", use_custom_coupling=True,
        measure_coupling_aux_mz=measure_coupling_aux_mz,
        measure_coupling_aux_mx=measure_coupling_aux_mx,
        reset_shared_data=reset_shared_meas,
    )
    
    # Create cube plaquettes dict: position -> (init, bulk, meas)
    cube_plaquettes = {
        (0, 0): (xzz_init, xzz_bulk, xzz_meas),  # XZZ at origin
        (0, 1): (zxx_init, zxx_bulk, zxx_meas),  # ZXX above (y+1)
    }
    
    # Create the layer structure
    layers = create_multi_cube_layers(template, cube_plaquettes)
    
    # Create a single combined observable for the merged blocks
    # For Y-axis: XZZ stores Z (read Z at end), ZXX stores X (read X at end)
    # Use the appropriate cube types for the observable
    combined_observable = create_combined_observable([
        ("XZZ", (0, 0, 0)),  # XZZ cube readouts
        ("ZXX", (0, 1, 0)),  # ZXX cube readouts
    ])
    
    # Create the LayerTree for automatic detector computation
    layer_tree = create_layer_tree(layers, [combined_observable])
    
    # Generate the circuit with detectors (manhattan_radius=0 disables automatic detector computation)
    circuit = layer_tree.generate_circuit(k, manhattan_radius=manhattan_radius)
    
    # Apply noise model if provided
    if noise_model is not None:
        circuit = noise_model.noisy_circuit(circuit)
    
    return circuit


def generate_two_cube_circuit_y_axis_with_flag_detectors(
    k: int = 2,
    noise_model: NoiseModel | None = None,
    measure_shared_data_final_only: bool = False,
) -> stim.Circuit:
    """
    Generate a Y-axis two-cube circuit with proper flag detectors.
    
    This is a convenience wrapper around generate_spatial_hadamard_circuit(axis='y').
    See that function for full documentation.
    
    Args:
        k: Scaling factor (code distance ≈ 2k+1)
        noise_model: Optional noise model to apply to the circuit
        measure_shared_data_final_only: If True, only measure/reset shared data at first/last round
        
    Returns:
        Complete stim circuit with flag detectors for the Y-axis spatial Hadamard
    """
    return generate_spatial_hadamard_circuit(
        k=k,
        axis='y',
        noise_model=noise_model,
        measure_shared_data_final_only=measure_shared_data_final_only,
    )


def generate_spatial_hadamard_circuit(
    k: int = 2,
    axis: str = 'x',
    noise_model: NoiseModel | None = None,
    measure_shared_data_final_only: bool = False,
    flag_config: str | None = None,
) -> stim.Circuit:
    """
    Generate a spatial Hadamard circuit with two cubes along the specified axis.
    
    This is the unified interface for generating spatial Hadamard circuits.
    Uses a two-pass approach for proper flag detector handling.
    
    For X-axis (axis='x'):
        - ZXZ cube at (0, 0, 0) stores Z
        - XZX cube at (1, 0, 0) stores X
        - Known parity plaquettes: (7, 11)
    
    For Y-axis (axis='y'):
        - XZZ cube at (0, 0, 0) stores Z
        - ZXX cube at (0, 1, 0) stores X
        - Known parity plaquettes: (5, 13)
    
    Args:
        k: Scaling factor (code distance ≈ 2k+1)
        axis: 'x' or 'y' - direction of the two-cube layout
        noise_model: Optional noise model to apply to the circuit
        measure_shared_data_final_only: If True, only measure/reset shared data at first/last round
            (deprecated, use flag_config instead)
        flag_config: Flag measurement configuration:
            - 'all': flags measured every round (measure_shared_data_final_only=False)
            - 'partial': flags measured only in final round (measure_shared_data_final_only=True)
            - 'none': no flags at all (measure_coupling_aux_mz=False, measure_shared_data=False)
            - None: use measure_shared_data_final_only parameter for backwards compatibility
        
    Returns:
        Complete stim circuit with:
        - Core detectors mapped from the non-flag circuit
        - Tagged single-measurement detectors for each flag measurement
        - Properly mapped observable
    """
    # Handle flag_config parameter
    # Determine whether to include MZ flags and shared data measurements
    include_mz_flags = True  # Default: include MZ aux measurements (flags)
    include_shared_data = True  # Default: include shared data measurements
    
    if flag_config is not None:
        if flag_config == 'all':
            measure_shared_data_final_only = False
        elif flag_config == 'partial':
            measure_shared_data_final_only = True
        elif flag_config == 'none':
            # No MZ flags or shared data measurements
            measure_shared_data_final_only = False
            include_mz_flags = False
            include_shared_data = False
        else:
            raise ValueError(f"flag_config must be 'all', 'partial', or 'none', got '{flag_config}'")
    
    # Select the appropriate base circuit generator and parameters based on axis
    if axis == 'x':
        base_generator = generate_two_cube_circuit
        known_parity_plaquettes = (7, 11)
    elif axis == 'y':
        base_generator = generate_two_cube_circuit_y_axis
        known_parity_plaquettes = (5, 13)
    else:
        raise ValueError(f"axis must be 'x' or 'y', got '{axis}'")
    
    # Step 1: Generate circuit WITHOUT any stretched stabilizer measurements
    circuit_no_flags = base_generator(
        k=k,
        noise_model=None,
        measure_coupling_aux_mz=False,
        measure_coupling_aux_mx=False,
        measure_shared_data=False,
        measure_shared_data_final_only=measure_shared_data_final_only,
        manhattan_radius=2,
    )
    
    # Step 2: Generate circuit WITH stretched stabilizer measurements
    # For 'none' config: no MZ flags or shared data, but still MX aux
    circuit_with_flags = base_generator(
        k=k,
        noise_model=None,
        measure_coupling_aux_mz=include_mz_flags,
        measure_coupling_aux_mx=True,
        measure_shared_data=include_shared_data,
        measure_shared_data_final_only=measure_shared_data_final_only,
        manhattan_radius=0,
    )
    
    # Step 3: Map detectors and add flag detectors
    # Interface coordinates are the same formula for both axes
    interface_min = 4 * k + 2
    interface_max = 4 * k + 4
    
    circuit_with_detectors = add_mapped_detectors_and_flag_detectors(
        circuit_with_flags,
        circuit_no_flags,
        known_parity_plaquettes=known_parity_plaquettes,
        interface_coord_range=(interface_min, interface_max),
        interface_axis=axis,
        flag_detector_tag="flag",
    )
    
    # Step 4: Apply noise if provided
    if noise_model is not None:
        circuit_with_detectors = noise_model.noisy_circuit(circuit_with_detectors)
    
    return circuit_with_detectors


# ============================================================================
# Two-pass detector generation for flag measurements
# ============================================================================

def _get_measurement_qubits(circuit: stim.Circuit) -> list[int]:
    """
    Extract the list of qubit indices for each measurement in the circuit.
        
    Returns:
        List of qubit indices, one per measurement in circuit order.
    """
    measurement_qubits = []
    for inst in circuit.flattened():
        if inst.name in ('M', 'MX', 'MY', 'MZ', 'MR', 'MRX', 'MRY', 'MRZ'):
            for target in inst.targets_copy():
                measurement_qubits.append(target.value)
    return measurement_qubits


def build_measurement_mapping(
    circuit_no_flags: stim.Circuit,
    circuit_with_flags: stim.Circuit,
) -> tuple[dict[int, int], set[int]]:
    """
    Build a mapping from measurement indices in the no-flag circuit to the with-flag circuit.
    
    The key insight is that non-flag measurements appear in the same relative order in both
    circuits. Flag measurements are "extra" measurements that appear only in the with-flag circuit.
    
    Args:
        circuit_no_flags: Circuit without flag measurements
        circuit_with_flags: Circuit with flag measurements
        
    Returns:
        Tuple of:
        - mapping: Dict mapping no-flag measurement index -> with-flag measurement index
        - flag_indices: Set of measurement indices in with-flag circuit that are flag measurements
    """
    # Get measurement qubit lists
    meas_qubits_no_flags = _get_measurement_qubits(circuit_no_flags)
    meas_qubits_with_flags = _get_measurement_qubits(circuit_with_flags)
    
    # The non-flag qubits are those that appear in the no-flag circuit
    # We match measurements by qubit index in order
    mapping: dict[int, int] = {}
    flag_indices: set[int] = set()
    
    no_flag_idx = 0
    for with_flag_idx, qubit in enumerate(meas_qubits_with_flags):
        if no_flag_idx < len(meas_qubits_no_flags) and qubit == meas_qubits_no_flags[no_flag_idx]:
            # This measurement exists in both circuits - it's a non-flag measurement
            mapping[no_flag_idx] = with_flag_idx
            no_flag_idx += 1
        else:
            # This measurement only exists in the with-flag circuit - it's a flag measurement
            flag_indices.add(with_flag_idx)
    
    # Verify we matched all no-flag measurements
    if no_flag_idx != len(meas_qubits_no_flags):
        raise ValueError(
            f"Could not match all no-flag measurements. "
            f"Matched {no_flag_idx}/{len(meas_qubits_no_flags)}"
        )
    
    return mapping, flag_indices


def _extract_detectors_from_circuit(circuit: stim.Circuit) -> list[tuple[list[int], tuple[float, ...]]]:
    """
    Extract all detectors from a circuit.
        
    Returns:
        List of (measurement_offsets, coords) tuples for each detector.
        measurement_offsets are absolute indices (0-indexed from start of circuit).
    """
    detectors = []
    current_measurement_count = 0
    
    for inst in circuit.flattened():
        if inst.name in ('M', 'MX', 'MY', 'MZ', 'MR', 'MRX', 'MRY', 'MRZ'):
            current_measurement_count += len(inst.targets_copy())
        elif inst.name == 'DETECTOR':
            # Extract measurement record offsets and convert to absolute indices
            measurement_offsets = []
            for target in inst.targets_copy():
                if target.is_measurement_record_target:
                    # Convert relative offset to absolute index
                    abs_idx = current_measurement_count + target.value
                    measurement_offsets.append(abs_idx)
            # Get detector coordinates
            coords = tuple(inst.gate_args_copy())
            detectors.append((measurement_offsets, coords))
    
    return detectors


def _get_measurement_info(circuit: stim.Circuit) -> list[dict]:
    """
    Extract detailed measurement info from circuit.
        
    Returns:
        List of dicts with keys: 'qubit', 'type', 'index', 'coords'
        where coords is (x, y) from QUBIT_COORDS
    """
    # First pass: collect qubit coordinates
    qubit_coords = {}
    for inst in circuit.flattened():
        if inst.name == 'QUBIT_COORDS':
            args = inst.gate_args_copy()
            targets = [t.value for t in inst.targets_copy() if t.is_qubit_target]
            for t in targets:
                qubit_coords[t] = (args[0], args[1])
    
    # Second pass: collect measurement info
    measurements = []
    meas_idx = 0
    for inst in circuit.flattened():
        if inst.name in ('M', 'MX', 'MY', 'MZ', 'MR', 'MRX', 'MRY', 'MRZ'):
            for target in inst.targets_copy():
                qubit = target.value
                measurements.append({
                    'qubit': qubit,
                    'type': inst.name,
                    'index': meas_idx,
                    'coords': qubit_coords.get(qubit, (None, None)),
                })
                meas_idx += 1
    
    return measurements


def _identify_known_parity_aux_qubits(
    circuit: stim.Circuit,
    known_parity_plaquettes: tuple[int, int],
    interface_coord_range: tuple[float, float],
    interface_axis: str,
) -> set[int]:
    """
    Identify the MX aux qubits that belong to known parity stretched stabilizers.
    
    For known parity plaquettes (e.g., 7 and 11 for X-axis, 5 and 13 for Y-axis),
    we need to find the MX aux qubits at the interface.
    
    The known parity stretched stabilizers are formed by pairs of plaquettes:
    - X-axis: plaquettes 7 (XZX, MZ aux) + 11 (ZXZ, MX aux) at y=0 (Z-boundary)
    - Y-axis: plaquettes 5 (ZXX, MX aux) + 13 (XZZ, MZ aux) at non-corner positions
    
    For Y-axis: The MX aux of plaquette 5 (ZXX) is at positions where plaquette 5 occurs,
    i.e., at x = 2, 6, 10, ... (starting from 2, every 4 units). These are NOT at corners.
    The interface y-coordinate for the MX aux is at the max of the interface range (in ZXX cube).
    
    For X-axis: The MX aux of plaquette 11 (ZXZ) is at y=0 (Z-boundary), interface x at min.
    
    Args:
        circuit: The circuit to analyze
        known_parity_plaquettes: Tuple of plaquette indices that have known parity
        interface_coord_range: (min, max) coordinate range for the interface
        interface_axis: 'x' or 'y' - which axis the interface is along
        
    Returns:
        Set of qubit indices for the known parity MX aux qubits
    """
    # Get qubit coordinates
    qubit_coords = {}
    for inst in circuit.flattened():
        if inst.name == 'QUBIT_COORDS':
            args = inst.gate_args_copy()
            targets = [t.value for t in inst.targets_copy() if t.is_qubit_target]
            for t in targets:
                qubit_coords[t] = (args[0], args[1])
    
    # Find MX measurements at the interface
    meas_info = _get_measurement_info(circuit)
    coord_min, coord_max = interface_coord_range
    
    # For stretched stabilizers, the MX aux is on one side of the interface
    # and the MZ aux is on the other side.
    # For Y-axis: MX aux (plaquette 5, ZXX) is at y = coord_max (top of interface)
    # For X-axis: MX aux (plaquette 11, ZXZ) is at x = coord_min (left side of interface)
    
    if interface_axis == 'y':
        # Y-axis interface: MX aux is at y = coord_max (ZXX cube side)
        mx_interface_coord = coord_max
    else:
        # X-axis interface: MX aux is at x = coord_min (ZXZ cube side)
        mx_interface_coord = coord_min
    
    # Find all MX aux at the interface on the correct side
    mx_aux_at_interface = {}  # qubit -> perpendicular coordinate
    for m in meas_info:
        x, y = m['coords']
        if x is None:
            continue
        coord = x if interface_axis == 'x' else y
        perp_coord = y if interface_axis == 'x' else x
        
        # Check if this is an MX measurement at the correct interface position
        if coord == mx_interface_coord and m['type'] == 'MX':
            q = m['qubit']
            if q not in mx_aux_at_interface:
                mx_aux_at_interface[q] = perp_coord
    
    if not mx_aux_at_interface:
        return set()
    
    # For known parity stretched stabilizers, we need to exclude corners
    # Corners are at min and max of the perpendicular coordinate
    perp_coords = set(mx_aux_at_interface.values())
    perp_min = min(perp_coords)
    perp_max = max(perp_coords)
    
    # Known parity MX aux are those NOT at corners
    # (i.e., not at perp_min or perp_max which would be corners)
    known_parity_aux = set()
    for q, perp_coord in mx_aux_at_interface.items():
        if perp_coord != perp_min and perp_coord != perp_max:
            known_parity_aux.add(q)
    
    return known_parity_aux


def _get_data_qubits_for_known_parity_stretched_stabilizer(
    circuit: stim.Circuit,
    mx_aux_qubit: int,
    interface_coord_range: tuple[float, float],
    interface_axis: str,
) -> list[int]:
    """
    Find ALL data qubits for a known parity stretched stabilizer.
    
    This includes data qubits from BOTH the MX aux (e.g., plaquette 7) AND
    its paired MZ aux (e.g., plaquette 11), but EXCLUDES the shared data qubit
    at the interface.
    
    For X-axis (interface at x=10-12):
    - MX aux at x=12, y=Y is paired with MZ aux at x=10, y=Y
    - Returns 4 data qubits: 2 from each side, excluding shared at x=11
    
    For Y-axis (interface at y=4-6):
    - MX aux at y=6, x=X is paired with MZ aux at y=4, x=X
    - Returns 4 data qubits: 2 from each side, excluding shared at y=5
    
    Args:
        circuit: The circuit to analyze
        mx_aux_qubit: The MX aux qubit index (at coord_max side)
        interface_coord_range: (min, max) coordinate range for the interface
        interface_axis: 'x' or 'y' - which axis the interface is along
        
    Returns:
        List of 4 data qubit indices (2 from each plaquette, excluding shared)
    """
    # Get qubit coordinates
    qubit_coords = {}
    for inst in circuit.flattened():
        if inst.name == 'QUBIT_COORDS':
            args = inst.gate_args_copy()
            targets = [t.value for t in inst.targets_copy() if t.is_qubit_target]
            for t in targets:
                qubit_coords[t] = (args[0], args[1])
    
    coord_min, coord_max = interface_coord_range
    
    # Get MX aux coordinates
    if mx_aux_qubit not in qubit_coords:
        return []
    mx_aux_x, mx_aux_y = qubit_coords[mx_aux_qubit]
    
    # Find the paired MZ aux (same perpendicular coordinate, but at coord_min)
    # For X-axis: MX aux at (12, Y) pairs with MZ aux at (10, Y)
    # For Y-axis: MX aux at (X, 6) pairs with MZ aux at (X, 4)
    mz_aux_qubit = None
    for q, (x, y) in qubit_coords.items():
        if interface_axis == 'x':
            # MZ aux should be at x=coord_min with same y
            if x == coord_min and y == mx_aux_y:
                mz_aux_qubit = q
                break
        else:  # y-axis
            # MZ aux should be at y=coord_min with same x
            if y == coord_min and x == mx_aux_x:
                mz_aux_qubit = q
                break
    
    # Find data qubits for both aux via CX/CZ interactions
    all_data_qubits = set()
    for inst in circuit.flattened():
        if inst.name in ('CX', 'CZ'):
            targets = [t.value for t in inst.targets_copy() if t.is_qubit_target]
            for i in range(0, len(targets), 2):
                if i + 1 < len(targets):
                    ctrl, tgt = targets[i], targets[i+1]
                    # Check if this involves our aux qubits
                    if ctrl == mx_aux_qubit or tgt == mx_aux_qubit:
                        other = tgt if ctrl == mx_aux_qubit else ctrl
                        all_data_qubits.add(other)
                    if mz_aux_qubit is not None and (ctrl == mz_aux_qubit or tgt == mz_aux_qubit):
                        other = tgt if ctrl == mz_aux_qubit else ctrl
                        all_data_qubits.add(other)
    
    # Remove the paired aux qubits themselves (they might interact with each other via shared data)
    all_data_qubits.discard(mx_aux_qubit)
    if mz_aux_qubit is not None:
        all_data_qubits.discard(mz_aux_qubit)
    
    # Remove shared data qubits (those at the interface middle)
    # For X-axis: shared data is at x=11 (between 10 and 12)
    # For Y-axis: shared data is at y=5 (between 4 and 6)
    non_shared = []
    for dq in all_data_qubits:
        if dq in qubit_coords:
            x, y = qubit_coords[dq]
            coord = x if interface_axis == 'x' else y
            # Exclude qubits strictly within the interface range
            if coord < coord_min or coord > coord_max:
                non_shared.append(dq)
    
    return sorted(non_shared)


def _get_final_data_measurement_indices(
    meas_info: list[dict],
    data_qubits: list[int],
) -> list[int]:
    """
    Get the final measurement indices for specified data qubits.
    
    Returns:
        List of measurement indices (the last measurement of each data qubit)
    """
    # Find the last measurement for each data qubit
    last_meas = {}
    for m in meas_info:
        if m['qubit'] in data_qubits:
            last_meas[m['qubit']] = m['index']
    
    return [last_meas[dq] for dq in data_qubits if dq in last_meas]


def add_mapped_detectors_and_flag_detectors(
    circuit_with_flags: stim.Circuit,
    circuit_no_flags: stim.Circuit,
    known_parity_plaquettes: tuple[int, int],
    interface_coord_range: tuple[float, float],
    interface_axis: str,
    flag_detector_tag: str = "flag",
) -> stim.Circuit:
    """
    Add detectors to the with-flag circuit based on the no-flag circuit detectors,
    plus manually constructed detectors for ALL stretched stabilizer measurements.
    
    The base circuit (no_flags) has NO stretched stabilizer measurements at all,
    so automatic detector generation only covers regular plaquettes.
    
    Detector structure for stretched stabilizers (all added manually):
    1. MZ measurements: single-measurement flag detectors (tagged)
    2. MX consecutive pairs: MX(N) × MX(N+1) paired detectors
    3. For known parity stretched stabilizers only:
       - First MX: single-measurement detector
       - Final MX × final data qubit measurements: detector
    
    Args:
        circuit_with_flags: Circuit with ALL measurements but no detectors (manhattan_radius=0)
        circuit_no_flags: Circuit WITHOUT stretched stabilizer measurements, with automatic detectors
        known_parity_plaquettes: Tuple of plaquette indices that have known parity
        interface_coord_range: (min, max) coordinate range for the interface
        interface_axis: 'x' or 'y' - which axis the interface is along
        flag_detector_tag: Tag to apply to flag detectors (default: "flag")
        
    Returns:
        New circuit with properly mapped detectors and stretched stabilizer detectors.
    """
    # Build the measurement mapping
    mapping, flag_indices = build_measurement_mapping(circuit_no_flags, circuit_with_flags)
    
    # Extract detectors from no-flag circuit
    no_flag_detectors = _extract_detectors_from_circuit(circuit_no_flags)
    
    # Also extract observable instructions from no-flag circuit
    observables = []
    current_measurement_count = 0
    for inst in circuit_no_flags.flattened():
        if inst.name in ('M', 'MX', 'MY', 'MZ', 'MR', 'MRX', 'MRY', 'MRZ'):
            current_measurement_count += len(inst.targets_copy())
        elif inst.name == 'OBSERVABLE_INCLUDE':
            measurement_offsets = []
            for target in inst.targets_copy():
                if target.is_measurement_record_target:
                    abs_idx = current_measurement_count + target.value
                    measurement_offsets.append(abs_idx)
            obs_idx = int(inst.gate_args_copy()[0])
            observables.append((measurement_offsets, obs_idx))
    
    # Get total measurements in with-flag circuit
    total_measurements_with_flags = circuit_with_flags.num_measurements
    
    # Get detailed measurement info for with-flag circuit
    meas_info = _get_measurement_info(circuit_with_flags)
    
    # Identify known parity aux qubits
    known_parity_aux = _identify_known_parity_aux_qubits(
        circuit_with_flags, known_parity_plaquettes, interface_coord_range, interface_axis
    )
    
    # ALL stretched stabilizer measurements are in flag_indices (since base has none)
    # Separate them into MZ and MX measurements
    coord_min, coord_max = interface_coord_range
    mz_flag_indices = []  # MZ measurements (single-measurement flags)
    mx_by_qubit = {}  # qubit -> list of measurement indices (for consecutive pairing)
    
    for flag_idx in flag_indices:
        m = meas_info[flag_idx]
        x, y = m['coords']
        if x is None:
            continue
        coord = x if interface_axis == 'x' else y
        if coord_min <= coord <= coord_max:
            if m['type'] in ('M', 'MZ'):
                mz_flag_indices.append(flag_idx)
            elif m['type'] == 'MX':
                q = m['qubit']
                if q not in mx_by_qubit:
                    mx_by_qubit[q] = []
                mx_by_qubit[q].append(flag_idx)
    
    # Build the new circuit
    new_circuit = stim.Circuit()
    
    # Copy all instructions except DETECTOR and OBSERVABLE_INCLUDE
    for inst in circuit_with_flags.flattened():
        if inst.name not in ('DETECTOR', 'OBSERVABLE_INCLUDE'):
            new_circuit.append(inst)
    
    # 1. Add mapped detectors from no-flag circuit (these are the "core" detectors)
    for meas_offsets, coords in no_flag_detectors:
        new_offsets = []
        for old_idx in meas_offsets:
            if old_idx in mapping:
                new_offsets.append(mapping[old_idx])
            else:
                raise ValueError(f"Measurement index {old_idx} not found in mapping")
        
        relative_offsets = [idx - total_measurements_with_flags for idx in new_offsets]
        targets = [stim.target_rec(offset) for offset in relative_offsets]
        new_circuit.append("DETECTOR", targets, list(coords))
    
    # 2. Add MZ flag detectors (single-measurement, tagged)
    for flag_idx in sorted(mz_flag_indices):
        m = meas_info[flag_idx]
        x, y = m['coords']
        relative_offset = flag_idx - total_measurements_with_flags
        detector_inst = stim.CircuitInstruction(
            "DETECTOR",
            [stim.target_rec(relative_offset)],
            [float(x), float(y), 0.0],
            tag=flag_detector_tag,
        )
        new_circuit.append(detector_inst)
    
    # 3. Add MX detectors for each aux qubit
    # For stretched stabilizers, the automatic detector generation creates:
    # - Initial single-measurement detector for MX aux on the "far" side of interface
    # - Consecutive pairs for all MX aux
    # The "far" side depends on interface axis: x=max for x-axis, y=max for y-axis
    
    # Determine which aux qubits should get initial single detectors
    # These are aux at the "far" coordinate of the interface (opposite side from cube origin)
    far_coord = coord_max  # e.g., x=12 for x-axis interface
    
    for qubit, indices in mx_by_qubit.items():
        sorted_indices = sorted(indices)
        m_first = meas_info[sorted_indices[0]]
        x, y = m_first['coords']
        qubit_coord = x if interface_axis == 'x' else y
        
        # Add initial single-measurement detector for aux at the far side of interface
        if qubit_coord == far_coord and sorted_indices:
            first_idx = sorted_indices[0]
            rel_offset = first_idx - total_measurements_with_flags
            new_circuit.append("DETECTOR", [stim.target_rec(rel_offset)], [float(x), float(y), 0.0])
        
        # Consecutive pairs: MX(N) × MX(N+1)
        for i in range(len(sorted_indices) - 1):
            idx1 = sorted_indices[i]
            idx2 = sorted_indices[i + 1]
            m = meas_info[idx2]
            x, y = m['coords']
            rel1 = idx1 - total_measurements_with_flags
            rel2 = idx2 - total_measurements_with_flags
            targets = [stim.target_rec(rel1), stim.target_rec(rel2)]
            new_circuit.append("DETECTOR", targets, [float(x), float(y), 0.0])
    
    # 4. Add mapped observables
    for meas_offsets, obs_idx in observables:
        new_offsets = []
        for old_idx in meas_offsets:
            if old_idx in mapping:
                new_offsets.append(mapping[old_idx])
            else:
                raise ValueError(f"Observable measurement index {old_idx} not found in mapping")
        
        relative_offsets = [idx - total_measurements_with_flags for idx in new_offsets]
        targets = [stim.target_rec(offset) for offset in relative_offsets]
        new_circuit.append("OBSERVABLE_INCLUDE", targets, [float(obs_idx)])
    
    # 5. Add final MX × data qubit detectors for known parity stretched stabilizers
    # For known parity aux: final MX × final measurements of all 4 data qubits from both plaquettes
    # (excluding the shared data qubit at the interface)
    for qubit, indices in mx_by_qubit.items():
        sorted_indices = sorted(indices)
        m_first = meas_info[sorted_indices[0]]
        x, y = m_first['coords']
        qubit_coord = x if interface_axis == 'x' else y
        
        # Known parity aux are at the "far" side of interface (e.g., x=12 for x-axis)
        if qubit_coord == coord_max and sorted_indices:
            final_mx_idx = sorted_indices[-1]
            m = meas_info[final_mx_idx]
            x, y = m['coords']
            
            # Get data qubits from BOTH plaquettes (MX aux + paired MZ aux), excluding shared
            data_qubits = _get_data_qubits_for_known_parity_stretched_stabilizer(
                circuit_with_flags, qubit, interface_coord_range, interface_axis
            )
            
            # Get final measurement indices for these data qubits
            data_meas_indices = _get_final_data_measurement_indices(meas_info, data_qubits)
            
            if data_meas_indices:
                # Build detector: final MX × final data measurements (should be 5 total: 1 aux + 4 data)
                targets = [stim.target_rec(final_mx_idx - total_measurements_with_flags)]
                for data_idx in data_meas_indices:
                    targets.append(stim.target_rec(data_idx - total_measurements_with_flags))
                new_circuit.append("DETECTOR", targets, [float(x), float(y), 0.0])
    
    return new_circuit


def get_flag_detector_indices_from_circuit(
    circuit: stim.Circuit,
    flag_detector_tag: str = "flag",
) -> set[int]:
    """
    Extract flag detector indices from a circuit by looking for tagged detectors.
    
    Args:
        circuit: A stim circuit with tagged flag detectors
        flag_detector_tag: The tag used to identify flag detectors
        
    Returns:
        Set of detector indices that have the flag tag
    """
    flag_indices = set()
    detector_idx = 0
    
    for inst in circuit.flattened():
        if inst.name == "DETECTOR":
            # Check if this detector has the flag tag
            # The tag is accessible via the instruction's string representation
            inst_str = str(inst)
            if f"[{flag_detector_tag}]" in inst_str:
                flag_indices.add(detector_idx)
            detector_idx += 1
    
    return flag_indices


def generate_two_cube_circuit_with_flag_detectors(
    k: int = 2,
    noise_model: NoiseModel | None = None,
    measure_shared_data_final_only: bool = False,
) -> stim.Circuit:
    """
    Generate an X-axis two-cube circuit with proper flag detectors.
    
    This is a convenience wrapper around generate_spatial_hadamard_circuit(axis='x').
    See that function for full documentation.
    
    Args:
        k: Scaling factor (code distance ≈ 2k+1)
        noise_model: Optional noise model to apply to the circuit
        measure_shared_data_final_only: If True, only measure/reset shared data at first/last round
        
    Returns:
        Complete stim circuit with flag detectors for the X-axis spatial Hadamard
    """
    return generate_spatial_hadamard_circuit(
        k=k,
        axis='x',
        noise_model=noise_model,
        measure_shared_data_final_only=measure_shared_data_final_only,
    )


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
            dont_explore_detection_event_sets_with_size_above=6,
            dont_explore_edges_with_degree_above=9999,
            dont_explore_edges_increasing_symptom_degree=False,
            canonicalize_circuit_errors=True
        )
        return len(logical_errors)
    except Exception as e:
        print(f"Error calculating circuit distance: {e}")
        return None


def save_circuit_to_file(
    circuit: stim.Circuit,
    filepath: str,
) -> None:
    """Save a stim circuit to a file."""
    with open(filepath, 'w') as f:
        f.write(str(circuit))
    print(f"Circuit saved to: {filepath}")


def main():
    """Generate and analyze two-cube spatial Hadamard circuits (X-axis and Y-axis)."""
    k = 2
    noise_level = 0.001
    noise_model = NoiseModel.uniform_depolarizing(noise_level)
    
    # Configuration for each axis
    axis_configs = {
        'x': {'name': 'X-Axis', 'cubes': 'ZXZ + XZX', 'filename': 'spatial_hadamard_x_axis.stim'},
        'y': {'name': 'Y-Axis', 'cubes': 'XZZ + ZXX', 'filename': 'spatial_hadamard_y_axis.stim'},
    }
    
    circuits = {}
    
    for axis, config in axis_configs.items():
        print("=" * 70)
        print(f"Spatial Hadamard: {config['name']} ({config['cubes']})")
        print("=" * 70)
        print(f"k = {k} (code distance ≈ {2*k+1})")
        print(f"Noise level: {noise_level}")
        print()
        
        # Generate circuit with flag detectors
        print(f"Generating {config['name']} two-cube circuit with flag detectors...")
        circuit = generate_spatial_hadamard_circuit(
            k=k,
            axis=axis,
            noise_model=noise_model,
            measure_shared_data_final_only=False,
        )
        circuits[axis] = circuit
        
        # Extract flag detector info
        flag_indices = get_flag_detector_indices_from_circuit(circuit)
        print(f"Flag detectors: {len(flag_indices)} (tagged with 'flag')")
        
        # Verify no missing detectors
        missing = circuit.missing_detectors()
        if missing.num_detectors > 0:
            print(f"WARNING: {missing.num_detectors} missing detectors found!")
        else:
            print("All measurements covered by detectors ✓")
        
        print()
        print("Circuit Statistics:")
        print(f"  Instructions: {len(circuit)}")
        print(f"  Qubits: {circuit.num_qubits}")
        print(f"  Detectors: {circuit.num_detectors}")
        print(f"  Observables: {circuit.num_observables}")
        print(f"  Measurements: {circuit.num_measurements}")
        
        print()
        print("Distance Calculations:")
        graphlike_dist = calculate_graphlike_distance(circuit)
        print(f"  Graph-like distance: {graphlike_dist}")
        circuit_dist = calculate_circuit_distance(circuit)
        print(f"  Circuit distance: {circuit_dist}")
        
        # Save circuit
        save_circuit_to_file(circuit, config['filename'])
        print()
    
    # Print Crumble URLs
    print("=" * 70)
    print("Crumble URLs")
    print("=" * 70)
    for axis, config in axis_configs.items():
        print(f"{config['name']} circuit:")
        print(shift_to_only_positive(circuits[axis]).to_crumble_url())
        print()


if __name__ == "__main__":
    main()
