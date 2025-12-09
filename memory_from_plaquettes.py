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
) -> Plaquette:
    """
    Create a custom coupling plaquette for index 11 (right boundary of ZXZ cube).
    
    This plaquette interacts with the bottom-right data qubit twice, which cannot
    be expressed in RPNG notation.
    
    Circuit (most CX/CZ gates delayed by 1 to avoid conflicts):
        Step 0: RX(aux), RZ(bottom-right)
        Step 1: CX(aux, bottom-right)      # aux is control (NOT delayed)
        Step 4: CZ(aux, bottom-left)
        Step 7: CZ(aux, top-left)
        Step 8: CX(bottom-right, aux)      # data qubit is control!
        Step 9: MZ(aux)
    
    Qubit layout (SquarePlaquetteQubits):
        - Index 0: top-left (-1, -1)
        - Index 1: top-right (1, -1)  -- NOT USED
        - Index 2: bottom-left (-1, 1)
        - Index 3: bottom-right (1, 1)
        - Index 4: auxiliary/syndrome (0, 0)
    
    Args:
        reset_data: Reset basis for data qubits ('z' or None for no reset)
        measurement_data: Measurement basis for data qubits ('z' or None for no measurement)
        
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
    
    # Step 0: Reset auxiliary in X basis, reset shared data qubit in Z basis
    circuit.append("RX", [AUX], [])
    circuit.append("RZ", [BOTTOM_RIGHT], [])  # Shared qubit - always reset in Z basis
    circuit.append("TICK", [], [])
    
    # Step 1: CX between auxiliary (control) and bottom-right (target) - NOT delayed
    circuit.append("CX", [AUX, BOTTOM_RIGHT], [])
    circuit.append("TICK", [], [])
    
    # Steps 2, 3: (empty)
    circuit.append("TICK", [], [])
    circuit.append("TICK", [], [])
    
    # Step 4: CZ between auxiliary and bottom-left
    circuit.append("CZ", [AUX, BOTTOM_LEFT], [])
    circuit.append("TICK", [], [])
    
    # Steps 5, 6: (empty)
    circuit.append("TICK", [], [])
    circuit.append("TICK", [], [])
    
    # Step 7: CZ between auxiliary and top-left
    circuit.append("CZ", [AUX, TOP_LEFT], [])
    circuit.append("TICK", [], [])
    
    # Step 8: CX between bottom-right (control) and auxiliary (target)
    circuit.append("CX", [BOTTOM_RIGHT, AUX], [])
    circuit.append("TICK", [], [])
    
    # Step 9: Measure auxiliary in Z basis
    circuit.append("MZ", [AUX], [])
    
    # Create scheduled circuit with explicit schedule
    schedule = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
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
    name = f"coupling_11_r{r_str}_m{m_str}"
    
    return Plaquette(
        name=name,
        qubits=plaquette_qubits,
        circuit=filtered_circuit,
        mergeable_instructions=frozenset(["RX", "RZ", "MZ", "MX", "H"]),
    )


def create_custom_coupling_plaquette_2(
    reset_data: str | None = None,
    measurement_data: str | None = None,
) -> Plaquette:
    """
    Create a custom coupling plaquette for index 2 (corner of ZXZ cube).
    
    Same as plaquette 12 but without the gate towards top-left.
    This plaquette interacts with the bottom-right data qubit twice.
    
    Circuit:
        Step 0: RX(aux), RZ(bottom-right)
        Step 1: CX(aux, bottom-right)      # aux is control (NOT delayed)
        Step 4: CX(aux, bottom-left)
        Step 8: CX(bottom-right, aux)      # data qubit is control!
        Step 9: MZ(aux)
    
    Qubit layout (SquarePlaquetteQubits):
        - Index 0: top-left (-1, -1)  -- NOT USED
        - Index 1: top-right (1, -1)  -- NOT USED
        - Index 2: bottom-left (-1, 1)
        - Index 3: bottom-right (1, 1)
        - Index 4: auxiliary/syndrome (0, 0)
    
    Args:
        reset_data: Reset basis for data qubits ('z' or None for no reset)
        measurement_data: Measurement basis for data qubits ('z' or None for no measurement)
        
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
    
    # Step 0: Reset auxiliary in X basis, reset shared data qubit in Z basis
    circuit.append("RX", [AUX], [])
    circuit.append("RZ", [BOTTOM_RIGHT], [])  # Shared qubit - always reset in Z basis
    circuit.append("TICK", [], [])
    
    # Step 1: CX between auxiliary (control) and bottom-right (target) - NOT delayed
    circuit.append("CX", [AUX, BOTTOM_RIGHT], [])
    circuit.append("TICK", [], [])
    
    # Steps 2, 3: (empty - no top-left gate)
    circuit.append("TICK", [], [])
    circuit.append("TICK", [], [])
    
    # Step 4: CX between auxiliary and bottom-left
    circuit.append("CX", [AUX, BOTTOM_LEFT], [])
    circuit.append("TICK", [], [])
    
    # Steps 5, 6, 7: (empty)
    circuit.append("TICK", [], [])
    circuit.append("TICK", [], [])
    circuit.append("TICK", [], [])
    
    # Step 8: CX between bottom-right (control) and auxiliary (target)
    circuit.append("CX", [BOTTOM_RIGHT, AUX], [])
    circuit.append("TICK", [], [])
    
    # Step 9: Measure auxiliary in Z basis
    circuit.append("MZ", [AUX], [])
    
    # Create scheduled circuit with explicit schedule
    schedule = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
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
    name = f"coupling_2_r{r_str}_m{m_str}"
    
    return Plaquette(
        name=name,
        qubits=plaquette_qubits,
        circuit=filtered_circuit,
        mergeable_instructions=frozenset(["RX", "RZ", "MZ", "MX", "H"]),
    )


def create_custom_coupling_plaquette_12(
    reset_data: str | None = None,
    measurement_data: str | None = None,
) -> Plaquette:
    """
    Create a custom coupling plaquette for index 12 (right boundary of ZXZ cube).
    
    This plaquette interacts with the bottom-right data qubit twice, which cannot
    be expressed in RPNG notation.
    
    Circuit:
        Step 0: RX(aux), RZ(bottom-right)
        Step 1: CX(aux, bottom-right)      # aux is control (NOT delayed)
        Step 2: CX(aux, top-left)
        Step 4: CX(aux, bottom-left)
        Step 8: CX(bottom-right, aux)      # data qubit is control!
        Step 9: MZ(aux)
    
    Qubit layout (SquarePlaquetteQubits):
        - Index 0: top-left (-1, -1)
        - Index 1: top-right (1, -1)  -- NOT USED
        - Index 2: bottom-left (-1, 1)
        - Index 3: bottom-right (1, 1)
        - Index 4: auxiliary/syndrome (0, 0)
    
    Args:
        reset_data: Reset basis for data qubits ('z' or None for no reset)
        measurement_data: Measurement basis for data qubits ('z' or None for no measurement)
        
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
    
    # Step 0: Reset auxiliary in X basis, reset shared data qubit in Z basis
    circuit.append("RX", [AUX], [])
    circuit.append("RZ", [BOTTOM_RIGHT], [])  # Shared qubit - always reset in Z basis
    circuit.append("TICK", [], [])
    
    # Step 1: CX between auxiliary (control) and bottom-right (target) - NOT delayed
    circuit.append("CX", [AUX, BOTTOM_RIGHT], [])
    circuit.append("TICK", [], [])
    
    # Step 2: CX between auxiliary and top-left
    circuit.append("CX", [AUX, TOP_LEFT], [])
    circuit.append("TICK", [], [])
    
    # Step 3: (empty)
    circuit.append("TICK", [], [])
    
    # Step 4: CX between auxiliary and bottom-left
    circuit.append("CX", [AUX, BOTTOM_LEFT], [])
    circuit.append("TICK", [], [])
    
    # Steps 5, 6, 7: (empty)
    circuit.append("TICK", [], [])
    circuit.append("TICK", [], [])
    circuit.append("TICK", [], [])
    
    # Step 8: CX between bottom-right (control) and auxiliary (target)
    circuit.append("CX", [BOTTOM_RIGHT, AUX], [])
    circuit.append("TICK", [], [])
    
    # Step 9: Measure auxiliary in Z basis
    circuit.append("MZ", [AUX], [])
    
    # Create scheduled circuit with explicit schedule
    schedule = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
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
    name = f"coupling_12_r{r_str}_m{m_str}"
    
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
) -> Plaquette:
    """
    Create a custom coupling plaquette for index 7 (left boundary of XZX cube).
    
    This plaquette interacts with the bottom-left data qubit twice.
    
    Circuit:
        Step 0: RZ(aux), RZ(bottom-left)
        Step 2: CX(bottom-left, aux)
        Step 5: CX(aux, top-right)
        Step 6: CX(aux, bottom-right)
        Step 9: CX(aux, bottom-left)
        Step 10: MX(aux)
    
    Qubit layout (SquarePlaquetteQubits):
        - Index 0: top-left (-1, -1)  -- NOT USED
        - Index 1: top-right (1, -1)
        - Index 2: bottom-left (-1, 1)
        - Index 3: bottom-right (1, 1)
        - Index 4: auxiliary/syndrome (0, 0)
    
    Args:
        reset_data: Reset basis for data qubits ('z' or None for no reset)
        measurement_data: Measurement basis for data qubits ('x' or None for no measurement)
        
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
    
    # Step 0: Reset auxiliary in Z basis, reset shared data qubit in Z basis
    circuit.append("RZ", [AUX], [])
    circuit.append("RZ", [BOTTOM_LEFT], [])  # Shared qubit - always reset in Z basis
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
    
    # Steps 7, 8: (empty)
    circuit.append("TICK", [], [])
    circuit.append("TICK", [], [])
    
    # Step 9: CX between auxiliary and bottom-left
    circuit.append("CX", [AUX, BOTTOM_LEFT], [])
    circuit.append("TICK", [], [])
    
    # Step 10: Measure auxiliary in X basis
    circuit.append("MX", [AUX], [])
    
    # Create scheduled circuit with explicit schedule
    schedule = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
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
    name = f"coupling_xzx_7_r{r_str}_m{m_str}"
    
    return Plaquette(
        name=name,
        qubits=plaquette_qubits,
        circuit=filtered_circuit,
        mergeable_instructions=frozenset(["RX", "RZ", "MZ", "MX", "H"]),
    )


def create_custom_coupling_plaquette_xzx_8(
    reset_data: str | None = None,
    measurement_data: str | None = None,
) -> Plaquette:
    """
    Create a custom coupling plaquette for index 8 (left boundary of XZX cube).
    
    This plaquette interacts with the bottom-left data qubit twice.
    
    Circuit:
        Step 0: RZ(aux), RZ(bottom-left)
        Step 2: CX(bottom-left, aux)
        Step 3: CZ(aux, bottom-right)
        Step 5: CZ(aux, top-right)
        Step 9: CX(aux, bottom-left)
        Step 10: MX(aux)
    
    Qubit layout (SquarePlaquetteQubits):
        - Index 0: top-left (-1, -1)  -- NOT USED
        - Index 1: top-right (1, -1)
        - Index 2: bottom-left (-1, 1)
        - Index 3: bottom-right (1, 1)
        - Index 4: auxiliary/syndrome (0, 0)
    
    Args:
        reset_data: Reset basis for data qubits ('x' or None for no reset)
        measurement_data: Measurement basis for data qubits ('x' or None for no measurement)
        
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
    
    # Step 0: Reset auxiliary in Z basis, reset shared data qubit in Z basis
    circuit.append("RZ", [AUX], [])
    circuit.append("RZ", [BOTTOM_LEFT], [])  # Shared qubit - always reset in Z basis
    circuit.append("TICK", [], [])
    
    # Step 1: (empty)
    circuit.append("TICK", [], [])
    
    # Step 2: CX between bottom-left (control) and auxiliary (target)
    circuit.append("CX", [BOTTOM_LEFT, AUX], [])
    circuit.append("TICK", [], [])
    
    # Step 3: CZ between auxiliary and bottom-right
    circuit.append("CZ", [AUX, BOTTOM_RIGHT], [])
    circuit.append("TICK", [], [])
    
    # Step 4: (empty)
    circuit.append("TICK", [], [])
    
    # Step 5: CZ between auxiliary and top-right
    circuit.append("CZ", [AUX, TOP_RIGHT], [])
    circuit.append("TICK", [], [])
    
    # Steps 6, 7, 8: (empty)
    circuit.append("TICK", [], [])
    circuit.append("TICK", [], [])
    circuit.append("TICK", [], [])
    
    # Step 9: CX between auxiliary and bottom-left
    circuit.append("CX", [AUX, BOTTOM_LEFT], [])
    circuit.append("TICK", [], [])
    
    # Step 10: Measure auxiliary in X basis
    circuit.append("MX", [AUX], [])
    
    # Create scheduled circuit with explicit schedule
    schedule = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
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
    name = f"coupling_xzx_8_r{r_str}_m{m_str}"
    
    return Plaquette(
        name=name,
        qubits=plaquette_qubits,
        circuit=filtered_circuit,
        mergeable_instructions=frozenset(["RX", "RZ", "MZ", "MX", "H"]),
    )


def create_custom_coupling_plaquette_xzx_1(
    reset_data: str | None = None,
    measurement_data: str | None = None,
) -> Plaquette:
    """
    Create a custom coupling plaquette for index 1 (corner of XZX cube).
    
    Same as plaquette 8 but without the gate towards top-right.
    This plaquette interacts with the bottom-left data qubit twice.
    
    Circuit:
        Step 0: RZ(aux), RZ(bottom-left)
        Step 2: CX(bottom-left, aux)
        Step 3: CZ(aux, bottom-right)
        Step 9: CX(aux, bottom-left)
        Step 10: MX(aux)
    
    Qubit layout (SquarePlaquetteQubits):
        - Index 0: top-left (-1, -1)  -- NOT USED
        - Index 1: top-right (1, -1)  -- NOT USED
        - Index 2: bottom-left (-1, 1)
        - Index 3: bottom-right (1, 1)
        - Index 4: auxiliary/syndrome (0, 0)
    
    Args:
        reset_data: Reset basis for data qubits ('x' or None for no reset)
        measurement_data: Measurement basis for data qubits ('x' or None for no measurement)
        
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
    
    # Step 0: Reset auxiliary in Z basis, reset shared data qubit in Z basis
    circuit.append("RZ", [AUX], [])
    circuit.append("RZ", [BOTTOM_LEFT], [])  # Shared qubit - always reset in Z basis
    circuit.append("TICK", [], [])
    
    # Step 1: (empty)
    circuit.append("TICK", [], [])
    
    # Step 2: CX between bottom-left (control) and auxiliary (target)
    circuit.append("CX", [BOTTOM_LEFT, AUX], [])
    circuit.append("TICK", [], [])
    
    # Step 3: CZ between auxiliary and bottom-right
    circuit.append("CZ", [AUX, BOTTOM_RIGHT], [])
    circuit.append("TICK", [], [])
    
    # Steps 4, 5, 6, 7, 8: (empty - no top-right gate)
    circuit.append("TICK", [], [])
    circuit.append("TICK", [], [])
    circuit.append("TICK", [], [])
    circuit.append("TICK", [], [])
    circuit.append("TICK", [], [])
    
    # Step 9: CX between auxiliary and bottom-left
    circuit.append("CX", [AUX, BOTTOM_LEFT], [])
    circuit.append("TICK", [], [])
    
    # Step 10: Measure auxiliary in X basis
    circuit.append("MX", [AUX], [])
    
    # Create scheduled circuit with explicit schedule
    schedule = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
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
    name = f"coupling_xzx_1_r{r_str}_m{m_str}"
    
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
        use_custom_coupling: If True, use custom coupling plaquette for index 11
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
        
        # Top boundary - X-basis 2-body (positions 2,3 → timings 4,3)
        5: RPNGDescription.from_string(f"---- ---- {r}x4{m} {r}x3{m}"),
        6: RPNGDescription.empty(),
        
        # Left boundary - Z-basis 2-body (positions 1,3 → timings 5,6)
        7: RPNGDescription.empty(),
        8: RPNGDescription.from_string(f"---- {r}z5{m} ---- {r}z6{m}"),
        
        # Bulk plaquettes (diagonal schedules, delayed by 1)
        9: RPNGDescription.from_string(f"{r}z7{m} {r}z5{m} {r}z4{m} {r}z6{m}"),   # Z-basis: 7,5,4,6
        10: RPNGDescription.from_string(f"{r}x2{m} {r}x5{m} {r}x4{m} {r}x3{m}"),  # X-basis: 2,5,4,3
        
        # Right boundary - Z-basis 2-body (positions 0,2 → timings 7,4)
        # Index 11 may be replaced with custom coupling plaquette
        11: RPNGDescription.from_string(f"{r}z7{m} ---- {r}z4{m} ----"),
        12: RPNGDescription.empty(),
        
        # Bottom boundary - X-basis 2-body (positions 0,1 → timings 2,5)
        13: RPNGDescription.empty(),
        14: RPNGDescription.from_string(f"{r}x2{m} {r}x5{m} ---- ----"),
    }
    
    plaquette_collection = {
        idx: rpng_to_plaquette(rpng, translator)
        for idx, rpng in rpng_descriptions.items()
    }
    
    # Replace indices 2, 11, and 12 with custom coupling plaquettes if requested
    if use_custom_coupling:
        plaquette_collection[2] = create_custom_coupling_plaquette_2(
            reset_data=reset,
            measurement_data=measurement,
        )
        plaquette_collection[11] = create_custom_coupling_plaquette_11(
            reset_data=reset,
            measurement_data=measurement,
        )
        plaquette_collection[12] = create_custom_coupling_plaquette_12(
            reset_data=reset,
            measurement_data=measurement,
        )
    
    empty_plaquette = rpng_to_plaquette(RPNGDescription.empty(), translator)
    return Plaquettes(FrozenDefaultDict(plaquette_collection, default_value=empty_plaquette))


def create_xzx_plaquettes(
    translator,
    reset: str | None = None,
    measurement: str | None = None,
    use_custom_coupling: bool = False,
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
        use_custom_coupling: If True, use custom coupling plaquettes for indices 1, 7, 8
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
        
        # Top boundary - Z-basis 2-body (positions 2,3 → timings 4,3)
        5: RPNGDescription.empty(),
        6: RPNGDescription.from_string(f"---- ---- {r}z4{m} {r}z3{m}"),
        
        # Left boundary - X-basis 2-body (positions 1,3 → timings 5,6)
        7: RPNGDescription.from_string(f"---- {r}x5{m} ---- {r}x6{m}"),
        8: RPNGDescription.empty(),
        
        # Bulk plaquettes (diagonal schedules - opposite of ZXZ, delayed by 1)
        9: RPNGDescription.from_string(f"{r}z2{m} {r}z5{m} {r}z4{m} {r}z3{m}"),   # Z-basis: 2,5,4,3
        10: RPNGDescription.from_string(f"{r}x7{m} {r}x5{m} {r}x4{m} {r}x6{m}"),  # X-basis: 7,5,4,6
        
        # Right boundary - X-basis 2-body (positions 0,2 → timings 7,4)
        11: RPNGDescription.empty(),
        12: RPNGDescription.from_string(f"{r}x7{m} ---- {r}x4{m} ----"),
        
        # Bottom boundary - Z-basis 2-body (positions 0,1 → timings 2,5)
        13: RPNGDescription.from_string(f"{r}z2{m} {r}z5{m} ---- ----"),
        14: RPNGDescription.empty(),
    }
    
    plaquette_collection = {
        idx: rpng_to_plaquette(rpng, translator)
        for idx, rpng in rpng_descriptions.items()
    }
    
    # Replace indices 1, 7, and 8 with custom coupling plaquettes if requested
    if use_custom_coupling:
        plaquette_collection[1] = create_custom_coupling_plaquette_xzx_1(
            reset_data=reset,
            measurement_data=measurement,
        )
        plaquette_collection[7] = create_custom_coupling_plaquette_xzx_7(
            reset_data=reset,
            measurement_data=measurement,
        )
        plaquette_collection[8] = create_custom_coupling_plaquette_xzx_8(
            reset_data=reset,
            measurement_data=measurement,
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
    zxz_init = create_zxz_plaquettes(translator, reset="z", use_custom_coupling=True)
    zxz_bulk = create_zxz_plaquettes(translator, use_custom_coupling=True)
    zxz_meas = create_zxz_plaquettes(translator, measurement="z", use_custom_coupling=True)
    
    # Create plaquettes for XZX cube at (1, 0) - stores X
    xzx_init = create_xzx_plaquettes(translator, reset="x", use_custom_coupling=True)
    xzx_bulk = create_xzx_plaquettes(translator, use_custom_coupling=True)
    xzx_meas = create_xzx_plaquettes(translator, measurement="x", use_custom_coupling=True)
    
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

