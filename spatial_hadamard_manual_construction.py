#!/usr/bin/env python3
"""
Example: Generate a memory experiment circuit directly from plaquettes.

This bypasses the BlockGraph abstraction and directly defines plaquette layouts
with custom RPNG descriptions to generate a stim circuit.
"""

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
    
    Circuit (most CX/CZ gates delayed by 1 to avoid conflicts):
        Step 0: RX(aux), RZ(bottom-right)
        Step 1: CX(aux, bottom-right)      # aux is control (NOT delayed)
        Step 4: CZ(aux, bottom-left)
        Step 7: CZ(aux, top-left)
        Step 8: CX(bottom-right, aux)      # data qubit is control!
        Step 9: MZ(aux)  [optional], MZ(bottom-right) [optional]
    
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
    
    # Step 9: Measure auxiliary in Z basis (optional)
    if measure_aux:
        circuit.append("TICK", [], [])
        circuit.append("MZ", [AUX], [])
    
    # Step 10: Measure shared data qubit in Z basis (optional)
    # This must be at step 10 (after step 9) to avoid conflict with XZX plaquettes
    if measure_shared_data:
        if not measure_aux:
            circuit.append("TICK", [], [])  # Need TICK if we didn't have one for aux
        circuit.append("TICK", [], [])
        circuit.append("MZ", [BOTTOM_RIGHT], [])
    
    # Create scheduled circuit with explicit schedule
    if measure_shared_data:
        schedule = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    elif measure_aux:
        schedule = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    else:
        schedule = [0, 1, 2, 3, 4, 5, 6, 7, 8]
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
    
    Circuit:
        Step 0: RX(aux), RZ(bottom-right)
        Step 1: CX(aux, bottom-right)      # aux is control (NOT delayed)
        Step 4: CX(aux, bottom-left)
        Step 8: CX(bottom-right, aux)      # data qubit is control!
        Step 9: MZ(aux)  [optional], MZ(bottom-right) [optional]
    
    Qubit layout (SquarePlaquetteQubits):
        - Index 0: top-left (-1, -1)  -- NOT USED
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
    
    # Step 0: Reset auxiliary in X basis, optionally reset shared data qubit in Z basis
    circuit.append("RX", [AUX], [])
    if reset_shared_data:
        circuit.append("RZ", [BOTTOM_RIGHT], [])  # Shared qubit - reset in Z basis
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
    
    # Step 9: Measure auxiliary in Z basis (optional)
    if measure_aux:
        circuit.append("TICK", [], [])
        circuit.append("MZ", [AUX], [])
    
    # Step 10: Measure shared data qubit in Z basis (optional)
    # This must be at step 10 (after step 9) to avoid conflict with XZX plaquettes
    if measure_shared_data:
        if not measure_aux:
            circuit.append("TICK", [], [])  # Need TICK if we didn't have one for aux
        circuit.append("TICK", [], [])
        circuit.append("MZ", [BOTTOM_RIGHT], [])
    
    # Create scheduled circuit with explicit schedule
    if measure_shared_data:
        schedule = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    elif measure_aux:
        schedule = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    else:
        schedule = [0, 1, 2, 3, 4, 5, 6, 7, 8]
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
    
    Circuit:
        Step 0: RX(aux), RZ(bottom-right)
        Step 1: CX(aux, bottom-right)      # aux is control (NOT delayed)
        Step 2: CX(aux, top-left)
        Step 4: CX(aux, bottom-left)
        Step 8: CX(bottom-right, aux)      # data qubit is control!
        Step 9: MZ(aux)  [optional], MZ(bottom-right) [optional]
    
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
    
    # Step 0: Reset auxiliary in X basis, optionally reset shared data qubit in Z basis
    circuit.append("RX", [AUX], [])
    if reset_shared_data:
        circuit.append("RZ", [BOTTOM_RIGHT], [])  # Shared qubit - reset in Z basis
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
    
    # Step 9: Measure auxiliary in Z basis (optional)
    if measure_aux:
        circuit.append("TICK", [], [])
        circuit.append("MZ", [AUX], [])
    
    # Step 10: Measure shared data qubit in Z basis (optional)
    # This must be at step 10 (after step 9) to avoid conflict with XZX plaquettes
    if measure_shared_data:
        if not measure_aux:
            circuit.append("TICK", [], [])  # Need TICK if we didn't have one for aux
        circuit.append("TICK", [], [])
        circuit.append("MZ", [BOTTOM_RIGHT], [])
    
    # Create scheduled circuit with explicit schedule
    if measure_shared_data:
        schedule = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    elif measure_aux:
        schedule = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    else:
        schedule = [0, 1, 2, 3, 4, 5, 6, 7, 8]
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
    
    Circuit:
        Step 0: RZ(aux), RZ(bottom-left)
        Step 2: CX(bottom-left, aux)
        Step 5: CX(aux, top-right)
        Step 6: CX(aux, bottom-right)
        Step 9: CX(aux, bottom-left)
        Step 10: MX(aux)  [optional]
    
    Qubit layout (SquarePlaquetteQubits):
        - Index 0: top-left (-1, -1)  -- NOT USED
        - Index 1: top-right (1, -1)
        - Index 2: bottom-left (-1, 1)
        - Index 3: bottom-right (1, 1)
        - Index 4: auxiliary/syndrome (0, 0)
    
    Args:
        reset_data: Reset basis for data qubits ('z' or None for no reset)
        measurement_data: Measurement basis for data qubits ('x' or None for no measurement)
        measure_aux: If True, measure the auxiliary qubit at the end
        
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
    
    # Steps 7, 8: (empty)
    circuit.append("TICK", [], [])
    circuit.append("TICK", [], [])
    
    # Step 9: CX between auxiliary and bottom-left
    circuit.append("CX", [AUX, BOTTOM_LEFT], [])
    
    # Step 10: Measure auxiliary in X basis (optional)
    if measure_aux:
        circuit.append("TICK", [], [])
        circuit.append("MX", [AUX], [])
    
    # Create scheduled circuit with explicit schedule
    schedule = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] if measure_aux else [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
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
    
    Circuit:
        Step 0: RZ(aux), RZ(bottom-left)
        Step 2: CX(bottom-left, aux)
        Step 3: CZ(aux, bottom-right)
        Step 5: CZ(aux, top-right)
        Step 9: CX(aux, bottom-left)
        Step 10: MX(aux)  [optional]
    
    Qubit layout (SquarePlaquetteQubits):
        - Index 0: top-left (-1, -1)  -- NOT USED
        - Index 1: top-right (1, -1)
        - Index 2: bottom-left (-1, 1)
        - Index 3: bottom-right (1, 1)
        - Index 4: auxiliary/syndrome (0, 0)
    
    Args:
        reset_data: Reset basis for data qubits ('x' or None for no reset)
        measurement_data: Measurement basis for data qubits ('x' or None for no measurement)
        measure_aux: If True, measure the auxiliary qubit at the end
        
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
    
    # Step 10: Measure auxiliary in X basis (optional)
    if measure_aux:
        circuit.append("TICK", [], [])
        circuit.append("MX", [AUX], [])
    
    # Create scheduled circuit with explicit schedule
    schedule = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] if measure_aux else [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
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
    
    Circuit:
        Step 0: RZ(aux), RZ(bottom-left)
        Step 2: CX(bottom-left, aux)
        Step 3: CZ(aux, bottom-right)
        Step 9: CX(aux, bottom-left)
        Step 10: MX(aux)  [optional]
    
    Qubit layout (SquarePlaquetteQubits):
        - Index 0: top-left (-1, -1)  -- NOT USED
        - Index 1: top-right (1, -1)  -- NOT USED
        - Index 2: bottom-left (-1, 1)
        - Index 3: bottom-right (1, 1)
        - Index 4: auxiliary/syndrome (0, 0)
    
    Args:
        reset_data: Reset basis for data qubits ('x' or None for no reset)
        measurement_data: Measurement basis for data qubits ('x' or None for no measurement)
        measure_aux: If True, measure the auxiliary qubit at the end
        
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
    
    # Step 10: Measure auxiliary in X basis (optional)
    if measure_aux:
        circuit.append("TICK", [], [])
        circuit.append("MX", [AUX], [])
    
    # Create scheduled circuit with explicit schedule
    schedule = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] if measure_aux else [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
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


def create_zxz_plaquettes(
    translator,
    reset: str | None = None,
    measurement: str | None = None,
    use_custom_coupling: bool = False,
    measure_coupling_aux: bool = True,
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
            measure_aux=measure_coupling_aux,
            measure_shared_data=measure_shared_data,
            reset_shared_data=reset_shared_data,
        )
        plaquette_collection[11] = create_custom_coupling_plaquette_11(
            reset_data=reset,
            measurement_data=measurement,
            measure_aux=measure_coupling_aux,
            measure_shared_data=measure_shared_data,
            reset_shared_data=reset_shared_data,
        )
        plaquette_collection[12] = create_custom_coupling_plaquette_12(
            reset_data=reset,
            measurement_data=measurement,
            measure_aux=measure_coupling_aux,
            measure_shared_data=measure_shared_data,
            reset_shared_data=reset_shared_data,
        )
    
    empty_plaquette = rpng_to_plaquette(RPNGDescription.empty(), translator)
    return Plaquettes(FrozenDefaultDict(plaquette_collection, default_value=empty_plaquette))


def create_xzx_plaquettes(
    translator,
    reset: str | None = None,
    measurement: str | None = None,
    use_custom_coupling: bool = False,
    measure_coupling_aux: bool = True,
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
            measure_aux=measure_coupling_aux,
            reset_shared_data=reset_shared_data,
        )
        plaquette_collection[7] = create_custom_coupling_plaquette_xzx_7(
            reset_data=reset,
            measurement_data=measurement,
            measure_aux=measure_coupling_aux,
            reset_shared_data=reset_shared_data,
        )
        plaquette_collection[8] = create_custom_coupling_plaquette_xzx_8(
            reset_data=reset,
            measurement_data=measurement,
            measure_aux=measure_coupling_aux,
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
    measure_zxz_coupling_aux: bool = True,
    measure_xzx_coupling_aux: bool = True,
    measure_shared_data: bool = False,
    measure_shared_data_final_only: bool = False,
    manhattan_radius: int = 2,
) -> stim.Circuit:
    """
    Generate a circuit with two decoupled cubes:
    - ZXZ cube at position (0, 0, 0) - stores Z
    - XZX cube at position (1, 0, 0) - stores X
    
    The cubes are decoupled (no pipe connecting them).
    
    Args:
        k: Scaling factor (code distance ≈ 2k+1)
        noise_model: Optional noise model to apply to the circuit
        measure_zxz_coupling_aux: If True, measure auxiliary qubits in ZXZ coupling plaquettes
        measure_xzx_coupling_aux: If True, measure auxiliary qubits in XZX coupling plaquettes
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
    
    # Reset shared data: always reset in init, optionally reset in bulk (if not final_only), never in meas
    reset_shared_init = True  # Always reset in init to initialize qubits
    reset_shared_bulk = not measure_shared_data_final_only  # Don't reset in bulk if final_only
    reset_shared_meas = False  # Never reset in meas - about to measure them
    
    zxz_init = create_zxz_plaquettes(translator, reset="z", use_custom_coupling=True, measure_coupling_aux=measure_zxz_coupling_aux, measure_shared_data=measure_shared_init_bulk, reset_shared_data=reset_shared_init)
    zxz_bulk = create_zxz_plaquettes(translator, use_custom_coupling=True, measure_coupling_aux=measure_zxz_coupling_aux, measure_shared_data=measure_shared_init_bulk, reset_shared_data=reset_shared_bulk)
    zxz_meas = create_zxz_plaquettes(translator, measurement="z", use_custom_coupling=True, measure_coupling_aux=measure_zxz_coupling_aux, measure_shared_data=measure_shared_meas, reset_shared_data=reset_shared_meas)
    
    # Create plaquettes for XZX cube at (1, 0) - stores X
    xzx_init = create_xzx_plaquettes(translator, reset="x", use_custom_coupling=True, measure_coupling_aux=measure_xzx_coupling_aux, reset_shared_data=reset_shared_init)
    xzx_bulk = create_xzx_plaquettes(translator, use_custom_coupling=True, measure_coupling_aux=measure_xzx_coupling_aux, reset_shared_data=reset_shared_bulk)
    xzx_meas = create_xzx_plaquettes(translator, measurement="x", use_custom_coupling=True, measure_coupling_aux=measure_xzx_coupling_aux, reset_shared_data=reset_shared_meas)
    
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


def add_mapped_detectors_and_flag_detectors(
    circuit_with_flags: stim.Circuit,
    circuit_no_flags: stim.Circuit,
    flag_detector_tag: str = "flag",
) -> stim.Circuit:
    """
    Add detectors to the with-flag circuit based on the no-flag circuit detectors,
    plus single-measurement detectors for each flag measurement.
    
    Flag detectors are tagged with the specified tag for easy identification.
    
    Args:
        circuit_with_flags: Circuit with flag measurements but no detectors (manhattan_radius=0)
        circuit_no_flags: Circuit without flag measurements but with good detectors
        flag_detector_tag: Tag to apply to flag detectors (default: "flag")
        
    Returns:
        New circuit with properly mapped detectors and tagged flag detectors.
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
    
    # Build the new circuit by stripping existing detectors/observables and adding new ones
    new_circuit = stim.Circuit()
    
    # Copy all instructions except DETECTOR and OBSERVABLE_INCLUDE
    for inst in circuit_with_flags.flattened():
        if inst.name not in ('DETECTOR', 'OBSERVABLE_INCLUDE'):
            new_circuit.append(inst)
    
    # Add mapped detectors from no-flag circuit (these are the "core" detectors)
    for meas_offsets, coords in no_flag_detectors:
        # Map measurement indices
        new_offsets = []
        for old_idx in meas_offsets:
            if old_idx in mapping:
                new_offsets.append(mapping[old_idx])
            else:
                # This shouldn't happen if our mapping is correct
                raise ValueError(f"Measurement index {old_idx} not found in mapping")
        
        # Convert to relative offsets from end of circuit
        relative_offsets = [idx - total_measurements_with_flags for idx in new_offsets]
        
        # Create detector instruction (no tag for core detectors)
        targets = [stim.target_rec(offset) for offset in relative_offsets]
        new_circuit.append("DETECTOR", targets, list(coords))
    
    # Add single-measurement detectors for each flag measurement
    # These are tagged with flag_detector_tag for easy identification
    for flag_idx in sorted(flag_indices):
        relative_offset = flag_idx - total_measurements_with_flags
        # Create tagged detector instruction using stim.CircuitInstruction
        detector_inst = stim.CircuitInstruction(
            "DETECTOR",
            [stim.target_rec(relative_offset)],
            [0.0, 0.0, 0.0],
            tag=flag_detector_tag,
        )
        new_circuit.append(detector_inst)
    
    # Add mapped observables
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
    Generate a two-cube circuit with proper detectors for flag measurements.
    
    This uses a two-pass approach:
    1. Generate circuit WITHOUT flags to get good automatic detectors
    2. Generate circuit WITH flags but skip automatic detector computation
    3. Map detectors from step 1 to step 2, and add simple single-measurement
       detectors for each flag measurement (tagged with "flag")
    
    Flag measurements are:
    - Auxiliary qubit measurements of ZXZ custom plaquettes (indices 2, 11, 12)
    - Shared data qubit measurements
    
    Flag detectors are tagged with "flag" and can be extracted using
    get_flag_detector_indices_from_circuit().
    
    Args:
        k: Scaling factor (code distance ≈ 2k+1)
        noise_model: Optional noise model to apply to the circuit
        measure_shared_data_final_only: If True, only measure/reset shared data at first/last round
        
    Returns:
        Complete stim circuit with:
        - Core detectors mapped from the non-flag circuit
        - Tagged single-measurement detectors for each flag measurement
        - Properly mapped observable
    """
    # Step 1: Generate circuit WITHOUT flags to get good detectors
    circuit_no_flags = generate_two_cube_circuit(
        k=k,
        noise_model=None,  # No noise yet - apply at the end
        measure_zxz_coupling_aux=False,
        measure_xzx_coupling_aux=True,
        measure_shared_data=False,
        measure_shared_data_final_only=measure_shared_data_final_only,
        manhattan_radius=2,  # Normal detector computation
    )
    
    # Step 2: Generate circuit WITH flags, but skip automatic detectors
    circuit_with_flags = generate_two_cube_circuit(
        k=k,
        noise_model=None,  # No noise yet
        measure_zxz_coupling_aux=True,
        measure_xzx_coupling_aux=True,
        measure_shared_data=True,
        measure_shared_data_final_only=measure_shared_data_final_only,
        manhattan_radius=0,  # Disable automatic detector computation
    )
    
    # Step 3: Map detectors and add flag detectors (tagged with "flag")
    circuit_with_detectors = add_mapped_detectors_and_flag_detectors(
        circuit_with_flags,
        circuit_no_flags,
        flag_detector_tag="flag",
    )
    
    # Step 4: Apply noise if provided
    if noise_model is not None:
        circuit_with_detectors = noise_model.noisy_circuit(circuit_with_detectors)
    
    return circuit_with_detectors


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


def load_circuit_from_file(filepath: str) -> stim.Circuit:
    """Load a stim circuit from a file."""
    with open(filepath, 'r') as f:
        return stim.Circuit(f.read())


def main():
    """Generate and analyze the two-cube spatial Hadamard circuit."""
    k = 2
    noise_level = 0.001
    
    print("=" * 70)
    print("Spatial Hadamard: Two-Cube Circuit (ZXZ + XZX)")
    print("=" * 70)
    print(f"k = {k} (code distance ≈ {2*k+1})")
    print(f"Noise level: {noise_level}")
    print()
    
    noise_model = NoiseModel.uniform_depolarizing(noise_level)
    
    # Generate two-cube circuit with proper flag detectors (two-pass approach)
    print("Generating two-cube circuit with flag detectors...")
    circuit = generate_two_cube_circuit_with_flag_detectors(
        k=k,
        noise_model=noise_model,
        measure_shared_data_final_only=False,
    )
    
    # Extract flag detector info from tags
    flag_detector_indices = get_flag_detector_indices_from_circuit(circuit)
    print(f"Flag detectors: {len(flag_detector_indices)} (tagged with 'flag')")
    
    # Verify no missing detectors
    missing_detectors = circuit.missing_detectors()
    if missing_detectors.num_detectors > 0:
        print(f"WARNING: {missing_detectors.num_detectors} missing detectors found!")
    else:
        print("All measurements covered by detectors ✓")
    
    print()
    print("=" * 70)
    print("Circuit Statistics")
    print("=" * 70)
    print(f"Instructions: {len(circuit)}")
    print(f"Qubits: {circuit.num_qubits}")
    print(f"Detectors: {circuit.num_detectors}")
    print(f"Observables: {circuit.num_observables}")
    print(f"Measurements: {circuit.num_measurements}")
    
    print()
    print("=" * 70)
    print("Distance Calculations")
    print("=" * 70)
    
    graphlike_dist = calculate_graphlike_distance(circuit)
    print(f"Graph-like distance: {graphlike_dist}")
    
    circuit_dist = calculate_circuit_distance(circuit)
    print(f"Circuit distance: {circuit_dist}")
    
    # Save circuit to file for debugging
    print()
    print("=" * 70)
    print("Saving Circuit")
    print("=" * 70)
    save_circuit_to_file(circuit, "spatial_hadamard_circuit.stim")
    
    print()
    print("=" * 70)
    print("Crumble URL")
    print("=" * 70)
    print(shift_to_only_positive(circuit).to_crumble_url())


if __name__ == "__main__":
    main()
