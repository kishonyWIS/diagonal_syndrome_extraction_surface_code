#!/usr/bin/env python3
"""
Transpilation pass to remove idle times from a stim circuit.

Each gate is moved to the earliest possible time step where all qubits it acts on
are available (i.e., right after their previous operations complete).
"""

import stim
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional


@dataclass
class ScheduledInstruction:
    """An instruction with its scheduled time step."""
    instruction: stim.CircuitInstruction
    time_step: int


def flatten_circuit(circuit: stim.Circuit) -> stim.Circuit:
    """
    Flatten a stim circuit by expanding all REPEAT blocks.
    
    This converts REPEAT blocks into their unrolled form, which allows
    the compaction pass to work on the full circuit structure.
    
    Args:
        circuit: A stim circuit potentially containing REPEAT blocks
        
    Returns:
        A flattened circuit with no REPEAT blocks
    """
    output = stim.Circuit()
    
    for instruction in circuit:
        if instruction.name == 'REPEAT':
            # Get repeat count and body
            repeat_count = instruction.repeat_count
            body = instruction.body_copy()
            
            # Recursively flatten the body
            flattened_body = flatten_circuit(body)
            
            # Append the flattened body repeat_count times
            for _ in range(repeat_count):
                for body_inst in flattened_body:
                    output.append(body_inst)
        else:
            output.append(instruction)
    
    return output
    

def get_instruction_qubits(instruction: stim.CircuitInstruction) -> list[int]:
    """Extract qubit indices from an instruction's targets."""
    qubits = []
    for target in instruction.targets_copy():
        if target.is_qubit_target:
            qubits.append(target.value)
        elif target.is_x_target or target.is_y_target or target.is_z_target:
            qubits.append(target.value)
        elif target.is_combiner:
            # Skip combiners (used in multi-target gates)
            pass
        elif target.is_measurement_record_target:
            # Skip measurement record references
            pass
    return qubits


def is_two_qubit_gate(name: str) -> bool:
    """Check if instruction is a two-qubit gate."""
    two_qubit_gates = {
        'CX', 'CY', 'CZ', 'CNOT', 'SWAP', 'ISWAP', 'ISWAP_DAG',
        'SQRT_XX', 'SQRT_XX_DAG', 'SQRT_YY', 'SQRT_YY_DAG', 
        'SQRT_ZZ', 'SQRT_ZZ_DAG', 'XCX', 'XCY', 'XCZ', 'YCX', 'YCY', 'YCZ',
        'ZCX', 'ZCY', 'ZCZ', 'DEPOLARIZE2'
    }
    return name in two_qubit_gates


def split_two_qubit_instruction(instruction: stim.CircuitInstruction) -> list[tuple[str, list[int], list[float]]]:
    """
    Split a multi-target two-qubit gate into individual qubit pairs.
    
    Returns list of (gate_name, [qubit1, qubit2], gate_args) tuples.
    """
    name = instruction.name
    targets = instruction.targets_copy()
    args = list(instruction.gate_args_copy())
    
    # Extract qubit indices
    qubits = []
    for target in targets:
        if target.is_qubit_target:
            qubits.append(target.value)
    
    # Split into pairs
    pairs = []
    for i in range(0, len(qubits), 2):
        if i + 1 < len(qubits):
            pairs.append((name, [qubits[i], qubits[i+1]], args))
    
    return pairs


def split_single_qubit_instruction(instruction: stim.CircuitInstruction) -> list[tuple[str, list[int], list[float]]]:
    """
    Split a multi-target single-qubit gate into individual qubit operations.
    
    Returns list of (gate_name, [qubit], gate_args) tuples.
    """
    name = instruction.name
    targets = instruction.targets_copy()
    args = list(instruction.gate_args_copy())
    
    # Extract qubit indices
    singles = []
    for target in targets:
        if target.is_qubit_target:
            singles.append((name, [target.value], args))
        elif target.is_x_target or target.is_y_target or target.is_z_target:
            singles.append((name, [target.value], args))
    
    return singles


def is_timing_instruction(name: str) -> bool:
    """Check if instruction is a timing/annotation instruction that should be preserved."""
    timing_instructions = {
        'TICK', 'QUBIT_COORDS', 'SHIFT_COORDS', 'DETECTOR', 'OBSERVABLE_INCLUDE',
        'REPEAT', 'MPP', 'MPAD'
    }
    return name in timing_instructions


def is_noise_instruction(name: str) -> bool:
    """Check if instruction is a noise instruction."""
    noise_instructions = {
        'DEPOLARIZE1', 'DEPOLARIZE2', 'X_ERROR', 'Y_ERROR', 'Z_ERROR',
        'PAULI_CHANNEL_1', 'PAULI_CHANNEL_2', 'HERALDED_ERASE', 'HERALDED_PAULI_CHANNEL_1'
    }
    return name in noise_instructions


def compact_circuit(circuit: stim.Circuit, remove_noise: bool = False) -> stim.Circuit:
    """
    Compact a stim circuit by removing idle times.
    
    Each gate is moved to the earliest time step where all its qubits are available.
    Two-qubit gates with multiple targets are split so each pair can be scheduled independently.
    
    Args:
        circuit: The input stim circuit
        remove_noise: If True, also remove noise instructions
        
    Returns:
        A new compacted stim circuit
    """
    # Track when each qubit becomes available (time step after its last operation)
    qubit_available_at: dict[int, int] = defaultdict(int)
    
    # Scheduled gates: (time_step, gate_name, qubits, args)
    scheduled_gates: list[tuple[int, str, list[int], list[float]]] = []
    
    # Annotations to preserve (detectors, observables, coords)
    annotations: list[stim.CircuitInstruction] = []
    
    # Process each instruction
    for instruction in circuit:
        name = instruction.name
        
        # Handle REPEAT blocks recursively
        if name == 'REPEAT':
            # Get the repeat count and body
            repeat_count = instruction.repeat_count
            body = instruction.body_copy()
            # Compact the body
            compacted_body = compact_circuit(body, remove_noise)
            # Add as a single scheduled item at current max time
            max_time = max(qubit_available_at.values()) if qubit_available_at else 0
            # We'll handle REPEAT specially when building output
            scheduled_gates.append((max_time, 'REPEAT', [], [repeat_count]))
            # Update qubit availability based on body
            body_qubits = set()
            for body_inst in compacted_body:
                body_qubits.update(get_instruction_qubits(body_inst))
            for q in body_qubits:
                qubit_available_at[q] = max_time + 1
            continue
            
        # Skip TICK - we'll add our own
        if name == 'TICK':
            continue
            
        # Preserve annotations but don't schedule them
        if name in ('DETECTOR', 'OBSERVABLE_INCLUDE', 'QUBIT_COORDS', 'SHIFT_COORDS'):
            annotations.append(instruction)
            continue
            
        # Optionally skip noise
        if remove_noise and is_noise_instruction(name):
            continue
        
        # Handle two-qubit gates by splitting into pairs
        if is_two_qubit_gate(name):
            pairs = split_two_qubit_instruction(instruction)
            for gate_name, qubits, args in pairs:
                # Find earliest time when both qubits are available
                earliest_time = max(qubit_available_at.get(q, 0) for q in qubits)
                # Schedule this pair
                scheduled_gates.append((earliest_time, gate_name, qubits, args))
                # Update qubit availability
                for q in qubits:
                    qubit_available_at[q] = earliest_time + 1
            continue
        
        # Get qubits this instruction acts on
        qubits = get_instruction_qubits(instruction)
        args = list(instruction.gate_args_copy())
        
        if not qubits:
            # Instructions without qubits - preserve at current time
            max_time = max(qubit_available_at.values()) if qubit_available_at else 0
            scheduled_gates.append((max_time, name, qubits, args))
            continue
        
        # Split single-qubit gates so each qubit can be scheduled independently
        singles = split_single_qubit_instruction(instruction)
        for gate_name, gate_qubits, gate_args in singles:
            # Find earliest time when this qubit is available
            earliest_time = qubit_available_at.get(gate_qubits[0], 0)
            # Schedule this single-qubit operation
            scheduled_gates.append((earliest_time, gate_name, gate_qubits, gate_args))
            # Update qubit availability
            qubit_available_at[gate_qubits[0]] = earliest_time + 1
    
    # Build output circuit grouped by time step, merging same-type gates
    time_to_gates: dict[int, dict[str, list[tuple[list[int], list[float]]]]] = defaultdict(lambda: defaultdict(list))
    for time_step, gate_name, qubits, args in scheduled_gates:
        time_to_gates[time_step][gate_name].append((qubits, args))
    
    # Create output circuit
    output = stim.Circuit()
    
    # Add qubit coordinates first (if any)
    for ann in annotations:
        if ann.name == 'QUBIT_COORDS':
            output.append(ann)
    
    # Add instructions in time order with TICKs between time steps
    sorted_times = sorted(time_to_gates.keys())
    for i, t in enumerate(sorted_times):
        gates_at_time = time_to_gates[t]
        for gate_name, qubit_lists in gates_at_time.items():
            # Merge all qubits for same gate type into one instruction
            all_qubits = []
            args = []
            for qubits, gate_args in qubit_lists:
                all_qubits.extend(qubits)
                if gate_args and not args:
                    args = gate_args
            if all_qubits:
                output.append(gate_name, all_qubits, args)
            else:
                output.append(gate_name, [], args)
        
        # Add TICK between time steps (but not after the last one)
        if i < len(sorted_times) - 1:
            output.append('TICK')
    
    # Add remaining annotations at the end
    for ann in annotations:
        if ann.name != 'QUBIT_COORDS':
            output.append(ann)
    
    return output


def is_init_gate(name: str) -> bool:
    """Check if instruction is an initialization/reset gate."""
    init_gates = {'R', 'RX', 'RY', 'RZ'}
    return name in init_gates


def alap_with_fixed_last(circuit: stim.Circuit) -> stim.Circuit:
    """
    ALAP scheduling with the last operation on each qubit fixed.
    
    After ASAP scheduling, this pushes all operations as late as possible while:
    - Keeping the last operation on each qubit at its ASAP-scheduled time
    - Respecting dependencies (each op must happen before the next op on its qubits)
    
    This eliminates all idle time by ensuring each operation happens right before
    its qubits are next needed.
    
    Args:
        circuit: An ASAP-scheduled stim circuit
        
    Returns:
        A circuit with ALAP scheduling (last ops fixed)
    """
    # Parse the circuit into timesteps
    timesteps: list[list[tuple[str, list[int], list[float]]]] = []
    current_timestep: list[tuple[str, list[int], list[float]]] = []
    annotations: list[stim.CircuitInstruction] = []
    
    for instruction in circuit:
        name = instruction.name
        
        if name == 'TICK':
            if current_timestep:
                timesteps.append(current_timestep)
                current_timestep = []
            continue
        
        if name in ('DETECTOR', 'OBSERVABLE_INCLUDE', 'QUBIT_COORDS', 'SHIFT_COORDS'):
            annotations.append(instruction)
            continue
        
        qubits = get_instruction_qubits(instruction)
        args = list(instruction.gate_args_copy())
        current_timestep.append((name, qubits, args))
    
    # Don't forget the last timestep
    if current_timestep:
        timesteps.append(current_timestep)
    
    if not timesteps:
        return circuit
    
    # Flatten all operations with their original timestep and unique ID
    # (orig_time, op_id, gate_name, qubits, args)
    all_ops: list[tuple[int, int, str, list[int], list[float]]] = []
    op_id = 0
    for t, gates in enumerate(timesteps):
        for gate_name, qubits, args in gates:
            all_ops.append((t, op_id, gate_name, qubits, args))
            op_id += 1
    
    # Find the last operation on each qubit
    qubit_last_op: dict[int, int] = {}  # qubit -> op_id of last operation
    for orig_t, op_id, gate_name, qubits, args in all_ops:
        for q in qubits:
            qubit_last_op[q] = op_id
    
    # Set of op_ids that are "last" on at least one qubit (these are anchored)
    anchored_ops = set(qubit_last_op.values())
    
    # For ALAP, we process operations in REVERSE order
    # For each qubit, track when it's next used (initially: after the last timestep)
    num_timesteps = len(timesteps)
    qubit_next_use: dict[int, int] = defaultdict(lambda: num_timesteps)
    
    # New scheduled time for each operation
    op_new_time: dict[int, int] = {}
    
    # Process operations in reverse order (by original time, then by op_id)
    for orig_t, op_id, gate_name, qubits, args in reversed(all_ops):
        if op_id in anchored_ops:
            # This is the last operation on at least one qubit - keep at original time
            new_time = orig_t
        else:
            # Schedule as late as possible: min(next_use[q] for q in qubits) - 1
            # But not earlier than original time (to maintain order within same ASAP time)
            latest_possible = min(qubit_next_use[q] for q in qubits) - 1
            new_time = max(orig_t, latest_possible)
        
        op_new_time[op_id] = new_time
        
        # Update next_use for all qubits in this operation
        for q in qubits:
            qubit_next_use[q] = new_time
    
    # Build new timesteps
    new_timesteps: list[list[tuple[str, list[int], list[float]]]] = [[] for _ in range(num_timesteps)]
    
    for orig_t, op_id, gate_name, qubits, args in all_ops:
        new_t = op_new_time[op_id]
        new_timesteps[new_t].append((gate_name, qubits, args))
    
    # Build the output circuit, merging same-type gates
    output = stim.Circuit()
    
    # Add qubit coordinates first
    for ann in annotations:
        if ann.name == 'QUBIT_COORDS':
            output.append(ann)
    
    # Filter out empty timesteps and build circuit
    non_empty_times = [(t, gates) for t, gates in enumerate(new_timesteps) if gates]
    
    for i, (t, gates) in enumerate(non_empty_times):
        # Group gates by type for merging
        gate_groups: dict[str, list[tuple[list[int], list[float]]]] = defaultdict(list)
        for gate_name, qubits, args in gates:
            gate_groups[gate_name].append((qubits, args))
        
        for gate_name, qubit_lists in gate_groups.items():
            all_qubits = []
            args = []
            for qubits, gate_args in qubit_lists:
                all_qubits.extend(qubits)
                if gate_args and not args:
                    args = gate_args
            if all_qubits:
                output.append(gate_name, all_qubits, args)
            else:
                output.append(gate_name, [], args)
        
        # Add TICK between timesteps
        if i < len(non_empty_times) - 1:
            output.append('TICK')
    
    # Add remaining annotations
    for ann in annotations:
        if ann.name != 'QUBIT_COORDS':
            output.append(ann)
    
    return output


def compact_and_delay_init(circuit: stim.Circuit, remove_noise: bool = False) -> stim.Circuit:
    """
    Fully compact a circuit: ASAP scheduling + ALAP with fixed endpoints.
    
    1. Flatten: Expand all REPEAT blocks
    2. ASAP: Move each gate to the earliest time its qubits are available
    3. ALAP: Move each gate to the latest time before its qubits are next needed
           (keeping the last operation on each qubit fixed)
    
    This eliminates all idle time - no qubit sits idle between operations.
    
    Args:
        circuit: The input stim circuit
        remove_noise: If True, also remove noise instructions
        
    Returns:
        A fully compacted stim circuit with no idle time
    """
    # First flatten REPEAT blocks
    flattened = flatten_circuit(circuit)
    
    # Then do both ASAP and ALAP passes
    return asap_alap_schedule(flattened, remove_noise)


def is_measurement_gate(name: str) -> bool:
    """Check if instruction is a measurement gate."""
    measurement_gates = {'M', 'MX', 'MY', 'MZ', 'MR', 'MRX', 'MRY', 'MRZ'}
    return name in measurement_gates


def asap_alap_schedule(circuit: stim.Circuit, remove_noise: bool = False) -> stim.Circuit:
    """
    Full ASAP + ALAP scheduling with gates kept split until the end.
    
    This ensures individual qubit operations can be scheduled independently.
    Also properly tracks measurement order to update detector/observable references.
    """
    # Track when each qubit becomes available (time step after its last operation)
    qubit_available_at: dict[int, int] = defaultdict(int)
    
    # Scheduled gates: list of (time_step, gate_name, qubits, args, measurement_id)
    # measurement_id is None for non-measurement gates, or a unique ID for measurements
    scheduled_ops: list[tuple[int, str, list[int], list[float], Optional[int]]] = []
    
    # Track measurements in original order: list of (measurement_id, qubit)
    original_measurement_order: list[tuple[int, int]] = []
    measurement_id_counter = 0
    
    # Annotations to preserve (we'll update detector/observable refs later)
    # Store detectors/observables with their absolute measurement references
    # (meas_count_at_definition, instruction, absolute_refs)
    detectors: list[tuple[int, stim.CircuitInstruction, list[int]]] = []
    observables: list[tuple[int, stim.CircuitInstruction, list[int]]] = []
    shift_coords: list[stim.CircuitInstruction] = []
    qubit_coords: list[stim.CircuitInstruction] = []
    
    # Track measurement count during parsing (for converting relative to absolute refs)
    parse_meas_count = 0
    
    # ===== ASAP Pass =====
    for instruction in circuit:
        name = instruction.name
        
        # Handle REPEAT blocks recursively
        if name == 'REPEAT':
            repeat_count = instruction.repeat_count
            body = instruction.body_copy()
            compacted_body = asap_alap_schedule(body, remove_noise)
            max_time = max(qubit_available_at.values()) if qubit_available_at else 0
            scheduled_ops.append((max_time, 'REPEAT', [], [repeat_count], None))
            body_qubits = set()
            for body_inst in compacted_body:
                body_qubits.update(get_instruction_qubits(body_inst))
            for q in body_qubits:
                qubit_available_at[q] = max_time + 1
            continue
            
        if name == 'TICK':
            continue
            
        if name == 'QUBIT_COORDS':
            qubit_coords.append(instruction)
            continue
        
        if name == 'SHIFT_COORDS':
            shift_coords.append(instruction)
            continue
            
        if name == 'DETECTOR':
            # Convert relative refs to absolute measurement IDs
            abs_refs = []
            for target in instruction.targets_copy():
                if target.is_measurement_record_target:
                    rel_ref = target.value  # Negative number
                    abs_ref = parse_meas_count + rel_ref  # Convert to absolute
                    abs_refs.append(abs_ref)
            detectors.append((parse_meas_count, instruction, abs_refs))
            continue
            
        if name == 'OBSERVABLE_INCLUDE':
            # Convert relative refs to absolute measurement IDs
            abs_refs = []
            for target in instruction.targets_copy():
                if target.is_measurement_record_target:
                    rel_ref = target.value
                    abs_ref = parse_meas_count + rel_ref
                    abs_refs.append(abs_ref)
            observables.append((parse_meas_count, instruction, abs_refs))
            continue
            
        if remove_noise and is_noise_instruction(name):
            continue
        
        # Handle two-qubit gates by splitting into pairs
        if is_two_qubit_gate(name):
            pairs = split_two_qubit_instruction(instruction)
            for gate_name, qubits, args in pairs:
                earliest_time = max(qubit_available_at.get(q, 0) for q in qubits)
                scheduled_ops.append((earliest_time, gate_name, qubits, args, None))
                for q in qubits:
                    qubit_available_at[q] = earliest_time + 1
            continue
        
        # Handle single-qubit gates by splitting
        qubits = get_instruction_qubits(instruction)
        args = list(instruction.gate_args_copy())
        
        if not qubits:
            max_time = max(qubit_available_at.values()) if qubit_available_at else 0
            scheduled_ops.append((max_time, name, qubits, args, None))
            continue
        
        singles = split_single_qubit_instruction(instruction)
        for gate_name, gate_qubits, gate_args in singles:
            earliest_time = qubit_available_at.get(gate_qubits[0], 0)
            
            # Track measurements with unique IDs
            if is_measurement_gate(gate_name):
                meas_id = measurement_id_counter
                measurement_id_counter += 1
                original_measurement_order.append((meas_id, gate_qubits[0]))
                scheduled_ops.append((earliest_time, gate_name, gate_qubits, gate_args, meas_id))
                parse_meas_count += 1  # Update count for detector/observable parsing
            else:
                scheduled_ops.append((earliest_time, gate_name, gate_qubits, gate_args, None))
            
            qubit_available_at[gate_qubits[0]] = earliest_time + 1
    
    if not scheduled_ops:
        return circuit
    
    # ===== ALAP Pass =====
    # Find the last operation on each qubit (these are anchored)
    qubit_last_op_idx: dict[int, int] = {}
    for idx, (time, gate_name, qubits, args, meas_id) in enumerate(scheduled_ops):
        for q in qubits:
            qubit_last_op_idx[q] = idx
    
    anchored_ops = set(qubit_last_op_idx.values())
    
    # Process in reverse order
    max_time = max(t for t, _, _, _, _ in scheduled_ops)
    qubit_next_use: dict[int, int] = defaultdict(lambda: max_time + 1)
    
    op_new_time: dict[int, int] = {}
    
    for idx in range(len(scheduled_ops) - 1, -1, -1):
        orig_t, gate_name, qubits, args, meas_id = scheduled_ops[idx]
        
        if idx in anchored_ops:
            # Keep at original time
            new_time = orig_t
        else:
            # Schedule as late as possible
            if qubits:
                latest_possible = min(qubit_next_use[q] for q in qubits) - 1
                new_time = max(0, latest_possible)  # Don't go negative
            else:
                new_time = orig_t
        
        op_new_time[idx] = new_time
        
        # Update next_use for all qubits
        for q in qubits:
            qubit_next_use[q] = new_time
    
    # ===== Build Output and Track New Measurement Order =====
    # Group by new time, preserving the idx for measurement tracking
    time_to_ops: dict[int, list[tuple[int, str, list[int], list[float], Optional[int]]]] = defaultdict(list)
    for idx, (orig_t, gate_name, qubits, args, meas_id) in enumerate(scheduled_ops):
        new_t = op_new_time[idx]
        time_to_ops[new_t].append((idx, gate_name, qubits, args, meas_id))
    
    # Determine the new order of measurements
    # IMPORTANT: Must match the actual output order, which groups by gate type within each timestep
    new_measurement_order: list[tuple[int, int]] = []  # (meas_id, qubit)
    sorted_times = sorted(time_to_ops.keys())
    
    for t in sorted_times:
        ops = time_to_ops[t]
        # Sort ops by original idx
        ops_sorted = sorted(ops, key=lambda x: x[0])
        
        # Group by gate type (same as output circuit building)
        gate_groups: dict[str, list[tuple[int, str, list[int], list[float], Optional[int]]]] = defaultdict(list)
        for idx, gate_name, qubits, args, meas_id in ops_sorted:
            gate_groups[gate_name].append((idx, gate_name, qubits, args, meas_id))
        
        # Extract measurements in the same order they'll appear in output
        # (grouped by gate type, with original idx order within each group)
        for gate_name, group_ops in gate_groups.items():
            if is_measurement_gate(gate_name):
                for idx, gn, qubits, args, meas_id in group_ops:
                    if meas_id is not None:
                        new_measurement_order.append((meas_id, qubits[0]))
    
    # Build mapping from measurement_id to new index (from end)
    total_measurements = len(new_measurement_order)
    meas_id_to_new_idx: dict[int, int] = {}
    for new_pos, (meas_id, qubit) in enumerate(new_measurement_order):
        # rec[-1] is the last measurement, rec[-N] is N-th from end
        new_idx_from_end = total_measurements - new_pos
        meas_id_to_new_idx[meas_id] = new_idx_from_end
    
    # Build mapping from old index to new index
    old_idx_to_new_idx: dict[int, int] = {}
    for old_pos, (meas_id, qubit) in enumerate(original_measurement_order):
        old_idx_from_end = total_measurements - old_pos
        new_idx_from_end = meas_id_to_new_idx[meas_id]
        old_idx_to_new_idx[old_idx_from_end] = new_idx_from_end
    
    # ===== Build Output Circuit =====
    output = stim.Circuit()
    
    # Add qubit coordinates first
    for ann in qubit_coords:
        output.append(ann)
    
    # Build circuit with merged gates
    for i, t in enumerate(sorted_times):
        ops = time_to_ops[t]
        
        # Sort ops by original idx to preserve relative order
        ops_sorted = sorted(ops, key=lambda x: x[0])
        
        # Group by gate type for merging, preserving order within each group
        gate_groups: dict[str, list[tuple[int, list[int], list[float]]]] = defaultdict(list)
        for idx, gate_name, qubits, args, meas_id in ops_sorted:
            if gate_name == 'REPEAT':
                # REPEAT blocks are not supported in compaction - skip them
                continue
            gate_groups[gate_name].append((idx, qubits, args))
        
        for gate_name, idx_qubit_lists in gate_groups.items():
            # Already sorted by idx, just extract qubits in order
            all_qubits = []
            args = []
            for idx, qubits, gate_args in idx_qubit_lists:
                all_qubits.extend(qubits)
                if gate_args and not args:
                    args = gate_args
            if all_qubits:
                output.append(gate_name, all_qubits, args)
            elif args:
                # Gates with args but no qubits (shouldn't happen normally)
                output.append(gate_name, [], args)
        
        if i < len(sorted_times) - 1:
            output.append('TICK')
    
    # ===== Add Detectors and Observables with Updated Measurement References =====
    # Build mapping from absolute measurement index to new position from end
    # abs_meas_to_new_idx[abs_meas_id] = new index from end (for rec[-N])
    abs_meas_to_new_idx: dict[int, int] = {}
    for new_pos, (meas_id, qubit) in enumerate(new_measurement_order):
        # Original absolute index is the same as meas_id (since we assigned them in order)
        abs_meas_idx = meas_id
        new_idx_from_end = total_measurements - new_pos
        abs_meas_to_new_idx[abs_meas_idx] = new_idx_from_end
    
    # Add all detectors at the end, converting absolute refs back to relative
    for meas_count_at_def, orig_inst, abs_refs in detectors:
        new_targets = []
        for abs_ref in abs_refs:
            if abs_ref in abs_meas_to_new_idx:
                new_idx_from_end = abs_meas_to_new_idx[abs_ref]
                new_targets.append(stim.target_rec(-new_idx_from_end))
            else:
                # Fallback - shouldn't happen
                new_targets.append(stim.target_rec(abs_ref - total_measurements))
        output.append('DETECTOR', new_targets, orig_inst.gate_args_copy())
    
    # Add all observables at the end
    for meas_count_at_def, orig_inst, abs_refs in observables:
        new_targets = []
        for abs_ref in abs_refs:
            if abs_ref in abs_meas_to_new_idx:
                new_idx_from_end = abs_meas_to_new_idx[abs_ref]
                new_targets.append(stim.target_rec(-new_idx_from_end))
            else:
                new_targets.append(stim.target_rec(abs_ref - total_measurements))
        output.append('OBSERVABLE_INCLUDE', new_targets, orig_inst.gate_args_copy())
    
    # Add SHIFT_COORDS at the end (though they may not be meaningful after restructuring)
    for sc in shift_coords:
        output.append(sc)
    
    return output


def compact_circuit_simple(circuit: stim.Circuit, remove_noise: bool = False) -> stim.Circuit:
    """
    Simpler version: just remove TICKs and noise, keeping gate order.
    
    This is a quick way to visualize the circuit structure without timing.
    """
    output = stim.Circuit()
    
    for instruction in circuit:
        name = instruction.name
        
        # Skip TICKs
        if name == 'TICK':
            continue
            
        # Optionally skip noise
        if remove_noise and is_noise_instruction(name):
            continue
            
        # Handle REPEAT blocks
        if name == 'REPEAT':
            body = instruction.body_copy()
            compacted_body = compact_circuit_simple(body, remove_noise)
            output.append(stim.CircuitRepeatBlock(instruction.repeat_count, compacted_body))
            continue
            
        output.append(instruction)
    
    return output


def print_crumble_url(circuit: stim.Circuit, name: str = "Circuit"):
    """Print a Crumble URL for the circuit."""
    url = circuit.to_crumble_url()
    print(f"\n=== {name} ===")
    print(url)


def demo():
    """Demonstrate the compaction on memory experiment circuits."""
    import sys
    sys.path.insert(0, 'venv/lib/python3.13/site-packages')
    
    import tqec.plaquette.constants as constants
    constants.MEASUREMENT_SCHEDULE = 8
    
    import importlib
    import tqec.plaquette.rpng.translators.default as default_translator
    importlib.reload(default_translator)
    
    from tqec.gallery import memory
    from tqec import compile_block_graph, NoiseModel
    from tqec.utils.enums import Basis
    
    # Import diagonal circuit creator
    from benchmark_memory import create_diagonal_convention
    
    print("=" * 70)
    print("Compact Circuit Transpilation Demo")
    print("=" * 70)
    
    # Create original (standard schedule) circuit
    print("\nGenerating standard schedule circuit...")
    mem_graph = memory(Basis.Z)
    compiled = compile_block_graph(mem_graph)
    noise_model = NoiseModel.uniform_depolarizing(0.001)
    original_circuit = compiled.generate_stim_circuit(k=1, noise_model=noise_model)
    
    # Create diagonal schedule circuit
    print("Generating diagonal schedule circuit...")
    diagonal_convention = create_diagonal_convention()
    compiled_diag = compile_block_graph(mem_graph, convention=diagonal_convention)
    diagonal_circuit = compiled_diag.generate_stim_circuit(k=1, noise_model=noise_model)
    
    # Compact both circuits (remove noise for cleaner visualization)
    print("\nCompacting circuits...")
    original_compact = compact_circuit_simple(original_circuit, remove_noise=True)
    diagonal_compact = compact_circuit_simple(diagonal_circuit, remove_noise=True)
    
    # Print stats
    print(f"\nOriginal circuit: {len(original_circuit)} instructions -> {len(original_compact)} after compaction")
    print(f"Diagonal circuit: {len(diagonal_circuit)} instructions -> {len(diagonal_compact)} after compaction")
    
    # Print Crumble URLs
    print_crumble_url(original_compact, "Standard Schedule (compacted, no noise)")
    print_crumble_url(diagonal_compact, "Diagonal Schedule (compacted, no noise)")
    
    # Also show with proper scheduling (gates at earliest available time)
    print("\n" + "=" * 70)
    print("With proper ASAP scheduling (gates moved to earliest time):")
    print("=" * 70)
    
    original_asap = compact_circuit(original_circuit, remove_noise=True)
    diagonal_asap = compact_circuit(diagonal_circuit, remove_noise=True)
    
    print_crumble_url(original_asap, "Standard Schedule (ASAP scheduled, no noise)")
    print_crumble_url(diagonal_asap, "Diagonal Schedule (ASAP scheduled, no noise)")


if __name__ == "__main__":
    demo()
