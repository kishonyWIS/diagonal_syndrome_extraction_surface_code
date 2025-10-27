#!/usr/bin/env python3
"""Compare standard vs diagonal schedule X junction implementations."""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, 'venv/lib/python3.13/site-packages')

from tqec.computation.block_graph import BlockGraph
from tqec.computation.cube import ZXCube
from tqec.utils.position import Position3D
from tqec.compile.compile import compile_block_graph
from tqec.compile.convention import FIXED_BULK_CONVENTION
from tqec.utils.noise_model import NoiseModel
from tqec.utils.enums import Basis
import stim
import sinter

try:
    import pymatching
    PYMATCHING_AVAILABLE = True
except ImportError:
    PYMATCHING_AVAILABLE = False

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


def compile_and_generate(graph, convention_name, convention, k=1, use_diagonal=False):
    """Compile the graph and generate a Stim circuit."""
    print(f"\nCompiling with {convention_name} convention (k={k})...")
    
    try:
        compiled_graph = compile_block_graph(
            block_graph=graph,
            convention=convention
        )
        print(f"✓ Successfully compiled block graph")
        
        # If use_diagonal, replace RPNG descriptions
        if use_diagonal:
            compiled_graph = replace_plaquettes_in_graph(compiled_graph)
        
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


def calculate_logical_error_rate(circuit, shots=50000, noise_levels=[0.001, 0.002, 0.005]):
    """Calculate logical error rate using sinter."""
    if not PYMATCHING_AVAILABLE:
        print("Skipping logical error rate calculation - pymatching not available")
        return {}
    
    print(f"Calculating logical error rate with {shots} shots...")
    
    results = {}
    
    for noise_level in noise_levels:
        print(f"  Testing noise level: {noise_level}")
        
        # Create a circuit with the specified noise level
        noisy_circuit = circuit.copy()
        
        # Replace noise parameters with the desired noise level
        noisy_circuit_str = str(noisy_circuit)
        noisy_circuit_str = noisy_circuit_str.replace('0.001', str(noise_level))
        noisy_circuit = stim.Circuit(noisy_circuit_str)
        
        # Use sinter to collect statistics
        task = sinter.Task(
            circuit=noisy_circuit,
            decoder='pymatching',
            json_metadata={'noise_level': noise_level}
        )
        
        # Collect statistics using sinter
        stats = sinter.collect(
            tasks=[task],
            max_shots=shots,
            max_errors=3000,
            num_workers=10
        )
        
        # Extract results
        if stats:
            stat = stats[0]
            logical_error_rate = stat.errors / stat.shots
            logical_errors = stat.errors
            
            # Calculate error bars using binomial distribution
            error_bar = np.sqrt(logical_error_rate * (1 - logical_error_rate) / stat.shots)
            
            results[noise_level] = {
                'logical_error_rate': logical_error_rate,
                'logical_errors': logical_errors,
                'shots': stat.shots,
                'error_bar': error_bar
            }
            
            print(f"    Logical error rate: {logical_error_rate:.6f} ± {error_bar:.6f} ({logical_errors}/{stat.shots})")
        else:
            print(f"    No statistics collected for noise level {noise_level}")
            results[noise_level] = {
                'logical_error_rate': 0.0,
                'logical_errors': 0,
                'shots': 0,
                'error_bar': 0.0
            }
    
    return results


def plot_logical_error_rates(data_dict, save_path="x_junction_error_rates.png"):
    """Plot logical error rates for multiple k values.
    
    Args:
        data_dict: Dict with structure {k_value: {circuit_type: {noise_level: results}}}
    """
    if not data_dict:
        print("Cannot create error rate plot - missing data")
        return
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Define colors for different k values
    k_colors = {1: 'blue', 2: 'red', 3: 'green', 4: 'orange', 5: 'purple'}
    
    # Define line styles for different circuit types
    circuit_styles = {
        'standard': '--',
        'diagonal': '-'
    }
    
    # Define markers for different circuit types
    circuit_markers = {
        'standard': 'o',
        'diagonal': 's'
    }
    
    # Determine all noise levels
    all_noise_levels = set()
    for k_data in data_dict.values():
        for circuit_data in k_data.values():
            all_noise_levels.update(circuit_data.keys())
    noise_levels = sorted(all_noise_levels)
    
    # Plot for each circuit type and k value
    for circuit_type in ['standard', 'diagonal']:
        circuit_label = 'Standard Fixed-Bulk' if circuit_type == 'standard' else 'Diagonal Schedule'
        
        for k in sorted(data_dict.keys()):
            if circuit_type in data_dict[k]:
                circuit_results = data_dict[k][circuit_type]
                if not circuit_results:
                    continue
                    
                # Extract data
                rates = [circuit_results.get(nl, {}).get('logical_error_rate', None) for nl in noise_levels]
                errors = [circuit_results.get(nl, {}).get('error_bar', 0) for nl in noise_levels]
                
                # Get color for this k value and style for this circuit
                color = k_colors.get(k, 'black')
                linestyle = circuit_styles.get(circuit_type, '-')
                marker = circuit_markers.get(circuit_type, 'o')
                
                # Plot with error bars
                plt.errorbar(
                    noise_levels, rates, yerr=errors, 
                    label=f'{circuit_label} (k={k})', 
                    color=color, 
                    linestyle=linestyle,
                    marker=marker,
                    capsize=3, capthick=1, linewidth=2, markersize=6,
                    alpha=0.8
                )
    
    # Customize the plot
    plt.xlabel('Physical Error Rate', fontsize=12)
    plt.ylabel('Logical Error Rate', fontsize=12)
    plt.title('X-Junction Logical Error Rate Comparison', fontsize=14)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.xscale('log')
    
    # Set axis limits
    all_rates = []
    for k_data in data_dict.values():
        for circuit_data in k_data.values():
            if circuit_data:
                all_rates.extend([circuit_data.get(nl, {}).get('logical_error_rate', 0) for nl in noise_levels])
    
    if all_rates:
        min_rate = min([r for r in all_rates if r > 0])
        max_rate = max([r for r in all_rates if r > 0])
        plt.ylim(min_rate * 0.5, max_rate * 2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved error rate plot to {save_path}")
    plt.close()


def compare_results(data_by_k, include_error_rates=False, shots=500000, noise_levels=[0.001, 0.002, 0.005]):
    """Compare and print results side-by-side.
    
    Args:
        data_by_k: Dict with structure {k_value: {'standard': results, 'diagonal': results}}
    """
    print("\n" + "=" * 60)
    print("X-Junction Comparison: Standard vs Diagonal Schedule")
    print("=" * 60)
    print()
    
    # Print basic comparison (using k=1 data if available)
    k_1_data = data_by_k.get(1, {})
    standard = k_1_data.get('standard')
    diagonal = k_1_data.get('diagonal')
    
    if standard and diagonal:
        print(f"{'Metric':<30} {'Standard':<20} {'Diagonal':<20}")
        print("-" * 70)
        print(f"{'Graph-like distance':<30} {standard.get('graphlike_distance', 'N/A'):<20} {diagonal.get('graphlike_distance', 'N/A'):<20}")
        print(f"{'Number of instructions':<30} {standard.get('num_instructions', 'N/A'):<20} {diagonal.get('num_instructions', 'N/A'):<20}")
        print(f"{'Number of qubits':<30} {standard.get('num_qubits', 'N/A'):<20} {diagonal.get('num_qubits', 'N/A'):<20}")
        print(f"{'Number of detectors':<30} {standard.get('num_detectors', 'N/A'):<20} {diagonal.get('num_detectors', 'N/A'):<20}")
        print(f"{'Number of observables':<30} {standard.get('num_observables', 'N/A'):<20} {diagonal.get('num_observables', 'N/A'):<20}")
        print()
        
        print("Standard Crumble URL (k=1):")
        print(standard.get('crumble_url', 'N/A'))
        print()
        
        print("Diagonal Crumble URL (k=1):")
        print(diagonal.get('crumble_url', 'N/A'))
    
    # Add logical error rate comparison if requested
    if include_error_rates and PYMATCHING_AVAILABLE:
        print("\n" + "=" * 60)
        print("Logical Error Rate Analysis")
        print("=" * 60)
        
        all_error_rates = {}
        
        for k in sorted(data_by_k.keys()):
            print(f"\n--- k={k} ---")
            k_data = data_by_k[k]
            
            all_error_rates[k] = {}
            
            if 'standard' in k_data and k_data['standard']:
                print("\nStandard Circuit:")
                standard_error_rates = calculate_logical_error_rate(
                    k_data['standard']['circuit'], 
                    shots=shots, 
                    noise_levels=noise_levels
                )
                all_error_rates[k]['standard'] = standard_error_rates
            
            if 'diagonal' in k_data and k_data['diagonal']:
                print("\nDiagonal Circuit:")
                diagonal_error_rates = calculate_logical_error_rate(
                    k_data['diagonal']['circuit'], 
                    shots=shots, 
                    noise_levels=noise_levels
                )
                all_error_rates[k]['diagonal'] = diagonal_error_rates
        
        # Plot results
        plot_logical_error_rates(all_error_rates)
    elif not PYMATCHING_AVAILABLE:
        print("\n⚠ Skipping error rate calculation - pymatching not available")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare standard vs diagonal X-junction circuits')
    parser.add_argument('--error-rates', action='store_true', 
                       help='Calculate and plot logical error rates (requires pymatching)')
    parser.add_argument('--shots', type=int, default=300000000,
                       help='Number of shots for logical error rate calculation (default: 50000)')
    parser.add_argument('--noise-levels', nargs='+', type=float,
                       default=np.logspace(-3.5, -2, 4),
                       help='Physical error rates to test (default: 0.001 0.002 0.005)')
    
    args = parser.parse_args()
    
    # If error rates are requested, we need to test multiple k values
    k_values = [1, 2, 3] if args.error_rates else [1]
    
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
    
    # Import diagonal convention
    from complete_comparison import create_diagonal_convention
    diagonal_convention = create_diagonal_convention()
    
    # Visualize the block graph as HTML
    print("\nGenerating block graph visualization...")
    try:
        html_viewer = graph.view_as_html(write_html_filepath="x_junction_block_graph.html")
        print("✓ Block graph visualization saved to x_junction_block_graph.html")
    except Exception as e:
        print(f"⚠ Could not generate block graph visualization: {e}")
    
    # Compile for each k value
    all_results = {}
    
    for k in k_values:
        print("\n" + "=" * 60)
        print(f"Processing k={k}")
        print("=" * 60)
        
        # Standard convention
        standard_result = compile_and_generate(
            graph, "Standard Fixed-Bulk", FIXED_BULK_CONVENTION, k=k, use_diagonal=False
        )
        
        # Diagonal convention
        diagonal_result = compile_and_generate(
            graph, "Diagonal Schedule", diagonal_convention, k=k, use_diagonal=False
        )
        
        all_results[k] = {
            'standard': standard_result,
            'diagonal': diagonal_result
        }
    
    # Compare results
    compare_results(
        all_results, 
        include_error_rates=args.error_rates,
        shots=args.shots,
        noise_levels=args.noise_levels
    )
    
    print("\n" + "=" * 60)
    print("✓ Comparison complete!")
    print("=" * 60)

