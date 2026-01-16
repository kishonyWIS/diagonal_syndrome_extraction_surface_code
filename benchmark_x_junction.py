#!/usr/bin/env python3
"""Compare standard vs diagonal schedule X junction implementations."""

import csv
import os
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
constants.MEASUREMENT_SCHEDULE = 8  # Allow schedule values up to 7

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
    """Replace standard RPNG schedule with diagonal schedule.
    
    Standard schedules: X=[1,2,3,5], Z=[1,2,3,5]
    Diagonal schedules: X=[7,5,4,6], Z=[1,3,4,2]
    
    Note: This function is largely unused as we use diagonal convention directly.
    """
    # Convert to string
    desc_str = str(rpng_desc)
    
    # Handle X plaquettes: change from [1,2,3,5] to [7,5,4,6]
    # Pattern: "-x1- -x2- -x3- -x5-" -> "-x7- -x5- -x4- -x6-"
    if 'x' in desc_str:
        desc_str = desc_str.replace('-x1-', '-x7-')
        desc_str = desc_str.replace('-x2-', '-x5-')
        desc_str = desc_str.replace('-x3-', '-x4-')
        desc_str = desc_str.replace('-x5-', '-x6-')
    
    # Handle Z plaquettes: change from [1,2,3,5] to [1,3,4,2]
    # Pattern: "-z1- -z2- -z3- -z5-" -> "-z1- -z3- -z4- -z2-"
    if 'z' in desc_str:
        # z1 stays z1
        desc_str = desc_str.replace('-z2-', '-z3-')
        desc_str = desc_str.replace('-z3-', '-z4-')
        desc_str = desc_str.replace('-z5-', '-z2-')
    
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
    """Compile the graph and generate a Stim circuit (without noise, compacted)."""
    from compact_circuit import compact_and_delay_init
    
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
        
        # Generate circuit WITHOUT noise
        circuit_before = compiled_graph.generate_stim_circuit(
            k=k, 
            manhattan_radius=manhattan_radius,
            noise_model=None
        )
        
        # Compact the circuit (ASAP + ALAP scheduling)
        circuit = compact_and_delay_init(circuit_before)
        print(f"✓ Successfully generated and compacted Stim circuit for k={k}")
        print(f"  Number of instructions: {len(circuit)}")
        print(f"  Number of qubits: {circuit.num_qubits}")
        print(f"  Number of detectors: {circuit.num_detectors}")
        print(f"  Number of observables: {circuit.num_observables}")
        
        # Generate crumble URLs (before and after compactification)
        print("\nGenerating Crumble URLs...")
        try:
            crumble_url_before = circuit_before.to_crumble_url()
            crumble_url_after = circuit.to_crumble_url()
            print(f"✓ Crumble URLs generated")
        except Exception as e:
            print(f"✗ Error generating Crumble URLs: {e}")
            crumble_url_before = None
            crumble_url_after = None
        
        # Calculate graph-like distance (needs noise)
        print("\nCalculating graph-like distance...")
        try:
            # Add noise for distance calculation
            distance_noise_model = NoiseModel.uniform_depolarizing(0.001)
            noisy_circuit_for_distance = distance_noise_model.noisy_circuit(circuit)
            graphlike_errors = noisy_circuit_for_distance.shortest_graphlike_error(canonicalize_circuit_errors=True)
            graphlike_distance = len(graphlike_errors)
            print(f"✓ Graph-like distance: {graphlike_distance}")
        except Exception as e:
            print(f"✗ Error calculating graph-like distance: {e}")
            import traceback
            traceback.print_exc()
            graphlike_distance = None
        
        return {
            'circuit': circuit,
            'circuit_before_compact': circuit_before,
            'crumble_url_before': crumble_url_before,
            'crumble_url_after': crumble_url_after,
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
    """Calculate logical error rate using sinter.
    
    Args:
        circuit: A noise-free stim circuit (noise will be added)
        shots: Number of shots per noise level
        noise_levels: List of physical error rates to test
    """
    if not PYMATCHING_AVAILABLE:
        print("Skipping logical error rate calculation - pymatching not available")
        return {}
    
    print(f"Calculating logical error rate with {shots} shots...")
    
    results = {}
    
    for noise_level in noise_levels:
        print(f"  Testing noise level: {noise_level}")
        
        # Add noise to the circuit using NoiseModel.noisy_circuit()
        noise_model = NoiseModel.uniform_depolarizing(noise_level)
        noisy_circuit = noise_model.noisy_circuit(circuit)
        
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


# =============================================================================
# CSV Data Saving/Loading
# =============================================================================

def save_error_rates_to_csv(all_error_rates, filepath="benchmark_data/x_junction_error_rates_low_error_rates.csv"):
    """Save logical error rate results to CSV file.
    
    Args:
        all_error_rates: Dict with structure {k: {circuit_type: {noise_level: {logical_error_rate, logical_errors, shots, error_bar}}}}
        filepath: Path to save CSV file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    fieldnames = ['k', 'circuit_type', 'physical_error_rate', 'logical_error_rate', 'logical_errors', 'shots', 'error_bar']
    
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for k in sorted(all_error_rates.keys()):
            for circuit_type, circuit_results in all_error_rates[k].items():
                for noise_level, result in circuit_results.items():
                    writer.writerow({
                        'k': k,
                        'circuit_type': circuit_type,
                        'physical_error_rate': noise_level,
                        'logical_error_rate': result['logical_error_rate'],
                        'logical_errors': result['logical_errors'],
                        'shots': result['shots'],
                        'error_bar': result['error_bar'],
                    })
    
    print(f"Saved error rate results to {filepath}")


def save_circuit_info_to_csv(data_by_k, filepath="benchmark_data/x_junction_circuit_info.csv"):
    """Save circuit information to CSV file.
    
    Args:
        data_by_k: Dict with structure {k: {circuit_type: {graphlike_distance, num_instructions, ...}}}
        filepath: Path to save CSV file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    fieldnames = ['k', 'circuit_type', 'graphlike_distance', 'num_instructions', 'num_qubits', 'num_detectors', 'num_observables']
    
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for k in sorted(data_by_k.keys()):
            for circuit_type, info in data_by_k[k].items():
                if info is None:
                    continue
                writer.writerow({
                    'k': k,
                    'circuit_type': circuit_type,
                    'graphlike_distance': info.get('graphlike_distance', 'N/A'),
                    'num_instructions': info.get('num_instructions', 'N/A'),
                    'num_qubits': info.get('num_qubits', 'N/A'),
                    'num_detectors': info.get('num_detectors', 'N/A'),
                    'num_observables': info.get('num_observables', 'N/A'),
                })
    
    print(f"Saved circuit info to {filepath}")


def load_error_rates_from_csv(filepath="benchmark_data/x_junction_error_rates.csv"):
    """Load logical error rate results from CSV file.
    
    Returns:
        Dict with structure {k: {circuit_type: {noise_level: {logical_error_rate, logical_errors, shots, error_bar}}}}
    """
    all_error_rates = {}
    
    with open(filepath, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            k = int(row['k'])
            circuit_type = row['circuit_type']
            noise_level = float(row['physical_error_rate'])
            
            if k not in all_error_rates:
                all_error_rates[k] = {}
            if circuit_type not in all_error_rates[k]:
                all_error_rates[k][circuit_type] = {}
            
            all_error_rates[k][circuit_type][noise_level] = {
                'logical_error_rate': float(row['logical_error_rate']),
                'logical_errors': int(row['logical_errors']),
                'shots': int(row['shots']),
                'error_bar': float(row['error_bar']),
            }
    
    print(f"Loaded error rate results from {filepath}")
    return all_error_rates


def save_crumble_urls_html(urls_dict, output_dir="crumble_urls", experiment_name="x_junction"):
    """Save Crumble URLs as HTML files with clickable links.
    
    Args:
        urls_dict: Dict with structure {k: {circuit_name: {'before': url, 'after': url}}}
        output_dir: Directory to save HTML files
        experiment_name: Name of the experiment for the HTML title
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create an index HTML file
    index_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{experiment_name.replace('_', ' ').title()} - Crumble URLs</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #555; margin-top: 30px; }}
        h3 {{ color: #777; }}
        .circuit {{ margin: 10px 0; padding: 10px; background: #f5f5f5; border-radius: 5px; }}
        a {{ color: #0066cc; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        .before {{ color: #cc6600; }}
        .after {{ color: #006600; }}
    </style>
</head>
<body>
    <h1>{experiment_name.replace('_', ' ').title()} - Crumble Circuit Visualizations</h1>
"""
    
    for k in sorted(urls_dict.keys()):
        index_html += f"    <h2>k = {k}</h2>\n"
        
        for circuit_name, urls in urls_dict[k].items():
            index_html += f"    <div class='circuit'>\n"
            index_html += f"        <h3>{circuit_name}</h3>\n"
            
            if 'before' in urls and urls['before']:
                index_html += f"        <p class='before'>Before compactification: <a href='{urls['before']}' target='_blank'>Open in Crumble</a></p>\n"
            
            if 'after' in urls and urls['after']:
                index_html += f"        <p class='after'>After compactification: <a href='{urls['after']}' target='_blank'>Open in Crumble</a></p>\n"
            
            index_html += f"    </div>\n"
    
    index_html += """</body>
</html>
"""
    
    # Save the index file
    index_path = os.path.join(output_dir, f"{experiment_name}_crumble_urls.html")
    with open(index_path, 'w') as f:
        f.write(index_html)
    
    print(f"Saved Crumble URLs to {index_path}")


def results_to_sinter_stats(results_data):
    """Convert results dictionary to list of sinter.TaskStats for plotting.
    
    Args:
        results_data: Dict with structure {k: {circuit_type: {noise_level: {logical_error_rate, logical_errors, shots, error_bar}}}}
    
    Returns:
        List of sinter.TaskStats objects
    """
    stats_list = []
    
    for k in results_data:
        for circuit_type in results_data[k]:
            circuit_results = results_data[k][circuit_type]
            if not circuit_results:
                continue
            
            # Map circuit_type to display name
            circuit_name = 'N/Z' if circuit_type == 'standard' else 'Diagonal'
            
            for noise_level, result in circuit_results.items():
                # Create TaskStats object
                stats = sinter.TaskStats(
                    strong_id=f"{circuit_type}_k{k}_p{noise_level}",
                    decoder='pymatching',
                    json_metadata={
                        'p': noise_level,
                        'k': k,
                        'd': 2 * k + 1,  # Surface code distance
                        'circuit': circuit_name,
                    },
                    shots=result['shots'],
                    errors=result['logical_errors'],
                )
                stats_list.append(stats)
    
    return stats_list


def fit_and_plot_distance(ax, stats_list, group_func, x_func, plot_args_func, min_points=2):
    """Fit p_logical = A * p^((d+1)/2) to data, add fit lines, and create inset d vs k plot.
    
    Args:
        ax: matplotlib axes object
        stats_list: List of sinter.TaskStats objects
        group_func: Function to group stats into curves (same as used in sinter.plot_error_rate)
        x_func: Function to extract x value (physical error rate)
        plot_args_func: Function to get plot styling for each curve
        min_points: Minimum number of points with reasonable error bars needed for fitting
    """
    from collections import defaultdict
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    # Group stats by curve
    curves = defaultdict(list)
    for s in stats_list:
        curve_id = group_func(s)
        curves[curve_id].append(s)
    
    # Collect fitted distances for inset plot: list of (k, d_eff, plot_args, curve_id)
    fitted_distances = []
    
    for idx, (curve_id, stats) in enumerate(curves.items()):
        # Extract data points: (p, p_logical, error_bar)
        points = []
        k_value = None
        for s in stats:
            if k_value is None:
                k_value = s.json_metadata.get('k')
            if s.errors > 0 and s.shots > 0:
                p = x_func(s)
                p_logical = s.errors / s.shots
                # Approximate error bar (binomial standard error)
                error_bar = np.sqrt(p_logical * (1 - p_logical) / s.shots)
                # Only include points with reasonable error bars (< 10% of value)
                if error_bar < 0.1 * p_logical and p_logical > 0:
                    points.append((p, p_logical, error_bar))
        
        # Sort by physical error rate and take the two lowest
        points.sort(key=lambda x: x[0])
        
        if len(points) < min_points:
            continue
        
        # Use the two points with lowest physical error rate
        fit_points = points[:min_points]
        
        # Fit in log-log space: log(p_logical) = log(A) + slope * log(p)
        # where slope = (d+1)/2, so d = 2*slope - 1
        log_p = np.array([np.log(pt[0]) for pt in fit_points])
        log_p_logical = np.array([np.log(pt[1]) for pt in fit_points])
        
        # Linear fit in log-log space
        slope, intercept = np.polyfit(log_p, log_p_logical, 1)
        
        # Calculate effective distance: slope = (d+1)/2 => d = 2*slope - 1
        d_eff = 2 * slope - 1
        
        # Get plot styling
        plot_args = plot_args_func(idx, curve_id)
        color = plot_args.get('color', 'black')
        marker = plot_args.get('marker', 'o')
        linestyle = plot_args.get('linestyle', '-')
        
        # Generate fit line across fixed x range (8e-5 to 1e-3)
        p_range = np.logspace(np.log10(8e-5), np.log10(1e-3), 100)
        p_logical_fit = np.exp(intercept) * p_range ** slope
        
        # Plot fit line as very faint dotted line
        ax.plot(p_range, p_logical_fit, color=color, linestyle=':', alpha=0.25, linewidth=1.5)
        
        # Store for inset plot
        if k_value is not None:
            fitted_distances.append((k_value, d_eff, color, marker, linestyle, curve_id))
    
    # Create inset plot for d vs k
    if fitted_distances:
        # Create inset axes in bottom right corner
        inset_ax = inset_axes(ax, width="30%", height="30%", loc='lower right', 
                              bbox_to_anchor=(0, 0.08, 1, 1), bbox_transform=ax.transAxes, borderpad=2)
        
        # Group by linestyle to draw black connecting lines (since different k values have same linestyle)
        from collections import defaultdict
        linestyle_groups = defaultdict(list)
        for k, d, color, marker, linestyle, curve_id in fitted_distances:
            linestyle_groups[linestyle].append((k, d, color, marker))
        
        # First draw black connecting lines by linestyle
        for linestyle, points in linestyle_groups.items():
            points.sort(key=lambda x: x[0])  # Sort by k
            ks = [p[0] for p in points]
            ds = [p[1] for p in points]
            inset_ax.plot(ks, ds, color='black', linestyle=linestyle, linewidth=1.5, zorder=1)
        
        # Then plot markers on top with their colors
        for k, d, color, marker, linestyle, curve_id in fitted_distances:
            inset_ax.plot(k, d, color=color, marker=marker, linestyle='none', 
                         markersize=6, zorder=2)
        
        # Style the inset
        inset_ax.set_xlabel('k', fontsize=22)
        inset_ax.set_ylabel('$d_{eff}$', fontsize=22)
        inset_ax.tick_params(axis='both', labelsize=22)
        inset_ax.set_xticks([1, 2, 3])
        # Set y-ticks to integers only
        from matplotlib.ticker import MaxNLocator
        inset_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        inset_ax.grid(True, alpha=0.3)


def plot_logical_error_rates(data_dict, save_path="benchmark_plots/x_junction_error_rates.pdf"):
    """Plot logical error rates using sinter.plot_error_rate (same style as benchmark_memory.py).
    
    Args:
        data_dict: Dict with structure {k_value: {circuit_type: {noise_level: results}}}
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if not data_dict:
        print("Cannot create error rate plot - missing data")
        return
    
    # Convert to sinter stats format
    stats_list = results_to_sinter_stats(data_dict)
    
    if not stats_list:
        print("No data to plot")
        return
    
    # Define colors and markers by k value
    k_colors = {1: 'C0', 2: 'C1', 3: 'C2', 4: 'C3', 5: 'C4'}
    k_markers = {1: 'o', 2: 's', 3: '^', 4: 'D', 5: 'v'}
    
    # Create combined comparison plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Custom plot_args_func for combined plot
    def combined_plot_args(index, curve_id):
        # curve_id is like "N/Z k=1" or "Diagonal k=2"
        parts = curve_id.split()
        circuit_type = parts[0]  # "N/Z" or "Diagonal"
        k = int(parts[1].split('=')[1])  # Extract k value
        
        return {
            'color': k_colors.get(k, 'black'),
            'marker': k_markers.get(k, 'o'),
            'linestyle': '--' if circuit_type == 'N/Z' else '-',
        }
    
    group_func = lambda s: f"{s.json_metadata['circuit']} k={s.json_metadata['k']}"
    x_func = lambda s: s.json_metadata['p']
    
    sinter.plot_error_rate(
        ax=ax,
        stats=stats_list,
        x_func=x_func,
        group_func=group_func,
        plot_args_func=combined_plot_args,
    )
    
    # Add fit lines with distance labels
    fit_and_plot_distance(ax, stats_list, group_func, x_func, combined_plot_args)
    
    ax.loglog()
    ax.set_xlabel("Physical Error Rate", fontsize=22)
    ax.set_ylabel("Logical Error Rate", fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.tick_params(axis='both', which='minor', labelsize=22)
    ax.grid(which='major', alpha=0.5)
    ax.grid(which='minor', alpha=0.2)
    ax.legend(fontsize=22, ncol=2, loc='upper left', columnspacing=0.5, handletextpad=0.3)
    fig.set_dpi(150)
    fig.tight_layout()
    
    # Save plot
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved error rate plot to {save_path}")
    
    plt.close()
    
    return fig


def compare_results(data_by_k, include_error_rates=False, shots=500000, noise_levels=[0.001, 0.002, 0.005]):
    """Compare and print results side-by-side.
    
    Args:
        data_by_k: Dict with structure {k_value: {'standard': results, 'diagonal': results}}
    """
    print("\n" + "=" * 60)
    print("X-Junction Comparison: N/Z vs Diagonal Schedule")
    print("=" * 60)
    print()
    
    # Print basic comparison (using k=1 data if available)
    k_1_data = data_by_k.get(1, {})
    standard = k_1_data.get('standard')
    diagonal = k_1_data.get('diagonal')
    
    if standard and diagonal:
        print(f"{'Metric':<30} {'N/Z':<20} {'Diagonal':<20}")
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
                print("\nN/Z Circuit:")
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
        
        # Save results to CSV
        save_error_rates_to_csv(all_error_rates)
        
        # Plot results
        plot_logical_error_rates(all_error_rates)
    elif not PYMATCHING_AVAILABLE:
        print("\n⚠ Skipping error rate calculation - pymatching not available")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare standard vs diagonal X-junction circuits')
    parser.add_argument('--error-rates', action='store_true', 
                       help='Calculate and plot logical error rates (requires pymatching)')
    parser.add_argument('--shots', type=int, default=2000_000_000,
                       help='Number of shots for logical error rate calculation (default: 50000)')
    parser.add_argument('--noise-levels', nargs='+', type=float,
                       default=np.logspace(-4, -2, 9)[2:],
                       help='Physical error rates to test (default: 0.001 0.002 0.005)')
    parser.add_argument('--plot-only', action='store_true',
                       help='Only generate plots from existing CSV data (skip all computations)')
    parser.add_argument('--load-error-rates', type=str, default=None,
                       help='Path to CSV file with error rates (default: benchmark_data/x_junction_error_rates.csv)')
    
    args = parser.parse_args()
    
    # Handle plot-only mode
    if args.plot_only:
        print("=== PLOT-ONLY MODE ===")
        print("Loading data from CSV files and generating plots...")
        print()
        
        # Load error rates
        error_rates_file = args.load_error_rates or "benchmark_data/x_junction_error_rates.csv"
        try:
            all_error_rates = load_error_rates_from_csv(error_rates_file)
            print(f"Loaded error rates from {error_rates_file}")
            plot_logical_error_rates(all_error_rates)
            print("\nPlot-only mode complete!")
        except FileNotFoundError:
            print(f"Error: {error_rates_file} not found")
            sys.exit(1)
        sys.exit(0)
    
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
    from benchmark_memory import create_diagonal_convention
    diagonal_convention = create_diagonal_convention()
    
    # Visualize the block graph as HTML
    print("\nGenerating block graph visualization...")
    try:
        os.makedirs("benchmark_plots", exist_ok=True)
        html_viewer = graph.view_as_html(write_html_filepath="benchmark_plots/x_junction_block_graph.html")
        print("✓ Block graph visualization saved to benchmark_plots/x_junction_block_graph.html")
    except Exception as e:
        print(f"⚠ Could not generate block graph visualization: {e}")
    
    # Compile for each k value
    all_results = {}
    all_crumble_urls = {}
    
    for k in k_values:
        print("\n" + "=" * 60)
        print(f"Processing k={k}")
        print("=" * 60)
        
        # N/Z convention
        standard_result = compile_and_generate(
            graph, "N/Z Fixed-Bulk", FIXED_BULK_CONVENTION, k=k, use_diagonal=False
        )
        
        # Diagonal convention
        diagonal_result = compile_and_generate(
            graph, "Diagonal Schedule", diagonal_convention, k=k, use_diagonal=False
        )
        
        all_results[k] = {
            'standard': standard_result,
            'diagonal': diagonal_result
        }
        
        # Collect Crumble URLs
        all_crumble_urls[k] = {}
        if standard_result:
            all_crumble_urls[k]['N/Z Fixed-Bulk'] = {
                'before': standard_result.get('crumble_url_before'),
                'after': standard_result.get('crumble_url_after')
            }
        if diagonal_result:
            all_crumble_urls[k]['Diagonal Schedule'] = {
                'before': diagonal_result.get('crumble_url_before'),
                'after': diagonal_result.get('crumble_url_after')
            }
    
    # Save circuit info to CSV
    save_circuit_info_to_csv(all_results)
    
    # Save Crumble URLs to HTML
    if all_crumble_urls:
        save_crumble_urls_html(all_crumble_urls, output_dir="crumble_urls", experiment_name="x_junction")
    
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

