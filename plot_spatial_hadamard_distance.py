#!/usr/bin/env python3
"""
Plot circuit-level distance vs. k for spatial Hadamard circuits.

Uses ILP-based exact distance computation with a fixed number of syndrome
extraction cycles (num_cycles=2) for fast computation regardless of k.

Expected results:
- With 'partial' or 'all' flags: full distance = 2k+1
- With 'none' flags: reduced distance ≈ k+1 (factor of ~2 reduction)
"""

import argparse
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from tqec import NoiseModel

from spatial_hadamard_manual_construction import generate_spatial_hadamard_circuit
from compact_circuit import compact_and_delay_init
from ilp_circuit_distance import mip_circuit_distance
from benchmark_spatial_hadamard import apply_interface_only_noise

# Default CSV path for saving/loading data
DEFAULT_CSV_PATH = 'benchmark_data/spatial_hadamard_distance.csv'


def get_circuit_distance(
    k: int, 
    direction: str, 
    flag_config: str, 
    num_cycles: int = 2,
    time_limit: int = 300,
    noise_mode: str = 'full',
    interface_margin: float = 1.0,
) -> int:
    """Generate a spatial Hadamard circuit and compute its exact distance via ILP.
    
    Args:
        k: Scaling factor (expected distance = 2k+1 for full distance)
        direction: 'x' or 'y'
        flag_config: 'all', 'partial', or 'none'
        num_cycles: Number of syndrome extraction cycles (default 2 for fast ILP)
        time_limit: ILP solver time limit in seconds
        noise_mode: 'full' (all qubits) or 'interface_only' (only qubits near interface)
        interface_margin: How far from interface to include qubits (for interface_only mode)
        
    Returns:
        Exact circuit-level distance
    """
    # Generate circuit without noise, with fixed number of cycles
    circuit = generate_spatial_hadamard_circuit(
        k=k,
        axis=direction,
        noise_model=None,
        flag_config=flag_config,
        num_cycles=num_cycles,
    )
    
    # Compact the circuit
    circuit = compact_and_delay_init(circuit)
    
    # Add noise for distance calculation
    noise_model = NoiseModel.uniform_depolarizing(0.001)
    if noise_mode == 'interface_only':
        noisy_circuit = apply_interface_only_noise(
            circuit, noise_model, k, direction, interface_margin
        )
    else:
        noisy_circuit = noise_model.noisy_circuit(circuit)
    
    # Compute exact distance via ILP
    dem = noisy_circuit.detector_error_model(decompose_errors=False)
    result = mip_circuit_distance(dem, time_limit=time_limit, verbose=False)
    
    if result["status"] not in ["OPTIMAL", "FEASIBLE"]:
        raise RuntimeError(f"ILP solver failed with status: {result['status']}")
    
    return result["distance"]


def compute_all_distances(
    k_values: list[int], 
    num_cycles: int = 2, 
    time_limit: int = 300,
    noise_mode: str = 'full',
    interface_margin: float = 1.0,
) -> dict:
    """Compute distances for all configurations using ILP.
    
    Args:
        k_values: List of k values to test
        num_cycles: Number of syndrome extraction cycles (default 2 for speed)
        time_limit: ILP solver time limit in seconds
        noise_mode: 'full' (all qubits) or 'interface_only' (only qubits near interface)
        interface_margin: How far from interface to include qubits (for interface_only mode)
    
    Returns:
        Dictionary mapping (direction, flag_config) to list of distances for each k
    """
    directions = ['x', 'y']
    flag_configs = ['none', 'partial', 'all']
    
    results = {}
    
    total = len(directions) * len(flag_configs) * len(k_values)
    count = 0
    
    print(f"Using {num_cycles} syndrome extraction cycles for all circuits")
    print(f"Noise mode: {noise_mode}" + (f" (margin={interface_margin})" if noise_mode == 'interface_only' else ""))
    print()
    
    for direction in directions:
        for flag_config in flag_configs:
            key = (direction, flag_config)
            results[key] = []
            
            for k in k_values:
                count += 1
                print(f"[{count}/{total}] k={k}, dir={direction}, flags={flag_config}...", end=" ", flush=True)
                
                try:
                    distance = get_circuit_distance(
                        k, direction, flag_config, 
                        num_cycles=num_cycles,
                        time_limit=time_limit,
                        noise_mode=noise_mode,
                        interface_margin=interface_margin,
                    )
                    results[key].append(distance)
                    print(f"d={distance}")
                except Exception as e:
                    print(f"ERROR: {e}")
                    results[key].append(None)
    
    return results


def save_results_to_csv(k_values: list[int], results: dict, csv_path: str):
    """Save distance results to a CSV file.
    
    CSV format:
    direction,flag_config,k,distance
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['direction', 'flag_config', 'k', 'distance'])
        
        for (direction, flag_config), distances in results.items():
            for k, d in zip(k_values, distances):
                writer.writerow([direction, flag_config, k, d if d is not None else ''])
    
    print(f"\nResults saved to: {csv_path}")


def load_results_from_csv(csv_path: str) -> tuple[list[int], dict]:
    """Load distance results from a CSV file.
    
    Returns:
        Tuple of (k_values, results dict)
    """
    results = {}
    k_values_set = set()
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            direction = row['direction']
            flag_config = row['flag_config']
            k = int(row['k'])
            distance = int(row['distance']) if row['distance'] else None
            
            k_values_set.add(k)
            key = (direction, flag_config)
            
            if key not in results:
                results[key] = {}
            results[key][k] = distance
    
    # Convert to sorted list and ensure results are in order
    k_values = sorted(k_values_set)
    
    # Convert results dict of dicts to dict of lists (ordered by k)
    results_ordered = {}
    for key, k_dict in results.items():
        results_ordered[key] = [k_dict.get(k) for k in k_values]
    
    print(f"Loaded results from: {csv_path}")
    print(f"  k values: {k_values}")
    print(f"  Configurations: {list(results_ordered.keys())}")
    
    return k_values, results_ordered


def plot_distances(k_values: list[int], results: dict, output_path: str = None):
    """Plot distance vs. k for all configurations."""
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Define styles for each configuration
    # Use different colors for flag configs, different markers for directions
    colors = {'none': 'red', 'partial': 'blue', 'all': 'green'}
    markers = {'x': 'o', 'y': 's'}
    linestyles = {'none': '--', 'partial': '-', 'all': '-'}
    
    # Track plotted lines for labels
    plotted_lines = []
    
    # Plot each configuration
    for (direction, flag_config), distances in results.items():
        color = colors[flag_config]
        marker = markers[direction]
        linestyle = linestyles[flag_config]
        
        # Extract valid data
        valid_k = [k for k, d in zip(k_values, distances) if d is not None]
        valid_d = [d for d in distances if d is not None]
        
        if valid_d:
            # Offset slightly to reduce overlap
            offset = 0.02 if direction == 'y' else 0
            
            label = f"{direction}, {flag_config}"
            line, = ax.plot(
                [k + offset for k in valid_k], 
                valid_d, 
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=2, 
                markersize=10,
                markerfacecolor=color if direction == 'x' else 'white',
                markeredgecolor=color,
                markeredgewidth=2,
                label=label,
                alpha=0.8,
            )
            plotted_lines.append((line, label, valid_k[-1], valid_d[-1], color))
    
    # Add text labels at the end of lines with vertical offsets to avoid overlap
    # Group lines by their ending distance value to apply offsets
    distance_groups = {}
    for item in plotted_lines:
        line, label, last_k, last_d, color = item
        if last_d not in distance_groups:
            distance_groups[last_d] = []
        distance_groups[last_d].append(item)
    
    # Apply vertical offsets for overlapping labels
    for last_d, items in distance_groups.items():
        n_items = len(items)
        if n_items == 1:
            offsets = [0]
        else:
            # Spread labels vertically
            total_spread = 0.35 * (n_items - 1)
            offsets = [i * 0.35 - total_spread / 2 for i in range(n_items)]
        
        for (line, label, last_k, item_last_d, color), v_offset in zip(items, offsets):
            ax.annotate(
                label,
                xy=(last_k, item_last_d),
                xytext=(last_k + 0.15, item_last_d + v_offset),
                fontsize=9,
                color=color,
                ha='left',
                va='center',
                fontweight='bold',
            )
    
    # Plot expected full distance line (2k+1)
    k_range = np.array(k_values)
    ax.plot(k_range, 2 * k_range + 1, 'k:', linewidth=2, alpha=0.4, label='Full distance (2k+1)')
    
    # Plot expected reduced distance line (k+1)
    ax.plot(k_range, k_range + 1, 'k-.', linewidth=2, alpha=0.4, label='Reduced distance (k+1)')
    
    # Formatting
    ax.set_xlabel('k (code distance parameter)', fontsize=12)
    ax.set_ylabel('Circuit-level distance', fontsize=12)
    ax.set_title('Spatial Hadamard Circuit Distance\n(ILP exact computation, 2 cycles)', fontsize=14)
    
    # Set integer ticks
    ax.set_xticks(k_values)
    all_distances = [d for distances in results.values() for d in distances if d is not None]
    if all_distances:
        max_d = max(all_distances)
        ax.set_yticks(range(1, max_d + 2))
    
    # Extend x-axis for labels
    ax.set_xlim(min(k_values) - 0.3, max(k_values) + 0.8)
    
    ax.grid(True, alpha=0.3)
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', linestyle=':', linewidth=2, label='Full (2k+1)'),
        Line2D([0], [0], color='black', linestyle='-.', linewidth=2, label='Reduced (k+1)'),
        Line2D([0], [0], color='red', linestyle='--', marker='o', markersize=8, label='flags=none'),
        Line2D([0], [0], color='blue', linestyle='-', marker='o', markersize=8, label='flags=partial'),
        Line2D([0], [0], color='green', linestyle='-', marker='o', markersize=8, label='flags=all'),
        Line2D([0], [0], color='gray', marker='o', markersize=8, linestyle='None', markerfacecolor='gray', label='dir=x (filled)'),
        Line2D([0], [0], color='gray', marker='s', markersize=8, linestyle='None', markerfacecolor='white', markeredgewidth=2, label='dir=y (hollow)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Plot circuit-level distance vs. k for spatial Hadamard circuits'
    )
    parser.add_argument(
        '--k-values', 
        nargs='+', 
        type=int, 
        default=[1, 2, 3],
        help='k values to test (default: 1 2 3)'
    )
    parser.add_argument(
        '--num-cycles',
        type=int,
        default=2,
        help='Number of syndrome extraction cycles (default: 2 for fast ILP)'
    )
    parser.add_argument(
        '--time-limit',
        type=int,
        default=300,
        help='ILP solver time limit in seconds (default: 300)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='benchmark_plots/spatial_hadamard_distance_vs_k.pdf',
        help='Output file path for the plot'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default=DEFAULT_CSV_PATH,
        help=f'CSV file path for saving/loading data (default: {DEFAULT_CSV_PATH})'
    )
    parser.add_argument(
        '--from-csv',
        action='store_true',
        help='Load data from CSV instead of computing (regenerate plot from saved data)'
    )
    parser.add_argument(
        '--skip-csv-save',
        action='store_true',
        help='Do not save results to CSV after computing'
    )
    parser.add_argument(
        '--noise-mode',
        type=str,
        choices=['full', 'interface_only'],
        default='full',
        help='Noise mode: full (all qubits) or interface_only (only near interface)'
    )
    parser.add_argument(
        '--interface-margin',
        type=float,
        default=1.0,
        help='For interface_only mode: how far from interface to include qubits (default: 1.0)'
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("Spatial Hadamard Circuit Distance vs. k")
    print("=" * 70)
    
    if args.from_csv:
        # Load from CSV
        print(f"Loading data from: {args.csv}")
        print(f"Output: {args.output}")
        print("=" * 70)
        print()
        
        k_values, results = load_results_from_csv(args.csv)
    else:
        # Compute distances
        print(f"k values: {args.k_values}")
        print(f"Number of syndrome extraction cycles: {args.num_cycles}")
        print(f"Noise mode: {args.noise_mode}" + (f" (margin={args.interface_margin})" if args.noise_mode == 'interface_only' else ""))
        print(f"Method: ILP (exact)")
        print(f"Time limit per circuit: {args.time_limit}s")
        print(f"Output: {args.output}")
        print(f"CSV: {args.csv}")
        print("=" * 70)
        print()
        
        # Compute all distances
        results = compute_all_distances(
            args.k_values, 
            num_cycles=args.num_cycles, 
            time_limit=args.time_limit,
            noise_mode=args.noise_mode,
            interface_margin=args.interface_margin,
        )
        k_values = args.k_values
        
        # Save to CSV
        if not args.skip_csv_save:
            save_results_to_csv(k_values, results, args.csv)
    
    # Print summary table
    print("\n" + "=" * 70)
    print("Summary: Distance vs. k")
    print("=" * 70)
    print(f"{'Config':<25} " + " ".join(f"k={k:>3}" for k in k_values))
    print("-" * 70)
    for (direction, flag_config), distances in results.items():
        label = f"dir={direction}, flags={flag_config}"
        dist_strs = [str(d) if d is not None else "N/A" for d in distances]
        print(f"{label:<25} " + " ".join(f"{d:>4}" for d in dist_strs))
    
    print("-" * 70)
    print(f"{'Expected full (2k+1)':<25} " + " ".join(f"{2*k+1:>4}" for k in k_values))
    print(f"{'Expected reduced (k+1)':<25} " + " ".join(f"{k+1:>4}" for k in k_values))
    
    # Plot
    print()
    plot_distances(k_values, results, output_path=args.output)


if __name__ == "__main__":
    main()
