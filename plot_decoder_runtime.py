#!/usr/bin/env python3
"""
Plot decoder runtime per cycle as a function of physical error rate.

Compares different decoders (pymatching, correlated_pymatching, tesseract)
across k values, using only the "partial" flag configuration.

Usage:
    python plot_decoder_runtime.py --csv cluster_benchmark/combined_results.csv
    python plot_decoder_runtime.py --csv cluster_benchmark/combined_results.csv --output-dir benchmark_plots
"""

import argparse
import csv
import os
from collections import defaultdict
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.ticker import LogLocator, NullFormatter

# Use LaTeX-like fonts (STIX)
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

# =============================================================================
# Style Constants
# =============================================================================

# k determines color (same palette as plot_logical_error_rates.py)
K_COLORS = {
    1: '#1f77b4',  # blue (C0)
    2: '#ff7f0e',  # orange (C1)
    3: '#2ca02c',  # green (C2)
    4: '#d62728',  # red (C3)
    5: '#9467bd',  # purple (C4)
    6: '#8c564b',  # brown (C5)
}

# Decoder determines marker AND linestyle
# Using markers not used in other plots (avoiding 'o', 's', '^')
DECODER_STYLES = {
    'pymatching': {'marker': 'D', 'linestyle': ':', 'label': 'Match.'},              # dotted
    'correlated_pymatching': {'marker': 'v', 'linestyle': '--', 'label': 'Corr. Match.'},  # dashed
    'tesseract': {'marker': 'p', 'linestyle': '-', 'label': 'Tesseract'},             # solid
}

# Plot styling constants
FONT_SIZE = 25.5
LEGEND_FONT_SIZE = 23
MARKER_SIZE = 8


# =============================================================================
# Core Functions
# =============================================================================

def load_runtime_data(
    csv_paths: list,
    flag_config: str = 'partial',
    direction: str = 'y',
    exclude_decoders: Optional[dict] = None,
) -> list:
    """
    Load CSV files and compute decode time per cycle.
    
    Supports both formats:
    - Full format with direction, flag_config columns (from spatial_hadamard benchmarks)
    - Simplified format without those columns (from dedicated runtime benchmark)
    
    Args:
        csv_paths: List of paths to CSV files
        flag_config: Filter for this flag_config value (default: 'partial')
        direction: Filter for this direction value (default: 'y')
        exclude_decoders: Dict mapping csv_path to list of decoder names to exclude from that file
        
    Returns:
        List of dicts with keys: k, decoder, physical_error_rate, decode_time_per_cycle
    """
    data = []
    exclude_decoders = exclude_decoders or {}
    
    for csv_path in csv_paths:
        excluded = exclude_decoders.get(csv_path, [])
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Check if this is the simplified format (no direction/flag_config columns)
                has_direction = 'direction' in row
                has_flag_config = 'flag_config' in row
                
                # Filter for specified flag_config and direction if columns exist
                if has_flag_config and row['flag_config'] != flag_config:
                    continue
                if has_direction and row['direction'] != direction:
                    continue
                
                decoder = row['decoder']
                
                # Skip excluded decoders for this file
                if decoder in excluded:
                    continue
                
                try:
                    k = int(row['k'])
                    physical_error_rate = float(row['physical_error_rate'])
                    decode_time = float(row['decode_time'])
                    shots = int(row['shots'])
                    
                    # Skip rows with no data
                    if shots == 0 or decode_time <= 0:
                        continue
                    
                    # Compute decode time per cycle
                    # d = 2k + 1 is the number of syndrome extraction rounds per shot
                    d = 2 * k + 1
                    decode_time_per_cycle = decode_time / (shots * d)
                    
                    # Convert to microseconds for readability
                    decode_time_per_cycle_us = decode_time_per_cycle * 1e6
                    
                    data.append({
                        'k': k,
                        'decoder': decoder,
                        'physical_error_rate': physical_error_rate,
                        'decode_time_per_cycle_us': decode_time_per_cycle_us,
                    })
                    
                except (ValueError, KeyError) as e:
                    print(f"Warning: skipping row due to error: {e}")
    
    return data


def plot_decoder_runtime(
    csv_paths: list,
    output_dir: str = 'benchmark_plots',
    output_filename: str = 'decoder_runtime.pdf',
    exclude_decoders_override: Optional[dict] = None,
    y_max_multiplier: float = 1.0,
):
    """
    Plot decoder runtime per cycle vs physical error rate.
    
    Args:
        csv_paths: List of paths to CSV files with benchmark data
        output_dir: Output directory for plots
        output_filename: Output filename
        exclude_decoders_override: Optional dict to override default exclude_decoders
    """
    print(f"Loading data from {csv_paths}")
    
    # Default: exclude tesseract from backup file (use dedicated tesseract file instead)
    if exclude_decoders_override is None:
        exclude_decoders = {
            'benchmark_data/spatial_hadamard_benchmark_backup.csv': ['tesseract'],
        }
    else:
        exclude_decoders = exclude_decoders_override
    
    data = load_runtime_data(csv_paths, flag_config='partial', direction='y',
                             exclude_decoders=exclude_decoders)
    
    if not data:
        print("No data to plot")
        return
    
    # Group data by (decoder, k)
    curves = defaultdict(list)
    for d in data:
        key = (d['decoder'], d['k'])
        curves[key].append((d['physical_error_rate'], d['decode_time_per_cycle_us']))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot each curve
    for (decoder, k), points in sorted(curves.items()):
        # Sort by physical error rate
        points.sort(key=lambda x: x[0])
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        
        style = DECODER_STYLES.get(decoder, {'marker': 'o', 'linestyle': '-'})
        color = K_COLORS.get(k, 'gray')
        
        ax.plot(
            x_vals, y_vals,
            color=color,
            marker=style['marker'],
            linestyle=style['linestyle'],
            markersize=MARKER_SIZE,
            label=f"{decoder} k={k}",
        )
    
    # Style the axes
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Physical Error Rate", fontsize=FONT_SIZE)
    ax.set_ylabel("Decode Time per Cycle (μs)", fontsize=FONT_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    
    # Minor ticks
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    
    ax.grid(which='major', alpha=0.5)
    ax.grid(which='minor', alpha=0.2)
    
    # Set y-axis limits if multiplier specified
    if y_max_multiplier > 1.0:
        y_max = max(d['decode_time_per_cycle_us'] for d in data)
        ax.set_ylim(top=y_max * y_max_multiplier)
    
    # Create dual legend
    _create_dual_legend(ax, data)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    fig.set_dpi(150)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def _create_dual_legend(ax: plt.Axes, data: list[dict]):
    """
    Create dual legend: k values (color patches) on top, decoder types below.
    
    Args:
        ax: matplotlib axes
        data: List of data dicts to extract unique k and decoder values
    """
    # Get unique k values and decoders
    k_values = sorted(set(d['k'] for d in data))
    # Define decoder order for legend
    decoder_order = ['pymatching', 'correlated_pymatching', 'tesseract']
    available_decoders = set(d['decoder'] for d in data)
    decoders = [dec for dec in decoder_order if dec in available_decoders]
    
    # Create k legend handles using rectangular color patches
    k_handles = [
        mpatches.Patch(color=K_COLORS.get(k, 'gray'), label=f'k={k}')
        for k in k_values
    ]
    
    # Add k legend (top)
    legend1 = ax.legend(
        handles=k_handles, loc='upper left',
        ncol=min(len(k_values), 6), fontsize=LEGEND_FONT_SIZE,
        columnspacing=0.5, handletextpad=0.3, handlelength=1.0
    )
    ax.add_artist(legend1)
    
    # Create decoder legend handles
    decoder_handles = [
        Line2D([0], [0], color='gray',
               linestyle=DECODER_STYLES[dec]['linestyle'],
               marker=DECODER_STYLES[dec]['marker'],
               markersize=6, label=DECODER_STYLES[dec]['label'])
        for dec in decoders if dec in DECODER_STYLES
    ]
    
    # Add decoder legend (below k legend) in 3 columns
    ax.legend(
        handles=decoder_handles, loc='upper left',
        ncol=3, fontsize=LEGEND_FONT_SIZE * 0.85,
        bbox_to_anchor=(0.0, 0.9), handlelength=2.0,
        columnspacing=0.8, handletextpad=0.3
    )


# =============================================================================
# CLI Interface
# =============================================================================

def plot_all():
    """Generate all decoder runtime plots."""
    output_dir = 'benchmark_plots'
    
    # Plot 1: Standard benchmark data
    print("\n=== Generating standard decoder runtime plot ===")
    plot_decoder_runtime(
        csv_paths=['benchmark_data/spatial_hadamard_benchmark_backup.csv',
                   'benchmark_data/spatial_hadamard_benchmark_y_only_tesseract_only.csv'],
        output_dir=output_dir,
        output_filename='decoder_runtime.pdf',
        exclude_decoders_override={
            'benchmark_data/spatial_hadamard_benchmark_backup.csv': ['tesseract'],
        },
    )
    
    # Plot 2: Interface (cluster benchmark) data
    print("\n=== Generating interface decoder runtime plot ===")
    plot_decoder_runtime(
        csv_paths=['cluster_benchmark/combined_results.csv'],
        output_dir=output_dir,
        output_filename='decoder_runtime_interface.pdf',
        exclude_decoders_override=None,
    )
    
    # Plot 3: Dedicated runtime benchmark (if exists)
    dedicated_csv = 'benchmark_data/decoder_runtime_benchmark.csv'
    if os.path.exists(dedicated_csv):
        print("\n=== Generating dedicated runtime benchmark plot ===")
        plot_decoder_runtime(
            csv_paths=[dedicated_csv],
            output_dir=output_dir,
            output_filename='decoder_runtime_dedicated.pdf',
            exclude_decoders_override=None,
            y_max_multiplier=10.0,
        )


def main():
    parser = argparse.ArgumentParser(
        description='Plot decoder runtime per cycle vs physical error rate',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all                    # Generate both standard and interface plots
  %(prog)s --csv file1.csv file2.csv
  %(prog)s --csv benchmark_data/spatial_hadamard_benchmark_backup.csv --output-dir benchmark_plots
        """
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Generate both standard and interface plots'
    )
    parser.add_argument(
        '--csv', '-c',
        nargs='+',
        default=['benchmark_data/spatial_hadamard_benchmark_backup.csv',
                 'benchmark_data/spatial_hadamard_benchmark_y_only_tesseract_only.csv'],
        help='Path(s) to CSV file(s) with benchmark data'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='benchmark_plots',
        help='Output directory for plots (default: benchmark_plots)'
    )
    parser.add_argument(
        '--output-filename', '-f',
        default='decoder_runtime.pdf',
        help='Output filename (default: decoder_runtime.pdf)'
    )
    parser.add_argument(
        '--y-max-multiplier',
        type=float,
        default=1.0,
        help='Multiplier for y-axis max (e.g., 15 to extend to 15x max data point)'
    )
    
    args = parser.parse_args()
    
    if args.all:
        plot_all()
    else:
        plot_decoder_runtime(
            csv_paths=args.csv,
            output_dir=args.output_dir,
            output_filename=args.output_filename,
            y_max_multiplier=args.y_max_multiplier,
        )


if __name__ == '__main__':
    main()
