#!/usr/bin/env python3
"""
Plot logical error rates for spatial Hadamard benchmarks.
Generates one plot per decoder with effective distance inset.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
import sinter

# Style configuration
COLORS = {
    1: '#1f77b4',  # blue
    2: '#ff7f0e',  # orange  
    3: '#2ca02c',  # green
    4: '#d62728',  # red
    5: '#9467bd',  # purple
    6: '#8c564b',  # brown
}

MARKERS = {
    'none': 'o',
    'partial': 's', 
    'all': '^',
}

LINESTYLES = {
    'none': ':',
    'partial': '--',
    'all': '-',
}

FLAG_LABELS = {
    'none': 'none',
    'partial': 'partial',
    'all': 'all',
}


def load_results(csv_path: str) -> list[dict]:
    """Load benchmark results from CSV file."""
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
            results.append({
                    'decoder': row['decoder'],
                'k': int(row['k']),
                'physical_error_rate': float(row['physical_error_rate']),
                'logical_error_rate': float(row['logical_error_rate']),
                'errors': int(row['errors']),
                'shots': int(row['shots']),
                'error_bar': float(row['error_bar']),
                    'flag_config': row['flag_config'],
                    'direction': row['direction'],
            })
            except (ValueError, KeyError) as e:
                print(f"Warning: skipping row due to error: {e}")
    return results


def results_to_sinter_stats(results: list[dict], decoder_filter: str = None) -> list[sinter.TaskStats]:
    """Convert results to sinter.TaskStats for plotting.
    
    Args:
        results: List of result dictionaries
        decoder_filter: If specified, only include results for this decoder
    
    Returns:
        List of sinter.TaskStats objects
    """
    stats_list = []
    
    for result in results:
        if decoder_filter and result['decoder'] != decoder_filter:
            continue
        
        stats = sinter.TaskStats(
            strong_id=f"{result['direction']}_{result['flag_config']}_k{result['k']}_p{result['physical_error_rate']}_{result['decoder']}",
            decoder=result['decoder'],
            json_metadata={
                'p': result['physical_error_rate'],
                'k': result['k'],
                'd': 2 * result['k'] + 1,
                'direction': result['direction'],
                'flag_config': result['flag_config'],
                'decoder': result['decoder'],
            },
            shots=result['shots'],
            errors=result['errors'],
        )
        stats_list.append(stats)
    
    return stats_list


def fit_and_plot_distance(ax, stats_list, group_func, x_func, plot_args_func, min_points=2):
    """
    Fit power law and plot fit lines with distance labels.
    Works with sinter.TaskStats objects.
    Returns dict of (group_key -> d_eff) for inset plotting.
    """
    from collections import defaultdict
    
    # Group data
    groups = defaultdict(list)
    for stat in stats_list:
        key = group_func(stat)
        groups[key].append(stat)
    
    d_eff_results = {}
    
    for key, stats in sorted(groups.items()):
        # Filter for valid points (errors > 0, shots > 0)
        valid_stats = []
        for s in stats:
            if s.errors > 0 and s.shots > 0:
                p_logical = s.errors / s.shots
                error_bar = np.sqrt(p_logical * (1 - p_logical) / s.shots)
                # Only include points with reasonable error bars (< 10% of value)
                if error_bar < 0.1 * p_logical:
                    valid_stats.append(s)
        
        if len(valid_stats) < min_points:
            print(f"Skipping {key}: only {len(valid_stats)} valid points")
            continue
        
        # Sort by physical error rate and take the lowest min_points
        valid_stats.sort(key=lambda s: x_func(s))
        fit_stats = valid_stats[:min_points]
        
        if len(fit_stats) < min_points:
            print(f"Using {len(fit_stats)} points for {key} (requested {min_points})")
        
        # Fit log(p_logical) = d * log(p) + const
        log_p = np.array([np.log10(x_func(s)) for s in fit_stats])
        log_p_logical = np.array([np.log10(s.errors / s.shots) for s in fit_stats])
        
        # Weight by inverse variance (error bar)
        weights = []
        for s in fit_stats:
            p_logical = s.errors / s.shots
            error_bar = np.sqrt(p_logical * (1 - p_logical) / s.shots)
            weights.append(1.0 / (error_bar / p_logical)**2)
        weights = np.array(weights)
        
        slope, intercept = np.polyfit(log_p, log_p_logical, 1, w=weights)
        d_eff = slope  # effective distance
        
        fit_points_str = ", ".join([f"{x_func(s):.2e}" for s in fit_stats])
        print(f"{key}: d_eff = {d_eff:.2f} (fit points: {fit_points_str})")
        
        d_eff_results[key] = d_eff
        
        # Plot fit line
        p_range = np.logspace(np.log10(8e-5), np.log10(2e-3), 100)
        fit_line = 10**(slope * np.log10(p_range) + intercept)
        
        plot_args = plot_args_func(0, key)
        ax.plot(p_range, fit_line, 
                color=plot_args.get('color', 'gray'),
                linestyle=':',
                alpha=0.25, linewidth=1.5)
        
    return d_eff_results


def plot_decoder_results(decoder: str, results: list[dict], output_dir: str):
    """Create plot for a single decoder using sinter.plot_error_rate."""
    # Filter results for this decoder
    decoder_results = [r for r in results if r['decoder'] == decoder]
    
    if not decoder_results:
        print(f"No results for decoder: {decoder}")
        return
    
    print(f"\nLoaded {len(decoder_results)} results for {decoder} (out of {len(results)} total)")
        
    # Convert to sinter stats
    stats_list = results_to_sinter_stats(decoder_results)
        
    # Create figure (matching benchmark_spatial_hadamard.py)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get unique k values and flag configs present in data
    k_values_present = sorted(set(r['k'] for r in decoder_results))
    flag_configs_present = sorted(set(r['flag_config'] for r in decoder_results))
    
    # Define the desired order for flag_config (used for sorting legend)
    flag_order = {'none': 0, 'partial': 1, 'all': 2}
    
    # Group function for sinter plotting
    def group_func(s):
        return f"{flag_order[s.json_metadata['flag_config']]}_{s.json_metadata['flag_config']} k={s.json_metadata['k']}"
    
    def x_func(s):
        return s.json_metadata['p']
    
    def plot_args_func(index, curve_id):
        # curve_id format: "0_none k=1" or "1_partial k=2" (with sort prefix)
        parts = curve_id.split()
        # Remove the sort prefix (e.g., "0_none" -> "none")
        flag_config = parts[0].split('_')[1] if '_' in parts[0] else parts[0]
        k = int(parts[1].split('=')[1])
        
        return {
            'color': COLORS.get(k, 'gray'),
            'marker': MARKERS.get(flag_config, 'o'),
            'linestyle': LINESTYLES.get(flag_config, '-'),
        }
    
    # Use sinter.plot_error_rate for proper error bar display
    sinter.plot_error_rate(
        ax=ax,
        stats=stats_list,
        x_func=x_func,
        group_func=group_func,
        plot_args_func=plot_args_func,
    )
    
    # Add fit lines with distance labels (using 2 lowest error rate points for fitting)
    d_eff_results = fit_and_plot_distance(ax, stats_list, group_func, x_func, plot_args_func, min_points=2)
    
    # Fix legend labels by removing the sort prefix from sinter's auto-generated legend
    handles, labels = ax.get_legend_handles_labels()
    # Remove the legend created by sinter (if it exists)
    if ax.legend_ is not None:
        ax.legend_.remove()
    
    # Create two separate legends: k values (colors) and flag configs (markers/linestyles)
    # Legend 1: k values (colors)
    k_handles = [Line2D([0], [0], color=COLORS.get(k, 'gray'), linestyle='-', marker='o', 
                        markersize=6, label=f'k={k}') 
                 for k in k_values_present]
    
    # Legend 2: flag configs (markers/linestyles)  
    flag_handles = [Line2D([0], [0], color='gray', linestyle=LINESTYLES.get(fc, '-'),
                           marker=MARKERS.get(fc, 'o'), markersize=6, 
                           label=FLAG_LABELS.get(fc, fc))
                    for fc in ['none', 'partial', 'all'] if fc in flag_configs_present]
    
    # Add first legend (k values) at top left (fontsize=18 to match benchmark_spatial_hadamard.py)
    legend1 = ax.legend(handles=k_handles, loc='upper left', 
                        ncol=len(k_values_present), fontsize=18,
                        columnspacing=0.5, handletextpad=0.3)
    ax.add_artist(legend1)
    
    # Add second legend (flag configs) below first
    legend2 = ax.legend(handles=flag_handles, loc='upper left',
                        ncol=len(flag_configs_present), fontsize=18,
                        bbox_to_anchor=(0.0, 0.88), handlelength=3.0,
                        columnspacing=0.5, handletextpad=0.3)
    
    # Set axis properties (matching benchmark_spatial_hadamard.py)
    ax.loglog()
    ax.set_xlim(left=7e-5)
    ax.set_ylim(bottom=5e-9, top=2e-1)
    ax.set_xlabel("Physical Error Rate", fontsize=22)
    ax.set_ylabel("Logical Error Rate", fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.tick_params(axis='both', which='minor', labelsize=22)
    ax.grid(which='major', alpha=0.5)
    ax.grid(which='minor', alpha=0.2)
    # No title for standalone figures
    
    # Create inset for effective distance vs k
    if d_eff_results:
        ax_inset = inset_axes(ax, width="30%", height="30%", loc='lower right',
                              bbox_to_anchor=(0, 0.08, 1, 1), bbox_transform=ax.transAxes, borderpad=2)
        
        # Plot d_eff vs k for each flag config, with markers colored by k
        for flag_config in ['none', 'partial', 'all']:
            k_vals = []
            d_vals = []
            for key, d_eff in d_eff_results.items():
                if flag_config in key:
                    # Extract k from key like "0_none k=2"
                    k = int(key.split('k=')[1])
                    k_vals.append(k)
                    d_vals.append(d_eff)
            
            if k_vals:
                # Sort by k
                sorted_pairs = sorted(zip(k_vals, d_vals))
                k_vals, d_vals = zip(*sorted_pairs)
                
                # Plot line connecting points (black, matching benchmark_spatial_hadamard.py)
                ax_inset.plot(k_vals, d_vals, 
                             linestyle=LINESTYLES.get(flag_config, '-'),
                             color='black', linewidth=1.5, zorder=1)
                
                # Plot individual markers colored by k
                for k, d in zip(k_vals, d_vals):
                    ax_inset.plot(k, d, 
                                 marker=MARKERS.get(flag_config, 'o'),
                                 color=COLORS.get(k, 'gray'), 
                                 linestyle='none',
                                 markersize=6, zorder=2)
        
        # Style inset (matching benchmark_spatial_hadamard.py)
        ax_inset.set_xlabel('k', fontsize=22)
        ax_inset.set_ylabel('$d_{eff}$', fontsize=22)
        ax_inset.tick_params(axis='both', labelsize=22)
        ax_inset.set_xticks(k_values_present)
        # Set y-ticks to integers only
        from matplotlib.ticker import MaxNLocator
        ax_inset.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax_inset.grid(True, alpha=0.3)
    
    # Save figure
    fig.set_dpi(150)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'spatial_hadamard_{decoder}_interface_only.pdf')
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def main():
    # Load results
    csv_path = 'cluster_benchmark/combined_results.csv'
    results = load_results(csv_path)
    
    if not results:
        print(f"No results found in {csv_path}")
        return
    
    # Output directory
    output_dir = 'benchmark_plots'
    
    # Get unique decoders
    decoders = sorted(set(r['decoder'] for r in results))
    
    # Plot for each decoder
    for decoder in decoders:
        plot_decoder_results(decoder, results, output_dir)


if __name__ == '__main__':
    main()
