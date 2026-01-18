#!/usr/bin/env python3
"""
Plot logical error rate vs physical error rate for tesseract decoder on spatial hadamard.
"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import sinter
from collections import defaultdict
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MaxNLocator

def load_results(filepath):
    """Load benchmark results from CSV file."""
    results = []
    with open(filepath, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            results.append({
                'direction': row['direction'],
                'flag_config': row['flag_config'],
                'k': int(row['k']),
                'distance': int(row['distance']) if row['distance'] else None,
                'physical_error_rate': float(row['physical_error_rate']),
                'decoder': row['decoder'],
                'logical_error_rate': float(row['logical_error_rate']),
                'errors': int(row['errors']),
                'shots': int(row['shots']),
                'error_bar': float(row['error_bar']),
                'decode_time': float(row['decode_time']),
            })
    return results

def results_to_sinter_stats(results):
    """Convert results to sinter.TaskStats for plotting."""
    stats_list = []
    for result in results:
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

def fit_and_plot_distance(ax, stats_list, group_func, x_func, plot_args_func, min_points=4):
    """Fit p_logical = A * p^((d+1)/2) to data, add fit lines, and create inset d vs k plot."""
    
    # Group stats by curve
    curves = defaultdict(list)
    for s in stats_list:
        curve_id = group_func(s)
        curves[curve_id].append(s)
    
    # Collect fitted distances for inset plot
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
                error_bar = np.sqrt(p_logical * (1 - p_logical) / s.shots)
                # Only include points with reasonable error bars (< 10% of value)
                if error_bar < 0.1 * p_logical and p_logical > 0:
                    points.append((p, p_logical, error_bar))
        
        # Sort by physical error rate (ascending) and take the lowest
        points.sort(key=lambda x: x[0])
        
        # Use available points if we have at least 2
        actual_min_points = min(min_points, len(points))
        if actual_min_points < 2:
            print(f"Skipping {curve_id}: only {len(points)} valid points")
            continue
        
        # Use the lowest error rate points available (up to min_points)
        fit_points = points[:actual_min_points]
        if actual_min_points < min_points:
            print(f"Using {actual_min_points} points for {curve_id} (requested {min_points})")
        
        # Fit in log-log space with weighted least squares
        log_p = np.array([np.log(pt[0]) for pt in fit_points])
        log_p_logical = np.array([np.log(pt[1]) for pt in fit_points])
        weights = np.array([(pt[1] / pt[2])**2 for pt in fit_points])
        
        # Linear fit in log-log space
        slope, intercept = np.polyfit(log_p, log_p_logical, 1, w=weights)
        
        # Calculate effective distance: slope = (d+1)/2 => d = 2*slope - 1
        d_eff = 2 * slope - 1
        
        # Get plot styling
        plot_args = plot_args_func(idx, curve_id)
        color = plot_args.get('color', 'black')
        marker = plot_args.get('marker', 'o')
        linestyle = plot_args.get('linestyle', '-')
        
        # Generate fit line across appropriate x range for this data
        p_range = np.logspace(np.log10(4e-4), np.log10(1.5e-2), 100)
        p_logical_fit = np.exp(intercept) * p_range ** slope
        
        # Plot fit line as very faint dotted line
        ax.plot(p_range, p_logical_fit, color=color, linestyle=':', alpha=0.25, linewidth=1.5)
        
        # Store for inset plot
        if k_value is not None:
            fitted_distances.append((k_value, d_eff, color, marker, linestyle, curve_id))
            print(f"{curve_id}: d_eff = {d_eff:.2f}")
    
    # Create inset plot for d vs k
    if fitted_distances:
        inset_ax = inset_axes(ax, width="30%", height="30%", loc='lower right', 
                              bbox_to_anchor=(0, 0.08, 1, 1), bbox_transform=ax.transAxes, borderpad=2)
        
        # Group by linestyle to draw black connecting lines
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
        inset_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        inset_ax.grid(True, alpha=0.3)

def plot_tesseract_results(csv_path, save_path):
    """Plot tesseract decoder results."""
    
    results = load_results(csv_path)
    print(f"Loaded {len(results)} results")
    
    stats_list = results_to_sinter_stats(results)
    
    # Define colors by k value
    k_colors = {1: 'C0', 2: 'C1', 3: 'C2', 4: 'C3'}
    
    # Define markers by flag_config
    flag_markers = {'all': 'o', 'partial': 's', 'none': '^'}
    
    # Define linestyles by flag_config
    flag_styles = {'all': '-', 'partial': '--', 'none': ':'}
    
    # Define order for legend
    flag_order = {'none': '0', 'partial': '1', 'all': '2'}
    
    def plot_args_func(index, curve_id):
        parts = curve_id.split()
        flag_config = parts[0].split('_')[1] if '_' in parts[0] else parts[0]
        k = int(parts[1].split('=')[1])
        return {
            'color': k_colors.get(k, 'black'),
            'marker': flag_markers.get(flag_config, 'o'),
            'linestyle': flag_styles.get(flag_config, '-'),
        }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    group_func = lambda s: f"{flag_order[s.json_metadata['flag_config']]}_{s.json_metadata['flag_config']} k={s.json_metadata['k']}"
    x_func = lambda s: s.json_metadata['p']
    
    sinter.plot_error_rate(
        ax=ax,
        stats=stats_list,
        x_func=x_func,
        group_func=group_func,
        plot_args_func=plot_args_func,
    )
    
    # Add fit lines with distance labels (using 4 points for fitting)
    fit_and_plot_distance(ax, stats_list, group_func, x_func, plot_args_func, min_points=4)
    
    # Fix legend labels
    handles, labels = ax.get_legend_handles_labels()
    new_labels = [label.split('_', 1)[1] if '_' in label else label for label in labels]
    ax.legend(handles, new_labels, fontsize=18, ncol=3, loc='upper left', columnspacing=0.5, handletextpad=0.3)
    
    ax.loglog()
    # Adjust x-limits based on data: from ~4e-4 to ~1.5e-2
    ax.set_xlim(4e-4, 1.5e-2)
    ax.set_xlabel("Physical Error Rate", fontsize=22)
    ax.set_ylabel("Logical Error Rate", fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.tick_params(axis='both', which='minor', labelsize=22)
    ax.grid(which='major', alpha=0.5)
    ax.grid(which='minor', alpha=0.2)
    ax.set_title("Spatial Hadamard - Tesseract Decoder", fontsize=24)
    
    fig.set_dpi(150)
    # Skip tight_layout due to inset axes compatibility issues
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    
    plt.close()

if __name__ == "__main__":
    csv_path = "benchmark_data/spatial_hadamard_benchmark_y_only_tesseract_only.csv"
    save_path = "benchmark_plots/spatial_hadamard_tesseract.pdf"
    plot_tesseract_results(csv_path, save_path)
