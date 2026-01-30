#!/usr/bin/env python3
"""
Unified plotting script for logical error rate benchmarks.

Consolidates plotting logic from benchmark_memory.py, benchmark_x_junction.py,
benchmark_patch_rotation.py, benchmark_spatial_hadamard.py into a single module.

Usage:
    python plot_logical_error_rates.py --experiment memory --csv benchmark_data/memory_error_rates_correlated_pymatching.csv
    python plot_logical_error_rates.py --experiment spatial_hadamard --csv ... --direction y --decoder tesseract
    python plot_logical_error_rates.py --experiment spatial_hadamard_interface --csv cluster_benchmark/combined_results.csv
    python plot_logical_error_rates.py --experiment spatial_hadamard_full_tesseract --csv cluster_benchmark/output/spatial_hadamard_full_tesseract/result_spatial_hadamard_full_tesseract_from_cluster.csv
    python plot_logical_error_rates.py --all
"""

import argparse
import csv
import os
from collections import defaultdict
from typing import Callable, Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import sinter
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, LogLocator, NullFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Use LaTeX-like fonts (STIX)
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

# =============================================================================
# Style Constants
# =============================================================================

# k determines color only
K_COLORS = {
    1: '#1f77b4',  # blue (C0)
    2: '#ff7f0e',  # orange (C1)
    3: '#2ca02c',  # green (C2)
    4: '#d62728',  # red (C3)
    5: '#9467bd',  # purple (C4)
}

# circuit_type determines both marker AND linestyle (memory, x_junction)
CIRCUIT_TYPE_STYLES = {
    'N/Z': {'marker': 'o', 'linestyle': '--'},
    'N/Z Circuit': {'marker': 'o', 'linestyle': '--'},
    'Diagonal': {'marker': 's', 'linestyle': '-'},
    'Diagonal Circuit': {'marker': 's', 'linestyle': '-'},
    'diagonal': {'marker': 's', 'linestyle': '-'},  # lowercase version
    'standard': {'marker': 'o', 'linestyle': '--'},  # x_junction uses 'standard' = N/Z
}

# flag_config determines both marker AND linestyle (spatial_hadamard)
FLAG_CONFIG_STYLES = {
    'none': {'marker': 'o', 'linestyle': ':'},
    'partial': {'marker': 's', 'linestyle': '--'},
    'all': {'marker': '^', 'linestyle': '-'},
}

# Plot styling constants
# Figures are 10 inches wide, scaled to ~3.4 inch column -> scale factor ~2.94
# To get 10pt text after scaling: 10 * (10/3.4) ≈ 29pt
FONT_SIZE = 25.5
LEGEND_FONT_SIZE = 23  # Slightly smaller than main text
X_LIM_LEFT = 7e-5
FIT_LINE_ALPHA = 0.25
INSET_SIZE = "30%"
INSET_OFFSET = 0.12

# Extra comparison markers (rectangular memory @ p=1e-3).
RECTANGULAR_COMPARISON_P = 1e-3
RECTANGULAR_COMPARISON_MARKER = 'X'
RECTANGULAR_COMPARISON_MARKER_SIZE = 90  # scatter "area" in points^2
RECTANGULAR_COMPARISON_ALPHA = 0.9
RECTANGULAR_COMPARISON_LEGEND_LABEL = r"$2\times 1\times 1$ memory"

# Legend placement constants (axes fraction coordinates).
# The style legend sits below the k legend using this anchor y-value.
DUAL_LEGEND_STYLE_ANCHOR_Y = 0.91
# Place the rectangular-memory legend below the style legend with the same
# vertical spacing as between the k legend (top) and the style legend.
RECTANGULAR_COMPARISON_LEGEND_X = 0.021
RECTANGULAR_COMPARISON_LEGEND_Y = 2 * DUAL_LEGEND_STYLE_ANCHOR_Y - 1.025


# =============================================================================
# Core Functions
# =============================================================================

def load_csv_to_sinter_stats(
    csv_path: str,
    metadata_columns: list[str],
    errors_column: str = 'errors',
    shots_column: str = 'shots',
) -> list[sinter.TaskStats]:
    """
    Load CSV file and convert to list of sinter.TaskStats.
    
    Args:
        csv_path: Path to CSV file
        metadata_columns: List of column names to include in json_metadata
        errors_column: Column name for error count (default: 'errors')
        shots_column: Column name for shot count (default: 'shots')
        
    Returns:
        List of sinter.TaskStats objects
    """
    stats_list = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            try:
                # Build metadata dict from specified columns
                metadata = {}
                for col in metadata_columns:
                    if col in row:
                        # Try to convert to appropriate type
                        val = row[col]
                        try:
                            if '.' in val:
                                metadata[col] = float(val)
                            else:
                                metadata[col] = int(val)
                        except ValueError:
                            metadata[col] = val
                
                # Handle different column names for errors
                errors_col = errors_column
                if errors_col not in row:
                    errors_col = 'logical_errors'  # fallback
                
                errors = int(row[errors_col])
                shots = int(row[shots_column])
                
                # Skip rows with no data
                if shots == 0:
                    continue
                
                # Create unique strong_id from metadata
                strong_id = "_".join(str(metadata.get(col, '')) for col in metadata_columns)
                
                stats = sinter.TaskStats(
                    strong_id=strong_id,
                    decoder=metadata.get('decoder', 'unknown'),
                    json_metadata=metadata,
                    shots=shots,
                    errors=errors,
                )
                stats_list.append(stats)
                
            except (ValueError, KeyError) as e:
                print(f"Warning: skipping row due to error: {e}")
    
    return stats_list


def load_cluster_benchmark_csv(
    csv_path: str,
    experiment_type: str,
) -> list[sinter.TaskStats]:
    """
    Load cluster benchmark CSV file and convert to list of sinter.TaskStats.
    
    Cluster benchmark CSV format:
    - experiment_type, schedule, k, physical_error_rate, decoder, logical_error_rate, errors, shots, error_bar, ...
    
    Args:
        csv_path: Path to cluster benchmark CSV file
        experiment_type: Expected experiment type ('memory', 'x_junction', 'patch_rotation')
        
    Returns:
        List of sinter.TaskStats objects with circuit_type mapped from schedule
    """
    stats_list = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            try:
                # Filter by experiment_type and decoder
                if row.get('experiment_type') != experiment_type:
                    continue
                if row.get('decoder') != 'correlated_pymatching':
                    continue
                
                # Map schedule to circuit_type
                schedule = row.get('schedule', '').strip()
                if schedule == 'diagonal':
                    circuit_type = 'Diagonal Circuit'
                else:
                    # Empty or other schedule -> N/Z or standard
                    if experiment_type == 'x_junction':
                        circuit_type = 'N/Z Circuit'  # x_junction uses 'standard' = N/Z
                    else:
                        circuit_type = 'N/Z Circuit'
                
                # Extract metadata
                k = int(row['k'])
                physical_error_rate = float(row['physical_error_rate'])
                errors = int(row['errors'])
                shots = int(row['shots'])
                
                # Skip rows with no data
                if shots == 0:
                    continue
                
                # Build metadata
                metadata = {
                    'k': k,
                    'circuit_type': circuit_type,
                    'physical_error_rate': physical_error_rate,
                    'p': physical_error_rate,  # Also include 'p' for compatibility
                    'decoder': 'correlated_pymatching',
                }
                
                # Handle patch_rotation differently (uses basis, not circuit_type)
                if experiment_type == 'patch_rotation':
                    if 'basis' in row and row['basis'].strip():
                        metadata['basis'] = row['basis'].strip()
                    # For patch_rotation, we don't use circuit_type, remove it
                    metadata.pop('circuit_type', None)
                    strong_id = f"patch_rotation_k{k}_basis{metadata.get('basis', 'unknown')}_p{physical_error_rate}"
                else:
                    # Create unique strong_id for memory and x_junction
                    strong_id = f"{circuit_type}_k{k}_p{physical_error_rate}"
                
                stats = sinter.TaskStats(
                    strong_id=strong_id,
                    decoder='correlated_pymatching',
                    json_metadata=metadata,
                    shots=shots,
                    errors=errors,
                )
                stats_list.append(stats)
                
            except (ValueError, KeyError) as e:
                print(f"Warning: skipping row due to error: {e}")
    
    return stats_list


def load_rectangular_memory_diagonal_points(
    decoder: str,
    p: float = RECTANGULAR_COMPARISON_P,
    csv_dir: str = 'benchmark_data',
) -> dict[int, float]:
    """
    Load rectangular-memory diagonal-schedule points for a given decoder at a fixed p.

    Expected filename: benchmark_data/rectangular_memory_error_rates_<decoder>.csv
    Expected columns: k, circuit_type, physical_error_rate, logical_error_rate, ...

    Returns:
        Dict mapping k -> logical_error_rate (only y>0 points).
    """
    csv_path = os.path.join(csv_dir, f"rectangular_memory_error_rates_{decoder}.csv")
    if not os.path.exists(csv_path):
        return {}

    out: dict[int, float] = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                circuit_type = (row.get('circuit_type') or '').strip().lower()
                if circuit_type not in ('diagonal circuit', 'diagonal'):
                    continue

                p_row = float(row['physical_error_rate'])
                if not np.isclose(p_row, p, rtol=0.0, atol=0.0):
                    continue

                k = int(row['k'])
                y = float(row['logical_error_rate'])
                if y <= 0:
                    continue

                out[k] = y
            except (KeyError, ValueError):
                continue
    return out


def overlay_rectangular_memory_diagonal_points(
    ax: plt.Axes,
    decoder: str,
    direction: str,
    p: float = RECTANGULAR_COMPARISON_P,
) -> bool:
    """
    Overlay rectangular-memory diagonal-schedule points on spatial_hadamard plots.

    Only adds points for the y-direction (as requested).
    """
    if direction != 'y':
        return False

    pts = load_rectangular_memory_diagonal_points(decoder=decoder, p=p)
    if not pts:
        return False

    y_values = []
    for k, y in sorted(pts.items()):
        color = K_COLORS.get(k, 'gray')
        ax.scatter(
            [p],
            [y],
            marker=RECTANGULAR_COMPARISON_MARKER,
            s=RECTANGULAR_COMPARISON_MARKER_SIZE,
            c=[color],
            edgecolors='black',
            alpha=RECTANGULAR_COMPARISON_ALPHA,
            linewidths=1.6,
            zorder=6,
        )
        y_values.append(y)

    # Ensure points aren't clipped by y-limits chosen from the main dataset.
    if y_values:
        cur_ymin, cur_ymax = ax.get_ylim()
        y_min2 = min(cur_ymin, min(y_values) * 0.8)
        y_max2 = max(cur_ymax, max(y_values) * 1.2)
        if y_min2 > 0 and y_max2 > 0:
            ax.set_ylim(y_min2, y_max2)

    return True


def _add_rectangular_memory_comparison_legend(ax: plt.Axes) -> None:
    """
    Add a third legend entry for the rectangular-memory comparison marker.

    This is intended to appear below the k legend and the style legend.
    """
    handle = Line2D(
        [0], [0],
        linestyle='None',
        marker=RECTANGULAR_COMPARISON_MARKER,
        markersize=9,
        markerfacecolor='gray',
        markeredgecolor='black',
        markeredgewidth=1.6,
        label=RECTANGULAR_COMPARISON_LEGEND_LABEL,
    )

    # Place below the style legend (which uses bbox_to_anchor y≈0.91).
    legend3 = Legend(
        ax,
        handles=[handle],
        labels=[RECTANGULAR_COMPARISON_LEGEND_LABEL],
        loc='upper left',
        bbox_to_anchor=(RECTANGULAR_COMPARISON_LEGEND_X, RECTANGULAR_COMPARISON_LEGEND_Y),
        fontsize=LEGEND_FONT_SIZE,
        frameon=True,
        handlelength=1.0,
        handletextpad=0.4,
        borderaxespad=0.0,
    )
    ax.add_artist(legend3)


def fit_and_plot_distance(
    ax: plt.Axes,
    stats_list: list[sinter.TaskStats],
    group_func: Callable,
    x_func: Callable,
    plot_args_func: Callable,
    min_points: int = 2,
    max_error_bar_ratio: Optional[float] = 0.1,
    x_lim_left: float = X_LIM_LEFT,
) -> tuple[dict[str, float], list[tuple]]:
    """
    Fit power law to data, add fit lines, and return data for d_eff inset.
    
    Fits: log(p_logical) = slope * log(p) + intercept
    where d_eff = 2 * slope - 1
    
    Args:
        ax: matplotlib axes object
        stats_list: List of sinter.TaskStats objects
        group_func: Function to group stats into curves
        x_func: Function to extract x value (physical error rate)
        plot_args_func: Function to get plot styling for each curve
        min_points: Number of lowest-p points to use for fitting
        max_error_bar_ratio: Filter points with error_bar > ratio * value (None to disable)
        x_lim_left: Left x-limit for fit line extension
        
    Returns:
        Tuple of (d_eff_results dict, fitted_distances list for inset)
    """
    # Group stats by curve
    curves = defaultdict(list)
    for s in stats_list:
        curve_id = group_func(s)
        curves[curve_id].append(s)
    
    # Collect fitted distances for inset plot
    fitted_distances = []  # list of (k, d_eff, color, marker, linestyle, curve_id)
    d_eff_results = {}
    
    for idx, (curve_id, stats) in enumerate(sorted(curves.items())):
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
                
                # Filter by error bar ratio if enabled
                if max_error_bar_ratio is None or (error_bar < max_error_bar_ratio * p_logical):
                    points.append((p, p_logical, error_bar))
        
        # Sort by physical error rate
        points.sort(key=lambda x: x[0])
        
        # Select fit points: use the lowest-p points.
        if len(points) < min_points:
            continue
        fit_points = points[:min_points]
        
        # Fit in log-log space
        log_p = np.array([np.log(pt[0]) for pt in fit_points])
        log_p_logical = np.array([np.log(pt[1]) for pt in fit_points])
        
        # Weighted fit: weight = (p_logical / error_bar)^2
        weights = np.array([(pt[1] / max(pt[2], 1e-300))**2 for pt in fit_points])
        
        slope, intercept = np.polyfit(log_p, log_p_logical, 1, w=weights)
        d_eff = 2 * slope - 1
        
        d_eff_results[curve_id] = d_eff
        
        # Get plot styling
        plot_args = plot_args_func(idx, curve_id)
        color = plot_args.get('color', 'black')
        marker = plot_args.get('marker', 'o')
        linestyle = plot_args.get('linestyle', '-')
        
        # Generate fit line from slightly above highest fit point to left edge
        p_max_fit = max(pt[0] for pt in fit_points) * 1.2  # slightly above highest fit point
        p_range = np.logspace(np.log10(x_lim_left), np.log10(p_max_fit), 100)
        p_logical_fit = np.exp(intercept) * p_range ** slope
        
        # Plot fit line as faint dotted line
        ax.plot(p_range, p_logical_fit, color=color, linestyle=':', 
                alpha=FIT_LINE_ALPHA, linewidth=1.5)
        
        # Store for inset plot
        if k_value is not None:
            fitted_distances.append((k_value, d_eff, color, marker, linestyle, curve_id))
    
    return d_eff_results, fitted_distances


def _create_deff_inset(ax: plt.Axes, fitted_distances: list[tuple]):
    """
    Create inset plot showing d_eff vs k.
    
    Args:
        ax: Parent axes
        fitted_distances: List of (k, d_eff, color, marker, linestyle, curve_id) tuples
    """
    inset_ax = inset_axes(
        ax, width=INSET_SIZE, height=INSET_SIZE, loc='lower right',
        bbox_to_anchor=(0, INSET_OFFSET, 1, 1), bbox_transform=ax.transAxes, borderpad=1.0
    )
    
    # Group by linestyle to draw black connecting lines
    linestyle_groups = defaultdict(list)
    for k, d, color, marker, linestyle, curve_id in fitted_distances:
        linestyle_groups[linestyle].append((k, d, color, marker))
    
    # Draw black connecting lines by linestyle
    for linestyle, points in linestyle_groups.items():
        points.sort(key=lambda x: x[0])  # Sort by k
        ks = [p[0] for p in points]
        ds = [p[1] for p in points]
        inset_ax.plot(ks, ds, color='black', linestyle=linestyle, linewidth=1.5, zorder=1)
    
    # Plot markers on top with their colors
    for k, d, color, marker, linestyle, curve_id in fitted_distances:
        inset_ax.plot(k, d, color=color, marker=marker, linestyle='none',
                     markersize=6, zorder=2)
    
    # Style the inset
    inset_ax.set_xlabel('k', fontsize=FONT_SIZE)
    inset_ax.set_ylabel('$d_{eff}$', fontsize=FONT_SIZE)
    inset_ax.tick_params(axis='both', labelsize=FONT_SIZE)
    
    # Set x/y limits by data extent
    k_values = [k for k, _, _, _, _, _ in fitted_distances]
    d_values = [d for _, d, _, _, _, _ in fitted_distances]
    
    k_min, k_max = min(k_values), max(k_values)
    d_min, d_max = min(d_values), max(d_values)
    
    # Add small padding (at least 0.3 to ensure non-integer limits)
    k_padding = 0.2
    d_padding = max((d_max - d_min) * 0.15, 0.3)
    
    # Set x-ticks to k values present
    inset_ax.set_xticks(sorted(set(k_values)))
    
    # Y-ticks only for odd d_eff values
    y_int_min = int(np.floor(d_min - d_padding))
    y_int_max = int(np.ceil(d_max + d_padding))
    odd_values = [y for y in range(y_int_min, y_int_max + 1) if y % 2 == 1]
    inset_ax.set_yticks(odd_values)
    
    inset_ax.grid(True, alpha=0.3)
    
    # Set limits LAST to ensure they aren't overridden
    inset_ax.set_xlim(k_min - k_padding, k_max + k_padding)
    inset_ax.set_ylim(d_min - d_padding, d_max + d_padding)


def _get_data_ylim(
    stats_list: list[sinter.TaskStats],
    buffer_low: float = 0.1,
    buffer_high: float = 0.1,
) -> tuple[float, float]:
    """Get y-axis limits based on data extent with proportional padding in log scale.
    
    Args:
        stats_list: List of sinter.TaskStats objects
        buffer_low: Fraction of log-range to add below min (default 0.1 = 10%)
        buffer_high: Fraction of log-range to add above max (default 0.1 = 10%)
    """
    y_values = []
    for s in stats_list:
        if s.errors > 0 and s.shots > 0:
            y_values.append(s.errors / s.shots)
    
    if not y_values:
        return (1e-8, 1e-1)
    
    y_min = min(y_values)
    y_max = max(y_values)
    
    # Buffer as a fraction of the log-scale data range
    log_range = np.log10(y_max) - np.log10(y_min)
    
    y_min = 10 ** (np.log10(y_min) - log_range * buffer_low)
    y_max = 10 ** (np.log10(y_max) + log_range * buffer_high)
    
    return (y_min, y_max)


def _create_dual_legend(
    ax: plt.Axes,
    k_values: list[int],
    style_items: list[tuple[str, dict]],  # list of (label, style_dict)
    style_label: str = None,
):
    """
    Create one or two legends: k values on top (always), styles below (if multiple).
    
    Args:
        ax: matplotlib axes
        k_values: List of k values to show
        style_items: List of (label, {'marker': ..., 'linestyle': ...}) tuples
        style_label: Optional label for the style legend
    """
    import matplotlib.patches as mpatches
    
    # Remove any existing legend
    if ax.legend_ is not None:
        ax.legend_.remove()
    
    # Create k legend handles using rectangular color patches
    k_handles = [
        mpatches.Patch(color=K_COLORS.get(k, 'gray'), label=f'k={k}')
        for k in sorted(k_values)
    ]
    
    # Add k legend (top)
    legend1 = ax.legend(
        handles=k_handles, loc='upper left',
        ncol=len(k_values), fontsize=LEGEND_FONT_SIZE,
        columnspacing=0.5, handletextpad=0.3, handlelength=1.0
    )
    ax.add_artist(legend1)
    
    # Add style legend (below k legend) only if there are multiple styles
    if len(style_items) > 1:
        style_handles = [
            Line2D([0], [0], color='gray',
                   linestyle=style['linestyle'],
                   marker=style['marker'],
                   markersize=6, label=label)
            for label, style in style_items
        ]
        ax.legend(
            handles=style_handles, loc='upper left',
            ncol=len(style_items), fontsize=LEGEND_FONT_SIZE,
            bbox_to_anchor=(0.0, DUAL_LEGEND_STYLE_ANCHOR_Y), handlelength=2.0,
            columnspacing=0.5, handletextpad=0.3
        )


def plot_error_rate_panel(
    ax: plt.Axes,
    stats_list: list[sinter.TaskStats],
    group_func: Callable,
    x_func: Callable,
    plot_args_func: Callable,
    k_values: list[int],
    style_items: list[tuple[str, dict]],
    add_inset: bool = True,
    min_fit_points: int = 2,
    max_error_bar_ratio: Optional[float] = 0.1,
    ylim_buffer_low: float = 0.1,
    ylim_buffer_high: float = 0.1,
) -> dict[str, float]:
    """
    Plot logical error rates on a single panel with optional d_eff inset.
    
    Args:
        ax: matplotlib axes object
        stats_list: List of sinter.TaskStats objects
        group_func: Function to group stats into curves
        x_func: Function to extract x value (physical error rate)
        plot_args_func: Function to get plot styling for each curve
        k_values: List of k values for legend
        style_items: List of (label, style_dict) for legend
        add_inset: Whether to add d_eff inset
        min_fit_points: Number of points for fitting
        max_error_bar_ratio: Error bar filter threshold
        ylim_buffer_low: Fraction of log-range to add below min (default 0.1)
        ylim_buffer_high: Fraction of log-range to add above max (default 0.1)
        
    Returns:
        Dict mapping curve_id to fitted d_eff value
    """
    # Use sinter.plot_error_rate for the main plot
    sinter.plot_error_rate(
        ax=ax,
        stats=stats_list,
        x_func=x_func,
        group_func=group_func,
        plot_args_func=plot_args_func,
    )
    
    # Add fit lines and get inset data
    d_eff_results = {}
    fitted_distances = []
    if add_inset:
        d_eff_results, fitted_distances = fit_and_plot_distance(
            ax, stats_list, group_func, x_func, plot_args_func,
            min_points=min_fit_points,
            max_error_bar_ratio=max_error_bar_ratio,
        )
    
    # Style the axes
    ax.loglog()
    ax.set_xlim(left=X_LIM_LEFT)
    
    # Set y-limits by data extent (not extending for fit lines)
    y_min, y_max = _get_data_ylim(stats_list, ylim_buffer_low, ylim_buffer_high)
    ax.set_ylim(y_min, y_max)
    
    ax.set_xlabel("Physical Error Rate", fontsize=FONT_SIZE)
    ax.set_ylabel("Logical Error Rate", fontsize=FONT_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    # Ensure minor ticks are visible on all plots
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100))
    # Explicitly hide minor tick labels with NullFormatter
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(which='major', alpha=0.5)
    ax.grid(which='minor', alpha=0.2)
    
    # Create dual legend (k on top, styles below)
    _create_dual_legend(ax, k_values, style_items)
    
    # Create inset if we have fitted distances
    if fitted_distances:
        _create_deff_inset(ax, fitted_distances)
    
    return d_eff_results


# =============================================================================
# Experiment-Specific Plot Functions
# =============================================================================

def plot_memory(
    csv_path: str,
    output_dir: str = 'benchmark_plots',
    output_filename: str = None,
):
    """
    Plot memory experiment logical error rates.
    
    Compares N/Z (original) vs Diagonal circuit types.
    If csv_path points to cluster benchmark data, also loads pymatching data from benchmark_data/.
    """
    if output_filename is None:
        output_filename = 'memory_error_rates.pdf'
    
    print(f"Loading data from {csv_path}")
    
    # Check if this is a cluster benchmark CSV - if so, create separate plots for each decoder
    if 'cluster_benchmark' in csv_path or 'result_memory_from_cluster' in csv_path:
        # Load correlated_pymatching from cluster benchmark
        correlated_stats = load_cluster_benchmark_csv(csv_path, experiment_type='memory')
        
        # Load pymatching from benchmark_data
        pymatching_csv = 'benchmark_data/memory_error_rates_backup.csv'
        pymatching_stats = []
        if os.path.exists(pymatching_csv):
            print(f"Also loading pymatching data from {pymatching_csv}")
            pymatching_stats = load_csv_to_sinter_stats(
                pymatching_csv,
                metadata_columns=['k', 'circuit_type', 'physical_error_rate'],
                errors_column='logical_errors',
            )
        else:
            print(f"Warning: {pymatching_csv} not found")
        
        # Plot correlated_pymatching
        if correlated_stats:
            _plot_memory_single_decoder(
                correlated_stats, 'correlated_pymatching', output_dir,
                'memory_error_rates_correlated_pymatching.pdf'
            )
        
        # Plot pymatching
        if pymatching_stats:
            _plot_memory_single_decoder(
                pymatching_stats, 'pymatching', output_dir,
                'memory_error_rates_pymatching.pdf'
            )
        
        return
    else:
        # Regular CSV - just plot what's in the file
        stats_list = load_csv_to_sinter_stats(
            csv_path,
            metadata_columns=['k', 'circuit_type', 'physical_error_rate'],
            errors_column='logical_errors',
        )
        
        if not stats_list:
            print("No data to plot")
            return
        
        # Determine decoder from filename or default
        decoder = 'correlated_pymatching' if 'correlated_pymatching' in csv_path else 'pymatching'
        if output_filename is None:
            output_filename = f'memory_error_rates_{decoder}.pdf'
        
        _plot_memory_single_decoder(stats_list, decoder, output_dir, output_filename)


def _plot_memory_single_decoder(
    stats_list: list[sinter.TaskStats],
    decoder: str,
    output_dir: str,
    output_filename: str,
):
    """Helper function to plot memory data for a single decoder."""
    if not stats_list:
        return
    
    # Rename metadata key for consistency
    for s in stats_list:
        s.json_metadata['p'] = s.json_metadata.pop('physical_error_rate', 0)
    
    def group_func(s):
        circuit = s.json_metadata['circuit_type'].replace(' Circuit', '')
        return f"{circuit} k={s.json_metadata['k']}"
    
    def x_func(s):
        return s.json_metadata['p']
    
    def plot_args_func(index, curve_id):
        parts = curve_id.split()
        circuit_type = parts[0]
        k = int(parts[1].split('=')[1])
        
        style = CIRCUIT_TYPE_STYLES.get(circuit_type, CIRCUIT_TYPE_STYLES.get('N/Z'))
        return {
            'color': K_COLORS.get(k, 'gray'),
            'marker': style['marker'],
            'linestyle': style['linestyle'],
        }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get unique values for legend
    k_values = sorted(set(s.json_metadata['k'] for s in stats_list))
    circuit_types = sorted(set(s.json_metadata['circuit_type'].replace(' Circuit', '') 
                               for s in stats_list))
    
    # Build style items for legend
    style_items = [
        (ct, CIRCUIT_TYPE_STYLES.get(ct, CIRCUIT_TYPE_STYLES['N/Z']))
        for ct in circuit_types
    ]
    
    plot_error_rate_panel(
        ax, stats_list, group_func, x_func, plot_args_func,
        k_values=k_values,
        style_items=style_items,
        min_fit_points=4,
        max_error_bar_ratio=None,
    )
    
    # Memory plot spans many decades - ensure major ticks at every decade
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    fig.set_dpi(150)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def plot_x_junction(
    csv_path: str,
    output_dir: str = 'benchmark_plots',
    output_filename: str = None,
):
    """
    Plot X-junction experiment logical error rates.
    
    Compares N/Z vs Diagonal circuit types.
    """
    if output_filename is None:
        # Auto-detect decoder type from CSV filename
        if 'correlated_pymatching' in csv_path:
            output_filename = 'x_junction_error_rates_correlated_pymatching.pdf'
        else:
            output_filename = 'x_junction_error_rates.pdf'
    
    print(f"Loading data from {csv_path}")
    
    # Check if this is a cluster benchmark CSV - if so, create separate plots for each decoder
    if 'cluster_benchmark' in csv_path or 'result_x_junction_from_cluster' in csv_path:
        # Load correlated_pymatching from cluster benchmark
        correlated_stats = load_cluster_benchmark_csv(csv_path, experiment_type='x_junction')
        
        # Load pymatching from benchmark_data
        pymatching_csv = 'benchmark_data/x_junction_error_rates_backup.csv'
        pymatching_stats = []
        if os.path.exists(pymatching_csv):
            print(f"Also loading pymatching data from {pymatching_csv}")
            pymatching_stats = load_csv_to_sinter_stats(
                pymatching_csv,
                metadata_columns=['k', 'circuit_type', 'physical_error_rate'],
                errors_column='logical_errors',
            )
        else:
            print(f"Warning: {pymatching_csv} not found")
        
        # Plot correlated_pymatching
        if correlated_stats:
            _plot_x_junction_single_decoder(
                correlated_stats, 'correlated_pymatching', output_dir,
                'x_junction_error_rates_correlated_pymatching.pdf'
            )
        
        # Plot pymatching
        if pymatching_stats:
            _plot_x_junction_single_decoder(
                pymatching_stats, 'pymatching', output_dir,
                'x_junction_error_rates_pymatching.pdf'
            )
        
        return
    else:
        # Regular CSV - just plot what's in the file
        stats_list = load_csv_to_sinter_stats(
            csv_path,
            metadata_columns=['k', 'circuit_type', 'physical_error_rate'],
            errors_column='logical_errors',
        )
        
        if not stats_list:
            print("No data to plot")
            return
        
        # Determine decoder from filename or default
        decoder = 'correlated_pymatching' if 'correlated_pymatching' in csv_path else 'pymatching'
        if output_filename is None:
            output_filename = f'x_junction_error_rates_{decoder}.pdf'
        
        _plot_x_junction_single_decoder(stats_list, decoder, output_dir, output_filename)


def _plot_x_junction_single_decoder(
    stats_list: list[sinter.TaskStats],
    decoder: str,
    output_dir: str,
    output_filename: str,
):
    """Helper function to plot x_junction data for a single decoder."""
    if not stats_list:
        return
    
    # Rename metadata key for consistency and normalize circuit_type names
    for s in stats_list:
        s.json_metadata['p'] = s.json_metadata.pop('physical_error_rate', 0)
        # Map to display names
        ct = s.json_metadata['circuit_type']
        if ct == 'standard':
            s.json_metadata['circuit_type_display'] = 'N/Z'
        elif ct == 'diagonal':
            s.json_metadata['circuit_type_display'] = 'Diagonal'
        else:
            s.json_metadata['circuit_type_display'] = ct.replace(' Circuit', '')
    
    def group_func(s):
        circuit = s.json_metadata['circuit_type_display']
        return f"{circuit} k={s.json_metadata['k']}"
    
    def x_func(s):
        return s.json_metadata['p']
    
    def plot_args_func(index, curve_id):
        # Parse "N/Z k=1" or "Diagonal k=2"
        parts = curve_id.split()
        circuit_type_display = parts[0]
        k = int(parts[1].split('=')[1])
        
        style = CIRCUIT_TYPE_STYLES.get(circuit_type_display, CIRCUIT_TYPE_STYLES.get('N/Z'))
        return {
            'color': K_COLORS.get(k, 'gray'),
            'marker': style['marker'],
            'linestyle': style['linestyle'],
        }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get unique values for legend
    k_values = sorted(set(s.json_metadata['k'] for s in stats_list))
    circuit_types = sorted(set(s.json_metadata['circuit_type_display'] for s in stats_list))
    
    # Build style items for legend
    style_items = [
        (ct, CIRCUIT_TYPE_STYLES.get(ct, CIRCUIT_TYPE_STYLES['N/Z']))
        for ct in circuit_types
    ]
    
    plot_error_rate_panel(
        ax, stats_list, group_func, x_func, plot_args_func,
        k_values=k_values,
        style_items=style_items,
        min_fit_points=2,
        max_error_bar_ratio=0.1,
    )
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    fig.set_dpi(150)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def plot_patch_rotation(
    csv_path: str,
    output_dir: str = 'benchmark_plots',
    basis: str = None,
):
    """
    Plot patch rotation experiment logical error rates.
    
    Creates one plot per basis (z, x) or both if basis is None.
    """
    print(f"Loading data from {csv_path}")
    
    # Check if this is a cluster benchmark CSV - if so, create separate plots for each decoder
    if 'cluster_benchmark' in csv_path or 'result_patch_rotation_from_cluster' in csv_path:
        # Load correlated_pymatching from cluster benchmark
        correlated_stats = load_cluster_benchmark_csv(csv_path, experiment_type='patch_rotation')
        
        # Load pymatching from benchmark_data
        pymatching_csv = 'benchmark_data/patch_rotation_benchmark_backup.csv'
        pymatching_stats = []
        if os.path.exists(pymatching_csv):
            print(f"Also loading pymatching data from {pymatching_csv}")
            pymatching_stats = load_csv_to_sinter_stats(
                pymatching_csv,
                metadata_columns=['k', 'basis', 'physical_error_rate'],
                errors_column='errors',
            )
        else:
            print(f"Warning: {pymatching_csv} not found")
        
        # Determine which bases to plot
        all_stats = correlated_stats + pymatching_stats
        available_bases = sorted(set(s.json_metadata.get('basis') for s in all_stats if s.json_metadata.get('basis')))
        if basis is not None:
            bases_to_plot = [basis] if basis in available_bases else []
        else:
            bases_to_plot = available_bases
        
        # Plot correlated_pymatching
        if correlated_stats:
            _plot_patch_rotation_single_decoder(
                correlated_stats, 'correlated_pymatching', output_dir, bases_to_plot
            )
        
        # Plot pymatching
        if pymatching_stats:
            _plot_patch_rotation_single_decoder(
                pymatching_stats, 'pymatching', output_dir, bases_to_plot
            )
        return
    else:
        # Regular CSV - just plot what's in the file
        stats_list = load_csv_to_sinter_stats(
            csv_path,
            metadata_columns=['k', 'basis', 'physical_error_rate'],
            errors_column='errors',
        )
        
        if not stats_list:
            print("No data to plot")
            return
        
        # Filter by basis if specified
        if basis:
            stats_list = [s for s in stats_list if s.json_metadata.get('basis') == basis]
        
        # Determine decoder from filename or default
        decoder = 'correlated_pymatching' if 'correlated_pymatching' in csv_path else 'pymatching'
        available_bases = sorted(set(s.json_metadata.get('basis') for s in stats_list if s.json_metadata.get('basis')))
        
        _plot_patch_rotation_single_decoder(stats_list, decoder, output_dir, available_bases)


def _plot_patch_rotation_single_decoder(
    stats_list: list[sinter.TaskStats],
    decoder: str,
    output_dir: str,
    bases_to_plot: list[str],
):
    """Helper function to plot patch_rotation data for a single decoder."""
    if not stats_list:
        return
    
    # Rename metadata key for consistency
    for s in stats_list:
        s.json_metadata['p'] = s.json_metadata.pop('physical_error_rate', 0)
    
    decoder_suffix = f'_{decoder}'
    
    for b in bases_to_plot:
        basis_stats = [s for s in stats_list if s.json_metadata.get('basis') == b]
        
        if not basis_stats:
            continue
        
        def group_func(s):
            return f"k={s.json_metadata['k']}"
        
        def x_func(s):
            return s.json_metadata['p']
        
        def plot_args_func(index, curve_id):
            k = int(curve_id.split('=')[1])
            return {
                'color': K_COLORS.get(k, 'gray'),
                'marker': 'o',
                'linestyle': '-',
            }
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get unique k values for legend
        k_values = sorted(set(s.json_metadata['k'] for s in basis_stats))
        
        # For patch rotation, only one style (no comparison)
        style_items = [(f'{b}-basis', {'marker': 'o', 'linestyle': '-'})]
        
        plot_error_rate_panel(
            ax, basis_stats, group_func, x_func, plot_args_func,
            k_values=k_values,
            style_items=style_items,
            min_fit_points=2,
            max_error_bar_ratio=0.1,
        )
        
        # Save
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'patch_rotation_{b}{decoder_suffix}.pdf')
        fig.set_dpi(150)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved plot to {output_path}")


def plot_spatial_hadamard(
    csv_path: str,
    output_dir: str = 'benchmark_plots',
    direction: str = None,
    decoder: str = None,
    filename_suffix: str = '',
):
    """
    Plot spatial Hadamard experiment logical error rates.
    
    Creates one plot per direction/decoder combination.
    """
    print(f"Loading data from {csv_path}")
    
    stats_list = load_csv_to_sinter_stats(
        csv_path,
        metadata_columns=['k', 'flag_config', 'direction', 'decoder', 'physical_error_rate'],
        errors_column='errors',
    )
    
    if not stats_list:
        print("No data to plot")
        return
    
    # Rename metadata key for consistency
    for s in stats_list:
        s.json_metadata['p'] = s.json_metadata.pop('physical_error_rate', 0)
    
    # Determine which directions and decoders to plot
    available_directions = sorted(set(s.json_metadata['direction'] for s in stats_list))
    available_decoders = sorted(set(s.json_metadata['decoder'] for s in stats_list))
    
    directions_to_plot = [direction] if direction else available_directions
    decoders_to_plot = [decoder] if decoder else available_decoders
    
    # Define sort order for flag_config in legend
    flag_order = {'none': 0, 'partial': 1, 'all': 2}
    
    for dec in decoders_to_plot:
        for dir_ in directions_to_plot:
            # The non-full spatial Hadamard Tesseract plots have been replaced by the
            # dedicated full-noise tesseract plot. Skip generating the legacy outputs:
            #   spatial_hadamard_tesseract_x.pdf
            #   spatial_hadamard_tesseract_y.pdf
            if dec == 'tesseract' and filename_suffix == '':
                continue

            filtered_stats = [
                s for s in stats_list
                if s.json_metadata['direction'] == dir_ and s.json_metadata['decoder'] == dec
            ]

            # Full-noise tesseract: use only k=1, 2, 3
            if dec == 'tesseract' and filename_suffix == '_full':
                filtered_stats = [s for s in filtered_stats if s.json_metadata.get('k') in (1, 2, 3)]

            if not filtered_stats:
                continue
            
            def group_func(s):
                fc = s.json_metadata['flag_config']
                return f"{flag_order[fc]}_{fc} k={s.json_metadata['k']}"
            
            def x_func(s):
                return s.json_metadata['p']
            
            def plot_args_func(index, curve_id):
                parts = curve_id.split()
                flag_config = parts[0].split('_')[1] if '_' in parts[0] else parts[0]
                k = int(parts[1].split('=')[1])
                
                style = FLAG_CONFIG_STYLES.get(flag_config, FLAG_CONFIG_STYLES['none'])
                return {
                    'color': K_COLORS.get(k, 'gray'),
                    'marker': style['marker'],
                    'linestyle': style['linestyle'],
                }
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Get unique values for legend
            k_values = sorted(set(s.json_metadata['k'] for s in filtered_stats))
            flag_configs = sorted(
                set(s.json_metadata['flag_config'] for s in filtered_stats),
                key=lambda x: flag_order.get(x, 99)
            )
            
            # Build style items for legend
            style_items = [
                (fc, FLAG_CONFIG_STYLES[fc])
                for fc in flag_configs
            ]
            
            # Custom y-limit buffers for specific plots
            ylim_buffer_low = 0.1
            ylim_buffer_high = 0.1
            if dec == 'tesseract' and dir_ == 'y' and filename_suffix == '_interface':
                ylim_buffer_low = 0.22  # larger buffer at low end
                ylim_buffer_high = 0.05  # smaller buffer at high end
            
            plot_error_rate_panel(
                ax, filtered_stats, group_func, x_func, plot_args_func,
                k_values=k_values,
                style_items=style_items,
                min_fit_points=2,
                max_error_bar_ratio=0.1,
                ylim_buffer_low=ylim_buffer_low,
                ylim_buffer_high=ylim_buffer_high,
            )

            # Add rectangular-memory comparison markers (diagonal schedule only).
            # Note: these CSVs only contain points at p=1e-3 on purpose.
            # Do NOT add these markers to the interface-noise plots.
            if filename_suffix != '_interface':
                did_overlay = overlay_rectangular_memory_diagonal_points(ax, decoder=dec, direction=dir_)
                if did_overlay:
                    _add_rectangular_memory_comparison_legend(ax)
            
            # Save
            os.makedirs(output_dir, exist_ok=True)
            # Rename: spatial_hadamard_tesseract_y_full.pdf -> spatial_hadamard_tesseract_y.pdf
            save_suffix = '' if (dec == 'tesseract' and filename_suffix == '_full') else filename_suffix
            output_path = os.path.join(output_dir, f'spatial_hadamard_{dec}_{dir_}{save_suffix}.pdf')
            fig.set_dpi(150)
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved plot to {output_path}")


def plot_spatial_hadamard_interface(
    csv_path: str,
    output_dir: str = 'benchmark_plots',
    direction: str = None,
    decoder: str = None,
):
    """
    Plot spatial Hadamard with interface-only noise (from cluster_benchmark).
    
    Same format as plot_spatial_hadamard but with '_interface' suffix in filename.
    """
    plot_spatial_hadamard(
        csv_path=csv_path,
        output_dir=output_dir,
        direction=direction,
        decoder=decoder,
        filename_suffix='_interface',
    )


def plot_spatial_hadamard_full_tesseract(
    csv_path: str,
    output_dir: str = 'benchmark_plots',
    direction: str = 'y',
):
    """
    Plot spatial Hadamard with full noise, decoded using tesseract (from cluster_benchmark).

    Saved as spatial_hadamard_tesseract_<direction>.pdf (without a '_full' suffix).
    """
    plot_spatial_hadamard(
        csv_path=csv_path,
        output_dir=output_dir,
        direction=direction,
        decoder='tesseract',
        filename_suffix='_full',
    )


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Plot logical error rates from benchmark CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --experiment memory --csv benchmark_data/memory_error_rates_correlated_pymatching.csv
  %(prog)s --experiment patch_rotation --csv benchmark_data/patch_rotation_benchmark_correlated_pymatching.csv --basis z
  %(prog)s --experiment spatial_hadamard --csv benchmark_data/spatial_hadamard_benchmark_backup.csv --direction y --decoder correlated_pymatching
  %(prog)s --experiment spatial_hadamard_interface --csv cluster_benchmark/combined_results.csv
  %(prog)s --experiment spatial_hadamard_full_tesseract --csv cluster_benchmark/output/spatial_hadamard_full_tesseract/result_spatial_hadamard_full_tesseract_from_cluster.csv
  %(prog)s --all
        """
    )
    
    parser.add_argument(
        '--experiment', '-e',
        choices=[
            'memory',
            'x_junction',
            'patch_rotation',
            'spatial_hadamard',
            'spatial_hadamard_interface',
            'spatial_hadamard_full_tesseract',
        ],
        help='Experiment type to plot'
    )
    parser.add_argument(
        '--csv', '-c',
        help='Path to CSV file with benchmark data'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='benchmark_plots',
        help='Output directory for plots (default: benchmark_plots)'
    )
    parser.add_argument(
        '--basis', '-b',
        choices=['x', 'z'],
        help='Basis for patch_rotation (default: plot both)'
    )
    parser.add_argument(
        '--direction', '-d',
        choices=['x', 'y'],
        help='Direction for spatial_hadamard (default: plot all)'
    )
    parser.add_argument(
        '--decoder',
        help='Decoder for spatial_hadamard (default: plot all)'
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Plot all experiments using default CSV paths'
    )
    
    args = parser.parse_args()
    
    if args.all:
        # Default CSV paths (using cluster benchmark data for correlated PyMatching)
        default_csvs = {
            'memory': 'cluster_benchmark/output/memory/result_memory_from_cluster.csv',
            'x_junction': 'cluster_benchmark/output/x_junction/result_x_junction_from_cluster.csv',
            'patch_rotation': 'cluster_benchmark/output/patch_rotation/result_patch_rotation_from_cluster.csv',
            'spatial_hadamard': 'benchmark_data/spatial_hadamard_benchmark_backup.csv',
            'spatial_hadamard_interface': 'cluster_benchmark/combined_results.csv',
            'spatial_hadamard_full_tesseract': 'cluster_benchmark/output/spatial_hadamard_full_tesseract/result_spatial_hadamard_full_tesseract_from_cluster.csv',
        }
        
        for exp, csv_path in default_csvs.items():
            if os.path.exists(csv_path):
                print(f"\n{'='*60}")
                print(f"Plotting {exp}")
                print('='*60)
                
                if exp == 'memory':
                    plot_memory(csv_path, args.output_dir)
                elif exp == 'x_junction':
                    plot_x_junction(csv_path, args.output_dir)
                elif exp == 'patch_rotation':
                    plot_patch_rotation(csv_path, args.output_dir)
                elif exp == 'spatial_hadamard':
                    plot_spatial_hadamard(csv_path, args.output_dir)
                elif exp == 'spatial_hadamard_interface':
                    plot_spatial_hadamard_interface(csv_path, args.output_dir)
                elif exp == 'spatial_hadamard_full_tesseract':
                    plot_spatial_hadamard_full_tesseract(csv_path, args.output_dir)
            else:
                print(f"Skipping {exp}: {csv_path} not found")
    
    elif args.experiment:
        if not args.csv:
            parser.error(f"--csv is required for --experiment {args.experiment}")
        
        if args.experiment == 'memory':
            plot_memory(args.csv, args.output_dir)
        elif args.experiment == 'x_junction':
            plot_x_junction(args.csv, args.output_dir)
        elif args.experiment == 'patch_rotation':
            plot_patch_rotation(args.csv, args.output_dir, basis=args.basis)
        elif args.experiment == 'spatial_hadamard':
            plot_spatial_hadamard(args.csv, args.output_dir, 
                                  direction=args.direction, decoder=args.decoder)
        elif args.experiment == 'spatial_hadamard_interface':
            plot_spatial_hadamard_interface(args.csv, args.output_dir,
                                            direction=args.direction, decoder=args.decoder)
        elif args.experiment == 'spatial_hadamard_full_tesseract':
            plot_spatial_hadamard_full_tesseract(args.csv, args.output_dir,
                                                 direction=args.direction or 'y')
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
