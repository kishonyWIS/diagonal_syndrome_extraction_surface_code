#!/usr/bin/env python3
"""
Plot benchmark results from spatial Hadamard circuit benchmarks.

Reads CSV data and generates two sets of plots:
1. By decoder: One figure per decoder, with different line styles for flag configs
2. By flag config: One figure per flag config, with different line styles for decoders

Each figure has side-by-side subplots for x and y directions.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# Configuration
# =============================================================================

CSV_FILE = 'spatial_hadamard_benchmark.csv'
OUTPUT_DIR = 'benchmark_plots'

# Color mapping for k values
K_COLORS = {
    1: '#1f77b4',  # blue
    2: '#d62728',  # red
    3: '#2ca02c',  # green
}

# Line styles for flag configs (consistent across all plots)
FLAG_CONFIG_STYLES = {
    'all': '-',       # solid
    'partial': '--',  # dashed
    'none': ':',      # dotted
}

# Markers for decoders (consistent across all plots)
DECODER_MARKERS = {
    'pymatching': 'o',             # circle
    'correlated_pymatching': 's',  # square
    'tesseract': '^',              # triangle
}

# Display names
DECODER_DISPLAY_NAMES = {
    'pymatching': 'PyMatching',
    'correlated_pymatching': 'Correlated PyMatching',
    'tesseract': 'Tesseract',
}

FLAG_CONFIG_DISPLAY_NAMES = {
    'all': 'All Flags',
    'partial': 'Partial Flags',
    'none': 'No Flags',
}


# =============================================================================
# Data Loading
# =============================================================================

def load_data(csv_file: str) -> pd.DataFrame:
    """Load benchmark data from CSV file."""
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"  Loaded {len(df)} rows")
    print(f"  Directions: {df['direction'].unique()}")
    print(f"  Flag configs: {df['flag_config'].unique()}")
    print(f"  k values: {df['k'].unique()}")
    print(f"  Decoders: {df['decoder'].unique()}")
    print(f"  Physical error rates: {sorted(df['physical_error_rate'].unique())}")
    return df


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_by_decoder(df: pd.DataFrame, output_dir: str) -> None:
    """
    Create one figure per decoder with subplots for x and y directions.
    Different line styles for each flag config, colors for k values.
    """
    decoders = df['decoder'].unique()
    
    for decoder in decoders:
        print(f"\nGenerating plot for decoder: {decoder}")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        fig.suptitle(f'Logical Error Rate - {DECODER_DISPLAY_NAMES.get(decoder, decoder)}', 
                     fontsize=14, fontweight='bold')
        
        for ax_idx, direction in enumerate(['x', 'y']):
            ax = axes[ax_idx]
            ax.set_title(f'Direction: {direction.upper()}', fontsize=12)
            
            # Filter data for this decoder and direction
            mask = (df['decoder'] == decoder) & (df['direction'] == direction)
            subset = df[mask]
            
            # Plot each combination of k and flag_config
            # Since this is a single decoder plot, we use a fixed marker for the decoder
            decoder_marker = DECODER_MARKERS.get(decoder, 'o')
            
            for k in sorted(df['k'].unique()):
                for flag_config in ['all', 'partial', 'none']:
                    data = subset[(subset['k'] == k) & (subset['flag_config'] == flag_config)]
                    
                    if data.empty:
                        continue
                    
                    # Sort by physical error rate
                    data = data.sort_values('physical_error_rate')
                    
                    color = K_COLORS.get(k, 'black')
                    linestyle = FLAG_CONFIG_STYLES.get(flag_config, '-')
                    
                    label = f"k={k}, {FLAG_CONFIG_DISPLAY_NAMES.get(flag_config, flag_config)}"
                    
                    ax.errorbar(
                        data['physical_error_rate'],
                        data['logical_error_rate'],
                        yerr=data['error_bar'],
                        label=label,
                        color=color,
                        linestyle=linestyle,
                        marker=decoder_marker,
                        markersize=6,
                        capsize=3,
                        capthick=1,
                        linewidth=1.5,
                        alpha=0.8,
                    )
            
            # Configure axes
            ax.set_xlabel('Physical Error Rate', fontsize=11)
            ax.set_ylabel('Logical Error Rate', fontsize=11)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3, which='both')
            ax.legend(fontsize=8, loc='lower right', ncol=1)
        
        plt.tight_layout()
        
        # Save figure
        filename = f"decoder_{decoder}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filepath}")
        plt.close()


def plot_by_flag_config(df: pd.DataFrame, output_dir: str) -> None:
    """
    Create one figure per flag config with subplots for x and y directions.
    Different line styles for each decoder, colors for k values.
    """
    flag_configs = ['all', 'partial', 'none']
    
    for flag_config in flag_configs:
        print(f"\nGenerating plot for flag config: {flag_config}")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        fig.suptitle(f'Logical Error Rate - {FLAG_CONFIG_DISPLAY_NAMES.get(flag_config, flag_config)}', 
                     fontsize=14, fontweight='bold')
        
        for ax_idx, direction in enumerate(['x', 'y']):
            ax = axes[ax_idx]
            ax.set_title(f'Direction: {direction.upper()}', fontsize=12)
            
            # Filter data for this flag config and direction
            mask = (df['flag_config'] == flag_config) & (df['direction'] == direction)
            subset = df[mask]
            
            # Plot each combination of k and decoder
            # Since this is a single flag_config plot, we use a fixed line style for the flag config
            flag_linestyle = FLAG_CONFIG_STYLES.get(flag_config, '-')
            
            for k in sorted(df['k'].unique()):
                for decoder in ['pymatching', 'correlated_pymatching', 'tesseract']:
                    data = subset[(subset['k'] == k) & (subset['decoder'] == decoder)]
                    
                    if data.empty:
                        continue
                    
                    # Sort by physical error rate
                    data = data.sort_values('physical_error_rate')
                    
                    color = K_COLORS.get(k, 'black')
                    marker = DECODER_MARKERS.get(decoder, 'o')
                    
                    label = f"k={k}, {DECODER_DISPLAY_NAMES.get(decoder, decoder)}"
                    
                    ax.errorbar(
                        data['physical_error_rate'],
                        data['logical_error_rate'],
                        yerr=data['error_bar'],
                        label=label,
                        color=color,
                        linestyle=flag_linestyle,
                        marker=marker,
                        markersize=6,
                        capsize=3,
                        capthick=1,
                        linewidth=1.5,
                        alpha=0.8,
                    )
            
            # Configure axes
            ax.set_xlabel('Physical Error Rate', fontsize=11)
            ax.set_ylabel('Logical Error Rate', fontsize=11)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3, which='both')
            ax.legend(fontsize=8, loc='lower right', ncol=1)
        
        plt.tight_layout()
        
        # Save figure
        filename = f"flagconfig_{flag_config}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filepath}")
        plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    print("=" * 60)
    print("Benchmark Results Visualization")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}/")
    
    # Load data
    df = load_data(CSV_FILE)
    
    # Generate plots by decoder
    print("\n" + "-" * 60)
    print("Generating plots by decoder...")
    print("-" * 60)
    plot_by_decoder(df, OUTPUT_DIR)
    
    # Generate plots by flag config
    print("\n" + "-" * 60)
    print("Generating plots by flag config...")
    print("-" * 60)
    plot_by_flag_config(df, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("Done! All plots saved to:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()

