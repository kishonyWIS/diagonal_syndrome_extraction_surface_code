#!/usr/bin/env python3
"""Complete implementation with custom cube/pipe builders for diagonal plaquettes."""

import csv
import os
import sys
import importlib
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Modify MEASUREMENT_SCHEDULE BEFORE importing any tqec modules
sys.path.insert(0, 'venv/lib/python3.13/site-packages')

# Change the constant before any imports
import tqec.plaquette.constants as constants
constants.MEASUREMENT_SCHEDULE = 8  # Allow schedule values up to 7

# Now import tqec modules
from tqec.gallery import memory
from tqec import compile_block_graph, NoiseModel
from tqec.utils.enums import Basis, Orientation
from diagonal_plaquettes import DiagonalPlaquetteGenerator
import sinter
try:
    import pymatching
    PYMATCHING_AVAILABLE = True
except ImportError:
    PYMATCHING_AVAILABLE = False
    print("Warning: pymatching not available. Logical error rate calculation will be skipped.")

# Reload the translator module to pick up the new MEASUREMENT_SCHEDULE
import tqec.plaquette.rpng.translators.default as default_translator
importlib.reload(default_translator)
DefaultRPNGTranslator = default_translator.DefaultRPNGTranslator

# Import necessary classes for custom compilation
from tqec.compile.convention import Convention, ConventionTriplet
from tqec.compile.specs.base import CubeBuilder, PipeBuilder
from tqec.compile.specs.library.fixed_bulk import FixedBulkPipeBuilder
from tqec.compile.observables.fixed_bulk_builder import FIXED_BULK_OBSERVABLE_BUILDER
from tqec.computation.cube import Port, YHalfCube, ZXCube
from tqec.compile.specs.enums import SpatialArms
from tqec.templates.base import RectangularTemplate
from tqec.plaquette.plaquette import Plaquettes
from tqec.utils.position import Direction3D
from tqec.utils.scale import LinearFunction
from tqec.compile.blocks.block import Block
from tqec.compile.blocks.layers.atomic.plaquettes import PlaquetteLayer
from tqec.compile.blocks.layers.composed.repeated import RepeatedLayer
from tqec.utils.exceptions import TQECError
from typing import override

# Create a custom CSS compiler that handles schedule 7
from tqec.plaquette.compilation.base import PlaquetteCompiler
from tqec.plaquette.compilation.passes.controlled_gate_basis import ChangeControlledGateBasisPass
from tqec.plaquette.compilation.passes.measurement_basis import ChangeMeasurementBasisPass
from tqec.plaquette.compilation.passes.reset_basis import ChangeResetBasisPass
from tqec.plaquette.compilation.passes.scheduling import ChangeSchedulePass
from tqec.plaquette.compilation.passes.sort_targets import SortTargetsPass
from tqec.plaquette.compilation.passes.transformer import ScheduleConstant

def _add_hadamard(mergeable_instructions: frozenset[str]) -> frozenset[str]:
    return mergeable_instructions | frozenset(["H"])

# Custom Identity compiler that handles schedule values up to 8 but keeps original basis
CustomIdentityPlaquetteCompiler = PlaquetteCompiler(
    "CustomIdentity",
    [
        # Compact schedule map: {0: 0, 1: 1, ..., 8: 8}
        # No gaps - direct mapping to avoid idle moments
        ChangeSchedulePass({i: i for i in range(9)}),
        # Sort the instruction targets to normalize the circuits.
        SortTargetsPass(),
    ],
    mergeable_instructions_modifier=lambda x: x | frozenset(["H"]),  # Keep original mergeable instructions
)


class DiagonalCubeBuilder(CubeBuilder):
    """Custom cube builder that uses the diagonal plaquette generator."""
    
    def __init__(self) -> None:
        """Initialize with diagonal generator."""
        translator = DefaultRPNGTranslator()
        compiler = CustomIdentityPlaquetteCompiler
        self._generator = DiagonalPlaquetteGenerator(translator, compiler)

    def _get_template_and_plaquettes(
        self, spec
    ) -> tuple[RectangularTemplate, tuple[Plaquettes, Plaquettes, Plaquettes]]:
        """Get the template and plaquettes using diagonal generator."""
        from tqec.compile.specs.base import CubeSpec
        from tqec.computation.cube import ZXCube
        
        assert isinstance(spec.kind, ZXCube)
        x, _, z = spec.kind.as_tuple()
        if not spec.is_spatial:
            orientation = Orientation.HORIZONTAL if x == Basis.Z else Orientation.VERTICAL
            return self._generator.get_memory_qubit_raw_template(), (
                self._generator.get_memory_qubit_plaquettes(orientation, z, None),
                self._generator.get_memory_qubit_plaquettes(orientation, None, None),
                self._generator.get_memory_qubit_plaquettes(orientation, None, z),
            )
        # else: spatial cube
        # Use x_basis as the spatial boundary basis (matching FixedBulkCubeBuilder)
        return self._generator.get_spatial_cube_qubit_raw_template(), (
            self._generator.get_spatial_cube_qubit_plaquettes(x, spec.spatial_arms, z, None),
            self._generator.get_spatial_cube_qubit_plaquettes(x, spec.spatial_arms, None, None),
            self._generator.get_spatial_cube_qubit_plaquettes(x, spec.spatial_arms, None, z),
        )

    @override
    def __call__(self, spec, block_temporal_height: LinearFunction):
        """Build a block using diagonal plaquettes."""
        kind = spec.kind
        if isinstance(kind, Port):
            raise TQECError("Cannot build a block for a Port.")
        elif isinstance(kind, YHalfCube):
            raise NotImplementedError("Y cube is not implemented.")
        # else
        template, (init, repeat, measure) = self._get_template_and_plaquettes(spec)
        layers = [
            PlaquetteLayer(template, init),
            RepeatedLayer(PlaquetteLayer(template, repeat), repetitions=block_temporal_height),
            PlaquetteLayer(template, measure),
        ]
        return Block(layers)


class DiagonalPipeBuilder(PipeBuilder):
    """Custom pipe builder that uses the diagonal plaquette generator."""
    
    def __init__(self) -> None:
        """Initialize with diagonal generator."""
        translator = DefaultRPNGTranslator()
        compiler = CustomIdentityPlaquetteCompiler
        self._generator = DiagonalPlaquetteGenerator(translator, compiler)

    @override
    def __call__(self, spec, block_temporal_height: LinearFunction):
        """Build a pipe using diagonal plaquettes."""
        from tqec.utils.scale import LinearFunction
        from tqec.compile.blocks.block import Block
        from tqec.compile.blocks.layers.atomic.plaquettes import PlaquetteLayer
        from tqec.compile.blocks.layers.composed.repeated import RepeatedLayer
        from tqec.compile.specs.enums import SpatialArms
        from tqec.utils.position import Direction3D
        from tqec.utils.exceptions import TQECError
        
        if spec.pipe_kind.is_temporal:
            # For temporal pipes, delegate to original
            from tqec.compile.specs.library.fixed_bulk import FixedBulkPipeBuilder
            from tqec.plaquette.compilation.base import IdentityPlaquetteCompiler
            original_builder = FixedBulkPipeBuilder(IdentityPlaquetteCompiler, DefaultRPNGTranslator())
            return original_builder(spec, block_temporal_height)
        
        # Spatial pipe
        x, y, z = spec.pipe_kind.x, spec.pipe_kind.y, spec.pipe_kind.z
        assert x is not None or y is not None
        spatial_boundary_basis: Basis = x if x is not None else y  # type: ignore
        
        # Get the arm(s)
        arms = self._get_spatial_cube_arms(spec)
        
        # Get template and plaquettes
        pipe_template = self._generator.get_spatial_cube_arm_raw_template(arms)
        initialisation_plaquettes = self._generator.get_spatial_cube_arm_plaquettes(
            spatial_boundary_basis, arms, spec.cube_specs, z, None
        )
        temporal_bulk_plaquettes = self._generator.get_spatial_cube_arm_plaquettes(
            spatial_boundary_basis, arms, spec.cube_specs, None, None
        )
        measurement_plaquettes = self._generator.get_spatial_cube_arm_plaquettes(
            spatial_boundary_basis, arms, spec.cube_specs, None, z
        )
        
        return Block(
            [
                PlaquetteLayer(pipe_template, initialisation_plaquettes),
                RepeatedLayer(
                    PlaquetteLayer(pipe_template, temporal_bulk_plaquettes),
                    repetitions=block_temporal_height,
                ),
                PlaquetteLayer(pipe_template, measurement_plaquettes),
            ]
        )
    
    @staticmethod
    def _get_spatial_cube_arms(spec) -> SpatialArms:
        """Return the arm(s) corresponding to the provided spec."""
        assert spec.pipe_kind.is_spatial
        assert any(spec.is_spatial for spec in spec.cube_specs)
        u, v = spec.cube_specs
        pipedir = spec.pipe_kind.direction
        arms = SpatialArms.NONE
        if u.is_spatial:
            arms |= SpatialArms.RIGHT if pipedir == Direction3D.X else SpatialArms.DOWN
        if v.is_spatial:
            arms |= SpatialArms.LEFT if pipedir == Direction3D.X else SpatialArms.UP
        return arms


def create_diagonal_convention():
    """Create a custom convention using diagonal plaquettes."""
    cube_builder = DiagonalCubeBuilder()
    pipe_builder = DiagonalPipeBuilder()
    
    return Convention(
        "diagonal_plaquettes",
        ConventionTriplet(
            cube_builder,
            pipe_builder,
            FIXED_BULK_OBSERVABLE_BUILDER
        )
    )


def create_original_memory_circuit(k=2, return_both=False):
    """Create the original memory experiment circuit (without noise, compacted).
    
    Args:
        k: Scale factor
        return_both: If True, return tuple (before_compact, after_compact)
    """
    from compact_circuit import compact_and_delay_init
    
    print("Creating original memory experiment circuit...")
    
    # Create memory block graph
    mem_graph = memory(Basis.Z)
    
    # Compile with default convention
    compiled = compile_block_graph(mem_graph)
    
    # Generate stim circuit WITHOUT noise
    circuit_before = compiled.generate_stim_circuit(k=k, noise_model=None)
    
    # Compact the circuit (ASAP + ALAP scheduling)
    circuit_after = compact_and_delay_init(circuit_before)
    
    if return_both:
        return circuit_before, circuit_after
    return circuit_after


def create_diagonal_memory_circuit(k=2, return_both=False):
    """Create the diagonal memory experiment circuit using diagonal plaquettes (without noise, compacted).
    
    Args:
        k: Scale factor
        return_both: If True, return tuple (before_compact, after_compact)
    """
    from compact_circuit import compact_and_delay_init
    
    print("Creating diagonal memory experiment circuit with diagonal plaquettes...")
    
    # Create memory block graph
    mem_graph = memory(Basis.Z)
    
    # Create custom convention with diagonal plaquettes
    diagonal_convention = create_diagonal_convention()
    
    # Compile with diagonal convention
    compiled = compile_block_graph(mem_graph, convention=diagonal_convention)
    
    # Generate stim circuit WITHOUT noise
    circuit_before = compiled.generate_stim_circuit(k=k, noise_model=None)
    
    # Compact the circuit (ASAP + ALAP scheduling)
    circuit_after = compact_and_delay_init(circuit_before)
    
    if return_both:
        return circuit_before, circuit_after
    return circuit_after




def calculate_logical_error_rate(circuit, shots=100000, noise_levels=[0.001, 0.002, 0.005]):
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
        
        # Create a task for sinter
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


def plot_distance_vs_k(distance_data, save_path="benchmark_plots/memory_distance_vs_k.png"):
    """Plot circuit distance vs k for both circuit types."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if not distance_data:
        print("Cannot create distance plot - missing data")
        return
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Extract data
    k_values = sorted(distance_data.keys())
    original_distances = []
    diagonal_distances = []
    
    for k in k_values:
        if 'Original Circuit' in distance_data[k] and distance_data[k]['Original Circuit']['circuit'] != 'Error':
            original_distances.append(distance_data[k]['Original Circuit']['circuit'])
        else:
            original_distances.append(None)
            
        if 'Diagonal Circuit' in distance_data[k] and distance_data[k]['Diagonal Circuit']['circuit'] != 'Error':
            diagonal_distances.append(distance_data[k]['Diagonal Circuit']['circuit'])
        else:
            diagonal_distances.append(None)
    
    # Plot both circuit types
    plt.plot(k_values, original_distances, 'o--', color='blue', linewidth=2, markersize=8, 
             label='Original Circuit', alpha=0.8)
    plt.plot(k_values, diagonal_distances, 's-', color='red', linewidth=2, markersize=8, 
             label='Diagonal Circuit', alpha=0.8)
    
    # Customize the plot
    plt.xlabel('k (Surface Code Parameter)', fontsize=14)
    plt.xticks(k_values)
    plt.ylabel('Circuit Distance', fontsize=14)
    plt.title('Circuit Distance vs k\nSurface Code Memory Experiment', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Set axis limits
    plt.xlim(min(k_values) - 0.1, max(k_values) + 0.1)
    all_distances = [d for d in original_distances + diagonal_distances if d is not None]
    if all_distances:
        plt.ylim(min(all_distances) - 0.5, max(all_distances) + 0.5)
    
    # Add annotations for each point
    for i, (k, orig_dist, diag_dist) in enumerate(zip(k_values, original_distances, diagonal_distances)):
        if orig_dist is not None:
            plt.annotate(f'{orig_dist}', (k, orig_dist), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=10, color='blue')
        if diag_dist is not None:
            plt.annotate(f'{diag_dist}', (k, diag_dist), textcoords="offset points", 
                        xytext=(0,-15), ha='center', fontsize=10, color='red')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Distance plot saved as: {save_path}")
    
    # Show the plot
    plt.show()
    
    return plt


def results_to_sinter_stats(results_data):
    """Convert results dictionary to list of sinter.TaskStats for plotting.
    
    Args:
        results_data: Dict with structure {k: {circuit_name: {noise_level: {logical_error_rate, logical_errors, shots, error_bar}}}}
        
    Returns:
        List of sinter.TaskStats objects
    """
    stats_list = []
    
    for k in results_data:
        for circuit_name in results_data[k]:
            circuit_results = results_data[k][circuit_name]
            if not circuit_results:
                continue
            
            for noise_level, result in circuit_results.items():
                # Create TaskStats object
                stats = sinter.TaskStats(
                    strong_id=f"{circuit_name}_k{k}_p{noise_level}",
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


def plot_logical_error_rates_multi_k(results_data, save_path="benchmark_plots/memory_error_rates.png"):
    """Plot logical error rate vs physical error rate using sinter.plot_error_rate."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if not results_data:
        print("Cannot create plot - missing data")
        return
    
    # Convert to sinter stats format
    stats_list = results_to_sinter_stats(results_data)
    
    if not stats_list:
        print("No data to plot")
        return
    
    # Define colors and markers by k value
    k_colors = {1: 'C0', 2: 'C1', 3: 'C2', 4: 'C3', 5: 'C4'}
    k_markers = {1: 'o', 2: 's', 3: '^', 4: 'D', 5: 'v'}
    
    # Create combined comparison plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    
    # Custom plot_args_func for combined plot
    def combined_plot_args(index, curve_id):
        # curve_id is like "Original k=1" or "Diagonal k=2"
        parts = curve_id.split()
        circuit_type = parts[0]  # "Original" or "Diagonal"
        k = int(parts[1].split('=')[1])  # Extract k value
        
        return {
            'color': k_colors.get(k, 'black'),
            'marker': k_markers.get(k, 'o'),
            'linestyle': '--' if circuit_type == 'Original' else '-',
        }
    
    sinter.plot_error_rate(
        ax=ax,
        stats=stats_list,
        x_func=lambda s: s.json_metadata['p'],
        group_func=lambda s: f"{s.json_metadata['circuit'].replace(' Circuit', '')} k={s.json_metadata['k']}",
        plot_args_func=combined_plot_args,
    )
    
    ax.loglog()
    ax.set_xlabel("Physical Error Rate (Uniform Depolarizing)", fontsize=16)
    ax.set_ylabel("Logical Error Rate (per Shot)", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    ax.grid(which='major', alpha=0.5)
    ax.grid(which='minor', alpha=0.2)
    ax.legend(fontsize=14, ncol=2)
    fig.set_dpi(150)
    fig.tight_layout()
    
    # Save plot
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Logical error rate plot saved as: {save_path}")
    
    plt.show()
    
    return fig


def calculate_circuit_distance(circuit):
    """Calculate the circuit level distance."""
    try:
        logical_errors = circuit.search_for_undetectable_logical_errors(
            dont_explore_detection_event_sets_with_size_above=4,
            dont_explore_edges_with_degree_above=9999,
            dont_explore_edges_increasing_symptom_degree=False,
            canonicalize_circuit_errors=True
        )
        return len(logical_errors)
    except Exception as e:
        print(f"Error calculating circuit distance: {e}")
        return None


# =============================================================================
# CSV Data Saving/Loading
# =============================================================================

def save_error_rates_to_csv(all_results, filepath="benchmark_data/memory_error_rates.csv"):
    """Save logical error rate results to CSV file.
    
    Args:
        all_results: Dict with structure {k: {circuit_name: {noise_level: {logical_error_rate, logical_errors, shots, error_bar}}}}
        filepath: Path to save CSV file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    fieldnames = ['k', 'circuit_type', 'physical_error_rate', 'logical_error_rate', 'logical_errors', 'shots', 'error_bar']
    
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for k in sorted(all_results.keys()):
            for circuit_name, circuit_results in all_results[k].items():
                for noise_level, result in circuit_results.items():
                    writer.writerow({
                        'k': k,
                        'circuit_type': circuit_name,
                        'physical_error_rate': noise_level,
                        'logical_error_rate': result['logical_error_rate'],
                        'logical_errors': result['logical_errors'],
                        'shots': result['shots'],
                        'error_bar': result['error_bar'],
                    })
    
    print(f"Saved error rate results to {filepath}")


def save_distances_to_csv(all_distances, filepath="benchmark_data/memory_distances.csv"):
    """Save circuit distance results to CSV file.
    
    Args:
        all_distances: Dict with structure {k: {circuit_name: {graphlike, circuit}}}
        filepath: Path to save CSV file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    fieldnames = ['k', 'circuit_type', 'graphlike_distance', 'circuit_distance']
    
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for k in sorted(all_distances.keys()):
            for circuit_name, distances in all_distances[k].items():
                writer.writerow({
                    'k': k,
                    'circuit_type': circuit_name,
                    'graphlike_distance': distances.get('graphlike', 'N/A'),
                    'circuit_distance': distances.get('circuit', 'N/A'),
                })
    
    print(f"Saved distance results to {filepath}")


def load_error_rates_from_csv(filepath="benchmark_data/memory_error_rates.csv"):
    """Load logical error rate results from CSV file.
    
    Returns:
        Dict with structure {k: {circuit_name: {noise_level: {logical_error_rate, logical_errors, shots, error_bar}}}}
    """
    all_results = {}
    
    with open(filepath, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            k = int(row['k'])
            circuit_type = row['circuit_type']
            noise_level = float(row['physical_error_rate'])
            
            if k not in all_results:
                all_results[k] = {}
            if circuit_type not in all_results[k]:
                all_results[k][circuit_type] = {}
            
            all_results[k][circuit_type][noise_level] = {
                'logical_error_rate': float(row['logical_error_rate']),
                'logical_errors': int(row['logical_errors']),
                'shots': int(row['shots']),
                'error_bar': float(row['error_bar']),
            }
    
    print(f"Loaded error rate results from {filepath}")
    return all_results


def load_distances_from_csv(filepath="benchmark_data/memory_distances.csv"):
    """Load circuit distance results from CSV file.
    
    Returns:
        Dict with structure {k: {circuit_name: {graphlike, circuit}}}
    """
    all_distances = {}
    
    with open(filepath, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            k = int(row['k'])
            circuit_type = row['circuit_type']
            
            if k not in all_distances:
                all_distances[k] = {}
            
            graphlike = row['graphlike_distance']
            circuit = row['circuit_distance']
            
            all_distances[k][circuit_type] = {
                'graphlike': int(graphlike) if graphlike != 'N/A' else 'N/A',
                'circuit': int(circuit) if circuit != 'N/A' else 'N/A',
            }
    
    print(f"Loaded distance results from {filepath}")
    return all_distances


def generate_crumble_url(circuit, name):
    """Generate Crumble URL for circuit visualization."""
    try:
        crumble_url = circuit.to_crumble_url()
        print(f"{name} Crumble URL: {crumble_url}")
        return crumble_url
    except Exception as e:
        print(f"Error generating {name} Crumble URL: {e}")
        return None


def save_crumble_urls_html(urls_dict, output_dir="crumble_urls", experiment_name="memory"):
    """Save Crumble URLs as HTML files with clickable links.
    
    Args:
        urls_dict: Dict with structure {k: {circuit_name: {'before': url, 'after': url}}}
        output_dir: Directory to save HTML files
        experiment_name: Name of the experiment for the HTML title
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Create an index HTML file
    index_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{experiment_name.title()} Experiment - Crumble URLs</title>
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
    <h1>{experiment_name.title()} Experiment - Crumble Circuit Visualizations</h1>
"""
    
    for k in sorted(urls_dict.keys()):
        index_html += f"    <h2>k = {k} (distance = {2*k+1})</h2>\n"
        
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


def main():
    """Main comparison function for original vs diagonal circuits."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compare original vs diagonal surface code circuits')
    parser.add_argument('--k-values', nargs='+', type=int, default=[1, 2, 3],
                       help='k values to test (default: 1 2 3)')
    parser.add_argument('--shots', type=int, default=300000000,
                       help='Number of shots for logical error rate calculation (default: 50000)')
    parser.add_argument('--noise-levels', nargs='+', type=float, default=None,
                       help='Physical error rates to test (default: logspace from 0.0001 to 0.01)')
    parser.add_argument('--skip-distance', action='store_true',
                       help='Skip circuit distance calculation and plotting')
    parser.add_argument('--skip-logical-error', action='store_true',
                       help='Skip logical error rate calculation and plotting')
    parser.add_argument('--skip-crumble', action='store_true',
                       help='Skip Crumble URL generation')
    parser.add_argument('--load-distances', type=str, default=None,
                       help='Load distances from CSV file instead of recomputing')
    parser.add_argument('--load-error-rates', type=str, default=None,
                       help='Load error rates from CSV file instead of recomputing')
    
    args = parser.parse_args()
    
    print("=== Memory Experiment Comparison: Original vs Diagonal Circuits ===")
    print(f"MEASUREMENT_SCHEDULE modified to: {constants.MEASUREMENT_SCHEDULE}")
    print("Using custom CSS compiler to handle qubit index 6")
    print()
    
    # Configuration
    k_values = args.k_values
    shots = args.shots
    if args.noise_levels:
        noise_levels = args.noise_levels
    else:
        noise_levels = np.logspace(-3.5, -2, 7)  # Default: 5 noise levels from 0.0001 to 0.01
    
    print(f"Testing k values: {k_values}")
    print(f"Surface code distances: {[2*k+1 for k in k_values]}")
    print(f"Shots per noise level: {shots}")
    print(f"Skip distance calculation: {args.skip_distance}")
    print(f"Skip logical error calculation: {args.skip_logical_error}")
    print(f"Skip Crumble URLs: {args.skip_crumble}")
    print(f"Load distances from: {args.load_distances}")
    print(f"Load error rates from: {args.load_error_rates}")
    print()
    
    # Show plaquette configurations
    print("=== Plaquette Configurations ===")
    print("Original X-basis bulk: \"-x1- -x2- -x3- -x5-\" (schedule: [1,2,3,5])")
    print("Diagonal X-basis bulk: \"-x7- -x5- -x4- -x6-\" (schedule: [7,5,4,6])")
    print()
    print("Original Z-basis bulk: \"-z1- -z2- -z3- -z5-\" (schedule: [1,2,3,5])")
    print("Diagonal Z-basis bulk: \"-z1- -z3- -z4- -z2-\" (schedule: [1,3,4,2])")
    print()
    
    # Store results for all k values and circuit types
    all_results = {}
    all_distances = {}
    all_crumble_urls = {}
    
    # Load distances from CSV if provided
    if args.load_distances:
        print(f"=== Loading Distances from {args.load_distances} ===")
        try:
            all_distances = load_distances_from_csv(args.load_distances)
            print()
        except FileNotFoundError:
            print(f"Warning: File {args.load_distances} not found. Will compute distances.")
            args.load_distances = None
            print()
    
    # Load error rates from CSV if provided
    if args.load_error_rates:
        print(f"=== Loading Error Rates from {args.load_error_rates} ===")
        try:
            all_results = load_error_rates_from_csv(args.load_error_rates)
            print()
        except FileNotFoundError:
            print(f"Warning: File {args.load_error_rates} not found. Will compute error rates.")
            args.load_error_rates = None
            print()
    
    # Test each k value
    for k in k_values:
        distance = 2 * k + 1
        print(f"=== Testing k={k} (surface code distance={distance}) ===")
        
        # Create circuits (with both before and after compactification)
        print("Creating circuits...")
        original_before, original_circuit = create_original_memory_circuit(k, return_both=True)
        diagonal_before, diagonal_circuit = create_diagonal_memory_circuit(k, return_both=True)
        print()
        
        # Store circuit info (use compacted versions for simulation)
        circuits = {
            'Original Circuit': original_circuit,
            'Diagonal Circuit': diagonal_circuit
        }
        
        # Store both versions for Crumble URLs
        circuits_both = {
            'Original Circuit': {'before': original_before, 'after': original_circuit},
            'Diagonal Circuit': {'before': diagonal_before, 'after': diagonal_circuit}
        }
        
        print(f"Circuit sizes for k={k}:")
        for name, circuit in circuits.items():
            print(f"  {name}: {len(circuit)} instructions")
        print()
        
        # Generate Crumble URLs (optional)
        if not args.skip_crumble:
            print("Generating Crumble URLs...")
            k_urls = {}
            for name, circuit_versions in circuits_both.items():
                k_urls[name] = {}
                # Before compactification
                url_before = generate_crumble_url(circuit_versions['before'], f"{name} k={k} (before compact)")
                k_urls[name]['before'] = url_before
                # After compactification
                url_after = generate_crumble_url(circuit_versions['after'], f"{name} k={k} (after compact)")
                k_urls[name]['after'] = url_after
            all_crumble_urls[k] = k_urls
            print()
        
        # Calculate logical error rates (optional, skip if loaded from CSV)
        if args.load_error_rates:
            if k in all_results:
                print(f"Using loaded error rates for k={k}")
            else:
                print(f"Warning: No loaded error rates for k={k}")
            print()
        elif not args.skip_logical_error and PYMATCHING_AVAILABLE:
            print(f"Calculating logical error rates for k={k}...")
            k_results = {}
            
            for name, circuit in circuits.items():
                print(f"  {name}:")
                error_rates = calculate_logical_error_rate(circuit, shots=shots, noise_levels=noise_levels)
                k_results[name] = error_rates
            
            all_results[k] = k_results
            print()
        elif args.skip_logical_error:
            print(f"Skipping logical error rate calculation for k={k}")
            print()
        elif not PYMATCHING_AVAILABLE:
            print(f"Skipping logical error rate calculation for k={k} (pymatching not available)")
            print()
        
        # Calculate distances (optional, skip if loaded from CSV)
        if args.load_distances:
            if k in all_distances:
                print(f"Using loaded distances for k={k}")
            else:
                print(f"Warning: No loaded distances for k={k}")
            print()
        elif not args.skip_distance:
            print(f"Calculating circuit distances for k={k}...")
            k_distances = {}
            
            # Need to add noise for distance calculation
            distance_noise_model = NoiseModel.uniform_depolarizing(0.001)
            
            for name, circuit in circuits.items():
                try:
                    # Add noise for distance calculation
                    noisy_circuit = distance_noise_model.noisy_circuit(circuit)
                    graphlike_dist = len(noisy_circuit.shortest_graphlike_error(canonicalize_circuit_errors=True))
                    circuit_dist = calculate_circuit_distance(noisy_circuit)
                    k_distances[name] = {
                        'graphlike': graphlike_dist,
                        'circuit': circuit_dist
                    }
                    print(f"  {name}: graphlike={graphlike_dist}, circuit={circuit_dist}")
                except Exception as e:
                    print(f"  {name}: Error calculating distance - {e}")
                    k_distances[name] = {'graphlike': 'Error', 'circuit': 'Error'}
            
            all_distances[k] = k_distances
            print()
        else:
            print(f"Skipping circuit distance calculation for k={k}")
            print()
    
    # Save data to CSV and create plots
    if all_distances:
        if not args.load_distances:
            print("=== Saving Distance Data ===")
            save_distances_to_csv(all_distances)
        print("=== Creating Distance vs k Plot ===")
        plot_distance_vs_k(all_distances)
        print()
    elif args.skip_distance:
        print("Skipping distance calculation")
        print()
    
    if all_results:
        if not args.load_error_rates:
            print("=== Saving Error Rate Data ===")
            save_error_rates_to_csv(all_results)
        print("=== Creating Logical Error Rate Plot ===")
        plot_logical_error_rates_multi_k(all_results)
        print()
    elif args.skip_logical_error:
        print("Skipping logical error rate calculation")
        print()
    elif not PYMATCHING_AVAILABLE:
        print("Skipping logical error rate calculation (pymatching not available)")
        print()
    
    # Print summary
    print("=== Summary ===")
    
    # Distance summary (if available)
    if all_distances:
        print("Circuit Distances:")
        for k in k_values:
            distance = 2 * k + 1
            print(f"k={k} (surface code distance={distance}):")
            if k in all_distances:
                for circuit_name, dists in all_distances[k].items():
                    if dists['circuit'] != 'Error':
                        print(f"  {circuit_name}: circuit distance = {dists['circuit']}")
                    else:
                        print(f"  {circuit_name}: Error calculating distance")
            print()
    
    # Crumble URLs summary and HTML save (optional)
    if not args.skip_crumble and all_crumble_urls:
        print("Crumble URLs for circuit visualization:")
        for k in k_values:
            print(f"k={k}:")
            if k in all_crumble_urls:
                for circuit_name, urls in all_crumble_urls[k].items():
                    print(f"  {circuit_name}:")
                    print(f"    Before compact: {urls.get('before', 'N/A')}")
                    print(f"    After compact: {urls.get('after', 'N/A')}")
            print()
        
        # Save Crumble URLs to HTML file
        save_crumble_urls_html(all_crumble_urls, output_dir="crumble_urls", experiment_name="memory")
    
if __name__ == "__main__":
    main()
