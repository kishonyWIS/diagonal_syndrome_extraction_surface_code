#!/usr/bin/env python3
"""Complete implementation with custom cube/pipe builders for diagonal plaquettes."""

import sys
import importlib
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Modify MEASUREMENT_SCHEDULE BEFORE importing any tqec modules
sys.path.insert(0, 'venv/lib/python3.13/site-packages')

# Change the constant before any imports
import tqec.plaquette.constants as constants
constants.MEASUREMENT_SCHEDULE = 7  # Allow schedule index 6

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

# Custom Identity compiler that handles schedule 7 (for qubit index 6) but keeps original basis
CustomIdentityPlaquetteCompiler = PlaquetteCompiler(
    "CustomIdentity",
    [
        # Compact schedule map: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
        # No gaps - direct mapping to avoid idle moments
        ChangeSchedulePass({i: i for i in range(8)}),
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
        # Spatial cube uses Z boundary basis for the spatial boundaries
        # The x basis here is actually the temporal boundary basis
        return self._generator.get_spatial_cube_qubit_raw_template(), (
            self._generator.get_spatial_cube_qubit_plaquettes(Basis.Z, spec.spatial_arms, z, None),
            self._generator.get_spatial_cube_qubit_plaquettes(Basis.Z, spec.spatial_arms, None, None),
            self._generator.get_spatial_cube_qubit_plaquettes(Basis.Z, spec.spatial_arms, None, z),
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


def create_original_memory_circuit(k=2):
    """Create the original memory experiment circuit."""
    print("Creating original memory experiment circuit...")
    
    # Create memory block graph
    mem_graph = memory(Basis.Z)
    
    # Compile with default convention
    compiled = compile_block_graph(mem_graph)
    
    # Generate stim circuit
    noise_model = NoiseModel.uniform_depolarizing(0.001)
    circuit = compiled.generate_stim_circuit(k=k, noise_model=noise_model)
    
    return circuit


def create_diagonal_memory_circuit(k=2):
    """Create the diagonal memory experiment circuit using diagonal plaquettes."""
    print("Creating diagonal memory experiment circuit with diagonal plaquettes...")
    
    # Create memory block graph
    mem_graph = memory(Basis.Z)
    
    # Create custom convention with diagonal plaquettes
    diagonal_convention = create_diagonal_convention()
    
    # Compile with diagonal convention
    compiled = compile_block_graph(mem_graph, convention=diagonal_convention)
    
    # Generate stim circuit
    noise_model = NoiseModel.uniform_depolarizing(0.001)
    circuit = compiled.generate_stim_circuit(k=k, noise_model=noise_model)
    
    return circuit




def calculate_logical_error_rate(circuit, shots=100000, noise_levels=[0.001, 0.002, 0.005]):
    """Calculate logical error rate using sinter."""
    if not PYMATCHING_AVAILABLE:
        print("Skipping logical error rate calculation - pymatching not available")
        return {}
    
    print(f"Calculating logical error rate with {shots} shots...")
    
    # Define noise levels to test
    results = {}
    
    for noise_level in noise_levels:
        print(f"  Testing noise level: {noise_level}")
        
        # Create a circuit with the specified noise level
        noisy_circuit = circuit.copy()
        
        # Replace noise parameters with the desired noise level
        noisy_circuit_str = str(noisy_circuit)
        noisy_circuit_str = noisy_circuit_str.replace('0.001', str(noise_level))
        import stim
        noisy_circuit = stim.Circuit(noisy_circuit_str)
        
        # Use sinter to collect statistics
        import sinter
        
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
            max_errors=3000,  # Allow all shots to be errors if needed
            num_workers=10
        )
        
        # Extract results
        if stats:
            stat = stats[0]
            logical_error_rate = stat.errors / stat.shots
            logical_errors = stat.errors
            
            # Calculate error bars using binomial distribution
            # Standard error for binomial distribution: sqrt(p*(1-p)/n)
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


def plot_distance_vs_k(distance_data, save_path="distance_vs_k.png"):
    """Plot circuit distance vs k for both circuit types."""
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


def plot_logical_error_rates_multi_k(results_data, save_path="logical_error_rates_multi_k.png"):
    """Plot logical error rate vs physical error rate for multiple k values and circuit types."""
    if not results_data:
        print("Cannot create plot - missing data")
        return
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Define colors for different k values
    k_colors = {1: 'blue', 2: 'red', 3: 'green', 4: 'orange', 5: 'purple'}
    
    # Define line styles for different circuit types
    circuit_styles = {
        'Original Circuit': '--',
        'Diagonal Circuit': '-'
    }
    
    # Define markers for different circuit types
    circuit_markers = {
        'Original Circuit': 'o',
        'Diagonal Circuit': 's'
    }
    
    for circuit_name in ['Original Circuit', 'Diagonal Circuit']:
        for k in sorted(results_data.keys()):
            if circuit_name in results_data[k]:
                circuit_results = results_data[k][circuit_name]
                if not circuit_results:
                    continue
                    
                # Extract data
                noise_levels = sorted(circuit_results.keys())
                rates = [circuit_results[n]['logical_error_rate'] for n in noise_levels]
                errors = [circuit_results[n]['error_bar'] for n in noise_levels]
                
                # Get color for this k value and style for this circuit
                color = k_colors.get(k, 'black')
                linestyle = circuit_styles.get(circuit_name, '-')
                marker = circuit_markers.get(circuit_name, 'o')
                
                # Plot with error bars
                plt.errorbar(noise_levels, rates, yerr=errors, 
                           label=f'{circuit_name} (k={k})', 
                           color=color, 
                           linestyle=linestyle,
                           marker=marker,
                           capsize=3, capthick=1, linewidth=2, markersize=6,
                           alpha=0.8)
    
    # Customize the plot
    plt.xlabel('Physical Error Rate', fontsize=14)
    plt.ylabel('Logical Error Rate', fontsize=14)
    plt.title('Logical Error Rate vs Physical Error Rate\nSurface Code Memory Experiment (Multiple k)', fontsize=16)
    plt.legend(fontsize=10, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.xscale('log')
    
    # Set axis limits
    all_rates = []
    all_noise_levels = []
    for k_data in results_data.values():
        for circuit_data in k_data.values():
            if circuit_data:
                all_rates.extend([circuit_data[n]['logical_error_rate'] for n in circuit_data.keys()])
                all_noise_levels.extend(circuit_data.keys())
    
    if all_rates and all_noise_levels:
        plt.xlim(min(all_noise_levels) * 0.5, max(all_noise_levels) * 2)
        plt.ylim(min(all_rates) * 0.5, max(all_rates) * 2)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Logical error rate plot saved as: {save_path}")
    
    # Show the plot
    plt.show()
    
    return plt


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


def generate_crumble_url(circuit, name):
    """Generate Crumble URL for circuit visualization."""
    try:
        crumble_url = circuit.to_crumble_url()
        print(f"{name} Crumble URL: {crumble_url}")
        return crumble_url
    except Exception as e:
        print(f"Error generating {name} Crumble URL: {e}")
        return None


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
        noise_levels = np.logspace(-3.5, -2, 4)  # Default: 5 noise levels from 0.0001 to 0.01
    
    print(f"Testing k values: {k_values}")
    print(f"Surface code distances: {[2*k+1 for k in k_values]}")
    print(f"Shots per noise level: {shots}")
    print(f"Skip distance calculation: {args.skip_distance}")
    print(f"Skip logical error calculation: {args.skip_logical_error}")
    print(f"Skip Crumble URLs: {args.skip_crumble}")
    print()
    
    # Show plaquette configurations
    print("=== Plaquette Configurations ===")
    print("Original X-basis bulk: \"-x1- -x2- -x3- -x5-\" (schedule: [1,2,3,5])")
    print("Diagonal X-basis bulk: \"-x1- -x4- -x3- -x2-\" (schedule: [1,4,3,2])")
    print()
    print("Original Z-basis bulk: \"-z1- -z2- -z3- -z5-\" (schedule: [1,2,3,5])")
    print("Diagonal Z-basis bulk: \"-z6- -z4- -z3- -z5-\" (schedule: [6,4,3,5])")
    print()
    
    # Store results for all k values and circuit types
    all_results = {}
    all_distances = {}
    all_crumble_urls = {}
    
    # Test each k value
    for k in k_values:
        distance = 2 * k + 1
        print(f"=== Testing k={k} (surface code distance={distance}) ===")
        
        # Create circuits
        print("Creating circuits...")
        original_circuit = create_original_memory_circuit(k)
        diagonal_circuit = create_diagonal_memory_circuit(k)
        print()
        
        # Store circuit info
        circuits = {
            'Original Circuit': original_circuit,
            'Diagonal Circuit': diagonal_circuit
        }
        
        print(f"Circuit sizes for k={k}:")
        for name, circuit in circuits.items():
            print(f"  {name}: {len(circuit)} instructions")
        print()
        
        # Generate Crumble URLs (optional)
        if not args.skip_crumble:
            print("Generating Crumble URLs...")
            k_urls = {}
            for name, circuit in circuits.items():
                url = generate_crumble_url(circuit, f"{name} k={k}")
                k_urls[name] = url
                print(f"  {name}: {url}")
            all_crumble_urls[k] = k_urls
            print()
        
        # Calculate logical error rates (optional)
        if not args.skip_logical_error and PYMATCHING_AVAILABLE:
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
        
        # Calculate distances (optional)
        if not args.skip_distance:
            print(f"Calculating circuit distances for k={k}...")
            k_distances = {}
            
            for name, circuit in circuits.items():
                try:
                    graphlike_dist = len(circuit.shortest_graphlike_error(canonicalize_circuit_errors=True))
                    circuit_dist = calculate_circuit_distance(circuit)
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
    
    # Create distance vs k plot (optional)
    if not args.skip_distance and all_distances:
        print("=== Creating Distance vs k Plot ===")
        plot_distance_vs_k(all_distances)
        print()
    elif args.skip_distance:
        print("Skipping distance vs k plot")
        print()
    
    # Create logical error rate plot (optional)
    if not args.skip_logical_error and PYMATCHING_AVAILABLE and all_results:
        print("=== Creating Logical Error Rate Plot ===")
        plot_logical_error_rates_multi_k(all_results)
        print()
    elif args.skip_logical_error:
        print("Skipping logical error rate plot")
        print()
    elif not PYMATCHING_AVAILABLE:
        print("Skipping logical error rate plot (pymatching not available)")
        print()
    
    # Print summary
    print("=== Summary ===")
    
    # Distance summary (optional)
    if not args.skip_distance and all_distances:
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
    
    # Crumble URLs summary (optional)
    if not args.skip_crumble and all_crumble_urls:
        print("Crumble URLs for circuit visualization:")
        for k in k_values:
            print(f"k={k}:")
            if k in all_crumble_urls:
                for circuit_name, url in all_crumble_urls[k].items():
                    print(f"  {circuit_name}: {url}")
            print()
    
if __name__ == "__main__":
    main()
