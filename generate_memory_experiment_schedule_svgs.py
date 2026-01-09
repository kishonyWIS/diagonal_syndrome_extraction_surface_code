#!/usr/bin/env python3
"""Generate SVG visualizations of the memory experiment schedules."""

import sys
sys.path.insert(0, 'venv/lib/python3.13/site-packages')

# Modify MEASUREMENT_SCHEDULE BEFORE importing any tqec modules
import tqec.plaquette.constants as constants
constants.MEASUREMENT_SCHEDULE = 8  # Allow schedule values up to 7

import importlib
import tqec.plaquette.rpng.translators.default as default_translator
importlib.reload(default_translator)

from tqec.gallery import memory
from tqec import compile_block_graph
from tqec.utils.enums import Basis

# Modify the hook error line coefficient for our visualizations
# Monkey-patch DrawerConfiguration to use 0.8 instead of 0.9 for hook error lines
from tqec.visualisation.configuration import DrawerConfiguration

_original_drawer_config_init = DrawerConfiguration.__init__

def _patched_drawer_config_init(self, *args, **kwargs):
    # Set our custom default if not explicitly provided
    if 'hook_error_line_lerp_coefficient' not in kwargs:
        kwargs['hook_error_line_lerp_coefficient'] = 0.8
    _original_drawer_config_init(self, *args, **kwargs)

DrawerConfiguration.__init__ = _patched_drawer_config_init

# Also patch the default arguments in the grid module functions
# which contain pre-instantiated DrawerConfiguration objects
import tqec.visualisation.computation.plaquette.grid as grid_module

# plaquette_grid_svg_viewer defaults
grid_module.plaquette_grid_svg_viewer.__defaults__ = (
    None, None, None, None, True, True, True, (0, 0), (), 
    DrawerConfiguration(),  # This now uses our patched __init__ with 0.8
    None,
)

# plaquette_grid_to_svg defaults
grid_module.plaquette_grid_to_svg.__defaults__ = (
    True, True, True, (0, 0), (), 
    DrawerConfiguration(),  # This now uses our patched __init__ with 0.8
    None,
)

# Import the diagonal convention from benchmark_memory
from benchmark_memory import create_diagonal_convention


def generate_memory_schedule_svgs(k=2, output_dir="benchmark_plots/svgs"):
    """Generate SVG visualizations for standard and diagonal schedules."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Create memory block graph
    mem_graph = memory(Basis.Z)
    
    print("=== Generating Standard Schedule SVGs ===")
    # Compile with standard convention
    compiled_std = compile_block_graph(mem_graph)
    layer_tree_std = compiled_std.to_layer_tree()
    
    # Generate SVG - layers_to_svg returns a list of SVG strings, one per layer
    svg_list_std = layer_tree_std.layers_to_svg(k=k)
    
    layer_names = ['init', 'bulk_1', 'bulk_2', 'bulk_3', 'final']
    for i, svg_string in enumerate(svg_list_std):
        layer_name = layer_names[i] if i < len(layer_names) else f'layer_{i}'
        filepath = os.path.join(output_dir, f"memory_standard_{layer_name}.svg")
        with open(filepath, "w") as f:
            f.write(svg_string)
        print(f"Saved: {filepath}")
    
    print("\n=== Generating Diagonal Schedule SVGs ===")
    # Compile with diagonal convention
    diagonal_convention = create_diagonal_convention()
    compiled_diag = compile_block_graph(mem_graph, convention=diagonal_convention)
    layer_tree_diag = compiled_diag.to_layer_tree()
    
    # Generate SVG
    svg_list_diag = layer_tree_diag.layers_to_svg(k=k)
    
    for i, svg_string in enumerate(svg_list_diag):
        layer_name = layer_names[i] if i < len(layer_names) else f'layer_{i}'
        filepath = os.path.join(output_dir, f"memory_diagonal_{layer_name}.svg")
        with open(filepath, "w") as f:
            f.write(svg_string)
        print(f"Saved: {filepath}")
    
    print(f"\nDone! Generated {len(svg_list_std) + len(svg_list_diag)} SVG files in {output_dir}/")


if __name__ == "__main__":
    generate_memory_schedule_svgs(k=2)
