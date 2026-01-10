#!/usr/bin/env python3
"""Generate SVG and PDF visualizations of circuit schedules for memory, X junction, and spatial hadamard experiments."""

import sys
sys.path.insert(0, 'venv/lib/python3.13/site-packages')

# Modify MEASUREMENT_SCHEDULE BEFORE importing any tqec modules
import tqec.plaquette.constants as constants
constants.MEASUREMENT_SCHEDULE = 8  # Allow schedule values up to 7

import importlib
import tqec.plaquette.rpng.translators.default as default_translator
importlib.reload(default_translator)

from tqec.gallery import memory
from tqec.computation.block_graph import BlockGraph
from tqec.computation.cube import ZXCube
from tqec.utils.position import Position3D
from tqec import compile_block_graph
from tqec.utils.enums import Basis

# Import spatial hadamard layer tree function
from spatial_hadamard_manual_construction import get_spatial_hadamard_layer_tree

# Import extended plaquette visualization
from tqec.visualisation.computation.plaquette.extended import (
    ExtendedPlaquetteDrawer,
    ExtendedPlaquetteType,
    ExtendedPlaquettePosition,
)

# Modify the hook error line coefficient for our visualizations
from tqec.visualisation.configuration import DrawerConfiguration

_original_drawer_config_init = DrawerConfiguration.__init__

def _patched_drawer_config_init(self, *args, **kwargs):
    if 'hook_error_line_lerp_coefficient' not in kwargs:
        kwargs['hook_error_line_lerp_coefficient'] = 0.7
    if 'font_size' not in kwargs:
        kwargs['font_size'] = 0.15
    _original_drawer_config_init(self, *args, **kwargs)

DrawerConfiguration.__init__ = _patched_drawer_config_init

# Patch the default arguments in the grid module functions
import tqec.visualisation.computation.plaquette.grid as grid_module

grid_module.plaquette_grid_svg_viewer.__defaults__ = (
    None, None, None, None, True, True, True, (0, 0), (), 
    DrawerConfiguration(),
    None,
)

grid_module.plaquette_grid_to_svg.__defaults__ = (
    True, True, True, (0, 0), (), 
    DrawerConfiguration(),
    None,
)

# Remove the "Moments: X -> Y" title text from SVGs
import svg
from tqec.visualisation.computation.tree import LayerVisualiser

def _empty_moment_text(self, start: int, end: int) -> svg.G:
    return svg.G()

LayerVisualiser.get_moment_text = _empty_moment_text

# Fix the hardcoded font_size in get_interaction_order_text
from tqec.visualisation.computation.plaquette.rpng import RPNGPlaquetteDrawer
from tqec.visualisation.computation.plaquette.base import lerp

def _patched_get_interaction_order_text(self, configuration: DrawerConfiguration = DrawerConfiguration()) -> list[svg.Text]:
    interaction_order_texts: list[svg.Text] = []
    for corner, rpng in zip(self._corners, self._rpngs):
        if rpng.n is None:
            continue
        text_position = lerp(self._center, corner, configuration.text_lerp_coefficient)
        interaction_order_texts.append(
            svg.Text(
                x=text_position.real,
                y=text_position.imag,
                fill="black",
                font_size=configuration.font_size,
                text_anchor="middle",
                dominant_baseline="central",
                text=str(rpng.n),
            )
        )
    return interaction_order_texts

RPNGPlaquetteDrawer.get_interaction_order_text = _patched_get_interaction_order_text

# Import the diagonal convention
from benchmark_memory import create_diagonal_convention


def create_memory_block_graph():
    """Create a block graph for the memory experiment."""
    return memory(Basis.Z)


def create_x_junction_block_graph():
    """Create a block graph for an X junction with central block and 4 neighbors."""
    graph = BlockGraph("X Junction")
    
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
    
    graph.add_pipe(central_pos, north_pos)
    graph.add_pipe(central_pos, south_pos)
    graph.add_pipe(central_pos, west_pos)
    graph.add_pipe(central_pos, east_pos)
    
    return graph


def create_l_junction_block_graph():
    """Create a block graph for an L junction with central block and 2 neighbors (+x and -y)."""
    graph = BlockGraph("L Junction")
    
    central_pos = Position3D(0, 1, 0)
    east_pos = Position3D(1, 1, 0)   # +x direction
    north_pos = Position3D(0, 0, 0)  # -y direction
    
    # Same cube types as X junction
    graph.add_cube(central_pos, ZXCube.from_str("XXZ"), "central")
    graph.add_cube(east_pos, ZXCube.from_str("ZXZ"), "east")
    graph.add_cube(north_pos, ZXCube.from_str("XZZ"), "north")
    
    graph.add_pipe(central_pos, east_pos)
    graph.add_pipe(central_pos, north_pos)
    
    return graph


def svg_to_pdf(svg_string: str, pdf_path: str, width: int = 600, height: int = 600):
    """Convert SVG string to PDF file using cairosvg."""
    import cairosvg
    import re
    
    svg_with_dims = re.sub(
        r'<svg\s+xmlns',
        f'<svg width="{width}" height="{height}" xmlns',
        svg_string
    )
    
    cairosvg.svg2pdf(bytestring=svg_with_dims.encode('utf-8'), write_to=pdf_path, dpi=72)


def generate_schedule_visualizations(
    block_graph,
    experiment_name: str,
    k: int = 2,
    layer_index: int = 1,
    output_dir: str = "benchmark_plots/svgs"
):
    """Generate PDF visualizations for standard and diagonal schedules.
    
    Args:
        block_graph: The block graph to compile
        experiment_name: Name prefix for output files (e.g., "memory", "x_junction")
        k: Scaling factor
        layer_index: Which layer to generate (default: 1 for bulk layer)
        output_dir: Output directory for files
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    diagonal_convention = create_diagonal_convention()
    
    for schedule_name, convention in [("standard", None), ("diagonal", diagonal_convention)]:
        print(f"=== Generating {experiment_name} {schedule_name} schedule ===")
        
        if convention is None:
            compiled = compile_block_graph(block_graph)
        else:
            compiled = compile_block_graph(block_graph, convention=convention)
        
        layer_tree = compiled.to_layer_tree()
        svg_list = layer_tree.layers_to_svg(k=k)
        
        if layer_index >= len(svg_list):
            print(f"Warning: layer_index {layer_index} >= {len(svg_list)} layers, using last layer")
            layer_index = len(svg_list) - 1
        
        svg_string = svg_list[layer_index]
        
        pdf_path = os.path.join(output_dir, f"{experiment_name}_{schedule_name}_layer_{layer_index}.pdf")
        svg_to_pdf(svg_string, pdf_path)
        print(f"Saved: {pdf_path}")


def create_custom_arm_shape(
    position: str,  # 'top' or 'bottom'
    basis: Basis,
    configuration: DrawerConfiguration,
    is_corner: bool = True,
) -> 'svg.Element':
    """Create a shape for the stretched stabilizer interface plaquettes.
    
    For corner position (x=0): trapezoid shape occupying right half
    - 'top' position: narrow at TOP, wider at BOTTOM (toward interface)
    - 'bottom' position: wider at TOP (at interface), narrow at BOTTOM
    
    For bulk positions (x>0): full-width rectangle
    
    For Y-axis interface, the plaquettes are:
    - Top (XZZ cube): 
      - Corner (x=0): plaquette 3 (X-basis) - only top-right at step 6
      - Bulk: plaquette 13 (Z-basis) at steps 2,3 or plaquette 14 (X-basis) at steps 5,6
    - Bottom (ZXX cube):
      - Corner (x=0): plaquette 1 (Z-basis) - only bottom-right at step 4
      - Bulk: plaquette 5 (X-basis) at steps 3,4 or plaquette 6 (Z-basis) at steps 4,5
    """
    import svg as svg_module
    from tqec.visualisation.computation.plaquette.base import SVGPlaquetteDrawer
    
    fill = SVGPlaquetteDrawer.get_colour(basis)
    stroke_color = configuration.stroke_color
    stroke_width = configuration.stroke_width
    
    elements = []
    
    # Get the actual schedule numbers based on position, basis, and whether it's a corner
    # These come from the custom coupling plaquette implementations
    if is_corner:
        # Corner plaquettes have only ONE non-shared data qubit (on the right)
        if position == 'top':
            # XZZ plaquette 3 (X-basis): top-right at step 6
            right_time = 6
            left_time = None  # No left data qubit at corner
        else:  # bottom
            # ZXX plaquette 1 (Z-basis): bottom-right at step 4
            right_time = 4
            left_time = None  # No left data qubit at corner
    else:
        # Bulk plaquettes have TWO non-shared data qubits
        if position == 'top':
            # XZZ cube plaquettes
            if basis == Basis.X:
                # Plaquette 14: top-left at step 5, top-right at step 6
                left_time, right_time = 5, 6
            else:  # Z-basis
                # Plaquette 13: top-right at step 2, top-left at step 3
                left_time, right_time = 3, 2
        else:  # bottom
            # ZXX cube plaquettes
            if basis == Basis.X:
                # Plaquette 5: bottom-left at step 3, bottom-right at step 4
                left_time, right_time = 3, 4
            else:  # Z-basis
                # Plaquette 6: bottom-right at step 4, bottom-left at step 5
                left_time, right_time = 5, 4
    
    # Use same text positioning as TQEC: lerp from center (0.5, 0.5) to corner with coefficient 0.8
    # This gives: 0.5 + 0.8 * (corner - 0.5) = 0.1 for corner=0, 0.9 for corner=1
    text_near_edge = 0.9  # 0.5 + 0.8 * 0.5
    text_far_edge = 0.1   # 0.5 - 0.8 * 0.5
    
    # Hook error line positioning (same lerp coefficient as text)
    hook_line_coef = configuration.hook_error_line_lerp_coefficient  # default 0.8
    
    if is_corner:
        # Trapezoid shape for corner (right half only)
        narrow_left = 0.7
        wide_left = 0.5
        right_edge = 1.0
        
        if position == 'top':
            # Upper half: narrow at TOP (y=0), wider at BOTTOM (y=1, at interface)
            path_d = f"M {narrow_left} 0 L {right_edge} 0 L {right_edge} 1 L {wide_left} 1 Z"
            elements.append(svg_module.Path(d=path_d, fill=fill, stroke=stroke_color, stroke_width=stroke_width))
            # Schedule number at top-right corner
            elements.append(svg_module.Text(
                x=text_near_edge, y=text_far_edge, fill="black", font_size=configuration.font_size,
                text_anchor="middle", dominant_baseline="central", text=str(right_time),
            ))
        else:  # bottom
            # Lower half: wider at TOP (y=0, at interface), narrow at BOTTOM (y=1)
            path_d = f"M {wide_left} 0 L {right_edge} 0 L {right_edge} 1 L {narrow_left} 1 Z"
            elements.append(svg_module.Path(d=path_d, fill=fill, stroke=stroke_color, stroke_width=stroke_width))
            # Schedule number at bottom-right corner
            elements.append(svg_module.Text(
                x=text_near_edge, y=text_near_edge, fill="black", font_size=configuration.font_size,
                text_anchor="middle", dominant_baseline="central", text=str(right_time),
            ))
    else:
        # Full-width rectangle for bulk positions
        left_edge = 0.0
        right_edge = 1.0
        center = 0.5
        
        # Calculate hook error line positions using lerp from center to corners
        hook_left_x = center + hook_line_coef * (left_edge - center)  # 0.1
        hook_right_x = center + hook_line_coef * (right_edge - center)  # 0.9
        
        if position == 'top':
            # Full rectangle
            path_d = f"M {left_edge} 0 L {right_edge} 0 L {right_edge} 1 L {left_edge} 1 Z"
            elements.append(svg_module.Path(d=path_d, fill=fill, stroke=stroke_color, stroke_width=stroke_width))
            
            # Hook error line between top corners (dashed)
            hook_y = center + hook_line_coef * (0 - center)  # 0.1
            elements.append(svg_module.Line(
                x1=hook_left_x, y1=hook_y, x2=hook_right_x, y2=hook_y,
                stroke=stroke_color, stroke_width=stroke_width,
                stroke_dasharray="0.05 0.03",
            ))
            
            # Schedule numbers at top corners (same positions as TQEC)
            elements.append(svg_module.Text(
                x=text_far_edge, y=text_far_edge, fill="black", font_size=configuration.font_size,
                text_anchor="middle", dominant_baseline="central", text=str(left_time),
            ))
            elements.append(svg_module.Text(
                x=text_near_edge, y=text_far_edge, fill="black", font_size=configuration.font_size,
                text_anchor="middle", dominant_baseline="central", text=str(right_time),
            ))
        else:  # bottom
            # Full rectangle
            path_d = f"M {left_edge} 0 L {right_edge} 0 L {right_edge} 1 L {left_edge} 1 Z"
            elements.append(svg_module.Path(d=path_d, fill=fill, stroke=stroke_color, stroke_width=stroke_width))
            
            # Hook error line between bottom corners (dashed)
            hook_y = center + hook_line_coef * (1 - center)  # 0.9
            elements.append(svg_module.Line(
                x1=hook_left_x, y1=hook_y, x2=hook_right_x, y2=hook_y,
                stroke=stroke_color, stroke_width=stroke_width,
                stroke_dasharray="0.05 0.03",
            ))
            
            # Schedule numbers at bottom corners (same positions as TQEC)
            elements.append(svg_module.Text(
                x=text_far_edge, y=text_near_edge, fill="black", font_size=configuration.font_size,
                text_anchor="middle", dominant_baseline="central", text=str(left_time),
            ))
            elements.append(svg_module.Text(
                x=text_near_edge, y=text_near_edge, fill="black", font_size=configuration.font_size,
                text_anchor="middle", dominant_baseline="central", text=str(right_time),
            ))
    
    return svg_module.G(elements=elements)


def create_custom_arm_shape_x_axis(
    position: str,  # 'left' or 'right'
    basis: Basis,
    configuration: DrawerConfiguration,
    is_corner: bool = True,
) -> 'svg.Element':
    """Create a shape for the X-axis stretched stabilizer interface plaquettes.
    
    For corner position (y=corner): trapezoid shape occupying bottom half
    - 'left' position: narrow at LEFT, wider at RIGHT (toward interface)
    - 'right' position: wider at LEFT (at interface), narrow at RIGHT
    
    For bulk positions: full-height rectangle
    
    For X-axis interface, the plaquettes are:
    - Left (ZXZ cube): 
      - Corner: plaquette 2 (X-basis) - bottom-left at step 3
      - Bulk: plaquette 11 (Z-basis) at steps 4,5 or plaquette 12 (X-basis) at steps 4,3
    - Right (XZX cube):
      - Corner: plaquette 1 (Z-basis) - bottom-right at step 2
      - Bulk: plaquette 7 (X-basis) at steps 5,6 or plaquette 8 (Z-basis) at steps 3,2
    """
    import svg as svg_module
    from tqec.visualisation.computation.plaquette.base import SVGPlaquetteDrawer
    
    fill = SVGPlaquetteDrawer.get_colour(basis)
    stroke_color = configuration.stroke_color
    stroke_width = configuration.stroke_width
    
    elements = []
    
    # Use same text positioning as TQEC
    text_near_edge = 0.9
    text_far_edge = 0.1
    
    # Hook error line positioning
    hook_line_coef = configuration.hook_error_line_lerp_coefficient
    
    # Get the actual schedule numbers based on position, basis, and whether it's a corner
    if is_corner:
        # Corner plaquettes have only ONE non-shared data qubit (at bottom)
        if position == 'left':
            # ZXZ plaquette 2 (X-basis): bottom-left at step 3
            bottom_time = 3
            top_time = None
        else:  # right
            # XZX plaquette 1 (Z-basis): bottom-right at step 2
            bottom_time = 2
            top_time = None
    else:
        # Bulk plaquettes have TWO non-shared data qubits (top and bottom)
        if position == 'left':
            # ZXZ cube plaquettes
            if basis == Basis.X:
                # Plaquette 12: top-left at step 4, bottom-left at step 3
                top_time, bottom_time = 4, 3
            else:  # Z-basis
                # Plaquette 11: top-left at step 4, bottom-left at step 5
                top_time, bottom_time = 4, 5
        else:  # right
            # XZX cube plaquettes
            if basis == Basis.X:
                # Plaquette 7: top-right at step 5, bottom-right at step 6
                top_time, bottom_time = 5, 6
            else:  # Z-basis
                # Plaquette 8: top-right at step 3, bottom-right at step 2
                top_time, bottom_time = 3, 2
    
    if is_corner:
        # Trapezoid shape for corner (bottom half only, similar to Y-axis but rotated)
        # The shape should have parallel edges at the interface (vertical interface)
        # narrow_y is the y-coordinate of the narrow top edge (away from interface)
        # wide_y is the y-coordinate of the wide top edge (toward interface)
        narrow_y = 0.7  # narrow tip on the outer side
        wide_y = 0.5    # wide base toward interface
        bottom_edge = 1.0
        
        if position == 'left':
            # Left half: narrow at top-left (x=0), wider at top-right (x=1, toward interface)
            # Bottom edge is full width
            path_d = f"M 0 {narrow_y} L 0 {bottom_edge} L 1 {bottom_edge} L 1 {wide_y} Z"
            elements.append(svg_module.Path(d=path_d, fill=fill, stroke=stroke_color, stroke_width=stroke_width))
            # Schedule number at bottom-left corner (where the data qubit is)
            elements.append(svg_module.Text(
                x=text_far_edge, y=text_near_edge, fill="black", font_size=configuration.font_size,
                text_anchor="middle", dominant_baseline="central", text=str(bottom_time),
            ))
        else:  # right
            # Right half: wider at top-left (x=0, toward interface), narrow at top-right (x=1)
            # Bottom edge is full width
            path_d = f"M 0 {wide_y} L 0 {bottom_edge} L 1 {bottom_edge} L 1 {narrow_y} Z"
            elements.append(svg_module.Path(d=path_d, fill=fill, stroke=stroke_color, stroke_width=stroke_width))
            # Schedule number at bottom-right corner (where the data qubit is)
            elements.append(svg_module.Text(
                x=text_near_edge, y=text_near_edge, fill="black", font_size=configuration.font_size,
                text_anchor="middle", dominant_baseline="central", text=str(bottom_time),
            ))
    else:
        # Full-height rectangle for bulk positions
        top_edge = 0.0
        bottom_edge = 1.0
        center = 0.5
        
        # Calculate hook error line positions
        hook_top_y = center + hook_line_coef * (top_edge - center)
        hook_bottom_y = center + hook_line_coef * (bottom_edge - center)
        
        if position == 'left':
            # Full rectangle
            path_d = f"M 0 0 L 1 0 L 1 1 L 0 1 Z"
            elements.append(svg_module.Path(d=path_d, fill=fill, stroke=stroke_color, stroke_width=stroke_width))
            
            # Hook error line between left corners (dashed, vertical)
            hook_x = center + hook_line_coef * (0 - center)  # 0.1
            elements.append(svg_module.Line(
                x1=hook_x, y1=hook_top_y, x2=hook_x, y2=hook_bottom_y,
                stroke=stroke_color, stroke_width=stroke_width,
                stroke_dasharray="0.05 0.03",
            ))
            
            # Schedule numbers at left corners
            elements.append(svg_module.Text(
                x=text_far_edge, y=text_far_edge, fill="black", font_size=configuration.font_size,
                text_anchor="middle", dominant_baseline="central", text=str(top_time),
            ))
            elements.append(svg_module.Text(
                x=text_far_edge, y=text_near_edge, fill="black", font_size=configuration.font_size,
                text_anchor="middle", dominant_baseline="central", text=str(bottom_time),
            ))
        else:  # right
            # Full rectangle
            path_d = f"M 0 0 L 1 0 L 1 1 L 0 1 Z"
            elements.append(svg_module.Path(d=path_d, fill=fill, stroke=stroke_color, stroke_width=stroke_width))
            
            # Hook error line between right corners (dashed, vertical)
            hook_x = center + hook_line_coef * (1 - center)  # 0.9
            elements.append(svg_module.Line(
                x1=hook_x, y1=hook_top_y, x2=hook_x, y2=hook_bottom_y,
                stroke=stroke_color, stroke_width=stroke_width,
                stroke_dasharray="0.05 0.03",
            ))
            
            # Schedule numbers at right corners
            elements.append(svg_module.Text(
                x=text_near_edge, y=text_far_edge, fill="black", font_size=configuration.font_size,
                text_anchor="middle", dominant_baseline="central", text=str(top_time),
            ))
            elements.append(svg_module.Text(
                x=text_near_edge, y=text_near_edge, fill="black", font_size=configuration.font_size,
                text_anchor="middle", dominant_baseline="central", text=str(bottom_time),
            ))
    
    return svg_module.G(elements=elements)


def create_extended_plaquette_svg_overlay(
    axis: str,
    k: int,
    configuration: DrawerConfiguration = None
) -> str:
    """Create SVG elements for extended/stretched stabilizers at the interface.
    
    For Y-axis: Interface is between XZZ (top, rows 0-5) and ZXX (bottom, rows 6-11)
                Interface row: y=5 (bottom of XZZ) and y=6 (top of ZXX)
    For X-axis: Interface is between ZXZ (left, cols 0-5) and XZX (right, cols 6-11)
                Interface col: x=5 (right of ZXZ) and x=6 (left of XZX)
    
    Args:
        axis: 'x' or 'y' - direction of the two-cube layout
        k: Scaling factor
        configuration: Drawing configuration
        
    Returns:
        SVG group element with extended plaquette overlays
    """
    import svg as svg_module
    
    if configuration is None:
        configuration = DrawerConfiguration()
    
    elements = []
    
    # Each cube has 2k+2 rows/cols in the visualization (including borders)
    cube_viz_size = 2 * k + 2  # 6 for k=2
    
    if axis == 'y':
        # Y-axis: Interface is horizontal between XZZ (top) and ZXX (bottom)
        # Interface at y=5 (last row of first cube) and y=6 (first row of second cube)
        interface_y_top = cube_viz_size - 1  # y=5 for k=2
        interface_y_bottom = cube_viz_size   # y=6 for k=2
        
        # User-specified pattern (from left to right):
        # x=1: narrow (right half), top=X (red), bottom=Z (blue)
        # x=2: full width, top=Z (blue), bottom=X (red)
        # x=3: full width, top=X (red), bottom=Z (blue)
        # x=4: full width, top=Z (blue), bottom=X (red)
        # x=5 (last): nothing
        
        # Stretched stabilizer schedule (from custom coupling plaquettes)
        z_schedule = (1, 3, 4, 2)  # Z-basis schedule
        x_schedule = (7, 5, 4, 6)  # X-basis schedule
        
        # Draw extended plaquettes for interface positions (x=0 to x=cube_viz_size-2)
        # Skip the last position (x=cube_viz_size-1)
        for x_pos in range(0, cube_viz_size - 1):
            # Determine plaquette type based on position
            use_custom_arm = (x_pos == 0)  # Use custom arm shape for first position
            
            if not use_custom_arm:
                # Other positions: full width
                plaq_type_top = ExtendedPlaquetteType.BULK
                plaq_type_bottom = ExtendedPlaquetteType.BULK
            
            # Determine basis for top (y=5) and bottom (y=6)
            # Note: In SVG, y increases downward, so y=5 is visually ABOVE y=6
            # User wants (visually from top to bottom):
            #   x=0: visual top (y=5)=X (red), visual bottom (y=6)=Z (blue) - narrow
            #   x=1: visual top (y=5)=Z (blue), visual bottom (y=6)=X (red) - full
            #   x=2: visual top (y=5)=X (red), visual bottom (y=6)=Z (blue) - full
            #   x=3: visual top (y=5)=Z (blue), visual bottom (y=6)=X (red) - full
            # So: even x -> top=X, bottom=Z; odd x -> top=Z, bottom=X
            if x_pos % 2 == 0:
                top_basis = Basis.X  # red (visually on top)
                bottom_basis = Basis.Z  # blue (visually on bottom)
            else:
                top_basis = Basis.Z  # blue (visually on top)
                bottom_basis = Basis.X  # red (visually on bottom)
            
            top_schedule = x_schedule if top_basis == Basis.X else z_schedule
            bottom_schedule = x_schedule if bottom_basis == Basis.X else z_schedule
            
            # Use custom arm shapes for all interface positions
            # x=0 is corner, x>0 are bulk positions
            if use_custom_arm:
                # Corner position (x=0): XZZ plaquette 3 (X-basis) and ZXX plaquette 1 (Z-basis)
                # These are fixed bases for the corner position
                top_svg = create_custom_arm_shape(
                    position='top',
                    basis=Basis.X,  # XZZ plaquette 3 is always X-basis
                    configuration=configuration,
                    is_corner=True,
                )
                elements.append(svg_module.G(
                    elements=[top_svg],
                    transform=[svg_module.Translate(x=float(x_pos), y=float(interface_y_top))]
                ))
                
                bottom_svg = create_custom_arm_shape(
                    position='bottom',
                    basis=Basis.Z,  # ZXX plaquette 1 is always Z-basis
                    configuration=configuration,
                    is_corner=True,
                )
                elements.append(svg_module.G(
                    elements=[bottom_svg],
                    transform=[svg_module.Translate(x=float(x_pos), y=float(interface_y_bottom))]
                ))
            else:
                # Bulk positions (x>0): Use custom arm shapes with correct interface schedules
                # Top: XZZ plaquettes 13 (Z-basis) or 14 (X-basis)
                # Bottom: ZXX plaquettes 5 (X-basis) or 6 (Z-basis)
                top_svg = create_custom_arm_shape(
                    position='top',
                    basis=top_basis,
                    configuration=configuration,
                    is_corner=False,
                )
                elements.append(svg_module.G(
                    elements=[top_svg],
                    transform=[svg_module.Translate(x=float(x_pos), y=float(interface_y_top))]
                ))
                
                bottom_svg = create_custom_arm_shape(
                    position='bottom',
                    basis=bottom_basis,
                    configuration=configuration,
                    is_corner=False,
                )
                elements.append(svg_module.G(
                    elements=[bottom_svg],
                    transform=[svg_module.Translate(x=float(x_pos), y=float(interface_y_bottom))]
                ))
    
    elif axis == 'x':
        # X-axis: Interface is vertical between ZXZ (left) and XZX (right)
        # Interface at x=5 (last col of first cube) and x=6 (first col of second cube)
        interface_x_left = cube_viz_size - 1   # x=5 for k=2
        interface_x_right = cube_viz_size      # x=6 for k=2
        
        # Draw extended plaquettes for interface positions (y=0 to y=cube_viz_size-2)
        # Stop one position before the end (similar to Y-axis)
        for y_pos in range(0, cube_viz_size - 1):
            # y=0 is corner position
            use_custom_arm = (y_pos == 0)
            
            # Determine basis for left and right based on checkerboard pattern
            # y=0: corner (fixed bases), y>0: alternating
            if y_pos % 2 == 0:
                left_basis = Basis.X  # ZXZ plaquette 12 or 2
                right_basis = Basis.Z  # XZX plaquette 8 or 1
            else:
                left_basis = Basis.Z  # ZXZ plaquette 11
                right_basis = Basis.X  # XZX plaquette 7
            
            if use_custom_arm:
                # Corner position (y=0): ZXZ plaquette 2 (X-basis) and XZX plaquette 1 (Z-basis)
                left_svg = create_custom_arm_shape_x_axis(
                    position='left',
                    basis=Basis.X,  # ZXZ plaquette 2 is always X-basis
                    configuration=configuration,
                    is_corner=True,
                )
                elements.append(svg_module.G(
                    elements=[left_svg],
                    transform=[svg_module.Translate(x=float(interface_x_left), y=float(y_pos))]
                ))
                
                right_svg = create_custom_arm_shape_x_axis(
                    position='right',
                    basis=Basis.Z,  # XZX plaquette 1 is always Z-basis
                    configuration=configuration,
                    is_corner=True,
                )
                elements.append(svg_module.G(
                    elements=[right_svg],
                    transform=[svg_module.Translate(x=float(interface_x_right), y=float(y_pos))]
                ))
            else:
                # Bulk positions: use custom arm shapes with correct interface schedules
                left_svg = create_custom_arm_shape_x_axis(
                    position='left',
                    basis=left_basis,
                    configuration=configuration,
                    is_corner=False,
                )
                elements.append(svg_module.G(
                    elements=[left_svg],
                    transform=[svg_module.Translate(x=float(interface_x_left), y=float(y_pos))]
                ))
                
                right_svg = create_custom_arm_shape_x_axis(
                    position='right',
                    basis=right_basis,
                    configuration=configuration,
                    is_corner=False,
                )
                elements.append(svg_module.G(
                    elements=[right_svg],
                    transform=[svg_module.Translate(x=float(interface_x_right), y=float(y_pos))]
                ))
    
    # Combine all elements
    return svg_module.G(id="extended_plaquettes", elements=elements)


def generate_spatial_hadamard_visualizations(
    axis: str = 'x',
    k: int = 2,
    layer_index: int = 1,
    output_dir: str = "benchmark_plots/svgs"
):
    """Generate PDF visualization for spatial hadamard with extended plaquette overlays.
    
    Args:
        axis: 'x' or 'y' - direction of the two-cube layout
        k: Scaling factor
        layer_index: Which layer to generate (default: 1 for bulk layer)
        output_dir: Output directory for files
    """
    import os
    import re
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== Generating spatial_hadamard_{axis} schedule ===")
    
    layer_tree = get_spatial_hadamard_layer_tree(axis=axis)
    svg_list = layer_tree.layers_to_svg(k=k)
    
    if layer_index >= len(svg_list):
        print(f"Warning: layer_index {layer_index} >= {len(svg_list)} layers, using last layer")
        layer_index = len(svg_list) - 1
    
    svg_string = svg_list[layer_index]
    
    # Create extended plaquette overlay
    extended_overlay = create_extended_plaquette_svg_overlay(axis, k)
    
    # Insert the overlay into the SVG
    # Find the closing </svg> tag and insert the overlay before it
    overlay_str = str(extended_overlay)
    
    # The overlay needs to be scaled and positioned to match the main SVG
    # The main SVG uses a viewBox and transforms, so we need to match that
    # For now, let's just add it as a group that will be positioned by the SVG viewer
    
    # Insert overlay before closing </svg> tag
    svg_string = svg_string.replace('</svg>', f'{overlay_str}</svg>')
    
    pdf_path = os.path.join(output_dir, f"spatial_hadamard_{axis}_layer_{layer_index}.pdf")
    svg_to_pdf(svg_string, pdf_path)
    print(f"Saved: {pdf_path}")


def main():
    import os
    output_dir = "benchmark_plots/svgs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate memory experiment visualizations
    print("\n" + "=" * 50)
    print("MEMORY EXPERIMENT")
    print("=" * 50)
    memory_graph = create_memory_block_graph()
    generate_schedule_visualizations(memory_graph, "memory", k=2, layer_index=1, output_dir=output_dir)
    
    # Generate L junction visualizations
    print("\n" + "=" * 50)
    print("L JUNCTION")
    print("=" * 50)
    l_junction_graph = create_l_junction_block_graph()
    generate_schedule_visualizations(l_junction_graph, "l_junction", k=2, layer_index=1, output_dir=output_dir)
    
    # Generate X junction visualizations
    print("\n" + "=" * 50)
    print("X JUNCTION")
    print("=" * 50)
    x_junction_graph = create_x_junction_block_graph()
    generate_schedule_visualizations(x_junction_graph, "x_junction", k=2, layer_index=1, output_dir=output_dir)
    
    # Generate spatial hadamard visualizations
    print("\n" + "=" * 50)
    print("SPATIAL HADAMARD (X-axis)")
    print("=" * 50)
    generate_spatial_hadamard_visualizations(axis='x', k=2, layer_index=1, output_dir=output_dir)
    
    print("\n" + "=" * 50)
    print("SPATIAL HADAMARD (Y-axis)")
    print("=" * 50)
    generate_spatial_hadamard_visualizations(axis='y', k=2, layer_index=1, output_dir=output_dir)
    
    print("\n" + "=" * 50)
    print("Done!")
    print("=" * 50)


if __name__ == "__main__":
    main()
