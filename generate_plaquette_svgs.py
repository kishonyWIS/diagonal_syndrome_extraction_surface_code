#!/usr/bin/env python3
"""Generate PDF plaquette drawings with schedule numbers using TQEC visualization code."""

import sys
sys.path.insert(0, 'venv/lib/python3.13/site-packages')

import os
import cairosvg
import svg

# Set up TQEC constants before imports
import tqec.plaquette.constants as constants
constants.MEASUREMENT_SCHEDULE = 8

import importlib
import tqec.plaquette.rpng.translators.default as default_translator
importlib.reload(default_translator)

from tqec.visualisation.computation.plaquette.rpng import RPNGPlaquetteDrawer
from tqec.visualisation.computation.plaquette.base import SVGPlaquetteDrawer, lerp
from tqec.visualisation.configuration import DrawerConfiguration
from tqec.plaquette.rpng import RPNGDescription
from tqec.utils.enums import Basis

# Patch get_interaction_order_text to actually use the font_size from configuration
# (TQEC has a hardcoded font size by default)
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


def create_schedule_order_line(schedule: list[int], config: DrawerConfiguration) -> svg.G:
    """Create a polyline connecting corners in schedule order with an arrow at the end.
    
    Args:
        schedule: List of 4 schedule numbers [tl, tr, bl, br]
        config: Drawing configuration
        
    Returns:
        SVG group containing the polyline with arrow marker
    """
    # Corner positions (in plaquette coordinates 0-1)
    # Order: [tl, tr, bl, br]
    corner_positions = [
        complex(0, 0),      # top-left
        complex(1, 0),      # top-right
        complex(0, 1),      # bottom-left
        complex(1, 1),      # bottom-right
    ]
    
    center = complex(0.5, 0.5)
    lerp_coef = config.hook_error_line_lerp_coefficient
    
    # Get lerped positions (same as hook error line positions)
    lerped_positions = [lerp(center, corner, lerp_coef) for corner in corner_positions]
    
    # Create list of (schedule_time, position, corner_index) and sort by schedule time
    schedule_with_pos = []
    for i, (sched_time, pos) in enumerate(zip(schedule, lerped_positions)):
        schedule_with_pos.append((sched_time, pos, i))
    
    # Sort by schedule time
    schedule_with_pos.sort(key=lambda x: x[0])
    
    # Extract the ordered positions
    ordered_positions = [item[1] for item in schedule_with_pos]
    
    # Shorten the final edge so the arrow doesn't block the last number
    # Move the last point 30% back along the final edge
    if len(ordered_positions) >= 2:
        last_pos = ordered_positions[-1]
        second_last_pos = ordered_positions[-2]
        # Interpolate: new_last = second_last + 0.7 * (last - second_last)
        shortened_last = second_last_pos + 0.95 * (last_pos - second_last_pos)
        ordered_positions[-1] = shortened_last
    
    # Create polyline points
    points = [f"{pos.real},{pos.imag}" for pos in ordered_positions]
    points_str = " ".join(points)
    
    # Create arrow marker definition
    arrow_marker = svg.Marker(
        id="arrow",
        viewBox="0 0 10 10",
        refX=5,
        refY=5,
        markerWidth=6,
        markerHeight=6,
        orient="auto-start-reverse",
        elements=[
            svg.Path(
                d="M 0 0 L 10 5 L 0 10 z",
                fill="black",
            )
        ]
    )
    
    # Create defs element for the marker
    defs = svg.Defs(elements=[arrow_marker])
    
    # Create the polyline with arrow marker at the end
    polyline = svg.Polyline(
        points=points_str,
        fill="none",
        stroke="black",
        stroke_width=config.stroke_width,
        marker_end="url(#arrow)",
    )
    
    return svg.G(elements=[defs, polyline])


def create_plaquette_pdf(
    filename: str,
    basis: str,
    schedule: list[int],
    hook_orientation: str = "horizontal",
    size: int = 100,
):
    """Create a PDF plaquette with schedule numbers and schedule order line using TQEC style.
    
    Args:
        filename: Output PDF filename
        basis: 'X' or 'Z' - determines color
        schedule: List of 4 schedule numbers [tl, tr, bl, br]
        hook_orientation: 'horizontal', 'vertical', or 'diagonal' - for reference only
        size: Size of the square in pixels
    """
    # Create RPNG description string: "-{basis}{schedule}-" for each corner
    # Format: "tl tr bl br" where each is like "-x1-" or "-z3-"
    b = basis.lower()
    rpng_str = f"-{b}{schedule[0]}- -{b}{schedule[1]}- -{b}{schedule[2]}- -{b}{schedule[3]}-"
    rpng_desc = RPNGDescription.from_string(rpng_str)
    
    # Create the drawer
    drawer = RPNGPlaquetteDrawer(rpng_desc)
    
    # Configuration with larger font for standalone plaquettes
    config = DrawerConfiguration(
        font_size=0.2,
        hook_error_line_lerp_coefficient=0.65,
        text_lerp_coefficient=0.8,
        stroke_width=0.02,
    )
    
    # Draw the plaquette WITHOUT the default hook error line
    plaquette_svg = drawer.draw(
        id=f"plaquette_{basis}",
        show_interaction_order=True,
        show_hook_errors=False,  # Disable default hook error line
        show_data_qubit_reset_measurements=False,
        configuration=config,
    )
    
    # Create custom schedule order line with arrow
    schedule_line = create_schedule_order_line(schedule, config)
    
    # Create a complete SVG with viewBox
    margin = 0.1
    viewbox = f"{-margin} {-margin} {1 + 2*margin} {1 + 2*margin}"
    
    full_svg = svg.SVG(
        xmlns="http://www.w3.org/2000/svg",
        width=size,
        height=size,
        viewBox=viewbox,
        elements=[plaquette_svg, schedule_line],
    )
    
    svg_string = str(full_svg)
    
    # Convert to PDF
    cairosvg.svg2pdf(bytestring=svg_string.encode('utf-8'), write_to=filename)
    print(f"Created: {filename}")


def main():
    # Create output directory
    output_dir = "plaquette_svgs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define plaquettes: (name, basis, schedule [tl, tr, bl, br], hook_orientation)
    # The hook orientation is determined by which pair of qubits is touched first
    # Vertical: first two are tl,bl (left column) -> vertical hook line
    # Horizontal: first two are tl,tr (top row) -> horizontal hook line
    plaquettes = [
        # Original schedules
        ("original_x_vertical", "X", [1, 4, 3, 5], "vertical"),      # tl(1), bl(3), tr(4), br(5) -> left column first
        ("original_x_horizontal", "X", [1, 2, 3, 5], "horizontal"),  # tl(1), tr(2), bl(3), br(5) -> top row first
        ("original_z_vertical", "Z", [1, 4, 3, 5], "vertical"),
        ("original_z_horizontal", "Z", [1, 2, 3, 5], "horizontal"),
        # Diagonal schedules
        ("diagonal_x", "X", [7, 5, 4, 6], "diagonal"),  # bl(4), tr(5), br(6), tl(7)
        ("diagonal_z", "Z", [1, 3, 4, 2], "diagonal"),  # tl(1), br(2), tr(3), bl(4)
    ]
    
    for name, basis, schedule, hook_orient in plaquettes:
        filename = os.path.join(output_dir, f"{name}.pdf")
        create_plaquette_pdf(filename, basis, schedule, hook_orient)
    
    print(f"\nAll PDFs created in {output_dir}/")


if __name__ == "__main__":
    main()
