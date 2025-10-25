#!/usr/bin/env python3
"""Custom diagonal plaquette configurations for memory experiments."""

from tqec.compile.specs.library.generators.fixed_bulk import FixedBulkConventionGenerator
from tqec.plaquette.rpng.translators.default import DefaultRPNGTranslator
from tqec.plaquette.compilation.base import PlaquetteCompiler
from tqec.plaquette.compilation.passes.scheduling import ChangeSchedulePass
from tqec.plaquette.compilation.passes.sort_targets import SortTargetsPass
from tqec.plaquette.rpng import RPNGDescription
from tqec.utils.enums import Basis, Orientation
from tqec.plaquette.enums import PlaquetteOrientation
from tqec.compile.specs.library.generators.utils import PlaquetteMapper
from tqec.utils.frozendefaultdict import FrozenDefaultDict
from collections import defaultdict
from typing import Literal

class DiagonalPlaquetteGenerator(FixedBulkConventionGenerator):
    """Custom generator with diagonal plaquette configurations."""
    
    def __init__(self, translator=None, compiler=None):
        if translator is None:
            translator = DefaultRPNGTranslator()
        if compiler is None:
            # Create custom Identity compiler that handles schedule 7 but keeps original basis
            compiler = PlaquetteCompiler(
                "CustomIdentity",
                [
                    # Compact schedule map: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
                    # No gaps - direct mapping to avoid idle moments
                    ChangeSchedulePass({i: i for i in range(8)}),
                    SortTargetsPass(),
                ],
                mergeable_instructions_modifier=lambda x: x | frozenset(["H"]),
            )
        super().__init__(translator, compiler)
    
    def get_diagonal_bulk_rpng_descriptions(
        self,
        reset: Basis | None = None,
        measurement: Basis | None = None,
        reset_and_measured_indices: tuple[Literal[0, 1, 2, 3], ...] = (0, 1, 2, 3),
    ) -> dict[Basis, dict[Orientation, RPNGDescription]]:
        """Get diagonal plaquettes with custom schedules.
        
        Args:
            reset: basis of the reset operation performed on data-qubits
            measurement: basis of the measurement operation performed on data-qubits
            reset_and_measured_indices: data-qubit indices that should be impacted
            
        Returns:
            a mapping with 4 plaquettes: X and Z basis, both orientations
        """
        # _r/_m: reset/measurement basis applied to each data-qubit
        _r = reset.value.lower() if reset is not None else "-"
        _m = measurement.value.lower() if measurement is not None else "-"
        
        # rs/ms: resets/measurements basis applied for each data-qubit
        rs = [_r if i in reset_and_measured_indices else "-" for i in range(4)]
        ms = [_m if i in reset_and_measured_indices else "-" for i in range(4)]
        
        # Use the original diagonal schedules with qubit index 6
        # Original: X-vertical=(1,4,3,5), X-horizontal=(1,2,3,5), Z-vertical=(1,4,3,5), Z-horizontal=(1,2,3,5)
        # Diagonal: X-vertical=(1,4,3,2), X-horizontal=(1,4,3,2), Z-vertical=(6,4,3,5), Z-horizontal=(6,4,3,5)
        return {
            Basis.X: {
                Orientation.VERTICAL: RPNGDescription.from_string(
                    " ".join(f"{r}x{s}{m}" for r, s, m in zip(rs, [1, 4, 3, 2], ms))
                ),
                Orientation.HORIZONTAL: RPNGDescription.from_string(
                    " ".join(f"{r}x{s}{m}" for r, s, m in zip(rs, [1, 4, 3, 2], ms))
                ),
            },
            Basis.Z: {
                Orientation.VERTICAL: RPNGDescription.from_string(
                    " ".join(f"{r}z{s}{m}" for r, s, m in zip(rs, [6, 4, 3, 5], ms))
                ),
                Orientation.HORIZONTAL: RPNGDescription.from_string(
                    " ".join(f"{r}z{s}{m}" for r, s, m in zip(rs, [6, 4, 3, 5], ms))
                ),
            },
        }
    
    def get_diagonal_2_body_rpng_descriptions(
        self,
    ) -> dict[Basis, dict[PlaquetteOrientation, RPNGDescription]]:
        """Get diagonal 2-body boundary plaquettes.
        
        These are derived from the diagonal bulk plaquettes by omitting qubits.
        """
        return {
            Basis.X: {
                # Derived from "-x1- -x4- -x3- -x2-"
                PlaquetteOrientation.DOWN: RPNGDescription.from_string("-x1- -x4- ---- ----"),
                PlaquetteOrientation.LEFT: RPNGDescription.from_string("---- -x4- ---- -x2-"),
                PlaquetteOrientation.UP: RPNGDescription.from_string("---- ---- -x3- -x2-"),
                PlaquetteOrientation.RIGHT: RPNGDescription.from_string("-x1- ---- -x3- ----"),
            },
            Basis.Z: {
                # Derived from "-z6- -z4- -z3- -z5-" (back to original with z6)
                PlaquetteOrientation.DOWN: RPNGDescription.from_string("-z6- -z4- ---- ----"),
                PlaquetteOrientation.LEFT: RPNGDescription.from_string("---- -z4- ---- -z5-"),
                PlaquetteOrientation.UP: RPNGDescription.from_string("---- ---- -z3- -z5-"),
                PlaquetteOrientation.RIGHT: RPNGDescription.from_string("-z6- ---- -z3- ----"),
            },
        }
    
    def get_memory_qubit_plaquettes(
        self,
        z_orientation: Orientation = Orientation.HORIZONTAL,
        reset: Basis | None = None,
        measurement: Basis | None = None,
    ):
        """Return the plaquettes needed to implement a standard memory operation on a logical qubit."""
        return self._mapper(self.get_memory_qubit_rpng_descriptions)(
            z_orientation, reset, measurement
        )

    def get_spatial_cube_qubit_raw_template(self):
        """Return the template instance needed to implement a spatial cube qubit."""
        from tqec.templates.qubit import QubitTemplate
        return QubitTemplate()

    def get_spatial_cube_qubit_plaquettes(
        self,
        x_basis: Basis,
        spatial_arms,
        reset: Basis | None = None,
        measurement: Basis | None = None,
    ):
        """Return the plaquettes needed to implement a spatial cube qubit."""
        return self._mapper(self.get_spatial_cube_qubit_rpng_descriptions)(
            x_basis, spatial_arms, reset, measurement
        )

    def get_spatial_cube_qubit_rpng_descriptions(
        self,
        x_basis: Basis,
        spatial_arms,
        reset: Basis | None = None,
        measurement: Basis | None = None,
    ):
        """Return RPNG descriptions for spatial cube qubit plaquettes."""
        # For now, use the same as memory qubit
        return self.get_memory_qubit_rpng_descriptions(reset, measurement)
    
    def get_memory_qubit_rpng_descriptions(
        self,
        z_orientation: Orientation = Orientation.HORIZONTAL,
        reset: Basis | None = None,
        measurement: Basis | None = None,
    ):
        """Override to use diagonal plaquettes."""
        # Border plaquette indices
        up, down, left, right = (
            (6, 13, 7, 12) if z_orientation == Orientation.VERTICAL else (5, 14, 8, 11)
        )
        # Basis for top/bottom and left/right boundary plaquettes
        hbasis = Basis.Z if z_orientation == Orientation.HORIZONTAL else Basis.X
        vbasis = hbasis.flipped()
        # Hook errors orientations
        zhook = z_orientation.flip()
        xhook = zhook.flip()
        
        # Get diagonal plaquette descriptions
        bulk_descriptions = self.get_diagonal_bulk_rpng_descriptions(reset, measurement)
        two_body_descriptions = self.get_diagonal_2_body_rpng_descriptions()
        
        # Return a FrozenDefaultDict like the original
        return FrozenDefaultDict(
            {
                up: two_body_descriptions[vbasis][PlaquetteOrientation.UP],
                left: two_body_descriptions[hbasis][PlaquetteOrientation.LEFT],
                # Bulk - using diagonal configurations
                9: bulk_descriptions[Basis.Z][zhook],
                10: bulk_descriptions[Basis.X][xhook],
                right: two_body_descriptions[hbasis][PlaquetteOrientation.RIGHT],
                down: two_body_descriptions[vbasis][PlaquetteOrientation.DOWN],
            },
            default_value=RPNGDescription.empty(),
        )

def demonstrate_diagonal_plaquettes():
    """Demonstrate the new diagonal plaquette configurations."""
    
    print("=== Diagonal Plaquette Configurations ===")
    print()
    
    generator = DiagonalPlaquetteGenerator()
    
    # Show bulk plaquettes
    print("Diagonal Bulk Plaquettes:")
    bulk_descriptions = generator.get_diagonal_bulk_rpng_descriptions()
    
    print("X-basis diagonal plaquettes:")
    print("  VERTICAL orientation:", bulk_descriptions[Basis.X][Orientation.VERTICAL])
    print("  HORIZONTAL orientation:", bulk_descriptions[Basis.X][Orientation.HORIZONTAL])
    print()
    
    print("Z-basis diagonal plaquettes:")
    print("  VERTICAL orientation:", bulk_descriptions[Basis.Z][Orientation.VERTICAL])
    print("  HORIZONTAL orientation:", bulk_descriptions[Basis.Z][Orientation.HORIZONTAL])
    print()
    
    # Show boundary plaquettes
    print("Diagonal Boundary Plaquettes:")
    two_body_descriptions = generator.get_diagonal_2_body_rpng_descriptions()
    
    print("X-basis diagonal boundary plaquettes:")
    for orientation, desc in two_body_descriptions[Basis.X].items():
        print(f"  {orientation}: {desc}")
    print()
    
    print("Z-basis diagonal boundary plaquettes:")
    for orientation, desc in two_body_descriptions[Basis.Z].items():
        print(f"  {orientation}: {desc}")
    print()
    
    # Show comparison with original
    print("=== Comparison with Original ===")
    print()
    print("Original X-basis bulk: \"-x1- -x2- -x3- -x5-\"")
    print("New diagonal X-basis: \"-x1- -x4- -x3- -x2-\"")
    print()
    print("Original Z-basis bulk: \"-z1- -z2- -z3- -z5-\"")
    print("New diagonal Z-basis: \"-z6- -z4- -z3- -z5-\"")
    print()
    print("Key differences:")
    print("- X-basis: Changed schedule from [1,2,3,5] to [1,4,3,2]")
    print("- Z-basis: Changed schedule from [1,2,3,5] to [6,4,3,5]")

if __name__ == "__main__":
    demonstrate_diagonal_plaquettes()
