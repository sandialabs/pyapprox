"""Post-processing utilities for Galerkin finite element solutions.

Organized by physics type:

- ``elasticity``: Stress/strain recovery, von Mises stress
"""

from pyapprox.typing.pde.galerkin.postprocessing.elasticity import (
    von_mises_stress_2d,
    strain_from_displacement_2d,
    stress_from_strain_2d,
)

__all__ = [
    "von_mises_stress_2d",
    "strain_from_displacement_2d",
    "stress_from_strain_2d",
]
