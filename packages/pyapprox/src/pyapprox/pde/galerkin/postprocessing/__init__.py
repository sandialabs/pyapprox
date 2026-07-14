"""Post-processing utilities for Galerkin finite element solutions.

Organized by physics type:

- ``elasticity``: Stress/strain recovery, von Mises stress
"""

from pyapprox.pde.galerkin.postprocessing.elasticity import (
    integrate,
    strain_from_displacement,
    stress_from_strain,
    von_mises_stress,
)

__all__ = [
    "integrate",
    "strain_from_displacement",
    "stress_from_strain",
    "von_mises_stress",
]
