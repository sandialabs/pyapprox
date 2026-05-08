"""PDE-specific prediction OED problems for obstructed advection-diffusion.

Re-exports the two public classes:

- :class:`AdvectionDiffusionOEDProblem` — full 13-dim parameter
  space (10 KLE terms + 2 inlet shape + 1 Reynolds).
- :class:`FixedVelocityAdvectionDiffusionOEDProblem` — pinned
  velocity, pre-cached Stokes, reduced KLE-only parameter space.

Mesh, Stokes, and KLE helper functions live in the sibling private
modules (``_mesh``, ``_stokes``, ``_kle``) and are not part of the
public API.
"""

from pyapprox_benchmarks.problems.oed.advection_diffusion.problem import (
    AdvectionDiffusionOEDProblem,
    FixedVelocityAdvectionDiffusionOEDProblem,
)

__all__ = [
    "AdvectionDiffusionOEDProblem",
    "FixedVelocityAdvectionDiffusionOEDProblem",
]
