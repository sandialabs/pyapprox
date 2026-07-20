"""PDE solvers, parameterizations, and parameterized models.

Layered architecture — imports must point strictly downward:

    zoo                       example problems
    models                    parameterized solver+parameterization wiring
    parameterizations         act ON physics; import solver interfaces
    galerkin | collocation    solver peers (galerkin may import
                              collocation only for manufactured
                              solutions and stress models, until those
                              move to solver-neutral homes)
    field_maps                solver-free general math
    decomposition             self-contained

Direction rules (review-enforced convention):

- Solver packages (collocation, galerkin) must never import
  ``pde.parameterizations`` or ``pde.models`` — physics knows nothing
  about parameterizations; the typed method surface on a physics class
  is its only parameterization-facing declaration.
- ``pyapprox.ode`` must never import ``pde.parameterizations`` — the
  stepper protocols stay ignorant of the ParamDerivatives bundle.
- Capability tiers (adapters wired from a parameterization's
  ParamDerivatives bundle) live in ``pde.models``, never solver-side.
"""
