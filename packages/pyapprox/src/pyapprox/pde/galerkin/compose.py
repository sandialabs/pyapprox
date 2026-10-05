"""Join a Galerkin physics with its boundary conditions.

The physics is the interior weak form and the mass. The boundary
conditions are split by role: natural terms join the interior in
``F = F_Omega + F_Gamma``, and essential conditions become the constraint
set. ``GalerkinSystem`` holds the resulting parts and knows nothing of how
they were built.
"""

from typing import Sequence

from pyapprox.pde.boundary import (
    BoundaryConditionRole,
    DirichletConstraintSet,
    NaturalBCOperator,
    split_by_role,
)
from pyapprox.pde.galerkin.physics.bc_mixin import GalerkinBCMixin
from pyapprox.pde.galerkin.protocols.physics import GalerkinPhysicsProtocol
from pyapprox.pde.galerkin.spatial_operator import ComposedSpatialOperator
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.util.backends.protocols import Array


def compose_galerkin_system(
    physics: GalerkinPhysicsProtocol[Array],
    boundary_conditions: Sequence[BoundaryConditionRole[Array]] = (),
) -> GalerkinSystem[Array]:
    """Compose a physics with its boundary conditions into a system.

    Parameters
    ----------
    physics : GalerkinPhysicsProtocol
        The interior operator ``F_Omega`` and the mass.
    boundary_conditions : Sequence[BoundaryConditionRole]
        Each in exactly one role, natural or essential.

    Returns
    -------
    GalerkinSystem
        ``F``, the essential constraints, and the mass.

    Raises
    ------
    TypeError
        If ``physics`` does not satisfy ``GalerkinPhysicsProtocol``, or a
        condition is in neither role or both.
    ValueError
        If ``physics`` was constructed with its own boundary conditions,
        which composing would silently drop.
    """
    if not isinstance(physics, GalerkinPhysicsProtocol):
        raise TypeError(
            "physics must satisfy GalerkinPhysicsProtocol, got "
            f"{type(physics).__name__}"
        )
    if isinstance(physics, GalerkinBCMixin) and (
        physics.weak_form_bcs() or physics.essential_bcs()
    ):
        raise ValueError(
            f"{type(physics).__name__} was constructed with "
            "boundary_conditions=, which compose_galerkin_system would "
            "ignore; construct it without them and pass them here instead"
        )
    roles = split_by_role(boundary_conditions)
    return GalerkinSystem(
        ComposedSpatialOperator(physics, NaturalBCOperator(roles.terms())),
        DirichletConstraintSet(
            roles.essentials(), physics.nstates(), physics.bkd()
        ),
        physics,
    )
