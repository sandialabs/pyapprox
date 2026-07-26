"""Factory for parameterizing collocation elasticity by Young's modulus."""

from pyapprox.pde.collocation.physics.linear_elasticity import (
    LinearElasticityPhysics,
)
from pyapprox.pde.field_maps.protocol import FieldMapProtocol
from pyapprox.pde.parameterizations.collocation_elasticity import (
    CollocationElasticityParameterization,
)
from pyapprox.util.backends.protocols import Array, Backend


def create_youngs_modulus_parameterization(
    physics: LinearElasticityPhysics[Array],
    bkd: Backend[Array],
    field_map: FieldMapProtocol[Array],
    poisson_ratio: float,
) -> CollocationElasticityParameterization[Array]:
    """Factory: parameterize E through the elasticity facade.

    Parameters
    ----------
    physics : LinearElasticityPhysics
        Collocation elasticity physics to bind.
    bkd : Backend
        Computational backend.
    field_map : FieldMapProtocol
        Field map for the Young's modulus field E(x).
    poisson_ratio : float
        Fixed Poisson ratio, -1 < nu < 0.5.
    """
    return CollocationElasticityParameterization(
        physics,
        youngs_modulus_map=field_map,
        poisson_ratio=poisson_ratio,
        bkd=bkd,
    )
