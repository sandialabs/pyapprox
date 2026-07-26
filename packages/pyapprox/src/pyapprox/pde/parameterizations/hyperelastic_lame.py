"""Factory for parameterizing collocation hyperelasticity by Young's modulus."""

from pyapprox.pde.collocation.physics.hyperelasticity import (
    HyperelasticityPhysics,
)
from pyapprox.pde.field_maps.protocol import FieldMapProtocol
from pyapprox.pde.parameterizations.collocation_hyperelasticity import (
    CollocationHyperelasticityParameterization,
)
from pyapprox.util.backends.protocols import Array, Backend


def create_hyperelastic_youngs_modulus_parameterization(
    physics: HyperelasticityPhysics[Array],
    bkd: Backend[Array],
    field_map: FieldMapProtocol[Array],
    poisson_ratio: float,
) -> CollocationHyperelasticityParameterization[Array]:
    """Factory: parameterize E through the hyperelasticity facade.

    Parameters
    ----------
    physics : HyperelasticityPhysics
        Collocation hyperelastic physics to bind (1D or 2D).
    bkd : Backend
        Computational backend.
    field_map : FieldMapProtocol
        Field map for the Young's modulus field E(x).
    poisson_ratio : float
        Fixed Poisson ratio, -1 < nu < 0.5.
    """
    return CollocationHyperelasticityParameterization(
        physics,
        youngs_modulus_map=field_map,
        poisson_ratio=poisson_ratio,
        bkd=bkd,
    )
