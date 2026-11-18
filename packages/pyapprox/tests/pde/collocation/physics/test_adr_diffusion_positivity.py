"""Strict positivity of the advection-diffusion-reaction diffusion field.

The physics raises when it first uses a non-positive field, not the
parameterization that wrote it: positivity is a property of the operator,
so it must hold for a physics built directly or changed after
construction, neither of which passes through a parameterization. Five
nodes on the reference interval; the field is :math:`0 + p_0`.
"""

from typing import Tuple

import pytest

from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.mesh import TransformedMesh1D
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.field_maps.basis_expansion import BasisExpansion
from pyapprox.pde.parameterizations.collocation_advection_diffusion import (
    CollocationAdvectionDiffusionParameterization,
)
from pyapprox.pde.parameterizations.diffusion import (
    create_diffusion_parameterization,
)
from pyapprox.util.backends.protocols import Array, Backend


def _param_and_physics(
    bkd: Backend[Array],
) -> Tuple[
    CollocationAdvectionDiffusionParameterization[Array],
    AdvectionDiffusionReaction[Array],
]:
    mesh = TransformedMesh1D(5, bkd)
    physics = AdvectionDiffusionReaction(
        ChebyshevBasis1D(mesh, bkd), bkd, diffusion=1.0
    )
    field_map = BasisExpansion(bkd, 0.0, [bkd.ones((5,))])
    return create_diffusion_parameterization(physics, bkd, field_map), physics


class TestDiffusionPositivity:
    def test_negative_diffusion_raises(self, bkd: Backend[Array]) -> None:
        param, physics = _param_and_physics(bkd)
        param.apply(bkd.array([-0.1]))
        with pytest.raises(ValueError, match="positive"):
            physics._get_diffusion(0.0)

    def test_zero_diffusion_raises(self, bkd: Backend[Array]) -> None:
        """Zero is inadmissible too: the diffusion term vanishes."""
        param, physics = _param_and_physics(bkd)
        param.apply(bkd.array([0.0]))
        with pytest.raises(ValueError):
            physics._get_diffusion(0.0)

    def test_positive_diffusion_succeeds(self, bkd: Backend[Array]) -> None:
        param, physics = _param_and_physics(bkd)
        param.apply(bkd.array([1.5]))
        physics._get_diffusion(0.0)
