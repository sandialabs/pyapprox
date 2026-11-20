"""The ADR state Jacobian in 1D, 2D and 3D, checked by finite
differences.

Variable diffusion, velocity (not divergence-free) and reaction, both
advection forms.
"""

from typing import Generic, List

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.pde.collocation.basis import (
    ChebyshevBasis1D,
    ChebyshevBasis2D,
    ChebyshevBasis3D,
)
from pyapprox.pde.collocation.mesh import (
    TransformedMesh1D,
    TransformedMesh2D,
    TransformedMesh3D,
)
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.collocation.physics.coefficients import NodalField
from pyapprox.pde.collocation.protocols.basis import TensorProductBasisProtocol
from pyapprox.pde.constitutive.advection import (
    AdvectionFormProtocol,
    ConservativeAdvection,
    NonConservativeAdvection,
)
from pyapprox.util.backends.protocols import Array, Backend


class _Residual(Generic[Array]):
    """The physics residual at time 0 as a function of the state, for
    DerivativeChecker: samples ``(nstates, 1)``, values ``(nstates, 1)``."""

    def __init__(self, physics: AdvectionDiffusionReaction[Array]) -> None:
        self._physics = physics

    def bkd(self) -> Backend[Array]:
        return self._physics.bkd()

    def nvars(self) -> int:
        return self._physics.nstates()

    def nqoi(self) -> int:
        return self._physics.nstates()

    def __call__(self, sample: Array) -> Array:
        return self._physics.residual(sample[:, 0], 0.0)[:, None]

    def jacobian(self, sample: Array) -> Array:
        return self._physics.jacobian(sample[:, 0], 0.0)

    def derivatives(self) -> Derivatives[Array]:
        return Derivatives.first_order(jacobian=self.jacobian)


def _basis(bkd: Backend[Array], ndim: int) -> TensorProductBasisProtocol[Array]:
    if ndim == 1:
        return ChebyshevBasis1D(TransformedMesh1D(8, bkd), bkd)
    if ndim == 2:
        return ChebyshevBasis2D(TransformedMesh2D(6, 5, bkd), bkd)
    return ChebyshevBasis3D(TransformedMesh3D(5, 4, 4, bkd), bkd)


@pytest.mark.parametrize("conservative", [False, True])
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_state_jacobian(bkd: Backend[Array], ndim: int, conservative: bool) -> None:
    """Every term on: D in [1.5, 2.5], v in [-1, 1] per component and
    r in [-1, 1] at random per node; error ratio at most 1e-6 at a
    random state."""
    rng = np.random.default_rng(1)
    basis = _basis(bkd, ndim)
    npts = basis.npts()
    velocity: List[Array] = [
        bkd.asarray(rng.uniform(-1.0, 1.0, npts)) for _ in range(ndim)
    ]
    form: AdvectionFormProtocol = (
        ConservativeAdvection() if conservative else NonConservativeAdvection()
    )
    physics = AdvectionDiffusionReaction(
        basis,
        bkd,
        diffusion=NodalField(bkd.asarray(rng.uniform(1.5, 2.5, npts))),
        velocity=velocity,
        reaction=NodalField(bkd.asarray(rng.uniform(-1.0, 1.0, npts))),
        advection_form=form,
    )
    checker = DerivativeChecker(_Residual(physics))
    state = bkd.asarray(rng.uniform(-1.0, 1.0, (npts, 1)))
    errors = checker.check_derivatives(state, verbosity=0)
    assert checker.error_ratio(errors[0]) <= 1e-6
