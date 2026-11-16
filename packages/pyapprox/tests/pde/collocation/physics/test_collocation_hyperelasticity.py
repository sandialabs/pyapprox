"""Tests for collocation hyperelasticity: the one Jacobian assembly
(1D, 2D and 3D), the 3D stress, and the interior-operator bundle."""

from typing import Generic, Optional

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.ode.residual_derivatives_check import ODEResidualBundleChecker
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
from pyapprox.pde.collocation.physics.coefficients import (
    CoefficientInput,
    NodalSupplier,
)
from pyapprox.pde.collocation.physics.hyperelasticity import (
    HyperelasticityPhysics,
)
from pyapprox.pde.collocation.protocols.basis import TensorProductBasisProtocol
from pyapprox.pde.collocation.protocols.physics import (
    CollocationInteriorOperatorProtocol,
)
from pyapprox.pde.constitutive.coefficient_functions import TimeDependent
from pyapprox.pde.constitutive.neo_hookean import NeoHookeanStress
from pyapprox.pde.models.collocation.right_hand_side import (
    CollocationRightHandSide,
)
from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.physics_test_utils import PhysicsDerivativeWrapper


def _basis(bkd: Backend[Array], ndim: int) -> TensorProductBasisProtocol[Array]:
    if ndim == 1:
        return ChebyshevBasis1D(TransformedMesh1D(8, bkd), bkd)
    if ndim == 2:
        return ChebyshevBasis2D(TransformedMesh2D(5, 5, bkd), bkd)
    return ChebyshevBasis3D(TransformedMesh3D(4, 4, 4, bkd), bkd)


def _physics(
    bkd: Backend[Array],
    ndim: int,
    forcing: Optional[CoefficientInput[Array]] = None,
) -> HyperelasticityPhysics[Array]:
    stress: NeoHookeanStress[Array] = NeoHookeanStress(lamda=1.5, mu=1.0)
    return HyperelasticityPhysics(_basis(bkd, ndim), bkd, stress, forcing)


def _state(bkd: Backend[Array], nstates: int) -> Array:
    """A small displacement, so that det F stays positive."""
    rng = np.random.default_rng(11)
    return bkd.asarray(0.05 * rng.standard_normal(nstates))


class _GrowingLoad(Generic[Array]):
    """A stacked two-component load growing in time. Shape: (2*npts,)"""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def __call__(self, coords: Array, time: float) -> Array:
        return (1.0 + time) * self._bkd.concatenate([coords[0], coords[1]])


class TestHyperelasticJacobian:
    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_jacobian_derivative_checker(self, bkd: Backend[Array], ndim: int) -> None:
        """The one assembly is the residual's Jacobian in every dimension;
        3D had no Jacobian before."""
        physics = _physics(bkd, ndim)
        checker = DerivativeChecker(PhysicsDerivativeWrapper(physics, time=0.0))
        # Steps from 1e-3 down: the derivative of a random direction is
        # O(n^2), so a larger step inverts the deformation (det F < 0),
        # where the neo-Hookean log is undefined.
        fd_eps = bkd.asarray([10.0**-k for k in range(3, 14)])
        errors = checker.check_derivatives(
            _state(bkd, physics.nstates())[:, None], fd_eps=fd_eps, verbosity=0
        )
        # The sweep decays linearly to ~1e-7 then climbs with roundoff; a
        # wrong Jacobian plateaus instead. The shortened sweep starts at
        # ~1e-2 or ~1e-3, so the ratio reaches only ~1e-5.
        assert float(bkd.min(errors[0])) <= 1e-6
        assert checker.error_ratio(errors[0]) <= 1e-4

    def test_stress_free_at_rest_3d(self, bkd: Backend[Array]) -> None:
        """At zero displacement F = I, so the neo-Hookean stress and the
        residual vanish; the 3D stress raised before."""
        physics = _physics(bkd, 3)
        zero = bkd.zeros((physics.nstates(),))
        for row in physics.pk1_stress(zero):
            for entry in row:
                bkd.assert_allclose(entry, bkd.zeros(entry.shape), atol=1e-14)
        bkd.assert_allclose(physics.residual(zero, 0.0), zero, atol=1e-12)


class TestHyperelasticInterior:
    def test_time_dependent_body_force(self, bkd: Backend[Array]) -> None:
        """A time-dependent body force leaves the Jacobian time-invariant
        but not the operator; the first-order bundle passes the checker."""
        basis = ChebyshevBasis2D(TransformedMesh2D(5, 5, bkd), bkd)
        nodes = bkd.stack(
            [
                bkd.flatten(xx)
                for xx in bkd.meshgrid(basis.nodes_x(), basis.nodes_y(), indexing="xy")
            ],
            axis=0,
        )
        load: TimeDependent[Array] = TimeDependent(_GrowingLoad(bkd))
        physics = _physics(bkd, 2, NodalSupplier(load, nodes))
        assert isinstance(physics, CollocationInteriorOperatorProtocol)
        assert physics.interior_jacobian_is_time_invariant()
        assert not physics.interior_is_time_invariant()
        assert _physics(bkd, 2).interior_is_time_invariant()
        rhs = CollocationRightHandSide(physics)
        report = ODEResidualBundleChecker(rhs).check(
            _state(bkd, physics.nstates()), 0.3
        )
        assert report.failures() == []
