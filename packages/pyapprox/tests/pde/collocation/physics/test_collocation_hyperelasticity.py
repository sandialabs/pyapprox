"""Tests for collocation hyperelasticity: the one Jacobian assembly
(1D, 2D and 3D), the 3D stress, and the interior-operator bundle."""

from typing import Callable, Generic, List, Optional

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
    NodalField,
    NodalSupplier,
)
from pyapprox.pde.collocation.physics.hyperelasticity import (
    HyperelasticityPhysics,
)
from pyapprox.pde.collocation.physics.neo_hookean_law import NeoHookeanLaw
from pyapprox.pde.collocation.protocols.basis import TensorProductBasisProtocol
from pyapprox.pde.collocation.protocols.physics import (
    CollocationInteriorOperatorProtocol,
)
from pyapprox.pde.constitutive.coefficient_functions import TimeDependent
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
    return HyperelasticityPhysics(
        _basis(bkd, ndim), bkd, NeoHookeanLaw(bkd), mu=1.0, lamda=1.5, forcing=forcing
    )


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


def _fields(
    bkd: Backend[Array], ndim: int
) -> tuple[HyperelasticityPhysics[Array], NodalField[Array], NodalField[Array]]:
    """A physics whose Lame parameters are rewritable, varying fields."""
    basis = _basis(bkd, ndim)
    npts = basis.npts()
    rng = np.random.default_rng(5)
    mu = NodalField(bkd.asarray(1.0 + 0.3 * rng.uniform(size=npts)))
    lamda = NodalField(bkd.asarray(1.5 + 0.3 * rng.uniform(size=npts)))
    law: NeoHookeanLaw[Array] = NeoHookeanLaw(bkd)
    return HyperelasticityPhysics(basis, bkd, law, mu=mu, lamda=lamda), mu, lamda


def _central_difference(
    bkd: Backend[Array],
    field: NodalField[Array],
    output: Callable[[], Array],
    eps: float = 1e-6,
) -> Array:
    """Central differences of ``output()`` in each value of ``field``.
    Shape: (noutputs, npts)"""
    base = field.values(0.0)
    columns = []
    for k in range(base.shape[0]):
        step = bkd.asarray(np.eye(base.shape[0])[k] * eps)
        field.set_values(base + step)
        plus = output()
        field.set_values(base - step)
        minus = output()
        columns.append((plus - minus) / (2.0 * eps))
    field.set_values(base)
    return bkd.stack(columns, axis=1)


class _StressOnlyLaw(Generic[Array]):
    """A law with stress and tangent but no Lame sensitivities."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._law: NeoHookeanLaw[Array] = NeoHookeanLaw(bkd)

    def stress(
        self, F: List[List[Array]], mu: Array, lamda: Array
    ) -> List[List[Array]]:
        return self._law.stress(F, mu, lamda)

    def tangent(
        self, F: List[List[Array]], mu: Array, lamda: Array
    ) -> List[List[List[List[Array]]]]:
        return self._law.tangent(F, mu, lamda)


class TestHyperelasticMaterial:
    def test_reads_rewritten_field(self, bkd: Backend[Array]) -> None:
        """The stress follows the physics' mu field after set_values: the
        parameterization and every reader of the physics see one
        material."""
        physics, mu, lamda = _fields(bkd, 2)
        state = _state(bkd, physics.nstates())
        before = physics.residual(state, 0.0)
        mu.set_values(2.0 * mu.values(0.0))
        F = physics._deformation_gradient(state)
        expected = NeoHookeanLaw(bkd).stress(F, mu.values(0.0), lamda.values(0.0))
        P = physics.pk1_stress(state)
        for i in range(2):
            for J in range(2):
                bkd.assert_allclose(P[i][J], expected[i][J], rtol=1e-14)
        assert float(bkd.norm(physics.residual(state, 0.0) - before)) > 1e-3

    def test_invalid_values_raise_on_read(self, bkd: Backend[Array]) -> None:
        physics, mu, lamda = _fields(bkd, 1)
        state = _state(bkd, physics.nstates())
        good_mu = mu.values(0.0)
        mu.set_values(-good_mu)
        with pytest.raises(ValueError, match="mu must be positive"):
            physics.residual(state, 0.0)
        mu.set_values(good_mu)
        lamda.set_values(-lamda.values(0.0))
        with pytest.raises(ValueError, match="lamda must be non-negative"):
            physics.residual(state, 0.0)

    def test_time_dependent_material_refused(self, numpy_bkd: Backend[Array]) -> None:
        basis = _basis(numpy_bkd, 1)
        nodes = basis.mesh().points()
        growing: TimeDependent[Array] = TimeDependent(_UniformGrowth(numpy_bkd))
        with pytest.raises(ValueError, match="mu must be time-invariant"):
            HyperelasticityPhysics(
                basis,
                numpy_bkd,
                NeoHookeanLaw(numpy_bkd),
                mu=NodalSupplier(growing, nodes),
                lamda=1.0,
            )

    def test_non_law_refused(self, numpy_bkd: Backend[Array]) -> None:
        with pytest.raises(TypeError, match="HyperelasticLawProtocol"):
            HyperelasticityPhysics(
                _basis(numpy_bkd, 1),
                numpy_bkd,
                object(),  # type: ignore[arg-type]
                mu=1.0,
                lamda=1.0,
            )

    @pytest.mark.parametrize("ndim", [1, 2])
    def test_residual_lame_jacobians(self, bkd: Backend[Array], ndim: int) -> None:
        """dR/dmu and dR/dlamda equal central differences of the residual
        in the fields."""
        physics, mu, lamda = _fields(bkd, ndim)
        state = _state(bkd, physics.nstates())
        for field, assembly in (
            (mu, physics.residual_mu_jacobian),
            (lamda, physics.residual_lamda_jacobian),
        ):
            fd = _central_difference(bkd, field, lambda: physics.residual(state, 0.0))
            bkd.assert_allclose(assembly(state), fd, rtol=1e-6, atol=1e-7)

    @pytest.mark.parametrize("ndim", [1, 2])
    def test_boundary_traction_lame_jacobian(
        self, bkd: Backend[Array], ndim: int
    ) -> None:
        """The [mu; lamda] Jacobian of the traction sum_J P_iJ n_J at one
        BC's rows equals central differences, for each component."""
        physics, mu, lamda = _fields(bkd, ndim)
        npts = physics.npts()
        state = _state(bkd, physics.nstates())
        mesh = physics.basis().mesh()
        idx = mesh.boundary_indices(1)
        normals = mesh.boundary_normals(1)
        for comp in range(ndim):

            def traction(comp: int = comp) -> Array:
                row = physics.pk1_stress(state)[comp]
                t = row[0][idx] * normals[:, 0]
                for J in range(1, ndim):
                    t = t + row[J][idx] * normals[:, J]
                return t

            jac = physics.boundary_traction_lame_jacobian(
                state, 0.0, idx + comp * npts, normals
            )
            fd = bkd.concatenate(
                [
                    _central_difference(bkd, mu, traction),
                    _central_difference(bkd, lamda, traction),
                ],
                axis=1,
            )
            bkd.assert_allclose(jac, fd, rtol=1e-6, atol=1e-7)

    def test_law_without_sensitivities(self, bkd: Backend[Array]) -> None:
        """A law with no Lame sensitivities solves, but the Lame
        assemblies a parameterization needs are refused."""
        basis = _basis(bkd, 1)
        physics = HyperelasticityPhysics(
            basis, bkd, _StressOnlyLaw(bkd), mu=1.0, lamda=1.5
        )
        state = _state(bkd, physics.nstates())
        reference = _physics(bkd, 1)
        bkd.assert_allclose(
            physics.residual(state, 0.0), reference.residual(state, 0.0)
        )
        with pytest.raises(NotImplementedError, match="no Lame-parameter"):
            physics.residual_mu_jacobian(state)

    def test_lame_jacobians_refused_in_3d(self, numpy_bkd: Backend[Array]) -> None:
        physics = _physics(numpy_bkd, 3)
        with pytest.raises(NotImplementedError, match="1D and 2D"):
            physics.residual_mu_jacobian(_state(numpy_bkd, physics.nstates()))


class _UniformGrowth(Generic[Array]):
    """A material value growing in time."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def __call__(self, coords: Array, time: float) -> Array:
        return (1.0 + time) * self._bkd.ones((coords.shape[1],))
