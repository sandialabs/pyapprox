"""Tests for the linear-elasticity full-matrix field-derivative assemblies.

DerivativeChecker finite differences are the ground truth, the
bilinearity identities cross-validate the field and mixed assemblies
exactly, and the boundary-traction direct-formula test pins the
component-stacked row convention from the public derivative matrices.
"""

from typing import Callable

import numpy as np
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.pde.collocation.mesh import TransformedMesh2D
from pyapprox.pde.collocation.physics.linear_elasticity import (
    LinearElasticityPhysics,
)
from pyapprox.util.backends.protocols import Array, Backend

_NPTS_1D = 5
_NU = 0.3


def _make_physics(bkd, mu_field, lam_field):
    mesh = TransformedMesh2D(_NPTS_1D, _NPTS_1D, bkd)
    basis = ChebyshevBasis2D(mesh, bkd)
    physics = LinearElasticityPhysics(basis, bkd, lamda=1.0, mu=1.0)
    physics.set_mu(mu_field)
    physics.set_lamda(lam_field)
    return basis, physics


def _random(bkd, shape, lb=-1.0, ub=1.0):
    return bkd.asarray(np.random.uniform(lb, ub, shape))


class _VectorFunctionWrapper:
    """Adapts a 1D-array function and its assembly for DerivativeChecker."""

    def __init__(
        self,
        bkd: Backend[Array],
        nvars: int,
        nqoi: int,
        fun: Callable[[Array], Array],
        jac: Callable[[Array], Array],
    ) -> None:
        self._bkd = bkd
        self._nvars = nvars
        self._nqoi = nqoi
        self._fun = fun
        self._jac = jac

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._nvars

    def nqoi(self) -> int:
        return self._nqoi

    def __call__(self, sample: Array) -> Array:
        return self._fun(sample[:, 0])[:, None]

    def jacobian(self, sample: Array) -> Array:
        return self._jac(sample[:, 0])

    def derivatives(self) -> Derivatives[Array]:
        return Derivatives.first_order(jacobian=self.jacobian)


class TestAssemblies:
    def _setup(self, bkd):
        npts = _NPTS_1D**2
        mu_field = _random(bkd, (npts,), 1.0, 2.0)
        lam_field = _random(bkd, (npts,), 1.5, 2.5)
        basis, physics = _make_physics(bkd, mu_field, lam_field)
        state = _random(bkd, (2 * npts,))
        return basis, physics, state, npts, mu_field, lam_field

    def test_mu_bilinearity_identity(self, bkd):
        """A_mu(delta) w == S_mu(w) delta: stress bilinear in
        (strain, mu)."""
        _, physics, state, npts, _, _ = self._setup(bkd)
        delta = _random(bkd, (npts,))
        wvec = _random(bkd, (2 * npts,))
        amat = physics.residual_mu_state_jacobian(delta, state)
        smat_w = physics.residual_mu_jacobian(wvec)
        bkd.assert_allclose(amat @ wvec, smat_w @ delta, rtol=1e-12)

    def test_lamda_bilinearity_identity(self, bkd):
        _, physics, state, npts, _, _ = self._setup(bkd)
        delta = _random(bkd, (npts,))
        wvec = _random(bkd, (2 * npts,))
        amat = physics.residual_lamda_state_jacobian(delta, state)
        smat_w = physics.residual_lamda_jacobian(wvec)
        bkd.assert_allclose(amat @ wvec, smat_w @ delta, rtol=1e-12)

    def test_mu_jacobian_derivative_checker(self, bkd):
        """The residual is affine in the mu field."""
        _, physics, state, npts, mu_field, _ = self._setup(bkd)

        def fun(mfield):
            physics.set_mu(mfield)
            return physics.residual(state, 0.0)

        def jac(mfield):
            physics.set_mu(mfield)
            return physics.residual_mu_jacobian(state)

        checker = DerivativeChecker(
            _VectorFunctionWrapper(bkd, npts, 2 * npts, fun, jac)
        )
        errors = checker.check_derivatives(mu_field[:, None], verbosity=0)
        assert checker.error_ratio(errors[0]) <= 1e-6

    def test_lamda_jacobian_derivative_checker(self, bkd):
        """The residual is affine in the lambda field."""
        _, physics, state, npts, _, lam_field = self._setup(bkd)

        def fun(lfield):
            physics.set_lamda(lfield)
            return physics.residual(state, 0.0)

        def jac(lfield):
            physics.set_lamda(lfield)
            return physics.residual_lamda_jacobian(state)

        checker = DerivativeChecker(
            _VectorFunctionWrapper(bkd, npts, 2 * npts, fun, jac)
        )
        errors = checker.check_derivatives(
            lam_field[:, None], verbosity=0
        )
        assert checker.error_ratio(errors[0]) <= 1e-6

    def test_mixed_assemblies_derivative_checker(self, bkd):
        """A(delta) is d/d(state) of S(state) delta for both fields."""
        _, physics, state, npts, _, _ = self._setup(bkd)
        delta = _random(bkd, (npts,))

        def fun_mu(uvec):
            return physics.residual_mu_jacobian(uvec) @ delta

        def jac_mu(uvec):
            return physics.residual_mu_state_jacobian(delta, uvec)

        checker = DerivativeChecker(
            _VectorFunctionWrapper(bkd, 2 * npts, 2 * npts, fun_mu, jac_mu)
        )
        errors = checker.check_derivatives(state[:, None], verbosity=0)
        assert checker.error_ratio(errors[0]) <= 1e-6

        def fun_lam(uvec):
            return physics.residual_lamda_jacobian(uvec) @ delta

        def jac_lam(uvec):
            return physics.residual_lamda_state_jacobian(delta, uvec)

        checker = DerivativeChecker(
            _VectorFunctionWrapper(
                bkd, 2 * npts, 2 * npts, fun_lam, jac_lam
            )
        )
        errors = checker.check_derivatives(state[:, None], verbosity=0)
        assert checker.error_ratio(errors[0]) <= 1e-6

    def test_boundary_traction_direct_formula(self, bkd):
        """Each traction row has exactly two nonzeros — the local
        strain contractions — recomputed here from the public
        derivative matrices."""
        basis, physics, state, npts, _, _ = self._setup(bkd)
        bc_indices = bkd.array([0, 3, 7], dtype=int)
        raw = np.random.uniform(-1.0, 1.0, (3, 2))
        raw /= np.linalg.norm(raw, axis=1)[:, None]
        normals = bkd.asarray(raw)
        bmat = physics.boundary_traction_lame_jacobian(
            state, 0.0, bc_indices, normals
        )

        u = state[:npts]
        v = state[npts:]
        dx = basis.derivative_matrix(1, 0)
        dy = basis.derivative_matrix(1, 1)
        exx = (dx @ u)[bc_indices]
        exy = (0.5 * ((dy @ u) + (dx @ v)))[bc_indices]
        eyy = (dy @ v)[bc_indices]
        trace = exx + eyy
        nx = normals[:, 0]
        ny = normals[:, 1]

        delta_mu = _random(bkd, (npts,))
        delta_lam = _random(bkd, (npts,))
        stacked = bkd.concatenate([delta_mu, delta_lam])
        result = bmat @ stacked
        expected_tx = (2.0 * exx * nx + 2.0 * exy * ny) * delta_mu[
            bc_indices
        ] + trace * nx * delta_lam[bc_indices]
        expected_ty = (2.0 * exy * nx + 2.0 * eyy * ny) * delta_mu[
            bc_indices
        ] + trace * ny * delta_lam[bc_indices]
        bkd.assert_allclose(result[:3], expected_tx, rtol=1e-12)
        bkd.assert_allclose(result[3:], expected_ty, rtol=1e-12)
