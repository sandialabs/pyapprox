"""Tests for 3D linear elasticity collocation physics.

DerivativeChecker finite differences are the ground truth, the
bilinearity identities cross-validate the field and mixed assemblies
exactly, the patch test pins the residual and interface traction on a
uniform-strain state, and the boundary-traction direct-formula test
pins the component-stacked row convention.

3D transient time-stepping is intentionally not tested: time
integration is dimension-agnostic and covered by the 2D transient
tests, while the 3D residual/jacobian are covered here and by the 3D
manufactured-solution tests.
"""

from typing import Callable

import numpy as np
import pytest
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.pde.collocation.basis import ChebyshevBasis1D, ChebyshevBasis3D
from pyapprox.pde.collocation.mesh import (
    AffineTransform3D,
    TransformedMesh1D,
    TransformedMesh3D,
)
from pyapprox.pde.collocation.physics.linear_elasticity import (
    LinearElasticityPhysics,
)
from pyapprox.util.backends.protocols import Array, Backend

from tests._helpers.physics_test_utils import PhysicsDerivativeWrapper

_NPTS_1D = 4


def _make_basis(bkd, npts_1d=_NPTS_1D, transform=None):
    mesh = TransformedMesh3D(npts_1d, npts_1d, npts_1d, bkd, transform)
    return mesh, ChebyshevBasis3D(mesh, bkd)


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


class TestConstruction:
    def test_constructor_rejects_1d(self, bkd):
        mesh = TransformedMesh1D(5, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        with pytest.raises(ValueError, match="2D or 3D"):
            LinearElasticityPhysics(basis, bkd, 1.0, 1.0)

    def test_dimensions(self, bkd):
        _, basis = _make_basis(bkd)
        physics = LinearElasticityPhysics(basis, bkd, 1.0, 1.0)
        npts = _NPTS_1D**3
        assert physics.ncomponents() == 3
        assert physics.nstates() == 3 * npts

    def test_state_state_hvp_zero(self, bkd):
        _, basis = _make_basis(bkd)
        physics = LinearElasticityPhysics(basis, bkd, 1.0, 1.0)
        npts = _NPTS_1D**3
        state = _random(bkd, (3 * npts,))
        adj = _random(bkd, (3 * npts,))
        wvec = _random(bkd, (3 * npts,))
        hvp = physics.state_state_hvp(state, adj, wvec, 0.0)
        assert hvp.shape == (3 * npts,)
        bkd.assert_allclose(hvp, bkd.zeros((3 * npts,)), atol=1e-15)


class TestPatchUniformStrain:
    """Uniform uniaxial extension u = a*x, v = -b*y, w = -b*z.

    Constant strain: the residual vanishes identically (derivative
    matrices are exact on linears) and the traction on every face is
    the constant analytic sigma . n.
    """

    _A = 0.1
    _B = 0.03
    _LAM = 1.0
    _MU = 1.0

    def _patch_state(self, bkd, mesh):
        pts = mesh.points()
        return bkd.concatenate(
            [self._A * pts[0], -self._B * pts[1], -self._B * pts[2]]
        )

    def test_residual_zero(self, bkd):
        mesh, basis = _make_basis(bkd, npts_1d=6)
        physics = LinearElasticityPhysics(
            basis, bkd, self._LAM, self._MU
        )
        state = self._patch_state(bkd, mesh)
        res = physics.residual(state, 0.0)
        bkd.assert_allclose(
            res, bkd.zeros((physics.nstates(),)), atol=1e-9
        )

    def test_interface_flux_faces(self, bkd):
        mesh, basis = _make_basis(bkd, npts_1d=6)
        physics = LinearElasticityPhysics(
            basis, bkd, self._LAM, self._MU
        )
        state = self._patch_state(bkd, mesh)
        trace = self._A - 2.0 * self._B
        sigma_xx = self._LAM * trace + 2.0 * self._MU * self._A
        sigma_yy = self._LAM * trace - 2.0 * self._MU * self._B

        # Face with outward normal (1, 0, 0): t = (sigma_xx, 0, 0)
        idx_x = mesh.boundary_indices(1)
        normal_x = mesh.boundary_normals(1)[0]
        flux = physics.compute_interface_flux(state, idx_x, normal_x)
        nb = idx_x.shape[0]
        assert flux.shape == (3 * nb,)
        bkd.assert_allclose(
            flux[:nb], bkd.full((nb,), sigma_xx), atol=1e-8
        )
        bkd.assert_allclose(
            flux[nb:], bkd.zeros((2 * nb,)), atol=1e-8
        )

        # Face with outward normal (0, 1, 0): t = (0, sigma_yy, 0)
        idx_y = mesh.boundary_indices(3)
        normal_y = mesh.boundary_normals(3)[0]
        flux = physics.compute_interface_flux(state, idx_y, normal_y)
        nb = idx_y.shape[0]
        bkd.assert_allclose(
            flux[nb : 2 * nb], bkd.full((nb,), sigma_yy), atol=1e-8
        )
        bkd.assert_allclose(flux[:nb], bkd.zeros((nb,)), atol=1e-8)
        bkd.assert_allclose(
            flux[2 * nb :], bkd.zeros((nb,)), atol=1e-8
        )


class TestJacobian:
    def _setup(self, bkd, transform=None, field_lame=False):
        mesh, basis = _make_basis(bkd, transform=transform)
        npts = _NPTS_1D**3
        physics = LinearElasticityPhysics(basis, bkd, 1.5, 1.0)
        if field_lame:
            physics.set_mu(_random(bkd, (npts,), 1.0, 2.0))
            physics.set_lamda(_random(bkd, (npts,), 1.5, 2.5))
        state = _random(bkd, (3 * npts,))
        return physics, state, npts

    @pytest.mark.parametrize("field_lame", [False, True])
    def test_linearity_identity(self, bkd, field_lame):
        """J @ state == residual(state) - residual(0) exactly: the
        residual is affine in the displacement."""
        physics, state, npts = self._setup(bkd, field_lame=field_lame)
        jac = physics.jacobian(state, 0.0)
        res = physics.residual(state, 0.0)
        res0 = physics.residual(bkd.zeros((3 * npts,)), 0.0)
        bkd.assert_allclose(jac @ state, res - res0, rtol=1e-12)

    def test_jacobian_derivative_checker(self, bkd):
        physics, state, _ = self._setup(bkd)
        checker = DerivativeChecker(
            PhysicsDerivativeWrapper(physics, time=0.0)
        )
        errors = checker.check_derivatives(state[:, None], verbosity=0)
        assert checker.error_ratio(errors[0]) <= 1e-6

    def test_jacobian_affine_transform(self, bkd):
        """Curvilinear-ordering guard: on a non-unit box the mixed
        blocks D_i @ diag @ D_j depend on operator order."""
        transform = AffineTransform3D(
            (0.0, 2.0, -1.0, 0.5, 0.0, 1.0), bkd
        )
        physics, state, npts = self._setup(bkd, transform=transform)
        jac = physics.jacobian(state, 0.0)
        res = physics.residual(state, 0.0)
        res0 = physics.residual(bkd.zeros((3 * npts,)), 0.0)
        bkd.assert_allclose(jac @ state, res - res0, rtol=1e-12)
        checker = DerivativeChecker(
            PhysicsDerivativeWrapper(physics, time=0.0)
        )
        errors = checker.check_derivatives(state[:, None], verbosity=0)
        assert checker.error_ratio(errors[0]) <= 1e-6


class TestAssemblies:
    def _setup(self, bkd):
        npts = _NPTS_1D**3
        mu_field = _random(bkd, (npts,), 1.0, 2.0)
        lam_field = _random(bkd, (npts,), 1.5, 2.5)
        _, basis = _make_basis(bkd)
        physics = LinearElasticityPhysics(basis, bkd, 1.0, 1.0)
        physics.set_mu(mu_field)
        physics.set_lamda(lam_field)
        state = _random(bkd, (3 * npts,))
        return basis, physics, state, npts, mu_field, lam_field

    def test_mu_bilinearity_identity(self, bkd):
        """A_mu(delta) w == S_mu(w) delta: stress bilinear in
        (strain, mu)."""
        _, physics, state, npts, _, _ = self._setup(bkd)
        delta = _random(bkd, (npts,))
        wvec = _random(bkd, (3 * npts,))
        amat = physics.residual_mu_state_jacobian(delta, state)
        smat_w = physics.residual_mu_jacobian(wvec)
        bkd.assert_allclose(amat @ wvec, smat_w @ delta, rtol=1e-12)

    def test_lamda_bilinearity_identity(self, bkd):
        _, physics, state, npts, _, _ = self._setup(bkd)
        delta = _random(bkd, (npts,))
        wvec = _random(bkd, (3 * npts,))
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
            _VectorFunctionWrapper(bkd, npts, 3 * npts, fun, jac)
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
            _VectorFunctionWrapper(bkd, npts, 3 * npts, fun, jac)
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
            _VectorFunctionWrapper(bkd, 3 * npts, 3 * npts, fun_mu, jac_mu)
        )
        errors = checker.check_derivatives(state[:, None], verbosity=0)
        assert checker.error_ratio(errors[0]) <= 1e-6

        def fun_lam(uvec):
            return physics.residual_lamda_jacobian(uvec) @ delta

        def jac_lam(uvec):
            return physics.residual_lamda_state_jacobian(delta, uvec)

        checker = DerivativeChecker(
            _VectorFunctionWrapper(
                bkd, 3 * npts, 3 * npts, fun_lam, jac_lam
            )
        )
        errors = checker.check_derivatives(state[:, None], verbosity=0)
        assert checker.error_ratio(errors[0]) <= 1e-6

    def test_boundary_traction_direct_formula(self, bkd):
        """Each traction row has exactly two nonzeros — the local
        strain contractions — recomputed here from the public
        derivative matrices, for all three component offsets."""
        basis, physics, state, npts, _, _ = self._setup(bkd)
        mesh_idx = bkd.array([0, 3, 7], dtype=int)
        raw = np.random.uniform(-1.0, 1.0, (3, 3))
        raw /= np.linalg.norm(raw, axis=1)[:, None]
        normals = bkd.asarray(raw)

        comps = [state[i * npts : (i + 1) * npts] for i in range(3)]
        dmats = [basis.derivative_matrix(1, d) for d in range(3)]
        grads = [
            [dmats[j] @ comps[i] for j in range(3)] for i in range(3)
        ]
        eps = [
            [
                grads[i][j]
                if i == j
                else 0.5 * (grads[i][j] + grads[j][i])
                for j in range(3)
            ]
            for i in range(3)
        ]
        trace = (eps[0][0] + eps[1][1] + eps[2][2])[mesh_idx]

        delta_mu = _random(bkd, (npts,))
        delta_lam = _random(bkd, (npts,))
        stacked = bkd.concatenate([delta_mu, delta_lam])

        for comp in range(3):
            bmat = physics.boundary_traction_lame_jacobian(
                state, 0.0, mesh_idx + comp * npts, normals
            )
            dt_dmu = sum(
                2.0 * eps[comp][j][mesh_idx] * normals[:, j]
                for j in range(3)
            )
            expected = dt_dmu * delta_mu[mesh_idx] + trace * normals[
                :, comp
            ] * delta_lam[mesh_idx]
            bkd.assert_allclose(bmat @ stacked, expected, rtol=1e-12)
