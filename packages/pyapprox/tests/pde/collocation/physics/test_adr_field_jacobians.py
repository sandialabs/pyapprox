"""Tests for the ADR full-matrix field-derivative assemblies.

DerivativeChecker finite differences are the ground truth, the
bilinearity identity cross-validates the diffusion and mixed
assemblies exactly, and the bc-flux direct-formula test pins the row
convention from the public derivative matrices. (Transition-scoped
parity tests validated the assemblies against the deleted
delta-contracted physics methods at rtol 1e-12 before their removal.)
"""

from typing import Callable

import numpy as np
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.pde.collocation.mesh import TransformedMesh2D
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.parameterizations.fields import ConstantInTimeField
from pyapprox.util.backends.protocols import Array, Backend

_NPTS_1D = 5


def _make_basis(bkd):
    mesh = TransformedMesh2D(_NPTS_1D, _NPTS_1D, bkd)
    return ChebyshevBasis2D(mesh, bkd)


def _random(bkd, shape, lb=-1.0, ub=1.0):
    return bkd.asarray(np.random.uniform(lb, ub, shape))


class _VectorFunctionWrapper:
    """Adapts a 1D-array function and its assembly for DerivativeChecker.

    DerivativeChecker expects __call__(sample) with sample (nvars, 1)
    returning (nqoi, 1) and jacobian(sample) returning (nqoi, nvars).
    """

    def __init__(
        self,
        bkd: Backend[Array],
        nvars: int,
        fun: Callable[[Array], Array],
        jac: Callable[[Array], Array],
    ) -> None:
        self._bkd = bkd
        self._nvars = nvars
        self._fun = fun
        self._jac = jac

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._nvars

    def nqoi(self) -> int:
        return self._nvars

    def __call__(self, sample: Array) -> Array:
        return self._fun(sample[:, 0])[:, None]

    def jacobian(self, sample: Array) -> Array:
        return self._jac(sample[:, 0])

    def derivatives(self) -> Derivatives[Array]:
        return Derivatives.first_order(jacobian=self.jacobian)


class TestDiffusionAssemblies:
    def _setup(self, bkd):
        basis = _make_basis(bkd)
        npts = basis.npts()
        diffusion = _random(bkd, (npts,), 1.5, 2.5)
        physics = AdvectionDiffusionReaction(basis, bkd)
        physics.set_diffusion(ConstantInTimeField(diffusion))
        state = _random(bkd, (npts,))
        return basis, physics, diffusion, state

    def test_bilinearity_identity(self, bkd):
        """A(delta) w == S(w) delta: the diffusion term is bilinear."""
        basis, physics, _, state = self._setup(bkd)
        npts = basis.npts()
        delta = _random(bkd, (npts,))
        wvec = _random(bkd, (npts,))
        amat = physics.residual_diffusion_state_jacobian(delta, state)
        smat_w = physics.residual_diffusion_jacobian(wvec)
        bkd.assert_allclose(amat @ wvec, smat_w @ delta, rtol=1e-12)

    def test_jacobian_derivative_checker(self, bkd):
        basis, physics, diffusion, state = self._setup(bkd)
        npts = basis.npts()

        def fun(dfield):
            physics.set_diffusion(ConstantInTimeField(dfield))
            return physics.residual(state, 0.0)

        def jac(dfield):
            physics.set_diffusion(ConstantInTimeField(dfield))
            return physics.residual_diffusion_jacobian(state)

        checker = DerivativeChecker(
            _VectorFunctionWrapper(bkd, npts, fun, jac)
        )
        errors = checker.check_derivatives(diffusion[:, None], verbosity=0)
        assert checker.error_ratio(errors[0]) <= 1e-6

    def test_state_jacobian_derivative_checker(self, bkd):
        basis, physics, _, state = self._setup(bkd)
        npts = basis.npts()
        delta = _random(bkd, (npts,))

        def fun(uvec):
            return physics.residual_diffusion_jacobian(uvec) @ delta

        def jac(uvec):
            return physics.residual_diffusion_state_jacobian(delta, uvec)

        checker = DerivativeChecker(
            _VectorFunctionWrapper(bkd, npts, fun, jac)
        )
        errors = checker.check_derivatives(state[:, None], verbosity=0)
        assert checker.error_ratio(errors[0]) <= 1e-6


class TestReactionForcingAssemblies:
    def _setup(self, bkd):
        basis = _make_basis(bkd)
        npts = basis.npts()
        physics = AdvectionDiffusionReaction(basis, bkd)
        reaction = _random(bkd, (npts,), 0.5, 1.5)
        physics.set_reaction(ConstantInTimeField(reaction))
        state = _random(bkd, (npts,))
        return basis, physics, reaction, state

    def test_reaction_jacobian_is_diag_state(self, bkd):
        basis, physics, _, state = self._setup(bkd)
        npts = basis.npts()
        smat = physics.residual_reaction_jacobian(state)
        delta = _random(bkd, (npts,))
        bkd.assert_allclose(smat @ delta, state * delta, rtol=1e-12)

    def test_reaction_state_jacobian(self, bkd):
        basis, physics, _, state = self._setup(bkd)
        npts = basis.npts()
        delta = _random(bkd, (npts,))
        wvec = _random(bkd, (npts,))
        amat = physics.residual_reaction_state_jacobian(delta, state)
        bkd.assert_allclose(amat @ wvec, delta * wvec, rtol=1e-12)

    def test_reaction_jacobian_derivative_checker(self, bkd):
        basis, physics, reaction, state = self._setup(bkd)
        npts = basis.npts()

        def fun(rfield):
            physics.set_reaction(ConstantInTimeField(rfield))
            return physics.residual(state, 0.0)

        def jac(rfield):
            physics.set_reaction(ConstantInTimeField(rfield))
            return physics.residual_reaction_jacobian(state)

        checker = DerivativeChecker(
            _VectorFunctionWrapper(bkd, npts, fun, jac)
        )
        errors = checker.check_derivatives(reaction[:, None], verbosity=0)
        assert checker.error_ratio(errors[0]) <= 1e-6

    def test_forcing_jacobian_derivative_checker(self, bkd):
        basis, physics, _, state = self._setup(bkd)
        npts = basis.npts()
        smat = physics.residual_forcing_jacobian()
        bkd.assert_allclose(smat, bkd.eye(npts), rtol=1e-14)
        forcing = _random(bkd, (npts,))

        def fun(ffield):
            physics.set_forcing(ConstantInTimeField(ffield))
            return physics.residual(state, 0.0)

        def jac(ffield):
            physics.set_forcing(ConstantInTimeField(ffield))
            return physics.residual_forcing_jacobian()

        checker = DerivativeChecker(
            _VectorFunctionWrapper(bkd, npts, fun, jac)
        )
        errors = checker.check_derivatives(forcing[:, None], verbosity=0)
        assert checker.error_ratio(errors[0]) <= 1e-6


class TestBoundaryFluxAssembly:
    def _setup(self, bkd):
        basis = _make_basis(bkd)
        npts = basis.npts()
        diffusion = _random(bkd, (npts,), 1.5, 2.5)
        physics = AdvectionDiffusionReaction(basis, bkd)
        physics.set_diffusion(ConstantInTimeField(diffusion))
        state = _random(bkd, (npts,))
        bc_indices = bkd.array([0, 3, 7], dtype=int)
        raw = np.random.uniform(-1.0, 1.0, (3, basis.ndim()))
        raw /= np.linalg.norm(raw, axis=1)[:, None]
        normals = bkd.asarray(raw)
        return basis, physics, diffusion, state, bc_indices, normals

    def test_direct_formula(self, bkd):
        """B delta == -(grad u . n)[bc] * delta[bc], computed here from
        the public derivative matrices."""
        basis, physics, _, state, bc_indices, normals = self._setup(bkd)
        npts = basis.npts()
        bmat = physics.boundary_flux_diffusion_jacobian(
            state, 0.0, bc_indices, normals
        )
        delta = _random(bkd, (npts,))
        grad_u_dot_n = bkd.zeros((3,))
        for dim in range(basis.ndim()):
            grad_u = basis.derivative_matrix(1, dim) @ state
            grad_u_dot_n = (
                grad_u_dot_n + grad_u[bc_indices] * normals[:, dim]
            )
        expected = -grad_u_dot_n * delta[bc_indices]
        bkd.assert_allclose(bmat @ delta, expected, rtol=1e-12)

    def test_row_convention_through_field_map(self, bkd):
        """B @ G'(p) matches -(grad u . n)_i * dD_dp[bc_i, :] — the
        dflux_n_dp row convention, recomputed here from the public
        derivative matrices and the field-map jacobian."""
        basis, physics, _, state, bc_indices, normals = self._setup(bkd)
        npts = basis.npts()
        nmodes = 3
        modes = bkd.asarray(np.random.uniform(-0.4, 0.4, (npts, nmodes)))
        field_map = MeshKLEFieldMap(bkd, bkd.full((npts,), 2.0), modes)
        params = _random(bkd, (nmodes,))
        grad_u_dot_n = bkd.zeros((3,))
        for dim in range(basis.ndim()):
            grad_u = basis.derivative_matrix(1, dim) @ state
            grad_u_dot_n = (
                grad_u_dot_n + grad_u[bc_indices] * normals[:, dim]
            )
        dd_dp = field_map.jacobian(params)
        expected = -grad_u_dot_n[:, None] * dd_dp[bc_indices]
        bmat = physics.boundary_flux_diffusion_jacobian(
            state, 0.0, bc_indices, normals
        )
        bkd.assert_allclose(
            bmat @ field_map.jacobian(params), expected, rtol=1e-12
        )
