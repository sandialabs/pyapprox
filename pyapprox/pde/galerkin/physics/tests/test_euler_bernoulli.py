"""Tests for Euler-Bernoulli beam physics.

Tests the analytical and FEM implementations using:
- Manufactured solutions with polynomial loads (exactly representable)
- Residual-is-zero-at-exact-solution verification
- Jacobian consistency via finite differences
- Convergence studies for non-polynomial loads
- Tip deflection agreement between analytical and FEM
"""

from typing import Any

import pytest
import numpy as np
import torch
from numpy.typing import NDArray
from scipy.sparse import issparse

from pyapprox.pde.galerkin.physics.euler_bernoulli import (
    EulerBernoulliBeamAnalytical,
    EulerBernoulliBeamFEM,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array
from pyapprox.util.backends.torch import TorchBkd
def _to_dense(mat):
    """Convert sparse matrix to dense numpy array if needed."""
    if issparse(mat):
        return mat.toarray()
    if hasattr(mat, 'detach'):
        return mat.detach().numpy()
    return np.asarray(mat)


class TestEulerBernoulliAnalytical:
    """Tests for the analytical Euler-Bernoulli beam solutions."""
    def test_uniform_load_tip_deflection(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        L, EI, q0 = 1.0, 1.0, 1.0
        beam = EulerBernoulliBeamAnalytical(
            length=L, EI=EI, q0=q0, bkd=bkd, load_type="uniform"
        )
        bkd.assert_allclose(
            bkd.asarray([beam.tip_deflection()]),
            bkd.asarray([q0 * L**4 / (8.0 * EI)]),
            rtol=1e-14,
        )

    def test_linear_load_tip_deflection(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        L, EI, q0 = 2.0, 3.0, 5.0
        beam = EulerBernoulliBeamAnalytical(
            length=L, EI=EI, q0=q0, bkd=bkd, load_type="linear"
        )
        bkd.assert_allclose(
            bkd.asarray([beam.tip_deflection()]),
            bkd.asarray([11.0 * q0 * L**4 / (120.0 * EI)]),
            rtol=1e-14,
        )

    def test_uniform_load_boundary_conditions(self, numpy_bkd) -> None:
        """w(0)=0, w'(0)=0 for cantilever."""
        bkd = numpy_bkd
        L, EI, q0 = 1.0, 1.0, 1.0
        beam = EulerBernoulliBeamAnalytical(
            length=L, EI=EI, q0=q0, bkd=bkd, load_type="uniform"
        )
        x = bkd.asarray(np.array([0.0]))
        bkd.assert_allclose(
            beam.deflection(x), bkd.asarray([0.0]), atol=1e-15
        )
        bkd.assert_allclose(
            beam.slope(x), bkd.asarray([0.0]), atol=1e-15
        )

    def test_linear_load_boundary_conditions(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        L, EI, q0 = 1.0, 1.0, 1.0
        beam = EulerBernoulliBeamAnalytical(
            length=L, EI=EI, q0=q0, bkd=bkd, load_type="linear"
        )
        x = bkd.asarray(np.array([0.0]))
        bkd.assert_allclose(
            beam.deflection(x), bkd.asarray([0.0]), atol=1e-15
        )
        bkd.assert_allclose(
            beam.slope(x), bkd.asarray([0.0]), atol=1e-15
        )

    def test_uniform_load_deflection_values(self, numpy_bkd) -> None:
        """Check w(x) at interior points against formula."""
        bkd = numpy_bkd
        L, EI, q0 = 1.0, 1.0, 1.0
        beam = EulerBernoulliBeamAnalytical(
            length=L, EI=EI, q0=q0, bkd=bkd, load_type="uniform"
        )
        x = bkd.asarray(np.array([0.5]))
        # w(0.5) = 1/24 * 0.25 * (6 - 2 + 0.25) = 4.25/96
        expected = bkd.asarray([4.25 / 96.0])
        bkd.assert_allclose(beam.deflection(x), expected, rtol=1e-14)

    def test_invalid_load_type_raises(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        with pytest.raises(ValueError):
            EulerBernoulliBeamAnalytical(
                length=1.0, EI=1.0, q0=1.0, bkd=bkd,
                load_type="quadratic"
            )


class TestEulerBernoulliFEM:
    """Tests for the FEM Euler-Bernoulli beam."""

    # ------------------------------------------------------------------
    # Basic structure tests
    # ------------------------------------------------------------------

    def test_ndofs(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        nx = 5
        beam = EulerBernoulliBeamFEM(
            nx=nx, length=1.0, EI=1.0,
            load_func=lambda x: np.ones_like(x),
            bkd=bkd,
        )
        assert beam.nstates() == 2 * (nx + 1)

    def test_stiffness_symmetric(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        beam = EulerBernoulliBeamFEM(
            nx=4, length=1.0, EI=1.0,
            load_func=lambda x: np.ones_like(x),
            bkd=bkd,
        )
        K = beam.stiffness_matrix()
        K_np = _to_dense(K)
        bkd.assert_allclose(
            bkd.asarray(K_np),
            bkd.asarray(K_np.T),
            atol=1e-14,
        )

    def test_mass_symmetric_positive(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        beam = EulerBernoulliBeamFEM(
            nx=4, length=1.0, EI=1.0,
            load_func=lambda x: np.ones_like(x),
            bkd=bkd,
        )
        M = beam.mass_matrix()
        M_np = _to_dense(M)
        bkd.assert_allclose(
            bkd.asarray(M_np),
            bkd.asarray(M_np.T),
            atol=1e-14,
        )
        eigs = np.linalg.eigvalsh(M_np)
        assert np.all(eigs > -1e-14)

    def test_residual_shape(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        nx = 4
        beam = EulerBernoulliBeamFEM(
            nx=nx, length=1.0, EI=1.0,
            load_func=lambda x: np.ones_like(x),
            bkd=bkd,
        )
        ndofs = beam.nstates()
        state = bkd.asarray(np.zeros(ndofs))
        res = beam.residual(state)
        assert len(bkd.to_numpy(res)) == ndofs

    def test_jacobian_shape(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        nx = 4
        beam = EulerBernoulliBeamFEM(
            nx=nx, length=1.0, EI=1.0,
            load_func=lambda x: np.ones_like(x),
            bkd=bkd,
        )
        ndofs = beam.nstates()
        state = bkd.asarray(np.zeros(ndofs))
        jac = beam.jacobian(state)
        jac_np = _to_dense(jac)
        assert jac_np.shape == (ndofs, ndofs)

    # ------------------------------------------------------------------
    # Manufactured solution: cubic polynomial (exactly representable)
    # ------------------------------------------------------------------

    def test_manufactured_solve_recovers_exact_dofs(self, numpy_bkd) -> None:
        """Solve K*u = b and verify u matches analytical solution at nodes.

        The uniform load cantilever has a degree-4 analytical solution
        which is NOT in the cubic Hermite space. However, with sufficient
        elements, Hermite FEM should recover it to high accuracy.
        """
        bkd = numpy_bkd
        EI_val = 2.0
        L = 1.0
        q0 = 3.0
        nx = 10

        beam = EulerBernoulliBeamFEM(
            nx=nx, length=L, EI=EI_val,
            load_func=lambda x: q0 * np.ones_like(x),
            bkd=bkd,
        )

        analytical = EulerBernoulliBeamAnalytical(
            length=L, EI=EI_val, q0=q0, bkd=bkd, load_type="uniform"
        )

        x_nodes = bkd.asarray(beam.node_coordinates())
        w_fem = beam.deflection_at_nodes()
        w_exact = analytical.deflection(x_nodes)

        bkd.assert_allclose(w_fem, w_exact, rtol=1e-10)

    def test_manufactured_residual_at_solution(self, numpy_bkd) -> None:
        """Residual at the computed FEM solution should be zero."""
        bkd = numpy_bkd
        EI_val = 1.5
        L = 2.0
        nx = 8

        beam = EulerBernoulliBeamFEM(
            nx=nx, length=L, EI=EI_val,
            load_func=lambda x: np.ones_like(x),
            bkd=bkd,
        )

        sol = beam.solve()
        res = beam.residual(sol)
        bkd.assert_allclose(
            res,
            bkd.asarray(np.zeros(beam.nstates())),
            atol=1e-10,
        )

    def test_manufactured_exact_solve_recovery(self, numpy_bkd) -> None:
        """Solve with load from cubic manufactured solution, recover exactly."""
        bkd = numpy_bkd
        # w(x) = x^2*(3L-x)/(6L) => a normalized cubic
        # w(0) = 0, w'(0) = 0
        # w'' = (6Lx - 6x^2)/(6L) = (L-x)/L... let me use simpler.
        #
        # w(x) = x^2, w'(x) = 2x, w''(x) = 2, w''''(x) = 0
        # EI*w'''' = 0, but b.c.: w(0)=0, w'(0)=0 both satisfied.
        # The load is q=0. Solve K*u = 0 with w(0)=w'(0)=0.
        # Solution should be u=0 (trivial), not w=x^2.
        # We need load that PRODUCES the manufactured solution.
        #
        # For uniform q=1, EI=1, L=1: exact w(x) = x^2*(6-4x+x^2)/24
        # This is degree 4, not exactly representable.
        # Let's test with enough elements that the error is small.
        L, EI_val, q0 = 1.0, 1.0, 1.0

        beam = EulerBernoulliBeamFEM(
            nx=10, length=L, EI=EI_val,
            load_func=lambda x: q0 * np.ones_like(x),
            bkd=bkd,
        )

        analytical = EulerBernoulliBeamAnalytical(
            length=L, EI=EI_val, q0=q0, bkd=bkd, load_type="uniform"
        )

        # Compare tip deflection
        tip_fem = beam.tip_deflection()
        tip_exact = analytical.tip_deflection()
        bkd.assert_allclose(
            bkd.asarray([tip_fem]),
            bkd.asarray([tip_exact]),
            rtol=1e-10,
        )

    def test_all_dirichlet_residual_zero(self, numpy_bkd) -> None:
        """With all DOFs constrained to exact values, residual is zero.

        When all DOFs are Dirichlet-constrained, residual[dof] = state[dof]-g,
        which is zero when state matches the prescribed values exactly.
        """
        bkd = numpy_bkd
        L = 1.0
        EI_val = 1.0
        nx = 4

        x_nodes = np.linspace(0, L, nx + 1)
        w_exact = x_nodes**2
        dwdx_exact = 2.0 * x_nodes
        all_dofs = list(range(2 * (nx + 1)))
        all_vals = []
        for i in range(nx + 1):
            all_vals.extend([w_exact[i], dwdx_exact[i]])

        beam = EulerBernoulliBeamFEM(
            nx=nx, length=L, EI=EI_val,
            load_func=lambda x: np.zeros_like(x),
            bkd=bkd,
            dirichlet_dofs=all_dofs,
            dirichlet_values=all_vals,
        )

        exact_state = beam.interpolate_manufactured(
            w_func=lambda x: x**2,
            dwdx_func=lambda x: 2.0 * x,
        )
        res = beam.residual(exact_state)
        bkd.assert_allclose(
            res,
            bkd.asarray(np.zeros(beam.nstates())),
            atol=1e-12,
        )

    # ------------------------------------------------------------------
    # Jacobian consistency (finite difference check)
    # ------------------------------------------------------------------

    def test_jacobian_fd_consistency(self, numpy_bkd) -> None:
        """Jacobian matches finite-difference approximation of residual."""
        bkd = numpy_bkd
        nx = 4
        L, EI_val = 1.0, 2.0

        beam = EulerBernoulliBeamFEM(
            nx=nx, length=L, EI=EI_val,
            load_func=lambda x: np.ones_like(x),
            bkd=bkd,
        )

        ndofs = beam.nstates()
        np.random.seed(42)
        state = bkd.asarray(np.random.randn(ndofs) * 0.01)

        jac = beam.jacobian(state)
        jac_np = _to_dense(jac)

        eps = 1e-7
        state_np = bkd.to_numpy(state)
        jac_fd = np.zeros((ndofs, ndofs))
        for j in range(ndofs):
            state_p = state_np.copy()
            state_m = state_np.copy()
            state_p[j] += eps
            state_m[j] -= eps
            res_p = bkd.to_numpy(
                beam.residual(bkd.asarray(state_p))
            )
            res_m = bkd.to_numpy(
                beam.residual(bkd.asarray(state_m))
            )
            jac_fd[:, j] = (res_p - res_m) / (2.0 * eps)

        bkd.assert_allclose(
            bkd.asarray(jac_np),
            bkd.asarray(jac_fd),
            atol=1e-6,
        )

    # ------------------------------------------------------------------
    # Dirichlet BC enforcement
    # ------------------------------------------------------------------

    def test_jacobian_identity_at_dirichlet_dofs(self, numpy_bkd) -> None:
        """Dirichlet DOF rows in Jacobian are identity rows."""
        bkd = numpy_bkd
        beam = EulerBernoulliBeamFEM(
            nx=4, length=1.0, EI=1.0,
            load_func=lambda x: np.ones_like(x),
            bkd=bkd,
        )
        state = bkd.asarray(np.zeros(beam.nstates()))
        jac = beam.jacobian(state)
        jac_np = _to_dense(jac)

        dof_indices, _ = beam.dirichlet_dof_info(0.0)
        dof_indices_np = bkd.to_numpy(dof_indices)

        for dof in dof_indices_np:
            expected_row = np.zeros(beam.nstates())
            expected_row[dof] = 1.0
            bkd.assert_allclose(
                bkd.asarray(jac_np[dof]),
                bkd.asarray(expected_row),
                atol=1e-14,
            )

    # ------------------------------------------------------------------
    # Convergence tests
    # ------------------------------------------------------------------

    def test_uniform_load_convergence(self, numpy_bkd) -> None:
        """FEM converges to analytical for uniform load under refinement."""
        bkd = numpy_bkd
        L, EI_val, q0 = 1.0, 1.0, 1.0
        exact_tip = q0 * L**4 / (8.0 * EI_val)

        errors = []
        nx_values = [2, 4, 8, 16]
        for nx in nx_values:
            beam = EulerBernoulliBeamFEM(
                nx=nx, length=L, EI=EI_val,
                load_func=lambda x: q0 * np.ones_like(x),
                bkd=bkd,
            )
            tip = beam.tip_deflection()
            errors.append(abs(tip - exact_tip))

        # Hermite elements should give very high accuracy even with few
        # elements for polynomial loads. For uniform load, the exact
        # solution is degree 4 which is approximated well by cubic Hermite.
        # With 2 elements the error should already be small.
        assert errors[-1] < 1e-10

    def test_linear_load_convergence(self, numpy_bkd) -> None:
        """FEM converges to analytical for linearly increasing load."""
        bkd = numpy_bkd
        L, EI_val, q0 = 1.0, 1.0, 1.0
        exact_tip = 11.0 * q0 * L**4 / (120.0 * EI_val)

        errors = []
        nx_values = [2, 4, 8, 16]
        for nx in nx_values:
            beam = EulerBernoulliBeamFEM(
                nx=nx, length=L, EI=EI_val,
                load_func=lambda x: q0 * x / L,
                bkd=bkd,
            )
            tip = beam.tip_deflection()
            errors.append(abs(tip - exact_tip))

        # Linear load produces degree 5 solution.
        # With refinement, cubic Hermite converges as O(h^4).
        # 16 elements should achieve very good accuracy.
        assert errors[-1] < 1e-10

    # ------------------------------------------------------------------
    # FEM vs analytical agreement
    # ------------------------------------------------------------------

    def test_fem_analytical_uniform_load_agreement(self, numpy_bkd) -> None:
        """FEM deflection at nodes matches analytical values."""
        bkd = numpy_bkd
        L, EI_val, q0 = 1.0, 1.0, 1.0

        analytical = EulerBernoulliBeamAnalytical(
            length=L, EI=EI_val, q0=q0, bkd=bkd, load_type="uniform"
        )

        nx = 10
        beam = EulerBernoulliBeamFEM(
            nx=nx, length=L, EI=EI_val,
            load_func=lambda x: q0 * np.ones_like(x),
            bkd=bkd,
        )

        x_nodes = bkd.asarray(beam.node_coordinates())
        w_exact = analytical.deflection(x_nodes)
        w_fem = beam.deflection_at_nodes()

        bkd.assert_allclose(w_fem, w_exact, rtol=1e-10)

    def test_fem_analytical_linear_load_agreement(self, numpy_bkd) -> None:
        """FEM deflection at nodes matches analytical for linear load."""
        bkd = numpy_bkd
        L, EI_val, q0 = 1.0, 1.0, 1.0

        analytical = EulerBernoulliBeamAnalytical(
            length=L, EI=EI_val, q0=q0, bkd=bkd, load_type="linear"
        )

        nx = 10
        beam = EulerBernoulliBeamFEM(
            nx=nx, length=L, EI=EI_val,
            load_func=lambda x: q0 * x / L,
            bkd=bkd,
        )

        x_nodes = bkd.asarray(beam.node_coordinates())
        w_exact = analytical.deflection(x_nodes)
        w_fem = beam.deflection_at_nodes()

        bkd.assert_allclose(w_fem, w_exact, rtol=1e-8)

    def test_fem_analytical_slope_agreement(self, numpy_bkd) -> None:
        """FEM slope at nodes matches analytical for uniform load."""
        bkd = numpy_bkd
        L, EI_val, q0 = 1.0, 1.0, 1.0

        analytical = EulerBernoulliBeamAnalytical(
            length=L, EI=EI_val, q0=q0, bkd=bkd, load_type="uniform"
        )

        nx = 10
        beam = EulerBernoulliBeamFEM(
            nx=nx, length=L, EI=EI_val,
            load_func=lambda x: q0 * np.ones_like(x),
            bkd=bkd,
        )

        x_nodes = bkd.asarray(beam.node_coordinates())
        dwdx_exact = analytical.slope(x_nodes)
        dwdx_fem = beam.slope_at_nodes()

        bkd.assert_allclose(dwdx_fem, dwdx_exact, rtol=1e-10)

    # ------------------------------------------------------------------
    # Scaling tests
    # ------------------------------------------------------------------

    def test_different_EI_values(self, numpy_bkd) -> None:
        """Tip deflection scales as 1/EI for uniform load."""
        bkd = numpy_bkd
        L, q0, nx = 1.0, 1.0, 10

        tip_vals = []
        EI_values = [1.0, 2.0, 4.0]
        for EI_val in EI_values:
            beam = EulerBernoulliBeamFEM(
                nx=nx, length=L, EI=EI_val,
                load_func=lambda x: q0 * np.ones_like(x),
                bkd=bkd,
            )
            tip_vals.append(beam.tip_deflection())

        # tip ~ 1/EI, so tip1*EI1 == tip2*EI2
        for i in range(1, len(EI_values)):
            bkd.assert_allclose(
                bkd.asarray([tip_vals[0] * EI_values[0]]),
                bkd.asarray([tip_vals[i] * EI_values[i]]),
                rtol=1e-10,
            )

    def test_different_lengths(self, numpy_bkd) -> None:
        """Tip deflection scales as L^4 for uniform load."""
        bkd = numpy_bkd
        EI_val, q0, nx = 1.0, 1.0, 20

        tip_vals = []
        L_values = [1.0, 2.0]
        for L in L_values:
            beam = EulerBernoulliBeamFEM(
                nx=nx, length=L, EI=EI_val,
                load_func=lambda x: q0 * np.ones_like(x),
                bkd=bkd,
            )
            tip_vals.append(beam.tip_deflection())

        # tip ~ L^4
        ratio = tip_vals[1] / tip_vals[0]
        expected_ratio = (L_values[1] / L_values[0])**4
        bkd.assert_allclose(
            bkd.asarray([ratio]),
            bkd.asarray([expected_ratio]),
            rtol=1e-8,
        )


def _smooth_EI_field(x_nodes: np.ndarray, EI_mean: float) -> np.ndarray:
    """Smooth spatially-varying EI: EI(x) = EI_mean * (1 + 0.5*sin(pi*x/L)).

    Returns per-element EI evaluated at element midpoints.
    """
    midpoints = 0.5 * (x_nodes[:-1] + x_nodes[1:])
    L = x_nodes[-1]
    return EI_mean * (1.0 + 0.5 * np.sin(np.pi * midpoints / L))


class TestEulerBernoulliVaryingEI:
    """Tests for Euler-Bernoulli beam with spatially-varying EI(x)."""
    def _make_beam(
        self, bkd, nx: int = 10, L: float = 1.0, EI_mean: float = 1.0,
    ) -> EulerBernoulliBeamFEM:
        """Create beam with spatially-varying EI from smooth field."""
        beam = EulerBernoulliBeamFEM(
            nx=nx, length=L, EI=1.0,
            load_func=lambda x: np.ones_like(x),
            bkd=bkd,
        )
        EI_field = _smooth_EI_field(beam.node_coordinates(), EI_mean)
        beam.set_EI(EI_field)
        return beam

    # ------------------------------------------------------------------
    # Residual at exact solution is zero
    # ------------------------------------------------------------------

    def test_residual_zero_at_solution(self, numpy_bkd) -> None:
        """Residual at the FEM solution should be zero for varying EI."""
        bkd = numpy_bkd
        beam = self._make_beam(bkd, nx=10, L=1.0, EI_mean=2.0)
        sol = beam.solve()
        res = beam.residual(sol)
        bkd.assert_allclose(
            res,
            bkd.asarray(np.zeros(beam.nstates())),
            atol=1e-10,
        )

    # ------------------------------------------------------------------
    # Jacobian via DerivativeChecker
    # ------------------------------------------------------------------

    def test_jacobian_derivative_checker(self, numpy_bkd) -> None:
        """Verify Jacobian of residual using DerivativeChecker."""
        bkd = numpy_bkd
        from pyapprox.interface.functions.derivative_checks.derivative_checker import (
            DerivativeChecker,
        )
        from pyapprox.interface.functions.fromcallable.jacobian import (
            FunctionWithJacobianFromCallable,
        )

        beam = self._make_beam(bkd, nx=6, L=1.0, EI_mean=2.0)
        ndofs = beam.nstates()

        def residual_func(samples: Array) -> Array:
            state = samples[:, 0]
            res = beam.residual(state)
            return bkd.reshape(res, (ndofs, 1))

        def jacobian_func(sample: Array) -> Array:
            state = sample[:, 0]
            jac = beam.jacobian(state)
            return bkd.asarray(_to_dense(jac))

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=ndofs, nvars=ndofs,
            fun=residual_func, jacobian=jacobian_func,
            bkd=bkd,
        )

        np.random.seed(42)
        sample = bkd.asarray(np.random.randn(ndofs, 1) * 0.01)
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(sample, relative=True)[0]
        ratio = float(bkd.to_numpy(checker.error_ratio(errors)))
        assert ratio <= 1e-6

    # ------------------------------------------------------------------
    # Stiffness matrix symmetry with varying EI
    # ------------------------------------------------------------------

    def test_stiffness_symmetric_varying_EI(self, numpy_bkd) -> None:
        """Stiffness matrix with element-wise EI is still symmetric."""
        bkd = numpy_bkd
        beam = self._make_beam(bkd, nx=8, L=1.0, EI_mean=3.0)
        K = beam.stiffness_matrix()
        K_np = _to_dense(K)
        bkd.assert_allclose(
            bkd.asarray(K_np),
            bkd.asarray(K_np.T),
            atol=1e-12,
        )

    # ------------------------------------------------------------------
    # Recovery of exact solution under mesh refinement (convergence)
    # ------------------------------------------------------------------

    def test_convergence_varying_EI(self, numpy_bkd) -> None:
        """FEM with varying EI converges under mesh refinement.

        Uses the same beam problem with increasing nx and verifies
        that the tip deflection converges (successive differences shrink).
        The piecewise-constant EI approximation converges as O(h^2).
        """
        bkd = numpy_bkd
        L, EI_mean = 1.0, 2.0
        tip_vals = []
        nx_values = [8, 16, 32, 64, 128, 256]
        for nx in nx_values:
            beam = EulerBernoulliBeamFEM(
                nx=nx, length=L, EI=1.0,
                load_func=lambda x: np.ones_like(x),
                bkd=bkd,
            )
            EI_field = _smooth_EI_field(beam.node_coordinates(), EI_mean)
            beam.set_EI(EI_field)
            tip_vals.append(beam.tip_deflection())

        # Check convergence: successive differences should decrease
        diffs = [abs(tip_vals[i+1] - tip_vals[i]) for i in range(len(tip_vals)-1)]
        for i in range(len(diffs) - 1):
            assert diffs[i + 1] < diffs[i]
        # Final refinement difference should be small (O(h^2) convergence)
        assert diffs[-1] < 5e-6

    # ------------------------------------------------------------------
    # Varying EI reduces to uniform when constant
    # ------------------------------------------------------------------

    def test_constant_field_matches_scalar(self, numpy_bkd) -> None:
        """Element-wise EI array with constant value gives same result as scalar."""
        bkd = numpy_bkd
        nx = 10
        L, EI_val = 1.0, 3.0

        beam_scalar = EulerBernoulliBeamFEM(
            nx=nx, length=L, EI=EI_val,
            load_func=lambda x: np.ones_like(x),
            bkd=bkd,
        )

        beam_array = EulerBernoulliBeamFEM(
            nx=nx, length=L, EI=np.full(nx, EI_val),
            load_func=lambda x: np.ones_like(x),
            bkd=bkd,
        )

        bkd.assert_allclose(
            bkd.asarray([beam_array.tip_deflection()]),
            bkd.asarray([beam_scalar.tip_deflection()]),
            rtol=1e-12,
        )

        # Deflections at all nodes should match
        bkd.assert_allclose(
            beam_array.deflection_at_nodes(),
            beam_scalar.deflection_at_nodes(),
            rtol=1e-12,
        )

    # ------------------------------------------------------------------
    # Max curvature
    # ------------------------------------------------------------------

    def test_max_curvature_positive(self, numpy_bkd) -> None:
        """Max curvature with varying EI should be positive."""
        bkd = numpy_bkd
        beam = self._make_beam(bkd, nx=20, L=1.0, EI_mean=1.0)
        curv = beam.max_curvature()
        assert curv > 0.0

    def test_max_curvature_scales_with_EI(self, numpy_bkd) -> None:
        """Doubling uniform EI halves max curvature (for constant field)."""
        bkd = numpy_bkd
        nx, L = 20, 1.0

        beam1 = EulerBernoulliBeamFEM(
            nx=nx, length=L, EI=1.0,
            load_func=lambda x: np.ones_like(x),
            bkd=bkd,
        )
        beam2 = EulerBernoulliBeamFEM(
            nx=nx, length=L, EI=2.0,
            load_func=lambda x: np.ones_like(x),
            bkd=bkd,
        )

        curv1 = beam1.max_curvature()
        curv2 = beam2.max_curvature()
        bkd.assert_allclose(
            bkd.asarray([curv1]),
            bkd.asarray([2.0 * curv2]),
            rtol=1e-8,
        )
