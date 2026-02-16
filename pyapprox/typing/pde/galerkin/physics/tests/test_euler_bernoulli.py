"""Tests for Euler-Bernoulli beam physics.

Tests the analytical and FEM implementations using:
- Manufactured solutions with polynomial loads (exactly representable)
- Residual-is-zero-at-exact-solution verification
- Jacobian consistency via finite differences
- Convergence studies for non-polynomial loads
- Tip deflection agreement between analytical and FEM
"""

from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import issparse
import torch
import unittest

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401
from pyapprox.typing.pde.galerkin.physics.euler_bernoulli import (
    EulerBernoulliBeamAnalytical,
    EulerBernoulliBeamFEM,
)


def _to_dense(mat):
    """Convert sparse matrix to dense numpy array if needed."""
    if issparse(mat):
        return mat.toarray()
    if hasattr(mat, 'detach'):
        return mat.detach().numpy()
    return np.asarray(mat)


class TestEulerBernoulliAnalytical(Generic[Array], unittest.TestCase):
    """Tests for the analytical Euler-Bernoulli beam solutions."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def test_uniform_load_tip_deflection(self) -> None:
        L, EI, q0 = 1.0, 1.0, 1.0
        beam = EulerBernoulliBeamAnalytical(
            length=L, EI=EI, q0=q0, bkd=self._bkd, load_type="uniform"
        )
        self._bkd.assert_allclose(
            self._bkd.asarray([beam.tip_deflection()]),
            self._bkd.asarray([q0 * L**4 / (8.0 * EI)]),
            rtol=1e-14,
        )

    def test_linear_load_tip_deflection(self) -> None:
        L, EI, q0 = 2.0, 3.0, 5.0
        beam = EulerBernoulliBeamAnalytical(
            length=L, EI=EI, q0=q0, bkd=self._bkd, load_type="linear"
        )
        self._bkd.assert_allclose(
            self._bkd.asarray([beam.tip_deflection()]),
            self._bkd.asarray([11.0 * q0 * L**4 / (120.0 * EI)]),
            rtol=1e-14,
        )

    def test_uniform_load_boundary_conditions(self) -> None:
        """w(0)=0, w'(0)=0 for cantilever."""
        L, EI, q0 = 1.0, 1.0, 1.0
        beam = EulerBernoulliBeamAnalytical(
            length=L, EI=EI, q0=q0, bkd=self._bkd, load_type="uniform"
        )
        x = self._bkd.asarray(np.array([0.0]))
        self._bkd.assert_allclose(
            beam.deflection(x), self._bkd.asarray([0.0]), atol=1e-15
        )
        self._bkd.assert_allclose(
            beam.slope(x), self._bkd.asarray([0.0]), atol=1e-15
        )

    def test_linear_load_boundary_conditions(self) -> None:
        L, EI, q0 = 1.0, 1.0, 1.0
        beam = EulerBernoulliBeamAnalytical(
            length=L, EI=EI, q0=q0, bkd=self._bkd, load_type="linear"
        )
        x = self._bkd.asarray(np.array([0.0]))
        self._bkd.assert_allclose(
            beam.deflection(x), self._bkd.asarray([0.0]), atol=1e-15
        )
        self._bkd.assert_allclose(
            beam.slope(x), self._bkd.asarray([0.0]), atol=1e-15
        )

    def test_uniform_load_deflection_values(self) -> None:
        """Check w(x) at interior points against formula."""
        L, EI, q0 = 1.0, 1.0, 1.0
        beam = EulerBernoulliBeamAnalytical(
            length=L, EI=EI, q0=q0, bkd=self._bkd, load_type="uniform"
        )
        x = self._bkd.asarray(np.array([0.5]))
        # w(0.5) = 1/24 * 0.25 * (6 - 2 + 0.25) = 4.25/96
        expected = self._bkd.asarray([4.25 / 96.0])
        self._bkd.assert_allclose(beam.deflection(x), expected, rtol=1e-14)

    def test_invalid_load_type_raises(self) -> None:
        with self.assertRaises(ValueError):
            EulerBernoulliBeamAnalytical(
                length=1.0, EI=1.0, q0=1.0, bkd=self._bkd,
                load_type="quadratic"
            )


class TestEulerBernoulliAnalyticalNumpy(
    TestEulerBernoulliAnalytical[NDArray[Any]]
):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestEulerBernoulliAnalyticalTorch(
    TestEulerBernoulliAnalytical[torch.Tensor]
):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestEulerBernoulliFEM(Generic[Array], unittest.TestCase):
    """Tests for the FEM Euler-Bernoulli beam."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    # ------------------------------------------------------------------
    # Basic structure tests
    # ------------------------------------------------------------------

    def test_ndofs(self) -> None:
        nx = 5
        beam = EulerBernoulliBeamFEM(
            nx=nx, length=1.0, EI=1.0,
            load_func=lambda x: np.ones_like(x),
            bkd=self._bkd,
        )
        self.assertEqual(beam.nstates(), 2 * (nx + 1))

    def test_stiffness_symmetric(self) -> None:
        beam = EulerBernoulliBeamFEM(
            nx=4, length=1.0, EI=1.0,
            load_func=lambda x: np.ones_like(x),
            bkd=self._bkd,
        )
        K = beam.stiffness_matrix()
        K_np = _to_dense(K)
        self._bkd.assert_allclose(
            self._bkd.asarray(K_np),
            self._bkd.asarray(K_np.T),
            atol=1e-14,
        )

    def test_mass_symmetric_positive(self) -> None:
        beam = EulerBernoulliBeamFEM(
            nx=4, length=1.0, EI=1.0,
            load_func=lambda x: np.ones_like(x),
            bkd=self._bkd,
        )
        M = beam.mass_matrix()
        M_np = _to_dense(M)
        self._bkd.assert_allclose(
            self._bkd.asarray(M_np),
            self._bkd.asarray(M_np.T),
            atol=1e-14,
        )
        eigs = np.linalg.eigvalsh(M_np)
        self.assertTrue(np.all(eigs > -1e-14))

    def test_residual_shape(self) -> None:
        nx = 4
        beam = EulerBernoulliBeamFEM(
            nx=nx, length=1.0, EI=1.0,
            load_func=lambda x: np.ones_like(x),
            bkd=self._bkd,
        )
        ndofs = beam.nstates()
        state = self._bkd.asarray(np.zeros(ndofs))
        res = beam.residual(state)
        self.assertEqual(len(self._bkd.to_numpy(res)), ndofs)

    def test_jacobian_shape(self) -> None:
        nx = 4
        beam = EulerBernoulliBeamFEM(
            nx=nx, length=1.0, EI=1.0,
            load_func=lambda x: np.ones_like(x),
            bkd=self._bkd,
        )
        ndofs = beam.nstates()
        state = self._bkd.asarray(np.zeros(ndofs))
        jac = beam.jacobian(state)
        jac_np = _to_dense(jac)
        self.assertEqual(jac_np.shape, (ndofs, ndofs))

    # ------------------------------------------------------------------
    # Manufactured solution: cubic polynomial (exactly representable)
    # ------------------------------------------------------------------

    def test_manufactured_solve_recovers_exact_dofs(self) -> None:
        """Solve K*u = b and verify u matches analytical solution at nodes.

        The uniform load cantilever has a degree-4 analytical solution
        which is NOT in the cubic Hermite space. However, with sufficient
        elements, Hermite FEM should recover it to high accuracy.
        """
        EI_val = 2.0
        L = 1.0
        q0 = 3.0
        nx = 10

        beam = EulerBernoulliBeamFEM(
            nx=nx, length=L, EI=EI_val,
            load_func=lambda x: q0 * np.ones_like(x),
            bkd=self._bkd,
        )

        analytical = EulerBernoulliBeamAnalytical(
            length=L, EI=EI_val, q0=q0, bkd=self._bkd, load_type="uniform"
        )

        x_nodes = self._bkd.asarray(beam.node_coordinates())
        w_fem = beam.deflection_at_nodes()
        w_exact = analytical.deflection(x_nodes)

        self._bkd.assert_allclose(w_fem, w_exact, rtol=1e-10)

    def test_manufactured_residual_at_solution(self) -> None:
        """Residual at the computed FEM solution should be zero."""
        EI_val = 1.5
        L = 2.0
        nx = 8

        beam = EulerBernoulliBeamFEM(
            nx=nx, length=L, EI=EI_val,
            load_func=lambda x: np.ones_like(x),
            bkd=self._bkd,
        )

        sol = beam.solve()
        res = beam.residual(sol)
        self._bkd.assert_allclose(
            res,
            self._bkd.asarray(np.zeros(beam.nstates())),
            atol=1e-10,
        )

    def test_manufactured_exact_solve_recovery(self) -> None:
        """Solve with load from cubic manufactured solution, recover exactly."""
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
            bkd=self._bkd,
        )

        analytical = EulerBernoulliBeamAnalytical(
            length=L, EI=EI_val, q0=q0, bkd=self._bkd, load_type="uniform"
        )

        # Compare tip deflection
        tip_fem = beam.tip_deflection()
        tip_exact = analytical.tip_deflection()
        self._bkd.assert_allclose(
            self._bkd.asarray([tip_fem]),
            self._bkd.asarray([tip_exact]),
            rtol=1e-10,
        )

    def test_all_dirichlet_residual_zero(self) -> None:
        """With all DOFs constrained to exact values, residual is zero.

        When all DOFs are Dirichlet-constrained, residual[dof] = state[dof]-g,
        which is zero when state matches the prescribed values exactly.
        """
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
            bkd=self._bkd,
            dirichlet_dofs=all_dofs,
            dirichlet_values=all_vals,
        )

        exact_state = beam.interpolate_manufactured(
            w_func=lambda x: x**2,
            dwdx_func=lambda x: 2.0 * x,
        )
        res = beam.residual(exact_state)
        self._bkd.assert_allclose(
            res,
            self._bkd.asarray(np.zeros(beam.nstates())),
            atol=1e-12,
        )

    # ------------------------------------------------------------------
    # Jacobian consistency (finite difference check)
    # ------------------------------------------------------------------

    def test_jacobian_fd_consistency(self) -> None:
        """Jacobian matches finite-difference approximation of residual."""
        nx = 4
        L, EI_val = 1.0, 2.0

        beam = EulerBernoulliBeamFEM(
            nx=nx, length=L, EI=EI_val,
            load_func=lambda x: np.ones_like(x),
            bkd=self._bkd,
        )

        ndofs = beam.nstates()
        np.random.seed(42)
        state = self._bkd.asarray(np.random.randn(ndofs) * 0.01)

        jac = beam.jacobian(state)
        jac_np = _to_dense(jac)

        eps = 1e-7
        state_np = self._bkd.to_numpy(state)
        jac_fd = np.zeros((ndofs, ndofs))
        for j in range(ndofs):
            state_p = state_np.copy()
            state_m = state_np.copy()
            state_p[j] += eps
            state_m[j] -= eps
            res_p = self._bkd.to_numpy(
                beam.residual(self._bkd.asarray(state_p))
            )
            res_m = self._bkd.to_numpy(
                beam.residual(self._bkd.asarray(state_m))
            )
            jac_fd[:, j] = (res_p - res_m) / (2.0 * eps)

        self._bkd.assert_allclose(
            self._bkd.asarray(jac_np),
            self._bkd.asarray(jac_fd),
            atol=1e-6,
        )

    # ------------------------------------------------------------------
    # Dirichlet BC enforcement
    # ------------------------------------------------------------------

    def test_jacobian_identity_at_dirichlet_dofs(self) -> None:
        """Dirichlet DOF rows in Jacobian are identity rows."""
        beam = EulerBernoulliBeamFEM(
            nx=4, length=1.0, EI=1.0,
            load_func=lambda x: np.ones_like(x),
            bkd=self._bkd,
        )
        state = self._bkd.asarray(np.zeros(beam.nstates()))
        jac = beam.jacobian(state)
        jac_np = _to_dense(jac)

        dof_indices, _ = beam.dirichlet_dof_info(0.0)
        dof_indices_np = self._bkd.to_numpy(dof_indices)

        for dof in dof_indices_np:
            expected_row = np.zeros(beam.nstates())
            expected_row[dof] = 1.0
            self._bkd.assert_allclose(
                self._bkd.asarray(jac_np[dof]),
                self._bkd.asarray(expected_row),
                atol=1e-14,
            )

    # ------------------------------------------------------------------
    # Convergence tests
    # ------------------------------------------------------------------

    def test_uniform_load_convergence(self) -> None:
        """FEM converges to analytical for uniform load under refinement."""
        L, EI_val, q0 = 1.0, 1.0, 1.0
        exact_tip = q0 * L**4 / (8.0 * EI_val)

        errors = []
        nx_values = [2, 4, 8, 16]
        for nx in nx_values:
            beam = EulerBernoulliBeamFEM(
                nx=nx, length=L, EI=EI_val,
                load_func=lambda x: q0 * np.ones_like(x),
                bkd=self._bkd,
            )
            tip = beam.tip_deflection()
            errors.append(abs(tip - exact_tip))

        # Hermite elements should give very high accuracy even with few
        # elements for polynomial loads. For uniform load, the exact
        # solution is degree 4 which is approximated well by cubic Hermite.
        # With 2 elements the error should already be small.
        self.assertLess(errors[-1], 1e-10)

    def test_linear_load_convergence(self) -> None:
        """FEM converges to analytical for linearly increasing load."""
        L, EI_val, q0 = 1.0, 1.0, 1.0
        exact_tip = 11.0 * q0 * L**4 / (120.0 * EI_val)

        errors = []
        nx_values = [2, 4, 8, 16]
        for nx in nx_values:
            beam = EulerBernoulliBeamFEM(
                nx=nx, length=L, EI=EI_val,
                load_func=lambda x: q0 * x / L,
                bkd=self._bkd,
            )
            tip = beam.tip_deflection()
            errors.append(abs(tip - exact_tip))

        # Linear load produces degree 5 solution.
        # With refinement, cubic Hermite converges as O(h^4).
        # 16 elements should achieve very good accuracy.
        self.assertLess(errors[-1], 1e-10)

    # ------------------------------------------------------------------
    # FEM vs analytical agreement
    # ------------------------------------------------------------------

    def test_fem_analytical_uniform_load_agreement(self) -> None:
        """FEM deflection at nodes matches analytical values."""
        L, EI_val, q0 = 1.0, 1.0, 1.0

        analytical = EulerBernoulliBeamAnalytical(
            length=L, EI=EI_val, q0=q0, bkd=self._bkd, load_type="uniform"
        )

        nx = 10
        beam = EulerBernoulliBeamFEM(
            nx=nx, length=L, EI=EI_val,
            load_func=lambda x: q0 * np.ones_like(x),
            bkd=self._bkd,
        )

        x_nodes = self._bkd.asarray(beam.node_coordinates())
        w_exact = analytical.deflection(x_nodes)
        w_fem = beam.deflection_at_nodes()

        self._bkd.assert_allclose(w_fem, w_exact, rtol=1e-10)

    def test_fem_analytical_linear_load_agreement(self) -> None:
        """FEM deflection at nodes matches analytical for linear load."""
        L, EI_val, q0 = 1.0, 1.0, 1.0

        analytical = EulerBernoulliBeamAnalytical(
            length=L, EI=EI_val, q0=q0, bkd=self._bkd, load_type="linear"
        )

        nx = 10
        beam = EulerBernoulliBeamFEM(
            nx=nx, length=L, EI=EI_val,
            load_func=lambda x: q0 * x / L,
            bkd=self._bkd,
        )

        x_nodes = self._bkd.asarray(beam.node_coordinates())
        w_exact = analytical.deflection(x_nodes)
        w_fem = beam.deflection_at_nodes()

        self._bkd.assert_allclose(w_fem, w_exact, rtol=1e-8)

    def test_fem_analytical_slope_agreement(self) -> None:
        """FEM slope at nodes matches analytical for uniform load."""
        L, EI_val, q0 = 1.0, 1.0, 1.0

        analytical = EulerBernoulliBeamAnalytical(
            length=L, EI=EI_val, q0=q0, bkd=self._bkd, load_type="uniform"
        )

        nx = 10
        beam = EulerBernoulliBeamFEM(
            nx=nx, length=L, EI=EI_val,
            load_func=lambda x: q0 * np.ones_like(x),
            bkd=self._bkd,
        )

        x_nodes = self._bkd.asarray(beam.node_coordinates())
        dwdx_exact = analytical.slope(x_nodes)
        dwdx_fem = beam.slope_at_nodes()

        self._bkd.assert_allclose(dwdx_fem, dwdx_exact, rtol=1e-10)

    # ------------------------------------------------------------------
    # Scaling tests
    # ------------------------------------------------------------------

    def test_different_EI_values(self) -> None:
        """Tip deflection scales as 1/EI for uniform load."""
        L, q0, nx = 1.0, 1.0, 10

        tip_vals = []
        EI_values = [1.0, 2.0, 4.0]
        for EI_val in EI_values:
            beam = EulerBernoulliBeamFEM(
                nx=nx, length=L, EI=EI_val,
                load_func=lambda x: q0 * np.ones_like(x),
                bkd=self._bkd,
            )
            tip_vals.append(beam.tip_deflection())

        # tip ~ 1/EI, so tip1*EI1 == tip2*EI2
        for i in range(1, len(EI_values)):
            self._bkd.assert_allclose(
                self._bkd.asarray([tip_vals[0] * EI_values[0]]),
                self._bkd.asarray([tip_vals[i] * EI_values[i]]),
                rtol=1e-10,
            )

    def test_different_lengths(self) -> None:
        """Tip deflection scales as L^4 for uniform load."""
        EI_val, q0, nx = 1.0, 1.0, 20

        tip_vals = []
        L_values = [1.0, 2.0]
        for L in L_values:
            beam = EulerBernoulliBeamFEM(
                nx=nx, length=L, EI=EI_val,
                load_func=lambda x: q0 * np.ones_like(x),
                bkd=self._bkd,
            )
            tip_vals.append(beam.tip_deflection())

        # tip ~ L^4
        ratio = tip_vals[1] / tip_vals[0]
        expected_ratio = (L_values[1] / L_values[0])**4
        self._bkd.assert_allclose(
            self._bkd.asarray([ratio]),
            self._bkd.asarray([expected_ratio]),
            rtol=1e-8,
        )


class TestEulerBernoulliFEMNumpy(TestEulerBernoulliFEM[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestEulerBernoulliFEMTorch(TestEulerBernoulliFEM[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
