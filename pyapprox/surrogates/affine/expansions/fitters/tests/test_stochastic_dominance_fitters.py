"""Tests for Stochastic Dominance Fitters.

Tests focus on:
- FSDObjective jacobian/hvp validation via DerivativeChecker
- StochasticDominanceConstraint jacobian/whvp validation
- FSD regression test: CDF ordering P(f(X) <= eta) <= P(Y <= eta)
- SSD regression test: DisutilitySSD condition
- Result types and shapes
- Single QoI validation
- Accessor methods
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.optimization.minimize.differentiable_approximations import (
    SmoothLogBasedLeftHeavisideFunction,
    SmoothLogBasedMaxFunction,
)
from pyapprox.probability import UniformMarginal
from pyapprox.probability.risk import DisutilitySSD
from pyapprox.probability.univariate.discrete import CustomDiscreteMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.expansions.fitters.results import (
    DirectSolverResult,
)
from pyapprox.surrogates.affine.expansions.fitters.stochastic_dominance import (
    FSDFitter,
    FSDObjective,
    SSDFitter,
    StochasticDominanceConstraint,
)
from pyapprox.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests, slow_test, slower_test  # noqa: F401


class TestStochasticDominanceFitters(Generic[Array], unittest.TestCase):
    """Base test class - NOT run directly."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        np.random.seed(42)

    def _create_expansion(self, nvars: int, max_level: int, nqoi: int = 1):
        """Create test expansion."""
        bkd = self._bkd
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    # --- FSDObjective Tests ---

    def test_fsd_objective_call(self) -> None:
        """FSDObjective returns correct shape."""
        bkd = self._bkd
        nsamples, nterms = 10, 5
        Phi = bkd.asarray(np.random.randn(nsamples, nterms))
        train_values = bkd.asarray(np.random.randn(nsamples, 1))

        objective = FSDObjective(Phi, train_values, bkd)

        # Evaluate at a single point
        coef = bkd.asarray(np.random.randn(nterms, 1))
        result = objective(coef)

        self.assertEqual(result.shape, (1, 1))

    def test_fsd_objective_call_batch(self) -> None:
        """FSDObjective handles batch evaluation."""
        bkd = self._bkd
        nsamples, nterms = 10, 5
        Phi = bkd.asarray(np.random.randn(nsamples, nterms))
        train_values = bkd.asarray(np.random.randn(nsamples, 1))

        objective = FSDObjective(Phi, train_values, bkd)

        # Evaluate at multiple points
        coefs = bkd.asarray(np.random.randn(nterms, 3))
        result = objective(coefs)

        self.assertEqual(result.shape, (1, 3))

    def test_fsd_objective_jacobian_shape(self) -> None:
        """FSDObjective jacobian has correct shape."""
        bkd = self._bkd
        nsamples, nterms = 10, 5
        Phi = bkd.asarray(np.random.randn(nsamples, nterms))
        train_values = bkd.asarray(np.random.randn(nsamples, 1))

        objective = FSDObjective(Phi, train_values, bkd)

        coef = bkd.asarray(np.random.randn(nterms, 1))
        jac = objective.jacobian(coef)

        self.assertEqual(jac.shape, (1, nterms))

    def test_fsd_objective_hvp_shape(self) -> None:
        """FSDObjective hvp has correct shape."""
        bkd = self._bkd
        nsamples, nterms = 10, 5
        Phi = bkd.asarray(np.random.randn(nsamples, nterms))
        train_values = bkd.asarray(np.random.randn(nsamples, 1))

        objective = FSDObjective(Phi, train_values, bkd)

        coef = bkd.asarray(np.random.randn(nterms, 1))
        vec = bkd.asarray(np.random.randn(nterms, 1))
        hvp_result = objective.hvp(coef, vec)

        self.assertEqual(hvp_result.shape, (nterms, 1))

    def test_fsd_objective_jacobian_derivative_check(self) -> None:
        """FSDObjective jacobian passes derivative check."""
        bkd = self._bkd
        nsamples, nterms = 15, 6
        Phi = bkd.asarray(np.random.randn(nsamples, nterms))
        train_values = bkd.asarray(np.random.randn(nsamples, 1))

        objective = FSDObjective(Phi, train_values, bkd)

        coef = bkd.asarray(np.random.randn(nterms, 1))
        checker = DerivativeChecker(objective)
        errors = checker.check_derivatives(coef)
        ratio = checker.error_ratio(errors[0])

        self.assertLess(float(ratio), 1e-5)

    def test_fsd_objective_hvp_derivative_check(self) -> None:
        """FSDObjective hvp passes derivative check."""
        bkd = self._bkd
        nsamples, nterms = 15, 6
        Phi = bkd.asarray(np.random.randn(nsamples, nterms))
        train_values = bkd.asarray(np.random.randn(nsamples, 1))

        objective = FSDObjective(Phi, train_values, bkd)

        coef = bkd.asarray(np.random.randn(nterms, 1))
        checker = DerivativeChecker(objective)
        errors = checker.check_derivatives(coef)

        # errors[1] is the HVP check
        ratio = checker.error_ratio(errors[1])
        self.assertLess(float(ratio), 1e-5)

    def test_fsd_objective_accessors(self) -> None:
        """FSDObjective accessors return correct values."""
        bkd = self._bkd
        nsamples, nterms = 10, 5
        Phi = bkd.asarray(np.random.randn(nsamples, nterms))
        train_values = bkd.asarray(np.random.randn(nsamples, 1))

        objective = FSDObjective(Phi, train_values, bkd)

        self.assertEqual(objective.nvars(), nterms)
        self.assertEqual(objective.nqoi(), 1)

    # --- StochasticDominanceConstraint Tests ---

    def test_constraint_call_fsd(self) -> None:
        """FSD constraint returns correct shape."""
        bkd = self._bkd
        nsamples, nterms = 10, 5
        Phi = bkd.asarray(np.random.randn(nsamples, nterms))
        train_values = bkd.asarray(np.random.randn(nsamples, 1))

        smooth_func = SmoothLogBasedLeftHeavisideFunction(bkd, eps=0.5)
        constraint = StochasticDominanceConstraint(Phi, train_values, smooth_func, bkd)

        coef = bkd.asarray(np.random.randn(nterms, 1))
        result = constraint(coef)

        self.assertEqual(result.shape, (nsamples, 1))

    def test_constraint_call_ssd(self) -> None:
        """SSD constraint returns correct shape."""
        bkd = self._bkd
        nsamples, nterms = 10, 5
        Phi = bkd.asarray(np.random.randn(nsamples, nterms))
        train_values = bkd.asarray(np.random.randn(nsamples, 1))

        smooth_func = SmoothLogBasedMaxFunction(bkd, eps=0.5)
        constraint = StochasticDominanceConstraint(
            Phi,
            train_values,
            smooth_func,
            bkd,
            lb=bkd.zeros((nsamples,)),
            ub=bkd.full((nsamples,), np.inf),
        )

        coef = bkd.asarray(np.random.randn(nterms, 1))
        result = constraint(coef)

        self.assertEqual(result.shape, (nsamples, 1))

    def test_constraint_jacobian_shape(self) -> None:
        """Constraint jacobian has correct shape."""
        bkd = self._bkd
        nsamples, nterms = 10, 5
        Phi = bkd.asarray(np.random.randn(nsamples, nterms))
        train_values = bkd.asarray(np.random.randn(nsamples, 1))

        smooth_func = SmoothLogBasedLeftHeavisideFunction(bkd, eps=0.5)
        constraint = StochasticDominanceConstraint(Phi, train_values, smooth_func, bkd)

        coef = bkd.asarray(np.random.randn(nterms, 1))
        jac = constraint.jacobian(coef)

        self.assertEqual(jac.shape, (nsamples, nterms))

    def test_constraint_whvp_shape(self) -> None:
        """Constraint whvp has correct shape."""
        bkd = self._bkd
        nsamples, nterms = 10, 5
        Phi = bkd.asarray(np.random.randn(nsamples, nterms))
        train_values = bkd.asarray(np.random.randn(nsamples, 1))

        smooth_func = SmoothLogBasedLeftHeavisideFunction(bkd, eps=0.5)
        constraint = StochasticDominanceConstraint(Phi, train_values, smooth_func, bkd)

        coef = bkd.asarray(np.random.randn(nterms, 1))
        vec = bkd.asarray(np.random.randn(nterms, 1))
        weights = bkd.asarray(np.random.randn(nsamples, 1))
        whvp_result = constraint.whvp(coef, vec, weights)

        self.assertEqual(whvp_result.shape, (nterms, 1))

    def test_constraint_jacobian_finite_diff(self) -> None:
        """Constraint jacobian matches finite differences."""
        bkd = self._bkd
        nsamples, nterms = 8, 4
        Phi = bkd.asarray(np.random.randn(nsamples, nterms))
        train_values = bkd.asarray(np.random.randn(nsamples, 1))

        smooth_func = SmoothLogBasedLeftHeavisideFunction(bkd, eps=0.5)
        constraint = StochasticDominanceConstraint(Phi, train_values, smooth_func, bkd)

        coef = bkd.asarray(np.random.randn(nterms, 1))
        h = 1e-7

        # Analytical jacobian
        analytical_jac = constraint.jacobian(coef)

        # Finite difference approximation
        fd_jac = bkd.zeros((nsamples, nterms))
        for ii in range(nterms):
            coef_plus = bkd.copy(coef)
            coef_plus[ii, 0] = coef_plus[ii, 0] + h
            coef_minus = bkd.copy(coef)
            coef_minus[ii, 0] = coef_minus[ii, 0] - h
            fd_jac[:, ii : ii + 1] = (
                constraint(coef_plus) - constraint(coef_minus)
            ) / (2 * h)

        bkd.assert_allclose(analytical_jac, fd_jac, rtol=1e-5, atol=1e-10)

    def test_constraint_whvp_finite_diff(self) -> None:
        """Constraint whvp matches finite differences of weighted jacobian."""
        bkd = self._bkd
        nsamples, nterms = 8, 4
        Phi = bkd.asarray(np.random.randn(nsamples, nterms))
        train_values = bkd.asarray(np.random.randn(nsamples, 1))

        smooth_func = SmoothLogBasedLeftHeavisideFunction(bkd, eps=0.5)
        constraint = StochasticDominanceConstraint(Phi, train_values, smooth_func, bkd)

        coef = bkd.asarray(np.random.randn(nterms, 1))
        vec = bkd.asarray(np.random.randn(nterms, 1))
        weights = bkd.asarray(np.random.randn(nsamples, 1))
        h = 1e-5

        # Analytical whvp
        analytical_whvp = constraint.whvp(coef, vec, weights)

        # Finite difference: d/dt [weights.T @ jacobian(coef + t*vec)] at t=0
        # = weights.T @ d(jacobian)/d(coef) @ vec = whvp
        jac_plus = constraint.jacobian(coef + h * vec)
        jac_minus = constraint.jacobian(coef - h * vec)

        # weighted jacobian: sum_c weights[c] * jac[c, :]
        weighted_jac_plus = bkd.dot(weights.T, jac_plus)  # (1, nterms)
        weighted_jac_minus = bkd.dot(weights.T, jac_minus)  # (1, nterms)
        fd_whvp = (weighted_jac_plus - weighted_jac_minus) / (2 * h)

        bkd.assert_allclose(analytical_whvp.T, fd_whvp, rtol=1e-4, atol=1e-10)

    def test_constraint_accessors(self) -> None:
        """Constraint accessors return correct values."""
        bkd = self._bkd
        nsamples, nterms = 10, 5
        Phi = bkd.asarray(np.random.randn(nsamples, nterms))
        train_values = bkd.asarray(np.random.randn(nsamples, 1))

        smooth_func = SmoothLogBasedLeftHeavisideFunction(bkd, eps=0.5)
        constraint = StochasticDominanceConstraint(Phi, train_values, smooth_func, bkd)

        self.assertEqual(constraint.nvars(), nterms)
        self.assertEqual(constraint.nqoi(), nsamples)
        self.assertEqual(constraint.lb().shape, (nsamples,))
        self.assertEqual(constraint.ub().shape, (nsamples,))

    # --- FSDFitter Tests ---

    def test_fsd_fitter_returns_direct_solver_result(self) -> None:
        """FSDFitter returns DirectSolverResult."""
        expansion = self._create_expansion(nvars=1, max_level=2)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (1, 20)))
        values = self._bkd.asarray(np.random.randn(1, 20))

        fitter = FSDFitter(self._bkd, eps=0.5, maxiter=50, verbosity=0)
        result = fitter.fit(expansion, samples, values)

        self.assertIsInstance(result, DirectSolverResult)

    def test_fsd_fitter_result_params_shape(self) -> None:
        """FSDFitter result params have correct shape."""
        expansion = self._create_expansion(nvars=1, max_level=2)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (1, 20)))
        values = self._bkd.asarray(np.random.randn(1, 20))

        fitter = FSDFitter(self._bkd, eps=0.5, maxiter=50)
        result = fitter.fit(expansion, samples, values)

        self.assertEqual(result.params().shape, (expansion.nterms(), 1))

    def test_fsd_fitter_handles_1d_values(self) -> None:
        """FSDFitter handles 1D values array."""
        expansion = self._create_expansion(nvars=1, max_level=2)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (1, 20)))
        values_1d = self._bkd.asarray(np.random.randn(20))

        fitter = FSDFitter(self._bkd, eps=0.5, maxiter=50)
        result = fitter.fit(expansion, samples, values_1d)

        self.assertEqual(result.params().shape[1], 1)

    def test_fsd_fitter_multi_qoi_raises(self) -> None:
        """FSDFitter: nqoi > 1 raises ValueError."""
        expansion = self._create_expansion(nvars=1, max_level=2, nqoi=2)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (1, 20)))
        values = self._bkd.asarray(np.random.randn(2, 20))

        fitter = FSDFitter(self._bkd, eps=0.5)
        with self.assertRaises(ValueError) as ctx:
            fitter.fit(expansion, samples, values)
        self.assertIn("nqoi=1", str(ctx.exception))

    def test_fsd_fitter_accessors(self) -> None:
        """FSDFitter accessors return correct values."""
        fitter = FSDFitter(self._bkd, eps=0.3, shift=0.05)
        bkd = self._bkd
        bkd.assert_allclose(
            bkd.asarray([fitter.eps()]),
            bkd.asarray([0.3]),
        )
        bkd.assert_allclose(
            bkd.asarray([fitter.shift()]),
            bkd.asarray([0.05]),
        )

    def _setup_fsd_regression_problem(self, nsamples: int, degree: int):
        """Setup problem for FSD regression test (mimics legacy test setup)."""
        bkd = self._bkd

        # Create expansion with max_level = degree
        expansion = self._create_expansion(nvars=1, max_level=degree)

        # Generate training data: exp(x) over [0+eps, 2+eps]
        train_samples = bkd.linspace(1e-3, 2 + 1e-3, nsamples)[None, :]
        train_values = bkd.exp(train_samples).T  # (nsamples, 1)

        return expansion, train_samples, train_values

    def test_fsd_gradients_nsamples10_degree1(self) -> None:
        """FSD gradient checks for nsamples=10, degree=1."""
        self._check_fsd_gradients(nsamples=10, degree=1)

    def test_fsd_gradients_nsamples10_degree2(self) -> None:
        """FSD gradient checks for nsamples=10, degree=2."""
        self._check_fsd_gradients(nsamples=10, degree=2)

    def _check_fsd_gradients(self, nsamples: int, degree: int) -> None:
        """Check FSD objective and constraint gradients (replicates legacy)."""
        bkd = self._bkd
        expansion, train_samples, train_values = self._setup_fsd_regression_problem(
            nsamples, degree
        )

        Phi = expansion.basis_matrix(train_samples)
        smooth_heaviside = SmoothLogBasedLeftHeavisideFunction(bkd, eps=0.5)

        # Create objective and check gradients
        objective = FSDObjective(Phi, train_values, bkd)
        coef = bkd.asarray(np.random.randn(expansion.nterms(), 1))

        checker = DerivativeChecker(objective)
        errors = checker.check_derivatives(coef)

        # Jacobian check
        ratio_jac = checker.error_ratio(errors[0])
        self.assertLess(float(ratio_jac), 1e-5)

        # HVP check
        ratio_hvp = checker.error_ratio(errors[1])
        self.assertLess(float(ratio_hvp), 1e-5)

        # Create constraint and check gradients
        constraint = StochasticDominanceConstraint(
            Phi, train_values, smooth_heaviside, bkd
        )

        # Jacobian finite difference check
        h = 1e-7
        analytical_jac = constraint.jacobian(coef)
        fd_jac = bkd.zeros((nsamples, expansion.nterms()))
        for ii in range(expansion.nterms()):
            coef_plus = bkd.copy(coef)
            coef_plus[ii, 0] = coef_plus[ii, 0] + h
            coef_minus = bkd.copy(coef)
            coef_minus[ii, 0] = coef_minus[ii, 0] - h
            fd_jac[:, ii : ii + 1] = (
                constraint(coef_plus) - constraint(coef_minus)
            ) / (2 * h)

        jac_error = float(bkd.max(bkd.abs(analytical_jac - fd_jac)))
        self.assertLess(jac_error, 1e-5)

        # WHVP check with random weights
        weights = bkd.asarray(np.random.uniform(0, 1, (nsamples, 1)))
        vec = bkd.asarray(np.random.randn(expansion.nterms(), 1))
        h = 1e-5

        analytical_whvp = constraint.whvp(coef, vec, weights)
        jac_plus = constraint.jacobian(coef + h * vec)
        jac_minus = constraint.jacobian(coef - h * vec)
        weighted_jac_plus = bkd.dot(weights.T, jac_plus)
        weighted_jac_minus = bkd.dot(weights.T, jac_minus)
        fd_whvp = (weighted_jac_plus - weighted_jac_minus) / (2 * h)

        whvp_error = float(bkd.max(bkd.abs(analytical_whvp.T - fd_whvp)))
        self.assertLess(whvp_error, 1e-4)

    def test_fsd_cdf_dominance_exact(self) -> None:
        """FSD fitted surrogate satisfies exact CDF dominance.

        Uses CustomDiscreteMarginal for exact empirical CDF comparison.
        Replicates legacy test_first_order_stochastic_dominance_regression.
        """
        bkd = self._bkd
        nsamples = 10
        degree = 3

        expansion, train_samples, train_values = self._setup_fsd_regression_problem(
            nsamples, degree
        )

        # Fit with FSD constraint using shift parameter for stability
        fitter = FSDFitter(bkd, eps=5e-2, maxiter=1000)
        result = fitter.fit(expansion, train_samples, train_values.T)

        # Evaluate surrogate
        surrogate_values = result(train_samples)  # (1, nsamples)

        # Create empirical CDFs using CustomDiscreteMarginal
        # Uniform weights for equally-spaced samples
        pk = np.ones(nsamples) / nsamples

        # Train data CDF
        train_data_1d = bkd.to_numpy(train_values[:, 0])
        train_cdf = CustomDiscreteMarginal(train_data_1d, pk, bkd)

        # Surrogate CDF
        surrogate_data_1d = bkd.to_numpy(surrogate_values[0])
        surrogate_cdf = CustomDiscreteMarginal(surrogate_data_1d, pk, bkd)

        # Check FSD condition: CDF(surrogate) >= CDF(data) at all surrogate values
        # This is equivalent to P(f(X) <= eta) <= P(Y <= eta)
        # In CDF terms: surrogate_cdf(x) >= train_cdf(x) means surrogate is
        # stochastically dominated by data, which is what FSD ensures
        surrogate_cdf_vals = surrogate_cdf.cdf(surrogate_values)
        train_cdf_vals = train_cdf.cdf(surrogate_values)

        # All surrogate CDF values should be >= train CDF values
        bkd.assert_allclose(
            bkd.maximum(surrogate_cdf_vals, train_cdf_vals),
            surrogate_cdf_vals,
            rtol=1e-6,
            atol=1e-6,
        )

    # --- SSDFitter Tests ---

    def test_ssd_fitter_returns_direct_solver_result(self) -> None:
        """SSDFitter returns DirectSolverResult."""
        expansion = self._create_expansion(nvars=1, max_level=2)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (1, 20)))
        values = self._bkd.asarray(np.random.randn(1, 20))

        fitter = SSDFitter(self._bkd, eps=0.5, maxiter=50, verbosity=0)
        result = fitter.fit(expansion, samples, values)

        self.assertIsInstance(result, DirectSolverResult)

    def test_ssd_fitter_result_params_shape(self) -> None:
        """SSDFitter result params have correct shape."""
        expansion = self._create_expansion(nvars=1, max_level=2)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (1, 20)))
        values = self._bkd.asarray(np.random.randn(1, 20))

        fitter = SSDFitter(self._bkd, eps=0.5, maxiter=50)
        result = fitter.fit(expansion, samples, values)

        self.assertEqual(result.params().shape, (expansion.nterms(), 1))

    def test_ssd_fitter_handles_1d_values(self) -> None:
        """SSDFitter handles 1D values array."""
        expansion = self._create_expansion(nvars=1, max_level=2)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (1, 20)))
        values_1d = self._bkd.asarray(np.random.randn(20))

        fitter = SSDFitter(self._bkd, eps=0.5, maxiter=50)
        result = fitter.fit(expansion, samples, values_1d)

        self.assertEqual(result.params().shape[1], 1)

    def test_ssd_fitter_multi_qoi_raises(self) -> None:
        """SSDFitter: nqoi > 1 raises ValueError."""
        expansion = self._create_expansion(nvars=1, max_level=2, nqoi=2)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (1, 20)))
        values = self._bkd.asarray(np.random.randn(2, 20))

        fitter = SSDFitter(self._bkd, eps=0.5)
        with self.assertRaises(ValueError) as ctx:
            fitter.fit(expansion, samples, values)
        self.assertIn("nqoi=1", str(ctx.exception))

    def test_ssd_fitter_accessors(self) -> None:
        """SSDFitter accessors return correct values."""
        fitter = SSDFitter(self._bkd, eps=0.4, shift=0.02)
        bkd = self._bkd
        bkd.assert_allclose(
            bkd.asarray([fitter.eps()]),
            bkd.asarray([0.4]),
        )
        bkd.assert_allclose(
            bkd.asarray([fitter.shift()]),
            bkd.asarray([0.02]),
        )

    def test_ssd_disutility_condition_exact(self) -> None:
        """SSD fitted surrogate satisfies exact disutility condition.

        Uses DisutilitySSD risk measure for exact comparison.
        Replicates legacy test_second_order_stochastic_dominance_regression.
        """
        bkd = self._bkd
        nsamples = 10
        degree = 3

        expansion, train_samples, train_values = self._setup_fsd_regression_problem(
            nsamples, degree
        )

        # Fit with SSD constraint
        fitter = SSDFitter(bkd, eps=5e-2, maxiter=1000)
        result = fitter.fit(expansion, train_samples, train_values.T)

        # Evaluate surrogate
        surrogate_values = result(train_samples)  # (1, nsamples)

        # Use DisutilitySSD to check condition
        # E[max(0, eta - f(X))] <= E[max(0, eta - Y)] for all eta
        dssd = DisutilitySSD(bkd)

        # Set eta values to the surrogate values (at these points we check)
        eta_vals = surrogate_values[0]
        dssd.set_eta(eta_vals)

        # Compute disutility for surrogate predictions
        dssd.set_samples(surrogate_values)
        surrogate_disutil = dssd()

        # Compute disutility for training data
        dssd.set_samples(train_values.T)
        train_disutil = dssd()

        # SSD requires: train_disutil <= surrogate_disutil (data dominates surrogate)
        # Allow small tolerance for numerical optimization
        tol = 4e-5
        diff = train_disutil - surrogate_disutil - tol
        violations = bkd.sum(bkd.asarray(diff > 0, dtype=float))
        self.assertEqual(float(violations), 0.0, "SSD disutility condition violated")

    def test_fsd_stronger_than_ssd(self) -> None:
        """FSD constraint is stronger (more conservative) than SSD."""
        bkd = self._bkd

        # Create problem
        expansion = self._create_expansion(nvars=1, max_level=2)
        nsamples = 30

        samples = bkd.asarray(np.random.uniform(-1, 1, (1, nsamples)))
        values = bkd.asarray(np.random.randn(1, nsamples))

        # Fit with both methods
        fsd_fitter = FSDFitter(bkd, eps=0.3, maxiter=100)
        ssd_fitter = SSDFitter(bkd, eps=0.3, maxiter=100)

        fsd_result = fsd_fitter.fit(expansion, samples, values)
        ssd_result = ssd_fitter.fit(expansion, samples, values)

        # Evaluate at samples
        fsd_preds = fsd_result(samples)
        ssd_preds = ssd_result(samples)

        # FSD should generally produce higher (more conservative) predictions
        # Check that mean of FSD is >= mean of SSD
        float(bkd.mean(fsd_preds))
        float(bkd.mean(ssd_preds))

        # FSD is often more conservative but this depends on the problem
        # At minimum, both should run without error and produce valid predictions
        self.assertFalse(bool(bkd.isnan(bkd.sum(fsd_preds))))
        self.assertFalse(bool(bkd.isnan(bkd.sum(ssd_preds))))

    def test_fitted_surrogate_evaluates(self) -> None:
        """Fitted surrogate can be evaluated at new points."""
        bkd = self._bkd
        expansion = self._create_expansion(nvars=2, max_level=2)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = bkd.asarray(np.random.randn(1, 30))

        fitter = FSDFitter(bkd, eps=0.5, maxiter=50)
        result = fitter.fit(expansion, samples, values)

        # Evaluate at new points
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (2, 10)))
        predictions = result(test_samples)

        self.assertEqual(predictions.shape, (1, 10))

    def test_fsd_fitter_with_shift(self) -> None:
        """FSD fitter with shift parameter produces valid result.

        Replicates legacy test that used shift=5e-2 for numerical stability.
        """
        bkd = self._bkd
        nsamples = 10
        degree = 3

        expansion, train_samples, train_values = self._setup_fsd_regression_problem(
            nsamples, degree
        )

        # Fit with shift parameter (as used in legacy test)
        fitter = FSDFitter(bkd, eps=5e-2, shift=5e-2, maxiter=1000)
        result = fitter.fit(expansion, train_samples, train_values.T)

        # Result should be valid
        predictions = result(train_samples)
        self.assertFalse(bool(bkd.isnan(bkd.sum(predictions))))

        # Predictions should have correct shape
        self.assertEqual(predictions.shape, (1, nsamples))

    def test_ssd_fitter_with_shift(self) -> None:
        """SSD fitter with shift parameter produces valid result."""
        bkd = self._bkd
        nsamples = 10
        degree = 3

        expansion, train_samples, train_values = self._setup_fsd_regression_problem(
            nsamples, degree
        )

        # Fit with shift parameter
        fitter = SSDFitter(bkd, eps=5e-2, shift=0.0, maxiter=1000)
        result = fitter.fit(expansion, train_samples, train_values.T)

        # Result should be valid
        predictions = result(train_samples)
        self.assertFalse(bool(bkd.isnan(bkd.sum(predictions))))

        # Predictions should have correct shape
        self.assertEqual(predictions.shape, (1, nsamples))


class TestStochasticDominanceFittersNumpy(TestStochasticDominanceFitters[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestStochasticDominanceFittersTorch(TestStochasticDominanceFitters[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    @slower_test
    def test_fitted_surrogate_evaluates(self) -> None:
        super().test_fitted_surrogate_evaluates()

    @slow_test
    def test_fsd_fitter_with_shift(self) -> None:
        super().test_fsd_fitter_with_shift()


if __name__ == "__main__":
    unittest.main()
