"""
Comparison: legacy vs typing D-optimal linear model OED.

Compares design matrices, objective values, Jacobians, exact EIG,
and optimal weights between legacy and typing implementations for
the linear Gaussian benchmark.
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.optimize import minimize, LinearConstraint

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401


def _solve_legacy_d_optimal(legacy_obj, nobs):
    """Solve D-optimal design using scipy directly on legacy objective."""
    x0 = np.ones(nobs) / nobs

    def fun(w):
        val = legacy_obj(w[:, None])
        return float(val[0, 0])

    def jac(w):
        j = legacy_obj.jacobian(w[:, None])
        return np.asarray(j).flatten()

    result = minimize(
        fun, x0, jac=jac, method="trust-constr",
        bounds=[(0.0, 1.0)] * nobs,
        constraints=LinearConstraint(np.ones((1, nobs)), 1.0, 1.0),
        options={"maxiter": 1000, "gtol": 1e-8},
    )
    return result.x


def _run_comparison(bkd: Backend):
    """Run full legacy vs typing D-optimal comparison."""
    nobs = 5
    degree = 2
    min_degree = 0
    noise_std = 0.5
    prior_std = 0.5

    results = {}

    # ================================================================
    # Legacy setup
    # ================================================================
    from pyapprox.util.backends.numpy import NumpyMixin
    from pyapprox.expdesign.bayesoed_benchmarks import (
        LinearGaussianBayesianOEDBenchmark,
    )
    from pyapprox.expdesign.bayesoed import (
        DOptimalLinearModelObjective as LegacyDOptObj,
    )

    legacy_bm = LinearGaussianBayesianOEDBenchmark(
        nobs, min_degree, degree, noise_std, prior_std, NumpyMixin,
    )
    legacy_obs_model = legacy_bm.get_observation_model()
    legacy_obj = LegacyDOptObj(
        legacy_obs_model,
        NumpyMixin.asarray(noise_std**2),
        NumpyMixin.asarray(prior_std**2),
    )

    # ================================================================
    # Typing setup
    # ================================================================
    from pyapprox.typing.expdesign.benchmarks import (
        LinearGaussianOEDBenchmark,
    )
    from pyapprox.typing.expdesign.objective import (
        DOptimalLinearModelObjective as TypingDOptObj,
    )

    typing_bm = LinearGaussianOEDBenchmark(
        nobs, degree, noise_std, prior_std, bkd, min_degree=min_degree,
    )
    typing_obj = TypingDOptObj(
        typing_bm.design_matrix(),
        bkd.asarray(noise_std**2),
        bkd.asarray(prior_std**2),
        bkd,
    )

    # ================================================================
    # 1. Compare design matrices
    # ================================================================
    legacy_A = np.asarray(legacy_obs_model._matrix)
    typing_A = bkd.to_numpy(typing_bm.design_matrix())

    results["design_matrix_match"] = np.allclose(legacy_A, typing_A, atol=1e-14)
    results["design_matrix_diff"] = float(np.max(np.abs(legacy_A - typing_A)))

    # ================================================================
    # 2. Compare objective values and Jacobians at random weights
    # ================================================================
    np.random.seed(42)
    obj_diffs = []
    jac_diffs = []
    for _ in range(5):
        raw = np.abs(np.random.randn(nobs))
        w_np = raw / raw.sum()

        w_legacy = NumpyMixin.asarray(w_np)[:, None]
        w_typing = bkd.reshape(bkd.asarray(w_np), (nobs, 1))

        legacy_val = float(legacy_obj(w_legacy)[0, 0])
        typing_val = float(bkd.to_numpy(typing_obj(w_typing))[0, 0])
        obj_diffs.append(abs(legacy_val - typing_val))

        legacy_jac = np.asarray(legacy_obj.jacobian(w_legacy))
        if legacy_jac.ndim == 1:
            legacy_jac = legacy_jac[None, :]
        typing_jac = bkd.to_numpy(typing_obj.jacobian(w_typing))
        jac_diffs.append(float(np.max(np.abs(legacy_jac - typing_jac))))

    # Also test uniform weights
    w_np = np.ones(nobs) / nobs
    w_legacy = NumpyMixin.asarray(w_np)[:, None]
    w_typing = bkd.reshape(bkd.asarray(w_np), (nobs, 1))
    legacy_val = float(legacy_obj(w_legacy)[0, 0])
    typing_val = float(bkd.to_numpy(typing_obj(w_typing))[0, 0])
    obj_diffs.append(abs(legacy_val - typing_val))

    results["obj_max_diff"] = max(obj_diffs)
    results["jac_max_diff"] = max(jac_diffs)

    # ================================================================
    # 3. Compare exact EIG
    # ================================================================
    legacy_eig = legacy_bm.exact_expected_information_gain(w_legacy)
    typing_eig = typing_bm.exact_eig(w_typing)

    results["eig_legacy"] = float(legacy_eig)
    results["eig_typing"] = float(typing_eig)
    results["eig_diff"] = abs(float(legacy_eig) - float(typing_eig))

    # D-optimal obj should equal -EIG
    typing_obj_val = float(bkd.to_numpy(typing_obj(w_typing))[0, 0])
    results["d_opt_equals_neg_eig"] = abs(typing_obj_val - (-float(typing_eig)))

    # ================================================================
    # 4. Compare optimal designs (scipy on legacy, RelaxedOEDSolver on typing)
    # ================================================================
    legacy_opt_w = _solve_legacy_d_optimal(legacy_obj, nobs)

    from pyapprox.typing.expdesign.solver import (
        RelaxedOEDSolver,
        RelaxedOEDConfig,
    )
    typing_solver = RelaxedOEDSolver(
        typing_obj, RelaxedOEDConfig(maxiter=1000),
    )
    typing_opt_w, _ = typing_solver.solve()
    typing_opt_w_np = bkd.to_numpy(typing_opt_w).flatten()

    results["legacy_opt_weights"] = legacy_opt_w
    results["typing_opt_weights"] = typing_opt_w_np

    # Compare EIG at optimal weights (better than comparing weights directly
    # since multiple weight vectors may achieve similar EIG)
    legacy_eig_opt = legacy_bm.exact_expected_information_gain(
        NumpyMixin.asarray(legacy_opt_w)[:, None],
    )
    typing_eig_opt = typing_bm.exact_eig(
        bkd.reshape(bkd.asarray(typing_opt_w_np), (nobs, 1)),
    )
    results["legacy_eig_at_opt"] = float(legacy_eig_opt)
    results["typing_eig_at_opt"] = float(typing_eig_opt)
    results["eig_at_opt_diff"] = abs(
        float(legacy_eig_opt) - float(typing_eig_opt)
    )
    results["opt_weights_diff"] = float(
        np.max(np.abs(legacy_opt_w - typing_opt_w_np))
    )

    return results


class TestDOptimalComparison(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        self._results = _run_comparison(self._bkd)

    def test_design_matrices_match(self):
        self.assertTrue(
            self._results["design_matrix_match"],
            f"Design matrix diff: {self._results['design_matrix_diff']}",
        )

    def test_objective_values_match(self):
        self._bkd.assert_allclose(
            self._bkd.asarray([self._results["obj_max_diff"]]),
            self._bkd.asarray([0.0]),
            atol=1e-12,
        )

    def test_jacobians_match(self):
        self._bkd.assert_allclose(
            self._bkd.asarray([self._results["jac_max_diff"]]),
            self._bkd.asarray([0.0]),
            atol=1e-12,
        )

    def test_exact_eig_match(self):
        self._bkd.assert_allclose(
            self._bkd.asarray([self._results["eig_legacy"]]),
            self._bkd.asarray([self._results["eig_typing"]]),
            rtol=1e-10,
        )

    def test_d_opt_equals_neg_eig(self):
        self._bkd.assert_allclose(
            self._bkd.asarray([self._results["d_opt_equals_neg_eig"]]),
            self._bkd.asarray([0.0]),
            atol=1e-10,
        )

    def test_optimal_weights_match(self):
        # Near-zero weights are sensitive to optimizer numerics, so we
        # compare with loose tolerance. The EIG at optimal test verifies
        # both solutions achieve the same information gain.
        self._bkd.assert_allclose(
            self._bkd.asarray(self._results["legacy_opt_weights"]),
            self._bkd.asarray(self._results["typing_opt_weights"]),
            atol=5e-4,
        )

    def test_eig_at_optimal_match(self):
        self._bkd.assert_allclose(
            self._bkd.asarray([self._results["legacy_eig_at_opt"]]),
            self._bkd.asarray([self._results["typing_eig_at_opt"]]),
            rtol=1e-4,
        )


class TestDOptimalComparisonNumpy(TestDOptimalComparison[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestDOptimalComparisonTorch(TestDOptimalComparison[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()
