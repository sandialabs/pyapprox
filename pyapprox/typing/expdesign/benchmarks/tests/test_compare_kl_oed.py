"""
Comparison: legacy vs typing KL-OED objective and solver.

Compares KL-OED objective values, Jacobians, and optimal designs
between legacy and typing implementations using the linear Gaussian
benchmark where the exact EIG is available for verification.

Both implementations use the same quadrature data (shapes, latent
samples) generated from the same random seed to ensure comparability.
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401


def _run_kl_comparison(bkd: Backend):
    """Run legacy vs typing KL-OED comparison.

    Uses Monte Carlo quadrature with the same seed for both systems.
    """
    nobs = 5
    degree = 2
    min_degree = 0
    noise_std = 0.5
    prior_std = 0.5
    nouter = 500
    ninner = 200

    results = {}

    # ================================================================
    # Legacy setup
    # ================================================================
    from pyapprox.util.backends.numpy import NumpyMixin
    from pyapprox.expdesign.bayesoed_benchmarks import (
        LinearGaussianBayesianOEDBenchmark,
    )
    from pyapprox.expdesign.bayesoed import (
        IndependentGaussianOEDInnerLoopLogLikelihood as LegacyInnerLike,
        KLBayesianOED as LegacyKLOED,
    )

    legacy_bm = LinearGaussianBayesianOEDBenchmark(
        nobs, min_degree, degree, noise_std, prior_std, NumpyMixin,
    )
    legacy_obs_model = legacy_bm.get_observation_model()
    legacy_prior = legacy_bm.get_prior()
    nparams = legacy_obs_model.nvars()

    # ================================================================
    # Generate shared quadrature data
    # ================================================================
    np.random.seed(42)

    # Outer loop: joint samples from [prior params, latent noise]
    theta_outer_np = np.random.randn(nparams, nouter) * prior_std
    latent_np = np.random.randn(nobs, nouter)
    outloop_samples_np = np.vstack([theta_outer_np, latent_np])

    # Inner loop: prior samples only
    theta_inner_np = np.random.randn(nparams, ninner) * prior_std

    # Compute forward model outputs (shapes)
    # Legacy model: input (nvars, nsamp), output (nsamp, nqoi)
    legacy_outloop_shapes = legacy_obs_model(theta_outer_np).T  # (nobs, nouter)
    legacy_inloop_shapes = legacy_obs_model(theta_inner_np).T   # (nobs, ninner)

    # ================================================================
    # Typing setup
    # ================================================================
    from pyapprox.typing.expdesign.benchmarks import (
        LinearGaussianOEDBenchmark,
    )

    typing_bm = LinearGaussianOEDBenchmark(
        nobs, degree, noise_std, prior_std, bkd, min_degree=min_degree,
    )

    # Use typing observation_model to compute shapes
    typing_obs_model = typing_bm.observation_model()
    typing_outloop_shapes = typing_obs_model(
        bkd.asarray(theta_outer_np),
    )  # (nobs, nouter)
    typing_inloop_shapes = typing_obs_model(
        bkd.asarray(theta_inner_np),
    )  # (nobs, ninner)

    # Verify shapes match
    results["shapes_match"] = np.allclose(
        bkd.to_numpy(typing_outloop_shapes),
        np.asarray(legacy_outloop_shapes),
        atol=1e-12,
    )

    # ================================================================
    # Create legacy KL-OED objective
    # ================================================================
    noise_diag = NumpyMixin.full((nobs, 1), noise_std**2)
    legacy_inloop_loglike = LegacyInnerLike(noise_diag, backend=NumpyMixin)
    legacy_kl_oed = LegacyKLOED(legacy_inloop_loglike)

    outloop_weights = NumpyMixin.full((nouter, 1), 1.0 / nouter)
    inloop_weights = NumpyMixin.full((ninner, 1), 1.0 / ninner)

    legacy_kl_oed.set_data(
        legacy_outloop_shapes,
        outloop_samples_np,
        outloop_weights,
        legacy_inloop_shapes,
        inloop_weights,
    )
    legacy_obj = legacy_kl_oed.objective()

    # ================================================================
    # Create typing KL-OED objective
    # ================================================================
    from pyapprox.typing.expdesign.objective import (
        create_kl_oed_objective_from_data,
    )

    latent_typing = bkd.asarray(latent_np)
    noise_var_typing = bkd.full((nobs,), noise_std**2)

    typing_obj = create_kl_oed_objective_from_data(
        noise_var_typing,
        typing_outloop_shapes,
        typing_inloop_shapes,
        latent_typing,
        bkd,
    )

    # ================================================================
    # 1. Compare objective values at random weights
    # ================================================================
    np.random.seed(123)
    obj_diffs = []
    jac_diffs = []
    for _ in range(5):
        raw = np.abs(np.random.randn(nobs)) + 0.01
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

    results["obj_max_diff"] = max(obj_diffs)
    results["jac_max_diff"] = max(jac_diffs)

    # ================================================================
    # 2. Compare EIG at uniform weights vs analytical
    # ================================================================
    w_uniform_np = np.ones(nobs) / nobs
    w_uniform_legacy = NumpyMixin.asarray(w_uniform_np)[:, None]
    w_uniform_typing = bkd.reshape(bkd.asarray(w_uniform_np), (nobs, 1))

    legacy_eig_mc = -float(legacy_obj(w_uniform_legacy)[0, 0])
    typing_eig_mc = typing_obj.expected_information_gain(w_uniform_typing)
    exact_eig = typing_bm.exact_eig(w_uniform_typing)

    results["legacy_eig_mc"] = legacy_eig_mc
    results["typing_eig_mc"] = typing_eig_mc
    results["exact_eig"] = float(exact_eig)
    results["legacy_mc_vs_exact_ratio"] = abs(legacy_eig_mc - exact_eig) / abs(exact_eig)
    results["typing_mc_vs_exact_ratio"] = abs(typing_eig_mc - exact_eig) / abs(exact_eig)

    # ================================================================
    # 3. Compare optimal designs
    # ================================================================
    from scipy.optimize import minimize, LinearConstraint as ScipyLC

    def _solve_legacy_kl(obj_fn, n):
        x0 = np.ones(n) / n
        def fun(w):
            return float(obj_fn(w[:, None])[0, 0])
        def jac(w):
            j = obj_fn.jacobian(w[:, None])
            return np.asarray(j).flatten()
        res = minimize(
            fun, x0, jac=jac, method="trust-constr",
            bounds=[(1e-6, 1.0)] * n,
            constraints=ScipyLC(np.ones((1, n)), 1.0, 1.0),
            options={"maxiter": 500, "gtol": 1e-7},
        )
        return res.x

    legacy_opt_w = _solve_legacy_kl(legacy_obj, nobs)

    from pyapprox.typing.expdesign.solver import (
        RelaxedKLOEDSolver,
        RelaxedOEDConfig,
    )
    typing_solver = RelaxedKLOEDSolver(
        typing_obj, RelaxedOEDConfig(maxiter=500),
    )
    typing_opt_w, typing_opt_eig = typing_solver.solve()
    typing_opt_w_np = bkd.to_numpy(typing_opt_w).flatten()

    results["legacy_opt_weights"] = legacy_opt_w
    results["typing_opt_weights"] = typing_opt_w_np

    # Compare EIG at optimal
    legacy_opt_eig = -float(legacy_obj(legacy_opt_w[:, None])[0, 0])
    results["legacy_opt_eig"] = legacy_opt_eig
    results["typing_opt_eig"] = typing_opt_eig

    # Verify both optimal EIGs are close to exact at those weights
    exact_eig_at_legacy = float(typing_bm.exact_eig(
        bkd.reshape(bkd.asarray(legacy_opt_w), (nobs, 1)),
    ))
    exact_eig_at_typing = float(typing_bm.exact_eig(typing_opt_w))

    results["exact_eig_at_legacy_opt"] = exact_eig_at_legacy
    results["exact_eig_at_typing_opt"] = exact_eig_at_typing

    return results


class TestKLOEDComparison(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        self._results = _run_kl_comparison(self._bkd)

    def test_shapes_match(self):
        self.assertTrue(self._results["shapes_match"])

    def test_objective_values_match(self):
        self._bkd.assert_allclose(
            self._bkd.asarray([self._results["obj_max_diff"]]),
            self._bkd.asarray([0.0]),
            atol=1e-10,
        )

    def test_jacobians_match(self):
        self._bkd.assert_allclose(
            self._bkd.asarray([self._results["jac_max_diff"]]),
            self._bkd.asarray([0.0]),
            atol=1e-10,
        )

    def test_mc_eig_matches_between_implementations(self):
        self._bkd.assert_allclose(
            self._bkd.asarray([self._results["legacy_eig_mc"]]),
            self._bkd.asarray([self._results["typing_eig_mc"]]),
            rtol=1e-10,
        )

    def test_mc_eig_close_to_exact(self):
        # MC estimates should be within ~20% of exact for 500 outer samples
        self.assertLess(self._results["legacy_mc_vs_exact_ratio"], 0.3)
        self.assertLess(self._results["typing_mc_vs_exact_ratio"], 0.3)

    def test_optimal_eig_matches(self):
        self._bkd.assert_allclose(
            self._bkd.asarray([self._results["legacy_opt_eig"]]),
            self._bkd.asarray([self._results["typing_opt_eig"]]),
            rtol=1e-4,
        )

    def test_exact_eig_at_optimal_close(self):
        # Both optimizers should find designs with similar exact EIG
        self._bkd.assert_allclose(
            self._bkd.asarray([self._results["exact_eig_at_legacy_opt"]]),
            self._bkd.asarray([self._results["exact_eig_at_typing_opt"]]),
            rtol=1e-2,
        )


class TestKLOEDComparisonNumpy(TestKLOEDComparison[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestKLOEDComparisonTorch(TestKLOEDComparison[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()
