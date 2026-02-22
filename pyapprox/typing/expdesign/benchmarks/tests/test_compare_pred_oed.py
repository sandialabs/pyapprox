"""
Comparison: legacy vs typing prediction OED objective and solver.

Compares prediction OED objective values (std deviation measure,
mean risk) between legacy and typing implementations using the
LinearGaussianPredOED and NonLinearGaussianOED benchmarks.

Both implementations use the same quadrature data generated from
the same random seed to ensure comparability.
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


def _run_pred_comparison(bkd: Backend, use_nonlinear: bool = False):
    """Run legacy vs typing prediction OED comparison.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    use_nonlinear : bool
        If True, use nonlinear (exponential) QoI benchmark.
        If False, use linear QoI benchmark.
    """
    nobs = 5
    degree = 2
    min_degree = 0
    noise_std = 0.5
    prior_std = 0.5
    npred = 3
    nouter = 300
    ninner = 150

    results = {}

    # ================================================================
    # Legacy setup
    # ================================================================
    from pyapprox.util.backends.numpy import NumpyMixin
    from pyapprox.expdesign.bayesoed_benchmarks import (
        LinearGaussianBayesianOEDForPredictionBenchmark,
        ExponentialQoIModel,
    )
    from pyapprox.expdesign.bayesoed import (
        IndependentGaussianOEDInnerLoopLogLikelihood as LegacyInnerLike,
        BayesianOEDForPrediction as LegacyPredOED,
        NoiseStatistic,
        PredictionOEDDeviationMeasure,
    )
    from pyapprox.optimization.sampleaverage import SampleAverageMean

    legacy_bm = LinearGaussianBayesianOEDForPredictionBenchmark(
        nobs, min_degree, degree, noise_std, prior_std, npred, NumpyMixin,
    )
    legacy_obs_model = legacy_bm.get_observation_model()
    legacy_prior = legacy_bm.get_prior()
    nparams = legacy_obs_model.nvars()

    if use_nonlinear:
        legacy_qoi_model = ExponentialQoIModel(legacy_bm.get_qoi_model())
    else:
        legacy_qoi_model = legacy_bm.get_qoi_model()

    # ================================================================
    # Generate shared quadrature data
    # ================================================================
    np.random.seed(42)
    theta_outer_np = np.random.randn(nparams, nouter) * prior_std
    latent_np = np.random.randn(nobs, nouter)
    outloop_samples_np = np.vstack([theta_outer_np, latent_np])
    theta_inner_np = np.random.randn(nparams, ninner) * prior_std

    # Legacy forward model shapes: output (nsamp, nqoi), transpose to (nobs, nsamp)
    legacy_outloop_shapes = legacy_obs_model(theta_outer_np).T
    legacy_inloop_shapes = legacy_obs_model(theta_inner_np).T
    # Legacy QoI values: output (ninner, npred) — no transpose needed
    legacy_qoi_vals = legacy_qoi_model(theta_inner_np)

    # ================================================================
    # Typing setup
    # ================================================================
    from pyapprox.typing.expdesign.benchmarks import (
        LinearGaussianPredOEDBenchmark,
        NonLinearGaussianOEDBenchmark,
    )

    if use_nonlinear:
        typing_bm = NonLinearGaussianOEDBenchmark(
            nobs, degree, noise_std, prior_std, bkd,
            npred=npred, min_degree=min_degree,
        )
    else:
        typing_bm = LinearGaussianPredOEDBenchmark(
            nobs, degree, noise_std, prior_std, npred, bkd,
            min_degree=min_degree,
        )

    typing_obs_model = typing_bm.observation_model()
    typing_pred_model = typing_bm.prediction_model()

    typing_outloop_shapes = typing_obs_model(bkd.asarray(theta_outer_np))
    typing_inloop_shapes = typing_obs_model(bkd.asarray(theta_inner_np))
    # Typing prediction model: output (npred, ninner), need (ninner, npred)
    typing_qoi_vals_raw = typing_pred_model(bkd.asarray(theta_inner_np))
    typing_qoi_vals = bkd.to_numpy(typing_qoi_vals_raw).T  # (ninner, npred)

    # Verify shapes match
    results["obs_shapes_match"] = np.allclose(
        bkd.to_numpy(typing_outloop_shapes),
        np.asarray(legacy_outloop_shapes),
        atol=1e-12,
    )
    results["qoi_vals_match"] = np.allclose(
        typing_qoi_vals,
        np.asarray(legacy_qoi_vals),
        atol=1e-10,
    )

    # ================================================================
    # Create legacy prediction OED objective
    # ================================================================
    noise_diag = NumpyMixin.full((nobs, 1), noise_std**2)
    legacy_inloop_loglike = LegacyInnerLike(noise_diag, backend=NumpyMixin)
    legacy_pred_oed = LegacyPredOED(legacy_inloop_loglike)

    outloop_weights = NumpyMixin.full((nouter, 1), 1.0 / nouter)
    inloop_weights = NumpyMixin.full((ninner, 1), 1.0 / ninner)
    qoi_quad_weights = NumpyMixin.full((npred, 1), 1.0 / npred)

    # Standard deviation measure
    from pyapprox.expdesign.bayesoed import OEDStandardDeviationMeasure
    legacy_deviation = OEDStandardDeviationMeasure(npred, backend=NumpyMixin)
    legacy_risk = SampleAverageMean(NumpyMixin)
    legacy_noise_stat = NoiseStatistic(SampleAverageMean(NumpyMixin))

    legacy_pred_oed.set_data(
        legacy_outloop_shapes,
        outloop_samples_np,
        outloop_weights,
        legacy_inloop_shapes,
        inloop_weights,
        legacy_qoi_vals,
        qoi_quad_weights,
        legacy_deviation,
        legacy_risk,
        legacy_noise_stat,
    )
    legacy_obj = legacy_pred_oed.objective()

    # ================================================================
    # Create typing prediction OED objective
    # ================================================================
    from pyapprox.typing.expdesign.objective import (
        create_prediction_oed_objective,
    )

    typing_obj = create_prediction_oed_objective(
        bkd.full((nobs,), noise_std**2),
        typing_outloop_shapes,
        typing_inloop_shapes,
        bkd.asarray(latent_np),
        bkd.asarray(typing_qoi_vals),
        bkd,
        deviation_type="stdev",
        risk_type="mean",
    )

    # ================================================================
    # Compare objective values at random weights
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
    # Compare optimal designs
    # ================================================================
    from scipy.optimize import minimize, LinearConstraint as ScipyLC

    def _solve_legacy(obj_fn, n):
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
        return res.x, float(res.fun)

    legacy_opt_w, legacy_opt_val = _solve_legacy(legacy_obj, nobs)

    from pyapprox.typing.expdesign.solver import (
        RelaxedOEDSolver,
        RelaxedOEDConfig,
    )
    typing_solver = RelaxedOEDSolver(
        typing_obj, RelaxedOEDConfig(maxiter=500),
    )
    typing_opt_w, typing_opt_val = typing_solver.solve()
    typing_opt_w_np = bkd.to_numpy(typing_opt_w).flatten()

    results["legacy_opt_val"] = legacy_opt_val
    results["typing_opt_val"] = typing_opt_val
    results["opt_val_diff"] = abs(legacy_opt_val - typing_opt_val)

    return results


class TestPredOEDComparisonLinear(Generic[Array], unittest.TestCase):
    """Compare prediction OED with linear QoI."""
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        self._results = _run_pred_comparison(
            self._bkd, use_nonlinear=False,
        )

    def test_obs_shapes_match(self):
        self.assertTrue(self._results["obs_shapes_match"])

    def test_qoi_vals_match(self):
        self.assertTrue(self._results["qoi_vals_match"])

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

    def test_optimal_values_match(self):
        self._bkd.assert_allclose(
            self._bkd.asarray([self._results["legacy_opt_val"]]),
            self._bkd.asarray([self._results["typing_opt_val"]]),
            rtol=1e-3,
        )


class TestPredOEDComparisonNonlinear(Generic[Array], unittest.TestCase):
    """Compare prediction OED with nonlinear (exponential) QoI."""
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        self._results = _run_pred_comparison(
            self._bkd, use_nonlinear=True,
        )

    def test_obs_shapes_match(self):
        self.assertTrue(self._results["obs_shapes_match"])

    def test_qoi_vals_match(self):
        self.assertTrue(self._results["qoi_vals_match"])

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

    def test_optimal_values_match(self):
        self._bkd.assert_allclose(
            self._bkd.asarray([self._results["legacy_opt_val"]]),
            self._bkd.asarray([self._results["typing_opt_val"]]),
            rtol=1e-3,
        )


# Linear QoI
class TestPredOEDLinearNumpy(TestPredOEDComparisonLinear[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestPredOEDLinearTorch(TestPredOEDComparisonLinear[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


# Nonlinear QoI
class TestPredOEDNonlinearNumpy(TestPredOEDComparisonNonlinear[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestPredOEDNonlinearTorch(TestPredOEDComparisonNonlinear[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()
