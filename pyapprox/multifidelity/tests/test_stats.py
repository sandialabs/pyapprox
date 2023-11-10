import unittest
from functools import partial

import numpy as np

from pyapprox.multifidelity.stats import (
    _nqoisq_nqoisq_subproblem,
    _nqoi_nqoisq_subproblem, MultiOutputMeanAndVariance)
from pyapprox.benchmarks.multifidelity_benchmarks import (
    MultioutputModelEnsemble)


def _single_qoi(qoi, fun, xx):
    return fun(xx)[:, qoi:qoi+1]


def _two_qoi(ii, jj, fun, xx):
    return fun(xx)[:, [ii, jj]]


def _setup_multioutput_model_subproblem(model_idx, qoi_idx):
    model = MultioutputModelEnsemble()
    cov = model.covariance()
    funs = [model.funs[ii] for ii in model_idx]
    if len(qoi_idx) == 1:
        funs = [partial(_single_qoi, qoi_idx[0], f) for f in funs]
    elif len(qoi_idx) == 2:
        funs = [partial(_two_qoi, *qoi_idx, f) for f in funs]
    idx = np.arange(9).reshape(3, 3)[np.ix_(model_idx, qoi_idx)].flatten()
    cov = cov[np.ix_(idx, idx)]
    costs = model.costs()[model_idx]
    return funs, cov, costs, model


class TestMOSTATS(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_multioutput_model(self):
        model = MultioutputModelEnsemble()
        assert np.allclose(model.covariance(), model._covariance_quadrature())

    def test_covariance_einsum(self):
        model = MultioutputModelEnsemble()
        npilot_samples = int(1e5)
        pilot_samples = model.variable.rvs(npilot_samples)
        pilot_values = np.hstack([f(pilot_samples) for f in model.funs])
        means = pilot_values.mean(axis=0)
        centered_values = pilot_values - means
        centered_values_sq = np.einsum(
            'nk,nl->nkl', centered_values, centered_values).reshape(
                npilot_samples, -1)
        nqoi = pilot_values.shape[1]
        mc_cov = (centered_values_sq.sum(axis=0)/(npilot_samples-1)).reshape(
            nqoi, nqoi)
        assert np.allclose(mc_cov, model.covariance(), rtol=1e-2)

    def test_variance_double_loop(self):
        NN = 5
        model = MultioutputModelEnsemble()
        samples = model.variable.rvs(NN)
        variance = 0
        for ii in range(samples.shape[1]):
            for jj in range(samples.shape[1]):
                variance += (samples[:, ii]-samples[:, jj])**2
        variance /= (2*NN*(NN-1))
        variance1 = ((samples - samples.mean(axis=1))**2).sum(axis=1)/(NN-1)
        assert np.allclose(variance, variance1)
        assert np.allclose(variance, samples.var(axis=1, ddof=1))

    def _check_pilot_covariances(self, model_idx, qoi_idx):
        funs, cov_exact, costs, model = _setup_multioutput_model_subproblem(
            model_idx, qoi_idx)
        nmodels = len(funs)
        # atol is needed for terms close to zero
        rtol, atol = 1e-2, 5.5e-4
        npilot_samples = int(1e6)
        pilot_samples = model.variable.rvs(npilot_samples)
        pilot_values = np.hstack([f(pilot_samples) for f in funs])
        cov, W, B = MultiOutputMeanAndVariance.compute_pilot_quantities(
            pilot_values, nmodels)
        assert np.allclose(cov, cov_exact, atol=atol, rtol=rtol)
        W_exact = model.covariance_of_centered_values_kronker_product()
        W_exact = _nqoisq_nqoisq_subproblem(
            W_exact, model.nmodels, model.nqoi, model_idx, qoi_idx)
        assert np.allclose(W, W_exact, atol=atol, rtol=rtol)
        B_exact = model.covariance_of_mean_and_variance_estimators()
        B_exact = _nqoi_nqoisq_subproblem(
            B_exact, model.nmodels, model.nqoi, model_idx, qoi_idx)
        assert np.allclose(B, B_exact, atol=atol, rtol=rtol)

    def test_pilot_covariances(self):
        fast_test = True
        test_cases = [
            [[0], [0]],
            [[1], [0, 1]],
            [[1], [0, 2]],
            [[1, 2], [0]],
            [[0, 1], [0, 2]],
            [[0, 1], [0, 1, 2]],
            [[0, 1, 2], [0]],
            [[0, 1, 2], [0, 2]],
            [[0, 1, 2], [0, 1, 2]],
        ]
        if fast_test:
            test_cases = [test_cases[1], test_cases[-1]]
        for test_case in test_cases:
            np.random.seed(123)
            self._check_pilot_covariances(*test_case)


if __name__ == "__main__":
    mostats_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestMOSTATS)
    unittest.TextTestRunner(verbosity=2).run(mostats_test_suite)
