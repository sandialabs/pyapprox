import unittest
from functools import partial

import numpy as np

from pyapprox.multifidelity.stats import (
    _nqoisq_nqoisq_subproblem,
    _nqoi_nqoisq_subproblem, MultiOutputMeanAndVariance,
    _get_V_from_covariance, _covariance_of_variance_estimator)
from pyapprox.benchmarks.multifidelity_benchmarks import (
    MultioutputModelEnsemble)


def _single_qoi(qoi, fun, xx):
    return fun(xx)[:, qoi:qoi+1]


def _two_qoi(ii, jj, fun, xx):
    return fun(xx)[:, [ii, jj]]


def _setup_multioutput_model_subproblem(model_idx, qoi_idx):
    model = MultioutputModelEnsemble()
    cov = model.get_covariance_matrix()
    funs = [model.funs[ii] for ii in model_idx]
    if len(qoi_idx) == 1:
        funs = [partial(_single_qoi, qoi_idx[0], f) for f in funs]
    elif len(qoi_idx) == 2:
        funs = [partial(_two_qoi, *qoi_idx, f) for f in funs]
    idx = np.arange(9).reshape(3, 3)[np.ix_(model_idx, qoi_idx)].flatten()
    cov = cov[np.ix_(idx, idx)]
    costs = model.costs()[model_idx]
    means = model.get_means()[np.ix_(model_idx, qoi_idx)]
    return funs, cov, costs, model, means


class TestMOSTATS(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

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
        assert np.allclose(mc_cov, model.get_covariance_matrix(), rtol=1e-2)

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
        funs, cov_exact, costs, model, means = (
            _setup_multioutput_model_subproblem(model_idx, qoi_idx))
        # atol is needed for terms close to zero
        rtol, atol = 1e-2, 5.5e-4
        npilot_samples = int(1e6)
        pilot_samples = model.variable.rvs(npilot_samples)
        pilot_values = [f(pilot_samples) for f in funs]
        cov, W, B = MultiOutputMeanAndVariance.compute_pilot_quantities(
            pilot_values)
        assert np.allclose(cov, cov_exact, atol=atol, rtol=rtol)
        W_exact = model.covariance_of_centered_values_kronker_product()
        W_exact = _nqoisq_nqoisq_subproblem(
            W_exact, model.nmodels, model.nqoi, model_idx, qoi_idx)
        print(cov.shape, W.shape, B.shape)
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

    def _mean_variance_realizations(self, funs, variable, nsamples, ntrials):
        nmodels = len(funs)
        means, covariances = [], []
        for ii in range(ntrials):
            samples = variable.rvs(nsamples)
            vals = np.hstack([f(samples) for f in funs])
            nqoi = vals.shape[1]//nmodels
            means.append(vals.mean(axis=0))
            covariance = np.hstack(
                [np.cov(vals[:, ii*nqoi:(ii+1)*nqoi].T, ddof=1).flatten()
                 for ii in range(nmodels)])
            covariances.append(covariance)
        means = np.array(means)
        covariances = np.array(covariances)
        return means, covariances

    def _check_mean_variance_covariances(self, model_idx, qoi_idx):
        nsamples, ntrials = 20, int(1e5)
        funs, cov, costs, model, means = _setup_multioutput_model_subproblem(
            model_idx, qoi_idx)
        means, covariances = self._mean_variance_realizations(
            funs, model.variable, nsamples, ntrials)
        nmodels = len(funs)
        nqoi = cov.shape[0]//nmodels

        # atol is needed for terms close to zero
        rtol, atol = 1e-2, 1e-4
        B_exact = model.covariance_of_mean_and_variance_estimators()
        B_exact = _nqoi_nqoisq_subproblem(
            B_exact, model.nmodels, model.nqoi, model_idx, qoi_idx)
        mc_mean_cov_var = np.cov(means.T, covariances.T, ddof=1)
        B_mc = mc_mean_cov_var[:nqoi*nmodels, nqoi*nmodels:]
        assert np.allclose(B_mc, B_exact/nsamples, atol=atol, rtol=rtol)

        # no need to extract subproblem for V_exact as cov has already
        # been downselected
        V_exact = _get_V_from_covariance(cov, nmodels)
        W_exact = model.covariance_of_centered_values_kronker_product()
        W_exact = _nqoisq_nqoisq_subproblem(
            W_exact, model.nmodels, model.nqoi, model_idx, qoi_idx)
        cov_var_exact = _covariance_of_variance_estimator(
            W_exact, V_exact, nsamples)
        print(cov)
        # print(V_exact)
        assert np.allclose(
           cov_var_exact, mc_mean_cov_var[nqoi*nmodels:, nqoi*nmodels:],
           atol=atol, rtol=rtol)

    def test_mean_variance_covariances(self):
        # fast_test = True
        test_cases = [
            [[0], [0]],
            [[1], [0, 1]],
            [[1], [0, 2]],
            [[1, 2], [0]],
            [[0, 1], [0, 2]],
            [[0, 1], [0, 1, 2]],
            [[0, 1, 2], [0]],
            [[0, 1, 2], [0, 1]],
            [[0, 1, 2], [0, 1, 2]],
        ]
        # if fast_test:
        #     test_cases = [test_cases[1], test_cases[-1]]
        for test_case in test_cases:
            np.random.seed(123)
            self._check_mean_variance_covariances(*test_case)


if __name__ == "__main__":
    mostats_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestMOSTATS)
    unittest.TextTestRunner(verbosity=2).run(mostats_test_suite)
