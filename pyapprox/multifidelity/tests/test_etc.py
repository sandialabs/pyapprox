import unittest

import numpy as np
from scipy import stats

from pyapprox.benchmarks.multifidelity_benchmarks import (
    TunableModelEnsemble)
from pyapprox.multifidelity.etc import AETCBLUE, _AETC_optimal_loss
from pyapprox.multifidelity.factory import get_estimator


class TestETC(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    @staticmethod
    def _setup_model_ensemble_tunable(shifts=None):
        example = TunableModelEnsemble(np.pi/4, shifts)
        funs = example.models
        cov = example.get_covariance_matrix()
        costs = 10.**(-np.arange(cov.shape[0]))
        return funs, cov, costs, example.variable

    def test_AETC_optimal_loss(self):
        alpha = 1000
        nsamples = int(1e6)
        shifts = np.array([1, 2])
        funs, cov, costs, variable = self._setup_model_ensemble_tunable(shifts)
        target_cost = np.sum(costs)*(nsamples+10)

        true_means = np.hstack((0, shifts))[:, None]
        oracle_stats = [cov, true_means]

        samples = variable.rvs(nsamples)
        values = np.hstack([fun(samples) for fun in funs])

        exploit_cost = 0.5*target_cost
        covariate_subset = np.asarray([0, 1])
        hf_values = values[:, :1]
        covariate_values = values[:, covariate_subset+1]
        result_oracle = _AETC_optimal_loss(
            target_cost, hf_values, covariate_values, costs, covariate_subset,
            alpha, 0, 0, oracle_stats, {}, exploit_cost)
        result_mc = _AETC_optimal_loss(
            target_cost, hf_values, covariate_values, costs, covariate_subset,
            alpha, 0, 0, None, {}, exploit_cost)
        assert np.allclose(result_mc[-2], result_oracle[-2], rtol=1e-2)

    def test_aetc_blue(self):
        target_cost = 1e4
        shifts = np.array([1, 2])
        funs, cov, costs, variable = self._setup_model_ensemble_tunable(shifts)

        true_means = np.hstack((0, shifts))[:, None]
        oracle_stats = [cov, true_means]
        # oracle_stats = None

        # provide oracle covariance so numerical and theoretical estimates
        # will coincide
        opt_options = {"method": "cvxpy"}
        # opt_options = {"method": "trust-constr"}
        print("#")
        estimator = AETCBLUE(
            funs, variable.rvs, costs,  oracle_stats, 0, 0,
            opt_options=opt_options)
        mean, values, result = estimator.estimate(
            target_cost, return_dict=False, subsets=[np.array([0, 1])])
        result_dict = estimator._explore_result_to_dict(result)
        print(result_dict)

        subset = result_dict["subset"]+1
        mlblue_est = get_estimator(
            "mlblue", "mean", 1, costs[subset], cov[np.ix_(subset, subset)],
            asketch=result_dict["beta_Sp"][1:])
        true_var = mlblue_est._covariance_from_npartition_samples(
            result_dict["rounded_nsamples_per_subset"])
        unrounded_true_var = mlblue_est._covariance_from_npartition_samples(
            result_dict["nsamples_per_subset"])
        print((mlblue_est._costs*mlblue_est._compute_nsamples_per_model(
            result_dict["nsamples_per_subset"])).sum())

        ntrials = int(1e3)
        means = np.empty(ntrials)
        for ii in range(ntrials):
            means[ii] = estimator.exploit(result)
        numerical_var = means.var()
        print(numerical_var, "NV")
        print(true_var.numpy(), "TV")
        print(result_dict["BLUE_variance"])
        print(unrounded_true_var.numpy()[0, 0], "UN")
        print(result_dict["exploit_budget"], 'e')
        print(true_var)
        assert np.allclose(numerical_var, true_var, rtol=3e-2)
        assert np.allclose(
            result_dict["BLUE_variance"],
            unrounded_true_var, rtol=3e-2)

        ntrials = int(1e3)
        means = np.empty(ntrials)
        for ii in range(ntrials):
            means[ii] = estimator.estimate(target_cost)[0]
        true_mean = 0
        mse = np.mean((means-true_mean)**2)
        print(mse, result_dict["loss"])
        print((mse-result_dict["loss"])/result_dict["loss"])
        # this is just a regression test to make sure this does not get worse
        # when code is changed
        assert np.allclose(mse, result_dict["loss"], rtol=2e-1)



if __name__ == "__main__":
    etc_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestETC)
    unittest.TextTestRunner(verbosity=2).run(etc_test_suite)
