import unittest

import numpy as np
from scipy import stats

from pyapprox.benchmarks.multifidelity_benchmarks import (
    TunableModelEnsemble)
from pyapprox.multifidelity.etc import AETCBLUE, _AETC_optimal_loss


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

        covariate_subset = np.asarray([0, 1])
        hf_values = values[:, :1]
        covariate_values = values[:, covariate_subset+1]
        result_oracle = _AETC_optimal_loss(
            target_cost, hf_values, covariate_values, costs, covariate_subset,
            alpha, 0, 0, oracle_stats)
        result_mc = _AETC_optimal_loss(
            target_cost, hf_values, covariate_values, costs, covariate_subset,
            alpha, 0, 0, None)
        # for r in result_oracle:
        #     print(r)
        #     print("##")
        # for r in result_mc:
        #     print(r)
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
        estimator = AETCBLUE(
            funs, variable.rvs, costs,  oracle_stats, 1e-12, 0)
        mean, values, result = estimator.estimate(
            target_cost, return_dict=False)
        result_dict = estimator._explore_result_to_dict(result)
        print(result_dict)

        from pyapprox.multifidelity.multilevelblue import BLUE_variance
        true_var = BLUE_variance(
            result_dict["beta_Sp"][1:],
            cov[np.ix_(result_dict["subset"]+1, result_dict["subset"]+1)],
            None, estimator._reg_blue, result_dict["nsamples_per_subset"])

        ntrials = int(1e3)
        means = np.empty(ntrials)
        for ii in range(ntrials):
            means[ii] = estimator.exploit(result)[0]
        numerical_var = means.var()
        print(numerical_var, "NV")
        print(true_var, "TV")
        assert np.allclose(numerical_var, true_var, rtol=3e-2)
        assert np.allclose(
            result_dict["BLUE_variance"]/result_dict["exploit_budget"],
            true_var, rtol=3e-2)

        ntrials = int(1e3)
        means = np.empty(ntrials)
        for ii in range(ntrials):
            means[ii] = estimator.estimate(target_cost)[0]
        true_mean = 0
        mse = np.mean((means-true_mean)**2)
        print(mse, result_dict["loss"])
        # this is just a regression test to make sure this does not get worse
        # when code is changed
        assert np.allclose(mse, result_dict["loss"], rtol=2e-1)



if __name__ == "__main__":
    etc_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestETC)
    unittest.TextTestRunner(verbosity=2).run(etc_test_suite)
