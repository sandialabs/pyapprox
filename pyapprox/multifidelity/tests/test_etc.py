import unittest

import numpy as np

from pyapprox.benchmarks.multifidelity_benchmarks import (
    TunableModelEnsemble)
from pyapprox.multifidelity.etc import (
    AETCBLUE, _AETC_optimal_loss, _AETC_least_squares)
from pyapprox.multifidelity.factory import get_estimator, multioutput_stats
from pyapprox.multifidelity.groupacv import _cvx_available


class TestETC(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    @staticmethod
    def _setup_model_ensemble_tunable(shifts=None, angle=np.pi/4):
        example = TunableModelEnsemble(angle, shifts)
        cov = example.get_covariance_matrix()
        costs = 10.**(-np.arange(cov.shape[0]))
        return example.funs, cov, costs, example.variable

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

    #@unittest.skipIf(not _cvx_available, "cvxpy not installed")
    @unittest.skipIf(True, "not released yet")
    def test_aetc_blue(self):
        target_cost = 300  # 1e3
        shifts = np.array([1, 2])
        funs, cov, costs, variable = self._setup_model_ensemble_tunable(shifts)
        print(costs)

        true_means = np.hstack((0, shifts))[:, None]

        # funs = [funs[0], funs[2], funs[1]]
        # cov = cov[np.ix_([0, 2, 1], [0, 2, 1])]
        # costs = costs[[0, 2, 1]]
        # true_means = true_means[[0, 2, 1]]

        # oracle_stats = [cov, true_means]
        oracle_stats = None

        # provide oracle covariance so numerical and theoretical estimates
        # will coincide
        # subsets = None
        subsets = [np.array([0, 1])]
        # subsets = [np.array([0])]
        # subsets = [np.array([1])]

        print("the threshold below is for trust-constr without a global search with nelder mead. I need to change tolerance or allow original init_guess to be passed to trust-constr. Right now I can hack old behavior by commenting out nelder mead optimization and just using init guess from self._init_guess. I have also changed the constraints slightly so this will also make it hard to meet the tolerance. I suggest just changing it for cxcpy but it is slow")
        # opt_options = {"method": "trust-constr"}
        opt_options = {"method": "cvxpy"}
        print("#")
        np.set_printoptions(precision=16)
        estimator = AETCBLUE(
            funs, variable.rvs, costs,  oracle_stats, 0, 0,
            opt_options=opt_options)
        mean, values, result = estimator.estimate(
            target_cost, return_dict=False, subsets=subsets)
        result_dict = estimator._explore_result_to_dict(result)
        print(result_dict)
        cov_exe = np.cov(values, rowvar=False, ddof=1)

        # todo switch on and off oracle stats

        subset = result_dict["subset"]+1
        stat = multioutput_stats["mean"](1)
        stat.set_pilot_quantities(cov_exe[np.ix_(subset, subset)])
        mlblue_est = get_estimator(
            "mlblue", stat, costs[subset],
            asketch=result_dict["beta_Sp"][1:])
        true_var = mlblue_est._covariance_from_npartition_samples(
            result_dict["rounded_nsamples_per_subset"])
        unrounded_true_var = mlblue_est._covariance_from_npartition_samples(
            result_dict["nsamples_per_subset"])
        print(result_dict["sigma_S"], cov_exe)
        assert np.allclose(result_dict["sigma_S"],
                           cov_exe[np.ix_(subset, subset)])

        ntrials = int(1e4)
        means = np.empty(ntrials)
        for ii in range(ntrials):
            means[ii] = estimator.exploit(result)
        numerical_var = means.var()
        print(numerical_var, "NV")
        print(true_var.numpy(), "TV")
        print(unrounded_true_var.numpy(), result_dict["BLUE_variance"])
        assert np.allclose(unrounded_true_var, result_dict["BLUE_variance"])
        # assert np.allclose(numerical_var, true_var, rtol=3e-2)
        assert np.allclose(
            result_dict["BLUE_variance"],
            unrounded_true_var, rtol=3e-2)

        noracle_samples = 1e5
        oracle_samples = variable.rvs(noracle_samples)
        oracle_hf_values = funs[0](oracle_samples)
        active_funs_idx = []
        for ii in range(1, len(funs)):
            for subset in subsets:
                if ii-1 in subset and ii not in active_funs_idx:
                    active_funs_idx.append(ii)
                    break
        print(active_funs_idx)
        oracle_covariate_values = np.hstack(
            [funs[ii](oracle_samples) for ii in active_funs_idx])
        true_beta_Sp = _AETC_least_squares(
            oracle_hf_values, oracle_covariate_values)[0]

        ntrials = int(1e3)
        means = np.empty(ntrials)
        sq_biases, variances = [], []
        print(true_means[0], true_means[active_funs_idx])
        true_active_means = np.hstack(
            (true_means[0], true_means[active_funs_idx, 0]))
        for ii in range(ntrials):
            print(ii)
            # print(estimator)
            means[ii], values_per_model, result = estimator.estimate(
                target_cost, subsets=subsets)
            sq_biases.append(
                (true_active_means.T @ (true_beta_Sp-result["beta_Sp"]))**2)
            variances.append(result["BLUE_variance"])

        mse = np.mean((means-true_means[0])**2)
        sq_bias = np.mean(sq_biases)
        variance = np.mean(variances)
        print(sq_bias, 'mc_bias')
        print(variance, "mc_var")
        print(sq_bias+variance, "mc_loss")
        print(mse, result_dict["loss"])
        print((mse-result_dict["loss"])/result_dict["loss"])
        # this is just a regression test to make sure this does not get worse
        # when code is changed
        assert np.allclose(mse, result_dict["loss"], rtol=3e-2)


if __name__ == "__main__":
    etc_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestETC)
    unittest.TextTestRunner(verbosity=2).run(etc_test_suite)
