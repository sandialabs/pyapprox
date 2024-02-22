import unittest
from functools import partial

import numpy as np
from scipy import stats

from pyapprox.util.utilities import (
    get_correlation_from_covariance, check_gradients)
from pyapprox.multifidelity.groupacv import (
    get_model_subsets, GroupACVEstimator,
    _get_allocation_matrix_is, _get_allocation_matrix_nested, _nest_subsets,
    _cvx_available, MLBLUEEstimator)
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.surrogates.autogp._torch_wrappers import (
    arange, full)
from pyapprox.multifidelity.factory import multioutput_stats


class TestGroupACV(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_allocation_mat(self):
        nmodels = 3
        subsets = get_model_subsets(nmodels)
        allocation_mat = _get_allocation_matrix_is(subsets)
        assert np.allclose(allocation_mat, np.eye(len(subsets)))

        # remove subset 0
        subsets = get_model_subsets(nmodels)[1:]
        subsets = _nest_subsets(subsets, nmodels)[0]
        print(subsets)
        idx = sorted(
            list(range(len(subsets))),
            key=lambda ii: (len(subsets[ii]), tuple(nmodels-subsets[ii])),
            reverse=True)
        subsets = [subsets[ii] for ii in idx]
        nsubsets = len(subsets)
        allocation_mat = _get_allocation_matrix_nested(subsets)
        assert np.allclose(
            allocation_mat, np.tril(np.ones((nsubsets, nsubsets))))

        # import matplotlib.pyplot as plt
        # from group_acv import _plot_allocation_matrix
        # ax = plt.subplots(1, 1, figsize=(8, 6))[1]
        # _plot_allocation_matrix(allocation_mat, subsets, ax)
        # plt.savefig("groupacvnested.pdf")

    def _check_separate_samples(self, est):
        NN = 2
        npartition_samples = full((est.nsubsets,), NN)
        est._set_optimized_params(npartition_samples)

        samples_per_model = est.generate_samples_per_model(
            lambda n: np.arange(n)[None, :])
        for ii in range(est.nmodels):
            assert (samples_per_model[ii].shape[1] ==
                    est._rounded_nsamples_per_model[ii])
        values_per_model = [
            (ii+1)*s.T for ii, s in enumerate(samples_per_model)]
        values_per_subset = est._separate_values_per_model(values_per_model)

        test_samples = np.arange(
            est._rounded_npartition_samples.sum())[None, :]
        test_values = [(ii+1)*test_samples.T for ii in range(est.nmodels)]
        for ii in range(est.nsubsets):
            active_partitions = np.where(est.allocation_mat[ii] == 1)[0]
            indices = np.arange(test_samples.shape[1], dtype=int).reshape(
                est.npartitions, NN)[active_partitions].flatten()
            assert np.allclose(
                values_per_subset[ii].shape,
                (est._nintersect_samples(npartition_samples)[ii][ii],
                 len(est.subsets[ii])))
            for jj, s in enumerate(est.subsets[ii]):
                assert np.allclose(
                    values_per_subset[ii][:,  jj], test_values[s][indices, 0])

    def test_nsamples_per_model(self):
        nmodels = 3
        cov = np.random.normal(0, 1, (nmodels, nmodels))
        cov = cov.T @ cov
        costs = np.arange(nmodels, 0, -1)

        stat = multioutput_stats["mean"](1)
        stat.set_pilot_quantities(cov)
        est = GroupACVEstimator(stat, costs)
        npartition_samples = arange(2., 2+est.nsubsets)
        assert np.allclose(
            est._compute_nsamples_per_model(npartition_samples),
            np.array([21, 23, 25]))
        assert np.allclose(
            est._estimator_cost(npartition_samples), 21*3+23*2+25*1)
        assert np.allclose(est._nintersect_samples(npartition_samples),
                           np.diag(npartition_samples))
        self._check_separate_samples(est)

        est = GroupACVEstimator(
            stat, costs, est_type="nested")
        npartition_samples = arange(2., 2+est.nsubsets)
        assert np.allclose(
            est._compute_nsamples_per_model(npartition_samples),
            np.array([9, 20, 27]))
        assert np.allclose(
            est._estimator_cost(npartition_samples), 9*3+20*2+27*1)
        assert np.allclose(
            est._nintersect_samples(npartition_samples),
            np.array([[2.,  2.,  2.,  2.,  2.,  2.],
                      [2.,  5.,  5.,  5.,  5.,  5.],
                      [2.,  5.,  9.,  9.,  9.,  9.],
                      [2.,  5.,  9., 14., 14., 14.],
                      [2.,  5.,  9., 14., 20., 20.],
                      [2.,  5.,  9., 14., 20., 27.]]))
        self._check_separate_samples(est)

        # import matplotlib.pyplot as plt
        # from group_acv import _plot_partitions_per_model
        # ax = plt.subplots(1, 1, figsize=(8, 6))[1]
        # _plot_partitions_per_model(
        #     est.partitions_per_model, ax,
        #     npartition_samples=npartition_samples)
        # plt.show()

    def _generate_correlated_values(self, chol_factor, means, samples):
        return (chol_factor @ samples + means[:, None]).T

    def _check_mean_estimator_variance(self, nmodels, ntrials, group_type,
                                       asketch=None):
        ntrials = int(ntrials)
        cov = np.random.normal(0, 1, (nmodels, nmodels))
        cov = cov.T @ cov
        cov = get_correlation_from_covariance(cov)
        costs = np.arange(nmodels, 0, -1)
        variable = IndependentMarginalsVariable(
            [stats.norm(0, 1) for ii in range(nmodels)])
        stat = multioutput_stats["mean"](1)
        stat.set_pilot_quantities(cov)
        est = GroupACVEstimator(stat, costs, est_type=group_type,
                                asketch=asketch)
        npartition_samples = arange(2., 2+est.nsubsets)
        est._set_optimized_params(
            npartition_samples)
        est_var = est._covariance_from_npartition_samples(
            est._rounded_npartition_samples)

        chol_factor = np.linalg.cholesky(cov)
        exact_means = np.arange(nmodels)
        generate_values = partial(
            self._generate_correlated_values, chol_factor, exact_means)

        subset_means = []
        acv_means = []

        for nn in range(ntrials):
            samples_per_model = est.generate_samples_per_model(variable.rvs)
            values_per_model = [
                generate_values(samples_per_model[ii])[:, ii:ii+1]
                for ii in range(est.nmodels)]
            subset_values = est._separate_values_per_model(values_per_model)
            subset_means_nn = []
            acv_mean = 0
            for kk in range(est.nsubsets):
                if subset_values[kk].shape[0] > 0:
                    subset_mean = subset_values[kk].mean(axis=0)
                else:
                    subset_mean = np.zeros(len(est.subsets[kk]))
                subset_means_nn.append(subset_mean)
            acv_mean = est(values_per_model)
            subset_means.append(np.hstack(subset_means_nn))
            acv_means.append(acv_mean)
        acv_means = np.array(acv_means)
        subset_means = np.array(subset_means)
        mc_group_cov = np.cov(subset_means, ddof=1, rowvar=False)
        Sigma = est._sigma(est._rounded_npartition_samples)
        atol, rtol = 4e-3, 2e-2
        np.set_printoptions(linewidth=1000, precision=4)
        # print(np.diag(mc_group_cov))
        # print(np.diag(Sigma.numpy()))
        # print(np.diag(mc_group_cov)-np.diag(Sigma.numpy()))
        assert np.allclose(
            np.diag(mc_group_cov), np.diag(Sigma.numpy()), rtol=rtol)
        # print(mc_group_cov)
        # print(Sigma.numpy())
        # print(mc_group_cov-Sigma.numpy())
        assert np.allclose(mc_group_cov, Sigma.numpy(), rtol=rtol, atol=atol)
        est_var_mc = acv_means.var(ddof=1)
        # print(est_var_mc, est_var)
        assert np.allclose(est_var_mc, est_var, rtol=rtol, atol=atol)

    def test_mean_estimator_variance(self):
        test_cases = [
            [2, 5e4, "is"],
            [2, 5e4, "is", [0.5, 0.5]],
            [2, 2e4, "nested"],
            [3, 5e4, "is"],
            [3, 2e4, "nested"],
            [4, 5e4, "nested"],
        ]
        # ignore last test until I can speed up code
        for test_case in test_cases:
            np.random.seed(1)
            print(test_case)
            self._check_mean_estimator_variance(*test_case)

    def _check_mlblue_objective(self, nmodels, min_nhf_samples):
        # check specialized mlblue objective is consitent with
        # more general groupacv estimator when computing a single mean
        cov = np.random.normal(0, 1, (nmodels, nmodels))
        cov = cov.T @ cov
        cov = get_correlation_from_covariance(cov)

        target_cost = 100
        costs = np.logspace(-nmodels+1, 0, nmodels)[::-1].copy()
        stat = multioutput_stats["mean"](1)
        stat.set_pilot_quantities(cov)
        gest = GroupACVEstimator(stat, costs, reg_blue=0)
        gest.allocate_samples(
            target_cost,
            optim_options={"disp": False, "maxiter": 1000,
                           "method": "slsqp"},
            min_nhf_samples=min_nhf_samples)
        assert gest._nhf_samples(
            gest._rounded_npartition_samples) >= min_nhf_samples

        mlest = MLBLUEEstimator(stat, costs, reg_blue=0)
        mlest.allocate_samples(
            target_cost, optim_options={"method": "slsqp"},
            min_nhf_samples=min_nhf_samples)
        assert mlest._nhf_samples(
            mlest._rounded_npartition_samples) >= min_nhf_samples
        assert np.allclose(mlest._covariance_from_npartition_samples(
            gest._rounded_npartition_samples),
                           gest._covariance_from_npartition_samples(
                               gest._rounded_npartition_samples))

        init_guess = mlest._init_guess(target_cost).numpy()
        # init_guess = np.array([99., 1e-2, 1e-2])
        errors = check_gradients(
            lambda x: gest._objective(x[:, 0], True), True,
            init_guess[:, None],
            disp=False)
        assert errors.min()/errors.max() < 1e-6 and errors[0] < 1

        # the answers will be different because group acv optimization
        # currently uses finite differences
        print(gest._optimized_criteria, mlest._optimized_criteria)
        assert np.allclose(gest._optimized_criteria,
                           mlest._optimized_criteria, rtol=5e-3)

    def test_mlblue_objective(self):
        test_cases = [
            [2, 1], [3, 1], [4, 1], [3, 10],
        ]
        for test_case in test_cases:
            np.random.seed(1)
            print(test_case)
            self._check_mlblue_objective(*test_case)

    def _check_mlblue_spd(self, nmodels, min_nhf_samples):
        cov = np.random.normal(0, 1, (nmodels, nmodels))
        cov = cov.T @ cov
        cov = get_correlation_from_covariance(cov)

        target_cost = 100
        costs = np.logspace(-nmodels+1, 0, nmodels)[::-1].copy()

        stat = multioutput_stats["mean"](1)
        stat.set_pilot_quantities(cov)
        gest = MLBLUEEstimator(stat, costs, reg_blue=0)
        gest.allocate_samples(
            target_cost,
            optim_options={"disp": False, "verbose": 0, "maxiter": 1000,
                           "gtol": 1e-9, "method": "trust-constr"},
            min_nhf_samples=min_nhf_samples)

        mlest = MLBLUEEstimator(stat, costs, reg_blue=0)
        mlest.allocate_samples(target_cost, optim_options={"method": "cvxpy"},
                               min_nhf_samples=min_nhf_samples)
        assert mlest._nhf_samples(
            mlest._rounded_npartition_samples) >= min_nhf_samples

        print(gest._optimized_criteria, mlest._optimized_criteria)
        assert np.allclose(gest._optimized_criteria,
                           mlest._optimized_criteria, rtol=1e-3)

    @unittest.skipIf(not _cvx_available, "cvxpy not installed")
    def test_mlblue_spd(self):
        test_cases = [
            [2, 1], [3, 1], [4, 1], [3, 10],
        ]
        for test_case in test_cases:
            np.random.seed(1)
            self._check_mlblue_spd(*test_case)

    def _check_objective_constraint_gradients(self, nmodels):
        cov = np.random.normal(0, 1, (nmodels, nmodels))
        cov = cov.T @ cov
        cov = get_correlation_from_covariance(cov)

        target_cost = 100
        costs = np.logspace(-nmodels+1, 0, nmodels)[::-1].copy()
        stat = multioutput_stats["mean"](1)
        stat.set_pilot_quantities(cov)
        gest = GroupACVEstimator(stat, costs, reg_blue=1e-12)

        init_guess = gest._init_guess(target_cost).numpy()
        # init_guess = np.array([99., 1e-2, 1e-2])
        errors = check_gradients(
            lambda x: gest._objective(x[:, 0], True), True,
            init_guess[:, None],
            disp=True)
        assert errors.min()/errors.max() < 1e-6 and errors[0] < 1

        errors = check_gradients(
            lambda x: gest._cost_constraint(
                x[:, 0], target_cost, return_grad=False),
            lambda x: gest._cost_constraint_jac(
                x[:, 0], target_cost),
            init_guess[:, None], disp=False)
        assert errors.min()/errors.max() < 1e-6 and errors.max() < 1

    def test_objective_constraint_gradients(self):
        test_cases = [
            [2], [3], [4],
        ]
        for test_case in test_cases:
            np.random.seed(1)
            self._check_objective_constraint_gradients(*test_case)

    def _check_insert_pilot_samples(self, nmodels, min_nhf_samples, seed):
        np.random.seed(seed)
        cov = np.random.normal(0, 1, (nmodels, nmodels))
        cov = cov.T @ cov
        cov = get_correlation_from_covariance(cov)

        variable = IndependentMarginalsVariable(
            [stats.norm(0, 1) for ii in range(nmodels)])
        chol_factor = np.linalg.cholesky(cov)
        exact_means = np.arange(nmodels)
        generate_values = partial(
            self._generate_correlated_values, chol_factor, exact_means)

        npilot_samples = 8
        assert min_nhf_samples > npilot_samples

        target_cost = 100
        costs = np.logspace(-nmodels+1, 0, nmodels)[::-1].copy()
        stat = multioutput_stats["mean"](1)
        stat.set_pilot_quantities(cov)
        est = MLBLUEEstimator(stat, costs, reg_blue=1e-10)
        # trust-const method does not work on some platforms.
        # It produces an interate and the lower bound and the SubBarrierProblem
        # returns a nan because it take the log of 0=x-lb.
        est.allocate_samples(
            target_cost, min_nhf_samples=min_nhf_samples,
            optim_options={"maxiter": 1000, "init_guess": {"maxfev": 1000},
                           "bounds": [1e-10, 1e6], "method": "slsqp"})

        # the following test only works if variable.num_vars()==1 because
        # variable.rvs does not produce nested samples when this condition does
        # not hold

        np.random.seed(seed)
        samples_per_model = est.generate_samples_per_model(
            variable.rvs)
        pilot_samples = est._remove_pilot_samples(
            npilot_samples, samples_per_model)[1]
        pilot_values = [
            generate_values(pilot_samples)[:, ii:ii+1]
            for ii in range(nmodels)]

        np.random.seed(seed)
        samples_per_model_wo_pilot = est.generate_samples_per_model(
            variable.rvs, npilot_samples)
        values_per_model_wo_pilot = [
            generate_values(samples_per_model_wo_pilot[ii])[:, ii:ii+1]
            for ii in range(est.nmodels)]
        values_per_model_recovered = est.insert_pilot_values(
            pilot_values, values_per_model_wo_pilot)

        np.random.seed(seed)
        samples_per_model = est.generate_samples_per_model(
            variable.rvs)
        values_per_model = [
            generate_values(samples_per_model[ii])[:, ii:ii+1]
            for ii in range(est.nmodels)]

        for v1, v2 in zip(values_per_model, values_per_model_recovered):
            assert np.allclose(v1, v2)

    def test_insert_pilot_samples(self):
        test_cases = [
            [2, 11], [3, 11], [4, 11],
        ]
        ntrials = 20
        for test_case in test_cases:
            for seed in range(ntrials):
                self._check_insert_pilot_samples(*test_case, seed)


if __name__ == '__main__':
    gacv_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestGroupACV)
    unittest.TextTestRunner(verbosity=2).run(gacv_test_suite)
