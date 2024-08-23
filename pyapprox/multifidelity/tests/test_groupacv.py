import unittest
from functools import partial

import numpy as np
from scipy import stats

from pyapprox.util.utilities import (
    get_correlation_from_covariance, check_gradients)
from pyapprox.multifidelity.groupacv import (
    get_model_subsets, GroupACVEstimator,
    _get_allocation_matrix_is, _get_allocation_matrix_nested, _nest_subsets,
    MLBLUEEstimator, GroupACVGradientOptimizer, MLBLUESPDOptimizer,
    ChainedACVOptimizer
)
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.multifidelity.factory import multioutput_stats
from pyapprox.optimization.pya_minimize import (
    ScipyConstrainedOptimizer, ScipyConstrainedNelderMeadOptimizer
)
from pyapprox.multifidelity._optim import _allocate_samples_mfmc
from pyapprox.util.sys_utilities import package_available
from pyapprox.multifidelity.acv import MFMCEstimator


class TestGroupACV:
    def setUp(self):
        np.random.seed(1)

    def test_allocation_mat(self):
        bkd = self.get_backend()
        nmodels = 3
        subsets = get_model_subsets(nmodels, bkd)
        allocation_mat = _get_allocation_matrix_is(subsets, bkd)
        assert np.allclose(allocation_mat, np.eye(len(subsets)))

        # remove subset 0
        subsets = get_model_subsets(nmodels, bkd)[1:]
        subsets = _nest_subsets(subsets, nmodels, bkd)[0]
        idx = sorted(
            list(range(len(subsets))),
            key=lambda ii: (len(subsets[ii]), tuple(nmodels-subsets[ii])),
            reverse=True)
        subsets = [subsets[ii] for ii in idx]
        nsubsets = len(subsets)
        allocation_mat = _get_allocation_matrix_nested(subsets, bkd)
        assert np.allclose(
            allocation_mat, np.tril(np.ones((nsubsets, nsubsets))))

        # import matplotlib.pyplot as plt
        # from group_acv import _plot_allocation_matrix
        # ax = plt.subplots(1, 1, figsize=(8, 6))[1]
        # _plot_allocation_matrix(allocation_mat, subsets, ax)
        # plt.savefig("groupacvnested.pdf")

    def _check_separate_samples(self, est):
        bkd = self.get_backend()
        NN = 2
        npartition_samples = bkd.full((est.nsubsets(),), NN)
        est._set_optimized_params(npartition_samples)

        samples_per_model = est.generate_samples_per_model(
            lambda n: bkd.arange(n)[None, :])
        for ii in range(est.nmodels()):
            assert (samples_per_model[ii].shape[1] ==
                    est._rounded_nsamples_per_model[ii])
        values_per_model = [
            (ii+1)*s.T for ii, s in enumerate(samples_per_model)]
        values_per_subset = est._separate_values_per_model(values_per_model)

        test_samples = bkd.arange(
            est._rounded_npartition_samples.sum())[None, :]
        test_values = [(ii+1)*test_samples.T for ii in range(est.nmodels())]
        for ii in range(est.nsubsets()):
            active_partitions = bkd.where(est._allocation_mat[ii] == 1)[0]
            indices = bkd.arange(test_samples.shape[1], dtype=int).reshape(
                est.npartitions(), NN)[active_partitions].flatten()
            assert np.allclose(
                values_per_subset[ii].shape,
                (est._nintersect_samples(npartition_samples)[ii][ii],
                 len(est._subsets[ii])))
            for jj, s in enumerate(est._subsets[ii]):
                assert bkd.allclose(
                    values_per_subset[ii][:,  jj], test_values[s][indices, 0])

    def test_nsamples_per_model(self):
        bkd = self.get_backend()
        nmodels = 3
        cov = bkd.array(np.random.normal(0, 1, (nmodels, nmodels)))
        cov = cov.T @ cov
        costs = bkd.arange(nmodels, 0, -1)

        stat = multioutput_stats["mean"](1)
        stat.set_pilot_quantities(cov)
        est = GroupACVEstimator(stat, costs)
        npartition_samples = bkd.arange(2., 2+est.nsubsets(), dtype=float)
        assert bkd.allclose(
            est._compute_nsamples_per_model(npartition_samples),
            bkd.array([21, 23, 25]))
        assert np.allclose(
            est._estimator_cost(npartition_samples), 21*3+23*2+25*1)
        assert np.allclose(est._nintersect_samples(npartition_samples),
                           np.diag(npartition_samples))
        self._check_separate_samples(est)

        est = GroupACVEstimator(
            stat, costs, est_type="nested")
        npartition_samples = bkd.arange(2., 2.+est.nsubsets(), dtype=float)
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
        bkd = self.get_backend()
        ntrials = int(ntrials)
        cov = bkd.array(np.random.normal(0, 1, (nmodels, nmodels)))
        cov = cov.T @ cov
        cov = bkd.get_correlation_from_covariance(cov)
        costs = bkd.arange(nmodels, 0, -1)
        variable = IndependentMarginalsVariable(
            [stats.norm(0, 1) for ii in range(nmodels)], backend=bkd
        )
        stat = multioutput_stats["mean"](1)
        stat.set_pilot_quantities(cov)
        est = GroupACVEstimator(
            stat, costs, est_type=group_type, asketch=asketch
        )
        npartition_samples = bkd.arange(2., 2+est.nsubsets(), dtype=float)
        est._set_optimized_params(npartition_samples)
        est_var = est._covariance_from_npartition_samples(
            est._rounded_npartition_samples)

        chol_factor = bkd.cholesky(cov)
        exact_means = bkd.arange(nmodels)
        generate_values = partial(
            self._generate_correlated_values, chol_factor, exact_means
        )

        subset_means = []
        acv_means = []

        for nn in range(ntrials):
            samples_per_model = est.generate_samples_per_model(variable.rvs)
            values_per_model = [
                generate_values(samples_per_model[ii])[:, ii:ii+1]
                for ii in range(est.nmodels())]
            subset_values = est._separate_values_per_model(values_per_model)
            subset_means_nn = []
            acv_mean = 0
            for kk in range(est.nsubsets()):
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
        # np.set_printoptions(linewidth=1000, precision=4)
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

    def _check_gradient_optimization(self, nmodels, min_nhf_samples):
        # check specialized mlblue objective is consitent with
        # more general groupacv estimator when computing a single mean
        bkd = self.get_backend()
        cov = bkd.array(np.random.normal(0, 1, (nmodels, nmodels)))
        cov = cov.T @ cov
        cov = bkd.get_correlation_from_covariance(cov)

        target_cost = 100
        costs = bkd.copy(bkd.flip(bkd.logspace(-nmodels+1, 0, nmodels)))
        stat = multioutput_stats["mean"](1)
        stat.set_pilot_quantities(cov)
        gest = GroupACVEstimator(stat, costs, reg_blue=0)
        # todo use hyperparameter to set npartition_samples
        # todo move all member variables private and add functions to access
        # iterate = bkd.full((gest.npartitions(), 1), 1.)
        iterate = gest._init_guess(target_cost)[:, None]
        # assert bkd.min(errors)/bkd.max(errors) < 1e-6 and errors[0] > 0.2

        opt = GroupACVGradientOptimizer(ScipyConstrainedOptimizer())
        opt.set_estimator(gest)
        opt.set_budget(target_cost)
        errors = opt._optimizer._objective.check_apply_jacobian(iterate)
        assert errors.min()/errors.max() < 1e-6 and errors.max() > 0.1
        errors = opt._optimizer._objective.check_apply_hessian(iterate)
        assert errors.min()/errors.max() < 1e-6 and errors.max() > 0.1
        errors = opt._constraint.check_apply_jacobian(iterate)
        # constraints are linear so jacobian will be exact with largest
        # finite difference size, so check just the first entry of errors
        assert errors[0] < 1e-12
        weights = bkd.ones((opt._constraint.nqoi(), 1))
        errors = opt._constraint.check_apply_hessian(
            iterate, weights=weights, relative=False
        )
        assert errors[0] < 1e-12

        mlest = MLBLUEEstimator(stat, costs, reg_blue=0)
        opt = GroupACVGradientOptimizer(ScipyConstrainedOptimizer())
        opt.set_estimator(mlest)
        opt.set_budget(target_cost)
        errors = opt._optimizer._objective.check_apply_jacobian(iterate)
        assert errors.min()/errors.max() < 1e-6 and errors.max() > 0.1
        errors = opt._optimizer._objective.check_apply_hessian(iterate)
        assert errors.min()/errors.max() < 1e-6 and errors.max() > 0.1

        gest.set_optimizer(opt)
        gest.allocate_samples(target_cost, min_nhf_samples, iterate=iterate)
        mlest.set_optimizer(opt)
        mlest.allocate_samples(target_cost, min_nhf_samples, iterate=iterate)
        assert np.allclose(
            mlest._covariance_from_npartition_samples(
                gest._rounded_npartition_samples
            ),
            gest._covariance_from_npartition_samples(
                gest._rounded_npartition_samples)
        )

        # todo test mlblue with subsets that result in mfmc allocation
        # and compare optimzed answer with analytic answer

    def test_gradient_optimization(self):
        test_cases = [
            [2, 1], [3, 1], [4, 1], [3, 10],
        ]
        for test_case in test_cases:
            np.random.seed(1)
            self._check_gradient_optimization(*test_case)

    def _check_mlblue_spd(self, nmodels, min_nhf_samples):
        bkd = self.get_backend()
        cov = bkd.array(np.random.normal(0, 1, (nmodels, nmodels)))
        cov = cov.T @ cov
        cov = bkd.get_correlation_from_covariance(cov)

        target_cost = 100
        costs = bkd.copy(bkd.flip(bkd.logspace(-nmodels+1, 0, nmodels)))

        stat = multioutput_stats["mean"](1)
        stat.set_pilot_quantities(cov)
        mlest = MLBLUEEstimator(stat, costs, reg_blue=0)
        opt = MLBLUESPDOptimizer()
        opt.set_estimator(mlest)
        mlest.set_optimizer(opt)
        mlest.allocate_samples(target_cost, min_nhf_samples)

        gest = MLBLUEEstimator(stat, costs, reg_blue=0)
        opt1 = GroupACVGradientOptimizer(
            ScipyConstrainedNelderMeadOptimizer(opts={"maxiter": 100})
        )
        opt1.set_estimator(gest)
        opt2 = GroupACVGradientOptimizer(ScipyConstrainedOptimizer())
        opt2.set_estimator(gest)
        opt = ChainedACVOptimizer(opt1, opt2)
        gest.set_optimizer(opt)
        iterate = gest._init_guess(target_cost)[:, None]
        gest.allocate_samples(target_cost, min_nhf_samples, iterate=iterate)

        print(gest._optimized_criteria, mlest._optimized_criteria)
        assert np.allclose(gest._optimized_criteria,
                           mlest._optimized_criteria, rtol=1e-3)

    @unittest.skipIf(not package_available("cvxpy"), "cvxpy not installed")
    def test_mlblue_spd(self):
        test_cases = [
            [2, 1], [3, 1], [4, 1], [3, 10],
        ]
        for test_case in test_cases:
            np.random.seed(1)
            self._check_mlblue_spd(*test_case)

    def _check_insert_pilot_samples(self, nmodels, min_nhf_samples, seed):
        np.random.seed(seed)
        bkd = self.get_backend()
        cov = bkd.array(np.random.normal(0, 1, (nmodels, nmodels)))
        cov = cov.T @ cov
        cov = bkd.get_correlation_from_covariance(cov)

        variable = IndependentMarginalsVariable(
            [stats.norm(0, 1) for ii in range(nmodels)], backend=bkd)
        chol_factor = bkd.cholesky(cov)
        exact_means = np.arange(nmodels)
        generate_values = partial(
            self._generate_correlated_values, chol_factor, exact_means)

        npilot_samples = 8
        assert min_nhf_samples > npilot_samples

        target_cost = 100
        costs = bkd.copy(bkd.flip(bkd.logspace(-nmodels+1, 0, nmodels)))
        stat = multioutput_stats["mean"](1)
        stat.set_pilot_quantities(cov)
        est = MLBLUEEstimator(stat, costs, reg_blue=1e-10)
        # opt = GroupACVGradientOptimizer(ScipyConstrainedOptimizer())
        # opt.set_estimator(est)
        # est.set_optimizer(opt)
        # iterate = est._init_guess(target_cost)[:, None]
        opt = MLBLUESPDOptimizer()
        opt.set_estimator(est)
        est.set_optimizer(opt)
        iterate = None
        est.allocate_samples(target_cost, min_nhf_samples, iterate=iterate)
        # trust-const method does not work on some platforms.
        # It produces an interate and the lower bound and the SubBarrierProblem
        # returns a nan because it take the log of 0=x-lb.
        # est.allocate_samples(
        #     target_cost, min_nhf_samples=min_nhf_samples,
        #     optim_options={"maxiter": 1000, "init_guess": {"maxfev": 1000},
        #                    "bounds": [1e-10, 1e6], "method": "slsqp"})

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
            for ii in range(est.nmodels())]
        values_per_model_recovered = est.insert_pilot_values(
            pilot_values, values_per_model_wo_pilot)

        np.random.seed(seed)
        samples_per_model = est.generate_samples_per_model(
            variable.rvs)
        values_per_model = [
            generate_values(samples_per_model[ii])[:, ii:ii+1]
            for ii in range(est.nmodels())]

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

    def test_mlblue_recovers_mfmc(self):
        nmodels = 3
        bkd = self.get_backend()
        cov = bkd.array(np.random.normal(0, 1, (nmodels, nmodels)))
        cov = cov.T @ cov
        cov = bkd.get_correlation_from_covariance(cov)

        target_cost = 100
        costs = bkd.copy(bkd.flip(bkd.logspace(-nmodels+1, 0, nmodels)))
        subsets = [[0, 1], [1, 2], [2]]
        subsets = [bkd.array(s, dtype=int) for s in subsets]
        stat = multioutput_stats["mean"](1)
        stat.set_pilot_quantities(cov)
        est = GroupACVEstimator(
            stat, costs, reg_blue=0, subsets=subsets, est_type="nested")
        opt1 = GroupACVGradientOptimizer(
            ScipyConstrainedNelderMeadOptimizer(opts={"maxiter": 100})
        )
        opt1.set_estimator(est)
        opt2 = GroupACVGradientOptimizer(
            ScipyConstrainedOptimizer(opts={"gtol": 1e-10})
        )
        opt2.set_estimator(est)
        opt = ChainedACVOptimizer(opt1, opt2)
        est.set_optimizer(opt)
        iterate = est._init_guess(target_cost)[:, None]
        est.allocate_samples(
            target_cost, iterate=iterate, round_nsamples=False
        )
        mfmc_model_ratios, mfmc_log_variance = _allocate_samples_mfmc(
            bkd.to_numpy(cov), bkd.to_numpy(costs), target_cost
        )
        partition_ratios = MFMCEstimator._native_ratios_to_npartition_ratios(
            mfmc_model_ratios
        )
        mfmc_est = MFMCEstimator(stat, bkd.to_numpy(costs))
        npartition_samples = (
            mfmc_est._npartition_samples_from_partition_ratios(
                target_cost, bkd.array(partition_ratios)
            )
        )
        assert np.allclose(
            est._compute_nsamples_per_model(npartition_samples),
            mfmc_est._compute_nsamples_per_model(npartition_samples)
        )
        assert np.allclose(
            np.exp(mfmc_log_variance),
            est._covariance_from_npartition_samples(
                npartition_samples).item()
        )
        assert np.allclose(est._optimized_criteria, np.exp(mfmc_log_variance))


class TestTorchGroupACV(TestGroupACV, unittest.TestCase):
    def get_backend(self):
        return TorchLinAlgMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
