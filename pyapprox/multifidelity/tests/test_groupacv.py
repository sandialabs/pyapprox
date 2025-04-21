import unittest
from functools import partial

import numpy as np
from scipy import stats

from pyapprox.multifidelity.groupacv import (
    get_model_subsets,
    GroupACVEstimator,
    _get_allocation_matrix_is,
    _get_allocation_matrix_nested,
    _nest_subsets,
    MLBLUEEstimator,
    GroupACVGradientOptimizer,
    MLBLUESPDOptimizer,
    ChainedACVOptimizer,
    MLBLUEGradientOptimizer,
)
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.util.backends.torch import TorchMixin

# from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.multifidelity.factory import multioutput_stats, get_estimator
from pyapprox.optimization.scipy import (
    ScipyConstrainedOptimizer,
    ScipyConstrainedDifferentialEvolutionOptimizer,
)
from pyapprox.multifidelity._optim import _allocate_samples_mfmc
from pyapprox.util.sys_utilities import package_available
from pyapprox.multifidelity.acv import MFMCEstimator


from pyapprox.multifidelity.tests.test_stats import (
    _setup_multioutput_model_subproblem,
)
from pyapprox.multifidelity.stats import (
    _nqoisq_nqoisq_subproblem,
    _nqoi_nqoisq_subproblem,
)

if package_available("pyrol"):
    has_pyrol = True
    from pyapprox.optimization.rol import ROLConstrainedOptimizer
else:
    has_pyrol = False


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
            key=lambda ii: (len(subsets[ii]), tuple(nmodels - subsets[ii])),
            reverse=True,
        )
        subsets = [subsets[ii] for ii in idx]
        nsubsets = len(subsets)
        allocation_mat = _get_allocation_matrix_nested(subsets, bkd)
        assert np.allclose(
            allocation_mat, np.tril(np.ones((nsubsets, nsubsets)))
        )

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
            lambda n: bkd.arange(n)[None, :]
        )
        for ii in range(est.nmodels()):
            assert (
                samples_per_model[ii].shape[1]
                == est._rounded_nsamples_per_model[ii]
            )
        values_per_model = [
            (ii + 1) * s.T for ii, s in enumerate(samples_per_model)
        ]
        values_per_subset = est._separate_values_per_model(values_per_model)

        test_samples = bkd.arange(est._rounded_npartition_samples.sum())[
            None, :
        ]
        test_values = [
            (ii + 1) * test_samples.T for ii in range(est.nmodels())
        ]
        for ii in range(est.nsubsets()):
            active_partitions = bkd.where(est._allocation_mat[ii] == 1)[0]
            indices = (
                bkd.arange(test_samples.shape[1], dtype=int)
                .reshape(est.npartitions(), NN)[active_partitions]
                .flatten()
            )
            assert np.allclose(
                values_per_subset[ii].shape,
                (
                    est._nintersect_samples(npartition_samples)[ii][ii],
                    len(est._subsets[ii]),
                ),
            )
            for jj, s in enumerate(est._subsets[ii]):
                assert bkd.allclose(
                    values_per_subset[ii][:, jj], test_values[s][indices, 0]
                )

    def test_nsamples_per_model(self):
        bkd = self.get_backend()
        nmodels = 3
        cov = bkd.array(np.random.normal(0, 1, (nmodels, nmodels)))
        cov = cov.T @ cov
        costs = bkd.arange(nmodels, 0, -1)

        stat = multioutput_stats["mean"](1, backend=bkd)
        stat.set_pilot_quantities(cov)
        est = GroupACVEstimator(stat, costs)
        npartition_samples = bkd.arange(2.0, 2 + est.nsubsets(), dtype=float)
        assert bkd.allclose(
            est._compute_nsamples_per_model(npartition_samples),
            bkd.array([21, 23, 25]),
        )
        assert np.allclose(
            est._estimator_cost(npartition_samples), 21 * 3 + 23 * 2 + 25 * 1
        )
        assert np.allclose(
            est._nintersect_samples(npartition_samples),
            np.diag(npartition_samples),
        )
        self._check_separate_samples(est)

        est = GroupACVEstimator(stat, costs, est_type="nested")
        npartition_samples = bkd.arange(2.0, 2.0 + est.nsubsets(), dtype=float)

        assert bkd.allclose(
            est._compute_nsamples_per_model(npartition_samples),
            bkd.array([9, 20, 27]),
        )
        assert bkd.allclose(
            est._estimator_cost(npartition_samples),
            bkd.array([9 * 3 + 20 * 2 + 27 * 1]),
        )
        assert bkd.allclose(
            est._nintersect_samples(npartition_samples),
            bkd.array(
                [
                    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                    [2.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                    [2.0, 5.0, 9.0, 9.0, 9.0, 9.0],
                    [2.0, 5.0, 9.0, 14.0, 14.0, 14.0],
                    [2.0, 5.0, 9.0, 14.0, 20.0, 20.0],
                    [2.0, 5.0, 9.0, 14.0, 20.0, 27.0],
                ]
            ),
        )
        self._check_separate_samples(est)

    def _generate_correlated_values(self, chol_factor, means, samples):
        return (chol_factor @ samples + means[:, None]).T

    def _check_gradient_optimization(
        self, nmodels, min_nhf_samples, local_opt
    ):
        # check specialized mlblue objective is consitent with
        # more general groupacv estimator when computing a single mean
        bkd = self.get_backend()
        cov = bkd.array(np.random.normal(0, 1, (nmodels, nmodels)))
        cov = cov.T @ cov
        cov = bkd.covariance_to_correlation(cov)

        target_cost = 100
        costs = bkd.copy(bkd.flip(bkd.logspace(-nmodels + 1, 0, nmodels)))
        stat = multioutput_stats["mean"](1, backend=bkd)
        stat.set_pilot_quantities(cov)
        gest = GroupACVEstimator(stat, costs, reg_blue=0)
        # todo use hyperparameter to set npartition_samples
        # todo move all member variables private and add functions to access
        iterate = gest._init_guess(target_cost)
        opt = GroupACVGradientOptimizer(ScipyConstrainedOptimizer())
        opt.set_estimator(gest)
        opt.set_budget(target_cost)
        errors = opt._optimizer._objective.check_apply_jacobian(
            iterate, disp=True
        )
        assert errors.min() / errors.max() < 1e-6 and errors.max() > 0.1
        errors = opt._optimizer._objective.check_apply_hessian(iterate)
        assert errors.min() / errors.max() < 1e-6 and errors.max() > 0.05
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
        opt = GroupACVGradientOptimizer(local_opt)
        local_opt.set_verbosity(3)
        # opt = MLBLUEGradientOptimizer(local_opt)
        opt.set_estimator(mlest)
        opt.set_budget(target_cost)
        errors = opt._optimizer._objective.check_apply_jacobian(iterate)
        assert errors.min() / errors.max() < 1e-6 and errors.max() > 0.05
        errors = opt._optimizer._objective.check_apply_hessian(
            iterate, disp=True
        )
        assert errors.min() / errors.max() < 1e-6 and errors.max() > 0.01

        gest.set_optimizer(opt)
        gest.allocate_samples(target_cost, min_nhf_samples, iterate=iterate)
        mlest.set_optimizer(opt)
        mlest.allocate_samples(target_cost, min_nhf_samples, iterate=iterate)
        assert bkd.allclose(
            mlest._covariance_from_npartition_samples(
                gest._rounded_npartition_samples
            ),
            gest._covariance_from_npartition_samples(
                gest._rounded_npartition_samples
            ),
        )

        # Test multioutput estimation
        nqoi = 2
        cov = bkd.array(
            np.random.normal(0, 1, (nmodels * nqoi, nmodels * nqoi))
        )
        cov = cov.T @ cov
        cov = bkd.covariance_to_correlation(cov)
        target_cost = 100
        costs = bkd.copy(bkd.flip(bkd.logspace(-nmodels + 1, 0, nmodels)))

        stat = multioutput_stats["mean"](nqoi, backend=bkd)
        stat.set_pilot_quantities(cov)
        gest = GroupACVEstimator(stat, costs, reg_blue=0)
        iterate = gest._init_guess(target_cost)
        gopt = GroupACVGradientOptimizer(local_opt)
        gopt.set_estimator(gest)
        gopt.set_budget(target_cost)
        errors = gopt._optimizer._objective.check_apply_jacobian(
            iterate, disp=True
        )
        assert errors.min() / errors.max() < 1e-6 and errors.max() > 0.1
        errors = gopt._optimizer._objective.check_apply_hessian(iterate)
        assert errors.min() / errors.max() < 1e-6 and errors.max() > 0.05

        stat = multioutput_stats["mean"](nqoi, backend=bkd)
        stat.set_pilot_quantities(cov)
        mlest = MLBLUEEstimator(stat, costs, reg_blue=0)
        # opt = GroupACVGradientOptimizer(ScipyConstrainedOptimizer())
        local_opt = ScipyConstrainedOptimizer()
        mlopt = MLBLUEGradientOptimizer(local_opt)
        mlopt.set_estimator(mlest)
        mlopt.set_budget(target_cost)
        errors = mlopt._optimizer._objective.check_apply_jacobian(iterate)
        assert errors.min() / errors.max() < 1e-6 and errors.max() > 0.05
        errors = mlopt._optimizer._objective.check_apply_hessian(iterate)
        assert errors.min() / errors.max() < 1e-6 and errors.max() > 0.01

        gest.set_optimizer(gopt)
        gest.allocate_samples(target_cost, min_nhf_samples, iterate=iterate)
        mlest.set_optimizer(mlopt)
        mlest.allocate_samples(target_cost, min_nhf_samples, iterate=iterate)
        assert bkd.allclose(
            mlest._covariance_from_npartition_samples(
                gest._rounded_npartition_samples
            ),
            gest._covariance_from_npartition_samples(
                gest._rounded_npartition_samples
            ),
        )

    def test_gradient_optimization(self):
        test_cases = [
            [2, 1, ScipyConstrainedOptimizer()],
            [3, 1, ScipyConstrainedOptimizer()],
            [4, 1, ScipyConstrainedOptimizer()],
            [3, 10, ScipyConstrainedOptimizer()],
        ]
        if has_pyrol:
            test_cases += [
                [2, 1, ROLConstrainedOptimizer()],
                [3, 1, ROLConstrainedOptimizer()],
                [4, 1, ROLConstrainedOptimizer()],
                [3, 10, ROLConstrainedOptimizer()],
            ]
        for test_case in test_cases:
            np.random.seed(1)
            print(test_case)
            self._check_gradient_optimization(*test_case)

    def _check_mlblue_spd(self, nmodels, min_nhf_samples):
        bkd = self.get_backend()
        cov = bkd.array(np.random.normal(0, 1, (nmodels, nmodels)))
        cov = cov.T @ cov
        cov = bkd.covariance_to_correlation(cov)

        target_cost = 100
        costs = bkd.copy(bkd.flip(bkd.logspace(-nmodels + 1, 0, nmodels)))

        stat = multioutput_stats["mean"](1, backend=bkd)
        stat.set_pilot_quantities(cov)
        mlest = MLBLUEEstimator(stat, costs, reg_blue=0)
        opt = MLBLUESPDOptimizer()
        opt.set_estimator(mlest)
        mlest.set_optimizer(opt)
        mlest.allocate_samples(target_cost, min_nhf_samples)

        gest = MLBLUEEstimator(stat, costs, reg_blue=0)
        opt1 = GroupACVGradientOptimizer(
            ScipyConstrainedDifferentialEvolutionOptimizer(
                opts={"maxiter": 20, "disp": True}
            )
        )
        opt1.set_estimator(gest)
        scipy_opt = ScipyConstrainedOptimizer(opts={"gtol": 1e-9})
        scipy_opt.set_verbosity(3)
        opt2 = GroupACVGradientOptimizer(scipy_opt)
        opt2.set_estimator(gest)
        opt = ChainedACVOptimizer(opt1, opt2)
        gest.set_optimizer(opt)
        iterate = gest._init_guess(target_cost)
        gest.allocate_samples(target_cost, min_nhf_samples, iterate=iterate)
        # print(gest._optimized_criteria-mlest._optimized_criteria)
        assert np.allclose(
            gest._optimized_criteria, mlest._optimized_criteria, rtol=1e-3
        )

    @unittest.skipIf(not package_available("cvxpy"), "cvxpy not installed")
    def test_mlblue_spd(self):
        test_cases = [
            [2, 1],
            [3, 1],
            [4, 1],
            [3, 10],
        ]
        for test_case in test_cases:
            np.random.seed(1)
            print(test_case)
            self._check_mlblue_spd(*test_case)

    def _check_insert_pilot_samples(self, nmodels, min_nhf_samples, seed):
        np.random.seed(seed)
        bkd = self.get_backend()
        cov = bkd.array(np.random.normal(0, 1, (nmodels, nmodels)))
        cov = cov.T @ cov
        cov = bkd.covariance_to_correlation(cov)

        variable = IndependentMarginalsVariable(
            [stats.norm(0, 1) for ii in range(nmodels)], backend=bkd
        )
        chol_factor = bkd.cholesky(cov)
        exact_means = bkd.arange(nmodels)
        generate_values = partial(
            self._generate_correlated_values, chol_factor, exact_means
        )

        npilot_samples = 8
        assert min_nhf_samples > npilot_samples

        target_cost = 100
        costs = bkd.copy(bkd.flip(bkd.logspace(-nmodels + 1, 0, nmodels)))
        stat = multioutput_stats["mean"](1, backend=bkd)
        stat.set_pilot_quantities(cov)
        est = MLBLUEEstimator(stat, costs, reg_blue=1e-10)
        opt = GroupACVGradientOptimizer(ScipyConstrainedOptimizer())
        opt.set_estimator(est)
        est.set_optimizer(opt)
        iterate = est._init_guess(target_cost)
        est.allocate_samples(target_cost, min_nhf_samples, iterate=iterate)

        # the following test only works if variable.nvars()==1 because
        # variable.rvs does not produce nested samples when this condition does
        # not hold
        np.random.seed(seed)
        samples_per_model = est.generate_samples_per_model(variable.rvs)
        pilot_samples = est._remove_pilot_samples(
            npilot_samples, samples_per_model
        )[1]
        pilot_values = [
            generate_values(pilot_samples)[:, ii : ii + 1]
            for ii in range(nmodels)
        ]

        np.random.seed(seed)
        samples_per_model_wo_pilot = est.generate_samples_per_model(
            variable.rvs, npilot_samples
        )
        values_per_model_wo_pilot = [
            generate_values(samples_per_model_wo_pilot[ii])[:, ii : ii + 1]
            for ii in range(est.nmodels())
        ]
        values_per_model_recovered = est.insert_pilot_values(
            pilot_values, values_per_model_wo_pilot
        )

        np.random.seed(seed)
        samples_per_model = est.generate_samples_per_model(variable.rvs)
        values_per_model = [
            generate_values(samples_per_model[ii])[:, ii : ii + 1]
            for ii in range(est.nmodels())
        ]

        for v1, v2 in zip(values_per_model, values_per_model_recovered):
            assert bkd.allclose(v1, v2)

    def test_insert_pilot_samples(self):
        test_cases = [
            [2, 11],
            [3, 11],
            [4, 11],
        ]
        ntrials = 20
        for test_case in test_cases:
            for seed in range(ntrials):
                self._check_insert_pilot_samples(*test_case, seed)

    def test_groupacv_recovers_mfmc(self):
        nmodels = 3
        bkd = self.get_backend()
        cov = bkd.array(np.random.normal(0, 1, (nmodels, nmodels)))
        cov = cov.T @ cov
        cov = bkd.covariance_to_correlation(cov)

        target_cost = 100
        costs = bkd.copy(bkd.flip(bkd.logspace(-nmodels + 1, 0, nmodels)))
        subsets = [[0, 1], [1, 2], [2]]
        subsets = [bkd.array(s, dtype=int) for s in subsets]
        stat = multioutput_stats["mean"](1, backend=bkd)
        stat.set_pilot_quantities(cov)
        est = GroupACVEstimator(
            stat,
            costs,
            reg_blue=1e-8,
            model_subsets=subsets,
            est_type="nested",
        )
        mfmc_model_ratios, mfmc_log_variance = _allocate_samples_mfmc(
            cov, costs, target_cost, bkd
        )
        mfmc_est = MFMCEstimator(stat, costs)
        partition_ratios = mfmc_est._native_ratios_to_npartition_ratios(
            mfmc_model_ratios
        )
        npartition_samples = (
            mfmc_est._npartition_samples_from_partition_ratios(
                target_cost, partition_ratios
            )
        )
        assert np.allclose(
            est._compute_nsamples_per_model(npartition_samples),
            mfmc_est._compute_nsamples_per_model(npartition_samples),
        )
        assert np.allclose(
            np.exp(mfmc_log_variance),
            est._covariance_from_npartition_samples(npartition_samples).item(),
        )

    def _check_sigma_matrix_of_estimator(self, est, ntrials, funs, variable):
        bkd = self.get_backend()
        est_var = est._covariance_from_npartition_samples(
            est._rounded_npartition_samples
        )

        subset_vars = []
        acv_ests = []

        for nn in range(ntrials):
            samples_per_model = est.generate_samples_per_model(variable.rvs)
            values_per_model = [
                bkd.array(funs[ii](samples_per_model[ii]))
                for ii in range(est.nmodels())
            ]
            subset_values = est._separate_values_per_model(values_per_model)
            subset_vars_nn = []
            acv_est = 0
            for kk in range(est.nsubsets()):
                if subset_values[kk].shape[0] > 0:
                    subset_var = est._stat.sample_estimate(subset_values[kk])
                else:
                    subset_var = bkd.zeros(len(est.subsets[kk]))
                subset_vars_nn.append(subset_var)
            acv_est = est(values_per_model)
            # print(acv_est)
            # assert False
            subset_vars.append(bkd.hstack(subset_vars_nn))
            acv_ests.append(acv_est)
        # acv_ests = np.array(acv_ests)
        # subset_vars = np.array(subset_vars)
        # mc_group_cov = np.cov(subset_vars, ddof=1, rowvar=False)
        acv_ests = bkd.stack(acv_ests)
        subset_vars = bkd.stack(subset_vars)
        mc_group_cov = bkd.cov(subset_vars, ddof=1, rowvar=False)
        Sigma = est._sigma(est._rounded_npartition_samples)
        atol, rtol = 4e-3, 3e-2
        # print(est._stat)
        print(bkd.diag(mc_group_cov))
        print(bkd.diag(Sigma), "S")
        # print(bkd.diag(mc_group_cov) - bkd.diag(Sigma))
        # print(
        #     (bkd.diag(mc_group_cov) - bkd.diag(Sigma)) / bkd.diag(mc_group_cov)
        # )
        assert bkd.allclose(bkd.diag(mc_group_cov), bkd.diag(Sigma), rtol=rtol)
        assert bkd.allclose(mc_group_cov, Sigma, rtol=rtol, atol=atol)
        est_var_mc = bkd.cov(acv_ests, ddof=1, rowvar=False)
        # print(est_var_mc, "Z")
        # print(est_var)
        # print(est_var_mc - est_var)
        assert bkd.allclose(est_var_mc, est_var, rtol=rtol, atol=atol)

    def _check_sigma_matrix(
        self, nmodels, ntrials, group_type, stat_name, qoi_idx, asketch=None
    ):
        bkd = self.get_backend()
        ntrials = int(ntrials)
        cov, W, B, costs, funs, model = self._setup_variance_problem(
            nmodels, qoi_idx, bkd
        )
        variable = model.variable()
        pilot_quantities = {
            "mean": (cov,),
            "variance": (cov, W),
            "mean_variance": (cov, W, B),
        }
        stat_kwargs = {
            "mean": {},
            "variance": {"tril": True},
            "mean_variance": {"tril": True},
        }
        stat = multioutput_stats[stat_name](
            len(qoi_idx), backend=bkd, **stat_kwargs[stat_name]
        )
        stat.set_pilot_quantities(*pilot_quantities[stat_name])
        if nmodels == 3:
            model_subsets = [[0, 1], [1, 2], [2]]
            model_subsets = [bkd.array(s, dtype=int) for s in model_subsets]
        else:
            model_subsets = None
        est = GroupACVEstimator(
            stat,
            costs,
            est_type=group_type,
            asketch=asketch,
            reg_blue=0,
            model_subsets=model_subsets,
        )
        npartition_samples = bkd.arange(2.0, 2 + est.nsubsets(), dtype=float)
        est._set_optimized_params(npartition_samples)
        self._check_sigma_matrix_of_estimator(est, ntrials, funs, variable)

    def test_sigma_matrix(self):
        bkd = self.get_backend()

        def asketch_fun(n):
            return bkd.hstack((bkd.eye(n), bkd.zeros((n, n))))

        test_cases = [
            [2, 5e4, "is", "mean", [0]],
            [2, 5e4, "is", "mean", [0, 1]],
            [2, 5e4, "is", "mean", [0], [[0.5, 0.5]]],
            [2, 5e4, "nested", "mean", [0]],
            [3, 5e4, "is", "mean", [0]],
            [3, 5e4, "is", "mean", [0, 1, 2]],
            [3, 2e4, "nested", "mean", [0]],
            [2, 5e4, "is", "variance", [0]],
            [2, 5e4, "is", "variance", [0, 1]],
            [2, 5e4, "is", "variance", [0], [[0.5, 0.5]]],
            [2, 2e4, "nested", "variance", [0, 1]],
            [2, 5e4, "is", "mean_variance", [0], asketch_fun(2)],
            [2, 2e4, "nested", "mean_variance", [0, 1], asketch_fun(5)],
            [3, 5e4, "is", "variance", [0]],
            [3, 2e4, "nested", "variance", [0, 1]],
        ]
        for test_case in test_cases:
            np.random.seed(1)
            print(test_case)
            self._check_sigma_matrix(*test_case)

    def _setup_variance_problem(self, nmodels, qoi_idx, bkd, psd=False):
        model_idx = bkd.arange(nmodels)
        (
            funs,
            cov,
            costs,
            benchmark,
            means,
        ) = _setup_multioutput_model_subproblem(
            model_idx, qoi_idx, bkd, psd=psd
        )
        W = benchmark.covariance_of_centered_values_kronker_product()
        W = _nqoisq_nqoisq_subproblem(
            W, benchmark.nmodels(), benchmark.nqoi(), model_idx, qoi_idx, bkd
        )
        B = benchmark.covariance_of_mean_and_variance_estimators()
        B = _nqoi_nqoisq_subproblem(
            B, benchmark.nmodels(), benchmark.nqoi(), model_idx, qoi_idx, bkd
        )
        return cov, W, B, costs, funs, benchmark

    def _check_group_acv_to_traditional_acv(
        self, est, mfmc_est, benchmark, funs
    ):
        # assumes nested mfmc subsets e.g. [0, 1] [1, 2], [2] with 3 models
        bkd = self.get_backend()
        import torch

        torch.set_printoptions(precision=12, linewidth=1000)
        # print(est._traditional_acv_weights())
        # print(
        #     mfmc_est._optimized_weights, mfmc_est._optimized_weights.shape, "W"
        # )
        samples_per_model = est.generate_samples_per_model(
            benchmark.variable().rvs
        )
        values_per_model = [
            bkd.array(funs[ii](samples_per_model[ii]))
            for ii in range(est.nmodels())
        ]
        subset_values = est._separate_values_per_model(values_per_model)
        subset_ests = []
        for kk in range(est.nsubsets()):
            if subset_values[kk].shape[0] > 0:
                print(kk, "K")
                subset_est = est._stat.sample_estimate(subset_values[kk])
                print(subset_est.shape, "V")
            else:
                subset_est = bkd.zeros(len(est.subsets[kk]))
            subset_ests.append(subset_est)
        Q0, Qe, Qu = est._group_to_traditional_estimators(subset_ests)
        # Q0, Qe, Qu = est._group_to_traditional_estimators_from_alpha(
        #     subset_ests, mfmc_est._optimized_weights
        # )
        nqoi = est._stat.nqoi()
        print(subset_values[0].shape, subset_values[1].shape, nqoi)
        print(
            subset_values[0].mean(axis=0),
            bkd.var(subset_values[0], axis=0, ddof=1),
            "M0",
        )
        print(
            subset_values[1].mean(axis=0),
            bkd.var(subset_values[1], axis=0, ddof=1),
            "M2",
        )
        print(subset_ests)
        if est.nmodels() == 2:
            acv_values = [
                bkd.zeros((0, nqoi)),
                subset_values[0][:, :nqoi],
                subset_values[0][:, nqoi : 2 * nqoi],
                subset_values[1],
            ]
        else:
            acv_values = [
                bkd.zeros((0, nqoi)),
                subset_values[0][:, :nqoi],
                subset_values[0][:, nqoi : 2 * nqoi],
                subset_values[1][:, :nqoi],
                subset_values[1][:, nqoi : 2 * nqoi],
                subset_values[2],
            ]

        # print(acv_values[0])
        # print(acv_values[1], subset_values[0], "a")
        print(bkd.var(acv_values[1], axis=0, ddof=1))
        print(mfmc_est)
        Q0_acv = mfmc_est._stat.sample_estimate(acv_values[1])
        Qe_acv = bkd.stack(
            [
                mfmc_est._stat.sample_estimate(acv_values[2 * ii])
                for ii in range(1, est.nmodels())
            ],
            axis=1,
        )
        Qu_acv = bkd.stack(
            [
                mfmc_est._stat.sample_estimate(acv_values[2 * ii + 1])
                for ii in range(1, est.nmodels())
            ],
            axis=1,
        )
        print(Q0, "Q0")
        print(Q0_acv, "Q0_acv")
        assert bkd.allclose(Q0, Q0_acv)
        print(Qe, "Qe")
        print(Qe_acv, "Qe_acv")
        print(Qu, "Qu")
        print(Qu_acv, "Qu_acv")
        print(bkd.hstack((Qe, Qu)))
        print(bkd.hstack((Qe_acv, Qu_acv)))
        for ii in range(Qe.shape[0]):
            print(
                bkd.sort(bkd.hstack((Qe[ii], Qu[ii]))),
                bkd.sort(bkd.hstack((Qe_acv[ii], Qu_acv[ii]))),
            )
            assert bkd.allclose(
                bkd.sort(bkd.abs(bkd.hstack((Qe[ii], Qu[ii])))),
                bkd.sort(bkd.abs(bkd.hstack((Qe_acv[ii], Qu_acv[ii])))),
            )
        assert bkd.allclose(Qe, Qe_acv)
        assert bkd.allclose(Qu, Qu_acv)
        # weight conversion is only unqiue up to a sign
        print(bkd.abs(est._traditional_acv_weights()))
        print(bkd.abs(mfmc_est._optimized_weights))
        assert bkd.allclose(
            bkd.abs(est._traditional_acv_weights()),
            bkd.abs(mfmc_est._optimized_weights),
        )

    def _check_mfmc_nested_estimation(self, nmodels, qoi_idx, statname):
        bkd = self.get_backend()
        est_type = "nested"
        # est_type = "is"
        target_cost = 10  # #50  # 100
        cov, W, B, costs, funs, benchmark = self._setup_variance_problem(
            nmodels, qoi_idx, bkd, psd=True
        )
        # costs = bkd.copy(bkd.flip(bkd.logspace(-nmodels + 1, 0, nmodels)))
        # print(costs)
        # costs = bkd.array([3, 1.1, 1])[:nmodels]
        # check that group acv computes the same estimator variance
        # as mfmc when estimating variance using the optimal mfmc sample
        # allocation for estimating variance
        stat = multioutput_stats[statname](len(qoi_idx), backend=bkd)
        pilot_quantities = {
            "mean": (cov,),
            "variance": (cov, W),
            "mean_variance": (cov, W, B),
        }[statname]
        print(pilot_quantities)
        stat.set_pilot_quantities(*pilot_quantities)
        if nmodels == 3:
            subsets = [[0, 1], [1, 2], [2]]
            if est_type == "nested":
                recursion_index = bkd.array([0, 1], dtype=int)
                acv_est_type = "gmf"
            else:
                recursion_index = bkd.array([0, 0], dtype=int)
                acv_est_type = "grd"
        if nmodels == 2:
            subsets = [[0, 1], [1]]
            recursion_index = bkd.array([0], dtype=int)
            if est_type == "nested":
                acv_est_type = "gmf"
            else:
                acv_est_type = "grd"

        subsets = [bkd.array(s, dtype=int) for s in subsets]
        est = GroupACVEstimator(
            stat,
            costs,
            model_subsets=subsets,
            est_type=est_type,
            reg_blue=0,
            use_pseudo_inv=False,
        )
        print(recursion_index)
        mfmc_stat = multioutput_stats[statname](len(qoi_idx), backend=bkd)
        mfmc_stat.set_pilot_quantities(*pilot_quantities)
        mfmc_est = get_estimator(
            acv_est_type, mfmc_stat, costs, recursion_index=recursion_index
        )
        mfmc_est.allocate_samples(target_cost)
        print(mfmc_est)

        samples_per_model = mfmc_est.generate_samples_per_model(
            lambda n: benchmark.variable().rvs(n)
        )
        values_per_model = [
            fun(samples) for fun, samples in zip(funs, samples_per_model)
        ]
        mfmc_est_val = mfmc_est(values_per_model)

        # apply mfmc sample allocaiton to group acv
        est.set_optimizer(est.get_default_optimizer())
        est._set_optimized_params(
            mfmc_est._rounded_npartition_samples,
            round_nsamples=False,
        )

        # self._check_group_acv_to_traditional_acv(
        #     est, mfmc_est, benchmark, funs
        # )

        # self._check_sigma_matrix_of_estimator(
        #     est, int(2e4), funs, benchmark.variable()
        # )

        # from pyapprox.multifidelity.factory import (
        #     numerically_compute_estimator_variance,
        # )

        # hfcovar_mc, hfcovar, covar_mc, covar, est_vals, Q, delta = (
        #     numerically_compute_estimator_variance(
        #         funs,
        #         benchmark.variable(),
        #         mfmc_est,
        #         int(1e3),
        #         1,
        #         True,
        #     )
        # )
        # hfcovar = hfcovar.numpy()
        # atol, rtol = 1e-2, 1e-2
        # # print(covar_mc, covar, "Y")
        # assert bkd.allclose(covar_mc, covar, atol=atol, rtol=rtol)

        groupacv_est_val = est(values_per_model)
        print(mfmc_est_val, groupacv_est_val, "V")
        print(mfmc_est._optimized_covariance, "MFMC")
        print(est._optimized_covariance, "GROUPACV")
        # print(est._optimized_covariance - mfmc_est._optimized_covariance)
        assert bkd.allclose(
            est._optimized_covariance, mfmc_est._optimized_covariance
        )
        assert np.allclose(groupacv_est_val, mfmc_est_val)

        # check optimization of group acv
        stat = multioutput_stats[statname](len(qoi_idx), backend=bkd)
        stat.set_pilot_quantities(*pilot_quantities)
        est = GroupACVEstimator(
            stat,
            costs,
            model_subsets=subsets,
            est_type=est_type,
            use_pseudo_inv=False,
        )

        opt1 = GroupACVGradientOptimizer(
            ScipyConstrainedDifferentialEvolutionOptimizer(
                opts={"maxiter": 20, "disp": False}
            )
        )
        opt2 = GroupACVGradientOptimizer(ScipyConstrainedOptimizer())
        opt = ChainedACVOptimizer(opt1, opt2)
        est.set_optimizer(opt)
        iterate = est._init_guess(target_cost)
        est.allocate_samples(target_cost, iterate=iterate, round_nsamples=True)
        # print(est._optimized_covariance)
        # print(mfmc_est._optimized_covariance)
        # print(est._optimized_covariance - mfmc_est._optimized_covariance)
        # print(est._optimized_criteria, mfmc_est._optimized_criteria)
        assert np.allclose(
            est._optimized_covariance, mfmc_est._optimized_covariance
        )
        assert np.allclose(
            est._optimized_criteria, mfmc_est._optimized_criteria
        )

    def test_mfmc_nested_estimation(self):
        test_cases = [
            # [2, [0], "mean"],
            # [3, [0], "mean"],
            # [2, [0, 1], "mean"],
            # [2, [0, 1, 2], "mean"],
            # [3, [0, 1], "mean"],
            # [3, [0, 1, 2], "mean"],
            # [2, [0], "variance"],
            # [2, [0, 1, 2], "variance"],
            # [3, [0, 1], "variance"],
            # [2, [0], "mean_variance"],
            [2, [0, 1], "mean_variance"],
        ]
        for test_case in test_cases:
            np.random.seed(1)
            self._check_mfmc_nested_estimation(*test_case)

    def test_restriction_matrices(self):
        bkd = self.get_backend()
        # nqoi = 1
        qoi_idx = [0]
        costs = bkd.array([1, 0.5, 0.25])
        stat = multioutput_stats["mean"](len(qoi_idx), backend=bkd)
        subsets = [[0, 1, 2], [1], [2]]
        subsets = [bkd.array(s, dtype=int) for s in subsets]
        est = GroupACVEstimator(
            stat, costs, model_subsets=subsets, est_type="nested"
        )

        # vector containing model ids
        Lvec = bkd.arange(3, dtype=bkd.double_type())[:, None]
        # make sure each restriction matrix recovers correct subset mdoel ids
        # from Lvec (only works for nqoi = 1)
        cnt = 0
        for ii in range(len(subsets)):
            assert bkd.allclose(
                (est._R[:, cnt : cnt + len(subsets[ii])].T @ Lvec)[:, 0],
                bkd.asarray(subsets[ii], dtype=bkd.double_type()),
            )
            cnt += len(subsets[ii])

        # nqoi = 2
        qoi_idx = [0, 1]
        costs = bkd.array([1, 0.5, 0.25])
        stat = multioutput_stats["mean"](len(qoi_idx), backend=bkd)
        subsets = [[0, 1, 2], [1], [2]]
        subsets = [bkd.array(s, dtype=int) for s in subsets]
        est = GroupACVEstimator(
            stat, costs, model_subsets=subsets, est_type="nested"
        )
        # vector containing flattend model qoi ids
        Lvec = bkd.arange(3 * len(qoi_idx), dtype=bkd.double_type())[:, None]
        # make sure each restriction matrix recovers all the correct qoi of all
        # subset model ids
        assert bkd.allclose(
            (est._R[:, :6].T @ Lvec)[:, 0], bkd.array([0, 1, 2, 3, 4, 5])
        )
        assert bkd.allclose((est._R[:, 6:8].T @ Lvec)[:, 0], bkd.array([2, 3]))
        assert bkd.allclose(
            (est._R[:, 8:10].T @ Lvec)[:, 0], bkd.array([4, 5])
        )


class TestTorchGroupACV(TestGroupACV, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


# class TestNumpyGroupACV(TestGroupACV, unittest.TestCase):
#     def get_backend(self):
#         return NumpyMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
