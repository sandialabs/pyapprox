import unittest
import torch
from functools import partial

import numpy as np

from pyapprox.multifidelity.stats import (
    get_V_from_covariance, covariance_of_variance_estimator,
    _nqoisq_nqoisq_subproblem, _nqoi_nqoisq_subproblem, )
from pyapprox.multifidelity.multioutput_monte_carlo import (
    get_estimator, ACVEstimator)
from pyapprox.multifidelity.control_variate_monte_carlo import (
    allocate_samples_mfmc)
from pyapprox.multifidelity.multioutput_monte_carlo import (
    log_trace_variance)
from pyapprox.multifidelity.tests.test_stats import (
    _setup_multioutput_model_subproblem, _single_qoi, _two_qoi)

# import builtins
# from inspect import getframeinfo, stack
# original_print = print

# def print_wrap(*args, **kwargs):
#     caller = getframeinfo(stack()[1][0])
#     original_print("FN:",caller.filename,"Line:", caller.lineno,"Func:", caller.function,":::", *args, **kwargs)

# builtins.print = print_wrap


def _estimate_components(est, funs, ii):
    random_state = np.random.RandomState(ii)
    est.set_random_state(random_state)
    mc_est = est.stat.sample_estimate
    acv_samples, acv_values = est.generate_data(funs)
    est_val = est(acv_values)
    Q = mc_est(acv_values[0][1])
    if isinstance(est, ACVEstimator):
        delta = np.hstack([mc_est(acv_values[ii][0]) -
                           mc_est(acv_values[ii][1])
                           for ii in range(1, est.nmodels)])
    else:
        delta = Q*0
    return est_val, Q, delta


def _log_single_qoi_criteria(
        qoi_idx, stat_type, criteria_type, variance):
    # use if stat == Variance and target is variance
    # return torch.log(variance[qoi_idx[0], qoi_idx[0]])
    # use if stat == MeanAndVariance and target is variance
    if stat_type == "mean_var" and criteria_type == "var":
        return torch.log(variance[3+qoi_idx[0], 3+qoi_idx[0]])
    if stat_type == "mean_var" and criteria_type == "mean":
        return torch.log(variance[qoi_idx[0], qoi_idx[0]])
    raise ValueError


class TestMOMC(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

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

    def _check_estimator_covariances(self, model_idx, qoi_idx):
        nsamples, ntrials = 20, int(1e5)
        funs, cov, costs, model = _setup_multioutput_model_subproblem(
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
        V_exact = get_V_from_covariance(cov, nmodels)
        W_exact = model.covariance_of_centered_values_kronker_product()
        W_exact = _nqoisq_nqoisq_subproblem(
            W_exact, model.nmodels, model.nqoi, model_idx, qoi_idx)
        cov_var_exact = covariance_of_variance_estimator(
            W_exact, V_exact, nsamples)
        assert np.allclose(
           cov_var_exact, mc_mean_cov_var[nqoi*nmodels:, nqoi*nmodels:],
           atol=atol, rtol=rtol)

    def test_estimator_covariances(self):
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
            self._check_estimator_covariances(*test_case)

    def _check_separate_samples(self, est_type, max_nmodels=None):
        funs, cov, costs, model = _setup_multioutput_model_subproblem(
            [0, 1, 2], [0, 1, 2])
        costs = [3, 2, 1]
        est = get_estimator(
            est_type, "mean", model.variable, costs, cov,
            max_nmodels=max_nmodels, recursion_index=[2, 0])
        target_cost = 30
        est.allocate_samples(target_cost, verbosity=1)
        acv_samples, acv_values = est.generate_data(funs)
        samples_per_model = est.combine_acv_samples(acv_samples)
        values_per_model = est.combine_acv_values(acv_values)

        nmodels = len(acv_values)
        for ii in range(nmodels):
            print(est.nsamples_per_model[ii],
                  samples_per_model[ii].shape[1])
            assert np.allclose(est.nsamples_per_model[ii],
                               samples_per_model[ii].shape[1])
            assert np.allclose(est.nsamples_per_model[ii],
                               values_per_model[ii].shape[0])

        acv_values1 = est.separate_model_values(values_per_model)
        acv_samples1 = est.separate_model_samples(samples_per_model)

        assert np.allclose(acv_values[0][1], acv_values1[0][1])
        assert np.allclose(acv_samples[0][1], acv_samples1[0][1])
        for ii in range(1, nmodels):
            assert np.allclose(acv_values[ii][0], acv_values1[ii][0])
            assert np.allclose(acv_values[ii][1], acv_values1[ii][1])
            assert np.allclose(acv_samples[ii][0], acv_samples1[ii][0])
            assert np.allclose(acv_samples[ii][1], acv_samples1[ii][1])

    def test_separate_samples(self):
        test_cases = [
            ["gmf"], ["mfmc"], ["mlmc"], ["gmf", 3], ["gis"]
        ]
        for test_case in test_cases[-1:]:
            self._check_separate_samples(*test_case)

    def _estimate_components_loop(
            self, ntrials, est, funs, max_eval_concurrency):
        if max_eval_concurrency == 1:
            Q = []
            delta = []
            estimator_vals = []
            for ii in range(ntrials):
                est_val, Q_val, delta_val = _estimate_components(est, funs, ii)
                estimator_vals.append(est_val)
                Q.append(Q_val)
                delta.append(delta_val)
            Q = np.array(Q)
            delta = np.array(delta)
            estimator_vals = np.array(estimator_vals)
            return estimator_vals, Q, delta

        from multiprocessing import Pool
        # set flat funs to none so funs can be pickled
        pool = Pool(max_eval_concurrency)
        func = partial(_estimate_components, est, funs)
        result = pool.map(func, list(range(ntrials)))
        pool.close()
        estimator_vals = np.asarray([r[0] for r in result])
        Q = np.asarray([r[1] for r in result])
        delta = np.asarray([r[2] for r in result])
        return estimator_vals, Q, delta

    def _check_estimator_variances(self, model_idx, qoi_idx, recursion_index,
                                   est_type, stat_type, tree_depth=None,
                                   max_nmodels=None, ntrials=int(1e4)):
        rtol, atol = 4.6e-2, 1.01e-3
        funs, cov, costs, model = _setup_multioutput_model_subproblem(
            model_idx, qoi_idx)
        nqoi = len(qoi_idx)
        args = []
        if est_type == "gmf":
            if tree_depth is not None:
                kwargs = {"tree_depth": tree_depth}
            else:
                kwargs = {"recursion_index": np.asarray(recursion_index)}
        else:
            kwargs = {}
        if stat_type == "mean":
            idx = nqoi
        if "variance" in stat_type:
            W = model.covariance_of_centered_values_kronker_product()
            W = _nqoisq_nqoisq_subproblem(
                W, model.nmodels, model.nqoi, model_idx, qoi_idx)
            # npilot_samples = int(1e6)
            # pilot_samples = model.variable.rvs(npilot_samples)
            # pilot_values = np.hstack([f(pilot_samples) for f in funs])
            # W = get_W_from_pilot(pilot_values, nmodels)
            args.append(W)
            idx = nqoi**2
        if stat_type == "mean_variance":
            B = model.covariance_of_mean_and_variance_estimators()
            B = _nqoi_nqoisq_subproblem(
                B, model.nmodels, model.nqoi, model_idx, qoi_idx)
            args.append(B)
            idx = nqoi+nqoi**2
        est = get_estimator(
            est_type, stat_type, model.variable, costs, cov, *args,
            max_nmodels=max_nmodels, **kwargs)

        # must call opt otherwise best_est will not be set for
        # best model subset acv
        est.allocate_samples(100)
        # est.nsamples_per_model = torch.tensor([10, 20, 30][:len(funs)])
        # est.nsamples_per_model = torch.tensor([7, 8, 140][:len(funs)])

        CF, cf = est.stat.get_discrepancy_covariances(
            est, est.nsamples_per_model)
        print(CF.shape)
        # assert False

        max_eval_concurrency = 4
        estimator_vals, Q, delta = self._estimate_components_loop(
            ntrials, est, funs, max_eval_concurrency)

        CF_mc = torch.as_tensor(
            np.cov(delta.T, ddof=1), dtype=torch.double)
        cf_mc = torch.as_tensor(
            np.cov(Q.T, delta.T, ddof=1)[:idx, idx:], dtype=torch.double)

        # np.set_printoptions(linewidth=1000)
        # print(estimator_vals.mean(axis=0).reshape(nqoi, nqoi))
        # print(model.covariance()[:nqoi:, :nqoi])

        hf_var_mc = np.cov(Q.T, ddof=1)
        hf_var = est.stat.high_fidelity_estimator_covariance(
            est.nsamples_per_model)
        # print(hf_var_mc)
        # print(hf_var.numpy())
        print(((hf_var_mc-hf_var.numpy())/hf_var.numpy()).max())
        assert np.allclose(hf_var_mc, hf_var, atol=atol, rtol=rtol)

        if est_type != "mc":
            CF, cf = est.stat.get_discrepancy_covariances(
                est, est.nsamples_per_model)
            CF, cf = CF.numpy(), cf.numpy()
            # print(np.linalg.det(CF), 'determinant')
            # print(np.linalg.matrix_rank(CF), 'rank', CF.shape)
            # print(CF, "CF")
            # print(CF_mc, "MC CF")
            print(CF.shape, CF_mc.shape)
            print(est)
            assert np.allclose(CF_mc, CF, atol=atol, rtol=rtol)

            # print(cf, "cf")
            # print(cf_mc, "MC cf")
            # print(cf_mc-cf, "diff")
            # print(cf_mc.shape, cf.shape, idx)
            assert np.allclose(cf_mc, cf, atol=atol, rtol=rtol)

        var_mc = np.cov(estimator_vals.T, ddof=1)
        variance = est._get_variance(est.nsamples_per_model).numpy()
        # print(est.nsamples_per_model)
        # print(var_mc, 'v_mc')
        # print(variance, 'v')
        # print((var_mc-variance)/variance)
        assert np.allclose(var_mc, variance, atol=atol, rtol=rtol)

    def test_estimator_variances(self):
        test_cases = [
            [[0, 1, 2], [0, 1, 2], [0, 0], "gmf", "mean"],
            [[0, 1, 2], [0, 1, 2], [0, 1], "gmf", "mean", 2],
            [[0, 1, 2], [0, 1, 2], [0, 1], "gmf", "mean", None, 3],
            [[0, 1], [0, 2], [0], "gmf", "mean"],
            [[0, 1], [0], [0], "gmf", "variance"],
            [[0, 1], [0, 2], [0], "gmf", "variance"],
            [[0, 1, 2], [0], [0, 0], "gmf", "variance"],
            [[0, 1, 2], [0, 1, 2], [0, 0], "gmf", "variance"],
            [[0, 1], [0], [0], "gmf", "mean_variance"],
            # following is slow test remove for speed
            # [[0, 1, 2], [0, 1, 2], [0, 0], "gmf", "mean_variance", None,
            #   None, int(1e5)],
            [[0, 1, 2], [0], None, "mfmc", "mean"],
            [[0, 1, 2], [0], None, "mlmc", "mean"],
            [[0, 1, 2], [0, 1], None, "mlmc", "mean"],
            [[0, 1, 2], [0], None, "mlmc", "variance"],
            [[0, 1, 2], [0], None, "mlmc", "mean_variance"],
            [[0, 1, 2], [0, 1, 2], None, "gmf", "variance", 2],
            [[0], [0, 1, 2], None, "mc", "variance"],
        ]
        for test_case in test_cases[1:2]:
            np.random.seed(1)
            # print(test_case)
            self._check_estimator_variances(*test_case)

    def test_sample_optimization(self):
        # check for scalar output case we require MFMC analytical solution
        model_idx, qoi_idx = [0, 1, 2], [0]
        recursion_index = [0, 1]
        target_cost = 10
        funs, cov, costs, model = _setup_multioutput_model_subproblem(
            model_idx, qoi_idx)
        est = get_estimator("gmf", "mean", model.variable, costs, cov,
                            recursion_index=np.asarray(recursion_index))
        # get nsample ratios before rounding
        # avoid using est._allocate_samples so we do not start
        # from mfmc exact solution
        nsample_ratios, obj_val = est._allocate_samples_opt(
            est.cov, est.costs, target_cost, est.get_constraints(target_cost),
            initial_guess=est.initial_guess)
        mfmc_nsample_ratios, mfmc_log10_variance = allocate_samples_mfmc(
            cov, costs, target_cost)
        assert np.allclose(nsample_ratios, mfmc_nsample_ratios)
        # print(np.exp(obj_val), 10**mfmc_log10_variance)
        assert np.allclose(np.exp(obj_val), 10**mfmc_log10_variance)

        est = get_estimator("gmf", "mean", model.variable, costs, cov,
                            tree_depth=2)
        est.allocate_samples(target_cost, verbosity=1)
        assert np.allclose(est.recursion_index, [0, 0])

    def test_best_model_subset_estimator(self):
        funs, cov, costs, model = _setup_multioutput_model_subproblem(
            [0, 1, 2], [0, 1, 2])
        est = get_estimator("gmf", "mean", model.variable, costs, cov,
                            max_nmodels=3)
        target_cost = 10
        est.allocate_samples(target_cost, verbosity=1, nprocs=1)

        ntrials, max_eval_concurrency = int(1e3), 1
        estimator_vals, Q, delta = self._estimate_components_loop(
            ntrials, est, funs, max_eval_concurrency)

        var_mc = np.cov(estimator_vals.T, ddof=1)
        variance = est.get_variance(target_cost, est.nsample_ratios).numpy()
        rtol, atol = 2e-2, 1e-3
        assert np.allclose(var_mc, variance, atol=atol, rtol=rtol)

        ntrials, max_eval_concurrency = int(1e4), 4
        qoi_idx = [0, 1]
        target_cost = 50
        funs, cov, costs, model = _setup_multioutput_model_subproblem(
            [0, 1, 2], qoi_idx)
        W = model.covariance_of_centered_values_kronker_product()
        W = _nqoisq_nqoisq_subproblem(
            W, model.nmodels, model.nqoi, [0, 1, 2], qoi_idx)
        B = model.covariance_of_mean_and_variance_estimators()
        B = _nqoi_nqoisq_subproblem(
            B, model.nmodels, model.nqoi, [0, 1, 2], qoi_idx)
        est = get_estimator(
            "gmf", "mean_variance", model.variable, costs, cov, W, B,
            opt_criteria=log_trace_variance)
        est.allocate_samples(target_cost)
        estimator_vals, Q, delta = self._estimate_components_loop(
            ntrials, est, funs, max_eval_concurrency)
        var_mc = np.cov(estimator_vals.T, ddof=1)
        variance = est.get_variance(
            est.rounded_target_cost, est.nsample_ratios).numpy()
        rtol, atol = 2e-2, 1e-4
        # print(est.nsample_ratios)
        # print(var_mc)
        # print(variance)
        # print((var_mc-variance)/variance)
        assert np.allclose(var_mc, variance, atol=atol, rtol=rtol)

    def test_compare_estimator_variances(self):
        funs, cov, costs, model = _setup_multioutput_model_subproblem(
            [0, 1, 2], [0, 1, 2])

        qoi_idx = [1]

        estimator_types = [
            "mc", "gmf", "gmf", "gmf", "gmf", "gmf",
            "gmf", "gmf", "gmf"]
        from pyapprox.util.configure_plots import mathrm_labels, mathrm_label
        est_labels = mathrm_labels(
            ["MC-MV",
             "ACVMF-M",  # Single QoI, optimize mean
             "ACVMF-V",  # Single QoI, optimize var
             "ACVMF-MV",  # Single QoI, optimize var
             "ACVMF-MOMV-GM",  # Multiple QoI, optimize single QoI mean
             "ACVMF-MOMV-GV",  # Multiple QoI, optimize single QoI var
             "ACVMF-MOM",  # Multiple QoI, optimize all means
             "ACVMF-MOV",  # Multiple QoI, optimize all vars
             "ACVMF-MOMV",])  # Multiple QoI, optimize all mean and vars
        kwargs_list = [
            {},
            {},
            {},
            {},
            {"opt_criteria": partial(
                _log_single_qoi_criteria, qoi_idx, "mean_var", "mean")},
            {"opt_criteria": partial(
                _log_single_qoi_criteria, qoi_idx, "mean_var", "var")},
            {},
            {},
            {}]
        # estimators that can compute mean
        mean_indices = [0, 1, 3, 4, 5, 6, 8]
        # estimators that can compute variance
        var_indices = [0, 2, 3, 4, 5, 7, 8]

        from pyapprox.multifidelity.multioutput_monte_carlo import (
            _nqoi_nqoi_subproblem)
        cov_sub = _nqoi_nqoi_subproblem(
           cov, model.nmodels, model.nqoi, [0, 1, 2], qoi_idx)

        W = model.covariance_of_centered_values_kronker_product()
        W_sub = _nqoisq_nqoisq_subproblem(
           W, model.nmodels, model.nqoi, [0, 1, 2], qoi_idx)
        B = model.covariance_of_mean_and_variance_estimators()
        B_sub = _nqoi_nqoisq_subproblem(
            B, model.nmodels, model.nqoi, [0, 1, 2], qoi_idx)
        stat_types = ["mean_variance", "mean", "variance", "mean_variance",
                      "mean_variance", "mean_variance", "mean",
                      "variance", "mean_variance"]
        covs = [cov, cov_sub, cov_sub, cov_sub, cov, cov, cov, cov, cov]
        args_list = [
            [W, B],
            [],
            [W_sub],
            [W_sub, B_sub],
            [W, B],
            [W, B],
            [],
            [W],
            [W, B]
        ]

        if len(qoi_idx) == 1:
            funs_sub = [partial(_single_qoi, qoi_idx[0], f) for f in funs]
        elif len(qoi_idx) == 2:
            funs_sub = [partial(_two_qoi, *qoi_idx, f) for f in funs]
        funs_list = [funs, funs_sub, funs_sub, funs_sub, funs, funs,
                     funs, funs, funs]

        for ii in range(1, len(kwargs_list)):
            kwargs_list[ii]["recursion_index"] = np.asarray([0, 0])

        estimators = [
            get_estimator(et, st, model.variable, costs, cv, *args, **kwargs)
            for et, st, cv, args, kwargs in zip(
                    estimator_types, stat_types, covs, args_list, kwargs_list)]


        # target_costs = np.array([1e1, 1e2, 1e3, 1e4, 1e5], dtype=int)[1:-1]
        target_costs = np.array([1e2], dtype=int)
        from pyapprox import multifidelity
        optimized_estimators = multifidelity.compare_estimator_variances(
            target_costs, estimators)

        from pyapprox.multifidelity.multioutput_monte_carlo import (
            MultiOutputMean, MultiOutputVariance, MultiOutputMeanAndVariance)

        def criteria(stat_type, variance, est):
            if stat_type == "variance" and isinstance(
                    est.stat, MultiOutputMeanAndVariance) and est.nqoi > 1:
                val = variance[est.stat.nqoi+qoi_idx[0],
                               est.stat.nqoi+qoi_idx[0]]
            elif stat_type == "variance" and isinstance(
                    est.stat, MultiOutputMeanAndVariance) and est.nqoi == 1:
                val = variance[est.stat.nqoi+0, est.stat.nqoi+0]
            elif (isinstance(
                    est.stat, (MultiOutputVariance, MultiOutputMean)) or
                  stat_type == "mean") and est.nqoi > 1:
                val = variance[qoi_idx[0], qoi_idx[0]]
            elif (isinstance(
                    est.stat, (MultiOutputVariance, MultiOutputMean)) or
                  stat_type == "mean") and est.nqoi == 1:
                val = variance[0, 0]
            else:
                print(est, est.stat, stat_type)
                raise ValueError
            return val

        # rtol, atol = 4.6e-2, 1e-3
        # ntrials, max_eval_concurrency = int(5e3), 4
        # for est, funcs in zip(optimized_estimators[1:], funs_list[1:]):
        #     est = est[0]
        #     estimator_vals, Q, delta = self._estimate_components_loop(
        #         ntrials, est, funcs, max_eval_concurrency)
        #     hf_var_mc = np.cov(Q.T, ddof=1)
        #     hf_var = est.stat.high_fidelity_estimator_covariance(
        #         est.nsamples_per_model)
        #     # print(hf_var_mc, hf_var)
        #     assert np.allclose(hf_var_mc, hf_var, atol=atol, rtol=rtol)

        #     CF_mc = torch.as_tensor(
        #         np.cov(delta.T, ddof=1), dtype=torch.double)
        #     CF = est.stat.get_discrepancy_covariances(
        #         est, est.nsamples_per_model)[0].numpy()
        #     assert np.allclose(CF_mc, CF, atol=atol, rtol=rtol)

        #     var_mc = np.cov(estimator_vals.T, ddof=1)
        #     variance = est._get_variance(est.nsamples_per_model).numpy()
        #     assert np.allclose(var_mc, variance, atol=atol, rtol=rtol)

        import matplotlib.pyplot as plt
        from pyapprox.multifidelity.multioutput_monte_carlo import (
            plot_estimator_variances, plot_estimator_variance_reductions)
        fig, axs = plt.subplots(1, 2, figsize=(2*8, 6), sharey=True)
        mean_optimized_estimators = [
            optimized_estimators[ii] for ii in mean_indices]
        mean_est_labels = [
            est_labels[ii] for ii in mean_indices]

        plot_estimator_variance_reductions(
            mean_optimized_estimators, mean_est_labels, axs[0], ylabel=None,
            criteria=partial(criteria, "mean"))

        var_optimized_estimators = [
            optimized_estimators[ii] for ii in var_indices]
        var_est_labels = [
            est_labels[ii] for ii in var_indices]
        plot_estimator_variance_reductions(
                var_optimized_estimators, var_est_labels, axs[1], ylabel=None,
                criteria=partial(criteria, "variance"))
        axs[0].set_xticks(axs[0].get_xticks())
        axs[0].set_xticklabels(
            axs[0].get_xticklabels(), rotation=30, ha='right')
        axs[1].set_xticks(axs[1].get_xticks())
        axs[1].set_xticklabels(
            axs[1].get_xticklabels(), rotation=30, ha='right')

        estimator_types[1:] = ["gmf" for ii in range(len(estimator_types)-1)]
        for ii in range(1, len(kwargs_list)):
            kwargs_list[ii]["recursion_index"] = np.asarray([0, 1])
        estimators = [
            get_estimator(et, st, model.variable, costs, cv, *args, **kwargs)
            for et, st, cv, args, kwargs in zip(
                    estimator_types, stat_types, covs, args_list, kwargs_list)]
        optimized_estimators = multifidelity.compare_estimator_variances(
            target_costs, estimators)

        mean_optimized_estimators = [
            optimized_estimators[ii] for ii in mean_indices]
        plot_estimator_variance_reductions(
            mean_optimized_estimators, mean_est_labels, axs[0], ylabel=None,
            criteria=partial(criteria, "mean"), alpha=0.5)
        var_optimized_estimators = [
            optimized_estimators[ii] for ii in var_indices]
        plot_estimator_variance_reductions(
                var_optimized_estimators, var_est_labels, axs[1], ylabel=None,
                criteria=partial(criteria, "variance"), alpha=0.5)

        # fig, axs = plt.subplots(1, 1, figsize=(1*8, 6))
        # plot_estimator_variances(
        #     optimized_estimators, est_labels, axs,
        #     ylabel=mathrm_label("Relative Estimator Variance"),
        #     relative_id=0, criteria=partial(criteria, "mean"))
        # plt.show()

    def test_insert_pilot_samples(self):
        funs, cov, costs, model = _setup_multioutput_model_subproblem(
            [0, 1, 2], [0, 1, 2])

        # modify costs so more hf samples are used but all three models
        # are selected
        costs[1:] = 0.1, 0.05
        est = get_estimator("gmf", "mean", model.variable, costs, cov,
                            max_nmodels=3)
        target_cost = 100
        est.allocate_samples(target_cost, verbosity=0, nprocs=1)

        random_state = np.random.RandomState(1)
        est.set_random_state(random_state)
        acv_samples, acv_values = est.generate_data(funs)
        est_val = est(acv_values)

        # This test is specific to ACVMF sampling strategy
        npilot_samples = 5
        pilot_samples = acv_samples[0][1][:, :npilot_samples]
        pilot_values = [f(pilot_samples) for f in model.funs]
        assert np.allclose(pilot_values[0], acv_values[0][1][:npilot_samples])

        values_per_model = est.combine_acv_values(acv_values)
        values_per_model_wo_pilot = [
            vals[npilot_samples:] for vals in values_per_model]
        values_per_model = [
            np.vstack((pilot_values[ii], vals))
            for ii, vals in enumerate(values_per_model_wo_pilot)]
        acv_values = est.separate_model_values(values_per_model)
        est_stats = est(acv_values)
        assert np.allclose(est_stats, est_val)

        random_state = np.random.RandomState(1)
        est.set_random_state(random_state)
        acv_samples, acv_values = est.generate_data(
            funs, [pilot_samples, pilot_values])
        est_val_pilot = est(acv_values)
        assert np.allclose(est_val, est_val_pilot)
        for ii in range(1, 3):
            assert np.allclose(
                acv_samples[ii][0][:, :npilot_samples],
                acv_samples[0][1][:, :npilot_samples])
            assert np.allclose(
                acv_samples[ii][1][:, :npilot_samples],
                acv_samples[0][1][:, :npilot_samples])

        npilot_samples = 8
        pilot_samples = est.best_est.generate_samples(npilot_samples)
        self.assertRaises(ValueError, est.generate_data,
                          funs, [pilot_samples, pilot_values])

        # modify costs so more hf samples are used
        costs[1:] = 0.5, 0.05
        est = get_estimator("gmf", "mean", model.variable, costs, cov,
                            max_nmodels=3)
        target_cost = 100
        est.allocate_samples(target_cost, verbosity=0, nprocs=1)
        random_state = np.random.RandomState(1)
        est.set_random_state(random_state)
        acv_samples, acv_values = est.generate_data(funs)
        est_val = est(acv_values)

        npilot_samples = 5
        samples_per_model = est.combine_acv_samples(acv_samples)
        samples_per_model_wo_pilot = [
            s[:, npilot_samples:] for s in samples_per_model]
        values_per_model = est.combine_acv_values(acv_values)
        values_per_model_wo_pilot = [
            vals[npilot_samples:] for vals in values_per_model]

        pilot_samples = acv_samples[0][1][:, :npilot_samples]
        pilot_values = [f(pilot_samples) for f in model.funs]
        random_state = np.random.RandomState(1)
        est.set_random_state(random_state)
        acv_samples1, acv_values1 = est.generate_data(
            funs, [pilot_samples, pilot_values])
        est_val_pilot = est(acv_values1)
        assert np.allclose(est_val, est_val_pilot)

        pilot_data = pilot_samples, pilot_values
        acv_values2 = est.combine_pilot_data(
            samples_per_model_wo_pilot, values_per_model_wo_pilot, pilot_data)
        assert np.allclose(acv_values1[0][1], acv_values[0][1])
        for ii in range(1, len(acv_values2)):
            assert np.allclose(acv_values2[ii][0], acv_values[ii][0])
            assert np.allclose(acv_values2[ii][1], acv_values[ii][1])
        est_val_pilot = est(acv_values2)
        assert np.allclose(est_val, est_val_pilot)

    def test_bootstrap_estimator(self):
        funs, cov, costs, model = _setup_multioutput_model_subproblem(
            [0, 1, 2], [0, 1, 2])

        # modify costs so more hf samples are used but all three models
        # are selected
        costs[1:] = 0.1, 0.05
        est = get_estimator("gmf", "mean", model.variable, costs, cov,
                            max_nmodels=3)
        target_cost = 100
        est.allocate_samples(target_cost, verbosity=0, nprocs=1)

        samples_per_model = est._generate_estimator_samples(None)[0]
        print(samples_per_model)
        values_per_model = [
            f(samples) for f, samples in zip(funs, samples_per_model)]

        bootstrap_mean, bootstrap_variance = est.bootstrap(
            values_per_model, 100) # 10000


if __name__ == "__main__":
    momc_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestMOMC)
    unittest.TextTestRunner(verbosity=2).run(momc_test_suite)
