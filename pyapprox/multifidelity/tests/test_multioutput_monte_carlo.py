import unittest
import torch
import numpy as np
from scipy import stats
from functools import partial

from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.surrogates.orthopoly.quadrature import gauss_jacobi_pts_wts_1D
from pyapprox.multifidelity.multioutput_monte_carlo import (
    get_V_from_covariance, covariance_of_variance_estimator, get_W_from_pilot,
    get_B_from_pilot, get_estimator, _nqoisq_nqoisq_subproblem,
    _nqoi_nqoisq_subproblem, ACVEstimator)
from pyapprox.multifidelity.control_variate_monte_carlo import (
    allocate_samples_mfmc)
from pyapprox.multifidelity.multioutput_monte_carlo import (
    log_determinant_variance, log_trace_variance)


class MultioutputModelEnsemble():
    """
    Benchmark for testing multifidelity algorithms that estimate statistics
    for vector valued models of varying fidelity.
    """
    def __init__(self):
        self.variable = IndependentMarginalsVariable([stats.uniform(0, 1)])
        self.funs = [self.f0, self.f1, self.f2]
        self.nmodels = len(self.funs)  # number of models
        self.nqoi = 3  # nqoi per model

        # self._sp_funs = [
        #     ["sqrt(11)*x**5", "x**4", "sin(2*pi*x)"],
        #     ["sqrt(7)*x**3", "sqrt(7)*x**2", "cos(2*pi*x+pi/2)"],
        #     ["sqrt(3)/2*x**2", "sqrt(3)/2*x", "cos(2*pi*x+pi/4)"]]
        self.flatten_funs()

    def _flat_fun_wrapper(self, ii, jj, xx):
        return self.funs[ii](xx[None, :])[:, jj]

    def flatten_funs(self):
        # If sp.lambdify is called then this class cannot be pickled
        # sp_x = sp.Symbol("x")
        # self._flat_funs = [
        #     np.vectorize(sp.lambdify((sp_x), sp.sympify(f), "numpy"))
        #     for model_funs in self._sp_funs for f in model_funs]
        self._flat_funs = []
        for ii in range(self.nmodels):
            for jj in range(self.nqoi):
                self._flat_funs.append(
                    partial(self._flat_fun_wrapper, ii, jj))

    def costs(self) -> np.ndarray:
        """
        The nominal costs of each model for a single sample

        Returns
        -------
        values : np.ndarray (nmodels)
            Model costs
        """
        return np.array([1., 0.01, 0.001])

    def f0(self, samples: np.ndarray) -> np.ndarray:
        """
        Highest fidelity model

        Parameters
        ----------
        samples : np.ndarray (nvars, nsamples)
            Samples realizations

        Returns
        -------
        values : np.ndarray (nsamples, qoi)
            Model evaluations at the samples
        """
        return np.hstack(
            [np.sqrt(11)*samples.T**5,
             samples.T**4,
             np.sin(2*np.pi*samples.T)])

    def f1(self, samples: np.ndarray) -> np.ndarray:
        """
        A low fidelity model

        Parameters
        ----------
        samples : np.ndarray (nvars, nsamples)
            Samples realizations

        Returns
        -------
        values : np.ndarray (nsamples, qoi)
            Model evaluations at the samples
        """
        return np.hstack(
            [np.sqrt(7)*samples.T**3,
             np.sqrt(7)*samples.T**2,
             np.cos(2*np.pi*samples.T+np.pi/2)])

    def f2(self, samples: np.ndarray) -> np.ndarray:
        """
        A low fidelity model

        Parameters
        ----------
        samples : np.ndarray (nvars, nsamples)
            Samples realizations

        Returns
        -------
        values : np.ndarray (nsamples, qoi)
            Model evaluations at the samples
        """
        return np.hstack(
            [np.sqrt(3)/2*samples.T**2,
             np.sqrt(3)/2*samples.T,
             np.cos(2*np.pi*samples.T+np.pi/4)])

    def _uniform_means(self):
        return np.array([
            [np.sqrt(11)/6, 1/5, 0.0],
            [np.sqrt(7)/4, np.sqrt(7)/3, 0.0],
            [1/(2*np.sqrt(3)), np.sqrt(3)/4, 0.0],
        ])

    def means(self) -> np.ndarray:
        """
        Return the means of the QoI of each model

        Returns
        -------
        means : np.ndarray(nmodels, nqoi)
            The means of each model
        """
        return self._uniform_means()

    def _uniform_covariance_matrices(self):
        # compute diagonal blocks
        c13 = -np.sqrt(11)*(15-10*np.pi**2+2*np.pi**4)/(4*np.pi**5)
        c23 = (3-np.pi**2)/(2*np.pi**3)
        cov11 = np.array([
            [25/36, np.sqrt(11)/15., c13],
            [np.sqrt(11)/15., 16/225, c23],
            [c13, c23, 1/2]
        ])
        c13 = np.sqrt(7)*(-3+2*np.pi**2)/(4*np.pi**3)
        c23 = np.sqrt(7)/(2*np.pi)
        cov22 = np.array([
            [9/16, 7/12, c13],
            [7/12, 28/45, c23],
            [c13, c23, 1/2]
        ])
        c13 = np.sqrt(3/2)*(1+np.pi)/(4*np.pi**2)
        c23 = np.sqrt(3/2)/(4*np.pi)
        cov33 = np.array([
            [1/15, 1/16, c13],
            [1/16, 1/16, c23],
            [c13, c23, 1/2]
        ])
        # compute off digonal block covariance between model 0 and mode 1
        c13 = np.sqrt(11)*(15-10*np.pi**2+2*np.pi**4)/(4*np.pi**5)
        c31 = np.sqrt(7)*(3-2*np.pi**2)/(4*np.pi**3)
        cov12 = np.array([
            [5*np.sqrt(77)/72, 5*np.sqrt(77)/72, c13],
            [3*np.sqrt(7)/40, 8/(15*np.sqrt(7)), (-3+np.pi**2)/(2*np.pi**3)],
            [c31, -np.sqrt(7)/(2*np.pi), -1/2]
        ])

        # compute off digonal block covariance between model 0 and mode 2
        c13 = np.sqrt(11/2)*(15+np.pi*(
            -15+np.pi*(-10+np.pi*(5+2*np.pi))))/(4*np.pi**5)
        c23 = (-3+(-1+np.pi)*np.pi*(3+np.pi))/(2*np.sqrt(2)*np.pi**4)
        cov13 = np.array([
            [5*np.sqrt(11/3)/48, 5*np.sqrt(11/3)/56, c13],
            [4/(35*np.sqrt(3)), 1/(10*np.sqrt(3)), c23],
            [-np.sqrt(3)/(4*np.pi), -np.sqrt(3)/(4*np.pi), -1/(2*np.sqrt(2))]
        ])

        # compute off digonal block covariance between model 1 and mode 2
        c13 = np.sqrt(7/2)*(-3+3*np.pi+2*np.pi**2)/(4*np.pi**3)
        c23 = np.sqrt(7/2)*(1+np.pi)/(2*np.pi**2)
        cov23 = np.array([
            [np.sqrt(7/3)/8, 3*np.sqrt(21)/80, c13],
            [2*np.sqrt(7/3)/15, np.sqrt(7/3)/8, c23],
            [np.sqrt(3)/(4*np.pi), np.sqrt(3)/(4*np.pi), 1/(2*np.sqrt(2))]
        ])
        return cov11, cov22, cov33, cov12, cov13, cov23

    def covariance(self) -> np.ndarray:
        """
        The covariance between the qoi of each model

        Returns
        -------
        cov = np.ndarray (nmodels*nqoi, nmodels*nqoi)
            The covariance treating functions concatinating the qoi
            of each model f0, f1, f2
        """
        cov11, cov22, cov33, cov12, cov13, cov23 = (
            self._uniform_covariance_matrices())
        return np.block([[cov11, cov12, cov13],
                         [cov12.T, cov22, cov23],
                         [cov13.T, cov23.T, cov33]])

    def __repr__(self):
        return "{0}(nmodels=3, variable_type='uniform')".format(
            self.__class__.__name__)

    def _covariance_quadrature(self):
        xx, ww = gauss_jacobi_pts_wts_1D(201, 0, 0)
        xx = (xx+1)/2
        means = [f(xx).dot(ww)for f in self._flat_funs]
        cov = np.empty((self.nmodels*self.nqoi, self.nmodels*self.nqoi))
        ii = 0
        for fi, mi in zip(self._flat_funs, means):
            jj = 0
            for fj, mj in zip(self._flat_funs, means):
                cov[ii, jj] = ((fi(xx)-mi)*(fj(xx)-mj)).dot(ww)
                jj += 1
            ii += 1
        return cov

    def _V_fun_entry(self, jj, kk, ll, means, flat_covs, xx):
        idx1 = jj*self.nqoi + kk
        idx2 = jj*self.nqoi + ll
        return ((self._flat_funs[idx1](xx)-means[idx1])*(
            self._flat_funs[idx2](xx)-means[idx2]) -
                flat_covs[jj][kk*self.nqoi+ll])

    def _V_fun(self, jj1, kk1, ll1, jj2, kk2, ll2, means, flat_covs, xx):
        return (
            self._V_fun_entry(jj1, kk1, ll1, means, flat_covs, xx) *
            self._V_fun_entry(jj2, kk2, ll2, means, flat_covs, xx))

    def _B_fun(self, ii, jj, kk, ll, means, flat_covs, xx):
        return (
            (self._flat_funs[ii](xx)-means[ii]) *
            self._V_fun_entry(jj, kk, ll, means, flat_covs, xx))

    def _flat_covs(self):
        cov = self.covariance()
        # store covariance only between the QoI of a model with QoI of the same
        # model
        flat_covs = []
        for ii in range(self.nmodels):
            flat_covs.append([])
            for jj in range(self.nqoi):
                for kk in range(self.nqoi):
                    flat_covs[ii].append(cov[ii*self.nqoi+jj][ii*self.nqoi+kk])
        return flat_covs

    def covariance_of_centered_values_kronker_product(self) -> np.ndarray:
        r"""
        The B matrix used to compute the covariance between the
        Kroneker product of centered (mean is subtracted off) values.

        Returns
        -------
        np.ndarray (nmodels*nqoi**2, nmodels*nqoi**2)
            The covariance :math:`Cov[(f_i-\mathbb{E}[f_i])^{\otimes^2}, (f_j-\mathbb{E}[f_j])^{\otimes^2}]`
        """
        means = self.means().flatten()
        flat_covs = self._flat_covs()

        xx, ww = gauss_jacobi_pts_wts_1D(201, 0, 0)
        xx = (xx+1)/2
        est_cov = np.empty(
            (self.nmodels*self.nqoi**2, self.nmodels*self.nqoi**2))
        cnt1 = 0
        for jj1 in range(self.nmodels):
            for kk1 in range(self.nqoi):
                for ll1 in range(self.nqoi):
                    cnt2 = 0
                    for jj2 in range(self.nmodels):
                        for kk2 in range(self.nqoi):
                            for ll2 in range(self.nqoi):
                                quad_cov = self._V_fun(
                                    jj1, kk1, ll1, jj2, kk2, ll2,
                                    means, flat_covs, xx).dot(ww)
                                est_cov[cnt1, cnt2] = quad_cov
                                cnt2 += 1
                    cnt1 += 1
        return np.array(est_cov)

    def covariance_of_mean_and_variance_estimators(self) -> np.ndarray:
        r"""
        The B matrix used to compute the covariance between mean and variance
        estimators.

        Returns
        -------
        np.ndarray (nmodels*nqoi, nmodels*nqoi**2)
            The covariance :math:`Cov[f_i, (f_j-\mathbb{E}[f_j])^{\otimes^2}]`
        """
        means = self.means().flatten()
        flat_covs = self._flat_covs()
        xx, ww = gauss_jacobi_pts_wts_1D(201, 0, 0)
        xx = (xx+1)/2
        est_cov = np.empty((self.nmodels*self.nqoi, self.nmodels*self.nqoi**2))
        for ii in range(len(self._flat_funs)):
            cnt = 0
            for jj in range(self.nmodels):
                for kk in range(self.nqoi):
                    for ll in range(self.nqoi):
                        quad_cov = self._B_fun(
                            ii, jj, kk, ll, means, flat_covs, xx).dot(ww)
                        est_cov[ii, cnt] = quad_cov
                        cnt += 1
        return np.array(est_cov)


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

    def _check_pilot_covariances(self, model_idx, qoi_idx):
        funs, cov, costs, model = _setup_multioutput_model_subproblem(
            model_idx, qoi_idx)
        nmodels = len(funs)
        # atol is needed for terms close to zero
        rtol, atol = 1e-2, 5.5e-4
        npilot_samples = int(1e6)
        pilot_samples = model.variable.rvs(npilot_samples)
        pilot_values = np.hstack([f(pilot_samples) for f in funs])
        W = get_W_from_pilot(pilot_values, nmodels)
        W_exact = model.covariance_of_centered_values_kronker_product()
        W_exact = _nqoisq_nqoisq_subproblem(
            W_exact, model.nmodels, model.nqoi, model_idx, qoi_idx)
        assert np.allclose(W, W_exact, atol=atol, rtol=rtol)
        B = get_B_from_pilot(pilot_values, nmodels)
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

    def _check_separate_samples(self, est_type):
        print(est_type)
        funs, cov, costs, model = _setup_multioutput_model_subproblem(
            [0, 1, 2], [0, 1, 2])
        costs = [3, 2, 1]
        est = get_estimator(
            est_type, "mean", model.variable, costs, cov)
        target_cost = 30
        est.allocate_samples(target_cost, verbosity=1)
        acv_samples, acv_values = est.generate_data(funs)
        samples_per_model = est.combine_acv_samples(acv_samples)
        values_per_model = est.combine_acv_values(acv_values)

        nmodels = len(acv_values)
        for ii in range(nmodels):
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
            ["acvmf"], ["mfmc"], ["mlmc"]
        ]
        for test_case in test_cases:
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
                                   est_type, stat_type, ntrials=int(1e4)):
        rtol, atol = 4.6e-2, 1.01e-3
        funs, cov, costs, model = _setup_multioutput_model_subproblem(
            model_idx, qoi_idx)
        nqoi = len(qoi_idx)
        args = []
        if est_type == "acvmf":
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
            est_type, stat_type, model.variable, costs, cov, *args, **kwargs)

        # set nsamples per model tso generate data can be called
        # without optimizing the sample allocaiton
        est.nsamples_per_model = torch.tensor([10, 20, 30][:len(funs)])
        # est.nsamples_per_model = torch.tensor([7, 8, 140][:len(funs)])

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
        # print(((hf_var_mc-hf_var.numpy())/hf_var.numpy()).max())
        assert np.allclose(hf_var_mc, hf_var, atol=atol, rtol=rtol)

        if est_type != "mc":
            CF, cf = est.stat.get_discrepancy_covariances(
                est, est.nsamples_per_model)
            CF, cf = CF.numpy(), cf.numpy()
            # print(np.linalg.det(CF), 'determinant')
            # print(np.linalg.matrix_rank(CF), 'rank', CF.shape)
            # print(CF, "CF")
            # print(CF_mc, "MC CF")
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
            [[0, 1, 2], [0, 1, 2], [0, 0], "acvmf", "mean"],
            [[0, 1, 2], [0, 1, 2], [0, 1], "acvmf", "mean"],
            [[0, 1], [0, 2], [0], "acvmf", "mean"],
            [[0, 1], [0], [0], "acvmf", "variance"],
            [[0, 1], [0, 2], [0], "acvmf", "variance"],
            [[0, 1, 2], [0], [0, 0], "acvmf", "variance"],
            [[0, 1, 2], [0, 1, 2], [0, 0], "acvmf", "variance"],
            [[0, 1], [0], [0], "acvmf", "mean_variance"],
            # following is slow test remove for speed
            # [[0, 1, 2], [0, 1, 2], [0, 0], "acvmf", "mean_variance", int(1e5)],
            [[0, 1, 2], [0], None, "mfmc", "mean"],
            [[0, 1, 2], [0], None, "mlmc", "mean"],
            [[0, 1, 2], [0, 1], None, "mlmc", "mean"],
            [[0, 1, 2], [0], None, "mlmc", "variance"],
            [[0, 1, 2], [0], None, "mlmc", "mean_variance"],
            [[0, 1, 2], [0, 1, 2], None, "acvmfb", "variance"],
            [[0], [0, 1, 2], None, "mc", "variance"],
        ]
        for test_case in test_cases:
            np.random.seed(1)
            print(test_case)
            self._check_estimator_variances(*test_case)

    def test_sample_optimization(self):
        # check for scalar output case we require MFMC analytical solution
        model_idx, qoi_idx = [0, 1, 2], [0]
        recursion_index = [0, 1]
        target_cost = 10
        funs, cov, costs, model = _setup_multioutput_model_subproblem(
            model_idx, qoi_idx)
        est = get_estimator("acvmf", "mean", model.variable, costs, cov,
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

        est = get_estimator("acvmfb", "mean", model.variable, costs, cov)
        est.allocate_samples(target_cost, verbosity=1)
        assert np.allclose(est.recursion_index, [0, 0])

    def test_best_model_subset_estimator(self):
        funs, cov, costs, model = _setup_multioutput_model_subproblem(
            [0, 1, 2], [0, 1, 2])
        est = get_estimator("acvmf", "mean", model.variable, costs, cov,
                            max_nmodels=3)
        target_cost = 10
        est.allocate_samples(target_cost, verbosity=1, nprocs=1)
        print(est, est.nmodels, est.best_est.nmodels, est.best_est.costs)

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
            "acvmf", "mean_variance", model.variable, costs, cov, W, B,
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

        # estimator_types = ["mc", "acvmf", "acvmf", "acvmf", "acvmf", "acvmf",
        #                   "acvmf", "acvmf", "acvmf"]
        estimator_types = [
            "mc", "acvmfb", "acvmfb", "acvmfb", "acvmfb", "acvmfb",
            "acvmfb", "acvmfb", "acvmfb"]
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

        estimator_types[1:] = ["acvmf" for ii in range(len(estimator_types)-1)]
        for ii in range(1, len(kwargs_list)):
            kwargs_list[ii]["recursion_index"] = np.asarray([0, 0])

        # estimator_types = [estimator_types[0], estimator_types[4]]
        # est_labels = [est_labels[0], est_labels[4]]
        # stat_types = [stat_types[0], stat_types[4]]
        # args_list = [args_list[0], args_l[4]]
        # kwargs_list = [kwargs_list[0], kwargs_list[4]]
        # covs = [covs[0], covs[4]]
        # mean_indices = [0, 1]
        # var_indices = [0, 1]

        estimators = [
            get_estimator(et, st, model.variable, costs, cv, *args, **kwargs)
            for et, st, cv, args, kwargs in zip(
                    estimator_types, stat_types, covs, args_list, kwargs_list)]
        # for et, label in zip(estimators, est_labels):
        #    print(et, label, et.optimization_criteria)


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
        axs[0].set_xticklabels(
            axs[0].get_xticklabels(), rotation=30, ha='right')
        axs[1].set_xticklabels(
            axs[1].get_xticklabels(), rotation=30, ha='right')

        estimator_types[1:] = ["acvmf" for ii in range(len(estimator_types)-1)]
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
        est = get_estimator("acvmf", "mean", model.variable, costs, cov,
                            max_nmodels=3)
        target_cost = 100
        est.allocate_samples(target_cost, verbosity=0, nprocs=1)

        random_state = np.random.RandomState(1)
        est.set_random_state(random_state)
        acv_samples, acv_values = est.generate_data(funs)
        est_val = est(acv_values)
        print(est.nsamples_per_model)

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
        est = get_estimator("acvmf", "mean", model.variable, costs, cov,
                            max_nmodels=3)
        target_cost = 100
        est.allocate_samples(target_cost, verbosity=0, nprocs=1)
        random_state = np.random.RandomState(1)
        est.set_random_state(random_state)
        acv_samples, acv_values = est.generate_data(funs)
        est_val = est(acv_values)
        print(est)

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
        est = get_estimator("acvmf", "mean", model.variable, costs, cov,
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
    
