import unittest
import torch
import numpy as np
from scipy import stats
from functools import partial

from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.surrogates.orthopoly.quadrature import gauss_jacobi_pts_wts_1D
from pyapprox.multifidelity.multioutput_monte_carlo import (
    get_V_from_covariance, covariance_of_variance_estimator, get_W_from_pilot,
    get_B_from_pilot, MultiOutputACVMeanEstimator,
    MultiOutputACVVarianceEstimator, MultiOutputACVMeanAndVarianceEstimator)
from pyapprox.multifidelity.control_variate_monte_carlo import (
    allocate_samples_mfmc)


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
        return np.array([1., 0.1, 0.01])

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
    mc_est = est._sample_estimate
    acv_values = est.generate_data(funs)[1]
    est_val = est(acv_values)
    Q = mc_est(acv_values[0][1])
    delta = np.hstack([mc_est(acv_values[ii][0]) -
                       mc_est(acv_values[ii][1])
                       for ii in range(1, len(funs))])
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

    def _nqoisq_nqoisq_subproblem(self, V, nmodels, nqoi, model_idx, qoi_idx):
        nsub_models, nsub_qoi = len(model_idx), len(qoi_idx)
        V_new = np.empty(
            (nsub_models*nsub_qoi**2, nsub_models*nsub_qoi**2))
        cnt1 = 0
        for jj1 in model_idx:
            for kk1 in qoi_idx:
                for ll1 in qoi_idx:
                    cnt2 = 0
                    idx1 = jj1*nqoi**2 + kk1*nqoi + ll1
                    for jj2 in model_idx:
                        for kk2 in qoi_idx:
                            for ll2 in qoi_idx:
                                idx2 = jj2*nqoi**2 + kk2*nqoi + ll2
                                V_new[cnt1, cnt2] = V[idx1, idx2]
                                cnt2 += 1
                    cnt1 += 1
        return V_new

    def _nqoi_nqoisq_subproblem(self, B, nmodels, nqoi, model_idx, qoi_idx):
        nsub_models, nsub_qoi = len(model_idx), len(qoi_idx)
        B_new = np.empty(
            (nsub_models*nsub_qoi, nsub_models*nsub_qoi**2))
        cnt1 = 0
        for jj1 in model_idx:
            for kk1 in qoi_idx:
                cnt2 = 0
                idx1 = jj1*nqoi + kk1
                for jj2 in model_idx:
                    for kk2 in qoi_idx:
                        for ll2 in qoi_idx:
                            idx2 = jj2*nqoi**2 + kk2*nqoi + ll2
                            B_new[cnt1, cnt2] = B[idx1, idx2]
                            cnt2 += 1
                cnt1 += 1
        return B_new

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
        B_exact = self._nqoi_nqoisq_subproblem(
            B_exact, model.nmodels, model.nqoi, model_idx, qoi_idx)
        mc_mean_cov_var = np.cov(means.T, covariances.T, ddof=1)
        B_mc = mc_mean_cov_var[:nqoi*nmodels, nqoi*nmodels:]
        assert np.allclose(B_mc, B_exact/nsamples, atol=atol, rtol=rtol)

        # no need to extract subproblem for V_exact as cov has already
        # been downselected
        V_exact = get_V_from_covariance(cov, nmodels)
        W_exact = model.covariance_of_centered_values_kronker_product()
        W_exact = self._nqoisq_nqoisq_subproblem(
            W_exact, model.nmodels, model.nqoi, model_idx, qoi_idx)
        cov_var_exact = covariance_of_variance_estimator(
            W_exact, V_exact, nsamples)
        assert np.allclose(
           cov_var_exact, mc_mean_cov_var[nqoi*nmodels:, nqoi*nmodels:],
           atol=atol, rtol=rtol)

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
        W_exact = self._nqoisq_nqoisq_subproblem(
            W_exact, model.nmodels, model.nqoi, model_idx, qoi_idx)
        assert np.allclose(W, W_exact, atol=atol, rtol=rtol)
        B = get_B_from_pilot(pilot_values, nmodels)
        B_exact = model.covariance_of_mean_and_variance_estimators()
        B_exact = self._nqoi_nqoisq_subproblem(
            B_exact, model.nmodels, model.nqoi, model_idx, qoi_idx)
        assert np.allclose(B, B_exact, atol=atol, rtol=rtol)

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

    def _check_estimator_variances(self, model_idx, qoi_idx, recursion_index,
                                   est_type):
        rtol, atol = 4.6e-2, 1e-3
        funs, cov, costs, model = _setup_multioutput_model_subproblem(
            model_idx, qoi_idx)
        nmodels, nqoi = len(model_idx), len(qoi_idx)
        if est_type == "MOACVM":
            est = MultiOutputACVMeanEstimator(
                cov, costs, model.variable,
                recursion_index=np.asarray(recursion_index))
            idx = nqoi
        elif est_type == "MOACVV":
            W = model.covariance_of_centered_values_kronker_product()
            W = self._nqoisq_nqoisq_subproblem(
                W, model.nmodels, model.nqoi, model_idx, qoi_idx)
            # npilot_samples = int(1e6)
            # pilot_samples = model.variable.rvs(npilot_samples)
            # pilot_values = np.hstack([f(pilot_samples) for f in funs])
            # W = get_W_from_pilot(pilot_values, nmodels)
            est = MultiOutputACVVarianceEstimator(
                cov, costs, model.variable, W,
                recursion_index=np.asarray(recursion_index))
            idx = nqoi**2
        elif est_type == "MOACVMV":
            W = model.covariance_of_centered_values_kronker_product()
            W = self._nqoisq_nqoisq_subproblem(
                W, model.nmodels, model.nqoi, model_idx, qoi_idx)
            print(W.shape)
            B = model.covariance_of_mean_and_variance_estimators()
            B = self._nqoi_nqoisq_subproblem(
                B, model.nmodels, model.nqoi, model_idx, qoi_idx)
            est = MultiOutputACVMeanAndVarianceEstimator(
                cov, costs, model.variable, W, B,
                recursion_index=np.asarray(recursion_index))
            idx = nqoi+nqoi**2

        # set nsamples per model tso generate data can be called
        # without optimizing the sample allocaiton
        est.nsamples_per_model = torch.tensor([10, 20, 30][:len(funs)])

        ntrials = int(1e4)
        # Q = []
        # delta = []
        # estimator_vals = []
        # mc_est = est._sample_estimate
        # for ii in range(ntrials):
        #     est_val, Q_val, delta_val = _estimate_components(est, funs, ii)
        #     estimator_vals.append(est_val)
        #     Q.append(Q_val)
        #     delta.append(delta_val)
        # delta = np.array(delta)
        # Q = np.array(Q)
        # estimator_vals = np.array(estimator_vals)

        max_eval_concurrency = 4
        from multiprocessing import Pool
        # set flat funs to none so funs can be pickled
        pool = Pool(max_eval_concurrency)
        func = partial(_estimate_components, est, funs)
        result = pool.map(func, list(range(ntrials)))
        pool.close()
        estimator_vals = np.asarray([r[0] for r in result])
        Q = np.asarray([r[1] for r in result])
        delta = np.asarray([r[2] for r in result])

        CF_mc = torch.as_tensor(
            np.cov(delta.T, ddof=1), dtype=torch.double)
        cf_mc = torch.as_tensor(
            np.cov(Q.T, delta.T, ddof=1)[:idx, idx:], dtype=torch.double)

        np.set_printoptions(linewidth=1000)
        # print(estimator_vals.mean(axis=0).reshape(nqoi, nqoi))
        # print(model.covariance()[:nqoi:, :nqoi])

        hf_var_mc = np.cov(Q.T, ddof=1)
        hf_var = est.high_fidelity_estimator_covariance(est.nsamples_per_model)
        # print(hf_var_mc)
        # print(hf_var.numpy())
        assert np.allclose(hf_var_mc, hf_var, atol=atol, rtol=rtol)

        CF, cf = est._get_discrepancy_covariances(est.nsamples_per_model)
        CF, cf = CF.numpy(), cf.numpy()
        # print(np.linalg.det(CF), 'determinant')
        # print(np.linalg.matrix_rank(CF), 'rank', CF.shape)
        # print(CF, "CF")
        # print(CF_mc, "MC CF")
        assert np.allclose(CF_mc, CF, atol=atol, rtol=rtol)

        # print(cf, "cf")
        # print(cf_mc, "MC cf")
        # print(cf_mc.shape, cf.shape, idx)
        assert np.allclose(cf_mc, cf, atol=atol, rtol=rtol)

        var_mc = np.cov(estimator_vals.T, ddof=1)
        variance = est._get_variance(est.nsamples_per_model).numpy()
        # print(var_mc, 'v_mc')
        # print(variance, 'v')
        # print((var_mc-variance)/variance)
        assert np.allclose(var_mc, variance, atol=atol, rtol=rtol)

    def test_estimator_variances(self):
        test_cases = [
            [[0, 1, 2], [0, 1, 2], [0, 0], "MOACVM"],
            [[0, 1, 2], [0, 1, 2], [0, 1], "MOACVM"],
            [[0, 1], [0, 2], [0], "MOACVM"],
            [[0, 1], [0], [0], "MOACVV"],
            [[0, 1], [0, 2], [0], "MOACVV"],
            [[0, 1, 2], [0, 1, 2], [0, 0], "MOACVV"],
            [[0, 1], [0], [0], "MOACVMV"],
            [[0, 1, 2], [0], [0, 0], "MOACVMV"],
        ]
        for test_case in test_cases:
            np.random.seed(1)
            self._check_estimator_variances(*test_case)

    def test_sample_optimization(self):
        # check for scalar output case we require MFMC analytical solution
        model_idx, qoi_idx = [0, 1, 2], [0]
        recursion_index = [0, 1]
        target_cost = 10
        funs, cov, costs, model = _setup_multioutput_model_subproblem(
            model_idx, qoi_idx)
        est = MultiOutputACVMeanEstimator(
            cov, costs, model.variable,
            recursion_index=np.asarray(recursion_index))
        # get nsample ratios before rounding
        # avoid using est._allocate_samples_multistart so we do not start
        # from mfmc exact solution
        nsample_ratios, obj_val = est._allocate_samples_opt(
            est.cov, est.costs, target_cost, est.get_constraints(target_cost),
            initial_guess=est.initial_guess)
        mfmc_nsample_ratios, mfmc_log10_variance = allocate_samples_mfmc(
            cov, costs, target_cost)
        print(nsample_ratios)
        assert np.allclose(nsample_ratios, mfmc_nsample_ratios)
        assert np.allclose(obj_val, 10**mfmc_log10_variance)


if __name__ == "__main__":
    momc_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestMOMC)
    unittest.TextTestRunner(verbosity=2).run(momc_test_suite)
