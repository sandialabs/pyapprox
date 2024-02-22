from abc import ABC, abstractmethod
from functools import partial

import numpy as np
from scipy import stats

from pyapprox.surrogates.integrate import integrate
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.surrogates.orthopoly.quadrature import gauss_jacobi_pts_wts_1D
from pyapprox.interface.wrappers import ModelEnsemble


class ACVBenchmark(ABC):
    def __init__(self, *args, **kwargs):
        self.variable = self._set_variable(*args, **kwargs)
        self.nvars = self.variable.num_vars()
        self.funs, self.nqoi = self._set_funs(*args, **kwargs)
        self.nmodels = len(self.funs)
        self._quadrature_rule = self._set_quadrature_rule(*args, **kwargs)
        self._flat_funs = self._flatten_funs()
        self._model_ensemble = ModelEnsemble(self.funs)

    @abstractmethod
    def _set_funs(self):
        raise NotImplementedError

    def _set_quadrature_rule(self, *args, **kwargs):
        quad_x, quad_w = integrate(
            "tensorproduct", self.variable,
            levels=[10]*self.variable.num_vars())
        return quad_x, quad_w

    def _flat_fun_wrapper(self, ii, jj, xx):
        assert xx.ndim == 2 and xx.shape[0] == self.nvars
        return self.funs[ii](xx)[:, jj]

    def _flatten_funs(self):
        flat_funs = []
        for ii in range(self.nmodels):
            for jj in range(self.nqoi):
                flat_funs.append(
                    partial(self._flat_fun_wrapper, ii, jj))
        return flat_funs

    def _get_means(self) -> np.ndarray:
        # overide this function if you know the means exactly
        quad_xx, quad_ww = self._quadrature_rule
        means = [f(quad_xx).dot(quad_ww) for f in self._flat_funs]
        return np.array(means).reshape(self.nmodels, self.nqoi)

    def get_means(self) -> np.ndarray:
        """
        Return the means of the QoI of each model

        Returns
        -------
        means : np.ndarray(nmodels, nqoi)
            The means of each model
        """
        means = self._get_means()
        if (means.ndim != 2 or means.shape[0] != self.nmodels or
                means.shape[1] != self.nqoi):
            raise RuntimeError("_get_means() not implemented correctly")
        return means

    def _get_covariance_matrix(self) -> np.ndarray:
        # overide this function if you know the means exactly
        quad_xx, quad_ww = self._quadrature_rule
        means = [f(quad_xx).dot(quad_ww) for f in self._flat_funs]

        cov = np.empty((self.nmodels*self.nqoi, self.nmodels*self.nqoi))
        ii = 0
        for fi, mi in zip(self._flat_funs, means):
            jj = 0
            for fj, mj in zip(self._flat_funs, means):
                cov[ii, jj] = (
                    (fi(quad_xx)-mi)*(fj(quad_xx)-mj)).dot(quad_ww).item()
                jj += 1
            ii += 1
        return cov

    def get_covariance_matrix(self) -> np.ndarray:
        """
        The covariance between the qoi of each model

        Returns
        -------
        cov = np.ndarray (nmodels*nqoi, nmodels*nqoi)
            The covariance treating functions concatinating the qoi
            of each model f0, f1, f2
        """
        cov = self._get_covariance_matrix()
        if (cov.ndim != 2 or cov.shape[0] != self.nmodels*self.nqoi or
                cov.shape[1] != self.nmodels*self.nqoi):
            raise RuntimeError(
                "_get_covariance_matrix() not implemented correctly")
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
        cov = self.get_covariance_matrix()
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
        The W matrix used to compute the covariance between the
        Kroneker product of centered (mean is subtracted off) values.

        Returns
        -------
        np.ndarray (nmodels*nqoi**2, nmodels*nqoi**2)
            The covariance :math:`Cov[(f_i-\mathbb{E}[f_i])^{\otimes^2}, (f_j-\mathbb{E}[f_j])^{\otimes^2}]`
        """
        means = self.get_means().flatten()
        flat_covs = self._flat_covs()

        xx, ww = self._quadrature_rule
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
                                est_cov[cnt1, cnt2] = quad_cov.item()
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
        means = self.get_means().flatten()
        flat_covs = self._flat_covs()
        xx, ww = self._quadrature_rule
        est_cov = np.empty((self.nmodels*self.nqoi, self.nmodels*self.nqoi**2))
        for ii in range(len(self._flat_funs)):
            cnt = 0
            for jj in range(self.nmodels):
                for kk in range(self.nqoi):
                    for ll in range(self.nqoi):
                        quad_cov = self._B_fun(
                            ii, jj, kk, ll, means, flat_covs, xx).dot(ww)
                        est_cov[ii, cnt] = quad_cov.item()
                        cnt += 1
        return np.array(est_cov)

    def __repr__(self):
        return "{0}(nmodels={1}, nqoi={2})".format(
            self.__class__.__name__, self.nmodels, self.nqoi)

    def __call__(self, samples):
        # samples must include model id in last row
        return self._model_ensemble(samples)


class PolynomialModelEnsemble(ACVBenchmark):
    def m0(self, samples):
        return samples.T**5

    def m1(self, samples):
        return samples.T**4

    def m2(self, samples):
        return samples.T**3

    def m3(self, samples):
        return samples.T**2

    def m4(self, samples):
        return samples.T**1

    def _set_funs(self, nmodels):
        return [self.m0, self.m1, self.m2, self.m3, self.m4][:nmodels], 1

    def _get_means(self):
        return 1/np.arange(6, 1, -1)[:self.nmodels].reshape(
            self.nmodels, self.nqoi)

    def _set_variable(self, nmodels):
        univariate_variables = [stats.uniform(0, 1)]
        return IndependentMarginalsVariable(
            univariate_variables)


class TunableModelEnsemble(ACVBenchmark):

    def __init__(self, theta1, shifts=None):
        """
        Parameters
        ----------
        theta0 : float
            Angle controling
        Notes
        -----
        The choice of A0, A1, A2 here results in unit variance for each model
        """
        super().__init__(theta1, shifts=shifts)

    def m0(self, samples):
        assert samples.shape[0] == 2
        x, y = samples[0, :], samples[1, :]
        return (self.A0*(np.cos(self.theta0) * x**5 + np.sin(self.theta0) *
                         y**5))[:, np.newaxis]

    def m1(self, samples):
        assert samples.shape[0] == 2
        x, y = samples[0, :], samples[1, :]
        return (self.A1*(np.cos(self.theta1) * x**3 + np.sin(self.theta1) *
                         y**3)+self.shifts[0])[:, np.newaxis]

    def m2(self, samples):
        assert samples.shape[0] == 2
        x, y = samples[0, :], samples[1, :]
        return (self.A2*(np.cos(self.theta2) * x + np.sin(self.theta2) *
                         y)+self.shifts[1])[:, np.newaxis]

    def _set_funs(self, theta1, shifts=None):
        self.A0 = np.sqrt(11)
        self.A1 = np.sqrt(7)
        self.A2 = np.sqrt(3)
        self.nmodels = 3
        self.theta0 = np.pi/2
        self.theta1 = theta1
        self.theta2 = np.pi/6
        assert self.theta0 > self.theta1 and self.theta1 > self.theta2
        self.shifts = shifts
        if self.shifts is None:
            self.shifts = [0, 0]
        return [self.m0, self.m1, self.m2], 1

    def _set_variable(self, theta1, shifts):
        univariate_variables = [stats.uniform(-1, 2), stats.uniform(-1, 2)]
        return IndependentMarginalsVariable(univariate_variables)

    def _get_covariance_matrix(self):
        cov = np.eye(self.nmodels)
        cov[0, 1] = self.A0*self.A1/9*(np.sin(self.theta0)*np.sin(
            self.theta1)+np.cos(self.theta0)*np.cos(self.theta1))
        cov[1, 0] = cov[0, 1]
        cov[0, 2] = self.A0*self.A2/7*(np.sin(self.theta0)*np.sin(
            self.theta2)+np.cos(self.theta0)*np.cos(self.theta2))
        cov[2, 0] = cov[0, 2]
        cov[1, 2] = self.A1*self.A2/5*(
            np.sin(self.theta1)*np.sin(self.theta2)+np.cos(
                self.theta1)*np.cos(self.theta2))
        cov[2, 1] = cov[1, 2]
        return cov

    def _get_means(self):
        return np.array(
            [0, self.shifts[0], self.shifts[1]])[:, None]

    def get_kurtosis(self):
        return np.array([
            (self.A0**4*(213+29*np.cos(4*self.theta0)))/5082,
            (self.A1**4*(93+5*np.cos(4*self.theta1)))/1274,
            -(1/30)*self.A2**4*(-7+np.cos(4*self.theta2))])

    def _covariance_of_variances(
            self, nsamples, E_f0_sq_f1_sq, E_f0_sq, E_f1_sq, E_f1, E_f0_sq_f1,
            E_f0, E_f0_f1_sq, E_f0_f1):
        E_u20_u21 = (E_f0_sq_f1_sq-E_f0_sq*E_f1_sq -
                     2*E_f1*E_f0_sq_f1 + 2*E_f1**2*E_f0_sq -
                     2*E_f0*E_f0_f1_sq + 2*E_f0**2*E_f1_sq +
                     4*E_f0*E_f1*E_f0_f1-4*E_f0**2*E_f1**2)
        return E_u20_u21/nsamples + 1/(nsamples*(nsamples-1))*(
            E_f0_f1**2 - 2*E_f0_f1*E_f0*E_f1 + (E_f0*E_f1)**2)

    def get_covariance_of_variances(self, nsamples):
        t0, t1, t2 = self.theta0, self.theta1, self.theta2
        s1, s2 = self.shifts
        E_f0_sq_f1 = self.A0**2*s1/11
        E_f0_f1_sq = 2*self.A0*self.A1*s1*np.cos(t1-t0)/9
        E_f0_sq_f1_sq = (
            self.A0**2*(7614*self.A1**2+19278*s1**2 +
                        3739*self.A1**2*np.cos(2*(t1 - t0)) +
                        1121*self.A1**2*np.cos(2*(t1 + t0))))/212058

        E_f0_sq_f2 = self.A0**2*s2/11
        E_f0_f2_sq = 2*self.A0*self.A2*s2*np.cos(t2-t0)/7
        E_f0_sq_f2_sq = (
            self.A0**2*(98*(23*self.A2**2+39*s2**2) +
                        919*self.A2**2*np.cos(2*(t2-t0)) +
                        61*self.A2**2*np.cos(2*(t2+t0))))/42042

        E_f1_sq_f2 = (
            (self.A1**2*s2)/7 + s1**2*s2 +
            2/5*self.A1*self.A2*s1*np.cos(t1-t2))
        E_f1_f2_sq = (
            (self.A2**2*s1)/3 + s1*s2**2 +
            2/5*self.A1*self.A2*s2*np.cos(t1-t2))
        E_f1_sq_f2_sq = (
            (5*self.A1**2*self.A2**2)/63 + (self.A2**2*s1**2)/3 +
            (self.A1**2*s2**2)/7 + s1**2*s2**2 +
            4/5*self.A1*self.A2*s1*s2*np.cos(t1-t2) +
            (113*(self.A1**2*self.A2**2*np.cos(2*(t1-t2)))/3150 -
             (13*self.A1**2*self.A2**2*np.cos(2*(t1 + t2)))/3150))

        E_f0, E_f1, E_f2 = self.get_means()
        cov = self.get_covariance_matrix()
        E_f0_sq = cov[0, 0]+E_f0**2
        E_f1_sq = cov[0, 0]+E_f1**2
        E_f2_sq = cov[0, 0]+E_f2**2

        E_f0_f1 = cov[0, 1]+E_f0*E_f1
        E_f0_f2 = cov[0, 2]+E_f0*E_f2
        E_f1_f2 = cov[1, 2]+E_f1*E_f2

        Cmat = np.zeros((3, 3))
        Cmat[0, 1] = self._covariance_of_variances(
            nsamples, E_f0_sq_f1_sq, E_f0_sq, E_f1_sq, E_f1, E_f0_sq_f1,
            E_f0, E_f0_f1_sq, E_f0_f1)
        Cmat[1, 0] = Cmat[0, 1]

        Cmat[0, 2] = self._covariance_of_variances(
            nsamples, E_f0_sq_f2_sq, E_f0_sq, E_f2_sq, E_f2, E_f0_sq_f2,
            E_f0, E_f0_f2_sq, E_f0_f2)
        Cmat[2, 0] = Cmat[0, 2]

        Cmat[1, 2] = self._covariance_of_variances(
            nsamples, E_f1_sq_f2_sq, E_f1_sq, E_f2_sq, E_f2, E_f1_sq_f2,
            E_f1, E_f1_f2_sq, E_f1_f2)
        Cmat[2, 1] = Cmat[1, 2]

        variances = np.diag(cov)
        kurtosis = self.get_kurtosis()
        C_mat_diag = (kurtosis-(nsamples-3)/(nsamples-1)*variances**2)/nsamples
        for ii in range(3):
            Cmat[ii, ii] = C_mat_diag[ii]

        return Cmat

    def costs(self) -> np.ndarray:
        """
        The nominal costs of each model for a single sample

        Returns
        -------
        values : np.ndarray (nmodels)
            Model costs
        """
        return np.array([1., 0.01, 0.001])


class ShortColumnModelEnsemble(ACVBenchmark):
    def __init(self):
        super().__init__()

    def _extract_variables(self, samples):
        assert samples.shape[0] == 5
        b = samples[0, :]
        h = samples[1, :]
        P = samples[2, :]
        M = samples[3, :]
        Y = samples[4, :]
        if self._apply_lognormal:
            Y = np.exp(Y)
        return b, h, P, M, Y

    def m0(self, samples):
        b, h, P, M, Y = self._extract_variables(samples)
        return (1 - 4*M/(b*(h**2)*Y) - (P/(b*h*Y))**2)[:, None]

    def m1(self, samples):
        b, h, P, M, Y = self._extract_variables(samples)
        return (1 - 3.8*M/(b*(h**2)*Y) - (
            (P*(1 + (M-2000)/4000))/(b*h*Y))**2)[:, None]

    def m2(self, samples):
        b, h, P, M, Y = self._extract_variables(samples)
        return (1 - M/(b*(h**2)*Y) - (P/(b*h*Y))**2)[:, None]

    def m3(self, samples):
        b, h, P, M, Y = self._extract_variables(samples)
        return (1 - M/(b*(h**2)*Y) - (P*(1 + M)/(b*h*Y))**2)[:, None]

    def m4(self, samples):
        b, h, P, M, Y = self._extract_variables(samples)
        return (1 - M/(b*(h**2)*Y) - (P*(1 + M)/(h*Y))**2)[:, None]

    def _set_funs(self, nmodels):
        self._apply_lognormal = False
        return [self.m0, self.m1, self.m2, self.m3, self.m4][:nmodels], 1

    def _set_variable(self, nmodels):
        univariate_variables = [
            stats.uniform(5, 10), stats.uniform(15, 10), stats.norm(500, 100),
            stats.norm(2000, 400), stats.lognorm(s=0.5, scale=np.exp(5))]
        return IndependentMarginalsVariable(univariate_variables)

    def _set_quadrature_rule(self, *args, **kwargs):
        nsamples = int(1e7)
        quad_x = self.variable.rvs(nsamples)
        quad_w = np.full((nsamples), 1/nsamples)
        return quad_x, quad_w


class MultioutputModelEnsemble(ACVBenchmark):
    """
    Benchmark for testing multifidelity algorithms that estimate statistics
    for vector valued models of varying fidelity.
    """
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
             np.sqrt(7)*samples.T**2,  # alexs paper
             # np.sqrt(6.9)*samples.T**2, # test
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

    def _set_variable(self):
        return IndependentMarginalsVariable([stats.uniform(0, 1)])

    def _set_funs(self):
        return [self.f0, self.f1, self.f2], 3

    def costs(self) -> np.ndarray:
        """
        The nominal costs of each model for a single sample

        Returns
        -------
        values : np.ndarray (nmodels)
            Model costs
        """
        return np.array([1., 0.01, 0.001])

    def _uniform_means(self):
        return np.array([
            [np.sqrt(11)/6, 1/5, 0.0],
            [np.sqrt(7)/4, np.sqrt(7)/3, 0.0],
            [1/(2*np.sqrt(3)), np.sqrt(3)/4, 0.0],
        ])

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
