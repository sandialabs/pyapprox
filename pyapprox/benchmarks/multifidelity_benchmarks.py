import numpy as np
from scipy import stats
from functools import partial

from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.variables.transforms import AffineTransform
from pyapprox.util.utilities import (
    scipy_gauss_legendre_pts_wts_1D, scipy_gauss_hermite_pts_wts_1D,
    get_tensor_product_quadrature_rule
)


class PolynomialModelEnsemble(object):
    def __init__(self):
        self.nmodels = 5
        self.nvars = 1
        self.models = [self.m0, self.m1, self.m2, self.m3, self.m4]

        univariate_variables = [stats.uniform(0, 1)]
        self.variable = IndependentMarginalsVariable(
            univariate_variables)
        self.generate_samples = self.variable.rvs

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

    def get_means(self):
        x, w = scipy_gauss_legendre_pts_wts_1D(50)
        # scale to [0,1]
        x = (x[np.newaxis, :]+1)/2
        nsamples = x.shape[1]
        nqoi = len(self.models)
        vals = np.empty((nsamples, nqoi))
        for ii in range(nqoi):
            vals[:, ii] = self.models[ii](x)[:, 0]
        means = vals.T.dot(w)
        return means

    def get_covariance_matrix(self):
        x, w = scipy_gauss_legendre_pts_wts_1D(10)
        # scale to [0,1]
        x = (x[np.newaxis, :]+1)/2
        nsamples = x.shape[1]
        nqoi = len(self.models)
        vals = np.empty((nsamples, nqoi))
        for ii in range(nqoi):
            vals[:, ii] = self.models[ii](x)[:, 0]
        cov = np.cov(vals, aweights=w, rowvar=False, ddof=0)
        return cov


class TunableModelEnsemble(object):

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
        assert len(self.shifts) == 2
        self.models = [self.m0, self.m1, self.m2]

        univariate_variables = [stats.uniform(-1, 2), stats.uniform(-1, 2)]
        self.variable = IndependentMarginalsVariable(
            univariate_variables)
        self.generate_samples = self.variable.rvs

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

    def get_covariance_matrix(self):
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

    def get_means(self):
        return np.array(
            [0, self.shifts[0], self.shifts[1]])

    def get_kurtosis(self):
        return np.array([
            (self.A0**4*(213+29*np.cos(4*self.theta0)))/5082,
            (self.A1**4*(93+5*np.cos(4*self.theta1)))/1274,
            -(1/30)*self.A2**4*(-7+np.cos(4*self.theta2))])


class ShortColumnModelEnsemble(object):
    def __init__(self):
        self.nmodels = 5
        self.nvars = 5
        self.models = [self.m0, self.m1, self.m2, self.m3, self.m4]
        self.apply_lognormal = False

        univariate_variables = [
            stats.uniform(5, 10), stats.uniform(15, 10), stats.norm(500, 100),
            stats.norm(2000, 400), stats.lognorm(s=0.5, scale=np.exp(5))]
        self.variable = IndependentMarginalsVariable(
            univariate_variables)
        self.generate_samples = self.variable.rvs

    def extract_variables(self, samples):
        assert samples.shape[0] == 5
        b = samples[0, :]
        h = samples[1, :]
        P = samples[2, :]
        M = samples[3, :]
        Y = samples[4, :]
        if self.apply_lognormal:
            Y = np.exp(Y)
        return b, h, P, M, Y

    def m0(self, samples):
        b, h, P, M, Y = self.extract_variables(samples)
        return (1 - 4*M/(b*(h**2)*Y) - (P/(b*h*Y))**2)[:, np.newaxis]

    def m1(self, samples):
        b, h, P, M, Y = self.extract_variables(samples)
        return (1 - 3.8*M/(b*(h**2)*Y) - (
            (P*(1 + (M-2000)/4000))/(b*h*Y))**2)[:, np.newaxis]

    def m2(self, samples):
        b, h, P, M, Y = self.extract_variables(samples)
        return (1 - M/(b*(h**2)*Y) - (P/(b*h*Y))**2)[:, np.newaxis]

    def m3(self, samples):
        b, h, P, M, Y = self.extract_variables(samples)
        return (1 - M/(b*(h**2)*Y) - (P*(1 + M)/(b*h*Y))**2)[:, np.newaxis]

    def m4(self, samples):
        b, h, P, M, Y = self.extract_variables(samples)
        return (1 - M/(b*(h**2)*Y) - (P*(1 + M)/(h*Y))**2)[:, np.newaxis]

    def get_quadrature_rule(self):
        nvars = self.variable.num_vars()
        degrees = [10]*nvars
        var_trans = AffineTransform(self.variable)
        univariate_quadrature_rules = [
            scipy_gauss_legendre_pts_wts_1D, scipy_gauss_legendre_pts_wts_1D,
            scipy_gauss_hermite_pts_wts_1D,
            scipy_gauss_hermite_pts_wts_1D, scipy_gauss_hermite_pts_wts_1D]
        x, w = get_tensor_product_quadrature_rule(
            degrees, self.variable.num_vars(), univariate_quadrature_rules,
            var_trans.map_from_canonical)
        return x, w

    def get_covariance_matrix(self):
        x, w = self.get_quadrature_rule()

        nsamples = x.shape[1]
        nqoi = len(self.models)
        vals = np.empty((nsamples, nqoi))
        for ii in range(nqoi):
            vals[:, ii] = self.models[ii](x)[:, 0]
        cov = np.cov(vals, aweights=w, rowvar=False, ddof=0)
        return cov

    def get_means(self):
        x, w = self.get_quadrature_rule()
        nsamples = x.shape[1]
        nqoi = len(self.models)
        vals = np.empty((nsamples, nqoi))
        for ii in range(nqoi):
            vals[:, ii] = self.models[ii](x)[:, 0]
        return vals.T.dot(w).squeeze()
