import numpy as np
from typing import Union
from abc import ABC, abstractmethod

from pyapprox.surrogates.autogp._torch_wrappers import (
    full, asarray, sqrt, exp, inf, cdist, array, to_numpy, cholesky, empty,
    arange, sin, eye)
from pyapprox.surrogates.autogp.hyperparameter import (
    HyperParameter, HyperParameterList, IdentityHyperParameterTransform,
    LogHyperParameterTransform)
from pyapprox.surrogates.interp.indexing import compute_hyperbolic_indices


class Kernel(ABC):
    @abstractmethod
    def diag(self, X):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, X, Y=None):
        raise NotImplementedError()

    def __mul__(self, kernel):
        return ProductKernel(self, kernel)

    def __add__(self, kernel):
        return SumKernel(self, kernel)

    def __repr__(self):
        return "{0}({1})".format(
            self.__class__.__name__, self.hyp_list._short_repr())

    def _cholesky(self, kmat):
        return cholesky(kmat)


class MaternKernel(Kernel):
    def __init__(self, nu: float,
                 lenscale: Union[float, array],
                 lenscale_bounds: array,
                 nvars: int):
        self._nvars = nvars
        self.nu = nu
        self._lenscale = HyperParameter(
            "lenscale", nvars, lenscale, lenscale_bounds,
            LogHyperParameterTransform())
        self.hyp_list = HyperParameterList([self._lenscale])

    def diag(self, X):
        return full((X.shape[1],), 1)

    def _eval_distance_form(self, distances):
        if self.nu == 0.5:
            return exp(-distances)
        if self.nu == 1.5:
            tmp = distances * np.sqrt(3)
            return (1.0 + tmp) * exp(-tmp)
        if self.nu == 2.5:
            tmp = distances * np.sqrt(5)
            return (1.0 + tmp + tmp**2/3.0) * exp(-tmp)
        if self.nu == inf:
            return exp(-(distances**2)/2.0)
        raise ValueError("Matern kernel with nu={0} not supported".format(
            self.nu))

    def __call__(self, X, Y=None):
        lenscale = self._lenscale.get_values()
        X = asarray(X)
        if Y is None:
            Y = X
        else:
            Y = asarray(Y)
        distances = cdist(X.T/lenscale, Y.T/lenscale)
        return self._eval_distance_form(distances)

    def nvars(self):
        return self._nvars


class ConstantKernel(Kernel):
    def __init__(self, constant, constant_bounds=[-inf, inf],
                 transform=IdentityHyperParameterTransform()):
        self._const = HyperParameter(
            "const", 1, constant, constant_bounds, transform)
        self.hyp_list = HyperParameterList([self._const])

    def diag(self, X):
        return full((X.shape[1],), self.hyp_list.get_values()[0])

    def __call__(self, X, Y=None):
        X = asarray(X)
        if Y is None:
            Y = X
        else:
            Y = asarray(Y)
        # full does not work when const value requires grad
        # return full((X.shape[1], Y.shape[1]), self._const.get_values()[0])
        const = empty((X.shape[1], Y.shape[1]))
        const[:] = self._const.get_values()[0]
        return const


class GaussianNoiseKernel(Kernel):
    def __init__(self, constant, constant_bounds=[-inf, inf],
                 transform=IdentityHyperParameterTransform()):
        self._const = HyperParameter(
            "const", 1, constant, constant_bounds, transform)
        self.hyp_list = HyperParameterList([self._const])

    def diag(self, X):
        return full((X.shape[1],), self.hyp_list.get_values()[0])

    def __call__(self, X, Y=None):
        X = asarray(X)
        if Y is None:
            return self._const.get_values()[0]*eye(X.shape[1])
        # full does not work when const value requires grad
        # return full((X.shape[1], Y.shape[1]), self._const.get_values()[0])
        const = full((X.shape[1], Y.shape[1]), 0.)
        return const


class ProductKernel(Kernel):
    def __init__(self, kernel1, kernel2):
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.hyp_list = kernel1.hyp_list+kernel2.hyp_list

    def diag(self, X):
        return self.kernel1.diag(X) * self.kernel2.diag(X)

    def __repr__(self):
        return "{0} * {1}".format(self.kernel1, self.kernel2)

    def __call__(self, X, Y=None):
        return self.kernel1(X, Y) * self.kernel2(X, Y)


class SumKernel(Kernel):
    def __init__(self, kernel1, kernel2):
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.hyp_list = kernel1.hyp_list+kernel2.hyp_list

    def diag(self, X):
        return self.kernel1.diag(X) + self.kernel2.diag(X)

    def __repr__(self):
        return "{0} + {1}".format(self.kernel1, self.kernel2)

    def __call__(self, X, Y=None):
        return self.kernel1(X, Y) + self.kernel2(X, Y)


def univariate_monomial_basis_matrix(max_level, samples):
    assert samples.ndim == 1
    basis_matrix = samples[:, None]**arange(max_level+1)[None, :]
    return basis_matrix


def monomial_basis_matrix(indices, samples):
    """
    Evaluate a multivariate monomial basis at a set of samples.

    Parameters
    ----------
    indices : np.ndarray (num_vars, num_indices)
        The exponents of each monomial term

    samples : np.ndarray (num_vars, num_samples)
        Samples at which to evaluate the monomial

    Return
    ------
    basis_matrix : np.ndarray (num_samples, num_indices)
        The values of the monomial basis at the samples
    """
    num_vars, num_indices = indices.shape
    assert samples.shape[0] == num_vars
    num_samples = samples.shape[1]

    deriv_order = 0
    basis_matrix = empty(
        ((1+deriv_order*num_vars)*num_samples, num_indices))
    basis_vals_1d = [univariate_monomial_basis_matrix(
        indices[0, :].max(), samples[0, :])]
    basis_matrix[:num_samples, :] = basis_vals_1d[0][:, indices[0, :]]
    for dd in range(1, num_vars):
        basis_vals_1d.append(univariate_monomial_basis_matrix(
            indices[dd, :].max(), samples[dd, :]))
        basis_matrix[:num_samples, :] *= basis_vals_1d[dd][:, indices[dd, :]]
    return basis_matrix


class Monomial():
    def __init__(self, nvars, degree, coefs, coef_bounds,
                 transform=IdentityHyperParameterTransform(),
                 name="MonomialCoefficients"):
        self._nvars = nvars
        self.degree = degree
        self.indices = compute_hyperbolic_indices(self.nvars(), self.degree)
        self.nterms = self.indices.shape[1]
        self._coef = HyperParameter(
            name, self.nterms, coefs, coef_bounds, transform)
        self.hyp_list = HyperParameterList([self._coef])

    def nvars(self):
        return self._nvars

    def basis_matrix(self, samples):
        return monomial_basis_matrix(self.indices, asarray(samples))

    def __call__(self, samples):
        if self.degree == 0:
            vals = empty((samples.shape[1], 1))
            vals[:] = self._coef.get_values()
            return vals
        basis_mat = self.basis_matrix(samples)
        vals = basis_mat @ self._coef.get_values()
        return asarray(vals[:, None])

    def __repr__(self):
        return "{0}(name={1}, nvars={2}, degree={3}, nterms={4})".format(
            self.__class__.__name__, self._coef.name, self.nvars(),
            self.degree, self.nterms)


class PeriodicMaternKernel(MaternKernel):
    def __init__(self,
                 nu: float,
                 period: Union[float, array],
                 period_bounds: array,
                 lenscale: Union[float, array],
                 lenscale_bounds: array):
        super().__init__(nu, lenscale, lenscale_bounds, 1)
        self._period = HyperParameter(
            "period", 1, lenscale, lenscale_bounds,
            LogHyperParameterTransform())
        self.hyp_list += HyperParameterList([self._period])

    def __call__(self, X, Y=None):
        X = asarray(X)
        if Y is None:
            Y = X
        else:
            Y = asarray(Y)
        lenscale = self._lenscale.get_values()
        period = self._period.get_values()
        distances = cdist(X.T/period, Y.T/period)/lenscale
        return super()._eval_distance_form(distances)

    def diag(self, X):
        return super().diag(X)
