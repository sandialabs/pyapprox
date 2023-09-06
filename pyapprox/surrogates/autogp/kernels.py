from typing import Union
from abc import ABC, abstractmethod

from pyapprox.surrogates.autogp._torch_wrappers import (
    full, asarray, sqrt, exp, inf, cdist, array, to_numpy, cholesky)
from pyapprox.surrogates.autogp.hyperparameter import (
    HyperParameter, HyperParameterList, IdentityHyperParameterTransform,
    LogHyperParameterTransform)
from pyapprox.surrogates.interp.indexing import compute_hyperbolic_indices
from pyapprox.surrogates.interp.monomial import monomial_basis_matrix


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
        self.nvars = nvars
        self.nu = nu
        self._len_scale = HyperParameter(
            "lenscale", nvars, lenscale, lenscale_bounds,
            LogHyperParameterTransform())
        self.hyp_list = HyperParameterList([self._len_scale])

    def diag(self, X):
        return full((X.shape[1],), 1)

    def __call__(self, X, Y=None):
        len_scale = self._len_scale.get_values()
        X = asarray(X)
        if Y is None:
            Y = X
        else:
            Y = asarray(Y)
        distances = cdist(X.T/len_scale, Y.T/len_scale)
        if self.nu == 0.5:
            return exp(-distances)
        if self.nu == 1.5:
            tmp = distances * sqrt(3)
            return (1.0 + tmp) * exp(-tmp)
        if self.nu == 2.5:
            tmp = distances * sqrt(5)
            return (1.0 + tmp + tmp**2/3.0) * exp(-tmp)
        if self.nu == inf:
            return exp(-(distances**2)/2.0)
        raise ValueError("Matern kernel with nu={0} not supported".format(
            self.nu))


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
        return full((X.shape[1], Y.shape[1]), self._const.get_values()[0])


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
        X = asarray(X)
        if Y is None:
            Y = X
        else:
            Y = asarray(Y)
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
        X = asarray(X)
        if Y is None:
            Y = X
        else:
            Y = asarray(Y)
        return self.kernel1(X, Y) + self.kernel2(X, Y)


class Monomial():
    def __init__(self, nvars, degree, coefs, coef_bounds,
                 transform=IdentityHyperParameterTransform(),
                 name="MonomialCoefficients"):
        self.nvars = nvars
        self.degree = degree
        self.indices = compute_hyperbolic_indices(self.nvars, self.degree)
        self.nterms = self.indices.shape[1]
        self._coef = HyperParameter(
            name, self.nterms, coefs, coef_bounds, transform)
        self.hyp_list = HyperParameterList([self._coef])

    def __call__(self, samples):
        basis_mat = monomial_basis_matrix(
            self.indices, to_numpy(samples), deriv_order=0)
        vals = basis_mat.dot(self._coef.get_values())
        return asarray(vals[:, None])

    def __repr__(self):
        return "{0}(name={1}, nvars={2}, degree={3}, nterms={4})".format(
            self.__class__.__name__, self._coef.name, self.nvars,
            self.self.degree, self.nterms)
