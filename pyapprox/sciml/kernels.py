from typing import Union
from abc import ABC, abstractmethod

import numpy as np
import scipy

from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.surrogates.polychaos.gpc import get_polynomial_from_variable
from pyapprox.surrogates.interp.indexing import (
    compute_hyperbolic_indices)
from pyapprox.surrogates.interp.tensorprod import (
    UnivariatePiecewiseLinearBasis, UnivariatePiecewiseMidPointConstantBasis,
    UnivariatePiecewiseQuadraticBasis)
from pyapprox.surrogates.integrate import integrate

from pyapprox.sciml.util._torch_wrappers import (
    exp, cdist, asarray, inf, full, array, empty, get_diagonal, hstack, norm,
    to_numpy)
from pyapprox.sciml.util.hyperparameter import (
    HyperParameter, HyperParameterList, LogHyperParameterTransform,
    IdentityHyperParameterTransform)


class Kernel(ABC):
    def diag(self, X1):
        return get_diagonal(self(X1))

    @abstractmethod
    def __call__(self, X1, X2=None):
        raise NotImplementedError()

    def __mul__(self, kernel):
        return ProductKernel(self, kernel)

    def __add__(self, kernel):
        return SumKernel(self, kernel)

    def __repr__(self):
        return "{0}({1})".format(
            self.__class__.__name__, self.hyp_list._short_repr())


class ProductKernel(Kernel):
    def __init__(self, kernel1, kernel2):
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.hyp_list = kernel1.hyp_list+kernel2.hyp_list

    def diag(self, X1):
        return self.kernel1.diag(X1) * self.kernel2.diag(X1)

    def __repr__(self):
        return "{0} * {1}".format(self.kernel1, self.kernel2)

    def __call__(self, X1, X2=None):
        return self.kernel1(X1, X2) * self.kernel2(X1, X2)


class SumKernel(Kernel):
    def __init__(self, kernel1, kernel2):
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.hyp_list = kernel1.hyp_list+kernel2.hyp_list

    def diag(self, X1):
        return self.kernel1.diag(X1) + self.kernel2.diag(X1)

    def __repr__(self):
        return "{0} + {1}".format(self.kernel1, self.kernel2)

    def __call__(self, X1, X2=None):
        return self.kernel1(X1, X2) + self.kernel2(X1, X2)


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

    def diag(self, X1):
        return full((X1.shape[1],), 1)

    def _eval_distance_form(self, distances):
        if self.nu == inf:
            return exp(-(distances**2)/2.)
        if self.nu == 5/2:
            tmp = np.sqrt(5)*distances
            return (1.0+tmp+tmp**2/3.)*exp(-tmp)
        if self.nu == 3/2:
            tmp = np.sqrt(3)*distances
            return (1.+tmp)*exp(-tmp)
        if self.nu == 1/2:
            return exp(-distances)
        raise ValueError("Matern kernel with nu={0} not supported".format(
            self.nu))

    def __call__(self, X1, X2=None):
        lenscale = self._lenscale.get_values()
        X1 = asarray(X1)
        if X2 is None:
            X2 = X1
        else:
            X2 = asarray(X2)
        distances = cdist(X1.T/lenscale, X2.T/lenscale)
        return self._eval_distance_form(distances)

    def nvars(self):
        return self._nvars


class ConstantKernel(Kernel):
    def __init__(self, constant, constant_bounds=[-inf, inf],
                 transform=IdentityHyperParameterTransform()):
        self._const = HyperParameter(
            "const", 1, constant, constant_bounds, transform)
        self.hyp_list = HyperParameterList([self._const])

    def diag(self, X1):
        return full((X1.shape[1],), self.hyp_list.get_values()[0])

    def __call__(self, X1, X2=None):
        X1 = asarray(X1)
        if X2 is None:
            X2 = X1
        else:
            X2 = asarray(X2)
        # full does not work when const value requires grad
        # return full((X1.shape[1], X2.shape[1]), self._const.get_values()[0])
        const = empty((X1.shape[1], X2.shape[1]))
        const[:] = self._const.get_values()[0]
        return const


class PolynomialKernel(Kernel):
    def __init__(self,
                 degree: float,
                 sigmasq: Union[float, array],
                 sigmasq_bounds: array,
                 scale: float,
                 scale_bounds: array,
                 shift: float,
                 shift_bounds: array):
        self._nvars = 1
        self._degree = degree
        self._sigmasq = HyperParameter(
            "sigmasq", 1, sigmasq, sigmasq_bounds,
            LogHyperParameterTransform())
        self._scale = HyperParameter(
            "scale", 1, scale, scale_bounds,
            IdentityHyperParameterTransform())
        self._shift = HyperParameter(
            "shift", 1, shift, shift_bounds,
            IdentityHyperParameterTransform())
        self.hyp_list = HyperParameterList(
            [self._sigmasq, self._scale, self._shift])

    def __call__(self, X1, X2=None):
        sigmasq = self._sigmasq.get_values()
        scale = self._scale.get_values()
        shift = self._shift.get_values()
        X1 = asarray(X1)
        if X2 is None:
            X2 = X1
        else:
            X2 = asarray(X2)
        K = (scale*(X1-shift).T @ (X2-shift) + sigmasq)**self._degree
        return K


class Legendre1DHilbertSchmidtKernel(Kernel):
    def __init__(self,
                 nterms: float,
                 weights: Union[float, array],
                 weight_bounds: array,
                 normalize=True):
        self._nvars = 1
        self._nterms = nterms
        self._normalize = normalize
        self._weights = HyperParameter(
            "weights", self._nterms, weights, weight_bounds,
            LogHyperParameterTransform())
        self.hyp_list = HyperParameterList([self._weights])

    def __call__(self, X1, X2=None):
        weights = self._weights.get_values()
        X1 = asarray(X1)
        if X2 is None:
            X2 = X1
        else:
            X2 = asarray(X2)
        X1 = 2*X1-1
        X2 = 2*X2-1  # hack
        X1basis = hstack(
            [scipy.special.eval_legendre(dd, X1[0])[:, None]
             for dd in range(self._nterms)])
        X2basis = hstack(
            [scipy.special.eval_legendre(dd, X2[0])[:, None]
             for dd in range(self._nterms)])
        if self._normalize:
            X1basis /= norm(X1basis, axis=1)[:, None]
            X2basis /= norm(X2basis, axis=1)[:, None]
        K = (X1basis*weights) @ X2basis.T
        return K


class HilbertSchmidtBasis(ABC):
    @abstractmethod
    def __call__(self, samples: array):
        raise NotImplementedError

    @abstractmethod
    def nterms(self):
        raise NotImplementedError

    @abstractmethod
    def nvars(self):
        raise NotImplementedError

    def __repr__(self):
        return "{0}".format(self.__class__.__name__)


class PCEHilbertSchmidtBasis(HilbertSchmidtBasis):
    def __init__(self,
                 marginal_variables,
                 degree: int,
                 nquad: int = None):
        if hasattr(marginal_variables, 'rvs'):
            self._variables = (
                IndependentMarginalsVariable([marginal_variables]))
        elif hasattr(marginal_variables, '__iter__'):
            self._variables = IndependentMarginalsVariable(marginal_variables)
        self._poly = get_polynomial_from_variable(self._variables)
        indices = compute_hyperbolic_indices(
            self._variables.num_vars(), degree, 1.0)
        self._poly.set_indices(indices)
        if nquad is None:
            nquad = degree+2
        self._quadrule = integrate(
            "tensorproduct", self._variables,
            levels=[nquad]*self._variables.num_vars())
        # avoid error about negative strides thrown by torch
        self._quadrule = (self._quadrule[0].copy(), self._quadrule[1])

    def nterms(self):
        return self._poly.indices.shape[1]

    def nvars(self):
        return self._variables.num_vars()

    def __call__(self, samples):
        return asarray(self._poly.basis_matrix(to_numpy(samples)))

    def quadrature_rule(self):
        return self._quadrule

    def __repr__(self):
        return "{0}(nterms={1})".format(
            self.__class__.__name__, self.nterms())


class EquidistantPiecewisePolyBasis1D(HilbertSchmidtBasis):
    def __init__(self,
                 bounds: Union[list, array],
                 degree: int,
                 nmesh: int):
        self._degree = degree
        if self._degree == 0:
            self._basis = UnivariatePiecewiseMidPointConstantBasis()
        elif self._degree == 1:
            self._basis = UnivariatePiecewiseLinearBasis()
        elif self._degree == 2:
            self._basis = UnivariatePiecewiseQuadraticBasis()
        else:
            raise ValueError("degree {0} not supported".format(degree))
        self._mesh = np.linspace(*bounds, nmesh)[None, :]
        self._basis.set_nodes(self._mesh)

    def nterms(self):
        return self._basis.nterms()

    def nvars(self):
        return 1

    def __call__(self, samples):
        return self._basis(samples)

    def quadrature_rule(self):
        return self._basis.quadrature_rule()

    def __repr__(self):
        return "{0}(degree={1}, nterms={2})".format(
            self.__class__.__name__, self._degree, self.nterms())


class HilbertSchmidtKernel(Kernel):
    def __init__(self,
                 basis: HilbertSchmidtBasis,
                 weights: Union[float, array],
                 weight_bounds: array,
                 normalize: bool = False):
        self._nvars = basis.nvars()
        self._basis = basis
        self._nterms = basis.nterms()**2
        self._normalize = normalize
        self._weights = HyperParameter(
            "weights", self._nterms, weights, weight_bounds,
            LogHyperParameterTransform())
        self.hyp_list = HyperParameterList([self._weights])

    def _get_weights(self):
        return self._weights.get_values().reshape(
            (self._basis.nterms(), self._basis.nterms()))

    def __call__(self, X1, X2=None):
        weights = self._get_weights()
        X1 = asarray(X1)
        if X2 is None:
            X2 = X1
        else:
            X2 = asarray(X2)
        X1basis_mat = self._basis(X1)
        X2basis_mat = self._basis(X2)
        if self._normalize:
            X1basis_mat /= norm(X1basis_mat, axis=1)[:, None]
            X2basis_mat /= norm(X2basis_mat, axis=1)[:, None]
        K = (X1basis_mat @ weights) @ X2basis_mat.T
        return K
