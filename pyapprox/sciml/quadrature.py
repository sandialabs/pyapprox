from abc import ABC, abstractmethod

import numpy as np

from pyapprox.sciml.util._torch_wrappers import asarray, linspace, full


class IntegralOperatorQuadratureRule(ABC):
    @abstractmethod
    def get_samples_weights(self):
        raise NotImplementedError()

    def __repr__(self):
        return "{0}(nquad={1})".format(
            self.__class__.__name__, self._nquad)


class Fixed1DGaussLegendreIOQuadRule(IntegralOperatorQuadratureRule):
    def __init__(self, nquad, var_trans=None):
        self._nquad = nquad
        # xx in [-1, 1]
        xx, ww = np.polynomial.legendre.leggauss(nquad)
        self._z_k_samples = asarray(xx)[None, :]
        self._z_k_weights = asarray(ww)[:, None]
        # hack
        self._z_k_samples = (self._z_k_samples+1)/2
        self._z_k_weights /= 2

    def get_samples_weights(self):
        return self._z_k_samples, self._z_k_weights


class Fixed1DTrapezoidIOQuadRule(IntegralOperatorQuadratureRule):
    def __init__(self, nquad, var_trans=None):
        self._nquad = nquad
        # xx in [0, 1]
        quad_xx = linspace(0, 1, nquad)
        delta = quad_xx[1]-quad_xx[0]
        quad_ww = full((nquad, ), delta)
        quad_ww[[0, -1]] /= 2
        self._z_k_samples = quad_xx[None, :]
        self._z_k_weights = quad_ww[:, None]

    def get_samples_weights(self):
        return self._z_k_samples, self._z_k_weights


class Fixed1DGaussChebyshevIOQuadRule(IntegralOperatorQuadratureRule):
    def __init__(self, nquad, var_trans=None):
        self._nquad = nquad
        # xx in [-1, 1]
        xx, ww = np.polynomial.chebyshev.chebgauss(nquad)
        self._z_k_samples = asarray(xx)[None, :]
        self._z_k_weights = asarray(ww)[:, None]

    def get_samples_weights(self):
        return self._z_k_samples, self._z_k_weights
