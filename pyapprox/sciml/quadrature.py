from abc import ABC, abstractmethod

import numpy as np

from pyapprox.sciml.util._torch_wrappers import (
    asarray, linspace, full, prod, cartesian_product, outer_product)


class IntegralOperatorQuadratureRule(ABC):
    @abstractmethod
    def get_samples_weights(self):
        raise NotImplementedError()

    def nquad(self):
        return self._nquad

    def __repr__(self):
        return "{0}(nquad={1})".format(
            self.__class__.__name__, self.nquad())


class Fixed1DGaussLegendreIOQuadRule(IntegralOperatorQuadratureRule):
    def __init__(self, nquad):
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
    def __init__(self, nquad):
        self._nquad = nquad
        if nquad == 1:
            quad_xx = full((nquad, ), 0)
            quad_ww = full((nquad, ), 2)
        else:
            quad_xx = linspace(-1, 1, nquad)
            delta = quad_xx[1]-quad_xx[0]
            quad_ww = full((nquad, ), delta)
            quad_ww[[0, -1]] /= 2
        self._z_k_samples = quad_xx[None, :]
        self._z_k_weights = quad_ww[:, None]

    def get_samples_weights(self):
        return self._z_k_samples, self._z_k_weights


class Fixed1DGaussChebyshevIOQuadRule(IntegralOperatorQuadratureRule):
    def __init__(self, nquad):
        self._nquad = nquad
        # xx in [-1, 1]
        xx, ww = np.polynomial.chebyshev.chebgauss(nquad)
        self._z_k_samples = asarray(xx)[None, :]
        self._z_k_weights = asarray(ww)[:, None]

    def get_samples_weights(self):
        return self._z_k_samples, self._z_k_weights


class TransformedQuadRule(IntegralOperatorQuadratureRule):
    def __init__(self, quad_rule):
        self._quad_rule = quad_rule

    def nquad(self):
        return self._quad_rule.nquad()

    @abstractmethod
    def _transform(self, points, weights):
        raise NotImplementedError

    def get_samples_weights(self):
        return self._transform(
            *self._quad_rule.get_samples_weights())


class OnePointRule1D(IntegralOperatorQuadratureRule):
    def __init__(self, point, weight):
        self._z_k_samples = asarray([point])[None, :]
        self._z_k_weights = asarray([weight])[:, None]
        self._nquad = 1

    def get_samples_weights(self):
        return self._z_k_samples, self._z_k_weights


class Transformed1DQuadRule(TransformedQuadRule):
    # Ultimately this should be only transform for bounded quad rules
    # once all base quad rules in 1D are converted to return points in [-1, 1]
    # when this is done TransformedUnitIntervalQuadRule can be deleted
    # it is only ncessary for Fixed1DTrapezoidIOQuadRule and
    # Fixed1DGaussLegendreIOQuadRule which returns points in [0, 1]
    def __init__(self, quad_rule, bounds):
        self._quad_rule = quad_rule
        self._bounds = bounds

    def _transform(self, points, weights):
        length = self._bounds[1]-self._bounds[0]
        return (points+1)/2*length+self._bounds[0], weights/2*length


class TensorProduct2DQuadRule(IntegralOperatorQuadratureRule):
    def __init__(self, quad_1, quad_2):
        self._quad_1 = quad_1
        self._quad_2 = quad_2
        self._nquad = self._quad_1.nquad()*self._quad_2.nquad()

    def get_samples_weights(self):
        x1, w1 = self._quad_1.get_samples_weights()
        x2, w2 = self._quad_2.get_samples_weights()
        return (cartesian_product([x1[0], x2[0]]),
                outer_product([w1[:, 0], w2[:, 0]])[:, None])
