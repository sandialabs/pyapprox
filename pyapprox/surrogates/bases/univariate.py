from abc import ABC, abstractmethod
import math
import warnings


import numpy as np
import scipy

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.surrogates.orthopoly.quadrature import (
    clenshaw_curtis_in_polynomial_order,
    clenshaw_curtis_poly_indices_to_quad_rule_indices,
)
from pyapprox.util.transforms import IdentityTransform


class UnivariateBasis(ABC):
    def __init__(self, trans, backend):
        if backend is None:
            backend = NumpyLinAlgMixin
        self._bkd = backend
        if trans is None:
            trans = IdentityTransform(backend=self._bkd)
        self._trans = trans

    @abstractmethod
    def _values(samples):
        raise NotImplementedError

    @abstractmethod
    def nterms(self):
        raise NotImplementedError

    def _check_samples(self, samples):
        if samples.ndim != 2 or samples.shape[0] != 1:
            raise ValueError("samples must be a 2D row vector")

    def __call__(self, samples):
        self._check_samples(samples)
        vals = self._values(samples)
        if vals.ndim != 2 or vals.shape[0] != samples.shape[1]:
            raise ValueError("returned values must be a 2D array")
        return vals

    def _derivatives(self, samples, namx, order):
        raise NotImplementedError

    def derivatives(self, samples, order):
        self._check_samples(samples)
        return self._derivatives(samples, order)

    def _quadrature_rule(self):
        raise NotImplementedError

    def quadrature_rule(self):
        samples, weights = self._quadrature_rule()
        self._check_samples(samples)
        if weights.ndim != 2 or weights.shape[1] != 1:
            raise ValueError("weights must be a 2D column vector")
        if samples.shape[1] != weights.shape[0]:
            raise ValueError("number of samples and weights are different")
        return samples, weights

    def __repr__(self):
        return "{0}(nterms={1})".format(self.__class__.__name__, self.nterms())


class Monomial1D(UnivariateBasis):
    """Univariate monomial basis."""

    def __init__(self, nterms=None, trans=None, backend=None):
        super().__init__(trans, backend)
        self._nterms = None
        if nterms is not None:
            self.set_nterms(nterms)

    def set_nterms(self, nterms):
        self._nterms = nterms

    def nterms(self):
        return self._nterms

    def _values(self, samples):
        basis_matrix = samples.T ** self._bkd.arange(self._nterms)[None, :]
        return basis_matrix

    def _derivatives(self, samples, order):
        powers = self._bkd.hstack(
            (
                self._bkd.zeros((order,)),
                self._bkd.arange(self._nterms - order),
            )
        )
        # 1 x x^2 x^3  x^4 vals
        # 0 1 2x  3x^2 4x^3 1st derivs
        # 0 0 2   6x   12x^2  2nd derivs
        consts = self._bkd.hstack(
            (
                self._bkd.zeros((order,)),
                order * self._bkd.arange(1, self._nterms - order + 1),
            )
        )
        return (samples.T ** powers[None, :]) * consts


class UnivariateInterpolatingBasis(UnivariateBasis):
    def __init__(self, trans=None, backend=None):
        super().__init__(trans, backend)
        self._quad_samples = None
        self._quad_weights = None

    def nterms(self):
        return self._quad_samples.shape[1]

    def __repr__(self):
        if self._quad_samples is None:
            return "{0}(bkd={1})".format(
                self.__class__.__name__, self._bkd.__name__
            )
        return "{0}(nterms={1}, bkd={2})".format(
            self.__class__.__name__, self.nterms(), self._bkd.__name__
        )

    @abstractmethod
    def _semideep_copy(self):
        raise NotImplementedError


def irregular_piecewise_left_constant_basis(nodes, xx, bkd=NumpyLinAlgMixin):
    # abscissa are not equidistant
    assert xx.ndim == 1
    assert nodes.ndim == 1
    nnodes = nodes.shape[0]
    if nnodes == 1:
        return bkd.full((xx.shape[0], nnodes), 1.0)
    vals = bkd.zeros((xx.shape[0], nnodes - 1))
    for ii in range(nnodes - 1):
        xl = nodes[ii]
        xr = nodes[ii + 1]
        II = bkd.where((xx >= xl) & (xx < xr))[0]
        vals[II, ii] = bkd.full((II.shape[0],), 1.0, dtype=float)
    return vals


def irregular_piecewise_right_constant_basis(nodes, xx, bkd=NumpyLinAlgMixin):
    # abscissa are not equidistant
    assert xx.ndim == 1
    assert nodes.ndim == 1
    nnodes = nodes.shape[0]
    if nnodes == 1:
        return bkd.ones((xx.shape[0], nnodes))
    vals = bkd.zeros((xx.shape[0], nnodes - 1))
    for ii in range(1, nnodes):
        xr = nodes[ii]
        xl = nodes[ii - 1]
        II = bkd.where((xx > xl) & (xx <= xr))[0]
        vals[II, ii] = bkd.ones((II.shape[0],), dtype=float)
    return vals


def irregular_piecewise_midpoint_constant_basis(
    nodes, xx, bkd=NumpyLinAlgMixin
):
    # abscissa are not equidistant
    assert xx.ndim == 1
    assert nodes.ndim == 1
    nnodes = nodes.shape[0]
    if nnodes == 1:
        return bkd.ones((xx.shape[0], nnodes))
    vals = bkd.zeros((xx.shape[0], nnodes - 1))
    for ii in range(nnodes - 1):
        xl = nodes[ii]
        xr = nodes[ii + 1]
        if ii < nnodes - 1:
            II = bkd.where((xx >= xl) & (xx < xr))[0]
        else:
            II = bkd.where((xx >= xl) & (xx <= xr))[0]
        vals[II, ii] = bkd.ones((II.shape[0],), dtype=float)
    return vals


def irregular_piecewise_linear_basis(nodes, xx, bkd=NumpyLinAlgMixin):
    # abscissa are not equidistant
    assert xx.ndim == 1
    assert nodes.ndim == 1
    nnodes = nodes.shape[0]
    if nnodes == 1:
        return bkd.ones((xx.shape[0], nnodes))
    vals = bkd.zeros((xx.shape[0], nnodes))
    for ii in range(nnodes):
        xm = nodes[ii]
        if ii > 0:
            xl = nodes[ii - 1]
            II = bkd.where((xx >= xl) & (xx <= xm))[0]
            # vals[II, ii] = (xx[II] - xl) / (xm - xl)
            vals = bkd.up(vals, (II, ii), (xx[II] - xl) / (xm - xl))
        if ii < nnodes - 1:
            xr = nodes[ii + 1]
            JJ = bkd.where((xx >= xm) & (xx <= xr))[0]
            # vals[JJ, ii] = (xr - xx[JJ]) / (xr - xm)
            vals = bkd.up(vals, (JJ, ii), (xr - xx[JJ]) / (xr - xm))
    return vals


def irregular_piecewise_linear_quadrature_weights(nodes, bkd=NumpyLinAlgMixin):
    assert nodes.ndim == 1
    nnodes = nodes.shape[0]
    if nnodes == 1:
        raise ValueError(
            "Cant compute weights from single point without bounds"
        )
    # weights = bkd.zeros((nnodes,))
    # use list so not to have to use bkd.up below. Since we are using
    # python loop here speed is not going to be much different
    weights = [0.0 for ii in range(nnodes)]
    for ii in range(nnodes):
        xm = nodes[ii]
        if ii > 0:
            xl = nodes[ii - 1]
            weights[ii] += 0.5 * (xm - xl)
            # weights = bkd.up(weights, ii, weights[ii] + 0.5 * (xm - xl))
        if ii < nnodes - 1:
            xr = nodes[ii + 1]
            weights[ii] += 0.5 * (xr - xm)
            # weights = bkd.up(weights, ii, weights[ii] + 0.5 * (xr - xm))
    return bkd.asarray(weights)


def irregular_piecewise_quadratic_basis(nodes, xx, bkd=NumpyLinAlgMixin):
    # nodes are not equidistant
    assert xx.ndim == 1
    nnodes = nodes.shape[0]
    if nnodes == 1:
        return bkd.ones((xx.shape[0], nnodes))
    if nodes.ndim != 1 or nnodes % 2 != 1:
        raise ValueError("nodes has the wrong shape")
    vals = bkd.zeros((xx.shape[0], nnodes))
    for ii in range(nnodes):
        if ii % 2 == 1:
            xl, xm, xr = nodes[ii - 1 : ii + 2]
            II = bkd.where((xx >= xl) & (xx <= xr))[0]
            # vals[II, ii] = (
            #     (xx[II] - xl) / (xm - xl) * (xx[II] - xr) / (xm - xr)
            # )
            vals = bkd.up(
                vals,
                (II, ii),
                (xx[II] - xl) / (xm - xl) * (xx[II] - xr) / (xm - xr),
            )
            continue
        if ii < nnodes - 2:
            xl, xm, xr = nodes[ii : ii + 3]
            II = bkd.where((xx >= xl) & (xx <= xr))[0]
            # vals[II, ii] = (
            #     (xx[II] - xm) / (xl - xm) * (xx[II] - xr) / (xl - xr)
            # )
            vals = bkd.up(
                vals,
                (II, ii),
                (xx[II] - xm) / (xl - xm) * (xx[II] - xr) / (xl - xr),
            )
        if ii > 1:
            xl, xm, xr = nodes[ii - 2 : ii + 1]
            II = bkd.where((xx >= xl) & (xx <= xr))[0]
            # vals[II, ii] = (
            #     (xx[II] - xl) / (xr - xl) * (xx[II] - xm) / (xr - xm)
            # )
            vals = bkd.up(
                vals,
                (II, ii),
                (xx[II] - xl) / (xr - xl) * (xx[II] - xm) / (xr - xm),
            )
    return vals


def irregular_piecewise_quadratic_quadrature_weights(
    nodes, bkd=NumpyLinAlgMixin
):
    # nodes are not equidistant
    nnodes = nodes.shape[0]
    if nnodes == 1:
        raise ValueError(
            "Cant compute weights from single point without bounds"
        )
    if nodes.ndim != 1 or nnodes % 2 != 1:
        raise ValueError("nodes has the wrong shape, it must be an odd number")
    # weights = bkd.zeros((nnodes,))
    weights = [0.0 for ii in range(nnodes)]
    for ii in range(nnodes):
        if ii % 2 == 1:
            xl, xm, xr = nodes[ii - 1 : ii + 2]
            weights[ii] = (xl - xr) ** 3 / (6 * (xm - xl) * (xm - xr))
            # weights = bkd.up(weights, ii, (xl - xr) ** 3 / (6 * (xm - xl) * (xm - xr)))
            continue
        if ii < nnodes - 2:
            xl, xm, xr = nodes[ii : ii + 3]
            weights[ii] += ((xr - xl) * (2 * xl - 3 * xm + xr)) / (
                6 * (xl - xm)
            )
            # weights = bkd.up(weights, ii, weights[ii]+((xr - xl) * (2 * xl - 3 * xm + xr)) / (
            #     6 * (xl - xm)))
        if ii > 1:
            xl, xm, xr = nodes[ii - 2 : ii + 1]
            weights[ii] += ((xl - xr) * (xl - 3 * xm + 2 * xr)) / (
                6 * (xm - xr)
            )
            # weights = bkd.up(weights, ii, weights[ii]+((xl - xr) * (xl - 3 * xm + 2 * xr)) / (
            #          6 * (xm - xr)))
    return bkd.asarray(weights)


def irregular_piecewise_cubic_basis(nodes, xx, bkd=NumpyLinAlgMixin):
    # nodes are not equidistant
    assert xx.ndim == 1
    nnodes = nodes.shape[0]
    if nnodes == 1:
        return bkd.ones((xx.shape[0], nnodes))
    if nodes.ndim != 1 or nnodes < 4 or (nnodes - 4) % 3 != 0:
        raise ValueError("nodes has the wrong shape")
    vals = bkd.zeros((xx.shape[0], nnodes))
    for ii in range(nnodes):
        if ii % 3 == 1:
            x1, x2, x3, x4 = nodes[ii - 1 : ii + 3]
            II = bkd.where((xx >= x1) & (xx <= x4))[0]
            # vals[II, ii] = (
            #     (xx[II] - x1)
            #     / (x2 - x1)
            #     * (xx[II] - x3)
            #     / (x2 - x3)
            #     * (xx[II] - x4)
            #     / (x2 - x4)
            # )
            vals = bkd.up(
                vals,
                (II, ii),
                (xx[II] - x1)
                / (x2 - x1)
                * (xx[II] - x3)
                / (x2 - x3)
                * (xx[II] - x4)
                / (x2 - x4),
            )
            continue
        if ii % 3 == 2:
            x1, x2, x3, x4 = nodes[ii - 2 : ii + 2]
            II = bkd.where((xx >= x1) & (xx <= x4))[0]
            # vals[II, ii] = (
            #     (xx[II] - x1)
            #     / (x3 - x1)
            #     * (xx[II] - x2)
            #     / (x3 - x2)
            #     * (xx[II] - x4)
            #     / (x3 - x4)
            # )
            vals = bkd.up(
                vals,
                (II, ii),
                (xx[II] - x1)
                / (x3 - x1)
                * (xx[II] - x2)
                / (x3 - x2)
                * (xx[II] - x4)
                / (x3 - x4),
            )
            continue
        if ii % 3 == 0 and ii < nnodes - 3:
            x1, x2, x3, x4 = nodes[ii : ii + 4]
            II = bkd.where((xx >= x1) & (xx <= x4))[0]
            # vals[II, ii] = (
            #     (xx[II] - x2)
            #     / (x1 - x2)
            #     * (xx[II] - x3)
            #     / (x1 - x3)
            #     * (xx[II] - x4)
            #     / (x1 - x4)
            # )
            vals = bkd.up(
                vals,
                (II, ii),
                (xx[II] - x2)
                / (x1 - x2)
                * (xx[II] - x3)
                / (x1 - x3)
                * (xx[II] - x4)
                / (x1 - x4),
            )
        if ii % 3 == 0 and ii >= 3:
            x1, x2, x3, x4 = nodes[ii - 3 : ii + 1]
            II = bkd.where((xx >= x1) & (xx <= x4))[0]
            # vals[II, ii] = (
            #     (xx[II] - x1)
            #     / (x4 - x1)
            #     * (xx[II] - x2)
            #     / (x4 - x2)
            #     * (xx[II] - x3)
            #     / (x4 - x3)
            # )
            vals = bkd.up(
                vals,
                (II, ii),
                (xx[II] - x1)
                / (x4 - x1)
                * (xx[II] - x2)
                / (x4 - x2)
                * (xx[II] - x3)
                / (x4 - x3),
            )
    return vals


def irregular_piecewise_cubic_quadrature_weights(nodes, bkd=NumpyLinAlgMixin):
    # An interpolating quadrature with n + 1 nodes that are symmetrically
    # placed around the center of the interval will integrate polynomials up to
    # degree n exactly when n is odd, and up to degree n + 1 exactly when
    # n is even.
    nnodes = nodes.shape[0]
    if nnodes == 1:
        raise ValueError(
            "Cant compute weights from single point without bounds"
        )
    if nodes.ndim != 1 or nnodes < 4 or (nnodes - 4) % 3 != 0:
        raise ValueError(f"nodes has the wrong shape {nnodes}")
    # weights = bkd.zeros((nnodes,))
    weights = [0.0 for ii in range(nnodes)]
    for ii in range(nnodes):
        if ii % 3 == 1:
            a, b, c, d = nodes[ii - 1 : ii + 3]
            weights[ii] = ((a - d) ** 3 * (a - 2 * c + d)) / (
                12 * (-a + b) * (b - c) * (b - d)
            )
            continue
        if ii % 3 == 2:
            a, b, c, d = nodes[ii - 2 : ii + 2]
            weights[ii] = ((a - d) ** 3 * (a - 2 * b + d)) / (
                12 * (-a + c) * (-b + c) * (c - d)
            )
            continue
        if ii % 3 == 0 and ii < nnodes - 3:
            a, b, c, d = nodes[ii : ii + 4]
            weights[ii] += (
                (d - a)
                * (
                    3 * a**2
                    + 6 * b * c
                    - 2 * (b + c) * d
                    + d**2
                    + 2 * a * (-2 * (b + c) + d)
                )
            ) / (12 * (a - b) * (a - c))
        if ii % 3 == 0 and ii >= 3:
            a, b, c, d = nodes[ii - 3 : ii + 1]
            weights[ii] += (
                (a - d)
                * (
                    a**2
                    + 6 * b * c
                    - 2 * a * (b + c - d)
                    - 4 * b * d
                    - 4 * c * d
                    + 3 * d**2
                )
            ) / (12 * (b - d) * (-c + d))
    return bkd.asarray(weights)


def univariate_lagrange_polynomial(abscissa, samples, bkd=NumpyLinAlgMixin):
    assert abscissa.ndim == 1
    assert samples.ndim == 1
    nabscissa = abscissa.shape[0]
    denoms = abscissa[:, None] - abscissa[None, :]
    numers = samples[:, None] - abscissa[None, :]
    # values = bkd.empty((samples.shape[0], nabscissa))
    values = []
    for ii in range(nabscissa):
        # l_j(x) = prod_{i!=j} (x-x_i)/(x_j-x_i)
        denom = bkd.prod(denoms[ii, :ii]) * bkd.prod(denoms[ii, ii + 1 :])
        # denom = bkd.prod(bkd.delete(denoms[ii], ii))
        # numer = bkd.prod(bkd.delete(numers, ii, axis=1), axis=1)
        numer = bkd.prod(numers[:, :ii], axis=1) * bkd.prod(
            numers[:, ii + 1 :], axis=1
        )
        # values[:, ii] = numer/denom
        values.append(numer / denom)
    return bkd.stack(values, axis=1)
    # return values


class UnivariatePiecewisePolynomialNodeGenerator(ABC):
    def __init__(self, backend=None):
        if backend is None:
            backend = NumpyLinAlgMixin
        self._bkd = backend
        self._bounds = None

    def set_bounds(self, bounds):
        self._bounds = bounds

    def __call__(self, nnodes):
        if self._bounds is None:
            raise ValueError("must call set_bounds")
        nodes = self._nodes(nnodes)
        if nodes.ndim != 2 or nodes.shape[0] != 1:
            raise RuntimeError("nodes returned does must be a 2D row vector")
        return nodes

    @abstractmethod
    def _nodes(self, nnodes):
        raise NotImplementedError


class UnivariateEquidistantNodeGenerator(
    UnivariatePiecewisePolynomialNodeGenerator
):
    def _nodes(self, nnodes):
        return self._bkd.linspace(*self._bounds, nnodes)[None, :]


class DydadicEquidistantNodeGenerator(
    UnivariatePiecewisePolynomialNodeGenerator
):
    # useful for adaptive interpolation
    def _nodes(self, nnodes):
        if nnodes == 1:
            level = 0
            return self._bkd.array([[(self._bounds[0] + self._bounds[1]) / 2]])

        if not _is_power_of_two(nnodes - 1):
            raise ValueError("nnodes-1 must be a power of 2")

        level = int(round(math.log(nnodes - 1, 2), 0))
        idx = clenshaw_curtis_poly_indices_to_quad_rule_indices(level)
        return self._bkd.linspace(*self._bounds, nnodes)[None, idx]


class UnivariatePiecewisePolynomialBasis(UnivariateInterpolatingBasis):
    def __init__(self, bounds, node_gen=None, trans=None, backend=None):
        super().__init__(trans, backend)
        if node_gen is None:
            node_gen = UnivariateEquidistantNodeGenerator(self._bkd)

        self._node_gen = None
        # nodes may be different than quad_samples
        # e.g. when using piecewise constant basis
        self._nodes = None
        self._bounds = None

        self.set_node_generator(node_gen)
        self.set_bounds(bounds)

    def set_node_generator(self, node_gen):
        if not isinstance(
            node_gen, UnivariatePiecewisePolynomialNodeGenerator
        ):
            raise ValueError(
                "node_gen must be an instance of {0}".format(
                    "UnivariatePiecewisePolynomialNodeGenerator"
                )
            )
        self._node_gen = node_gen
        if self._bounds is not None:
            # self.set_bounds calls self._node_gen.set_bounds()
            # but this will be ignored when reseting node_gen here.
            raise RuntimeError("Do not set bounds before setting generator")

    def _copy(self):
        other = self.__class__(
            self._bounds, self._node_gen, self._trans, self._bkd
        )
        # need to copy nodes or changing nodes in one basis
        # may change it in anotehr unintentionally
        if self._nodes is not None:
            other._set_nodes(self._bkd.copy(self._nodes))
        return other

    def _semideep_copy(self):
        return self._copy()

    @abstractmethod
    def _evaluate_from_nodes(self, nodes):
        raise NotImplementedError

    def _values(self, samples):
        if self._nodes is None:
            raise RuntimeError("must call set_nodes")
        idx = self._bkd.argsort(self._nodes[0])
        vals = self._evaluate_from_nodes(self._nodes[:, idx], samples)
        inverse_idx = self._bkd.empty(self._nodes.shape[1], dtype=int)
        inverse_idx = self._bkd.up(
            inverse_idx, idx, self._bkd.arange(self._nodes.shape[1])
        )
        return vals[:, inverse_idx]

    def set_bounds(self, bounds):
        """Set the bounds of the quadrature rule"""
        if len(bounds) != 2:
            raise ValueError("must specifiy an upper and lower bound")
        self._bounds = bounds
        self._node_gen.set_bounds(bounds)

    @abstractmethod
    def _quadrature_rule_from_nodes(self, nodes):
        raise NotImplementedError

    def _quadrature_rule(self):
        return self._quad_samples, self._quad_weights

    def _set_nodes(self, nodes):
        if nodes.ndim != 2 or nodes.shape[0] != 1:
            raise ValueError("nodes must be a 2D row vector")
        self._nodes = nodes
        self._quad_samples, self._quad_weights = (
            self._quadrature_rule_from_nodes(self._nodes)
        )

    def _active_node_indices_for_quadrature(self):
        # used in time_integration.py
        return self._bkd.arange(self.nterms())

    def __repr__(self):
        if self._quad_samples is None:
            return "{0}(bkd={1})".format(self.__class__.__name__, self._bkd)
        return "{0}(nterms={1}, nnodes={2}, bkd={3})".format(
            self.__class__.__name__,
            self.nterms(),
            self._nodes.shape[1],
            self._bkd,
        )

    def set_nterms(self, nterms):
        self._set_nodes(self._node_gen(nterms))


class UnivariatePiecewiseConstantBasis(UnivariatePiecewisePolynomialBasis):
    def set_nterms(self, nterms):
        # need to use nterms + 1 because nterms = nnodes-1 for piecewise
        # constant basis
        self._set_nodes(self._node_gen(nterms + 1))


class UnivariatePiecewiseLeftConstantBasis(UnivariatePiecewiseConstantBasis):
    def _evaluate_from_nodes(self, nodes, samples):
        return irregular_piecewise_left_constant_basis(
            nodes[0], samples[0], self._bkd
        )

    def _quadrature_rule_from_nodes(self, nodes):
        return nodes[:, :-1], self._bkd.diff(nodes[0])[:, None]

    def nterms(self):
        return self._nodes.shape[1] - 1

    def _active_node_indices_for_quadrature(self):
        return self._bkd.arange(self.nterms() - 1)


class UnivariatePiecewiseRightConstantBasis(UnivariatePiecewiseConstantBasis):
    def _evaluate_from_nodes(self, nodes, samples):
        return irregular_piecewise_right_constant_basis(
            nodes[0], samples[0], self._bkd
        )

    def _quadrature_rule_from_nodes(self, nodes):
        # unlike other higherorder methods weights.shape[0] != nodes.shape[0]
        return nodes[:, 1:], self._bkd.diff(nodes[0])[:, None]

    def nterms(self):
        return self._nodes.shape[1] - 1

    def _active_node_indices_for_quadrature(self):
        return self._bkd.arange(1, self.nterms())


class UnivariatePiecewiseMidPointConstantBasis(
    UnivariatePiecewiseConstantBasis
):
    def _evaluate_from_nodes(self, nodes, samples):
        return irregular_piecewise_midpoint_constant_basis(
            nodes[0], samples[0], self._bkd
        )

    def _quadrature_rule_from_nodes(self, nodes):
        return (
            (nodes[:, 1:] + nodes[:, :-1]) / 2,
            self._bkd.diff(nodes[0])[:, None],
        )

    def nterms(self):
        return self._nodes.shape[1] - 1

    def _active_node_indices_for_quadrature(self):
        raise ValueError("Quadrature points do not coincide with nodes")


class UnivariatePiecewiseLinearBasis(UnivariatePiecewisePolynomialBasis):
    def _evaluate_from_nodes(self, nodes, samples):
        return irregular_piecewise_linear_basis(
            nodes[0], samples[0], self._bkd
        )

    def _quadrature_rule_from_nodes(self, nodes):
        if nodes.shape[1] == 1:
            return (
                self._nodes,
                self._bkd.full((1, 1), self._bounds[1] - self._bounds[0]),
            )
        return (
            nodes,
            irregular_piecewise_linear_quadrature_weights(nodes[0], self._bkd)[
                :, None
            ],
        )


class UnivariatePiecewiseQuadraticBasis(UnivariatePiecewisePolynomialBasis):
    def _evaluate_from_nodes(self, nodes, samples):
        return irregular_piecewise_quadratic_basis(
            nodes[0], samples[0], self._bkd
        )

    def _quadrature_rule_from_nodes(self, nodes):
        if nodes.shape[1] == 1:
            return (
                self._nodes,
                self._bkd.full((1, 1), self._bounds[1] - self._bounds[0]),
            )
        return (
            nodes,
            irregular_piecewise_quadratic_quadrature_weights(
                nodes[0], self._bkd
            )[:, None],
        )


class UnivariatePiecewiseCubicBasis(UnivariatePiecewisePolynomialBasis):
    def _evaluate_from_nodes(self, nodes, samples):
        return irregular_piecewise_cubic_basis(nodes[0], samples[0], self._bkd)

    def _quadrature_rule_from_nodes(self, nodes):
        if nodes.shape[1] == 1:
            return (
                self._nodes,
                self._bkd.full((1, 1), self._bounds[1] - self._bounds[0]),
            )
        return (
            nodes,
            irregular_piecewise_cubic_quadrature_weights(nodes[0], self._bkd)[
                :, None
            ],
        )


class UnivariateQuadratureRule(ABC):
    def __init__(self, backend=None, store=False):
        """
        Parameters
        ----------
        store : bool
            Store all quadrature rules computed. This is useful
            if repetedly calling the quadrature rule and
            the cost of computing the quadrature rule is nontrivial
        """
        if backend is None:
            backend = NumpyLinAlgMixin
        self._bkd = backend
        self._store = store
        self._quad_samples = dict()
        self._quad_weights = dict()
        self._nnodes = None

    def set_nnodes(self, nnodes):
        self._nnodes = nnodes

    @abstractmethod
    def _quad_rule(self, nnodes):
        raise NotImplementedError

    def __call__(self, nnodes=None):
        if nnodes is None:
            if self._nnodes is None:
                raise ValueError(
                    "If nnodes is None must first call set_nnodes"
                )
            nnodes = self._nnodes
        if self._store and nnodes in self._quad_samples:
            return self._quad_samples[nnodes], self._quad_weights[nnodes]
        quad_samples, quad_weights = self._quad_rule(nnodes)
        if self._store:
            self._quad_samples[nnodes] = quad_samples
            self._quad_weights[nnodes] = quad_weights
        return quad_samples, quad_weights

    def __repr__(self):
        return "{0}(bkd={1})".format(
            self.__class__.__name__, self._bkd.__name__
        )


def _is_power_of_two(integer):
    return (integer & (integer - 1) == 0) and integer != 0


class ClenshawCurtisQuadratureRule(UnivariateQuadratureRule):
    """Integrates functions on [-1, 1] with weight function 1/2 or 1."""

    def __init__(
        self, prob_measure=True, bounds=None, backend=None, store=False
    ):
        super().__init__(backend=backend, store=store)
        self._prob_measure = prob_measure

        if bounds is None:
            msg = (
                "Bounds not set. Proceed with caution. "
                "User is responsible for ensuring samples are in "
                "canonical domain of the polynomial."
            )
            warnings.warn(msg, UserWarning)
            bounds = [-1, 1]
        self._bounds = bounds

    def _quad_rule(self, nnodes):
        # rule requires nnodes = 2**l + 1 for l=1,2,3
        # so check n=nnodes-1 is a power of 2
        if nnodes == 1:
            level = 0
        else:
            if not _is_power_of_two(nnodes - 1):
                raise ValueError("nnodes-1 must be a power of 2")
            level = int(round(math.log(nnodes - 1, 2), 0))
        quad_samples, quad_weights = clenshaw_curtis_in_polynomial_order(
            level, False
        )
        length = self._bounds[1] - self._bounds[0]
        quad_samples = (quad_samples + 1) / 2 * length + self._bounds[0]
        if not self._prob_measure:
            # taking quadweights for Lebesque measure on [-1, 1] to [a,b]
            # requires multiplying weights by (b-a)/2 but clenshaw curtis
            # weights are for uniform measure 1/2 on [-1,1] so only
            # need to multiply by (b-a)
            quad_weights *= length

        return (
            self._bkd.asarray(quad_samples)[None, :],
            self._bkd.asarray(quad_weights)[:, None],
        )


class UnivariatePiecewisePolynomialQuadratureRule(UnivariateQuadratureRule):
    def __init__(
        self, basis_type, bounds=None, node_gen=None, backend=None, store=False
    ):
        super().__init__(backend=backend, store=store)
        self._basis = setup_univariate_piecewise_polynomial_basis(
            basis_type, bounds, node_gen, None, backend
        )

    def _quad_rule(self, nnodes):
        self._basis.set_nterms(nnodes)
        return self._basis.quadrature_rule()


class UnivariateLagrangeBasis(UnivariateInterpolatingBasis):
    def __init__(self, quadrature_rule, nterms=None):
        super().__init__(quadrature_rule._bkd)
        self._quad_rule = quadrature_rule
        if nterms is not None:
            self.set_nterms(nterms)

    def set_nterms(self, nterms):
        self._quad_samples, self._quad_weights = self._quad_rule(nterms)

    def _values(self, samples):
        return univariate_lagrange_polynomial(
            self._quad_samples[0], samples[0], self._bkd
        )

    def _quadrature_rule(self):
        return self._quad_samples, self._quad_weights

    def _semideep_copy(self):
        # do not copy quadrature rule as it may be used in multiple
        # dimensions of a tensor product and subspaces in a sparse grid
        return UnivariateLagrangeBasis(self._quad_rule, self.nterms())


class UnivariateBarycentricLagrangeBasis(UnivariateLagrangeBasis):
    def set_nterms(self, nterms, interval_length=None):
        super().set_nterms(nterms)
        self._interval_length = interval_length
        self._set_barycentric_weights()

    def _set_barycentric_weights(self):
        """
        Return barycentric weights for a sequence of samples. e.g. of sequence
        x0, x1, x2 where order represents the order in which the samples are
        added to the interpolant.

        E.g. using 3 points
        compites [1/((x0-x2)(x0-x1)),1/((x1-x2)(x1-x0)),1/((x2-x1)(x2-x0))]

        Note
        ----
        If length of interval [a,b]=4C then weights will grow or decay
        exponentially at C^{-n} where n is number of points causing overflow
        or underflow.

        To minimize this effect multiply each x_j-x_k by C^{-1}. This has effect
        of rescaling all weights by C^n. In rare situations where n is so large
        randomize or use Leja ordering of the samples before computing weights.
        See Barycentric Lagrange Interpolation by
        Jean-Paul Berrut and Lloyd N. Trefethen 2004
        """
        if self._interval_length is None:
            scaling_factor = 1.0
        else:
            scaling_factor = self._interval_length / 4.0

        C_inv = 1.0 / scaling_factor
        samples = self._quad_samples[0]
        nsamples = samples.shape[0]

        weights = self._bkd.empty((nsamples, nsamples), dtype=float)
        weights[0, 0] = 1.0
        # TODO speed up by using same concept as in lagrange polynomial
        for jj in range(1, nsamples):
            weights[jj, :jj] = (
                C_inv * (samples[:jj] - samples[jj]) * weights[jj - 1, :jj]
            )
            weights[jj, jj] = self._bkd.prod(
                C_inv * (samples[jj] - samples[:jj])
            )
            weights[jj - 1, :jj] = 1.0 / weights[jj - 1, :jj]

        weights[nsamples - 1, :nsamples] = (
            1.0 / weights[nsamples - 1, :nsamples]
        )

        if not self._bkd.all(np.isfinite(weights)):
            raise RuntimeError(
                "Samples are ill conditioned. set or change scale factor"
            )
        # todo consider storing all elements of result and updating
        # if necessary
        self._bary_weights = weights[nsamples-1]

    def _values(self, eval_samples):
        with warnings.catch_warnings():
            # ignore division by zero warning thrown when computing basis
            # at an interpolation point. It is faster to ignore then
            # only compute basis values at interpolation points
            warnings.simplefilter("ignore")
            diff = eval_samples.T-self._quad_samples
            diff_inv = 1/diff
            # factor = self._bkd.prod(diff, axis=1)
            factor = 1/self._bkd.sum(diff_inv * self._bary_weights, axis=1)
            basis_mat = factor[:, None] * (diff_inv * self._bary_weights)
            basis_mat[self._bkd.where(diff == 0)] = 1.
        return basis_mat


def setup_univariate_piecewise_polynomial_basis(
    basis_type,
    bounds,
    node_gen=None,
    trans=None,
    backend=None,
):
    basis_dict = {
        "leftconst": UnivariatePiecewiseLeftConstantBasis,
        "rightconst": UnivariatePiecewiseRightConstantBasis,
        "midconst": UnivariatePiecewiseMidPointConstantBasis,
        "linear": UnivariatePiecewiseLinearBasis,
        "quadratic": UnivariatePiecewiseQuadraticBasis,
        "cubic": UnivariatePiecewiseCubicBasis,
    }
    if basis_type not in basis_dict:
        raise ValueError(
            "basis_type {0} not supported must be in {1}".format(
                basis_type, list(basis_dict.keys())
            )
        )
    return basis_dict[basis_type](bounds, node_gen, trans, backend)


class UnivariateIntegrator(ABC):
    def __init__(self, backend):
        self._integrand = None
        if backend is None:
            backend = NumpyLinAlgMixin
        self._bkd = backend

    def set_integrand(self, integrand):
        self._integrand = integrand

    def set_options(self):
        raise NotImplementedError

    @abstractmethod
    def __call__(self):
        raise NotImplementedError

    def __repr__(self):
        return "{0}".format(self.__class__.__name__)


class ScipyUnivariateIntegrator(UnivariateIntegrator):
    def __init__(self, backend=None):
        super().__init__(backend)

        self._kwargs = {}
        self._bounds = None
        self._result = None

    def set_options(self, **kwargs):
        self._kwargs = kwargs

    def set_bounds(self, bounds):
        if len(bounds) != 2:
            raise ValueError("bounds must be an interable with two elements")
        self._bounds = bounds

    def _scipy_integrand(self, sample):
        val = self._integrand(self._bkd.atleast2d(sample))
        if val.ndim != 2:
            raise RuntimeError("integrand must return 2D array with 1 element")
        return self._bkd.to_numpy(val)[0, 0]

    def __call__(self):
        result = scipy.integrate.quad(
            self._scipy_integrand, *self._bounds, **self._kwargs
        )
        self._result = result
        return result[0]


class UnivariateUnboundedIntegrator(UnivariateIntegrator):
    """
    Compute unbounded integrals by moving left and right from origin.
    Assume that integral decays towards +/- infinity. And that once integral
    over a sub interval drops below tolerance it will not increase again if
    we keep moving in same direction.

    Warning: Do not use this if your integrand is expensive. This
    algorithm priortizes speed with vectorization over minimizing the
    number of calls to the integrand
    """

    def __init__(self, quad_rule, backend=None):
        super().__init__(backend)
        self._bounds = None
        self._inteval_size = None
        self._verbosity = None
        self._nquad_samples = None
        self._adaptive = None
        self._atol = None
        self._rtol = None
        self._maxiters = None
        self.set_options()
        if not self._bkd.bkd_equal(self._bkd, quad_rule._bkd):
            raise ValueError("quad_rule backend does not match mine")
        self._quad_rule = quad_rule

    def set_options(
        self,
        interval_size=2,
        nquad_samples=50,
        verbosity=0,
        adaptive=True,
        atol=1e-8,
        rtol=1e-8,
        maxiters=1000,
        maxinner_iters=10,
    ):
        if interval_size <= 0:
            raise ValueError("Interval size must be positive")
        self._interval_size = interval_size
        self._verbosity = verbosity
        self._nquad_samples = nquad_samples
        self._adaptive = adaptive
        self._atol = atol
        self._rtol = rtol
        self._maxiters = maxiters
        self._maxinner_iters = maxinner_iters

    def set_bounds(self, bounds):
        if len(bounds) != 2:
            raise ValueError("bounds must be an interable with two elements")
        self._bounds = bounds
        if np.isfinite(bounds[0]) and np.isfinite(bounds[1]):
            raise ValueError(
                "Do not use this integrator for bounded integrals"
            )

    def _initial_interval_bounds(self):
        lb, ub = self._bounds
        if np.isfinite(lb) and not np.isfinite(ub):
            return lb, lb + self._interval_size
        elif not np.isfinite(lb) and np.isfinite(ub):
            return ub - self._interval_size, ub
        return -self._interval_size / 2, self._interval_size / 2

    def _integrate_interval(self, lb, ub, nquad_samples):
        can_quad_x, can_quad_w = self._quad_rule(nquad_samples)
        quad_x = (can_quad_x + 1) / 2 * (ub - lb) + lb
        quad_w = can_quad_w * (ub - lb) / 2
        return self._integrand(quad_x).T @ quad_w[:, 0]

    def _adaptive_integrate_interval(self, lb, ub):
        nquad_samples = self._nquad_samples
        integral = self._integrate_interval(lb, ub, nquad_samples)
        if not self._adaptive:
            return integral
        it = 1
        while True:
            nquad_samples = (nquad_samples - 1) * 2 + 1
            prev_integral = integral
            integral = self._integrate_interval(lb, ub, nquad_samples)
            it += 1
            diff = self._bkd.abs(self._bkd.atleast1d(prev_integral - integral))
            if self._bkd.all(
                diff
                < self._rtol * self._bkd.abs(self._bkd.atleast1d(integral))
                + self._atol
            ):
                break
            if it >= self._maxinner_iters:
                break
        return integral

    def _left_integrate(self, lb, ub):
        integral = 0
        prev_integral = np.inf
        it = 0
        while (
            self._bkd.any(
                self._bkd.abs(self._bkd.atleast1d(integral - prev_integral))
                >= self._rtol
                * self._bkd.abs(self._bkd.atleast1d(prev_integral))
                + self._atol
            )
            and lb >= self._bounds[0]
            and it < self._maxiters
        ):
            result = self._adaptive_integrate_interval(lb, ub)
            if it == 0:
                prev_integral = integral
            else:
                prev_integral = self._bkd.copy(integral)
            integral += result
            ub = lb
            lb -= self._interval_size
            it += 1
        if self._verbosity > 0:
            print(
                "nleft iters={0}, error={1}".format(
                    it,
                    self._bkd.abs(
                        self._bkd.atleast1d(integral - prev_integral)
                    ),
                )
            )
        return integral

    def _right_integrate(self, lb, ub):
        integral = 0
        prev_integral = np.inf
        it = 0
        while (
            self._bkd.any(
                self._bkd.abs(self._bkd.atleast1d(integral - prev_integral))
                >= self._rtol
                * self._bkd.abs(self._bkd.atleast1d(prev_integral))
                + self._atol
            )
            and ub <= self._bounds[1]
            and it < self._maxiters
        ):
            result = self._adaptive_integrate_interval(lb, ub)
            if it == 0:
                prev_integral = integral
            else:
                prev_integral = self._bkd.copy(integral)
            integral += result
            lb = ub
            ub += self._interval_size
            it += 1

        if self._verbosity > 0:
            print(
                "nright iters={0}, error={1}".format(
                    it,
                    self._bkd.abs(
                        self._bkd.atleast1d(integral - prev_integral)
                    ),
                )
            )
        return integral

    def __call__(self):
        # compute left integral
        lb, ub = self._initial_interval_bounds()
        left_integral = self._left_integrate(lb, ub)
        right_integral = self._right_integrate(ub, ub + self._interval_size)
        integral = left_integral + right_integral
        if integral.shape[0] == 1:
            return integral[0]
        return integral
