from abc import ABC, abstractmethod
import math

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.surrogates.orthopoly.quadrature import (
    clenshaw_curtis_in_polynomial_order
)


class UnivariateBasis(ABC):
    def __init__(self, backend):
        if backend is None:
            backend = NumpyLinAlgMixin()
        self._bkd = backend

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

    def __init__(self, nterms=None, backend=NumpyLinAlgMixin()):
        super().__init__(backend)
        self._nterms = None
        if nterms is not None:
            self.set_nterms(nterms)

    def set_nterms(self, nterms):
        self._nterms = nterms

    def nterms(self):
        return self._nterms

    def _values(self, samples):
        basis_matrix = samples.T ** self._bkd._la_arange(self._nterms)[None, :]
        return basis_matrix

    def _derivatives(self, samples, order):
        powers = self._bkd._la_hstack(
            (
                self._bkd._la_zeros((order,)),
                self._bkd._la_arange(self._nterms - order),
            )
        )
        # 1 x x^2 x^3  x^4 vals
        # 0 1 2x  3x^2 4x^3 1st derivs
        # 0 0 2   6x   12x^2  2nd derivs
        consts = self._bkd._la_hstack(
            (
                self._bkd._la_zeros((order,)),
                order * self._bkd._la_arange(1, self._nterms - order + 1),
            )
        )
        return (samples.T ** powers[None, :]) * consts


class UnivariateInterpolatingBasis(UnivariateBasis):
    def __init__(self, backend=NumpyLinAlgMixin()):
        super().__init__(backend)
        self._quad_samples = None
        self._quad_weights = None

    def nterms(self):
        return self._quad_samples.shape[1]

    def __repr__(self):
        if self._quad_samples is None:
            return "{0}(bkd={1})".format(self.__class__.__name__, self._bkd)
        return "{0}(nterms={1}, bkd={2})".format(
            self.__class__.__name__, self.nterms(), self._bkd
        )


def irregular_piecewise_left_constant_basis(nodes, xx, bkd=NumpyLinAlgMixin()):
    # abscissa are not equidistant
    assert xx.ndim == 1
    assert nodes.ndim == 1
    nnodes = nodes.shape[0]
    if nnodes == 1:
        return bkd._la_full((xx.shape[0], nnodes), 1.0)
    vals = bkd._la_zeros((xx.shape[0], nnodes - 1))
    for ii in range(nnodes - 1):
        xl = nodes[ii]
        xr = nodes[ii + 1]
        II = bkd._la_where((xx >= xl) & (xx < xr))[0]
        vals[II, ii] = bkd._la_full((II.shape[0],), 1.0, dtype=float)
    return vals


def irregular_piecewise_right_constant_basis(
    nodes, xx, bkd=NumpyLinAlgMixin()
):
    # abscissa are not equidistant
    assert xx.ndim == 1
    assert nodes.ndim == 1
    nnodes = nodes.shape[0]
    if nnodes == 1:
        return bkd._la_ones((xx.shape[0], nnodes))
    vals = bkd._la_zeros((xx.shape[0], nnodes - 1))
    for ii in range(1, nnodes):
        xr = nodes[ii]
        xl = nodes[ii - 1]
        II = bkd._la_where((xx > xl) & (xx <= xr))[0]
        vals[II, ii] = bkd._la_ones((II.shape[0],), dtype=float)
    return vals


def irregular_piecewise_midpoint_constant_basis(
    nodes, xx, bkd=NumpyLinAlgMixin()
):
    # abscissa are not equidistant
    assert xx.ndim == 1
    assert nodes.ndim == 1
    nnodes = nodes.shape[0]
    if nnodes == 1:
        return bkd._la_ones((xx.shape[0], nnodes))
    vals = bkd._la_zeros((xx.shape[0], nnodes - 1))
    for ii in range(nnodes - 1):
        xl = nodes[ii]
        xr = nodes[ii + 1]
        if ii < nnodes - 1:
            II = bkd._la_where((xx >= xl) & (xx < xr))[0]
        else:
            II = bkd._la_where((xx >= xl) & (xx <= xr))[0]
        vals[II, ii] = bkd._la_ones((II.shape[0],), dtype=float)
    return vals


def irregular_piecewise_linear_basis(nodes, xx, bkd=NumpyLinAlgMixin()):
    # abscissa are not equidistant
    assert xx.ndim == 1
    assert nodes.ndim == 1
    nnodes = nodes.shape[0]
    if nnodes == 1:
        return bkd._la_ones((xx.shape[0], nnodes))
    vals = bkd._la_zeros((xx.shape[0], nnodes))
    for ii in range(nnodes):
        xm = nodes[ii]
        if ii > 0:
            xl = nodes[ii - 1]
            II = bkd._la_where((xx >= xl) & (xx <= xm))[0]
            # vals[II, ii] = (xx[II] - xl) / (xm - xl)
            vals = bkd._la_up(vals, (II, ii), (xx[II] - xl) / (xm - xl))
        if ii < nnodes - 1:
            xr = nodes[ii + 1]
            JJ = bkd._la_where((xx >= xm) & (xx <= xr))[0]
            # vals[JJ, ii] = (xr - xx[JJ]) / (xr - xm)
            vals = bkd._la_up(vals, (JJ, ii), (xr - xx[JJ]) / (xr - xm))
    return vals


def irregular_piecewise_linear_quadrature_weights(
    nodes, bkd=NumpyLinAlgMixin()
):
    assert nodes.ndim == 1
    nnodes = nodes.shape[0]
    if nnodes == 1:
        raise ValueError(
            "Cant compute weights from single point without bounds"
        )
    # weights = bkd._la_zeros((nnodes,))
    # use list so not to have to use bkd._la_up below. Since we are using
    # python loop here speed is not going to be much different
    weights = [0.0 for ii in range(nnodes)]
    for ii in range(nnodes):
        xm = nodes[ii]
        if ii > 0:
            xl = nodes[ii - 1]
            weights[ii] += 0.5 * (xm - xl)
            # weights = bkd._la_up(weights, ii, weights[ii] + 0.5 * (xm - xl))
        if ii < nnodes - 1:
            xr = nodes[ii + 1]
            weights[ii] += 0.5 * (xr - xm)
            # weights = bkd._la_up(weights, ii, weights[ii] + 0.5 * (xr - xm))
    return bkd._la_asarray(weights)


def irregular_piecewise_quadratic_basis(nodes, xx, bkd=NumpyLinAlgMixin()):
    # nodes are not equidistant
    assert xx.ndim == 1
    nnodes = nodes.shape[0]
    if nnodes == 1:
        return bkd._la_ones((xx.shape[0], nnodes))
    if nodes.ndim != 1 or nnodes % 2 != 1:
        raise ValueError("nodes has the wrong shape")
    vals = bkd._la_zeros((xx.shape[0], nnodes))
    for ii in range(nnodes):
        if ii % 2 == 1:
            xl, xm, xr = nodes[ii - 1 : ii + 2]
            II = bkd._la_where((xx >= xl) & (xx <= xr))[0]
            # vals[II, ii] = (
            #     (xx[II] - xl) / (xm - xl) * (xx[II] - xr) / (xm - xr)
            # )
            vals = bkd._la_up(
                vals,
                (II, ii),
                (xx[II] - xl) / (xm - xl) * (xx[II] - xr) / (xm - xr),
            )
            continue
        if ii < nnodes - 2:
            xl, xm, xr = nodes[ii : ii + 3]
            II = bkd._la_where((xx >= xl) & (xx <= xr))[0]
            # vals[II, ii] = (
            #     (xx[II] - xm) / (xl - xm) * (xx[II] - xr) / (xl - xr)
            # )
            vals = bkd._la_up(
                vals,
                (II, ii),
                (xx[II] - xm) / (xl - xm) * (xx[II] - xr) / (xl - xr),
            )
        if ii > 1:
            xl, xm, xr = nodes[ii - 2 : ii + 1]
            II = bkd._la_where((xx >= xl) & (xx <= xr))[0]
            # vals[II, ii] = (
            #     (xx[II] - xl) / (xr - xl) * (xx[II] - xm) / (xr - xm)
            # )
            vals = bkd._la_up(
                vals,
                (II, ii),
                (xx[II] - xl) / (xr - xl) * (xx[II] - xm) / (xr - xm),
            )
    return vals


def irregular_piecewise_quadratic_quadrature_weights(
    nodes, bkd=NumpyLinAlgMixin()
):
    # nodes are not equidistant
    nnodes = nodes.shape[0]
    if nnodes == 1:
        raise ValueError(
            "Cant compute weights from single point without bounds"
        )
    if nodes.ndim != 1 or nnodes % 2 != 1:
        raise ValueError("nodes has the wrong shape, it must be an odd number")
    # weights = bkd._la_zeros((nnodes,))
    weights = [0.0 for ii in range(nnodes)]
    for ii in range(nnodes):
        if ii % 2 == 1:
            xl, xm, xr = nodes[ii - 1 : ii + 2]
            weights[ii] = (xl - xr) ** 3 / (6 * (xm - xl) * (xm - xr))
            # weights = bkd._la_up(weights, ii, (xl - xr) ** 3 / (6 * (xm - xl) * (xm - xr)))
            continue
        if ii < nnodes - 2:
            xl, xm, xr = nodes[ii : ii + 3]
            weights[ii] += ((xr - xl) * (2 * xl - 3 * xm + xr)) / (
                6 * (xl - xm)
            )
            # weights = bkd._la_up(weights, ii, weights[ii]+((xr - xl) * (2 * xl - 3 * xm + xr)) / (
            #     6 * (xl - xm)))
        if ii > 1:
            xl, xm, xr = nodes[ii - 2 : ii + 1]
            weights[ii] += ((xl - xr) * (xl - 3 * xm + 2 * xr)) / (
                6 * (xm - xr)
            )
            # weights = bkd._la_up(weights, ii, weights[ii]+((xl - xr) * (xl - 3 * xm + 2 * xr)) / (
            #          6 * (xm - xr)))
    return bkd._la_asarray(weights)


def irregular_piecewise_cubic_basis(nodes, xx, bkd=NumpyLinAlgMixin()):
    # nodes are not equidistant
    assert xx.ndim == 1
    nnodes = nodes.shape[0]
    if nnodes == 1:
        return bkd._la_ones((xx.shape[0], nnodes))
    if nodes.ndim != 1 or nnodes < 4 or (nnodes - 4) % 3 != 0:
        raise ValueError("nodes has the wrong shape")
    vals = bkd._la_zeros((xx.shape[0], nnodes))
    for ii in range(nnodes):
        if ii % 3 == 1:
            x1, x2, x3, x4 = nodes[ii - 1 : ii + 3]
            II = bkd._la_where((xx >= x1) & (xx <= x4))[0]
            # vals[II, ii] = (
            #     (xx[II] - x1)
            #     / (x2 - x1)
            #     * (xx[II] - x3)
            #     / (x2 - x3)
            #     * (xx[II] - x4)
            #     / (x2 - x4)
            # )
            vals = bkd._la_up(
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
            II = bkd._la_where((xx >= x1) & (xx <= x4))[0]
            # vals[II, ii] = (
            #     (xx[II] - x1)
            #     / (x3 - x1)
            #     * (xx[II] - x2)
            #     / (x3 - x2)
            #     * (xx[II] - x4)
            #     / (x3 - x4)
            # )
            vals = bkd._la_up(
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
            II = bkd._la_where((xx >= x1) & (xx <= x4))[0]
            # vals[II, ii] = (
            #     (xx[II] - x2)
            #     / (x1 - x2)
            #     * (xx[II] - x3)
            #     / (x1 - x3)
            #     * (xx[II] - x4)
            #     / (x1 - x4)
            # )
            vals = bkd._la_up(
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
            II = bkd._la_where((xx >= x1) & (xx <= x4))[0]
            # vals[II, ii] = (
            #     (xx[II] - x1)
            #     / (x4 - x1)
            #     * (xx[II] - x2)
            #     / (x4 - x2)
            #     * (xx[II] - x3)
            #     / (x4 - x3)
            # )
            vals = bkd._la_up(
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


def irregular_piecewise_cubic_quadrature_weights(
    nodes, bkd=NumpyLinAlgMixin()
):
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
    # weights = bkd._la_zeros((nnodes,))
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
    return bkd._la_asarray(weights)


def univariate_lagrange_polynomial(abscissa, samples, bkd=NumpyLinAlgMixin()):
    assert abscissa.ndim == 1
    assert samples.ndim == 1
    nabscissa = abscissa.shape[0]
    denoms = abscissa[:, None]-abscissa[None, :]
    numers = samples[:, None]-abscissa[None, :]
    # values = bkd._la_empty((samples.shape[0], nabscissa))
    values = []
    for ii in range(nabscissa):
        # l_j(x) = prod_{i!=j} (x-x_i)/(x_j-x_i)
        denom = bkd._la_prod(denoms[ii, :ii])*bkd._la_prod(denoms[ii, ii+1:])
        # denom = bkd._la_prod(bkd._la_delete(denoms[ii], ii))
        # numer = bkd._la_prod(bkd._la_delete(numers, ii, axis=1), axis=1)
        numer = bkd._la_prod(numers[:, :ii], axis=1)*bkd._la_prod(numers[:, ii+1:], axis=1)
        # values[:, ii] = numer/denom
        values.append(numer/denom)
    return bkd._la_stack(values, axis=1)
    # return values


class UnivariatePiecewisePolynomialBasis(UnivariateInterpolatingBasis):
    def __init__(self, bounds, backend=NumpyLinAlgMixin()):
        super().__init__(backend)
        # nodes may be different than quad_samples
        # e.g. when using piecwise constant basis
        self._nodes = None
        self._bounds = None
        self.set_bounds(bounds)

    @abstractmethod
    def _evaluate_from_nodes(self, nodes):
        raise NotImplementedError

    def _values(self, samples):
        if self._nodes is None:
            raise RuntimeError("must call set_nodes")
        return self._evaluate_from_nodes(self._nodes, samples)

    def set_bounds(self, bounds):
        """Set the bounds of the quadrature rule"""
        if len(bounds) != 2:
            raise ValueError("must specifiy an upper and lower bound")
        self._bounds = bounds

    @abstractmethod
    def _quadrature_rule_from_nodes(self, nodes):
        raise NotImplementedError

    def _quadrature_rule(self):
        return self._quad_samples, self._quad_weights

    def set_nodes(self, nodes):
        if nodes.ndim != 2 or nodes.shape[0] != 1:
            raise ValueError("nodes must be a 2D row vector")
        self._nodes = nodes
        self._quad_samples, self._quad_weights = (
            self._quadrature_rule_from_nodes(self._nodes)
        )

    def _active_node_indices_for_quadrature(self):
        # used in time_integration.py
        return self._bkd._la_arange(self.nterms())

    def __repr__(self):
        if self._quad_samples is None:
            return "{0}(bkd={1})".format(self.__class__.__name__, self._bkd)
        return "{0}(nterms={1}, nnodes={2}, bkd={3})".format(
            self.__class__.__name__, self.nterms(), self._nodes.shape[1],
            self._bkd
        )

    def set_nterms(self, nterms):
        """Set equidistant nodes"""
        self.set_nodes(self._bkd._la_linspace(*self._bounds, nterms)[None, :])


class UnivariatePiecewiseConstantBasis(UnivariateInterpolatingBasis):
    def set_nterms(self, nterms):
        # need to use nterms + 1 because nterms = nnodes-1 for piecewise
        # constant basis
        self.set_nodes(
            self._bkd._la_linspace(*self._bounds, nterms+1)[None, :]
        )


class UnivariatePiecewiseLeftConstantBasis(UnivariatePiecewiseConstantBasis):
    def _evaluate_from_nodes(self, nodes, samples):
        return irregular_piecewise_left_constant_basis(
            nodes[0], samples[0], self._bkd
        )

    def _quadrature_rule_from_nodes(self, nodes):
        return nodes[:, :-1], self._bkd._la_diff(nodes[0])[:, None]

    def nterms(self):
        return self._nodes.shape[1] - 1

    def _active_node_indices_for_quadrature(self):
        return self._bkd._la_arange(self.nterms() - 1)


class UnivariatePiecewiseRightConstantBasis(UnivariatePiecewiseConstantBasis):
    def _evaluate_from_nodes(self, nodes, samples):
        return irregular_piecewise_right_constant_basis(
            nodes[0], samples[0], self._bkd
        )

    def _quadrature_rule_from_nodes(self, nodes):
        # unlike other higherorder methods weights.shape[0] != nodes.shape[0]
        return nodes[:, 1:], self._bkd._la_diff(nodes[0])[:, None]

    def nterms(self):
        return self._nodes.shape[1] - 1

    def _active_node_indices_for_quadrature(self):
        return self._bkd._la_arange(1, self.nterms())


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
            self._bkd._la_diff(nodes[0])[:, None],
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
            return self._nodes, self._bounds[1]-self._bounds[0]
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
            return self._nodes, self._bounds[1]-self._bounds[0]
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
            return self._nodes, self._bounds[1]-self._bounds[0]
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
            backend = NumpyLinAlgMixin()
        self._bkd = backend
        self._store = store
        self._quad_samples = dict()
        self._quad_weights = dict

    @abstractmethod
    def _quad_rule(self, nnodes):
        raise NotImplementedError

    def __call__(self, nnodes):
        if self._store and nnodes in self._quad_samples:
            return self._quad_samples[nnodes], self._quad_weights[nnodes]
        quad_samples, quad_weights = self._quad_rule(nnodes)
        if self._store:
            self._quad_samples[nnodes] = quad_samples
            self._quad_weights[nnodes] = quad_weights
        return quad_samples, quad_weights

    def __repr__(self):
        return "{0}(bkd={1})".format(self.__class__.__name__, self._bkd)


def _is_power_of_two(integer):
    return (integer & (integer-1) == 0) and integer != 0


class ClenshawCurtisQuadratureRule(UnivariateQuadratureRule):
    """Integrates functions on [-1, 1] with weight function 1/2
    """
    def _quad_rule(self, nnodes):
        # rule requires nnodes = 2**l + 1 for l=1,2,3
        # so check n=nnodes-1 is a power of 2
        if nnodes == 1:
            level = 0
        else:
            if not _is_power_of_two(nnodes-1):
                raise ValueError("nnodes-1 must be a power of 2")
            level = int(round(math.log(nnodes-1, 2), 0))
        quad_samples, quad_weights = clenshaw_curtis_in_polynomial_order(
            level, False
        )
        return (
            self._bkd._la_asarray(quad_samples)[None, :],
            self._bkd._la_asarray(quad_weights)[:, None]
        )


class UnivariateLagrangeBasis(UnivariateInterpolatingBasis):
    def __init__(self, quadrature_rule, nterms, backend=NumpyLinAlgMixin()):
        super().__init__(backend)
        self._quad_rule = quadrature_rule
        self.set_nterms(nterms)

    def set_nterms(self, nterms):
        self._quad_samples, self._quad_weights = self._quad_rule(nterms)

    def _values(self, samples):
        return univariate_lagrange_polynomial(
            self._quad_samples, samples[0], self._bkd)

    def _quadrature_rule(self):
        return self._quad_samples, self._quad_weights


def setup_univariate_piecewise_polynomial_basis(
        basis_type, bounds, backend=NumpyLinAlgMixin()
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
    return basis_dict[basis_type](bounds, backend)
