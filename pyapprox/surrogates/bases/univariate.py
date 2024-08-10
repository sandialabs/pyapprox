from abc import ABC, abstractmethod

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin


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
        if weights.ndim != 2 or weights.shape[1] != 1:
            raise ValueError("weights must be a 2D column vector")
        self._check_samples(samples)
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
        self._nodes = None

    def set_nodes(self, nodes):
        if nodes.ndim != 2 or nodes.shape[0] != 1:
            print(nodes)
            raise ValueError("nodes must be a 2D row vector")
        self._nodes = nodes

    @abstractmethod
    def _evaluate_from_nodes(self, nodes):
        raise NotImplementedError

    @abstractmethod
    def _quadrature_rule_from_nodes(self, nodes):
        raise NotImplementedError

    def _values(self, samples):
        if self._nodes is None:
            raise RuntimeError("must call set_nodes")
        return self._evaluate_from_nodes(self._nodes, samples)

    def _quadrature_rule(self):
        return self._quadrature_rule_from_nodes(self._nodes)

    def nterms(self):
        return self._nodes.shape[1]

    def __repr__(self):
        if self._nodes is None:
            return "{0}(bkd={1})".format(self.__class__.__name__, self._bkd)
        return "{0}(nnodes={1}, bkd={2})".format(
            self.__class__.__name__, self.nterms(), self._bkd
        )

    def _active_node_indices_for_quadrature(self):
        return self._bkd._la_arange(self.nterms())


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
    values = bkd._la_ones((samples.shape[0], nabscissa))
    # idx is needed for bkd_la_up until I can find a better way to pass in :
    idx = bkd._la_arange(samples.shape[0], dtype=int)
    for ii in range(nabscissa):
        x_ii = abscissa[ii]
        for jj in range(nabscissa):
            if ii == jj:
                continue
            # values[:, ii] *= (samples - abscissa[jj])/(x_ii-abscissa[jj])
            values = bkd._la_up(
                values,
                (idx, ii),
                values[:, ii]
                * (samples - abscissa[jj])
                / (x_ii - abscissa[jj]),
            )
    return values


class UnivariatePiecewiseLeftConstantBasis(UnivariateInterpolatingBasis):
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


class UnivariatePiecewiseRightConstantBasis(UnivariateInterpolatingBasis):
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


class UnivariatePiecewiseMidPointConstantBasis(UnivariateInterpolatingBasis):
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


class UnivariatePiecewiseLinearBasis(UnivariateInterpolatingBasis):
    def _evaluate_from_nodes(self, nodes, samples):
        return irregular_piecewise_linear_basis(
            nodes[0], samples[0], self._bkd
        )

    def _quadrature_rule_from_nodes(self, nodes):
        return (
            nodes,
            irregular_piecewise_linear_quadrature_weights(nodes[0], self._bkd)[
                :, None
            ],
        )


class UnivariatePiecewiseQuadraticBasis(UnivariateInterpolatingBasis):
    def _evaluate_from_nodes(self, nodes, samples):
        return irregular_piecewise_quadratic_basis(
            nodes[0], samples[0], self._bkd
        )

    def _quadrature_rule_from_nodes(self, nodes):
        return (
            nodes,
            irregular_piecewise_quadratic_quadrature_weights(
                nodes[0], self._bkd
            )[:, None],
        )


class UnivariatePiecewiseCubicBasis(UnivariateInterpolatingBasis):
    def _evaluate_from_nodes(self, nodes, samples):
        return irregular_piecewise_cubic_basis(nodes[0], samples[0], self._bkd)

    def _quadrature_rule_from_nodes(self, nodes):
        return (
            nodes,
            irregular_piecewise_cubic_quadrature_weights(nodes[0], self._bkd)[
                :, None
            ],
        )


class UnivariateLagrangeBasis(UnivariateInterpolatingBasis):
    def __init__(self, backend=NumpyLinAlgMixin()):
        super().__init__(backend)
        self._bounds = None

    def _evaluate_from_nodes(self, nodes, samples):
        return univariate_lagrange_polynomial(nodes[0], samples[0], self._bkd)

    def set_bounds(self, bounds):
        """Set the bounds of the quadrature rule"""
        if len(bounds) != 2:
            raise ValueError("must specifiy an upper and lower bound")
        self._bounds = bounds

    def _quadrature_rule_from_nodes(self, nodes):
        """Compute quadrature assuming a constant weight over self._bounds"""
        if self._bounds is None:
            raise ValueError("must call set_bounds")
        # return Gauss-Legendre quadrature rule for weight function 1 on [-1,1]
        # todo allowing user to pass in quadrature rule.
        import numpy as np
        xx, ww = np.polynomial.legendre.leggauss(nodes.shape[1])
        # scale to user domain
        scale = (self._bounds[1]-self._bounds[0])
        xx = (xx+1)/2*scale+self._bounds[0]
        ww = ww/2*scale
        # now integrate lagrange basis functions
        vals = self(self._bkd._la_asarray(xx)[None, :])
        weights = (vals.T @ ww)
        return (
            nodes,
            self._bkd._la_asarray(weights)[:, None],
        )


def get_univariate_interpolation_basis(basis_type, backend=NumpyLinAlgMixin()):
    basis_dict = {
        "linear": UnivariatePiecewiseLinearBasis,
        "quadratic": UnivariatePiecewiseQuadraticBasis,
        "cubic": UnivariatePiecewiseCubicBasis,
        "lagrange": UnivariateLagrangeBasis,
    }
    if basis_type not in basis_dict:
        raise ValueError(
            "basis_type {0} not supported must be in {1}".format(
                basis_type, list(basis_dict.keys())
            )
        )
    return basis_dict[basis_type](backend)
