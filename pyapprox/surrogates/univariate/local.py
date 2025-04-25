from abc import abstractmethod
from typing import Tuple

from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.util.backends.numpy import NumpyMixin

from pyapprox.surrogates.univariate.base import (
    UnivariateInterpolatingBasis,
    UnivariateEquidistantNodeGenerator,
    UnivariatePiecewisePolynomialNodeGenerator,
    UnivariateQuadratureRule,
)
from pyapprox.variables.transforms import Transform
from pyapprox.variables.marginals import Marginal


def irregular_piecewise_left_constant_basis(
    nodes: Array, xx: Array, bkd: BackendMixin = NumpyMixin
) -> Array:
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


def irregular_piecewise_right_constant_basis(
    nodes: Array, xx: Array, bkd: BackendMixin = NumpyMixin
) -> Array:
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
    nodes: Array, xx: Array, bkd: BackendMixin = NumpyMixin
) -> Array:
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


def irregular_piecewise_linear_basis(
    nodes: Array, xx: Array, bkd: BackendMixin = NumpyMixin
) -> Array:
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


def irregular_piecewise_linear_quadrature_weights(
    nodes: Array, bkd: BackendMixin = NumpyMixin
) -> Array:
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


def irregular_piecewise_quadratic_basis(
    nodes: Array, xx: Array, bkd: BackendMixin = NumpyMixin
) -> Array:
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
    nodes: Array, bkd: BackendMixin = NumpyMixin
) -> Array:
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


def irregular_piecewise_cubic_basis(
    nodes: Array, xx: Array, bkd: BackendMixin = NumpyMixin
) -> Array:
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


def irregular_piecewise_cubic_quadrature_weights(
    nodes: Array, bkd: BackendMixin = NumpyMixin
) -> Array:
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


class UnivariatePiecewisePolynomialBasis(UnivariateInterpolatingBasis):
    def __init__(
        self,
        bounds: Array,
        node_gen: UnivariatePiecewisePolynomialNodeGenerator = None,
        trans: Transform = None,
        backend: BackendMixin = NumpyMixin,
    ):
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

    def set_node_generator(
        self, node_gen: UnivariatePiecewisePolynomialNodeGenerator
    ):
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
    def _evaluate_from_nodes(self, nodes: Array) -> Array:
        raise NotImplementedError

    def _values(self, samples: Array) -> Array:
        if self._nodes is None:
            raise RuntimeError("must call set_nodes")
        idx = self._bkd.argsort(self._nodes[0])
        vals = self._evaluate_from_nodes(self._nodes[:, idx], samples)
        inverse_idx = self._bkd.empty(self._nodes.shape[1], dtype=int)
        inverse_idx = self._bkd.up(
            inverse_idx, idx, self._bkd.arange(self._nodes.shape[1])
        )
        return vals[:, inverse_idx]

    def set_bounds(self, bounds: Array):
        """Set the bounds of the quadrature rule"""
        if len(bounds) != 2:
            raise ValueError("must specifiy an upper and lower bound")
        self._bounds = bounds
        self._node_gen.set_bounds(bounds)

    @abstractmethod
    def _quadrature_rule_from_nodes(self, nodes: Array) -> Tuple[Array, Array]:
        raise NotImplementedError

    def _quadrature_rule(self) -> Tuple[Array, Array]:
        if self._quad_samples is None:
            raise RuntimeError("must call set_nterms()")
        return self._quad_samples, self._quad_weights

    def _set_nodes(self, nodes: Array):
        if nodes.ndim != 2 or nodes.shape[0] != 1:
            raise ValueError("nodes must be a 2D row vector")
        self._nodes = nodes
        self._quad_samples, self._quad_weights = (
            self._quadrature_rule_from_nodes(self._nodes)
        )

    def _active_node_indices_for_quadrature(self) -> Array:
        # used in time_integration.py
        return self._bkd.arange(self.nterms())

    def __repr__(self) -> str:
        if self._quad_samples is None:
            return "{0}(bkd={1})".format(self.__class__.__name__, self._bkd)
        return "{0}(nterms={1}, nnodes={2}, bkd={3})".format(
            self.__class__.__name__,
            self.nterms(),
            self._nodes.shape[1],
            self._bkd,
        )

    def set_nterms(self, nterms: int):
        self._set_nodes(self._node_gen(nterms))


class UnivariatePiecewiseConstantBasis(UnivariatePiecewisePolynomialBasis):
    def set_nterms(self, nterms: int):
        # need to use nterms + 1 because nterms = nnodes-1 for piecewise
        # constant basis
        self._set_nodes(self._node_gen(nterms + 1))


class UnivariatePiecewiseLeftConstantBasis(UnivariatePiecewiseConstantBasis):
    def _evaluate_from_nodes(self, nodes: Array, samples: Array) -> Array:
        return irregular_piecewise_left_constant_basis(
            nodes[0], samples[0], self._bkd
        )

    def _quadrature_rule_from_nodes(self, nodes: Array) -> Tuple[Array, Array]:
        return nodes[:, :-1], self._bkd.diff(nodes[0])[:, None]

    def nterms(self) -> int:
        return self._nodes.shape[1] - 1

    def _active_node_indices_for_quadrature(self) -> Array:
        return self._bkd.arange(self.nterms() - 1)


class UnivariatePiecewiseRightConstantBasis(UnivariatePiecewiseConstantBasis):
    def _evaluate_from_nodes(self, nodes, samples) -> Array:
        return irregular_piecewise_right_constant_basis(
            nodes[0], samples[0], self._bkd
        )

    def _quadrature_rule_from_nodes(self, nodes: Array) -> Tuple[Array, Array]:
        # unlike other higherorder methods weights.shape[0] != nodes.shape[0]
        return nodes[:, 1:], self._bkd.diff(nodes[0])[:, None]

    def nterms(self) -> int:
        return self._nodes.shape[1] - 1

    def _active_node_indices_for_quadrature(self) -> Array:
        return self._bkd.arange(1, self.nterms())


class UnivariatePiecewiseMidPointConstantBasis(
    UnivariatePiecewiseConstantBasis
):
    def _evaluate_from_nodes(self, nodes: Array, samples: Array) -> Array:
        return irregular_piecewise_midpoint_constant_basis(
            nodes[0], samples[0], self._bkd
        )

    def _quadrature_rule_from_nodes(self, nodes: Array) -> Tuple[Array, Array]:
        return (
            (nodes[:, 1:] + nodes[:, :-1]) / 2,
            self._bkd.diff(nodes[0])[:, None],
        )

    def nterms(self) -> int:
        return self._nodes.shape[1] - 1

    def _active_node_indices_for_quadrature(self) -> Array:
        raise ValueError("Quadrature points do not coincide with nodes")


class UnivariatePiecewiseLinearBasis(UnivariatePiecewisePolynomialBasis):
    def _evaluate_from_nodes(self, nodes: Array, samples: Array) -> Array:
        return irregular_piecewise_linear_basis(
            nodes[0], samples[0], self._bkd
        )

    def _quadrature_rule_from_nodes(self, nodes: Array) -> Tuple[Array, Array]:
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
    def _evaluate_from_nodes(self, nodes: Array, samples: Array) -> Array:
        return irregular_piecewise_quadratic_basis(
            nodes[0], samples[0], self._bkd
        )

    def _quadrature_rule_from_nodes(self, nodes: Array) -> Tuple[Array, Array]:
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
    def _evaluate_from_nodes(self, nodes: Array, samples: Array) -> Array:
        return irregular_piecewise_cubic_basis(nodes[0], samples[0], self._bkd)

    def _quadrature_rule_from_nodes(self, nodes: Array) -> Tuple[Array, Array]:
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


class UnivariatePiecewisePolynomialQuadratureRule(UnivariateQuadratureRule):
    def __init__(
        self,
        basis_type: str,
        bounds: Array = None,
        node_gen: UnivariatePiecewisePolynomialNodeGenerator = None,
        backend: BackendMixin = NumpyMixin,
        store: bool = False,
        marginal: Marginal = None,
    ):
        super().__init__(backend=backend, store=store)
        self._basis = setup_univariate_piecewise_polynomial_basis(
            basis_type, bounds, node_gen, None, backend
        )
        self._marginal = marginal

    def _quad_rule(self, nnodes: int):
        self._basis.set_nterms(nnodes)
        quadx, quadw = self._basis.quadrature_rule()
        if self._marginal is not None:
            quadw *= self._marginal.pdf(quadx[0])[:, None]
        return quadx, quadw


def setup_univariate_piecewise_polynomial_basis(
    basis_type: str,
    bounds: Array,
    node_gen: UnivariatePiecewisePolynomialNodeGenerator = None,
    trans: Transform = None,
    backend: BackendMixin = NumpyMixin,
) -> UnivariatePiecewisePolynomialBasis:
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
    return basis_dict[basis_type](bounds, node_gen, trans, backend=backend)
