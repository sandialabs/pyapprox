"""Tests for PiecewiseMeasureQuadratureRule.

The oracle for each weight is ``int phi_i p`` computed by adaptive
``scipy.integrate.quad`` on every interval between nodes, independent of
the rule's Gauss-Legendre computation.
"""

from typing import Callable, List, Tuple

import numpy as np
import pytest
from scipy import integrate, stats

from pyapprox.probability.protocols.distribution import MarginalProtocol
from pyapprox.probability.univariate.beta import BetaMarginal
from pyapprox.probability.univariate.gaussian import GaussianMarginal
from pyapprox.probability.univariate.scipy_continuous import (
    ScipyContinuousMarginal,
)
from pyapprox.probability.univariate.uniform import UniformMarginal
from pyapprox.surrogates.affine.univariate.piecewisepoly import (
    EquidistantNodeGenerator,
    PiecewiseCubic,
    PiecewiseLinear,
    PiecewiseMeasureQuadratureRule,
    PiecewisePolynomialProtocol,
    PiecewiseQuadratic,
)
from pyapprox.util.backends.protocols import Array, Backend

_BasisClass = Callable[[Array, Backend[Array]], PiecewisePolynomialProtocol[Array]]

# Basis classes with node counts each accepts.
_BASES_AND_COUNTS = [
    (PiecewiseLinear, [5, 9]),
    (PiecewiseQuadratic, [5, 9]),
    (PiecewiseCubic, [4, 7]),
]


def _oracle_weights(
    basis_class: _BasisClass[Array],
    npoints: int,
    bounds: Tuple[float, float],
    pdf: Callable[[float], float],
    bkd: Backend[Array],
) -> Array:
    """``int phi_i p`` by adaptive quadrature, normalized. Shape: (npoints, 1)"""
    nodes = np.linspace(bounds[0], bounds[1], npoints)
    basis = basis_class(bkd.asarray(nodes), bkd)
    weights = np.zeros(npoints)
    for left, right in zip(nodes[:-1], nodes[1:]):
        for ii in range(npoints):
            weights[ii] += integrate.quad(
                lambda x, ii=ii: bkd.to_float(basis(bkd.asarray([x]))[0, ii]) * pdf(x),
                left,
                right,
                epsabs=1e-15,
                epsrel=1e-14,
            )[0]
    return bkd.asarray((weights / weights.sum())[:, None])


def _rule(
    marginal: MarginalProtocol[Array],
    basis_class: _BasisClass[Array],
    bounds: Tuple[float, float],
    bkd: Backend[Array],
) -> PiecewiseMeasureQuadratureRule[Array]:
    return PiecewiseMeasureQuadratureRule(
        marginal, basis_class, EquidistantNodeGenerator(bkd, bounds)
    )


def _integrand(x: float) -> float:
    return float(np.cos(2.0 * x) + np.exp(0.3 * x))


class TestExactDensities:
    """A density polynomial on each interval gives weights exact to rounding."""

    @pytest.mark.parametrize("basis_class, counts", _BASES_AND_COUNTS)
    def test_uniform_is_lebesgue_over_width(
        self, bkd: Backend[Array], basis_class: _BasisClass[Array], counts: List[int]
    ) -> None:
        rule = _rule(UniformMarginal(-1.0, 3.0, bkd), basis_class, (-1.0, 3.0), bkd)
        for npoints in counts:
            points, weights = rule(npoints)
            nodes = bkd.linspace(-1.0, 3.0, npoints)
            _, lebesgue = basis_class(nodes, bkd).quadrature_rule()
            bkd.assert_allclose(points, bkd.reshape(nodes, (1, -1)))
            bkd.assert_allclose(
                weights, bkd.reshape(lebesgue, (-1, 1)) / 4.0, rtol=1e-13
            )

    @pytest.mark.parametrize("basis_class, counts", _BASES_AND_COUNTS)
    def test_integer_beta(
        self, bkd: Backend[Array], basis_class: _BasisClass[Array], counts: List[int]
    ) -> None:
        rule = _rule(BetaMarginal(2.0, 5.0, bkd), basis_class, (0.0, 1.0), bkd)
        beta = stats.beta(2.0, 5.0)
        for npoints in counts:
            expected = _oracle_weights(
                basis_class, npoints, (0.0, 1.0), lambda x: float(beta.pdf(x)), bkd
            )
            # A weight that is zero in theory is only zero to rounding.
            bkd.assert_allclose(rule(npoints)[1], expected, rtol=1e-12, atol=1e-15)


class TestGaussian:
    """A Gaussian density is not polynomial: the weights converge."""

    @pytest.mark.parametrize("basis_class, counts", _BASES_AND_COUNTS)
    def test_weights_match_oracle(
        self, bkd: Backend[Array], basis_class: _BasisClass[Array], counts: List[int]
    ) -> None:
        rule = _rule(GaussianMarginal(0.0, 1.0, bkd), basis_class, (-5.0, 5.0), bkd)
        for npoints in counts:
            expected = _oracle_weights(
                basis_class,
                npoints,
                (-5.0, 5.0),
                lambda x: float(stats.norm.pdf(x)),
                bkd,
            )
            bkd.assert_allclose(rule(npoints)[1], expected, atol=1e-11)

    @pytest.mark.parametrize(
        "basis_class, npoints, degree",
        [(PiecewiseLinear, 9, 1), (PiecewiseQuadratic, 9, 2), (PiecewiseCubic, 7, 3)],
    )
    def test_exact_for_polynomials_under_the_measure(
        self,
        bkd: Backend[Array],
        basis_class: _BasisClass[Array],
        npoints: int,
        degree: int,
    ) -> None:
        """The interpolant of a degree-``d`` polynomial is the polynomial,
        so the rule returns its truncated-normal moments."""
        rule = _rule(GaussianMarginal(0.0, 1.0, bkd), basis_class, (-5.0, 5.0), bkd)
        truncated = stats.truncnorm(-5.0, 5.0)
        points, weights = rule(npoints)
        moments = [
            bkd.sum(points[0] ** power * weights[:, 0]) for power in range(degree + 1)
        ]
        expected = [float(truncated.moment(power)) for power in range(degree + 1)]
        bkd.assert_allclose(bkd.stack(moments), bkd.asarray(expected), atol=1e-10)

    @pytest.mark.parametrize(
        "basis_class, counts, order",
        [
            (PiecewiseLinear, [33, 65, 129], 2.0),
            # Simpson's superconvergence survives the density weighting.
            (PiecewiseQuadratic, [33, 65, 129], 4.0),
            (PiecewiseCubic, [73, 145, 289], 4.0),
        ],
    )
    def test_expectation_converges_at_the_basis_order(
        self,
        bkd: Backend[Array],
        basis_class: _BasisClass[Array],
        counts: List[int],
        order: float,
    ) -> None:
        """For a smooth integrand the rule's error in ``E[f]`` falls like
        ``h^order``; the weights' own error is far below it."""
        rule = _rule(GaussianMarginal(0.0, 1.0, bkd), basis_class, (-5.0, 5.0), bkd)
        mass = stats.norm.cdf(5.0) - stats.norm.cdf(-5.0)
        exact = (
            integrate.quad(
                lambda x: _integrand(x) * float(stats.norm.pdf(x)),
                -5.0,
                5.0,
                epsabs=1e-15,
                epsrel=1e-15,
                limit=200,
            )[0]
            / mass
        )
        errors = []
        for npoints in counts:
            points, weights = rule(npoints)
            values = [_integrand(bkd.to_float(x)) for x in points[0]]
            estimate = bkd.to_float(bkd.sum(bkd.asarray(values) * weights[:, 0]))
            errors.append(abs(estimate - exact))
        spacing = 10.0 / (np.array(counts) - 1.0)
        slope = float(np.polyfit(np.log(spacing), np.log(errors), 1)[0])
        assert abs(slope - order) < 0.1, (slope, errors)


class TestRuntimeChecks:
    def test_kink_inside_an_interval_raises(self, bkd: Backend[Array]) -> None:
        """A triangular density peaked at 0.3 has a kink between nodes."""
        marginal = ScipyContinuousMarginal(stats.triang(0.65, loc=-1.0, scale=2.0), bkd)
        rule = _rule(marginal, PiecewiseLinear, (-1.0, 1.0), bkd)
        with pytest.raises(ValueError, match="kink"):
            rule(5)

    def test_kink_at_a_node_is_exact(self, bkd: Backend[Array]) -> None:
        """The same density peaked at 0, a node: polynomial on each interval."""
        triangular = stats.triang(0.5, loc=-1.0, scale=2.0)
        rule = _rule(
            ScipyContinuousMarginal(triangular, bkd), PiecewiseLinear, (-1.0, 1.0), bkd
        )
        expected = _oracle_weights(
            PiecewiseLinear, 5, (-1.0, 1.0), lambda x: float(triangular.pdf(x)), bkd
        )
        bkd.assert_allclose(rule(5)[1], expected, rtol=1e-12)

    def test_density_and_cdf_disagreeing_raises(self, bkd: Backend[Array]) -> None:
        class _HalfCDF(GaussianMarginal[Array]):
            def cdf(self, samples: Array) -> Array:
                return 0.5 * super().cdf(samples)

        rule = _rule(_HalfCDF(0.0, 1.0, bkd), PiecewiseLinear, (-5.0, 5.0), bkd)
        with pytest.raises(ValueError, match="CDF"):
            rule(9)


class TestInterface:
    def test_one_point_is_the_midpoint(self, bkd: Backend[Array]) -> None:
        rule = _rule(UniformMarginal(-1.0, 3.0, bkd), PiecewiseLinear, (-1.0, 3.0), bkd)
        points, weights = rule(1)
        bkd.assert_allclose(points, bkd.asarray([[1.0]]))
        bkd.assert_allclose(weights, bkd.asarray([[1.0]]))

    def test_store_caches_by_node_count(self, bkd: Backend[Array]) -> None:
        marginal = GaussianMarginal(0.0, 1.0, bkd)
        generator = EquidistantNodeGenerator(bkd, (-5.0, 5.0))
        stored = PiecewiseMeasureQuadratureRule(marginal, PiecewiseLinear, generator)
        assert stored(9) is stored(9)
        fresh = PiecewiseMeasureQuadratureRule(
            marginal, PiecewiseLinear, generator, store=False
        )
        assert fresh(9) is not fresh(9)
        bkd.assert_allclose(fresh(9)[1], stored(9)[1])

    def test_rejects_bad_arguments(self, bkd: Backend[Array]) -> None:
        marginal = GaussianMarginal(0.0, 1.0, bkd)
        generator = EquidistantNodeGenerator(bkd, (-5.0, 5.0))
        with pytest.raises(ValueError, match="rtol"):
            PiecewiseMeasureQuadratureRule(
                marginal, PiecewiseLinear, generator, rtol=0.0
            )
        with pytest.raises(ValueError, match="points_per_interval"):
            PiecewiseMeasureQuadratureRule(
                marginal, PiecewiseLinear, generator, points_per_interval=70
            )
        with pytest.raises(ValueError, match="npoints"):
            PiecewiseMeasureQuadratureRule(marginal, PiecewiseLinear, generator)(0)
