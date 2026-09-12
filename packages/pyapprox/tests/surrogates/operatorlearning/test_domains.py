r"""Tests for the spaces a fixed-basis surrogate is built over.

The property that matters is the *metric*, not the geometry: a domain's
job is to say what a best approximation is, and getting that wrong makes
a POD basis silently optimal in the wrong norm. So the tests here check
quadrature against integrals with known values, and check that
grid-boundedness is visible through the protocols rather than through a
flag.
"""

import numpy as np
import pytest
from pyapprox.surrogates.operatorlearning.domains import (
    MetricSpaceProtocol,
    OffGridEvaluatorProtocol,
    UniformGridDomain,
)
from pyapprox.util.backends.protocols import Backend
from pyapprox.util.linalg.inner_product import (
    DiagonalInnerProduct,
    EuclideanInnerProduct,
)


def _axis(bkd: Backend, lo: float, hi: float, npoints: int):
    return bkd.asarray(np.linspace(lo, hi, npoints))


class TestProtocolConformance:
    """Grid-boundedness is which protocols an object satisfies."""

    def test_uniform_grid_is_a_metric_space(self, bkd: Backend) -> None:
        domain = UniformGridDomain([_axis(bkd, 0.0, 1.0, 5)], bkd)
        assert isinstance(domain, MetricSpaceProtocol)

    def test_a_callers_own_object_satisfies_it(self, bkd: Backend) -> None:
        """Two methods, and no import in either direction.

        This is why the module ships one domain rather than several: a
        caller who already holds a mesh and its quadrature weights needs
        no class from here. The protocol is structural, so their object
        is a domain by having the methods.
        """

        class MeshDomain:
            def bkd(self):
                return bkd

            def inner_product(self):
                return EuclideanInnerProduct(6, bkd)

        assert isinstance(MeshDomain(), MetricSpaceProtocol)

    def test_does_not_evaluate_off_grid(self, bkd: Backend) -> None:
        """Absence of the capability is absence of the methods.

        Nothing in the fixed-basis path evaluates away from its sample
        points, so this domain implements no interpolation rule. A
        consumer asks the protocol rather than reading a boolean, and
        gets False because the methods are genuinely not there.
        """
        domain = UniformGridDomain([_axis(bkd, 0.0, 1.0, 5)], bkd)
        assert not isinstance(domain, OffGridEvaluatorProtocol)
        assert not hasattr(domain, "interpolate")


class TestUniformGridQuadrature:
    r"""The weights must integrate, not merely sum to something.

    A metric that weights every point equally is the failure this class
    exists to catch: it makes the POD basis optimal in a norm nobody
    chose, biased toward wherever the grid is fine.
    """

    def _integral(self, bkd: Backend, domain, f, g):
        return float(domain.inner_product().dot(f, g)[0, 0])

    def test_weights_sum_to_the_measure_1d(self, bkd: Backend) -> None:
        domain = UniformGridDomain([_axis(bkd, 0.0, 1.0, 5)], bkd)
        ones = bkd.ones((5, 1))
        assert self._integral(bkd, domain, ones, ones) == pytest.approx(1.0)

    def test_weights_sum_to_the_measure_2d(self, bkd: Backend) -> None:
        domain = UniformGridDomain(
            [_axis(bkd, 0.0, 1.0, 3), _axis(bkd, 0.0, 2.0, 4)], bkd
        )
        ones = bkd.ones((12, 1))
        assert self._integral(bkd, domain, ones, ones) == pytest.approx(2.0)

    def test_integrates_a_linear_function_exactly(
        self, bkd: Backend
    ) -> None:
        r"""Trapezoid is exact for linear integrands: :math:`\int_0^1 x = 1/2`."""
        domain = UniformGridDomain([_axis(bkd, 0.0, 1.0, 5)], bkd)
        x = domain.sample_points()[0][:, None]
        assert self._integral(
            bkd, domain, x, bkd.ones((5, 1))
        ) == pytest.approx(0.5)

    def test_endpoint_weights_are_halved(self, bkd: Backend) -> None:
        """The signature of trapezoid rather than a uniform sum."""
        domain = UniformGridDomain([_axis(bkd, 0.0, 1.0, 5)], bkd)
        weights = domain.inner_product().apply(bkd.ones((5, 1)))[:, 0]
        bkd.assert_allclose(
            weights, bkd.asarray([0.125, 0.25, 0.25, 0.25, 0.125])
        )

    def test_graded_axis_weights_follow_the_spacing(
        self, bkd: Backend
    ) -> None:
        """The case that motivates weighting at all.

        On a graded axis a uniform metric over-weights the refined
        region, which is how a POD basis ends up spending modes there.
        Each interior weight is half the interval on either side.
        """
        axis = bkd.asarray(np.array([0.0, 0.1, 0.2, 1.0, 2.0]))
        domain = UniformGridDomain([axis], bkd)
        weights = domain.inner_product().apply(bkd.ones((5, 1)))[:, 0]
        bkd.assert_allclose(
            weights, bkd.asarray([0.05, 0.1, 0.45, 0.9, 0.5])
        )
        assert float(bkd.sum(weights)) == pytest.approx(2.0)


class TestUniformGridPoints:
    def test_c_order_matches_ravel(self, bkd: Backend) -> None:
        """A caller building snapshots with ravel needs no permutation.

        Pins the enumeration against meshgrid's "ij" indexing, since a
        mismatch would permute every snapshot silently.
        """
        xs, ys = _axis(bkd, 0.0, 1.0, 3), _axis(bkd, 0.0, 2.0, 4)
        domain = UniformGridDomain([xs, ys], bkd)
        mesh = np.meshgrid(
            bkd.to_numpy(xs), bkd.to_numpy(ys), indexing="ij"
        )
        bkd.assert_allclose(
            domain.sample_points(),
            bkd.asarray(np.vstack([m.ravel() for m in mesh])),
        )

    def test_reports_its_shape(self, bkd: Backend) -> None:
        domain = UniformGridDomain(
            [_axis(bkd, 0.0, 1.0, 3), _axis(bkd, 0.0, 2.0, 4)], bkd
        )
        assert domain.ndim() == 2
        assert domain.nsites() == 12
        assert domain.sample_points().shape == (2, 12)


class TestRejects:
    def test_grid_rejects_empty_axes(self, bkd: Backend) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            UniformGridDomain([], bkd)

    def test_grid_rejects_single_point_axis(self, bkd: Backend) -> None:
        with pytest.raises(ValueError, match="at least two points"):
            UniformGridDomain([bkd.asarray(np.array([0.0]))], bkd)

    def test_grid_rejects_unsorted_axis(self, bkd: Backend) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            UniformGridDomain(
                [bkd.asarray(np.array([0.0, 0.5, 0.25, 1.0]))], bkd
            )

    def test_grid_rejects_mismatched_metric(self, bkd: Backend) -> None:
        with pytest.raises(ValueError, match="defined on"):
            UniformGridDomain(
                [_axis(bkd, 0.0, 1.0, 5)],
                bkd,
                EuclideanInnerProduct(4, bkd),
            )



class TestSuppliedMetric:
    def test_grid_accepts_an_override(self, bkd: Backend) -> None:
        """A measure that is not Lebesgue is a modeling choice."""
        weights = bkd.asarray(np.full(5, 0.2))
        domain = UniformGridDomain(
            [_axis(bkd, 0.0, 1.0, 5)],
            bkd,
            DiagonalInnerProduct(weights, bkd),
        )
        ones = bkd.ones((5, 1))
        assert float(
            domain.inner_product().dot(ones, ones)[0, 0]
        ) == pytest.approx(1.0)
        bkd.assert_allclose(
            domain.inner_product().apply(ones)[:, 0], weights
        )
