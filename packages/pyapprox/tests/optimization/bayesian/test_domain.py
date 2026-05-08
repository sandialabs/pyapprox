"""Tests for BoxDomain."""

import pytest

from pyapprox.optimization.bayesian.domain.box import BoxDomain


class TestBoxDomain:
    def test_constructor(self, bkd) -> None:
        bounds = bkd.array([[0.0, 1.0], [-1.0, 2.0]])
        domain = BoxDomain(bounds, bkd)
        assert domain.nvars() == 2
        bkd.assert_allclose(domain.bounds(), bounds)

    def test_contains_inside(self, bkd) -> None:
        bounds = bkd.array([[0.0, 1.0], [-1.0, 2.0]])
        domain = BoxDomain(bounds, bkd)
        X = bkd.array([[0.5], [0.5]])
        result = domain.contains(X)
        assert bool(bkd.to_numpy(result)[0])

    def test_contains_outside(self, bkd) -> None:
        bounds = bkd.array([[0.0, 1.0], [-1.0, 2.0]])
        domain = BoxDomain(bounds, bkd)
        X = bkd.array([[1.5], [0.5]])
        result = domain.contains(X)
        assert not bool(bkd.to_numpy(result)[0])

    def test_contains_on_boundary(self, bkd) -> None:
        bounds = bkd.array([[0.0, 1.0], [-1.0, 2.0]])
        domain = BoxDomain(bounds, bkd)
        X = bkd.array([[0.0], [-1.0]])
        result = domain.contains(X)
        assert bool(bkd.to_numpy(result)[0])

    def test_contains_batch(self, bkd) -> None:
        bounds = bkd.array([[0.0, 1.0], [-1.0, 2.0]])
        domain = BoxDomain(bounds, bkd)
        # 3 points: inside, outside, on boundary
        X = bkd.array([[0.5, 1.5, 1.0], [0.5, 0.5, 2.0]])
        result = domain.contains(X)
        result_np = bkd.to_numpy(result)
        assert bool(result_np[0])  # inside
        assert not bool(result_np[1])  # outside
        assert bool(result_np[2])  # on boundary

    def test_invalid_bounds_shape(self, bkd) -> None:
        with pytest.raises(ValueError, match="shape"):
            BoxDomain(bkd.array([0.0, 1.0]), bkd)

    def test_bkd(self, bkd) -> None:
        bounds = bkd.array([[0.0, 1.0]])
        domain = BoxDomain(bounds, bkd)
        assert domain.bkd() is bkd
