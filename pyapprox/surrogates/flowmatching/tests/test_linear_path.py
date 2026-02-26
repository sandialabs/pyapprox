"""Tests for LinearPath probability path."""

from pyapprox.surrogates.flowmatching.linear_path import LinearPath
from pyapprox.surrogates.flowmatching.protocols import (
    ProbabilityPathProtocol,
)


class TestLinearPath:
    """Dual-backend tests for LinearPath."""

    def _make_path(self, bkd):
        return LinearPath(bkd)

    def test_satisfies_protocol(self, bkd) -> None:
        path = self._make_path(bkd)
        assert isinstance(path, ProbabilityPathProtocol)

    def test_alpha_boundary(self, bkd) -> None:
        path = self._make_path(bkd)
        t0 = bkd.array([[0.0]])
        t1 = bkd.array([[1.0]])
        bkd.assert_allclose(
            path.alpha(t0), bkd.array([[0.0]]), rtol=1e-12
        )
        bkd.assert_allclose(
            path.alpha(t1), bkd.array([[1.0]]), rtol=1e-12
        )

    def test_sigma_boundary(self, bkd) -> None:
        path = self._make_path(bkd)
        t0 = bkd.array([[0.0]])
        t1 = bkd.array([[1.0]])
        bkd.assert_allclose(
            path.sigma(t0), bkd.array([[1.0]]), rtol=1e-12
        )
        bkd.assert_allclose(
            path.sigma(t1), bkd.array([[0.0]]), rtol=1e-12
        )

    def test_alpha_plus_sigma_equals_one(self, bkd) -> None:
        path = self._make_path(bkd)
        t = bkd.array([[0.0, 0.25, 0.5, 0.75, 1.0]])
        result = path.alpha(t) + path.sigma(t)
        bkd.assert_allclose(result, bkd.ones_like(t), rtol=1e-12)

    def test_interpolate_at_t0_returns_x0(self, bkd) -> None:
        path = self._make_path(bkd)
        t = bkd.array([[0.0, 0.0]])
        x0 = bkd.array([[1.0, 2.0], [3.0, 4.0]])
        x1 = bkd.array([[5.0, 6.0], [7.0, 8.0]])
        result = path.interpolate(t, x0, x1)
        bkd.assert_allclose(result, x0, rtol=1e-12)

    def test_interpolate_at_t1_returns_x1(self, bkd) -> None:
        path = self._make_path(bkd)
        t = bkd.array([[1.0, 1.0]])
        x0 = bkd.array([[1.0, 2.0], [3.0, 4.0]])
        x1 = bkd.array([[5.0, 6.0], [7.0, 8.0]])
        result = path.interpolate(t, x0, x1)
        bkd.assert_allclose(result, x1, rtol=1e-12)

    def test_interpolate_midpoint(self, bkd) -> None:
        path = self._make_path(bkd)
        t = bkd.array([[0.5]])
        x0 = bkd.array([[0.0], [0.0]])
        x1 = bkd.array([[2.0], [4.0]])
        result = path.interpolate(t, x0, x1)
        expected = bkd.array([[1.0], [2.0]])
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_interpolate_shape(self, bkd) -> None:
        path = self._make_path(bkd)
        d, ns = 3, 5
        t = bkd.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
        x0 = bkd.zeros((d, ns))
        x1 = bkd.ones_like(x0)
        result = path.interpolate(t, x0, x1)
        assert result.shape == (d, ns)

    def test_target_field_shape(self, bkd) -> None:
        path = self._make_path(bkd)
        d, ns = 3, 5
        t = bkd.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
        x0 = bkd.zeros((d, ns))
        x1 = bkd.ones_like(x0)
        result = path.target_field(t, x0, x1)
        assert result.shape == (d, ns)

    def test_target_field_equals_x1_minus_x0(self, bkd) -> None:
        path = self._make_path(bkd)
        t = bkd.array([[0.3, 0.7]])
        x0 = bkd.array([[1.0, 2.0], [3.0, 4.0]])
        x1 = bkd.array([[5.0, 6.0], [7.0, 8.0]])
        result = path.target_field(t, x0, x1)
        expected = x1 - x0
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_target_field_is_time_derivative_of_interpolate(self, bkd) -> None:
        """Verify u_t = d/dt interpolate(t, x0, x1) via finite differences."""
        path = self._make_path(bkd)
        x0 = bkd.array([[1.0, 2.0], [3.0, 4.0]])
        x1 = bkd.array([[5.0, 6.0], [7.0, 8.0]])

        t_val = 0.4
        dt = 1e-7
        t = bkd.array([[t_val, t_val]])
        t_plus = bkd.array([[t_val + dt, t_val + dt]])
        t_minus = bkd.array([[t_val - dt, t_val - dt]])

        fd_deriv = (
            path.interpolate(t_plus, x0, x1)
            - path.interpolate(t_minus, x0, x1)
        ) / (2.0 * dt)

        analytical = path.target_field(t, x0, x1)
        bkd.assert_allclose(fd_deriv, analytical, rtol=1e-6)

    def test_d_alpha_via_finite_differences(self, bkd) -> None:
        path = self._make_path(bkd)
        t_val = 0.3
        dt = 1e-7
        t = bkd.array([[t_val]])
        t_plus = bkd.array([[t_val + dt]])
        t_minus = bkd.array([[t_val - dt]])

        fd = (path.alpha(t_plus) - path.alpha(t_minus)) / (2.0 * dt)
        analytical = path.d_alpha(t)
        bkd.assert_allclose(fd, analytical, rtol=1e-6)

    def test_d_sigma_via_finite_differences(self, bkd) -> None:
        path = self._make_path(bkd)
        t_val = 0.6
        dt = 1e-7
        t = bkd.array([[t_val]])
        t_plus = bkd.array([[t_val + dt]])
        t_minus = bkd.array([[t_val - dt]])

        fd = (path.sigma(t_plus) - path.sigma(t_minus)) / (2.0 * dt)
        analytical = path.d_sigma(t)
        bkd.assert_allclose(fd, analytical, rtol=1e-6)

    def test_target_field_consistent_with_coefficients(self, bkd) -> None:
        """Verify u_t = d_alpha*x1 + d_sigma*x0 (general formula)."""
        path = self._make_path(bkd)
        t = bkd.array([[0.2, 0.5, 0.8]])
        x0 = bkd.array([[1.0, 2.0, 3.0]])
        x1 = bkd.array([[4.0, 5.0, 6.0]])

        u_general = path.d_alpha(t) * x1 + path.d_sigma(t) * x0
        u_target = path.target_field(t, x0, x1)
        bkd.assert_allclose(u_general, u_target, rtol=1e-12)

    def test_1d_single_sample(self, bkd) -> None:
        """Test with d=1, ns=1."""
        path = self._make_path(bkd)
        t = bkd.array([[0.3]])
        x0 = bkd.array([[2.0]])
        x1 = bkd.array([[5.0]])
        result = path.interpolate(t, x0, x1)
        expected = bkd.array([[0.7 * 2.0 + 0.3 * 5.0]])
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_batch_with_varying_times(self, bkd) -> None:
        """Test batch with different time values per sample."""
        path = self._make_path(bkd)
        t = bkd.array([[0.0, 0.5, 1.0]])
        x0 = bkd.array([[1.0, 1.0, 1.0]])
        x1 = bkd.array([[3.0, 3.0, 3.0]])
        result = path.interpolate(t, x0, x1)
        expected = bkd.array([[1.0, 2.0, 3.0]])
        bkd.assert_allclose(result, expected, rtol=1e-12)
