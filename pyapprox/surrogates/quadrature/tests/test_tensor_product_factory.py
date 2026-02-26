"""Tests for TensorProductQuadratureFactory."""

import numpy as np

from pyapprox.surrogates.quadrature.tensor_product_factory import (
    TensorProductQuadratureFactory,
)


class TestTensorProductQuadratureFactory:
    def test_affine_mapping_points_in_range(self, bkd) -> None:
        """Verify mapped points lie within [lb, ub]."""
        domain = bkd.asarray([[2.0, 5.0], [10.0, 20.0]])
        factory = TensorProductQuadratureFactory([5, 5], domain, bkd)
        quad = factory([0, 1])
        samples, weights = quad()
        assert samples.shape[0] == 2
        s_np = bkd.to_numpy(samples)
        assert np.all(s_np[0] >= 2.0 - 1e-14)
        assert np.all(s_np[0] <= 5.0 + 1e-14)
        assert np.all(s_np[1] >= 10.0 - 1e-14)
        assert np.all(s_np[1] <= 20.0 + 1e-14)

    def test_integrate_constant_gives_volume(self, bkd) -> None:
        """Integral of 1 over [a,b] should give b-a."""
        domain = bkd.asarray([[2.0, 5.0]])
        factory = TensorProductQuadratureFactory([5], domain, bkd)
        quad = factory([0])
        samples, weights = quad()
        integral = bkd.sum(weights)
        bkd.assert_allclose(bkd.asarray([integral]), bkd.asarray([3.0]), rtol=1e-12)

    def test_integrate_constant_2d(self, bkd) -> None:
        """Integral of 1 over [2,5] x [10,20] = 3 * 10 = 30."""
        domain = bkd.asarray([[2.0, 5.0], [10.0, 20.0]])
        factory = TensorProductQuadratureFactory([3, 3], domain, bkd)
        quad = factory([0, 1])
        _, weights = quad()
        bkd.assert_allclose(
            bkd.asarray([bkd.sum(weights)]),
            bkd.asarray([30.0]),
            rtol=1e-12,
        )

    def test_subset_selection(self, bkd) -> None:
        """3-variable factory, call with [1] -> 1D rule for var 1."""
        domain = bkd.asarray([[0.0, 1.0], [2.0, 4.0], [5.0, 10.0]])
        factory = TensorProductQuadratureFactory([3, 5, 7], domain, bkd)
        quad = factory([1])
        samples, weights = quad()
        assert samples.shape[0] == 1
        assert samples.shape[1] == 5
        # Points should be in [2, 4]
        s_np = bkd.to_numpy(samples)
        assert np.all(s_np >= 2.0 - 1e-14)
        assert np.all(s_np <= 4.0 + 1e-14)
        # Integral of 1 = 4 - 2 = 2
        bkd.assert_allclose(
            bkd.asarray([bkd.sum(weights)]),
            bkd.asarray([2.0]),
            rtol=1e-12,
        )

    def test_polynomial_integration(self, bkd) -> None:
        """Integrate x^2 on [0, 3] exactly."""
        domain = bkd.asarray([[0.0, 3.0]])
        factory = TensorProductQuadratureFactory([5], domain, bkd)
        quad = factory([0])
        samples, weights = quad()
        # int_0^3 x^2 dx = 9
        values = samples[0] ** 2
        integral = bkd.sum(weights * values)
        bkd.assert_allclose(bkd.asarray([integral]), bkd.asarray([9.0]), rtol=1e-12)

    def test_integrate_method(self, bkd) -> None:
        """Test the integrate() method of _AffinelyMappedQuadratureRule."""
        domain = bkd.asarray([[0.0, 2.0], [0.0, 3.0]])
        factory = TensorProductQuadratureFactory([5, 5], domain, bkd)
        quad = factory([0, 1])

        def func(samples):
            # f(x,y) = x*y, returns (nsamples, 1)
            return bkd.reshape(samples[0] * samples[1], (-1, 1))

        result = quad.integrate(func)
        # int_0^2 int_0^3 x*y dy dx = (2^2/2) * (3^2/2) = 2 * 4.5 = 9
        bkd.assert_allclose(result, bkd.asarray([9.0]), rtol=1e-12)
