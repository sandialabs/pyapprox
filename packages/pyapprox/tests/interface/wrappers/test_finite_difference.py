"""Tests for FiniteDifferenceWrapper."""

import pytest

from pyapprox.interface.functions.fromcallable.function import (
    FunctionFromCallable,
)
from pyapprox.interface.wrappers.finite_difference import (
    FiniteDifferenceWrapper,
)


class TestFiniteDifferenceJacobian:
    """Base tests for finite difference Jacobian computation."""

    def test_forward_difference_quadratic(self, bkd) -> None:
        """Test forward difference on f(x) = x^2."""

        def fun(samples):
            return samples[0:1, :] ** 2

        model = FunctionFromCallable(nqoi=1, nvars=1, fun=fun, bkd=bkd)
        fd_model = FiniteDifferenceWrapper(model, method="forward")

        sample = bkd.asarray([[2.0]])
        jacobian = fd_model.jacobian(sample)

        # Jacobian of x^2 at x=2 is 2*2 = 4
        assert jacobian.shape == (1, 1)
        bkd.assert_allclose(
            jacobian,
            bkd.asarray([[4.0]]),
            rtol=1e-5,  # Forward difference is O(h)
        )

    def test_centered_difference_quadratic(self, bkd) -> None:
        """Test centered difference on f(x) = x^2."""

        def fun(samples):
            return samples[0:1, :] ** 2

        model = FunctionFromCallable(nqoi=1, nvars=1, fun=fun, bkd=bkd)
        fd_model = FiniteDifferenceWrapper(model, method="centered")

        sample = bkd.asarray([[2.0]])
        jacobian = fd_model.jacobian(sample)

        # Jacobian of x^2 at x=2 is 2*2 = 4
        assert jacobian.shape == (1, 1)
        bkd.assert_allclose(
            jacobian,
            bkd.asarray([[4.0]]),
            rtol=1e-10,  # Centered difference is O(h^2)
        )

    def test_centered_more_accurate_than_forward(self, bkd) -> None:
        """Test that centered difference is more accurate than forward."""

        def fun(samples):
            return bkd.sin(samples[0:1, :])

        model = FunctionFromCallable(nqoi=1, nvars=1, fun=fun, bkd=bkd)

        sample = bkd.asarray([[1.0]])
        expected = bkd.cos(sample)  # True derivative

        # Use a larger step to show difference
        step = 1e-4

        fd_forward = FiniteDifferenceWrapper(model, method="forward", step=step)
        fd_centered = FiniteDifferenceWrapper(model, method="centered", step=step)

        jac_forward = fd_forward.jacobian(sample)
        jac_centered = fd_centered.jacobian(sample)

        error_forward = bkd.abs(jac_forward - expected)
        error_centered = bkd.abs(jac_centered - expected)

        # Centered should be more accurate
        assert bkd.all_bool(error_centered < error_forward)

    def test_multivariate_jacobian(self, bkd) -> None:
        """Test Jacobian for f(x,y) = x^2 + y^2."""

        def fun(samples):
            x, y = samples[0:1, :], samples[1:2, :]
            return x**2 + y**2

        model = FunctionFromCallable(nqoi=1, nvars=2, fun=fun, bkd=bkd)
        fd_model = FiniteDifferenceWrapper(model, method="centered")

        sample = bkd.asarray([[2.0], [3.0]])
        jacobian = fd_model.jacobian(sample)

        # Jacobian is [2x, 2y] = [4, 6]
        assert jacobian.shape == (1, 2)
        bkd.assert_allclose(
            jacobian,
            bkd.asarray([[4.0, 6.0]]),
            rtol=1e-8,
        )

    def test_multi_output_jacobian(self, bkd) -> None:
        """Test Jacobian for multi-output function."""

        def fun(samples):
            x, y = samples[0:1, :], samples[1:2, :]
            return bkd.vstack([x**2, y**2])

        model = FunctionFromCallable(nqoi=2, nvars=2, fun=fun, bkd=bkd)
        fd_model = FiniteDifferenceWrapper(model, method="centered")

        sample = bkd.asarray([[2.0], [3.0]])
        jacobian = fd_model.jacobian(sample)

        # Jacobian is [[2x, 0], [0, 2y]] = [[4, 0], [0, 6]]
        assert jacobian.shape == (2, 2)
        bkd.assert_allclose(
            jacobian,
            bkd.asarray([[4.0, 0.0], [0.0, 6.0]]),
            rtol=1e-8,
        )


class TestFiniteDifferenceJVP:
    """Base tests for finite difference JVP computation."""

    def test_jvp_quadratic(self, bkd) -> None:
        """Test JVP on f(x,y) = x^2 + y^2."""

        def fun(samples):
            x, y = samples[0:1, :], samples[1:2, :]
            return x**2 + y**2

        model = FunctionFromCallable(nqoi=1, nvars=2, fun=fun, bkd=bkd)
        fd_model = FiniteDifferenceWrapper(model, method="centered")

        sample = bkd.asarray([[2.0], [3.0]])
        vec = bkd.asarray([[1.0], [1.0]])
        jvp = fd_model.jvp(sample, vec)

        # JVP = [2x, 2y] @ [1, 1] = 4 + 6 = 10
        assert jvp.shape == (1, 1)
        bkd.assert_allclose(
            jvp,
            bkd.asarray([[10.0]]),
            rtol=1e-8,
        )


class TestFiniteDifferenceHessian:
    """Base tests for finite difference Hessian computation."""

    def test_hessian_quadratic(self, bkd) -> None:
        """Test Hessian on f(x,y) = x^2 + 2*y^2."""

        def fun(samples):
            x, y = samples[0:1, :], samples[1:2, :]
            return x**2 + 2 * y**2

        model = FunctionFromCallable(nqoi=1, nvars=2, fun=fun, bkd=bkd)
        fd_model = FiniteDifferenceWrapper(model, method="centered")

        sample = bkd.asarray([[1.0], [1.0]])
        hessian = fd_model.hessian(sample)

        # Hessian is [[2, 0], [0, 4]]
        assert hessian.shape == (2, 2)
        bkd.assert_allclose(
            hessian,
            bkd.asarray([[2.0, 0.0], [0.0, 4.0]]),
            rtol=1e-5,
        )

    def test_hessian_requires_nqoi_one(self, bkd) -> None:
        """Test that Hessian raises error for nqoi > 1."""

        def fun(samples):
            x = samples[0:1, :]
            return bkd.vstack([x**2, x**3])

        model = FunctionFromCallable(nqoi=2, nvars=1, fun=fun, bkd=bkd)
        fd_model = FiniteDifferenceWrapper(model, method="centered")

        sample = bkd.asarray([[1.0]])
        with pytest.raises(ValueError):
            fd_model.hessian(sample)


class TestFiniteDifferenceHVP:
    """Base tests for finite difference HVP computation."""

    def test_hvp_quadratic(self, bkd) -> None:
        """Test HVP on f(x,y) = x^2 + 2*y^2."""

        def fun(samples):
            x, y = samples[0:1, :], samples[1:2, :]
            return x**2 + 2 * y**2

        model = FunctionFromCallable(nqoi=1, nvars=2, fun=fun, bkd=bkd)
        fd_model = FiniteDifferenceWrapper(model, method="centered")

        sample = bkd.asarray([[1.0], [1.0]])
        vec = bkd.asarray([[1.0], [1.0]])
        hvp = fd_model.hvp(sample, vec)

        # Hessian is [[2, 0], [0, 4]], HVP = [2, 4]
        assert hvp.shape == (2, 1)
        bkd.assert_allclose(
            hvp,
            bkd.asarray([[2.0], [4.0]]),
            rtol=1e-5,
        )

    def test_hvp_requires_nqoi_one(self, bkd) -> None:
        """Test that HVP raises error for nqoi > 1."""

        def fun(samples):
            x = samples[0:1, :]
            return bkd.vstack([x**2, x**3])

        model = FunctionFromCallable(nqoi=2, nvars=1, fun=fun, bkd=bkd)
        fd_model = FiniteDifferenceWrapper(model, method="centered")

        sample = bkd.asarray([[1.0]])
        vec = bkd.asarray([[1.0]])
        with pytest.raises(ValueError):
            fd_model.hvp(sample, vec)


class TestFiniteDifferencePassthrough:
    """Base tests for passthrough functionality."""

    def test_call_passthrough(self, bkd) -> None:
        """Test that __call__ passes through to wrapped model."""

        def fun(samples):
            return samples[0:1, :] ** 2

        model = FunctionFromCallable(nqoi=1, nvars=1, fun=fun, bkd=bkd)
        fd_model = FiniteDifferenceWrapper(model)

        samples = bkd.asarray([[1.0, 2.0, 3.0]])
        values = fd_model(samples)
        expected = bkd.asarray([[1.0, 4.0, 9.0]])

        bkd.assert_allclose(values, expected)

    def test_nvars_nqoi_passthrough(self, bkd) -> None:
        """Test that nvars and nqoi pass through."""

        def fun(samples):
            return samples[0:1, :]

        model = FunctionFromCallable(nqoi=1, nvars=3, fun=fun, bkd=bkd)
        fd_model = FiniteDifferenceWrapper(model)

        assert fd_model.nvars() == 3
        assert fd_model.nqoi() == 1

    def test_set_step(self, bkd) -> None:
        """Test setting step size."""

        def fun(samples):
            return samples[0:1, :]

        model = FunctionFromCallable(nqoi=1, nvars=1, fun=fun, bkd=bkd)
        fd_model = FiniteDifferenceWrapper(model, step=1e-4)

        assert fd_model._step == 1e-4

        fd_model.set_step(1e-6)
        assert fd_model._step == 1e-6
