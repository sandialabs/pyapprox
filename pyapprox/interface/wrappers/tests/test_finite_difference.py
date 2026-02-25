"""Tests for FiniteDifferenceWrapper."""

import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray

from pyapprox.interface.functions.fromcallable.function import (
    FunctionFromCallable,
)
from pyapprox.interface.wrappers.finite_difference import (
    FiniteDifferenceWrapper,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd


class TestFiniteDifferenceJacobian(Generic[Array], unittest.TestCase):
    """Base tests for finite difference Jacobian computation."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_forward_difference_quadratic(self) -> None:
        """Test forward difference on f(x) = x^2."""

        def fun(samples: Array) -> Array:
            return samples[0:1, :] ** 2

        model = FunctionFromCallable(nqoi=1, nvars=1, fun=fun, bkd=self._bkd)
        fd_model = FiniteDifferenceWrapper(model, method="forward")

        sample = self._bkd.asarray([[2.0]])
        jacobian = fd_model.jacobian(sample)

        # Jacobian of x^2 at x=2 is 2*2 = 4
        self.assertEqual(jacobian.shape, (1, 1))
        self._bkd.assert_allclose(
            jacobian,
            self._bkd.asarray([[4.0]]),
            rtol=1e-5,  # Forward difference is O(h)
        )

    def test_centered_difference_quadratic(self) -> None:
        """Test centered difference on f(x) = x^2."""

        def fun(samples: Array) -> Array:
            return samples[0:1, :] ** 2

        model = FunctionFromCallable(nqoi=1, nvars=1, fun=fun, bkd=self._bkd)
        fd_model = FiniteDifferenceWrapper(model, method="centered")

        sample = self._bkd.asarray([[2.0]])
        jacobian = fd_model.jacobian(sample)

        # Jacobian of x^2 at x=2 is 2*2 = 4
        self.assertEqual(jacobian.shape, (1, 1))
        self._bkd.assert_allclose(
            jacobian,
            self._bkd.asarray([[4.0]]),
            rtol=1e-10,  # Centered difference is O(h^2)
        )

    def test_centered_more_accurate_than_forward(self) -> None:
        """Test that centered difference is more accurate than forward."""

        def fun(samples: Array) -> Array:
            return self._bkd.sin(samples[0:1, :])

        model = FunctionFromCallable(nqoi=1, nvars=1, fun=fun, bkd=self._bkd)

        sample = self._bkd.asarray([[1.0]])
        expected = self._bkd.cos(sample)  # True derivative

        # Use a larger step to show difference
        step = 1e-4

        fd_forward = FiniteDifferenceWrapper(model, method="forward", step=step)
        fd_centered = FiniteDifferenceWrapper(model, method="centered", step=step)

        jac_forward = fd_forward.jacobian(sample)
        jac_centered = fd_centered.jacobian(sample)

        error_forward = self._bkd.abs(jac_forward - expected)
        error_centered = self._bkd.abs(jac_centered - expected)

        # Centered should be more accurate
        self.assertTrue(self._bkd.all_bool(error_centered < error_forward))

    def test_multivariate_jacobian(self) -> None:
        """Test Jacobian for f(x,y) = x^2 + y^2."""

        def fun(samples: Array) -> Array:
            x, y = samples[0:1, :], samples[1:2, :]
            return x**2 + y**2

        model = FunctionFromCallable(nqoi=1, nvars=2, fun=fun, bkd=self._bkd)
        fd_model = FiniteDifferenceWrapper(model, method="centered")

        sample = self._bkd.asarray([[2.0], [3.0]])
        jacobian = fd_model.jacobian(sample)

        # Jacobian is [2x, 2y] = [4, 6]
        self.assertEqual(jacobian.shape, (1, 2))
        self._bkd.assert_allclose(
            jacobian,
            self._bkd.asarray([[4.0, 6.0]]),
            rtol=1e-8,
        )

    def test_multi_output_jacobian(self) -> None:
        """Test Jacobian for multi-output function."""

        def fun(samples: Array) -> Array:
            x, y = samples[0:1, :], samples[1:2, :]
            return self._bkd.vstack([x**2, y**2])

        model = FunctionFromCallable(nqoi=2, nvars=2, fun=fun, bkd=self._bkd)
        fd_model = FiniteDifferenceWrapper(model, method="centered")

        sample = self._bkd.asarray([[2.0], [3.0]])
        jacobian = fd_model.jacobian(sample)

        # Jacobian is [[2x, 0], [0, 2y]] = [[4, 0], [0, 6]]
        self.assertEqual(jacobian.shape, (2, 2))
        self._bkd.assert_allclose(
            jacobian,
            self._bkd.asarray([[4.0, 0.0], [0.0, 6.0]]),
            rtol=1e-8,
        )


class TestFiniteDifferenceJVP(Generic[Array], unittest.TestCase):
    """Base tests for finite difference JVP computation."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_jvp_quadratic(self) -> None:
        """Test JVP on f(x,y) = x^2 + y^2."""

        def fun(samples: Array) -> Array:
            x, y = samples[0:1, :], samples[1:2, :]
            return x**2 + y**2

        model = FunctionFromCallable(nqoi=1, nvars=2, fun=fun, bkd=self._bkd)
        fd_model = FiniteDifferenceWrapper(model, method="centered")

        sample = self._bkd.asarray([[2.0], [3.0]])
        vec = self._bkd.asarray([[1.0], [1.0]])
        jvp = fd_model.jvp(sample, vec)

        # JVP = [2x, 2y] @ [1, 1] = 4 + 6 = 10
        self.assertEqual(jvp.shape, (1, 1))
        self._bkd.assert_allclose(
            jvp,
            self._bkd.asarray([[10.0]]),
            rtol=1e-8,
        )


class TestFiniteDifferenceHessian(Generic[Array], unittest.TestCase):
    """Base tests for finite difference Hessian computation."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_hessian_quadratic(self) -> None:
        """Test Hessian on f(x,y) = x^2 + 2*y^2."""

        def fun(samples: Array) -> Array:
            x, y = samples[0:1, :], samples[1:2, :]
            return x**2 + 2 * y**2

        model = FunctionFromCallable(nqoi=1, nvars=2, fun=fun, bkd=self._bkd)
        fd_model = FiniteDifferenceWrapper(model, method="centered")

        sample = self._bkd.asarray([[1.0], [1.0]])
        hessian = fd_model.hessian(sample)

        # Hessian is [[2, 0], [0, 4]]
        self.assertEqual(hessian.shape, (2, 2))
        self._bkd.assert_allclose(
            hessian,
            self._bkd.asarray([[2.0, 0.0], [0.0, 4.0]]),
            rtol=1e-5,
        )

    def test_hessian_requires_nqoi_one(self) -> None:
        """Test that Hessian raises error for nqoi > 1."""

        def fun(samples: Array) -> Array:
            x = samples[0:1, :]
            return self._bkd.vstack([x**2, x**3])

        model = FunctionFromCallable(nqoi=2, nvars=1, fun=fun, bkd=self._bkd)
        fd_model = FiniteDifferenceWrapper(model, method="centered")

        sample = self._bkd.asarray([[1.0]])
        with self.assertRaises(ValueError):
            fd_model.hessian(sample)


class TestFiniteDifferenceHVP(Generic[Array], unittest.TestCase):
    """Base tests for finite difference HVP computation."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_hvp_quadratic(self) -> None:
        """Test HVP on f(x,y) = x^2 + 2*y^2."""

        def fun(samples: Array) -> Array:
            x, y = samples[0:1, :], samples[1:2, :]
            return x**2 + 2 * y**2

        model = FunctionFromCallable(nqoi=1, nvars=2, fun=fun, bkd=self._bkd)
        fd_model = FiniteDifferenceWrapper(model, method="centered")

        sample = self._bkd.asarray([[1.0], [1.0]])
        vec = self._bkd.asarray([[1.0], [1.0]])
        hvp = fd_model.hvp(sample, vec)

        # Hessian is [[2, 0], [0, 4]], HVP = [2, 4]
        self.assertEqual(hvp.shape, (2, 1))
        self._bkd.assert_allclose(
            hvp,
            self._bkd.asarray([[2.0], [4.0]]),
            rtol=1e-5,
        )

    def test_hvp_requires_nqoi_one(self) -> None:
        """Test that HVP raises error for nqoi > 1."""

        def fun(samples: Array) -> Array:
            x = samples[0:1, :]
            return self._bkd.vstack([x**2, x**3])

        model = FunctionFromCallable(nqoi=2, nvars=1, fun=fun, bkd=self._bkd)
        fd_model = FiniteDifferenceWrapper(model, method="centered")

        sample = self._bkd.asarray([[1.0]])
        vec = self._bkd.asarray([[1.0]])
        with self.assertRaises(ValueError):
            fd_model.hvp(sample, vec)


class TestFiniteDifferencePassthrough(Generic[Array], unittest.TestCase):
    """Base tests for passthrough functionality."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_call_passthrough(self) -> None:
        """Test that __call__ passes through to wrapped model."""

        def fun(samples: Array) -> Array:
            return samples[0:1, :] ** 2

        model = FunctionFromCallable(nqoi=1, nvars=1, fun=fun, bkd=self._bkd)
        fd_model = FiniteDifferenceWrapper(model)

        samples = self._bkd.asarray([[1.0, 2.0, 3.0]])
        values = fd_model(samples)
        expected = self._bkd.asarray([[1.0, 4.0, 9.0]])

        self._bkd.assert_allclose(values, expected)

    def test_nvars_nqoi_passthrough(self) -> None:
        """Test that nvars and nqoi pass through."""

        def fun(samples: Array) -> Array:
            return samples[0:1, :]

        model = FunctionFromCallable(nqoi=1, nvars=3, fun=fun, bkd=self._bkd)
        fd_model = FiniteDifferenceWrapper(model)

        self.assertEqual(fd_model.nvars(), 3)
        self.assertEqual(fd_model.nqoi(), 1)

    def test_set_step(self) -> None:
        """Test setting step size."""

        def fun(samples: Array) -> Array:
            return samples[0:1, :]

        model = FunctionFromCallable(nqoi=1, nvars=1, fun=fun, bkd=self._bkd)
        fd_model = FiniteDifferenceWrapper(model, step=1e-4)

        self.assertEqual(fd_model._step, 1e-4)

        fd_model.set_step(1e-6)
        self.assertEqual(fd_model._step, 1e-6)


# NumPy backend concrete test classes
class TestFiniteDifferenceJacobianNumpy(TestFiniteDifferenceJacobian[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestFiniteDifferenceJVPNumpy(TestFiniteDifferenceJVP[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestFiniteDifferenceHessianNumpy(TestFiniteDifferenceHessian[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestFiniteDifferenceHVPNumpy(TestFiniteDifferenceHVP[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestFiniteDifferencePassthroughNumpy(
    TestFiniteDifferencePassthrough[NDArray[Any]]
):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# PyTorch backend concrete test classes
class TestFiniteDifferenceJacobianTorch(TestFiniteDifferenceJacobian[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> TorchBkd:
        return self._bkd


class TestFiniteDifferenceJVPTorch(TestFiniteDifferenceJVP[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> TorchBkd:
        return self._bkd


class TestFiniteDifferenceHessianTorch(TestFiniteDifferenceHessian[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> TorchBkd:
        return self._bkd


class TestFiniteDifferenceHVPTorch(TestFiniteDifferenceHVP[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> TorchBkd:
        return self._bkd


class TestFiniteDifferencePassthroughTorch(
    TestFiniteDifferencePassthrough[torch.Tensor]
):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> TorchBkd:
        return self._bkd


if __name__ == "__main__":
    unittest.main()
