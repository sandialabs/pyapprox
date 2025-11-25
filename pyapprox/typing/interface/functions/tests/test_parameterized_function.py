import unittest
from typing import Generic, Any

from numpy.typing import NDArray
import torch

from pyapprox.typing.interface.functions.parameterized_wrappers import (
    FunctionOfParametersWithJacobian,
    FunctionOfParametersWithHVP,
)
from pyapprox.typing.util.backend import Array, Backend
from pyapprox.typing.util.numpy import NumpyBkd
from pyapprox.typing.util.torch import TorchBkd
from pyapprox.typing.util.abstracttestcase import AbstractTestCase


class ExampleParameterizedFunctionWithFullRankHVP:
    """
    Example implementation of a parameterized function with full-rank Hessian.

    The function is defined as:
        f(x, p) = p[0]**2 * x**2 + p[1]**3 * x + p[2]**4 + p[0] * p[1]
    """

    def __init__(self, bkd: Backend[Array]):
        self._bkd = bkd

    def __call__(self, x: Array) -> Array:
        """
        Evaluate f(x, p) = p[0]**2 * x**2 + p[1]**3 * x + p[2]**4 + p[0] * p[1].
        """
        if self._p is None:
            raise ValueError("Parameter p has not been set.")
        return (
            self._p[0] ** 2 * x**2
            + self._p[1] ** 3 * x
            + self._p[2] ** 4
            + self._p[0] * self._p[1]
        )

    def set_parameter(self, p: Array) -> None:
        """
        Set the parameter p.
        """
        self._p = p

    def jacobian_wrt_parameters(self, x: Array) -> Array:
        """
        Compute the Jacobian of f(x, p) with respect to p.

        Jacobian:
        - ∂f/∂p[0] = 2 * p[0] * x**2 + p[1]
        - ∂f/∂p[1] = 3 * p[1]**2 * x + p[0]
        - ∂f/∂p[2] = 4 * p[2]**3
        """
        return self._bkd.stack(
            [
                2 * self._p[0] * x**2 + self._p[1],  # ∂f/∂p[0]
                3 * self._p[1] ** 2 * x + self._p[0],  # ∂f/∂p[1]
                4 * self._p[2] ** 3,  # ∂f/∂p[2]
            ],
            axis=0,
        )

    def hvp_wrt_parameters(self, x: Array, vec: Array) -> Array:
        """
        Compute the Hessian-vector product of f(x, p) with respect to p.

        Hessian:
        - ∂²f/∂p[0]² = 2 * x**2
        - ∂²f/∂p[1]² = 6 * p[1] * x
        - ∂²f/∂p[2]² = 12 * p[2]**2
        - ∂²f/∂p[0]∂p[1] = 1
        """
        hessian = self._bkd.zeros((len(self._p), len(self._p), x.shape[1]))
        hessian[0, 0, :] = 2 * x**2  # ∂²f/∂p[0]²
        hessian[1, 1, :] = 6 * self._p[1] * x  # ∂²f/∂p[1]²
        hessian[2, 2, :] = 12 * self._p[2] ** 2  # ∂²f/∂p[2]²
        hessian[0, 1, :] = self._bkd.ones_like(x)  # ∂²f/∂p[0]∂p[1]
        hessian[1, 0, :] = self._bkd.ones_like(x)  # Symmetry of Hessian

        # Compute Hessian-vector product
        return self._bkd.tensordot(hessian, vec, axes=1)


class TestParameterizedFunction(Generic[Array], AbstractTestCase):
    def bkd(self) -> Backend[Array]:
        """
        Override this method in derived classes to provide the specific backend.
        """
        raise NotImplementedError(
            "Derived classes must implement this method."
        )

    def setUp(self) -> None:
        self.fixed_x = self.bkd().reshape(
            self.bkd().linspace(0, 1, 5), (1, -1)
        )  # Shape (1, npts)
        self.param_func_with_full_rank_hvp = (
            ExampleParameterizedFunctionWithFullRankHVP(self.bkd())
        )

    def test_function_of_parameters_with_jacobian(self):
        """
        Test the functionality of FunctionOfParametersWithJacobian.
        """
        self.param_func_with_full_rank_hvp.set_parameter(
            self.bkd().array([1.0, 2.0, 3.0])
        )
        func_of_params_with_jacobian = FunctionOfParametersWithJacobian(
            self.param_func_with_full_rank_hvp, self.fixed_x
        )

        param = self.bkd().array([1.0, 2.0, 3.0])  # Single parameter vector
        jacobian = func_of_params_with_jacobian.jacobian(param)

        # Expected Jacobian computation:
        expected_jacobian = self.bkd().stack(
            [
                2.0 * param[0] * self.fixed_x[0] ** 2 + param[1],  # ∂f/∂p[0]
                3.0 * param[1] ** 2 * self.fixed_x[0] + param[0],  # ∂f/∂p[1]
                4.0 * param[2] ** 3,  # ∂f/∂p[2]
            ],
            axis=0,
        )
        self.assertTrue(self.bkd().allclose(jacobian, expected_jacobian))

    def test_function_of_parameters_with_hvp(self):
        """
        Test the functionality of FunctionOfParametersWithHVP with full-rank Hessian.
        """
        self.param_func_with_full_rank_hvp.set_parameter(
            self.bkd().array([1.0, 2.0, 3.0])
        )
        func_of_params_with_hvp = FunctionOfParametersWithHVP(
            self.param_func_with_full_rank_hvp, self.fixed_x
        )

        param = self.bkd().array([1.0, 2.0, 3.0])  # Single parameter vector
        vec = self.bkd().array([1.0, 1.0, 1.0])  # Vector for HVP
        hvp = func_of_params_with_hvp.hvp(param, vec)

        # Expected HVP computation:
        expected_hvp = self.bkd().stack(
            [
                2.0 * self.fixed_x[0] ** 2 * vec[0]
                + vec[1],  # ∂²f/∂p[0]² * vec[0] + ∂²f/∂p[0]∂p[1] * vec[1]
                6.0 * param[1] * self.fixed_x[0] * vec[1]
                + vec[0],  # ∂²f/∂p[1]² * vec[1] + ∂²f/∂p[0]∂p[1] * vec[0]
                12.0 * param[2] ** 2 * vec[2],  # ∂²f/∂p[2]² * vec[2]
            ],
            axis=0,
        )
        self.assertTrue(self.bkd().allclose(hvp, expected_hvp))


# Derived test class for NumPy backend
class TestParameterizedFunctionNumpy(
    TestParameterizedFunction[NDArray[Any]], unittest.TestCase
):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# Derived test class for PyTorch backend
class TestParameterizedFunctionTorch(
    TestParameterizedFunction[torch.Tensor], unittest.TestCase
):
    def setUp(self) -> None:
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


if __name__ == "__main__":
    unittest.main()
