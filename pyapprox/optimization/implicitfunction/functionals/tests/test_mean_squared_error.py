import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray

from pyapprox.optimization.implicitfunction.functionals.mean_squared_error import (
    MSEFunctional,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd


class TestMSEFunctional(Generic[Array], unittest.TestCase):
    """
    Base test class for MSEFunctional.

    Derived classes must implement the bkd() method to provide the backend.
    """

    __test__ = False

    def setUp(self) -> None:
        """
        Set up the test environment for MSEFunctional.
        """
        self.nstates = 3
        self.nparams = 2
        self.obs = self.bkd().reshape(self.bkd().array([1.0, 2.0, 3.0]), (3, 1))

    def bkd(self) -> Backend:
        """
        Override this method in derived classes to provide the backend.
        """
        raise NotImplementedError("Derived classes must implement this method.")

    def test_initialization(self) -> None:
        """
        Test the initialization of MSEFunctional.
        """
        func = MSEFunctional(self.nstates, self.nparams, self.bkd())
        self.assertEqual(func.nstates(), self.nstates)
        self.assertEqual(func.nparams(), self.nparams)
        self.assertEqual(func.nqoi(), 1)  # MSE always returns scalar
        self.assertEqual(func.nunique_params(), 0)
        self.assertIsNotNone(func.bkd())

    def test_set_observations_valid(self) -> None:
        """
        Test setting observations with valid shape.
        """
        func = MSEFunctional(self.nstates, self.nparams, self.bkd())
        func.set_observations(self.obs)
        # No exception should be raised

    def test_set_observations_invalid_shape(self) -> None:
        """
        Test that set_observations raises ValueError for invalid shape.
        """
        func = MSEFunctional(self.nstates, self.nparams, self.bkd())
        bad_obs = self.bkd().array([1.0, 2.0])  # Wrong shape
        with self.assertRaises(ValueError):
            func.set_observations(bad_obs)

    def test_functional_evaluation(self) -> None:
        """
        Test the functional evaluation (__call__).

        MSE = sum((obs - state)^2) / 2
        """
        func = MSEFunctional(self.nstates, self.nparams, self.bkd())
        func.set_observations(self.obs)

        state = self.bkd().reshape(self.bkd().array([2.0, 3.0, 4.0]), (3, 1))
        param = self.bkd().zeros((2, 1))  # Not used in MSE

        result = func(state, param)

        # Expected: sum((obs - state)^2) / 2
        # = sum(([1, 2, 3] - [2, 3, 4])^2) / 2
        # = sum([-1, -1, -1]^2) / 2 = sum([1, 1, 1]) / 2 = 3 / 2 = 1.5
        expected = self.bkd().reshape(self.bkd().array([1.5]), (1, 1))
        self.bkd().assert_allclose(result, expected)

    def test_state_jacobian(self) -> None:
        """
        Test the state Jacobian computation.

        For MSE: dJ/dstate = (state - obs)^T
        """
        func = MSEFunctional(self.nstates, self.nparams, self.bkd())
        func.set_observations(self.obs)

        state = self.bkd().reshape(self.bkd().array([2.0, 3.0, 5.0]), (3, 1))
        param = self.bkd().zeros((2, 1))

        jac = func.state_jacobian(state, param)

        # Expected: (state - obs)^T = ([2, 3, 5] - [1, 2, 3])^T = [1, 1, 2]^T
        expected = self.bkd().reshape(self.bkd().array([[1.0, 1.0, 2.0]]), (1, 3))
        self.bkd().assert_allclose(jac, expected)

    def test_param_jacobian(self) -> None:
        """
        Test the parameter Jacobian computation.

        For MSE with respect to parameters: should be zeros
        (MSE doesn't depend on parameters).
        """
        func = MSEFunctional(self.nstates, self.nparams, self.bkd())
        func.set_observations(self.obs)

        state = self.bkd().zeros((3, 1))
        param = self.bkd().zeros((2, 1))

        jac = func.param_jacobian(state, param)

        # Expected: zeros (MSE doesn't depend on params)
        expected = self.bkd().zeros((1, self.nparams))
        self.bkd().assert_allclose(jac, expected)

    def test_state_state_hvp(self) -> None:
        """
        Test state-state Hessian-vector product.

        For MSE: d²J/dstate² = I (identity), so HVP returns the input vector.
        """
        func = MSEFunctional(self.nstates, self.nparams, self.bkd())
        func.set_observations(self.obs)

        state = self.bkd().zeros((3, 1))
        param = self.bkd().zeros((2, 1))
        vvec = self.bkd().reshape(self.bkd().array([1.0, 2.0, 3.0]), (3, 1))

        hvp = func.state_state_hvp(state, param, vvec)

        # Expected: identity matrix times vector = vector itself
        self.bkd().assert_allclose(hvp, vvec)

    def test_param_param_hvp(self) -> None:
        """
        Test parameter-parameter Hessian-vector product.

        For MSE: should be zeros (MSE doesn't depend on parameters).
        """
        func = MSEFunctional(self.nstates, self.nparams, self.bkd())
        func.set_observations(self.obs)

        state = self.bkd().zeros((3, 1))
        param = self.bkd().zeros((2, 1))
        vvec = self.bkd().ones((2, 1))

        hvp = func.param_param_hvp(state, param, vvec)

        # Expected: zeros
        expected = self.bkd().zeros((self.nparams, 1))
        self.bkd().assert_allclose(hvp, expected)

    def test_param_state_hvp(self) -> None:
        """
        Test parameter-state Hessian-vector product.

        For MSE: should be zeros (MSE doesn't depend on parameters).
        """
        func = MSEFunctional(self.nstates, self.nparams, self.bkd())
        func.set_observations(self.obs)

        state = self.bkd().zeros((3, 1))
        param = self.bkd().zeros((2, 1))
        vvec = self.bkd().ones((3, 1))

        hvp = func.param_state_hvp(state, param, vvec)

        # Expected: zeros
        expected = self.bkd().zeros((self.nparams, 1))
        self.bkd().assert_allclose(hvp, expected)

    def test_state_param_hvp(self) -> None:
        """
        Test state-parameter Hessian-vector product.

        For MSE: should be zeros (MSE doesn't depend on parameters).
        """
        func = MSEFunctional(self.nstates, self.nparams, self.bkd())
        func.set_observations(self.obs)

        state = self.bkd().zeros((3, 1))
        param = self.bkd().zeros((2, 1))
        vvec = self.bkd().ones((2, 1))

        hvp = func.state_param_hvp(state, param, vvec)

        # Expected: zeros
        expected = self.bkd().zeros((self.nstates, 1))
        self.bkd().assert_allclose(hvp, expected)

    def test_repr(self) -> None:
        """
        Test the string representation of MSEFunctional.
        """
        func = MSEFunctional(self.nstates, self.nparams, self.bkd())
        repr_str = repr(func)
        self.assertIn("MSEFunctional", repr_str)
        self.assertIn(f"nstates={self.nstates}", repr_str)
        self.assertIn(f"nparams={self.nparams}", repr_str)


class TestMSEFunctionalNumpy(TestMSEFunctional[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestMSEFunctionalTorch(TestMSEFunctional[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> TorchBkd:
        return self._bkd


# Custom test loader to exclude the base class
def load_tests(loader: unittest.TestLoader, tests, pattern: str) -> unittest.TestSuite:
    """
    Custom test loader to exclude the base class MSEFunctional.
    """
    test_suite = unittest.TestSuite()
    for test_class in [
        TestMSEFunctionalNumpy,
        TestMSEFunctionalTorch,
    ]:
        test_suite.addTests(loader.loadTestsFromTestCase(test_class))
    return test_suite


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = load_tests(loader, [], None)
    runner = unittest.TextTestRunner()
    runner.run(suite)
