import unittest
from typing import Generic, Any

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.protocols import Backend, Array
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.hyperparameter.cholesky_hyperparameter import (
    CholeskyHyperParameter,
)


class TestCholeskyHyperParameter(Generic[Array], unittest.TestCase):
    """
    Base test class for CholeskyHyperParameter.

    Derived classes must implement the bkd() method to provide the backend.
    """

    def setUp(self) -> None:
        """
        Set up the test environment for CholeskyHyperParameter.
        """
        self.name = "cholesky_param"
        self.nrows = 2
        self.user_values = self.bkd().array([[1.0, 0.0], [0.5, 1.0]])
        self.user_bounds = self.bkd().array(
            [[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, -np.inf]]
        )

    def bkd(self) -> Backend:
        """
        Override this method in derived classes to provide the backend.
        """
        raise NotImplementedError(
            "Derived classes must implement this method."
        )

    def test_get_values(self) -> None:
        """
        Test the get_values function of CholeskyHyperParameter.
        """
        cholesky_hyperparameter = CholeskyHyperParameter(
            name=self.name,
            nrows=self.nrows,
            user_values=self.user_values,
            user_bounds=self.user_bounds,
            bkd=self.bkd(),
        )
        self.bkd().assert_allclose(
            cholesky_hyperparameter.get_values(),
            self.bkd().array([1.0, 0.5, 1.0]),
        )

    def test_get_cholesky_factor(self) -> None:
        """
        Test retrieving the full Cholesky factor for CholeskyHyperParameter.
        """
        cholesky_hyperparameter = CholeskyHyperParameter(
            name=self.name,
            nrows=self.nrows,
            user_values=self.user_values,
            user_bounds=self.user_bounds,
            bkd=self.bkd(),
        )
        self.bkd().assert_allclose(
            cholesky_hyperparameter.factor(), self.user_values
        )


class TestCholeskyHyperParameterNumpy(
    TestCholeskyHyperParameter[NDArray[Any]]
):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestCholeskyHyperParameterTorch(
    TestCholeskyHyperParameter[torch.Tensor]
):
    def setUp(self) -> None:
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> TorchBkd:
        return self._bkd


# Custom test loader to exclude the base class
def load_tests(
    loader: unittest.TestLoader, tests, pattern: str
) -> unittest.TestSuite:
    """
    Custom test loader to exclude the base class CholeskyHyperParameter.
    """
    test_suite = unittest.TestSuite()
    for test_class in [
        TestCholeskyHyperParameterNumpy,
        TestCholeskyHyperParameterTorch,
    ]:
        test_suite.addTests(loader.loadTestsFromTestCase(test_class))
    return test_suite


# Main block to explicitly run tests using the custom loader
if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = load_tests(loader, [], None)
    runner = unittest.TextTestRunner()
    runner.run(suite)
