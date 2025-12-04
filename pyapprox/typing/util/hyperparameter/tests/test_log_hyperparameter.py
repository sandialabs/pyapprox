import unittest
from typing import Generic, Any

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.protocols import Backend, Array
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.hyperparameter.log_hyperparameter import (
    LogHyperParameter,
)


class TestLogHyperParameter(Generic[Array], unittest.TestCase):
    """
    Base test class for LogHyperParameter.

    Derived classes must implement the bkd() method to provide the backend.
    """

    def setUp(self) -> None:
        """
        Set up the test environment for LogHyperParameter.
        """
        self.name = "log_param"
        self.nparams = 3
        self.user_values = self.bkd().array([1.0, 2.0, 3.0])
        self.user_bounds = (0.1, 10.0)

    def bkd(self) -> Backend:
        """
        Override this method in derived classes to provide the backend.
        """
        raise NotImplementedError(
            "Derived classes must implement this method."
        )

    def test_get_values(self) -> None:
        """
        Test retrieving log-transformed values for LogHyperParameter.
        """
        log_hyperparameter = LogHyperParameter(
            name=self.name,
            nparams=self.nparams,
            user_values=self.user_values,
            user_bounds=self.user_bounds,
            bkd=self.bkd(),
        )
        self.bkd().assert_allclose(
            log_hyperparameter.get_values(),
            self.bkd().log(self.user_values),
        )

    def test_exp_values(self) -> None:
        """
        Test retrieving exponential values for LogHyperParameter.
        """
        log_hyperparameter = LogHyperParameter(
            name=self.name,
            nparams=self.nparams,
            user_values=self.user_values,
            user_bounds=self.user_bounds,
            bkd=self.bkd(),
        )
        self.bkd().assert_allclose(
            log_hyperparameter.exp_values(),
            self.user_values,
        )


class TestLogHyperParameterNumpy(TestLogHyperParameter[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestLogHyperParameterTorch(TestLogHyperParameter[torch.Tensor]):
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
    Custom test loader to exclude the base class LogHyperParameter.
    """
    test_suite = unittest.TestSuite()
    for test_class in [
        TestLogHyperParameterNumpy,
        TestLogHyperParameterTorch,
    ]:
        test_suite.addTests(loader.loadTestsFromTestCase(test_class))
    return test_suite


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = load_tests(loader, [], None)
    runner = unittest.TextTestRunner()
    runner.run(suite)
