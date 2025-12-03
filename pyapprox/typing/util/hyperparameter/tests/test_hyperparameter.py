import unittest
from typing import Generic, Any

import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.typing.util.backends.protocols import Backend, Array
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.abstracttestcase import AbstractTestCase
from pyapprox.typing.util.hyperparameter.hyperparameter import HyperParameter
from pyapprox.typing.util.hyperparameter.transforms import (
    IdentityHyperParameterTransform,
)


class TestHyperParameter(Generic[Array], AbstractTestCase):
    def bkd(self) -> Backend[Array]:
        """
        Override this method in derived classes to provide the specific
        backend.
        """
        raise NotImplementedError(
            "Derived classes must implement this method."
        )

    def setUp(self) -> None:
        """
        Set up the test environment for HyperParameter.
        """
        self.name = "example_param"
        self.nparams = 3
        self.values = self.bkd().array([1.0, 2.0, 3.0])
        self.bounds = (0.0, 5.0)
        self.transform = IdentityHyperParameterTransform(self.bkd())

    def test_initialization(self) -> None:
        """
        Test the initialization of HyperParameter.
        """
        hyperparameter = HyperParameter(
            name=self.name,
            nparams=self.nparams,
            values=self.values,
            bounds=self.bounds,
            bkd=self.bkd(),
            transform=self.transform,
        )
        self.assertEqual(hyperparameter.nparams(), self.nparams)
        self.bkd().assert_allclose(hyperparameter.get_values(), self.values)
        self.assertTrue(
            np.allclose(
                hyperparameter.get_bounds(), np.tile(self.bounds, self.nparams)
            )
        )

    def test_set_values(self) -> None:
        """
        Test setting values for HyperParameter.
        """
        hyperparameter = HyperParameter(
            name=self.name,
            nparams=self.nparams,
            values=self.values,
            bounds=self.bounds,
            bkd=self.bkd(),
            transform=self.transform,
        )
        new_values = np.array([4.0, 5.0, 6.0])
        hyperparameter.set_values(new_values)
        self.bkd().assert_allclose(hyperparameter.get_values(), new_values)

    def test_set_bounds(self) -> None:
        """
        Test setting bounds for HyperParameter.
        """
        hyperparameter = HyperParameter(
            name=self.name,
            nparams=self.nparams,
            values=self.values,
            bounds=self.bounds,
            bkd=self.bkd(),
            transform=self.transform,
        )
        new_bounds = np.array([0.0, 10.0])
        hyperparameter.set_bounds(new_bounds)
        self.assertTrue(
            np.allclose(
                hyperparameter.get_bounds(), np.tile(new_bounds, self.nparams)
            )
        )

    def test_active_indices(self) -> None:
        """
        Test setting and getting active indices for HyperParameter.
        """
        hyperparameter = HyperParameter(
            name=self.name,
            nparams=self.nparams,
            values=self.values,
            bounds=self.bounds,
            bkd=self.bkd(),
            transform=self.transform,
        )
        active_indices = np.array([0, 2])
        hyperparameter.set_active_indices(active_indices)
        self.assertTrue(
            np.allclose(hyperparameter.get_active_indices(), active_indices)
        )


# Derived test class for NumPy backend
class TestHyperParameterNumpy(
    TestHyperParameter[NDArray[Any]], unittest.TestCase
):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# Derived test class for PyTorch backend
class TestHyperParameterTorch(
    TestHyperParameter[torch.Tensor], unittest.TestCase
):
    def setUp(self) -> None:
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


if __name__ == "__main__":
    unittest.main()
