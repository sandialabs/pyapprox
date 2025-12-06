import unittest
from typing import Generic, Any

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.protocols import Backend, Array
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.hyperparameter.hyperparameter import HyperParameter


class TestHyperParameter(Generic[Array], unittest.TestCase):
    """
    Base test class for HyperParameter.

    Derived classes must implement the bkd() method to provide the backend.
    """

    __test__ = False

    def setUp(self) -> None:
        """
        Set up the test environment for HyperParameter.
        """
        self.name = "example_param"
        self.nparams = 3
        self.values = self.bkd().array([1.0, 2.0, 3.0])
        self.bounds = self.bkd().array([[0.1, 10.0], [0.1, 10.0], [0.1, 10.0]])

    def bkd(self) -> Backend:
        """
        Override this method in derived classes to provide the backend.
        """
        raise NotImplementedError(
            "Derived classes must implement this method."
        )

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
        )
        self.assertEqual(hyperparameter.nparams(), self.nparams)
        self.bkd().assert_allclose(hyperparameter.get_values(), self.values)
        self.bkd().assert_allclose(hyperparameter.get_bounds(), self.bounds)

    def test_get_values(self) -> None:
        """
        Test the get_values function of HyperParameter.
        """
        hyperparameter = HyperParameter(
            name=self.name,
            nparams=self.nparams,
            values=self.values,
            bounds=self.bounds,
            bkd=self.bkd(),
        )
        self.bkd().assert_allclose(hyperparameter.get_values(), self.values)

    def test_get_bounds(self) -> None:
        """
        Test retrieving the bounds for HyperParameter.
        """
        hyperparameter = HyperParameter(
            name=self.name,
            nparams=self.nparams,
            values=self.values,
            bounds=self.bounds,
            bkd=self.bkd(),
        )
        self.bkd().assert_allclose(hyperparameter.get_bounds(), self.bounds)

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
        )
        active_indices = self.bkd().array([0, 2])
        hyperparameter.set_active_indices(active_indices)
        self.bkd().assert_allclose(
            hyperparameter.get_active_indices(), active_indices
        )

    def test_set_all_active(self) -> None:
        """
        Test setting all parameters to active for HyperParameter.
        """
        hyperparameter = HyperParameter(
            name=self.name,
            nparams=self.nparams,
            values=self.values,
            bounds=self.bounds,
            bkd=self.bkd(),
        )
        hyperparameter.set_all_active()
        self.bkd().assert_allclose(
            hyperparameter.get_active_indices(),
            self.bkd().arange(self.nparams, dtype=int),
        )

    def test_set_all_inactive(self) -> None:
        """
        Test setting all parameters to inactive for HyperParameter.
        """
        hyperparameter = HyperParameter(
            name=self.name,
            nparams=self.nparams,
            values=self.values,
            bounds=self.bounds,
            bkd=self.bkd(),
        )
        hyperparameter.set_all_inactive()
        self.assertEqual(hyperparameter.get_active_indices().shape[0], 0)

    def test_get_active_values(self) -> None:
        """
        Test retrieving active values for HyperParameter.
        """
        hyperparameter = HyperParameter(
            name=self.name,
            nparams=self.nparams,
            values=self.values,
            bounds=self.bounds,
            bkd=self.bkd(),
        )
        active_indices = self.bkd().array([0, 2])
        hyperparameter.set_active_indices(active_indices)
        self.bkd().assert_allclose(
            hyperparameter.get_active_values(),
            self.bkd().array([1.0, 3.0]),
        )

    def test_set_active_values(self) -> None:
        """
        Test setting active values for HyperParameter.
        """
        hyperparameter = HyperParameter(
            name=self.name,
            nparams=self.nparams,
            values=self.values,
            bounds=self.bounds,
            bkd=self.bkd(),
        )
        active_indices = self.bkd().array([0, 2])
        hyperparameter.set_active_indices(active_indices)
        new_active_values = self.bkd().array([7.0, 8.0])
        hyperparameter.set_active_values(new_active_values)
        self.bkd().assert_allclose(
            hyperparameter.get_active_values(),
            new_active_values,
        )


class TestHyperParameterNumpy(TestHyperParameter[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestHyperParameterTorch(TestHyperParameter[torch.Tensor]):
    def setUp(self) -> None:
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> TorchBkd:
        return self._bkd


from pyapprox.typing.util.test_utils import load_tests


if __name__ == "__main__":
    unittest.main()
