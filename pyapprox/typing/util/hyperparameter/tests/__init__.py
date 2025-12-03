import unittest
from typing import Generic, Any

import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.typing.util.backend import Backend, Array
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.abstracttestcase import AbstractTestCase
from pyapprox.typing.optimization.hyperparameter import HyperParameter
from pyapprox.typing.optimization.hyperparameterlist import HyperParameterList
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
        self.values = np.array([1.0, 2.0, 3.0])
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
        self.assertTrue(np.allclose(hyperparameter.get_values(), self.values))
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
        self.assertTrue(np.allclose(hyperparameter.get_values(), new_values))

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


class TestHyperParameterList(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up the test environment for HyperParameterList.
        """
        self.hyperparameter1 = HyperParameter(
            name="param1",
            nparams=3,
            values=np.array([1.0, 2.0, 3.0]),
            bounds=(0.0, 5.0),
            bkd=self.bkd(),
            transform=IdentityHyperParameterTransform(self.bkd()),
        )
        self.hyperparameter2 = HyperParameter(
            name="param2",
            nparams=2,
            values=np.array([4.0, 5.0]),
            bounds=(1.0, 10.0),
            bkd=self.bkd(),
            transform=IdentityHyperParameterTransform(self.bkd()),
        )
        self.hyperparameter_list = HyperParameterList(
            [self.hyperparameter1, self.hyperparameter2]
        )

    def test_initialization(self) -> None:
        """
        Test the initialization of HyperParameterList.
        """
        self.assertEqual(len(self.hyperparameter_list.hyperparameters()), 2)
        self.assertEqual(self.hyperparameter_list.nvars(), 5)

    def test_get_values(self) -> None:
        """
        Test getting values from HyperParameterList.
        """
        expected_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertTrue(
            np.allclose(self.hyperparameter_list.get_values(), expected_values)
        )

    def test_set_values(self) -> None:
        """
        Test setting values for HyperParameterList.
        """
        new_values = np.array([6.0, 7.0, 8.0, 9.0, 10.0])
        self.hyperparameter_list.set_values(new_values)
        self.assertTrue(
            np.allclose(self.hyperparameter_list.get_values(), new_values)
        )

    def test_get_bounds(self) -> None:
        """
        Test getting bounds from HyperParameterList.
        """
        expected_bounds = np.array(
            [0.0, 5.0, 0.0, 5.0, 0.0, 5.0, 1.0, 10.0, 1.0, 10.0]
        )
        self.assertTrue(
            np.allclose(self.hyperparameter_list.get_bounds(), expected_bounds)
        )

    def test_active_indices(self) -> None:
        """
        Test setting and getting active indices for HyperParameterList.
        """
        active_indices = np.array([0, 1, 3])
        self.hyperparameter_list.set_active_indices(active_indices)
        self.assertTrue(
            np.allclose(
                self.hyperparameter_list.get_active_indices(), active_indices
            )
        )

    def test_addition(self) -> None:
        """
        Test addition of HyperParameterLists.
        """
        hyperparameter3 = HyperParameter(
            name="param3",
            nparams=1,
            values=np.array([11.0]),
            bounds=(0.0, 15.0),
            bkd=self.bkd(),
            transform=IdentityHyperParameterTransform(self.bkd()),
        )
        hyperparameter_list2 = HyperParameterList([hyperparameter3])
        combined_list = self.hyperparameter_list + hyperparameter_list2
        self.assertEqual(len(combined_list.hyperparameters()), 3)
        self.assertEqual(combined_list.nvars(), 6)


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
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


if __name__ == "__main__":
    unittest.main()
