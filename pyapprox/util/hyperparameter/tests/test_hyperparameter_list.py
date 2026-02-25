import unittest
from typing import Generic, Any

import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.util.backends.protocols import Backend, Array
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.hyperparameter import (
    HyperParameter,
    LogHyperParameter,
)
from pyapprox.util.hyperparameter.hyperparameter_list import (
    HyperParameterList,
)


class TestHyperParameterList(Generic[Array], unittest.TestCase):
    __test__ = False

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
        Set up the test environment for HyperParameterList.
        """
        self.hyperparameter1 = HyperParameter(
            name="param1",
            nparams=3,
            values=self.bkd().array([1.0, 2.0, 3.0]),
            bounds=(0.0, 5.0),
            bkd=self.bkd(),
        )
        self.hyperparameter2 = LogHyperParameter(
            name="param2",
            nparams=2,
            user_values=self.bkd().array([4.0, 5.0]),
            user_bounds=(1.0, 10.0),
            bkd=self.bkd(),
        )
        self.hyperparameter_list = HyperParameterList(
            [self.hyperparameter1, self.hyperparameter2]
        )

    def test_initialization(self) -> None:
        """
        Test the initialization of HyperParameterList.
        """
        self.assertEqual(len(self.hyperparameter_list.hyperparameters()), 2)
        self.assertEqual(self.hyperparameter_list.nparams(), 5)

    def test_get_values(self) -> None:
        """
        Test getting values from HyperParameterList.
        """
        expected_values = self.bkd().array(
            [1.0, 2.0, 3.0, np.log(4.0), np.log(5.0)]
        )
        self.bkd().assert_allclose(
            self.hyperparameter_list.get_values(), expected_values
        )

    def test_get_bounds(self) -> None:
        """
        Test getting bounds from HyperParameterList.
        """
        expected_bounds = self.bkd().vstack(
            (
                self.bkd().array([[0.0, 5.0], [0.0, 5.0], [0.0, 5.0]]),
                self.bkd().log(self.bkd().array([[1.0, 10.0], [1.0, 10.0]])),
            )
        )
        self.bkd().assert_allclose(
            self.hyperparameter_list.get_bounds(), expected_bounds
        )

    def test_active_indices(self) -> None:
        """
        Test setting and getting active indices for HyperParameterList.
        """
        active_indices = self.bkd().array([0, 1, 3])
        self.hyperparameter_list.set_active_indices(active_indices)
        self.bkd().assert_allclose(
            self.hyperparameter_list.get_active_indices(), active_indices
        )

    def test_addition(self) -> None:
        """
        Test addition of HyperParameterLists.
        """
        hyperparameter3 = HyperParameter(
            name="param3",
            nparams=1,
            values=self.bkd().array([11.0]),
            bounds=(0.0, 15.0),
            bkd=self.bkd(),
        )
        hyperparameter_list2 = HyperParameterList([hyperparameter3])
        combined_list = self.hyperparameter_list + hyperparameter_list2
        self.assertEqual(len(combined_list.hyperparameters()), 3)
        self.assertEqual(combined_list.nparams(), 6)

    def test_get_active_bounds(self) -> None:
        """
        Test getting bounds for active parameters only.
        """
        # Set only indices 0, 2, 4 as active (skipping 1, 3)
        active_indices = self.bkd().array([0, 2, 4])
        self.hyperparameter_list.set_active_indices(active_indices)

        active_bounds = self.hyperparameter_list.get_active_bounds()
        self.assertEqual(active_bounds.shape, (3, 2))

        # Check first active param (index 0 from param1)
        self.bkd().assert_allclose(active_bounds[0], self.bkd().array([0.0, 5.0]))
        # Check second active param (index 2 from param1)
        self.bkd().assert_allclose(active_bounds[1], self.bkd().array([0.0, 5.0]))
        # Check third active param (index 4, which is index 1 of param2 - log bounds)
        expected_log_bounds = self.bkd().log(self.bkd().array([1.0, 10.0]))
        self.bkd().assert_allclose(active_bounds[2], expected_log_bounds)

    def test_extract_active_1d(self) -> None:
        """
        Test extracting active elements from a 1D array (e.g., gradient).
        """
        # Set only indices 0, 2, 4 as active
        active_indices = self.bkd().array([0, 2, 4])
        self.hyperparameter_list.set_active_indices(active_indices)

        # Create a full gradient with all 5 parameters
        full_grad = self.bkd().array([10.0, 20.0, 30.0, 40.0, 50.0])

        # Extract active elements
        active_grad = self.hyperparameter_list.extract_active(full_grad)

        # Should get elements at indices 0, 2, 4
        expected = self.bkd().array([10.0, 30.0, 50.0])
        self.bkd().assert_allclose(active_grad, expected)

    def test_extract_active_multidim(self) -> None:
        """
        Test extracting active elements from a multi-dimensional array.

        This is useful for extracting active columns from kernel jacobians
        with shape (n, n, nparams).
        """
        # Set only indices 1, 3 as active
        active_indices = self.bkd().array([1, 3])
        self.hyperparameter_list.set_active_indices(active_indices)

        # Create a 3D array simulating kernel jacobian (2, 2, 5)
        full_array = self.bkd().arange(20.0).reshape((2, 2, 5))

        # Extract active elements along last axis
        active_array = self.hyperparameter_list.extract_active(full_array)

        # Should have shape (2, 2, 2) - extracting params 1 and 3
        self.assertEqual(active_array.shape, (2, 2, 2))
        # Check values
        self.bkd().assert_allclose(active_array[:, :, 0], full_array[:, :, 1])
        self.bkd().assert_allclose(active_array[:, :, 1], full_array[:, :, 3])

    def test_expand_to_full(self) -> None:
        """
        Test expanding active parameter array to full parameter space.
        """
        # Set only indices 0, 2, 4 as active
        active_indices = self.bkd().array([0, 2, 4])
        self.hyperparameter_list.set_active_indices(active_indices)

        # Create an active array with 3 elements
        active_array = self.bkd().array([1.0, 2.0, 3.0])

        # Expand to full space (5 params), fixed params get 0
        full_array = self.hyperparameter_list.expand_to_full(active_array)

        # Should have shape (5,) with zeros at indices 1, 3
        expected = self.bkd().array([1.0, 0.0, 2.0, 0.0, 3.0])
        self.bkd().assert_allclose(full_array, expected)

    def test_expand_to_full_custom_fill(self) -> None:
        """
        Test expanding with custom fill value for fixed parameters.
        """
        # Set only indices 1, 2 as active
        active_indices = self.bkd().array([1, 2])
        self.hyperparameter_list.set_active_indices(active_indices)

        active_array = self.bkd().array([10.0, 20.0])
        full_array = self.hyperparameter_list.expand_to_full(
            active_array, fill_value=-1.0
        )

        expected = self.bkd().array([-1.0, 10.0, 20.0, -1.0, -1.0])
        self.bkd().assert_allclose(full_array, expected)

    def test_fixed_hyperparameter(self) -> None:
        """
        Test that fixed=True makes hyperparameter inactive.
        """
        # Create a fixed hyperparameter
        fixed_param = HyperParameter(
            name="fixed_param",
            nparams=2,
            values=self.bkd().array([1.0, 2.0]),
            bounds=(0.0, 5.0),
            bkd=self.bkd(),
            fixed=True,
        )
        # Create an active hyperparameter
        active_param = HyperParameter(
            name="active_param",
            nparams=3,
            values=self.bkd().array([3.0, 4.0, 5.0]),
            bounds=(0.0, 10.0),
            bkd=self.bkd(),
            fixed=False,
        )

        hyp_list = HyperParameterList([fixed_param, active_param])

        # Total params = 5, active params = 3
        self.assertEqual(hyp_list.nparams(), 5)
        self.assertEqual(hyp_list.nactive_params(), 3)

        # Active indices should be [2, 3, 4] (the active_param indices)
        expected_active = self.bkd().array([2, 3, 4])
        self.bkd().assert_allclose(hyp_list.get_active_indices(), expected_active)

        # Extract active should get only the active_param values
        full_values = hyp_list.get_values()
        active_values = hyp_list.extract_active(full_values)
        self.bkd().assert_allclose(active_values, self.bkd().array([3.0, 4.0, 5.0]))

    def test_mixed_fixed_active_gradient_extraction(self) -> None:
        """
        Test gradient extraction with mixed fixed/active parameters.

        This simulates the use case of optimizing GP with some fixed
        hyperparameters (e.g., noise) and some active (e.g., length scales).
        """
        # Simulate: 2 noise params (fixed) + 3 length scale params (active)
        noise_param = HyperParameter(
            name="noise",
            nparams=2,
            values=self.bkd().array([0.1, 0.2]),
            bounds=(0.01, 1.0),
            bkd=self.bkd(),
            fixed=True,
        )
        lenscale_param = HyperParameter(
            name="lenscale",
            nparams=3,
            values=self.bkd().array([1.0, 1.0, 1.0]),
            bounds=(0.1, 10.0),
            bkd=self.bkd(),
            fixed=False,
        )

        hyp_list = HyperParameterList([noise_param, lenscale_param])

        # Simulate a full gradient (5 elements)
        full_grad = self.bkd().array([0.5, 0.6, 1.0, 2.0, 3.0])

        # Extract active gradient (should be last 3 elements)
        active_grad = hyp_list.extract_active(full_grad)
        expected = self.bkd().array([1.0, 2.0, 3.0])
        self.bkd().assert_allclose(active_grad, expected)

        # Expand a direction vector for HVP
        active_dir = self.bkd().array([0.1, 0.2, 0.3])
        full_dir = hyp_list.expand_to_full(active_dir)
        expected_full = self.bkd().array([0.0, 0.0, 0.1, 0.2, 0.3])
        self.bkd().assert_allclose(full_dir, expected_full)


# Derived test class for NumPy backend
class TestHyperParameterListNumpy(TestHyperParameterList[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# Derived test class for PyTorch backend
class TestHyperParameterListTorch(TestHyperParameterList[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


from pyapprox.util.test_utils import load_tests


if __name__ == "__main__":
    unittest.main()
