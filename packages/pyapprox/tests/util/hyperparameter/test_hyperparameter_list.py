import numpy as np

from pyapprox.util.hyperparameter import (
    HyperParameter,
    LogHyperParameter,
)
from pyapprox.util.hyperparameter.hyperparameter_list import (
    HyperParameterList,
)


class TestHyperParameterList:

    def _make_list(self, bkd):
        """Set up the test environment for HyperParameterList."""
        hyperparameter1 = HyperParameter(
            name="param1",
            nparams=3,
            values=bkd.array([1.0, 2.0, 3.0]),
            bounds=(0.0, 5.0),
            bkd=bkd,
        )
        hyperparameter2 = LogHyperParameter(
            name="param2",
            nparams=2,
            user_values=bkd.array([4.0, 5.0]),
            user_bounds=(1.0, 10.0),
            bkd=bkd,
        )
        hyperparameter_list = HyperParameterList(
            [hyperparameter1, hyperparameter2]
        )
        return hyperparameter1, hyperparameter2, hyperparameter_list

    def test_initialization(self, bkd) -> None:
        """
        Test the initialization of HyperParameterList.
        """
        _, _, hyperparameter_list = self._make_list(bkd)
        assert len(hyperparameter_list.hyperparameters()) == 2
        assert hyperparameter_list.nparams() == 5

    def test_get_values(self, bkd) -> None:
        """
        Test getting values from HyperParameterList.
        """
        _, _, hyperparameter_list = self._make_list(bkd)
        expected_values = bkd.array([1.0, 2.0, 3.0, np.log(4.0), np.log(5.0)])
        bkd.assert_allclose(
            hyperparameter_list.get_values(), expected_values
        )

    def test_get_bounds(self, bkd) -> None:
        """
        Test getting bounds from HyperParameterList.
        """
        _, _, hyperparameter_list = self._make_list(bkd)
        expected_bounds = bkd.vstack(
            (
                bkd.array([[0.0, 5.0], [0.0, 5.0], [0.0, 5.0]]),
                bkd.log(bkd.array([[1.0, 10.0], [1.0, 10.0]])),
            )
        )
        bkd.assert_allclose(
            hyperparameter_list.get_bounds(), expected_bounds
        )

    def test_active_indices(self, bkd) -> None:
        """
        Test setting and getting active indices for HyperParameterList.
        """
        _, _, hyperparameter_list = self._make_list(bkd)
        active_indices = bkd.array([0, 1, 3])
        hyperparameter_list.set_active_indices(active_indices)
        bkd.assert_allclose(
            hyperparameter_list.get_active_indices(), active_indices
        )

    def test_addition(self, bkd) -> None:
        """
        Test addition of HyperParameterLists.
        """
        _, _, hyperparameter_list = self._make_list(bkd)
        hyperparameter3 = HyperParameter(
            name="param3",
            nparams=1,
            values=bkd.array([11.0]),
            bounds=(0.0, 15.0),
            bkd=bkd,
        )
        hyperparameter_list2 = HyperParameterList([hyperparameter3])
        combined_list = hyperparameter_list + hyperparameter_list2
        assert len(combined_list.hyperparameters()) == 3
        assert combined_list.nparams() == 6

    def test_get_active_bounds(self, bkd) -> None:
        """
        Test getting bounds for active parameters only.
        """
        _, _, hyperparameter_list = self._make_list(bkd)
        # Set only indices 0, 2, 4 as active (skipping 1, 3)
        active_indices = bkd.array([0, 2, 4])
        hyperparameter_list.set_active_indices(active_indices)

        active_bounds = hyperparameter_list.get_active_bounds()
        assert active_bounds.shape == (3, 2)

        # Check first active param (index 0 from param1)
        bkd.assert_allclose(active_bounds[0], bkd.array([0.0, 5.0]))
        # Check second active param (index 2 from param1)
        bkd.assert_allclose(active_bounds[1], bkd.array([0.0, 5.0]))
        # Check third active param (index 4, which is index 1 of param2 - log bounds)
        expected_log_bounds = bkd.log(bkd.array([1.0, 10.0]))
        bkd.assert_allclose(active_bounds[2], expected_log_bounds)

    def test_extract_active_1d(self, bkd) -> None:
        """
        Test extracting active elements from a 1D array (e.g., gradient).
        """
        _, _, hyperparameter_list = self._make_list(bkd)
        # Set only indices 0, 2, 4 as active
        active_indices = bkd.array([0, 2, 4])
        hyperparameter_list.set_active_indices(active_indices)

        # Create a full gradient with all 5 parameters
        full_grad = bkd.array([10.0, 20.0, 30.0, 40.0, 50.0])

        # Extract active elements
        active_grad = hyperparameter_list.extract_active(full_grad)

        # Should get elements at indices 0, 2, 4
        expected = bkd.array([10.0, 30.0, 50.0])
        bkd.assert_allclose(active_grad, expected)

    def test_extract_active_multidim(self, bkd) -> None:
        """
        Test extracting active elements from a multi-dimensional array.

        This is useful for extracting active columns from kernel jacobians
        with shape (n, n, nparams).
        """
        _, _, hyperparameter_list = self._make_list(bkd)
        # Set only indices 1, 3 as active
        active_indices = bkd.array([1, 3])
        hyperparameter_list.set_active_indices(active_indices)

        # Create a 3D array simulating kernel jacobian (2, 2, 5)
        full_array = bkd.arange(20.0).reshape((2, 2, 5))

        # Extract active elements along last axis
        active_array = hyperparameter_list.extract_active(full_array)

        # Should have shape (2, 2, 2) - extracting params 1 and 3
        assert active_array.shape == (2, 2, 2)
        # Check values
        bkd.assert_allclose(active_array[:, :, 0], full_array[:, :, 1])
        bkd.assert_allclose(active_array[:, :, 1], full_array[:, :, 3])

    def test_expand_to_full(self, bkd) -> None:
        """
        Test expanding active parameter array to full parameter space.
        """
        _, _, hyperparameter_list = self._make_list(bkd)
        # Set only indices 0, 2, 4 as active
        active_indices = bkd.array([0, 2, 4])
        hyperparameter_list.set_active_indices(active_indices)

        # Create an active array with 3 elements
        active_array = bkd.array([1.0, 2.0, 3.0])

        # Expand to full space (5 params), fixed params get 0
        full_array = hyperparameter_list.expand_to_full(active_array)

        # Should have shape (5,) with zeros at indices 1, 3
        expected = bkd.array([1.0, 0.0, 2.0, 0.0, 3.0])
        bkd.assert_allclose(full_array, expected)

    def test_expand_to_full_custom_fill(self, bkd) -> None:
        """
        Test expanding with custom fill value for fixed parameters.
        """
        _, _, hyperparameter_list = self._make_list(bkd)
        # Set only indices 1, 2 as active
        active_indices = bkd.array([1, 2])
        hyperparameter_list.set_active_indices(active_indices)

        active_array = bkd.array([10.0, 20.0])
        full_array = hyperparameter_list.expand_to_full(
            active_array, fill_value=-1.0
        )

        expected = bkd.array([-1.0, 10.0, 20.0, -1.0, -1.0])
        bkd.assert_allclose(full_array, expected)

    def test_fixed_hyperparameter(self, bkd) -> None:
        """
        Test that fixed=True makes hyperparameter inactive.
        """
        # Create a fixed hyperparameter
        fixed_param = HyperParameter(
            name="fixed_param",
            nparams=2,
            values=bkd.array([1.0, 2.0]),
            bounds=(0.0, 5.0),
            bkd=bkd,
            fixed=True,
        )
        # Create an active hyperparameter
        active_param = HyperParameter(
            name="active_param",
            nparams=3,
            values=bkd.array([3.0, 4.0, 5.0]),
            bounds=(0.0, 10.0),
            bkd=bkd,
            fixed=False,
        )

        hyp_list = HyperParameterList([fixed_param, active_param])

        # Total params = 5, active params = 3
        assert hyp_list.nparams() == 5
        assert hyp_list.nactive_params() == 3

        # Active indices should be [2, 3, 4] (the active_param indices)
        expected_active = bkd.array([2, 3, 4])
        bkd.assert_allclose(hyp_list.get_active_indices(), expected_active)

        # Extract active should get only the active_param values
        full_values = hyp_list.get_values()
        active_values = hyp_list.extract_active(full_values)
        bkd.assert_allclose(active_values, bkd.array([3.0, 4.0, 5.0]))

    def test_mixed_fixed_active_gradient_extraction(self, bkd) -> None:
        """
        Test gradient extraction with mixed fixed/active parameters.

        This simulates the use case of optimizing GP with some fixed
        hyperparameters (e.g., noise) and some active (e.g., length scales).
        """
        # Simulate: 2 noise params (fixed) + 3 length scale params (active)
        noise_param = HyperParameter(
            name="noise",
            nparams=2,
            values=bkd.array([0.1, 0.2]),
            bounds=(0.01, 1.0),
            bkd=bkd,
            fixed=True,
        )
        lenscale_param = HyperParameter(
            name="lenscale",
            nparams=3,
            values=bkd.array([1.0, 1.0, 1.0]),
            bounds=(0.1, 10.0),
            bkd=bkd,
            fixed=False,
        )

        hyp_list = HyperParameterList([noise_param, lenscale_param])

        # Simulate a full gradient (5 elements)
        full_grad = bkd.array([0.5, 0.6, 1.0, 2.0, 3.0])

        # Extract active gradient (should be last 3 elements)
        active_grad = hyp_list.extract_active(full_grad)
        expected = bkd.array([1.0, 2.0, 3.0])
        bkd.assert_allclose(active_grad, expected)

        # Expand a direction vector for HVP
        active_dir = bkd.array([0.1, 0.2, 0.3])
        full_dir = hyp_list.expand_to_full(active_dir)
        expected_full = bkd.array([0.0, 0.0, 0.1, 0.2, 0.3])
        bkd.assert_allclose(full_dir, expected_full)
