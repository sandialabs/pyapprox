from pyapprox.util.hyperparameter.hyperparameter import HyperParameter


class TestHyperParameter:
    """
    Base test class for HyperParameter.

    Derived classes must implement the bkd() method to provide the backend.
    """

    def _make_params(self, bkd):
        """Set up the test data for HyperParameter."""
        name = "example_param"
        nparams = 3
        values = bkd.array([1.0, 2.0, 3.0])
        bounds = bkd.array([[0.1, 10.0], [0.1, 10.0], [0.1, 10.0]])
        return name, nparams, values, bounds

    def test_initialization(self, bkd) -> None:
        """
        Test the initialization of HyperParameter.
        """
        name, nparams, values, bounds = self._make_params(bkd)
        hyperparameter = HyperParameter(
            name=name,
            nparams=nparams,
            values=values,
            bounds=bounds,
            bkd=bkd,
        )
        assert hyperparameter.nparams() == nparams
        bkd.assert_allclose(hyperparameter.get_values(), values)
        bkd.assert_allclose(hyperparameter.get_bounds(), bounds)

    def test_get_values(self, bkd) -> None:
        """
        Test the get_values function of HyperParameter.
        """
        name, nparams, values, bounds = self._make_params(bkd)
        hyperparameter = HyperParameter(
            name=name,
            nparams=nparams,
            values=values,
            bounds=bounds,
            bkd=bkd,
        )
        bkd.assert_allclose(hyperparameter.get_values(), values)

    def test_get_bounds(self, bkd) -> None:
        """
        Test retrieving the bounds for HyperParameter.
        """
        name, nparams, values, bounds = self._make_params(bkd)
        hyperparameter = HyperParameter(
            name=name,
            nparams=nparams,
            values=values,
            bounds=bounds,
            bkd=bkd,
        )
        bkd.assert_allclose(hyperparameter.get_bounds(), bounds)

    def test_active_indices(self, bkd) -> None:
        """
        Test setting and getting active indices for HyperParameter.
        """
        name, nparams, values, bounds = self._make_params(bkd)
        hyperparameter = HyperParameter(
            name=name,
            nparams=nparams,
            values=values,
            bounds=bounds,
            bkd=bkd,
        )
        active_indices = bkd.array([0, 2])
        hyperparameter.set_active_indices(active_indices)
        bkd.assert_allclose(hyperparameter.get_active_indices(), active_indices)

    def test_set_all_active(self, bkd) -> None:
        """
        Test setting all parameters to active for HyperParameter.
        """
        name, nparams, values, bounds = self._make_params(bkd)
        hyperparameter = HyperParameter(
            name=name,
            nparams=nparams,
            values=values,
            bounds=bounds,
            bkd=bkd,
        )
        hyperparameter.set_all_active()
        bkd.assert_allclose(
            hyperparameter.get_active_indices(),
            bkd.arange(nparams, dtype=int),
        )

    def test_set_all_inactive(self, bkd) -> None:
        """
        Test setting all parameters to inactive for HyperParameter.
        """
        name, nparams, values, bounds = self._make_params(bkd)
        hyperparameter = HyperParameter(
            name=name,
            nparams=nparams,
            values=values,
            bounds=bounds,
            bkd=bkd,
        )
        hyperparameter.set_all_inactive()
        assert hyperparameter.get_active_indices().shape[0] == 0

    def test_get_active_values(self, bkd) -> None:
        """
        Test retrieving active values for HyperParameter.
        """
        name, nparams, values, bounds = self._make_params(bkd)
        hyperparameter = HyperParameter(
            name=name,
            nparams=nparams,
            values=values,
            bounds=bounds,
            bkd=bkd,
        )
        active_indices = bkd.array([0, 2])
        hyperparameter.set_active_indices(active_indices)
        bkd.assert_allclose(
            hyperparameter.get_active_values(),
            bkd.array([1.0, 3.0]),
        )

    def test_set_active_values(self, bkd) -> None:
        """
        Test setting active values for HyperParameter.
        """
        name, nparams, values, bounds = self._make_params(bkd)
        hyperparameter = HyperParameter(
            name=name,
            nparams=nparams,
            values=values,
            bounds=bounds,
            bkd=bkd,
        )
        active_indices = bkd.array([0, 2])
        hyperparameter.set_active_indices(active_indices)
        new_active_values = bkd.array([7.0, 8.0])
        hyperparameter.set_active_values(new_active_values)
        bkd.assert_allclose(
            hyperparameter.get_active_values(),
            new_active_values,
        )
