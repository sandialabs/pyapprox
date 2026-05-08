import numpy as np

from pyapprox.util.hyperparameter.cholesky_hyperparameter import (
    CholeskyHyperParameter,
)


class TestCholeskyHyperParameter:
    """
    Base test class for CholeskyHyperParameter.

    Derived classes must implement the bkd() method to provide the backend.
    """

    def _make_params(self, bkd):
        """Set up the test data for CholeskyHyperParameter."""
        name = "cholesky_param"
        nrows = 2
        user_values = bkd.array([[1.0, 0.0], [0.5, 1.0]])
        user_bounds = bkd.array(
            [[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, -np.inf]]
        )
        return name, nrows, user_values, user_bounds

    def test_get_values(self, bkd) -> None:
        """
        Test the get_values function of CholeskyHyperParameter.
        """
        name, nrows, user_values, user_bounds = self._make_params(bkd)
        cholesky_hyperparameter = CholeskyHyperParameter(
            name=name,
            nrows=nrows,
            user_values=user_values,
            user_bounds=user_bounds,
            bkd=bkd,
        )
        bkd.assert_allclose(
            cholesky_hyperparameter.get_values(),
            bkd.array([1.0, 0.5, 1.0]),
        )

    def test_get_cholesky_factor(self, bkd) -> None:
        """
        Test retrieving the full Cholesky factor for CholeskyHyperParameter.
        """
        name, nrows, user_values, user_bounds = self._make_params(bkd)
        cholesky_hyperparameter = CholeskyHyperParameter(
            name=name,
            nrows=nrows,
            user_values=user_values,
            user_bounds=user_bounds,
            bkd=bkd,
        )
        bkd.assert_allclose(cholesky_hyperparameter.factor(), user_values)
