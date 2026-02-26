from pyapprox.util.hyperparameter.log_hyperparameter import (
    LogHyperParameter,
)


class TestLogHyperParameter:
    """
    Base test class for LogHyperParameter.

    Derived classes must implement the bkd() method to provide the backend.
    """

    def _make_params(self, bkd):
        """Set up the test data for LogHyperParameter."""
        name = "log_param"
        nparams = 3
        user_values = bkd.array([1.0, 2.0, 3.0])
        user_bounds = (0.1, 10.0)
        return name, nparams, user_values, user_bounds

    def test_get_values(self, bkd) -> None:
        """
        Test retrieving log-transformed values for LogHyperParameter.
        """
        name, nparams, user_values, user_bounds = self._make_params(bkd)
        log_hyperparameter = LogHyperParameter(
            name=name,
            nparams=nparams,
            user_values=user_values,
            user_bounds=user_bounds,
            bkd=bkd,
        )
        bkd.assert_allclose(
            log_hyperparameter.get_values(),
            bkd.log(user_values),
        )

    def test_exp_values(self, bkd) -> None:
        """
        Test retrieving exponential values for LogHyperParameter.
        """
        name, nparams, user_values, user_bounds = self._make_params(bkd)
        log_hyperparameter = LogHyperParameter(
            name=name,
            nparams=nparams,
            user_values=user_values,
            user_bounds=user_bounds,
            bkd=bkd,
        )
        bkd.assert_allclose(
            log_hyperparameter.exp_values(),
            user_values,
        )
