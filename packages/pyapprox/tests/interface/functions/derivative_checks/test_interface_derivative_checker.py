from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.hessian import (
    FunctionWithJacobianAndHVPFromCallable,
)


class TestDerivativeChecker:
    def test_derivative_checker(self, bkd) -> None:
        """
        Test the derivative checker for a simple quadratic function.
        """

        # Define the value function
        def value_function(x):
            return bkd.reshape(x[0] ** 3 + x[1] ** 2, (1, x.shape[1]))

        def jacobian_function(x):
            return bkd.stack([3 * x[0] ** 2, 2 * x[1]], axis=1)

        def hvp_function(x, v):
            return bkd.stack([6 * x[0] * v[0], 2 * v[1]], axis=0)

        # Wrap the function using FunctionWithJacobianAndHVPFromCallable
        function_object = FunctionWithJacobianAndHVPFromCallable(
            nvars=2,
            fun=value_function,
            jacobian=jacobian_function,
            hvp=hvp_function,
            bkd=bkd,
        )

        # Initialize DerivativeChecker
        checker = DerivativeChecker(function_object)

        # Define a sample point
        sample = bkd.asarray([[2.0, 1.0]]).T

        # Check derivatives
        errors = checker.check_derivatives(sample)

        # Assert that the gradient errors are below a tolerance
        assert checker.error_ratio(errors[0]) <= 1e-5

        # Assert that the Hessian errors are below a tolerance
        assert checker.error_ratio(errors[1]) <= 1e-5
