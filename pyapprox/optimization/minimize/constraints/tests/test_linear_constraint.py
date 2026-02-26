import numpy as np
from scipy.optimize import LinearConstraint as ScipyLinearConstraint

from pyapprox.optimization.minimize.constraints.linear import (
    PyApproxLinearConstraint,
)


class TestPyApproxLinearConstraint:

    def test_generic_linear_constraint(self, bkd) -> None:
        """
        Test the PyApproxLinearConstraint class.
        """
        # Define coefficient matrix and bounds
        A = bkd.asarray([[1.0, 2.0], [3.0, 4.0]])
        lb = bkd.asarray([0.0, 1.0])
        ub = bkd.asarray([5.0, 6.0])

        # Create a generic linear constraint
        constraint = PyApproxLinearConstraint(A, lb, ub, bkd)

        # Assert that the backend is correct
        assert constraint.bkd() == bkd

        # Assert that the coefficient matrix matches the expected values
        np.testing.assert_allclose(
            bkd.to_numpy(constraint.A()), [[1.0, 2.0], [3.0, 4.0]]
        )

        # Assert that the lower bounds match the expected values
        np.testing.assert_allclose(bkd.to_numpy(constraint.lb()), [0.0, 1.0])

        # Assert that the upper bounds match the expected values
        np.testing.assert_allclose(bkd.to_numpy(constraint.ub()), [5.0, 6.0])

        # Convert to SciPy LinearConstraint
        scipy_constraint = constraint.to_scipy()

        # Assert that the converted constraint matches the expected values
        assert isinstance(scipy_constraint, ScipyLinearConstraint)
        np.testing.assert_allclose(
            scipy_constraint.A, np.asarray([[1.0, 2.0], [3.0, 4.0]])
        )
        np.testing.assert_allclose(scipy_constraint.lb, [0.0, 1.0])
        np.testing.assert_allclose(scipy_constraint.ub, [5.0, 6.0])
