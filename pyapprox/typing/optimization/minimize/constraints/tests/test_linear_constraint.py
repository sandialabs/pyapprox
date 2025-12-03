import unittest
from typing import Generic, Any

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.abstracttestcase import AbstractTestCase
from scipy.optimize import LinearConstraint as ScipyLinearConstraint
from pyapprox.typing.optimization.minimize.constraints.linear import (
    PyApproxLinearConstraint,
)


class TestPyApproxLinearConstraint(Generic[Array], AbstractTestCase):
    def bkd(self) -> Backend[Array]:
        """
        Override this method in derived classes to provide the specific
        backend.
        """
        raise NotImplementedError(
            "Derived classes must implement this method."
        )

    def test_generic_linear_constraint(self) -> None:
        """
        Test the PyApproxLinearConstraint class.
        """
        bkd = self.bkd()

        # Define coefficient matrix and bounds
        A = bkd.asarray([[1.0, 2.0], [3.0, 4.0]])
        lb = bkd.asarray([0.0, 1.0])
        ub = bkd.asarray([5.0, 6.0])

        # Create a generic linear constraint
        constraint = PyApproxLinearConstraint(A, lb, ub, bkd)

        # Assert that the backend is correct
        self.assertEqual(constraint.bkd(), bkd)

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
        self.assertIsInstance(scipy_constraint, ScipyLinearConstraint)
        np.testing.assert_allclose(
            scipy_constraint.A, np.asarray([[1.0, 2.0], [3.0, 4.0]])
        )
        np.testing.assert_allclose(scipy_constraint.lb, [0.0, 1.0])
        np.testing.assert_allclose(scipy_constraint.ub, [5.0, 6.0])


class TestPyApproxLinearConstraintNumpy(
    TestPyApproxLinearConstraint[NDArray[Any]], unittest.TestCase
):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestPyApproxLinearConstraintTorch(
    TestPyApproxLinearConstraint[torch.Tensor], unittest.TestCase
):
    def setUp(self) -> None:
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


if __name__ == "__main__":
    unittest.main()
