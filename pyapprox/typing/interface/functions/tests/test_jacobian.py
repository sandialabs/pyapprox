import unittest
import numpy as np
from pyapprox.typing.util.backend import Backend
from pyapprox.typing.interface.functions.jacobian import (
    FunctionWithJacobianFromCallable,
    validate_jacobian,
    validate_jacobians,
)


class TestJacobian(unittest.TestCase):
    def setUp(self):
        self.nqoi = 2
        self.nvars = 3
        self.samples = np.random.rand(self.nvars, 5)
        self.backend = Backend()

    def test_validate_jacobian(self):
        jac = np.random.rand(self.nqoi, self.nvars)
        validate_jacobian(
            self.nqoi, self.nvars, jac
        )  # Should not raise an error

        with self.assertRaises(ValueError):
            invalid_jac = np.random.rand(self.nqoi + 1, self.nvars)
            validate_jacobian(self.nqoi, self.nvars, invalid_jac)

    def test_validate_jacobians(self):
        jac = np.random.rand(self.samples.shape[1], self.nqoi, self.nvars)
        validate_jacobians(
            self.nqoi, self.nvars, self.samples, jac
        )  # Should not raise an error

        with self.assertRaises(ValueError):
            invalid_jac = np.random.rand(
                self.samples.shape[1], self.nqoi + 1, self.nvars
            )
            validate_jacobians(
                self.nqoi, self.nvars, self.samples, invalid_jac
            )

    def test_function_with_jacobian_from_callable(self):
        def fun(samples):
            return np.sum(samples, axis=0)

        def jacobian(samples):
            return np.ones((self.nqoi, self.nvars))

        function = FunctionWithJacobianFromCallable(
            self.nqoi, self.nvars, fun, jacobian, self.backend
        )

        self.assertEqual(function.nvars(), self.nvars)
        self.assertEqual(function.nqoi(), self.nqoi)

        jac = function.jacobian(self.samples)
        self.assertEqual(jac.shape, (self.nqoi, self.nvars))

        jacs = function.jacobians(self.samples)
        self.assertEqual(
            jacs.shape, (self.samples.shape[1], self.nqoi, self.nvars)
        )


if __name__ == "__main__":
    unittest.main()
