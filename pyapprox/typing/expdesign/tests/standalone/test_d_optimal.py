"""
Standalone tests for DOptimalLinearModelObjective.

PERMANENT - no legacy imports.

These tests verify correctness using:
1. DerivativeChecker for Jacobian and Hessian
2. Self-consistent property checks (symmetry, shapes)
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401
from pyapprox.typing.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.typing.interface.functions.fromcallable.hessian import (
    FunctionWithJacobianAndHVPFromCallable,
)

from pyapprox.typing.expdesign.objective import DOptimalLinearModelObjective


class TestDOptimalStandalone(Generic[Array], unittest.TestCase):
    """Standalone tests for DOptimalLinearModelObjective."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(42)

        self._nobs = 5
        self._nparams = 3
        self._design_matrix = self._bkd.asarray(
            np.random.randn(self._nobs, self._nparams)
        )
        self._noise_cov = self._bkd.asarray(np.array(0.1))
        self._prior_cov = self._bkd.asarray(np.array(1.0))
        self._weights = self._bkd.ones((self._nobs, 1)) / self._nobs

    def _create_objective(self) -> DOptimalLinearModelObjective[Array]:
        return DOptimalLinearModelObjective(
            self._design_matrix,
            self._noise_cov,
            self._prior_cov,
            self._bkd,
        )

    def _create_derivative_checker_function(
        self, obj: DOptimalLinearModelObjective[Array]
    ):
        """Wrap objective for DerivativeChecker compatibility.

        DerivativeChecker expects:
        - __call__(samples): (nvars, nsamples) -> (nqoi, nsamples)
        - jacobian(sample): (nvars, 1) -> (nqoi, nvars)
        - hvp(sample, vec): (nvars, 1), (nvars, 1) -> (nvars, 1)

        DOptimalLinearModelObjective has:
        - __call__(weights): (nobs, 1) -> (1, 1)
        - jacobian(weights): (nobs, 1) -> (1, nobs)
        - hvp(weights, vec): (nobs, 1), (nobs, 1) -> (1, nobs)
        """

        def value_fun(samples: Array) -> Array:
            # samples: (nvars, nsamples) = (nobs, nsamples)
            nsamples = samples.shape[1]
            results = []
            for ii in range(nsamples):
                w = samples[:, ii : ii + 1]  # (nobs, 1)
                val = obj(w)  # (1, 1)
                results.append(val[0, 0])
            return self._bkd.reshape(
                self._bkd.stack(results), (1, nsamples)
            )

        def jacobian_fun(sample: Array) -> Array:
            # sample: (nvars, 1) = (nobs, 1)
            jac = obj.jacobian(sample)  # (1, nobs)
            return jac  # (nqoi=1, nvars=nobs)

        def hvp_fun(sample: Array, vec: Array) -> Array:
            # sample: (nvars, 1), vec: (nvars, 1)
            hvp = obj.hvp(sample, vec)  # (1, nobs)
            return hvp.T  # (nvars=nobs, 1)

        return FunctionWithJacobianAndHVPFromCallable(
            nvars=obj.nvars(),
            fun=value_fun,
            jacobian=jacobian_fun,
            hvp=hvp_fun,
            bkd=self._bkd,
        )

    def test_objective_shape(self):
        """Test objective returns correct shape."""
        obj = self._create_objective()
        result = obj(self._weights)
        self.assertEqual(result.shape, (1, 1))

    def test_jacobian_shape(self):
        """Test Jacobian returns correct shape."""
        obj = self._create_objective()
        jac = obj.jacobian(self._weights)
        self.assertEqual(jac.shape, (1, self._nobs))

    def test_hessian_shape(self):
        """Test Hessian returns correct shape."""
        obj = self._create_objective()
        hess = obj.hessian(self._weights)
        self.assertEqual(hess.shape, (1, self._nobs, self._nobs))

    def test_jacobian_derivative_checker(self):
        """Test Jacobian using DerivativeChecker."""
        obj = self._create_objective()
        wrapped = self._create_derivative_checker_function(obj)
        checker = DerivativeChecker(wrapped)

        # Check at uniform weights
        errors = checker.check_derivatives(self._weights)

        # Jacobian errors should show second-order convergence
        # (error_ratio ~ 0.25 for correct implementation)
        ratio = checker.error_ratio(errors[0])
        self.assertLessEqual(float(self._bkd.to_numpy(ratio)), 1e-6)

    def test_hvp_derivative_checker(self):
        """Test HVP (Hessian-vector product) using DerivativeChecker."""
        obj = self._create_objective()
        wrapped = self._create_derivative_checker_function(obj)
        checker = DerivativeChecker(wrapped)

        # Check at uniform weights
        errors = checker.check_derivatives(self._weights)

        # HVP errors should show second-order convergence
        ratio = checker.error_ratio(errors[1])
        self.assertLessEqual(float(self._bkd.to_numpy(ratio)), 1e-6)

    def test_hessian_symmetric(self):
        """Test Hessian is symmetric."""
        obj = self._create_objective()
        hess = obj.hessian(self._weights)

        # Extract the 2D matrix
        hess_2d = hess[0]
        self._bkd.assert_allclose(hess_2d, hess_2d.T, rtol=1e-12)

    def test_different_weights(self):
        """Test with non-uniform weights."""
        np.random.seed(123)
        weights = self._bkd.asarray(
            np.random.dirichlet(np.ones(self._nobs))[:, None]
        )

        obj = self._create_objective()
        result = obj(weights)
        jac = obj.jacobian(weights)
        hess = obj.hessian(weights)

        # Check shapes
        self.assertEqual(result.shape, (1, 1))
        self.assertEqual(jac.shape, (1, self._nobs))
        self.assertEqual(hess.shape, (1, self._nobs, self._nobs))

        # Check finite
        self.assertTrue(
            np.isfinite(self._bkd.to_numpy(result)).all()
        )
        self.assertTrue(
            np.isfinite(self._bkd.to_numpy(jac)).all()
        )
        self.assertTrue(
            np.isfinite(self._bkd.to_numpy(hess)).all()
        )

    def test_scalar_validation(self):
        """Test that non-scalar covariances raise error."""
        with self.assertRaises(TypeError):
            DOptimalLinearModelObjective(
                self._design_matrix,
                self._bkd.asarray(np.array([0.1, 0.2])),  # Not scalar
                self._prior_cov,
                self._bkd,
            )

        with self.assertRaises(TypeError):
            DOptimalLinearModelObjective(
                self._design_matrix,
                self._noise_cov,
                self._bkd.asarray(np.array([1.0, 2.0])),  # Not scalar
                self._bkd,
            )

    def test_nobs_nparams(self):
        """Test nobs and nparams accessors."""
        obj = self._create_objective()
        self.assertEqual(obj.nobs(), self._nobs)
        self.assertEqual(obj.nparams(), self._nparams)
        self.assertEqual(obj.nvars(), self._nobs)
        self.assertEqual(obj.nqoi(), 1)

    def test_jacobian_implemented(self):
        """Test jacobian_implemented returns True."""
        obj = self._create_objective()
        self.assertTrue(obj.jacobian_implemented())

    def test_hessian_implemented(self):
        """Test hessian_implemented returns True."""
        obj = self._create_objective()
        self.assertTrue(obj.hessian_implemented())

    def test_hvp_implemented(self):
        """Test hvp_implemented returns True."""
        obj = self._create_objective()
        self.assertTrue(obj.hvp_implemented())

    def test_evaluate_alias(self):
        """Test evaluate is alias for __call__."""
        obj = self._create_objective()
        result_call = obj(self._weights)
        result_eval = obj.evaluate(self._weights)
        self._bkd.assert_allclose(result_call, result_eval, rtol=1e-12)

    def test_hvp_matches_hessian_times_vec(self):
        """Test HVP equals Hessian @ vec."""
        obj = self._create_objective()
        np.random.seed(456)
        vec = self._bkd.asarray(np.random.randn(self._nobs, 1))

        hvp_result = obj.hvp(self._weights, vec)
        hess = obj.hessian(self._weights)
        hess_2d = self._bkd.reshape(hess, (self._nobs, self._nobs))
        expected = self._bkd.dot(hess_2d, vec).T  # (1, nobs)

        self._bkd.assert_allclose(hvp_result, expected, rtol=1e-12)


class TestDOptimalStandaloneNumpy(TestDOptimalStandalone[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestDOptimalStandaloneTorch(TestDOptimalStandalone[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
