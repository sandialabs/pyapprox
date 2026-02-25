"""
Tests for KL-OED objective function.

Tests cover:
- Objective value correctness
- Jacobian verification via finite differences
- Expected information gain computation
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.expdesign.likelihood import GaussianOEDInnerLoopLikelihood
from pyapprox.expdesign.objective import KLOEDObjective
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401


class TestKLOEDObjective(Generic[Array], unittest.TestCase):
    """Base test class for KL-OED objective."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        # Set up test data
        self._nobs = 3
        self._ninner = 15
        self._nouter = 10

        np.random.seed(42)
        self._noise_variances = self._bkd.asarray(np.array([0.1, 0.2, 0.15]))

        # Model outputs (shapes)
        self._outer_shapes = self._bkd.asarray(
            np.random.randn(self._nobs, self._nouter)
        )
        self._inner_shapes = self._bkd.asarray(
            np.random.randn(self._nobs, self._ninner)
        )

        # Latent samples for reparameterization
        self._latent_samples = self._bkd.asarray(
            np.random.randn(self._nobs, self._nouter)
        )

        # Design weights
        self._design_weights = self._bkd.asarray(
            np.random.uniform(0.5, 1.5, (self._nobs, 1))
        )

        # Create inner likelihood
        self._inner_likelihood = GaussianOEDInnerLoopLikelihood(
            self._noise_variances, self._bkd
        )

    def _create_objective(self, outer_weights=None, inner_weights=None):
        """Helper to create KLOEDObjective."""
        return KLOEDObjective(
            self._inner_likelihood,
            self._outer_shapes,
            self._latent_samples,
            self._inner_shapes,
            outer_weights,
            inner_weights,
            self._bkd,
        )

    def test_objective_scalar(self):
        """Test that nqoi is 1."""
        objective = self._create_objective()
        self.assertEqual(objective.nqoi(), 1)
        self.assertEqual(objective.nvars(), self._nobs)

    def test_jacobian_finite_diff(self):
        """Test Jacobian against finite differences."""
        objective = self._create_objective()

        jac_analytical = objective.jacobian(self._design_weights)

        # Finite difference
        eps = 1e-5
        jac_fd = self._bkd.zeros((1, self._nobs))
        for k in range(self._nobs):
            weights_plus = self._bkd.copy(self._design_weights)
            weights_minus = self._bkd.copy(self._design_weights)
            weights_plus[k, 0] = weights_plus[k, 0] + eps
            weights_minus[k, 0] = weights_minus[k, 0] - eps

            val_plus = objective(weights_plus)
            val_minus = objective(weights_minus)

            jac_fd[0, k] = (val_plus[0, 0] - val_minus[0, 0]) / (2 * eps)

        self.assertTrue(
            self._bkd.allclose(jac_analytical, jac_fd, rtol=1e-3, atol=1e-6)
        )

    def test_expected_information_gain(self):
        """Test expected information gain is returned correctly."""
        objective = self._create_objective()

        eig = objective.expected_information_gain(self._design_weights)
        neg_eig = objective(self._design_weights)

        # EIG should be positive for meaningful designs
        # (though not guaranteed for all random data)
        expected_eig = -float(self._bkd.to_numpy(neg_eig)[0, 0])
        self._bkd.assert_allclose(
            self._bkd.asarray([eig]),
            self._bkd.asarray([expected_eig]),
            rtol=1e-10,
        )


class TestKLOEDObjectiveNumpy(TestKLOEDObjective[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestKLOEDObjectiveTorch(TestKLOEDObjective[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
