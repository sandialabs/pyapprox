"""
Tests for variable elimination inference.
"""

import unittest
from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.inverse.bayesnet.network import GaussianNetwork
from pyapprox.typing.inverse.bayesnet.inference import (
    cond_prob_variable_elimination,
    compute_marginal,
    compute_posterior,
)


class TestVariableEliminationBase(Generic[Array], unittest.TestCase):
    """Base tests for variable elimination inference."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def _build_chain_network(self) -> GaussianNetwork[Array]:
        """Build a simple chain network for testing."""
        network = GaussianNetwork(self.bkd())

        # X0 ~ N(0, 1)
        network.add_node(
            0, nvars=1,
            prior_mean=self.bkd().asarray(np.array([0.0])),
            prior_cov=self.bkd().asarray(np.array([[1.0]])),
        )

        # X1 = X0 + noise, noise ~ N(0, 0.5)
        network.add_node(
            1, nvars=1, parents=[0],
            cpd_coefficients=[self.bkd().asarray(np.array([[1.0]]))],
            cpd_offset=self.bkd().asarray(np.array([0.0])),
            cpd_noise_cov=self.bkd().asarray(np.array([[0.5]])),
        )

        return network

    def test_compute_marginal_root(self) -> None:
        """Test computing marginal of root node."""
        network = self._build_chain_network()

        marginal = compute_marginal(network, [0])

        mean, cov = marginal.to_moments()

        # X0 ~ N(0, 1)
        np.testing.assert_allclose(
            self.bkd().to_numpy(mean), [0.0], rtol=1e-6
        )
        np.testing.assert_allclose(
            self.bkd().to_numpy(cov), [[1.0]], rtol=1e-6
        )

    def test_compute_marginal_child(self) -> None:
        """Test computing marginal of child node."""
        network = self._build_chain_network()

        marginal = compute_marginal(network, [1])

        mean, cov = marginal.to_moments()

        # X1 = X0 + noise
        # E[X1] = E[X0] = 0
        # Var[X1] = Var[X0] + Var[noise] = 1 + 0.5 = 1.5
        np.testing.assert_allclose(
            self.bkd().to_numpy(mean), [0.0], rtol=1e-6
        )
        np.testing.assert_allclose(
            self.bkd().to_numpy(cov), [[1.5]], rtol=1e-6
        )

    def test_compute_joint(self) -> None:
        """Test computing joint distribution."""
        network = self._build_chain_network()

        joint = compute_marginal(network, [0, 1])

        mean, cov = joint.to_moments()

        # Joint mean
        np.testing.assert_allclose(
            self.bkd().to_numpy(mean), [0.0, 0.0], rtol=1e-6
        )

        # Joint covariance:
        # Var[X0] = 1
        # Var[X1] = 1.5
        # Cov[X0, X1] = Cov[X0, X0 + noise] = Var[X0] = 1
        cov_np = self.bkd().to_numpy(cov)
        np.testing.assert_allclose(cov_np[0, 0], 1.0, rtol=1e-6)
        np.testing.assert_allclose(cov_np[1, 1], 1.5, rtol=1e-6)
        np.testing.assert_allclose(cov_np[0, 1], 1.0, rtol=1e-6)
        np.testing.assert_allclose(cov_np[1, 0], 1.0, rtol=1e-6)

    def test_compute_posterior_single_observation(self) -> None:
        """Test posterior inference with single observation."""
        network = self._build_chain_network()

        # Observe X1 = 1.0
        evidence = {1: self.bkd().asarray(np.array([1.0]))}

        posterior = compute_posterior(network, [0], evidence)

        mean, cov = posterior.to_moments()

        # Posterior of X0 given X1 = 1:
        # Using conjugate Gaussian formula:
        # Prior: X0 ~ N(0, 1)
        # Likelihood: X1 | X0 ~ N(X0, 0.5)
        # Posterior precision = 1/1 + 1/0.5 = 3
        # Posterior variance = 1/3
        # Posterior mean = (1/3) * (0 * 1 + 1 * 2) = 2/3
        expected_var = 1.0 / 3.0
        expected_mean = 2.0 / 3.0

        np.testing.assert_allclose(
            float(self.bkd().to_numpy(mean)[0]), expected_mean, rtol=1e-5
        )
        np.testing.assert_allclose(
            float(self.bkd().to_numpy(cov)[0, 0]), expected_var, rtol=1e-5
        )

    def test_three_node_chain_posterior(self) -> None:
        """Test posterior inference in three-node chain."""
        network = GaussianNetwork(self.bkd())

        # X0 ~ N(0, 1)
        network.add_node(
            0, nvars=1,
            prior_mean=self.bkd().asarray(np.array([0.0])),
            prior_cov=self.bkd().asarray(np.array([[1.0]])),
        )

        # X1 = X0 + noise1
        network.add_node(
            1, nvars=1, parents=[0],
            cpd_coefficients=[self.bkd().asarray(np.array([[1.0]]))],
            cpd_noise_cov=self.bkd().asarray(np.array([[0.5]])),
        )

        # X2 = X1 + noise2
        network.add_node(
            2, nvars=1, parents=[1],
            cpd_coefficients=[self.bkd().asarray(np.array([[1.0]]))],
            cpd_noise_cov=self.bkd().asarray(np.array([[0.5]])),
        )

        # Observe X2 = 2.0, query X0
        evidence = {2: self.bkd().asarray(np.array([2.0]))}

        posterior = compute_posterior(network, [0], evidence)

        mean, cov = posterior.to_moments()

        # The posterior mean should shift toward observed value
        # (but less than with direct observation)
        mean_val = float(self.bkd().to_numpy(mean)[0])
        self.assertGreater(mean_val, 0.0)
        self.assertLess(mean_val, 2.0)

    def test_multiple_queries(self) -> None:
        """Test querying multiple variables."""
        network = self._build_chain_network()

        # Add another node for richer example
        network.add_node(
            2, nvars=1, parents=[1],
            cpd_coefficients=[self.bkd().asarray(np.array([[1.0]]))],
            cpd_noise_cov=self.bkd().asarray(np.array([[0.5]])),
        )

        # Observe X2 = 1.0, query X0 and X1
        evidence = {2: self.bkd().asarray(np.array([1.0]))}

        posterior = compute_posterior(network, [0, 1], evidence)

        self.assertEqual(set(posterior.var_ids()), {0, 1})

        # Both posterior means should be positive (shifted toward evidence)
        mean, _ = posterior.to_moments()
        mean_np = self.bkd().to_numpy(mean)
        self.assertGreater(mean_np[0], 0.0)

    def test_v_structure_network(self) -> None:
        """Test network with V-structure (two parents, one child)."""
        network = GaussianNetwork(self.bkd())

        # Two independent roots
        network.add_node(
            0, nvars=1,
            prior_mean=self.bkd().asarray(np.array([0.0])),
            prior_cov=self.bkd().asarray(np.array([[1.0]])),
        )
        network.add_node(
            1, nvars=1,
            prior_mean=self.bkd().asarray(np.array([0.0])),
            prior_cov=self.bkd().asarray(np.array([[1.0]])),
        )

        # X2 = X0 + X1 + noise
        network.add_node(
            2, nvars=1, parents=[0, 1],
            cpd_coefficients=[
                self.bkd().asarray(np.array([[1.0]])),
                self.bkd().asarray(np.array([[1.0]])),
            ],
            cpd_noise_cov=self.bkd().asarray(np.array([[0.1]])),
        )

        # Compute joint
        joint = compute_marginal(network, [0, 1, 2])
        mean, cov = joint.to_moments()

        # E[X2] = E[X0] + E[X1] = 0
        # Var[X2] = Var[X0] + Var[X1] + Var[noise] = 2.1
        mean_np = self.bkd().to_numpy(mean)
        cov_np = self.bkd().to_numpy(cov)

        np.testing.assert_allclose(mean_np[2], 0.0, atol=1e-6)
        np.testing.assert_allclose(cov_np[2, 2], 2.1, rtol=1e-5)

        # X0 and X1 are marginally independent but become dependent when X2 observed
        # (explaining away effect)

    def test_posterior_vs_analytical(self) -> None:
        """Test posterior matches analytical solution."""
        network = GaussianNetwork(self.bkd())

        # Simple linear regression setup:
        # Prior: theta ~ N(0, prior_var)
        # Likelihood: y | theta ~ N(theta, noise_var)
        prior_var = 2.0
        noise_var = 0.5

        network.add_node(
            0, nvars=1,
            prior_mean=self.bkd().asarray(np.array([0.0])),
            prior_cov=self.bkd().asarray(np.array([[prior_var]])),
        )
        network.add_node(
            1, nvars=1, parents=[0],
            cpd_coefficients=[self.bkd().asarray(np.array([[1.0]]))],
            cpd_noise_cov=self.bkd().asarray(np.array([[noise_var]])),
        )

        # Observe y = 1.5
        y_obs = 1.5
        evidence = {1: self.bkd().asarray(np.array([y_obs]))}

        posterior = compute_posterior(network, [0], evidence)
        post_mean, post_cov = posterior.to_moments()

        # Analytical solution:
        # Posterior precision = 1/prior_var + 1/noise_var
        # Posterior variance = 1 / (1/prior_var + 1/noise_var)
        # Posterior mean = post_var * (y_obs / noise_var)
        post_prec_analytical = 1 / prior_var + 1 / noise_var
        post_var_analytical = 1 / post_prec_analytical
        post_mean_analytical = post_var_analytical * (y_obs / noise_var)

        np.testing.assert_allclose(
            float(self.bkd().to_numpy(post_mean)[0]),
            post_mean_analytical,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            float(self.bkd().to_numpy(post_cov)[0, 0]),
            post_var_analytical,
            rtol=1e-5,
        )


# NumPy backend tests
class TestVariableEliminationNumpy(TestVariableEliminationBase[NDArray[Any]]):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# PyTorch backend tests
class TestVariableEliminationTorch(TestVariableEliminationBase[torch.Tensor]):
    __test__ = True

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

    def bkd(self) -> TorchBkd:
        return self._bkd


from pyapprox.typing.util.test_utils import load_tests


if __name__ == "__main__":
    unittest.main()
