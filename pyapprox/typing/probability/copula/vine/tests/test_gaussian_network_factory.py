"""
Tests for dvine_from_gaussian_network factory.

Verifies correctness by comparing the efficient local covariance
propagation against ground truth from the full joint distribution
obtained via variable elimination.
"""

import math
import unittest
from typing import Any, Dict, Generic, List

import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401
from pyapprox.typing.inverse.bayesnet.network import GaussianNetwork
from pyapprox.typing.inverse.bayesnet.inference import compute_marginal
from pyapprox.typing.probability.copula.vine.gaussian_network_factory import (
    dvine_from_gaussian_network,
)
from pyapprox.typing.probability.gaussian.dense import (
    DenseCholeskyMultivariateGaussian,
)


_SQRT2 = math.sqrt(2.0)


def _standard_normal_cdf(z: Array, bkd: Backend[Array]) -> Array:
    """Phi(z) using erf."""
    return 0.5 * (1.0 + bkd.erf(z / _SQRT2))


def _get_network_correlation_precision(
    network: GaussianNetwork[Array],
    bkd: Backend[Array],
) -> Array:
    """Compute inv(R) from network's joint, where R is the correlation matrix.

    DVineCopula.to_precision_matrix() returns inv(R), not inv(Sigma).
    The relationship is: inv(R) = D * inv(Sigma) * D where
    D = diag(sqrt(diag(Sigma))).
    """
    topo_order = network.topological_order()
    joint_factor = compute_marginal(network, topo_order)
    _, cov = joint_factor.to_moments()
    d = bkd.sqrt(bkd.get_diagonal(cov))
    d_inv = 1.0 / d
    R = cov * bkd.outer(d_inv, d_inv)
    return bkd.inv(R)


def _get_network_correlation(
    network: GaussianNetwork[Array],
    bkd: Backend[Array],
) -> Array:
    """Compute ground truth correlation matrix from network's joint."""
    topo_order = network.topological_order()
    joint_factor = compute_marginal(network, topo_order)
    _, cov = joint_factor.to_moments()
    d = bkd.sqrt(bkd.get_diagonal(cov))
    d_inv = 1.0 / d
    return cov * bkd.outer(d_inv, d_inv)


def _build_simple_chain(
    bkd: Backend[Array],
) -> GaussianNetwork[Array]:
    """Build a 4-node simple chain: X0 -> X1 -> X2 -> X3."""
    network = GaussianNetwork(bkd)
    network.add_node(
        0, nvars=1,
        prior_mean=bkd.asarray(np.array([0.0])),
        prior_cov=bkd.asarray(np.array([[1.0]])),
    )
    network.add_node(
        1, nvars=1, parents=[0],
        cpd_coefficients=[bkd.asarray(np.array([[0.6]]))],
        cpd_noise_cov=bkd.asarray(np.array([[0.64]])),
    )
    network.add_node(
        2, nvars=1, parents=[1],
        cpd_coefficients=[bkd.asarray(np.array([[0.5]]))],
        cpd_noise_cov=bkd.asarray(np.array([[0.75]])),
    )
    network.add_node(
        3, nvars=1, parents=[2],
        cpd_coefficients=[bkd.asarray(np.array([[0.7]]))],
        cpd_noise_cov=bkd.asarray(np.array([[0.51]])),
    )
    return network


def _build_skip_chain(
    bkd: Backend[Array],
) -> GaussianNetwork[Array]:
    """Build a 4-node chain with skip: X2 depends on X0 and X1.

    X0 -> X1 -> X2 -> X3
    X0 --------> X2
    """
    network = GaussianNetwork(bkd)
    network.add_node(
        0, nvars=1,
        prior_mean=bkd.asarray(np.array([0.0])),
        prior_cov=bkd.asarray(np.array([[1.0]])),
    )
    network.add_node(
        1, nvars=1, parents=[0],
        cpd_coefficients=[bkd.asarray(np.array([[0.6]]))],
        cpd_noise_cov=bkd.asarray(np.array([[0.64]])),
    )
    network.add_node(
        2, nvars=1, parents=[0, 1],
        cpd_coefficients=[
            bkd.asarray(np.array([[0.3]])),
            bkd.asarray(np.array([[0.4]])),
        ],
        cpd_noise_cov=bkd.asarray(np.array([[0.5]])),
    )
    network.add_node(
        3, nvars=1, parents=[2],
        cpd_coefficients=[bkd.asarray(np.array([[0.7]]))],
        cpd_noise_cov=bkd.asarray(np.array([[0.51]])),
    )
    return network


class TestGaussianNetworkFactoryBase(Generic[Array], unittest.TestCase):
    """Base tests for dvine_from_gaussian_network."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_simple_chain_bandwidth1(self) -> None:
        """4-node simple chain: precision matrix matches ground truth."""
        network = _build_simple_chain(self._bkd)
        dvine = dvine_from_gaussian_network(network, self._bkd)

        self.assertEqual(dvine.nvars(), 4)
        self.assertEqual(dvine.truncation_level(), 1)

        expected = _get_network_correlation_precision(network, self._bkd)
        actual = dvine.to_precision_matrix()
        self._bkd.assert_allclose(actual, expected, rtol=1e-10, atol=1e-14)

    def test_chain_with_skip_bandwidth2(self) -> None:
        """Chain with skip connection: bandwidth 2, precision matches."""
        network = _build_skip_chain(self._bkd)
        dvine = dvine_from_gaussian_network(network, self._bkd)

        self.assertEqual(dvine.nvars(), 4)
        self.assertEqual(dvine.truncation_level(), 2)

        expected = _get_network_correlation_precision(network, self._bkd)
        actual = dvine.to_precision_matrix()
        self._bkd.assert_allclose(actual, expected, rtol=1e-10, atol=1e-14)

    def test_logpdf_matches_mvn_density(self) -> None:
        """DVine copula logpdf matches MVN density decomposition.

        copula_logpdf(u) + sum marginal_logpdf(z_i) = mvn_logpdf(z)
        """
        np.random.seed(42)
        network = _build_simple_chain(self._bkd)
        dvine = dvine_from_gaussian_network(network, self._bkd)
        nvars = dvine.nvars()

        R = _get_network_correlation(network, self._bkd)
        mean_zero = self._bkd.zeros((nvars, 1))
        mvn = DenseCholeskyMultivariateGaussian(mean_zero, R, self._bkd)

        nsamples = 50
        x = mvn.rvs(nsamples)  # (nvars, nsamples)
        u = _standard_normal_cdf(x, self._bkd)

        mvn_logpdf = mvn.logpdf(x)  # (1, nsamples)

        log_2pi = math.log(2.0 * math.pi)
        marginal_logpdf_sum = self._bkd.sum(
            -0.5 * log_2pi - 0.5 * x * x, axis=0
        )
        marginal_logpdf_sum = self._bkd.reshape(
            marginal_logpdf_sum, (1, -1)
        )

        copula_logpdf = dvine.logpdf(u)

        self._bkd.assert_allclose(
            copula_logpdf + marginal_logpdf_sum, mvn_logpdf, rtol=1e-8
        )

    def test_single_node(self) -> None:
        """Single node network: trunc=0, no pair copulas."""
        network = GaussianNetwork(self._bkd)
        network.add_node(
            0, nvars=1,
            prior_mean=self._bkd.asarray(np.array([0.0])),
            prior_cov=self._bkd.asarray(np.array([[2.0]])),
        )
        dvine = dvine_from_gaussian_network(network, self._bkd)
        self.assertEqual(dvine.nvars(), 1)
        self.assertEqual(dvine.truncation_level(), 0)
        self.assertEqual(dvine.npair_copulas(), 0)

    def test_two_nodes(self) -> None:
        """Two-node chain: trunc=1, correlation matches."""
        network = GaussianNetwork(self._bkd)
        network.add_node(
            0, nvars=1,
            prior_mean=self._bkd.asarray(np.array([0.0])),
            prior_cov=self._bkd.asarray(np.array([[1.0]])),
        )
        network.add_node(
            1, nvars=1, parents=[0],
            cpd_coefficients=[self._bkd.asarray(np.array([[0.8]]))],
            cpd_noise_cov=self._bkd.asarray(np.array([[0.36]])),
        )
        dvine = dvine_from_gaussian_network(network, self._bkd)
        self.assertEqual(dvine.nvars(), 2)
        self.assertEqual(dvine.truncation_level(), 1)
        self.assertEqual(dvine.npair_copulas(), 1)

        R_expected = _get_network_correlation(network, self._bkd)
        R_actual = dvine.to_correlation_matrix()
        self._bkd.assert_allclose(R_actual, R_expected, rtol=1e-10)

    def test_rejects_multivariate_nodes(self) -> None:
        """Node with nvars=2 raises ValueError."""
        network = GaussianNetwork(self._bkd)
        network.add_node(
            0, nvars=2,
            prior_mean=self._bkd.asarray(np.array([0.0, 0.0])),
            prior_cov=self._bkd.asarray(np.eye(2)),
        )
        with self.assertRaises(ValueError):
            dvine_from_gaussian_network(network, self._bkd)

    def test_independent_nodes(self) -> None:
        """Multiple root nodes with no edges: trunc=0."""
        network = GaussianNetwork(self._bkd)
        for i in range(4):
            network.add_node(
                i, nvars=1,
                prior_mean=self._bkd.asarray(np.array([0.0])),
                prior_cov=self._bkd.asarray(np.array([[1.0]])),
            )
        dvine = dvine_from_gaussian_network(network, self._bkd)
        self.assertEqual(dvine.nvars(), 4)
        self.assertEqual(dvine.truncation_level(), 0)
        self.assertEqual(dvine.npair_copulas(), 0)

    def test_skip_chain_logpdf_matches_mvn(self) -> None:
        """Bandwidth-2 chain: copula logpdf matches MVN density."""
        np.random.seed(42)
        network = _build_skip_chain(self._bkd)
        dvine = dvine_from_gaussian_network(network, self._bkd)
        nvars = dvine.nvars()

        R = _get_network_correlation(network, self._bkd)
        mean_zero = self._bkd.zeros((nvars, 1))
        mvn = DenseCholeskyMultivariateGaussian(mean_zero, R, self._bkd)

        nsamples = 50
        x = mvn.rvs(nsamples)
        u = _standard_normal_cdf(x, self._bkd)

        mvn_logpdf = mvn.logpdf(x)
        log_2pi = math.log(2.0 * math.pi)
        marginal_logpdf_sum = self._bkd.sum(
            -0.5 * log_2pi - 0.5 * x * x, axis=0
        )
        marginal_logpdf_sum = self._bkd.reshape(
            marginal_logpdf_sum, (1, -1)
        )

        copula_logpdf = dvine.logpdf(u)
        self._bkd.assert_allclose(
            copula_logpdf + marginal_logpdf_sum, mvn_logpdf, rtol=1e-8
        )

    def test_correlation_matches_simple_chain(self) -> None:
        """Simple chain: correlation matrix from DVine matches network."""
        network = _build_simple_chain(self._bkd)
        dvine = dvine_from_gaussian_network(network, self._bkd)

        R_expected = _get_network_correlation(network, self._bkd)
        R_actual = dvine.to_correlation_matrix()
        self._bkd.assert_allclose(R_actual, R_expected, rtol=1e-10)

    def test_correlation_matches_skip_chain(self) -> None:
        """Skip chain: correlation matrix from DVine matches network."""
        network = _build_skip_chain(self._bkd)
        dvine = dvine_from_gaussian_network(network, self._bkd)

        R_expected = _get_network_correlation(network, self._bkd)
        R_actual = dvine.to_correlation_matrix()
        self._bkd.assert_allclose(R_actual, R_expected, rtol=1e-10)


class TestGaussianNetworkFactoryNumpy(
    TestGaussianNetworkFactoryBase[NDArray[Any]]
):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGaussianNetworkFactoryTorch(
    TestGaussianNetworkFactoryBase[torch.Tensor]
):
    __test__ = True

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = self.bkd()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
