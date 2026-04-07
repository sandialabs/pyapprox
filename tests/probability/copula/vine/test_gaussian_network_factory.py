"""
Tests for dvine_from_gaussian_network factory.

Verifies correctness by comparing the efficient local covariance
propagation against ground truth from the full joint distribution
obtained via variable elimination.
"""

import math

import numpy as np
import pytest

from pyapprox.inverse.bayesnet.inference import compute_marginal
from pyapprox.inverse.bayesnet.network import GaussianNetwork
from tests.probability.copula.vine.gaussian_network_factory import (
    dvine_from_gaussian_network,
)
from pyapprox.probability.gaussian.dense import (
    DenseCholeskyMultivariateGaussian,
)

_SQRT2 = math.sqrt(2.0)


def _standard_normal_cdf(z, bkd):
    """Phi(z) using erf."""
    return 0.5 * (1.0 + bkd.erf(z / _SQRT2))


def _get_network_correlation_precision(network, bkd):
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


def _get_network_correlation(network, bkd):
    """Compute ground truth correlation matrix from network's joint."""
    topo_order = network.topological_order()
    joint_factor = compute_marginal(network, topo_order)
    _, cov = joint_factor.to_moments()
    d = bkd.sqrt(bkd.get_diagonal(cov))
    d_inv = 1.0 / d
    return cov * bkd.outer(d_inv, d_inv)


def _build_simple_chain(bkd):
    """Build a 4-node simple chain: X0 -> X1 -> X2 -> X3."""
    network = GaussianNetwork(bkd)
    network.add_node(
        0,
        nvars=1,
        prior_mean=bkd.asarray(np.array([0.0])),
        prior_cov=bkd.asarray(np.array([[1.0]])),
    )
    network.add_node(
        1,
        nvars=1,
        parents=[0],
        cpd_coefficients=[bkd.asarray(np.array([[0.6]]))],
        cpd_noise_cov=bkd.asarray(np.array([[0.64]])),
    )
    network.add_node(
        2,
        nvars=1,
        parents=[1],
        cpd_coefficients=[bkd.asarray(np.array([[0.5]]))],
        cpd_noise_cov=bkd.asarray(np.array([[0.75]])),
    )
    network.add_node(
        3,
        nvars=1,
        parents=[2],
        cpd_coefficients=[bkd.asarray(np.array([[0.7]]))],
        cpd_noise_cov=bkd.asarray(np.array([[0.51]])),
    )
    return network


def _build_skip_chain(bkd):
    """Build a 4-node chain with skip: X2 depends on X0 and X1.

    X0 -> X1 -> X2 -> X3
    X0 --------> X2
    """
    network = GaussianNetwork(bkd)
    network.add_node(
        0,
        nvars=1,
        prior_mean=bkd.asarray(np.array([0.0])),
        prior_cov=bkd.asarray(np.array([[1.0]])),
    )
    network.add_node(
        1,
        nvars=1,
        parents=[0],
        cpd_coefficients=[bkd.asarray(np.array([[0.6]]))],
        cpd_noise_cov=bkd.asarray(np.array([[0.64]])),
    )
    network.add_node(
        2,
        nvars=1,
        parents=[0, 1],
        cpd_coefficients=[
            bkd.asarray(np.array([[0.3]])),
            bkd.asarray(np.array([[0.4]])),
        ],
        cpd_noise_cov=bkd.asarray(np.array([[0.5]])),
    )
    network.add_node(
        3,
        nvars=1,
        parents=[2],
        cpd_coefficients=[bkd.asarray(np.array([[0.7]]))],
        cpd_noise_cov=bkd.asarray(np.array([[0.51]])),
    )
    return network


class TestGaussianNetworkFactory:
    """Base tests for dvine_from_gaussian_network."""

    def test_simple_chain_bandwidth1(self, bkd) -> None:
        """4-node simple chain: precision matrix matches ground truth."""
        network = _build_simple_chain(bkd)
        dvine = dvine_from_gaussian_network(network, bkd)

        assert dvine.nvars() == 4
        assert dvine.truncation_level() == 1

        expected = _get_network_correlation_precision(network, bkd)
        actual = dvine.to_precision_matrix()
        bkd.assert_allclose(actual, expected, rtol=1e-10, atol=1e-14)

    def test_chain_with_skip_bandwidth2(self, bkd) -> None:
        """Chain with skip connection: bandwidth 2, precision matches."""
        network = _build_skip_chain(bkd)
        dvine = dvine_from_gaussian_network(network, bkd)

        assert dvine.nvars() == 4
        assert dvine.truncation_level() == 2

        expected = _get_network_correlation_precision(network, bkd)
        actual = dvine.to_precision_matrix()
        bkd.assert_allclose(actual, expected, rtol=1e-10, atol=1e-14)

    def test_logpdf_matches_mvn_density(self, bkd) -> None:
        """DVine copula logpdf matches MVN density decomposition.

        copula_logpdf(u) + sum marginal_logpdf(z_i) = mvn_logpdf(z)
        """
        np.random.seed(42)
        network = _build_simple_chain(bkd)
        dvine = dvine_from_gaussian_network(network, bkd)
        nvars = dvine.nvars()

        R = _get_network_correlation(network, bkd)
        mean_zero = bkd.zeros((nvars, 1))
        mvn = DenseCholeskyMultivariateGaussian(mean_zero, R, bkd)

        nsamples = 50
        x = mvn.rvs(nsamples)  # (nvars, nsamples)
        u = _standard_normal_cdf(x, bkd)

        mvn_logpdf = mvn.logpdf(x)  # (1, nsamples)

        log_2pi = math.log(2.0 * math.pi)
        marginal_logpdf_sum = bkd.sum(-0.5 * log_2pi - 0.5 * x * x, axis=0)
        marginal_logpdf_sum = bkd.reshape(marginal_logpdf_sum, (1, -1))

        copula_logpdf = dvine.logpdf(u)

        bkd.assert_allclose(
            copula_logpdf + marginal_logpdf_sum, mvn_logpdf, rtol=1e-8
        )

    def test_single_node(self, bkd) -> None:
        """Single node network: trunc=0, no pair copulas."""
        network = GaussianNetwork(bkd)
        network.add_node(
            0,
            nvars=1,
            prior_mean=bkd.asarray(np.array([0.0])),
            prior_cov=bkd.asarray(np.array([[2.0]])),
        )
        dvine = dvine_from_gaussian_network(network, bkd)
        assert dvine.nvars() == 1
        assert dvine.truncation_level() == 0
        assert dvine.npair_copulas() == 0

    def test_two_nodes(self, bkd) -> None:
        """Two-node chain: trunc=1, correlation matches."""
        network = GaussianNetwork(bkd)
        network.add_node(
            0,
            nvars=1,
            prior_mean=bkd.asarray(np.array([0.0])),
            prior_cov=bkd.asarray(np.array([[1.0]])),
        )
        network.add_node(
            1,
            nvars=1,
            parents=[0],
            cpd_coefficients=[bkd.asarray(np.array([[0.8]]))],
            cpd_noise_cov=bkd.asarray(np.array([[0.36]])),
        )
        dvine = dvine_from_gaussian_network(network, bkd)
        assert dvine.nvars() == 2
        assert dvine.truncation_level() == 1
        assert dvine.npair_copulas() == 1

        R_expected = _get_network_correlation(network, bkd)
        R_actual = dvine.to_correlation_matrix()
        bkd.assert_allclose(R_actual, R_expected, rtol=1e-10)

    def test_rejects_multivariate_nodes(self, bkd) -> None:
        """Node with nvars=2 raises ValueError."""
        network = GaussianNetwork(bkd)
        network.add_node(
            0,
            nvars=2,
            prior_mean=bkd.asarray(np.array([0.0, 0.0])),
            prior_cov=bkd.asarray(np.eye(2)),
        )
        with pytest.raises(ValueError):
            dvine_from_gaussian_network(network, bkd)

    def test_independent_nodes(self, bkd) -> None:
        """Multiple root nodes with no edges: trunc=0."""
        network = GaussianNetwork(bkd)
        for i in range(4):
            network.add_node(
                i,
                nvars=1,
                prior_mean=bkd.asarray(np.array([0.0])),
                prior_cov=bkd.asarray(np.array([[1.0]])),
            )
        dvine = dvine_from_gaussian_network(network, bkd)
        assert dvine.nvars() == 4
        assert dvine.truncation_level() == 0
        assert dvine.npair_copulas() == 0

    def test_skip_chain_logpdf_matches_mvn(self, bkd) -> None:
        """Bandwidth-2 chain: copula logpdf matches MVN density."""
        np.random.seed(42)
        network = _build_skip_chain(bkd)
        dvine = dvine_from_gaussian_network(network, bkd)
        nvars = dvine.nvars()

        R = _get_network_correlation(network, bkd)
        mean_zero = bkd.zeros((nvars, 1))
        mvn = DenseCholeskyMultivariateGaussian(mean_zero, R, bkd)

        nsamples = 50
        x = mvn.rvs(nsamples)
        u = _standard_normal_cdf(x, bkd)

        mvn_logpdf = mvn.logpdf(x)
        log_2pi = math.log(2.0 * math.pi)
        marginal_logpdf_sum = bkd.sum(-0.5 * log_2pi - 0.5 * x * x, axis=0)
        marginal_logpdf_sum = bkd.reshape(marginal_logpdf_sum, (1, -1))

        copula_logpdf = dvine.logpdf(u)
        bkd.assert_allclose(
            copula_logpdf + marginal_logpdf_sum, mvn_logpdf, rtol=1e-8
        )

    def test_correlation_matches_simple_chain(self, bkd) -> None:
        """Simple chain: correlation matrix from DVine matches network."""
        network = _build_simple_chain(bkd)
        dvine = dvine_from_gaussian_network(network, bkd)

        R_expected = _get_network_correlation(network, bkd)
        R_actual = dvine.to_correlation_matrix()
        bkd.assert_allclose(R_actual, R_expected, rtol=1e-10)

    def test_correlation_matches_skip_chain(self, bkd) -> None:
        """Skip chain: correlation matrix from DVine matches network."""
        network = _build_skip_chain(bkd)
        dvine = dvine_from_gaussian_network(network, bkd)

        R_expected = _get_network_correlation(network, bkd)
        R_actual = dvine.to_correlation_matrix()
        bkd.assert_allclose(R_actual, R_expected, rtol=1e-10)
