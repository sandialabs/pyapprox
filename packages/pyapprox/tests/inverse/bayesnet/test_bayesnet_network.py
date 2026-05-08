"""
Tests for GaussianNetwork.
"""

import numpy as np

from pyapprox.inverse.bayesnet.network import GaussianNetwork


class TestGaussianNetworkBase:
    """Base tests for GaussianNetwork."""

    def test_add_root_node(self, bkd) -> None:
        """Test adding a root node with prior."""
        network = GaussianNetwork(bkd)

        mean = bkd.asarray(np.array([0.0]))
        cov = bkd.asarray(np.array([[1.0]]))

        network.add_node(0, nvars=1, prior_mean=mean, prior_cov=cov)

        assert network.nodes() == [0]
        assert network.is_root(0)
        assert network.get_parents(0) == []

    def test_add_child_node(self, bkd) -> None:
        """Test adding a child node with CPD."""
        network = GaussianNetwork(bkd)

        # Root
        mean = bkd.asarray(np.array([0.0]))
        cov = bkd.asarray(np.array([[1.0]]))
        network.add_node(0, nvars=1, prior_mean=mean, prior_cov=cov)

        # Child
        A = bkd.asarray(np.array([[1.0]]))
        offset = bkd.asarray(np.array([0.0]))
        noise_cov = bkd.asarray(np.array([[0.5]]))
        network.add_node(
            1,
            nvars=1,
            parents=[0],
            cpd_coefficients=[A],
            cpd_offset=offset,
            cpd_noise_cov=noise_cov,
        )

        assert set(network.nodes()) == {0, 1}
        assert not network.is_root(1)
        assert network.get_parents(1) == [0]
        assert network.get_children(0) == [1]

    def test_three_node_chain(self, bkd) -> None:
        """Test a chain network: X0 -> X1 -> X2."""
        network = GaussianNetwork(bkd)

        # X0: root with N(0, 1)
        network.add_node(
            0,
            nvars=1,
            prior_mean=bkd.asarray(np.array([0.0])),
            prior_cov=bkd.asarray(np.array([[1.0]])),
        )

        # X1 = X0 + noise
        network.add_node(
            1,
            nvars=1,
            parents=[0],
            cpd_coefficients=[bkd.asarray(np.array([[1.0]]))],
            cpd_offset=bkd.asarray(np.array([0.0])),
            cpd_noise_cov=bkd.asarray(np.array([[0.5]])),
        )

        # X2 = X1 + noise
        network.add_node(
            2,
            nvars=1,
            parents=[1],
            cpd_coefficients=[bkd.asarray(np.array([[1.0]]))],
            cpd_offset=bkd.asarray(np.array([0.0])),
            cpd_noise_cov=bkd.asarray(np.array([[0.5]])),
        )

        assert len(network.nodes()) == 3
        assert network.topological_order() == [0, 1, 2]

    def test_convert_to_factors(self, bkd) -> None:
        """Test converting network to factors."""
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
            cpd_coefficients=[bkd.asarray(np.array([[1.0]]))],
            cpd_noise_cov=bkd.asarray(np.array([[0.5]])),
        )

        factors = network.convert_to_factors()

        assert len(factors) == 2

        # First factor is prior on X0
        assert factors[0].var_ids() == [0]

        # Second factor is CPD p(X1|X0)
        assert set(factors[1].var_ids()) == {0, 1}

    def test_sample(self, bkd) -> None:
        """Test sampling from network."""
        np.random.seed(42)
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
            cpd_coefficients=[bkd.asarray(np.array([[1.0]]))],
            cpd_offset=bkd.asarray(np.array([1.0])),
            cpd_noise_cov=bkd.asarray(np.array([[0.5]])),
        )

        samples = network.sample(1000)

        assert samples[0].shape == (1, 1000)
        assert samples[1].shape == (1, 1000)

        # X0 ~ N(0, 1), X1 = X0 + 1 + noise
        # E[X0] = 0, E[X1] = E[X0] + 1 = 1
        x0_mean = np.mean(bkd.to_numpy(samples[0]))
        x1_mean = np.mean(bkd.to_numpy(samples[1]))

        assert np.abs(x0_mean - 0.0) < 0.1
        assert np.abs(x1_mean - 1.0) < 0.1

    def test_multivariate_nodes(self, bkd) -> None:
        """Test network with multivariate nodes."""
        network = GaussianNetwork(bkd)

        # 2D root
        mean = bkd.asarray(np.array([1.0, 2.0]))
        cov = bkd.asarray(np.eye(2))
        network.add_node(0, nvars=2, prior_mean=mean, prior_cov=cov)

        # 2D child depends on 2D parent
        A = bkd.asarray(np.array([[1.0, 0.0], [0.0, 1.0]]))
        offset = bkd.asarray(np.zeros(2))
        noise_cov = bkd.asarray(np.eye(2) * 0.1)
        network.add_node(
            1,
            nvars=2,
            parents=[0],
            cpd_coefficients=[A],
            cpd_offset=offset,
            cpd_noise_cov=noise_cov,
        )

        assert network.get_node_nvars(0) == 2
        assert network.get_node_nvars(1) == 2

    def test_multiple_parents(self, bkd) -> None:
        """Test node with multiple parents."""
        network = GaussianNetwork(bkd)

        # Two root nodes
        network.add_node(
            0,
            nvars=1,
            prior_mean=bkd.asarray(np.array([0.0])),
            prior_cov=bkd.asarray(np.array([[1.0]])),
        )
        network.add_node(
            1,
            nvars=1,
            prior_mean=bkd.asarray(np.array([1.0])),
            prior_cov=bkd.asarray(np.array([[1.0]])),
        )

        # Child depends on both: X2 = 0.5*X0 + 0.5*X1 + noise
        A0 = bkd.asarray(np.array([[0.5]]))
        A1 = bkd.asarray(np.array([[0.5]]))
        network.add_node(
            2,
            nvars=1,
            parents=[0, 1],
            cpd_coefficients=[A0, A1],
            cpd_noise_cov=bkd.asarray(np.array([[0.1]])),
        )

        assert set(network.get_parents(2)) == {0, 1}

        factors = network.convert_to_factors()
        assert len(factors) == 3

        # CPD factor should include all three variables
        cpd_factor = factors[2]
        assert set(cpd_factor.var_ids()) == {0, 1, 2}

    def test_repr(self, bkd) -> None:
        """Test string representation."""
        network = GaussianNetwork(bkd)
        network.add_node(
            0,
            nvars=1,
            prior_mean=bkd.asarray(np.array([0.0])),
            prior_cov=bkd.asarray(np.array([[1.0]])),
        )

        repr_str = repr(network)
        assert "GaussianNetwork" in repr_str
        assert "nodes=1" in repr_str
