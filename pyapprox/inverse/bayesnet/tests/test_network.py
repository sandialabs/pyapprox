"""
Tests for GaussianNetwork.
"""

import unittest
from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.inverse.bayesnet.network import GaussianNetwork


class TestGaussianNetworkBase(Generic[Array], unittest.TestCase):
    """Base tests for GaussianNetwork."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_add_root_node(self) -> None:
        """Test adding a root node with prior."""
        network = GaussianNetwork(self.bkd())

        mean = self.bkd().asarray(np.array([0.0]))
        cov = self.bkd().asarray(np.array([[1.0]]))

        network.add_node(0, nvars=1, prior_mean=mean, prior_cov=cov)

        self.assertEqual(network.nodes(), [0])
        self.assertTrue(network.is_root(0))
        self.assertEqual(network.get_parents(0), [])

    def test_add_child_node(self) -> None:
        """Test adding a child node with CPD."""
        network = GaussianNetwork(self.bkd())

        # Root
        mean = self.bkd().asarray(np.array([0.0]))
        cov = self.bkd().asarray(np.array([[1.0]]))
        network.add_node(0, nvars=1, prior_mean=mean, prior_cov=cov)

        # Child
        A = self.bkd().asarray(np.array([[1.0]]))
        offset = self.bkd().asarray(np.array([0.0]))
        noise_cov = self.bkd().asarray(np.array([[0.5]]))
        network.add_node(
            1, nvars=1, parents=[0],
            cpd_coefficients=[A], cpd_offset=offset, cpd_noise_cov=noise_cov
        )

        self.assertEqual(set(network.nodes()), {0, 1})
        self.assertFalse(network.is_root(1))
        self.assertEqual(network.get_parents(1), [0])
        self.assertEqual(network.get_children(0), [1])

    def test_three_node_chain(self) -> None:
        """Test a chain network: X0 -> X1 -> X2."""
        network = GaussianNetwork(self.bkd())

        # X0: root with N(0, 1)
        network.add_node(
            0, nvars=1,
            prior_mean=self.bkd().asarray(np.array([0.0])),
            prior_cov=self.bkd().asarray(np.array([[1.0]])),
        )

        # X1 = X0 + noise
        network.add_node(
            1, nvars=1, parents=[0],
            cpd_coefficients=[self.bkd().asarray(np.array([[1.0]]))],
            cpd_offset=self.bkd().asarray(np.array([0.0])),
            cpd_noise_cov=self.bkd().asarray(np.array([[0.5]])),
        )

        # X2 = X1 + noise
        network.add_node(
            2, nvars=1, parents=[1],
            cpd_coefficients=[self.bkd().asarray(np.array([[1.0]]))],
            cpd_offset=self.bkd().asarray(np.array([0.0])),
            cpd_noise_cov=self.bkd().asarray(np.array([[0.5]])),
        )

        self.assertEqual(len(network.nodes()), 3)
        self.assertEqual(network.topological_order(), [0, 1, 2])

    def test_convert_to_factors(self) -> None:
        """Test converting network to factors."""
        network = GaussianNetwork(self.bkd())

        network.add_node(
            0, nvars=1,
            prior_mean=self.bkd().asarray(np.array([0.0])),
            prior_cov=self.bkd().asarray(np.array([[1.0]])),
        )
        network.add_node(
            1, nvars=1, parents=[0],
            cpd_coefficients=[self.bkd().asarray(np.array([[1.0]]))],
            cpd_noise_cov=self.bkd().asarray(np.array([[0.5]])),
        )

        factors = network.convert_to_factors()

        self.assertEqual(len(factors), 2)

        # First factor is prior on X0
        self.assertEqual(factors[0].var_ids(), [0])

        # Second factor is CPD p(X1|X0)
        self.assertEqual(set(factors[1].var_ids()), {0, 1})

    def test_sample(self) -> None:
        """Test sampling from network."""
        np.random.seed(42)
        network = GaussianNetwork(self.bkd())

        network.add_node(
            0, nvars=1,
            prior_mean=self.bkd().asarray(np.array([0.0])),
            prior_cov=self.bkd().asarray(np.array([[1.0]])),
        )
        network.add_node(
            1, nvars=1, parents=[0],
            cpd_coefficients=[self.bkd().asarray(np.array([[1.0]]))],
            cpd_offset=self.bkd().asarray(np.array([1.0])),
            cpd_noise_cov=self.bkd().asarray(np.array([[0.5]])),
        )

        samples = network.sample(1000)

        self.assertEqual(samples[0].shape, (1, 1000))
        self.assertEqual(samples[1].shape, (1, 1000))

        # X0 ~ N(0, 1), X1 = X0 + 1 + noise
        # E[X0] = 0, E[X1] = E[X0] + 1 = 1
        x0_mean = np.mean(self.bkd().to_numpy(samples[0]))
        x1_mean = np.mean(self.bkd().to_numpy(samples[1]))

        self.assertLess(np.abs(x0_mean - 0.0), 0.1)
        self.assertLess(np.abs(x1_mean - 1.0), 0.1)

    def test_multivariate_nodes(self) -> None:
        """Test network with multivariate nodes."""
        network = GaussianNetwork(self.bkd())

        # 2D root
        mean = self.bkd().asarray(np.array([1.0, 2.0]))
        cov = self.bkd().asarray(np.eye(2))
        network.add_node(0, nvars=2, prior_mean=mean, prior_cov=cov)

        # 2D child depends on 2D parent
        A = self.bkd().asarray(np.array([[1.0, 0.0], [0.0, 1.0]]))
        offset = self.bkd().asarray(np.zeros(2))
        noise_cov = self.bkd().asarray(np.eye(2) * 0.1)
        network.add_node(
            1, nvars=2, parents=[0],
            cpd_coefficients=[A], cpd_offset=offset, cpd_noise_cov=noise_cov
        )

        self.assertEqual(network.get_node_nvars(0), 2)
        self.assertEqual(network.get_node_nvars(1), 2)

    def test_multiple_parents(self) -> None:
        """Test node with multiple parents."""
        network = GaussianNetwork(self.bkd())

        # Two root nodes
        network.add_node(
            0, nvars=1,
            prior_mean=self.bkd().asarray(np.array([0.0])),
            prior_cov=self.bkd().asarray(np.array([[1.0]])),
        )
        network.add_node(
            1, nvars=1,
            prior_mean=self.bkd().asarray(np.array([1.0])),
            prior_cov=self.bkd().asarray(np.array([[1.0]])),
        )

        # Child depends on both: X2 = 0.5*X0 + 0.5*X1 + noise
        A0 = self.bkd().asarray(np.array([[0.5]]))
        A1 = self.bkd().asarray(np.array([[0.5]]))
        network.add_node(
            2, nvars=1, parents=[0, 1],
            cpd_coefficients=[A0, A1],
            cpd_noise_cov=self.bkd().asarray(np.array([[0.1]])),
        )

        self.assertEqual(set(network.get_parents(2)), {0, 1})

        factors = network.convert_to_factors()
        self.assertEqual(len(factors), 3)

        # CPD factor should include all three variables
        cpd_factor = factors[2]
        self.assertEqual(set(cpd_factor.var_ids()), {0, 1, 2})

    def test_repr(self) -> None:
        """Test string representation."""
        network = GaussianNetwork(self.bkd())
        network.add_node(
            0, nvars=1,
            prior_mean=self.bkd().asarray(np.array([0.0])),
            prior_cov=self.bkd().asarray(np.array([[1.0]])),
        )

        repr_str = repr(network)
        self.assertIn("GaussianNetwork", repr_str)
        self.assertIn("nodes=1", repr_str)


# NumPy backend tests
class TestGaussianNetworkNumpy(TestGaussianNetworkBase[NDArray[Any]]):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# PyTorch backend tests
class TestGaussianNetworkTorch(TestGaussianNetworkBase[torch.Tensor]):
    __test__ = True

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

    def bkd(self) -> TorchBkd:
        return self._bkd


from pyapprox.util.test_utils import load_tests


if __name__ == "__main__":
    unittest.main()
