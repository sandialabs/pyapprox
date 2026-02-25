"""Tests for Chebyshev-Gauss-Lobatto nodes."""

import unittest
from typing import Generic

import numpy as np

from pyapprox.pde.collocation.basis.chebyshev.nodes import (
    ChebyshevGaussLobattoNodes1D,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend


class TestChebyshevNodes(Generic[Array], unittest.TestCase):
    """Base test class for Chebyshev nodes."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_single_node(self):
        """Test single node case (n=1)."""
        bkd = self.bkd()
        gen = ChebyshevGaussLobattoNodes1D(bkd)
        nodes = gen.generate(1)

        self.assertEqual(nodes.shape, (1,))
        bkd.assert_allclose(nodes[0], 0.0, atol=1e-14)

    def test_two_nodes(self):
        """Test two nodes case (n=2) - endpoints."""
        bkd = self.bkd()
        gen = ChebyshevGaussLobattoNodes1D(bkd)
        nodes = gen.generate(2)

        self.assertEqual(nodes.shape, (2,))
        # First node is cos(0) = 1, second is cos(pi) = -1
        bkd.assert_allclose(nodes[0], 1.0, atol=1e-14)
        bkd.assert_allclose(nodes[1], -1.0, atol=1e-14)

    def test_three_nodes(self):
        """Test three nodes case."""
        bkd = self.bkd()
        gen = ChebyshevGaussLobattoNodes1D(bkd)
        nodes = gen.generate(3)

        self.assertEqual(nodes.shape, (3,))
        # cos(0) = 1, cos(pi/2) = 0, cos(pi) = -1
        expected = bkd.asarray([1.0, 0.0, -1.0])
        bkd.assert_allclose(nodes, expected, atol=1e-14)

    def test_nodes_in_range(self):
        """Test all nodes are in [-1, 1]."""
        bkd = self.bkd()
        gen = ChebyshevGaussLobattoNodes1D(bkd)

        for npts in [5, 10, 20]:
            nodes = gen.generate(npts)
            self.assertEqual(nodes.shape, (npts,))
            nodes_np = bkd.to_numpy(nodes)
            self.assertTrue(np.all(nodes_np >= -1.0 - 1e-14))
            self.assertTrue(np.all(nodes_np <= 1.0 + 1e-14))

    def test_endpoints_included(self):
        """Test that endpoints -1 and 1 are included."""
        bkd = self.bkd()
        gen = ChebyshevGaussLobattoNodes1D(bkd)

        for npts in [5, 10, 20]:
            nodes = gen.generate(npts)
            # First node is 1, last is -1 (decreasing order)
            bkd.assert_allclose(nodes[0], 1.0, atol=1e-14)
            bkd.assert_allclose(nodes[-1], -1.0, atol=1e-14)

    def test_nodes_decreasing(self):
        """Test nodes are in decreasing order."""
        bkd = self.bkd()
        gen = ChebyshevGaussLobattoNodes1D(bkd)

        nodes = gen.generate(10)
        for i in range(len(nodes) - 1):
            self.assertGreater(float(nodes[i]), float(nodes[i + 1]))

    def test_symmetric(self):
        """Test nodes are symmetric about 0."""
        bkd = self.bkd()
        gen = ChebyshevGaussLobattoNodes1D(bkd)

        for npts in [5, 10, 11]:
            nodes = gen.generate(npts)
            # Check x_i = -x_{n-1-i}
            reversed_nodes = nodes[::-1]
            bkd.assert_allclose(nodes, -reversed_nodes, atol=1e-14)

    def test_cosine_formula(self):
        """Test nodes match exact cosine formula."""
        bkd = self.bkd()
        gen = ChebyshevGaussLobattoNodes1D(bkd)

        npts = 7
        nodes = gen.generate(npts)

        # x_j = cos(j * pi / (n-1))
        import numpy as np

        expected = np.cos(np.arange(npts) * np.pi / (npts - 1))
        expected_arr = bkd.asarray(expected)
        bkd.assert_allclose(nodes, expected_arr, atol=1e-14)

    def test_invalid_npts(self):
        """Test that invalid npts raises error."""
        bkd = self.bkd()
        gen = ChebyshevGaussLobattoNodes1D(bkd)

        with self.assertRaises(ValueError):
            gen.generate(0)
        with self.assertRaises(ValueError):
            gen.generate(-1)


class TestChebyshevNodesNumpy(TestChebyshevNodes):
    """NumPy backend tests for Chebyshev nodes."""

    __test__ = True

    def bkd(self) -> Backend[Array]:
        return NumpyBkd()


if __name__ == "__main__":
    unittest.main()
