"""Tests for Chebyshev-Gauss-Lobatto nodes."""


import pytest
import numpy as np

from pyapprox.pde.collocation.basis.chebyshev.nodes import (
    ChebyshevGaussLobattoNodes1D,
)


class TestChebyshevNodes:
    """Base test class for Chebyshev nodes."""

    def test_single_node(self, bkd):
        """Test single node case (n=1)."""
        gen = ChebyshevGaussLobattoNodes1D(bkd)
        nodes = gen.generate(1)

        assert nodes.shape == (1,)
        bkd.assert_allclose(nodes[0], bkd.asarray(0.0), atol=1e-14)

    def test_two_nodes(self, bkd):
        """Test two nodes case (n=2) - endpoints."""
        gen = ChebyshevGaussLobattoNodes1D(bkd)
        nodes = gen.generate(2)

        assert nodes.shape == (2,)
        # First node is cos(0) = 1, second is cos(pi) = -1
        bkd.assert_allclose(nodes[0], bkd.asarray(1.0), atol=1e-14)
        bkd.assert_allclose(nodes[1], bkd.asarray(-1.0), atol=1e-14)

    def test_three_nodes(self, bkd):
        """Test three nodes case."""
        gen = ChebyshevGaussLobattoNodes1D(bkd)
        nodes = gen.generate(3)

        assert nodes.shape == (3,)
        # cos(0) = 1, cos(pi/2) = 0, cos(pi) = -1
        expected = bkd.asarray([1.0, 0.0, -1.0])
        bkd.assert_allclose(nodes, expected, atol=1e-14)

    def test_nodes_in_range(self, bkd):
        """Test all nodes are in [-1, 1]."""
        gen = ChebyshevGaussLobattoNodes1D(bkd)

        for npts in [5, 10, 20]:
            nodes = gen.generate(npts)
            assert nodes.shape == (npts,)
            nodes_np = bkd.to_numpy(nodes)
            assert np.all(nodes_np >= -1.0 - 1e-14)
            assert np.all(nodes_np <= 1.0 + 1e-14)

    def test_endpoints_included(self, bkd):
        """Test that endpoints -1 and 1 are included."""
        gen = ChebyshevGaussLobattoNodes1D(bkd)

        for npts in [5, 10, 20]:
            nodes = gen.generate(npts)
            # First node is 1, last is -1 (decreasing order)
            bkd.assert_allclose(nodes[0], bkd.asarray(1.0), atol=1e-14)
            bkd.assert_allclose(nodes[-1], bkd.asarray(-1.0), atol=1e-14)

    def test_nodes_decreasing(self, bkd):
        """Test nodes are in decreasing order."""
        gen = ChebyshevGaussLobattoNodes1D(bkd)

        nodes = gen.generate(10)
        for i in range(len(nodes) - 1):
            assert float(nodes[i]) > float(nodes[i + 1])

    def test_symmetric(self, bkd):
        """Test nodes are symmetric about 0."""
        gen = ChebyshevGaussLobattoNodes1D(bkd)

        for npts in [5, 10, 11]:
            nodes = gen.generate(npts)
            # Check x_i = -x_{n-1-i}
            reversed_nodes = bkd.flip(nodes)
            bkd.assert_allclose(nodes, -reversed_nodes, atol=1e-14)

    def test_cosine_formula(self, bkd):
        """Test nodes match exact cosine formula."""
        gen = ChebyshevGaussLobattoNodes1D(bkd)

        npts = 7
        nodes = gen.generate(npts)

        # x_j = cos(j * pi / (n-1))
        import numpy as np

        expected = np.cos(np.arange(npts) * np.pi / (npts - 1))
        expected_arr = bkd.asarray(expected)
        bkd.assert_allclose(nodes, expected_arr, atol=1e-14)

    def test_invalid_npts(self, bkd):
        """Test that invalid npts raises error."""
        gen = ChebyshevGaussLobattoNodes1D(bkd)

        with pytest.raises(ValueError):
            gen.generate(0)
        with pytest.raises(ValueError):
            gen.generate(-1)
