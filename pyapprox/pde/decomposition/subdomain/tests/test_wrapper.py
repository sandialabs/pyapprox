"""Tests for SubdomainWrapper class."""

import unittest
import math

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401
from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.mesh import TransformedMesh1D
from pyapprox.pde.collocation.boundary import (
    constant_dirichlet_bc,
    DirichletBC,
)
from pyapprox.pde.collocation.physics.advection_diffusion import (
    create_steady_diffusion,
)
from pyapprox.pde.decomposition.subdomain.wrapper import (
    SubdomainWrapper,
)
from pyapprox.pde.decomposition.subdomain.flux import (
    FluxComputer,
    compute_flux_mismatch,
    flux_mismatch_norm,
)
from pyapprox.pde.decomposition.interface import Interface1D


class TestSubdomainWrapperBasic(unittest.TestCase):
    """Basic tests for SubdomainWrapper."""

    def setUp(self):
        self.bkd = NumpyBkd()

    def test_initialization(self):
        """Test basic initialization."""
        bkd = self.bkd
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        physics = create_steady_diffusion(basis, bkd, diffusion=1.0)

        # Create a 1D interface at x=0.5
        interface = Interface1D(bkd, interface_id=0, subdomain_ids=(0, 1),
                                interface_point=0.5)

        wrapper = SubdomainWrapper(
            bkd,
            subdomain_id=0,
            physics=physics,
            interfaces={0: interface},
        )

        self.assertEqual(wrapper.subdomain_id(), 0)
        self.assertEqual(wrapper.nstates(), npts)
        self.assertEqual(wrapper.interface_ids(), [0])

    def test_set_interface_boundary_indices(self):
        """Test setting interface boundary indices."""
        bkd = self.bkd
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        physics = create_steady_diffusion(basis, bkd, diffusion=1.0)

        interface = Interface1D(bkd, interface_id=0, subdomain_ids=(0, 1),
                                interface_point=0.5)

        wrapper = SubdomainWrapper(
            bkd, subdomain_id=0, physics=physics, interfaces={0: interface}
        )

        # Set boundary indices (rightmost point)
        boundary_indices = bkd.asarray([npts - 1])
        wrapper.set_interface_boundary_indices(0, boundary_indices)

        # Should not raise
        interface_coeffs = bkd.asarray([1.0])
        interface.set_subdomain_boundary_points(0, bkd.asarray([0.5]))
        wrapper.set_interface_dirichlet(0, interface_coeffs)

    def test_invalid_interface_id(self):
        """Test error handling for invalid interface ID."""
        bkd = self.bkd
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        physics = create_steady_diffusion(basis, bkd, diffusion=1.0)

        interface = Interface1D(bkd, interface_id=0, subdomain_ids=(0, 1),
                                interface_point=0.5)

        wrapper = SubdomainWrapper(
            bkd, subdomain_id=0, physics=physics, interfaces={0: interface}
        )

        with self.assertRaises(ValueError):
            wrapper.set_interface_boundary_indices(999, bkd.asarray([0]))


class TestFluxComputer(unittest.TestCase):
    """Tests for FluxComputer class."""

    def setUp(self):
        self.bkd = NumpyBkd()

    def test_gradient_computation(self):
        """Test gradient computation for polynomial."""
        bkd = self.bkd
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)

        flux_computer = FluxComputer(bkd, basis, diffusion=1.0)

        # u = x^2, du/dx = 2x
        nodes = basis.nodes()
        u = nodes ** 2

        grad = flux_computer.compute_gradient(u)

        # Should have 1 component for 1D
        self.assertEqual(len(grad), 1)

        # Check gradient: du/dx = 2x
        expected_grad = 2 * nodes
        bkd.assert_allclose(grad[0], expected_grad, atol=1e-10)

    def test_flux_at_boundary(self):
        """Test flux computation at boundary points."""
        bkd = self.bkd
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)

        flux_computer = FluxComputer(bkd, basis, diffusion=1.0)

        # u = x^2, du/dx = 2x
        nodes = basis.nodes()
        u = nodes ** 2

        # Chebyshev nodes are ordered from x=-1 (index 0) to x=1 (index npts-1)
        # Compute flux at right boundary x=1 (index = npts-1)
        # Normal pointing right: n = [1]
        indices = bkd.asarray([npts - 1])
        normal = bkd.asarray([1.0])

        flux = flux_computer.compute_flux_at_indices(u, indices, normal)

        # At x = 1, du/dx = 2
        expected_flux = bkd.asarray([2.0])
        bkd.assert_allclose(flux, expected_flux, atol=1e-10)


class TestFluxMismatch(unittest.TestCase):
    """Tests for flux mismatch functions."""

    def setUp(self):
        self.bkd = NumpyBkd()

    def test_flux_mismatch_zero(self):
        """Test flux mismatch when conservation is satisfied."""
        bkd = self.bkd

        # Flux from left (normal points right) = 1
        # Flux from right (normal points left) = -1
        # These should sum to zero for conservation
        flux_left = bkd.asarray([1.0, 2.0, 3.0])
        flux_right = bkd.asarray([-1.0, -2.0, -3.0])

        mismatch = compute_flux_mismatch(flux_left, flux_right, bkd)
        bkd.assert_allclose(mismatch, bkd.zeros((3,)), atol=1e-12)

    def test_flux_mismatch_nonzero(self):
        """Test flux mismatch when conservation is violated."""
        bkd = self.bkd

        flux_left = bkd.asarray([1.0, 2.0])
        flux_right = bkd.asarray([0.5, -1.0])  # Not conserved

        mismatch = compute_flux_mismatch(flux_left, flux_right, bkd)
        expected = bkd.asarray([1.5, 1.0])
        bkd.assert_allclose(mismatch, expected, atol=1e-12)

    def test_flux_mismatch_norm(self):
        """Test flux mismatch norm."""
        bkd = self.bkd

        flux_left = bkd.asarray([3.0, 0.0])
        flux_right = bkd.asarray([1.0, 4.0])

        norm = flux_mismatch_norm(flux_left, flux_right, bkd)

        # mismatch = [4, 4], norm = sqrt(32) = 4*sqrt(2)
        import numpy as np
        expected = np.sqrt(32)
        self.assertAlmostEqual(norm, expected, places=10)


if __name__ == "__main__":
    unittest.main()
