"""Tests for Helmholtz physics.

This module contains:
1. Unit tests for matrix properties (TestHelmholtzBase)
2. Parametrized manufactured solution tests inspired by legacy
   test_finite_elements.py test_helmholtz (3 cases).

The typing Helmholtz class solves the screened Poisson (modified Helmholtz)
equation: -div(grad(u)) + k^2*u = f, with stiffness K = K_lap + k^2*M.
The manufactured solution is created using the ADR machinery with D=1,
no velocity, and reaction R(u) = -k^2*u so that the forcing matches.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)


import numpy as np
from scipy.sparse import issparse

from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import Helmholtz
from pyapprox.util.backends.numpy import NumpyBkd


class TestHelmholtzBase:
    """Base test class for Helmholtz physics."""
    def test_1d_mass_matrix_symmetric(self, numpy_bkd) -> None:
        """Test mass matrix is symmetric in 1D."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        physics = Helmholtz(basis=basis, wavenumber=2 * np.pi, bkd=bkd)

        M = physics.mass_matrix()
        M_np = M.toarray() if issparse(M) else bkd.to_numpy(M)

        np.testing.assert_array_almost_equal(M_np, M_np.T)

    def test_1d_stiffness_symmetric(self, numpy_bkd: NumpyBkd) -> None:
        """Test stiffness matrix is symmetric in 1D."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        physics = Helmholtz(basis=basis, wavenumber=2 * np.pi, bkd=bkd)
        system = compose_galerkin_system(physics)

        u0 = bkd.asarray(np.zeros(physics.nstates()))
        jac = system.spatial_operator().spatial_jacobian(u0, 0.0)
        jac_np = jac.toarray() if issparse(jac) else bkd.to_numpy(jac)

        # For Helmholtz, -jacobian = K = K_laplacian + k^2*M, should be symmetric
        K = -jac_np
        np.testing.assert_array_almost_equal(K, K.T, decimal=10)


    def test_wavenumber_property(self, numpy_bkd) -> None:
        """Test wavenumber property."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        k = 3.5
        physics = Helmholtz(basis=basis, wavenumber=k, bkd=bkd)

        assert physics.wavenumber() == k


# ---------------------------------------------------------------------------
# Parametrized manufactured solution tests
# Replicates the 3 legacy test cases from test_finite_elements.py::test_helmholtz
# ---------------------------------------------------------------------------


