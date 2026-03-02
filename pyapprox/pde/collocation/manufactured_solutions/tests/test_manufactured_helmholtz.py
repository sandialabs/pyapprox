"""Parameterized tests for Helmholtz manufactured solutions with physics.

Verifies:
1. Residual = 0 at exact solution (using polynomial solutions for machine precision)
2. Jacobian correctness
3. Multiple test cases via parameterization

Note: For residual tests, we use polynomial solutions that can be exactly
represented by Chebyshev interpolation.

Sign Convention Note:
---------------------
The ManufacturedHelmholtz class produces forcing for: -Δu - k²*u = f
(DiffusionMixin: adds -Δu, ReactionMixin: subtracts k²*u)

The HelmholtzPhysics computes residual for: -Δu + k²*u = f
(using ADR with diffusion=1, reaction=-k²)

To make them compatible, we negate the wave number squared when passing to physics.
"""

from typing import Generic

import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.pde.collocation.basis import (
    ChebyshevBasis1D,
    ChebyshevBasis2D,
)
from pyapprox.pde.collocation.boundary import (
    zero_dirichlet_bc,
)
from pyapprox.pde.collocation.manufactured_solutions import (
    ManufacturedHelmholtz,
)
from pyapprox.pde.collocation.mesh import (
    TransformedMesh1D,
    TransformedMesh2D,
    create_uniform_mesh_1d,
    create_uniform_mesh_2d,
)
from pyapprox.pde.collocation.physics import HelmholtzPhysics
from pyapprox.util.backends.protocols import Array


class PhysicsDerivativeWrapper(Generic[Array]):
    """Wrapper to adapt physics interface for DerivativeChecker.

    DerivativeChecker expects:
    - bkd() method
    - nvars() method
    - nqoi() method
    - __call__(samples) for batch evaluation returning (nqoi, nsamples)
    - jacobian(sample) for single sample returning (nqoi, nvars)
    """

    def __init__(self, physics, time=0.0):
        self._physics = physics
        self._time = time
        self._backend = physics._bkd

    def bkd(self):
        return self._backend

    def nvars(self):
        return self._physics.npts()

    def nqoi(self):
        return self._physics.npts()

    def __call__(self, samples):
        # samples shape: (nvars, nsamples), return (nqoi, nsamples)
        if samples.ndim == 2:
            return self._backend.stack(
                [
                    self._physics.residual(samples[:, i], self._time)
                    for i in range(samples.shape[1])
                ],
                axis=1,
            )
        # Single sample: return (nqoi, 1)
        return self._physics.residual(samples, self._time).reshape(-1, 1)

    def jacobian(self, sample):
        # sample shape: (nvars, 1), return (nqoi, nvars)
        if sample.ndim == 2:
            sample = sample[:, 0]
        return self._physics.jacobian(sample, self._time)


class TestManufacturedHelmholtz1D:
    """Test 1D Helmholtz physics with manufactured solutions."""

    def test_steady_helmholtz_residual(self, bkd):
        """Test steady Helmholtz residual with polynomial solution."""
        npts = 12
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        # Polynomial solution: u = (1 - x^2)
        k2 = 2.0
        man_sol = ManufacturedHelmholtz(
            sol_str="(1 - x**2)",
            nvars=1,
            sqwavenum_str=str(k2),
            bkd=bkd,
            oned=True,
        )

        nodes = basis.nodes()
        u_exact = man_sol.functions["solution"](nodes.reshape(1, -1))
        forcing = man_sol.functions["forcing"](nodes.reshape(1, -1))

        # Note: ManufacturedHelmholtz uses -Δu - k²u = f convention
        # HelmholtzPhysics uses -Δu + k²u = f convention
        # So we negate k² to match
        physics = HelmholtzPhysics(
            basis, bkd, wave_number_sq=-k2, forcing=lambda t: forcing
        )

        bcs = [
            zero_dirichlet_bc(bkd, mesh.boundary_indices(0)),
            zero_dirichlet_bc(bkd, mesh.boundary_indices(1)),
        ]
        physics.set_boundary_conditions(bcs)

        residual = physics.residual(u_exact, 0.0)
        residual_with_bc, _ = physics.apply_boundary_conditions(
            residual, physics.jacobian(u_exact, 0.0), u_exact, 0.0
        )

        interior = [i for i in range(npts) if i not in [0, npts - 1]]
        interior_residual = bkd.asarray([residual_with_bc[i] for i in interior])

        bkd.assert_allclose(
            interior_residual, bkd.zeros(interior_residual.shape), atol=1e-12
        )

    def test_steady_helmholtz_jacobian(self, bkd):
        """Test Helmholtz Jacobian via derivative checker."""
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        k2 = 1.5
        man_sol = ManufacturedHelmholtz(
            sol_str="(1 - x**2)",
            nvars=1,
            sqwavenum_str=str(k2),
            bkd=bkd,
            oned=True,
        )

        nodes = basis.nodes()
        u_exact = man_sol.functions["solution"](nodes.reshape(1, -1))
        forcing = man_sol.functions["forcing"](nodes.reshape(1, -1))

        # Negate k² to match manufactured solution convention
        physics = HelmholtzPhysics(
            basis, bkd, wave_number_sq=-k2, forcing=lambda t: forcing
        )

        bcs = [
            zero_dirichlet_bc(bkd, mesh.boundary_indices(0)),
            zero_dirichlet_bc(bkd, mesh.boundary_indices(1)),
        ]
        physics.set_boundary_conditions(bcs)

        wrapper = PhysicsDerivativeWrapper(physics)
        checker = DerivativeChecker(wrapper)
        # DerivativeChecker expects sample shape (nvars, 1)
        errors = checker.check_derivatives(u_exact.reshape(-1, 1))
        assert checker.error_ratio(errors[0]) <= 1e-6


class TestManufacturedHelmholtz2D:
    """Test 2D Helmholtz physics with manufactured solutions."""

    def test_steady_helmholtz_residual_2d(self, bkd):
        """Test 2D steady Helmholtz residual."""
        npts_x, npts_y = 10, 10
        mesh = TransformedMesh2D(npts_x, npts_y, bkd)

        basis = ChebyshevBasis2D(mesh, bkd)
        mesh = create_uniform_mesh_2d((npts_x, npts_y), (-1.0, 1.0, -1.0, 1.0), bkd)

        k2 = 1.0
        man_sol = ManufacturedHelmholtz(
            sol_str="(1 - x**2)*(1 - y**2)",
            nvars=2,
            sqwavenum_str=str(k2),
            bkd=bkd,
            oned=True,
        )

        nodes_x = basis.nodes_x()
        nodes_y = basis.nodes_y()
        xx, yy = bkd.meshgrid(nodes_x, nodes_y, indexing="xy")
        nodes = bkd.stack([xx.flatten(), yy.flatten()], axis=0)
        u_exact = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        # Negate k² to match manufactured solution convention
        physics = HelmholtzPhysics(
            basis, bkd, wave_number_sq=-k2, forcing=lambda t: forcing
        )

        bcs = []
        for side in range(4):
            boundary_idx = mesh.boundary_indices(side)
            bc = zero_dirichlet_bc(bkd, boundary_idx)
            bcs.append(bc)
        physics.set_boundary_conditions(bcs)

        residual = physics.residual(u_exact, 0.0)
        residual_with_bc, _ = physics.apply_boundary_conditions(
            residual, physics.jacobian(u_exact, 0.0), u_exact, 0.0
        )

        boundary_indices = set()
        for side in range(4):
            for idx in mesh.boundary_indices(side):
                boundary_indices.add(int(idx))
        interior_indices = [i for i in range(basis.npts()) if i not in boundary_indices]
        interior_residual = bkd.asarray([residual_with_bc[i] for i in interior_indices])

        bkd.assert_allclose(
            interior_residual, bkd.zeros(interior_residual.shape), atol=1e-10
        )


class TestHelmholtz1DParameterized:
    """Parameterized 1D Helmholtz residual tests."""

    @pytest.mark.parametrize(
        "name,sol_str,k2,npts",
        [
            ("quadratic", "(1 - x**2)", 1.0, 10),
            ("quartic", "(1 - x**2)**2", 2.0, 12),
            ("cubic_sym", "x*(1 - x**2)", 0.5, 10),
            ("high_k2", "(1 - x**2)", 5.0, 12),
        ],
    )
    def test_helmholtz_1d_residual(self, bkd, name, sol_str, k2, npts):
        """Test 1D Helmholtz residual for parameterized cases."""
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        man_sol = ManufacturedHelmholtz(
            sol_str=sol_str,
            nvars=1,
            sqwavenum_str=str(k2),
            bkd=bkd,
            oned=True,
        )

        nodes = basis.nodes()
        u_exact = man_sol.functions["solution"](nodes.reshape(1, -1))
        forcing = man_sol.functions["forcing"](nodes.reshape(1, -1))

        # Negate k² to match manufactured solution convention
        physics = HelmholtzPhysics(
            basis, bkd, wave_number_sq=-k2, forcing=lambda t: forcing
        )

        bcs = [
            zero_dirichlet_bc(bkd, mesh.boundary_indices(0)),
            zero_dirichlet_bc(bkd, mesh.boundary_indices(1)),
        ]
        physics.set_boundary_conditions(bcs)

        residual = physics.residual(u_exact, 0.0)
        residual_with_bc, _ = physics.apply_boundary_conditions(
            residual, physics.jacobian(u_exact, 0.0), u_exact, 0.0
        )

        interior = [i for i in range(npts) if i not in [0, npts - 1]]
        interior_residual = bkd.asarray([residual_with_bc[i] for i in interior])

        bkd.assert_allclose(
            interior_residual,
            bkd.zeros(interior_residual.shape),
            atol=1e-10,
        )


class TestHelmholtz2DParameterized:
    """Parameterized 2D Helmholtz residual tests."""

    @pytest.mark.parametrize(
        "name,sol_str,k2,npts_x,npts_y",
        [
            ("quadratic_2d", "(1 - x**2)*(1 - y**2)", 1.0, 8, 8),
            ("quartic_2d", "(1 - x**2)**2*(1 - y**2)", 2.0, 10, 10),
            ("asymmetric", "(1 - x**2)*(1 - y**2)*x", 1.5, 10, 8),
        ],
    )
    def test_helmholtz_2d_residual(self, bkd, name, sol_str, k2, npts_x, npts_y):
        """Test 2D Helmholtz residual for parameterized cases."""
        mesh = TransformedMesh2D(npts_x, npts_y, bkd)

        basis = ChebyshevBasis2D(mesh, bkd)
        mesh = create_uniform_mesh_2d((npts_x, npts_y), (-1.0, 1.0, -1.0, 1.0), bkd)

        man_sol = ManufacturedHelmholtz(
            sol_str=sol_str,
            nvars=2,
            sqwavenum_str=str(k2),
            bkd=bkd,
            oned=True,
        )

        nodes_x = basis.nodes_x()
        nodes_y = basis.nodes_y()
        xx, yy = bkd.meshgrid(nodes_x, nodes_y, indexing="xy")
        nodes = bkd.stack([xx.flatten(), yy.flatten()], axis=0)
        u_exact = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        # Negate k² to match manufactured solution convention
        physics = HelmholtzPhysics(
            basis, bkd, wave_number_sq=-k2, forcing=lambda t: forcing
        )

        bcs = []
        for side in range(4):
            boundary_idx = mesh.boundary_indices(side)
            bc = zero_dirichlet_bc(bkd, boundary_idx)
            bcs.append(bc)
        physics.set_boundary_conditions(bcs)

        residual = physics.residual(u_exact, 0.0)
        residual_with_bc, _ = physics.apply_boundary_conditions(
            residual, physics.jacobian(u_exact, 0.0), u_exact, 0.0
        )

        boundary_indices = set()
        for side in range(4):
            for idx in mesh.boundary_indices(side):
                boundary_indices.add(int(idx))
        interior_indices = [i for i in range(basis.npts()) if i not in boundary_indices]
        interior_residual = bkd.asarray([residual_with_bc[i] for i in interior_indices])

        bkd.assert_allclose(
            interior_residual,
            bkd.zeros(interior_residual.shape),
            atol=1e-10,
        )


# Concrete backend implementations
