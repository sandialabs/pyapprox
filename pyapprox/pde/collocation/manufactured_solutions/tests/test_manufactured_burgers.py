"""Parameterized tests for Burgers manufactured solutions with physics.

Verifies:
1. Residual = 0 at exact solution (using polynomial solutions for machine precision)
2. Jacobian correctness via DerivativeChecker
3. Multiple test cases via parameterization

Note: For residual tests, we use polynomial solutions that can be exactly
represented by Chebyshev interpolation.

Sign Convention Note:
---------------------
The ManufacturedBurgers1D class computes forcing for the conservative form:
    du/dt + d/dx(u²/2 - ν*du/dx) = f
    => forcing f = d/dx(u²/2 - ν*du/dx) = u*du/dx - ν*d²u/dx²

The BurgersPhysics1D computes residual for:
    du/dt = -u*du/dx + ν*d²u/dx² + f
    => residual = -u*du/dx + ν*d²u/dx² + f

For steady state (du/dt = 0), the physics gives:
    0 = -u*du/dx + ν*d²u/dx² + f
    => f = u*du/dx - ν*d²u/dx²

This matches the manufactured solution forcing, so no sign negation is needed.
"""

import pytest
from typing import Generic



from pyapprox.util.backends.protocols import Array
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.boundary import zero_dirichlet_bc
from pyapprox.pde.collocation.manufactured_solutions import (
    ManufacturedBurgers1D,
)
from pyapprox.pde.collocation.mesh import (
    TransformedMesh1D,
    create_uniform_mesh_1d,
)
from pyapprox.pde.collocation.physics import BurgersPhysics1D
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


class TestManufacturedBurgers1D:
    """Test 1D Burgers physics with manufactured solutions."""

    def test_steady_burgers_residual_quadratic(self, bkd):
        """Test steady Burgers residual with quadratic solution u = 1 - x²."""
        npts = 15
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        # Polynomial solution: u = (1 - x²)
        # This satisfies homogeneous Dirichlet BCs at x = ±1
        nu = 0.1
        man_sol = ManufacturedBurgers1D(
            sol_str="(1 - x**2)",
            visc_str=str(nu),
            bkd=bkd,
            oned=True,
        )

        nodes = basis.nodes()
        u_exact = man_sol.functions["solution"](nodes.reshape(1, -1))
        forcing = man_sol.functions["forcing"](nodes.reshape(1, -1))

        physics = BurgersPhysics1D(basis, bkd, viscosity=nu, forcing=lambda t: forcing)

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
            interior_residual, bkd.zeros(interior_residual.shape), atol=1e-10
        )

    def test_steady_burgers_jacobian(self, bkd):
        """Test Burgers Jacobian via derivative checker."""
        npts = 12
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        nu = 0.1
        man_sol = ManufacturedBurgers1D(
            sol_str="(1 - x**2)",
            visc_str=str(nu),
            bkd=bkd,
            oned=True,
        )

        nodes = basis.nodes()
        u_exact = man_sol.functions["solution"](nodes.reshape(1, -1))
        forcing = man_sol.functions["forcing"](nodes.reshape(1, -1))

        physics = BurgersPhysics1D(basis, bkd, viscosity=nu, forcing=lambda t: forcing)

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

    def test_steady_burgers_high_viscosity(self, bkd):
        """Test with high viscosity (diffusion-dominated)."""
        npts = 15
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        # High viscosity makes the equation more diffusion-dominated
        nu = 1.0
        man_sol = ManufacturedBurgers1D(
            sol_str="(1 - x**2)",
            visc_str=str(nu),
            bkd=bkd,
            oned=True,
        )

        nodes = basis.nodes()
        u_exact = man_sol.functions["solution"](nodes.reshape(1, -1))
        forcing = man_sol.functions["forcing"](nodes.reshape(1, -1))

        physics = BurgersPhysics1D(basis, bkd, viscosity=nu, forcing=lambda t: forcing)

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
            interior_residual, bkd.zeros(interior_residual.shape), atol=1e-10
        )

    def test_non_conservative_form(self, bkd):
        """Test non-conservative form of Burgers equation."""
        npts = 15
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        nu = 0.1
        man_sol = ManufacturedBurgers1D(
            sol_str="(1 - x**2)",
            visc_str=str(nu),
            bkd=bkd,
            oned=True,
        )

        nodes = basis.nodes()
        u_exact = man_sol.functions["solution"](nodes.reshape(1, -1))
        forcing = man_sol.functions["forcing"](nodes.reshape(1, -1))

        # Use non-conservative form
        physics = BurgersPhysics1D(
            basis, bkd, viscosity=nu, forcing=lambda t: forcing, conservative=False
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

        # Non-conservative form should also give zero residual
        bkd.assert_allclose(
            interior_residual, bkd.zeros(interior_residual.shape), atol=1e-10
        )


class TestBurgers1DParameterized:
    """Parameterized 1D Burgers residual tests."""

    @pytest.mark.parametrize(
        "name,sol_str,viscosity,npts",
        [
            ("quadratic_low_nu", "(1 - x**2)", 0.05, 15),
            ("quadratic_med_nu", "(1 - x**2)", 0.5, 15),
            ("quadratic_high_nu", "(1 - x**2)", 2.0, 15),
            ("quartic", "(1 - x**2)**2", 0.1, 18),
            ("cubic_odd", "x*(1 - x**2)", 0.1, 15),
            ("sixth_order", "(1 - x**2)**3", 0.1, 20),
        ],
    )
    def test_burgers_1d_residual(self, bkd, name, sol_str, viscosity, npts):
        """Test 1D Burgers residual for parameterized cases."""
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        man_sol = ManufacturedBurgers1D(
            sol_str=sol_str,
            visc_str=str(viscosity),
            bkd=bkd,
            oned=True,
        )

        nodes = basis.nodes()
        u_exact = man_sol.functions["solution"](nodes.reshape(1, -1))
        forcing = man_sol.functions["forcing"](nodes.reshape(1, -1))

        physics = BurgersPhysics1D(
            basis, bkd, viscosity=viscosity, forcing=lambda t: forcing
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
            atol=1e-9,
        )

    @pytest.mark.parametrize(
        "name,sol_str,viscosity,npts",
        [
            ("quadratic_jacobian", "(1 - x**2)", 0.1, 12),
            ("quartic_jacobian", "(1 - x**2)**2", 0.2, 14),
            ("cubic_jacobian", "x*(1 - x**2)", 0.05, 12),
        ],
    )
    def test_burgers_1d_jacobian(self, bkd, name, sol_str, viscosity, npts):
        """Test 1D Burgers Jacobian via DerivativeChecker."""
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        man_sol = ManufacturedBurgers1D(
            sol_str=sol_str,
            visc_str=str(viscosity),
            bkd=bkd,
            oned=True,
        )

        nodes = basis.nodes()
        u_exact = man_sol.functions["solution"](nodes.reshape(1, -1))
        forcing = man_sol.functions["forcing"](nodes.reshape(1, -1))

        physics = BurgersPhysics1D(
            basis, bkd, viscosity=viscosity, forcing=lambda t: forcing
        )

        bcs = [
            zero_dirichlet_bc(bkd, mesh.boundary_indices(0)),
            zero_dirichlet_bc(bkd, mesh.boundary_indices(1)),
        ]
        physics.set_boundary_conditions(bcs)

        wrapper = PhysicsDerivativeWrapper(physics)
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(u_exact.reshape(-1, 1))
        assert checker.error_ratio(errors[0]) <= 1e-6


# Concrete backend implementations
