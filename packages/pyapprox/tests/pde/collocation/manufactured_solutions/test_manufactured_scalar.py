"""Tests for scalar manufactured solutions (Helmholtz, Burgers, ShallowIce).

These tests verify:
1. Manufactured solutions can be created successfully
2. Sympy expressions are correctly computed
3. Functions (solution, forcing, flux) can be evaluated
4. Consistency checks between expressions and evaluations
"""

import math

from pyapprox.pde.collocation.manufactured_solutions import (
    ManufacturedBurgers1D,
    ManufacturedHelmholtz,
    ManufacturedShallowIce,
)


class TestManufacturedHelmholtz:
    """Test Helmholtz manufactured solutions.

    Helmholtz equation: -Δu + k²*u = f
    """

    def test_helmholtz_creation_1d(self, bkd):
        """Test 1D Helmholtz manufactured solution creation."""
        man_sol = ManufacturedHelmholtz(
            sol_str="sin(pi*x)",
            nvars=1,
            sqwavenum_str="1.0",
            bkd=bkd,
        )
        assert man_sol.ncomponents() == 1
        assert man_sol.nvars() == 1

    def test_helmholtz_creation_2d(self, bkd):
        """Test 2D Helmholtz manufactured solution creation."""
        man_sol = ManufacturedHelmholtz(
            sol_str="sin(pi*x)*sin(pi*y)",
            nvars=2,
            sqwavenum_str="2.0",
            bkd=bkd,
        )
        assert man_sol.ncomponents() == 1
        assert man_sol.nvars() == 2

    def test_helmholtz_functions_exist(self, bkd):
        """Test that all expected functions are created."""
        man_sol = ManufacturedHelmholtz(
            sol_str="sin(pi*x)",
            nvars=1,
            sqwavenum_str="1.0",
            bkd=bkd,
        )
        assert "solution" in man_sol.functions
        assert "forcing" in man_sol.functions
        assert "diffusion" in man_sol.functions
        assert "flux" in man_sol.functions
        assert "reaction" in man_sol.functions
        assert "sqwavenum" in man_sol.functions

    def test_helmholtz_evaluation_1d(self, bkd):
        """Test 1D Helmholtz function evaluation."""
        man_sol = ManufacturedHelmholtz(
            sol_str="sin(pi*x)",
            nvars=1,
            sqwavenum_str="1.0",
            bkd=bkd,
            oned=True,
        )
        # Create test points
        x = bkd.linspace(-1, 1, 10)
        nodes = x.reshape(1, -1)

        # Evaluate functions
        u = man_sol.functions["solution"](nodes)
        f = man_sol.functions["forcing"](nodes)

        # Check shapes
        assert u.shape == (10,)
        assert f.shape == (10,)

        # Check solution values
        expected_u = bkd.sin(math.pi * x)
        bkd.assert_allclose(u, expected_u, atol=1e-12)

    def test_helmholtz_evaluation_2d(self, bkd):
        """Test 2D Helmholtz function evaluation."""
        man_sol = ManufacturedHelmholtz(
            sol_str="sin(pi*x)*sin(pi*y)",
            nvars=2,
            sqwavenum_str="1.0",
            bkd=bkd,
            oned=True,
        )
        # Create test points
        x = bkd.linspace(0, 1, 5)
        y = bkd.linspace(0, 1, 5)
        xx, yy = bkd.meshgrid(x, y, indexing="xy")
        nodes = bkd.stack([xx.flatten(), yy.flatten()], axis=0)

        # Evaluate functions
        u = man_sol.functions["solution"](nodes)
        f = man_sol.functions["forcing"](nodes)

        # Check shapes
        assert u.shape == (25,)
        assert f.shape == (25,)

    def test_helmholtz_forcing_consistency(self, bkd):
        """Test that forcing is consistent with manufactured solution framework.

        The framework uses ADR-style conventions where:
        - DiffusionMixin adds -Δu to forcing
        - ReactionMixin subtracts R(u) from forcing

        For u = x*(1-x) in 1D on [0, 1]:
        - du/dx = 1 - 2x
        - d²u/dx² = -2
        - -Δu = 2 (diffusion contribution)
        - Reaction R(u) = k²*u, forcing -= R(u)

        So: f = -Δu - k²*u = 2 - k²*x*(1-x)

        Note: This corresponds to the PDE: -Δu - k²*u = f
        (different sign from standard Helmholtz -Δu + k²*u = f)
        """
        k2 = 3.0
        man_sol = ManufacturedHelmholtz(
            sol_str="x*(1 - x)",
            nvars=1,
            sqwavenum_str=str(k2),
            bkd=bkd,
            oned=True,
        )
        # Test at specific points
        x = bkd.linspace(0, 1, 5)
        nodes = x.reshape(1, -1)

        u = man_sol.functions["solution"](nodes)
        f = man_sol.functions["forcing"](nodes)

        # Expected: f = 2 - k²*u (ADR-style sign convention)
        expected_f = 2.0 - k2 * u
        bkd.assert_allclose(f, expected_f, atol=1e-12)


class TestManufacturedBurgers1D:
    """Test 1D Burgers manufactured solutions.

    Burgers equation: du/dt + d/dx(u²/2 - ν*du/dx) = f
    """

    def test_burgers_creation(self, bkd):
        """Test Burgers manufactured solution creation."""
        man_sol = ManufacturedBurgers1D(
            sol_str="sin(pi*x)",
            visc_str="0.1",
            bkd=bkd,
        )
        assert man_sol.ncomponents() == 1
        assert man_sol.nvars() == 1

    def test_burgers_functions_exist(self, bkd):
        """Test that all expected functions are created."""
        man_sol = ManufacturedBurgers1D(
            sol_str="sin(pi*x)",
            visc_str="0.1",
            bkd=bkd,
        )
        assert "solution" in man_sol.functions
        assert "forcing" in man_sol.functions
        assert "viscosity" in man_sol.functions
        assert "flux" in man_sol.functions

    def test_burgers_evaluation(self, bkd):
        """Test Burgers function evaluation."""
        man_sol = ManufacturedBurgers1D(
            sol_str="sin(pi*x)",
            visc_str="0.1",
            bkd=bkd,
            oned=True,
        )
        # Create test points
        x = bkd.linspace(-1, 1, 10)
        nodes = x.reshape(1, -1)

        # Evaluate functions
        u = man_sol.functions["solution"](nodes)
        f = man_sol.functions["forcing"](nodes)
        flux = man_sol.functions["flux"](nodes)

        # Check shapes
        assert u.shape == (10,)
        assert f.shape == (10,)
        # Flux is a list with 1 component
        assert flux.shape == (10, 1)

    def test_burgers_forcing_consistency(self, bkd):
        """Test forcing consistency for Burgers equation.

        For u = x*(1-x**2) with viscosity ν:
        - Flux F = u²/2 - ν*du/dx
        - Forcing = dF/dx
        """
        visc = 0.1
        man_sol = ManufacturedBurgers1D(
            sol_str="x*(1 - x**2)",  # = x - x**3
            visc_str=str(visc),
            bkd=bkd,
            oned=True,
        )
        # Test at specific points
        x = bkd.linspace(-0.9, 0.9, 5)  # Avoid boundaries
        nodes = x.reshape(1, -1)

        u = man_sol.functions["solution"](nodes)
        f = man_sol.functions["forcing"](nodes)

        # Expected solution: u = x - x**3
        expected_u = x - x**3
        bkd.assert_allclose(u, expected_u, atol=1e-12)

        # Verify forcing is non-zero (nonlinear equation)
        assert float(bkd.max(bkd.abs(f))) > 0.0

    def test_burgers_transient(self, bkd):
        """Test transient Burgers manufactured solution."""
        man_sol = ManufacturedBurgers1D(
            sol_str="(1 - T)*sin(pi*x)",
            visc_str="0.1",
            bkd=bkd,
            oned=True,
        )
        # Should be transient
        assert man_sol.is_transient()

        # Evaluate at different times
        x = bkd.linspace(-1, 1, 10)
        nodes = x.reshape(1, -1)

        u0 = man_sol.functions["solution"](nodes, 0.0)
        u1 = man_sol.functions["solution"](nodes, 0.5)

        # At t=0, u = sin(pi*x)
        expected_u0 = bkd.sin(math.pi * x)
        bkd.assert_allclose(u0, expected_u0, atol=1e-12)

        # At t=0.5, u = 0.5*sin(pi*x)
        expected_u1 = 0.5 * bkd.sin(math.pi * x)
        bkd.assert_allclose(u1, expected_u1, atol=1e-12)


class TestManufacturedShallowIce:
    """Test Shallow Ice Approximation manufactured solutions.

    SIA equation: dH/dt - div(D*grad(s)) = f
    where H = ice thickness, s = surface = H + bed
    """

    def test_shallow_ice_creation_1d(self, bkd):
        """Test 1D shallow ice manufactured solution creation."""
        man_sol = ManufacturedShallowIce(
            sol_str="1 + 0.5*(1 - x**2)",
            nvars=1,
            bed_str="0.1*x",
            friction_str="1e6",
            A=1e-16,
            rho=917.0,
            bkd=bkd,
        )
        assert man_sol.ncomponents() == 1
        assert man_sol.nvars() == 1

    def test_shallow_ice_creation_2d(self, bkd):
        """Test 2D shallow ice manufactured solution creation."""
        man_sol = ManufacturedShallowIce(
            sol_str="1 + 0.25*(1 - x**2)*(1 - y**2)",
            nvars=2,
            bed_str="0.1*x + 0.1*y",
            friction_str="1e6",
            A=1e-16,
            rho=917.0,
            bkd=bkd,
        )
        assert man_sol.ncomponents() == 1
        assert man_sol.nvars() == 2

    def test_shallow_ice_functions_exist(self, bkd):
        """Test that all expected functions are created."""
        man_sol = ManufacturedShallowIce(
            sol_str="1 + 0.5*(1 - x**2)",
            nvars=1,
            bed_str="0.0",
            friction_str="1e6",
            A=1e-16,
            rho=917.0,
            bkd=bkd,
        )
        assert "solution" in man_sol.functions
        assert "forcing" in man_sol.functions
        assert "bed" in man_sol.functions
        assert "friction" in man_sol.functions
        assert "surface" in man_sol.functions
        assert "diffusion" in man_sol.functions
        assert "flux" in man_sol.functions

    def test_shallow_ice_evaluation_1d(self, bkd):
        """Test 1D shallow ice function evaluation."""
        man_sol = ManufacturedShallowIce(
            sol_str="1 + 0.5*(1 - x**2)",
            nvars=1,
            bed_str="0.0",
            friction_str="1e6",
            A=1e-16,
            rho=917.0,
            bkd=bkd,
            oned=True,
        )
        # Create test points
        x = bkd.linspace(-0.9, 0.9, 10)
        nodes = x.reshape(1, -1)

        # Evaluate functions
        H = man_sol.functions["solution"](nodes)
        f = man_sol.functions["forcing"](nodes)
        bed = man_sol.functions["bed"](nodes)
        surface = man_sol.functions["surface"](nodes)

        # Check shapes
        assert H.shape == (10,)
        assert f.shape == (10,)
        assert bed.shape == (10,)
        assert surface.shape == (10,)

        # Check solution values: H = 1 + 0.5*(1 - x**2)
        expected_H = 1 + 0.5 * (1 - x**2)
        bkd.assert_allclose(H, expected_H, atol=1e-12)

        # Check surface = H + bed (bed = 0)
        bkd.assert_allclose(surface, H, atol=1e-12)

    def test_shallow_ice_evaluation_2d(self, bkd):
        """Test 2D shallow ice function evaluation."""
        man_sol = ManufacturedShallowIce(
            sol_str="1 + 0.25*(1 - x**2)*(1 - y**2)",
            nvars=2,
            bed_str="0.0",
            friction_str="1e6",
            A=1e-16,
            rho=917.0,
            bkd=bkd,
            oned=True,
        )
        # Create test points
        x = bkd.linspace(-0.9, 0.9, 5)
        y = bkd.linspace(-0.9, 0.9, 5)
        xx, yy = bkd.meshgrid(x, y, indexing="xy")
        nodes = bkd.stack([xx.flatten(), yy.flatten()], axis=0)

        # Evaluate functions
        H = man_sol.functions["solution"](nodes)
        f = man_sol.functions["forcing"](nodes)

        # Check shapes
        assert H.shape == (25,)
        assert f.shape == (25,)

    def test_shallow_ice_positive_thickness(self, bkd):
        """Test that ice thickness remains positive."""
        man_sol = ManufacturedShallowIce(
            sol_str="1 + 0.5*(1 - x**2)",
            nvars=1,
            bed_str="0.0",
            friction_str="1e6",
            A=1e-16,
            rho=917.0,
            bkd=bkd,
            oned=True,
        )
        x = bkd.linspace(-1, 1, 20)
        nodes = x.reshape(1, -1)
        H = man_sol.functions["solution"](nodes)

        # Ice thickness should be positive
        assert float(bkd.min(H)) > 0.0

    def test_shallow_ice_with_bed(self, bkd):
        """Test shallow ice with non-zero bed topography."""
        man_sol = ManufacturedShallowIce(
            sol_str="1 + 0.5*(1 - x**2)",
            nvars=1,
            bed_str="0.1*x",
            friction_str="1e6",
            A=1e-16,
            rho=917.0,
            bkd=bkd,
            oned=True,
        )
        x = bkd.linspace(-0.9, 0.9, 10)
        nodes = x.reshape(1, -1)

        H = man_sol.functions["solution"](nodes)
        bed = man_sol.functions["bed"](nodes)
        surface = man_sol.functions["surface"](nodes)

        # Surface = H + bed
        expected_surface = H + bed
        bkd.assert_allclose(surface, expected_surface, atol=1e-12)

    def test_shallow_ice_transient(self, bkd):
        """Test transient shallow ice manufactured solution."""
        man_sol = ManufacturedShallowIce(
            sol_str="(1 + 0.1*T)*(1 + 0.5*(1 - x**2))",
            nvars=1,
            bed_str="0.0",
            friction_str="1e6",
            A=1e-16,
            rho=917.0,
            bkd=bkd,
            oned=True,
        )
        # Should be transient
        assert man_sol.is_transient()

        # Evaluate at different times
        x = bkd.linspace(-0.9, 0.9, 10)
        nodes = x.reshape(1, -1)

        H0 = man_sol.functions["solution"](nodes, 0.0)
        H1 = man_sol.functions["solution"](nodes, 1.0)

        # At t=0: H = 1*(1 + 0.5*(1 - x**2))
        expected_H0 = 1 + 0.5 * (1 - x**2)
        bkd.assert_allclose(H0, expected_H0, atol=1e-12)

        # At t=1: H = 1.1*(1 + 0.5*(1 - x**2))
        expected_H1 = 1.1 * (1 + 0.5 * (1 - x**2))
        bkd.assert_allclose(H1, expected_H1, atol=1e-12)


class TestManufacturedHelmholtzNumpy(TestManufacturedHelmholtz):
    """Numpy implementation of Helmholtz tests."""


class TestManufacturedBurgers1DNumpy(TestManufacturedBurgers1D):
    """Numpy implementation of Burgers tests."""


class TestManufacturedShallowIceNumpy(TestManufacturedShallowIce):
    """Numpy implementation of Shallow Ice tests."""
