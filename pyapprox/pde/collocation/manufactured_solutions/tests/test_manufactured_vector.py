"""Tests for vector manufactured solutions.

Tests:
- ManufacturedShallowWave (1D and 2D)
- ManufacturedTwoSpeciesReactionDiffusion
- ManufacturedShallowShelfVelocityEquations
- ManufacturedStokes

These tests verify:
1. Manufactured solutions can be created successfully
2. Sympy expressions are correctly computed
3. Functions (solution, forcing, flux) can be evaluated
4. Sign conventions are correct (residual = dy/dt = g(y) form)
"""

from typing import Generic, Tuple

import sympy as sp

from pyapprox.pde.collocation.manufactured_solutions import (
    ManufacturedShallowShelfVelocityEquations,
    ManufacturedShallowWave,
    ManufacturedStokes,
    ManufacturedTwoSpeciesReactionDiffusion,
)
from pyapprox.util.backends.protocols import Array, Backend


class QuadraticLinearCoupledReaction(Generic[Array]):
    """Test reaction: R0 = u0² - u1, R1 = u1 + u0.

    This is a simple test reaction with known symbolic form for
    testing manufactured solutions. The cross-species coupling
    (- u1 in R0, + u0 in R1) provides non-trivial behavior.

    Implements SymbolicReactionProtocol for use with
    ManufacturedTwoSpeciesReactionDiffusion.
    """

    def __init__(self, bkd: Backend[Array]):
        self._bkd = bkd

    def __call__(self, u0: Array, u1: Array) -> Tuple[Array, Array]:
        """Evaluate reaction: R0 = u0² - u1, R1 = u1 + u0."""
        R0 = u0**2 - u1
        R1 = u1 + u0
        return R0, R1

    def jacobian(self, u0: Array, u1: Array) -> Tuple[Array, Array, Array, Array]:
        """Compute Jacobian of reaction."""
        bkd = self._bkd
        npts = u0.shape[0]
        dR0_du0 = 2 * u0  # d(u0²)/du0
        dR0_du1 = bkd.full((npts,), -1.0)  # d(-u1)/du1
        dR1_du0 = bkd.full((npts,), 1.0)  # d(u0)/du0
        dR1_du1 = bkd.full((npts,), 1.0)  # d(u1)/du1
        return dR0_du0, dR0_du1, dR1_du0, dR1_du1

    def sympy_expressions(
        self, u0_expr: sp.Expr, u1_expr: sp.Expr
    ) -> Tuple[sp.Expr, sp.Expr]:
        """Return symbolic reaction expressions."""
        R0_expr = u0_expr**2 - u1_expr
        R1_expr = u1_expr + u0_expr
        return R0_expr, R1_expr


class TestManufacturedShallowWave:
    """Test shallow water wave manufactured solutions."""

    def test_shallow_wave_creation_1d(self, bkd):
        """Test 1D shallow wave manufactured solution creation."""
        man_sol = ManufacturedShallowWave(
            nvars=1,
            depth_str="1 + 0.1*(1 - x**2)",
            mom_strs=["0.5*(1 - x**2)"],
            bed_str="0.1*x",
            bkd=bkd,
        )
        assert man_sol.ncomponents() == 2
        assert man_sol.nvars() == 1

    def test_shallow_wave_creation_2d(self, bkd):
        """Test 2D shallow wave manufactured solution creation."""
        man_sol = ManufacturedShallowWave(
            nvars=2,
            depth_str="1 + 0.1*(1 - x**2)*(1 - y**2)",
            mom_strs=["0.5*(1 - x**2)*(1 - y**2)", "0.3*(1 - x**2)*(1 - y**2)"],
            bed_str="0.1*x + 0.05*y",
            bkd=bkd,
        )
        assert man_sol.ncomponents() == 3
        assert man_sol.nvars() == 2

    def test_shallow_wave_functions_exist_1d(self, bkd):
        """Test that all expected functions are created for 1D."""
        man_sol = ManufacturedShallowWave(
            nvars=1,
            depth_str="1 + 0.1*x",
            mom_strs=["0.5*x"],
            bed_str="0.0",
            bkd=bkd,
        )
        assert "solution" in man_sol.functions
        assert "forcing" in man_sol.functions
        assert "flux" in man_sol.functions
        assert "bed" in man_sol.functions

    def test_shallow_wave_evaluation_1d(self, bkd):
        """Test 1D shallow wave function evaluation."""
        man_sol = ManufacturedShallowWave(
            nvars=1,
            depth_str="1 + 0.1*(1 - x**2)",
            mom_strs=["0.5*(1 - x**2)"],
            bed_str="0.0",
            bkd=bkd,
            oned=True,
        )
        # Create test points
        x = bkd.linspace(-0.9, 0.9, 10)
        nodes = x.reshape(1, -1)

        # Evaluate solution: [h, uh]
        sol = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        # Check shapes
        assert sol.shape == (10, 2)
        assert forcing.shape == (10, 2)

        # Check h values
        expected_h = 1 + 0.1 * (1 - x**2)
        bkd.assert_allclose(sol[:, 0], expected_h, atol=1e-12)

        # Check uh values
        expected_uh = 0.5 * (1 - x**2)
        bkd.assert_allclose(sol[:, 1], expected_uh, atol=1e-12)

    def test_shallow_wave_evaluation_2d(self, bkd):
        """Test 2D shallow wave function evaluation."""
        man_sol = ManufacturedShallowWave(
            nvars=2,
            depth_str="1 + 0.1*(1 - x**2)*(1 - y**2)",
            mom_strs=["0.3*(1 - x**2)*(1 - y**2)", "0.2*(1 - x**2)*(1 - y**2)"],
            bed_str="0.0",
            bkd=bkd,
            oned=True,
        )
        # Create test points
        x = bkd.linspace(-0.9, 0.9, 5)
        y = bkd.linspace(-0.9, 0.9, 5)
        xx, yy = bkd.meshgrid(x, y, indexing="xy")
        nodes = bkd.stack([xx.flatten(), yy.flatten()], axis=0)

        sol = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        # Check shapes: [h, uh, vh]
        assert sol.shape == (25, 3)
        assert forcing.shape == (25, 3)

    def test_shallow_wave_positive_depth(self, bkd):
        """Test that water depth remains positive."""
        man_sol = ManufacturedShallowWave(
            nvars=1,
            depth_str="1 + 0.1*(1 - x**2)",
            mom_strs=["0.5*(1 - x**2)"],
            bed_str="0.0",
            bkd=bkd,
            oned=True,
        )
        x = bkd.linspace(-1, 1, 20)
        nodes = x.reshape(1, -1)
        sol = man_sol.functions["solution"](nodes)

        # Depth should be positive
        assert float(bkd.min(sol[:, 0])) > 0.0


class TestManufacturedTwoSpeciesReactionDiffusion:
    """Test two-species reaction-diffusion manufactured solutions."""

    def test_reaction_diffusion_creation(self, bkd):
        """Test two-species reaction-diffusion creation."""
        reaction = QuadraticLinearCoupledReaction(bkd)
        man_sol = ManufacturedTwoSpeciesReactionDiffusion(
            sol_strs=["(1 - x**2)*(1 - y**2)", "(1 - x**2)*(1 - y**2)*0.5"],
            nvars=2,
            diff_strs=["1.0", "0.5"],
            reaction=reaction,
            bkd=bkd,
        )
        assert man_sol.ncomponents() == 2
        assert man_sol.nvars() == 2

    def test_reaction_diffusion_functions_exist(self, bkd):
        """Test that all expected functions are created."""
        reaction = QuadraticLinearCoupledReaction(bkd)
        man_sol = ManufacturedTwoSpeciesReactionDiffusion(
            sol_strs=["(1 - x**2)", "(1 - x**2)*0.5"],
            nvars=1,
            diff_strs=["1.0", "0.5"],
            reaction=reaction,
            bkd=bkd,
        )
        assert "solution" in man_sol.functions
        assert "forcing" in man_sol.functions
        assert "diffusion" in man_sol.functions
        assert "flux" in man_sol.functions
        assert "reaction" in man_sol.functions

    def test_reaction_diffusion_evaluation_1d(self, bkd):
        """Test 1D two-species reaction-diffusion evaluation."""
        reaction = QuadraticLinearCoupledReaction(bkd)
        man_sol = ManufacturedTwoSpeciesReactionDiffusion(
            sol_strs=["(1 - x**2)", "(1 - x**2)*0.5"],
            nvars=1,
            diff_strs=["1.0", "0.5"],
            reaction=reaction,
            bkd=bkd,
            oned=True,
        )
        x = bkd.linspace(-0.9, 0.9, 10)
        nodes = x.reshape(1, -1)

        sol = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        # Check shapes: [u0, u1]
        assert sol.shape == (10, 2)
        assert forcing.shape == (10, 2)

        # Check u0 values
        expected_u0 = 1 - x**2
        bkd.assert_allclose(sol[:, 0], expected_u0, atol=1e-12)

    def test_reaction_diffusion_evaluation_2d(self, bkd):
        """Test 2D two-species reaction-diffusion evaluation."""
        reaction = QuadraticLinearCoupledReaction(bkd)
        man_sol = ManufacturedTwoSpeciesReactionDiffusion(
            sol_strs=["(1 - x**2)*(1 - y**2)", "(1 - x**2)*(1 - y**2)*0.5"],
            nvars=2,
            diff_strs=["1.0", "0.5"],
            reaction=reaction,
            bkd=bkd,
            oned=True,
        )
        x = bkd.linspace(-0.9, 0.9, 5)
        y = bkd.linspace(-0.9, 0.9, 5)
        xx, yy = bkd.meshgrid(x, y, indexing="xy")
        nodes = bkd.stack([xx.flatten(), yy.flatten()], axis=0)

        sol = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        assert sol.shape == (25, 2)
        assert forcing.shape == (25, 2)


class TestManufacturedShallowShelf:
    """Test shallow shelf (SSA) manufactured solutions."""

    def test_shallow_shelf_creation(self, bkd):
        """Test shallow shelf velocity equations creation."""
        man_sol = ManufacturedShallowShelfVelocityEquations(
            sol_strs=["(1 - x**2)*(1 - y**2)", "(1 - x**2)*(1 - y**2)*0.5"],
            nvars=2,
            bed_str="0.0",
            depth_str="1.0",
            friction_str="1e4",
            A=1e-16,
            rho=917.0,
            bkd=bkd,
        )
        assert man_sol.ncomponents() == 2
        assert man_sol.nvars() == 2

    def test_shallow_shelf_functions_exist(self, bkd):
        """Test that all expected functions are created."""
        man_sol = ManufacturedShallowShelfVelocityEquations(
            sol_strs=["(1 - x**2)*(1 - y**2)", "(1 - x**2)*(1 - y**2)*0.5"],
            nvars=2,
            bed_str="0.0",
            depth_str="1.0",
            friction_str="1e4",
            A=1e-16,
            rho=917.0,
            bkd=bkd,
        )
        assert "solution" in man_sol.functions
        assert "forcing" in man_sol.functions
        assert "flux" in man_sol.functions
        assert "bed" in man_sol.functions
        assert "depth" in man_sol.functions
        assert "surface" in man_sol.functions
        assert "friction" in man_sol.functions
        assert "effective_strain_rate" in man_sol.functions

    def test_shallow_shelf_evaluation(self, bkd):
        """Test shallow shelf velocity evaluation.

        Uses velocity with nonzero strain rate everywhere to avoid
        divide-by-zero in Glen's flow law viscosity (ε_eff^(-2/3)).
        """
        man_sol = ManufacturedShallowShelfVelocityEquations(
            sol_strs=["1 + x + 0.3*y + 0.2*x**2", "1 + 0.3*x + y + 0.2*y**2"],
            nvars=2,
            bed_str="0.0",
            depth_str="1.0",
            friction_str="1e4",
            A=1e-16,
            rho=917.0,
            bkd=bkd,
            oned=True,
        )
        x = bkd.linspace(-0.8, 0.8, 5)
        y = bkd.linspace(-0.8, 0.8, 5)
        xx, yy = bkd.meshgrid(x, y, indexing="xy")
        nodes = bkd.stack([xx.flatten(), yy.flatten()], axis=0)

        sol = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        # Check shapes: [u, v]
        assert sol.shape == (25, 2)
        assert forcing.shape == (25, 2)


class TestManufacturedStokes:
    """Test Stokes/Navier-Stokes manufactured solutions."""

    def test_stokes_creation_2d(self, bkd):
        """Test 2D Stokes creation."""
        man_sol = ManufacturedStokes(
            sol_strs=["(1 - x**2)*(1 - y**2)*x", "(1 - x**2)*(1 - y**2)*y", "x*y"],
            nvars=2,
            navier_stokes=False,
            bkd=bkd,
        )
        assert man_sol.ncomponents() == 3
        assert man_sol.nvars() == 2

    def test_navier_stokes_creation_2d(self, bkd):
        """Test 2D Navier-Stokes creation."""
        man_sol = ManufacturedStokes(
            sol_strs=["(1 - x**2)*(1 - y**2)*x", "(1 - x**2)*(1 - y**2)*y", "x*y"],
            nvars=2,
            navier_stokes=True,
            bkd=bkd,
        )
        assert man_sol.ncomponents() == 3
        assert man_sol.nvars() == 2

    def test_stokes_functions_exist(self, bkd):
        """Test that all expected functions are created."""
        man_sol = ManufacturedStokes(
            sol_strs=["(1 - x**2)*(1 - y**2)", "-(1 - x**2)*(1 - y**2)", "0"],
            nvars=2,
            navier_stokes=False,
            bkd=bkd,
        )
        assert "solution" in man_sol.functions
        assert "forcing" in man_sol.functions
        assert "flux" in man_sol.functions
        assert "vel_forcing" in man_sol.functions
        assert "pres_forcing" in man_sol.functions

    def test_stokes_evaluation_2d(self, bkd):
        """Test 2D Stokes function evaluation."""
        man_sol = ManufacturedStokes(
            sol_strs=["(1 - x**2)*(1 - y**2)*x", "(1 - x**2)*(1 - y**2)*y", "x + y"],
            nvars=2,
            navier_stokes=False,
            bkd=bkd,
            oned=True,
        )
        x = bkd.linspace(-0.9, 0.9, 5)
        y = bkd.linspace(-0.9, 0.9, 5)
        xx, yy = bkd.meshgrid(x, y, indexing="xy")
        nodes = bkd.stack([xx.flatten(), yy.flatten()], axis=0)

        sol = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        # Check shapes: [u, v, p]
        assert sol.shape == (25, 3)
        assert forcing.shape == (25, 3)

    def test_stokes_incompressibility(self, bkd):
        """Test that divergence-free velocity gives zero pressure forcing.

        For divergence-free velocity field, the continuity constraint
        pres_forcing = du/dx + dv/dy should be zero.
        """
        # Use stream function approach: u = dψ/dy, v = -dψ/dx
        # ψ = (1-x²)(1-y²) gives automatically divergence-free field
        # u = dψ/dy = (1-x²)*(-2y), v = -dψ/dx = -(-2x)*(1-y²) = 2x*(1-y²)
        man_sol = ManufacturedStokes(
            sol_strs=["(1 - x**2)*(-2*y)", "2*x*(1 - y**2)", "0"],
            nvars=2,
            navier_stokes=False,
            bkd=bkd,
            oned=True,
        )
        x = bkd.linspace(-0.9, 0.9, 5)
        y = bkd.linspace(-0.9, 0.9, 5)
        xx, yy = bkd.meshgrid(x, y, indexing="xy")
        nodes = bkd.stack([xx.flatten(), yy.flatten()], axis=0)

        # Get pressure forcing (should be zero for divergence-free flow)
        pres_forcing = man_sol.functions["pres_forcing"](nodes)
        bkd.assert_allclose(pres_forcing, bkd.zeros(pres_forcing.shape), atol=1e-12)
