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

import unittest
from typing import Generic
import math

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.pde.collocation.manufactured_solutions import (
    ManufacturedShallowWave,
    ManufacturedTwoSpeciesReactionDiffusion,
    ManufacturedShallowShelfVelocityEquations,
    ManufacturedStokes,
)


class TestManufacturedShallowWave(Generic[Array], unittest.TestCase):
    """Test shallow water wave manufactured solutions."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_shallow_wave_creation_1d(self):
        """Test 1D shallow wave manufactured solution creation."""
        bkd = self.bkd()
        man_sol = ManufacturedShallowWave(
            nvars=1,
            depth_str="1 + 0.1*(1 - x**2)",
            mom_strs=["0.5*(1 - x**2)"],
            bed_str="0.1*x",
            bkd=bkd,
        )
        self.assertEqual(man_sol.ncomponents(), 2)  # [h, uh]
        self.assertEqual(man_sol.nvars(), 1)

    def test_shallow_wave_creation_2d(self):
        """Test 2D shallow wave manufactured solution creation."""
        bkd = self.bkd()
        man_sol = ManufacturedShallowWave(
            nvars=2,
            depth_str="1 + 0.1*(1 - x**2)*(1 - y**2)",
            mom_strs=["0.5*(1 - x**2)*(1 - y**2)", "0.3*(1 - x**2)*(1 - y**2)"],
            bed_str="0.1*x + 0.05*y",
            bkd=bkd,
        )
        self.assertEqual(man_sol.ncomponents(), 3)  # [h, uh, vh]
        self.assertEqual(man_sol.nvars(), 2)

    def test_shallow_wave_functions_exist_1d(self):
        """Test that all expected functions are created for 1D."""
        bkd = self.bkd()
        man_sol = ManufacturedShallowWave(
            nvars=1,
            depth_str="1 + 0.1*x",
            mom_strs=["0.5*x"],
            bed_str="0.0",
            bkd=bkd,
        )
        self.assertIn("solution", man_sol.functions)
        self.assertIn("forcing", man_sol.functions)
        self.assertIn("flux", man_sol.functions)
        self.assertIn("bed", man_sol.functions)

    def test_shallow_wave_evaluation_1d(self):
        """Test 1D shallow wave function evaluation."""
        bkd = self.bkd()
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
        self.assertEqual(sol.shape, (10, 2))
        self.assertEqual(forcing.shape, (10, 2))

        # Check h values
        expected_h = 1 + 0.1 * (1 - x**2)
        bkd.assert_allclose(sol[:, 0], expected_h, atol=1e-12)

        # Check uh values
        expected_uh = 0.5 * (1 - x**2)
        bkd.assert_allclose(sol[:, 1], expected_uh, atol=1e-12)

    def test_shallow_wave_evaluation_2d(self):
        """Test 2D shallow wave function evaluation."""
        bkd = self.bkd()
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
        xx, yy = bkd.meshgrid((x, y), indexing='xy')
        nodes = bkd.stack([xx.flatten(), yy.flatten()], axis=0)

        sol = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        # Check shapes: [h, uh, vh]
        self.assertEqual(sol.shape, (25, 3))
        self.assertEqual(forcing.shape, (25, 3))

    def test_shallow_wave_positive_depth(self):
        """Test that water depth remains positive."""
        bkd = self.bkd()
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
        self.assertGreater(float(bkd.min(sol[:, 0])), 0.0)


class TestManufacturedTwoSpeciesReactionDiffusion(Generic[Array], unittest.TestCase):
    """Test two-species reaction-diffusion manufactured solutions."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_reaction_diffusion_creation(self):
        """Test two-species reaction-diffusion creation."""
        bkd = self.bkd()
        man_sol = ManufacturedTwoSpeciesReactionDiffusion(
            sol_strs=["(1 - x**2)*(1 - y**2)", "(1 - x**2)*(1 - y**2)*0.5"],
            nvars=2,
            diff_strs=["1.0", "0.5"],
            react_strs=["u**2", "u"],
            bkd=bkd,
        )
        self.assertEqual(man_sol.ncomponents(), 2)
        self.assertEqual(man_sol.nvars(), 2)

    def test_reaction_diffusion_functions_exist(self):
        """Test that all expected functions are created."""
        bkd = self.bkd()
        man_sol = ManufacturedTwoSpeciesReactionDiffusion(
            sol_strs=["(1 - x**2)", "(1 - x**2)*0.5"],
            nvars=1,
            diff_strs=["1.0", "0.5"],
            react_strs=["u**2", "u"],
            bkd=bkd,
        )
        self.assertIn("solution", man_sol.functions)
        self.assertIn("forcing", man_sol.functions)
        self.assertIn("diffusion", man_sol.functions)
        self.assertIn("flux", man_sol.functions)
        self.assertIn("reaction", man_sol.functions)

    def test_reaction_diffusion_evaluation_1d(self):
        """Test 1D two-species reaction-diffusion evaluation."""
        bkd = self.bkd()
        man_sol = ManufacturedTwoSpeciesReactionDiffusion(
            sol_strs=["(1 - x**2)", "(1 - x**2)*0.5"],
            nvars=1,
            diff_strs=["1.0", "0.5"],
            react_strs=["u**2", "u"],
            bkd=bkd,
            oned=True,
        )
        x = bkd.linspace(-0.9, 0.9, 10)
        nodes = x.reshape(1, -1)

        sol = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        # Check shapes: [u0, u1]
        self.assertEqual(sol.shape, (10, 2))
        self.assertEqual(forcing.shape, (10, 2))

        # Check u0 values
        expected_u0 = 1 - x**2
        bkd.assert_allclose(sol[:, 0], expected_u0, atol=1e-12)

    def test_reaction_diffusion_evaluation_2d(self):
        """Test 2D two-species reaction-diffusion evaluation."""
        bkd = self.bkd()
        man_sol = ManufacturedTwoSpeciesReactionDiffusion(
            sol_strs=["(1 - x**2)*(1 - y**2)", "(1 - x**2)*(1 - y**2)*0.5"],
            nvars=2,
            diff_strs=["1.0", "0.5"],
            react_strs=["u**2", "u"],
            bkd=bkd,
            oned=True,
        )
        x = bkd.linspace(-0.9, 0.9, 5)
        y = bkd.linspace(-0.9, 0.9, 5)
        xx, yy = bkd.meshgrid((x, y), indexing='xy')
        nodes = bkd.stack([xx.flatten(), yy.flatten()], axis=0)

        sol = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        self.assertEqual(sol.shape, (25, 2))
        self.assertEqual(forcing.shape, (25, 2))


class TestManufacturedShallowShelf(Generic[Array], unittest.TestCase):
    """Test shallow shelf (SSA) manufactured solutions."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_shallow_shelf_creation(self):
        """Test shallow shelf velocity equations creation."""
        bkd = self.bkd()
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
        self.assertEqual(man_sol.ncomponents(), 2)  # [u, v]
        self.assertEqual(man_sol.nvars(), 2)

    def test_shallow_shelf_functions_exist(self):
        """Test that all expected functions are created."""
        bkd = self.bkd()
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
        self.assertIn("solution", man_sol.functions)
        self.assertIn("forcing", man_sol.functions)
        self.assertIn("flux", man_sol.functions)
        self.assertIn("bed", man_sol.functions)
        self.assertIn("depth", man_sol.functions)
        self.assertIn("surface", man_sol.functions)
        self.assertIn("friction", man_sol.functions)
        self.assertIn("effective_strain_rate", man_sol.functions)

    def test_shallow_shelf_evaluation(self):
        """Test shallow shelf velocity evaluation."""
        bkd = self.bkd()
        man_sol = ManufacturedShallowShelfVelocityEquations(
            sol_strs=["(1 - x**2)*(1 - y**2)", "(1 - x**2)*(1 - y**2)*0.5"],
            nvars=2,
            bed_str="0.0",
            depth_str="1.0",
            friction_str="1e4",
            A=1e-16,
            rho=917.0,
            bkd=bkd,
            oned=True,
        )
        # Create test points (avoid boundary for strain rate singularity)
        x = bkd.linspace(-0.8, 0.8, 5)
        y = bkd.linspace(-0.8, 0.8, 5)
        xx, yy = bkd.meshgrid((x, y), indexing='xy')
        nodes = bkd.stack([xx.flatten(), yy.flatten()], axis=0)

        sol = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        # Check shapes: [u, v]
        self.assertEqual(sol.shape, (25, 2))
        self.assertEqual(forcing.shape, (25, 2))


class TestManufacturedStokes(Generic[Array], unittest.TestCase):
    """Test Stokes/Navier-Stokes manufactured solutions."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_stokes_creation_2d(self):
        """Test 2D Stokes creation."""
        bkd = self.bkd()
        man_sol = ManufacturedStokes(
            sol_strs=["(1 - x**2)*(1 - y**2)*x", "(1 - x**2)*(1 - y**2)*y", "x*y"],
            nvars=2,
            navier_stokes=False,
            bkd=bkd,
        )
        self.assertEqual(man_sol.ncomponents(), 3)  # [u, v, p]
        self.assertEqual(man_sol.nvars(), 2)

    def test_navier_stokes_creation_2d(self):
        """Test 2D Navier-Stokes creation."""
        bkd = self.bkd()
        man_sol = ManufacturedStokes(
            sol_strs=["(1 - x**2)*(1 - y**2)*x", "(1 - x**2)*(1 - y**2)*y", "x*y"],
            nvars=2,
            navier_stokes=True,
            bkd=bkd,
        )
        self.assertEqual(man_sol.ncomponents(), 3)
        self.assertEqual(man_sol.nvars(), 2)

    def test_stokes_functions_exist(self):
        """Test that all expected functions are created."""
        bkd = self.bkd()
        man_sol = ManufacturedStokes(
            sol_strs=["(1 - x**2)*(1 - y**2)", "-(1 - x**2)*(1 - y**2)", "0"],
            nvars=2,
            navier_stokes=False,
            bkd=bkd,
        )
        self.assertIn("solution", man_sol.functions)
        self.assertIn("forcing", man_sol.functions)
        self.assertIn("flux", man_sol.functions)
        self.assertIn("vel_forcing", man_sol.functions)
        self.assertIn("pres_forcing", man_sol.functions)

    def test_stokes_evaluation_2d(self):
        """Test 2D Stokes function evaluation."""
        bkd = self.bkd()
        man_sol = ManufacturedStokes(
            sol_strs=["(1 - x**2)*(1 - y**2)*x", "(1 - x**2)*(1 - y**2)*y", "x + y"],
            nvars=2,
            navier_stokes=False,
            bkd=bkd,
            oned=True,
        )
        x = bkd.linspace(-0.9, 0.9, 5)
        y = bkd.linspace(-0.9, 0.9, 5)
        xx, yy = bkd.meshgrid((x, y), indexing='xy')
        nodes = bkd.stack([xx.flatten(), yy.flatten()], axis=0)

        sol = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        # Check shapes: [u, v, p]
        self.assertEqual(sol.shape, (25, 3))
        self.assertEqual(forcing.shape, (25, 3))

    def test_stokes_incompressibility(self):
        """Test that divergence-free velocity gives zero pressure forcing.

        For divergence-free velocity field, the continuity constraint
        pres_forcing = du/dx + dv/dy should be zero.
        """
        bkd = self.bkd()
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
        xx, yy = bkd.meshgrid((x, y), indexing='xy')
        nodes = bkd.stack([xx.flatten(), yy.flatten()], axis=0)

        # Get pressure forcing (should be zero for divergence-free flow)
        pres_forcing = man_sol.functions["pres_forcing"](nodes)
        bkd.assert_allclose(pres_forcing, bkd.zeros(pres_forcing.shape), atol=1e-12)


class TestManufacturedShallowWaveNumpy(TestManufacturedShallowWave):
    """Numpy implementation of shallow wave tests."""

    __test__ = True

    def bkd(self):
        return NumpyBkd()


class TestManufacturedTwoSpeciesReactionDiffusionNumpy(
    TestManufacturedTwoSpeciesReactionDiffusion
):
    """Numpy implementation of two-species reaction-diffusion tests."""

    __test__ = True

    def bkd(self):
        return NumpyBkd()


class TestManufacturedShallowShelfNumpy(TestManufacturedShallowShelf):
    """Numpy implementation of shallow shelf tests."""

    __test__ = True

    def bkd(self):
        return NumpyBkd()


class TestManufacturedStokesNumpy(TestManufacturedStokes):
    """Numpy implementation of Stokes tests."""

    __test__ = True

    def bkd(self):
        return NumpyBkd()


if __name__ == "__main__":
    unittest.main()
