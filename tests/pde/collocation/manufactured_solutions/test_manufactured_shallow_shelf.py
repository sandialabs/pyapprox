"""Parameterized tests for Shallow Shelf manufactured solutions with physics.

Verifies:
1. Residual = 0 at exact solution (using polynomial solutions)
2. Jacobian correctness via DerivativeChecker
3. Multiple test cases via parameterization

Note: The SSA equations have highly nonlinear viscosity (Glen's flow law with
n=3), requiring many points for spectral accuracy. Tests use normalized
parameters (A=1, rho=1) for numerical stability.

Sign Convention Note:
---------------------
The ManufacturedShallowShelfVelocityEquations computes forcing for:
    div(2*mu*H*epsilon) - C*vel - H*rho*g*grad(s) + f = 0

where the forcing f is computed symbolically to satisfy this equation.

The ShallowShelfVelocityPhysics computes residual as:
    residual = div(tau) - friction - driving_stress + forcing

For the exact manufactured solution with correct forcing, residual = 0.
"""

from typing import Generic

import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.pde.collocation.manufactured_solutions import (
    ManufacturedShallowShelfVelocityEquations,
)
from pyapprox.pde.collocation.mesh import (
    TransformedMesh2D,
)
from pyapprox.pde.collocation.physics import ShallowShelfVelocityPhysics
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array
from tests._helpers.markers import slow_test


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
        return self._physics.nstates()

    def nqoi(self):
        return self._physics.nstates()

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


class TestManufacturedShallowShelf2D:
    """Test 2D Shallow Shelf physics with manufactured solutions."""

    def test_steady_shallow_shelf_residual(self, bkd):
        """Test steady shallow shelf residual with manufactured solution."""
        npts_1d = 40  # Need many points for nonlinear SSA
        mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)

        basis = ChebyshevBasis2D(mesh, bkd)
        basis.npts()

        # Normalized parameters for numerical stability
        A = 1.0
        rho = 1.0

        man_sol = ManufacturedShallowShelfVelocityEquations(
            sol_strs=["x**2 + y", "x + y**2"],
            nvars=2,
            bed_str="0.0",
            depth_str="1.0",
            friction_str="1.0",
            A=A,
            rho=rho,
            bkd=bkd,
            oned=False,
        )

        # Get nodes using meshgrid
        nodes_x = basis.nodes_x()
        nodes_y = basis.nodes_y()
        X, Y = bkd.meshgrid(nodes_x, nodes_y, indexing="xy")
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        nodes_2d = bkd.stack([X_flat, Y_flat], axis=0)

        # Get exact solution and forcing
        exact_sol_vals = man_sol.functions["solution"](nodes_2d)
        u_exact = exact_sol_vals[:, 0]
        v_exact = exact_sol_vals[:, 1]
        exact_state = bkd.hstack([u_exact, v_exact])

        forcing_vals = man_sol.functions["forcing"](nodes_2d)
        forcing_u = forcing_vals[:, 0]
        forcing_v = forcing_vals[:, 1]
        forcing = bkd.hstack([forcing_u, forcing_v])

        # Get fields
        depth_vals = man_sol.functions["depth"](nodes_2d).flatten()
        bed_vals = man_sol.functions["bed"](nodes_2d).flatten()
        friction_vals = man_sol.functions["friction"](nodes_2d).flatten()

        # Create physics with forcing
        physics = ShallowShelfVelocityPhysics(
            basis,
            bkd,
            depth=depth_vals,
            bed=bed_vals,
            friction=friction_vals,
            A=A,
            rho=rho,
            forcing=lambda t: forcing,
        )

        # Compute residual at exact solution
        residual = physics.residual(exact_state, time=0.0)

        # Residual should be very small (spectral discretization error only)
        res_norm = float(bkd.norm(residual))
        assert res_norm < 1e-6

    def test_steady_shallow_shelf_jacobian(self, bkd):
        """Test Shallow Shelf Jacobian via derivative checker."""
        npts_1d = 6  # Smaller for Jacobian test
        mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)

        basis = ChebyshevBasis2D(mesh, bkd)
        basis.npts()

        # Normalized parameters
        A = 1.0
        rho = 1.0

        man_sol = ManufacturedShallowShelfVelocityEquations(
            sol_strs=["x**2 + y", "x + y**2"],
            nvars=2,
            bed_str="0.0",
            depth_str="1.0",
            friction_str="1.0",
            A=A,
            rho=rho,
            bkd=bkd,
            oned=False,
        )

        nodes_x = basis.nodes_x()
        nodes_y = basis.nodes_y()
        X, Y = bkd.meshgrid(nodes_x, nodes_y, indexing="xy")
        nodes_2d = bkd.stack([X.flatten(), Y.flatten()], axis=0)

        exact_sol_vals = man_sol.functions["solution"](nodes_2d)
        forcing_vals = man_sol.functions["forcing"](nodes_2d)
        forcing = bkd.hstack([forcing_vals[:, 0], forcing_vals[:, 1]])

        depth_vals = man_sol.functions["depth"](nodes_2d).flatten()
        bed_vals = man_sol.functions["bed"](nodes_2d).flatten()
        friction_vals = man_sol.functions["friction"](nodes_2d).flatten()

        physics = ShallowShelfVelocityPhysics(
            basis,
            bkd,
            depth=depth_vals,
            bed=bed_vals,
            friction=friction_vals,
            A=A,
            rho=rho,
            forcing=lambda t: forcing,
        )

        exact_state = bkd.hstack([exact_sol_vals[:, 0], exact_sol_vals[:, 1]])
        wrapper = PhysicsDerivativeWrapper(physics)
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(exact_state.reshape(-1, 1))
        assert checker.error_ratio(errors[0]) <= 1e-5

    def test_sloped_bed(self, bkd):
        """Test with sloped bed topography."""
        npts_1d = 40
        mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)

        basis = ChebyshevBasis2D(mesh, bkd)

        A = 1.0
        rho = 1.0

        man_sol = ManufacturedShallowShelfVelocityEquations(
            sol_strs=["x**2 + y", "x + y**2"],
            nvars=2,
            bed_str="0.1*x",  # Sloped bed
            depth_str="1.0",
            friction_str="1.0",
            A=A,
            rho=rho,
            bkd=bkd,
            oned=False,
        )

        nodes_x = basis.nodes_x()
        nodes_y = basis.nodes_y()
        X, Y = bkd.meshgrid(nodes_x, nodes_y, indexing="xy")
        nodes_2d = bkd.stack([X.flatten(), Y.flatten()], axis=0)

        exact_sol_vals = man_sol.functions["solution"](nodes_2d)
        forcing_vals = man_sol.functions["forcing"](nodes_2d)
        forcing = bkd.hstack([forcing_vals[:, 0], forcing_vals[:, 1]])

        depth_vals = man_sol.functions["depth"](nodes_2d).flatten()
        bed_vals = man_sol.functions["bed"](nodes_2d).flatten()
        friction_vals = man_sol.functions["friction"](nodes_2d).flatten()

        physics = ShallowShelfVelocityPhysics(
            basis,
            bkd,
            depth=depth_vals,
            bed=bed_vals,
            friction=friction_vals,
            A=A,
            rho=rho,
            forcing=lambda t: forcing,
        )

        exact_state = bkd.hstack([exact_sol_vals[:, 0], exact_sol_vals[:, 1]])
        residual = physics.residual(exact_state, time=0.0)

        res_norm = float(bkd.norm(residual))
        assert res_norm < 1e-6


class TestShallowShelf2DParameterized:
    """Parameterized 2D Shallow Shelf residual tests."""

    def bkd(self):
        return NumpyBkd()

    @pytest.mark.parametrize(
        "name,u_str,v_str,bed_str,depth_str,friction_str,npts_1d",
        [
            ("quadratic_flat", "x**2 + y", "x + y**2", "0.0", "1.0", "1.0", 40),
            ("quadratic_sloped", "x**2 + y", "x + y**2", "0.1*x", "1.0", "1.0", 40),
            ("linear_velocity", "x + 0.5*y", "0.5*x + y", "0.0", "1.0", "1.0", 30),
            ("higher_friction", "x**2 + y", "x + y**2", "0.0", "1.0", "10.0", 40),
            ("variable_depth", "x**2 + y", "x + y**2", "0.0", "1.0 + 0.1*x", "1.0", 45),
        ],
    )
    @slow_test
    def test_shallow_shelf_2d_residual(
        self, bkd, name, u_str, v_str, bed_str, depth_str, friction_str, npts_1d
    ):
        """Test 2D Shallow Shelf residual for parameterized cases."""
        mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)

        basis = ChebyshevBasis2D(mesh, bkd)

        A = 1.0
        rho = 1.0

        man_sol = ManufacturedShallowShelfVelocityEquations(
            sol_strs=[u_str, v_str],
            nvars=2,
            bed_str=bed_str,
            depth_str=depth_str,
            friction_str=friction_str,
            A=A,
            rho=rho,
            bkd=bkd,
            oned=False,
        )

        nodes_x = basis.nodes_x()
        nodes_y = basis.nodes_y()
        X, Y = bkd.meshgrid(nodes_x, nodes_y, indexing="xy")
        nodes_2d = bkd.stack([X.flatten(), Y.flatten()], axis=0)

        exact_sol_vals = man_sol.functions["solution"](nodes_2d)
        forcing_vals = man_sol.functions["forcing"](nodes_2d)
        forcing = bkd.hstack([forcing_vals[:, 0], forcing_vals[:, 1]])

        depth_vals = man_sol.functions["depth"](nodes_2d).flatten()
        bed_vals = man_sol.functions["bed"](nodes_2d).flatten()
        friction_vals = man_sol.functions["friction"](nodes_2d).flatten()

        physics = ShallowShelfVelocityPhysics(
            basis,
            bkd,
            depth=depth_vals,
            bed=bed_vals,
            friction=friction_vals,
            A=A,
            rho=rho,
            forcing=lambda t: forcing,
        )

        exact_state = bkd.hstack([exact_sol_vals[:, 0], exact_sol_vals[:, 1]])
        residual = physics.residual(exact_state, time=0.0)

        res_norm = float(bkd.norm(residual))
        # Use relative tolerance based on forcing magnitude
        forcing_norm = float(bkd.norm(forcing))
        if forcing_norm > 1e-10:
            rel_residual = res_norm / forcing_norm
            assert rel_residual < 1e-5
        else:
            assert res_norm < 1e-6

    @pytest.mark.parametrize(
        "name,u_str,v_str,npts_1d",
        [
            ("jacobian_quadratic", "x**2 + y", "x + y**2", 6),
            ("jacobian_linear", "x + 0.5*y", "0.5*x + y", 6),
            ("jacobian_cubic", "x**3 + y", "x + y**3", 7),
        ],
    )
    def test_shallow_shelf_2d_jacobian(self, bkd, name, u_str, v_str, npts_1d):
        """Test 2D Shallow Shelf Jacobian via DerivativeChecker."""
        mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)

        basis = ChebyshevBasis2D(mesh, bkd)

        A = 1.0
        rho = 1.0

        man_sol = ManufacturedShallowShelfVelocityEquations(
            sol_strs=[u_str, v_str],
            nvars=2,
            bed_str="0.0",
            depth_str="1.0",
            friction_str="1.0",
            A=A,
            rho=rho,
            bkd=bkd,
            oned=False,
        )

        nodes_x = basis.nodes_x()
        nodes_y = basis.nodes_y()
        X, Y = bkd.meshgrid(nodes_x, nodes_y, indexing="xy")
        nodes_2d = bkd.stack([X.flatten(), Y.flatten()], axis=0)

        exact_sol_vals = man_sol.functions["solution"](nodes_2d)
        forcing_vals = man_sol.functions["forcing"](nodes_2d)
        forcing = bkd.hstack([forcing_vals[:, 0], forcing_vals[:, 1]])

        depth_vals = man_sol.functions["depth"](nodes_2d).flatten()
        bed_vals = man_sol.functions["bed"](nodes_2d).flatten()
        friction_vals = man_sol.functions["friction"](nodes_2d).flatten()

        physics = ShallowShelfVelocityPhysics(
            basis,
            bkd,
            depth=depth_vals,
            bed=bed_vals,
            friction=friction_vals,
            A=A,
            rho=rho,
            forcing=lambda t: forcing,
        )

        exact_state = bkd.hstack([exact_sol_vals[:, 0], exact_sol_vals[:, 1]])
        wrapper = PhysicsDerivativeWrapper(physics)
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(exact_state.reshape(-1, 1))
        assert checker.error_ratio(errors[0]) <= 1e-5


# Concrete backend implementations
