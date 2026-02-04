"""Parameterized tests for Shallow Ice manufactured solutions with physics.

Verifies:
1. Residual = 0 at exact solution (using polynomial solutions)
2. Jacobian correctness via DerivativeChecker
3. Multiple test cases via parameterization

Note: Uses normalized parameters (A=1, rho=1) for numerical stability.
With glaciological parameters (A~1e-16, rho~917), forcing terms are O(1e10)
which causes numerical issues.

Sign Convention Note:
---------------------
Manufactured solution: dH/dt - div(D*grad(s)) = f
    => forcing f = -div(D*grad(s)) for steady state

Physics:            dH/dt = div(D*grad(s)) + f
    => residual = div(D*grad(s)) + f

For residual = 0: f must equal -div(D*grad(s)), which matches manufactured.
No sign negation needed.
"""

import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray
from unittest_parametrize import ParametrizedTestCase, parametrize

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401
from pyapprox.typing.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.typing.pde.collocation.mesh import create_uniform_mesh_1d
from pyapprox.typing.pde.collocation.boundary import constant_dirichlet_bc
from pyapprox.typing.pde.collocation.physics import ShallowIcePhysics
from pyapprox.typing.pde.collocation.manufactured_solutions import (
    ManufacturedShallowIce,
)
from pyapprox.typing.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)


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
                [self._physics.residual(samples[:, i], self._time)
                 for i in range(samples.shape[1])],
                axis=1
            )
        # Single sample: return (nqoi, 1)
        return self._physics.residual(samples, self._time).reshape(-1, 1)

    def jacobian(self, sample):
        # sample shape: (nvars, 1), return (nqoi, nvars)
        if sample.ndim == 2:
            sample = sample[:, 0]
        return self._physics.jacobian(sample, self._time)


class TestManufacturedShallowIce1D(Generic[Array], unittest.TestCase):
    """Test 1D Shallow Ice physics with manufactured solutions."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_steady_shallow_ice_residual(self):
        """Test steady Shallow Ice residual with manufactured solution."""
        bkd = self.bkd()
        npts = 20
        basis = ChebyshevBasis1D(npts, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)
        nodes = basis.nodes()

        # Normalized parameters for numerical stability
        A = 1.0
        rho = 1.0

        man_sol = ManufacturedShallowIce(
            sol_str="2 + 0.5*(1 - x**2)",
            nvars=1,
            bed_str="0.1*x",
            friction_str="1.0",
            A=A,
            rho=rho,
            bkd=bkd,
            oned=True,
        )

        # Get manufactured solution values
        u_exact = man_sol.functions["solution"](nodes.reshape(1, -1))
        forcing = man_sol.functions["forcing"](nodes.reshape(1, -1))
        bed = man_sol.functions["bed"](nodes.reshape(1, -1))

        # Create physics with manufactured forcing
        physics = ShallowIcePhysics(
            basis, bkd, bed=bed, friction=1.0, A=A, rho=rho,
            forcing=lambda t: forcing
        )

        # Set Dirichlet BCs at boundaries
        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)
        bc_left = constant_dirichlet_bc(
            bkd, left_idx, float(u_exact[int(left_idx)])
        )
        bc_right = constant_dirichlet_bc(
            bkd, right_idx, float(u_exact[int(right_idx)])
        )
        physics.set_boundary_conditions([bc_left, bc_right])

        residual = physics.residual(u_exact, 0.0)
        residual_with_bc, _ = physics.apply_boundary_conditions(
            residual, physics.jacobian(u_exact, 0.0), u_exact, 0.0
        )

        # Check interior residual only (boundary rows are modified by BCs)
        interior = [i for i in range(npts) if i not in [0, npts - 1]]
        interior_residual = bkd.asarray([residual_with_bc[i] for i in interior])

        bkd.assert_allclose(
            interior_residual, bkd.zeros(interior_residual.shape), atol=1e-6
        )

    def test_steady_shallow_ice_jacobian(self):
        """Test Shallow Ice Jacobian via derivative checker."""
        bkd = self.bkd()
        npts = 15
        basis = ChebyshevBasis1D(npts, bkd)
        nodes = basis.nodes()

        # Normalized parameters
        A = 1.0
        rho = 1.0

        man_sol = ManufacturedShallowIce(
            sol_str="2 + 0.5*(1 - x**2)",
            nvars=1,
            bed_str="0.1*x",
            friction_str="1.0",
            A=A,
            rho=rho,
            bkd=bkd,
            oned=True,
        )

        u_exact = man_sol.functions["solution"](nodes.reshape(1, -1))
        forcing = man_sol.functions["forcing"](nodes.reshape(1, -1))
        bed = man_sol.functions["bed"](nodes.reshape(1, -1))

        physics = ShallowIcePhysics(
            basis, bkd, bed=bed, friction=1.0, A=A, rho=rho,
            forcing=lambda t: forcing
        )

        wrapper = PhysicsDerivativeWrapper(physics)
        checker = DerivativeChecker(wrapper)
        # DerivativeChecker expects sample shape (nvars, 1)
        errors = checker.check_derivatives(u_exact.reshape(-1, 1))
        self.assertLessEqual(checker.error_ratio(errors[0]), 1e-6)

    def test_flat_bed(self):
        """Test with flat bed topography."""
        bkd = self.bkd()
        npts = 20
        basis = ChebyshevBasis1D(npts, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)
        nodes = basis.nodes()

        # Normalized parameters
        A = 1.0
        rho = 1.0

        man_sol = ManufacturedShallowIce(
            sol_str="2 + 0.5*(1 - x**2)",
            nvars=1,
            bed_str="0.0",  # Flat bed
            friction_str="1.0",
            A=A,
            rho=rho,
            bkd=bkd,
            oned=True,
        )

        u_exact = man_sol.functions["solution"](nodes.reshape(1, -1))
        forcing = man_sol.functions["forcing"](nodes.reshape(1, -1))
        bed = man_sol.functions["bed"](nodes.reshape(1, -1))

        physics = ShallowIcePhysics(
            basis, bkd, bed=bed, friction=1.0, A=A, rho=rho,
            forcing=lambda t: forcing
        )

        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)
        bc_left = constant_dirichlet_bc(
            bkd, left_idx, float(u_exact[int(left_idx)])
        )
        bc_right = constant_dirichlet_bc(
            bkd, right_idx, float(u_exact[int(right_idx)])
        )
        physics.set_boundary_conditions([bc_left, bc_right])

        residual = physics.residual(u_exact, 0.0)
        residual_with_bc, _ = physics.apply_boundary_conditions(
            residual, physics.jacobian(u_exact, 0.0), u_exact, 0.0
        )

        interior = [i for i in range(npts) if i not in [0, npts - 1]]
        interior_residual = bkd.asarray([residual_with_bc[i] for i in interior])

        bkd.assert_allclose(
            interior_residual, bkd.zeros(interior_residual.shape), atol=1e-6
        )


class TestShallowIce1DParameterized(ParametrizedTestCase):
    """Parameterized 1D Shallow Ice residual tests."""

    def bkd(self):
        return NumpyBkd()

    @parametrize(
        "name,sol_str,bed_str,friction_str,npts",
        [
            ("quadratic_flat_bed", "2 + 0.5*(1 - x**2)", "0.0", "1.0", 20),
            ("quadratic_sloped_bed", "2 + 0.5*(1 - x**2)", "0.1*x", "1.0", 20),
            ("quadratic_curved_bed", "2 + 0.5*(1 - x**2)", "0.05*x**2", "1.0", 20),
            ("higher_friction", "2 + 0.5*(1 - x**2)", "0.1*x", "10.0", 20),
            ("quartic_profile", "2 + 0.25*(1 - x**2)**2", "0.1*x", "1.0", 30),
        ],
    )
    def test_shallow_ice_1d_residual(
        self, name, sol_str, bed_str, friction_str, npts
    ):
        """Test 1D Shallow Ice residual for parameterized cases."""
        bkd = self.bkd()
        basis = ChebyshevBasis1D(npts, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)
        nodes = basis.nodes()

        # Normalized parameters for numerical stability
        A = 1.0
        rho = 1.0
        friction = float(friction_str)

        man_sol = ManufacturedShallowIce(
            sol_str=sol_str,
            nvars=1,
            bed_str=bed_str,
            friction_str=friction_str,
            A=A,
            rho=rho,
            bkd=bkd,
            oned=True,
        )

        u_exact = man_sol.functions["solution"](nodes.reshape(1, -1))
        forcing = man_sol.functions["forcing"](nodes.reshape(1, -1))
        bed = man_sol.functions["bed"](nodes.reshape(1, -1))

        physics = ShallowIcePhysics(
            basis, bkd, bed=bed, friction=friction, A=A, rho=rho,
            forcing=lambda t: forcing
        )

        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)
        bc_left = constant_dirichlet_bc(
            bkd, left_idx, float(u_exact[int(left_idx)])
        )
        bc_right = constant_dirichlet_bc(
            bkd, right_idx, float(u_exact[int(right_idx)])
        )
        physics.set_boundary_conditions([bc_left, bc_right])

        residual = physics.residual(u_exact, 0.0)
        residual_with_bc, _ = physics.apply_boundary_conditions(
            residual, physics.jacobian(u_exact, 0.0), u_exact, 0.0
        )

        interior = [i for i in range(npts) if i not in [0, npts - 1]]
        interior_residual = bkd.asarray([residual_with_bc[i] for i in interior])

        bkd.assert_allclose(
            interior_residual,
            bkd.zeros(interior_residual.shape),
            atol=1e-4,
        )

    @parametrize(
        "name,sol_str,bed_str,npts",
        [
            ("jacobian_basic", "2 + 0.5*(1 - x**2)", "0.1*x", 15),
            ("jacobian_flat_bed", "2 + 0.5*(1 - x**2)", "0.0", 15),
            ("jacobian_curved_bed", "2 + 0.5*(1 - x**2)", "0.05*x**2", 15),
        ],
    )
    def test_shallow_ice_1d_jacobian(self, name, sol_str, bed_str, npts):
        """Test 1D Shallow Ice Jacobian via DerivativeChecker."""
        bkd = self.bkd()
        basis = ChebyshevBasis1D(npts, bkd)
        nodes = basis.nodes()

        # Normalized parameters
        A = 1.0
        rho = 1.0

        man_sol = ManufacturedShallowIce(
            sol_str=sol_str,
            nvars=1,
            bed_str=bed_str,
            friction_str="1.0",
            A=A,
            rho=rho,
            bkd=bkd,
            oned=True,
        )

        u_exact = man_sol.functions["solution"](nodes.reshape(1, -1))
        forcing = man_sol.functions["forcing"](nodes.reshape(1, -1))
        bed = man_sol.functions["bed"](nodes.reshape(1, -1))

        physics = ShallowIcePhysics(
            basis, bkd, bed=bed, friction=1.0, A=A, rho=rho,
            forcing=lambda t: forcing
        )

        wrapper = PhysicsDerivativeWrapper(physics)
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(u_exact.reshape(-1, 1))
        self.assertLessEqual(checker.error_ratio(errors[0]), 1e-6)


# Concrete backend implementations
class TestManufacturedShallowIce1DNumpy(
    TestManufacturedShallowIce1D[NDArray[Any]]
):
    """NumPy backend tests for 1D Shallow Ice."""

    __test__ = True

    def bkd(self):
        return NumpyBkd()


class TestManufacturedShallowIce1DTorch(
    TestManufacturedShallowIce1D[torch.Tensor]
):
    """PyTorch backend tests for 1D Shallow Ice."""

    __test__ = True

    def bkd(self):
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = self.bkd()


if __name__ == "__main__":
    unittest.main()
