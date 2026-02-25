"""Parameterized tests for Linear Elasticity manufactured solutions with physics.

Verifies:
1. Residual = 0 at exact solution (using polynomial solutions)
2. Jacobian correctness via DerivativeChecker
3. Multiple test cases via parameterization
4. Dual backend support (NumPy and PyTorch)

Note: Linear elasticity is tested with polynomial solutions that can be exactly
represented by the tensor product Chebyshev basis, giving machine precision
residuals for interior points.
"""

import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray
from unittest_parametrize import ParametrizedTestCase, parametrize

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.pde.collocation.boundary import zero_dirichlet_bc
from pyapprox.pde.collocation.manufactured_solutions import (
    ManufacturedLinearElasticityEquations,
)
from pyapprox.pde.collocation.mesh import (
    TransformedMesh2D,
    create_uniform_mesh_2d,
)
from pyapprox.pde.collocation.physics import LinearElasticityPhysics
from pyapprox.pde.collocation.time_integration import CollocationModel
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401


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


class TestManufacturedLinearElasticity2D(Generic[Array], unittest.TestCase):
    """Test 2D Linear Elasticity physics with manufactured solutions."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_residual_polynomial(self):
        """Test residual = 0 at exact polynomial solution."""
        bkd = self.bkd()
        npts_x, npts_y = 10, 10
        mesh = TransformedMesh2D(npts_x, npts_y, bkd)

        basis = ChebyshevBasis2D(mesh, bkd)
        mesh = create_uniform_mesh_2d((npts_x, npts_y), (-1.0, 1.0, -1.0, 1.0), bkd)

        # Polynomial manufactured solution
        man_sol = ManufacturedLinearElasticityEquations(
            sol_strs=["(1 - x**2)*(1 - y**2)", "(1 - x**2)*(1 - y**2)*x"],
            nvars=2,
            lambda_str="1.0",
            mu_str="1.0",
            bkd=bkd,
            oned=True,
        )

        # Construct mesh nodes
        nodes_x = basis.nodes_x()
        nodes_y = basis.nodes_y()
        xx, yy = bkd.meshgrid(nodes_x, nodes_y, indexing="xy")
        nodes = bkd.stack([xx.flatten(), yy.flatten()], axis=0)

        # Get exact solution and forcing
        u_exact = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        # Flatten for physics
        npts = basis.npts()
        u_exact_flat = bkd.concatenate([u_exact[:, 0], u_exact[:, 1]])
        forcing_flat = bkd.concatenate([forcing[:, 0], forcing[:, 1]])

        # Create physics
        physics = LinearElasticityPhysics(
            basis, bkd, lamda=1.0, mu=1.0, forcing=lambda t: forcing_flat
        )

        # Set BCs on all boundaries
        bcs = []
        for side in range(4):
            boundary_idx = mesh.boundary_indices(side)
            bc_u = zero_dirichlet_bc(bkd, boundary_idx)
            bcs.append(bc_u)
            boundary_idx_v = bkd.asarray([idx + npts for idx in boundary_idx])
            bc_v = zero_dirichlet_bc(bkd, boundary_idx_v)
            bcs.append(bc_v)
        physics.set_boundary_conditions(bcs)

        # Compute residual
        residual = physics.residual(u_exact_flat, 0.0)
        jacobian = physics.jacobian(u_exact_flat, 0.0)
        residual_with_bc, _ = physics.apply_boundary_conditions(
            residual, jacobian, u_exact_flat, 0.0
        )

        # Get interior indices
        boundary_indices = set()
        for side in range(4):
            for idx in mesh.boundary_indices(side):
                boundary_indices.add(idx)
                boundary_indices.add(idx + npts)

        interior_indices = [i for i in range(2 * npts) if i not in boundary_indices]
        interior_residual = bkd.asarray([residual_with_bc[i] for i in interior_indices])

        bkd.assert_allclose(
            interior_residual, bkd.zeros(interior_residual.shape), atol=1e-10
        )

    def test_jacobian_derivative_checker(self):
        """Test Jacobian correctness via DerivativeChecker."""
        bkd = self.bkd()
        npts_x, npts_y = 6, 6
        mesh = TransformedMesh2D(npts_x, npts_y, bkd)

        basis = ChebyshevBasis2D(mesh, bkd)

        physics = LinearElasticityPhysics(basis, bkd, lamda=1.0, mu=1.0)

        wrapper = PhysicsDerivativeWrapper(physics)
        nstates = physics.nstates()
        sample = bkd.asarray([[0.1 * float(i) for i in range(nstates)]]).T

        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(sample, verbosity=0)
        self.assertLessEqual(checker.error_ratio(errors[0]), 1e-5)

    def test_different_lame_parameters(self):
        """Test with different Lamé parameters."""
        bkd = self.bkd()
        npts_x, npts_y = 10, 10
        mesh = TransformedMesh2D(npts_x, npts_y, bkd)

        basis = ChebyshevBasis2D(mesh, bkd)
        mesh = create_uniform_mesh_2d((npts_x, npts_y), (-1.0, 1.0, -1.0, 1.0), bkd)

        # Different Lamé parameters
        lamda, mu = 2.5, 0.5

        man_sol = ManufacturedLinearElasticityEquations(
            sol_strs=["(1 - x**2)*(1 - y**2)", "(1 - x**2)*(1 - y**2)*x"],
            nvars=2,
            lambda_str=str(lamda),
            mu_str=str(mu),
            bkd=bkd,
            oned=True,
        )

        nodes_x = basis.nodes_x()
        nodes_y = basis.nodes_y()
        xx, yy = bkd.meshgrid(nodes_x, nodes_y, indexing="xy")
        nodes = bkd.stack([xx.flatten(), yy.flatten()], axis=0)

        u_exact = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        npts = basis.npts()
        u_exact_flat = bkd.concatenate([u_exact[:, 0], u_exact[:, 1]])
        forcing_flat = bkd.concatenate([forcing[:, 0], forcing[:, 1]])

        physics = LinearElasticityPhysics(
            basis, bkd, lamda=lamda, mu=mu, forcing=lambda t: forcing_flat
        )

        bcs = []
        for side in range(4):
            boundary_idx = mesh.boundary_indices(side)
            bc_u = zero_dirichlet_bc(bkd, boundary_idx)
            bcs.append(bc_u)
            boundary_idx_v = bkd.asarray([idx + npts for idx in boundary_idx])
            bc_v = zero_dirichlet_bc(bkd, boundary_idx_v)
            bcs.append(bc_v)
        physics.set_boundary_conditions(bcs)

        residual = physics.residual(u_exact_flat, 0.0)
        jacobian = physics.jacobian(u_exact_flat, 0.0)
        residual_with_bc, _ = physics.apply_boundary_conditions(
            residual, jacobian, u_exact_flat, 0.0
        )

        boundary_indices = set()
        for side in range(4):
            for idx in mesh.boundary_indices(side):
                boundary_indices.add(idx)
                boundary_indices.add(idx + npts)

        interior_indices = [i for i in range(2 * npts) if i not in boundary_indices]
        interior_residual = bkd.asarray([residual_with_bc[i] for i in interior_indices])

        bkd.assert_allclose(
            interior_residual, bkd.zeros(interior_residual.shape), atol=1e-10
        )

    def test_numerical_solve(self):
        """Test numerical solution matches manufactured solution."""
        bkd = self.bkd()
        npts_x, npts_y = 10, 10
        mesh = TransformedMesh2D(npts_x, npts_y, bkd)

        basis = ChebyshevBasis2D(mesh, bkd)
        mesh = create_uniform_mesh_2d((npts_x, npts_y), (-1.0, 1.0, -1.0, 1.0), bkd)

        man_sol = ManufacturedLinearElasticityEquations(
            sol_strs=["(1 - x**2)*(1 - y**2)", "(1 - x**2)*(1 - y**2)*x"],
            nvars=2,
            lambda_str="1.0",
            mu_str="1.0",
            bkd=bkd,
            oned=True,
        )

        nodes_x = basis.nodes_x()
        nodes_y = basis.nodes_y()
        xx, yy = bkd.meshgrid(nodes_x, nodes_y, indexing="xy")
        nodes = bkd.stack([xx.flatten(), yy.flatten()], axis=0)

        u_exact = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        npts = basis.npts()
        u_exact_flat = bkd.concatenate([u_exact[:, 0], u_exact[:, 1]])
        forcing_flat = bkd.concatenate([forcing[:, 0], forcing[:, 1]])

        physics = LinearElasticityPhysics(
            basis, bkd, lamda=1.0, mu=1.0, forcing=lambda t: forcing_flat
        )

        bcs = []
        for side in range(4):
            boundary_idx = mesh.boundary_indices(side)
            bc_u = zero_dirichlet_bc(bkd, boundary_idx)
            bcs.append(bc_u)
            boundary_idx_v = bkd.asarray([idx + npts for idx in boundary_idx])
            bc_v = zero_dirichlet_bc(bkd, boundary_idx_v)
            bcs.append(bc_v)
        physics.set_boundary_conditions(bcs)

        model = CollocationModel(physics, bkd)
        initial_guess = bkd.zeros((2 * npts,))
        u_numerical = model.solve_steady(initial_guess, tol=1e-10, maxiter=50)

        bkd.assert_allclose(u_numerical, u_exact_flat, atol=1e-8)


class TestLinearElasticity2DParameterized(ParametrizedTestCase):
    """Parameterized 2D Linear Elasticity residual tests."""

    def bkd(self):
        return NumpyBkd()

    @parametrize(
        "name,u_str,v_str,lambda_str,mu_str,npts_1d",
        [
            # Polynomial solutions (exact for Chebyshev)
            (
                "poly_basic",
                "(1 - x**2)*(1 - y**2)",
                "(1 - x**2)*(1 - y**2)*x",
                "1.0",
                "1.0",
                10,
            ),
            (
                "poly_symmetric",
                "(1 - x**2)*(1 - y**2)*x*y",
                "(1 - x**2)*(1 - y**2)*x*y",
                "1.0",
                "1.0",
                12,
            ),
            (
                "poly_higher_degree",
                "(1 - x**2)*(1 - y**2)*x**2",
                "(1 - x**2)*(1 - y**2)*y**2",
                "1.0",
                "1.0",
                14,
            ),
            # Different Lamé parameters
            (
                "high_lambda",
                "(1 - x**2)*(1 - y**2)",
                "(1 - x**2)*(1 - y**2)*x",
                "10.0",
                "1.0",
                10,
            ),
            (
                "high_mu",
                "(1 - x**2)*(1 - y**2)",
                "(1 - x**2)*(1 - y**2)*x",
                "1.0",
                "10.0",
                10,
            ),
            (
                "incompressible_like",
                "(1 - x**2)*(1 - y**2)",
                "(1 - x**2)*(1 - y**2)*x",
                "100.0",
                "1.0",
                10,
            ),
        ],
    )
    def test_linear_elasticity_2d_residual(
        self, name, u_str, v_str, lambda_str, mu_str, npts_1d
    ):
        """Test 2D Linear Elasticity residual for parameterized cases."""
        bkd = self.bkd()
        npts_x, npts_y = npts_1d, npts_1d
        mesh = TransformedMesh2D(npts_x, npts_y, bkd)

        basis = ChebyshevBasis2D(mesh, bkd)
        mesh = create_uniform_mesh_2d((npts_x, npts_y), (-1.0, 1.0, -1.0, 1.0), bkd)

        man_sol = ManufacturedLinearElasticityEquations(
            sol_strs=[u_str, v_str],
            nvars=2,
            lambda_str=lambda_str,
            mu_str=mu_str,
            bkd=bkd,
            oned=True,
        )

        nodes_x = basis.nodes_x()
        nodes_y = basis.nodes_y()
        xx, yy = bkd.meshgrid(nodes_x, nodes_y, indexing="xy")
        nodes = bkd.stack([xx.flatten(), yy.flatten()], axis=0)

        u_exact = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        npts = basis.npts()
        u_exact_flat = bkd.concatenate([u_exact[:, 0], u_exact[:, 1]])
        forcing_flat = bkd.concatenate([forcing[:, 0], forcing[:, 1]])

        lamda = float(lambda_str)
        mu = float(mu_str)

        physics = LinearElasticityPhysics(
            basis, bkd, lamda=lamda, mu=mu, forcing=lambda t: forcing_flat
        )

        bcs = []
        for side in range(4):
            boundary_idx = mesh.boundary_indices(side)
            bc_u = zero_dirichlet_bc(bkd, boundary_idx)
            bcs.append(bc_u)
            boundary_idx_v = bkd.asarray([idx + npts for idx in boundary_idx])
            bc_v = zero_dirichlet_bc(bkd, boundary_idx_v)
            bcs.append(bc_v)
        physics.set_boundary_conditions(bcs)

        residual = physics.residual(u_exact_flat, 0.0)
        jacobian = physics.jacobian(u_exact_flat, 0.0)
        residual_with_bc, _ = physics.apply_boundary_conditions(
            residual, jacobian, u_exact_flat, 0.0
        )

        boundary_indices = set()
        for side in range(4):
            for idx in mesh.boundary_indices(side):
                boundary_indices.add(idx)
                boundary_indices.add(idx + npts)

        interior_indices = [i for i in range(2 * npts) if i not in boundary_indices]
        interior_residual = bkd.asarray([residual_with_bc[i] for i in interior_indices])

        bkd.assert_allclose(
            interior_residual, bkd.zeros(interior_residual.shape), atol=1e-9
        )

    @parametrize(
        "name,u_str,v_str,lambda_str,mu_str,npts_1d",
        [
            (
                "jacobian_basic",
                "(1 - x**2)*(1 - y**2)",
                "(1 - x**2)*(1 - y**2)*x",
                "1.0",
                "1.0",
                6,
            ),
            (
                "jacobian_high_lambda",
                "(1 - x**2)*(1 - y**2)",
                "(1 - x**2)*(1 - y**2)*x",
                "2.5",
                "0.5",
                6,
            ),
            (
                "jacobian_high_mu",
                "(1 - x**2)*(1 - y**2)",
                "(1 - x**2)*(1 - y**2)*x",
                "0.5",
                "2.5",
                6,
            ),
        ],
    )
    def test_linear_elasticity_2d_jacobian(
        self, name, u_str, v_str, lambda_str, mu_str, npts_1d
    ):
        """Test 2D Linear Elasticity Jacobian via DerivativeChecker."""
        bkd = self.bkd()
        npts_x, npts_y = npts_1d, npts_1d
        mesh = TransformedMesh2D(npts_x, npts_y, bkd)

        basis = ChebyshevBasis2D(mesh, bkd)

        lamda = float(lambda_str)
        mu = float(mu_str)

        physics = LinearElasticityPhysics(basis, bkd, lamda=lamda, mu=mu)

        wrapper = PhysicsDerivativeWrapper(physics)
        nstates = physics.nstates()
        sample = bkd.asarray([[0.1 * float(i) for i in range(nstates)]]).T

        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(sample, verbosity=0)
        self.assertLessEqual(checker.error_ratio(errors[0]), 1e-5)


# Concrete backend implementations
class TestManufacturedLinearElasticity2DNumpy(
    TestManufacturedLinearElasticity2D[NDArray[Any]]
):
    """NumPy backend tests for 2D Linear Elasticity."""

    __test__ = True

    def bkd(self):
        return NumpyBkd()


class TestManufacturedLinearElasticity2DTorch(
    TestManufacturedLinearElasticity2D[torch.Tensor]
):
    """PyTorch backend tests for 2D Linear Elasticity."""

    __test__ = True

    def bkd(self):
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = self.bkd()


if __name__ == "__main__":
    unittest.main()
