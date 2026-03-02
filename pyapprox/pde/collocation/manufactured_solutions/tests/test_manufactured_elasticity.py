"""Tests for linear elasticity manufactured solutions.

Verifies:
1. Residual = 0 at exact solution (using polynomial solutions for machine precision)
2. Jacobian correctness via DerivativeChecker
3. Numerical solution matches exact solution

Note: For residual tests, we use polynomial solutions that can be exactly
represented by the tensor product Chebyshev basis. This gives machine
precision residuals (< 1e-12).
"""

from typing import Generic

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
from pyapprox.util.backends.protocols import Array, Backend


class PhysicsDerivativeWrapper(Generic[Array]):
    """Wrapper to adapt physics interface for DerivativeChecker.

    DerivativeChecker expects:
    - bkd() -> Backend
    - nvars() -> int
    - nqoi() -> int
    - __call__(samples) -> Array
    - jacobian(sample) -> Array

    Physics provides:
    - residual(state, time)
    - jacobian(state, time)

    This wrapper fixes time=0 for derivative checking.
    """

    def __init__(self, physics, bkd: Backend[Array], time: float = 0.0):
        self._physics = physics
        self._bkd = bkd
        self._time = time

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._physics.nstates()

    def nqoi(self) -> int:
        return self._physics.nstates()

    def __call__(self, sample: Array) -> Array:
        """Evaluate residual at sample."""
        state = sample.flatten()
        result = self._physics.residual(state, self._time)
        return result.reshape(-1, 1)

    def jacobian(self, sample: Array) -> Array:
        """Evaluate Jacobian at sample."""
        state = sample.flatten()
        return self._physics.jacobian(state, self._time)


class TestLinearElasticity:
    """Test linear elasticity manufactured solutions.

    Uses polynomial solutions for machine precision:
    - u = (1-x**2)*(1-y**2)  (degree 2 in each direction)
    - v = (1-x**2)*(1-y**2)*x*y  (degree 3 in each direction)

    These satisfy u=v=0 on all boundaries.
    """

    def test_residual_polynomial(self, bkd):
        """Test residual = 0 at exact polynomial solution.

        Uses polynomial solution with degree <= 4 for machine precision.
        """
        npts_x, npts_y = 10, 10
        mesh = TransformedMesh2D(npts_x, npts_y, bkd)

        basis = ChebyshevBasis2D(mesh, bkd)
        mesh = create_uniform_mesh_2d((npts_x, npts_y), (-1.0, 1.0, -1.0, 1.0), bkd)

        # Polynomial manufactured solution
        # u = (1-x**2)*(1-y**2), v = (1-x**2)*(1-y**2)*x
        # Both zero on boundaries
        man_sol = ManufacturedLinearElasticityEquations(
            sol_strs=["(1 - x**2)*(1 - y**2)", "(1 - x**2)*(1 - y**2)*x"],
            nvars=2,
            lambda_str="1.0",
            mu_str="1.0",
            bkd=bkd,
            oned=True,
        )

        # Construct mesh nodes with 'xy' indexing
        nodes_x = basis.nodes_x()
        nodes_y = basis.nodes_y()
        xx, yy = bkd.meshgrid(nodes_x, nodes_y, indexing="xy")
        nodes = bkd.stack([xx.flatten(), yy.flatten()], axis=0)

        # Get exact solution and forcing
        u_exact = man_sol.functions["solution"](nodes)  # (npts, 2)
        forcing = man_sol.functions["forcing"](nodes)  # (npts, 2)

        # Flatten for physics (state is [u_all, v_all])
        npts = basis.npts()
        u_exact_flat = bkd.concatenate([u_exact[:, 0], u_exact[:, 1]])
        forcing_flat = bkd.concatenate([forcing[:, 0], forcing[:, 1]])

        # Create physics
        physics = LinearElasticityPhysics(
            basis, bkd, lamda=1.0, mu=1.0, forcing=lambda t: forcing_flat
        )

        # Set BCs on all 4 boundaries for both components
        bcs = []
        for side in range(4):
            boundary_idx = mesh.boundary_indices(side)
            # BC for u component
            bc_u = zero_dirichlet_bc(bkd, boundary_idx)
            bcs.append(bc_u)
            # BC for v component (offset by npts)
            boundary_idx_v = bkd.asarray([idx + npts for idx in boundary_idx])
            bc_v = zero_dirichlet_bc(bkd, boundary_idx_v)
            bcs.append(bc_v)
        physics.set_boundary_conditions(bcs)

        # Compute residual at exact solution
        residual = physics.residual(u_exact_flat, 0.0)

        # Apply BCs
        jacobian = physics.jacobian(u_exact_flat, 0.0)
        residual_with_bc, _ = physics.apply_boundary_conditions(
            residual, jacobian, u_exact_flat, 0.0
        )

        # Get interior indices (excluding all boundaries)
        boundary_indices = set()
        for side in range(4):
            for idx in mesh.boundary_indices(side):
                boundary_indices.add(idx)
                boundary_indices.add(idx + npts)  # v component

        interior_indices = [i for i in range(2 * npts) if i not in boundary_indices]
        interior_residual = bkd.asarray([residual_with_bc[i] for i in interior_indices])

        # Machine precision residual for polynomial solution
        bkd.assert_allclose(
            interior_residual, bkd.zeros(interior_residual.shape), atol=1e-10
        )

    def test_jacobian_derivative_checker(self, bkd):
        """Test Jacobian correctness via DerivativeChecker."""
        npts_x, npts_y = 6, 6
        mesh = TransformedMesh2D(npts_x, npts_y, bkd)

        basis = ChebyshevBasis2D(mesh, bkd)

        # Create physics without forcing
        physics = LinearElasticityPhysics(basis, bkd, lamda=1.0, mu=1.0)

        # Wrap for DerivativeChecker
        wrapper = PhysicsDerivativeWrapper(physics, bkd)

        # Create test state
        nstates = physics.nstates()
        sample = bkd.asarray([[0.1 * float(i) for i in range(nstates)]]).T

        # Check derivatives
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(sample, verbosity=0)

        # Should have small relative error at some epsilon
        min_error = float(bkd.min(errors[0]))
        assert min_error < 1e-5

    def test_jacobian_with_different_lame_params(self, bkd):
        """Test Jacobian with different Lamé parameters."""
        npts_x, npts_y = 6, 6
        mesh = TransformedMesh2D(npts_x, npts_y, bkd)

        basis = ChebyshevBasis2D(mesh, bkd)

        # Different Lamé parameters
        physics = LinearElasticityPhysics(basis, bkd, lamda=2.5, mu=0.5)

        wrapper = PhysicsDerivativeWrapper(physics, bkd)
        nstates = physics.nstates()
        sample = bkd.asarray([[0.05 * float(i) for i in range(nstates)]]).T

        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(sample, verbosity=0)

        min_error = float(bkd.min(errors[0]))
        assert min_error < 1e-5

    def test_numerical_solve(self, bkd):
        """Test numerical solution matches manufactured solution."""
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

        u_exact = man_sol.functions["solution"](nodes)
        forcing = man_sol.functions["forcing"](nodes)

        npts = basis.npts()
        u_exact_flat = bkd.concatenate([u_exact[:, 0], u_exact[:, 1]])
        forcing_flat = bkd.concatenate([forcing[:, 0], forcing[:, 1]])

        physics = LinearElasticityPhysics(
            basis, bkd, lamda=1.0, mu=1.0, forcing=lambda t: forcing_flat
        )

        # Set BCs
        bcs = []
        for side in range(4):
            boundary_idx = mesh.boundary_indices(side)
            bc_u = zero_dirichlet_bc(bkd, boundary_idx)
            bcs.append(bc_u)
            boundary_idx_v = bkd.asarray([idx + npts for idx in boundary_idx])
            bc_v = zero_dirichlet_bc(bkd, boundary_idx_v)
            bcs.append(bc_v)
        physics.set_boundary_conditions(bcs)

        # Solve
        model = CollocationModel(physics, bkd)
        initial_guess = bkd.zeros((2 * npts,))
        u_numerical = model.solve_steady(initial_guess, tol=1e-10, maxiter=50)

        # Should match exact solution
        bkd.assert_allclose(u_numerical, u_exact_flat, atol=1e-8)

    def test_ncomponents(self, bkd):
        """Test that physics reports correct number of components."""
        npts_x, npts_y = 5, 5
        mesh = TransformedMesh2D(npts_x, npts_y, bkd)

        basis = ChebyshevBasis2D(mesh, bkd)

        physics = LinearElasticityPhysics(basis, bkd, lamda=1.0, mu=1.0)

        assert physics.ncomponents() == 2
        assert physics.nstates() == 2 * basis.npts()


class TestLinearElasticityNumpy(TestLinearElasticity):
    """Numpy implementation of linear elasticity tests."""
