"""Tests for transient DtN solver with manufactured solutions.

Tests verify:
1. Time-dependent solutions converge to manufactured solutions
2. Flux conservation holds at each time step
3. Vector-valued transient problems work correctly

The transient DtN approach:
- At each time step, solve the DtN problem for interface values
- The physics objects handle the time-dependent residual
- The manufactured solutions provide exact reference values

NOTE: All tests use the reference domain [-1, 1] for each subdomain.
The domain decomposition connects:
- Subdomain 0: [-1, 1] with interface at x=1 (right)
- Subdomain 1: [-1, 1] with interface at x=-1 (left)
This is conceptually stitching two domains together at a shared interface.
"""

import unittest
import numpy as np
from typing import Callable, Tuple

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401
from pyapprox.typing.pde.collocation.basis import (
    ChebyshevBasis1D,
    ChebyshevBasis2D,
)
from pyapprox.typing.pde.collocation.mesh import (
    TransformedMesh1D,
    TransformedMesh2D,
    TransformedMesh3D,
)
from pyapprox.typing.pde.collocation.boundary import (
    DirichletBC,
    zero_dirichlet_bc,
)
from pyapprox.typing.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
    create_steady_diffusion,
)
from pyapprox.typing.pde.collocation.physics.reaction_diffusion import (
    TwoSpeciesReactionDiffusionPhysics,
    LinearReaction,
)
from pyapprox.typing.pde.decomposition.interface import (
    Interface1D,
    Interface,
    Interface2D,
    LegendreInterfaceBasis1D,
    LegendreInterfaceBasis2D,
)
from pyapprox.typing.pde.collocation.basis import ChebyshevBasis3D
from pyapprox.typing.pde.collocation.physics.linear_elasticity import (
    LinearElasticityPhysics,
    create_linear_elasticity,
)
from pyapprox.typing.pde.decomposition.subdomain import SubdomainWrapper
from pyapprox.typing.pde.decomposition.solver import (
    DtNResidual,
    DtNSolver,
)


class ManufacturedSolution1DWithForcingSymmetric:
    """Manufactured solution for two domains stitched at interface.

    Domain setup:
    - Subdomain 0: [-1, 1] with external BC at x=-1, interface at x=1
    - Subdomain 1: [-1, 1] with interface at x=-1, external BC at x=1

    Both subdomains solve: -D * u'' = f with manufactured solution.

    We use a symmetric parabolic solution that works on each subdomain:
    u(x, t) = (1 + t) * (1 - x²)

    On subdomain 0:
    - u(-1) = 0 (external BC)
    - u(1) = 0 (interface value - but wait, this doesn't match!)

    Let's use a different manufactured solution that has a non-zero interface value.
    u(x, t) = (1 + t) * (1 + x) / 2  (linear, satisfies Laplace -u'' = 0)
    - u(-1) = 0, u(1) = (1 + t), u(0) = (1 + t)/2

    For time-dependent with forcing, let's use:
    u(x, t) = A(t) + B(t)*x - C(t)*x²
    with -u'' = 2*C(t)

    Specifically:
    Subdomain 0: u = (1+t) * (1 + x) satisfies -u'' = 0
    u(-1) = 0, u(1) = 2(1+t), forcing = 0

    Subdomain 1: u = (1+t) * (2 - x - 1) = (1+t)*(1-x), also -u'' = 0
    u(-1) = 2(1+t), u(1) = 0

    Interface value: lambda = 2(1+t)
    Flux conservation:
    - Sub 0: u'(1) = (1+t), normal = +1, flux = (1+t)
    - Sub 1: u'(-1) = -(1+t), normal = -1, flux = (1+t)
    Total = 2(1+t) ≠ 0 unless we adjust...

    Actually let's use the standard approach from test_dtn_solver.py:
    Subdomain 0: u = lambda/2 * (x + 1), u(-1)=0, u(1)=lambda
    Subdomain 1: u = (lambda+1)/2 + (1-lambda)/2 * x, u(-1)=lambda, u(1)=1

    For time-dependent, let lambda = lambda(t).

    Simpler: use Laplace with time-dependent BCs.
    """
    pass


class ManufacturedSolution1DTransientLaplace:
    """Time-dependent Laplace equation with linear solution.

    On each subdomain [-1, 1]:
    -u'' = 0 (Laplace equation)

    Subdomain 0: u(-1) = 0, u(1) = lambda(t)
    Solution: u = lambda(t)/2 * (x + 1)
    u'(1) = lambda(t)/2

    Subdomain 1: u(-1) = lambda(t), u(1) = g(t) (external BC)
    Solution: u = (lambda + g)/2 + (g - lambda)/2 * x
    u'(-1) = (g - lambda)/2

    Flux conservation (with D=1):
    flux_0 = u'(1) * (+1) = lambda/2
    flux_1 = u'(-1) * (-1) = -(g - lambda)/2 = (lambda - g)/2
    Total = lambda/2 + (lambda - g)/2 = lambda - g/2 = 0
    => lambda = g/2

    So if we set external BC g(t) = 2*(1 + t), then lambda(t) = 1 + t.
    """

    def __init__(self, bkd: Backend, D: float = 1.0):
        self.bkd = bkd
        self.D = D

    def g(self, t: float) -> float:
        """External BC at x=1 of subdomain 1."""
        return 2.0 * (1.0 + t)

    def interface_value(self, t: float) -> float:
        """Value at interface (x=1 of sub0, x=-1 of sub1)."""
        return 1.0 + t  # lambda = g/2

    def solution_sub0(self, x: Array, t: float) -> Array:
        """Solution on subdomain 0: u = lambda(t)/2 * (x + 1)."""
        lam = self.interface_value(t)
        return lam / 2.0 * (x + 1.0)

    def solution_sub1(self, x: Array, t: float) -> Array:
        """Solution on subdomain 1: u = (lambda + g)/2 + (g - lambda)/2 * x."""
        lam = self.interface_value(t)
        g = self.g(t)
        return (lam + g) / 2.0 + (g - lam) / 2.0 * x


class ManufacturedSolution1DTransientForced:
    """Manufactured solution for 1D diffusion with forcing (steady-state approach).

    This treats the time parameter as just affecting the forcing/BCs,
    not actual time evolution. Each time step is an independent steady solve.

    Domain decomposition with two subdomains on [-1, 1] each:
    - Subdomain 0: u(-1) = 0 (external), u(1) = lambda (interface)
    - Subdomain 1: u(-1) = lambda (interface), u(1) = 0 (external)

    Manufactured solution on full conceptual domain:
    u(x, t) = (1 + t) * (1 - x²)

    For subdomain 0 (conceptually [-1, 0] mapped to [-1, 1]):
    We need to express u on the reference domain [-1, 1] that maps to [-1, 0].
    x_phys = (x_ref - 1)/2, so x_ref = 2*x_phys + 1

    At interface (x_phys=0): x_ref = 1, so interface is at right boundary.
    u(x_phys=0, t) = (1+t)*(1-0) = 1+t = interface value

    But the physics operates on reference domain, so we need to
    evaluate the solution at physical coordinates corresponding to ref nodes.

    Actually, the simpler approach is to use the symmetric setup from
    test_dtn_solver.py where each subdomain is on [-1, 1] and they share
    an interface conceptually. Let's use:

    -u'' = f on each subdomain
    with symmetric forcing that gives parabolic solution.

    Subdomain 0: u(-1) = 0, u(1) = lambda
    Subdomain 1: u(-1) = lambda, u(1) = 0

    With f = 2*(1+t), solution u = A + Bx - (1+t)*x² satisfying -u'' = 2(1+t).

    Sub 0: u(-1) = A - B - (1+t) = 0, u(1) = A + B - (1+t) = lambda
    => A - B = 1+t, A + B = lambda + 1+t
    => 2A = lambda + 2(1+t), A = (lambda + 2(1+t))/2
    => 2B = lambda, B = lambda/2
    u = (lambda/2 + 1+t) + (lambda/2)*x - (1+t)*x²
    u'(1) = lambda/2 - 2(1+t)

    Sub 1: u(-1) = C - D - (1+t) = lambda, u(1) = C + D - (1+t) = 0
    => C - D = lambda + 1+t, C + D = 1+t
    => 2C = lambda + 2(1+t), C = (lambda + 2(1+t))/2
    => 2D = (1+t) - (lambda + 1+t) = -lambda, D = -lambda/2
    u = (lambda/2 + 1+t) - (lambda/2)*x - (1+t)*x²
    u'(-1) = -lambda/2 + 2(1+t)

    Flux conservation:
    flux_0 = u'(1) = lambda/2 - 2(1+t)
    flux_1 = -u'(-1) = -(-lambda/2 + 2(1+t)) = lambda/2 - 2(1+t)
    Total = 2*(lambda/2 - 2(1+t)) = lambda - 4(1+t) = 0
    => lambda = 4(1+t)

    So interface value = 4*(1+t).
    """

    def __init__(self, bkd: Backend, D: float = 1.0):
        self.bkd = bkd
        self.D = D

    def forcing(self, x: Array, t: float) -> Array:
        """Forcing term f = 2*(1+t)."""
        return np.full_like(x, 2.0 * (1.0 + t))

    def interface_value(self, t: float) -> float:
        """Value at interface."""
        return 4.0 * (1.0 + t)

    def solution_sub0(self, x: Array, t: float) -> Array:
        """Solution on subdomain 0."""
        lam = self.interface_value(t)
        one_plus_t = 1.0 + t
        # u = (lambda/2 + 1+t) + (lambda/2)*x - (1+t)*x²
        return (lam / 2.0 + one_plus_t) + (lam / 2.0) * x - one_plus_t * x**2

    def solution_sub1(self, x: Array, t: float) -> Array:
        """Solution on subdomain 1."""
        lam = self.interface_value(t)
        one_plus_t = 1.0 + t
        # u = (lambda/2 + 1+t) - (lambda/2)*x - (1+t)*x²
        return (lam / 2.0 + one_plus_t) - (lam / 2.0) * x - one_plus_t * x**2

    def left_bc_sub0(self, t: float) -> float:
        """BC at x=-1 of subdomain 0."""
        return 0.0

    def right_bc_sub1(self, t: float) -> float:
        """BC at x=1 of subdomain 1."""
        return 0.0


class ManufacturedSolutionTwoSpecies1D:
    """Manufactured solution for 1D two-species diffusion (uncoupled).

    Each species satisfies -D_i * u_i'' = f_i on each subdomain.

    Using symmetric setup:
    Species 0: u0(-1) = 0, u0(1) = lambda0 (sub 0); u0(-1) = lambda0, u0(1) = 0 (sub 1)
    Species 1: u1(-1) = 0, u1(1) = lambda1 (sub 0); u1(-1) = lambda1, u1(1) = 0 (sub 1)

    With f0 = 2*D0*(1+t), f1 = 2*D1*(1+t):
    lambda0 = 4*D0*(1+t)/D0 = 4*(1+t)  (if we use diffusion=D0 in residual)
    Actually, for -D*u'' = f => -u'' = f/D, we need f such that -u'' = 2*(1+t)
    So f = 2*D*(1+t).

    For species 0 with D0: -D0*u0'' = f0 => f0 = 2*D0*(1+t), interface = 4*(1+t)
    For species 1 with D1: -D1*u1'' = f1 => f1 = 2*D1*(1+t), interface = 4*(1+t)

    Both have same interface value but different forcing scaled by diffusion.
    """

    def __init__(self, bkd: Backend, D0: float = 1.0, D1: float = 0.5):
        self.bkd = bkd
        self.D0 = D0
        self.D1 = D1

    def interface_values(self, t: float) -> Tuple[float, float]:
        """Values at interface for [u0, u1]."""
        return 4.0 * (1.0 + t), 4.0 * (1.0 + t)

    def forcing0(self, x: Array, t: float) -> Array:
        """Forcing for species 0."""
        return np.full_like(x, 2.0 * self.D0 * (1.0 + t))

    def forcing1(self, x: Array, t: float) -> Array:
        """Forcing for species 1."""
        return np.full_like(x, 2.0 * self.D1 * (1.0 + t))

    def solution0_sub0(self, x: Array, t: float) -> Array:
        """Species 0 solution on subdomain 0."""
        lam = 4.0 * (1.0 + t)
        one_plus_t = 1.0 + t
        return (lam / 2.0 + one_plus_t) + (lam / 2.0) * x - one_plus_t * x**2

    def solution0_sub1(self, x: Array, t: float) -> Array:
        """Species 0 solution on subdomain 1."""
        lam = 4.0 * (1.0 + t)
        one_plus_t = 1.0 + t
        return (lam / 2.0 + one_plus_t) - (lam / 2.0) * x - one_plus_t * x**2

    def solution1_sub0(self, x: Array, t: float) -> Array:
        """Species 1 solution on subdomain 0."""
        lam = 4.0 * (1.0 + t)
        one_plus_t = 1.0 + t
        return (lam / 2.0 + one_plus_t) + (lam / 2.0) * x - one_plus_t * x**2

    def solution1_sub1(self, x: Array, t: float) -> Array:
        """Species 1 solution on subdomain 1."""
        lam = 4.0 * (1.0 + t)
        one_plus_t = 1.0 + t
        return (lam / 2.0 + one_plus_t) - (lam / 2.0) * x - one_plus_t * x**2


class TestTransientDtN1DSimple(unittest.TestCase):
    """Test transient DtN for 1D diffusion with manufactured solution.

    Uses symmetric domain decomposition where each subdomain is on [-1, 1].
    """

    def setUp(self):
        self.bkd = NumpyBkd()
        self.npts = 16
        self.D = 1.0

    def _create_transient_problem(self, time: float = 0.0):
        """Create transient 1D diffusion problem on reference domains.

        Domain decomposition:
        - Subdomain 0: [-1, 1] with u(-1) = 0, u(1) = lambda (interface)
        - Subdomain 1: [-1, 1] with u(-1) = lambda (interface), u(1) = 0
        """
        bkd = self.bkd
        npts = self.npts
        D = self.D

        mms = ManufacturedSolution1DTransientForced(bkd, D)

        interface = Interface1D(
            bkd, interface_id=0, subdomain_ids=(0, 1), interface_point=0.0
        )

        # Subdomain 0: [-1, 1] with external BC at x=-1, interface at x=1
        mesh0 = TransformedMesh1D(npts, bkd)
        basis0 = ChebyshevBasis1D(mesh0, bkd)
        nodes0 = basis0.nodes()

        def forcing0(t):
            return mms.forcing(nodes0, t)

        physics0 = AdvectionDiffusionReaction(
            basis0, bkd, diffusion=D, forcing=forcing0
        )

        # External BC at x=-1 (index 0): u = 0
        left_bc0 = zero_dirichlet_bc(bkd, bkd.asarray([0]))

        wrapper0 = SubdomainWrapper(
            bkd, subdomain_id=0, physics=physics0,
            interfaces={0: interface}, external_bcs=[left_bc0]
        )
        # Interface at x=1 (index npts-1)
        wrapper0.set_interface_boundary_indices(0, bkd.asarray([npts - 1]))

        # Subdomain 1: [-1, 1] with interface at x=-1, external BC at x=1
        mesh1 = TransformedMesh1D(npts, bkd)
        basis1 = ChebyshevBasis1D(mesh1, bkd)
        nodes1 = basis1.nodes()

        def forcing1(t):
            return mms.forcing(nodes1, t)

        physics1 = AdvectionDiffusionReaction(
            basis1, bkd, diffusion=D, forcing=forcing1
        )

        # External BC at x=1 (index npts-1): u = 0
        right_bc1 = zero_dirichlet_bc(bkd, bkd.asarray([npts - 1]))

        wrapper1 = SubdomainWrapper(
            bkd, subdomain_id=1, physics=physics1,
            interfaces={0: interface}, external_bcs=[right_bc1]
        )
        # Interface at x=-1 (index 0)
        wrapper1.set_interface_boundary_indices(0, bkd.asarray([0]))

        # Setup interface interpolation
        interface.set_subdomain_boundary_points(0, bkd.asarray([1.0]))
        interface.set_subdomain_boundary_points(1, bkd.asarray([-1.0]))

        interface_dof_offsets = bkd.asarray([0, 1])

        return {
            "subdomain_solvers": {0: wrapper0, 1: wrapper1},
            "interfaces": {0: interface},
            "interface_dof_offsets": interface_dof_offsets,
            "mms": mms,
            "nodes": {0: nodes0, 1: nodes1},
        }

    def test_steady_state_at_t0(self):
        """Test DtN solves correctly at t=0."""
        bkd = self.bkd
        problem = self._create_transient_problem(time=0.0)

        residual = DtNResidual(
            bkd,
            interfaces=problem["interfaces"],
            subdomain_solvers=problem["subdomain_solvers"],
            interface_dof_offsets=problem["interface_dof_offsets"],
        )

        solver = DtNSolver(bkd, residual, max_iters=20, tol=1e-10)

        # Initial guess
        initial_guess = bkd.asarray([2.0])
        result = solver.solve(initial_guess)

        self.assertTrue(result.converged, "Solver should converge at t=0")

        mms = problem["mms"]
        exact_interface = mms.interface_value(0.0)  # lambda = 4*(1+0) = 4
        computed_interface = float(result.interface_dofs[0])

        self.assertAlmostEqual(computed_interface, exact_interface, places=6,
                               msg=f"Interface: computed={computed_interface}, "
                                   f"exact={exact_interface}")

    def test_residual_zero_at_exact_solution(self):
        """Test residual is zero at the exact interface solution."""
        bkd = self.bkd
        problem = self._create_transient_problem(time=0.0)

        residual = DtNResidual(
            bkd,
            interfaces=problem["interfaces"],
            subdomain_solvers=problem["subdomain_solvers"],
            interface_dof_offsets=problem["interface_dof_offsets"],
        )

        mms = problem["mms"]
        exact_interface = mms.interface_value(0.0)  # lambda = 4
        exact_dofs = bkd.asarray([exact_interface])

        res = residual(exact_dofs)
        res_norm = float(bkd.norm(res))

        self.assertLess(res_norm, 1e-10,
                        f"Residual norm {res_norm} should be < 1e-10 at exact solution")

    def test_solution_accuracy_at_collocation_points(self):
        """Verify full solution matches manufactured solution at t=0."""
        bkd = self.bkd
        t = 0.0  # Use t=0 since solver uses time=0 internally
        problem = self._create_transient_problem(time=t)

        residual = DtNResidual(
            bkd,
            interfaces=problem["interfaces"],
            subdomain_solvers=problem["subdomain_solvers"],
            interface_dof_offsets=problem["interface_dof_offsets"],
        )

        solver = DtNSolver(bkd, residual, max_iters=20, tol=1e-10)
        result = solver.solve(bkd.asarray([2.0]))

        self.assertTrue(result.converged)

        mms = problem["mms"]

        # Check solution on each subdomain
        for sub_id in [0, 1]:
            wrapper = problem["subdomain_solvers"][sub_id]
            computed_sol = wrapper.solution()
            nodes = problem["nodes"][sub_id]
            if sub_id == 0:
                exact_sol = mms.solution_sub0(nodes, t)
            else:
                exact_sol = mms.solution_sub1(nodes, t)

            max_error = float(bkd.max(bkd.abs(computed_sol - exact_sol)))
            self.assertLess(max_error, 1e-8,
                            f"Subdomain {sub_id}: max error = {max_error}")


class TestDtN2DWithForcing(unittest.TestCase):
    """Test 2D DtN with forcing on reference domains.

    Uses ASYMMETRIC boundary conditions to ensure non-zero flux at interface:
    - Each subdomain is on the full reference domain [-1, 1]²
    - Interface connects right edge of subdomain 0 to left edge of subdomain 1
    - Forcing is constant (no y-dependence), so solution is also y-independent

    Manufactured solution with non-zero interface flux:
    - Sub 0: u(-1) = 0, u(1) = lambda, -u'' = 2
    - Sub 1: u(-1) = lambda, u(1) = 2, -u'' = 2  (asymmetric external BC)
    - lambda = 5 for flux conservation
    - u'(1) on sub 0 = 0.5 (non-zero flux!)
    """

    def setUp(self):
        self.bkd = NumpyBkd()
        self.npts = 6  # points in each direction
        self.D = 1.0
        self.exact_lambda = 5.0  # Exact interface value

    def _compute_boundary_indices_2d(self, npts_x, npts_y):
        """Compute 2D boundary indices (x varies fastest in tensor product ordering)."""
        bkd = self.bkd
        # Left: x-index=0 (x_ref=-1), all y
        left = bkd.asarray([j * npts_x for j in range(npts_y)])
        # Right: x-index=npts_x-1 (x_ref=+1), all y
        right = bkd.asarray([j * npts_x + (npts_x - 1) for j in range(npts_y)])
        # Bottom: y-index=0 (y_ref=-1), all x
        bottom = bkd.asarray(list(range(npts_x)))
        # Top: y-index=npts_y-1 (y_ref=+1), all x
        top = bkd.asarray([(npts_y - 1) * npts_x + i for i in range(npts_x)])
        return {"left": left, "right": right, "bottom": bottom, "top": top}

    def _exact_sub0(self, x):
        """Exact solution on subdomain 0.

        u = (lambda/2 + 1) + (lambda/2)*x - x^2 = 3.5 + 2.5x - x^2
        """
        lam = self.exact_lambda
        return (lam / 2.0 + 1.0) + (lam / 2.0) * x - x**2

    def _exact_sub1(self, x):
        """Exact solution on subdomain 1.

        u = (lambda + 4)/2 + (2 - lambda)/2 * x - x^2 = 4.5 - 1.5x - x^2
        """
        lam = self.exact_lambda
        return (lam + 4.0) / 2.0 + (2.0 - lam) / 2.0 * x - x**2

    def test_2d_with_constant_forcing(self):
        """Test 2D DtN with constant forcing and asymmetric BCs.

        This uses asymmetric external BCs to ensure non-zero flux at interface:
        - Sub 0: u(-1) = 0 (external)
        - Sub 1: u(1) = 2 (external, NOT zero!)

        Interface value lambda = 5 satisfies flux conservation with non-zero flux.
        """
        bkd = self.bkd
        npts = self.npts
        D = self.D
        forcing_val = 2.0 * D

        # Interface basis with degree=npts-1 to match boundary points
        interface_basis = LegendreInterfaceBasis1D(
            bkd, degree=npts - 1, physical_bounds=(-1.0, 1.0)
        )
        interface = Interface(
            bkd, interface_id=0, subdomain_ids=(0, 1),
            basis=interface_basis, normal_direction=0, ambient_dim=2
        )

        # Subdomain 0
        mesh0 = TransformedMesh2D(npts, npts, bkd)
        basis0 = ChebyshevBasis2D(mesh0, bkd)
        nodes_x0 = basis0.nodes_x()
        nodes_y0 = basis0.nodes_y()
        npts_total = basis0.npts()

        forcing0 = bkd.full((npts_total,), forcing_val)
        physics0 = AdvectionDiffusionReaction(
            basis0, bkd, diffusion=D, forcing=lambda t: forcing0
        )

        bounds0 = self._compute_boundary_indices_2d(npts, npts)

        # External BC at x=-1: u = 0
        left_bc0 = zero_dirichlet_bc(bkd, bounds0["left"])

        # Y-boundary BCs using exact solution
        bottom_vals0 = bkd.asarray([self._exact_sub0(nodes_x0[i]) for i in range(npts)])
        bottom_bc0 = DirichletBC(bkd, bounds0["bottom"], bottom_vals0)
        top_vals0 = bkd.asarray([self._exact_sub0(nodes_x0[i]) for i in range(npts)])
        top_bc0 = DirichletBC(bkd, bounds0["top"], top_vals0)

        wrapper0 = SubdomainWrapper(
            bkd, subdomain_id=0, physics=physics0,
            interfaces={0: interface},
            external_bcs=[left_bc0, bottom_bc0, top_bc0]
        )
        wrapper0.set_interface_boundary_indices(0, bounds0["right"])

        # Subdomain 1
        mesh1 = TransformedMesh2D(npts, npts, bkd)
        basis1 = ChebyshevBasis2D(mesh1, bkd)
        forcing1 = bkd.full((npts_total,), forcing_val)
        physics1 = AdvectionDiffusionReaction(
            basis1, bkd, diffusion=D, forcing=lambda t: forcing1
        )

        bounds1 = self._compute_boundary_indices_2d(npts, npts)

        # External BC at x=1: u = 2 (asymmetric!)
        right_bc1_vals = bkd.full((npts,), 2.0)
        right_bc1 = DirichletBC(bkd, bounds1["right"], right_bc1_vals)

        # Y-boundary BCs using exact solution
        bottom_vals1 = bkd.asarray([self._exact_sub1(nodes_x0[i]) for i in range(npts)])
        bottom_bc1 = DirichletBC(bkd, bounds1["bottom"], bottom_vals1)
        top_vals1 = bkd.asarray([self._exact_sub1(nodes_x0[i]) for i in range(npts)])
        top_bc1 = DirichletBC(bkd, bounds1["top"], top_vals1)

        wrapper1 = SubdomainWrapper(
            bkd, subdomain_id=1, physics=physics1,
            interfaces={0: interface},
            external_bcs=[right_bc1, bottom_bc1, top_bc1]
        )
        wrapper1.set_interface_boundary_indices(0, bounds1["left"])

        # Setup interface interpolation
        interface.set_subdomain_boundary_points(0, nodes_y0)
        interface.set_subdomain_boundary_points(1, nodes_y0)

        interface_dof_offsets = bkd.asarray([0, interface.ndofs()])

        residual = DtNResidual(
            bkd,
            interfaces={0: interface},
            subdomain_solvers={0: wrapper0, 1: wrapper1},
            interface_dof_offsets=interface_dof_offsets,
        )

        solver = DtNSolver(bkd, residual, max_iters=30, tol=1e-10)

        ndofs = interface.ndofs()
        result = solver.solve(bkd.full((ndofs,), 3.0))

        self.assertTrue(result.converged, "2D solver should converge")

        # Interface value should be constant = 5
        computed_vals = interface.evaluate(result.interface_dofs)
        exact_val = self.exact_lambda

        max_error = float(bkd.max(bkd.abs(computed_vals - exact_val)))
        self.assertLess(max_error, 1e-6,
                        f"2D interface error = {max_error}")

    def test_2d_residual_zero_at_exact(self):
        """Test residual is zero at exact interface solution."""
        bkd = self.bkd
        npts = self.npts
        D = self.D
        forcing_val = 2.0 * D

        interface_basis = LegendreInterfaceBasis1D(
            bkd, degree=npts - 1, physical_bounds=(-1.0, 1.0)
        )
        interface = Interface(
            bkd, interface_id=0, subdomain_ids=(0, 1),
            basis=interface_basis, normal_direction=0, ambient_dim=2
        )

        mesh0 = TransformedMesh2D(npts, npts, bkd)
        basis0 = ChebyshevBasis2D(mesh0, bkd)
        nodes_x0 = basis0.nodes_x()
        nodes_y0 = basis0.nodes_y()
        npts_total = basis0.npts()

        forcing0 = bkd.full((npts_total,), forcing_val)
        physics0 = AdvectionDiffusionReaction(
            basis0, bkd, diffusion=D, forcing=lambda t: forcing0
        )

        bounds0 = self._compute_boundary_indices_2d(npts, npts)
        left_bc0 = zero_dirichlet_bc(bkd, bounds0["left"])

        bottom_bc0 = DirichletBC(bkd, bounds0["bottom"],
                                 bkd.asarray([self._exact_sub0(nodes_x0[i]) for i in range(npts)]))
        top_bc0 = DirichletBC(bkd, bounds0["top"],
                              bkd.asarray([self._exact_sub0(nodes_x0[i]) for i in range(npts)]))

        wrapper0 = SubdomainWrapper(
            bkd, subdomain_id=0, physics=physics0,
            interfaces={0: interface},
            external_bcs=[left_bc0, bottom_bc0, top_bc0]
        )
        wrapper0.set_interface_boundary_indices(0, bounds0["right"])

        mesh1 = TransformedMesh2D(npts, npts, bkd)
        basis1 = ChebyshevBasis2D(mesh1, bkd)
        forcing1 = bkd.full((npts_total,), forcing_val)
        physics1 = AdvectionDiffusionReaction(
            basis1, bkd, diffusion=D, forcing=lambda t: forcing1
        )

        bounds1 = self._compute_boundary_indices_2d(npts, npts)

        # Asymmetric external BC: u(1) = 2
        right_bc1_vals = bkd.full((npts,), 2.0)
        right_bc1 = DirichletBC(bkd, bounds1["right"], right_bc1_vals)

        bottom_bc1 = DirichletBC(bkd, bounds1["bottom"],
                                 bkd.asarray([self._exact_sub1(nodes_x0[i]) for i in range(npts)]))
        top_bc1 = DirichletBC(bkd, bounds1["top"],
                              bkd.asarray([self._exact_sub1(nodes_x0[i]) for i in range(npts)]))

        wrapper1 = SubdomainWrapper(
            bkd, subdomain_id=1, physics=physics1,
            interfaces={0: interface},
            external_bcs=[right_bc1, bottom_bc1, top_bc1]
        )
        wrapper1.set_interface_boundary_indices(0, bounds1["left"])

        interface.set_subdomain_boundary_points(0, nodes_y0)
        interface.set_subdomain_boundary_points(1, nodes_y0)

        interface_dof_offsets = bkd.asarray([0, interface.ndofs()])

        residual = DtNResidual(
            bkd,
            interfaces={0: interface},
            subdomain_solvers={0: wrapper0, 1: wrapper1},
            interface_dof_offsets=interface_dof_offsets,
        )

        # Exact interface value = 5 for all y points
        ndofs = interface.ndofs()
        exact_dofs = bkd.full((ndofs,), self.exact_lambda)

        res = residual(exact_dofs)
        res_norm = float(bkd.norm(res))

        self.assertLess(res_norm, 1e-10,
                        f"Residual norm {res_norm} should be < 1e-10 at exact solution")


class TestTransientDtNTwoSpecies(unittest.TestCase):
    """Test transient DtN for two-species diffusion (uncoupled).

    Verifies vector-valued problems with manufactured solutions.
    Uses same reference domain approach as 1D scalar tests.
    """

    def setUp(self):
        self.bkd = NumpyBkd()
        self.npts = 16
        self.D0 = 1.0
        self.D1 = 0.5

    def _create_two_species_problem(self, time: float = 0.0):
        """Create two-species problem on reference domains.

        Each species satisfies -D_i * u_i'' = f_i independently.
        Using same symmetric setup as 1D scalar tests:
        - Subdomain 0: u(-1) = 0, u(1) = lambda
        - Subdomain 1: u(-1) = lambda, u(1) = 0

        With f_i = 2*D_i*(1+t), both species have lambda = 4*(1+t).
        """
        bkd = self.bkd
        npts = self.npts
        D0, D1 = self.D0, self.D1

        mms = ManufacturedSolutionTwoSpecies1D(bkd, D0, D1)

        # Interface for 2-component system
        interface = Interface1D(
            bkd, interface_id=0, subdomain_ids=(0, 1),
            interface_point=0.0, ncomponents=2
        )

        # Subdomain 0: [-1, 1] with external BC at x=-1, interface at x=1
        mesh0 = TransformedMesh1D(npts, bkd)
        basis0 = ChebyshevBasis1D(mesh0, bkd)
        nodes0 = basis0.nodes()

        def forcing0_0(t):
            return mms.forcing0(nodes0, t)

        def forcing0_1(t):
            return mms.forcing1(nodes0, t)

        # No reaction coupling
        reaction0 = LinearReaction(0.0, 0.0, 0.0, 0.0, bkd)

        physics0 = TwoSpeciesReactionDiffusionPhysics(
            basis0, bkd, diffusion0=D0, diffusion1=D1,
            reaction=reaction0, forcing0=forcing0_0, forcing1=forcing0_1
        )

        # External BC at x=-1 (index 0): u0 = u1 = 0
        # Component-stacked: [u0 indices, u1 indices]
        left_indices = bkd.asarray([0, npts])
        left_vals = bkd.asarray([0.0, 0.0])
        left_bc0 = DirichletBC(bkd, left_indices, left_vals)

        wrapper0 = SubdomainWrapper(
            bkd, subdomain_id=0, physics=physics0,
            interfaces={0: interface}, external_bcs=[left_bc0]
        )
        # Interface at x=1 (index npts-1)
        wrapper0.set_interface_boundary_indices(0, bkd.asarray([npts - 1]))

        # Subdomain 1: [-1, 1] with interface at x=-1, external BC at x=1
        mesh1 = TransformedMesh1D(npts, bkd)
        basis1 = ChebyshevBasis1D(mesh1, bkd)
        nodes1 = basis1.nodes()

        def forcing1_0(t):
            return mms.forcing0(nodes1, t)

        def forcing1_1(t):
            return mms.forcing1(nodes1, t)

        reaction1 = LinearReaction(0.0, 0.0, 0.0, 0.0, bkd)

        physics1 = TwoSpeciesReactionDiffusionPhysics(
            basis1, bkd, diffusion0=D0, diffusion1=D1,
            reaction=reaction1, forcing0=forcing1_0, forcing1=forcing1_1
        )

        # External BC at x=1 (index npts-1): u0 = u1 = 0
        right_indices = bkd.asarray([npts - 1, 2 * npts - 1])
        right_vals = bkd.asarray([0.0, 0.0])
        right_bc1 = DirichletBC(bkd, right_indices, right_vals)

        wrapper1 = SubdomainWrapper(
            bkd, subdomain_id=1, physics=physics1,
            interfaces={0: interface}, external_bcs=[right_bc1]
        )
        # Interface at x=-1 (index 0)
        wrapper1.set_interface_boundary_indices(0, bkd.asarray([0]))

        # Setup interface
        interface.set_subdomain_boundary_points(0, bkd.asarray([1.0]))
        interface.set_subdomain_boundary_points(1, bkd.asarray([-1.0]))

        interface_dof_offsets = bkd.asarray([0, 2])  # 2 DOFs (1 per component)

        return {
            "subdomain_solvers": {0: wrapper0, 1: wrapper1},
            "interfaces": {0: interface},
            "interface_dof_offsets": interface_dof_offsets,
            "mms": mms,
            "nodes": {0: nodes0, 1: nodes1},
        }

    def test_two_species_at_t0(self):
        """Test two-species DtN at t=0."""
        bkd = self.bkd
        t = 0.0
        problem = self._create_two_species_problem(time=t)

        residual = DtNResidual(
            bkd,
            interfaces=problem["interfaces"],
            subdomain_solvers=problem["subdomain_solvers"],
            interface_dof_offsets=problem["interface_dof_offsets"],
        )

        solver = DtNSolver(bkd, residual, max_iters=30, tol=1e-10)

        # Exact interface: [u0(interface), u1(interface)] = [4, 4] at t=0
        mms = problem["mms"]
        exact_u0, exact_u1 = mms.interface_values(t)
        initial_guess = bkd.asarray([2.0, 2.0])

        result = solver.solve(initial_guess)

        self.assertTrue(result.converged, "Two-species DtN should converge")

        computed_u0 = float(result.interface_dofs[0])
        computed_u1 = float(result.interface_dofs[1])

        self.assertAlmostEqual(computed_u0, exact_u0, places=6,
                               msg=f"u0 at interface: {computed_u0} vs {exact_u0}")
        self.assertAlmostEqual(computed_u1, exact_u1, places=6,
                               msg=f"u1 at interface: {computed_u1} vs {exact_u1}")

    def test_residual_zero_at_exact_solution(self):
        """Test residual is zero at the exact interface solution."""
        bkd = self.bkd
        t = 0.0
        problem = self._create_two_species_problem(time=t)

        residual = DtNResidual(
            bkd,
            interfaces=problem["interfaces"],
            subdomain_solvers=problem["subdomain_solvers"],
            interface_dof_offsets=problem["interface_dof_offsets"],
        )

        mms = problem["mms"]
        exact_u0, exact_u1 = mms.interface_values(t)
        exact_dofs = bkd.asarray([exact_u0, exact_u1])

        res = residual(exact_dofs)
        res_norm = float(bkd.norm(res))

        self.assertLess(res_norm, 1e-10,
                        f"Residual norm {res_norm} should be < 1e-10 at exact solution")

    def test_two_species_solution_accuracy(self):
        """Verify full two-species solution matches manufactured solution at t=0."""
        bkd = self.bkd
        t = 0.0  # Use t=0 since solver uses time=0 internally
        problem = self._create_two_species_problem(time=t)

        residual = DtNResidual(
            bkd,
            interfaces=problem["interfaces"],
            subdomain_solvers=problem["subdomain_solvers"],
            interface_dof_offsets=problem["interface_dof_offsets"],
        )

        solver = DtNSolver(bkd, residual, max_iters=30, tol=1e-10)
        result = solver.solve(bkd.asarray([3.0, 3.0]))

        self.assertTrue(result.converged)

        mms = problem["mms"]
        npts = self.npts

        for sub_id in [0, 1]:
            wrapper = problem["subdomain_solvers"][sub_id]
            sol = wrapper.solution()
            nodes = problem["nodes"][sub_id]

            # Split solution into components
            u0_computed = sol[:npts]
            u1_computed = sol[npts:]

            if sub_id == 0:
                u0_exact = mms.solution0_sub0(nodes, t)
                u1_exact = mms.solution1_sub0(nodes, t)
            else:
                u0_exact = mms.solution0_sub1(nodes, t)
                u1_exact = mms.solution1_sub1(nodes, t)

            max_error_u0 = float(bkd.max(bkd.abs(u0_computed - u0_exact)))
            max_error_u1 = float(bkd.max(bkd.abs(u1_computed - u1_exact)))

            self.assertLess(max_error_u0, 1e-8,
                            f"Subdomain {sub_id} u0 error = {max_error_u0}")
            self.assertLess(max_error_u1, 1e-8,
                            f"Subdomain {sub_id} u1 error = {max_error_u1}")


class TestDtN2DVariableDiffusion(unittest.TestCase):
    """Test 2D DtN with variable diffusion coefficient.

    Uses a spatially varying diffusion coefficient D(x) = D0*(1 + alpha*x)
    to verify that the flux computation correctly handles variable diffusion.

    Manufactured solution for -div(D*grad(u)) = f with linear u:
    u(x) = A + B*x  (y-independent)
    grad(u) = [B, 0]
    D(x) * grad(u) = [B*D(x), 0]
    div(D*grad(u)) = B*D_x = B*D0*alpha
    -div(D*grad(u)) = -B*D0*alpha
    So f = B*D0*alpha

    For flux conservation with asymmetric BCs:
    - Sub 0: u(-1) = 0, u(1) = lambda
    - Sub 1: u(-1) = lambda, u(1) = g (g != 0)

    Solution on sub 0: u_0 = lambda/2 + (lambda/2)*x
      B_0 = lambda/2, f_0 = (lambda/2)*D0*alpha

    Solution on sub 1: u_1 = (lambda+g)/2 + ((g-lambda)/2)*x
      B_1 = (g-lambda)/2, f_1 = ((g-lambda)/2)*D0*alpha

    Flux conservation:
    D(1)*B_0*(+1) + D(-1)*B_1*(-1) = 0
    D0*(1+alpha)*(lambda/2) - D0*(1-alpha)*((g-lambda)/2) = 0
    (1+alpha)*lambda - (1-alpha)*(g-lambda) = 0
    2*lambda = (1-alpha)*g
    lambda = (1-alpha)*g/2

    For alpha=0.3 and g=2: lambda = 0.7
    """

    def setUp(self):
        self.bkd = NumpyBkd()
        self.npts = 8
        self.D0 = 1.0
        self.alpha = 0.3  # D(x) = D0*(1 + alpha*x)
        self.g = 2.0  # External BC on subdomain 1
        self.exact_lambda = (1.0 - self.alpha) * self.g / 2.0  # = 0.7

    def _compute_boundary_indices_2d(self, npts_x, npts_y):
        """Compute 2D boundary indices (x varies fastest in tensor product ordering)."""
        bkd = self.bkd
        # Left: x-index=0 (x_ref=-1), all y
        left = bkd.asarray([j * npts_x for j in range(npts_y)])
        # Right: x-index=npts_x-1 (x_ref=+1), all y
        right = bkd.asarray([j * npts_x + (npts_x - 1) for j in range(npts_y)])
        # Bottom: y-index=0 (y_ref=-1), all x
        bottom = bkd.asarray(list(range(npts_x)))
        # Top: y-index=npts_y-1 (y_ref=+1), all x
        top = bkd.asarray([(npts_y - 1) * npts_x + i for i in range(npts_x)])
        return {"left": left, "right": right, "bottom": bottom, "top": top}

    def _build_variable_diffusion(self, nodes_x, npts_x, npts_y):
        """Build variable diffusion array D(x) = D0*(1 + alpha*x)."""
        bkd = self.bkd
        D0, alpha = self.D0, self.alpha
        npts_total = npts_x * npts_y
        D = bkd.zeros((npts_total,))
        for j in range(npts_y):
            for i in range(npts_x):
                idx = j * npts_x + i
                D[idx] = D0 * (1.0 + alpha * nodes_x[i])
        return D

    def _build_forcing_sub0(self, nodes_x, npts_x, npts_y):
        """Build constant forcing for sub 0: f = -B_0*D0*alpha.

        For residual = div(D*grad(u)) + f = 0, we need f = -div(D*grad(u)).
        With u = A + B*x and D = D0*(1+alpha*x):
            div(D*grad(u)) = B*D0*alpha
        So f = -B*D0*alpha
        """
        bkd = self.bkd
        B_0 = self.exact_lambda / 2.0
        f_val = -B_0 * self.D0 * self.alpha  # Negative sign!
        return bkd.full((npts_x * npts_y,), f_val)

    def _build_forcing_sub1(self, nodes_x, npts_x, npts_y):
        """Build constant forcing for sub 1: f = -B_1*D0*alpha."""
        bkd = self.bkd
        B_1 = (self.g - self.exact_lambda) / 2.0
        f_val = -B_1 * self.D0 * self.alpha  # Negative sign!
        return bkd.full((npts_x * npts_y,), f_val)

    def _exact_solution_sub0(self, x):
        """Exact solution on sub 0: u = lambda/2 + (lambda/2)*x."""
        lam = self.exact_lambda
        return lam / 2.0 + (lam / 2.0) * x

    def _exact_solution_sub1(self, x):
        """Exact solution on sub 1: u = (lambda+g)/2 + ((g-lambda)/2)*x."""
        lam = self.exact_lambda
        g = self.g
        return (lam + g) / 2.0 + ((g - lam) / 2.0) * x

    def test_variable_diffusion_residual_at_exact(self):
        """Test residual is zero at exact interface solution with variable D."""
        bkd = self.bkd
        npts = self.npts

        interface_basis = LegendreInterfaceBasis1D(
            bkd, degree=npts - 1, physical_bounds=(-1.0, 1.0)
        )
        interface = Interface(
            bkd, interface_id=0, subdomain_ids=(0, 1),
            basis=interface_basis, normal_direction=0, ambient_dim=2
        )

        mesh0 = TransformedMesh2D(npts, npts, bkd)
        basis0 = ChebyshevBasis2D(mesh0, bkd)
        mesh1 = TransformedMesh2D(npts, npts, bkd)
        basis1 = ChebyshevBasis2D(mesh1, bkd)
        nodes_x0 = basis0.nodes_x()
        nodes_y0 = basis0.nodes_y()
        npts_total = basis0.npts()

        bounds0 = self._compute_boundary_indices_2d(npts, npts)
        bounds1 = self._compute_boundary_indices_2d(npts, npts)

        D0_arr = self._build_variable_diffusion(nodes_x0, npts, npts)
        D1_arr = self._build_variable_diffusion(nodes_x0, npts, npts)
        f0 = self._build_forcing_sub0(nodes_x0, npts, npts)
        f1 = self._build_forcing_sub1(nodes_x0, npts, npts)

        physics0 = AdvectionDiffusionReaction(
            basis0, bkd, diffusion=D0_arr, forcing=lambda t: f0
        )
        physics1 = AdvectionDiffusionReaction(
            basis1, bkd, diffusion=D1_arr, forcing=lambda t: f1
        )

        # External BC: u(-1) = 0 on sub 0
        left_bc0 = zero_dirichlet_bc(bkd, bounds0["left"])

        # External BC: u(1) = g on sub 1
        right_bc1_vals = bkd.full((npts,), self.g)
        right_bc1 = DirichletBC(bkd, bounds1["right"], right_bc1_vals)

        # Y-boundary BCs using exact solutions
        exact_y_bc0 = bkd.asarray([self._exact_solution_sub0(nodes_x0[i]) for i in range(npts)])
        bottom_bc0 = DirichletBC(bkd, bounds0["bottom"], exact_y_bc0)
        top_bc0 = DirichletBC(bkd, bounds0["top"], exact_y_bc0)

        exact_y_bc1 = bkd.asarray([self._exact_solution_sub1(nodes_x0[i]) for i in range(npts)])
        bottom_bc1 = DirichletBC(bkd, bounds1["bottom"], exact_y_bc1)
        top_bc1 = DirichletBC(bkd, bounds1["top"], exact_y_bc1)

        wrapper0 = SubdomainWrapper(
            bkd, subdomain_id=0, physics=physics0,
            interfaces={0: interface},
            external_bcs=[left_bc0, bottom_bc0, top_bc0]
        )
        wrapper0.set_interface_boundary_indices(0, bounds0["right"])

        wrapper1 = SubdomainWrapper(
            bkd, subdomain_id=1, physics=physics1,
            interfaces={0: interface},
            external_bcs=[right_bc1, bottom_bc1, top_bc1]
        )
        wrapper1.set_interface_boundary_indices(0, bounds1["left"])

        interface.set_subdomain_boundary_points(0, nodes_y0)
        interface.set_subdomain_boundary_points(1, nodes_y0)

        interface_dof_offsets = bkd.asarray([0, interface.ndofs()])

        residual = DtNResidual(
            bkd,
            interfaces={0: interface},
            subdomain_solvers={0: wrapper0, 1: wrapper1},
            interface_dof_offsets=interface_dof_offsets,
        )

        ndofs = interface.ndofs()
        exact_dofs = bkd.full((ndofs,), self.exact_lambda)

        res = residual(exact_dofs)
        res_norm = float(bkd.norm(res))

        self.assertLess(res_norm, 1e-9,
                        f"Residual norm {res_norm} should be < 1e-9 at exact solution")

    def test_variable_diffusion_solver_converges(self):
        """Test DtN solver converges with variable diffusion."""
        bkd = self.bkd
        npts = self.npts

        interface_basis = LegendreInterfaceBasis1D(
            bkd, degree=npts - 1, physical_bounds=(-1.0, 1.0)
        )
        interface = Interface(
            bkd, interface_id=0, subdomain_ids=(0, 1),
            basis=interface_basis, normal_direction=0, ambient_dim=2
        )

        mesh0 = TransformedMesh2D(npts, npts, bkd)
        basis0 = ChebyshevBasis2D(mesh0, bkd)
        mesh1 = TransformedMesh2D(npts, npts, bkd)
        basis1 = ChebyshevBasis2D(mesh1, bkd)
        nodes_x0 = basis0.nodes_x()
        nodes_y0 = basis0.nodes_y()

        bounds0 = self._compute_boundary_indices_2d(npts, npts)
        bounds1 = self._compute_boundary_indices_2d(npts, npts)

        D0_arr = self._build_variable_diffusion(nodes_x0, npts, npts)
        D1_arr = self._build_variable_diffusion(nodes_x0, npts, npts)
        f0 = self._build_forcing_sub0(nodes_x0, npts, npts)
        f1 = self._build_forcing_sub1(nodes_x0, npts, npts)

        physics0 = AdvectionDiffusionReaction(
            basis0, bkd, diffusion=D0_arr, forcing=lambda t: f0
        )
        physics1 = AdvectionDiffusionReaction(
            basis1, bkd, diffusion=D1_arr, forcing=lambda t: f1
        )

        left_bc0 = zero_dirichlet_bc(bkd, bounds0["left"])
        right_bc1_vals = bkd.full((npts,), self.g)
        right_bc1 = DirichletBC(bkd, bounds1["right"], right_bc1_vals)

        exact_y_bc0 = bkd.asarray([self._exact_solution_sub0(nodes_x0[i]) for i in range(npts)])
        bottom_bc0 = DirichletBC(bkd, bounds0["bottom"], exact_y_bc0)
        top_bc0 = DirichletBC(bkd, bounds0["top"], exact_y_bc0)

        exact_y_bc1 = bkd.asarray([self._exact_solution_sub1(nodes_x0[i]) for i in range(npts)])
        bottom_bc1 = DirichletBC(bkd, bounds1["bottom"], exact_y_bc1)
        top_bc1 = DirichletBC(bkd, bounds1["top"], exact_y_bc1)

        wrapper0 = SubdomainWrapper(
            bkd, subdomain_id=0, physics=physics0,
            interfaces={0: interface},
            external_bcs=[left_bc0, bottom_bc0, top_bc0]
        )
        wrapper0.set_interface_boundary_indices(0, bounds0["right"])

        wrapper1 = SubdomainWrapper(
            bkd, subdomain_id=1, physics=physics1,
            interfaces={0: interface},
            external_bcs=[right_bc1, bottom_bc1, top_bc1]
        )
        wrapper1.set_interface_boundary_indices(0, bounds1["left"])

        interface.set_subdomain_boundary_points(0, nodes_y0)
        interface.set_subdomain_boundary_points(1, nodes_y0)

        interface_dof_offsets = bkd.asarray([0, interface.ndofs()])

        residual = DtNResidual(
            bkd,
            interfaces={0: interface},
            subdomain_solvers={0: wrapper0, 1: wrapper1},
            interface_dof_offsets=interface_dof_offsets,
        )

        solver = DtNSolver(bkd, residual, max_iters=30, tol=1e-10)

        ndofs = interface.ndofs()
        result = solver.solve(bkd.full((ndofs,), 0.5))

        self.assertTrue(result.converged, "Variable diffusion DtN should converge")

        computed_vals = interface.evaluate(result.interface_dofs)
        exact_val = self.exact_lambda  # 0.7

        max_error = float(bkd.max(bkd.abs(computed_vals - exact_val)))
        self.assertLess(max_error, 1e-8,
                        f"Variable diffusion interface error = {max_error}")


class TestDtN3DScalar(unittest.TestCase):
    """Test 3D DtN with scalar diffusion and manufactured solutions.

    Problem: -Laplacian(u) = f on [-1, 1]^3
    Split into left subdomain and right subdomain at x=0.

    Manufactured solution with non-zero interface flux:
    - Linear solution: u = A + B*x (satisfies Laplace when f=0)
    - Sub 0: u(-1) = 0, u(1) = lambda
    - Sub 1: u(-1) = lambda, u(1) = g (asymmetric)
    - lambda chosen for flux conservation

    For f=0 (Laplace):
    u_0 = lambda/2 + (lambda/2)*x, u_0(-1)=0, u_0(1)=lambda
    u_1 = (lambda+g)/2 + ((g-lambda)/2)*x, u_1(-1)=lambda, u_1(1)=g

    Flux conservation: du_0/dx(1) + (-1)*du_1/dx(-1) = 0
    lambda/2 - (g-lambda)/2 = 0
    lambda = g/2

    For g=2: lambda=1
    """

    def setUp(self):
        self.bkd = NumpyBkd()
        self.npts = 5
        self.D = 1.0
        self.g = 2.0
        self.exact_lambda = self.g / 2.0  # = 1.0

    def _compute_boundary_indices_3d(self, npts_x, npts_y, npts_z):
        """Compute 3D boundary indices (x varies fastest, then y, then z)."""
        bkd = self.bkd
        npts_xy = npts_x * npts_y

        # Left: x-index=0 (x=-1), all y and z
        left = []
        for k in range(npts_z):
            for j in range(npts_y):
                left.append(k * npts_xy + j * npts_x)
        left = bkd.asarray(left)

        # Right: x-index=npts_x-1 (x=+1), all y and z
        right = []
        for k in range(npts_z):
            for j in range(npts_y):
                right.append(k * npts_xy + j * npts_x + (npts_x - 1))
        right = bkd.asarray(right)

        # Bottom: y-index=0 (y=-1), all x and z
        bottom = []
        for k in range(npts_z):
            for i in range(npts_x):
                bottom.append(k * npts_xy + i)
        bottom = bkd.asarray(bottom)

        # Top: y-index=npts_y-1 (y=+1), all x and z
        top = []
        for k in range(npts_z):
            for i in range(npts_x):
                top.append(k * npts_xy + (npts_y - 1) * npts_x + i)
        top = bkd.asarray(top)

        # Front: z-index=0 (z=-1), all x and y
        front = []
        for j in range(npts_y):
            for i in range(npts_x):
                front.append(j * npts_x + i)
        front = bkd.asarray(front)

        # Back: z-index=npts_z-1 (z=+1), all x and y
        back = []
        for j in range(npts_y):
            for i in range(npts_x):
                back.append((npts_z - 1) * npts_xy + j * npts_x + i)
        back = bkd.asarray(back)

        return {
            "left": left, "right": right,
            "bottom": bottom, "top": top,
            "front": front, "back": back,
        }

    def _exact_solution_sub0(self, x):
        """Exact solution on sub 0: u = lambda/2 + (lambda/2)*x."""
        lam = self.exact_lambda
        return lam / 2.0 + (lam / 2.0) * x

    def _exact_solution_sub1(self, x):
        """Exact solution on sub 1: u = (lambda+g)/2 + ((g-lambda)/2)*x."""
        lam = self.exact_lambda
        g = self.g
        return (lam + g) / 2.0 + ((g - lam) / 2.0) * x

    def test_3d_scalar_residual_at_exact(self):
        """Test residual is zero at exact interface solution for 3D scalar."""
        bkd = self.bkd
        npts = self.npts

        # 2D interface basis for 3D problem
        interface_basis = LegendreInterfaceBasis2D(
            bkd, degree_y=npts - 1, degree_z=npts - 1,
            physical_bounds_y=(-1.0, 1.0), physical_bounds_z=(-1.0, 1.0)
        )
        interface = Interface2D(
            bkd, interface_id=0, subdomain_ids=(0, 1),
            basis=interface_basis, normal_direction=0
        )

        mesh0 = TransformedMesh3D(npts, npts, npts, bkd)
        basis0 = ChebyshevBasis3D(mesh0, bkd)
        mesh1 = TransformedMesh3D(npts, npts, npts, bkd)
        basis1 = ChebyshevBasis3D(mesh1, bkd)
        nodes_x0 = basis0.nodes_x()
        nodes_y0 = basis0.nodes_y()
        nodes_z0 = basis0.nodes_z()

        bounds0 = self._compute_boundary_indices_3d(npts, npts, npts)
        bounds1 = self._compute_boundary_indices_3d(npts, npts, npts)

        # Laplace equation: -Laplacian(u) = 0, so forcing = 0
        f0 = bkd.zeros((basis0.npts(),))
        f1 = bkd.zeros((basis1.npts(),))

        physics0 = AdvectionDiffusionReaction(
            basis0, bkd, diffusion=self.D, forcing=lambda t: f0
        )
        physics1 = AdvectionDiffusionReaction(
            basis1, bkd, diffusion=self.D, forcing=lambda t: f1
        )

        # External BC: u(-1) = 0 on sub 0
        left_bc0 = zero_dirichlet_bc(bkd, bounds0["left"])

        # External BC: u(1) = g on sub 1
        right_bc1 = DirichletBC(bkd, bounds1["right"],
                                bkd.full((npts * npts,), self.g))

        # Y-boundary BCs for sub 0 (u depends only on x)
        bc_vals_y0 = bkd.zeros((npts * npts,))
        idx = 0
        for k in range(npts):
            for i in range(npts):
                bc_vals_y0[idx] = self._exact_solution_sub0(nodes_x0[i])
                idx += 1
        bottom_bc0 = DirichletBC(bkd, bounds0["bottom"], bc_vals_y0)
        top_bc0 = DirichletBC(bkd, bounds0["top"], bc_vals_y0)

        # Z-boundary BCs for sub 0
        bc_vals_z0 = bkd.zeros((npts * npts,))
        idx = 0
        for j in range(npts):
            for i in range(npts):
                bc_vals_z0[idx] = self._exact_solution_sub0(nodes_x0[i])
                idx += 1
        front_bc0 = DirichletBC(bkd, bounds0["front"], bc_vals_z0)
        back_bc0 = DirichletBC(bkd, bounds0["back"], bc_vals_z0)

        # Y-boundary BCs for sub 1
        bc_vals_y1 = bkd.zeros((npts * npts,))
        idx = 0
        for k in range(npts):
            for i in range(npts):
                bc_vals_y1[idx] = self._exact_solution_sub1(nodes_x0[i])
                idx += 1
        bottom_bc1 = DirichletBC(bkd, bounds1["bottom"], bc_vals_y1)
        top_bc1 = DirichletBC(bkd, bounds1["top"], bc_vals_y1)

        # Z-boundary BCs for sub 1
        bc_vals_z1 = bkd.zeros((npts * npts,))
        idx = 0
        for j in range(npts):
            for i in range(npts):
                bc_vals_z1[idx] = self._exact_solution_sub1(nodes_x0[i])
                idx += 1
        front_bc1 = DirichletBC(bkd, bounds1["front"], bc_vals_z1)
        back_bc1 = DirichletBC(bkd, bounds1["back"], bc_vals_z1)

        wrapper0 = SubdomainWrapper(
            bkd, subdomain_id=0, physics=physics0,
            interfaces={0: interface},
            external_bcs=[left_bc0, bottom_bc0, top_bc0, front_bc0, back_bc0]
        )
        wrapper0.set_interface_boundary_indices(0, bounds0["right"])

        wrapper1 = SubdomainWrapper(
            bkd, subdomain_id=1, physics=physics1,
            interfaces={0: interface},
            external_bcs=[right_bc1, bottom_bc1, top_bc1, front_bc1, back_bc1]
        )
        wrapper1.set_interface_boundary_indices(0, bounds1["left"])

        interface.set_subdomain_boundary_points_2d(0, nodes_y0, nodes_z0)
        interface.set_subdomain_boundary_points_2d(1, nodes_y0, nodes_z0)

        interface_dof_offsets = bkd.asarray([0, interface.ndofs()])

        residual = DtNResidual(
            bkd,
            interfaces={0: interface},
            subdomain_solvers={0: wrapper0, 1: wrapper1},
            interface_dof_offsets=interface_dof_offsets,
        )

        ndofs = interface.ndofs()
        exact_dofs = bkd.full((ndofs,), self.exact_lambda)

        res = residual(exact_dofs)
        res_norm = float(bkd.norm(res))

        self.assertLess(res_norm, 1e-9,
                        f"3D residual norm {res_norm} should be < 1e-9")

    def test_3d_scalar_solver_converges(self):
        """Test DtN solver converges for 3D scalar problem."""
        bkd = self.bkd
        npts = self.npts

        interface_basis = LegendreInterfaceBasis2D(
            bkd, degree_y=npts - 1, degree_z=npts - 1,
            physical_bounds_y=(-1.0, 1.0), physical_bounds_z=(-1.0, 1.0)
        )
        interface = Interface2D(
            bkd, interface_id=0, subdomain_ids=(0, 1),
            basis=interface_basis, normal_direction=0
        )

        mesh0 = TransformedMesh3D(npts, npts, npts, bkd)
        basis0 = ChebyshevBasis3D(mesh0, bkd)
        mesh1 = TransformedMesh3D(npts, npts, npts, bkd)
        basis1 = ChebyshevBasis3D(mesh1, bkd)
        nodes_x0 = basis0.nodes_x()
        nodes_y0 = basis0.nodes_y()
        nodes_z0 = basis0.nodes_z()

        bounds0 = self._compute_boundary_indices_3d(npts, npts, npts)
        bounds1 = self._compute_boundary_indices_3d(npts, npts, npts)

        f0 = bkd.zeros((basis0.npts(),))
        f1 = bkd.zeros((basis1.npts(),))

        physics0 = AdvectionDiffusionReaction(
            basis0, bkd, diffusion=self.D, forcing=lambda t: f0
        )
        physics1 = AdvectionDiffusionReaction(
            basis1, bkd, diffusion=self.D, forcing=lambda t: f1
        )

        left_bc0 = zero_dirichlet_bc(bkd, bounds0["left"])
        right_bc1 = DirichletBC(bkd, bounds1["right"],
                                bkd.full((npts * npts,), self.g))

        bc_vals_y0 = bkd.zeros((npts * npts,))
        idx = 0
        for k in range(npts):
            for i in range(npts):
                bc_vals_y0[idx] = self._exact_solution_sub0(nodes_x0[i])
                idx += 1
        bottom_bc0 = DirichletBC(bkd, bounds0["bottom"], bc_vals_y0)
        top_bc0 = DirichletBC(bkd, bounds0["top"], bc_vals_y0)

        bc_vals_z0 = bkd.zeros((npts * npts,))
        idx = 0
        for j in range(npts):
            for i in range(npts):
                bc_vals_z0[idx] = self._exact_solution_sub0(nodes_x0[i])
                idx += 1
        front_bc0 = DirichletBC(bkd, bounds0["front"], bc_vals_z0)
        back_bc0 = DirichletBC(bkd, bounds0["back"], bc_vals_z0)

        bc_vals_y1 = bkd.zeros((npts * npts,))
        idx = 0
        for k in range(npts):
            for i in range(npts):
                bc_vals_y1[idx] = self._exact_solution_sub1(nodes_x0[i])
                idx += 1
        bottom_bc1 = DirichletBC(bkd, bounds1["bottom"], bc_vals_y1)
        top_bc1 = DirichletBC(bkd, bounds1["top"], bc_vals_y1)

        bc_vals_z1 = bkd.zeros((npts * npts,))
        idx = 0
        for j in range(npts):
            for i in range(npts):
                bc_vals_z1[idx] = self._exact_solution_sub1(nodes_x0[i])
                idx += 1
        front_bc1 = DirichletBC(bkd, bounds1["front"], bc_vals_z1)
        back_bc1 = DirichletBC(bkd, bounds1["back"], bc_vals_z1)

        wrapper0 = SubdomainWrapper(
            bkd, subdomain_id=0, physics=physics0,
            interfaces={0: interface},
            external_bcs=[left_bc0, bottom_bc0, top_bc0, front_bc0, back_bc0]
        )
        wrapper0.set_interface_boundary_indices(0, bounds0["right"])

        wrapper1 = SubdomainWrapper(
            bkd, subdomain_id=1, physics=physics1,
            interfaces={0: interface},
            external_bcs=[right_bc1, bottom_bc1, top_bc1, front_bc1, back_bc1]
        )
        wrapper1.set_interface_boundary_indices(0, bounds1["left"])

        interface.set_subdomain_boundary_points_2d(0, nodes_y0, nodes_z0)
        interface.set_subdomain_boundary_points_2d(1, nodes_y0, nodes_z0)

        interface_dof_offsets = bkd.asarray([0, interface.ndofs()])

        residual = DtNResidual(
            bkd,
            interfaces={0: interface},
            subdomain_solvers={0: wrapper0, 1: wrapper1},
            interface_dof_offsets=interface_dof_offsets,
        )

        solver = DtNSolver(bkd, residual, max_iters=30, tol=1e-10)

        ndofs = interface.ndofs()
        result = solver.solve(bkd.full((ndofs,), 0.5))

        self.assertTrue(result.converged, "3D scalar DtN should converge")

        computed_vals = interface.evaluate(result.interface_dofs)
        max_error = float(bkd.max(bkd.abs(computed_vals - self.exact_lambda)))
        self.assertLess(max_error, 1e-8,
                        f"3D interface error = {max_error}")


class TestDtN2DVector(unittest.TestCase):
    """Test 2D DtN with vector-valued linear elasticity.

    Problem: 2D linear elasticity equilibrium
        -div(σ) = f
        σ = λ*tr(ε)*I + 2μ*ε

    Split domain at x=0, enforce traction continuity at interface.

    Manufactured solution (linear displacement):
    u(x,y) = (u_x, u_y) = (a*x, 0)  (uniaxial extension in x)

    This gives:
    ε_xx = a, ε_yy = 0, ε_xy = 0
    σ_xx = (λ + 2μ)*a, σ_yy = λ*a, σ_xy = 0

    For equilibrium with zero body force:
    div(σ)_x = dσ_xx/dx + dσ_xy/dy = 0 ✓
    div(σ)_y = dσ_xy/dx + dσ_yy/dy = 0 ✓

    Traction on interface (normal n = (1, 0)):
    t = σ·n = ((λ + 2μ)*a, 0)
    """

    def setUp(self):
        self.bkd = NumpyBkd()
        self.npts = 6
        self.lamda = 1.0  # Lamé first parameter
        self.mu = 1.0     # Shear modulus
        self.g = 2.0      # External BC on sub 1

    def _compute_boundary_indices_2d(self, npts_x, npts_y):
        """Compute 2D boundary indices (x varies fastest in tensor product ordering)."""
        bkd = self.bkd
        # Left: x-index=0 (x_ref=-1), all y
        left = bkd.asarray([j * npts_x for j in range(npts_y)])
        # Right: x-index=npts_x-1 (x_ref=+1), all y
        right = bkd.asarray([j * npts_x + (npts_x - 1) for j in range(npts_y)])
        # Bottom: y-index=0 (y_ref=-1), all x
        bottom = bkd.asarray(list(range(npts_x)))
        # Top: y-index=npts_y-1 (y_ref=+1), all x
        top = bkd.asarray([(npts_y - 1) * npts_x + i for i in range(npts_x)])
        return {"left": left, "right": right, "bottom": bottom, "top": top}

    def _exact_ux_sub0(self, x):
        """u_x on sub 0: linear displacement u_x = (lambda/2)*(x+1)."""
        # u_x(-1) = 0, u_x(1) = lambda
        lam = self.g / 2.0  # Interface value for u_x
        return (lam / 2.0) * (x + 1.0)

    def _exact_ux_sub1(self, x):
        """u_x on sub 1: linear displacement."""
        lam = self.g / 2.0
        g = self.g
        # u_x(-1) = lambda, u_x(1) = g
        return lam + ((g - lam) / 2.0) * (x + 1.0)

    def test_2d_vector_residual_at_exact(self):
        """Test residual at exact interface for 2D vector elasticity."""
        bkd = self.bkd
        npts = self.npts

        # Interface basis with 2 components (u_x, u_y)
        interface_basis = LegendreInterfaceBasis1D(
            bkd, degree=npts - 1, physical_bounds=(-1.0, 1.0)
        )
        interface = Interface(
            bkd, interface_id=0, subdomain_ids=(0, 1),
            basis=interface_basis, normal_direction=0, ambient_dim=2,
            ncomponents=2  # Vector-valued
        )

        mesh0 = TransformedMesh2D(npts, npts, bkd)
        basis0 = ChebyshevBasis2D(mesh0, bkd)
        mesh1 = TransformedMesh2D(npts, npts, bkd)
        basis1 = ChebyshevBasis2D(mesh1, bkd)
        nodes_x0 = basis0.nodes_x()
        nodes_y0 = basis0.nodes_y()
        npts_total = basis0.npts()

        bounds0 = self._compute_boundary_indices_2d(npts, npts)
        bounds1 = self._compute_boundary_indices_2d(npts, npts)

        # Zero forcing (equilibrium solution satisfies homogeneous eq)
        physics0 = LinearElasticityPhysics(
            basis0, bkd, lamda=self.lamda, mu=self.mu, forcing=None
        )
        physics1 = LinearElasticityPhysics(
            basis1, bkd, lamda=self.lamda, mu=self.mu, forcing=None
        )

        # External BCs for sub 0: u_x(-1)=0, u_y(-1)=0
        left_bc0_ux = zero_dirichlet_bc(bkd, bounds0["left"])
        left_bc0_uy = zero_dirichlet_bc(bkd, bounds0["left"] + npts_total)

        # External BCs for sub 1: u_x(1)=g, u_y(1)=0
        right_bc1_ux = DirichletBC(bkd, bounds1["right"],
                                   bkd.full((npts,), self.g))
        right_bc1_uy = zero_dirichlet_bc(bkd, bounds1["right"] + npts_total)

        # Y-boundary BCs for sub 0 (u_y=0, u_x depends on x)
        bc_vals_ux0 = bkd.asarray([self._exact_ux_sub0(nodes_x0[i]) for i in range(npts)])
        bottom_bc0_ux = DirichletBC(bkd, bounds0["bottom"], bc_vals_ux0)
        top_bc0_ux = DirichletBC(bkd, bounds0["top"], bc_vals_ux0)
        bottom_bc0_uy = zero_dirichlet_bc(bkd, bounds0["bottom"] + npts_total)
        top_bc0_uy = zero_dirichlet_bc(bkd, bounds0["top"] + npts_total)

        # Y-boundary BCs for sub 1
        bc_vals_ux1 = bkd.asarray([self._exact_ux_sub1(nodes_x0[i]) for i in range(npts)])
        bottom_bc1_ux = DirichletBC(bkd, bounds1["bottom"], bc_vals_ux1)
        top_bc1_ux = DirichletBC(bkd, bounds1["top"], bc_vals_ux1)
        bottom_bc1_uy = zero_dirichlet_bc(bkd, bounds1["bottom"] + npts_total)
        top_bc1_uy = zero_dirichlet_bc(bkd, bounds1["top"] + npts_total)

        wrapper0 = SubdomainWrapper(
            bkd, subdomain_id=0, physics=physics0,
            interfaces={0: interface},
            external_bcs=[left_bc0_ux, left_bc0_uy,
                          bottom_bc0_ux, top_bc0_ux,
                          bottom_bc0_uy, top_bc0_uy]
        )
        wrapper0.set_interface_boundary_indices(0, bounds0["right"])

        wrapper1 = SubdomainWrapper(
            bkd, subdomain_id=1, physics=physics1,
            interfaces={0: interface},
            external_bcs=[right_bc1_ux, right_bc1_uy,
                          bottom_bc1_ux, top_bc1_ux,
                          bottom_bc1_uy, top_bc1_uy]
        )
        wrapper1.set_interface_boundary_indices(0, bounds1["left"])

        interface.set_subdomain_boundary_points(0, nodes_y0)
        interface.set_subdomain_boundary_points(1, nodes_y0)

        # Total DOFs = ndofs * ncomponents
        total_ndofs = interface.total_ndofs()
        interface_dof_offsets = bkd.asarray([0, total_ndofs])

        residual = DtNResidual(
            bkd,
            interfaces={0: interface},
            subdomain_solvers={0: wrapper0, 1: wrapper1},
            interface_dof_offsets=interface_dof_offsets,
        )

        # Exact interface DOFs: [u_x values, u_y values]
        exact_ux = self.g / 2.0  # lambda = g/2
        ndofs_per_comp = interface.ndofs()
        exact_dofs = bkd.concatenate([
            bkd.full((ndofs_per_comp,), exact_ux),  # u_x
            bkd.zeros((ndofs_per_comp,))  # u_y = 0
        ])

        res = residual(exact_dofs)
        res_norm = float(bkd.norm(res))

        self.assertLess(res_norm, 1e-8,
                        f"2D vector residual {res_norm} should be < 1e-8")

    def test_2d_vector_solver_converges(self):
        """Test DtN solver converges for 2D vector elasticity."""
        bkd = self.bkd
        npts = self.npts

        interface_basis = LegendreInterfaceBasis1D(
            bkd, degree=npts - 1, physical_bounds=(-1.0, 1.0)
        )
        interface = Interface(
            bkd, interface_id=0, subdomain_ids=(0, 1),
            basis=interface_basis, normal_direction=0, ambient_dim=2,
            ncomponents=2
        )

        mesh0 = TransformedMesh2D(npts, npts, bkd)
        basis0 = ChebyshevBasis2D(mesh0, bkd)
        mesh1 = TransformedMesh2D(npts, npts, bkd)
        basis1 = ChebyshevBasis2D(mesh1, bkd)
        nodes_x0 = basis0.nodes_x()
        nodes_y0 = basis0.nodes_y()
        npts_total = basis0.npts()

        bounds0 = self._compute_boundary_indices_2d(npts, npts)
        bounds1 = self._compute_boundary_indices_2d(npts, npts)

        physics0 = LinearElasticityPhysics(
            basis0, bkd, lamda=self.lamda, mu=self.mu, forcing=None
        )
        physics1 = LinearElasticityPhysics(
            basis1, bkd, lamda=self.lamda, mu=self.mu, forcing=None
        )

        left_bc0_ux = zero_dirichlet_bc(bkd, bounds0["left"])
        left_bc0_uy = zero_dirichlet_bc(bkd, bounds0["left"] + npts_total)

        right_bc1_ux = DirichletBC(bkd, bounds1["right"],
                                   bkd.full((npts,), self.g))
        right_bc1_uy = zero_dirichlet_bc(bkd, bounds1["right"] + npts_total)

        bc_vals_ux0 = bkd.asarray([self._exact_ux_sub0(nodes_x0[i]) for i in range(npts)])
        bottom_bc0_ux = DirichletBC(bkd, bounds0["bottom"], bc_vals_ux0)
        top_bc0_ux = DirichletBC(bkd, bounds0["top"], bc_vals_ux0)
        bottom_bc0_uy = zero_dirichlet_bc(bkd, bounds0["bottom"] + npts_total)
        top_bc0_uy = zero_dirichlet_bc(bkd, bounds0["top"] + npts_total)

        bc_vals_ux1 = bkd.asarray([self._exact_ux_sub1(nodes_x0[i]) for i in range(npts)])
        bottom_bc1_ux = DirichletBC(bkd, bounds1["bottom"], bc_vals_ux1)
        top_bc1_ux = DirichletBC(bkd, bounds1["top"], bc_vals_ux1)
        bottom_bc1_uy = zero_dirichlet_bc(bkd, bounds1["bottom"] + npts_total)
        top_bc1_uy = zero_dirichlet_bc(bkd, bounds1["top"] + npts_total)

        wrapper0 = SubdomainWrapper(
            bkd, subdomain_id=0, physics=physics0,
            interfaces={0: interface},
            external_bcs=[left_bc0_ux, left_bc0_uy,
                          bottom_bc0_ux, top_bc0_ux,
                          bottom_bc0_uy, top_bc0_uy]
        )
        wrapper0.set_interface_boundary_indices(0, bounds0["right"])

        wrapper1 = SubdomainWrapper(
            bkd, subdomain_id=1, physics=physics1,
            interfaces={0: interface},
            external_bcs=[right_bc1_ux, right_bc1_uy,
                          bottom_bc1_ux, top_bc1_ux,
                          bottom_bc1_uy, top_bc1_uy]
        )
        wrapper1.set_interface_boundary_indices(0, bounds1["left"])

        interface.set_subdomain_boundary_points(0, nodes_y0)
        interface.set_subdomain_boundary_points(1, nodes_y0)

        total_ndofs = interface.total_ndofs()
        interface_dof_offsets = bkd.asarray([0, total_ndofs])

        residual = DtNResidual(
            bkd,
            interfaces={0: interface},
            subdomain_solvers={0: wrapper0, 1: wrapper1},
            interface_dof_offsets=interface_dof_offsets,
        )

        solver = DtNSolver(bkd, residual, max_iters=30, tol=1e-10)

        result = solver.solve(bkd.zeros((total_ndofs,)))

        self.assertTrue(result.converged, "2D vector DtN should converge")

        # Check u_x values
        ndofs_per_comp = interface.ndofs()
        computed_ux = result.interface_dofs[:ndofs_per_comp]
        computed_uy = result.interface_dofs[ndofs_per_comp:]

        exact_ux = self.g / 2.0
        max_error_ux = float(bkd.max(bkd.abs(computed_ux - exact_ux)))
        max_error_uy = float(bkd.max(bkd.abs(computed_uy)))

        self.assertLess(max_error_ux, 1e-8,
                        f"u_x error = {max_error_ux}")
        self.assertLess(max_error_uy, 1e-8,
                        f"u_y error = {max_error_uy}")


class TestDtN3DWithForcing(unittest.TestCase):
    """Test 3D DtN with non-zero forcing and asymmetric BCs.

    Similar to TestDtN2DWithForcing but in 3D.

    Manufactured solution with non-zero interface flux:
    - Sub 0: u = (lambda/2 + 1) + (lambda/2)*x - x², u(-1)=0, u(1)=lambda
    - Sub 1: u = (lambda+g)/2 + ((g-lambda)/2)*x - x², u(-1)=lambda, u(1)=g-2

    Forcing: -u'' = 2 (constant)

    For flux conservation with asymmetric BCs (g != 0):
    du_0/dx(1) = lambda/2 - 2 = lambda/2 - 2
    du_1/dx(-1) = (g-lambda)/2 + 2

    Flux conservation: (lambda/2 - 2) - ((g-lambda)/2 + 2) = 0
    lambda - 4 = g - lambda + 4 = 0
    2*lambda = g + 8
    lambda = (g + 8)/2

    For g = 0: lambda = 4
    """

    def setUp(self):
        self.bkd = NumpyBkd()
        self.npts = 6
        self.D = 1.0
        self.g = 2.0  # Asymmetric external BC (same as 2D test)
        self.exact_lambda = (self.g + 8.0) / 2.0  # = 5.0

    def _compute_boundary_indices_3d(self, npts_x, npts_y, npts_z):
        """Compute 3D boundary indices (x varies fastest, then y, then z)."""
        bkd = self.bkd
        npts_xy = npts_x * npts_y

        # Left: x-index=0 (x=-1), all y and z
        left = []
        for k in range(npts_z):
            for j in range(npts_y):
                left.append(k * npts_xy + j * npts_x)
        left = bkd.asarray(left)

        # Right: x-index=npts_x-1 (x=+1), all y and z
        right = []
        for k in range(npts_z):
            for j in range(npts_y):
                right.append(k * npts_xy + j * npts_x + (npts_x - 1))
        right = bkd.asarray(right)

        # Bottom: y-index=0 (y=-1), all x and z
        bottom = []
        for k in range(npts_z):
            for i in range(npts_x):
                bottom.append(k * npts_xy + i)
        bottom = bkd.asarray(bottom)

        # Top: y-index=npts_y-1 (y=+1), all x and z
        top = []
        for k in range(npts_z):
            for i in range(npts_x):
                top.append(k * npts_xy + (npts_y - 1) * npts_x + i)
        top = bkd.asarray(top)

        # Front: z-index=0 (z=-1), all x and y
        front = []
        for j in range(npts_y):
            for i in range(npts_x):
                front.append(j * npts_x + i)
        front = bkd.asarray(front)

        # Back: z-index=npts_z-1 (z=+1), all x and y
        back = []
        for j in range(npts_y):
            for i in range(npts_x):
                back.append((npts_z - 1) * npts_xy + j * npts_x + i)
        back = bkd.asarray(back)

        return {
            "left": left, "right": right,
            "bottom": bottom, "top": top,
            "front": front, "back": back,
        }

    def _exact_sub0(self, x):
        """Exact solution on sub 0.

        u = (lambda/2 + 1) + (lambda/2)*x - x²
        u(-1) = lambda/2 + 1 - lambda/2 - 1 = 0 ✓
        u(1) = lambda/2 + 1 + lambda/2 - 1 = lambda ✓
        """
        lam = self.exact_lambda
        return (lam / 2.0 + 1.0) + (lam / 2.0) * x - x**2

    def _exact_sub1(self, x):
        """Exact solution on sub 1.

        Must satisfy:
        - u(-1) = lambda (interface condition)
        - u(1) = g (external BC, where g is target value + 2 since we subtract x²)
        - -u'' = 2 (forcing)

        Form: u = A + B*x - x²
        At x=-1: A - B - 1 = lambda => A - B = lambda + 1
        At x=1: A + B - 1 = g => A + B = g + 1

        Solving: 2A = lambda + g + 2 => A = (lambda + g + 2)/2
                 2B = g - lambda => B = (g - lambda)/2
        """
        lam = self.exact_lambda
        g = self.g
        A = (lam + g + 2.0) / 2.0
        B = (g - lam) / 2.0
        return A + B * x - x**2

    def test_3d_forcing_residual_at_exact(self):
        """Test 3D residual with forcing at exact solution."""
        bkd = self.bkd
        npts = self.npts

        interface_basis = LegendreInterfaceBasis2D(
            bkd, degree_y=npts - 1, degree_z=npts - 1,
            physical_bounds_y=(-1.0, 1.0), physical_bounds_z=(-1.0, 1.0)
        )
        interface = Interface2D(
            bkd, interface_id=0, subdomain_ids=(0, 1),
            basis=interface_basis, normal_direction=0
        )

        mesh0 = TransformedMesh3D(npts, npts, npts, bkd)
        basis0 = ChebyshevBasis3D(mesh0, bkd)
        mesh1 = TransformedMesh3D(npts, npts, npts, bkd)
        basis1 = ChebyshevBasis3D(mesh1, bkd)
        nodes_x0 = basis0.nodes_x()
        nodes_y0 = basis0.nodes_y()
        nodes_z0 = basis0.nodes_z()

        bounds0 = self._compute_boundary_indices_3d(npts, npts, npts)
        bounds1 = self._compute_boundary_indices_3d(npts, npts, npts)

        # Forcing: div(D*grad(u)) + f = 0, so f = -div(D*grad(u)) = -D*laplacian(u)
        # For u = ... - x², laplacian(u) = -2, so f = -D*(-2) = 2*D = 2
        f_val = 2.0 * self.D
        f0 = bkd.full((basis0.npts(),), f_val)
        f1 = bkd.full((basis1.npts(),), f_val)

        physics0 = AdvectionDiffusionReaction(
            basis0, bkd, diffusion=self.D, forcing=lambda t: f0
        )
        physics1 = AdvectionDiffusionReaction(
            basis1, bkd, diffusion=self.D, forcing=lambda t: f1
        )

        # External BCs
        left_bc0 = zero_dirichlet_bc(bkd, bounds0["left"])
        # u(1) on sub 1 = g - 2 = -2 when g=0
        right_bc1 = DirichletBC(bkd, bounds1["right"],
                                bkd.full((npts * npts,), self._exact_sub1(1.0)))

        # Y-boundary BCs (u depends only on x)
        bc_vals_y0 = bkd.zeros((npts * npts,))
        idx = 0
        for k in range(npts):
            for i in range(npts):
                bc_vals_y0[idx] = self._exact_sub0(nodes_x0[i])
                idx += 1
        bottom_bc0 = DirichletBC(bkd, bounds0["bottom"], bc_vals_y0)
        top_bc0 = DirichletBC(bkd, bounds0["top"], bc_vals_y0)

        bc_vals_z0 = bkd.zeros((npts * npts,))
        idx = 0
        for j in range(npts):
            for i in range(npts):
                bc_vals_z0[idx] = self._exact_sub0(nodes_x0[i])
                idx += 1
        front_bc0 = DirichletBC(bkd, bounds0["front"], bc_vals_z0)
        back_bc0 = DirichletBC(bkd, bounds0["back"], bc_vals_z0)

        bc_vals_y1 = bkd.zeros((npts * npts,))
        idx = 0
        for k in range(npts):
            for i in range(npts):
                bc_vals_y1[idx] = self._exact_sub1(nodes_x0[i])
                idx += 1
        bottom_bc1 = DirichletBC(bkd, bounds1["bottom"], bc_vals_y1)
        top_bc1 = DirichletBC(bkd, bounds1["top"], bc_vals_y1)

        bc_vals_z1 = bkd.zeros((npts * npts,))
        idx = 0
        for j in range(npts):
            for i in range(npts):
                bc_vals_z1[idx] = self._exact_sub1(nodes_x0[i])
                idx += 1
        front_bc1 = DirichletBC(bkd, bounds1["front"], bc_vals_z1)
        back_bc1 = DirichletBC(bkd, bounds1["back"], bc_vals_z1)

        wrapper0 = SubdomainWrapper(
            bkd, subdomain_id=0, physics=physics0,
            interfaces={0: interface},
            external_bcs=[left_bc0, bottom_bc0, top_bc0, front_bc0, back_bc0]
        )
        wrapper0.set_interface_boundary_indices(0, bounds0["right"])

        wrapper1 = SubdomainWrapper(
            bkd, subdomain_id=1, physics=physics1,
            interfaces={0: interface},
            external_bcs=[right_bc1, bottom_bc1, top_bc1, front_bc1, back_bc1]
        )
        wrapper1.set_interface_boundary_indices(0, bounds1["left"])

        interface.set_subdomain_boundary_points_2d(0, nodes_y0, nodes_z0)
        interface.set_subdomain_boundary_points_2d(1, nodes_y0, nodes_z0)

        interface_dof_offsets = bkd.asarray([0, interface.ndofs()])

        residual = DtNResidual(
            bkd,
            interfaces={0: interface},
            subdomain_solvers={0: wrapper0, 1: wrapper1},
            interface_dof_offsets=interface_dof_offsets,
        )

        ndofs = interface.ndofs()
        exact_dofs = bkd.full((ndofs,), self.exact_lambda)

        res = residual(exact_dofs)
        res_norm = float(bkd.norm(res))

        self.assertLess(res_norm, 1e-9,
                        f"3D forcing residual {res_norm} should be < 1e-9")

    def test_3d_forcing_solver_converges(self):
        """Test DtN solver converges for 3D problem with forcing."""
        bkd = self.bkd
        npts = self.npts

        interface_basis = LegendreInterfaceBasis2D(
            bkd, degree_y=npts - 1, degree_z=npts - 1,
            physical_bounds_y=(-1.0, 1.0), physical_bounds_z=(-1.0, 1.0)
        )
        interface = Interface2D(
            bkd, interface_id=0, subdomain_ids=(0, 1),
            basis=interface_basis, normal_direction=0
        )

        mesh0 = TransformedMesh3D(npts, npts, npts, bkd)
        basis0 = ChebyshevBasis3D(mesh0, bkd)
        mesh1 = TransformedMesh3D(npts, npts, npts, bkd)
        basis1 = ChebyshevBasis3D(mesh1, bkd)
        nodes_x0 = basis0.nodes_x()
        nodes_y0 = basis0.nodes_y()
        nodes_z0 = basis0.nodes_z()

        bounds0 = self._compute_boundary_indices_3d(npts, npts, npts)
        bounds1 = self._compute_boundary_indices_3d(npts, npts, npts)

        f_val = 2.0 * self.D
        f0 = bkd.full((basis0.npts(),), f_val)
        f1 = bkd.full((basis1.npts(),), f_val)

        physics0 = AdvectionDiffusionReaction(
            basis0, bkd, diffusion=self.D, forcing=lambda t: f0
        )
        physics1 = AdvectionDiffusionReaction(
            basis1, bkd, diffusion=self.D, forcing=lambda t: f1
        )

        left_bc0 = zero_dirichlet_bc(bkd, bounds0["left"])
        right_bc1 = DirichletBC(bkd, bounds1["right"],
                                bkd.full((npts * npts,), self._exact_sub1(1.0)))

        bc_vals_y0 = bkd.zeros((npts * npts,))
        idx = 0
        for k in range(npts):
            for i in range(npts):
                bc_vals_y0[idx] = self._exact_sub0(nodes_x0[i])
                idx += 1
        bottom_bc0 = DirichletBC(bkd, bounds0["bottom"], bc_vals_y0)
        top_bc0 = DirichletBC(bkd, bounds0["top"], bc_vals_y0)

        bc_vals_z0 = bkd.zeros((npts * npts,))
        idx = 0
        for j in range(npts):
            for i in range(npts):
                bc_vals_z0[idx] = self._exact_sub0(nodes_x0[i])
                idx += 1
        front_bc0 = DirichletBC(bkd, bounds0["front"], bc_vals_z0)
        back_bc0 = DirichletBC(bkd, bounds0["back"], bc_vals_z0)

        bc_vals_y1 = bkd.zeros((npts * npts,))
        idx = 0
        for k in range(npts):
            for i in range(npts):
                bc_vals_y1[idx] = self._exact_sub1(nodes_x0[i])
                idx += 1
        bottom_bc1 = DirichletBC(bkd, bounds1["bottom"], bc_vals_y1)
        top_bc1 = DirichletBC(bkd, bounds1["top"], bc_vals_y1)

        bc_vals_z1 = bkd.zeros((npts * npts,))
        idx = 0
        for j in range(npts):
            for i in range(npts):
                bc_vals_z1[idx] = self._exact_sub1(nodes_x0[i])
                idx += 1
        front_bc1 = DirichletBC(bkd, bounds1["front"], bc_vals_z1)
        back_bc1 = DirichletBC(bkd, bounds1["back"], bc_vals_z1)

        wrapper0 = SubdomainWrapper(
            bkd, subdomain_id=0, physics=physics0,
            interfaces={0: interface},
            external_bcs=[left_bc0, bottom_bc0, top_bc0, front_bc0, back_bc0]
        )
        wrapper0.set_interface_boundary_indices(0, bounds0["right"])

        wrapper1 = SubdomainWrapper(
            bkd, subdomain_id=1, physics=physics1,
            interfaces={0: interface},
            external_bcs=[right_bc1, bottom_bc1, top_bc1, front_bc1, back_bc1]
        )
        wrapper1.set_interface_boundary_indices(0, bounds1["left"])

        interface.set_subdomain_boundary_points_2d(0, nodes_y0, nodes_z0)
        interface.set_subdomain_boundary_points_2d(1, nodes_y0, nodes_z0)

        interface_dof_offsets = bkd.asarray([0, interface.ndofs()])

        residual = DtNResidual(
            bkd,
            interfaces={0: interface},
            subdomain_solvers={0: wrapper0, 1: wrapper1},
            interface_dof_offsets=interface_dof_offsets,
        )

        solver = DtNSolver(bkd, residual, max_iters=30, tol=1e-10)

        ndofs = interface.ndofs()
        result = solver.solve(bkd.full((ndofs,), 2.0))

        self.assertTrue(result.converged, "3D with forcing should converge")

        computed_vals = interface.evaluate(result.interface_dofs)
        max_error = float(bkd.max(bkd.abs(computed_vals - self.exact_lambda)))
        self.assertLess(max_error, 1e-8,
                        f"3D forcing interface error = {max_error}")


class TestTimeSteppingDtN1D(unittest.TestCase):
    """Test DtN with proper time stepping using backward Euler.

    This test demonstrates how DtN domain decomposition integrates with
    the time stepping protocols. At each time step:
    1. Set up BCs for the new time
    2. Solve the DtN problem for interface DOFs
    3. Advance to the next time step

    For diffusion with time-dependent BCs:
        du/dt = D * laplacian(u) + f

    We use a manufactured solution where:
    - The exact solution is known at all times
    - We verify that the DtN solver recovers the correct interface values
    - Flux conservation holds at each time step
    """

    def setUp(self):
        self.bkd = NumpyBkd()
        self.npts = 12
        self.D = 1.0
        self.init_time = 0.0
        self.final_time = 1.0
        self.deltat = 0.25

    def _create_transient_problem(self):
        """Create transient 1D diffusion problem."""
        bkd = self.bkd
        npts = self.npts
        D = self.D

        mms = ManufacturedSolution1DTransientForced(bkd, D)

        interface = Interface1D(
            bkd, interface_id=0, subdomain_ids=(0, 1), interface_point=0.0
        )

        # Subdomain 0
        mesh0 = TransformedMesh1D(npts, bkd)
        basis0 = ChebyshevBasis1D(mesh0, bkd)
        nodes0 = basis0.nodes()

        def forcing0(t):
            return mms.forcing(nodes0, t)

        physics0 = AdvectionDiffusionReaction(
            basis0, bkd, diffusion=D, forcing=forcing0
        )
        # External BC at x=-1 (index 0)
        left_bc0 = zero_dirichlet_bc(bkd, bkd.asarray([0]))

        wrapper0 = SubdomainWrapper(
            bkd, subdomain_id=0, physics=physics0,
            interfaces={0: interface}, external_bcs=[left_bc0]
        )
        # Interface at x=1 (index npts-1)
        wrapper0.set_interface_boundary_indices(0, bkd.asarray([npts - 1]))

        # Subdomain 1
        mesh1 = TransformedMesh1D(npts, bkd)
        basis1 = ChebyshevBasis1D(mesh1, bkd)
        nodes1 = basis1.nodes()

        def forcing1(t):
            return mms.forcing(nodes1, t)

        physics1 = AdvectionDiffusionReaction(
            basis1, bkd, diffusion=D, forcing=forcing1
        )
        # External BC at x=1 (index npts-1)
        right_bc1 = zero_dirichlet_bc(bkd, bkd.asarray([npts - 1]))

        wrapper1 = SubdomainWrapper(
            bkd, subdomain_id=1, physics=physics1,
            interfaces={0: interface}, external_bcs=[right_bc1]
        )
        # Interface at x=-1 (index 0)
        wrapper1.set_interface_boundary_indices(0, bkd.asarray([0]))

        # Setup interface interpolation
        interface.set_subdomain_boundary_points(0, bkd.asarray([1.0]))
        interface.set_subdomain_boundary_points(1, bkd.asarray([-1.0]))

        interface_dof_offsets = bkd.asarray([0, 1])

        return {
            "subdomain_solvers": {0: wrapper0, 1: wrapper1},
            "interfaces": {0: interface},
            "interface_dof_offsets": interface_dof_offsets,
            "mms": mms,
            "nodes": {0: nodes0, 1: nodes1},
        }

    def test_time_stepping_multiple_steps(self):
        """Test DtN solver over multiple time steps."""
        bkd = self.bkd
        problem = self._create_transient_problem()
        mms = problem["mms"]

        times = []
        interface_values = []
        errors = []

        time = self.init_time
        while time <= self.final_time + 1e-10:
            # Create DtN residual for current time
            residual = DtNResidual(
                bkd,
                interfaces=problem["interfaces"],
                subdomain_solvers=problem["subdomain_solvers"],
                interface_dof_offsets=problem["interface_dof_offsets"],
            )

            # Set time for time-dependent forcing
            residual.set_time(time)

            solver = DtNSolver(bkd, residual, max_iters=20, tol=1e-10)

            # Initial guess from exact solution
            exact_interface = mms.interface_value(time)
            initial_guess = bkd.asarray([exact_interface * 0.9])  # Perturb

            result = solver.solve(initial_guess)

            self.assertTrue(result.converged,
                            f"DtN should converge at t={time}")

            computed_interface = float(result.interface_dofs[0])
            error = abs(computed_interface - exact_interface)

            times.append(time)
            interface_values.append(computed_interface)
            errors.append(error)

            time += self.deltat

        # All errors should be small
        max_error = max(errors)
        self.assertLess(max_error, 1e-8,
                        f"Max interface error across time = {max_error}")

    def test_flux_conservation_over_time(self):
        """Verify flux conservation holds at multiple time steps."""
        bkd = self.bkd
        problem = self._create_transient_problem()
        mms = problem["mms"]

        for t in [0.0, 0.5, 1.0]:
            residual = DtNResidual(
                bkd,
                interfaces=problem["interfaces"],
                subdomain_solvers=problem["subdomain_solvers"],
                interface_dof_offsets=problem["interface_dof_offsets"],
            )

            # Set time for time-dependent forcing
            residual.set_time(t)

            # Solve at exact interface value
            exact_interface = mms.interface_value(t)
            exact_dofs = bkd.asarray([exact_interface])

            res = residual(exact_dofs)
            res_norm = float(bkd.norm(res))

            self.assertLess(res_norm, 1e-9,
                            f"Flux conservation residual at t={t}: {res_norm}")

    def test_solution_trajectory(self):
        """Test solution accuracy over a trajectory of time steps."""
        bkd = self.bkd
        problem = self._create_transient_problem()
        mms = problem["mms"]

        ntimes = 5
        times = bkd.linspace(self.init_time, self.final_time, ntimes)

        for ii, t in enumerate(times):
            t = float(t)

            residual = DtNResidual(
                bkd,
                interfaces=problem["interfaces"],
                subdomain_solvers=problem["subdomain_solvers"],
                interface_dof_offsets=problem["interface_dof_offsets"],
            )

            # Set time for time-dependent forcing
            residual.set_time(t)

            solver = DtNSolver(bkd, residual, max_iters=20, tol=1e-10)
            result = solver.solve(bkd.asarray([2.0]))

            self.assertTrue(result.converged, f"Failed at t={t}")

            # Check solution on each subdomain
            for sub_id in [0, 1]:
                wrapper = problem["subdomain_solvers"][sub_id]
                computed_sol = wrapper.solution()
                nodes = problem["nodes"][sub_id]
                if sub_id == 0:
                    exact_sol = mms.solution_sub0(nodes, t)
                else:
                    exact_sol = mms.solution_sub1(nodes, t)

                max_error = float(bkd.max(bkd.abs(computed_sol - exact_sol)))
                self.assertLess(max_error, 1e-7,
                                f"Sub {sub_id} at t={t}: error = {max_error}")


# NOTE: Shallow wave DtN tests were removed because shallow water equations
# are hyperbolic, not elliptic. The DtN domain decomposition method is designed
# for elliptic/parabolic problems where we solve a steady-state subproblem.
# Shallow water equations don't have a meaningful steady state for Newton
# iteration from zero initial guess. For shallow water domain decomposition,
# use a time-stepping approach with explicit flux coupling instead of DtN.


if __name__ == "__main__":
    unittest.main()
