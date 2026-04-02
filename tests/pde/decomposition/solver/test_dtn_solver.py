"""Tests for DtN solver using manufactured solutions.

Tests verify:
1. Residual is zero (to tolerance) at exact solution
2. Solver converges to exact solution from non-exact initial guess
3. Jacobian matches finite differences via DerivativeChecker

Note: All tests use the reference domain [-1, 1] where the Chebyshev
basis operates directly, avoiding coordinate transformation issues.
"""

from typing import Generic

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.pde.collocation.basis import (
    ChebyshevBasis1D,
    ChebyshevBasis2D,
    ChebyshevBasis3D,
)
from pyapprox.pde.collocation.boundary import (
    DirichletBC,
    zero_dirichlet_bc,
)
from pyapprox.pde.collocation.mesh import (
    TransformedMesh1D,
    TransformedMesh2D,
    TransformedMesh3D,
)
from pyapprox.pde.collocation.physics.advection_diffusion import (
    create_steady_diffusion,
)
from pyapprox.pde.decomposition.interface import (
    Interface,
    Interface1D,
    Interface2D,
    LegendreInterfaceBasis1D,
    LegendreInterfaceBasis2D,
)
from pyapprox.pde.decomposition.solver import (
    DtNJacobian,
    DtNResidual,
    DtNSolver,
)
from pyapprox.pde.decomposition.subdomain import SubdomainWrapper
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend


class DtNResidualDerivativeWrapper(Generic[Array]):
    """Wrap DtN residual for DerivativeChecker compatibility.

    Adapts DtNResidual interface to FunctionWithJacobianProtocol interface.

    Parameters
    ----------
    residual : DtNResidual[Array]
        DtN residual to wrap.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, residual: DtNResidual, bkd: Backend[Array]):
        self._residual = residual
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return number of input variables (interface DOFs)."""
        return self._residual.total_dofs()

    def nqoi(self) -> int:
        """Return number of output quantities (residual components)."""
        return self._residual.total_dofs()

    def __call__(self, samples: Array) -> Array:
        """Evaluate residual at samples.

        Parameters
        ----------
        samples : Array
            Samples of shape (nvars, nsamples) or (nvars,).

        Returns
        -------
        Array
            Values of shape (nqoi, nsamples) or (nqoi, 1).
        """
        if samples.ndim == 1:
            interface_dofs = samples
        else:
            interface_dofs = samples[:, 0]
        residual = self._residual(interface_dofs)
        return residual[:, None]

    def jacobian(self, sample: Array) -> Array:
        """Compute Jacobian at sample.

        Parameters
        ----------
        sample : Array
            Sample of shape (nvars, 1) or (nvars,).

        Returns
        -------
        Array
            Jacobian of shape (nqoi, nvars).
        """
        if sample.ndim == 2:
            interface_dofs = sample[:, 0]
        else:
            interface_dofs = sample
        jacobian_computer = DtNJacobian(self._bkd, self._residual, epsilon=1e-8)
        return jacobian_computer(interface_dofs)


class TestDtNSolver1DSimple:
    """Test DtN solver for simple 1D Poisson on reference domain.

    Problem: -u'' = f on [-1, 1]
    BCs: u(-1) = 0, u(1) = 0
    Manufactured solution: u(x) = (1-x^2)  (parabola touching zero at boundaries)
    Forcing: f = -u'' = 2

    Domain decomposition:
    - Subdomain 0: [-1, 0]
    - Subdomain 1: [0, 1]
    - Interface at x = 0

    At interface: u(0) = 1, u'(0) = 0 (from both sides due to symmetry)
    """

    def setup_method(self):
        self.bkd = NumpyBkd()
        self.npts = 12

    def _create_problem(self, bkd) :
        """Create simple 1D problem on reference domain."""
        bkd = self.bkd
        npts = self.npts

        # Manufactured solution: u(x) = 1 - x^2
        # u(-1) = 0, u(1) = 0, u(0) = 1
        # u'(x) = -2x, u'(0) = 0
        # u''(x) = -2, so -u'' = 2

        interface = Interface1D(
            bkd,
            interface_id=0,
            subdomain_ids=(0, 1),
            interface_point=0.0,
        )

        # Subdomain 0: Chebyshev on [-1, 1], but we treat it as [-1, 0]
        # Since ChebyshevBasis1D uses [-1, 1], we need separate instances
        # For simplicity, let's use the full domain approach:
        # Solve on each half using Chebyshev nodes naturally

        # For subdomain 0, we want to solve on the left half
        # The Chebyshev nodes are on [-1, 1], at index 0 is x=1, at index n-1 is x=-1
        # So for left subdomain, right boundary (interface at x=0) needs special
        # handling

        # Actually, let's use a simpler approach: use the full domain [-1, 1]
        # and just have Dirichlet BCs at the boundaries

        # Subdomain 0: Use full Chebyshev grid, but set u(-1)=0 and u(0)=lambda
        mesh0 = TransformedMesh1D(npts, bkd)
        basis0 = ChebyshevBasis1D(mesh0, bkd)
        basis0.nodes()  # x=1 at index 0, x=-1 at index npts-1

        # Forcing: f = 2 (constant)
        forcing0 = bkd.full((npts,), 2.0)
        physics0 = create_steady_diffusion(
            basis0, bkd, diffusion=1.0, forcing=lambda t: forcing0
        )

        # For subdomain 0 representing [-1, 0]:
        # - Left BC at x=-1 (index npts-1): u = 0
        # - Right BC at x=0 (interface): u = lambda (set by DtN)
        # But Chebyshev nodes don't naturally split at x=0

        # Instead, let's use a different approach:
        # Subdomain 0 uses the full [-1, 1] Chebyshev grid
        # BC at x=-1: u = 0
        # BC at x=1: this is the interface (we'll call it "right")

        # Wait, this doesn't decompose the domain correctly.
        # Let me think about this differently.

        # The simplest test: two identical subdomains, each on [-1, 1]
        # Interface is at x=1 of subdomain 0 = x=-1 of subdomain 1

        # Subdomain 0: [-1, 1] with u(-1) = 0 (external), u(1) = lambda (interface)
        # Exact on this subdomain: u(x) = 1 - x^2, u(1) = 0
        # But we want u(1) = lambda, so we need to adjust...

        # Let me use a linear exact solution instead:
        # u(x) = (1-x)/2 on [-1, 1], u(-1) = 1, u(1) = 0
        # -u'' = 0, so forcing = 0

        # Actually, simplest: solve Laplace (-u'' = 0) with linear solution
        # u(x) = a*x + b, u(-1) = 0 => -a + b = 0, u(1) = 1 => a + b = 1
        # => b = a, 2a = 1, a = 0.5, b = 0.5
        # u(x) = 0.5*x + 0.5, u(-1) = 0, u(1) = 1, u(0) = 0.5

        # Domain decomposition:
        # Subdomain 0: [-1, 1] with u(-1) = 0 (left BC), u(1) = lambda (interface)
        # Subdomain 1: [-1, 1] with u(-1) = lambda (interface), u(1) = 1 (right BC)

        # For exact solution lambda = u(interface) = ?
        # If we set interface at the "right" of subdomain 0 and "left" of subdomain 1,
        # we need lambda such that flux conservation holds.

        # Let's simplify even more: use Laplace equation with known interface value
        # u(x) = 0.5*(1 + x) on [-1, 1], so u(-1) = 0, u(1) = 1, u(0) = 0.5

        # Subdomain 0: solves -u'' = 0 on [-1, 1] with u(-1) = 0, u(1) = lambda
        # Solution: u = lambda/2 * (x + 1)
        # u'(1) = lambda/2

        # Subdomain 1: solves -u'' = 0 on [-1, 1] with u(-1) = lambda, u(1) = 1
        # Solution: u = (lambda + 1)/2 + (1 - lambda)/2 * x
        # u'(-1) = (1 - lambda)/2

        # Flux conservation at interface:
        # From subdomain 0: flux = u'(1) * n0 = (lambda/2) * (+1) = lambda/2
        # From subdomain 1: flux = u'(-1) * n1 = ((1-lambda)/2) * (-1) = -(1-lambda)/2
        # Total flux: lambda/2 - (1-lambda)/2 = lambda/2 - 1/2 + lambda/2 = lambda - 1/2
        # = 0
        # => lambda = 0.5 ✓

        # Let's implement this

        # Subdomain 0
        forcing0 = bkd.zeros((npts,))  # Laplace equation
        physics0 = create_steady_diffusion(
            basis0, bkd, diffusion=1.0, forcing=lambda t: forcing0
        )
        # BC at x=-1 (index npts-1): u = 0
        left_bc0 = zero_dirichlet_bc(bkd, bkd.asarray([npts - 1]))

        wrapper0 = SubdomainWrapper(
            bkd,
            subdomain_id=0,
            physics=physics0,
            interfaces={0: interface},
            external_bcs=[left_bc0],
        )
        # Interface at x=1 (index 0)
        wrapper0.set_interface_boundary_indices(0, bkd.asarray([0]))

        # Subdomain 1
        mesh1 = TransformedMesh1D(npts, bkd)
        basis1 = ChebyshevBasis1D(mesh1, bkd)
        forcing1 = bkd.zeros((npts,))
        physics1 = create_steady_diffusion(
            basis1, bkd, diffusion=1.0, forcing=lambda t: forcing1
        )
        # BC at x=1 (index 0): u = 1
        right_bc1 = DirichletBC(bkd, bkd.asarray([0]), bkd.asarray([1.0]))

        wrapper1 = SubdomainWrapper(
            bkd,
            subdomain_id=1,
            physics=physics1,
            interfaces={0: interface},
            external_bcs=[right_bc1],
        )
        # Interface at x=-1 (index npts-1)
        wrapper1.set_interface_boundary_indices(0, bkd.asarray([npts - 1]))

        # Set up interface interpolation (single point)
        # Interface is between x=1 of subdomain 0 and x=-1 of subdomain 1
        # Both are at the same "logical" interface point
        interface.set_subdomain_boundary_points(0, bkd.asarray([1.0]))
        interface.set_subdomain_boundary_points(1, bkd.asarray([-1.0]))

        interface_dof_offsets = bkd.asarray([0, 1])

        return {
            "subdomain_solvers": {0: wrapper0, 1: wrapper1},
            "interfaces": {0: interface},
            "interface_dof_offsets": interface_dof_offsets,
            "exact_interface_value": 0.5,
        }

    def test_residual_zero_at_exact_solution(self, bkd):
        """Test residual is zero at exact interface solution."""
        bkd = self.bkd
        problem = self._create_problem(bkd)

        residual = DtNResidual(
            bkd,
            interfaces=problem["interfaces"],
            subdomain_solvers=problem["subdomain_solvers"],
            interface_dof_offsets=problem["interface_dof_offsets"],
        )

        exact_dofs = bkd.asarray([problem["exact_interface_value"]])
        res = residual(exact_dofs)
        res_norm = float(bkd.norm(res))

        assert res_norm < 1e-10

    def test_solver_converges_from_wrong_initial_guess(self, bkd):
        """Test solver converges from non-exact initial guess."""
        bkd = self.bkd
        problem = self._create_problem(bkd)

        residual = DtNResidual(
            bkd,
            interfaces=problem["interfaces"],
            subdomain_solvers=problem["subdomain_solvers"],
            interface_dof_offsets=problem["interface_dof_offsets"],
        )

        solver = DtNSolver(bkd, residual, max_iters=20, tol=1e-10)

        # Start from wrong initial guess
        initial_guess = bkd.asarray([0.0])  # Exact is 0.5

        result = solver.solve(initial_guess)

        assert result.converged, "Solver should converge"

        computed = float(result.interface_dofs[0])
        exact = problem["exact_interface_value"]
        assert abs(computed - exact) < 10**(-8)


class TestDtNSolver1DWithForcing:
    """Test DtN solver with non-zero forcing.

    Problem: -u'' = 2 on [-1, 1] (each subdomain)
    This gives parabolic solutions.

    Setup:
    - Subdomain 0: [-1, 1] with u(-1) = 0, u(1) = lambda
    - Subdomain 1: [-1, 1] with u(-1) = lambda, u(1) = 0

    Exact solution on subdomain 0: u = A + Bx - x^2
    BC: u(-1) = A - B - 1 = 0, u(1) = A + B - 1 = lambda
    => A - B = 1, A + B = lambda + 1
    => 2A = lambda + 2, A = (lambda + 2)/2
    => 2B = lambda, B = lambda/2
    u = (lambda + 2)/2 + (lambda/2)x - x^2
    u'(1) = lambda/2 - 2

    Exact solution on subdomain 1: u = C + Dx - x^2
    BC: u(-1) = C - D - 1 = lambda, u(1) = C + D - 1 = 0
    => C - D = lambda + 1, C + D = 1
    => 2C = lambda + 2, C = (lambda + 2)/2
    => 2D = 1 - lambda - 1 = -lambda, D = -lambda/2
    u = (lambda + 2)/2 - (lambda/2)x - x^2
    u'(-1) = -lambda/2 + 2

    Flux conservation:
    flux_0 = u'(1) = lambda/2 - 2 (normal +1)
    flux_1 = u'(-1) = -lambda/2 + 2 (normal -1), so contribution = -(-lambda/2 + 2) =
    lambda/2 - 2
    Total = (lambda/2 - 2) + (lambda/2 - 2) = lambda - 4 = 0
    => lambda = 4

    At lambda = 4:
    Subdomain 0: u = 3 + 2x - x^2, u(-1) = 3 - 2 - 1 = 0 ✓, u(1) = 3 + 2 - 1 = 4 ✓
    Subdomain 1: u = 3 - 2x - x^2, u(-1) = 3 + 2 - 1 = 4 ✓, u(1) = 3 - 2 - 1 = 0 ✓
    """

    def setup_method(self):
        self.bkd = NumpyBkd()
        self.npts = 12

    def _create_problem(self, bkd) :
        """Create problem with forcing."""
        bkd = self.bkd
        npts = self.npts

        interface = Interface1D(
            bkd, interface_id=0, subdomain_ids=(0, 1), interface_point=0.0
        )

        # Subdomain 0: -u'' = 2, u(-1) = 0, u(1) = lambda
        mesh0 = TransformedMesh1D(npts, bkd)
        basis0 = ChebyshevBasis1D(mesh0, bkd)
        forcing0 = bkd.full((npts,), 2.0)
        physics0 = create_steady_diffusion(
            basis0, bkd, diffusion=1.0, forcing=lambda t: forcing0
        )
        left_bc0 = zero_dirichlet_bc(bkd, bkd.asarray([npts - 1]))  # u(-1) = 0

        wrapper0 = SubdomainWrapper(
            bkd,
            subdomain_id=0,
            physics=physics0,
            interfaces={0: interface},
            external_bcs=[left_bc0],
        )
        wrapper0.set_interface_boundary_indices(0, bkd.asarray([0]))  # interface at x=1

        # Subdomain 1: -u'' = 2, u(-1) = lambda, u(1) = 0
        mesh1 = TransformedMesh1D(npts, bkd)
        basis1 = ChebyshevBasis1D(mesh1, bkd)
        forcing1 = bkd.full((npts,), 2.0)
        physics1 = create_steady_diffusion(
            basis1, bkd, diffusion=1.0, forcing=lambda t: forcing1
        )
        right_bc1 = zero_dirichlet_bc(bkd, bkd.asarray([0]))  # u(1) = 0

        wrapper1 = SubdomainWrapper(
            bkd,
            subdomain_id=1,
            physics=physics1,
            interfaces={0: interface},
            external_bcs=[right_bc1],
        )
        wrapper1.set_interface_boundary_indices(
            0, bkd.asarray([npts - 1])
        )  # interface at x=-1

        interface.set_subdomain_boundary_points(0, bkd.asarray([1.0]))
        interface.set_subdomain_boundary_points(1, bkd.asarray([-1.0]))

        interface_dof_offsets = bkd.asarray([0, 1])

        return {
            "subdomain_solvers": {0: wrapper0, 1: wrapper1},
            "interfaces": {0: interface},
            "interface_dof_offsets": interface_dof_offsets,
            "exact_interface_value": 4.0,
        }

    def test_residual_zero_at_exact_solution(self, bkd):
        """Test residual is zero at exact solution with forcing."""
        bkd = self.bkd
        problem = self._create_problem(bkd)

        residual = DtNResidual(
            bkd,
            interfaces=problem["interfaces"],
            subdomain_solvers=problem["subdomain_solvers"],
            interface_dof_offsets=problem["interface_dof_offsets"],
        )

        exact_dofs = bkd.asarray([problem["exact_interface_value"]])
        res = residual(exact_dofs)
        res_norm = float(bkd.norm(res))

        assert res_norm < 1e-10

    def test_solver_finds_correct_interface_value(self, bkd):
        """Test solver finds correct non-trivial interface value."""
        bkd = self.bkd
        problem = self._create_problem(bkd)

        residual = DtNResidual(
            bkd,
            interfaces=problem["interfaces"],
            subdomain_solvers=problem["subdomain_solvers"],
            interface_dof_offsets=problem["interface_dof_offsets"],
        )

        solver = DtNSolver(bkd, residual, max_iters=20, tol=1e-10)

        # Start from zero (exact is 4)
        result = solver.solve(bkd.asarray([0.0]))

        assert result.converged
        computed = float(result.interface_dofs[0])
        exact = problem["exact_interface_value"]
        assert abs(computed - exact) < 10**(-8)


class TestDtNSolverThreeSubdomains:
    """Test DtN solver with three subdomains (two interfaces).

    Setup with Laplace equation:
    - Subdomain 0: u(-1) = 0, u(1) = lambda0
    - Subdomain 1: u(-1) = lambda0, u(1) = lambda1
    - Subdomain 2: u(-1) = lambda1, u(1) = 1

    Linear solutions:
    Subdomain 0: u = lambda0/2 * (x + 1), u'(1) = lambda0/2
    Subdomain 1: u = (lambda0 + lambda1)/2 + (lambda1 - lambda0)/2 * x
                 u'(-1) = (lambda1 - lambda0)/2, u'(1) = (lambda1 - lambda0)/2
    Subdomain 2: u = (lambda1 + 1)/2 + (1 - lambda1)/2 * x, u'(-1) = (1 - lambda1)/2

    Flux conservation at interface 0:
    lambda0/2 + (-(lambda1 - lambda0)/2) = 0
    lambda0/2 - (lambda1 - lambda0)/2 = 0
    lambda0 - (lambda1 - lambda0) = 0
    2*lambda0 - lambda1 = 0

    Flux conservation at interface 1:
    (lambda1 - lambda0)/2 + (-(1 - lambda1)/2) = 0
    (lambda1 - lambda0)/2 - (1 - lambda1)/2 = 0
    lambda1 - lambda0 - 1 + lambda1 = 0
    2*lambda1 - lambda0 = 1

    System:
    2*lambda0 - lambda1 = 0
    -lambda0 + 2*lambda1 = 1

    From first: lambda1 = 2*lambda0
    Substitute: -lambda0 + 4*lambda0 = 1 => 3*lambda0 = 1 => lambda0 = 1/3
    lambda1 = 2/3
    """

    def setup_method(self):
        self.bkd = NumpyBkd()
        self.npts = 10

    def _create_problem(self, bkd) :
        """Create 3-subdomain problem."""
        bkd = self.bkd
        npts = self.npts

        interface0 = Interface1D(
            bkd, interface_id=0, subdomain_ids=(0, 1), interface_point=0.0
        )
        interface1 = Interface1D(
            bkd, interface_id=1, subdomain_ids=(1, 2), interface_point=0.0
        )

        # Subdomain 0
        mesh0 = TransformedMesh1D(npts, bkd)
        basis0 = ChebyshevBasis1D(mesh0, bkd)
        forcing0 = bkd.zeros((npts,))
        physics0 = create_steady_diffusion(
            basis0, bkd, diffusion=1.0, forcing=lambda t: forcing0
        )
        left_bc0 = zero_dirichlet_bc(bkd, bkd.asarray([npts - 1]))

        wrapper0 = SubdomainWrapper(
            bkd,
            subdomain_id=0,
            physics=physics0,
            interfaces={0: interface0},
            external_bcs=[left_bc0],
        )
        wrapper0.set_interface_boundary_indices(0, bkd.asarray([0]))

        # Subdomain 1 (no external BCs)
        mesh1 = TransformedMesh1D(npts, bkd)
        basis1 = ChebyshevBasis1D(mesh1, bkd)
        forcing1 = bkd.zeros((npts,))
        physics1 = create_steady_diffusion(
            basis1, bkd, diffusion=1.0, forcing=lambda t: forcing1
        )

        wrapper1 = SubdomainWrapper(
            bkd,
            subdomain_id=1,
            physics=physics1,
            interfaces={0: interface0, 1: interface1},
            external_bcs=[],
        )
        wrapper1.set_interface_boundary_indices(0, bkd.asarray([npts - 1]))
        wrapper1.set_interface_boundary_indices(1, bkd.asarray([0]))

        # Subdomain 2
        mesh2 = TransformedMesh1D(npts, bkd)
        basis2 = ChebyshevBasis1D(mesh2, bkd)
        forcing2 = bkd.zeros((npts,))
        physics2 = create_steady_diffusion(
            basis2, bkd, diffusion=1.0, forcing=lambda t: forcing2
        )
        right_bc2 = DirichletBC(bkd, bkd.asarray([0]), bkd.asarray([1.0]))

        wrapper2 = SubdomainWrapper(
            bkd,
            subdomain_id=2,
            physics=physics2,
            interfaces={1: interface1},
            external_bcs=[right_bc2],
        )
        wrapper2.set_interface_boundary_indices(1, bkd.asarray([npts - 1]))

        # Interface interpolation
        interface0.set_subdomain_boundary_points(0, bkd.asarray([1.0]))
        interface0.set_subdomain_boundary_points(1, bkd.asarray([-1.0]))
        interface1.set_subdomain_boundary_points(1, bkd.asarray([1.0]))
        interface1.set_subdomain_boundary_points(2, bkd.asarray([-1.0]))

        interface_dof_offsets = bkd.asarray([0, 1, 2])

        return {
            "subdomain_solvers": {0: wrapper0, 1: wrapper1, 2: wrapper2},
            "interfaces": {0: interface0, 1: interface1},
            "interface_dof_offsets": interface_dof_offsets,
            "exact_interface_values": [1.0 / 3.0, 2.0 / 3.0],
        }

    def test_residual_zero_at_exact_solution(self, bkd):
        """Test residual is zero at exact solution with multiple interfaces."""
        bkd = self.bkd
        problem = self._create_problem(bkd)

        residual = DtNResidual(
            bkd,
            interfaces=problem["interfaces"],
            subdomain_solvers=problem["subdomain_solvers"],
            interface_dof_offsets=problem["interface_dof_offsets"],
        )

        exact_dofs = bkd.asarray(problem["exact_interface_values"])
        res = residual(exact_dofs)
        res_norm = float(bkd.norm(res))

        assert res_norm < 1e-10

    def test_solver_converges_multiple_interfaces(self, bkd):
        """Test solver converges with multiple interface DOFs."""
        bkd = self.bkd
        problem = self._create_problem(bkd)

        residual = DtNResidual(
            bkd,
            interfaces=problem["interfaces"],
            subdomain_solvers=problem["subdomain_solvers"],
            interface_dof_offsets=problem["interface_dof_offsets"],
        )

        solver = DtNSolver(bkd, residual, max_iters=30, tol=1e-10)

        # Initial guess
        result = solver.solve(bkd.asarray([0.0, 0.0]))

        assert result.converged

        exact_values = problem["exact_interface_values"]
        for i, exact in enumerate(exact_values):
            computed = float(result.interface_dofs[i])
            assert abs(computed - exact) < 10**(-6)


class TestDtNJacobian:
    """Test DtN Jacobian computation with DerivativeChecker.

    Uses the same patterns as pyapprox physics tests for derivative checking.
    """

    def setup_method(self):
        self.bkd = NumpyBkd()

    def _create_two_subdomain_problem(self, bkd, npts=8, forcing_value=1.0) :
        """Create a simple two-subdomain problem for testing."""
        bkd = self.bkd

        interface = Interface1D(
            bkd, interface_id=0, subdomain_ids=(0, 1), interface_point=0.0
        )

        mesh0 = TransformedMesh1D(npts, bkd)
        basis0 = ChebyshevBasis1D(mesh0, bkd)
        forcing0 = bkd.full((npts,), forcing_value)
        physics0 = create_steady_diffusion(
            basis0, bkd, diffusion=1.0, forcing=lambda t: forcing0
        )
        left_bc0 = zero_dirichlet_bc(bkd, bkd.asarray([npts - 1]))

        wrapper0 = SubdomainWrapper(
            bkd,
            subdomain_id=0,
            physics=physics0,
            interfaces={0: interface},
            external_bcs=[left_bc0],
        )
        wrapper0.set_interface_boundary_indices(0, bkd.asarray([0]))

        mesh1 = TransformedMesh1D(npts, bkd)
        basis1 = ChebyshevBasis1D(mesh1, bkd)
        forcing1 = bkd.full((npts,), forcing_value)
        physics1 = create_steady_diffusion(
            basis1, bkd, diffusion=1.0, forcing=lambda t: forcing1
        )
        right_bc1 = zero_dirichlet_bc(bkd, bkd.asarray([0]))

        wrapper1 = SubdomainWrapper(
            bkd,
            subdomain_id=1,
            physics=physics1,
            interfaces={0: interface},
            external_bcs=[right_bc1],
        )
        wrapper1.set_interface_boundary_indices(0, bkd.asarray([npts - 1]))

        interface.set_subdomain_boundary_points(0, bkd.asarray([1.0]))
        interface.set_subdomain_boundary_points(1, bkd.asarray([-1.0]))

        interface_dof_offsets = bkd.asarray([0, 1])

        residual = DtNResidual(
            bkd,
            interfaces={0: interface},
            subdomain_solvers={0: wrapper0, 1: wrapper1},
            interface_dof_offsets=interface_dof_offsets,
        )

        return residual

    def _create_three_subdomain_problem(self, bkd, npts=8) :
        """Create a three-subdomain problem for testing (2 interface DOFs)."""
        bkd = self.bkd

        interface0 = Interface1D(
            bkd, interface_id=0, subdomain_ids=(0, 1), interface_point=0.0
        )
        interface1 = Interface1D(
            bkd, interface_id=1, subdomain_ids=(1, 2), interface_point=0.0
        )

        # Subdomain 0
        mesh0 = TransformedMesh1D(npts, bkd)
        basis0 = ChebyshevBasis1D(mesh0, bkd)
        forcing0 = bkd.zeros((npts,))
        physics0 = create_steady_diffusion(
            basis0, bkd, diffusion=1.0, forcing=lambda t: forcing0
        )
        left_bc0 = zero_dirichlet_bc(bkd, bkd.asarray([npts - 1]))

        wrapper0 = SubdomainWrapper(
            bkd,
            subdomain_id=0,
            physics=physics0,
            interfaces={0: interface0},
            external_bcs=[left_bc0],
        )
        wrapper0.set_interface_boundary_indices(0, bkd.asarray([0]))

        # Subdomain 1 (middle, no external BCs)
        mesh1 = TransformedMesh1D(npts, bkd)
        basis1 = ChebyshevBasis1D(mesh1, bkd)
        forcing1 = bkd.zeros((npts,))
        physics1 = create_steady_diffusion(
            basis1, bkd, diffusion=1.0, forcing=lambda t: forcing1
        )

        wrapper1 = SubdomainWrapper(
            bkd,
            subdomain_id=1,
            physics=physics1,
            interfaces={0: interface0, 1: interface1},
            external_bcs=[],
        )
        wrapper1.set_interface_boundary_indices(0, bkd.asarray([npts - 1]))
        wrapper1.set_interface_boundary_indices(1, bkd.asarray([0]))

        # Subdomain 2
        mesh2 = TransformedMesh1D(npts, bkd)
        basis2 = ChebyshevBasis1D(mesh2, bkd)
        forcing2 = bkd.zeros((npts,))
        physics2 = create_steady_diffusion(
            basis2, bkd, diffusion=1.0, forcing=lambda t: forcing2
        )
        right_bc2 = DirichletBC(bkd, bkd.asarray([0]), bkd.asarray([1.0]))

        wrapper2 = SubdomainWrapper(
            bkd,
            subdomain_id=2,
            physics=physics2,
            interfaces={1: interface1},
            external_bcs=[right_bc2],
        )
        wrapper2.set_interface_boundary_indices(1, bkd.asarray([npts - 1]))

        interface0.set_subdomain_boundary_points(0, bkd.asarray([1.0]))
        interface0.set_subdomain_boundary_points(1, bkd.asarray([-1.0]))
        interface1.set_subdomain_boundary_points(1, bkd.asarray([1.0]))
        interface1.set_subdomain_boundary_points(2, bkd.asarray([-1.0]))

        interface_dof_offsets = bkd.asarray([0, 1, 2])

        residual = DtNResidual(
            bkd,
            interfaces={0: interface0, 1: interface1},
            subdomain_solvers={0: wrapper0, 1: wrapper1, 2: wrapper2},
            interface_dof_offsets=interface_dof_offsets,
        )

        return residual

    def test_jacobian_derivative_checker_1dof(self, bkd):
        """Test Jacobian using DerivativeChecker with 1 interface DOF."""
        bkd = self.bkd
        residual = self._create_two_subdomain_problem(bkd)

        # Wrap for DerivativeChecker
        wrapper = DtNResidualDerivativeWrapper(residual, bkd)

        # Test at a point
        test_dofs = bkd.asarray([0.5])
        sample = test_dofs[:, None]  # Shape (nvars, 1)

        # Check derivatives using DerivativeChecker
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(sample, verbosity=0)

        # Check that minimum error is small
        min_error = float(bkd.min(errors[0]))
        assert min_error < 1e-5

    def test_jacobian_derivative_checker_2dof(self, bkd):
        """Test Jacobian using DerivativeChecker with 2 interface DOFs."""
        bkd = self.bkd
        residual = self._create_three_subdomain_problem(bkd)

        wrapper = DtNResidualDerivativeWrapper(residual, bkd)

        # Test at various points
        test_points = [
            bkd.asarray([0.3, 0.6]),
            bkd.asarray([0.1, 0.9]),
            bkd.asarray([0.5, 0.5]),
        ]

        for test_dofs in test_points:
            sample = test_dofs[:, None]
            checker = DerivativeChecker(wrapper)
            errors = checker.check_derivatives(sample, verbosity=0)

            min_error = float(bkd.min(errors[0]))
            assert min_error < 1e-5

    def test_jacobian_error_ratio(self, bkd):
        """Test error ratio from DerivativeChecker is small.

        When the Jacobian is correct, the error ratio (min_error/max_error)
        should be small, indicating that errors decrease with epsilon.
        """
        bkd = self.bkd
        residual = self._create_two_subdomain_problem(bkd)

        wrapper = DtNResidualDerivativeWrapper(residual, bkd)
        test_dofs = bkd.asarray([0.5])
        sample = test_dofs[:, None]

        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(sample, verbosity=0)

        # The error_ratio method computes min/max
        error_ratio = checker.error_ratio(errors[0])
        assert error_ratio < 1e-3

    def test_jacobian_symmetry_for_symmetric_problem(self, bkd):
        """Test Jacobian is symmetric for symmetric problems.

        For the Laplace equation with symmetric BCs and geometry,
        the Schur complement (and hence DtN Jacobian) should be symmetric.
        """
        bkd = self.bkd
        residual = self._create_three_subdomain_problem(bkd)

        # Test at a point
        test_dofs = bkd.asarray([0.3, 0.7])

        jacobian = DtNJacobian(bkd, residual, epsilon=1e-7)
        J = jacobian(test_dofs)

        # Check symmetry
        # J should be approximately symmetric for this problem
        # Note: DtN Jacobian is S = A_ΓΓ - A_ΓI A_II^{-1} A_IΓ
        # which is SPD for elliptic problems
        # Tolerance accounts for finite difference epsilon effects
        sym_error = float(bkd.norm(J - J.T))
        J_norm = float(bkd.norm(J))
        relative_sym_error = sym_error / J_norm if J_norm > 1e-14 else sym_error
        assert relative_sym_error < 1e-5


class TestDtNSolver2D:
    """Test DtN solver for 2D Poisson with domain decomposition.

    Problem: -Laplacian(u) = f on [-1, 1] x [-1, 1]
    Split into left subdomain [-1, 0] x [-1, 1] and right [0, 1] x [-1, 1]
    Interface at x = 0.

    Using linear solution u(x, y) = x + 0.5 which satisfies Laplace equation
    with zero forcing. This gives interface values u(0, y) = 0.5.
    """

    def setup_method(self):
        self.bkd = NumpyBkd()

    def _compute_boundary_indices_2d(self, bkd, npts_x, npts_y) :
        """Compute 2D boundary indices (x varies fastest)."""
        bkd = self.bkd
        # Left: x=0, all y
        left = bkd.asarray([j * npts_x for j in range(npts_y)])
        # Right: x=npts_x-1, all y
        right = bkd.asarray([j * npts_x + (npts_x - 1) for j in range(npts_y)])
        # Bottom: y=0, all x
        bottom = bkd.asarray(list(range(npts_x)))
        # Top: y=npts_y-1, all x
        top = bkd.asarray([(npts_y - 1) * npts_x + i for i in range(npts_x)])
        return {"left": left, "right": right, "bottom": bottom, "top": top}

    def _create_2d_problem_linear_solution(self, npts_x=6, npts_y=6):
        """Create 2D problem with linear manufactured solution.

        Simplified problem working in reference domain [-1, 1] x [-1, 1]:
        - Left subdomain: reference domain, physical domain [-1, 0]
        - Right subdomain: reference domain, physical domain [0, 1]

        Solution: u(x_phys, y) = x_phys (linear in physical x)
        This satisfies Laplace equation: -Laplacian(u) = 0
        Interface at x_phys = 0 has u = 0 for all y.
        """
        bkd = self.bkd

        # Interface: vertical line at physical x=0
        interface_basis = LegendreInterfaceBasis1D(
            bkd, degree=npts_y, physical_bounds=(-1.0, 1.0)
        )
        interface = Interface(
            bkd,
            interface_id=0,
            subdomain_ids=(0, 1),
            basis=interface_basis,
            normal_direction=0,  # x-direction
            ambient_dim=2,  # 2D problem
        )

        # Left subdomain: physical [-1, 0] mapped to reference [-1, 1]
        # x_phys = 0.5 * (x_ref - 1) => at x_ref=-1, x_phys=-1; at x_ref=1, x_phys=0
        mesh0 = TransformedMesh2D(npts_x, npts_y, bkd)
        basis0 = ChebyshevBasis2D(mesh0, bkd)
        forcing0 = bkd.zeros((basis0.npts(),))
        physics0 = create_steady_diffusion(
            basis0, bkd, diffusion=1.0, forcing=lambda t: forcing0
        )

        bounds0 = self._compute_boundary_indices_2d(bkd, npts_x, npts_y)
        nodes_x0 = basis0.nodes_x()
        nodes_y0 = basis0.nodes_y()

        # Physical x coords for left subdomain
        x_phys0 = 0.5 * (nodes_x0 - 1)  # maps [-1,1] to [-1, 0]

        # BCs for u = x_phys
        left_bc0 = DirichletBC(bkd, bounds0["left"], bkd.full((npts_y,), -1.0))
        bottom_bc0 = DirichletBC(bkd, bounds0["bottom"], x_phys0)
        top_bc0 = DirichletBC(bkd, bounds0["top"], x_phys0)

        wrapper0 = SubdomainWrapper(
            bkd,
            subdomain_id=0,
            physics=physics0,
            interfaces={0: interface},
            external_bcs=[left_bc0, bottom_bc0, top_bc0],
        )
        wrapper0.set_interface_boundary_indices(0, bounds0["right"])

        # Right subdomain: physical [0, 1] mapped to reference [-1, 1]
        # x_phys = 0.5 * (x_ref + 1) => at x_ref=-1, x_phys=0; at x_ref=1, x_phys=1
        mesh1 = TransformedMesh2D(npts_x, npts_y, bkd)
        basis1 = ChebyshevBasis2D(mesh1, bkd)
        forcing1 = bkd.zeros((basis1.npts(),))
        physics1 = create_steady_diffusion(
            basis1, bkd, diffusion=1.0, forcing=lambda t: forcing1
        )

        bounds1 = self._compute_boundary_indices_2d(bkd, npts_x, npts_y)
        nodes_x1 = basis1.nodes_x()

        # Physical x coords for right subdomain
        x_phys1 = 0.5 * (nodes_x1 + 1)  # maps [-1,1] to [0, 1]

        right_bc1 = DirichletBC(bkd, bounds1["right"], bkd.full((npts_y,), 1.0))
        bottom_bc1 = DirichletBC(bkd, bounds1["bottom"], x_phys1)
        top_bc1 = DirichletBC(bkd, bounds1["top"], x_phys1)

        wrapper1 = SubdomainWrapper(
            bkd,
            subdomain_id=1,
            physics=physics1,
            interfaces={0: interface},
            external_bcs=[right_bc1, bottom_bc1, top_bc1],
        )
        wrapper1.set_interface_boundary_indices(0, bounds1["left"])

        # Interface points in y direction
        interface.set_subdomain_boundary_points(0, nodes_y0)
        interface.set_subdomain_boundary_points(1, nodes_y0)

        interface_dof_offsets = bkd.asarray([0, interface.ndofs()])

        residual = DtNResidual(
            bkd,
            interfaces={0: interface},
            subdomain_solvers={0: wrapper0, 1: wrapper1},
            interface_dof_offsets=interface_dof_offsets,
        )

        solver = DtNSolver(bkd, residual, max_iters=30, tol=1e-10, verbose=False)

        # Exact interface value: u(0, y) = 0 for all y
        exact_interface_value = 0.0

        return {
            "residual": residual,
            "solver": solver,
            "exact_interface_value": exact_interface_value,
            "interface": interface,
            "npts_y": npts_y,
        }

    def test_residual_zero_at_exact_solution_2d(self, bkd):
        """Test residual = 0 at exact interface values."""
        bkd = self.bkd
        problem = self._create_2d_problem_linear_solution(npts_x=6, npts_y=6)

        residual = problem["residual"]
        interface = problem["interface"]
        exact_value = problem["exact_interface_value"]

        # Create interface coefficients that give constant value
        # For Legendre basis, constant = 0.5 means all coefficients give 0.5
        ndofs = interface.ndofs()
        # The constant function evaluated at Gauss-Lobatto points gives the value
        # For a constant, we need coefficients that produce constant 0.5
        interface_dofs = bkd.full((ndofs,), exact_value)

        # Compute residual
        res = residual(interface_dofs)
        res_norm = float(bkd.norm(res))

        assert res_norm < 1e-6

    def test_solver_converges_2d(self, bkd):
        """Test solver converges to correct interface values from wrong guess."""
        bkd = self.bkd
        problem = self._create_2d_problem_linear_solution(npts_x=6, npts_y=6)

        solver = problem["solver"]
        interface = problem["interface"]
        exact_value = problem["exact_interface_value"]
        ndofs = interface.ndofs()

        # Start from wrong initial guess
        initial_guess = bkd.zeros((ndofs,))
        result = solver.solve(initial_guess)

        assert result.converged, "Solver should converge"

        # Check interface values are close to exact (constant 0.5)
        computed_values = interface.evaluate(result.interface_dofs)
        expected_values = bkd.full(computed_values.shape, exact_value)

        max_error = float(bkd.max(bkd.abs(computed_values - expected_values)))
        assert max_error < 1e-6

    def test_jacobian_2d(self, bkd):
        """Test Jacobian using DerivativeChecker for 2D problem."""
        bkd = self.bkd
        problem = self._create_2d_problem_linear_solution(npts_x=5, npts_y=5)

        residual = problem["residual"]
        wrapper = DtNResidualDerivativeWrapper(residual, bkd)

        # Test at a point
        ndofs = residual.total_dofs()
        test_dofs = bkd.full((ndofs,), 0.3)
        sample = test_dofs[:, None]

        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(sample, verbosity=0)

        min_error = float(bkd.min(errors[0]))
        assert min_error < 1e-4


class TestDtNSolver3D:
    """Test DtN solver for 3D Poisson with domain decomposition.

    Problem: -Laplacian(u) = 0 on [-1, 1]^3
    Split into left subdomain [-1, 0] x [-1, 1]^2 and right [0, 1] x [-1, 1]^2
    Interface at x = 0.

    Using linear solution u(x, y, z) = x + 0.5 which satisfies Laplace equation.
    """

    def setup_method(self):
        self.bkd = NumpyBkd()

    def _compute_boundary_indices_3d(self, bkd, npts_x, npts_y, npts_z) :
        """Compute 3D boundary indices (x varies fastest, then y, then z)."""
        bkd = self.bkd
        npts_xy = npts_x * npts_y

        # Left: x=0
        left = []
        for k in range(npts_z):
            for j in range(npts_y):
                left.append(k * npts_xy + j * npts_x)
        left = bkd.asarray(left)

        # Right: x=npts_x-1
        right = []
        for k in range(npts_z):
            for j in range(npts_y):
                right.append(k * npts_xy + j * npts_x + (npts_x - 1))
        right = bkd.asarray(right)

        # Bottom: y=0
        bottom = []
        for k in range(npts_z):
            for i in range(npts_x):
                bottom.append(k * npts_xy + i)
        bottom = bkd.asarray(bottom)

        # Top: y=npts_y-1
        top = []
        for k in range(npts_z):
            for i in range(npts_x):
                top.append(k * npts_xy + (npts_y - 1) * npts_x + i)
        top = bkd.asarray(top)

        # Front: z=0
        front = []
        for j in range(npts_y):
            for i in range(npts_x):
                front.append(j * npts_x + i)
        front = bkd.asarray(front)

        # Back: z=npts_z-1
        back = []
        for j in range(npts_y):
            for i in range(npts_x):
                back.append((npts_z - 1) * npts_xy + j * npts_x + i)
        back = bkd.asarray(back)

        return {
            "left": left,
            "right": right,
            "bottom": bottom,
            "top": top,
            "front": front,
            "back": back,
        }

    def _create_3d_problem_linear_solution(self, npts=4):
        """Create 3D problem with linear manufactured solution.

        Problem setup in reference domain [-1, 1]^3:
        - Left subdomain: reference domain, physical domain [-1, 0] x [-1,1]^2
        - Right subdomain: reference domain, physical domain [0, 1] x [-1,1]^2

        Solution: u(x_phys, y, z) = x_phys (linear in physical x)
        This satisfies Laplace equation: -Laplacian(u) = 0
        Interface at x_phys = 0 has u = 0 for all y, z.

        Uses 2D interface basis (tensor product in y-z) for the 2D interface surface.
        """
        bkd = self.bkd
        npts_x = npts_y = npts_z = npts

        # Interface: 2D surface with tensor product Legendre basis
        interface_basis = LegendreInterfaceBasis2D(
            bkd,
            degree_y=npts_y - 1,
            degree_z=npts_z - 1,
            physical_bounds_y=(-1.0, 1.0),
            physical_bounds_z=(-1.0, 1.0),
        )
        interface = Interface2D(
            bkd,
            interface_id=0,
            subdomain_ids=(0, 1),
            basis=interface_basis,
            normal_direction=0,  # x-direction
        )

        # Left subdomain: physical [-1, 0] in x
        # x_phys = 0.5 * (x_ref - 1)
        mesh0 = TransformedMesh3D(npts_x, npts_y, npts_z, bkd)
        basis0 = ChebyshevBasis3D(mesh0, bkd)
        forcing0 = bkd.zeros((basis0.npts(),))
        physics0 = create_steady_diffusion(
            basis0, bkd, diffusion=1.0, forcing=lambda t: forcing0
        )

        bounds0 = self._compute_boundary_indices_3d(bkd, npts_x, npts_y, npts_z)
        nodes_x0 = basis0.nodes_x()
        nodes_y0 = basis0.nodes_y()
        x_phys0 = 0.5 * (nodes_x0 - 1)

        # BCs for u = x_phys
        left_bc0 = DirichletBC(bkd, bounds0["left"], bkd.full((npts_y * npts_z,), -1.0))

        # Bottom/top: y varies, need u = x_phys for each x,z
        bottom_vals0 = bkd.zeros((npts_x * npts_z,))
        idx = 0
        for k in range(npts_z):
            for i in range(npts_x):
                bottom_vals0[idx] = x_phys0[i]
                idx += 1
        bottom_bc0 = DirichletBC(bkd, bounds0["bottom"], bottom_vals0)

        top_vals0 = bkd.zeros((npts_x * npts_z,))
        idx = 0
        for k in range(npts_z):
            for i in range(npts_x):
                top_vals0[idx] = x_phys0[i]
                idx += 1
        top_bc0 = DirichletBC(bkd, bounds0["top"], top_vals0)

        # Front/back: z varies, need u = x_phys for each x,y
        front_vals0 = bkd.zeros((npts_x * npts_y,))
        idx = 0
        for j in range(npts_y):
            for i in range(npts_x):
                front_vals0[idx] = x_phys0[i]
                idx += 1
        front_bc0 = DirichletBC(bkd, bounds0["front"], front_vals0)

        back_vals0 = bkd.zeros((npts_x * npts_y,))
        idx = 0
        for j in range(npts_y):
            for i in range(npts_x):
                back_vals0[idx] = x_phys0[i]
                idx += 1
        back_bc0 = DirichletBC(bkd, bounds0["back"], back_vals0)

        wrapper0 = SubdomainWrapper(
            bkd,
            subdomain_id=0,
            physics=physics0,
            interfaces={0: interface},
            external_bcs=[left_bc0, bottom_bc0, top_bc0, front_bc0, back_bc0],
        )
        wrapper0.set_interface_boundary_indices(0, bounds0["right"])

        # Right subdomain: physical [0, 1] in x
        # x_phys = 0.5 * (x_ref + 1)
        mesh1 = TransformedMesh3D(npts_x, npts_y, npts_z, bkd)
        basis1 = ChebyshevBasis3D(mesh1, bkd)
        forcing1 = bkd.zeros((basis1.npts(),))
        physics1 = create_steady_diffusion(
            basis1, bkd, diffusion=1.0, forcing=lambda t: forcing1
        )

        bounds1 = self._compute_boundary_indices_3d(bkd, npts_x, npts_y, npts_z)
        nodes_x1 = basis1.nodes_x()
        x_phys1 = 0.5 * (nodes_x1 + 1)

        right_bc1 = DirichletBC(
            bkd, bounds1["right"], bkd.full((npts_y * npts_z,), 1.0)
        )

        bottom_vals1 = bkd.zeros((npts_x * npts_z,))
        idx = 0
        for k in range(npts_z):
            for i in range(npts_x):
                bottom_vals1[idx] = x_phys1[i]
                idx += 1
        bottom_bc1 = DirichletBC(bkd, bounds1["bottom"], bottom_vals1)

        top_vals1 = bkd.zeros((npts_x * npts_z,))
        idx = 0
        for k in range(npts_z):
            for i in range(npts_x):
                top_vals1[idx] = x_phys1[i]
                idx += 1
        top_bc1 = DirichletBC(bkd, bounds1["top"], top_vals1)

        front_vals1 = bkd.zeros((npts_x * npts_y,))
        idx = 0
        for j in range(npts_y):
            for i in range(npts_x):
                front_vals1[idx] = x_phys1[i]
                idx += 1
        front_bc1 = DirichletBC(bkd, bounds1["front"], front_vals1)

        back_vals1 = bkd.zeros((npts_x * npts_y,))
        idx = 0
        for j in range(npts_y):
            for i in range(npts_x):
                back_vals1[idx] = x_phys1[i]
                idx += 1
        back_bc1 = DirichletBC(bkd, bounds1["back"], back_vals1)

        wrapper1 = SubdomainWrapper(
            bkd,
            subdomain_id=1,
            physics=physics1,
            interfaces={0: interface},
            external_bcs=[right_bc1, bottom_bc1, top_bc1, front_bc1, back_bc1],
        )
        wrapper1.set_interface_boundary_indices(0, bounds1["left"])

        # Interface points (2D surface in y-z)
        # Get z nodes from basis
        nodes_z0 = basis0.nodes_z()
        interface.set_subdomain_boundary_points_2d(0, nodes_y0, nodes_z0)
        interface.set_subdomain_boundary_points_2d(1, nodes_y0, nodes_z0)

        interface_dof_offsets = bkd.asarray([0, interface.ndofs()])

        residual = DtNResidual(
            bkd,
            interfaces={0: interface},
            subdomain_solvers={0: wrapper0, 1: wrapper1},
            interface_dof_offsets=interface_dof_offsets,
        )

        solver = DtNSolver(bkd, residual, max_iters=30, tol=1e-10, verbose=False)

        # Exact interface value: u(0, y, z) = 0
        return {
            "residual": residual,
            "solver": solver,
            "exact_interface_value": 0.0,
            "interface": interface,
        }

    def test_residual_zero_at_exact_solution_3d(self, bkd):
        """Test residual = 0 at exact interface values for 3D."""
        bkd = self.bkd
        problem = self._create_3d_problem_linear_solution(npts=4)

        residual = problem["residual"]
        interface = problem["interface"]
        exact_value = problem["exact_interface_value"]

        ndofs = interface.ndofs()
        interface_dofs = bkd.full((ndofs,), exact_value)

        res = residual(interface_dofs)
        res_norm = float(bkd.norm(res))

        assert res_norm < 1e-5

    def test_solver_converges_3d(self, bkd):
        """Test solver converges for 3D problem."""
        bkd = self.bkd
        problem = self._create_3d_problem_linear_solution(npts=4)

        solver = problem["solver"]
        interface = problem["interface"]
        exact_value = problem["exact_interface_value"]
        ndofs = interface.ndofs()

        initial_guess = bkd.zeros((ndofs,))
        result = solver.solve(initial_guess)

        assert result.converged, "3D solver should converge"

        computed_values = interface.evaluate(result.interface_dofs)
        expected_values = bkd.full(computed_values.shape, exact_value)

        max_error = float(bkd.max(bkd.abs(computed_values - expected_values)))
        assert max_error < 1e-5


class TestVectorPhysicsFlux:
    """Test compute_interface_flux for vector-valued physics.

    These tests verify the flux computation methods work correctly for:
    1. Scalar diffusion physics
    2. Two-species reaction-diffusion physics
    3. Linear elasticity physics
    """

    def setup_method(self):
        self.bkd = NumpyBkd()

    def test_advection_diffusion_flux(self, bkd):
        """Test compute_interface_flux for advection-diffusion."""
        bkd = self.bkd
        npts = 10

        from pyapprox.pde.collocation.physics.advection_diffusion import (
            AdvectionDiffusionReaction,
        )

        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=1.0, velocity=None, reaction=None
        )

        # Create linear solution u(x) = x
        nodes = basis.nodes()
        state = nodes.copy()

        # Interface at x = 0 (somewhere in the middle)
        # For Chebyshev, index 0 is x=1, index npts-1 is x=-1
        # Find index closest to x=0
        mid_idx = npts // 2
        boundary_indices = bkd.asarray([mid_idx])
        normal = bkd.asarray([1.0])  # x-direction

        flux = physics.compute_interface_flux(state, boundary_indices, normal)

        # For u(x) = x, du/dx = 1, so flux = D * du/dx * n = 1.0 * 1.0 * 1.0 = 1.0
        expected_flux = 1.0
        assert abs(float(flux[0]) - expected_flux) < 10**(-10)

    def test_reaction_diffusion_flux(self, bkd):
        """Test compute_interface_flux for two-species reaction-diffusion."""
        bkd = self.bkd
        npts = 10

        from pyapprox.pde.collocation.physics.reaction_diffusion import (
            LinearReaction,
            TwoSpeciesReactionDiffusionPhysics,
        )

        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        reaction = LinearReaction(0.0, 0.0, 0.0, 0.0, bkd)  # No reaction

        physics = TwoSpeciesReactionDiffusionPhysics(
            basis, bkd, diffusion0=1.0, diffusion1=2.0, reaction=reaction
        )

        # Create linear solutions: u0(x) = x, u1(x) = 2*x
        nodes = basis.nodes()
        u0 = nodes.copy()
        u1 = 2.0 * nodes
        state = bkd.concatenate([u0, u1])

        mid_idx = npts // 2
        boundary_indices = bkd.asarray([mid_idx])
        normal = bkd.asarray([1.0])

        flux = physics.compute_interface_flux(state, boundary_indices, normal)

        # flux0 = D0 * du0/dx * n = 1.0 * 1.0 * 1.0 = 1.0
        # flux1 = D1 * du1/dx * n = 2.0 * 2.0 * 1.0 = 4.0
        # Component-stacked: [flux0, flux1]
        assert flux.shape[0] == 2
        assert abs(float(flux[0]) - 1.0) < 10**(-10)
        assert abs(float(flux[1]) - 4.0) < 10**(-10)

    def test_linear_elasticity_flux(self, bkd):
        """Test compute_interface_flux for linear elasticity."""
        bkd = self.bkd
        npts = 6

        from pyapprox.pde.collocation.physics.linear_elasticity import (
            LinearElasticityPhysics,
        )

        mesh = TransformedMesh2D(npts, npts, bkd)
        basis = ChebyshevBasis2D(mesh, bkd)
        lamda = 1.0
        mu = 1.0
        physics = LinearElasticityPhysics(basis, bkd, lamda, mu)

        # Simple displacement: u(x,y) = x, v(x,y) = 0
        # This gives strain: exx = 1, exy = 0, eyy = 0
        # Stress: sigma_xx = lambda + 2*mu = 3, sigma_xy = 0, sigma_yy = lambda = 1
        # Traction for n = [1, 0]: t_x = sigma_xx = 3, t_y = sigma_xy = 0
        nodes_x = basis.nodes_x()
        basis.nodes_y()
        npts_total = basis.npts()

        # u(x,y) = x, v(x,y) = 0
        u = bkd.zeros((npts_total,))
        v = bkd.zeros((npts_total,))
        for j in range(npts):
            for i in range(npts):
                idx = j * npts + i
                u[idx] = nodes_x[i]
                v[idx] = 0.0
        state = bkd.concatenate([u, v])

        # Use right boundary (x = nodes_x[0] which is +1 in Chebyshev)
        # Indices on right boundary: j * npts + 0 for j = 0..npts-1
        boundary_indices = bkd.asarray([j * npts for j in range(npts)])
        normal = bkd.asarray([1.0, 0.0])

        flux = physics.compute_interface_flux(state, boundary_indices, normal)

        # Expected: t_x = 3.0 for all points, t_y = 0.0 for all points
        # Shape: (2 * npts,)
        assert flux.shape[0] == 2 * npts

        # First npts entries are t_x, should be ~3.0
        for i in range(npts):
            assert abs(float(flux[i]) - 3.0) < 10**(-8)

        # Next npts entries are t_y, should be ~0.0
        for i in range(npts, 2 * npts):
            assert abs(float(flux[i]) - 0.0) < 10**(-10)


class TestVectorInterfaceComponents:
    """Test interface classes with ncomponents parameter."""

    def setup_method(self):
        self.bkd = NumpyBkd()

    def test_interface1d_with_ncomponents(self, bkd):
        """Test Interface1D handles multiple components."""
        bkd = self.bkd

        interface = Interface1D(
            bkd,
            interface_id=0,
            subdomain_ids=(0, 1),
            interface_point=0.0,
            ncomponents=2,
        )

        assert interface.ncomponents() == 2
        assert interface.ndofs() == 1
        assert interface.total_ndofs() == 2

        # Set up interpolation
        interface.set_subdomain_boundary_points(0, bkd.asarray([0.0]))
        interface.set_subdomain_boundary_points(1, bkd.asarray([0.0]))

        # Evaluate with 2 coefficients
        coeffs = bkd.asarray([1.0, 2.0])
        values = interface.evaluate(coeffs)
        assert values.shape[0] == 2
        assert abs(float(values[0]) - 1.0) < 1e-7
        assert abs(float(values[1]) - 2.0) < 1e-7

    def test_interface_with_ncomponents(self, bkd):
        """Test Interface handles multiple components."""
        bkd = self.bkd

        interface_basis = LegendreInterfaceBasis1D(
            bkd, degree=4, physical_bounds=(-1.0, 1.0)
        )

        interface = Interface(
            bkd,
            interface_id=0,
            subdomain_ids=(0, 1),
            basis=interface_basis,
            normal_direction=0,
            ambient_dim=2,
            ncomponents=2,
        )

        ndofs = interface.ndofs()
        assert interface.ncomponents() == 2
        assert interface.total_ndofs() == 2 * ndofs

        # Evaluate with component-stacked coefficients
        # First component: all zeros
        # Second component: all ones
        coeffs = bkd.concatenate(
            [
                bkd.zeros((ndofs,)),
                bkd.ones((ndofs,)),
            ]
        )

        values = interface.evaluate(coeffs)
        npts = interface.npts()

        # First npts values should be ~0
        for i in range(npts):
            assert abs(float(values[i]) - 0.0) < 10**(-10)

        # Second npts values should be ~1
        for i in range(npts, 2 * npts):
            assert abs(float(values[i]) - 1.0) < 10**(-10)
