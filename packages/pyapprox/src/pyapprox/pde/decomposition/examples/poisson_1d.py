"""1D Poisson domain decomposition example.

Demonstrates DtN domain decomposition for the 1D Poisson equation:
    -u'' = f    on [0, 2]
    u(0) = 0, u(2) = 0

Domain is split at x = 1 into:
    - Subdomain 0: [0, 1]
    - Subdomain 1: [1, 2]

Manufactured solution: u(x) = sin(pi*x)
which satisfies -u'' = pi^2 * sin(pi*x) with u(0) = u(2) = 0.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Dict, Tuple

from numpy.typing import NDArray

from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.boundary import (
    zero_dirichlet_bc,
)
from pyapprox.pde.collocation.mesh import TransformedMesh1D
from pyapprox.pde.collocation.physics.advection_diffusion import (
    create_steady_diffusion,
)
from pyapprox.pde.decomposition.interface import Interface1D
from pyapprox.pde.decomposition.solver import (
    DtNResidual,
    DtNSolver,
)
from pyapprox.pde.decomposition.subdomain import SubdomainWrapper
from pyapprox.util.backends.numpy import NumpyBkd


def create_poisson_1d_problem(
    npts_per_subdomain: int = 10,
) -> Tuple[dict, dict, NDArray, NumpyBkd]:  # type: ignore[type-arg]
    """Create 1D Poisson problem with two subdomains.

    Parameters
    ----------
    npts_per_subdomain : int
        Number of collocation points per subdomain.

    Returns
    -------
    tuple
        (subdomain_solvers, interfaces, interface_dof_offsets, bkd)
    """
    bkd = NumpyBkd()

    # Manufactured solution: u(x) = sin(pi*x)
    def exact_solution(x: NDArray) -> NDArray:
        return bkd.sin(math.pi * x)

    def forcing(
        time: float,
    ) -> "Callable[[NDArray], NDArray]":
        # For -u'' = f, with u = sin(pi*x), f = pi^2 * sin(pi*x)
        # But ADR residual is du/dt = D*laplacian(u) + f
        # For steady state: 0 = D*laplacian(u) + f => f = -D*laplacian(u)
        # With D=1, laplacian(sin(pi*x)) = -pi^2*sin(pi*x)
        # So residual = -pi^2*sin(pi*x) + f = 0 => f = pi^2*sin(pi*x)
        # But we want -u'' = f, so residual = 0 when D*u'' + f = 0
        # D*(-pi^2*sin(pi*x)) + f = 0 => f = pi^2*sin(pi*x)
        return lambda nodes: math.pi**2 * bkd.sin(math.pi * nodes)

    # Create interface at x = 1
    interface = Interface1D(
        bkd,
        interface_id=0,
        subdomain_ids=(0, 1),
        interface_point=1.0,
    )

    # Create subdomain 0: [0, 1]
    # Chebyshev nodes on [-1, 1], mapped to [0, 1]
    mesh0 = TransformedMesh1D(npts_per_subdomain, bkd)
    basis0 = ChebyshevBasis1D(mesh0, bkd)
    nodes0 = basis0.nodes()
    # Map to [0, 1]: x = 0.5 * (nodes + 1)
    physical_nodes0 = 0.5 * (nodes0 + 1.0)

    # Forcing at physical nodes
    forcing0 = math.pi**2 * bkd.sin(math.pi * physical_nodes0)
    physics0 = create_steady_diffusion(
        basis0, bkd, diffusion=1.0, forcing=lambda t: forcing0
    )

    # External BC: u(0) = 0 at left boundary
    # Chebyshev nodes: index 0 is x=1, index npts-1 is x=-1 (mapped: 0)
    left_bc0 = zero_dirichlet_bc(bkd, bkd.asarray([npts_per_subdomain - 1]))

    # Create subdomain 1: [1, 2]
    mesh1 = TransformedMesh1D(npts_per_subdomain, bkd)
    basis1 = ChebyshevBasis1D(mesh1, bkd)
    nodes1 = basis1.nodes()
    # Map to [1, 2]: x = 0.5 * (nodes + 1) + 1 = 0.5 * nodes + 1.5
    physical_nodes1 = 0.5 * nodes1 + 1.5

    # Forcing at physical nodes
    forcing1 = math.pi**2 * bkd.sin(math.pi * physical_nodes1)
    physics1 = create_steady_diffusion(
        basis1, bkd, diffusion=1.0, forcing=lambda t: forcing1
    )

    # External BC: u(2) = 0 at right boundary
    # Chebyshev nodes: index 0 is x=1 (mapped to 2), index npts-1 is x=-1 (mapped to 1)
    right_bc1 = zero_dirichlet_bc(bkd, bkd.asarray([0]))

    # Create subdomain wrappers
    wrapper0 = SubdomainWrapper(
        bkd,
        subdomain_id=0,
        physics=physics0,
        interfaces={0: interface},
        external_bcs=[left_bc0],
    )

    wrapper1 = SubdomainWrapper(
        bkd,
        subdomain_id=1,
        physics=physics1,
        interfaces={0: interface},
        external_bcs=[right_bc1],
    )

    # Set interface boundary indices
    # For subdomain 0 (right boundary at x=1): index 0
    wrapper0.set_interface_boundary_indices(0, bkd.asarray([0]))
    # For subdomain 1 (left boundary at x=1): index npts-1
    wrapper1.set_interface_boundary_indices(0, bkd.asarray([npts_per_subdomain - 1]))

    # Set up interface interpolation (single point, so trivial)
    interface.set_subdomain_boundary_points(0, bkd.asarray([1.0]))
    interface.set_subdomain_boundary_points(1, bkd.asarray([1.0]))

    # Interface DOF offsets: 1 interface with 1 DOF
    interface_dof_offsets = bkd.asarray([0, 1])

    subdomain_solvers = {0: wrapper0, 1: wrapper1}
    interfaces = {0: interface}

    return subdomain_solvers, interfaces, interface_dof_offsets, bkd


def solve_poisson_1d(
    npts_per_subdomain: int = 10, verbose: bool = False,
) -> Dict[str, object]:
    """Solve 1D Poisson with domain decomposition.

    Parameters
    ----------
    npts_per_subdomain : int
        Number of collocation points per subdomain.
    verbose : bool
        Print convergence info.

    Returns
    -------
    dict
        Results including solutions and error.
    """
    # Create problem
    subdomain_solvers, interfaces, interface_dof_offsets, bkd = (
        create_poisson_1d_problem(npts_per_subdomain)
    )

    # Create residual and solver
    residual = DtNResidual(
        bkd,
        interfaces=interfaces,
        subdomain_solvers=subdomain_solvers,
        interface_dof_offsets=interface_dof_offsets,
    )

    solver = DtNSolver(
        bkd,
        residual,
        max_iters=20,
        tol=1e-10,
        verbose=verbose,
    )

    # Solve
    result = solver.solve()

    # Compute error at interface
    # Exact solution at x=1: sin(pi) = 0
    interface_error = abs(bkd.to_float(result.interface_dofs[0]) - math.sin(math.pi))

    return {
        "result": result,
        "interface_dofs": result.interface_dofs,
        "interface_error": interface_error,
        "converged": result.converged,
        "iterations": result.iterations,
        "residual_norm": result.residual_norm,
    }


def main() -> None:
    """Run 1D Poisson example."""
    print("1D Poisson Domain Decomposition Example")
    print("=" * 50)
    print()
    print("Problem: -u'' = f on [0, 2]")
    print("         u(0) = 0, u(2) = 0")
    print("Exact solution: u(x) = sin(pi*x)")
    print("Decomposition: [0, 1] | [1, 2]")
    print()

    results = solve_poisson_1d(npts_per_subdomain=10, verbose=True)

    print()
    print("Results:")
    print(f"  Converged: {results['converged']}")
    print(f"  Iterations: {results['iterations']}")
    print(f"  Final residual norm: {results['residual_norm']:.2e}")
    iface_val = results['interface_dofs'][0]
    print(f"  Interface value: {iface_val:.10f}")
    print(f"  Interface error: {results['interface_error']:.2e}")


if __name__ == "__main__":
    main()
