"""Periodic viscous Burgers physics builder.

The periodic 1D Galerkin discretization identifies the domain
endpoints at the mesh-topology level (``PeriodicStructuredMesh1D``),
so the physics carries NO boundary-condition objects: the semi-discrete
system is ``M du/dt = f(u)`` on all ``nx`` dofs with a full-rank mass
matrix.  This is the form the operator-inference recovery benchmarks
require -- the projected operator is exactly polynomial in the state
with no boundary lift.
"""

from typing import Any, Callable, Optional

from pyapprox.pde.galerkin.physics.burgers import BurgersPhysics
from pyapprox.pde.galerkin.protocols.basis import GalerkinBasisProtocol
from pyapprox.util.backends.protocols import Array, Backend

from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.mesh import PeriodicStructuredMesh1D


def build_periodic_line_basis(
    nx: int,
    bounds: tuple[float, float],
    bkd: Backend[Array],
    degree: int = 1,
) -> LagrangeBasis[Array]:
    """Build a Lagrange basis on the endpoint-identified periodic line.

    Parameters
    ----------
    nx : int
        Number of elements (equals the number of P1 dofs -- the
        endpoints are identified).
    bounds : tuple[float, float]
        Domain bounds ``(xmin, xmax)``; ``xmax`` is identified with
        ``xmin``.
    bkd : Backend[Array]
        Computational backend.
    degree : int, optional
        Polynomial degree of the Lagrange basis. Default 1.

    Returns
    -------
    LagrangeBasis
        Basis on the periodic mesh.  Interpolated functions must be
        periodic-compatible (see ``PeriodicStructuredMesh1D``).
    """
    mesh = PeriodicStructuredMesh1D(nx=nx, bounds=bounds, bkd=bkd)
    return LagrangeBasis(mesh, degree=degree)


def build_periodic_burgers_physics(
    basis: GalerkinBasisProtocol[Array],
    viscosity: float,
    bkd: Backend[Array],
    forcing: Optional[Callable[..., Any]] = None,
) -> BurgersPhysics[Array]:
    """Build viscous Burgers physics on a periodic basis.

    The basis is taken as an argument (rather than built internally)
    so parameterized rebuilds -- e.g. ``PDEOpInfProblem.physics_at`` --
    share ONE basis across all parameter values: reduced-operator
    recovery requires snapshots, projections, and the intrusive
    reference to live on the identical discretization.

    Parameters
    ----------
    basis : GalerkinBasisProtocol[Array]
        Shared finite element basis, typically from
        :func:`build_periodic_line_basis`.
    viscosity : float
        Kinematic viscosity.
    bkd : Backend[Array]
        Computational backend.
    forcing : Callable, optional
        Forcing term ``f(x)`` or ``f(x, t)`` returning ``(npts,)``
        (numpy at the skfem assembly seam).

    Returns
    -------
    BurgersPhysics
        Physics with no boundary conditions (periodic topology).
    """
    return BurgersPhysics(
        basis=basis, viscosity=viscosity, bkd=bkd, forcing=forcing
    )
