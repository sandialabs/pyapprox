"""Builder of a 1D linear elastic bar.

The 1D bar equation -d/dx(E(x)*du/dx) = f(x) is mathematically identical
to steady diffusion. This builder reuses AdvectionDiffusionReaction with
diffusion = E(x), providing a mechanics-oriented interface.
"""

from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.mesh import BoundarySide, TransformedMesh1D
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.collocation.physics.coefficients import NodalSupplier
from pyapprox.pde.constitutive.coefficient_functions import (
    TimeAwareCallableProtocol,
)
from pyapprox.pde.field_maps.protocol import (
    FieldMapProtocol,
)
from pyapprox.pde.models.collocation.closed_system import CollocationSystem
from pyapprox.pde.models.collocation.steady_function import (
    ParameterizedCollocationSystem,
)
from pyapprox.pde.parameterizations.diffusion import (
    create_diffusion_parameterization,
)
from pyapprox.util.backends.protocols import Array
from pyapprox_benchmarks.functions.pde.collocation.boundaries import (
    BoundaryContext,
    Dirichlet,
    FluxNeumann,
    assemble_boundaries,
)


def build_linear_elastic_bar_1d(
    mesh: TransformedMesh1D[Array],
    forcing: TimeAwareCallableProtocol[Array],
    field_map: FieldMapProtocol[Array],
    traction: float = 0.0,
    dirichlet_left: float = 0.0,
) -> ParameterizedCollocationSystem[Array, ChebyshevBasis1D[Array]]:
    """Build a steady 1D linear elastic bar.

    Solves -d/dx(E(x)*du/dx) = f(x) on the physical interval [a, b] of
    ``mesh`` with:
    - Left end (x=a): Dirichlet BC u = dirichlet_left (fixed displacement).
    - Right end (x=b): Neumann BC E*du/dx = traction (applied traction).

    Parameters
    ----------
    mesh : TransformedMesh1D
        The Chebyshev mesh; its transform sets the physical interval.
    forcing : TimeAwareCallableProtocol
        Body force f, called with the physical node coordinates, shape
        (1, npts), and a time, and declared time-independent or not
        (``TimeIndependent`` / ``TimeDependent``).
    field_map : FieldMapProtocol
        Field map for E(x) at the physical nodes (e.g. from
        create_lognormal_kle_field_map), each value positive.
    traction : float
        Applied traction at x=b. Default 0.0 (free end).
    dirichlet_left : float
        Prescribed displacement at x=a. Default 0.0 (fixed end).

    Returns
    -------
    ParameterizedCollocationSystem
        Build its function with ``collocation_steady_function``.
    """
    bkd = mesh.bkd()
    basis = ChebyshevBasis1D(mesh, bkd)

    # The parameterization replaces the diffusion with E(p).
    physics = AdvectionDiffusionReaction(
        basis,
        bkd,
        diffusion=1.0,
        forcing=NodalSupplier(forcing, mesh.points()),
    )

    # Left BC: prescribed displacement at x=a
    # Right BC: traction at x=b
    # flux(u).n = (-E*du/dx)*(+1) = -E*du/dx
    # We want E*du/dx = T, so flux.n = -T
    essential, closures = assemble_boundaries(
        BoundaryContext(mesh, basis, physics),
        (
            Dirichlet(BoundarySide.LEFT, dirichlet_left),
            FluxNeumann(BoundarySide.RIGHT, -traction),
        ),
    )

    param = create_diffusion_parameterization(physics, bkd, field_map)

    return ParameterizedCollocationSystem(
        CollocationSystem(physics, essential, closures), param, basis
    )
