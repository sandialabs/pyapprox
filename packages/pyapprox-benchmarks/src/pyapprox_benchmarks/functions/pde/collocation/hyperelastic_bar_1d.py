"""Builder of a 1D hyperelastic bar.

Solves div(P) + f = 0 with Neo-Hookean constitutive law, where P is the
first Piola-Kirchhoff stress. Young's modulus E(x) is parameterized via
a field map (e.g., KLE), converted to Lame parameters with fixed Poisson ratio.
"""

from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.mesh import BoundarySide, TransformedMesh1D
from pyapprox.pde.collocation.physics.coefficients import NodalSupplier
from pyapprox.pde.collocation.physics.hyperelasticity import (
    HyperelasticityPhysics,
)
from pyapprox.pde.collocation.physics.neo_hookean_law import NeoHookeanLaw
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
from pyapprox.pde.parameterizations.hyperelastic_lame import (
    create_hyperelastic_youngs_modulus_parameterization,
)
from pyapprox.util.backends.protocols import Array
from pyapprox_benchmarks.functions.pde.collocation.boundaries import (
    BoundaryContext,
    Dirichlet,
    FluxNeumann,
    assemble_boundaries,
)


def build_hyperelastic_bar_1d(
    mesh: TransformedMesh1D[Array],
    forcing: TimeAwareCallableProtocol[Array],
    field_map: FieldMapProtocol[Array],
    poisson_ratio: float,
    traction: float = 0.0,
    dirichlet_left: float = 0.0,
) -> ParameterizedCollocationSystem[Array, ChebyshevBasis1D[Array]]:
    """Build a steady 1D hyperelastic bar.

    Solves div(P) + f = 0 on the physical interval [a, b] of ``mesh``
    with Neo-Hookean stress, where:
    - Left end (x=a): Dirichlet BC u = dirichlet_left (fixed displacement).
    - Right end (x=b): Neumann BC P*n = traction (applied traction).

    Young's modulus E(x) is parameterized via field_map and converted to
    Lame parameters (mu, lambda) using fixed Poisson ratio.

    Parameters
    ----------
    mesh : TransformedMesh1D
        The Chebyshev mesh; its transform sets the physical interval.
    forcing : TimeAwareCallableProtocol
        Body force f, called with the physical node coordinates, shape
        (1, npts), and a time, and declared time-independent or not
        (``TimeIndependent`` / ``TimeDependent``).
    field_map : FieldMapProtocol
        Field map for E(x) at the physical nodes (e.g. KLE), each value
        positive.
    poisson_ratio : float
        Fixed Poisson ratio nu, in [0, 1/2).
    traction : float
        Applied traction P*n at x=b. Default 0.0 (free end).
    dirichlet_left : float
        Prescribed displacement at x=a. Default 0.0 (fixed end).

    Returns
    -------
    ParameterizedCollocationSystem
        Build its function with ``collocation_steady_function``.
    """
    bkd = mesh.bkd()
    basis = ChebyshevBasis1D(mesh, bkd)

    # The parameterization replaces both Lame fields with those of E(p);
    # these are placeholders.
    physics = HyperelasticityPhysics(
        basis,
        bkd,
        NeoHookeanLaw(bkd),
        mu=1.0,
        lamda=1.0,
        forcing=NodalSupplier(forcing, mesh.points()),
    )

    # Left BC: prescribed displacement at x=a
    # Right BC: traction at x=b
    # For hyperelastic: flux = P (PK1 stress), so flux.n = P*n
    # At x=b with n=+1: flux.n = P, so values = traction directly
    essential, closures = assemble_boundaries(
        BoundaryContext(mesh, basis, physics),
        (
            Dirichlet(BoundarySide.LEFT, dirichlet_left),
            FluxNeumann(BoundarySide.RIGHT, traction),
        ),
    )

    # Parameterization
    param = create_hyperelastic_youngs_modulus_parameterization(
        physics,
        bkd,
        field_map,
        poisson_ratio,
    )

    return ParameterizedCollocationSystem(
        CollocationSystem(physics, essential, closures), param, basis
    )
