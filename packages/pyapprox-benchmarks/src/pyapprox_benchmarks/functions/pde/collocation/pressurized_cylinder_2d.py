"""Builder of a 2D linear elastic pressurized cylinder.

Solves 2D linear elasticity -div(sigma) + f = 0 on a quarter-annulus domain
[R_i, R_o] x [0, pi/2] with polar coordinate transform, internal pressure
loading, and symmetry boundary conditions.

Young's modulus E(x) is parameterized via a field map (e.g., KLE) and
converted to Lame parameters with fixed Poisson ratio.
"""

from typing import Optional

from pyapprox.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.pde.collocation.mesh import BoundarySide, TransformedMesh2D
from pyapprox.pde.collocation.physics import LinearElasticityPhysics
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
from pyapprox.pde.parameterizations.lame import (
    create_youngs_modulus_parameterization,
)
from pyapprox.util.backends.protocols import Array
from pyapprox_benchmarks.functions.pde.collocation.boundaries import (
    BoundaryContext,
    CauchyPressure,
    CauchyTraction,
    Dirichlet,
    assemble_boundaries,
)


def build_linear_pressurized_cylinder_2d(
    mesh: TransformedMesh2D[Array],
    poisson_ratio: float,
    inner_pressure: float,
    field_map: FieldMapProtocol[Array],
    forcing: Optional[TimeAwareCallableProtocol[Array]] = None,
) -> ParameterizedCollocationSystem[Array, ChebyshevBasis2D[Array]]:
    """Build a steady 2D linear elastic pressurized cylinder.

    Solves -div(sigma) + f = 0 on a quarter-annulus [R_i, R_o] x [0, pi/2]
    with:
    - Inner boundary (r=R_i): pressure traction t = -p * n_domain
    - Outer boundary (r=R_o): stress-free traction t = 0
    - Bottom (theta=0): symmetry (v=0 Dirichlet, sigma_xy=0 traction)
    - Top (theta=pi/2): symmetry (u=0 Dirichlet, sigma_xy=0 traction)

    Young's modulus E(x) is parameterized via field_map and converted to
    Lame parameters (mu, lambda) using fixed Poisson ratio.

    Parameters
    ----------
    mesh : TransformedMesh2D
        The quarter-annulus: a ``PolarTransform`` of (r, theta) over
        [R_i, R_o] x [0, pi/2], so the sides LEFT, RIGHT, BOTTOM and TOP
        are r=R_i, r=R_o, theta=0 and theta=pi/2.
    poisson_ratio : float
        Fixed Poisson ratio nu.
    inner_pressure : float
        Internal pressure p applied at r=R_i.
    field_map : FieldMapProtocol
        Field map for E(x) at the physical nodes (e.g. from
        create_lognormal_kle_field_map), each value positive.
    forcing : TimeAwareCallableProtocol, optional
        Body force f, called with the physical node coordinates, shape
        (2, npts), and a time, returning the components stacked, shape
        (2*npts,); declared time-independent or not. Default none.

    Returns
    -------
    ParameterizedCollocationSystem
        Build its function with ``collocation_steady_function``.
    """
    bkd = mesh.bkd()
    basis = ChebyshevBasis2D(mesh, bkd)

    # Physics; the parameterization replaces both Lame fields with those
    # of E(p), so these are placeholders.
    physics = LinearElasticityPhysics(
        basis,
        bkd,
        lamda=1.0,
        mu=1.0,
        forcing=None if forcing is None else NodalSupplier(forcing, mesh.points()),
    )

    # ------------------------------------------------------------------
    # Boundary conditions. Shared corner nodes: Dirichlet rows beat
    # traction rows, and among tractions the later spec wins, so the
    # order is the old last-applied-wins order. The traction reads its
    # stress, and so its material, from the physics.
    # ------------------------------------------------------------------
    essential, closures = assemble_boundaries(
        BoundaryContext(mesh, basis, physics),
        (
            # (a) Outer boundary (r=R_o): stress-free traction t=0
            CauchyTraction(BoundarySide.RIGHT, 0, 0.0),
            CauchyTraction(BoundarySide.RIGHT, 1, 0.0),
            # (b) Bottom (theta=0): shear-free traction t_x=0 (component 0)
            CauchyTraction(BoundarySide.BOTTOM, 0, 0.0),
            # (c) Top (theta=pi/2): shear-free traction t_y=0 (component 1)
            CauchyTraction(BoundarySide.TOP, 1, 0.0),
            # (d) Inner boundary (r=R_i): pressure loading
            # Domain outward normal at r=R_i points radially inward (-e_r).
            # Traction from pressure: t = sigma . n_domain = -p * n_domain
            CauchyPressure(BoundarySide.LEFT, inner_pressure),
            # (e) Bottom Dirichlet v=0 (theta=0)
            Dirichlet(BoundarySide.BOTTOM, 0.0, component=1),
            # (f) Top Dirichlet u=0 (theta=pi/2)
            Dirichlet(BoundarySide.TOP, 0.0, component=0),
        ),
    )

    # Parameterization: E field -> Lame parameters
    param = create_youngs_modulus_parameterization(
        physics,
        bkd,
        field_map,
        poisson_ratio,
    )

    return ParameterizedCollocationSystem(
        CollocationSystem(physics, essential, closures), param, basis
    )
