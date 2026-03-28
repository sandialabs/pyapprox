"""Factory function for a 1D hyperelastic bar forward model.

Solves div(P) + f = 0 with Neo-Hookean constitutive law, where P is the
first Piola-Kirchhoff stress. Young's modulus E(x) is parameterized via
a field map (e.g., KLE), converted to Lame parameters with fixed Poisson ratio.
"""

from typing import Any, Callable, Optional

from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.boundary import (
    constant_dirichlet_bc,
    flux_neumann_bc,
    zero_dirichlet_bc,
)
from pyapprox.pde.collocation.forward_models.steady import (
    SteadyForwardModel,
)
from pyapprox.pde.collocation.mesh import (
    AffineTransform1D,
    TransformedMesh1D,
)
from pyapprox.pde.collocation.physics.hyperelasticity import (
    HyperelasticityPhysics,
)
from pyapprox.pde.collocation.physics.stress_models import (
    NeoHookeanStress,
)
from pyapprox.pde.field_maps.protocol import (
    FieldMapProtocol,
)
from pyapprox.pde.parameterizations.hyperelastic_lame import (
    create_hyperelastic_youngs_modulus_parameterization,
)
from pyapprox.util.backends.protocols import Array, Backend


def create_hyperelastic_bar_1d(
    bkd: Backend[Array],
    npts: int,
    length: float,
    E_mean: float,
    poisson_ratio: float,
    forcing: Callable[..., Any],
    field_map: FieldMapProtocol[Array],
    traction: Optional[float] = None,
    dirichlet_left: float = 0.0,
    functional: object = None,
) -> SteadyForwardModel[Array]:
    """Create a steady 1D hyperelastic bar forward model.

    Solves div(P) + f = 0 on [0, length] with Neo-Hookean stress, where:
    - Left end (x=0): Dirichlet BC u = dirichlet_left (fixed displacement).
    - Right end (x=L): Neumann BC P*n = traction (applied traction).

    Young's modulus E(x) is parameterized via field_map and converted to
    Lame parameters (mu, lambda) using fixed Poisson ratio.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    npts : int
        Number of Chebyshev collocation points.
    length : float
        Bar length L.
    E_mean : float
        Mean Young's modulus for initial Lame parameter computation.
    poisson_ratio : float
        Fixed Poisson ratio nu.
    forcing : callable
        Body force f(x). Accepts time, returns Array of shape (npts,).
    field_map : FieldMapProtocol
        Field map for E(x) parameterization (e.g. KLE).
    traction : float, optional
        Applied traction P*n at x=L. If None, uses zero Neumann.
    dirichlet_left : float
        Prescribed displacement at x=0. Default 0.0 (fixed end).
    functional : optional
        QoI functional. If None, returns full displacement field.

    Returns
    -------
    SteadyForwardModel
        Configured forward model.
    """
    # Mesh and basis
    transform = AffineTransform1D((0.0, length), bkd)
    mesh = TransformedMesh1D(npts, bkd, transform)
    basis = ChebyshevBasis1D(mesh, bkd)

    # Convert E_mean to Lame parameters
    dmu_dE = 1.0 / (2.0 * (1.0 + poisson_ratio))
    dlam_dE = poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    mu_init = E_mean * dmu_dE
    lamda_init = E_mean * dlam_dE

    # Stress model and physics
    stress_model = NeoHookeanStress(lamda=lamda_init, mu=mu_init)
    physics = HyperelasticityPhysics(basis, bkd, stress_model, forcing)

    # Left BC: prescribed displacement at x=0
    left_idx = mesh.boundary_indices(0)
    if dirichlet_left == 0.0:
        bc_left = zero_dirichlet_bc(bkd, left_idx)
    else:
        bc_left = constant_dirichlet_bc(bkd, left_idx, dirichlet_left)

    # Right BC: traction at x=L
    # For hyperelastic: flux = P (PK1 stress), so flux.n = P*n
    # At x=L with n=+1: flux.n = P, so values = traction directly
    right_idx = mesh.boundary_indices(1)
    right_normals = mesh.boundary_normals(1)
    traction_val = 0.0 if traction is None else traction
    bc_right = flux_neumann_bc(
        bkd,
        right_idx,
        right_normals,
        physics,
        traction_val,
    )

    physics.set_boundary_conditions([bc_left, bc_right])

    # Parameterization
    param = create_hyperelastic_youngs_modulus_parameterization(
        bkd,
        basis,
        field_map,
        poisson_ratio,
    )

    init_state = bkd.zeros((npts,))
    return SteadyForwardModel(
        physics,
        bkd,
        init_state,
        functional=functional,
        parameterization=param,
    )
