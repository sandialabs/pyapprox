"""Factory for a 2D hyperelastic pressurized cylinder forward model.

Solves 2D hyperelasticity -div(P) + f = 0 on a quarter-annulus domain
[R_i, R_o] x [0, pi/2] with polar coordinate transform, internal pressure
loading, and symmetry boundary conditions, where P is the first
Piola-Kirchhoff stress from a Neo-Hookean constitutive model.

Young's modulus E(x) is parameterized via a field map (e.g., KLE) and
converted to Lame parameters with fixed Poisson ratio.
"""

import math
from typing import Any, Callable, Optional

from pyapprox.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.pde.collocation.boundary import zero_dirichlet_bc
from pyapprox.pde.collocation.boundary.hyperelastic_traction import (
    hyperelastic_traction_neumann_bc,
)
from pyapprox.pde.collocation.forward_models.steady import (
    SteadyForwardModel,
)
from pyapprox.pde.collocation.mesh import TransformedMesh2D
from pyapprox.pde.collocation.mesh.transforms import PolarTransform
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


def create_hyperelastic_pressurized_cylinder_2d(
    bkd: Backend[Array],
    npts_r: int,
    npts_theta: int,
    r_inner: float,
    r_outer: float,
    E_mean: float,
    poisson_ratio: float,
    inner_pressure: float,
    field_map: FieldMapProtocol[Array],
    forcing: Optional[Callable[..., Any]] = None,
    functional=None,
) -> SteadyForwardModel[Array]:
    """Create a steady 2D hyperelastic pressurized cylinder forward model.

    Solves -div(P) + f = 0 on a quarter-annulus [R_i, R_o] x [0, pi/2]
    using Neo-Hookean hyperelasticity with:
    - Inner boundary (r=R_i): pressure traction t = -p * n
    - Outer boundary (r=R_o): stress-free traction t = 0
    - Bottom (theta=0): symmetry (v=0 Dirichlet, t_x=0 traction)
    - Top (theta=pi/2): symmetry (u=0 Dirichlet, t_y=0 traction)

    Young's modulus E(x) is parameterized via field_map and converted to
    Lame parameters (mu, lambda) using fixed Poisson ratio.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    npts_r : int
        Number of Chebyshev collocation points in radial direction.
    npts_theta : int
        Number of Chebyshev collocation points in angular direction.
    r_inner : float
        Inner radius R_i.
    r_outer : float
        Outer radius R_o.
    E_mean : float
        Mean Young's modulus for initial Lame parameter computation.
    poisson_ratio : float
        Fixed Poisson ratio nu.
    inner_pressure : float
        Internal pressure p applied at r=R_i.
    field_map : FieldMapProtocol
        Field map for E(x) parameterization (e.g. from
        create_lognormal_kle_field_map).
    forcing : callable, optional
        Body force f. Accepts time, returns Array of shape (2*npts,).
    functional : optional
        QoI functional. If None, returns full displacement field.

    Returns
    -------
    SteadyForwardModel
        Configured forward model mapping KLE coefficients to displacement.
    """
    # Mesh and basis on quarter-annulus
    transform = PolarTransform(
        (r_inner, r_outer),
        (0.0, math.pi / 2.0),
        bkd,
    )
    mesh = TransformedMesh2D(npts_r, npts_theta, bkd, transform)
    basis = ChebyshevBasis2D(mesh, bkd)

    # Convert E_mean to Lame parameters
    dmu_dE = 1.0 / (2.0 * (1.0 + poisson_ratio))
    dlam_dE = poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    mu_init = E_mean * dmu_dE
    lamda_init = E_mean * dlam_dE

    # Stress model shared between physics and traction BCs
    stress_model = NeoHookeanStress(lamda=lamda_init, mu=mu_init)

    # Physics
    physics = HyperelasticityPhysics(
        basis,
        bkd,
        stress_model,
        forcing=forcing,
    )

    # Derivative matrices for traction BCs
    D_matrices = [
        basis.derivative_matrix(1, 0),
        basis.derivative_matrix(1, 1),
    ]
    npts = basis.npts()

    # ------------------------------------------------------------------
    # Boundary conditions (order: traction first, Dirichlet last for
    # correct corner priority via last-applied-wins)
    # ------------------------------------------------------------------
    bcs = []

    # (a) Outer boundary (bnd 1, r=R_o): stress-free traction t=0
    outer_idx = mesh.boundary_indices(1)
    outer_normals = mesh.boundary_normals(1)
    for comp in (0, 1):
        bcs.append(
            hyperelastic_traction_neumann_bc(
                bkd,
                outer_idx,
                outer_normals,
                D_matrices,
                stress_model,
                npts,
                comp,
                values=0.0,
            )
        )

    # (b) Bottom (bnd 2, theta=0): shear-free traction t_x=0 (component 0)
    bottom_idx = mesh.boundary_indices(2)
    bottom_normals = mesh.boundary_normals(2)
    bcs.append(
        hyperelastic_traction_neumann_bc(
            bkd,
            bottom_idx,
            bottom_normals,
            D_matrices,
            stress_model,
            npts,
            component=0,
            values=0.0,
        )
    )

    # (c) Top (bnd 3, theta=pi/2): shear-free traction t_y=0 (component 1)
    top_idx = mesh.boundary_indices(3)
    top_normals = mesh.boundary_normals(3)
    bcs.append(
        hyperelastic_traction_neumann_bc(
            bkd,
            top_idx,
            top_normals,
            D_matrices,
            stress_model,
            npts,
            component=1,
            values=0.0,
        )
    )

    # (d) Inner boundary (bnd 0, r=R_i): pressure loading
    # Traction from pressure: t = -p * n
    inner_idx = mesh.boundary_indices(0)
    inner_normals = mesh.boundary_normals(0)
    for comp in (0, 1):
        pressure_vals = -inner_pressure * inner_normals[:, comp]
        bcs.append(
            hyperelastic_traction_neumann_bc(
                bkd,
                inner_idx,
                inner_normals,
                D_matrices,
                stress_model,
                npts,
                comp,
                values=pressure_vals,
            )
        )

    # (e) Bottom Dirichlet v=0 (theta=0, v-component offset by npts)
    bottom_v_idx = bottom_idx + npts
    bcs.append(zero_dirichlet_bc(bkd, bottom_v_idx))

    # (f) Top Dirichlet u=0 (theta=pi/2, u-component = mesh indices)
    bcs.append(zero_dirichlet_bc(bkd, top_idx))

    physics.set_boundary_conditions(bcs)

    # Parameterization: E field -> Lame parameters
    param = create_hyperelastic_youngs_modulus_parameterization(
        bkd,
        basis,
        field_map,
        poisson_ratio,
    )

    init_state = bkd.zeros((2 * npts,))
    return SteadyForwardModel(
        physics,
        bkd,
        init_state,
        functional=functional,
        parameterization=param,
    )
