"""Factory function for a 1D linear elastic bar forward model.

The 1D bar equation -d/dx(E(x)*du/dx) = f(x) is mathematically identical
to steady diffusion. This factory reuses AdvectionDiffusionReaction with
diffusion = E(x), providing a mechanics-oriented interface.
"""

from typing import Callable, Optional, Union

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.typing.pde.collocation.mesh import (
    AffineTransform1D,
    TransformedMesh1D,
)
from pyapprox.typing.pde.collocation.boundary import (
    constant_dirichlet_bc,
    flux_neumann_bc,
    zero_dirichlet_bc,
)
from pyapprox.typing.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.typing.forward_models.field_maps.protocol import (
    FieldMapProtocol,
)
from pyapprox.typing.forward_models.parameterizations.diffusion import (
    create_diffusion_parameterization,
)
from pyapprox.typing.pde.collocation.forward_models.steady import (
    SteadyForwardModel,
)


def create_linear_elastic_bar_1d(
    bkd: Backend[Array],
    npts: int,
    length: float,
    E_mean_field: Union[float, Array],
    forcing: Callable,
    field_map: FieldMapProtocol,
    traction: Optional[float] = None,
    dirichlet_left: float = 0.0,
    functional=None,
) -> SteadyForwardModel[Array]:
    """Create a steady 1D linear elastic bar forward model.

    Solves -d/dx(E(x)*du/dx) = f(x) on [0, length] with:
    - Left end (x=0): Dirichlet BC u = dirichlet_left (fixed displacement).
    - Right end (x=L): Neumann BC E*du/dx = traction (applied traction).

    If field_map is provided, the model is parameterized via KLE or similar
    and supports parameter jacobians for UQ.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    npts : int
        Number of Chebyshev collocation points.
    length : float
        Bar length L.
    E_mean_field : float or Array
        Mean Young's modulus. If float, constant across bar.
        If Array, shape (npts,) for spatially varying initial E.
    forcing : callable
        Body force f(x). Accepts time, returns Array of shape (npts,).
    traction : float, optional
        Applied traction at x=L. If None, uses zero Neumann (free end).
    dirichlet_left : float
        Prescribed displacement at x=0. Default 0.0 (fixed end).
    field_map : FieldMapProtocol
        Field map for E(x) parameterization (e.g. from
        create_lognormal_kle_field_map).
    functional : optional
        QoI functional. If None, returns full displacement field.

    Returns
    -------
    SteadyForwardModel
        Configured forward model.
    """
    transform = AffineTransform1D((0.0, length), bkd)
    mesh = TransformedMesh1D(npts, bkd, transform)
    basis = ChebyshevBasis1D(mesh, bkd)

    init_E = E_mean_field if not isinstance(E_mean_field, float) else E_mean_field
    physics = AdvectionDiffusionReaction(
        basis, bkd, diffusion=init_E, forcing=forcing,
    )

    # Left BC: prescribed displacement at x=0
    left_idx = mesh.boundary_indices(0)
    if dirichlet_left == 0.0:
        bc_left = zero_dirichlet_bc(bkd, left_idx)
    else:
        bc_left = constant_dirichlet_bc(bkd, left_idx, dirichlet_left)

    # Right BC: traction at x=L
    # flux(u).n = (-E*du/dx)*(+1) = -E*du/dx
    # We want E*du/dx = T, so flux.n = -T
    right_idx = mesh.boundary_indices(1)
    right_normals = mesh.boundary_normals(1)
    traction_val = 0.0 if traction is None else -traction
    bc_right = flux_neumann_bc(
        bkd, right_idx, right_normals, physics, traction_val,
    )

    physics.set_boundary_conditions([bc_left, bc_right])

    param = create_diffusion_parameterization(bkd, basis, field_map)

    init_state = bkd.zeros((npts,))
    return SteadyForwardModel(
        physics, bkd, init_state,
        functional=functional, parameterization=param,
    )
