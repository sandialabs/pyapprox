"""Factory functions for 1D diffusion forward models."""

from typing import Any, Callable, List, Optional, Tuple

from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.boundary import zero_dirichlet_bc
from pyapprox.pde.collocation.forward_models.steady import (
    SteadyForwardModel,
)
from pyapprox.pde.collocation.forward_models.transient import (
    TransientForwardModel,
)
from pyapprox.pde.collocation.mesh import (
    TransformedMesh1D,
    create_uniform_mesh_1d,
)
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.field_maps.basis_expansion import (
    BasisExpansion,
)
from pyapprox.pde.field_maps.protocol import (
    FieldMapProtocol,
)
from pyapprox.pde.parameterizations.diffusion import (
    create_diffusion_parameterization,
)
from pyapprox.pde.time.config import TimeIntegrationConfig
from pyapprox.util.backends.protocols import Array, Backend


def _build_field_map(
    bkd: Backend[Array],
    diffusion_base: Optional[float],
    basis_funs: Optional[List[Array]],
    field_map: Optional[FieldMapProtocol[Array]],
) -> FieldMapProtocol[Array]:
    """Resolve field map from either explicit field_map or basis_funs args."""
    if field_map is not None:
        if basis_funs is not None:
            raise ValueError("Cannot specify both field_map and basis_funs")
        return field_map
    if basis_funs is None:
        raise ValueError("Must specify either field_map or basis_funs")
    if diffusion_base is None:
        raise ValueError("diffusion_base is required when using basis_funs")
    return BasisExpansion(bkd, diffusion_base, basis_funs)


def create_steady_diffusion_1d(
    bkd: Backend[Array],
    npts: int,
    domain: Tuple[float, float],
    forcing: Callable[..., Any],
    diffusion_base: Optional[float] = None,
    basis_funs: Optional[List[Array]] = None,
    field_map: Optional[FieldMapProtocol[Array]] = None,
    bcs: Optional[list[Any]] = None,
    functional: object = None,
) -> SteadyForwardModel[Array]:
    """Create a steady-state 1D diffusion forward model.

    Parameterization is specified via either basis_funs (BasisExpansion)
    or field_map (any FieldMapProtocol, e.g. from create_lognormal_kle_field_map).

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    npts : int
        Number of collocation points.
    domain : tuple
        Domain bounds (left, right).
    forcing : callable
        Forcing term. Accepts time, returns Array of shape (npts,).
    diffusion_base : float, optional
        Base diffusion coefficient. Required when using basis_funs.
    basis_funs : list of Array, optional
        Basis functions for BasisExpansion parameterization. Each shape (npts,).
    field_map : FieldMapProtocol, optional
        Pre-built field map for parameterization.
    bcs : list, optional
        Boundary conditions. If None, uses homogeneous Dirichlet on both ends.
    functional : optional
        QoI functional. If None, returns full solution (identity).

    Returns
    -------
    SteadyForwardModel
        Configured forward model.
    """
    mesh = TransformedMesh1D(npts, bkd)
    basis = ChebyshevBasis1D(mesh, bkd)
    mesh_obj = create_uniform_mesh_1d(npts, domain, bkd)

    init_diffusion = diffusion_base if diffusion_base is not None else 1.0
    physics = AdvectionDiffusionReaction(
        basis,
        bkd,
        diffusion=init_diffusion,
        forcing=forcing,
    )

    if bcs is None:
        left_idx = mesh_obj.boundary_indices(0)
        right_idx = mesh_obj.boundary_indices(1)
        bcs = [
            zero_dirichlet_bc(bkd, left_idx),
            zero_dirichlet_bc(bkd, right_idx),
        ]
    physics.set_boundary_conditions(bcs)

    fm = _build_field_map(bkd, diffusion_base, basis_funs, field_map)
    param = create_diffusion_parameterization(bkd, basis, fm)

    init_state = bkd.zeros((npts,))
    return SteadyForwardModel(
        physics,
        bkd,
        init_state,
        functional=functional,
        parameterization=param,
    )


def create_transient_diffusion_1d(
    bkd: Backend[Array],
    npts: int,
    domain: Tuple[float, float],
    init_state_func: Callable[..., Any],
    time_config: TimeIntegrationConfig,
    forcing: Optional[Callable[..., Any]] = None,
    diffusion_base: Optional[float] = None,
    basis_funs: Optional[List[Array]] = None,
    field_map: Optional[FieldMapProtocol[Array]] = None,
    bcs: Optional[list[Any]] = None,
    functional: object = None,
) -> TransientForwardModel[Array]:
    """Create a transient 1D diffusion forward model.

    Parameterization is specified via either basis_funs (BasisExpansion)
    or field_map (any FieldMapProtocol, e.g. from create_lognormal_kle_field_map).

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    npts : int
        Number of collocation points.
    domain : tuple
        Domain bounds (left, right).
    init_state_func : callable
        Initial condition function. Accepts nodes Array, returns Array (npts,).
    time_config : TimeIntegrationConfig
        Time integration configuration.
    forcing : callable, optional
        Forcing term. Accepts time, returns Array (npts,).
    diffusion_base : float, optional
        Base diffusion coefficient. Required when using basis_funs.
    basis_funs : list of Array, optional
        Basis functions for BasisExpansion parameterization. Each shape (npts,).
    field_map : FieldMapProtocol, optional
        Pre-built field map for parameterization.
    bcs : list, optional
        Boundary conditions. If None, uses homogeneous Dirichlet on both ends.
    functional : optional
        QoI functional. If None, returns all states at final time.

    Returns
    -------
    TransientForwardModel
        Configured forward model.
    """
    mesh = TransformedMesh1D(npts, bkd)
    basis = ChebyshevBasis1D(mesh, bkd)
    mesh_obj = create_uniform_mesh_1d(npts, domain, bkd)
    nodes = basis.nodes()

    init_diffusion = diffusion_base if diffusion_base is not None else 1.0
    physics = AdvectionDiffusionReaction(
        basis,
        bkd,
        diffusion=init_diffusion,
        forcing=forcing,
    )

    if bcs is None:
        left_idx = mesh_obj.boundary_indices(0)
        right_idx = mesh_obj.boundary_indices(1)
        bcs = [
            zero_dirichlet_bc(bkd, left_idx),
            zero_dirichlet_bc(bkd, right_idx),
        ]
    physics.set_boundary_conditions(bcs)

    fm = _build_field_map(bkd, diffusion_base, basis_funs, field_map)
    param = create_diffusion_parameterization(bkd, basis, fm)

    init_state = init_state_func(nodes)
    return TransientForwardModel(
        physics,
        bkd,
        init_state,
        time_config,
        functional=functional,
        parameterization=param,
    )
