"""Advection-Diffusion-Reaction physics for spectral collocation.

Implements the scalar ADR equation:
    du/dt = div(D * grad(u)) - v . grad(u) + r*u + f

where:
- D: diffusion coefficient (scalar or field)
- v: velocity field (vector)
- r: reaction coefficient (scalar or field)
- f: forcing/source term (field)
"""

from typing import Callable, List, Optional

from pyapprox.pde.collocation.operators.differential import (
    Divergence,
    Gradient,
    Laplacian,
)
from pyapprox.pde.collocation.physics.base import AbstractScalarPhysics
from pyapprox.pde.collocation.protocols.basis import (
    TensorProductBasisProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class AdvectionDiffusionReaction(AbstractScalarPhysics[Array]):
    """Advection-Diffusion-Reaction physics.

    Implements the scalar ADR equation:
        du/dt = div(D * grad(u)) - v . grad(u) + r*u + f

    In conservative form:
        du/dt = -div(flux) + r*u + f
    where flux = -D * grad(u) + u * v

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        Collocation basis (provides nodes, derivative matrices).
    bkd : Backend
        Computational backend.
    diffusion : float or Array, optional
        Diffusion coefficient. If Array, shape: (npts,). Default: 0.0
    velocity : List[Array], optional
        Velocity field components. Each has shape: (npts,). Default: None (no advection)
    reaction : float or Array, optional
        Reaction coefficient. If Array, shape: (npts,). Default: 0.0
    forcing : Callable[[float], Array] or Array, optional
        Forcing term. If callable, takes time and returns (npts,) array.
        If Array, shape: (npts,). Default: None (no forcing)
    """

    def __init__(
        self,
        basis: TensorProductBasisProtocol[Array],
        bkd: Backend[Array],
        diffusion: Optional[float] = None,
        velocity: Optional[List[Array]] = None,
        reaction: Optional[float] = None,
        forcing: Optional[Callable[[float], Array]] = None,
    ):
        super().__init__(basis, bkd)
        self._laplacian = Laplacian(basis, bkd)
        self._gradient = Gradient(basis, bkd)
        self._divergence = Divergence(basis, bkd)

        npts = basis.npts()
        ndim = basis.ndim()

        # Store diffusion coefficient
        self._diffusion_value = diffusion if diffusion is not None else 0.0
        self._is_variable_diffusion = not isinstance(
            self._diffusion_value, (int, float)
        )
        if isinstance(self._diffusion_value, (int, float)):
            self._diffusion_array = bkd.full((npts,), float(self._diffusion_value))
            self._has_diffusion = self._diffusion_value != 0.0
        else:
            self._diffusion_array = self._diffusion_value
            self._has_diffusion = True  # Assume variable diffusion is non-zero
        self._diffusion_func: Optional[Callable[[float], Array]] = None

        # Store velocity field
        if velocity is not None:
            if len(velocity) != ndim:
                raise ValueError(
                    f"velocity must have {ndim} components, got {len(velocity)}"
                )
            self._velocity = velocity
        else:
            self._velocity = None

        # Store reaction coefficient
        self._reaction_value = reaction if reaction is not None else 0.0
        if isinstance(self._reaction_value, (int, float)):
            self._reaction_array = bkd.full((npts,), float(self._reaction_value))
        else:
            self._reaction_array = self._reaction_value
        self._has_reaction = reaction is not None and not (
            isinstance(reaction, (int, float)) and reaction == 0.0
        )
        self._reaction_func: Optional[Callable[[float], Array]] = None

        # Store forcing function
        self._forcing_func = forcing

        # Precompute derivative matrices for efficiency
        self._D_matrices = [basis.derivative_matrix(1, dim) for dim in range(ndim)]
        self._D2_matrix = self._compute_laplacian_matrix()

    def _compute_laplacian_matrix(self) -> Array:
        """Compute the Laplacian matrix (sum of second derivatives)."""
        bkd = self._bkd
        ndim = self._basis.ndim()
        npts = self._basis.npts()
        D2 = bkd.zeros((npts, npts))
        for dim in range(ndim):
            D2_dim = self._basis.derivative_matrix(2, dim)
            D2 = D2 + D2_dim
        return D2

    def _get_forcing(self, time: float) -> Array:
        """Get forcing array at given time."""
        npts = self.npts()
        if self._forcing_func is None:
            return self._bkd.zeros((npts,))
        if callable(self._forcing_func):
            return self._forcing_func(time)
        return self._forcing_func

    def set_diffusion(self, func: Callable[[float], Array]) -> None:
        """Set diffusion coefficient as time-dependent callable."""
        self._diffusion_func = func
        self._is_variable_diffusion = True
        self._has_diffusion = True

    def set_forcing(self, func: Callable[[float], Array]) -> None:
        """Set forcing term as time-dependent callable."""
        self._forcing_func = func

    def set_reaction(self, func: Callable[[float], Array]) -> None:
        """Set reaction coefficient as time-dependent callable."""
        self._reaction_func = func
        self._has_reaction = True

    def _get_diffusion(self, time: float) -> Array:
        """Get diffusion array at given time."""
        if self._diffusion_func is not None:
            self._diffusion_array = self._diffusion_func(time)
        return self._diffusion_array

    def _get_reaction(self, time: float) -> Array:
        """Get reaction array at given time."""
        if self._reaction_func is not None:
            self._reaction_array = self._reaction_func(time)
        return self._reaction_array

    def residual_diffusion_sensitivity(
        self,
        state: Array,
        time: float,
        delta_D: Array,
        grad_delta_D: List[Array],
    ) -> Array:
        """Compute d(residual)/d(D_field) applied to perturbation delta_D.

        Returns delta_D * laplacian(u) + grad(delta_D) . grad(u).

        Parameters
        ----------
        state : Array
            Solution state. Shape: (npts,)
        time : float
            Current time.
        delta_D : Array
            Perturbation of diffusion field. Shape: (npts,)
        grad_delta_D : List[Array]
            Gradient of perturbation, one per dimension. Each shape: (npts,)

        Returns
        -------
        Array
            Sensitivity. Shape: (npts,)
        """
        lap_u = self._D2_matrix @ state
        result = delta_D * lap_u
        for dim in range(self._basis.ndim()):
            grad_u_dim = self._D_matrices[dim] @ state
            result = result + grad_delta_D[dim] * grad_u_dim
        return result

    def residual_reaction_sensitivity(self, state: Array, time: float) -> Array:
        """Compute d(residual)/d(r_field) pointwise = state.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (npts,)
        time : float
            Current time.

        Returns
        -------
        Array
            Sensitivity. Shape: (npts,)
        """
        return state

    def residual(self, state: Array, time: float) -> Array:
        """Compute spatial residual f(u, t).

        For transient problems: du/dt = residual(u, t)

        Parameters
        ----------
        state : Array
            Solution state. Shape: (npts,)
        time : float
            Current time.

        Returns
        -------
        Array
            Residual. Shape: (npts,)
        """
        bkd = self._bkd
        npts = self.npts()

        # Initialize residual
        residual = bkd.zeros((npts,))

        # Diffusion term: div(D * grad(u)) = D * laplacian(u) for constant D
        # For variable D: div(D * grad(u))
        if self._has_diffusion:
            diff_array = self._get_diffusion(time)
            if not self._is_variable_diffusion:
                # Constant diffusion: D * laplacian(u)
                D2_u = self._D2_matrix @ state
                residual = residual + float(self._diffusion_value) * D2_u
            else:
                # Variable diffusion: div(D * grad(u))
                # = D * laplacian(u) + grad(D) . grad(u)
                D2_u = self._D2_matrix @ state
                residual = residual + diff_array * D2_u
                # Add grad(D) . grad(u) term
                for dim in range(self._basis.ndim()):
                    D_dim = self._D_matrices[dim]
                    grad_D_dim = D_dim @ diff_array
                    grad_u_dim = D_dim @ state
                    residual = residual + grad_D_dim * grad_u_dim

        # Advection term: -v . grad(u)
        if self._velocity is not None:
            for dim in range(self._basis.ndim()):
                D_dim = self._D_matrices[dim]
                grad_u_dim = D_dim @ state
                residual = residual - self._velocity[dim] * grad_u_dim

        # Reaction term: r * u
        if self._has_reaction:
            react_array = self._get_reaction(time)
            residual = residual + react_array * state

        # Forcing term
        residual = residual + self._get_forcing(time)

        return residual

    def jacobian(self, state: Array, time: float) -> Array:
        """Compute state Jacobian df/du.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (npts,)
        time : float
            Current time.

        Returns
        -------
        Array
            Jacobian matrix. Shape: (npts, npts)
        """
        bkd = self._bkd
        npts = self.npts()

        # Initialize Jacobian
        jacobian = bkd.zeros((npts, npts))

        # Diffusion term Jacobian: d/du[D * laplacian(u)] = D * laplacian
        if self._has_diffusion:
            diff_array = self._get_diffusion(time)
            if not self._is_variable_diffusion:
                jacobian = jacobian + float(self._diffusion_value) * self._D2_matrix
            else:
                # Variable diffusion: diag(D) @ laplacian + grad(D) terms
                D_diag = bkd.diag(diff_array)
                jacobian = jacobian + D_diag @ self._D2_matrix
                # Add grad(D) . grad term: sum_i diag(grad_D_i) @ D_i
                for dim in range(self._basis.ndim()):
                    D_dim = self._D_matrices[dim]
                    grad_D_dim = D_dim @ diff_array
                    jacobian = jacobian + bkd.diag(grad_D_dim) @ D_dim

        # Advection term Jacobian: d/du[-v . grad(u)] = -sum_i diag(v_i) @ D_i
        if self._velocity is not None:
            for dim in range(self._basis.ndim()):
                D_dim = self._D_matrices[dim]
                v_diag = bkd.diag(self._velocity[dim])
                jacobian = jacobian - v_diag @ D_dim

        # Reaction term Jacobian: d/du[r * u] = diag(r)
        if self._has_reaction:
            react_array = self._get_reaction(time)
            jacobian = jacobian + bkd.diag(react_array)

        return jacobian

    def compute_flux(self, state: Array) -> list:
        """Compute total conservative flux at all mesh points.

        Returns the sum of diffusive and advective flux:
          flux_d = -D * du/dx_d + v_d * u  (for each dimension d)

        For pure diffusion (no velocity): flux_d = -D * du/dx_d

        Parameters
        ----------
        state : Array
            Solution state. Shape: (npts,)

        Returns
        -------
        List[Array]
            Flux components, one per dimension. Each shape: (npts,)
        """
        ndim = self._basis.ndim()
        flux = []
        for dim in range(ndim):
            D_dim = self._D_matrices[dim]
            grad_u_dim = D_dim @ state
            # Diffusive flux: -D * du/dx_d
            if self._is_variable_diffusion:
                flux_dim = -self._diffusion_array * grad_u_dim
            else:
                flux_dim = -float(self._diffusion_value) * grad_u_dim
            # Advective flux: v_d * u
            if self._velocity is not None:
                flux_dim = flux_dim + self._velocity[dim] * state
            flux.append(flux_dim)
        return flux

    def compute_flux_jacobian(self, state: Array) -> list:
        """Compute Jacobian of total conservative flux w.r.t. state.

        d(flux_d)/du = -D * D_d + diag(v_d)
        For variable D: -diag(D) @ D_d + diag(v_d)

        Parameters
        ----------
        state : Array
            Solution state. Shape: (npts,)

        Returns
        -------
        List[Array]
            Jacobian of each flux component. Each shape: (npts, npts)
        """
        bkd = self._bkd
        ndim = self._basis.ndim()
        flux_jacs = []
        for dim in range(ndim):
            D_dim = self._D_matrices[dim]
            if self._is_variable_diffusion:
                jac_dim = -bkd.diag(self._diffusion_array) @ D_dim
            else:
                jac_dim = -float(self._diffusion_value) * D_dim
            if self._velocity is not None:
                jac_dim = jac_dim + bkd.diag(self._velocity[dim])
            flux_jacs.append(jac_dim)
        return flux_jacs

    def compute_interface_flux(
        self, state: Array, boundary_indices: Array, normal: Array
    ) -> Array:
        """Compute diffusive flux at boundary for DtN domain decomposition.

        Computes D * grad(u) · n at the specified boundary points.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (npts,)
        boundary_indices : Array
            Mesh indices at interface. Shape: (nboundary,)
        normal : Array
            Outward unit normal. Shape: (ndim,)

        Returns
        -------
        Array
            Flux at boundary points. Shape: (nboundary,)
        """
        ndim = self._basis.ndim()
        nboundary = boundary_indices.shape[0]

        # Compute grad(u) · n at boundary points
        flux = self._bkd.zeros((nboundary,))
        for dim in range(ndim):
            D_dim = self._D_matrices[dim]
            grad_u_dim = D_dim @ state
            flux = flux + grad_u_dim[boundary_indices] * float(normal[dim])

        # Scale by diffusion coefficient
        if self._is_variable_diffusion:
            # Variable diffusion: multiply by D(x) at boundary
            flux = self._diffusion_array[boundary_indices] * flux
        else:
            # Constant diffusion: multiply by scalar D
            flux = float(self._diffusion_value) * flux

        return flux


def create_steady_diffusion(
    basis: TensorProductBasisProtocol[Array],
    bkd: Backend[Array],
    diffusion: float,
    forcing: Optional[Callable[[float], Array]] = None,
) -> AdvectionDiffusionReaction[Array]:
    """Create steady diffusion physics (Poisson equation).

    Solves: -D * laplacian(u) = f  or  D * laplacian(u) + f = 0

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        Collocation basis.
    bkd : Backend
        Computational backend.
    diffusion : float
        Diffusion coefficient (positive).
    forcing : Callable or Array, optional
        Source term.

    Returns
    -------
    AdvectionDiffusionReaction
        Physics for steady diffusion.
    """
    return AdvectionDiffusionReaction(basis, bkd, diffusion=diffusion, forcing=forcing)


def create_advection_diffusion(
    basis: TensorProductBasisProtocol[Array],
    bkd: Backend[Array],
    diffusion: float,
    velocity: List[Array],
    forcing: Optional[Callable[[float], Array]] = None,
) -> AdvectionDiffusionReaction[Array]:
    """Create advection-diffusion physics.

    Solves: du/dt = D * laplacian(u) - v . grad(u) + f

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        Collocation basis.
    bkd : Backend
        Computational backend.
    diffusion : float
        Diffusion coefficient.
    velocity : List[Array]
        Velocity field components.
    forcing : Callable or Array, optional
        Source term.

    Returns
    -------
    AdvectionDiffusionReaction
        Advection-diffusion physics.
    """
    return AdvectionDiffusionReaction(
        basis, bkd, diffusion=diffusion, velocity=velocity, forcing=forcing
    )
