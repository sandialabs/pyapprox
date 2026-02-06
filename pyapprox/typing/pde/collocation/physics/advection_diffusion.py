"""Advection-Diffusion-Reaction physics for spectral collocation.

Implements the scalar ADR equation:
    du/dt = div(D * grad(u)) - v . grad(u) + r*u + f

where:
- D: diffusion coefficient (scalar or field)
- v: velocity field (vector)
- r: reaction coefficient (scalar or field)
- f: forcing/source term (field)
"""

from typing import Generic, List, Optional, Callable, Tuple

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.pde.collocation.protocols.basis import (
    TensorProductBasisProtocol,
)
from pyapprox.typing.pde.collocation.physics.base import AbstractScalarPhysics
from pyapprox.typing.pde.collocation.operators.field import (
    Field,
    input_field,
    constant_field,
)
from pyapprox.typing.pde.collocation.operators.differential import (
    Laplacian,
    Gradient,
    Divergence,
)


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
        self._is_variable_diffusion = not isinstance(self._diffusion_value, (int, float))
        if isinstance(self._diffusion_value, (int, float)):
            self._diffusion_array = bkd.full((npts,), float(self._diffusion_value))
            self._has_diffusion = self._diffusion_value != 0.0
        else:
            self._diffusion_array = self._diffusion_value
            self._has_diffusion = True  # Assume variable diffusion is non-zero

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
            if not self._is_variable_diffusion:
                # Constant diffusion: D * laplacian(u)
                D2_u = self._D2_matrix @ state
                residual = residual + float(self._diffusion_value) * D2_u
            else:
                # Variable diffusion: div(D * grad(u))
                # = D * laplacian(u) + grad(D) . grad(u)
                D2_u = self._D2_matrix @ state
                residual = residual + self._diffusion_array * D2_u
                # Add grad(D) . grad(u) term
                for dim in range(self._basis.ndim()):
                    D_dim = self._D_matrices[dim]
                    grad_D_dim = D_dim @ self._diffusion_array
                    grad_u_dim = D_dim @ state
                    residual = residual + grad_D_dim * grad_u_dim

        # Advection term: -v . grad(u)
        if self._velocity is not None:
            for dim in range(self._basis.ndim()):
                D_dim = self._D_matrices[dim]
                grad_u_dim = D_dim @ state
                residual = residual - self._velocity[dim] * grad_u_dim

        # Reaction term: r * u
        if self._reaction_value != 0.0:
            residual = residual + self._reaction_array * state

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
            if not self._is_variable_diffusion:
                jacobian = jacobian + float(self._diffusion_value) * self._D2_matrix
            else:
                # Variable diffusion: diag(D) @ laplacian + grad(D) terms
                D_diag = bkd.diag(self._diffusion_array)
                jacobian = jacobian + D_diag @ self._D2_matrix
                # Add grad(D) . grad term: sum_i diag(grad_D_i) @ D_i
                for dim in range(self._basis.ndim()):
                    D_dim = self._D_matrices[dim]
                    grad_D_dim = D_dim @ self._diffusion_array
                    jacobian = jacobian + bkd.diag(grad_D_dim) @ D_dim

        # Advection term Jacobian: d/du[-v . grad(u)] = -sum_i diag(v_i) @ D_i
        if self._velocity is not None:
            for dim in range(self._basis.ndim()):
                D_dim = self._D_matrices[dim]
                v_diag = bkd.diag(self._velocity[dim])
                jacobian = jacobian - v_diag @ D_dim

        # Reaction term Jacobian: d/du[r * u] = diag(r)
        if self._reaction_value != 0.0:
            jacobian = jacobian + bkd.diag(self._reaction_array)

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


class AdvectionDiffusionReactionWithParam(AdvectionDiffusionReaction[Array]):
    """ADR physics with parameterized diffusion coefficient.

    The diffusion coefficient is parameterized as:
        D(x) = D_base + sum_i param_i * basis_i(x)

    This enables adjoint-based sensitivity analysis and optimization.

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        Collocation basis.
    bkd : Backend
        Computational backend.
    diffusion_base : float
        Base diffusion coefficient.
    diffusion_basis_funs : List[Array]
        Basis functions for parameterized diffusion. Each has shape: (npts,)
    velocity : List[Array], optional
        Velocity field components.
    reaction : float or Array, optional
        Reaction coefficient.
    forcing : Callable[[float], Array] or Array, optional
        Forcing term.
    initial_condition : Callable[[Array], Array], optional
        Function that takes parameters and returns initial condition.
        If None, initial condition has zero parameter Jacobian.
    """

    def __init__(
        self,
        basis: TensorProductBasisProtocol[Array],
        bkd: Backend[Array],
        diffusion_base: float,
        diffusion_basis_funs: List[Array],
        velocity: Optional[List[Array]] = None,
        reaction: Optional[float] = None,
        forcing: Optional[Callable[[float], Array]] = None,
        initial_condition: Optional[Callable[[Array], Tuple[Array, Array]]] = None,
    ):
        # Initialize with base diffusion
        super().__init__(basis, bkd, diffusion_base, velocity, reaction, forcing)

        self._diffusion_base = diffusion_base
        self._diffusion_basis_funs = diffusion_basis_funs
        self._nparams = len(diffusion_basis_funs)
        self._param = bkd.zeros((self._nparams,))
        self._initial_condition_func = initial_condition

        # Precompute gradient of basis functions
        self._grad_basis_funs = []
        for basis_fun in diffusion_basis_funs:
            grad_basis = []
            for dim in range(basis.ndim()):
                grad_basis.append(self._D_matrices[dim] @ basis_fun)
            self._grad_basis_funs.append(grad_basis)

    def nparams(self) -> int:
        """Return number of parameters."""
        return self._nparams

    def set_param(self, param: Array) -> None:
        """Set parameter values.

        Parameters
        ----------
        param : Array
            Parameter vector. Shape: (nparams,)
        """
        bkd = self._bkd
        if param.shape[0] != self._nparams:
            raise ValueError(
                f"param length {param.shape[0]} != nparams {self._nparams}"
            )
        self._param = param

        # Update diffusion array: D = D_base + sum_i param_i * basis_i
        npts = self.npts()
        self._diffusion_array = bkd.full((npts,), self._diffusion_base)
        for i in range(self._nparams):
            self._diffusion_array = (
                self._diffusion_array + float(param[i]) * self._diffusion_basis_funs[i]
            )
        # Use variable diffusion path
        self._is_variable_diffusion = True
        self._has_diffusion = True

    def param_jacobian(self, state: Array, time: float) -> Array:
        """Compute parameter Jacobian df/dp.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (npts,)
        time : float
            Current time.

        Returns
        -------
        Array
            Parameter Jacobian. Shape: (npts, nparams)
        """
        bkd = self._bkd
        npts = self.npts()

        # Parameter Jacobian: df/dp_i where D = D_base + sum_j p_j * phi_j
        # Diffusion residual: div(D * grad(u))
        # For constant basis functions:
        #   d/dp_i[div(D * grad(u))] = div(phi_i * grad(u))
        #                            = phi_i * laplacian(u) + grad(phi_i) . grad(u)

        param_jac = bkd.zeros((npts, self._nparams))

        laplacian_u = self._D2_matrix @ state
        grad_u = [self._D_matrices[dim] @ state for dim in range(self._basis.ndim())]

        for i in range(self._nparams):
            phi_i = self._diffusion_basis_funs[i]
            # phi_i * laplacian(u)
            term1 = phi_i * laplacian_u
            # grad(phi_i) . grad(u)
            term2 = bkd.zeros((npts,))
            for dim in range(self._basis.ndim()):
                grad_phi_i_dim = self._grad_basis_funs[i][dim]
                term2 = term2 + grad_phi_i_dim * grad_u[dim]

            col = term1 + term2
            for j in range(npts):
                param_jac[j, i] = col[j]

        return param_jac

    def initial_param_jacobian(self) -> Array:
        """Compute initial condition parameter Jacobian d(u_0)/dp.

        Returns
        -------
        Array
            Initial condition Jacobian. Shape: (npts, nparams)
        """
        bkd = self._bkd
        npts = self.npts()

        if self._initial_condition_func is None:
            # Initial condition does not depend on parameters
            return bkd.zeros((npts, self._nparams))

        # Get initial condition and its Jacobian
        _, ic_jac = self._initial_condition_func(self._param)
        return ic_jac


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
    return AdvectionDiffusionReaction(
        basis, bkd, diffusion=diffusion, forcing=forcing
    )


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
