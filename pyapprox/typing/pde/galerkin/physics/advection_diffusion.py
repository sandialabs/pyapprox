"""Advection-diffusion-reaction physics for Galerkin FEM.

Supports two advection forms:

Non-conservative (default):
    du/dt + v . grad(u) = div(D * grad(u)) + R(u) + f
  Weak form:
    (w, v.grad(u)) + (grad(w), D*grad(u)) - (w, R(u)) = (w, f)

Conservative:
    du/dt + div(v * u) = div(D * grad(u)) + R(u) + f
  Weak form (after integration by parts of div(v*u)):
    -(v*u, grad(w)) + (grad(w), D*grad(u)) - (w, R(u)) = (w, f)

where:
    D = diffusivity (scalar or function of x)
    v = velocity field
    R(u) = reaction term (general nonlinear function of u)
           Positive R(u) = source/production (standard physics convention)
    f = forcing/source term
"""

from typing import Generic, Optional, Callable, List, Union, Tuple

import numpy as np
from scipy.sparse import csr_matrix

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.pde.galerkin.protocols.basis import GalerkinBasisProtocol
from pyapprox.typing.pde.galerkin.protocols.boundary import BoundaryConditionProtocol
from pyapprox.typing.pde.galerkin.physics.galerkin_base import GalerkinPhysicsBase
from pyapprox.typing.pde.galerkin.physics.helpers import ScalarMassAssembler

# Import skfem for assembly
try:
    from skfem import asm, LinearForm, BilinearForm
    from skfem.helpers import dot, grad
except ImportError:
    raise ImportError(
        "scikit-fem is required for Galerkin module. "
        "Install with: pip install scikit-fem"
    )


# Type alias for reaction functions
# R(x, u) -> values at quadrature points
ReactionFunc = Callable[[np.ndarray, np.ndarray], np.ndarray]
# R'(x, u) -> derivative w.r.t. u at quadrature points
ReactionDerivFunc = Callable[[np.ndarray, np.ndarray], np.ndarray]


class AdvectionDiffusionReaction(GalerkinPhysicsBase[Array]):
    """Advection-diffusion-reaction physics with general reaction term.

    Solves:
        du/dt + v . grad(u) = div(D * grad(u)) + R(u) + f

    where R(u) is a general (possibly nonlinear) reaction term.
    Positive R(u) represents a source/production term.

    The weak form is:
        (w, du/dt) + (w, v.grad(u)) + (grad(w), D*grad(u)) = (w, R(u)) + (w, f)

    For the residual F in M * du/dt = F:
        F_i = integral(f * phi_i) + integral(R(u) * phi_i)
              - integral(D * grad(u) . grad(phi_i))
              - integral(v . grad(u) * phi_i)

    In steady state (F=0), the linear system K*u = b is solved where the
    reaction term contributes to both the stiffness matrix (for linear R)
    or is evaluated at each Newton iteration (for nonlinear R).

    Parameters
    ----------
    basis : GalerkinBasisProtocol
        Finite element basis.
    diffusivity : float or Callable
        Diffusion coefficient D. If callable, takes coordinates (ndim, npts)
        and returns values (npts,).
    bkd : Backend
        Computational backend.
    velocity : Array or Callable, optional
        Velocity field v. If array, shape (ndim,). If callable, takes
        coordinates and returns (ndim, npts).
    reaction : float, Callable, or Tuple[Callable, Callable], optional
        Reaction term R(u). Can be:
        - float: Linear reaction R(u) = coeff * u (positive = source)
        - Callable: R(x, u) returning reaction values
        - Tuple[R, R']: (reaction function, derivative function)
          where R(x, u) and R'(x, u) are callables
        Default is None (no reaction).
    forcing : Callable, optional
        Forcing/source term f. Takes coordinates and returns (npts,).
        For time-dependent problems, takes (coordinates, time).
    boundary_conditions : List[BoundaryConditionProtocol], optional
        List of boundary conditions.
    conservative : bool, default=False
        If True, use conservative advection form div(v*u) with weak form
        bilinear term -(v*u, grad(w)). If False, use non-conservative
        form v.grad(u) with weak form bilinear term (w, v.grad(u)).

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.pde.galerkin.mesh import StructuredMesh1D
    >>> from pyapprox.typing.pde.galerkin.basis import LagrangeBasis
    >>> bkd = NumpyBkd()
    >>> mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
    >>> basis = LagrangeBasis(mesh, degree=1)
    >>>
    >>> # Pure diffusion
    >>> physics = AdvectionDiffusionReaction(
    ...     basis=basis, diffusivity=0.01, bkd=bkd
    ... )
    >>>
    >>> # Linear reaction R(u) = 2*u (source term)
    >>> physics = AdvectionDiffusionReaction(
    ...     basis=basis, diffusivity=0.01, reaction=2.0, bkd=bkd
    ... )
    >>>
    >>> # Nonlinear reaction R(u) = u^2
    >>> def R(x, u): return u**2
    >>> def R_prime(x, u): return 2*u
    >>> physics = AdvectionDiffusionReaction(
    ...     basis=basis, diffusivity=0.01, reaction=(R, R_prime), bkd=bkd
    ... )
    """

    def __init__(
        self,
        basis: GalerkinBasisProtocol[Array],
        diffusivity: Union[float, Callable],
        bkd: Backend[Array],
        velocity: Optional[Union[Array, Callable]] = None,
        reaction: Optional[Union[float, Callable, Tuple[Callable, Callable]]] = None,
        forcing: Optional[Callable] = None,
        boundary_conditions: Optional[List[BoundaryConditionProtocol[Array]]] = None,
        conservative: bool = False,
    ):
        super().__init__(basis, bkd, boundary_conditions)
        self._mass = ScalarMassAssembler(basis, bkd)

        # Store coefficients
        self._diffusivity = diffusivity
        self._velocity = velocity
        self._forcing = forcing
        self._conservative = conservative

        # Parse reaction term
        self._reaction_func: Optional[ReactionFunc] = None
        self._reaction_deriv: Optional[ReactionDerivFunc] = None
        self._reaction_is_linear = False
        self._reaction_coeff: Optional[float] = None

        if reaction is not None:
            if isinstance(reaction, (int, float)):
                # Linear reaction: R(u) = coeff * u, R'(u) = coeff
                self._reaction_coeff = float(reaction)
                self._reaction_is_linear = True
                self._reaction_func = lambda x, u: self._reaction_coeff * u
                self._reaction_deriv = lambda x, u: np.full_like(u, self._reaction_coeff)
            elif isinstance(reaction, tuple):
                # Tuple of (R, R')
                self._reaction_func, self._reaction_deriv = reaction
                self._reaction_is_linear = False
            elif callable(reaction):
                # Just the reaction function, no derivative provided
                self._reaction_func = reaction
                self._reaction_deriv = None
                self._reaction_is_linear = False
            else:
                raise TypeError(
                    f"reaction must be float, callable, or tuple of callables, "
                    f"got {type(reaction)}"
                )

        # Cache assembled matrices for linear problems
        self._stiffness_cached: Optional[Array] = None
        self._load_cached: Optional[Array] = None

    def is_linear(self) -> bool:
        """Return True if the problem is linear (linear or no reaction)."""
        return self._reaction_func is None or self._reaction_is_linear

    def _get_diffusivity(self, coords: np.ndarray) -> np.ndarray:
        """Get diffusivity values at given coordinates."""
        if callable(self._diffusivity):
            return self._diffusivity(coords)
        else:
            return np.full(coords.shape[1], self._diffusivity)

    def _get_velocity(self, coords: np.ndarray) -> np.ndarray:
        """Get velocity values at given coordinates."""
        if self._velocity is None:
            return np.zeros_like(coords)
        elif callable(self._velocity):
            return self._velocity(coords)
        else:
            # Constant velocity - broadcast to all points
            vel = self._bkd.to_numpy(self._velocity)
            return np.broadcast_to(vel[:, np.newaxis], coords.shape)

    def _get_forcing(self, coords: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Get forcing values at given coordinates."""
        if self._forcing is None:
            return np.zeros(coords.shape[1])
        else:
            # Try calling with time first, fall back to without
            try:
                return self._forcing(coords, time)
            except TypeError:
                return self._forcing(coords)

    def _assemble_stiffness(self, state: Array, time: float) -> Array:
        """Assemble stiffness matrix K.

        For the weak form, K includes:
        - Diffusion: (grad(w), D*grad(u))
        - Advection: (w, v.grad(u))
        - Linear reaction (if applicable): -(w, r*u) where R(u) = r*u

        Note: For nonlinear reaction, the contribution is handled in
        the residual and Jacobian separately.
        """
        # Check cache for linear problems
        if self._stiffness_cached is not None and self.is_linear():
            return self._stiffness_cached

        skfem_basis = self._basis.skfem_basis()

        # Get constant coefficients or prepare for callable
        diff_const = self._diffusivity if not callable(self._diffusivity) else None

        # For linear reaction, include in stiffness matrix
        react_coeff = self._reaction_coeff if self._reaction_is_linear else None

        def bilinear_form(u, v, w):
            # Diffusion coefficient
            if diff_const is not None:
                diff = diff_const
            else:
                diff = self._diffusivity(np.asarray(w.x))

            # Diffusion term: (grad(w), D*grad(u)) contributes D*grad(u).grad(v)
            result = diff * dot(grad(u), grad(v))

            # Linear reaction term: -(w, r*u) contributes -r*u*v
            # (negative because it's moved to LHS of weak form)
            if react_coeff is not None:
                result = result - react_coeff * u * v

            return result

        stiffness = asm(BilinearForm(bilinear_form), skfem_basis)

        # Add advection if present
        if self._velocity is not None:
            vel_np = self._bkd.to_numpy(self._velocity) if not callable(self._velocity) else None

            conservative = self._conservative

            def advection_form(u, v, w):
                if vel_np is not None:
                    vel = vel_np
                else:
                    vel = self._velocity(np.asarray(w.x))
                if conservative:
                    # Conservative: -(v*u, grad(w)) from div(v*u)
                    return -u * dot(vel, grad(v))
                else:
                    # Non-conservative: (w, v.grad(u))
                    return dot(vel, grad(u)) * v

            advection = asm(BilinearForm(advection_form), skfem_basis)
            stiffness = stiffness + advection

        # Cache if linear problem with constant coefficients
        if self.is_linear() and not callable(self._diffusivity) and not callable(self._velocity):
            self._stiffness_cached = stiffness

        return stiffness


    def _assemble_load(self, state: Array, time: float) -> Array:
        """Assemble load vector b.

        For the weak form, b includes:
        - Forcing: (w, f)
        - Nonlinear reaction: (w, R(u)) evaluated at current state

        For linear reaction, the term is already in the stiffness matrix.
        """
        skfem_basis = self._basis.skfem_basis()
        state_np = self._bkd.to_numpy(state)

        # Start with forcing contribution
        if self._forcing is None and (self._reaction_func is None or self._reaction_is_linear):
            # No forcing and no nonlinear reaction - use cached zero vector
            if self._load_cached is not None:
                return self._load_cached
            load_np = np.zeros(self.nstates())
            self._load_cached = self._bkd.asarray(load_np.astype(np.float64))
            return self._load_cached

        load_np = np.zeros(self.nstates())

        # Forcing contribution: (w, f)
        if self._forcing is not None:
            forcing_func = self._forcing
            current_time = time

            def forcing_form(v, w):
                x_np = np.asarray(w.x)
                x_shape = x_np.shape
                if len(x_shape) == 3:
                    ndim, nelem, nquad = x_shape
                    x_flat = x_np.reshape(ndim, -1)
                    try:
                        forc_flat = forcing_func(x_flat, current_time)
                    except TypeError:
                        forc_flat = forcing_func(x_flat)
                    forc = forc_flat.reshape(nelem, nquad)
                else:
                    try:
                        forc = forcing_func(x_np, current_time)
                    except TypeError:
                        forc = forcing_func(x_np)
                return forc * v

            load_np += asm(LinearForm(forcing_form), skfem_basis)

        # Nonlinear reaction contribution: (w, R(u))
        if self._reaction_func is not None and not self._reaction_is_linear:
            reaction_func = self._reaction_func

            # Interpolate state to get u values at quadrature points
            state_interp = skfem_basis.interpolate(state_np)

            def reaction_form(v, w):
                x_np = np.asarray(w.x)
                u_prev = w.u_prev  # Interpolated state values

                x_shape = x_np.shape
                if len(x_shape) == 3:
                    ndim, nelem, nquad = x_shape
                    x_flat = x_np.reshape(ndim, -1)
                    u_flat = u_prev.reshape(-1)
                    react_flat = reaction_func(x_flat, u_flat)
                    react = react_flat.reshape(nelem, nquad)
                else:
                    react = reaction_func(x_np, u_prev)
                return react * v

            load_np += asm(
                LinearForm(reaction_form), skfem_basis, u_prev=state_interp
            )

        return self._bkd.asarray(load_np.astype(np.float64))

    def _assemble_reaction_jacobian(self, state: Array, time: float) -> Array:
        """Assemble Jacobian contribution from nonlinear reaction.

        For R(u), the Jacobian term is: -(w, R'(u) * du)
        where du is the trial function.

        This returns the matrix J where J_ij = -integral(R'(u) * phi_j * phi_i)
        """
        if self._reaction_deriv is None or self._reaction_is_linear:
            # No nonlinear reaction or linear reaction (already in stiffness)
            return csr_matrix((self.nstates(), self.nstates()))

        skfem_basis = self._basis.skfem_basis()
        state_np = self._bkd.to_numpy(state)
        reaction_deriv = self._reaction_deriv

        # Interpolate state
        state_interp = skfem_basis.interpolate(state_np)

        def reaction_jacobian_form(u, v, w):
            x_np = np.asarray(w.x)
            u_prev = w.u_prev

            x_shape = x_np.shape
            if len(x_shape) == 3:
                ndim, nelem, nquad = x_shape
                x_flat = x_np.reshape(ndim, -1)
                u_flat = u_prev.reshape(-1)
                react_deriv_flat = reaction_deriv(x_flat, u_flat)
                react_deriv = react_deriv_flat.reshape(nelem, nquad)
            else:
                react_deriv = reaction_deriv(x_np, u_prev)

            # Negative because moved to LHS: -(w, R'(u)*du)
            return -react_deriv * u * v

        return asm(
            BilinearForm(reaction_jacobian_form), skfem_basis, u_prev=state_interp
        )

    def mass_matrix(self):
        """Return the scalar mass matrix."""
        return self._mass.mass_matrix()

    def mass_solve(self, rhs: Array) -> Array:
        """Solve M * x = rhs for x."""
        return self._mass.mass_solve(rhs)

    def spatial_residual(self, state: Array, time: float) -> Array:
        """Compute spatial residual F = b - K*u without Dirichlet enforcement.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Spatial residual. Shape: (nstates,)
        """
        stiffness = self._assemble_stiffness(state, time)
        load = self._assemble_load(state, time)
        stiffness = self._apply_bc_to_stiffness(stiffness, time)
        load = self._apply_bc_to_load(load, time)
        return load - stiffness @ state

    def spatial_jacobian(self, state: Array, time: float) -> Array:
        """Compute dF/du without Dirichlet enforcement.

        Includes nonlinear reaction Jacobian if applicable.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Jacobian dF/du. Shape: (nstates, nstates)
        """
        stiffness = self._assemble_stiffness(state, time)
        stiffness = self._apply_bc_to_stiffness(stiffness, time)
        jacobian = -stiffness
        if not self._reaction_is_linear and self._reaction_deriv is not None:
            jacobian = jacobian + self._assemble_reaction_jacobian(state, time)
        return jacobian

    def initial_condition(self, func: Callable) -> Array:
        """Create initial condition by interpolating a function.

        Parameters
        ----------
        func : Callable
            Function to interpolate. Takes coordinates (ndim, npts)
            and returns (npts,).

        Returns
        -------
        Array
            Initial DOF values. Shape: (nstates,)
        """
        return self._basis.interpolate(func)

    def __repr__(self) -> str:
        react_str = "None"
        if self._reaction_is_linear:
            react_str = f"linear({self._reaction_coeff})"
        elif self._reaction_func is not None:
            react_str = "nonlinear"
        return (
            f"AdvectionDiffusionReaction("
            f"nstates={self.nstates()}, "
            f"diffusivity={self._diffusivity}, "
            f"reaction={react_str})"
        )


# Backwards compatibility alias
LinearAdvectionDiffusionReaction = AdvectionDiffusionReaction
