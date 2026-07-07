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

from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from skfem.assembly.form.form import FormExtraParams
    from skfem.element.discrete_field import DiscreteField

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from pyapprox.pde.galerkin.physics.galerkin_base import GalerkinPhysicsBase
from pyapprox.pde.galerkin.physics.helpers import ScalarMassAssembler
from pyapprox.pde.galerkin.protocols.basis import GalerkinBasisProtocol
from pyapprox.pde.galerkin.protocols.boundary import BoundaryConditionProtocol
from pyapprox.util.backends.protocols import Array, Backend

# Import skfem for assembly
try:
    from skfem import BilinearForm, LinearForm, asm
    from skfem.helpers import dot, grad
except ImportError:
    from pyapprox.util.optional_deps import import_optional_dependency

    import_optional_dependency(
        "skfem", feature_name="Galerkin module", extra_name="fem"
    )


# Type alias for reaction functions
# R(x, u) -> values at quadrature points
ReactionFunc = Callable[[np.ndarray, np.ndarray], np.ndarray]
# R'(x, u) -> derivative w.r.t. u at quadrature points
ReactionDerivFunc = Callable[[np.ndarray, np.ndarray], np.ndarray]


class _DiffusionReactionKernel:
    """Picklable kernel for the diffusion + linear-reaction form.

    Module-level callable class (not a closure) so the forms returned
    by ``stiffness_forms`` remain picklable — the _ExpTransform
    precedent in pde.field_maps.
    """

    __name__ = "diffusion_reaction"

    def __init__(
        self,
        diff_const: Optional[float],
        diff_callable: Optional[
            Callable[[NDArray[np.floating[Any]]], NDArray[np.floating[Any]]]
        ],
        react_coeff: Optional[float],
    ) -> None:
        self._diff_const = diff_const
        self._diff_callable = diff_callable
        self._react_coeff = react_coeff

    def __call__(
        self,
        u: "DiscreteField",
        v: "DiscreteField",
        w: "FormExtraParams",
    ) -> np.ndarray:
        # Diffusion coefficient
        diff: Union[float, NDArray[np.floating[Any]]]
        if self._diff_const is not None:
            diff = self._diff_const
        else:
            assert self._diff_callable is not None
            diff = self._diff_callable(np.asarray(w.x))

        # Diffusion term: (grad(w), D*grad(u)) contributes D*grad(u).grad(v)
        result: NDArray[np.floating[Any]] = diff * dot(grad(u), grad(v))

        # Linear reaction term: -(w, r*u) contributes -r*u*v
        # (negative because it's moved to LHS of weak form)
        if self._react_coeff is not None:
            result = result - self._react_coeff * u * v

        return result


class _AdvectionKernel:
    """Picklable kernel for the advection form."""

    __name__ = "advection"

    def __init__(
        self,
        vel_np: Optional[NDArray[np.floating[Any]]],
        vel_callable: Optional[
            Callable[[NDArray[np.floating[Any]]], NDArray[np.floating[Any]]]
        ],
        conservative: bool,
    ) -> None:
        self._vel_np = vel_np
        self._vel_callable = vel_callable
        self._conservative = conservative

    def __call__(
        self,
        u: "DiscreteField",
        v: "DiscreteField",
        w: "FormExtraParams",
    ) -> np.ndarray:
        if self._vel_np is not None:
            vel = self._vel_np
        else:
            assert self._vel_callable is not None
            vel = self._vel_callable(np.asarray(w.x))
        if self._conservative:
            # Conservative: -(v*u, grad(w)) from div(v*u)
            ret: NDArray[np.floating[Any]] = -u * dot(vel, grad(v))
            return ret
        else:
            # Non-conservative: (w, v.grad(u))
            ret2: NDArray[np.floating[Any]] = dot(vel, grad(u)) * v
            return ret2


class _ForcingKernel:
    """Picklable kernel for the forcing load form (w, f)."""

    __name__ = "forcing"

    def __init__(
        self, forcing_func: Callable[..., Any], time: float
    ) -> None:
        self._forcing_func = forcing_func
        self._time = time

    def __call__(
        self, v: "DiscreteField", w: "FormExtraParams"
    ) -> np.ndarray:
        x_np = np.asarray(w.x)
        x_shape = x_np.shape
        if len(x_shape) == 3:
            ndim, nelem, nquad = x_shape
            x_flat = x_np.reshape(ndim, -1)
            try:
                forc_flat = self._forcing_func(x_flat, self._time)
            except TypeError:
                forc_flat = self._forcing_func(x_flat)
            forc = forc_flat.reshape(nelem, nquad)
        else:
            try:
                forc = self._forcing_func(x_np, self._time)
            except TypeError:
                forc = self._forcing_func(x_np)
        ret: NDArray[np.floating[Any]] = forc * v
        return ret


class _ReactionKernel:
    """Picklable kernel for the nonlinear reaction load (w, R(u)).

    Assembled with the interpolated state as the ``u_prev`` form
    parameter.
    """

    __name__ = "reaction"

    def __init__(self, reaction_func: ReactionFunc) -> None:
        self._reaction_func = reaction_func

    def __call__(
        self, v: "DiscreteField", w: "FormExtraParams"
    ) -> np.ndarray:
        x_np = np.asarray(w.x)
        u_prev = w.u_prev  # Interpolated state values

        x_shape = x_np.shape
        if len(x_shape) == 3:
            ndim, nelem, nquad = x_shape
            x_flat = x_np.reshape(ndim, -1)
            u_flat = np.asarray(u_prev).reshape(-1)
            react_flat = self._reaction_func(x_flat, u_flat)
            react = react_flat.reshape(nelem, nquad)
        else:
            react = self._reaction_func(x_np, u_prev)
        ret: NDArray[np.floating[Any]] = react * v
        return ret


class _ReactionJacobianKernel:
    """Picklable kernel for the reaction Jacobian (w, R'(u)*du).

    Assembled with the interpolated state as the ``u_prev`` form
    parameter.
    """

    __name__ = "reaction_jacobian"

    def __init__(self, reaction_deriv: ReactionDerivFunc) -> None:
        self._reaction_deriv = reaction_deriv

    def __call__(
        self,
        u: "DiscreteField",
        v: "DiscreteField",
        w: "FormExtraParams",
    ) -> np.ndarray:
        x_np = np.asarray(w.x)
        u_prev = w.u_prev

        x_shape = x_np.shape
        if len(x_shape) == 3:
            ndim, nelem, nquad = x_shape
            x_flat = x_np.reshape(ndim, -1)
            u_flat = np.asarray(u_prev).reshape(-1)
            react_deriv_flat = self._reaction_deriv(x_flat, u_flat)
            react_deriv = react_deriv_flat.reshape(nelem, nquad)
        else:
            react_deriv = self._reaction_deriv(x_np, u_prev)

        # Derivative of the load term (w, R(u)) with respect to the
        # state: F = load - K*u, so dF/du gains +(w, R'(u)*du)
        ret: NDArray[np.floating[Any]] = react_deriv * u * v
        return ret


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
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.pde.galerkin.mesh import StructuredMesh1D
    >>> from pyapprox.pde.galerkin.basis import LagrangeBasis
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
        diffusivity: Union[float, Callable[..., Any]],
        bkd: Backend[Array],
        velocity: Optional[Union[Array, Callable[..., Any]]] = None,
        reaction: Optional[
            Union[
                float, Callable[..., Any], Tuple[Callable[..., Any], Callable[..., Any]]
            ]
        ] = None,
        forcing: Optional[Callable[..., Any]] = None,
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
                self._reaction_deriv = lambda x, u: np.full_like(
                    u, self._reaction_coeff
                )
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
            return np.asarray(self._diffusivity(coords))
        else:
            return np.full(coords.shape[1], self._diffusivity)

    def _get_velocity(self, coords: np.ndarray) -> np.ndarray:
        """Get velocity values at given coordinates."""
        if self._velocity is None:
            return np.zeros_like(coords)
        elif callable(self._velocity):
            return np.asarray(self._velocity(coords))
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
                return np.asarray(self._forcing(coords, time))
            except TypeError:
                return np.asarray(self._forcing(coords))

    def _diffusion_reaction_form(self) -> "BilinearForm":
        """Bilinear form for diffusion plus linear reaction."""
        # Get constant coefficients or prepare for callable
        diff_const = self._diffusivity if not callable(self._diffusivity) else None

        # For linear reaction, include in stiffness matrix
        react_coeff = self._reaction_coeff if self._reaction_is_linear else None

        # Capture callable diffusivity for use in the kernel
        diff_callable = self._diffusivity if callable(self._diffusivity) else None

        return BilinearForm(
            _DiffusionReactionKernel(diff_const, diff_callable, react_coeff)
        )

    def _advection_form(self) -> Optional["BilinearForm"]:
        """Bilinear form for advection, or None when velocity is absent."""
        if self._velocity is None:
            return None

        vel_np = (
            self._bkd.to_numpy(self._velocity)
            if not callable(self._velocity)
            else None
        )

        vel_callable = self._velocity if callable(self._velocity) else None

        return BilinearForm(
            _AdvectionKernel(vel_np, vel_callable, self._conservative)
        )

    def forcing_form(self, time: float) -> Optional["LinearForm"]:
        """Linear form for the forcing contribution (w, f), or None."""
        if self._forcing is None:
            return None
        return LinearForm(_ForcingKernel(self._forcing, time))

    def reaction_form(self) -> Optional["LinearForm"]:
        """Linear form for the nonlinear reaction (w, R(u)), or None.

        Assemble with the interpolated state as a form parameter,
        ``asm(form, basis, u_prev=basis.interpolate(state))``; a raw
        (nelems, nquad) array of state values at the quadrature points
        also works, enabling element-restricted assembly.
        """
        if self._reaction_func is None or self._reaction_is_linear:
            return None
        return LinearForm(_ReactionKernel(self._reaction_func))

    def reaction_jacobian_form(self) -> Optional["BilinearForm"]:
        """Bilinear form (w, R'(u)*du) for the reaction Jacobian, or None.

        Assemble with the interpolated state as a form parameter, as in
        :meth:`reaction_form`.
        """
        if self._reaction_deriv is None or self._reaction_is_linear:
            return None
        return BilinearForm(_ReactionJacobianKernel(self._reaction_deriv))

    def stiffness_forms(self) -> List["BilinearForm"]:
        """Return the bilinear forms whose sum assembles the stiffness.

        Each form can be assembled on any compatible skfem basis — in
        particular a ``basis.with_elements(...)``-restricted basis — so
        consumers such as hyper-reduction can extract per-element
        contributions without changing the global assembly path.

        Returns
        -------
        List[BilinearForm]
            Diffusion (+ linear reaction) form, followed by the
            advection form when a velocity is present.
        """
        forms = [self._diffusion_reaction_form()]
        advection = self._advection_form()
        if advection is not None:
            forms.append(advection)
        return forms

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

        forms = self.stiffness_forms()
        stiffness = asm(forms[0], skfem_basis)
        # Add advection if present
        for form in forms[1:]:
            stiffness = stiffness + asm(form, skfem_basis)

        # Cache if linear problem with constant coefficients
        if (
            self.is_linear()
            and not callable(self._diffusivity)
            and not callable(self._velocity)
        ):
            self._stiffness_cached = stiffness

        result: Array = stiffness
        return result

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
        if self._forcing is None and (
            self._reaction_func is None or self._reaction_is_linear
        ):
            # No forcing and no nonlinear reaction - use cached zero vector
            if self._load_cached is not None:
                return self._load_cached
            load_np = np.zeros(self.nstates())
            self._load_cached = self._bkd.asarray(load_np.astype(np.float64))
            return self._load_cached

        load_np = np.zeros(self.nstates())

        # Forcing contribution: (w, f)
        forcing = self.forcing_form(time)
        if forcing is not None:
            load_np += asm(forcing, skfem_basis)

        # Nonlinear reaction contribution: (w, R(u))
        reaction = self.reaction_form()
        if reaction is not None:
            # Interpolate state to get u values at quadrature points
            state_interp = skfem_basis.interpolate(state_np)
            load_np += asm(reaction, skfem_basis, u_prev=state_interp)

        return self._bkd.asarray(load_np.astype(np.float64))

    def _assemble_reaction_jacobian(self, state: Array, time: float) -> Array:
        """Assemble Jacobian contribution from nonlinear reaction.

        For R(u), the Jacobian term is: (w, R'(u) * du)
        where du is the trial function.

        This returns the matrix J where J_ij = integral(R'(u) * phi_j * phi_i)
        """
        form = self.reaction_jacobian_form()
        if form is None:
            # No nonlinear reaction or linear reaction (already in stiffness)
            empty: Array = csr_matrix((self.nstates(), self.nstates()))
            return empty

        skfem_basis = self._basis.skfem_basis()
        state_np = self._bkd.to_numpy(state)

        # Interpolate state
        state_interp = skfem_basis.interpolate(state_np)

        jacobian: Array = asm(form, skfem_basis, u_prev=state_interp)
        return jacobian

    def mass_matrix(self) -> Array:
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

    def initial_condition(self, func: Callable[..., Any]) -> Array:
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
