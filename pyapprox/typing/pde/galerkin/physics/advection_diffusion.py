"""Linear advection-diffusion-reaction physics for Galerkin FEM.

Solves the equation:
    du/dt = div(D * grad(u)) - v . grad(u) + r*u + f

where:
    D = diffusivity (scalar or function)
    v = velocity field
    r = reaction coefficient
    f = forcing/source term
"""

from typing import Generic, Optional, Callable, List

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.pde.galerkin.protocols.basis import GalerkinBasisProtocol
from pyapprox.typing.pde.galerkin.protocols.boundary import BoundaryConditionProtocol
from pyapprox.typing.pde.galerkin.physics.base import AbstractGalerkinPhysics

# Import skfem for assembly
try:
    from skfem import asm, LinearForm, BilinearForm
    from skfem.helpers import dot, grad
except ImportError:
    raise ImportError(
        "scikit-fem is required for Galerkin module. "
        "Install with: pip install scikit-fem"
    )


class LinearAdvectionDiffusionReaction(AbstractGalerkinPhysics[Array]):
    """Linear advection-diffusion-reaction physics.

    Solves:
        du/dt = div(D * grad(u)) - v . grad(u) + r*u + f

    In weak form (for M * du/dt = F):
        F = b - K*u
    where:
        K_ij = integral(D * grad(phi_j) . grad(phi_i))
               + integral(v . grad(phi_j) * phi_i)
               - integral(r * phi_j * phi_i)
        b_i = integral(f * phi_i)

    Parameters
    ----------
    basis : GalerkinBasisProtocol
        Finite element basis.
    diffusivity : float or Callable
        Diffusion coefficient. If callable, takes coordinates (ndim, npts)
        and returns values (npts,).
    velocity : Array or Callable, optional
        Velocity field. If array, shape (ndim,). If callable, takes
        coordinates and returns (ndim, npts).
    reaction : float or Callable, optional
        Reaction coefficient. Positive for decay, negative for growth.
    forcing : Callable, optional
        Forcing/source term. Takes coordinates and returns (npts,).
    boundary_conditions : List[BoundaryConditionProtocol], optional
        List of boundary conditions.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.pde.galerkin.mesh import StructuredMesh1D
    >>> from pyapprox.typing.pde.galerkin.basis import LagrangeBasis
    >>> bkd = NumpyBkd()
    >>> mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
    >>> basis = LagrangeBasis(mesh, degree=1)
    >>> physics = LinearAdvectionDiffusionReaction(
    ...     basis=basis,
    ...     diffusivity=0.01,
    ...     bkd=bkd,
    ... )
    """

    def __init__(
        self,
        basis: GalerkinBasisProtocol[Array],
        diffusivity: float,
        bkd: Backend[Array],
        velocity: Optional[Array] = None,
        reaction: float = 0.0,
        forcing: Optional[Callable] = None,
        boundary_conditions: Optional[List[BoundaryConditionProtocol[Array]]] = None,
    ):
        super().__init__(basis, boundary_conditions)
        self._bkd = bkd

        # Store coefficients
        self._diffusivity = diffusivity
        self._velocity = velocity
        self._reaction = reaction
        self._forcing = forcing

        # Cache assembled matrices if coefficients are constant
        self._stiffness_cached: Optional[Array] = None
        self._load_cached: Optional[Array] = None

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

    def _get_reaction(self, coords: np.ndarray) -> np.ndarray:
        """Get reaction values at given coordinates."""
        if callable(self._reaction):
            return self._reaction(coords)
        else:
            return np.full(coords.shape[1], self._reaction)

    def _get_forcing(self, coords: np.ndarray) -> np.ndarray:
        """Get forcing values at given coordinates."""
        if self._forcing is None:
            return np.zeros(coords.shape[1])
        else:
            return self._forcing(coords)

    def _assemble_stiffness(self, state: Array, time: float) -> Array:
        """Assemble stiffness matrix K.

        K combines diffusion, advection, and reaction:
            K_ij = integral(D * grad(phi_j) . grad(phi_i))
                   + integral(v . grad(phi_j) * phi_i)
                   - integral(r * phi_j * phi_i)
        """
        if self._stiffness_cached is not None:
            return self._stiffness_cached

        skfem_basis = self._basis.skfem_basis()

        # Get constant coefficients or prepare for callable
        diff_const = self._diffusivity if not callable(self._diffusivity) else None
        react_const = self._reaction if not callable(self._reaction) else None

        # Define the bilinear form using skfem conventions
        # In skfem, u and v are basis function values/gradients
        # w.x gives quadrature point coordinates
        def bilinear_form(u, v, w):
            # Diffusion coefficient
            if diff_const is not None:
                diff = diff_const
            else:
                diff = self._diffusivity(w.x)

            # Reaction coefficient
            if react_const is not None:
                react = react_const
            else:
                react = self._reaction(w.x)

            # Diffusion term: D * grad(u) . grad(v)
            result = diff * dot(grad(u), grad(v))

            # Reaction term: -r * u * v
            result = result - react * u * v

            return result

        stiffness_np = asm(
            BilinearForm(bilinear_form), skfem_basis
        ).toarray()

        # Add advection if present
        if self._velocity is not None:
            vel_np = self._bkd.to_numpy(self._velocity) if not callable(self._velocity) else None

            def advection_form(u, v, w):
                if vel_np is not None:
                    # Constant velocity
                    vel = vel_np
                else:
                    vel = self._velocity(w.x)
                # Advection term: v . grad(u) * test_func
                return dot(vel, grad(u)) * v

            advection_np = asm(
                BilinearForm(advection_form), skfem_basis
            ).toarray()
            stiffness_np = stiffness_np + advection_np

        stiffness = self._bkd.asarray(stiffness_np.astype(np.float64))

        # Cache if coefficients are constant
        if not callable(self._diffusivity) and not callable(self._velocity):
            self._stiffness_cached = stiffness

        return stiffness

    def _assemble_load(self, state: Array, time: float) -> Array:
        """Assemble load vector b.

        b_i = integral(f * phi_i)
        """
        if self._load_cached is not None:
            return self._load_cached

        skfem_basis = self._basis.skfem_basis()

        if self._forcing is None:
            load_np = np.zeros(self.nstates())
        else:
            # Store forcing function for closure
            forcing_func = self._forcing

            def linear_form(v, w):
                # w.x shape: (ndim, nelem, nquad)
                # Need to reshape for the forcing function
                x_shape = w.x.shape
                if len(x_shape) == 3:
                    # Reshape to (ndim, nelem*nquad), apply, reshape back
                    ndim, nelem, nquad = x_shape
                    x_flat = w.x.reshape(ndim, -1)
                    forc_flat = forcing_func(x_flat)
                    forc = forc_flat.reshape(nelem, nquad)
                else:
                    forc = forcing_func(w.x)
                return forc * v

            load_np = asm(LinearForm(linear_form), skfem_basis)

        load = self._bkd.asarray(load_np.astype(np.float64))

        # Cache if forcing is constant
        if self._forcing is None:
            self._load_cached = load

        return load

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
        return (
            f"LinearAdvectionDiffusionReaction("
            f"nstates={self.nstates()}, "
            f"diffusivity={self._diffusivity})"
        )
