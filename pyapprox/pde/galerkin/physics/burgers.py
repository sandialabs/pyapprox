"""Viscous Burgers equation physics for Galerkin FEM.

Solves the 1D viscous Burgers equation in conservative form:

    du/dt + d/dx(u²/2) = d/dx(ν * du/dx) + f

The nonlinear term u*du/dx requires Newton linearization. At each Newton
iteration, the bilinear and linear forms are reassembled using the current
state (u_prev):

Bilinear form (Jacobian/stiffness):
    ν*grad(u)·grad(v) + v*u_prev*du/dx + v*u*du_prev/dx

Linear form (residual/load):
    f*v - ν*grad(u_prev)·grad(v) - v*u_prev*du_prev/dx
"""

from typing import Callable, Generic, List, Optional, Union

import numpy as np
from scipy.sparse import csr_matrix

from pyapprox.pde.galerkin.physics.galerkin_base import GalerkinPhysicsBase
from pyapprox.pde.galerkin.physics.helpers import ScalarMassAssembler
from pyapprox.pde.galerkin.protocols.basis import GalerkinBasisProtocol
from pyapprox.pde.galerkin.protocols.boundary import (
    BoundaryConditionProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend

try:
    from skfem import BilinearForm, LinearForm, asm
    from skfem.helpers import dot, grad
except ImportError:
    raise ImportError(
        "scikit-fem is required for Galerkin module. "
        "Install with: pip install scikit-fem"
    )


class BurgersPhysics(GalerkinPhysicsBase[Array], Generic[Array]):
    """Viscous Burgers equation physics for Galerkin FEM.

    Solves:
        du/dt + d/dx(u²/2) = d/dx(ν * du/dx) + f

    or equivalently (using chain rule on conservative form):
        du/dt + u * du/dx = ν * d²u/dx² + f

    The Newton linearization produces state-dependent bilinear and linear
    forms that are reassembled at every Newton iteration.

    Parameters
    ----------
    basis : GalerkinBasisProtocol
        Finite element basis.
    viscosity : float or Callable
        Kinematic viscosity ν. If callable, takes coordinates (ndim, npts)
        and returns values (npts,).
    bkd : Backend
        Computational backend.
    forcing : Callable, optional
        Forcing/source term f. Takes coordinates and returns (npts,).
        For time-dependent problems, takes (coordinates, time).
    boundary_conditions : List[BoundaryConditionProtocol], optional
        List of boundary conditions.
    """

    def __init__(
        self,
        basis: GalerkinBasisProtocol[Array],
        viscosity: Union[float, Callable],
        bkd: Backend[Array],
        forcing: Optional[Callable] = None,
        boundary_conditions: Optional[List[BoundaryConditionProtocol[Array]]] = None,
    ):
        super().__init__(basis, bkd, boundary_conditions)
        self._mass = ScalarMassAssembler(basis, bkd)
        self._viscosity = viscosity
        self._forcing = forcing

    def is_linear(self) -> bool:
        """Burgers equation is always nonlinear."""
        return False

    def _get_viscosity(self, coords: np.ndarray) -> np.ndarray:
        """Get viscosity values at given coordinates."""
        if callable(self._viscosity):
            return self._viscosity(coords)
        else:
            return np.full(coords.shape[-1], self._viscosity)

    def _get_forcing(self, coords: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Get forcing values at given coordinates."""
        if self._forcing is None:
            return np.zeros(coords.shape[-1])
        try:
            return self._forcing(coords, time)
        except TypeError:
            return self._forcing(coords)

    def mass_matrix(self):
        """Return the scalar mass matrix."""
        return self._mass.mass_matrix()

    def mass_solve(self, rhs: Array) -> Array:
        """Solve M * x = rhs for x."""
        return self._mass.mass_solve(rhs)

    def _assemble_stiffness(self, state: Array, time: float) -> Array:
        """Assemble Newton-linearized stiffness matrix.

        The bilinear form for the Newton linearization of Burgers:
            ν*grad(u)·grad(v) + v*u_prev*du/dx + v*u*du_prev/dx

        where u_prev is the interpolated current state.
        """
        skfem_basis = self._basis.skfem_basis()
        state_np = self._bkd.to_numpy(state)
        state_interp = skfem_basis.interpolate(state_np)

        visc_const = self._viscosity if not callable(self._viscosity) else None

        def bilinear_form(u, v, w):
            if visc_const is not None:
                visc = visc_const
            else:
                visc = self._viscosity(np.asarray(w.x))
            return (
                dot(visc * grad(u), grad(v))
                + v * w.u_prev * u.grad[0]
                + v * u * w.u_prev.grad[0]
            )

        return asm(BilinearForm(bilinear_form), skfem_basis, u_prev=state_interp)

    def _assemble_load(self, state: Array, time: float) -> Array:
        """Assemble Newton-linearized load vector.

        The linear form for the Newton linearization of Burgers:
            f*v - ν*grad(u_prev)·grad(v) - v*u_prev*du_prev/dx

        where u_prev is the interpolated current state.
        """
        skfem_basis = self._basis.skfem_basis()
        state_np = self._bkd.to_numpy(state)
        state_interp = skfem_basis.interpolate(state_np)

        visc_const = self._viscosity if not callable(self._viscosity) else None
        current_time = time

        def linear_form(v, w):
            if visc_const is not None:
                visc = visc_const
            else:
                visc = self._viscosity(np.asarray(w.x))

            x_np = np.asarray(w.x)
            x_shape = x_np.shape
            if len(x_shape) == 3:
                ndim, nelem, nquad = x_shape
                x_flat = x_np.reshape(ndim, -1)
                forc_flat = self._get_forcing(x_flat, current_time)
                forc = forc_flat.reshape(nelem, nquad)
            else:
                forc = self._get_forcing(x_np, current_time)

            return (
                forc * v
                - dot(visc * grad(w.u_prev), grad(v))
                - v * w.u_prev * w.u_prev.grad[0]
            )

        load_np = asm(LinearForm(linear_form), skfem_basis, u_prev=state_interp)
        return self._bkd.asarray(load_np.astype(np.float64))

    def spatial_residual(self, state: Array, time: float) -> Array:
        """Compute Burgers spatial residual without Dirichlet enforcement.

        The load (linear form) already evaluates the full nonlinear interior
        residual at the current state. The Newton-linearized stiffness is NOT
        part of the residual — it only enters the Jacobian. Robin BCs add a
        linear boundary mass term via a zero-initialized stiffness.
        """
        load = self._assemble_load(state, time)
        load = self._apply_bc_to_load(load, time)

        # Zero stiffness — only BC contributions (Robin alpha*M_bnd) matter
        n = self.nstates()
        bc_stiffness = csr_matrix((n, n))
        bc_stiffness = self._apply_bc_to_stiffness(bc_stiffness, time)

        return load - bc_stiffness @ state

    def spatial_jacobian(self, state: Array, time: float) -> Array:
        """Compute dR/du without Dirichlet enforcement.

        For Burgers, the Jacobian is -K where K is the Newton-linearized
        stiffness (bilinear form) evaluated at the current state.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Jacobian dR/du. Shape: (nstates, nstates)
        """
        stiffness = self._assemble_stiffness(state, time)
        stiffness = self._apply_bc_to_stiffness(stiffness, time)
        return -stiffness

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
        return f"BurgersPhysics(nstates={self.nstates()}, viscosity={self._viscosity})"
