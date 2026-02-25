"""Boundary condition utilities for manufactured solution testing.

Provides functions to create boundary conditions from manufactured solutions
for verifying Galerkin finite element implementations.
"""

from typing import Generic, List, Callable, Tuple, Optional

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.pde.galerkin.protocols.basis import GalerkinBasisProtocol
from pyapprox.pde.galerkin.boundary.implementations import (
    DirichletBC,
    NeumannBC,
    RobinBC,
    BoundaryConditionSet,
)


def canonical_boundary_normal(
    boundary_index: int, samples: np.ndarray
) -> np.ndarray:
    """Compute outward unit normal for canonical rectangular domain boundaries.

    For a rectangular domain [x0, x1] x [y0, y1] x ..., the boundaries are:
    - Index 0: left (x = x0), normal = [-1, 0, ...]
    - Index 1: right (x = x1), normal = [+1, 0, ...]
    - Index 2: bottom (y = y0), normal = [0, -1, ...]
    - Index 3: top (y = y1), normal = [0, +1, ...]
    - Index 4: front (z = z0), normal = [0, 0, -1, ...]
    - Index 5: back (z = z1), normal = [0, 0, +1, ...]

    Parameters
    ----------
    boundary_index : int
        Boundary index (0=left, 1=right, 2=bottom, 3=top, ...).
    samples : np.ndarray
        Sample coordinates. Shape: (ndim, nsamples) or (ndim, nelem, nquad).

    Returns
    -------
    np.ndarray
        Normal vectors. Same shape as samples.
    """
    if samples.ndim == 2:
        # Shape: (ndim, nsamples)
        normal_vals = np.zeros_like(samples)
        active_var = boundary_index // 2
        sign = (-1) ** ((boundary_index + 1) % 2)
        normal_vals[active_var, :] = sign
    else:
        # Shape: (ndim, nelem, nquad) - used in skfem assembly
        normal_vals = np.zeros_like(samples)
        active_var = boundary_index // 2
        sign = (-1) ** ((boundary_index + 1) % 2)
        normal_vals[active_var, :, :] = sign

    return normal_vals


def _compute_normal_flux(
    flux_func: Callable,
    normal_func: Callable,
    coords: np.ndarray,
    time: Optional[float] = None,
) -> np.ndarray:
    """Compute flux dot normal at given coordinates.

    Parameters
    ----------
    flux_func : Callable
        Function returning flux vector. Takes (ndim, npts), returns (ndim, npts).
    normal_func : Callable
        Function returning outward normal. Takes (ndim, npts), returns (ndim, npts).
    coords : np.ndarray
        Coordinates. Shape: (ndim, npts).
    time : float, optional
        Current time for time-dependent flux.

    Returns
    -------
    np.ndarray
        Normal flux values. Shape: (npts,).
    """
    normal = normal_func(coords)
    if time is not None:
        flux = flux_func(coords, time)
    else:
        flux = flux_func(coords)

    # flux dot normal: sum over dimensions
    return np.sum(flux * normal, axis=0)


class ManufacturedSolutionBC(Generic[Array]):
    """Factory for creating boundary conditions from manufactured solutions.

    Given a manufactured solution with known exact solution u and flux
    grad(u) * D, this class creates appropriate boundary conditions for
    each boundary of the domain.

    Parameters
    ----------
    basis : GalerkinBasisProtocol[Array]
        Finite element basis.
    solution_func : Callable
        Exact solution function u(x, t). Takes (ndim, npts), returns (npts,).
    flux_func : Callable
        Exact flux function D * grad(u). Takes (ndim, npts), returns (ndim, npts).
    bkd : Backend[Array]
        Computational backend.
    time_dependent : bool, default=False
        Whether the solution is time-dependent.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.pde.galerkin.mesh import StructuredMesh1D
    >>> from pyapprox.pde.galerkin.basis import LagrangeBasis
    >>> import numpy as np
    >>> bkd = NumpyBkd()
    >>> mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
    >>> basis = LagrangeBasis(mesh, degree=1)
    >>> # u = x, flux = D * grad(u) = D * [1]
    >>> def sol(x, t=None): return x[0]
    >>> def flux(x, t=None): return np.ones_like(x)
    >>> ms_bc = ManufacturedSolutionBC(basis, sol, flux, bkd)
    >>> bc_set = ms_bc.create_boundary_conditions(["D", "D"])
    """

    def __init__(
        self,
        basis: GalerkinBasisProtocol[Array],
        solution_func: Callable,
        flux_func: Callable,
        bkd: Backend[Array],
        time_dependent: bool = False,
    ):
        self._basis = basis
        self._solution_func = solution_func
        self._flux_func = flux_func
        self._bkd = bkd
        self._time_dependent = time_dependent

        # Get mesh dimension for determining number of boundaries
        self._ndim = basis.mesh().ndim()

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def _get_boundary_names(self) -> List[str]:
        """Get ordered list of boundary names for this mesh dimension."""
        if self._ndim == 1:
            return ["left", "right"]
        elif self._ndim == 2:
            return ["left", "right", "bottom", "top"]
        elif self._ndim == 3:
            return ["left", "right", "bottom", "top", "front", "back"]
        else:
            raise ValueError(f"Unsupported dimension: {self._ndim}")

    def _create_dirichlet_bc(
        self, boundary_name: str, boundary_index: int
    ) -> DirichletBC[Array]:
        """Create a Dirichlet BC from the manufactured solution."""
        sol_func = self._solution_func
        time_dep = self._time_dependent

        if time_dep:
            def value_func(x, t):
                return sol_func(x, t)
        else:
            def value_func(x, t=None):
                return sol_func(x)

        return DirichletBC(
            basis=self._basis,
            boundary_name=boundary_name,
            value_func=value_func,
            bkd=self._bkd,
        )

    def _create_neumann_bc(
        self, boundary_name: str, boundary_index: int
    ) -> NeumannBC[Array]:
        """Create a Neumann BC from the manufactured solution.

        The Neumann value is: flux . n = D * grad(u) . n
        """
        flux_func = self._flux_func
        time_dep = self._time_dependent

        def normal_func(x):
            return canonical_boundary_normal(boundary_index, x)

        if time_dep:
            def neumann_value(x, t):
                return _compute_normal_flux(flux_func, normal_func, x, t)
        else:
            def neumann_value(x, t=None):
                return _compute_normal_flux(flux_func, normal_func, x)

        return NeumannBC(
            basis=self._basis,
            boundary_name=boundary_name,
            flux_func=neumann_value,
            bkd=self._bkd,
        )

    def _create_robin_bc(
        self, boundary_name: str, boundary_index: int, alpha: float = 1.0
    ) -> RobinBC[Array]:
        """Create a Robin BC from the manufactured solution.

        Robin BC: alpha * u - flux . n = g
        So g = alpha * u - D * grad(u) . n
        """
        sol_func = self._solution_func
        flux_func = self._flux_func
        time_dep = self._time_dependent

        def normal_func(x):
            return canonical_boundary_normal(boundary_index, x)

        if time_dep:
            def robin_value(x, t):
                u_val = sol_func(x, t)
                flux_dot_n = _compute_normal_flux(flux_func, normal_func, x, t)
                return alpha * u_val - flux_dot_n
        else:
            def robin_value(x, t=None):
                u_val = sol_func(x)
                flux_dot_n = _compute_normal_flux(flux_func, normal_func, x)
                return alpha * u_val - flux_dot_n

        return RobinBC(
            basis=self._basis,
            boundary_name=boundary_name,
            alpha=alpha,
            value_func=robin_value,
            bkd=self._bkd,
        )

    def create_boundary_conditions(
        self, bc_types: List[str], robin_alpha: float = 1.0
    ) -> BoundaryConditionSet[Array]:
        """Create boundary condition set from type specification.

        Parameters
        ----------
        bc_types : List[str]
            List of boundary condition types in order [left, right, bottom, top, ...].
            Valid types:
            - "D": Dirichlet (u = g)
            - "N": Neumann (flux . n = g)
            - "R": Robin (alpha * u - flux . n = g)
            - "P": Periodic (not yet implemented)
        robin_alpha : float, default=1.0
            Coefficient for Robin BCs.

        Returns
        -------
        BoundaryConditionSet[Array]
            Collection of boundary conditions.

        Raises
        ------
        ValueError
            If bc_types length doesn't match number of boundaries.
        """
        boundary_names = self._get_boundary_names()

        if len(bc_types) != len(boundary_names):
            raise ValueError(
                f"Expected {len(boundary_names)} BC types for {self._ndim}D problem, "
                f"got {len(bc_types)}"
            )

        bc_set = BoundaryConditionSet(self._bkd)

        for i, (bc_type, boundary_name) in enumerate(
            zip(bc_types, boundary_names)
        ):
            if bc_type == "D":
                bc = self._create_dirichlet_bc(boundary_name, i)
                bc_set.add_dirichlet(bc)
            elif bc_type == "N":
                bc = self._create_neumann_bc(boundary_name, i)
                bc_set.add_neumann(bc)
            elif bc_type == "R":
                bc = self._create_robin_bc(boundary_name, i, robin_alpha)
                bc_set.add_robin(bc)
            elif bc_type == "P":
                # Periodic BCs need special handling
                raise NotImplementedError(
                    "Periodic boundary conditions not yet implemented"
                )
            else:
                raise ValueError(
                    f"Unknown BC type '{bc_type}'. "
                    "Valid types: 'D' (Dirichlet), 'N' (Neumann), 'R' (Robin)"
                )

        return bc_set

    def __repr__(self) -> str:
        return (
            f"ManufacturedSolutionBC(ndim={self._ndim}, "
            f"time_dependent={self._time_dependent})"
        )
