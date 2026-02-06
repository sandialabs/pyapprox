"""Adapter for using collocation manufactured solutions with Galerkin tests.

Bridges the existing manufactured solution infrastructure in
`pyapprox.typing.pde.collocation.manufactured_solutions` with the
Galerkin boundary conditions and physics implementations.
"""

from typing import Generic, List, Tuple, Callable, Optional, Dict

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.pde.galerkin.protocols.basis import GalerkinBasisProtocol
from pyapprox.typing.pde.galerkin.boundary import (
    DirichletBC,
    NeumannBC,
    RobinBC,
    BoundaryConditionSet,
    canonical_boundary_normal,
)
from pyapprox.typing.pde.collocation.manufactured_solutions import (
    ManufacturedAdvectionDiffusionReaction,
    ManufacturedHelmholtz,
)


class GalerkinManufacturedSolutionAdapter(Generic[Array]):
    """Adapter to use manufactured solutions with Galerkin finite elements.

    Takes a manufactured solution object (from collocation module) and
    provides methods to create boundary conditions and forcing functions
    compatible with Galerkin physics implementations.

    Parameters
    ----------
    basis : GalerkinBasisProtocol[Array]
        Finite element basis.
    functions : Dict[str, Callable]
        Dictionary of manufactured solution functions. Must contain:
        - "solution": exact solution function
        - "forcing": forcing function for the PDE
        Optional:
        - "flux": flux function D * grad(u) for boundary conditions
        - "diffusion": diffusion coefficient
        - "velocity": velocity field for advection
        - "reaction": reaction coefficient/function
    bkd : Backend[Array]
        Computational backend.
    time_dependent : bool, default=False
        Whether the solution is time-dependent.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.pde.collocation.manufactured_solutions import (
    ...     ManufacturedAdvectionDiffusionReaction
    ... )
    >>> from pyapprox.typing.pde.galerkin.mesh import StructuredMesh1D
    >>> from pyapprox.typing.pde.galerkin.basis import LagrangeBasis
    >>> bkd = NumpyBkd()
    >>> man_sol = ManufacturedAdvectionDiffusionReaction(
    ...     sol_str="x", nvars=1, diff_str="1.0",
    ...     react_str="0", vel_strs=["0"], bkd=bkd, oned=True
    ... )
    >>> mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
    >>> basis = LagrangeBasis(mesh, degree=1)
    >>> adapter = GalerkinManufacturedSolutionAdapter(
    ...     basis, man_sol.functions, bkd
    ... )
    """

    def __init__(
        self,
        basis: GalerkinBasisProtocol[Array],
        functions: Dict[str, Callable],
        bkd: Backend[Array],
        time_dependent: bool = False,
    ):
        self._basis = basis
        self._functions = functions
        self._bkd = bkd
        self._time_dependent = time_dependent
        self._ndim = basis.mesh().ndim()

        # Validate required functions
        if "solution" not in functions:
            raise ValueError("functions must contain 'solution'")
        if "forcing" not in functions:
            raise ValueError("functions must contain 'forcing'")

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def basis(self) -> GalerkinBasisProtocol[Array]:
        """Return the finite element basis."""
        return self._basis

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

    def solution_function(self) -> Callable:
        """Return the exact solution function.

        Returns
        -------
        Callable
            Function u(x) or u(x, t) returning exact solution values.
        """
        return self._functions["solution"]

    def forcing_function(self) -> Callable:
        """Return the forcing function.

        Returns
        -------
        Callable
            Function f(x) or f(x, t) returning forcing values.
        """
        return self._functions["forcing"]

    def forcing_for_galerkin(self) -> Callable:
        """Return forcing function adapted for Galerkin physics interface.

        The Galerkin physics expects forcing as f(x) returning shape (npts,).
        The manufactured solution returns f(x) with shape (npts, 1).

        Returns
        -------
        Callable
            Forcing function compatible with Galerkin physics.
        """
        forcing_func = self._functions["forcing"]

        if self._time_dependent:
            def adapted_forcing(x, time):
                vals = forcing_func(x, time)
                if hasattr(vals, "shape") and vals.ndim > 1:
                    return vals[:, 0] if vals.shape[1] == 1 else vals
                return vals
        else:
            def adapted_forcing(x):
                vals = forcing_func(x)
                if hasattr(vals, "shape") and vals.ndim > 1:
                    return vals[:, 0] if vals.shape[1] == 1 else vals
                return vals

        return adapted_forcing

    def _compute_normal_flux(
        self,
        boundary_index: int,
        coords: np.ndarray,
        time: Optional[float] = None,
    ) -> np.ndarray:
        """Compute flux dot normal at boundary coordinates.

        Parameters
        ----------
        boundary_index : int
            Boundary index (0=left, 1=right, 2=bottom, etc.).
        coords : np.ndarray
            Boundary coordinates. Shape: (ndim, npts).
        time : float, optional
            Time for transient problems.

        Returns
        -------
        np.ndarray
            Normal flux values. Shape: (npts,).
        """
        flux_func = self._functions.get("flux")
        if flux_func is None:
            raise ValueError(
                "Manufactured solution must provide 'flux' function "
                "to create Neumann or Robin boundary conditions"
            )

        # Compute normal
        normal = canonical_boundary_normal(boundary_index, coords)

        # Compute flux
        if self._time_dependent and time is not None:
            flux = flux_func(coords, time)
        else:
            flux = flux_func(coords)

        # Handle different output formats from manufactured solutions
        if flux.ndim == 2:
            # flux shape: (npts, ndim) - need to transpose
            if flux.shape[0] == coords.shape[1]:
                flux = flux.T
            # flux shape should now be (ndim, npts)

        # flux dot normal
        return np.sum(flux * normal, axis=0)

    def _create_dirichlet_bc(
        self, boundary_name: str, boundary_index: int
    ) -> DirichletBC[Array]:
        """Create a Dirichlet BC from the manufactured solution."""
        sol_func = self._functions["solution"]

        if self._time_dependent:
            def value_func(x, t):
                vals = sol_func(x, t)
                if hasattr(vals, "shape") and vals.ndim > 1:
                    return vals[:, 0] if vals.shape[1] == 1 else vals
                return vals
        else:
            def value_func(x, t=None):
                vals = sol_func(x)
                if hasattr(vals, "shape") and vals.ndim > 1:
                    return vals[:, 0] if vals.shape[1] == 1 else vals
                return vals

        return DirichletBC(
            basis=self._basis,
            boundary_name=boundary_name,
            value_func=value_func,
            bkd=self._bkd,
        )

    def _create_neumann_bc(
        self, boundary_name: str, boundary_index: int
    ) -> NeumannBC[Array]:
        """Create a Neumann BC from the manufactured solution."""

        if self._time_dependent:
            def neumann_value(x, t):
                return self._compute_normal_flux(boundary_index, x, t)
        else:
            def neumann_value(x, t=None):
                return self._compute_normal_flux(boundary_index, x)

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
        sol_func = self._functions["solution"]

        if self._time_dependent:
            def robin_value(x, t):
                u_vals = sol_func(x, t)
                if hasattr(u_vals, "shape") and u_vals.ndim > 1:
                    u_vals = u_vals[:, 0] if u_vals.shape[1] == 1 else u_vals
                flux_dot_n = self._compute_normal_flux(boundary_index, x, t)
                return alpha * u_vals - flux_dot_n
        else:
            def robin_value(x, t=None):
                u_vals = sol_func(x)
                if hasattr(u_vals, "shape") and u_vals.ndim > 1:
                    u_vals = u_vals[:, 0] if u_vals.shape[1] == 1 else u_vals
                flux_dot_n = self._compute_normal_flux(boundary_index, x)
                return alpha * u_vals - flux_dot_n

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
            List of boundary condition types in order
            [left, right, bottom, top, ...].
            Valid types:
            - "D": Dirichlet (u = g)
            - "N": Neumann (flux . n = g)
            - "R": Robin (alpha * u - flux . n = g)
        robin_alpha : float, default=1.0
            Coefficient for Robin BCs.

        Returns
        -------
        BoundaryConditionSet[Array]
            Collection of boundary conditions.
        """
        boundary_names = self._get_boundary_names()

        if len(bc_types) != len(boundary_names):
            raise ValueError(
                f"Expected {len(boundary_names)} BC types for {self._ndim}D "
                f"problem, got {len(bc_types)}"
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
            else:
                raise ValueError(
                    f"Unknown BC type '{bc_type}'. "
                    "Valid types: 'D' (Dirichlet), 'N' (Neumann), 'R' (Robin)"
                )

        return bc_set

    def __repr__(self) -> str:
        return (
            f"GalerkinManufacturedSolutionAdapter(ndim={self._ndim}, "
            f"time_dependent={self._time_dependent})"
        )


def create_adr_manufactured_test(
    bounds: List[float],
    sol_str: str,
    diff_str: str,
    react_str: str,
    vel_strs: List[str],
    bkd: Backend[Array],
    time_dependent: bool = False,
) -> Tuple[Dict[str, Callable], int]:
    """Create ADR manufactured solution functions for Galerkin tests.

    Parameters
    ----------
    bounds : List[float]
        Domain bounds [xmin, xmax] or [xmin, xmax, ymin, ymax] etc.
    sol_str : str
        Solution string (uses 'x', 'y', 'z' for coords, 'T' for time).
    diff_str : str
        Diffusion coefficient string.
    react_str : str
        Reaction term string (uses 'u' for solution).
    vel_strs : List[str]
        Velocity component strings.
    bkd : Backend
        Computational backend.
    time_dependent : bool
        Whether solution is time-dependent.

    Returns
    -------
    functions : Dict[str, Callable]
        Dictionary containing solution, forcing, flux, etc.
    nvars : int
        Number of spatial dimensions.
    """
    nvars = len(bounds) // 2

    man_sol = ManufacturedAdvectionDiffusionReaction(
        sol_str=sol_str,
        nvars=nvars,
        diff_str=diff_str,
        react_str=react_str,
        vel_strs=vel_strs,
        bkd=bkd,
        oned=True,  # Return 1D arrays for Galerkin
    )

    return man_sol.functions, nvars


def create_helmholtz_manufactured_test(
    bounds: List[float],
    sol_str: str,
    sqwavenum_str: str,
    bkd: Backend[Array],
) -> Tuple[Dict[str, Callable], int]:
    """Create Helmholtz manufactured solution functions for Galerkin tests.

    Parameters
    ----------
    bounds : List[float]
        Domain bounds [xmin, xmax] or [xmin, xmax, ymin, ymax] etc.
    sol_str : str
        Solution string (uses 'x', 'y', 'z' for coords).
    sqwavenum_str : str
        Squared wavenumber string (k²).
    bkd : Backend
        Computational backend.

    Returns
    -------
    functions : Dict[str, Callable]
        Dictionary containing solution, forcing, sqwavenum, etc.
    nvars : int
        Number of spatial dimensions.
    """
    nvars = len(bounds) // 2

    man_sol = ManufacturedHelmholtz(
        sol_str=sol_str,
        nvars=nvars,
        sqwavenum_str=sqwavenum_str,
        bkd=bkd,
        oned=True,  # Return 1D arrays for Galerkin
    )

    return man_sol.functions, nvars
