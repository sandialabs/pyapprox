"""Adapter for using collocation manufactured solutions with Galerkin tests.

Bridges the existing manufactured solution infrastructure in
`pyapprox.pde.collocation.manufactured_solutions` with the
Galerkin boundary conditions and physics implementations.
"""

from typing import Any, Callable, Dict, Generic, List, Optional, Tuple

import numpy as np

from pyapprox.pde.collocation.manufactured_solutions import (
    ManufacturedAdvectionDiffusionReaction,
    ManufacturedHelmholtz,
)
from pyapprox.pde.collocation.manufactured_solutions.hyperelasticity import (
    ManufacturedHyperelasticityEquations,
)
from pyapprox.pde.collocation.manufactured_solutions.linear_elasticity import (
    ManufacturedLinearElasticityEquations,
)
from pyapprox.pde.collocation.physics.stress_models.protocols import (
    SymbolicStressModelProtocol,
)
from pyapprox.pde.galerkin.boundary import (
    BoundaryConditionSet,
    DirichletBC,
    NeumannBC,
    RobinBC,
    canonical_boundary_normal,
)
from pyapprox.pde.galerkin.protocols.basis import GalerkinBasisProtocol
from pyapprox.util.backends.protocols import Array, Backend


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
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.pde.collocation.manufactured_solutions import (
    ...     ManufacturedAdvectionDiffusionReaction
    ... )
    >>> from pyapprox.pde.galerkin.mesh import StructuredMesh1D
    >>> from pyapprox.pde.galerkin.basis import LagrangeBasis
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
        functions: Dict[str, Callable[..., Any]],
        bkd: Backend[Array],
        time_dependent: bool = False,
        conservative: bool = False,
    ):
        self._basis = basis
        self._functions = functions
        self._bkd = bkd
        self._time_dependent = time_dependent
        self._conservative = conservative
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

    def solution_function(self) -> Callable[..., Any]:
        """Return the exact solution function.

        Returns
        -------
        Callable
            Function u(x) or u(x, t) returning exact solution values.
        """
        return self._functions["solution"]

    def forcing_function(self) -> Callable[..., Any]:
        """Return the forcing function.

        Returns
        -------
        Callable
            Function f(x) or f(x, t) returning forcing values.
        """
        return self._functions["forcing"]

    def velocity_for_galerkin(self) -> Optional[Callable[..., Any]]:
        """Return velocity function adapted for Galerkin physics interface.

        The Galerkin physics (skfem) expects velocity as a callable returning
        shape matching the input coordinates: (ndim, nelem, nquad) or
        (ndim, npts). The manufactured solution returns (npts, ndim).
        This method transposes the output to match the expected convention.

        Returns
        -------
        Callable or None
            Velocity function compatible with Galerkin physics, or None
            if no velocity is defined.
        """
        vel_func = self._functions.get("velocity")
        if vel_func is None:
            return None

        def adapted_velocity(
            x: np.ndarray,
        ) -> np.ndarray:
            # x: (ndim, ...) from skfem
            orig_shape = x.shape
            ndim = orig_shape[0]
            trailing = orig_shape[1:]
            flat = x.reshape(ndim, -1)
            # Manufactured velocity returns (npts, ndim)
            vals = vel_func(flat)
            # Transpose to (ndim, npts) then reshape to (ndim, ...)
            return vals.T.reshape(ndim, *trailing)

        return adapted_velocity

    def forcing_for_galerkin(self) -> Callable[..., Any]:
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

            def adapted_forcing(
                x: np.ndarray, time: float,
            ) -> np.ndarray:
                vals = forcing_func(x, time)
                if hasattr(vals, "shape") and vals.ndim > 1:
                    return vals[:, 0] if vals.shape[1] == 1 else vals
                return vals
        else:

            def adapted_forcing(
                x: np.ndarray,
            ) -> np.ndarray:
                vals = forcing_func(x)
                if hasattr(vals, "shape") and vals.ndim > 1:
                    return vals[:, 0] if vals.shape[1] == 1 else vals
                return vals

        return adapted_forcing

    def _eval_flux(
        self,
        flux_func: Callable[..., Any],
        coords: np.ndarray,
        time: Optional[float] = None,
    ) -> np.ndarray:
        """Evaluate a flux function and return shape (ndim, npts)."""
        if self._time_dependent and time is not None:
            flux = flux_func(coords, time)
        else:
            flux = flux_func(coords)

        if flux.ndim == 2:
            # flux shape: (npts, ndim) - need to transpose
            if flux.shape[0] == coords.shape[1]:
                flux = flux.T
            # flux shape should now be (ndim, npts)
        return flux

    def _compute_natural_bc_value(
        self,
        boundary_index: int,
        coords: np.ndarray,
        time: Optional[float] = None,
    ) -> np.ndarray:
        """Compute the natural BC value at boundary coordinates.

        Returns the quantity that appears as the natural boundary condition
        in the Galerkin weak form after integration by parts.

        Non-conservative form:
            Natural BC = D * grad(u) · n

        Conservative form:
            Natural BC = D * grad(u) · n - v * u · n
            (includes the advective boundary term from IBP of div(v*u))

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
            Natural BC value. Shape: (npts,).
        """
        # Use diffusive_flux (pure -D*grad(u)) if available
        flux_func = self._functions.get("diffusive_flux")
        if flux_func is None:
            flux_func = self._functions.get("flux")
        if flux_func is None:
            raise ValueError(
                "Manufactured solution must provide 'diffusive_flux' or "
                "'flux' function to create Neumann or Robin boundary "
                "conditions"
            )

        normal = canonical_boundary_normal(boundary_index, coords)
        diff_flux = self._eval_flux(flux_func, coords, time)

        # D * grad(u) · n  (negate because diffusive flux = -D * grad(u))
        result = -np.sum(diff_flux * normal, axis=0)

        if self._conservative:
            # Subtract advective boundary term: v * u · n
            vel_func = self._functions.get("velocity")
            sol_func = self._functions["solution"]
            if vel_func is not None:
                vel = self._eval_flux(vel_func, coords, time)
                if self._time_dependent and time is not None:
                    u_vals = sol_func(coords, time)
                else:
                    u_vals = sol_func(coords)
                if u_vals.ndim > 1:
                    u_vals = u_vals[:, 0] if u_vals.shape[1] == 1 else u_vals.flatten()
                # v * u · n
                advective_flux_dot_n = np.sum(vel * u_vals * normal, axis=0)
                result -= advective_flux_dot_n

        return result

    def _create_dirichlet_bc(
        self, boundary_name: str, boundary_index: int
    ) -> DirichletBC[Array]:
        """Create a Dirichlet BC from the manufactured solution."""
        sol_func = self._functions["solution"]

        if self._time_dependent:

            def value_func(
                x: np.ndarray, t: float,
            ) -> np.ndarray:
                vals = sol_func(x, t)
                if hasattr(vals, "shape") and vals.ndim > 1:
                    return vals[:, 0] if vals.shape[1] == 1 else vals
                return vals
        else:

            def value_func(
                x: np.ndarray,
                t: Optional[float] = None,
            ) -> np.ndarray:
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

            def neumann_value(
                x: np.ndarray, t: float,
            ) -> np.ndarray:
                return self._compute_natural_bc_value(boundary_index, x, t)
        else:

            def neumann_value(
                x: np.ndarray,
                t: Optional[float] = None,
            ) -> np.ndarray:
                return self._compute_natural_bc_value(boundary_index, x)

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

        Robin BC weak form: natural_bc = g - alpha*u
        So g = alpha * u + natural_bc
        where natural_bc is D*grad(u)·n for non-conservative, or
        D*grad(u)·n - v*u·n for conservative advection.
        """
        sol_func = self._functions["solution"]

        if self._time_dependent:

            def robin_value(
                x: np.ndarray, t: float,
            ) -> np.ndarray:
                u_vals = sol_func(x, t)
                if hasattr(u_vals, "shape") and u_vals.ndim > 1:
                    u_vals = u_vals[:, 0] if u_vals.shape[1] == 1 else u_vals
                nat_bc = self._compute_natural_bc_value(boundary_index, x, t)
                return alpha * u_vals + nat_bc
        else:

            def robin_value(
                x: np.ndarray,
                t: Optional[float] = None,
            ) -> np.ndarray:
                u_vals = sol_func(x)
                if hasattr(u_vals, "shape") and u_vals.ndim > 1:
                    u_vals = u_vals[:, 0] if u_vals.shape[1] == 1 else u_vals
                nat_bc = self._compute_natural_bc_value(boundary_index, x)
                return alpha * u_vals + nat_bc

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

        for i, (bc_type, boundary_name) in enumerate(zip(bc_types, boundary_names)):
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
    conservative: bool = False,
) -> Tuple[Dict[str, Callable[..., Any]], int]:
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
    conservative : bool
        If True, the flux includes the advective contribution v*u in
        addition to the diffusive flux -D*grad(u). If False (default),
        the flux contains only the diffusive part. Use False for
        standard Galerkin FEM which uses the non-conservative advection
        form v.grad(u).

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
        conservative=conservative,
    )

    return man_sol.functions, nvars


def create_helmholtz_manufactured_test(
    bounds: List[float],
    sol_str: str,
    sqwavenum_str: str,
    bkd: Backend[Array],
) -> Tuple[Dict[str, Callable[..., Any]], int]:
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


def create_elasticity_manufactured_test(
    bounds: List[float],
    sol_strs: List[str],
    lambda_str: str,
    mu_str: str,
    bkd: Backend[Array],
) -> Tuple[Dict[str, Callable[..., Any]], int]:
    """Create linear elasticity manufactured solution functions for Galerkin tests.

    Parameters
    ----------
    bounds : List[float]
        Domain bounds [xmin, xmax, ymin, ymax].
    sol_strs : List[str]
        Displacement component strings [u_x, u_y].
        May contain 'x', 'y' for coords and 'T' for time.
    lambda_str : str
        Lame first parameter string.
    mu_str : str
        Shear modulus string.
    bkd : Backend
        Computational backend.

    Returns
    -------
    functions : Dict[str, Callable]
        Dictionary containing solution, forcing, flux, etc.
    nvars : int
        Number of spatial dimensions.
    """
    nvars = len(bounds) // 2

    man_sol = ManufacturedLinearElasticityEquations(
        sol_strs=sol_strs,
        nvars=nvars,
        lambda_str=lambda_str,
        mu_str=mu_str,
        bkd=bkd,
        oned=True,
    )

    return man_sol.functions, nvars


def create_hyperelasticity_manufactured_test(
    bounds: List[float],
    sol_strs: List[str],
    stress_model: SymbolicStressModelProtocol,
    bkd: Backend[Array],
) -> Tuple[Dict[str, Callable[..., Any]], int]:
    """Create hyperelasticity manufactured solution functions for Galerkin tests.

    Parameters
    ----------
    bounds : List[float]
        Domain bounds [xmin, xmax] or [xmin, xmax, ymin, ymax] etc.
    sol_strs : List[str]
        Displacement component expressions. Length must equal nvars.
        May contain 'x', 'y', 'z' for coords and 'T' for time.
    stress_model : SymbolicStressModelProtocol
        Stress model with sympy expression support (e.g., NeoHookeanStress).
    bkd : Backend
        Computational backend.

    Returns
    -------
    functions : Dict[str, Callable]
        Dictionary containing solution, forcing, flux functions.
        - solution(x): (npts, ncomponents) exact displacement
        - forcing(x): (npts, ncomponents) body force
        - flux(x): (ncomponents, npts, ncomponents) PK1 stress tensor
    nvars : int
        Number of spatial dimensions.
    """
    nvars = len(bounds) // 2

    man_sol = ManufacturedHyperelasticityEquations(
        sol_strs=sol_strs,
        nvars=nvars,
        stress_model=stress_model,
        bkd=bkd,
        oned=True,
    )

    return man_sol.functions, nvars


class GalerkinHyperelasticityAdapter(Generic[Array]):
    """Adapter for vector-valued hyperelasticity manufactured solutions.

    Provides methods to create boundary conditions and forcing functions
    for hyperelasticity Galerkin tests. Handles the interleaved DOF
    ordering of VectorLagrangeBasis and vector-valued BC values.

    Parameters
    ----------
    basis : GalerkinBasisProtocol[Array]
        Vector finite element basis (VectorLagrangeBasis).
    functions : Dict[str, Callable]
        Manufactured solution functions from
        ``create_hyperelasticity_manufactured_test()``.
    bkd : Backend[Array]
        Computational backend.
    time_dependent : bool, default=False
        Whether the solution is time-dependent.
    """

    def __init__(
        self,
        basis: GalerkinBasisProtocol[Array],
        functions: Dict[str, Callable[..., Any]],
        bkd: Backend[Array],
        time_dependent: bool = False,
    ):
        self._basis = basis
        self._functions = functions
        self._bkd = bkd
        self._time_dependent = time_dependent
        self._ndim = basis.mesh().ndim()

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

    def solution_function(self) -> Callable[..., Any]:
        """Return the exact solution function."""
        return self._functions["solution"]

    def forcing_for_galerkin(self) -> Callable[..., Any]:
        """Return body force adapted for Galerkin physics.

        The manufactured solution returns forcing as (npts, ncomponents).
        Galerkin physics expects body_force(x, time) returning (ndim, npts).

        Returns
        -------
        Callable
            body_force(x, time) -> (ndim, npts)
        """
        forcing_func = self._functions["forcing"]
        time_dep = self._time_dependent

        if time_dep:

            def adapted_forcing(
                x: np.ndarray, time: float,
            ) -> np.ndarray:
                vals = forcing_func(x, time)  # (npts, ncomponents)
                return vals.T  # (ncomponents, npts) = (ndim, npts)
        else:

            def adapted_forcing(
                x: np.ndarray, time: float = 0.0,
            ) -> np.ndarray:
                vals = forcing_func(x)  # (npts, ncomponents)
                return vals.T  # (ncomponents, npts) = (ndim, npts)

        return adapted_forcing

    def _create_dirichlet_bc(self, boundary_name: str) -> DirichletBC[Array]:
        """Create a Dirichlet BC for vector-valued displacement.

        Handles the interleaved DOF ordering: DOF j corresponds to
        component j % ndim.
        """
        sol_func = self._functions["solution"]
        ndim = self._ndim
        time_dep = self._time_dependent

        def value_func(
            coords: np.ndarray, time: float = 0.0,
        ) -> np.ndarray:
            # coords: (ndim, nbndry_dofs) from dof_coordinates
            nbndry_dofs = coords.shape[1]
            if time_dep:
                vals = sol_func(coords, time)  # (nbndry_dofs, ncomponents)
            else:
                vals = sol_func(coords)  # (nbndry_dofs, ncomponents)
            # Extract correct component for each interleaved DOF
            result = np.zeros(nbndry_dofs)
            for j in range(nbndry_dofs):
                result[j] = vals[j, j % ndim]
            return result

        return DirichletBC(
            basis=self._basis,
            boundary_name=boundary_name,
            value_func=value_func,
            bkd=self._bkd,
        )

    def _compute_traction(
        self,
        boundary_index: int,
        coords: np.ndarray,
        time: Optional[float] = None,
    ) -> np.ndarray:
        """Compute traction t = P.n at boundary coordinates.

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
            Traction components. Shape: (ndim, npts).
        """
        flux_func = self._functions.get("flux")
        if flux_func is None:
            raise ValueError(
                "Manufactured solution must provide 'flux' function "
                "(PK1 stress tensor) for Neumann/Robin BCs."
            )

        normal = canonical_boundary_normal(boundary_index, coords)
        ndim = self._ndim

        if self._time_dependent and time is not None:
            P = flux_func(coords, time)
        else:
            P = flux_func(coords)
        # P may be a backend array (e.g. torch tensor); convert to numpy
        # since this is called inside skfem assembly which is numpy-based
        P = self._bkd.to_numpy(P)
        # P shape: (ncomponents, npts, ncomponents) = (ndim, npts, ndim)
        # P[i, :, j] = P_{ij} at each point

        # Traction: t_i = sum_j P_{ij} * n_j
        npts = coords.shape[1]
        traction = np.zeros((ndim, npts))
        for i in range(ndim):
            for j in range(ndim):
                traction[i] += P[i, :, j] * normal[j]

        return traction

    def _create_neumann_bc(
        self, boundary_name: str, boundary_index: int
    ) -> NeumannBC[Array]:
        """Create a Neumann BC for vector-valued traction.

        The Neumann flux for hyperelasticity is the traction: t = P.n.
        Returns (ndim, npts) traction at spatial coordinates for
        quadrature-point assembly in the NeumannBC LinearForm.
        """
        time_dep = self._time_dependent
        bndry_idx = boundary_index

        def neumann_flux(
            coords: np.ndarray, time: float = 0.0,
        ) -> np.ndarray:
            # coords: (ndim, npts) — quadrature point coordinates
            traction = self._compute_traction(
                bndry_idx, coords, time if time_dep else None
            )
            return traction  # (ndim, npts)

        return NeumannBC(
            basis=self._basis,
            boundary_name=boundary_name,
            flux_func=neumann_flux,
            bkd=self._bkd,
        )

    def _create_robin_bc(
        self, boundary_name: str, boundary_index: int, alpha: float = 1.0
    ) -> RobinBC[Array]:
        """Create a Robin BC: alpha * u + P.n = g.

        The Robin value g = alpha * u + t where t = P.n (traction).
        Returns (ndim, npts) at spatial coordinates for
        quadrature-point assembly in the RobinBC LinearForm.
        """
        sol_func = self._functions["solution"]
        time_dep = self._time_dependent
        bndry_idx = boundary_index
        alpha_val = alpha

        def robin_value(
            coords: np.ndarray, time: float = 0.0,
        ) -> np.ndarray:
            # coords: (ndim, npts)
            traction = self._compute_traction(
                bndry_idx, coords, time if time_dep else None
            )
            if time_dep:
                u_vals = sol_func(coords, time)  # (npts, ncomponents)
            else:
                u_vals = sol_func(coords)  # (npts, ncomponents)
            # Convert to numpy since this runs inside skfem assembly
            u_vals = self._bkd.to_numpy(u_vals)
            # u_vals: (npts, ndim) → transpose to (ndim, npts)
            return alpha_val * u_vals.T + traction  # (ndim, npts)

        return RobinBC(
            basis=self._basis,
            boundary_name=boundary_name,
            alpha=alpha_val,
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
            List of BC types in order [left, right, ...].
            "D" for Dirichlet, "N" for Neumann, "R" for Robin.
        robin_alpha : float, default=1.0
            Coefficient for Robin BCs.

        Returns
        -------
        BoundaryConditionSet[Array]
        """
        boundary_names = self._get_boundary_names()

        if len(bc_types) != len(boundary_names):
            raise ValueError(
                f"Expected {len(boundary_names)} BC types for "
                f"{self._ndim}D problem, got {len(bc_types)}"
            )

        bc_set = BoundaryConditionSet(self._bkd)

        for i, (bc_type, boundary_name) in enumerate(zip(bc_types, boundary_names)):
            if bc_type == "D":
                bc = self._create_dirichlet_bc(boundary_name)
                bc_set.add_dirichlet(bc)
            elif bc_type == "N":
                bc = self._create_neumann_bc(boundary_name, i)
                bc_set.add_neumann(bc)
            elif bc_type == "R":
                bc = self._create_robin_bc(boundary_name, i, robin_alpha)
                bc_set.add_robin(bc)
            else:
                raise ValueError(
                    f"Unknown BC type '{bc_type}'. Valid types: 'D', 'N', 'R'"
                )

        return bc_set

    def __repr__(self) -> str:
        return (
            f"GalerkinHyperelasticityAdapter(ndim={self._ndim}, "
            f"time_dependent={self._time_dependent})"
        )
