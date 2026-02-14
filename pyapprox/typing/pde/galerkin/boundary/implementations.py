"""Concrete boundary condition implementations for Galerkin FEM.

Provides implementations of Dirichlet, Neumann, and Robin boundary conditions
that integrate with scikit-fem for assembly.
"""

from typing import Generic, List, Optional, Callable, Union

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.pde.galerkin.protocols.basis import GalerkinBasisProtocol

try:
    from skfem import asm, LinearForm, BilinearForm, Basis
except ImportError:
    raise ImportError(
        "scikit-fem is required for Galerkin module. "
        "Install with: pip install scikit-fem"
    )


class DirichletBC(Generic[Array]):
    """Dirichlet boundary condition: u = g(x, t) on boundary.

    Enforces the constraint by modifying the residual and Jacobian
    at boundary DOFs.

    Parameters
    ----------
    basis : GalerkinBasisProtocol[Array]
        Finite element basis.
    boundary_name : str
        Name of the boundary (e.g., "left", "right", "bottom", "top").
    value_func : Callable or float
        Function g(x, t) returning boundary values, or constant value.
        If callable, takes coordinates (ndim, npts) and time, returns (npts,).
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.pde.galerkin.mesh import StructuredMesh1D
    >>> from pyapprox.typing.pde.galerkin.basis import LagrangeBasis
    >>> bkd = NumpyBkd()
    >>> mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
    >>> basis = LagrangeBasis(mesh, degree=1)
    >>> bc = DirichletBC(basis, "left", value_func=0.0, bkd=bkd)
    """

    def __init__(
        self,
        basis: GalerkinBasisProtocol[Array],
        boundary_name: str,
        value_func: Union[Callable, float],
        bkd: Backend[Array],
    ):
        self._basis = basis
        self._boundary_name = boundary_name
        self._bkd = bkd

        # Store value function
        if callable(value_func):
            self._value_func = value_func
        else:
            # Constant value
            const = float(value_func)
            self._value_func = lambda x, t=None: np.full(x.shape[1], const)

        # Get and cache boundary DOF indices
        self._boundary_dofs = basis.get_dofs(boundary_name)

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def boundary_name(self) -> str:
        """Return the boundary name."""
        return self._boundary_name

    def boundary_dofs(self) -> Array:
        """Return indices of DOFs on this boundary.

        Returns
        -------
        Array
            Integer DOF indices. Shape: (nboundary_dofs,)
        """
        return self._boundary_dofs

    def boundary_values(self, time: float = 0.0) -> Array:
        """Return Dirichlet boundary values at given time.

        Parameters
        ----------
        time : float
            Current time.

        Returns
        -------
        Array
            Boundary values. Shape: (nboundary_dofs,)
        """
        # Get DOF coordinates on boundary
        dof_coords = self._basis.dof_coordinates()
        dof_coords_np = self._bkd.to_numpy(dof_coords)
        bndry_dofs_np = self._bkd.to_numpy(self._boundary_dofs)

        # Extract boundary coordinates
        bndry_coords = dof_coords_np[:, bndry_dofs_np]

        # Evaluate boundary function
        values_np = self._value_func(bndry_coords, time)

        return self._bkd.asarray(values_np.astype(np.float64))

    def apply_to_residual(
        self, residual: Array, state: Array, time: float
    ) -> Array:
        """Apply Dirichlet BC to residual.

        Sets residual[dof] = state[dof] - g(x, t) for boundary DOFs.

        Parameters
        ----------
        residual : Array
            Residual vector. Shape: (nstates,)
        state : Array
            Current solution. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Modified residual. Shape: (nstates,)
        """
        res_np = self._bkd.to_numpy(residual).copy()
        state_np = self._bkd.to_numpy(state)
        bndry_dofs_np = self._bkd.to_numpy(self._boundary_dofs)
        bndry_vals_np = self._bkd.to_numpy(self.boundary_values(time))

        # Set residual to constraint violation: u - g
        res_np[bndry_dofs_np] = state_np[bndry_dofs_np] - bndry_vals_np

        return self._bkd.asarray(res_np)

    def apply_to_jacobian(
        self, jacobian: Array, state: Array, time: float
    ) -> Array:
        """Apply Dirichlet BC to Jacobian.

        Sets Jacobian rows to identity for boundary DOFs.

        Parameters
        ----------
        jacobian : Array
            Jacobian matrix. Shape: (nstates, nstates)
        state : Array
            Current solution. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Modified Jacobian. Shape: (nstates, nstates)
        """
        jac_np = self._bkd.to_numpy(jacobian).copy()
        bndry_dofs_np = self._bkd.to_numpy(self._boundary_dofs)

        # Set Jacobian rows to identity for boundary DOFs
        for dof in bndry_dofs_np:
            jac_np[dof, :] = 0.0
            jac_np[dof, dof] = 1.0

        return self._bkd.asarray(jac_np)

    def apply_to_param_jacobian(
        self, param_jacobian: Array, state: Array, time: float,
    ) -> Array:
        """Apply Dirichlet BC to parameter Jacobian.

        Dirichlet constraint u = g(x, t) does not depend on material
        parameters, so the parameter Jacobian rows at boundary DOFs
        are set to zero.

        Parameters
        ----------
        param_jacobian : Array
            Parameter Jacobian. Shape: (nstates, nparams)
        state : Array
            Current solution. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Modified parameter Jacobian. Shape: (nstates, nparams)
        """
        pj_np = self._bkd.to_numpy(param_jacobian).copy()
        bndry_dofs_np = self._bkd.to_numpy(self._boundary_dofs)
        pj_np[bndry_dofs_np, :] = 0.0
        return self._bkd.asarray(pj_np)

    def __repr__(self) -> str:
        return (
            f"DirichletBC(boundary='{self._boundary_name}', "
            f"ndofs={len(self._bkd.to_numpy(self._boundary_dofs))})"
        )


class NeumannBC(Generic[Array]):
    """Neumann boundary condition: flux . n = g(x, t) on boundary.

    In weak form, contributes to the load vector via boundary integral:
        integral_{Gamma} g * phi ds

    Parameters
    ----------
    basis : GalerkinBasisProtocol[Array]
        Finite element basis.
    boundary_name : str
        Name of the boundary.
    flux_func : Callable or float
        Function g(x, t) returning flux values, or constant value.
        If callable, takes coordinates (ndim, npts) and time, returns (npts,).
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        basis: GalerkinBasisProtocol[Array],
        boundary_name: str,
        flux_func: Union[Callable, float],
        bkd: Backend[Array],
    ):
        self._basis = basis
        self._boundary_name = boundary_name
        self._bkd = bkd

        # Store flux function
        if callable(flux_func):
            self._flux_func = flux_func
        else:
            const = float(flux_func)
            self._flux_func = lambda x, t=None: np.full(x.shape[1], const)

        # Get boundary DOFs
        self._boundary_dofs = basis.get_dofs(boundary_name)

        # Cache boundary basis
        self._boundary_basis: Optional[Basis] = None

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def boundary_name(self) -> str:
        """Return the boundary name."""
        return self._boundary_name

    def boundary_dofs(self) -> Array:
        """Return indices of DOFs on this boundary."""
        return self._boundary_dofs

    def flux_values(self, time: float = 0.0) -> Array:
        """Return Neumann flux values at given time.

        Parameters
        ----------
        time : float
            Current time.

        Returns
        -------
        Array
            Flux values. Shape: (nboundary_dofs,)
        """
        dof_coords = self._basis.dof_coordinates()
        dof_coords_np = self._bkd.to_numpy(dof_coords)
        bndry_dofs_np = self._bkd.to_numpy(self._boundary_dofs)
        bndry_coords = dof_coords_np[:, bndry_dofs_np]

        values_np = self._flux_func(bndry_coords, time)
        return self._bkd.asarray(values_np.astype(np.float64))

    def _get_boundary_basis(self) -> Basis:
        """Get or create the boundary basis for assembly."""
        if self._boundary_basis is None:
            skfem_basis = self._basis.skfem_basis()
            self._boundary_basis = skfem_basis.boundary(self._boundary_name)
        return self._boundary_basis

    def apply_to_load(self, load: Array, time: float) -> Array:
        """Apply Neumann BC contribution to load vector.

        Adds boundary integral: integral_{Gamma} g . phi ds

        For scalar elements, flux_func returns (npts,).
        For vector elements, flux_func returns (ndim, npts) and the
        form computes sum_i(flux_i * v_i).

        Parameters
        ----------
        load : Array
            Load vector. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Modified load vector. Shape: (nstates,)
        """
        load_np = self._bkd.to_numpy(load).copy()
        bndry_basis = self._get_boundary_basis()

        # Store flux function for closure
        flux_func = self._flux_func
        current_time = time

        def neumann_form(v, w):
            x_np = np.asarray(w.x)
            x_shape = x_np.shape
            if len(x_shape) == 3:
                ndim, nelem, nquad = x_shape
                x_flat = x_np.reshape(ndim, -1)
                flux_flat = np.asarray(flux_func(x_flat, current_time))
                # Detect vector flux: shape (ndim, npts) vs scalar (npts,)
                if flux_flat.ndim == 2 and flux_flat.shape[0] == ndim:
                    # Vector: flux_flat is (ndim, npts)
                    flux = flux_flat.reshape(ndim, nelem, nquad)
                    return sum(flux[i] * v[i] for i in range(ndim))
                else:
                    flux = flux_flat.reshape(nelem, nquad)
                    return flux * v
            else:
                flux = np.asarray(flux_func(x_np, current_time))
                if flux.ndim == 2:
                    ndim = flux.shape[0]
                    return sum(flux[i] * v[i] for i in range(ndim))
                return flux * v

        contribution = asm(LinearForm(neumann_form), bndry_basis)
        load_np += contribution

        return self._bkd.asarray(load_np.astype(np.float64))

    def __repr__(self) -> str:
        return (
            f"NeumannBC(boundary='{self._boundary_name}', "
            f"ndofs={len(self._bkd.to_numpy(self._boundary_dofs))})"
        )


class RobinBC(Generic[Array]):
    """Robin boundary condition: alpha * u + beta * (flux . n) = g(x, t).

    This is a mixed boundary condition that combines Dirichlet and Neumann.
    Special cases:
    - alpha=1, beta=0: Dirichlet BC
    - alpha=0, beta=1: Neumann BC

    In weak form for diffusion: -D * du/dn = alpha * u - g
    Contributes:
    - To stiffness matrix: alpha * integral_{Gamma} u * phi ds
    - To load vector: integral_{Gamma} g * phi ds

    Parameters
    ----------
    basis : GalerkinBasisProtocol[Array]
        Finite element basis.
    boundary_name : str
        Name of the boundary.
    alpha : float
        Coefficient for u term.
    value_func : Callable or float
        Function g(x, t) returning Robin values, or constant value.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        basis: GalerkinBasisProtocol[Array],
        boundary_name: str,
        alpha: float,
        value_func: Union[Callable, float],
        bkd: Backend[Array],
    ):
        self._basis = basis
        self._boundary_name = boundary_name
        self._alpha = alpha
        self._bkd = bkd

        # Store value function
        if callable(value_func):
            self._value_func = value_func
        else:
            const = float(value_func)
            self._value_func = lambda x, t=None: np.full(x.shape[1], const)

        # Get boundary DOFs
        self._boundary_dofs = basis.get_dofs(boundary_name)

        # Cache boundary basis
        self._boundary_basis: Optional[Basis] = None

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def boundary_name(self) -> str:
        """Return the boundary name."""
        return self._boundary_name

    def boundary_dofs(self) -> Array:
        """Return indices of DOFs on this boundary."""
        return self._boundary_dofs

    def alpha(self) -> float:
        """Return coefficient for u term."""
        return self._alpha

    def boundary_values(self, time: float = 0.0) -> Array:
        """Return Robin boundary values g at given time."""
        dof_coords = self._basis.dof_coordinates()
        dof_coords_np = self._bkd.to_numpy(dof_coords)
        bndry_dofs_np = self._bkd.to_numpy(self._boundary_dofs)
        bndry_coords = dof_coords_np[:, bndry_dofs_np]

        values_np = self._value_func(bndry_coords, time)
        return self._bkd.asarray(values_np.astype(np.float64))

    def _get_boundary_basis(self) -> Basis:
        """Get or create the boundary basis for assembly."""
        if self._boundary_basis is None:
            skfem_basis = self._basis.skfem_basis()
            self._boundary_basis = skfem_basis.boundary(self._boundary_name)
        return self._boundary_basis

    def apply_to_stiffness(self, stiffness: Array, time: float) -> Array:
        """Apply Robin BC contribution to stiffness matrix.

        Adds: alpha * integral_{Gamma} u . phi ds

        For vector elements uses sum_i(u_i * v_i); for scalar uses u * v.

        Parameters
        ----------
        stiffness : Array
            Stiffness matrix. Shape: (nstates, nstates)
        time : float
            Current time.

        Returns
        -------
        Array
            Modified stiffness matrix.
        """
        stiff_np = self._bkd.to_numpy(stiffness).copy()
        bndry_basis = self._get_boundary_basis()
        alpha = self._alpha
        ncomps = getattr(self._basis, 'ncomponents', lambda: 1)()

        if ncomps > 1:
            def robin_bilinear(u, v, w):
                return alpha * sum(u[i] * v[i] for i in range(ncomps))
        else:
            def robin_bilinear(u, v, w):
                return alpha * u * v

        contribution = asm(BilinearForm(robin_bilinear), bndry_basis).toarray()
        stiff_np += contribution

        return self._bkd.asarray(stiff_np.astype(np.float64))

    def apply_to_load(self, load: Array, time: float) -> Array:
        """Apply Robin BC contribution to load vector.

        Adds: integral_{Gamma} g . phi ds

        For vector elements, value_func returns (ndim, npts) and the
        form computes sum_i(g_i * v_i). For scalar, returns (npts,).

        Parameters
        ----------
        load : Array
            Load vector. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Modified load vector.
        """
        load_np = self._bkd.to_numpy(load).copy()
        bndry_basis = self._get_boundary_basis()
        value_func = self._value_func
        current_time = time

        def robin_linear(v, w):
            x_np = np.asarray(w.x)
            x_shape = x_np.shape
            if len(x_shape) == 3:
                ndim, nelem, nquad = x_shape
                x_flat = x_np.reshape(ndim, -1)
                vals_flat = np.asarray(value_func(x_flat, current_time))
                if vals_flat.ndim == 2 and vals_flat.shape[0] == ndim:
                    vals = vals_flat.reshape(ndim, nelem, nquad)
                    return sum(vals[i] * v[i] for i in range(ndim))
                else:
                    vals = vals_flat.reshape(nelem, nquad)
                    return vals * v
            else:
                vals = np.asarray(value_func(x_np, current_time))
                if vals.ndim == 2:
                    ndim = vals.shape[0]
                    return sum(vals[i] * v[i] for i in range(ndim))
                return vals * v

        contribution = asm(LinearForm(robin_linear), bndry_basis)
        load_np += contribution

        return self._bkd.asarray(load_np.astype(np.float64))

    def apply_to_residual(
        self, residual: Array, state: Array, time: float
    ) -> Array:
        """Apply Robin BC to residual.

        For residual form, adds: alpha * u - g contribution on boundary.

        Parameters
        ----------
        residual : Array
            Residual vector. Shape: (nstates,)
        state : Array
            Current solution. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Modified residual.
        """
        res_np = self._bkd.to_numpy(residual).copy()
        state_np = self._bkd.to_numpy(state)
        bndry_basis = self._get_boundary_basis()
        alpha = self._alpha
        value_func = self._value_func
        current_time = time

        # Add alpha * u * phi contribution to residual
        def robin_residual_u(v, w):
            return alpha * w.u_prev * v

        # Need to interpolate state onto boundary
        state_interp = bndry_basis.interpolate(state_np)
        contribution_u = asm(
            LinearForm(robin_residual_u), bndry_basis, u_prev=state_interp
        )
        res_np += contribution_u

        # Subtract g * phi contribution
        def robin_residual_g(v, w):
            # Note: w.x may be a DiscreteField, so convert to ndarray
            x_np = np.asarray(w.x)
            x_shape = x_np.shape
            if len(x_shape) == 3:
                ndim, nelem, nquad = x_shape
                x_flat = x_np.reshape(ndim, -1)
                vals_flat = value_func(x_flat, current_time)
                vals = vals_flat.reshape(nelem, nquad)
            else:
                vals = value_func(x_np, current_time)
            return vals * v

        contribution_g = asm(LinearForm(robin_residual_g), bndry_basis)
        res_np -= contribution_g

        return self._bkd.asarray(res_np.astype(np.float64))

    def apply_to_jacobian(
        self, jacobian: Array, state: Array, time: float
    ) -> Array:
        """Apply Robin BC to Jacobian.

        Adds: alpha * integral_{Gamma} u * phi ds to Jacobian.
        """
        return self.apply_to_stiffness(jacobian, time)

    def __repr__(self) -> str:
        return (
            f"RobinBC(boundary='{self._boundary_name}', "
            f"alpha={self._alpha}, "
            f"ndofs={len(self._bkd.to_numpy(self._boundary_dofs))})"
        )


class BoundaryConditionSet(Generic[Array]):
    """Collection of boundary conditions for a problem.

    Manages multiple boundary conditions and provides methods to apply
    them collectively to residuals, Jacobians, and load vectors.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]):
        self._bkd = bkd
        self._dirichlet_bcs: List[DirichletBC[Array]] = []
        self._neumann_bcs: List[NeumannBC[Array]] = []
        self._robin_bcs: List[RobinBC[Array]] = []

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def add_dirichlet(self, bc: DirichletBC[Array]) -> None:
        """Add a Dirichlet boundary condition."""
        self._dirichlet_bcs.append(bc)

    def add_neumann(self, bc: NeumannBC[Array]) -> None:
        """Add a Neumann boundary condition."""
        self._neumann_bcs.append(bc)

    def add_robin(self, bc: RobinBC[Array]) -> None:
        """Add a Robin boundary condition."""
        self._robin_bcs.append(bc)

    def ndirichlet(self) -> int:
        """Return number of Dirichlet BCs."""
        return len(self._dirichlet_bcs)

    def nneumann(self) -> int:
        """Return number of Neumann BCs."""
        return len(self._neumann_bcs)

    def nrobin(self) -> int:
        """Return number of Robin BCs."""
        return len(self._robin_bcs)

    def all_conditions(self) -> List[Union[DirichletBC[Array], NeumannBC[Array], RobinBC[Array]]]:
        """Return all boundary conditions as a flat list.

        The order is: Dirichlet, then Neumann, then Robin.
        This can be passed directly to physics classes that accept
        a list of BoundaryConditionProtocol objects.

        Returns
        -------
        List
            All boundary conditions.
        """
        return self._dirichlet_bcs + self._neumann_bcs + self._robin_bcs

    def dirichlet_dofs(self) -> Array:
        """Return all Dirichlet DOF indices."""
        if not self._dirichlet_bcs:
            return self._bkd.asarray(np.array([], dtype=np.int64))

        all_dofs = []
        for bc in self._dirichlet_bcs:
            all_dofs.append(self._bkd.to_numpy(bc.boundary_dofs()))

        return self._bkd.asarray(
            np.concatenate(all_dofs).astype(np.int64)
        )

    def dirichlet_values(self, time: float = 0.0) -> Array:
        """Return all Dirichlet values at given time."""
        if not self._dirichlet_bcs:
            return self._bkd.asarray(np.array([], dtype=np.float64))

        all_vals = []
        for bc in self._dirichlet_bcs:
            all_vals.append(self._bkd.to_numpy(bc.boundary_values(time)))

        return self._bkd.asarray(
            np.concatenate(all_vals).astype(np.float64)
        )

    def apply_to_residual(
        self, residual: Array, state: Array, time: float
    ) -> Array:
        """Apply all boundary conditions to residual."""
        res = residual

        # Apply Dirichlet BCs
        for bc in self._dirichlet_bcs:
            res = bc.apply_to_residual(res, state, time)

        # Apply Robin BCs
        for bc in self._robin_bcs:
            res = bc.apply_to_residual(res, state, time)

        return res

    def apply_to_jacobian(
        self, jacobian: Array, state: Array, time: float
    ) -> Array:
        """Apply all boundary conditions to Jacobian."""
        jac = jacobian

        # Apply Robin BCs (they modify interior of Jacobian)
        for bc in self._robin_bcs:
            jac = bc.apply_to_jacobian(jac, state, time)

        # Apply Dirichlet BCs (they replace rows)
        for bc in self._dirichlet_bcs:
            jac = bc.apply_to_jacobian(jac, state, time)

        return jac

    def apply_to_load(self, load: Array, time: float) -> Array:
        """Apply all boundary conditions to load vector."""
        # Apply Neumann BCs
        for bc in self._neumann_bcs:
            load = bc.apply_to_load(load, time)

        # Apply Robin BCs
        for bc in self._robin_bcs:
            load = bc.apply_to_load(load, time)

        return load

    def apply_to_stiffness(self, stiffness: Array, time: float) -> Array:
        """Apply all boundary conditions to stiffness matrix."""
        # Apply Robin BCs (they add boundary mass terms)
        for bc in self._robin_bcs:
            stiffness = bc.apply_to_stiffness(stiffness, time)

        return stiffness

    def set_time(self, time: float) -> None:
        """Set time for all time-dependent boundary conditions.

        This is a no-op for the current implementation since we pass
        time to each method. Provided for compatibility with legacy API.
        """
        pass

    def __repr__(self) -> str:
        return (
            f"BoundaryConditionSet("
            f"dirichlet={self.ndirichlet()}, "
            f"neumann={self.nneumann()}, "
            f"robin={self.nrobin()})"
        )
