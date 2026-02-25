"""Euler-Bernoulli beam models: analytical and FEM.

Solves the cantilever beam equation:

    EI * w''''(x) = q(x)

where w is the transverse deflection, EI is the bending stiffness,
and q(x) is the distributed transverse load.

The FEM formulation uses cubic Hermite (C1) elements with DOFs
[w, dw/dx] at each node. The weak form is:

    integral(EI * w'' * v'') dx = integral(q * v) dx

for all test functions v in the Hermite finite element space.
"""

from typing import Callable, Generic, List, Optional, Union

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.pde.galerkin.physics.bc_mixin import GalerkinBCMixin
from pyapprox.pde.galerkin.boundary.implementations import DirectDirichletBC

try:
    from skfem import Basis, BilinearForm, LinearForm, asm
    from skfem.element import ElementLineHermite
    from skfem.mesh import MeshLine
    from skfem.utils import condense, solve as skfem_solve
except ImportError:
    raise ImportError(
        "scikit-fem is required for Galerkin module. "
        "Install with: pip install scikit-fem"
    )


class EulerBernoulliBeamAnalytical(Generic[Array]):
    """Analytical solution for a cantilever Euler-Bernoulli beam.

    Clamped at x=0 (w=0, w'=0), free at x=L (M=0, V=0).

    Supports two load types:
    - Uniform load: q(x) = q0
    - Linearly increasing load: q(x) = q0 * x / L

    Parameters
    ----------
    length : float
        Beam length L.
    EI : float
        Bending stiffness (Young's modulus times second moment of area).
    q0 : float
        Load magnitude.
    load_type : str, default="uniform"
        Either "uniform" for q(x)=q0 or "linear" for q(x)=q0*x/L.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        length: float,
        EI: float,
        q0: float,
        bkd: Backend[Array],
        load_type: str = "uniform",
    ):
        self._length = length
        self._EI = EI
        self._q0 = q0
        self._bkd = bkd
        if load_type not in ("uniform", "linear"):
            raise ValueError(
                f"load_type must be 'uniform' or 'linear', got '{load_type}'"
            )
        self._load_type = load_type

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def length(self) -> float:
        return self._length

    def EI(self) -> float:
        return self._EI

    def q0(self) -> float:
        return self._q0

    def deflection(self, x: Array) -> Array:
        """Compute deflection w(x).

        Parameters
        ----------
        x : Array
            Evaluation points. Shape: (npts,)

        Returns
        -------
        Array
            Deflection values. Shape: (npts,)
        """
        L = self._length
        EI = self._EI
        q0 = self._q0
        bkd = self._bkd

        if self._load_type == "uniform":
            # w(x) = q0/(24*EI) * x^2 * (6*L^2 - 4*L*x + x^2)
            return (q0 / (24.0 * EI)) * x**2 * (
                6.0 * L**2 - 4.0 * L * x + x**2
            )
        else:
            # w(x) = q0/(120*L*EI) * x^2 * (20*L^3 - 10*L^2*x + x^3)
            return (q0 / (120.0 * L * EI)) * x**2 * (
                20.0 * L**3 - 10.0 * L**2 * x + x**3
            )

    def slope(self, x: Array) -> Array:
        """Compute slope dw/dx(x).

        Parameters
        ----------
        x : Array
            Evaluation points. Shape: (npts,)

        Returns
        -------
        Array
            Slope values. Shape: (npts,)
        """
        L = self._length
        EI = self._EI
        q0 = self._q0

        if self._load_type == "uniform":
            # w'(x) = q0/(6*EI) * x * (3*L^2 - 3*L*x + x^2)
            return (q0 / (6.0 * EI)) * x * (
                3.0 * L**2 - 3.0 * L * x + x**2
            )
        else:
            # w'(x) = q0/(120*L*EI) * x * (40*L^3 - 30*L^2*x + 4*x^3)
            return (q0 / (120.0 * L * EI)) * x * (
                40.0 * L**3 - 30.0 * L**2 * x + 4.0 * x**3
            )

    def tip_deflection(self) -> float:
        """Compute tip deflection w(L).

        Returns
        -------
        float
            Deflection at x=L.
        """
        L = self._length
        EI = self._EI
        q0 = self._q0
        if self._load_type == "uniform":
            # w(L) = q0*L^4 / (8*EI)
            return q0 * L**4 / (8.0 * EI)
        else:
            # w(L) = 11*q0*L^4 / (120*EI)
            return 11.0 * q0 * L**4 / (120.0 * EI)


class EulerBernoulliBeamFEM(GalerkinBCMixin[Array], Generic[Array]):
    """FEM solution for Euler-Bernoulli beam using Hermite elements.

    Uses cubic Hermite (C1) elements with DOFs [w, dw/dx] at each node.
    Cantilever boundary conditions: clamped at x=0 (w=0, w'=0),
    free at x=L (natural BCs from weak form).

    The physics interface follows the standard pattern:
    - ``residual(state, time) = b - K*u`` with Dirichlet rows replaced
    - ``jacobian(state, time) = -K`` with Dirichlet rows replaced

    Parameters
    ----------
    nx : int
        Number of elements.
    length : float
        Beam length L.
    EI : Union[float, np.ndarray]
        Bending stiffness. Scalar for uniform EI, or array of length
        ``nx`` for per-element EI.
    load_func : Callable
        Distributed load function q(x). Takes numpy array of x-coordinates
        and returns array of load values.
    bkd : Backend[Array]
        Computational backend.
    dirichlet_dofs : list of int, optional
        DOF indices to clamp. Default: clamped left end (DOFs 0, 1).
    dirichlet_values : list of float, optional
        Values for clamped DOFs. Default: zeros.
    """

    def __init__(
        self,
        nx: int,
        length: float,
        EI: Union[float, np.ndarray],
        load_func: Callable,
        bkd: Backend[Array],
        dirichlet_dofs: Optional[List[int]] = None,
        dirichlet_values: Optional[List[float]] = None,
    ):
        self._nx = nx
        self._length = length
        self._EI = EI
        self._load_func = load_func
        self._bkd = bkd

        # Build mesh
        nodes = np.linspace(0, length, nx + 1)
        self._mesh = MeshLine(nodes).with_boundaries(
            {
                "left": lambda x: np.abs(x[0] - 0.0) < 1e-12,
                "right": lambda x: np.abs(x[0] - length) < 1e-12,
            }
        )

        # Build Hermite basis
        self._element = ElementLineHermite()
        self._skfem_basis = Basis(self._mesh, self._element)

        # DOF info: 2 per node (w, dw/dx), total = 2*(nx+1)
        self._ndofs = self._skfem_basis.N

        # Dirichlet DOFs
        if dirichlet_dofs is not None:
            dof_indices = np.array(dirichlet_dofs, dtype=np.int64)
            dof_values = np.array(
                dirichlet_values if dirichlet_values is not None
                else [0.0] * len(dirichlet_dofs),
                dtype=np.float64,
            )
        else:
            # Default: clamped left end (w=0, dw/dx=0)
            left_dof_indices = self._skfem_basis.get_dofs("left").flatten()
            dof_indices = left_dof_indices.astype(np.int64)
            dof_values = np.zeros(len(dof_indices), dtype=np.float64)

        self._boundary_conditions = [
            DirectDirichletBC(dof_indices, dof_values, bkd)
        ]

        # Assembly caches
        self._stiffness: Optional[Array] = None
        self._mass: Optional[Array] = None
        self._solution: Optional[Array] = None

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nstates(self) -> int:
        """Return total number of DOFs (2 per node)."""
        return self._ndofs

    def nnodes(self) -> int:
        """Return number of mesh nodes."""
        return self._nx + 1

    def length(self) -> float:
        return self._length

    def EI(self) -> Union[float, np.ndarray]:
        return self._EI

    def set_EI(self, EI: Union[float, np.ndarray]) -> None:
        """Update bending stiffness and invalidate cached matrices/solution."""
        self._EI = EI
        self._stiffness = None
        self._solution = None

    def set_load_func(self, load_func: Callable) -> None:
        """Update load function and invalidate cached solution."""
        self._load_func = load_func
        self._solution = None

    def node_coordinates(self) -> np.ndarray:
        """Return node x-coordinates. Shape: (nnodes,)"""
        return self._mesh.p[0]

    def stiffness_matrix(self) -> Array:
        """Return beam stiffness matrix K_ij = integral(EI * w''_i * w''_j).

        Returns
        -------
        Array
            Stiffness matrix. Shape: (ndofs, ndofs)
        """
        if self._stiffness is not None:
            return self._stiffness

        EI_val = self._EI

        if isinstance(EI_val, np.ndarray):
            x_nodes = self.node_coordinates()
            ei_array = EI_val

            @BilinearForm
            def beam_stiffness_form(u, v, w):
                x = w.x[0]
                elem_idx = np.clip(
                    np.searchsorted(x_nodes[1:], x), 0, len(ei_array) - 1
                )
                return ei_array[elem_idx] * u.hess[0, 0] * v.hess[0, 0]
        else:
            @BilinearForm
            def beam_stiffness_form(u, v, w):
                return EI_val * u.hess[0, 0] * v.hess[0, 0]

        self._stiffness = asm(beam_stiffness_form, self._skfem_basis)
        return self._stiffness

    def mass_matrix(self) -> Array:
        """Return beam mass matrix M_ij = integral(w_i * w_j).

        Returns
        -------
        Array
            Mass matrix. Shape: (ndofs, ndofs)
        """
        if self._mass is not None:
            return self._mass

        @BilinearForm
        def beam_mass_form(u, v, w):
            return u.value * v.value

        self._mass = asm(beam_mass_form, self._skfem_basis)
        return self._mass

    def load_vector(self) -> Array:
        """Return load vector b_i = integral(q(x) * w_i).

        Returns
        -------
        Array
            Load vector. Shape: (ndofs,)
        """
        load_func = self._load_func

        @LinearForm
        def beam_load_form(v, w):
            x = w.x[0]
            return load_func(x) * v.value

        f = asm(beam_load_form, self._skfem_basis)
        return self._bkd.asarray(f.astype(np.float64))

    def spatial_residual(self, state: Array, time: float = 0.0) -> Array:
        """Compute F = b - K*u without BC enforcement.

        Parameters
        ----------
        state : Array
            DOF vector. Shape: (ndofs,)
        time : float
            Current time.

        Returns
        -------
        Array
            Residual. Shape: (ndofs,)
        """
        K = self.stiffness_matrix()
        b = self.load_vector()
        return b - K @ state

    def residual(self, state: Array, time: float = 0.0) -> Array:
        """Compute residual with Dirichlet BCs enforced.

        At Dirichlet DOFs, the residual is ``state[dof] - g``.

        Parameters
        ----------
        state : Array
            DOF vector. Shape: (ndofs,)
        time : float
            Current time.

        Returns
        -------
        Array
            Residual. Shape: (ndofs,)
        """
        res = self.spatial_residual(state, time)
        return self._apply_dirichlet_to_residual(res, state, time)

    def spatial_jacobian(self, state: Array, time: float = 0.0) -> Array:
        """Compute dF/du = -K without BC enforcement.

        Parameters
        ----------
        state : Array
            DOF vector. Shape: (ndofs,)
        time : float
            Current time.

        Returns
        -------
        Array
            Jacobian. Shape: (ndofs, ndofs)
        """
        return -self.stiffness_matrix()

    def jacobian(self, state: Array, time: float = 0.0) -> Array:
        """Compute Jacobian with Dirichlet BCs enforced.

        At Dirichlet DOFs, the row becomes an identity row.

        Parameters
        ----------
        state : Array
            DOF vector. Shape: (ndofs,)
        time : float
            Current time.

        Returns
        -------
        Array
            Jacobian. Shape: (ndofs, ndofs)
        """
        jac = self.spatial_jacobian(state, time)
        return self._apply_dirichlet_to_jacobian(jac, state, time)

    def interpolate_manufactured(
        self,
        w_func: Callable,
        dwdx_func: Callable,
    ) -> Array:
        """Build DOF vector from manufactured solution functions.

        Parameters
        ----------
        w_func : Callable
            Deflection function w(x). Takes array, returns array.
        dwdx_func : Callable
            Slope function dw/dx(x). Takes array, returns array.

        Returns
        -------
        Array
            DOF vector. Shape: (ndofs,)
        """
        x_nodes = self.node_coordinates()
        w_vals = w_func(x_nodes)
        dwdx_vals = dwdx_func(x_nodes)

        # Interleave: [w0, dw0/dx, w1, dw1/dx, ...]
        dofs = np.zeros(self._ndofs)
        dofs[0::2] = w_vals
        dofs[1::2] = dwdx_vals
        return self._bkd.asarray(dofs.astype(np.float64))

    def solve(self) -> Array:
        """Solve the static beam problem.

        Returns
        -------
        Array
            DOF solution vector [w0, dw0/dx, w1, dw1/dx, ...].
            Shape: (ndofs,)
        """
        if self._solution is not None:
            return self._solution

        K_sp = self.stiffness_matrix()
        f_np = self._bkd.to_numpy(self.load_vector())

        dof_set = self._skfem_basis.get_dofs("left")
        u = skfem_solve(*condense(K_sp, f_np, D=dof_set))
        self._solution = self._bkd.asarray(u.astype(np.float64))
        return self._solution

    def deflection_at_nodes(self) -> Array:
        """Extract deflection values at mesh nodes from solution.

        Returns
        -------
        Array
            Deflection at each node. Shape: (nnodes,)
        """
        sol = self.solve()
        sol_np = self._bkd.to_numpy(sol)
        w_nodes = sol_np[0::2]
        return self._bkd.asarray(w_nodes.astype(np.float64))

    def slope_at_nodes(self) -> Array:
        """Extract slope values at mesh nodes from solution.

        Returns
        -------
        Array
            Slope (dw/dx) at each node. Shape: (nnodes,)
        """
        sol = self.solve()
        sol_np = self._bkd.to_numpy(sol)
        dwdx_nodes = sol_np[1::2]
        return self._bkd.asarray(dwdx_nodes.astype(np.float64))

    def tip_deflection(self) -> float:
        """Compute tip deflection w(L).

        Returns
        -------
        float
            Deflection at x=L.
        """
        sol = self.solve()
        sol_np = self._bkd.to_numpy(sol)
        return float(sol_np[-2])

    def curvature_at_elements(self) -> np.ndarray:
        """Element-wise absolute curvature |d^2w/dx^2|.

        Approximated by finite-differencing nodal slopes dw/dx.

        Returns
        -------
        np.ndarray
            Absolute curvature per element, shape (nelements,).
        """
        slopes = self._bkd.to_numpy(self.slope_at_nodes())
        x_nodes = self.node_coordinates()
        return np.abs(np.diff(slopes) / np.diff(x_nodes))

    def max_curvature(self) -> float:
        """Compute max absolute curvature max|d^2w/dx^2|.

        Returns
        -------
        float
            Maximum absolute curvature.
        """
        return float(np.max(self.curvature_at_elements()))
