"""Hyperelastic traction normal operator for nonlinear Neumann BCs.

Computes one component of PK1 traction t_i = P_iJ * n_J at boundary
points for hyperelastic materials. Unlike TractionNormalOperator (linear
elasticity), the Jacobian is state-dependent because PK1 stress is
nonlinear in the deformation gradient.

Factory functions create RobinBC objects wrapping this operator.
"""

from typing import Generic, List, Union, Callable

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.pde.collocation.boundary.robin import RobinBC


class HyperelasticTractionNormalOperator(Generic[Array]):
    """Computes one component of PK1 traction t = P·n at boundary points.

    For 2D hyperelasticity with component-stacked state [u_x, u_y]:
        F = I + grad(u)
        P = stress_model.compute_stress_2d(F11, F12, F21, F22)
        t_i = P_{i1} * n_x + P_{i2} * n_y

    This operator returns t_x (component=0) or t_y (component=1).

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    mesh_boundary_indices : Array
        Mesh point indices (0..npts-1) on boundary. Shape: (nboundary,)
    normals : Array
        Outward unit normal vectors at boundary points.
        Shape: (nboundary, 2)
    derivative_matrices : List[Array]
        [Dx, Dy], each shape (npts, npts). Physical derivative matrices.
    stress_model
        Stress model providing compute_stress_2d and compute_tangent_2d.
    npts : int
        Number of mesh points (state length = 2*npts).
    component : int
        0 for t_x, 1 for t_y.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        mesh_boundary_indices: Array,
        normals: Array,
        derivative_matrices: List[Array],
        stress_model,
        npts: int,
        component: int,
    ):
        self._bkd = bkd
        self._mesh_boundary_indices = mesh_boundary_indices
        self._normals = normals
        self._Dx = derivative_matrices[0]
        self._Dy = derivative_matrices[1]
        self._stress_model = stress_model
        self._npts = npts
        self._component = component
        self._nboundary = mesh_boundary_indices.shape[0]

    def normals(self) -> Array:
        """Return outward unit normals. Shape: (nboundary, 2)."""
        return self._normals

    def has_coefficient_dependence(self) -> bool:
        """Return True: PK1 stress depends on material parameters."""
        return True

    def _compute_F_full(self, state: Array):
        """Compute deformation gradient components at all mesh points."""
        npts = self._npts
        u = state[:npts]
        v = state[npts:]
        Dx, Dy = self._Dx, self._Dy

        F11 = 1.0 + Dx @ u
        F12 = Dy @ u
        F21 = Dx @ v
        F22 = 1.0 + Dy @ v
        return F11, F12, F21, F22

    def _subset_to_boundary(self, *arrays):
        """Extract boundary-point values from full-field arrays."""
        idx = self._mesh_boundary_indices
        return tuple(a[idx] for a in arrays)

    def __call__(self, state: Array) -> Array:
        """Compute one component of PK1 traction t = P·n at boundary.

        Parameters
        ----------
        state : Array
            Full component-stacked solution [u_x, u_y]. Shape: (2*npts,)

        Returns
        -------
        Array
            Traction component at boundary points. Shape: (nboundary,)
        """
        bkd = self._bkd
        # Compute stress at all points (handles spatially-varying material)
        F11, F12, F21, F22 = self._compute_F_full(state)
        P11, P12, P21, P22 = self._stress_model.compute_stress_2d(
            F11, F12, F21, F22, bkd
        )
        # Extract boundary subset
        P11, P12, P21, P22 = self._subset_to_boundary(P11, P12, P21, P22)

        nx = self._normals[:, 0]
        ny = self._normals[:, 1]
        if self._component == 0:
            return P11 * nx + P12 * ny
        else:
            return P21 * nx + P22 * ny

    def jacobian(self, state: Array) -> Array:
        """Return Jacobian of traction component w.r.t. state.

        State-dependent (nonlinear hyperelastic).

        Parameters
        ----------
        state : Array
            Full solution vector. Shape: (2*npts,)

        Returns
        -------
        Array
            Jacobian. Shape: (nboundary, 2*npts)
        """
        bkd = self._bkd
        npts = self._npts
        idx = self._mesh_boundary_indices
        Dx, Dy = self._Dx, self._Dy
        D = [Dx, Dy]

        # Compute tangent at all points, then subset to boundary
        F11, F12, F21, F22 = self._compute_F_full(state)
        A_full = self._stress_model.compute_tangent_2d(
            F11, F12, F21, F22, bkd
        )
        A = {k: v[idx] for k, v in A_full.items()}

        # Map (i,J) to key index: i in {1,2}, J in {1,2}
        # Component i+1 of traction: t_{i+1} = sum_J P_{i+1,J} * n_J
        # d(t_{i+1})/d(state) = sum_J n_J * d(P_{i+1,J})/d(state)
        # d(P_{i+1,J})/d(state) = sum_k sum_L A_{i+1,J,k,L} * D_L[bdry,:]
        #   where D_L acts on component k: state block [k*npts:(k+1)*npts]

        comp = self._component + 1  # 1-indexed for A keys
        nx = self._normals[:, 0]
        ny = self._normals[:, 1]
        n_vec = [nx, ny]

        jac = bkd.zeros((self._nboundary, 2 * npts))
        jac = bkd.copy(jac)

        for J_idx in range(2):  # J = 1,2
            J = J_idx + 1
            for k_idx in range(2):  # k = 1,2
                k = k_idx + 1
                for L_idx in range(2):  # L = 1,2
                    L = L_idx + 1
                    key = f"A_{comp}{J}{k}{L}"
                    A_vals = A[key]  # shape (nboundary,)
                    # Contribution: n_J * A_{comp,J,k,L} * D_L[bdry, :]
                    # goes into block k (columns k_idx*npts:(k_idx+1)*npts)
                    for b in range(self._nboundary):
                        mesh_idx = int(idx[b])
                        coeff = n_vec[J_idx][b] * A_vals[b]
                        jac[b, k_idx * npts: (k_idx + 1) * npts] = (
                            jac[b, k_idx * npts: (k_idx + 1) * npts]
                            + coeff * D[L_idx][mesh_idx, :]
                        )

        return jac


def hyperelastic_traction_neumann_bc(
    bkd: Backend[Array],
    mesh_boundary_indices: Array,
    normals: Array,
    derivative_matrices: List[Array],
    stress_model,
    npts: int,
    component: int,
    values: Union[float, Array, Callable[[float], Array]] = 0.0,
) -> RobinBC[Array]:
    """Create Neumann BC for one component of hyperelastic PK1 traction.

    Enforces t_comp = g_comp at boundary, where t = P·n.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    mesh_boundary_indices : Array
        Mesh point indices (0..npts-1). Shape: (nboundary,)
    normals : Array
        Outward unit normals. Shape: (nboundary, 2)
    derivative_matrices : List[Array]
        [Dx, Dy], each shape (npts, npts).
    stress_model
        Hyperelastic stress model (e.g., NeoHookeanStress).
    npts : int
        Number of mesh points.
    component : int
        0 for t_x, 1 for t_y.
    values : float, Array, or Callable
        Traction values g_comp. Default: 0.0 (traction-free).

    Returns
    -------
    RobinBC
        Traction Neumann BC as Robin with alpha=0, beta=1.
    """
    normal_op = HyperelasticTractionNormalOperator(
        bkd, mesh_boundary_indices, normals, derivative_matrices,
        stress_model, npts, component,
    )
    state_indices = mesh_boundary_indices + component * npts
    return RobinBC(bkd, state_indices, normal_op, 0.0, 1.0, values)


def hyperelastic_traction_robin_bc(
    bkd: Backend[Array],
    mesh_boundary_indices: Array,
    normals: Array,
    derivative_matrices: List[Array],
    stress_model,
    npts: int,
    component: int,
    alpha: Union[float, Array],
    beta: Union[float, Array],
    values: Union[float, Array, Callable[[float], Array]],
) -> RobinBC[Array]:
    """Create Robin BC for one component of hyperelastic PK1 traction.

    Enforces alpha * u_comp + beta * t_comp = g_comp at boundary.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    mesh_boundary_indices : Array
        Mesh point indices (0..npts-1). Shape: (nboundary,)
    normals : Array
        Outward unit normals. Shape: (nboundary, 2)
    derivative_matrices : List[Array]
        [Dx, Dy], each shape (npts, npts).
    stress_model
        Hyperelastic stress model.
    npts : int
        Number of mesh points.
    component : int
        0 for u_x/t_x, 1 for u_y/t_y.
    alpha : float or Array
        Coefficient for u_comp term.
    beta : float or Array
        Coefficient for t_comp term.
    values : float, Array, or Callable
        Boundary values g_comp.

    Returns
    -------
    RobinBC
        Robin BC with hyperelastic traction normal operator.
    """
    normal_op = HyperelasticTractionNormalOperator(
        bkd, mesh_boundary_indices, normals, derivative_matrices,
        stress_model, npts, component,
    )
    state_indices = mesh_boundary_indices + component * npts
    return RobinBC(bkd, state_indices, normal_op, alpha, beta, values)
