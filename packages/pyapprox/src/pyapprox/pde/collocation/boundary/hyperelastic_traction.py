"""Hyperelastic traction normal operator for nonlinear Neumann BCs.

Computes one component of PK1 traction t_i = P_iJ * n_J at boundary
points for hyperelastic materials. Unlike TractionNormalOperator (linear
elasticity), the Jacobian is state-dependent because PK1 stress is
nonlinear in the deformation gradient.

Factory functions create RobinBC objects wrapping this operator.
"""

from typing import Callable, Generic, List, Protocol, Union, runtime_checkable

from pyapprox.pde.collocation.boundary.robin import RobinBC
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class PK1StressSourceProtocol(Protocol, Generic[Array]):
    """What the traction operator needs: the first Piola-Kirchhoff stress
    at every node and its Jacobian. A hyperelastic physics satisfies it,
    so the traction reads the material the physics holds."""

    def pk1_stress(self, state: Array) -> List[List[Array]]:
        """``P[i][J]`` at every node, each shape (npts,)."""
        ...

    def pk1_stress_jacobian(self, state: Array) -> List[List[Array]]:
        """``dP[i][J]/du``, each shape (npts, nstates)."""
        ...


class HyperelasticTractionNormalOperator(Generic[Array]):
    """Computes one component of PK1 traction t = P·n at boundary points.

    For hyperelasticity with component-stacked state, the traction
    component i is :math:`t_i = \\sum_J P_{iJ} n_J`, with P taken from the
    stress source.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    mesh_boundary_indices : Array
        Mesh point indices (0..npts-1) on boundary. Shape: (nboundary,)
    normals : Array
        Outward unit normal vectors at boundary points.
        Shape: (nboundary, ndim)
    stress_source : PK1StressSourceProtocol
        The PK1 stress and its Jacobian, usually the physics.
    component : int
        The traction component i returned.

    Raises
    ------
    TypeError
        If ``stress_source`` does not satisfy ``PK1StressSourceProtocol``.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        mesh_boundary_indices: Array,
        normals: Array,
        stress_source: PK1StressSourceProtocol[Array],
        component: int,
    ) -> None:
        if not isinstance(stress_source, PK1StressSourceProtocol):
            raise TypeError(
                "stress_source must satisfy PK1StressSourceProtocol, got "
                f"{type(stress_source).__name__}"
            )
        self._bkd = bkd
        self._mesh_boundary_indices = mesh_boundary_indices
        self._normals = normals
        self._stress_source = stress_source
        self._component = component

    def normals(self) -> Array:
        """Return outward unit normals. Shape: (nboundary, ndim)."""
        return self._normals

    def has_coefficient_dependence(self) -> bool:
        """Return True: PK1 stress depends on material parameters."""
        return True

    def __call__(self, state: Array, time: float) -> Array:
        """Compute one component of PK1 traction t = P·n at boundary.

        Parameters
        ----------
        state : Array
            Full component-stacked solution. Shape: (nstates,)
        time : float
            Unused: the stress source takes no time.

        Returns
        -------
        Array
            Traction component at boundary points. Shape: (nboundary,)
        """
        idx = self._mesh_boundary_indices
        stress_row = self._stress_source.pk1_stress(state)[self._component]
        traction = stress_row[0][idx] * self._normals[:, 0]
        for J in range(1, len(stress_row)):
            traction = traction + stress_row[J][idx] * self._normals[:, J]
        return traction

    def jacobian(self, state: Array, time: float) -> Array:
        """Return Jacobian of traction component w.r.t. state:
        :math:`\\sum_J n_J \\, \\partial P_{iJ}/\\partial u` at the
        boundary rows.

        Parameters
        ----------
        state : Array
            Full solution vector. Shape: (nstates,)
        time : float
            Unused: the stress source takes no time.

        Returns
        -------
        Array
            Jacobian. Shape: (nboundary, nstates)
        """
        idx = self._mesh_boundary_indices
        row = self._stress_source.pk1_stress_jacobian(state)[self._component]
        jac = self._normals[:, 0:1] * row[0][idx]
        for J in range(1, len(row)):
            jac = jac + self._normals[:, J : J + 1] * row[J][idx]
        return jac


def hyperelastic_traction_neumann_bc(
    bkd: Backend[Array],
    mesh_boundary_indices: Array,
    normals: Array,
    stress_source: PK1StressSourceProtocol[Array],
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
    stress_source : PK1StressSourceProtocol
        The PK1 stress and its Jacobian, usually the physics.
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
        bkd, mesh_boundary_indices, normals, stress_source, component
    )
    state_indices = mesh_boundary_indices + component * npts
    return RobinBC(bkd, state_indices, normal_op, 0.0, 1.0, values)


def hyperelastic_traction_robin_bc(
    bkd: Backend[Array],
    mesh_boundary_indices: Array,
    normals: Array,
    stress_source: PK1StressSourceProtocol[Array],
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
    stress_source : PK1StressSourceProtocol
        The PK1 stress and its Jacobian, usually the physics.
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
        bkd, mesh_boundary_indices, normals, stress_source, component
    )
    state_indices = mesh_boundary_indices + component * npts
    return RobinBC(bkd, state_indices, normal_op, alpha, beta, values)
