"""Dirichlet boundary conditions for spectral collocation methods.

Enforces u = g(x, t) on the boundary.
"""

from typing import Callable, Generic, Union

from pyapprox.util.backends.protocols import Array, Backend


class DirichletBC(Generic[Array]):
    """Dirichlet boundary condition.

    Enforces u = g on the boundary, where g can be a constant,
    array of values, or a time-dependent function.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    boundary_indices : Array
        Indices of mesh points on this boundary. Shape: (nboundary_pts,)
    values : float, Array, or Callable
        Boundary values. Can be:
        - float: constant value for all boundary points
        - Array: values at each boundary point, shape (nboundary_pts,)
        - Callable[[float], Array]: time-dependent function returning
          boundary values
    """

    def __init__(
        self,
        bkd: Backend[Array],
        boundary_indices: Array,
        values: Union[float, Array, Callable[[float], Array]],
    ):
        self._bkd = bkd
        self._boundary_indices = boundary_indices
        self._nboundary_pts = boundary_indices.shape[0]

        # Store values in a normalized form
        if callable(values):
            self._values_func = values
            self._is_time_dependent = True
        elif isinstance(values, (int, float)):
            self._constant_values = bkd.full((self._nboundary_pts,), float(values))
            self._is_time_dependent = False
        else:
            # Array of values
            if values.shape != (self._nboundary_pts,):
                raise ValueError(
                    f"values shape {values.shape} must match boundary points "
                    f"({self._nboundary_pts},)"
                )
            self._constant_values = values
            self._is_time_dependent = False

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def boundary_indices(self) -> Array:
        """Return indices of mesh points on this boundary."""
        return self._boundary_indices

    def is_essential(self) -> bool:
        """Return True: Dirichlet BCs directly constrain DOF values."""
        return True

    def boundary_values(self, time: float) -> Array:
        """Return Dirichlet boundary values at given time.

        Parameters
        ----------
        time : float
            Current time.

        Returns
        -------
        Array
            Boundary values. Shape: (nboundary_pts,)
        """
        if self._is_time_dependent:
            return self._values_func(time)
        return self._constant_values

    def apply_to_residual(self, residual: Array, state: Array, time: float) -> Array:
        """Apply Dirichlet BC to residual.

        Sets residual at boundary points to: u - g

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
        idx = self._boundary_indices
        g = self.boundary_values(time)
        residual = self._bkd.copy(residual)
        for i in range(self._nboundary_pts):
            residual[idx[i]] = state[idx[i]] - g[i]
        return residual

    def apply_to_jacobian(self, jacobian: Array, state: Array, time: float) -> Array:
        """Apply Dirichlet BC to Jacobian.

        Sets Jacobian rows at boundary points to identity rows.

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
        idx = self._boundary_indices
        jacobian = self._bkd.copy(jacobian)
        nstates = jacobian.shape[0]
        for i in range(self._nboundary_pts):
            # Zero out row
            for j in range(nstates):
                jacobian[idx[i], j] = 0.0
            # Set diagonal to 1
            jacobian[idx[i], idx[i]] = 1.0
        return jacobian

    def apply_to_param_jacobian(
        self,
        param_jacobian: Array,
        state: Array,
        time: float,
        physical_sensitivities=None,
    ) -> Array:
        """Apply Dirichlet BC to parameter Jacobian.

        Sets parameter Jacobian rows at boundary points to zero
        (boundary values don't depend on parameters in this implementation).

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
        idx = self._boundary_indices
        param_jacobian = self._bkd.copy(param_jacobian)
        nparams = param_jacobian.shape[1]
        for i in range(self._nboundary_pts):
            for j in range(nparams):
                param_jacobian[idx[i], j] = 0.0
        return param_jacobian


def constant_dirichlet_bc(
    bkd: Backend[Array],
    boundary_indices: Array,
    value: float,
) -> DirichletBC[Array]:
    """Create a constant Dirichlet boundary condition.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    boundary_indices : Array
        Indices of mesh points on this boundary.
    value : float
        Constant boundary value.

    Returns
    -------
    DirichletBC
        Dirichlet boundary condition.
    """
    return DirichletBC(bkd, boundary_indices, value)


def zero_dirichlet_bc(
    bkd: Backend[Array],
    boundary_indices: Array,
) -> DirichletBC[Array]:
    """Create a homogeneous Dirichlet boundary condition (u = 0).

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    boundary_indices : Array
        Indices of mesh points on this boundary.

    Returns
    -------
    DirichletBC
        Zero Dirichlet boundary condition.
    """
    return DirichletBC(bkd, boundary_indices, 0.0)
