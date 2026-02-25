"""Neumann boundary conditions for spectral collocation methods.

Enforces du/dn = g(x, t) on the boundary, where n is the outward normal.

Conventions:
- "gradient": g represents the normal derivative du/dn
- "flux": g represents the diffusive flux -D*du/dn (requires diffusivity)

Note: Both conventions use the same implementation. The convention parameter
is primarily for documentation and to compute appropriate g values from
manufactured solutions.
"""

from typing import Callable, Generic, Optional, Union

from pyapprox.util.backends.protocols import Array, Backend


class NeumannBC(Generic[Array]):
    """Neumann boundary condition.

    Enforces du/dn = g on the boundary, where n is the outward normal
    and g can be a constant, array, or time-dependent function.

    Two conventions are supported:
    - "gradient" (default): g = du/dn (normal derivative)
    - "flux": g = -D*du/dn (diffusive flux, requires diffusivity)

    For gradient convention, the BC enforces: du/dn = g
    For flux convention with diffusivity D, the BC enforces: du/dn = -g/D
    (the diffusivity scaling is handled by passing scaled values)

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    boundary_indices : Array
        Indices of mesh points on this boundary. Shape: (nboundary_pts,)
    derivative_matrix : Array
        Derivative matrix for computing du/dn at boundary points.
        Shape: (nboundary_pts, npts)
    normal_sign : float
        Sign of the outward normal (+1 or -1). For 1D left boundary,
        normal points left so sign is -1. For right boundary, +1.
    values : float, Array, or Callable
        Boundary values. Can be:
        - float: constant value for all boundary points
        - Array: values at each boundary point, shape (nboundary_pts,)
        - Callable[[float], Array]: time-dependent function
    convention : str, optional
        Boundary condition convention: "gradient" (default) or "flux".
        This affects how the values are interpreted (see class docstring).
    diffusivity : float or Array, optional
        Diffusivity for flux convention. Required if convention="flux".
        Scalar or shape (nboundary_pts,).
    """

    def __init__(
        self,
        bkd: Backend[Array],
        boundary_indices: Array,
        derivative_matrix: Array,
        normal_sign: float,
        values: Union[float, Array, Callable[[float], Array]],
        convention: str = "gradient",
        diffusivity: Optional[Union[float, Array]] = None,
    ):
        self._bkd = bkd
        self._boundary_indices = boundary_indices
        self._derivative_matrix = derivative_matrix
        self._normal_sign = normal_sign
        self._nboundary_pts = boundary_indices.shape[0]

        # Store values in normalized form
        if callable(values):
            self._values_func = values
            self._is_time_dependent = True
        elif isinstance(values, (int, float)):
            self._constant_values = bkd.full((self._nboundary_pts,), float(values))
            self._is_time_dependent = False
        else:
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

    def boundary_values(self, time: float) -> Array:
        """Return Neumann boundary values at given time.

        Parameters
        ----------
        time : float
            Current time.

        Returns
        -------
        Array
            Boundary flux values. Shape: (nboundary_pts,)
        """
        if self._is_time_dependent:
            return self._values_func(time)
        return self._constant_values

    def apply_to_residual(self, residual: Array, state: Array, time: float) -> Array:
        """Apply Neumann BC to residual.

        Sets residual at boundary points to: normal_sign * D @ u - g

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

        # Compute du/dn = normal_sign * D @ u
        flux = self._normal_sign * (self._derivative_matrix @ state)

        for i in range(self._nboundary_pts):
            residual[idx[i]] = flux[i] - g[i]
        return residual

    def apply_to_jacobian(self, jacobian: Array, state: Array, time: float) -> Array:
        """Apply Neumann BC to Jacobian.

        Sets Jacobian rows at boundary points to derivative matrix rows.

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
            # Set row to normal_sign * derivative matrix row
            for j in range(nstates):
                jacobian[idx[i], j] = self._normal_sign * self._derivative_matrix[i, j]
        return jacobian

    def apply_to_param_jacobian(
        self,
        param_jacobian: Array,
        state: Array,
        time: float,
        physical_sensitivities=None,
    ) -> Array:
        """Apply Neumann BC to parameter Jacobian.

        Sets parameter Jacobian rows at boundary points to zero.

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


def zero_neumann_bc(
    bkd: Backend[Array],
    boundary_indices: Array,
    derivative_matrix: Array,
    normal_sign: float,
) -> NeumannBC[Array]:
    """Create a homogeneous Neumann BC (du/dn = 0).

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    boundary_indices : Array
        Indices of mesh points on this boundary.
    derivative_matrix : Array
        Derivative matrix for computing du/dn.
    normal_sign : float
        Sign of outward normal (+1 or -1).

    Returns
    -------
    NeumannBC
        Zero flux Neumann boundary condition.
    """
    return NeumannBC(bkd, boundary_indices, derivative_matrix, normal_sign, 0.0)
