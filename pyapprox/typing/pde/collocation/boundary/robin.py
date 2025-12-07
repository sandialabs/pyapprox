"""Robin boundary conditions for spectral collocation methods.

Enforces alpha * u + beta * du/dn = g(x, t) on the boundary.

Special cases:
- alpha=1, beta=0: Dirichlet (u = g)
- alpha=0, beta=1: Neumann (du/dn = g)
"""

from typing import Generic, Callable, Union

from pyapprox.typing.util.backends.protocols import Array, Backend


class RobinBC(Generic[Array]):
    """Robin boundary condition.

    Enforces alpha * u + beta * du/dn = g on the boundary.

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
        Sign of the outward normal (+1 or -1).
    alpha : float or Array
        Coefficient for u term. Scalar or shape (nboundary_pts,).
    beta : float or Array
        Coefficient for du/dn term. Scalar or shape (nboundary_pts,).
    values : float, Array, or Callable
        Boundary values g. Can be:
        - float: constant value for all boundary points
        - Array: values at each boundary point, shape (nboundary_pts,)
        - Callable[[float], Array]: time-dependent function
    """

    def __init__(
        self,
        bkd: Backend[Array],
        boundary_indices: Array,
        derivative_matrix: Array,
        normal_sign: float,
        alpha: Union[float, Array],
        beta: Union[float, Array],
        values: Union[float, Array, Callable[[float], Array]],
    ):
        self._bkd = bkd
        self._boundary_indices = boundary_indices
        self._derivative_matrix = derivative_matrix
        self._normal_sign = normal_sign
        self._nboundary_pts = boundary_indices.shape[0]

        # Normalize alpha and beta to arrays
        if isinstance(alpha, (int, float)):
            self._alpha = bkd.full((self._nboundary_pts,), float(alpha))
        else:
            self._alpha = alpha

        if isinstance(beta, (int, float)):
            self._beta = bkd.full((self._nboundary_pts,), float(beta))
        else:
            self._beta = beta

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

    def alpha(self) -> Array:
        """Return coefficient for u term."""
        return self._alpha

    def beta(self) -> Array:
        """Return coefficient for du/dn term."""
        return self._beta

    def boundary_values(self, time: float) -> Array:
        """Return Robin boundary values at given time.

        Parameters
        ----------
        time : float
            Current time.

        Returns
        -------
        Array
            Boundary values g. Shape: (nboundary_pts,)
        """
        if self._is_time_dependent:
            return self._values_func(time)
        return self._constant_values

    def apply_to_residual(
        self, residual: Array, state: Array, time: float
    ) -> Array:
        """Apply Robin BC to residual.

        Sets residual at boundary points to:
        alpha * u + beta * (normal_sign * D @ u) - g

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
            u_bndry = state[idx[i]]
            residual[idx[i]] = (
                self._alpha[i] * u_bndry + self._beta[i] * flux[i] - g[i]
            )
        return residual

    def apply_to_jacobian(
        self, jacobian: Array, state: Array, time: float
    ) -> Array:
        """Apply Robin BC to Jacobian.

        Sets Jacobian rows at boundary points to:
        alpha * I_row + beta * normal_sign * D_row

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
            row_idx = idx[i]
            for j in range(nstates):
                # beta * normal_sign * D[i, j]
                jac_val = self._beta[i] * self._normal_sign * self._derivative_matrix[i, j]
                # Add alpha contribution only on diagonal
                if j == row_idx:
                    jac_val = jac_val + self._alpha[i]
                jacobian[row_idx, j] = jac_val
        return jacobian

    def apply_to_param_jacobian(
        self, param_jacobian: Array, state: Array, time: float
    ) -> Array:
        """Apply Robin BC to parameter Jacobian.

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


def homogeneous_robin_bc(
    bkd: Backend[Array],
    boundary_indices: Array,
    derivative_matrix: Array,
    normal_sign: float,
    alpha: Union[float, Array],
    beta: Union[float, Array],
) -> RobinBC[Array]:
    """Create a homogeneous Robin BC (alpha * u + beta * du/dn = 0).

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
    alpha : float or Array
        Coefficient for u term.
    beta : float or Array
        Coefficient for du/dn term.

    Returns
    -------
    RobinBC
        Homogeneous Robin boundary condition.
    """
    return RobinBC(
        bkd, boundary_indices, derivative_matrix, normal_sign, alpha, beta, 0.0
    )
