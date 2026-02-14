"""Robin boundary conditions for spectral collocation methods.

Enforces alpha * u + beta * N(u) = g(x, t) on the boundary,
where N(u) is a normal operator (gradient or flux convention).

Special cases:
- alpha=1, beta=0: Dirichlet (u = g)
- alpha=0, beta=1: Neumann (N(u) = g)

Factory functions:
- gradient_robin_bc: N(u) = grad(u) . n
- flux_robin_bc: N(u) = flux(u) . n (total conservative flux)
- gradient_neumann_bc: grad(u) . n = g
- flux_neumann_bc: flux(u) . n = g
- homogeneous_robin_bc: backward-compatible API
"""

from typing import Generic, Callable, List, Union

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.pde.collocation.protocols.boundary import (
    NormalOperatorProtocol,
    FluxProviderProtocol,
)
from pyapprox.typing.pde.collocation.boundary.normal_operators import (
    GradientNormalOperator,
    FluxNormalOperator,
    TractionNormalOperator,
    _LegacyNormalOperator,
)


class RobinBC(Generic[Array]):
    """Robin boundary condition.

    Enforces alpha * u + beta * N(u) = g on the boundary,
    where N(u) is a normal operator providing the boundary normal term.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    boundary_indices : Array
        Indices of mesh points on this boundary. Shape: (nboundary_pts,)
    normal_operator : NormalOperatorProtocol
        Operator computing the normal term N(u) at boundary points.
    alpha : float or Array
        Coefficient for u term. Scalar or shape (nboundary_pts,).
    beta : float or Array
        Coefficient for N(u) term. Scalar or shape (nboundary_pts,).
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
        normal_operator: NormalOperatorProtocol[Array],
        alpha: Union[float, Array],
        beta: Union[float, Array],
        values: Union[float, Array, Callable[[float], Array]],
    ):
        self._bkd = bkd
        self._boundary_indices = boundary_indices
        self._normal_operator = normal_operator
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

    def is_essential(self) -> bool:
        """Return False: Robin BCs are natural boundary conditions."""
        return False

    def alpha(self) -> Array:
        """Return coefficient for u term."""
        return self._alpha

    def beta(self) -> Array:
        """Return coefficient for N(u) term."""
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
        alpha * u + beta * N(u) - g

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

        # Compute normal term: N(u) at boundary points
        normal_term = self._normal_operator(state)

        for i in range(self._nboundary_pts):
            u_bndry = state[idx[i]]
            residual[idx[i]] = (
                self._alpha[i] * u_bndry + self._beta[i] * normal_term[i] - g[i]
            )
        return residual

    def apply_to_jacobian(
        self, jacobian: Array, state: Array, time: float
    ) -> Array:
        """Apply Robin BC to Jacobian.

        Sets Jacobian rows at boundary points to:
        alpha * I_row + beta * N_jacobian_row

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

        # Get Jacobian of the normal operator
        normal_jac = self._normal_operator.jacobian(state)

        for i in range(self._nboundary_pts):
            row_idx = idx[i]
            for j in range(nstates):
                jac_val = self._beta[i] * normal_jac[i, j]
                if j == row_idx:
                    jac_val = jac_val + self._alpha[i]
                jacobian[row_idx, j] = jac_val
        return jacobian

    def normal_operator(self) -> NormalOperatorProtocol[Array]:
        """Return the normal operator for this Robin BC."""
        return self._normal_operator

    def apply_to_param_jacobian(
        self, param_jacobian: Array, state: Array, time: float,
        physical_sensitivities=None,
    ) -> Array:
        """Apply Robin BC to parameter Jacobian.

        If physical_sensitivities provides "dflux_n_dp", applies the
        unified BC sensitivity formula for any physics type (diffusion,
        hyperelastic, etc.).

        Parameters
        ----------
        param_jacobian : Array
            Parameter Jacobian. Shape: (nstates, nparams)
        state : Array
            Current solution. Shape: (nstates,)
        time : float
            Current time.
        physical_sensitivities : dict, optional
            Dict with key "dflux_n_dp" of shape (nbnd, nparams) —
            sensitivity of the normal operator's output to parameters.

        Returns
        -------
        Array
            Modified parameter Jacobian. Shape: (nstates, nparams)
        """
        idx = self._boundary_indices
        param_jacobian = self._bkd.copy(param_jacobian)

        dflux_n_dp = None
        if physical_sensitivities is not None:
            dflux_n_dp = physical_sensitivities.get("dflux_n_dp")

        if dflux_n_dp is not None:
            param_jacobian[idx, :] = self._beta[:, None] * dflux_n_dp
        else:
            param_jacobian[idx, :] = 0.0

        return param_jacobian


def gradient_robin_bc(
    bkd: Backend[Array],
    boundary_indices: Array,
    normals: Array,
    derivative_matrices: List[Array],
    alpha: Union[float, Array],
    beta: Union[float, Array],
    values: Union[float, Array, Callable[[float], Array]],
) -> RobinBC[Array]:
    """Create Robin BC using gradient convention: alpha*u + beta*grad(u).n = g.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    boundary_indices : Array
        Indices of mesh points on this boundary. Shape: (nboundary_pts,)
    normals : Array
        Outward unit normals at boundary points. Shape: (nboundary_pts, ndim)
    derivative_matrices : List[Array]
        Physical derivative matrices (d/dx_d), one per dimension.
        Each shape: (npts, npts)
    alpha : float or Array
        Coefficient for u term.
    beta : float or Array
        Coefficient for grad(u).n term.
    values : float, Array, or Callable
        Boundary values g.

    Returns
    -------
    RobinBC
        Robin boundary condition with gradient normal operator.
    """
    normal_op = GradientNormalOperator(
        bkd, boundary_indices, normals, derivative_matrices
    )
    return RobinBC(bkd, boundary_indices, normal_op, alpha, beta, values)


def flux_robin_bc(
    bkd: Backend[Array],
    boundary_indices: Array,
    normals: Array,
    flux_provider: FluxProviderProtocol[Array],
    alpha: Union[float, Array],
    beta: Union[float, Array],
    values: Union[float, Array, Callable[[float], Array]],
) -> RobinBC[Array]:
    """Create Robin BC using flux convention: alpha*u + beta*flux(u).n = g.

    Flux is the total conservative flux (diffusive + advective).

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    boundary_indices : Array
        Indices of mesh points on this boundary. Shape: (nboundary_pts,)
    normals : Array
        Outward unit normals at boundary points. Shape: (nboundary_pts, ndim)
    flux_provider : FluxProviderProtocol
        Physics providing compute_flux() and compute_flux_jacobian().
    alpha : float or Array
        Coefficient for u term.
    beta : float or Array
        Coefficient for flux(u).n term.
    values : float, Array, or Callable
        Boundary values g.

    Returns
    -------
    RobinBC
        Robin boundary condition with flux normal operator.
    """
    normal_op = FluxNormalOperator(
        bkd, boundary_indices, normals, flux_provider
    )
    return RobinBC(bkd, boundary_indices, normal_op, alpha, beta, values)


def gradient_neumann_bc(
    bkd: Backend[Array],
    boundary_indices: Array,
    normals: Array,
    derivative_matrices: List[Array],
    values: Union[float, Array, Callable[[float], Array]] = 0.0,
) -> RobinBC[Array]:
    """Create Neumann BC using gradient convention: grad(u).n = g.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    boundary_indices : Array
        Indices of mesh points on this boundary. Shape: (nboundary_pts,)
    normals : Array
        Outward unit normals. Shape: (nboundary_pts, ndim)
    derivative_matrices : List[Array]
        Physical derivative matrices, one per dimension.
    values : float, Array, or Callable
        Boundary values g. Default: 0.0 (homogeneous).

    Returns
    -------
    RobinBC
        Neumann BC as Robin with alpha=0, beta=1.
    """
    return gradient_robin_bc(
        bkd, boundary_indices, normals, derivative_matrices, 0.0, 1.0, values
    )


def flux_neumann_bc(
    bkd: Backend[Array],
    boundary_indices: Array,
    normals: Array,
    flux_provider: FluxProviderProtocol[Array],
    values: Union[float, Array, Callable[[float], Array]] = 0.0,
) -> RobinBC[Array]:
    """Create Neumann BC using flux convention: flux(u).n = g.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    boundary_indices : Array
        Indices of mesh points on this boundary. Shape: (nboundary_pts,)
    normals : Array
        Outward unit normals. Shape: (nboundary_pts, ndim)
    flux_provider : FluxProviderProtocol
        Physics providing compute_flux() and compute_flux_jacobian().
    values : float, Array, or Callable
        Boundary values g. Default: 0.0 (homogeneous).

    Returns
    -------
    RobinBC
        Neumann BC as Robin with alpha=0, beta=1.
    """
    return flux_robin_bc(
        bkd, boundary_indices, normals, flux_provider, 0.0, 1.0, values
    )


def homogeneous_robin_bc(
    bkd: Backend[Array],
    boundary_indices: Array,
    derivative_matrix: Array,
    normal_sign: float,
    alpha: Union[float, Array],
    beta: Union[float, Array],
) -> RobinBC[Array]:
    """Create a homogeneous Robin BC using legacy (derivative_matrix, normal_sign) API.

    Enforces alpha * u + beta * (normal_sign * D @ u) = 0.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    boundary_indices : Array
        Indices of mesh points on this boundary.
    derivative_matrix : Array
        Boundary-extracted derivative matrix rows. Shape: (nboundary_pts, npts)
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
    normal_op = _LegacyNormalOperator(bkd, derivative_matrix, normal_sign)
    return RobinBC(bkd, boundary_indices, normal_op, alpha, beta, 0.0)


def traction_robin_bc(
    bkd: Backend[Array],
    mesh_boundary_indices: Array,
    normals: Array,
    derivative_matrices: List[Array],
    lamda: float,
    mu: float,
    npts: int,
    component: int,
    alpha: Union[float, Array],
    beta: Union[float, Array],
    values: Union[float, Array, Callable[[float], Array]],
) -> RobinBC[Array]:
    """Create Robin BC for one component of 2D elasticity traction.

    Enforces alpha * u_comp + beta * t_comp = g_comp at boundary,
    where t = σ·n is the traction vector.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    mesh_boundary_indices : Array
        Mesh point indices (0..npts-1) on this boundary. Shape: (nboundary,)
    normals : Array
        Outward unit normals at boundary points. Shape: (nboundary, 2)
    derivative_matrices : List[Array]
        [Dx, Dy], each shape (npts, npts). Physical derivative matrices.
    lamda : float
        Lamé's first parameter λ.
    mu : float
        Shear modulus μ.
    npts : int
        Number of mesh points (state length = 2*npts).
    component : int
        0 for u_x (state indices = mesh_boundary_indices),
        1 for u_y (state indices = mesh_boundary_indices + npts).
    alpha : float or Array
        Coefficient for u_comp term.
    beta : float or Array
        Coefficient for t_comp term.
    values : float, Array, or Callable
        Boundary values g_comp.

    Returns
    -------
    RobinBC
        Robin BC for one displacement component with traction normal operator.
    """
    normal_op = TractionNormalOperator(
        bkd, mesh_boundary_indices, normals, derivative_matrices,
        lamda, mu, component, npts
    )
    # State indices for this component: offset by component * npts
    state_indices = mesh_boundary_indices + component * npts
    return RobinBC(bkd, state_indices, normal_op, alpha, beta, values)


def traction_neumann_bc(
    bkd: Backend[Array],
    mesh_boundary_indices: Array,
    normals: Array,
    derivative_matrices: List[Array],
    lamda: float,
    mu: float,
    npts: int,
    component: int,
    values: Union[float, Array, Callable[[float], Array]] = 0.0,
) -> RobinBC[Array]:
    """Create Neumann BC for one component of 2D elasticity traction.

    Enforces t_comp = g_comp at boundary, where t = σ·n.

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
    lamda : float
        Lamé's first parameter λ.
    mu : float
        Shear modulus μ.
    npts : int
        Number of mesh points.
    component : int
        0 for t_x, 1 for t_y.
    values : float, Array, or Callable
        Traction values g_comp. Default: 0.0 (homogeneous).

    Returns
    -------
    RobinBC
        Traction Neumann BC as Robin with alpha=0, beta=1.
    """
    return traction_robin_bc(
        bkd, mesh_boundary_indices, normals, derivative_matrices,
        lamda, mu, npts, component, 0.0, 1.0, values
    )
