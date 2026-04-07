"""Shallow Ice Approximation physics for spectral collocation.

Implements the Shallow Ice Approximation (SIA) equation:
    dH/dt = div(D * grad(s)) + f

where:
    H = ice thickness (solution variable)
    s = H + b = surface elevation
    b = bed elevation
    D = nonlinear diffusion coefficient:
        D = gamma * H^(n+2) * |grad(s)|^(n-1) + (rho*g/C) * H^2

    gamma = 2*A*(rho*g)^n / (n+2)
    n = Glen's flow law exponent (typically 3)
    A = rate factor
    rho = ice density
    g = gravitational acceleration
    C = friction coefficient
"""

from typing import Any, Callable, Optional, Union

from pyapprox.pde.collocation.physics.base import AbstractScalarPhysics
from pyapprox.pde.collocation.protocols.basis import (
    TensorProductBasisProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class ShallowIcePhysics(AbstractScalarPhysics[Array]):
    """Shallow Ice Approximation physics.

    Implements the SIA equation for ice thickness evolution:
        dH/dt = div(D * grad(s)) + f

    where s = H + b is the surface elevation and D is a nonlinear
    diffusion coefficient.

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        Collocation basis (provides nodes, derivative matrices).
    bkd : Backend
        Computational backend.
    bed : Array
        Bed elevation b. Shape: (npts,)
    friction : float or Array
        Friction coefficient C. If Array, shape: (npts,)
    A : float
        Rate factor in Glen's flow law (typical: 1e-16 to 1e-17 Pa^-3 s^-1)
    rho : float
        Ice density (kg/m^3). Typical: 917.0
    forcing : Callable[[float], Array] or Array, optional
        Forcing/source term f (e.g., accumulation - ablation).
    eps : float
        Small regularization parameter to avoid division by zero
        in gradient norm. Default: 1e-12

    Examples
    --------
    >>> bkd = NumpyBkd()
    >>> mesh = TransformedMesh1D(30, bkd)
    >>> basis = ChebyshevBasis1D(mesh, bkd)
    >>> bed = 0.1 * basis.nodes()  # Sloped bed
    >>> physics = ShallowIcePhysics(
    ...     basis, bkd, bed=bed, friction=1e6, A=1e-16, rho=917.0
    ... )
    """

    def __init__(
        self,
        basis: TensorProductBasisProtocol[Array],
        bkd: Backend[Array],
        bed: Array,
        friction: Union[float, Array],
        A: float,
        rho: float,
        forcing: Optional[Callable[[float], Array]] = None,
        eps: float = 1e-12,
    ):
        super().__init__(basis, bkd)

        self._bed = bed
        self._A = A
        self._rho = rho
        self._g = 9.81  # Gravitational acceleration
        self._n = 3  # Glen's flow law exponent
        self._eps = eps

        npts = basis.npts()
        ndim = basis.ndim()

        # Compute gamma = 2*A*(rho*g)^n / (n+2)
        self._gamma = 2 * self._A * (self._rho * self._g) ** self._n / (self._n + 2)

        # Store friction coefficient
        if isinstance(friction, (int, float)):
            self._friction_array = bkd.full((npts,), float(friction))
        else:
            self._friction_array = friction

        # Friction fraction: rho*g/C
        self._friction_frac = self._rho * self._g / self._friction_array

        self._forcing_func = forcing

        # Precompute derivative matrices
        self._D_matrices = [basis.derivative_matrix(1, dim) for dim in range(ndim)]

    def _get_forcing(self, time: float) -> Array:
        """Get forcing array at given time."""
        npts = self.npts()
        if self._forcing_func is None:
            return self._bkd.zeros((npts,))
        if callable(self._forcing_func):
            return self._forcing_func(time)
        return self._forcing_func

    def _compute_surface_gradient(self, state: Array) -> tuple[Any, ...]:
        """Compute surface gradient components.

        Parameters
        ----------
        state : Array
            Ice thickness H. Shape: (npts,)

        Returns
        -------
        tuple
            (grad_s_components, grad_s_sq) where grad_s_components is list
            of gradient components and grad_s_sq is squared magnitude.
        """
        ndim = self._basis.ndim()

        # Surface elevation s = H + bed
        surface = state + self._bed

        # Surface gradient components
        grad_s = [self._D_matrices[dim] @ surface for dim in range(ndim)]

        # |grad(s)|^2
        grad_s_sq = sum(gs**2 for gs in grad_s)

        return grad_s, grad_s_sq

    def _compute_diffusion(self, state: Array, grad_s_sq: Array) -> Array:
        """Compute nonlinear diffusion coefficient.

        D = gamma * H^(n+2) * |grad(s)|^(n-1) + friction_frac * H^2

        Parameters
        ----------
        state : Array
            Ice thickness H. Shape: (npts,)
        grad_s_sq : Array
            Squared magnitude of surface gradient. Shape: (npts,)

        Returns
        -------
        Array
            Diffusion coefficient D. Shape: (npts,)
        """
        n = self._n

        # Deformation component: gamma * H^(n+2) * |grad(s)|^(n-1)
        # For n=3: gamma * H^5 * |grad(s)|^2
        # |grad(s)|^(n-1) = (|grad(s)|^2)^((n-1)/2) = grad_s_sq for n=3
        H_power = state ** (n + 2)  # H^5 for n=3
        grad_power = (grad_s_sq + self._eps) ** ((n - 1) / 2)  # |grad(s)|^2 for n=3

        deformation = self._gamma * H_power * grad_power

        # Sliding component: friction_frac * H^2
        sliding = self._friction_frac * state**2

        return deformation + sliding

    def residual(self, state: Array, time: float) -> Array:
        """Compute spatial residual f(u, t).

        For transient problems: dH/dt = residual(H, t)

        Parameters
        ----------
        state : Array
            Ice thickness H. Shape: (npts,)
        time : float
            Current time.

        Returns
        -------
        Array
            Residual div(D*grad(s)) + f. Shape: (npts,)
        """
        ndim = self._basis.ndim()

        # Compute surface gradient and its squared magnitude
        grad_s, grad_s_sq = self._compute_surface_gradient(state)

        # Compute nonlinear diffusion
        D = self._compute_diffusion(state, grad_s_sq)

        # Flux = D * grad(s)
        flux = [D * gs for gs in grad_s]

        # Divergence of flux: div(D*grad(s)) = sum_i d/dx_i(D * ds/dx_i)
        div_flux = sum(self._D_matrices[dim] @ flux[dim] for dim in range(ndim))

        # Residual = div(D*grad(s)) + f
        residual = div_flux + self._get_forcing(time)

        return residual

    def jacobian(self, state: Array, time: float) -> Array:
        """Compute state Jacobian df/dH.

        This is computed via automatic differentiation-style chain rule
        or finite differences if exact derivatives are too complex.

        Parameters
        ----------
        state : Array
            Ice thickness H. Shape: (npts,)
        time : float
            Current time.

        Returns
        -------
        Array
            Jacobian matrix. Shape: (npts, npts)
        """
        bkd = self._bkd
        npts = self.npts()
        self._basis.ndim()

        # Compute surface gradient
        grad_s, grad_s_sq = self._compute_surface_gradient(state)

        # Compute diffusion and its derivatives
        self._compute_diffusion(state, grad_s_sq)

        # Initialize Jacobian
        jacobian = bkd.zeros((npts, npts))

        # The residual is: div(D*grad(s)) + f
        # where D depends on H and |grad(s)|, and s = H + bed

        # grad(s) = grad(H) + grad(bed), but grad(bed) is constant
        # d[grad(s)]/dH = D1 (each component)

        # dD/dH has two parts:
        # 1. Through H directly: d/dH[gamma*H^(n+2)*...] = (n+2)*gamma*H^(n+1)*...
        # 2. Through |grad(s)|: dD/d(|grad(s)|^2) * d(|grad(s)|^2)/dH

        # For simplicity, we compute Jacobian numerically via finite differences
        # This avoids complex chain rule for the highly nonlinear diffusion

        eps = 1e-7
        for j in range(npts):
            state_plus = bkd.copy(state)
            state_plus[j] = state_plus[j] + eps
            state_minus = bkd.copy(state)
            state_minus[j] = state_minus[j] - eps

            res_plus = self.residual(state_plus, time)
            res_minus = self.residual(state_minus, time)

            for i in range(npts):
                jacobian[i, j] = (res_plus[i] - res_minus[i]) / (2 * eps)

        return jacobian


def create_shallow_ice(
    basis: TensorProductBasisProtocol[Array],
    bkd: Backend[Array],
    bed: Array,
    friction: Union[float, Array],
    A: float = 1e-16,
    rho: float = 917.0,
    forcing: Optional[Callable[[float], Array]] = None,
    eps: float = 1e-12,
) -> ShallowIcePhysics[Array]:
    """Create Shallow Ice Approximation physics.

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        Collocation basis.
    bkd : Backend
        Computational backend.
    bed : Array
        Bed elevation.
    friction : float or Array
        Friction coefficient C.
    A : float
        Rate factor (default: 1e-16).
    rho : float
        Ice density (default: 917.0).
    forcing : Callable or Array, optional
        Source term.
    eps : float
        Regularization parameter.

    Returns
    -------
    ShallowIcePhysics
        SIA physics.
    """
    return ShallowIcePhysics(
        basis=basis,
        bkd=bkd,
        bed=bed,
        friction=friction,
        A=A,
        rho=rho,
        forcing=forcing,
        eps=eps,
    )
