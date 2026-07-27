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

from typing import Callable, Optional, Union

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
    glen_exponent : float
        Glen's flow law exponent n. Default: 3.0

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
        glen_exponent: float = 3.0,
    ):
        super().__init__(basis, bkd)

        self._bed = bed
        self._A = A
        self._rho = rho
        self._g = 9.81  # Gravitational acceleration
        self._n = float(glen_exponent)  # Glen's flow law exponent
        self._eps = eps

        npts = basis.npts()
        ndim = basis.ndim()

        # Compute gamma = 2*A*(rho*g)^n / (n+2). The annotation is
        # required: float ** float is typed Any in typeshed.
        self._gamma: float = (
            2 * self._A * (self._rho * self._g) ** self._n / (self._n + 2)
        )

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

    def _compute_surface_gradient(
        self, state: Array
    ) -> tuple[list[Array], Array]:
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
        grad_s_sq = grad_s[0] ** 2
        for gs in grad_s[1:]:
            grad_s_sq = grad_s_sq + gs**2

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

    def _compute_diffusion_derivatives(
        self, state: Array, grad_s_sq: Array
    ) -> tuple[Array, Array]:
        """Compute partial derivatives of the diffusion coefficient.

        With :math:`G = |\\nabla s|^2 + \\epsilon` (the same regularized
        quantity used by ``_compute_diffusion``) and

        .. math:: D = \\gamma H^{n+2} G^{(n-1)/2} + \\phi H^2,

        the partial derivatives are

        .. math::

            \\kappa_H = \\partial D/\\partial H
                = \\gamma (n+2) H^{n+1} G^{(n-1)/2} + 2 \\phi H,

            \\kappa_G = \\partial D/\\partial G
                = \\gamma \\tfrac{n-1}{2} H^{n+2} G^{(n-3)/2}.

        For n=3 these collapse to
        :math:`\\kappa_H = 5 \\gamma H^4 G + 2 \\phi H` and
        :math:`\\kappa_G = \\gamma H^5` (no eps dependence).

        Parameters
        ----------
        state : Array
            Ice thickness H. Shape: (npts,)
        grad_s_sq : Array
            Squared magnitude of surface gradient. Shape: (npts,)

        Returns
        -------
        tuple
            (kappa_h, kappa_g), each shape (npts,).
        """
        n = self._n
        reg = grad_s_sq + self._eps
        kappa_h = (
            self._gamma * (n + 2) * state ** (n + 1) * reg ** ((n - 1) / 2)
            + 2.0 * self._friction_frac * state
        )
        kappa_g = (
            self._gamma * ((n - 1) / 2) * state ** (n + 2) * reg ** ((n - 3) / 2)
        )
        return kappa_h, kappa_g

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
        """Compute exact state Jacobian dR/dH of the SIA residual.

        With :math:`R = \\sum_d D_d (\\kappa \\odot g_d) + f`,
        :math:`g_d = D_d (H + b)` and :math:`G = |\\nabla s|^2 + \\epsilon`,
        the chain rule gives, per dimension d:

        .. math::

            D_d \\left[ \\mathrm{diag}(\\kappa_H g_d)
            + \\mathrm{diag}(\\kappa) D_d
            + \\mathrm{diag}(2 \\kappa_G g_d)
              \\sum_e \\mathrm{diag}(g_e) D_e \\right]

        where :math:`\\kappa_H, \\kappa_G` come from
        ``_compute_diffusion_derivatives``. The forcing is
        state-independent. The result is nonsymmetric.

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
        ndim = self._basis.ndim()

        grad_s, grad_s_sq = self._compute_surface_gradient(state)
        kappa = self._compute_diffusion(state, grad_s_sq)
        kappa_h, kappa_g = self._compute_diffusion_derivatives(state, grad_s_sq)

        # grad_dot_mat @ delta = grad(s) . grad(delta); shared across d
        grad_dot_mat = bkd.zeros((npts, npts))
        for dim in range(ndim):
            grad_dot_mat = (
                grad_dot_mat + bkd.diag(grad_s[dim]) @ self._D_matrices[dim]
            )

        jacobian = bkd.zeros((npts, npts))
        for dim in range(ndim):
            inner = (
                bkd.diag(kappa_h * grad_s[dim])
                + bkd.diag(kappa) @ self._D_matrices[dim]
                + bkd.diag(2.0 * kappa_g * grad_s[dim]) @ grad_dot_mat
            )
            jacobian = jacobian + self._D_matrices[dim] @ inner

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
    glen_exponent: float = 3.0,
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
    glen_exponent : float
        Glen's flow law exponent n (default: 3.0).

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
        glen_exponent=glen_exponent,
    )
