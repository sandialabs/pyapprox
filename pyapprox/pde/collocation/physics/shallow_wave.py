"""Shallow water equations physics for spectral collocation.

Implements the 1D and 2D shallow water equations:

1D:
    dh/dt + d(hu)/dx = 0
    d(hu)/dt + d(hu^2 + 0.5*g*h^2)/dx = -g*h*db/dx + f_u

2D:
    dh/dt + d(hu)/dx + d(hv)/dy = 0
    d(hu)/dt + d(hu^2 + 0.5*g*h^2)/dx + d(huv)/dy = -g*h*db/dx + f_u
    d(hv)/dt + d(huv)/dx + d(hv^2 + 0.5*g*h^2)/dy = -g*h*db/dy + f_v

where:
    h = water depth
    u, v = velocity components
    b = bed elevation (bathymetry)
    g = gravitational acceleration
    f = forcing/source terms
"""

from typing import Callable, Optional, Tuple

from pyapprox.pde.collocation.physics.base import AbstractVectorPhysics
from pyapprox.pde.collocation.protocols.basis import (
    TensorProductBasisProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class ShallowWavePhysics(AbstractVectorPhysics[Array]):
    """Shallow water equations physics.

    Implements the shallow water equations in conservative form.
    Supports both 1D (2 components: h, hu) and 2D (3 components: h, hu, hv).

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        Collocation basis (provides nodes, derivative matrices).
    bkd : Backend
        Computational backend.
    bed : Array
        Bed elevation (bathymetry). Shape: (npts,)
    g : float
        Gravitational acceleration (default: 9.81).
    forcing : Callable[[float], Array], optional
        Forcing term for all components. If callable, takes time and returns
        array of shape (ncomponents * npts,) as [f_h, f_hu] (1D) or
        [f_h, f_hu, f_hv] (2D).
        Forcing term for momentum equations. For 1D, returns shape (npts,).
        For 2D, returns shape (2*npts,) as [f_u, f_v].

    Examples
    --------
    >>> bkd = NumpyBkd()
    >>> mesh = TransformedMesh1D(30, bkd)
    >>> basis = ChebyshevBasis1D(mesh, bkd)
    >>> bed = bkd.zeros((30,))  # Flat bottom
    >>> physics = ShallowWavePhysics(basis, bkd, bed=bed)
    """

    def __init__(
        self,
        basis: TensorProductBasisProtocol[Array],
        bkd: Backend[Array],
        bed: Array,
        g: float = 9.81,
        forcing: Optional[Callable[[float], Array]] = None,
    ):
        ndim = basis.ndim()
        ncomponents = ndim + 1  # h + velocity components
        super().__init__(basis, bkd, ncomponents=ncomponents)

        self._bed = bed
        self._g = g
        self._forcing_func = forcing

        basis.npts()

        # Precompute derivative matrices
        self._D1_matrices = [basis.derivative_matrix(1, dim) for dim in range(ndim)]

        # Precompute bed slope gradient
        self._bed_gradient = [D @ bed for D in self._D1_matrices]

    def g(self) -> float:
        """Return gravitational acceleration."""
        return self._g

    def bed(self) -> Array:
        """Return bed elevation."""
        return self._bed

    def _get_forcing(self, time: float) -> Array:
        """Get forcing array at given time.

        Returns array of shape (ncomponents * npts,) with forcing for
        all components: [f_h, f_hu] (1D) or [f_h, f_hu, f_hv] (2D).
        """
        npts = self.npts()
        bkd = self._bkd
        ncomponents = self.ncomponents()

        if self._forcing_func is None:
            return bkd.zeros((ncomponents * npts,))
        if callable(self._forcing_func):
            return self._forcing_func(time)
        return self._forcing_func

    def _split_state(self, state: Array) -> Tuple[Array, ...]:
        """Split combined state into components (h, hu, [hv]).

        Parameters
        ----------
        state : Array
            Combined state. Shape: (ncomponents * npts,)

        Returns
        -------
        Tuple[Array, ...]
            (h, hu) for 1D or (h, hu, hv) for 2D.
        """
        npts = self.npts()
        ndim = self._basis.ndim()

        h = state[:npts]
        hu = state[npts : 2 * npts]

        if ndim == 1:
            return (h, hu)
        else:
            hv = state[2 * npts : 3 * npts]
            return (h, hu, hv)

    def _combine_state(self, *components: Array) -> Array:
        """Combine components into single state vector."""
        return self._bkd.hstack(list(components))

    def residual(self, state: Array, time: float) -> Array:
        """Compute spatial residual f(u, t).

        For transient problems: du/dt = residual(u, t)

        Parameters
        ----------
        state : Array
            Combined state [h, hu, (hv)]. Shape: (ncomponents * npts,)
        time : float
            Current time.

        Returns
        -------
        Array
            Residual. Shape: (ncomponents * npts,)
        """
        self.npts()
        ndim = self._basis.ndim()

        if ndim == 1:
            h, hu = self._split_state(state)
            return self._residual_1d(h, hu, time)
        else:
            h, hu, hv = self._split_state(state)
            return self._residual_2d(h, hu, hv, time)

    def _residual_1d(self, h: Array, hu: Array, time: float) -> Array:
        """Compute 1D shallow water residual."""
        bkd = self._bkd
        npts = self.npts()
        g = self._g
        D = self._D1_matrices[0]

        # Check for negative depth
        min_h = float(bkd.min(h))
        if min_h <= 0:
            raise RuntimeError(f"Depth became non-positive: min(h) = {min_h}")

        # Compute velocity
        u = hu / h

        # Continuity: dh/dt = -d(hu)/dx + f_h
        res_h = -D @ hu

        # Momentum: d(hu)/dt = -d(hu^2 + 0.5*g*h^2)/dx - g*h*db/dx + f_hu
        flux_u = hu * u + 0.5 * g * h**2
        res_hu = -D @ flux_u - g * h * self._bed_gradient[0]

        # Add forcing to all components
        forcing = self._get_forcing(time)
        res_h = res_h + forcing[:npts]
        res_hu = res_hu + forcing[npts : 2 * npts]

        return self._combine_state(res_h, res_hu)

    def _residual_2d(self, h: Array, hu: Array, hv: Array, time: float) -> Array:
        """Compute 2D shallow water residual."""
        bkd = self._bkd
        npts = self.npts()
        g = self._g
        Dx = self._D1_matrices[0]
        Dy = self._D1_matrices[1]

        # Check for negative depth
        min_h = float(bkd.min(h))
        if min_h <= 0:
            raise RuntimeError(f"Depth became non-positive: min(h) = {min_h}")

        # Compute velocities
        u = hu / h
        v = hv / h

        # Continuity: dh/dt = -d(hu)/dx - d(hv)/dy
        res_h = -Dx @ hu - Dy @ hv

        # x-momentum: d(hu)/dt = -d(hu^2 + 0.5*g*h^2)/dx - d(huv)/dy - g*h*db/dx
        g_hsq = 0.5 * g * h**2
        flux_uu = hu * u + g_hsq
        flux_uv = hu * v
        res_hu = -Dx @ flux_uu - Dy @ flux_uv - g * h * self._bed_gradient[0]

        # y-momentum: d(hv)/dt = -d(huv)/dx - d(hv^2 + 0.5*g*h^2)/dy - g*h*db/dy
        flux_vu = hv * u
        flux_vv = hv * v + g_hsq
        res_hv = -Dx @ flux_vu - Dy @ flux_vv - g * h * self._bed_gradient[1]

        # Add forcing to all components
        forcing = self._get_forcing(time)
        res_h = res_h + forcing[:npts]
        res_hu = res_hu + forcing[npts : 2 * npts]
        res_hv = res_hv + forcing[2 * npts : 3 * npts]

        return self._combine_state(res_h, res_hu, res_hv)

    def jacobian(self, state: Array, time: float) -> Array:
        """Compute state Jacobian df/du.

        Parameters
        ----------
        state : Array
            Combined state [h, hu, (hv)]. Shape: (ncomponents * npts,)
        time : float
            Current time.

        Returns
        -------
        Array
            Jacobian matrix. Shape: (ncomponents * npts, ncomponents * npts)
        """
        ndim = self._basis.ndim()

        if ndim == 1:
            h, hu = self._split_state(state)
            return self._jacobian_1d(h, hu)
        else:
            h, hu, hv = self._split_state(state)
            return self._jacobian_2d(h, hu, hv)

    def _jacobian_1d(self, h: Array, hu: Array) -> Array:
        """Compute 1D Jacobian."""
        bkd = self._bkd
        npts = self.npts()
        g = self._g
        D = self._D1_matrices[0]

        u = hu / h
        u2 = u**2

        # Jacobian is 2x2 block matrix:
        # [[J_hh, J_h_hu], [J_hu_h, J_hu_hu]]

        jacobian = bkd.zeros((2 * npts, 2 * npts))

        # J_hh = d(res_h)/dh = 0
        # res_h = -D @ hu doesn't depend on h

        # J_h_hu = d(res_h)/d(hu) = -D
        jacobian[:npts, npts : 2 * npts] = -D

        # J_hu_h = d(res_hu)/dh
        # res_hu = -D @ (hu*u + 0.5*g*h^2) - g*h*db/dx
        #        = -D @ (hu^2/h + 0.5*g*h^2) - g*h*db/dx
        # d/dh = -D @ (-hu^2/h^2 + g*h) - g*db/dx
        #      = -D @ (-u^2 + g*h) - g*db/dx
        #      = D @ diag(u^2) - g*D @ diag(h) - g*diag(db/dx)
        jacobian[npts : 2 * npts, :npts] = (
            D @ bkd.diag(u2) - g * D @ bkd.diag(h) - g * bkd.diag(self._bed_gradient[0])
        )

        # J_hu_hu = d(res_hu)/d(hu)
        # res_hu = -D @ (hu^2/h + 0.5*g*h^2) - g*h*db/dx
        # d/d(hu) = -D @ (2*hu/h) = -2*D @ diag(u)
        jacobian[npts : 2 * npts, npts : 2 * npts] = -2.0 * D @ bkd.diag(u)

        return jacobian

    def _jacobian_2d(self, h: Array, hu: Array, hv: Array) -> Array:
        """Compute 2D Jacobian."""
        bkd = self._bkd
        npts = self.npts()
        g = self._g
        Dx = self._D1_matrices[0]
        Dy = self._D1_matrices[1]

        u = hu / h
        v = hv / h
        u2 = u**2
        v2 = v**2
        uv = u * v

        # Jacobian is 3x3 block matrix
        jacobian = bkd.zeros((3 * npts, 3 * npts))

        # Row 1: d(res_h)/d(h, hu, hv)
        # res_h = -Dx @ hu - Dy @ hv
        jacobian[:npts, npts : 2 * npts] = -Dx  # d/d(hu)
        jacobian[:npts, 2 * npts : 3 * npts] = -Dy  # d/d(hv)

        # Row 2: d(res_hu)/d(h, hu, hv)
        # res_hu = -Dx @ (hu*u + 0.5*g*h^2) - Dy @ (hu*v) - g*h*db/dx
        #        = -Dx @ (hu^2/h + 0.5*g*h^2) - Dy @ (hu*hv/h) - g*h*db/dx

        # d/dh: -Dx @ (-u^2 + g*h) - Dy @ (-uv) - g*db/dx
        jacobian[npts : 2 * npts, :npts] = (
            Dx @ bkd.diag(u2)
            - g * Dx @ bkd.diag(h)
            + Dy @ bkd.diag(uv)
            - g * bkd.diag(self._bed_gradient[0])
        )

        # d/d(hu): -Dx @ (2*u) - Dy @ (v) = -2*Dx @ diag(u) - Dy @ diag(v)
        jacobian[npts : 2 * npts, npts : 2 * npts] = -2.0 * Dx @ bkd.diag(
            u
        ) - Dy @ bkd.diag(v)

        # d/d(hv): -Dy @ (u) = -Dy @ diag(u/h) @ diag(h) ... wait
        # hu*v = hu * hv/h, so d/d(hv) = hu/h = u
        jacobian[npts : 2 * npts, 2 * npts : 3 * npts] = -Dy @ bkd.diag(u)

        # Row 3: d(res_hv)/d(h, hu, hv)
        # res_hv = -Dx @ (hv*u) - Dy @ (hv*v + 0.5*g*h^2) - g*h*db/dy
        #        = -Dx @ (hv*hu/h) - Dy @ (hv^2/h + 0.5*g*h^2) - g*h*db/dy

        # d/dh: -Dx @ (-uv) - Dy @ (-v^2 + g*h) - g*db/dy
        jacobian[2 * npts : 3 * npts, :npts] = (
            Dx @ bkd.diag(uv)
            + Dy @ bkd.diag(v2)
            - g * Dy @ bkd.diag(h)
            - g * bkd.diag(self._bed_gradient[1])
        )

        # d/d(hu): -Dx @ (v) = -Dx @ diag(hv/h) / ... = -Dx @ diag(v)
        jacobian[2 * npts : 3 * npts, npts : 2 * npts] = -Dx @ bkd.diag(v)

        # d/d(hv): -Dx @ (u) - Dy @ (2*v) = -Dx @ diag(u) - 2*Dy @ diag(v)
        jacobian[2 * npts : 3 * npts, 2 * npts : 3 * npts] = -Dx @ bkd.diag(
            u
        ) - 2.0 * Dy @ bkd.diag(v)

        return jacobian

    def compute_interface_flux(
        self, state: Array, boundary_indices: Array, normal: Array
    ) -> Array:
        """Compute flux at boundary for DtN domain decomposition.

        For shallow water equations, the interface flux includes:
        - Mass flux: h*u·n (continuity)
        - Momentum flux: (hu*u + 0.5*g*h²)*n_x + hu*v*n_y (momentum)

        For 1D: flux = [hu*n, (hu*u + 0.5*g*h²)*n]
        For 2D: flux = [hu*n_x + hv*n_y,
                        (hu²/h + 0.5*g*h²)*n_x + (hu*hv/h)*n_y,
                        (hu*hv/h)*n_x + (hv²/h + 0.5*g*h²)*n_y]

        Parameters
        ----------
        state : Array
            Combined state [h, hu, (hv)]. Shape: (ncomponents * npts,)
        boundary_indices : Array
            Mesh indices at interface. Shape: (nboundary,)
        normal : Array
            Outward unit normal. Shape: (ndim,)

        Returns
        -------
        Array
            Flux at boundary. Shape: (ncomponents * nboundary,)
            Component-stacked: [flux_h, flux_hu, (flux_hv)]
        """
        bkd = self._bkd
        ndim = self._basis.ndim()
        boundary_indices.shape[0]
        g = self._g

        if ndim == 1:
            h, hu = self._split_state(state)
            h_b = h[boundary_indices]
            hu_b = hu[boundary_indices]
            u_b = hu_b / h_b

            n = float(normal[0])

            # Mass flux: hu * n
            flux_h = hu_b * n

            # Momentum flux: (hu*u + 0.5*g*h²) * n
            flux_hu = (hu_b * u_b + 0.5 * g * h_b**2) * n

            return bkd.concatenate([flux_h, flux_hu])

        else:  # 2D
            h, hu, hv = self._split_state(state)
            h_b = h[boundary_indices]
            hu_b = hu[boundary_indices]
            hv_b = hv[boundary_indices]
            u_b = hu_b / h_b
            v_b = hv_b / h_b

            nx = float(normal[0])
            ny = float(normal[1])

            g_hsq = 0.5 * g * h_b**2

            # Mass flux: hu*n_x + hv*n_y
            flux_h = hu_b * nx + hv_b * ny

            # x-momentum flux: (hu*u + 0.5*g*h²)*n_x + hu*v*n_y
            flux_hu = (hu_b * u_b + g_hsq) * nx + (hu_b * v_b) * ny

            # y-momentum flux: hv*u*n_x + (hv*v + 0.5*g*h²)*n_y
            flux_hv = (hv_b * u_b) * nx + (hv_b * v_b + g_hsq) * ny

            return bkd.concatenate([flux_h, flux_hu, flux_hv])


def create_shallow_wave(
    basis: TensorProductBasisProtocol[Array],
    bkd: Backend[Array],
    bed: Array,
    g: float = 9.81,
    forcing: Optional[Callable[[float], Array]] = None,
) -> ShallowWavePhysics[Array]:
    """Create shallow water equations physics.

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        Collocation basis.
    bkd : Backend
        Computational backend.
    bed : Array
        Bed elevation (bathymetry).
    g : float
        Gravitational acceleration (default: 9.81).
    forcing : Callable, optional
        Forcing for all components [f_h, f_hu] (1D) or [f_h, f_hu, f_hv] (2D).

    Returns
    -------
    ShallowWavePhysics
        Shallow water physics.
    """
    return ShallowWavePhysics(
        basis=basis,
        bkd=bkd,
        bed=bed,
        g=g,
        forcing=forcing,
    )
