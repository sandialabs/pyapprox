r"""Divergence-free velocity from a random stream function.

On the unit square, the velocity is the perpendicular gradient of a mean
stream function plus a random one,

.. math::

    \beta(\eta) = \nabla^\perp \psi_{\mathrm{mean}}
    + \nabla^\perp \psi(\eta), \qquad
    \psi(\eta) = \sum_{i,j=1}^{n_\psi} \sqrt{\lambda_{ij}}\, \eta_{ij}
    \sin(i\pi x_1) \sin(j\pi x_2),

with :math:`\nabla^\perp = (\partial_{x_2}, -\partial_{x_1})`. Taking the
mean as a stream function too makes every velocity divergence-free by
construction, rather than by a promise about an injected velocity.
:math:`\psi(\eta)` vanishes on the boundary, so the perturbation has no
normal component there; the mean may have one (e.g. an inflow).
"""

import math
from typing import Generic, Protocol, runtime_checkable

from pyapprox.pde.field_maps.protocol import validate_params_1d
from pyapprox.pde.field_maps.vector_layout import VectorFieldLayoutProtocol
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class StreamFunctionProtocol(Protocol, Generic[Array]):
    r"""A stream function :math:`\psi`, given by its gradient."""

    def gradient(self, points: Array) -> Array:
        r"""Return :math:`(\partial_{x_1}\psi, \partial_{x_2}\psi)` at
        ``points``.

        Parameters
        ----------
        points : Array
            Shape: ``(2, npts)``.

        Returns
        -------
        Array
            Shape: ``(2, npts)``.
        """
        ...


class StreamFunctionVelocityMap(Generic[Array]):
    r"""Linear field map from stream-function coefficients to nodal velocity.

    The output is the velocity at ``points`` in the injected layout's DOF
    order. For a Galerkin ``VectorLagrangeBasis`` (and
    ``NodalFieldVelocity``) pass the basis's node coordinates,
    ``vector_basis.dof_coordinates()[:, 0::2]``, and its layout,
    ``vector_basis.component_layout()``.

    The derivatives of each mode are evaluated analytically at the points.
    The map is linear in :math:`\eta`, so its Jacobian is a constant matrix
    and its Hessian is zero.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    points : Array
        Points in the unit square. Shape: ``(2, npts)``.
    spectrum : Array
        Mode variances :math:`\lambda_{ij} \ge 0`, the caller's choice.
        Shape: ``(n_psi, n_psi)``; ``spectrum[i - 1, j - 1]`` scales the
        mode :math:`\sin(i\pi x_1)\sin(j\pi x_2)`.
    mean_stream_function : StreamFunctionProtocol
        :math:`\psi_{\mathrm{mean}}`; the mean velocity is its
        perpendicular gradient. A uniform flow :math:`(U, 0)` is
        :math:`\psi = U x_2`.
    layout : VectorFieldLayoutProtocol
        How the two velocity components are ordered in the output.

    Notes
    -----
    The coefficients are standardized and ordered with ``i`` outer and
    ``j`` inner: :math:`\eta_{ij}` is entry ``(i - 1) * n_psi + (j - 1)``.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        points: Array,
        spectrum: Array,
        mean_stream_function: StreamFunctionProtocol[Array],
        layout: VectorFieldLayoutProtocol[Array],
    ) -> None:
        if not isinstance(mean_stream_function, StreamFunctionProtocol):
            raise TypeError(
                "mean_stream_function must satisfy StreamFunctionProtocol, "
                f"got {type(mean_stream_function).__name__}"
            )
        if not isinstance(layout, VectorFieldLayoutProtocol):
            raise TypeError(
                "layout must satisfy VectorFieldLayoutProtocol, got "
                f"{type(layout).__name__}"
            )
        if points.ndim != 2 or points.shape[0] != 2:
            raise ValueError(
                f"points must have shape (2, npts), got {tuple(points.shape)}"
            )
        if (
            spectrum.ndim != 2
            or spectrum.shape[0] != spectrum.shape[1]
            or spectrum.shape[0] < 1
        ):
            raise ValueError(
                "spectrum must be square, shape (n_psi, n_psi), got "
                f"{tuple(spectrum.shape)}"
            )
        if bool(bkd.any_bool(spectrum < 0.0)):
            raise ValueError("spectrum entries are variances; must be >= 0")
        self._bkd = bkd
        npts = int(points.shape[1])
        gradient = mean_stream_function.gradient(points)
        if tuple(gradient.shape) != (2, npts):
            raise ValueError(
                "mean_stream_function.gradient must return shape "
                f"(2, {npts}), got {tuple(gradient.shape)}"
            )
        # The perpendicular gradient (d psi/dx2, -d psi/dx1).
        self._mean = layout.flatten(
            self._bkd.stack([gradient[1], -gradient[0]], axis=0)
        )
        self._jacobian = layout.flatten(self._mode_components(points, spectrum))

    def _mode_components(self, points: Array, spectrum: Array) -> Array:
        """Each mode's velocity components, shape ``(2, npts, n_psi^2)``."""
        bkd = self._bkd
        npsi = int(spectrum.shape[0])
        wavenumbers = bkd.arange(1.0, npsi + 1.0) * math.pi
        kx = wavenumbers[:, None] * points[0][None, :]
        ky = wavenumbers[:, None] * points[1][None, :]
        sin_x, cos_x = bkd.sin(kx), bkd.cos(kx)
        sin_y, cos_y = bkd.sin(ky), bkd.cos(ky)
        amplitude = bkd.sqrt(spectrum)
        # Mode (i, j): sqrt(lam_ij) (j pi sin(i pi x) cos(j pi y),
        #                           -i pi cos(i pi x) sin(j pi y)).
        u = (
            amplitude[:, :, None]
            * wavenumbers[None, :, None]
            * sin_x[:, None, :]
            * cos_y[None, :, :]
        )
        v = -(
            amplitude[:, :, None]
            * wavenumbers[:, None, None]
            * cos_x[:, None, :]
            * sin_y[None, :, :]
        )
        npts = int(points.shape[1])
        # (n_psi, n_psi, npts) -> (npts, n_psi^2), modes i-outer, j-inner.
        u = bkd.transpose(bkd.reshape(u, (npsi * npsi, npts)))
        v = bkd.transpose(bkd.reshape(v, (npsi * npsi, npts)))
        return bkd.stack([u, v], axis=0)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return int(self._jacobian.shape[1])

    def __call__(self, params_1d: Array) -> Array:
        """Return the interleaved nodal velocity. Shape: ``(2 npts,)``."""
        validate_params_1d(params_1d, self.nvars())
        return self._mean + self._bkd.dot(self._jacobian, params_1d)

    def jacobian(self, params_1d: Array) -> Array:
        """Return the constant Jacobian. Shape: ``(2 npts, n_psi^2)``."""
        return self._jacobian

    def hvp(self, params_1d: Array, adj_state: Array, vvec: Array) -> Array:
        """Adjoint-weighted HVP: exactly zero (the map is linear).

        Shape: ``(nvars,)``.
        """
        return self._bkd.zeros((self.nvars(),))

    def is_linear(self) -> bool:
        """Linear in the coefficients."""
        return True

    def __repr__(self) -> str:
        return (
            f"StreamFunctionVelocityMap(npts={self._jacobian.shape[0] // 2}, "
            f"nvars={self.nvars()})"
        )
