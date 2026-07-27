"""Zone weights: pluggable spatial weighting for integrated QoIs.

A zone weight defines the ``W`` of quadratic spatial functionals
``u^T W u`` (e.g. time-integrated zone contamination): a weighting
function ``w(x)`` assembled against the finite-element basis as
``W_ij = int w(x) phi_i phi_j dx``. Implementations differ in how
``w`` interacts with the mesh:

- :class:`ElementAlignedRectangleZone`: a SHARP indicator whose edges
  must coincide with element edges. No element is cut, so element-wise
  assembly is EXACT (``W`` is the sub-mesh mass matrix) at every
  nested-refinement level; misalignment raises instead of silently
  degrading quadrature.
- :class:`SmoothDiscZone`: a tanh-mollified disc indicator assembled
  by quadrature; the transition width should resolve the mesh.

Swapping implementations changes the QoI's spatial weighting without
touching the functional, which consumes only the assembled ``W``.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Protocol,
    Tuple,
    runtime_checkable,
)

if TYPE_CHECKING:
    from skfem.assembly.form.form import FormExtraParams
    from skfem.element.discrete_field import DiscreteField

    from pyapprox.pde.galerkin.basis.lagrange import LagrangeBasis

import numpy as np
from numpy.typing import NDArray

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class ZoneWeightProtocol(Protocol, Generic[Array]):
    """A spatial weighting w(x) assembled against a scalar basis."""

    def assemble_weighted_mass(
        self, basis: "LagrangeBasis[Array]", bkd: Backend[Array]
    ) -> Array:
        """Assemble ``W_ij = int w(x) phi_i phi_j dx``.

        Parameters
        ----------
        basis : LagrangeBasis
            Scalar basis of the state the weight applies to.
        bkd : Backend
            Backend for the returned dense array.

        Returns
        -------
        Array
            Symmetric weighted mass matrix. Shape: ``(ndofs, ndofs)``.
        """
        ...

    def outline_vertices(self) -> np.ndarray:
        """Return a closed outline polyline for plotting.

        Returns
        -------
        np.ndarray
            Vertex coordinates. Shape: ``(2, nvertices)``.
        """
        ...


class ElementAlignedRectangleZone(Generic[Array]):
    """Sharp rectangle indicator whose edges lie on element edges.

    Because no element is cut, ``w`` is constant on every element and
    the assembled ``W`` is exactly the mass matrix of the sub-mesh
    covering the rectangle — at every nested-refinement level. An
    element straddling the rectangle boundary raises at assembly:
    silent cut-element quadrature error would make ``W`` (and every
    quantity downstream of it) mesh-dependent.

    Parameters
    ----------
    xlim : tuple of 2 floats
        Rectangle x-extent ``(xmin, xmax)``.
    ylim : tuple of 2 floats
        Rectangle y-extent ``(ymin, ymax)``.
    """

    def __init__(
        self, xlim: Tuple[float, float], ylim: Tuple[float, float]
    ) -> None:
        if not (xlim[0] < xlim[1] and ylim[0] < ylim[1]):
            raise ValueError(
                f"rectangle must have positive extent, got xlim={xlim}, "
                f"ylim={ylim}"
            )
        self._xlim = (float(xlim[0]), float(xlim[1]))
        self._ylim = (float(ylim[0]), float(ylim[1]))

    def xlim(self) -> Tuple[float, float]:
        """Return the x-extent."""
        return self._xlim

    def ylim(self) -> Tuple[float, float]:
        """Return the y-extent."""
        return self._ylim

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Return a boolean mask of points inside the rectangle.

        Parameters
        ----------
        points : np.ndarray
            Coordinates. Shape: ``(2, npoints)``.
        """
        return np.asarray(
            (points[0] >= self._xlim[0])
            & (points[0] <= self._xlim[1])
            & (points[1] >= self._ylim[0])
            & (points[1] <= self._ylim[1])
        )

    def _member_elements(
        self, mesh_p: np.ndarray, mesh_t: np.ndarray
    ) -> np.ndarray:
        """Classify elements, raising on any cut element."""
        tol = 1e-12
        xin = (mesh_p[0] >= self._xlim[0] - tol) & (
            mesh_p[0] <= self._xlim[1] + tol
        )
        yin = (mesh_p[1] >= self._ylim[0] - tol) & (
            mesh_p[1] <= self._ylim[1] + tol
        )
        vert_in = xin & yin
        xstrict = (mesh_p[0] > self._xlim[0] + tol) & (
            mesh_p[0] < self._xlim[1] - tol
        )
        ystrict = (mesh_p[1] > self._ylim[0] + tol) & (
            mesh_p[1] < self._ylim[1] - tol
        )
        vert_strict_in = xstrict & ystrict

        elem_vert_in = vert_in[mesh_t]
        elem_vert_strict_in = vert_strict_in[mesh_t]
        cut = elem_vert_strict_in.any(axis=0) & (~elem_vert_in).any(axis=0)
        if cut.any():
            raise ValueError(
                f"{int(cut.sum())} elements straddle the rectangle "
                f"boundary x={self._xlim}, y={self._ylim}: its edges "
                "must lie on mesh grid lines for exact element-aligned "
                "assembly. Align the rectangle with the mesh or use a "
                "quadrature-assembled zone (SmoothDiscZone)."
            )
        centroids = mesh_p[:, mesh_t].mean(axis=1)
        member = self.contains(centroids)
        return np.where(member)[0]

    def assemble_weighted_mass(
        self, basis: "LagrangeBasis[Array]", bkd: Backend[Array]
    ) -> Array:
        """Assemble the exact sub-mesh mass matrix of the rectangle."""
        from skfem import Basis, BilinearForm, asm

        skfem_basis = basis.skfem_basis()
        mesh = skfem_basis.mesh
        elements = self._member_elements(
            np.asarray(mesh.p), np.asarray(mesh.t)
        )
        if elements.shape[0] == 0:
            raise ValueError(
                f"no elements lie inside the rectangle x={self._xlim}, "
                f"y={self._ylim}"
            )
        sub_basis = Basis(mesh, skfem_basis.elem, elements=elements)

        def mass_form(
            u: "DiscreteField",
            v: "DiscreteField",
            w: "FormExtraParams",
        ) -> np.ndarray:
            ret: NDArray[np.floating[Any]] = u * v
            return ret

        return bkd.asarray(
            asm(BilinearForm(mass_form), sub_basis).toarray()
        )

    def outline_vertices(self) -> np.ndarray:
        """Return the closed rectangle outline. Shape: ``(2, 5)``."""
        xmin, xmax = self._xlim
        ymin, ymax = self._ylim
        return np.array(
            [
                [xmin, xmax, xmax, xmin, xmin],
                [ymin, ymin, ymax, ymax, ymin],
            ]
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"xlim={self._xlim}, ylim={self._ylim})"
        )


class SmoothDiscZone(Generic[Array]):
    """Tanh-mollified disc indicator assembled by quadrature.

    ``w(x) = (1 - tanh((|x - center| - radius)/transition_width))/2``:
    approximately one inside the disc, zero outside, with a transition
    layer of width ``transition_width`` — choose it comparable to the
    mesh size so quadrature resolves it.

    Parameters
    ----------
    center : tuple of 2 floats
        Disc center.
    radius : float
        Disc radius.
    transition_width : float
        Width of the tanh transition layer.
    """

    def __init__(
        self,
        center: Tuple[float, float],
        radius: float,
        transition_width: float,
    ) -> None:
        if radius <= 0 or transition_width <= 0:
            raise ValueError(
                "radius and transition_width must be positive, got "
                f"{radius} and {transition_width}"
            )
        self._center = (float(center[0]), float(center[1]))
        self._radius = float(radius)
        self._transition_width = float(transition_width)

    def center(self) -> Tuple[float, float]:
        """Return the disc center."""
        return self._center

    def radius(self) -> float:
        """Return the disc radius."""
        return self._radius

    def weight_values(self, points: np.ndarray) -> np.ndarray:
        """Evaluate w at points of shape ``(2, ...)``."""
        distance = np.sqrt(
            (points[0] - self._center[0]) ** 2
            + (points[1] - self._center[1]) ** 2
        )
        return np.asarray(
            0.5
            * (
                1.0
                - np.tanh(
                    (distance - self._radius) / self._transition_width
                )
            )
        )

    def assemble_weighted_mass(
        self, basis: "LagrangeBasis[Array]", bkd: Backend[Array]
    ) -> Array:
        """Assemble ``W`` with w evaluated at quadrature points."""
        from skfem import BilinearForm, asm

        weight_values = self.weight_values

        def weighted_mass_form(
            u: "DiscreteField",
            v: "DiscreteField",
            w: "FormExtraParams",
        ) -> np.ndarray:
            ret: NDArray[np.floating[Any]] = (
                weight_values(np.asarray(w.x)) * u * v
            )
            return ret

        return bkd.asarray(
            asm(BilinearForm(weighted_mass_form), basis.skfem_basis())
            .toarray()
        )

    def outline_vertices(self) -> np.ndarray:
        """Return a closed circle outline. Shape: ``(2, 65)``."""
        angles = np.linspace(0.0, 2.0 * np.pi, 65)
        return np.vstack(
            [
                self._center[0] + self._radius * np.cos(angles),
                self._center[1] + self._radius * np.sin(angles),
            ]
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"center={self._center}, radius={self._radius}, "
            f"transition_width={self._transition_width})"
        )
