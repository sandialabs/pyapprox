from abc import ABC, abstractmethod
import math

from pyapprox.util.linearalgebra.numpylinalg import (
    LinAlgMixin,
    NumpyLinAlgMixin,
)
from pyapprox.pde.autopde._mesh_transforms import (
    OrthogonalCoordinateTransform,
    ScaleAndTranslationTransform1D,
)
from pyapprox.surrogates.bases.orthopoly import GaussLegendreQuadratureRule


class OrthogonalCoordinateMesh(ABC):
    def __init__(self, npts_1d, transform: OrthogonalCoordinateTransform):
        self._nphys_vars = transform.nphys_vars()
        if len(npts_1d) != self._nphys_vars:
            raise ValueError("len(npts_1d) must match nphys_vars()")
        self._npts_1d = npts_1d
        if not isinstance(transform, OrthogonalCoordinateTransform):
            raise ValueError(
                "transform must be an instance of "
                "OrthogonalCoordinateTransform"
            )
        self._bkd = transform._bkd
        self.trans = transform
        self._set_orthogonal_mesh_pts(npts_1d)
        self._mesh_pts = self.trans.map_from_orthogonal(self._orth_mesh_pts)
        self._set_boundaries()
        self._set_boundary_indices()

    @abstractmethod
    def _set_boundaries(self):
        raise NotImplementedError

    @abstractmethod
    def _univariate_orthogonal_mesh_pts(self, npts):
        raise NotImplementedError

    def nphys_vars(self):
        return self._nphys_vars

    def _set_orthogonal_mesh_pts(self, npts_1d):
        self._orth_mesh_pts_1d = [
            self._univariate_orthogonal_mesh_pts(self._npts_1d[ii])
            for ii in range(self.nphys_vars())
        ]
        self._orth_mesh_pts = self._bkd.cartesian_product(
            self._orth_mesh_pts_1d
        )

    def mesh_pts(self):
        return self._mesh_pts

    def _set_boundary_indices(self):
        self._bndry_indices = [[] for ii in range(2 * self.nphys_vars())]
        for ii in range(2 * self.nphys_vars()):
            self._bndry_indices[ii] = self._bndrys[
                ii
            ].orth_samples_on_boundary(self._orth_mesh_pts)

    def boundary_indices(self):
        return self._bndry_indices


class ChebyshevCollocationMesh(OrthogonalCoordinateMesh):
    def _univariate_orthogonal_mesh_pts(self, npts):
        return -self._bkd.cos(self._bkd.linspace(0.0, math.pi, npts))


class OrthogonalCoordinateMeshBoundary(ABC):
    def __init__(
        self,
        tol: float = 1e-15,
        bkd: LinAlgMixin = NumpyLinAlgMixin,
    ):
        self._bkd = bkd
        self._tol = tol

    @abstractmethod
    def orth_normals(self, samples):
        raise NotImplementedError

    @abstractmethod
    def orth_quadrature_rule(self):
        raise NotImplementedError

    @abstractmethod
    def orth_samples_on_boundary(self):
        raise NotImplementedError


class OrthogonalCoordinateMeshBoundary1D(OrthogonalCoordinateMeshBoundary):
    def __init__(
        self,
        bndry_name: str,
        tol: float = 1e-15,
        bkd: LinAlgMixin = NumpyLinAlgMixin,
    ):
        super().__init__(tol, bkd)
        self._bndry_index = {"left": 0, "right": 1}[bndry_name]
        self._orth_normal = self._bkd.asarray([[-1.0], [1.0]])[
            self._bndry_index
        ]
        self._inactive_orth_coord = {"left": -1.0, "right": 1.0}[bndry_name]

    def orth_normals(self, samples):
        return self._bkd.tile(self._orth_normal, (1, samples.shape[1])).T

    def orth_quadrature_rule(self):
        return self._bkd.ones((1, 1)), self._bkd.ones((1, 1))

    def orth_samples_on_boundary(self, orth_samples):
        return self._bkd.where(
            self._bkd.abs(self._inactive_orth_coord - orth_samples[0, :])
            < self._tol
        )[0]


class OrthogonalCoordinateMeshBoundary2D(OrthogonalCoordinateMeshBoundary):
    def __init__(
        self,
        bndry_name: str,
        order: int,
        tol: float = 1e-15,
        bkd: LinAlgMixin = NumpyLinAlgMixin,
    ):
        super().__init__(tol, bkd)
        self._bndry_index = {"left": 0, "right": 1, "bottom": 2, "top": 3}[
            bndry_name
        ]
        self._orth_normal = self._bkd.asarray(
            [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
        )[self._bndry_index]
        self._order = order
        self._active_orth_bounds = self._bkd.asarray([-1.0, 1.0])
        self._inactive_orth_coord = {
            "left": -1.0,
            "right": 1.0,
            "bottom": -1.0,
            "top": 1.0,
        }[bndry_name]

    def orth_normals(self, samples):
        return self._bkd.tile(
            self._orth_normal[:, None], (1, samples.shape[1])
        ).T

    def orth_quadrature_rule(self):
        nsamples = self._order + 3
        active_quad = GaussLegendreQuadratureRule(self._active_orth_bounds)
        active_quadx, active_quadw = active_quad(nsamples)
        inactive_quadx = self._bkd.array([self._inactive_coord])
        xlist = [None, None]
        xlist[int(self._bndry_index < 2)] = active_quadx[0, :]
        xlist[int(self._bndry_index >= 2)] = inactive_quadx
        xquad = self._bkd.cartesian_product(xlist)
        return xquad, active_quadw

    def orth_samples_on_boundary(self, orth_samples):
        dd = int(self._bndry_index >= 2)
        indices = self._bkd.where(
            self._bkd.abs(self._inactive_coord - orth_samples[dd, :])
            < self._tol
        )[0]
        # To avoid corners appear twice. Remove second reference to corners
        # from top and bottom boundaries
        if self._bndry_index in ["bottom", "top"]:
            indices = indices[1:-1]
        return indices


class OrthogonalCoordinateMesh1DMixin(OrthogonalCoordinateMesh):
    def __init__(self, npts_1d, transform: ScaleAndTranslationTransform1D):
        if not isinstance(transform, ScaleAndTranslationTransform1D):
            raise ValueError(
                "transform must be an instance of "
                "ScaleAndTranslationTransform1D"
            )
        super().__init__(npts_1d, transform)

    def _set_boundaries(self):
        self._bndrys = [
            OrthogonalCoordinateMeshBoundary1D(name)
            for name in ["left", "right"]
        ]


class OrthogonalCoordinateMesh2DMixin:
    def _set_boundaries(self):
        self._bndrys = [
            OrthogonalCoordinateMeshBoundary2D(name)
            for name in ["left", "right", "bottom", "top"]
        ]


class ChebyshevCollocationMesh1D(
    ChebyshevCollocationMesh, OrthogonalCoordinateMesh1DMixin
):
    pass


class ChebyshevCollocationMesh2D(
    ChebyshevCollocationMesh, OrthogonalCoordinateMesh2DMixin
):
    pass
