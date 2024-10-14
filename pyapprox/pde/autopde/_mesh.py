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
        self._npts_1d = npts_1d
        self._orth_mesh_pts_1d = [
            self._univariate_orthogonal_mesh_pts(self._npts_1d[ii])
            for ii in range(self.nphys_vars())
        ]
        self._orth_mesh_pts = self._bkd.cartesian_product(
            self._orth_mesh_pts_1d
        )

    def mesh_pts(self):
        return self._mesh_pts

    def nmesh_pts(self):
        return self._mesh_pts.shape[1]

    def _set_boundary_indices(self):
        for name, bndry in self._bndrys.items():
            bndry.set_orth_mesh_pts_boundary_idx(self._orth_mesh_pts)
            bndry.set_mesh_pts_on_boundary(self._mesh_pts)

    def get_boundaries(self):
        return self._bndrys

    def __repr__(self):
        return "{0}(nphys_vars={1}, npts_1d={2}, npts={3}, bndrys={4})".format(
            self.__class__.__name__,
            self.nphys_vars(),
            self._npts_1d,
            self.nmesh_pts(),
            list(self._bndrys.keys()),
        )


class OrthogonalCoordinateMeshBoundary(ABC):
    def __init__(
        self,
        bndry_name: str,
        tol: float = 1e-15,
        bkd: LinAlgMixin = NumpyLinAlgMixin,
    ):
        self._bndry_name = bndry_name
        self._bkd = bkd
        self._tol = tol

    @abstractmethod
    def orth_normals(self, samples):
        raise NotImplementedError

    @abstractmethod
    def orth_quadrature_rule(self):
        raise NotImplementedError

    @abstractmethod
    def _orth_samples_on_boundary(self):
        raise NotImplementedError

    def set_orth_mesh_pts_boundary_idx(self, orth_mesh_pts):
        self._bndry_idx = self._orth_samples_on_boundary(orth_mesh_pts)

    def set_mesh_pts_on_boundary(self, mesh_pts):
        self._bndry_mesh_pts = mesh_pts[:, self._bndry_idx]

    def __repr__(self):
        return "{0}(name={1})".format(
            self.__class__.__name__, self._bndry_name
        )


class OrthogonalCoordinateMeshBoundary1D(OrthogonalCoordinateMeshBoundary):
    def __init__(
        self,
        bndry_name: str,
        tol: float = 1e-15,
        bkd: LinAlgMixin = NumpyLinAlgMixin,
    ):
        super().__init__(bndry_name, tol, bkd)
        self._bndry_index = {"left": 0, "right": 1}[bndry_name]
        self._orth_normal = self._bkd.asarray([[-1.0], [1.0]])[
            self._bndry_index
        ]
        self._inactive_orth_coord = {"left": -1.0, "right": 1.0}[bndry_name]

    def orth_normals(self, samples):
        return self._bkd.tile(self._orth_normal, (1, samples.shape[1])).T

    def orth_quadrature_rule(self):
        return self._bkd.ones((1, 1)), self._bkd.ones((1, 1))

    def _orth_samples_on_boundary(self, orth_samples):
        return self._bkd.where(
            self._bkd.abs(self._inactive_orth_coord - orth_samples[0, :])
            < self._tol
        )[0]


class OrthogonalCoordinateMeshBoundary2D(OrthogonalCoordinateMeshBoundary):
    def __init__(
        self,
        bndry_name: str,
        npts: int,
        tol: float = 1e-15,
        bkd: LinAlgMixin = NumpyLinAlgMixin,
    ):
        super().__init__(bndry_name, tol, bkd)
        self._bndry_index = {"left": 0, "right": 1, "bottom": 2, "top": 3}[
            bndry_name
        ]
        self._orth_normal = self._bkd.asarray(
            [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
        )[self._bndry_index]
        self._npts = npts
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
        nsamples = self._npts + 2
        active_orth_quad = GaussLegendreQuadratureRule(
            self._active_orth_bounds
        )
        active_orth_quadx, active_orth_quadw = active_orth_quad(nsamples)
        inactive_orth_quadx = self._bkd.array([self._inactive_orth_coord])
        orth_xlist = [None, None]
        orth_xlist[int(self._bndry_index < 2)] = active_orth_quadx[0, :]
        orth_xlist[int(self._bndry_index >= 2)] = inactive_orth_quadx
        orth_xquad = self._bkd.cartesian_product(orth_xlist)
        return orth_xquad, active_orth_quadw

    def _orth_samples_on_boundary(self, orth_samples):
        dd = int(self._bndry_index >= 2)
        indices = self._bkd.where(
            self._bkd.abs(self._inactive_orth_coord - orth_samples[dd, :])
            < self._tol
        )[0]
        # Avoid corners appearing twice. Remove second reference to corners
        # from top and bottom boundaries
        if self._bndry_index in ["bottom", "top"]:
            indices = indices[1:-1]
        return indices


class OrthogonalCoordinateMeshBoundary3D(OrthogonalCoordinateMeshBoundary):
    def __init__(
        self,
        bndry_name: str,
        npts_1d: int,
        tol: float = 1e-15,
        bkd: LinAlgMixin = NumpyLinAlgMixin,
    ):
        super().__init__(bndry_name, tol, bkd)
        self._bndry_index = {
            "left": 0, "right": 1, "front": 2, "back": 3, "bottom": 4, "top": 5
        }[bndry_name]
        self._orth_normal = self._bkd.asarray(
            [
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0],
            ]
        )[self._bndry_index]
        self._npts_1d = npts_1d
        self._active_orth_bounds = self._bkd.asarray([-1.0, 1.0])
        self._inactive_orth_coord = {
            "left": -1.0,
            "right": 1.0,
            "front": -1.0,
            "back": 1.0,
            "bottom": -1.0,
            "top": 1.0,
        }[bndry_name]

    def orth_normals(self, samples):
        return self._bkd.tile(
            self._orth_normal[:, None], (1, samples.shape[1])
        ).T

    def orth_quadrature_rule(self):
        active_quad = [
            GaussLegendreQuadratureRule(self._active_orth_bounds[0:2]),
            GaussLegendreQuadratureRule(self._active_orth_bounds[2:4])
        ]
        if self._bndry_index < 2:
            # x boundaries
            jdx, idx0, idx1 = 0, 1, 2
        elif self._bndry_index < 4:
            # y boundaries
            jdx, idx0, idx1 = 1, 0, 2
        else:
            # z boundaries
            jdx, idx0, idx1 = 2, 0, 1
        active_quadx0, active_quadw0 = active_quad[0](self._npts_1d[idx0]+2)
        active_quadx1, active_quadw1 = active_quad[1](self._npts_1d[idx1]+2)
        active_quadw = self._bkd.outer_product(
            [active_quadw0[:, 0], active_quadw1[:, 0]]
        )
        inactive_quadx = self._bkd.array([self._inactive_coord])
        xlist = [None, None, None]
        xlist[jdx] = inactive_quadx
        xlist[idx0] = active_quadx0
        xlist[idx1] = active_quadx1
        xquad = self._bkd.cartesian_product(xlist)
        return xquad, active_quadw

    def _orth_samples_on_boundary(self, orth_samples):
        dd = int(self._bndry_index >= 2)
        indices = self._bkd.where(
            self._bkd.abs(self._inactive_coord - orth_samples[dd, :])
            < self._tol
        )[0]
        # To avoid points on edge lines appearing twice.
        # Remove second reference to edge lines
        # from top and bottom boundaries
        raise NotImplementedError("TODO")
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
        self._bndrys = {
            name: OrthogonalCoordinateMeshBoundary1D(name)
            for name in ["left", "right"]
        }


class OrthogonalCoordinateMesh2DMixin:
    def _set_boundaries(self):
        self._bndrys = {
            name: OrthogonalCoordinateMeshBoundary2D(
                name, self._npts_1d[ii < 2]
            )
            for ii, name in enumerate(["left", "right", "bottom", "top"])
        }


class OrthogonalCoordinateMesh3DMixin:
    def _set_boundaries(self):
        self._bndrys = {
            name: OrthogonalCoordinateMeshBoundary3D(name, self._npts_1d)
            # bounaries are in x, then y then z
            for name in ["left", "right", "front", "back", "bottom", "top"]
        }


class ChebyshevCollocationMesh(OrthogonalCoordinateMesh):
    def _univariate_orthogonal_mesh_pts(self, npts):
        return -self._bkd.cos(self._bkd.linspace(0.0, math.pi, npts))


class ChebyshevCollocationMesh1D(
    ChebyshevCollocationMesh, OrthogonalCoordinateMesh1DMixin
):
    pass


class ChebyshevCollocationMesh2D(
        OrthogonalCoordinateMesh2DMixin, ChebyshevCollocationMesh,
):
    pass


class ChebyshevCollocationMesh3D(
        OrthogonalCoordinateMesh3DMixin, ChebyshevCollocationMesh,
):
    pass
