from abc import ABC, abstractmethod
import math
import textwrap

from pyapprox.util.linearalgebra.linalgbase import Array
from pyapprox.surrogates.bases.basis import TensorProductInterpolatingBasis
from pyapprox.surrogates.bases.basisexp import TensorProductInterpolant
from pyapprox.pde.collocation.mesh import (
    OrthogonalCoordinateMesh,
    ChebyshevCollocationMesh,
)
from pyapprox.surrogates.bases.orthopoly import (
    UnivariateChebyhsev1stKindGaussLobattoBarycentricLagrangeBasis,
)
from pyapprox.util.linearalgebra.linalg import qr_solve


class OrthogonalCoordinateCollocationBasis(ABC):
    def __init__(self, mesh: OrthogonalCoordinateMesh):
        if not isinstance(mesh, OrthogonalCoordinateMesh):
            raise ValueError(
                "transform must be an instance of " "OrthogonalCoordinateMesh"
            )
        self._bkd = mesh._bkd
        self.mesh = mesh
        self._set_derivative_matrices()

    def _set_derivative_matrices(self):
        orth_deriv_mats_1d = [
            self._form_1d_orth_derivative_matrix(pts)
            for pts in self.mesh._orth_mesh_pts_1d
        ]
        orth_deriv_mats = self._form_orth_derivative_matrices(
            orth_deriv_mats_1d
        )
        gradient_factors = self.mesh.trans.gradient_factors(
            self.mesh._orth_mesh_pts
        )
        self._deriv_mats = []
        for dd in range(self.nphys_vars()):
            self._deriv_mats.append(0)
            for ii in range(self.nphys_vars()):
                self._deriv_mats[-1] += (
                    gradient_factors[:, dd, ii : ii + 1] * orth_deriv_mats[ii]
                )

    def nphys_vars(self):
        return self.mesh.nphys_vars()

    @abstractmethod
    def _form_1d_orth_derivative_matrix(self):
        raise NotImplementedError

    @abstractmethod
    def _interpolate(self, values_at_mesh: Array):
        raise NotImplementedError

    def interpolate(self, values_at_mesh: Array, new_samples: Array):
        if (
            values_at_mesh.ndim != 2
            or values_at_mesh.shape[0] != self.mesh.nmesh_pts()
        ):
            raise ValueError(
                "values_at_mesh shape {0} is wrong".format(
                    values_at_mesh.shape
                )
            )
        new_orth_samples = self.mesh.trans.map_to_orthogonal(new_samples)
        return self._interpolate(values_at_mesh, new_orth_samples)

    def __call__(self):
        raise NotImplementedError("use interpolate")

    def __repr__(self):
        return "{0}".format(self.__class__.__name__)


class ChebyshevCollocationBasis(OrthogonalCoordinateCollocationBasis):
    def __init__(
        self,
        mesh: ChebyshevCollocationMesh,
    ):
        if not isinstance(mesh, ChebyshevCollocationMesh):
            raise ValueError(
                "transform must be an instance of " "ChebyshevCollocationMesh"
            )
        super().__init__(mesh)
        # quadrule only used to define mesh of UnivariateLagrangeBasis
        bases_1d = [
            UnivariateChebyhsev1stKindGaussLobattoBarycentricLagrangeBasis(
                [-1, 1], backend=mesh._bkd
            )
            for dd in range(self.nphys_vars())
        ]
        basis = TensorProductInterpolatingBasis(bases_1d)
        basis.set_tensor_product_indices(self.mesh._npts_1d)
        self._bexp = TensorProductInterpolant(basis)

    def _form_1d_orth_derivative_matrix(self, pts: Array):
        npts = pts.shape[0]
        if npts == 1:
            return self._bkd.array([0], dtype=float)

        # the matrix returned reverse order used by matlab cheb function
        scalars = self._bkd.ones((npts), dtype=float)
        scalars[0] = 2.0
        scalars[npts - 1] = 2.0
        scalars[1:npts:2] *= -1
        derivative_matrix = self._bkd.empty((npts, npts), dtype=float)
        for ii in range(npts):
            row_sum = 0.0
            for jj in range(npts):
                if ii == jj:
                    denominator = 1.0
                else:
                    denominator = pts[ii] - pts[jj]
                numerator = scalars[ii] / scalars[jj]
                derivative_matrix[ii, jj] = numerator / denominator
                row_sum += derivative_matrix[ii, jj]
            derivative_matrix[ii, ii] -= row_sum
        # I return points and calculate derivatives using reverse order of
        # points compared to what is used by Matlab cheb function thus the
        # derivative matrix I return will be the negative of the matlab version
        return derivative_matrix

    def _second_derivative_matrix_entry(self, degree, pts: Array, ii, jj):
        if (ii == 0 and jj == 0) or (ii == degree and jj == degree):
            return (degree**4 - 1) / 15

        if ii == jj and ((ii > 0) and (ii < degree)):
            return -((degree**2 - 1) * (1 - pts[ii] ** 2) + 3) / (
                3 * (1 - pts[ii] ** 2) ** 2
            )

        if ii != jj and (ii > 0 and ii < degree):
            deriv = (
                (-1) ** (ii + jj)
                * (pts[ii] ** 2 + pts[ii] * pts[jj] - 2)
                / ((1 - pts[ii] ** 2) * (pts[ii] - pts[jj]) ** 2)
            )
            if jj == 0 or jj == degree:
                deriv /= 2
            return deriv

        # because I define pts from left to right instead of right to left
        # the next two formulas are different to those in the book
        # Roger Peyret. Spectral Methods for Incompressible Viscous Flow
        # I.e. pts  = -x
        if ii == 0 and jj > 0:
            deriv = (
                2
                / 3
                * (-1) ** jj
                * ((2 * degree**2 + 1) * (1 + pts[jj]) - 6)
                / (1 + pts[jj]) ** 2
            )
            if jj == degree:
                deriv /= 2
            return deriv

        # if ii == degree and jj < degree:
        deriv = (
            2
            / 3
            * (-1) ** (jj + degree)
            * ((2 * degree**2 + 1) * (1 - pts[jj]) - 6)
            / (1 - pts[jj]) ** 2
        )
        if jj == 0:
            deriv /= 2
        return deriv

    def _form_1d_orth_second_derivative_matrix(self, degree):
        # this is reverse order used in book
        pts = -self._bkd.cos(self._bkd.linspace(0.0, math.pi, degree + 1))
        derivative_matrix = self._bkd.empty((degree + 1, degree + 1))
        for ii in range(degree + 1):
            for jj in range(degree + 1):
                derivative_matrix[ii, jj] = (
                    self._chebyshev_second_derivative_matrix_entry(
                        degree, pts, ii, jj
                    )
                )
        return pts, derivative_matrix

    def _interpolate(self, values_at_mesh: Array, new_samples: Array):
        self._bexp.fit(values_at_mesh)
        return self._bexp(new_samples)

    def __repr__(self):
        return "{0}(\n{1}\n)".format(
            self.__class__.__name__,
            textwrap.indent("mesh=" + str(self.mesh), prefix="    "),
        )


class OrthogonalCoordinateBasis1DMixin:
    def _form_orth_derivative_matrices(self, orth_deriv_mats_1d: list[Array]):
        return [self._bkd.copy(orth_deriv_mats_1d[0])]


class OrthogonalCoordinateBasis2DMixin:
    def _form_orth_derivative_matrices(self, orth_deriv_mats_1d):
        # assumes that 2d-mesh_pts varies in x1 faster than x2,
        # e.g. points are
        # [[x11,x21],[x12,x21],[x13,x12],[x11,x22],[x12,x22],...]

        # The following fails with PyTorch 2.3.0
        # I thought it may have been caused by converting numpy to tensor
        # but this code suggests it is not that explicitly.
        # What is confusing is this works in ipython as standalone code
        # For now setting setup to only use pytorch<=2.2
        # mat1 = torch.eye(31, dtype=torch.double)
        # mat2 = torch.ones((31, 31), dtype=torch.double)
        # C = torch.kron(mat1, mat2)
        # print("A", C.shape)
        return [
            self._bkd.kron(
                self._bkd.eye(self.mesh._npts_1d[1]), orth_deriv_mats_1d[0]
            ),
            self._bkd.kron(
                orth_deriv_mats_1d[1], self._bkd.eye(self.mesh._npts_1d[0])
            ),
        ]


class OrthogonalCoordinateBasis3DMixin:
    def _form_orth_derivative_matrices(self, orth_deriv_mats_1d: list[Array]):
        # assumes that 2d-mesh_pts varies in x1 faster than x2,
        # which is faster than x3
        # TODO Need to check this is correct. I just derived it
        Dx = self._bkd.kron(
            self._bkd.eye(self.mesh._npts_1d[2]),
            self._bkd.kron(
                self._bkd.eye(self.mesh._npts_1d[1]), orth_deriv_mats_1d[0]
            ),
        )
        Dy = self._bkd.kron(
            self._bkd.eye(self.mesh._npts_1d[2]),
            self._bkd.kron(
                orth_deriv_mats_1d[1], self._bkd.eye(self.mesh._npts_1d[0])
            ),
        )
        Dz = self._bkd.kron(
            self._bkd.kron(
                orth_deriv_mats_1d[2], self._bkd.eye(self.mesh._npts_1d[1])
            ),
            self._bkd.eye(self.mesh._npts_1d[0]),
        )
        return [Dx, Dy, Dz]


class ChebyshevCollocationBasis1D(
    ChebyshevCollocationBasis, OrthogonalCoordinateBasis1DMixin
):
    pass


class ChebyshevCollocationBasis2D(
    ChebyshevCollocationBasis, OrthogonalCoordinateBasis2DMixin
):
    pass


class ChebyshevCollocationBasis3D(
    ChebyshevCollocationBasis, OrthogonalCoordinateBasis3DMixin
):
    pass
