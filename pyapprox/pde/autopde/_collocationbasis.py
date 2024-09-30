from abc import abstractmethod
import math


from pyapprox.surrogates.bases.basis import Basis
from pyapprox.pde.autopde._mesh import (
    OrthogonalCoordinateMesh, ChebyshevCollocationMesh
)


class OrthogonalCoordinateCollocationBasis(Basis):
    def __init_(self, mesh: OrthogonalCoordinateMesh):
        if not isinstance(mesh,  OrthogonalCoordinateMesh):
            raise ValueError(
                "transform must be an instance of "
                "OrthogonalCoordinateMesh"
            )
        self.mesh = mesh
        self._form_orth_derivative_matrices()

    def _form_orth_derivative_matrices(self):
        orth_deriv_mats_1d = [
            self._form_1d_orth_derivative_matrices(self.mesh._npts_1d)
            for npts in self._npts_1d
        ]
        return self.mesh._derivative_matrices(orth_deriv_mats_1d)

    @abstractmethod
    def _form_1d_orth_derivative_matrices(self):
        raise NotImplementedError

    @abstractmethod
    def nabla(self):
        raise NotImplementedError

    @abstractmethod
    def laplace(self):
        raise NotImplementedError

    @abstractmethod
    def _interpolate(self, values_at_mesh):
        raise NotImplementedError

    def interpolate(self, values_at_mesh, new_samples):
        if values_at_mesh.shape[0] != self.mesh.nmesh_pts():
            raise ValueError("values_at_mesh has the wrong shape")
        new_orth_samples = self.trans.map_to_orthogonal(new_samples)
        return self._interpolate(values_at_mesh, new_orth_samples)


class ChebyshevCollocationBasis(OrthogonalCoordinateCollocationBasis):
    def __init__(
            self,
            mesh: ChebyshevCollocationMesh,
    ):
        if not isinstance(mesh, ChebyshevCollocationMesh):
            raise ValueError(
                "transform must be an instance of "
                "ChebyshevCollocationMesh"
            )
        super().__init__(mesh._bkd)
        self._mesh = mesh

    def _orth_derivative_1d_matrix(self, order):
        if order == 0:
            pts = self._bkd.array([1], dtype=float)
            derivative_matrix = self._bkd.array([0], dtype=float)
        else:
            # this is reverse order used by matlab cheb function
            pts = -self._bkd.cos(self._bkd.linspace(0., math.pi, order+1))
            scalars = self._bkd.ones((order+1), dtype=float)
            scalars[0] = 2.
            scalars[order] = 2.
            scalars[1:order+1:2] *= -1
            derivative_matrix = self._bkd.empty(
                (order+1, order+1), dtype=float
            )
            for ii in range(order+1):
                row_sum = 0.
                for jj in range(order+1):
                    if (ii == jj):
                        denominator = 1.
                    else:
                        denominator = pts[ii]-pts[jj]
                    numerator = scalars[ii] / scalars[jj]
                    derivative_matrix[ii, jj] = numerator / denominator
                    row_sum += derivative_matrix[ii, jj]
                derivative_matrix[ii, ii] -= row_sum

        # I return points and calculate derivatives using reverse order of
        # points compared to what is used by Matlab cheb function thus the
        # derivative matrix I return will be the negative of the matlab version
        return pts, derivative_matrix

    def _second_derivative_matrix_entry(self, degree, pts, ii, jj):
        if (ii == 0 and jj == 0) or (ii == degree and jj == degree):
            return (degree**4-1)/15

        if (ii == jj and ((ii > 0) and (ii < degree))):
            return -((degree**2-1)*(1-pts[ii]**2)+3)/(
                3*(1-pts[ii]**2)**2)

        if (ii != jj and (ii > 0 and ii < degree)):
            deriv = (-1)**(ii+jj)*(
                pts[ii]**2+pts[ii]*pts[jj]-2)/(
                    (1-pts[ii]**2)*(pts[ii]-pts[jj])**2)
            if jj == 0 or jj == degree:
                deriv /= 2
            return deriv

        # because I define pts from left to right instead of right to left
        # the next two formulas are different to those in the book
        # Roger Peyret. Spectral Methods for Incompressible Viscous Flow
        # I.e. pts  = -x
        if (ii == 0 and jj > 0):
            deriv = 2/3*(-1)**jj*(
                (2*degree**2+1)*(1+pts[jj])-6)/(1+pts[jj])**2
            if jj == degree:
                deriv /= 2
            return deriv

        # if ii == degree and jj < degree:
        deriv = 2/3*(-1)**(jj+degree)*(
            (2*degree**2+1)*(1-pts[jj])-6)/(1-pts[jj])**2
        if jj == 0:
            deriv /= 2
        return deriv

    def _second_derivative_1d_matrix(self, degree):
        # this is reverse order used in book
        pts = -self._bkd.cos(self._bkd.linspace(0., math.pi, degree+1))
        derivative_matrix = self._bkd.empty((degree+1, degree+1))
        for ii in range(degree+1):
            for jj in range(degree+1):
                derivative_matrix[ii, jj] = \
                    self._chebyshev_second_derivative_matrix_entry(
                        degree, pts, ii, jj
                    )
        return pts, derivative_matrix


class OrthogonalCoordinateBasis1DMixin:
    def derivative_matrices(self, orth_deriv_mats_1d):
        return [self._bkd.copy(orth_deriv_mats_1d[0])]

    def _interpolate(self):
        pass


class OrthogonalCoordinateBasis2DMixin:
    def derivative_matrices(self, orth_deriv_mats_1d):
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
                    self._bkd.eye(self._npts_1d[1]), orth_deriv_mats_1d[0]
                ),
                self._bkd.kron(
                    orth_deriv_mats_1d[1], self._bkd.eye(self._npts_1d[0])
                )
            ]


class OrthogonalCoordinateBasis3DMixin:
    def derivative_matrices(self, orth_deriv_mats_1d):
        # assumes that 2d-mesh_pts varies in x1 faster than x2,
        # which is faster than x3
        # TODO Need to check this is correct. I just derived it
        Dx = self._bkd.kron(
            self._bkd.eye(self._npts_1d[2]),
            self._bkd.kron(
                self._bkd.eye(self._npts_1d[1]), orth_deriv_mats_1d[0]
            )
        )
        Dy = self._bkd.kron(
            self._bkd.eye(self._npts_1d[2]),
            self._bkd.kron(
                    orth_deriv_mats_1d[1], self._bkd.eye(self._npts_1d[0])
            )
        )
        Dz = self._bkd.kron(
            self._bkd.kron(
                orth_deriv_mats_1d[2],
                self._bkd.eye(self._npts_1d[1])
            ),
            self._bkd.eye(self._npts_1d[0])
        )
        return [Dx, Dy, Dz]
