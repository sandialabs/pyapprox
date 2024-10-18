from abc import ABC, abstractmethod
import math
from typing import Union
import textwrap

from pyapprox.util.visualization import get_meshgrid_samples
from pyapprox.surrogates.bases.basis import TensorProductInterpolatingBasis
from pyapprox.surrogates.bases.basisexp import TensorProductInterpolant
from pyapprox.pde.collocation.mesh import (
    OrthogonalCoordinateMesh,
    ChebyshevCollocationMesh,
    OrthogonalCoordinateMeshBoundary,
)
from pyapprox.pde.collocation.newton import NewtonResidual
from pyapprox.surrogates.bases.orthopoly import (
    UnivariateChebyhsev1stKindGaussLobattoBarycentricLagrangeBasis
)
from pyapprox.util.linearalgebra.linalg import qr_solve
from pyapprox.util.linearalgebra.linalgbase import Array


class OrthogonalCoordinateCollocationBasis(ABC):
    def __init__(self, mesh: OrthogonalCoordinateMesh):
        if not isinstance(mesh,  OrthogonalCoordinateMesh):
            raise ValueError(
                "transform must be an instance of "
                "OrthogonalCoordinateMesh"
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
                    gradient_factors[:, dd, ii:ii+1]*orth_deriv_mats[ii]
                )

    def nphys_vars(self):
        return self.mesh.nphys_vars()

    @abstractmethod
    def _form_1d_orth_derivative_matrix(self):
        raise NotImplementedError

    @abstractmethod
    def _interpolate(self, values_at_mesh):
        raise NotImplementedError

    def interpolate(self, values_at_mesh, new_samples):
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
                "transform must be an instance of "
                "ChebyshevCollocationMesh"
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

    def _form_1d_orth_derivative_matrix(self, pts):
        npts = pts.shape[0]
        if npts == 1:
            return self._bkd.array([0], dtype=float)

        # the matrix returned reverse order used by matlab cheb function
        scalars = self._bkd.ones((npts), dtype=float)
        scalars[0] = 2.
        scalars[npts-1] = 2.
        scalars[1:npts:2] *= -1
        derivative_matrix = self._bkd.empty((npts, npts), dtype=float)
        for ii in range(npts):
            row_sum = 0.
            for jj in range(npts):
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
        return derivative_matrix

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

    def _form_1d_orth_second_derivative_matrix(self, degree):
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

    def _interpolate(self, values_at_mesh, new_samples):
        self._bexp.fit(values_at_mesh)
        return self._bexp(new_samples)

    def __repr__(self):
        return "{0}(\n{1}\n)".format(
            self.__class__.__name__,
            textwrap.indent("mesh="+str(self.mesh), prefix="    ")
        )


class OrthogonalCoordinateBasis1DMixin:
    def _form_orth_derivative_matrices(self, orth_deriv_mats_1d):
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
                )
            ]


class OrthogonalCoordinateBasis3DMixin:
    def _form_orth_derivative_matrices(self, orth_deriv_mats_1d):
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


class Function(ABC):
    def __init__(
            self, basis: OrthogonalCoordinateCollocationBasis,
            values_at_mesh=None,
            jac=None
    ):
        if not isinstance(basis, OrthogonalCoordinateCollocationBasis):
            raise ValueError(
                "basis must be an instance of "
                "OrthogonalCoordinateCollocationBasis"
            )
        self._bkd = basis._bkd
        self.basis = basis
        if values_at_mesh is not None:
            self.set_values(values_at_mesh)
        if jac is not None:
            self.set_jacobian(jac)

    def nphys_vars(self):
        return self.basis.nphys_vars()

    @abstractmethod
    def shape(self):
        raise NotImplementedError

    @abstractmethod
    def set_values(self):
        raise NotImplementedError

    @abstractmethod
    def get_values(self):
        raise NotImplementedError

    @abstractmethod
    def set_jacobian(self):
        raise NotImplementedError

    @abstractmethod
    def get_jacobian(self):
        raise NotImplementedError

    @abstractmethod
    def __call__(self):
        raise NotImplementedError

    def __repr__(self):
        return "{0}(\n{1}\n)".format(
            self.__class__.__name__,
            textwrap.indent("basis="+str(self.basis), prefix="    ")
        )

    def _plot_1d(self, ax, nplot_pts_1d, **kwargs):
        plot_samples = self._bkd.linspace(
            *self.basis.mesh.trans._ranges, nplot_pts_1d
        )[None, :]
        return ax.plot(
            plot_samples[0], self(plot_samples)[:, 0, :].T, **kwargs
        )

    def _plot_2d(self, ax, npts_1d, **kwargs):
        # TODO generate samples and interpolate on orth space
        X, Y, plot_samples = get_meshgrid_samples(
            self.basis.mesh.trans._ranges, npts_1d, bkd=self._bkd
        )
        return ax.contourf(
            X, Y, self._bkd.reshape(self(plot_samples), X.shape), **kwargs
        )

    def plot(self, ax, npts_1d=51, **kwargs):
        if self.nphys_vars() == 1:
            return self._plot_1d(ax, npts_1d, **kwargs)
        if self.nphys_vars() == 2:
            return self._plot_2d(ax, npts_1d, **kwargs)


class MatrixFunction(Function):
    def __init__(
            self,
            basis: OrthogonalCoordinateCollocationBasis,
            nrows: int,
            ncols: int,
            values_at_mesh: Array = None,
            jac: Union[Array, str] = None,
    ):
        self._nrows = nrows
        self._ncols = ncols
        super().__init__(basis, values_at_mesh, jac)

    def shape(self):
        return [self._nrows, self._ncols]

    def set_values(self, values_at_mesh):
        """
        Parameters
        ----------
        values_at_mesh : Array (nrows, ncols, nmesh_pts)
            The values of each function at the mesh points
        """
        if (
                values_at_mesh.ndim != 3
                or values_at_mesh.shape != (
                    self._nrows, self._ncols, self.basis.mesh.nmesh_pts()
                )
        ):
            raise ValueError(
                "values_at_mesh shape {0} should be {1}".format(
                    values_at_mesh.shape,
                    (self._nrows, self._ncols, self.basis.mesh.nmesh_pts())
                )
            )
        self._values_at_mesh = values_at_mesh

    def get_jacobian(self):
        return self._jac

    def nmesh_pts(self):
        return self.basis.mesh.nmesh_pts()

    def jacobian_shape(self):
        return (
            self._nrows,
            self._ncols,
            self.basis.mesh.nmesh_pts(),
            self.nmesh_pts()
        )

    def set_jacobian(self, jac):
        """
        Parameters
        ----------
        jac : Array (nrows, ncols, nmesh_pts, nmesh_pts)
            The jacobian of each function at the mesh points
        """
        if (jac.shape != self.jacobian_shape()):
            raise ValueError(
                "jac shape {0} should be {1}".format(
                    jac.shape, self.jacobian_shape()
                )
            )

        self._jac = jac

    def get_values(self):
        return self._values_at_mesh

    def dot(self, other):
        values = self.get_values()
        other_values = self.get_values()
        return MatrixFunction(
            values._nrows,
            other_values._ncols,
            self._bkd.einsum("ijk,jkl->ilk", values, other_values),
            self._bkd
        )

    def __mul__(self, other):
        if self._nrows != 1 or self._ncols != 1:
            raise ValueError(
                "multiplication only defined when other is a scalar valued "
                "function"
            )
        values = self.get_values() * other.get_values()
        # values = self._bkd.copy(other.get_values())
        # use product rule
        jac = (
            other.get_jacobian() * self.get_values()[..., None]
            + other.get_values()[..., None] * self.get_jacobian()
        )
        return MatrixFunction(
            other.basis, other._nrows, other._ncols, values, jac
        )

    def __add__(self, other):
        if self.jacobian_shape() != other.jacobian_shape():
            raise ValueError("self and other have different shapes")
        return MatrixFunction(
            self.basis,
            self._nrows,
            self._ncols,
            self.get_values()+other.get_values(),
            self.get_jacobian()+other.get_jacobian(),
        )

    def __sub__(self, other):
        if self.jacobian_shape() != other.jacobian_shape():
            raise ValueError("self and other have different shapes")
        return MatrixFunction(
            self.basis,
            self._nrows,
            self._ncols,
            self.get_values()-other.get_values(),
            self.get_jacobian()-other.get_jacobian(),
        )

    def __call__(self, eval_samples):
        values_at_mesh = self.get_values()
        flat_values = values_at_mesh.reshape(
            (self._nrows*self._ncols, self.basis.mesh.nmesh_pts())
        ).T
        interp_vals = self.basis.interpolate(flat_values, eval_samples)
        # may not work for matrix valued function
        return self._bkd.stack(
            [vals for vals in interp_vals.T], axis=0).reshape(
                self._nrows, self._ncols, interp_vals.shape[0]
        )


class VectorFunction(MatrixFunction):
    def __init__(
            self,
            basis: OrthogonalCoordinateCollocationBasis,
            nrows: int,
            values_at_mesh: Array = None,
            jac: Union[Array, str] = None,
    ):
        super().__init__(basis, nrows, 1, values_at_mesh, jac)

    def set_values(self, values_at_mesh: Array):
        # This class is to just make initialization of vector function more
        # intuitive. So set_jacobian is not overidden because this is typically
        # called in a manner that is hidden from the user
        if (
                values_at_mesh.ndim != 2
                or values_at_mesh.shape != (
                    self.basis.mesh.nmesh_pts(), self._nrows
                )
        ):
            raise ValueError(
                "values_at_mesh shape {0} should be {1}".format(
                    values_at_mesh.shape,
                    (self.basis.mesh.nmesh_pts(), self._nrows)
                )
            )
        super().set_values(values_at_mesh[None, None, :])


class ScalarFunction(MatrixFunction):
    def __init__(
            self,
            basis: OrthogonalCoordinateCollocationBasis,
            values_at_mesh: Array = None,
            jac: Union[Array, str] = None,
    ):
        super().__init__(basis, 1, 1, values_at_mesh, jac)

    def set_values(self, values_at_mesh: Array):
        if (
                values_at_mesh.ndim != 1
                or values_at_mesh.shape != (self.basis.mesh.nmesh_pts(),)
        ):
            raise ValueError(
                "values_at_mesh shape {0} should be {1}".format(
                    values_at_mesh.shape,  (self.basis.mesh.nmesh_pts(),))
            )
        super().set_values(values_at_mesh[None, None, :])



class ImutableMixin:
    def _initial_jacobian(self):
        return self._bkd.zeros(self.jacobian_shape())


class ImutableScalarFunction(ImutableMixin, ScalarFunction):
    pass


class FunctionFromCallableMixin:
    def _setup(self, fun: callable):
        self._fun = fun
        self.set_values(self._fun(self.basis.mesh.mesh_pts()))
        self.set_jacobian(self._initial_jacobian())




class Operator(ABC):
    @abstractmethod
    def _values(self, fun: Function):
        raise NotImplementedError

    @abstractmethod
    def _jacobian(self, fun: Function):
        raise NotImplementedError

    def __call__(self, fun: Function):
        values = self._values(fun)
        return MatrixFunction(
            fun.basis,
            values.shape[0],
            values.shape[1],
            values,
            self._jacobian(fun),
        )


class ScalarOperatorFromCallable(Operator):
    def __init__(self, op_values_fun, op_jac_fun):
        self._op_values_fun = op_values_fun
        self._op_jac_fun = op_jac_fun

    def _values(self, fun):
        vals = self._op_values_fun(fun.get_values())
        if vals.shape != fun.get_values().shape:
            raise RuntimeError(
                "op_values_fun returned array with the wrong shape"
            )
        return vals

    def _jacobian(self, fun):
        jac = self._op_jac_fun(fun.get_values())
        if jac.shape != fun.get_jacobian().shape:
            raise RuntimeError(
                "op_jac_fun returned array with the wrong shape"
            )
        return jac


class BoundaryFunction(ABC):
    def __init__(self, mesh_bndry: OrthogonalCoordinateMeshBoundary):
        if not isinstance(mesh_bndry, OrthogonalCoordinateMeshBoundary):
            raise ValueError(
                "mesh_bndry must be an instance of "
                "OrthogonalCoordinateMeshBoundary"
            )
        self._mesh_bndry = mesh_bndry
        self._bkd = self._mesh_bndry._bkd

    @abstractmethod
    def apply_to_residual(self, sol: Array, res_array: Array, jac: Array):
        raise NotImplementedError

    @abstractmethod
    def apply_to_jacobian(self, sol: Array, jac: Array):
        raise NotImplementedError

    def _bndry_slice(self, vec, idx, axis):
        # avoid copying data
        if len(idx) == 1:
            if axis == 0:
                return vec[idx]
            return vec[:, idx]

        stride = idx[1]-idx[0]
        if axis == 0:
            return vec[idx[0]:idx[-1]+1:stride]
        return vec[:, idx[0]:idx[-1]+1:stride]

    @abstractmethod
    def __call__(self):
        raise NotImplementedError

    def __repr__(self):
        return "{0}(bndry={1})".format(
            self.__class__.__name__, self._mesh_bndry
        )


class DirichletBoundary(BoundaryFunction):
    def apply_to_residual(self, sol: Array, res: Array, jac: Array):
        idx = self._mesh_bndry._bndry_idx
        bndry_vals = self._bkd.flatten(
            self(self._mesh_bndry._bndry_mesh_pts)
        )
        res[idx] = self._bndry_slice(sol, idx, 0)-bndry_vals
        return res

    def apply_to_jacobian(self, sol: Array, jac: Array):
        idx = self._mesh_bndry._bndry_idx
        jac[idx, ] = 0.
        jac[idx, idx] = 1.
        return jac


class BoundaryFromFunctionMixin:
    def _set_function(self, fun: Function):
        if not isinstance(fun, Function):
            raise ValueError("fun must be an instance of Function")
        self._fun = fun

    def __call__(self, bndry_mesh_pts):
        return self._fun(bndry_mesh_pts)


class DirichletBoundaryFromFunction(
        BoundaryFromFunctionMixin, DirichletBoundary
):
    def __init__(
            self, mesh_bndry: OrthogonalCoordinateMeshBoundary, fun: Function
    ):
        super().__init__(mesh_bndry)
        self._set_function(fun)


class RobinBoundary(BoundaryFunction):
    def __init__(
            self,
            mesh_bndry: OrthogonalCoordinateMeshBoundary,
            alpha: float,
            beta: float,
    ):
        super().__init__(mesh_bndry)
        self._alpha = alpha
        self._beta = beta
        self._normal_vals = self._mesh_bndry.normals(
            self._mesh_bndry._bndry_mesh_pts
        )
        self._flux_jac = None

    def set_flux_jacobian(self, flux_jac):
        self._flux_jac = flux_jac

    def _flux_normal_jacobian(self, sol_array: Array):
        # pass in sol in case flux depends on sol
        # todo. (determine if sol is ever needed an remove if not)
        # todo only compute once for linear transient problems
        # todo: flux_jac called here evaluates flux jac on all grid points
        #       find way to only evalyate on boundary
        idx = self._mesh_bndry._bndry_idx
        flux_jac = self._flux_jac(sol_array)
        flux_jac = [self._bndry_slice(f, idx, 0) for f in flux_jac]
        flux_normal_jac = [
            self._normal_vals[:, dd:dd+1]*flux_jac[dd]
            for dd in range(self._mesh_bndry.nphys_vars())]
        return flux_normal_jac

    def apply_to_residual(
            self, sol_array: Array, res_array: Array, jac: Array
    ):
        idx = self._mesh_bndry._bndry_idx
        bndry_vals = self._bkd.flatten(
            self(self._mesh_bndry._bndry_mesh_pts)
        )
        # todo flux_normal vals gets called here and in apply_to_jacobian
        # remove this computational redundancy
        jac[idx] = sum(self._flux_normal_jacobian(sol_array))
        res_array[idx] = (
            self._alpha * self._bndry_slice(sol_array, idx, 0)
            + self._beta*(self._bndry_slice(jac, idx, 0) @ sol_array)
            - bndry_vals
        )
        return res_array

    def apply_to_jacobian(self, sol: Array, jac: Array):
        # ignoring normals
        # res = D1 * u + D2 * u + alpha * I
        # so jac(res) = D1 + D2 + alpha * I
        idx = self._mesh_bndry._bndry_idx
        # D1 + D2
        jac[idx] = sum(self._flux_normal_jacobian(sol))
        # alpha * I
        jac[idx, idx] += self._alpha
        return jac


class RobinBoundaryFromFunction(
        BoundaryFromFunctionMixin, RobinBoundary
):
    def __init__(
            self,
            mesh_bndry: OrthogonalCoordinateMeshBoundary,
            fun: Function,
            alpha: float,
    ):
        super().__init__(mesh_bndry, alpha)
        self._set_function(fun)


class PeriodicBoundary(BoundaryFunction):
    def __init__(
            self,
            mesh_bndry: OrthogonalCoordinateMeshBoundary,
            partner_mesh_bndry: OrthogonalCoordinateMeshBoundary,
            basis: OrthogonalCoordinateCollocationBasis,
    ):
        super().__init__(mesh_bndry)
        self._partner_mesh_bndry = partner_mesh_bndry
        self._basis = basis
        self._normal_vals = self._mesh_bndry.normals(
            self._mesh_bndry._bndry_mesh_pts
        )
        self._partner_normal_vals = self._partner_mesh_bndry.normals(
            self._partner_mesh_bndry._bndry_mesh_pts
        )

    def _gradient_dot_normal_jacobian(
            self,
            mesh_bndry: OrthogonalCoordinateMeshBoundary,
            normal_vals: Array,
     ):
        idx = mesh_bndry._bndry_idx
        return sum(
            [
                normal_vals[:, dd:dd+1] * (
                    self._bndry_slice(
                        self._basis._deriv_mats[dd], idx, 0
                    )
                )
                for dd in range(self._mesh_bndry.nphys_vars())
            ]
        )

    def _gradient_dot_normal(
            self,
            mesh_bndry: OrthogonalCoordinateMeshBoundary,
            normal_vals: Array,
            sol_array: Array,
    ):
        jac = self._gradient_dot_normal_jacobian(mesh_bndry, normal_vals)
        return jac @ sol_array

    def apply_to_residual(
            self, sol_array: Array, res_array: Array, jac: Array
    ):
        idx1 = self._mesh_bndry._bndry_idx
        idx2 = self._partner_mesh_bndry._bndry_idx
        # match solution values
        res_array[idx1] = sol_array[idx1]-sol_array[idx2]
        # match flux
        res_array[idx2] = (
            self._gradient_dot_normal(
                self._mesh_bndry, self._normal_vals, sol_array
            )
            + self._gradient_dot_normal(
                self._partner_mesh_bndry, self._partner_normal_vals, sol_array
            )
        )
        return res_array

    def apply_to_jacobian(self, sol_array: Array, jac: Array):
        idx1 = self._mesh_bndry._bndry_idx
        idx2 = self._partner_mesh_bndry._bndry_idx
        jac[idx1, :] = 0
        jac[idx1, idx1] = 1
        jac[idx1, idx2] = -1
        jac[idx2] = (
            self._gradient_dot_normal_jacobian(
                self._mesh_bndry, self._normal_vals
            )
            + self._gradient_dot_normal_jacobian(
                self._partner_mesh_bndry, self._partner_normal_vals
            )
        )
        return jac

    def __call__(self, samples):
        raise NotImplementedError("Periodic Boundary does not need __call__")


class SolutionMixin:
    def _initial_jacobian(self):
        return self._bkd.stack(
                [
                    self._bkd.stack(
                        [
                            self._bkd.eye(self.nmesh_pts())
                            for jj in range(self._ncols)
                        ],
                        axis=0
                    )
                    for ii in range(self._nrows)
                ],
                axis=0
            )


class ScalarSolution(SolutionMixin, ScalarFunction):
    pass


class ImutableScalarFunctionFromCallable(
        ImutableScalarFunction, FunctionFromCallableMixin
):
    """Scalar function that does not depend on the solution of a PDE"""
    def __init__(
            self,
            basis: OrthogonalCoordinateCollocationBasis,
            fun: callable,
    ):
        ScalarFunction.__init__(self, basis)
        self._setup(fun)


class ScalarSolutionFromCallable(
        ScalarSolution, FunctionFromCallableMixin
):
    """Scalar solution of a PDE"""
    def __init__(
            self,
            basis: OrthogonalCoordinateCollocationBasis,
            fun: callable,
    ):
        ScalarSolution.__init__(self, basis)
        self._setup(fun)


class TransientFunctionFromCallableMixin:
    def _check_time_set(self):
        if not hasattr(self, "_time"):
            raise ValueError(
                "Must call set_time before evaluating the function"
            )

    def set_time(self, time):
        self._time = time

    def get_time(self):
        self._check_time_set()
        return self._time

    def _eval(self, mesh_pts):
        self._check_time_set()
        return self._fun(mesh_pts, time=self._time)


class Physics(NewtonResidual):
    def __init__(self, basis: OrthogonalCoordinateCollocationBasis):
        super().__init__(basis._bkd)
        if not isinstance(basis, OrthogonalCoordinateCollocationBasis):
            raise ValueError(
                "basis must be an instance of "
                "OrthogonalCoordinateCollocationBasis"
            )
        self.basis = basis
        self._flux_jacobian_implemented = False

    def set_boundaries(self, bndrys: list[BoundaryFunction]):
        nperiodic_boundaries = 0
        for bndry in bndrys:
            if isinstance(bndry, PeriodicBoundary):
                nperiodic_boundaries += 1
        if len(bndrys) + nperiodic_boundaries != len(self.basis.mesh._bndrys):
            raise ValueError("Must set all boundaries")
        self._bndrys = bndrys
        for bndry in self._bndrys:
            if isinstance(bndry, RobinBoundary):
                if not self._flux_jacobian_implemented:
                    raise ValueError(
                        f"RobinBoundary requested but {self} "
                        "does not define _flux_jacobian"
                    )
                bndry.set_flux_jacobian(self._flux_jacobian)

    def apply_boundary_conditions_to_residual(
            self, sol_array: Array, res_array: Array, jac: Array
    ):
        for bndry in self._bndrys:
            res_array = bndry.apply_to_residual(sol_array, res_array, jac)
        return res_array

    def apply_boundary_conditions_to_jacobian(
            self, sol_array: Array, jac: Array
    ):
        for bndry in self._bndrys:
            res_array = bndry.apply_to_jacobian(sol_array, jac)
        return res_array

    @abstractmethod
    def residual(self, sol: Function):
        raise NotImplementedError

    def _residual_function_from_solution_array(self, sol_array: Array):
        self._bkd.assert_isarray(self._bkd, sol_array)
        sol = self._separate_solutions(sol_array)
        return self.residual(sol)

    def _residual_array_and_jacobian_from_solution_array(
            self, sol_array: Array
    ):
        res = self._residual_function_from_solution_array(sol_array)
        res_array = self._bkd.flatten(res.get_values())
        jac = res.get_jacobian()
        jac = self._bkd.reshape(
            jac, (jac.shape[0]*jac.shape[2], jac.shape[3])
        )
        return res_array, jac

    def _residual_array_from_solution_array(self, sol_array: Array):
        # TODO add option to matrix function that stops jacobian being computed
        # when computing residual. useful for explicit time stepping
        # and linear problems where jacobian does not depend on time or
        # on uncertain parameters of PDE
        return self._residual_array_and_jacobian_from_solution_array(
            sol_array
        )[0]

    @abstractmethod
    def _separate_solutions(self, array):
        raise NotImplementedError

    def separate_solutions(self, sol_array: Array):
        sol = self._separate_solutions(sol_array)
        if not isinstance(sol, SolutionMixin) or not isinstance(sol, Function):
            raise RuntimeError(
                "sol must be derived from SolutionMixin and Function"
            )
        return sol

    def __call__(self, sol_array: Array):
        res_array, jac = self._residual_array_and_jacobian_from_solution_array(
            sol_array
        )
        self._jac = jac
        return self.apply_boundary_conditions_to_residual(
            sol_array, res_array, jac
        )

    def jacobian(self, sol_array: Array):
        # assumes jac called after __call__
        return self.apply_boundary_conditions_to_jacobian(
            sol_array, self._jac)

    def linsolve(self, sol_array: Array, res_array: Array):
        self._bkd.assert_isarray(self._bkd, sol_array)
        self._bkd.assert_isarray(self._bkd, res_array)
        return self._linsolve(sol_array, res_array)

    def _flux_jacobian(self, sol_array: Array):
        raise NotImplementedError


class LinearPhysicsMixin:
    #TODO add option of prefactoring jacobian
    #def _linsolve(self, sol_array: Array, res_array: Array):
    #    return qr_solve(self._Q, self._R, res_array, bkd=self._bkd)

    def _linsolve(self, sol_array: Array, res_array: Array):
        return self._bkd.solve(self.jacobian(sol_array), res_array)

    # def _residual_from_array(self, sol_array: Array):
    #     # Only compute linear jacobian once.
    #     # This is useful for transient problems or for steady state
    #     # parameterized PDEs with jacobians that are not dependent on
    #     # the uncertain parameters
    #     if hasattr(self, "_linear_jac"):
    #         return self._linear_jac @ sol_array
    #     return self._linear_residual_from_function().get_values()


class NonLinearPhysicsMixin:
    def _linsolve(self, sol_array: Array, res_array: Array):
        return self._bkd.solve(self.jacobian(sol_array), res_array)


class ScalarPhysicsMixin:
    def _separate_solutions(self, array: Array):
        sol = ScalarSolution(self.basis)
        sol.set_values(array)
        sol.set_jacobian(sol._initial_jacobian())
        return sol


class LinearDiffusionEquation(LinearPhysicsMixin, ScalarPhysicsMixin, Physics):
    def __init__(
            self,
            forcing: ImutableScalarFunction,
            diffusion: ImutableScalarFunction,
    ):
        if not isinstance(forcing, ImutableScalarFunction):
            raise ValueError(
                "forcing must be an instance of ImutableScalarFunction"
            )
        if not isinstance(diffusion, ImutableScalarFunction):
            raise ValueError(
                "diffusion must be an instance of ImutableScalarFunction"
            )
        super().__init__(forcing.basis)
        self._forcing = forcing
        self._diffusion = diffusion
        self._flux_jacobian_implemented = True

    def residual(self, sol: ScalarFunction):
        if not isinstance(sol, ScalarSolution):
            raise ValueError("sol must be an instance of ScalarFunction")
        return div(self._diffusion*nabla(sol)) + self._forcing

    def _flux_jacobian(self, sol_array: Array):
        sol = self.separate_solutions(sol_array)
        flux_jac = (self._diffusion*nabla(sol)).get_jacobian()
        return flux_jac[:, 0, :, :]


class LinearReactionDiffusionEquation(LinearDiffusionEquation):
    def __init__(
            self,
            forcing: ImutableScalarFunction,
            diffusion: ImutableScalarFunction,
            reaction: ImutableScalarFunction,
    ):
        super().__init__(forcing, diffusion)
        if not isinstance(reaction, ImutableScalarFunction):
            raise ValueError(
                "reaction must be an instance of ImutableScalarFunction"
            )
        self._reaction = reaction

    def residual(self, sol: ScalarFunction):
        diff_res = super().residual(sol)
        react_res = self._reaction * sol
        return diff_res - react_res


# nabla f(u) = [D_1f_1,    0  ], d/du (nabla f(u)) = [D_1f_1'(u),     0     ]
#            = [  0   , D_2f_2]                      [   0      , D_2 f'(u) ]
# where f'(u) = d/du f(u)
def nabla(fun: Function):
    """Gradient of a scalar valued function"""
    funvalues = fun.get_values()[0, 0]
    fun_jac = fun.get_jacobian()
    # todo need to create 3d array
    grad_vals = fun._bkd.stack(
        [
            fun.basis._deriv_mats[dd] @ funvalues
            for dd in range(fun.nphys_vars())
        ],
        axis=0
    )[:, None, :]
    grad_jacs = fun._bkd.stack(
        [
            (fun.basis._deriv_mats[dd] @ fun_jac[0, 0])[None, :]
            for dd in range(fun.nphys_vars())
        ],
        axis=0
    )
    return MatrixFunction(
        fun.basis, fun.nphys_vars(), 1, grad_vals, grad_jacs
    )


# div f = [D_1 f_1(u) + D_2f_2(u)],  (div f)' = [D_1f'_1(u) + D_2f'_2(u)]
def div(fun: MatrixFunction):
    """Divergence of a vector valued function."""
    if fun._ncols != 1:
        raise ValueError("Fun must be a vector valued function")
    fun_values = fun.get_values()[:, 0, ...]
    fun_jacs = fun.get_jacobian()[:, 0, ...]
    dmats = fun._bkd.stack(fun.basis._deriv_mats, axis=0)
    # dmats: (nrows, n, n)
    # fun_values : (nrows, n)
    div_vals = fun._bkd.sum(
        fun._bkd.einsum("ijk,ik->ij", dmats, fun_values), axis=0
    )
    # dmats: (nrows, n, n)
    # fun_jacs : (nrows, n, n)
    div_jac = fun._bkd.sum(
        fun._bkd.einsum("ijk,ikm->ijm", dmats, fun_jacs), axis=0
    )
    return MatrixFunction(
        fun.basis, 1, 1, div_vals[None, None, :], div_jac[None, None, ...]
    )


# div (nabla f)  = [D_1, D_2][D_1f_1,    0  ] = [D_1D_1f_1,    0     ]
#                            [  0   , D_2f_2] = [  0      , D_2D_2f_2]
# d/du (nabla f(u)) = [D_1D_1f_1'(u),     0        ]
#                     [   0      ,    D_2D_2 f'(u) ]
def laplace(fun : MatrixFunction):
    """Laplacian of a scalar valued function"""
    return div(nabla(fun))


def fdotgradf(fun: MatrixFunction):
    r"""(f \cdot nabla f)f of a vector-valued function f"""
    pass
