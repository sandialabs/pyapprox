from abc import ABC, abstractmethod
import math

from pyapprox.util.visualization import get_meshgrid_samples
from pyapprox.surrogates.bases.basis import TensorProductInterpolatingBasis
from pyapprox.surrogates.bases.basisexp import TensorProductInterpolant
from pyapprox.pde.autopde._mesh import (
    OrthogonalCoordinateMesh, ChebyshevCollocationMesh
)
from pyapprox.surrogates.bases.orthopoly import (
    UnivariateChebyhsev1stKindGaussLobattoBarycentricLagrangeBasis
)


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
        return "{0}(mesh={1})".format(self.__class__.__name__, self.mesh)


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


class CollocationFunction(ABC):
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
        return "{0}(basis={1})".format(self.__class__.__name__, self.basis)

    def _plot_1d(self, ax, nplot_pts_1d, **kwargs):
        plot_samples = self._bkd.linspace(
            *self.basis.mesh.trans._ranges, nplot_pts_1d
        )[None, :]
        return ax.plot(plot_samples[0], self(plot_samples), **kwargs)

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


class MatrixCollocationFunction(CollocationFunction):
    def __init__(
            self, basis, nrows, ncols, values_at_mesh=None, jac=None,
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
        values_at_mesh : array (nrows, ncols, nmesh_pts)
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
        jac : array (nrows, ncols, nmesh_pts, nmesh_pts)
            The jacobian of each function at the mesh points
        """
        if isinstance(jac, str) and jac == "identity":
            self._fun_type = "solution"
            jac = self._bkd.stack(
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
        elif isinstance(jac, str) and jac == "zero":
            self._fun_type = "solution_independent"
            jac = self._bkd.zeros(self.jacobian_shape())
        elif isinstance(jac, str):
            raise ValueError("jac str can only be identity or zero")

        self._fun_type = "operator"
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
        return MatrixCollocationFunction(
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
        # use product rule
        jac = (
            other.get_jacobian() * self.get_values()[..., None]
            + other.get_values()[..., None] * self.get_jacobian()
        )
        return MatrixCollocationFunction(
            other.basis, other._nrows, other._ncols, values, jac
        )

    def __add__(self, other):
        if self.jacobian_shape() != other.jacobian_shape():
            raise ValueError("self and other have different shapes")
        return MatrixCollocationFunction(
            self.basis,
            self._nrows,
            self._ncols,
            self.get_values()+other.get_values(),
            self.get_jacobian()+other.get_jacobian(),
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


class ScalarCollocationFunction(MatrixCollocationFunction):
    def __init__(
            self, basis, values_at_mesh=None, jac=None,
    ):
        super().__init__(basis, 1, 1, values_at_mesh, jac)

    def shape(self):
        return [1,]

    def set_values(self, values_at_mesh):
        if (
                values_at_mesh.ndim != 1
                or values_at_mesh.shape != (self.basis.mesh.nmesh_pts(),)
        ):
            raise ValueError(
                "values_at_mesh shape {0} should be {1}".format(
                    values_at_mesh.shape,  (self.basis.mesh.nmesh_pts(),))
            )
        super().set_values(values_at_mesh[None, None, :])

    def set_jacobian(self, jac):
        """
        Parameters
        ----------
        jac : array or str
            "zero" - corresponds to all entries of jacobian being equal to zero
            "identity"  - corresponds to the jacobian being the diagonal matrix
            otherwise - array (nmesh_pts, nmesh_pts)
        """
        if isinstance(jac, str):
            return super().set_jacobian(jac)

        if (
                jac.shape
                != (self.basis.mesh.nmesh_pts(), self.basis.mesh.nmesh_pts())
        ):
            raise ValueError(
                "jac shape {0} should be {1}".format(
                    jac.shape,
                    (self.basis.mesh.nmesh_pts(), self.basis.mesh.nmesh_pts())
                )
            )
        super().set_jacobian(jac[None, None, :, :])

    def __call__(self, eval_samples):
        return super().__call__(eval_samples)


class CollocationOperator(ABC):
    @abstractmethod
    def _values(self, fun: CollocationFunction):
        raise NotImplementedError

    @abstractmethod
    def _jacobian(self, fun: CollocationFunction):
        raise NotImplementedError

    def __call__(self, fun: CollocationFunction):
        values = self._values(fun)
        return MatrixCollocationFunction(
            fun.basis,
            values.shape[0],
            values.shape[1],
            values,
            self._jacobian(fun),
        )


class ScalarCollocationOperatorFromCallable(CollocationOperator):
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


class Physics(ABC):
    def __init__(self):
        self._nonlinear_res_implemented = False

    @abstractmethod
    def _linear_residual(self, sol: CollocationFunction):
        raise NotImplementedError

    def linear_residual(self, sol: CollocationFunction):
        # Only compute linear jacobian once.
        # This is useful for transient problems
        if hasattr(self, "_linear_jac"):
            return self._linear_jac @ sol, self._linear_jac
        vals, jac = self._linear_residual(sol)
        self._linear_jac = jac
        return vals, jac

    def nonlinear_residual(self, sol: CollocationFunction):
        return 0

    def residual(self, sol: CollocationFunction):
        res_values, res_jac = self.linear_residual(sol)
        if not self._nonlinear_res_implemented:
            return res_values, res_jac
        nonlinear_res_values, nonlinear_res_jac = self.nonlinear_residual(sol)
        return (
            res_values + nonlinear_res_values,
            res_jac + nonlinear_res_jac
        )


class LinearDiffusionEquation(Physics):
    def __init__(
            self, forcing: CollocationFunction, diffusion: CollocationFunction
    ):
        super().__init__()
        self._forcing = forcing
        self._diffusion = diffusion

    def _linear_residual(self, sol: CollocationFunction):
        res = div(self._diffusion*nabla(sol)) + self._forcing
        return res.get_values(), res.get_jacobian()


# nabla f(u) = [D_1f_1,    0  ], d/du (nabla f(u)) = [D_1f_1'(u),     0     ]
#            = [  0   , D_2f_2]                      [   0      , D_2 f'(u) ]
# where f'(u) = d/du f(u)
def nabla(fun: CollocationFunction):
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
    return MatrixCollocationFunction(
        fun.basis, fun.nphys_vars(), 1, grad_vals, grad_jacs
    )


# div f = [D_1 f_1(u) + D_2f_2(u)],  (div f)' = [D_1f'_1(u) + D_2f'_2(u)]
def div(fun: MatrixCollocationFunction):
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
    print(div_vals.shape)
    # dmats: (nrows, n, n)
    # fun_jacs : (nrows, n, n)
    div_jac = fun._bkd.sum(
        fun._bkd.einsum("ijk,ikm->ijm", dmats, fun_jacs), axis=0
    )
    return ScalarCollocationFunction(fun.basis, div_vals, div_jac)


# div (nabla f)  = [D_1, D_2][D_1f_1,    0  ] = [D_1D_1f_1,    0     ]
#                            [  0   , D_2f_2] = [  0      , D_2D_2f_2]
# d/du (nabla f(u)) = [D_1D_1f_1'(u),     0        ]
#                     [   0      ,    D_2D_2 f'(u) ]
def laplace(fun: ScalarCollocationFunction):
    """Laplacian of a scalar valued function"""
    return div(nabla(fun))


def fdotgradf(fun: ScalarCollocationFunction):
    r"""(f \cdot nabla f)f of a vector-valued function f"""
    pass
