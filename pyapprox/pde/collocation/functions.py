import textwrap
from abc import ABC, abstractmethod
from typing import Union

from pyapprox.util.linearalgebra.linalgbase import Array
from pyapprox.util.visualization import get_meshgrid_samples
from pyapprox.pde.collocation.basis import OrthogonalCoordinateCollocationBasis


class MatrixFunction(ABC):
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

    def shape(self):
        return [self._nrows, self._ncols]

    def set_matrix_values(self, values_at_mesh):
        """
        Parameters
        ----------
        values_at_mesh : Array (nrows, ncols, nmesh_pts)
            The values of each function at the mesh points
        """
        if values_at_mesh.ndim != 3 or values_at_mesh.shape != (
            self._nrows,
            self._ncols,
            self.basis.mesh.nmesh_pts(),
        ):
            raise ValueError(
                "values_at_mesh shape {0} should be {1}".format(
                    values_at_mesh.shape,
                    (self._nrows, self._ncols, self.basis.mesh.nmesh_pts()),
                )
            )
        self._values_at_mesh = values_at_mesh

    def get_matrix_jacobian(self):
        return self._jac

    def nmesh_pts(self):
        return self.basis.mesh.nmesh_pts()

    def matrix_jacobian_shape(self):
        return (
            self._nrows,
            self._ncols,
            self.basis.mesh.nmesh_pts(),
            self.nmesh_pts(),
        )

    def set_matrix_jacobian(self, jac):
        """
        Parameters
        ----------
        jac : Array (nrows, ncols, nmesh_pts, nmesh_pts)
            The jacobian of each function at the mesh points
        """
        if jac.shape != self.matrix_jacobian_shape():
            raise ValueError(
                "jac shape {0} should be {1}".format(
                    jac.shape, self.matrix_jacobian_shape()
                )
            )

        self._jac = jac

    def get_matrix_values(self):
        return self._values_at_mesh

    def dot(self, other):
        values = self.get_matrix_values()
        other_values = self.get_matrix_values()
        return MatrixFunction(
            values._nrows,
            other_values._ncols,
            self._bkd.einsum("ijk,jkl->ilk", values, other_values),
            self._bkd,
        )

    def __mul__(self, other):
        if self._nrows != 1 or self._ncols != 1:
            raise ValueError(
                "multiplication only defined when other is a scalar valued "
                "function"
            )
        values = self.get_matrix_values() * other.get_matrix_values()
        # values = self._bkd.copy(other.get_values())
        # use product rule
        jac = (
            other.get_matrix_jacobian() * self.get_matrix_values()[..., None]
            + other.get_matrix_values()[..., None] * self.get_matrix_jacobian()
        )
        return MatrixFunction(
            other.basis, other._nrows, other._ncols, values, jac
        )

    def __add__(self, other):
        if self.matrix_jacobian_shape() != other.matrix_jacobian_shape():
            raise ValueError("self and other have different shapes")
        return MatrixFunction(
            self.basis,
            self._nrows,
            self._ncols,
            self.get_matrix_values() + other.get_matrix_values(),
            self.get_matrix_jacobian() + other.get_matrix_jacobian(),
        )

    def __sub__(self, other):
        if self.matrix_jacobian_shape() != other.matrix_jacobian_shape():
            raise ValueError("self and other have different shapes")
        return MatrixFunction(
            self.basis,
            self._nrows,
            self._ncols,
            self.get_matrix_values() - other.get_matrix_values(),
            self.get_matrix_jacobian() - other.get_matrix_jacobian(),
        )

    def __call__(self, eval_samples: Array):
        values_at_mesh = self.get_matrix_values()
        flat_values = values_at_mesh.reshape(
            (self._nrows * self._ncols, self.basis.mesh.nmesh_pts())
        ).T
        interp_vals = self.basis.interpolate(flat_values, eval_samples)
        # may not work for matrix valued function
        return self._bkd.stack(
            [vals for vals in interp_vals.T], axis=0
        ).reshape(self._nrows, self._ncols, interp_vals.shape[0])

    def set_values(self, values_at_mesh: Array):
        self.set_matrix_values(values_at_mesh)

    def get_values(self):
        return self.get_matrix_values()

    def set_jacobian(self, jac):
        return self.set_matrix_jacobian(jac)

    def get_jacobian(self):
        return self.get_matrix_jacobian()

    def __repr__(self):
        return "{0}(\n{1}\n)".format(
            self.__class__.__name__,
            textwrap.indent("basis=" + str(self.basis), prefix="    "),
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
        self.set_matrix_values(self._reshape_to_matrix(values_at_mesh))

    def _reshape_from_matrix(self, array: Array):
        return array[:, 0]

    def _reshape_to_matrix(self, array: Array):
        if array.ndim != 2 or array.shape != (
            self.basis.mesh.nmesh_pts(),
            self._nrows,
        ):
            raise ValueError(
                "array shape {0} should be {1}".format(
                    array.shape, (self.basis.mesh.nmesh_pts(), self._nrows)
                )
            )
        return array[:, None, :]

    def _reshape_jacobian_to_matrix(self, jac: Array):
        # note reshape_to_matrix takes nrows as last axis.
        # This is to be consistent with rest of pyapprox.
        # But we do not do that here
        jac_shape = (
            self._nrows,
            self.basis.mesh.nmesh_pts(),
            self.basis.mesh.nmesh_pts(),
        )
        if jac.ndim != 2 or jac.shape != jac_shape:
            raise ValueError(
                "jac shape {0} should be {1}".format(
                    jac.shape, jac_shape
                )
            )
        return jac[:, None, ...]

    def _reshape_jacobian_from_matrix(self, jac: Array):
        return jac[:, 0, ...]

    def get_values(self):
        return self._reshape_from_matrix(self.get_matrix_values())

    def get_jacobian(self):
        return self._reshape_jacobian_from_matrix(self.get_matrix_jacobian())

    def set_jacobian(self, jac):
        return self.set_matrix_jacobian(self._reshape_jacobian_to_matrix(jac))

    def __call__(self, eval_samples: Array):
        return self._reshape_from_matrix(super().__call__(eval_samples))


class ScalarFunction(MatrixFunction):
    def __init__(
        self,
        basis: OrthogonalCoordinateCollocationBasis,
        values_at_mesh: Array = None,
        jac: Union[Array, str] = None,
    ):
        super().__init__(basis, 1, 1, values_at_mesh, jac)

    def set_values(self, values_at_mesh: Array):
        self.set_matrix_values(self._reshape_to_matrix(values_at_mesh))

    def _reshape_from_matrix(self, array: Array):
        return array[0, 0]

    def _reshape_to_matrix(self, array: Array):
        if array.ndim != 1 or array.shape != (self.basis.mesh.nmesh_pts(),):
            raise ValueError(
                "array shape {0} should be {1}".format(
                    array.shape, (self.basis.mesh.nmesh_pts(),)
                )
            )
        return array[None, None, :]

    def _reshape_jacobian_to_matrix(self, jac: Array):
        jac_shape = (
            self.basis.mesh.nmesh_pts(),
            self.basis.mesh.nmesh_pts()
        )
        if jac.ndim != 2 or jac.shape != jac_shape:
            raise ValueError(
                "jac shape {0} should be {1}".format(
                    jac.shape, jac_shape
                )
            )
        return jac[None, None, :]

    def _reshape_jacobian_from_matrix(self, jac: Array):
        return jac[0, 0]

    def get_values(self):
        return self._reshape_from_matrix(self.get_matrix_values())

    def get_jacobian(self):
        return self._reshape_jacobian_from_matrix(self.get_matrix_jacobian())

    def set_jacobian(self, jac):
        return self.set_matrix_jacobian(self._reshape_jacobian_to_matrix(jac))

    def __call__(self, eval_samples: Array):
        return self._reshape_from_matrix(super().__call__(eval_samples))

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


class ImutableMixin:
    def __init__(
        self,
        basis: OrthogonalCoordinateCollocationBasis,
        values_at_mesh: Array = None,
    ):
        super().__init__(basis, values_at_mesh)
        self.set_matrix_jacobian(self._initial_matrix_jacobian())

    def _initial_matrix_jacobian(self):
        return self._bkd.zeros(self.matrix_jacobian_shape())


class ImutableScalarFunction(ImutableMixin, ScalarFunction):
    pass


class ImutableVectorFunction(ImutableMixin, VectorFunction):
    pass


class FunctionFromCallableMixin:
    def _setup(self, fun: callable):
        self._fun = fun
        self.set_values(self._fun(self.basis.mesh.mesh_pts()))
        self.set_matrix_jacobian(self._initial_matrix_jacobian())


class Operator(ABC):
    @abstractmethod
    def __new__(
        cls,
        fun: MatrixFunction,
    ):
        raise NotImplementedError


class OperatorFromCallable(Operator):
    @staticmethod
    def _values(fun: MatrixFunction, op_values_fun: callable):
        vals = op_values_fun(fun.get_values())
        if vals.shape != fun.get_values().shape:
            raise RuntimeError(
                "op_values_fun returned array with the wrong shape"
            )
        return vals

    @staticmethod
    def _jacobian(fun: MatrixFunction, op_jac_fun):
        jac = op_jac_fun(fun.get_values())
        if jac.shape != fun.get_jacobian().shape:
            raise RuntimeError(
                "op_jac_fun returned array with the wrong shape"
            )
        return jac


class ScalarOperatorFromCallable(OperatorFromCallable):
    def __new__(
        cls,
        fun: MatrixFunction,
        op_values_fun: callable,
        op_jac_fun: callable,
    ):
        return ScalarFunction(
            fun.basis,
            cls._values(fun, op_values_fun),
            cls._jacobian(fun, op_jac_fun),
        )


class ScalarMonomialOperator(Operator):
    def __new__(
        cls,
        fun: MatrixFunction,
        degree: int,
    ):
        vals = fun.get_values()
        return ScalarFunction(
            fun.basis,
            vals**degree,
            (degree * fun._bkd.diag(vals) ** (degree-1)),
        )


class SolutionMixin:
    def __init__(
        self,
        basis: OrthogonalCoordinateCollocationBasis,
        values_at_mesh: Array = None,
    ):
        super().__init__(basis, values_at_mesh)
        self.set_matrix_jacobian(self._initial_matrix_jacobian())

    def _initial_matrix_jacobian(self):
        return self._bkd.stack(
            [
                self._bkd.stack(
                    [
                        self._bkd.eye(self.nmesh_pts())
                        for jj in range(self._ncols)
                    ],
                    axis=0,
                )
                for ii in range(self._nrows)
            ],
            axis=0,
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


class ImutableVectorFunctionFromCallable(
    ImutableVectorFunction, FunctionFromCallableMixin
):
    """Vector function that does not depend on the solution of a PDE"""

    def __init__(
        self,
        basis: OrthogonalCoordinateCollocationBasis,
        fun: callable,
    ):
        VectorFunction.__init__(self, basis)
        self._setup(fun)


class ScalarSolutionFromCallable(ScalarSolution, FunctionFromCallableMixin):
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
