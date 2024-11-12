import textwrap
from abc import ABC, abstractmethod
from typing import Union, List

import matplotlib as mpl
import matplotlib.pyplot as plt

from pyapprox.util.linearalgebra.linalgbase import Array
from pyapprox.util.visualization import get_meshgrid_samples
from pyapprox.pde.collocation.basis import OrthogonalCoordinateCollocationBasis


# Note from typing documentation
# When a type hint contains names that have not been defined yet, that
# definition may be expressed as a string literal, to be resolved later.
# The string literal should contain a valid Python expression and it
# should evaluate without errors once the module has been fully loaded.


class MatrixOperator(ABC):
    """
    Matrix valued operator, which depends on the PDE solution.

    Functions can be recovered by setting Jacobian (w.r.t solution) to zero
    """

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
            self.nmesh_pts(),
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

    def dot(self, other: "MatrixOperator") -> "MatrixOperator":
        values = self.get_matrix_values()
        other_values = self.get_matrix_values()
        return MatrixOperator(
            values._nrows,
            other_values._ncols,
            self._bkd.einsum("ijk,jkl->ilk", values, other_values),
            self._bkd,
        )

    def _multiply_functions(
        self, other: Union["MatrixOperator", float]
    ) -> "MatrixOperator":
        if not isinstance(other, (MatrixOperator, float)):
            raise ValueError(
                f"cannot multiply Matrixfunction by {type(other)}"
            )

        if isinstance(other, float):
            return MatrixOperator(
                self.basis,
                self._nrows,
                self._ncols,
                self.get_matrix_values() * other,
                self.get_matrix_jacobian() * other,
            )

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
        return MatrixOperator(
            other.basis, other._nrows, other._ncols, values, jac
        )

    def __mul__(
        self, other: Union["MatrixOperator", float]
    ) -> "MatrixOperator":
        return self._multiply_functions(other)

    def __rmul__(
        self, other: Union["MatrixOperator", float]
    ) -> "MatrixOperator":
        return self._multiply_functions(other)

    def _divide_functions(self, fun1, fun2) -> "MatrixOperator":
        if isinstance(fun2, float):
            if fun2 == 0:
                raise ValueError("Cannot divide by zero")
            return MatrixOperator(
                fun1.basis,
                fun1._nrows,
                fun1._ncols,
                fun1.get_matrix_values() / fun2,
                fun1.get_matrix_jacobian() / fun2,
            )

        if isinstance(fun1, float):
            if self._bkd.any(fun2.get_matrix_values() == 0.0):
                raise ValueError("Cannot divide by zero")
            return MatrixOperator(
                fun2.basis,
                fun2._nrows,
                fun2._ncols,
                fun1 / fun2.get_matrix_values(),
                -fun1
                * fun2.get_matrix_jacobian()
                / (fun2.get_matrix_values() ** 2),
            )

        if fun1._nrows != 1 or fun1._ncols != 1:
            raise ValueError(
                "multiplication only defined when other is a scalar valued "
                "function"
            )
        values = fun1.get_matrix_values() / fun2.get_matrix_values()
        # use quotient rule
        jac = (
            fun2.get_matrix_values()[..., None] * fun1.get_matrix_jacobian()
            - fun2.get_matrix_jacobian() * fun1.get_matrix_values()[..., None]
        ) / fun2.get_matrix_values()[..., None] ** 2
        return MatrixOperator(
            fun2.basis, fun2._nrows, fun2._ncols, values, jac
        )

    def __truediv__(
        self, other: Union["MatrixOperator", float]
    ) -> "MatrixOperator":
        if not isinstance(other, (float, MatrixOperator)):
            raise ValueError(f"Cannot divde function by {type(other)}")
        return self._divide_functions(self, other)

    def __rtruediv__(
        self, other: Union["MatrixOperator", float]
    ) -> "MatrixOperator":
        if not isinstance(other, (float, MatrixOperator)):
            raise ValueError(f"Cannot divide {type(other)} by function")
        return self._divide_functions(other, self)

    def _add_functions(
        self, other: Union["MatrixOperator", float]
    ) -> "MatrixOperator":
        if isinstance(other, float):
            return MatrixOperator(
                self.basis,
                self._nrows,
                self._ncols,
                self.get_matrix_values() + other,
                self.get_matrix_jacobian(),
            )

        if self.matrix_jacobian_shape() != other.matrix_jacobian_shape():
            raise ValueError("self and other have different shapes")
        return MatrixOperator(
            self.basis,
            self._nrows,
            self._ncols,
            self.get_matrix_values() + other.get_matrix_values(),
            self.get_matrix_jacobian() + other.get_matrix_jacobian(),
        )

    def __add__(
        self, other: Union["MatrixOperator", float]
    ) -> "MatrixOperator":
        if not isinstance(other, (MatrixOperator, float)):
            raise ValueError(f"cannot add {type(other)} to a Matrixfunction")
        return self._add_functions(other)

    def __radd__(
        self, other: Union["MatrixOperator", float]
    ) -> "MatrixOperator":
        if not isinstance(other, (MatrixOperator, float)):
            raise ValueError(f"cannot add {type(other)} to a Matrixfunction")
        return self._add_functions(other)

    def __pow__(self, other: int) -> "MatrixOperator":
        if not isinstance(other, int) or other < 2:
            raise ValueError("Can only use power with int and power > 1")
        result = self
        for ii in range(2, other + 1):
            result = self * result
        return result

    def sqrt(self) -> "MatrixOperator":
        vals = self.get_matrix_values()
        if self._bkd.any(vals < 0.0):
            raise ValueError("Cannot take sqrt of negative values")
        sqrt_vals = self._bkd.sqrt(vals)
        return MatrixOperator(
            self.basis,
            self._nrows,
            self._ncols,
            sqrt_vals,
            self.get_matrix_jacobian() / (2 * sqrt_vals[..., None]),
        )

    def _subtract_functions(self, fun1, fun2) -> "MatrixOperator":
        if fun1.matrix_jacobian_shape() != fun2.matrix_jacobian_shape():
            raise ValueError("self and other have different shapes")
        return MatrixOperator(
            fun1.basis,
            fun1._nrows,
            fun1._ncols,
            fun1.get_matrix_values() - fun2.get_matrix_values(),
            fun1.get_matrix_jacobian() - fun2.get_matrix_jacobian(),
        )

    def __sub__(
        self, other: Union["MatrixOperator", float]
    ) -> "MatrixOperator":
        if isinstance(other, float):
            return MatrixOperator(
                self.basis,
                self._nrows,
                self._ncols,
                self.get_matrix_values() - other,
                self.get_matrix_jacobian(),
            )

        return self._subtract_functions(self, other)

    def __rsub__(
        self, other: Union["MatrixOperator", float]
    ) -> "MatrixOperator":
        if isinstance(other, float):
            return MatrixOperator(
                self.basis,
                self._nrows,
                self._ncols,
                other - self.get_matrix_values(),
                -self.get_matrix_jacobian(),
            )
        return self._subtract_functions(other, self)

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


class VectorOperator(MatrixOperator):
    """Vector Valued Operator, which depends on the PDE solution."""

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
        return array.T[:, None, :]

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
                "jac shape {0} should be {1}".format(jac.shape, jac_shape)
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


class ScalarOperator(MatrixOperator):
    """Scalar Valued Operator, which depends on the PDE solution."""

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
        jac_shape = (self.basis.mesh.nmesh_pts(), self.basis.mesh.nmesh_pts())
        if jac.ndim != 2 or jac.shape != jac_shape:
            raise ValueError(
                "jac shape {0} should be {1}".format(jac.shape, jac_shape)
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
        orth_range = self.basis.mesh.trans._orthog_ranges
        orth_X, orth_Y, orth_pts_2d = get_meshgrid_samples(
            self._bkd.tile(orth_range, (2,)), npts_1d, bkd=self._bkd
        )
        pts = self.basis.mesh.trans.map_from_orthogonal(orth_pts_2d)
        X = self._bkd.reshape(pts[0], orth_X.shape)
        Y = self._bkd.reshape(pts[1], orth_X.shape)
        return ax.contourf(
            X, Y, self._bkd.reshape(self(pts), X.shape), **kwargs
        )

    def _set_plot_3d_limits(self, ax, orth_range):
        orth_ranges = self._bkd.reshape(
            self._bkd.tile(orth_range, (3,)), (3, 2)
        )
        ranges = self.basis.mesh.trans.map_from_orthogonal(orth_ranges)
        ax.set_xlim(*ranges[0])
        ax.set_ylim(*ranges[1])
        ax.set_zlim(*ranges[2])

    def _prepare_edge_3d(
        self, ax, orth_pts_2d, X_shape, idx, inactive_orth_pt
    ):
        orth_pts = self._bkd.empty((3, orth_pts_2d.shape[1]))
        orth_pts[idx] = self._bkd.full(
            (orth_pts_2d.shape[1],), inactive_orth_pt
        )
        jdx = self._bkd.arange(3)
        jdx = self._bkd.delete(jdx, idx)
        orth_pts[jdx] = orth_pts_2d
        pts = self.basis.mesh.trans.map_from_orthogonal(orth_pts)
        vals = self._bkd.reshape(self(pts), X_shape)
        X = self._bkd.reshape(pts[0], X_shape)
        Y = self._bkd.reshape(pts[1], X_shape)
        Z = self._bkd.reshape(pts[2], X_shape)
        return X, Y, Z, vals

    def _plot_3d_internal(self, ax, npts_1d, fig):
        orth_range = self.basis.mesh.trans._orthog_ranges
        orth_X, orth_Y, orth_pts_2d = get_meshgrid_samples(
            self._bkd.tile(orth_range, (2,)), npts_1d
        )
        # ax.contourf(X1, Y1, Z1, levels=levels, zdir="z", offset=pts1[2, 0])
        mid = sum(orth_range) / 2
        edges = [
            self._prepare_edge_3d(ax, orth_pts_2d, orth_X.shape, ii, mid)
            for ii in range(3)
        ]
        vmin = self._bkd.min(self._bkd.hstack([edge[-1] for edge in edges]))
        vmax = self._bkd.max(self._bkd.hstack([edge[-1] for edge in edges]))
        zdirs = ["x", "y", "z"]
        ims = []
        for ii in range(3):
            jdx = self._bkd.arange(3)
            jdx = self._bkd.delete(jdx, ii)
            data = [None, None, None]
            data[ii] = edges[ii][3]
            data[jdx[0]] = edges[ii][jdx[0]]
            data[jdx[1]] = edges[ii][jdx[1]]
            im = ax.contourf(
                *data,
                zdir=zdirs[ii],
                offset=edges[ii][ii][0, 0],
                levels=self._bkd.linspace(vmin, vmax, 50),
            )
            ims.append(im)
        plt.colorbar(ims[0], ax=ax)
        return ims

    def _plot_3d_external(self, ax, npts_1d, fig):
        orth_range = self.basis.mesh.trans._orthog_ranges
        orth_X, orth_Y, orth_pts_2d = get_meshgrid_samples(
            self._bkd.tile(orth_range, (2,)), npts_1d, bkd=self._bkd
        )
        edges = [
            self._prepare_edge_3d(
                ax, orth_pts_2d, orth_X.shape, ii, orth_range[0]
            )
            for ii in range(3)
        ]
        edges += [
            self._prepare_edge_3d(
                ax, orth_pts_2d, orth_X.shape, ii, orth_range[1]
            )
            for ii in range(3)
        ]
        vmin = self._bkd.min(self._bkd.hstack([edge[-1] for edge in edges]))
        vmax = self._bkd.max(self._bkd.hstack([edge[-1] for edge in edges]))
        ims = []
        for edge in edges:
            vals = (edge[3] - vmin) / (vmax - vmin)
            im = ax.plot_surface(
                *edge[:3],
                facecolors=plt.cm.jet(vals),
                antialiased=False,
            )
            ims.append(im)
        if fig is None:
            return ims

        fig.subplots_adjust(right=0.85)
        ax1 = fig.add_axes([0.85, 0.10, 0.05, 0.8])
        cmap = plt.cm.jet
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        mpl.colorbar.ColorbarBase(
            ax1, cmap=cmap, norm=norm, orientation="vertical"
        )
        return ims

    def plot(self, ax, npts_1d=51, fig=None, **kwargs):
        if self.nphys_vars() == 1:
            return self._plot_1d(ax, npts_1d, **kwargs)
        if self.nphys_vars() == 2:
            return self._plot_2d(ax, npts_1d, **kwargs)
        if self.nphys_vars() == 3:
            ptype = kwargs.pop("ptype", "external")
            if ptype == "internal":
                return self._plot_3d_internal(ax, npts_1d, fig, **kwargs)
            return self._plot_3d_external(ax, npts_1d, fig, **kwargs)

    def get_plot_axis(self, figsize=(8, 6)):
        if self.nphys_vars() < 3:
            fig = plt.figure(figsize=figsize)
            return fig, fig.gca()
        fig = plt.figure(figsize=figsize)
        return fig, fig.add_subplot(111, projection="3d")


class FunctionMixin:
    """Function of physical variables, which is independent of solution"""

    def __init__(
        self,
        basis: OrthogonalCoordinateCollocationBasis,
        values_at_mesh: Array = None,
    ):
        super().__init__(basis, values_at_mesh)
        self.set_matrix_jacobian(self._initial_matrix_jacobian())

    def _initial_matrix_jacobian(self):
        return self._bkd.zeros(self.matrix_jacobian_shape())


class ScalarFunction(FunctionMixin, ScalarOperator):
    pass


class VectorFunction(FunctionMixin, VectorOperator):
    pass


class OperatorFromCallableMixin:
    def _setup(self, fun: callable):
        self._fun = fun
        self.set_values(self._fun(self.basis.mesh.mesh_pts()))
        self.set_matrix_jacobian(self._initial_matrix_jacobian())


class Operator(ABC):
    @abstractmethod
    def __call__(self, fun: MatrixOperator) -> MatrixOperator:
        raise NotImplementedError

    def __repr__(self):
        return "{0}".format(self.__class__.__name__)


class OperatorFromCallable(Operator):
    def __init__(self, op_values_fun: callable, op_jac_fun: callable):
        self._op_values_fun = op_values_fun
        self._op_jac_fun = op_jac_fun

    def _values(self, fun: MatrixOperator):
        vals = self._op_values_fun(fun.get_values())
        if vals.shape != fun.get_values().shape:
            raise RuntimeError(
                "op_values_fun returned array with the wrong shape"
            )
        return vals

    def _jacobian(self, fun: MatrixOperator):
        jac = self._op_jac_fun(fun.get_values())
        if jac.shape != fun.get_jacobian().shape:
            raise RuntimeError(
                "op_jac_fun returned array with the wrong shape"
            )
        return jac


class ScalarOperatorFromCallable(OperatorFromCallable):
    def __call__(
        self,
        fun: MatrixOperator,
    ) -> ScalarOperator:
        return ScalarOperator(
            fun.basis,
            self._values(fun),
            self._jacobian(fun),
        )


class ScalarMonomialOperator(Operator):
    def __init__(self, degree: int, coef: ScalarOperator = None):
        self._degree = degree
        if coef is not None and not isinstance(coef, ScalarOperator):
            raise ValueError("coef must be an instance ScalarOperator")
        self._coef = coef

    def __call__(self, fun: MatrixOperator) -> ScalarOperator:
        if self._degree > 0:
            vals = fun.get_values()
            if self._degree > 1:
                jac = self._degree * fun._bkd.diag(vals) ** (self._degree - 1)
            else:
                jac = self._degree * fun._bkd.diag(vals)
            poly = ScalarOperator(fun.basis, vals**self._degree, jac)
        else:
            poly = 1.0
        if self._coef is None:
            return poly
        return self._coef * poly

    def __repr__(self):
        return "{0}(degree={1}, coef={2})".format(
            self.__class__.__name__, self._degree, self._coef
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


class ScalarSolution(SolutionMixin, ScalarOperator):
    pass


class VectorSolution(SolutionMixin, VectorOperator):
    pass


class ScalarFunctionFromCallable(ScalarFunction, OperatorFromCallableMixin):
    """Scalar function that does not depend on the solution of a PDE"""

    def __init__(
        self,
        basis: OrthogonalCoordinateCollocationBasis,
        fun: callable,
    ):
        ScalarOperator.__init__(self, basis)
        self._setup(fun)


class VectorFunctionFromCallable(VectorFunction, OperatorFromCallableMixin):
    """Vector function that does not depend on the solution of a PDE"""

    def __init__(
        self,
        basis: OrthogonalCoordinateCollocationBasis,
        nrows: int,
        fun: callable,
    ):
        VectorOperator.__init__(self, basis, nrows)
        self._setup(fun)


class ScalarSolutionFromCallable(ScalarSolution, OperatorFromCallableMixin):
    """Steady scalar solution of a PDE"""

    def __init__(
        self,
        basis: OrthogonalCoordinateCollocationBasis,
        fun: callable,
    ):
        ScalarSolution.__init__(self, basis)
        self._setup(fun)


class VectorSolutionFromCallable(VectorSolution, OperatorFromCallableMixin):
    """Steady vector solution of a PDE"""

    def __init__(
        self,
        basis: OrthogonalCoordinateCollocationBasis,
        nrows: int,
        fun: callable,
    ):
        VectorSolution.__init__(self, basis, nrows)
        self._setup(fun)


class TransientOperatorMixin(ABC):
    """Operator that depends on time."""

    def __init__(self, *args, **kwargs):
        super()._init__(*args, **kwargs)

    @abstractmethod
    def _eval(self, time: float):
        raise NotImplementedError

    def set_time(self, time: float):
        self._time = time
        self.set_values(self._eval(self.basis.mesh.mesh_pts()))
        self.set_matrix_jacobian(self._initial_matrix_jacobian())

    def _check_time_set(self):
        if not hasattr(self, "_time"):
            raise ValueError(
                "Must call set_time before evaluating the function"
            )

    def get_time(self):
        self._check_time_set()
        return self._time

    def __repr__(self):
        return "{0}(\ntime={1}\n{2}\n)".format(
            self.__class__.__name__,
            self._time,
            textwrap.indent("basis=" + str(self.basis), prefix="    "),
        )


class TransientOperatorFromCallableMixin(TransientOperatorMixin):
    def _eval(self, mesh_pts):
        self._check_time_set()
        return self._fun(mesh_pts, time=self._time)


class TransientScalarFunctionFromCallable(
    TransientOperatorFromCallableMixin, ScalarFunction
):
    def __init__(
        self,
        basis: OrthogonalCoordinateCollocationBasis,
        fun: callable,
        time: float = 0,
    ):
        self._fun = fun
        ScalarOperator.__init__(self, basis)
        self.set_time(time)


class TransientScalarSolutionFromCallable(
    TransientOperatorFromCallableMixin, ScalarSolution
):
    """Transient scalar solution of a PDE"""

    def __init__(
        self,
        basis: OrthogonalCoordinateCollocationBasis,
        fun: callable,
        time: float = 0,
    ):
        self._fun = fun
        ScalarSolution.__init__(self, basis)
        self.set_time(time)


class TransientVectorFunctionFromCallable(
    TransientOperatorFromCallableMixin, VectorFunction
):
    def __init__(
        self,
        basis: OrthogonalCoordinateCollocationBasis,
        nrows: int,
        fun: callable,
        time: float = 0,
    ):
        self._fun = fun
        VectorOperator.__init__(self, basis, nrows)
        self.set_time(time)


class TransientVectorSolutionFromCallable(
    TransientOperatorFromCallableMixin, VectorSolution
):
    """Transient scalar solution of a PDE"""

    def __init__(
        self,
        basis: OrthogonalCoordinateCollocationBasis,
        nrows: int,
        fun: callable,
        time: float = 0,
    ):
        self._fun = fun
        ScalarSolution.__init__(self, basis, nrows)
        self.set_time(time)


class VectorSolutionOperator(Operator):
    def __init__(self, scalar_vec_operators: List):
        self._ops = scalar_vec_operators

    def split_into_scalar_solutions(self, sol: VectorSolution):
        vals = sol.get_values()
        if vals.shape[1] != 1:
            raise ValueError("sol must be a vector-valued funciton")
        return [
            ScalarSolution(sol.basis, vals[ii, 0])
            for ii in range(vals.shape[0])
        ]

    def split_into_immutable_functions(self, sol: VectorSolution):
        vals = sol.get_values()
        if vals.shape[1] != 1:
            raise ValueError("sol must be a vector-valued funciton")
        return [
            ScalarFunction(sol.basis, vals[ii, 0])
            for ii in range(vals.shape[0])
        ]

    def __call__(self, sol: VectorSolution):
        # split into sols with jacobian and without to easily compute
        # jacobian entries
        nop_rows = len(self._ops)
        sols = self.split_into_scalar_solutions(sol)
        funs = self.split_into_immutable_functions(sol)
        result = MatrixOperator(basis, nop_rows, nop_cols)
        values = self._bkd.zeros(result.matrix_jacobian_shape()[:-1])
        jac = self._bkd.zeros(result.matrix_jacobian_shape())
        # loop over element of output vector
        for ii in range(nop_rows):
            # compute derivative of ith output vector operator with respect to
            # each input vector element
            for jj in range(sol._nrows):
                quantities = [
                    funs[jj] if jj != kk else sols[kk]
                    for kk in range(sol._nrows)
                ]
                result_ii = op(quantities)
                values[ii, 0] = result_ii.get_values()
                # jac (npts, npts)
                jac[ii, jj, ...] = result_ii.get_jacobian()


# nabla f(u) = [D_1f_1,    0  ], d/du (nabla f(u)) = [D_1f_1'(u),     0     ]
#            = [  0   , D_2f_2]                      [   0      , D_2 f'(u) ]
# where f'(u) = d/du f(u)
def nabla(fun: MatrixOperator):
    """Gradient of a scalar valued function"""
    funvalues = fun.get_matrix_values()[0, 0]
    fun_jac = fun.get_matrix_jacobian()
    # todo need to create 3d array
    grad_vals = fun._bkd.stack(
        [
            fun.basis._deriv_mats[dd] @ funvalues
            for dd in range(fun.nphys_vars())
        ],
        axis=0,
    )[:, None, :]
    grad_jacs = fun._bkd.stack(
        [
            (fun.basis._deriv_mats[dd] @ fun_jac[0, 0])[None, :]
            for dd in range(fun.nphys_vars())
        ],
        axis=0,
    )
    return MatrixOperator(fun.basis, fun.nphys_vars(), 1, grad_vals, grad_jacs)


# div f = [D_1 f_1(u) + D_2f_2(u)],  (div f)' = [D_1f'_1(u) + D_2f'_2(u)]
def div(fun: MatrixOperator):
    """Divergence of a vector valued function."""
    if fun._ncols != 1:
        raise ValueError("Fun must be a vector valued function")
    fun_values = fun.get_values()[:, 0, ...]
    fun_jacs = fun.get_matrix_jacobian()[:, 0, ...]
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
    return MatrixOperator(
        fun.basis, 1, 1, div_vals[None, None, :], div_jac[None, None, ...]
    )


def sqmagnitude(fun: MatrixOperator):
    """Squared magnitude of a vector valued function."""
    if fun._ncols != 1:
        raise ValueError("Fun must be a vector valued function")
    fun_values = fun.get_values()[:, 0, ...]
    fun_jacs = fun.get_matrix_jacobian()[:, 0, ...]
    mag_vals = fun._bkd.sum(fun_values**2, axis=0)
    mag_jacs = fun._bkd.sum(2 * fun_jacs * fun_values[..., None], axis=0)
    return ScalarOperator(fun.basis, mag_vals, mag_jacs)


# div (nabla f)  = [D_1, D_2][D_1f_1,    0  ] = [D_1D_1f_1,    0     ]
#                            [  0   , D_2f_2] = [  0      , D_2D_2f_2]
# d/du (nabla f(u)) = [D_1D_1f_1'(u),     0        ]
#                     [   0      ,    D_2D_2 f'(u) ]
def laplace(fun: MatrixOperator):
    """Laplacian of a scalar valued function"""
    return div(nabla(fun))


def fdotgradf(fun: MatrixOperator):
    r"""(f \cdot nabla f)f of a vector-valued function f"""
    pass
