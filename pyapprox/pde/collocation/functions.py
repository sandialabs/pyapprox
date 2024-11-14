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


class ScalarOperator:
    def __init__(
        self,
        basis: OrthogonalCoordinateCollocationBasis,
        ninput_funs: int,
        values: Array = None,
        jac: Array = None,
    ):
        """
        Parameters
        ----------
        basis : OrthogonalCoordinateCollocationBasis
            The basis used to represent the functions

        ninput_funs: integer
            The number of input functions

        values : Array (nrows, ncols, nmesh_pts)
            The values of each function at the mesh points

        jac : Array (nrows, ncols, nsols, nmesh_pts, nmesh_pts)
            The jacobian of each function at the mesh points
        """
        self._ninput_funs = ninput_funs
        if not isinstance(basis, OrthogonalCoordinateCollocationBasis):
            raise ValueError(
                "basis must be an instance of "
                "OrthogonalCoordinateCollocationBasis"
            )
        self._bkd = basis._bkd
        self.basis = basis
        if values is not None:
            self.set_values(values)
        if jac is not None:
            self.set_jacobian(jac)

    def ninput_funs(self) -> int:
        return self._ninput_funs

    def nphys_vars(self) -> int:
        return self.basis.nphys_vars()

    def nmesh_pts(self) -> int:
        return self.basis.mesh.nmesh_pts()

    def set_values(self, values: Array):
        if values.shape != (self.nmesh_pts(),):
            raise ValueError(
                "values.shape {0} should be {1}".format(
                    values.shape, (self.nmesh_pts(),)
                )
            )
        self._values = values

    def get_values(self) -> Array:
        if not hasattr(self, "_values"):
            print(self)
            raise RuntimeError("must first call set_values()")
        return self._values

    def get_jacobian(self) -> Array:
        if not hasattr(self, "_jac"):
            raise RuntimeError("must call set_jacobian()")
        return self._jac

    def set_jacobian(self, jac: Array):
        jac_shape = (self.nmesh_pts(), self.nmesh_pts() * self.ninput_funs())
        if jac.shape != jac_shape:
            raise ValueError(
                "values.shape {0} should be {1}".format(jac.shape, jac_shape)
            )
        self._jac = jac

    def get_flattened_values(self):
        return self.get_values()

    def get_flattened_jacobian(self):
        return self.get_jacobian()

    def _add_functions(
        self, other: Union["ScalarOperator", float]
    ) -> "ScalarOperator":
        if isinstance(other, float):
            return ScalarOperator(
                self.basis,
                self.ninput_funs(),
                self.get_values() + other,
                self.get_jacobian(),
            )

        return ScalarOperator(
            self.basis,
            self.ninput_funs(),
            self.get_values() + other.get_values(),
            self.get_jacobian() + other.get_jacobian(),
        )

    def __add__(
        self, other: Union["ScalarOperator", float]
    ) -> "ScalarOperator":
        if not isinstance(other, (ScalarOperator, float)):
            raise ValueError(f"cannot add {type(other)} to a ScalarOperator")
        return self._add_functions(other)

    def __radd__(
        self, other: Union["ScalarOperator", float]
    ) -> "ScalarOperator":
        if isinstance(other, int):
            # allow use of sum(List[ops])
            other = float(other)
        if not isinstance(other, (ScalarOperator, float)):
            raise ValueError(f"cannot add {type(other)} to a ScalarOperator")
        return self._add_functions(other)

    def _subtract_functions(self, fun1, fun2) -> "ScalarOperator":
        return ScalarOperator(
            fun1.basis,
            self.ninput_funs(),
            fun1.get_values() - fun2.get_values(),
            fun1.get_jacobian() - fun2.get_jacobian(),
        )

    def __sub__(
        self, other: Union["ScalarOperator", float]
    ) -> "ScalarOperator":
        if isinstance(other, float):
            return ScalarOperator(
                self.basis,
                self.ninput_funs(),
                self.get_values() - other,
                self.get_jacobian(),
            )
        if not isinstance(other, ScalarOperator):
            raise ValueError(
                f"cannot subtract {type(other)} from a ScalarOperator"
            )
        return self._subtract_functions(self, other)

    def __rsub__(
        self, other: Union["ScalarOperator", float]
    ) -> "ScalarOperator":
        if isinstance(other, float):
            return ScalarOperator(
                self.basis,
                self.ninput_funs(),
                other - self.get_values(),
                -self.get_jacobian(),
            )
        if not isinstance(other, ScalarOperator):
            raise ValueError(
                f"cannot subtract a ScalarOperator from {type(other)}"
            )
        return self._subtract_functions(other, self)

    def _multiply_functions(
        self, other: Union["ScalarOperator", float]
    ) -> "ScalarOperator":
        if not isinstance(other, (ScalarOperator, float)):
            raise ValueError(
                f"cannot multiply ScalarOperator by {type(other)}"
            )

        if isinstance(other, float):
            return ScalarOperator(
                self.basis,
                self.ninput_funs(),
                self.get_values() * other,
                self.get_jacobian() * other,
            )

        values = self.get_values() * other.get_values()
        # use product rule
        jac = (
            other.get_jacobian() * self.get_values()[..., None]
            + other.get_values()[..., None] * self.get_jacobian()
        )
        return ScalarOperator(other.basis, self.ninput_funs(), values, jac)

    def __mul__(
        self, other: Union["ScalarOperator", "MatrixOperator", float]
    ) -> "ScalarOperator":
        if isinstance(other, MatrixOperator):
            # use __mul__ defined in MatrixOperator
            return other * self
        return self._multiply_functions(other)

    def __rmul__(
        self, other: Union["ScalarOperator", "MatrixOperator", float]
    ) -> "ScalarOperator":
        if isinstance(other, MatrixOperator):
            # use __mul__ defined in MatrixOperator
            return other * self
        return self._multiply_functions(other)

    def __neg__(self):
        return -1.0 * self

    def __pow__(self, other: int) -> "ScalarOperator":
        if (
            not (isinstance(other, int) or isinstance(other, float))
            or other == 0
        ):
            raise ValueError(
                "power must be an integer or float and must not be zero"
            )
        return ScalarOperator(
            self.basis,
            self.ninput_funs(),
            self.get_values() ** other,
            other
            * self.get_jacobian()
            * self.get_values()[..., None] ** (other - 1),
        )

    def _divide_functions(self, fun1, fun2) -> "ScalarOperator":
        if isinstance(fun2, float):
            if fun2 == 0:
                raise ValueError("Cannot divide by zero")
            return ScalarOperator(
                fun1.basis,
                self.ninput_funs(),
                fun1.get_values() / fun2,
                fun1.get_jacobian() / fun2,
            )

        if isinstance(fun1, float):
            if self._bkd.any(fun2.get_values() == 0.0):
                raise ValueError("Cannot divide by zero")
            return ScalarOperator(
                fun2.basis,
                self.ninput_funs(),
                fun1 / fun2.get_values(),
                -fun1
                * fun2.get_jacobian()
                / (fun2.get_values()[..., None] ** 2),
            )

        values = fun1.get_values() / fun2.get_values()
        # use quotient rule
        jac = (
            fun2.get_values()[..., None] * fun1.get_jacobian()
            - fun2.get_jacobian() * fun1.get_values()[..., None]
        ) / fun2.get_values()[..., None] ** 2
        return ScalarOperator(fun1.basis, self.ninput_funs(), values, jac)

    def __truediv__(
        self, other: Union["ScalarOperator", float]
    ) -> "ScalarOperator":
        if not isinstance(other, (float, ScalarOperator)):
            raise ValueError(f"Cannot divde function by {type(other)}")
        return self._divide_functions(self, other)

    def __rtruediv__(
        self, other: Union["ScalarOperator", float]
    ) -> "ScalarOperator":
        if not isinstance(other, (float, ScalarOperator)):
            raise ValueError(f"Cannot divide {type(other)} by function")
        return self._divide_functions(other, self)

    def deriv(self, physvar_id: int) -> "ScalarOperator":
        return ScalarOperator(
            self.basis,
            self.ninput_funs(),
            self.basis._deriv_mats[physvar_id] @ self.get_values(),
            self.basis._deriv_mats[physvar_id] @ self.get_jacobian(),
        )

    def __call__(self, eval_samples: Array) -> Array:
        values = self.get_values()[:, None]
        return self.basis.interpolate(values, eval_samples)[:, 0]

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

    def __repr__(self):
        return "{0}(\n{1}\n)".format(
            self.__class__.__name__,
            textwrap.indent("basis=" + str(self.basis), prefix="    "),
        )

    def values_shape(self) -> tuple:
        return (self.basis.mesh.nmesh_pts(),)


class ScalarFunction(ScalarOperator):
    def __init__(
        self,
        basis: OrthogonalCoordinateCollocationBasis,
        values: Array = None,
        ninput_funs: int = 1,
    ):
        super().__init__(basis, ninput_funs, values)

    def set_values(self, values: Array):
        super().set_values(values)
        zero_jac = self.basis._bkd.zeros(
            (self.basis.mesh.nmesh_pts(), self.ninput_funs()*self.basis.mesh.nmesh_pts())
        )
        super().set_jacobian(zero_jac)

    def set_jacobian(self, jac: Array):
        raise NotImplementedError(
            "do not call set_jacobian because it is called by set_values"
        )


class ScalarSolution(ScalarOperator):
    def __init__(
        self,
        basis: OrthogonalCoordinateCollocationBasis,
        values: Array = None,
        ninput_funs: int = 1,
    ):
        super().__init__(basis, ninput_funs, values)

    def set_values(self, values: Array):
        super().set_values(values)
        ident_jac = self.basis._bkd.eye(self.basis.mesh.nmesh_pts())
        super().set_jacobian(ident_jac)

    def set_jacobian(self, jac: Array):
        raise NotImplementedError(
            "do not call set_jacobian because it is called by set_values"
        )


class ScalarOperatorFromCallableMixin:
    def __init__(
        self,
        basis: OrthogonalCoordinateCollocationBasis,
        fun: callable,
    ):
        super().__init__(basis, self._get_values(basis, fun))

    def _get_values(
        self, basis: OrthogonalCoordinateCollocationBasis, fun: callable
    ):
        self._fun = fun
        return self._fun(basis.mesh.mesh_pts())


class ScalarFunctionFromCallable(
    ScalarOperatorFromCallableMixin, ScalarFunction
):
    pass


class ScalarSolutionFromCallable(
    ScalarOperatorFromCallableMixin, ScalarSolution
):
    pass


class ScalarOperatorOperation(ABC):
    """Shortcut to create complex operator operation on ScalarOperators"""

    @abstractmethod
    def __call__(self, fun: ScalarOperator) -> ScalarOperator:
        raise NotImplementedError

    def __repr__(self):
        return "{0}".format(self.__class__.__name__)


class ScalarMonomialOperator(ScalarOperatorOperation):
    def __init__(self, degree: int, coef: ScalarOperator = None):
        self._degree = degree
        if coef is not None and not isinstance(coef, ScalarOperator):
            raise ValueError("coef must be an instance ScalarOperator")
        self._coef = coef

    def __call__(self, fun: ScalarOperator) -> ScalarOperator:
        if not isinstance(fun, ScalarOperator):
            raise ValueError("Can only be used for scalar solutions")
        if self._degree > 0:
            poly = fun**self._degree
        else:
            poly = 1.0
        if self._coef is None:
            return poly
        return self._coef * poly

    def __repr__(self):
        return "{0}(degree={1}, coef={2})".format(
            self.__class__.__name__, self._degree, self._coef
        )


class TransientOperatorMixin:
    """Operator that depends on time."""

    def __init__(self, *args, **kwargs):
        self._time = None
        super().__init__(*args, **kwargs)

    @abstractmethod
    def _eval(self, time: float):
        raise NotImplementedError

    def set_time(self, time: float):
        self._time = time
        self.set_values(self._eval(self.basis.mesh.mesh_pts()).T)
        # It is assumed set_jacobian is set in set_values, e.g.
        # as done by Function and Solution

    def _check_time_set(self):
        if not hasattr(self, "_time"):
            raise ValueError(
                "Must call set_time before evaluating the function"
            )

    def get_time(self):
        self._check_time_set()
        return self._time

    def __repr__(self):
        return "{0}(\n{1},\n{2}\n)".format(
            self.__class__.__name__,
            textwrap.indent(f"time={self._time}", prefix="    "),
            textwrap.indent("basis=" + str(self.basis), prefix="    "),
        )


class TransientScalarFunction(TransientOperatorMixin, ScalarFunction):
    pass


class TransientOperatorFromCallableMixin(TransientOperatorMixin):
    def __init__(
        self,
        basis: OrthogonalCoordinateCollocationBasis,
        fun: callable,
        ninput_funs: int = 1,
    ):
        self._fun = fun
        super().__init__(basis, ninput_funs=ninput_funs)

    def _eval(self, mesh_pts):
        self._check_time_set()
        print(self._fun(mesh_pts, time=self._time).shape)
        return self._fun(mesh_pts, time=self._time)


class TransientScalarFunctionFromCallable(
    TransientOperatorFromCallableMixin, TransientScalarFunction
):
    pass


class TransientScalarSolution(TransientOperatorMixin, ScalarSolution):
    pass


class TransientScalarSolutionFromCallable(
    TransientOperatorFromCallableMixin, TransientScalarSolution
):
    pass


class VectorSolutionComponent(ScalarOperator):
    def __init__(
        self,
        basis: OrthogonalCoordinateCollocationBasis,
        ninput_funs: int,
        input_id: int,
        values: Array,
    ):
        if input_id >= ninput_funs:
            raise ValueError("input_id must be smaller than ninput_funs")
        self._input_id = input_id
        zero_jac = basis._bkd.zeros(
            (basis.mesh.nmesh_pts(), basis.mesh.nmesh_pts())
        )
        jac = [None for ii in range(ninput_funs)]
        jac[input_id] = basis._bkd.eye(basis.mesh.nmesh_pts())
        for ii in range(ninput_funs):
            if ii == input_id:
                jac[input_id] = basis._bkd.eye(basis.mesh.nmesh_pts())
            else:
                jac[ii] = zero_jac
        jac = basis._bkd.hstack(jac)
        super().__init__(basis, ninput_funs, values, jac)


class MatrixOperator:
    def __init__(
        self,
        basis: OrthogonalCoordinateCollocationBasis,
        ninput_funs: int,
        nrows: int,
        ncols: int,
    ):
        self.basis = basis
        self._bkd = basis._bkd
        self._ninput_funs = ninput_funs
        self._nrows = nrows
        self._ncols = ncols

    def ninput_funs(self) -> int:
        return self._ninput_funs

    def _check_components(self, components):
        # todo check all components have the same ninput_funs()
        if len(components) != self.nrows():
            print(components)
            raise ValueError(
                "len(components)={0} must equal self.nrows()={1}".format(
                    len(components), self.nrows()
                )
            )
        for ii in range(self.nrows()):
            if len(components[ii]) != self.ncols():
                raise ValueError(
                    f"len(components[{ii}]) must equal self.ncols()"
                )
            for comp in components[ii]:
                if not isinstance(comp, ScalarOperator):
                    raise ValueError(
                        f"component {comp} is not an instance "
                        "of ScalarOperator"
                    )

    def set_components(self, components: List[ScalarOperator]):
        self._check_components(components)
        self._components = components
        self._bkd = components[0][0]._bkd

    def get_values(self) -> Array:
        if not hasattr(self, "_components"):
            raise RuntimeError("must call set_commponents()")
        rows = []
        for ii in range(self.nrows()):
            row = []
            for jj in range(self.ncols()):
                row.append(self._components[ii][jj].get_values())
            rows.append(self._bkd.stack(row, axis=0))
        return self._bkd.stack(rows, axis=0)

    def get_jacobian(self) -> Array:
        rows = []
        for ii in range(self.nrows()):
            row = []
            for jj in range(self.ncols()):
                row.append(self._components[ii][jj].get_jacobian())
            rows.append(self._bkd.stack(row, axis=0))
        return self._bkd.stack(rows, axis=0)

    def get_flattened_values(self):
        return self._bkd.flatten(self.get_values())

    def get_flattened_jacobian(self):
        jac = self.get_jacobian()
        return self._bkd.reshape(
            jac, ((jac.shape[0] * jac.shape[2], jac.shape[3]))
        )

    def nrows(self) -> int:
        return self._nrows

    def ncols(self) -> int:
        return self._ncols

    def nphys_vars(self) -> int:
        return self._components[0][0].nphys_vars()

    def __repr__(self) -> str:
        return "{0}(nrows={1}, ncols={2})".format(
            self.__class__.__name__, self.nrows(), self.ncols()
        )

    def sqnorm(self) -> ScalarOperator:
        """Squared Frobenius norm of a matrix valued function."""
        ops = []
        for ii in range(self.nrows()):
            for jj in range(self.ncols()):
                ops.append(self._components[ii][jj] ** 2)
        return sum(ops)

    def _multiply_functions(
        self, other: Union["MatrixOperator", ScalarOperator, float]
    ) -> "MatrixOperator":
        if not hasattr(self, "_components"):
            raise RuntimeError("must call set_commponents()")

        if not isinstance(other, (MatrixOperator, ScalarOperator, float)):
            raise ValueError(
                f"cannot multiply MatrixOperator by {type(other)}"
            )

        if isinstance(other, MatrixOperator):
            op = MatrixOperator(
                self.basis, self.ninput_funs(), self.nrows(), self.ncols()
            )
            components = [
                [
                    other._components[ii][jj] * self._components[ii][jj]
                    for jj in range(self.ncols())
                ]
                for ii in range(self.nrows())
            ]
            op.set_components(components)
            return op

        op = MatrixOperator(
            self.basis, self.ninput_funs(), self.nrows(), self.ncols()
        )
        components = [
            [self._components[ii][jj] * other for jj in range(self.ncols())]
            for ii in range(self.nrows())
        ]
        op.set_components(components)
        return op

    def __mul__(
        self, other: Union["MatrixOperator", ScalarOperator, float]
    ) -> "MatrixOperator":
        return self._multiply_functions(other)

    def __rmul__(
        self, other: Union["MatrixOperator", ScalarOperator, float]
    ) -> "MatrixOperator":
        return self._multiply_functions(other)

    def __neg__(self):
        return -1.0 * self

    def _add_functions(
        self, other: Union["MatrixOperator", ScalarOperator, float]
    ) -> "MatrixOperator":
        if not hasattr(self, "_components"):
            raise RuntimeError("must call set_commponents()")

        if not isinstance(other, (MatrixOperator, ScalarOperator, float)):
            raise ValueError(f"cannot add {type(other)} to a ScalarOperator")

        if isinstance(other, MatrixOperator):
            op = MatrixOperator(
                self.basis, self.ninput_funs(), self.nrows(), self.ncols()
            )
            components = [
                [
                    other._components[ii][jj] + self._components[ii][jj]
                    for jj in range(self.ncols())
                ]
                for ii in range(self.nrows())
            ]
            op.set_components(components)
            return op

        op = MatrixOperator(
            self.basis, self.ninput_funs(), self.nrows(), self.ncols()
        )
        components = [
            [self._components[ii][jj] + other for jj in range(self.ncols())]
            for ii in range(self.nrows())
        ]
        op.set_components(components)
        return op

    def __add__(
        self, other: Union["MatrixOperator", ScalarOperator, float]
    ) -> "MatrixOperator":
        return self._add_functions(other)

    def __radd__(
        self, other: Union["MatrixOperator", ScalarOperator, float]
    ) -> "MatrixOperator":
        return self._add_functions(other)

    def __sub__(
        self, other: Union["MatrixOperator", ScalarOperator, float]
    ) -> "MatrixOperator":
        if not hasattr(self, "_components"):
            raise RuntimeError("must call set_commponents()")

        if not isinstance(other, (MatrixOperator, ScalarOperator, float)):
            raise ValueError(
                f"cannot subtract {type(other)} from a MatrixOperator"
            )
        if isinstance(other, MatrixOperator):
            op = MatrixOperator(
                self.basis, self.ninput_funs(), self.nrows(), self.ncols()
            )
            components = [
                [
                    self._components[ii][jj] - other._components[ii][jj]
                    for jj in range(self.ncols())
                ]
                for ii in range(self.nrows())
            ]
            op.set_components(components)
            return op

        op = MatrixOperator(
            self.basis, self.ninput_funs(), self.nrows(), self.ncols()
        )
        components = [
            [self._components[ii][jj] - other for jj in range(self.ncols())]
            for ii in range(self.nrows())
        ]
        op.set_components(components)
        return op

    def __rsub__(
        self, other: Union["MatrixOperator", ScalarOperator, float]
    ) -> "MatrixOperator":
        if not hasattr(self, "_components"):
            raise RuntimeError("must call set_commponents()")

        if not isinstance(other, (MatrixOperator, ScalarOperator, float)):
            raise ValueError(
                f"cannot subtract MatrixOperator from {type(other)}"
            )

        if isinstance(other, MatrixOperator):
            op = MatrixOperator(
                self.basis, self.ninput_funs(), self.nrows(), self.ncols()
            )
            components = [
                [
                    other._components[ii][jj] - self._components[ii][jj]
                    for jj in range(self.ncols())
                ]
                for ii in range(self.nrows())
            ]
            op.set_components(components)
            return op

        op = MatrixOperator(
            self.basis, self.ninput_funs(), self.nrows(), self.ncols()
        )
        components = [
            [other - self._components[ii][jj] for jj in range(self.ncols())]
            for ii in range(self.nrows())
        ]
        op.set_components(components)
        return op

    def create_scalar_operator_from_values(
        self, values: Array, row: int, col: int
    ) -> ScalarOperator:
        return ScalarOperator(self.basis, self.ninput_funs(), values)

    def _set_matrix_components(self, components: List[ScalarOperator]):
        MatrixOperator.set_components(self, components)

    def set_flattened_values(self, values: Array):
        values_shape = (
            self.nrows() * self.ncols() * self.basis.mesh.nmesh_pts(),
        )
        if values.shape != values_shape:
            raise ValueError(
                f"values.shape {values.shape} must be {values_shape}"
            )
        reshaped_values = self._bkd.reshape(
            values, (self.nrows(), self.ncols(), self.basis.mesh.nmesh_pts())
        )
        components = []
        for ii in range(self.nrows()):
            row = []
            for jj in range(self.ncols()):
                row.append(
                    self.create_scalar_operator_from_values(
                        reshaped_values[ii, jj], ii, jj
                    )
                )
            components.append(row)
        self._set_matrix_components(components)

    def set_values(self, values: Array):
        if values.shape != self.values_shape():
            raise ValueError("values has the wrong shape")
        self.set_flattened_values(self._bkd.flatten(values))

    def __call__(self, eval_samples: Array) -> Array:
        rows = []
        for ii in range(self.nrows()):
            row = []
            for jj in range(self.ncols()):
                row.append(
                    self._components[ii][jj](eval_samples)
                )
            rows.append(self._bkd.stack(row, axis=0))
        # todo consider passing stacked values of each component
        # to interpolate just once and then separting
        return self._bkd.stack(rows, axis=0)

    def get_components(self) -> List[VectorSolutionComponent]:
        if not hasattr(self, "_components"):
            raise RuntimeError("must call set_commponents()")
        return self._components

    def values_shape(self) -> tuple:
        return (
            self.nrows(),
            self.ncols(),
            self.basis.mesh.nmesh_pts(),
        )


class VectorOperator(MatrixOperator):
    def __init__(
        self,
        basis: OrthogonalCoordinateCollocationBasis,
        ninput_funs: int,
        nrows: int,
    ):
        super().__init__(basis, ninput_funs, nrows, 1)

    def set_components(self, components: List[ScalarOperator]):
        super().set_components([[comp] for comp in components])

    def get_values(self) -> Array:
        return super().get_values()[:, 0]

    def __repr__(self) -> str:
        return "{0}(nrows={1})".format(self.__class__.__name__, self.nrows())

    def set_values(self, values: Array):
        values_shape = self.values_shape()
        if values.shape != values_shape:
            raise ValueError(
                "values shape {0} must be {1}".format(
                    values.shape, values_shape
                )
            )
        self.set_flattened_values(self._bkd.flatten(values))

    def __call__(self, eval_samples: Array) -> Array:
        return super().__call__(eval_samples)[:, 0]

    def get_components(self) -> List[VectorSolutionComponent]:
        if not hasattr(self, "_components"):
            raise RuntimeError("must call set_commponents()")
        return [row[0] for row in self._components]

    def values_shape(self) -> tuple:
        return (self.nrows(), self.basis.mesh.nmesh_pts(),)


class VectorSolution(VectorOperator):
    def create_scalar_operator_from_values(
        self, values: Array, row: int, col: int
    ) -> VectorSolutionComponent:
        return VectorSolutionComponent(
            self.basis, self.ninput_funs(), row, values
        )

    def _set_matrix_components(
        self, components: List[VectorSolutionComponent]
    ):
        for row in components:
            if not isinstance(row[0], VectorSolutionComponent):
                raise ValueError(
                    "component must be an instance of VectorSolutionComponent"
                )
        MatrixOperator.set_components(self, components)


class VectorFunction(VectorOperator):
    def set_components(self, components: List[ScalarFunction]):
        for comp in components:
            if not isinstance(comp, ScalarFunction):
                raise ValueError("component must be an instance of Function")
        super().set_components([comp for comp in components])

    def set_values(self, values: Array):
        super().set_values(values)


class TransientVectorSolution(TransientOperatorMixin, VectorSolution):
    pass


class TransientVectorFunction(TransientOperatorMixin, VectorFunction):
    def create_scalar_operator_from_values(
        self, values: Array, row: int, col: int
    ) -> TransientScalarFunction:
        return TransientScalarFunction(
            self.basis, values, ninput_funs=self.ninput_funs()
        )


class VectorOperatorFromCallableMixin:
    def __init__(
        self,
        basis: OrthogonalCoordinateCollocationBasis,
        ninput_funs: int,
        nrows: int,
        fun: callable,
    ):
        super().__init__(basis, ninput_funs, nrows)
        self._fun = fun
        values = self._fun(basis.mesh.mesh_pts())
        if values.shape[1] != nrows:
            raise ValueError("values returned by fun has the wrong shape")
        self.set_values(values.T)


class VectorSolutionFromCallable(
    VectorOperatorFromCallableMixin, VectorSolution
):
    pass


class VectorFunctionFromCallable(
    VectorOperatorFromCallableMixin, VectorFunction
):
    def create_scalar_operator_from_values(
        self, values: Array, row: int, col: int
    ) -> ScalarFunction:
        return ScalarFunction(self.basis, values, ninput_funs=self.ninput_funs())


class TransientVectorOperatorFromCallableMixin(TransientOperatorMixin):
    def __init__(
        self,
        basis: OrthogonalCoordinateCollocationBasis,
        ninput_funs: int,
        nrows: int,
        fun: callable,
    ):
        self._fun = fun
        super().__init__(basis, ninput_funs, nrows)

    def _eval(self, mesh_pts):
        self._check_time_set()
        return self._fun(mesh_pts, time=self._time)


class TransientVectorSolutionFromCallable(
    TransientVectorOperatorFromCallableMixin, TransientVectorSolution
):
    # if RuntimeError: must call set_commponents() is raised then
    # set_time was likely not called.
    pass


class TransientVectorFunctionFromCallable(
    TransientVectorOperatorFromCallableMixin, TransientVectorFunction
):
    # if RuntimeError: must call set_commponents() is raised then
    # set_time was likely not called.
    def create_scalar_operator_from_values(
        self, values: Array, row: int, col: int
    ) -> TransientScalarFunctionFromCallable:
        fun = TransientScalarFunctionFromCallable(
            self.basis, lambda xx, time: self._fun(xx, time)[:, row],
            ninput_funs = self.ninput_funs()
        )
        fun.set_time(self._time)
        return fun


def nabla(op: ScalarOperator) -> VectorOperator:
    """Gradient of a scalar valued function"""
    vec_op = VectorOperator(op.basis, op.ninput_funs(), op.nphys_vars())
    ops = [op.deriv(dd) for dd in range(op.nphys_vars())]
    vec_op.set_components(ops)
    return vec_op


def div(mat_op: MatrixOperator):
    """
    Divergence of a Matrix valued function.

    The divergence opeerator is applied to each column of the matrix
    independently.
    """
    if mat_op.nrows() != mat_op.nphys_vars():
        raise ValueError("vec_op.nrows() does not match nphys_vars")
    ops = []
    for jj in range(mat_op.ncols()):
        ops.append(
            sum(
                [
                    mat_op._components[dd][jj].deriv(dd)
                    for dd in range(mat_op.nphys_vars())
                ]
            )
        )
    if len(ops) == 1:
        return ops[0]

    op = VectorOperator(mat_op.basis, mat_op.ninput_funs(), len(ops))
    op.set_components(ops)
    return op


# class TransientOperatorMixin(ABC):
#     """Operator that depends on time."""

#     def __init__(self, *args, **kwargs):
#         super()._init__(*args, **kwargs)

#     @abstractmethod
#     def _eval(self, time: float):
#         raise NotImplementedError

#     def set_time(self, time: float):
#         self._time = time
#         self.set_values(self._eval(self.basis.mesh.mesh_pts()))
#         self.set_matrix_jacobian(self._initial_matrix_jacobian())

#     def _check_time_set(self):
#         if not hasattr(self, "_time"):
#             raise ValueError(
#                 "Must call set_time before evaluating the function"
#             )

#     def get_time(self):
#         self._check_time_set()
#         return self._time

#     def __repr__(self):
#         return "{0}(\ntime={1}\n{2}\n)".format(
#             self.__class__.__name__,
#             self._time,
#             textwrap.indent("basis=" + str(self.basis), prefix="    "),
#         )


# class TransientOperatorFromCallableMixin(TransientOperatorMixin):
#     def _eval(self, mesh_pts):
#         self._check_time_set()
#         return self._fun(mesh_pts, time=self._time)


# class TransientScalarFunctionFromCallable(
#     TransientOperatorFromCallableMixin, ScalarFunction
# ):
#     def __init__(
#         self,
#         basis: OrthogonalCoordinateCollocationBasis,
#         fun: callable,
#         time: float = 0,
#     ):
#         self._fun = fun
#         ScalarOperator.__init__(self, basis)
#         self.set_time(time)


# class TransientScalarSolutionFromCallable(
#     TransientOperatorFromCallableMixin, ScalarSolution
# ):
#     """Transient scalar solution of a PDE"""

#     def __init__(
#         self,
#         basis: OrthogonalCoordinateCollocationBasis,
#         fun: callable,
#         time: float = 0,
#     ):
#         self._fun = fun
#         ScalarSolution.__init__(self, basis)
#         self.set_time(time)


# class TransientVectorFunctionFromCallable(
#     TransientOperatorFromCallableMixin, VectorFunction
# ):
#     def __init__(
#         self,
#         basis: OrthogonalCoordinateCollocationBasis,
#         nrows: int,
#         fun: callable,
#         time: float = 0,
#     ):
#         self._fun = fun
#         VectorOperator.__init__(self, basis, nrows)
#         self.set_time(time)


# class TransientVectorSolutionFromCallable(
#     TransientOperatorFromCallableMixin, VectorSolution
# ):
#     """Transient scalar solution of a PDE"""

#     def __init__(
#         self,
#         basis: OrthogonalCoordinateCollocationBasis,
#         nrows: int,
#         fun: callable,
#         time: float = 0,
#     ):
#         self._fun = fun
#         ScalarSolution.__init__(self, basis, nrows)
#         self.set_time(time)


# # class VectorSolutionOperator(Operator):
# #     def __init__(self, scalar_vec_operators: List):
# #         self._ops = scalar_vec_operators

# #     def split_into_scalar_solutions(self, sol: VectorSolution):
# #         vals = sol.get_values()
# #         if vals.shape[1] != 1:
# #             raise ValueError("sol must be a vector-valued funciton")
# #         return [
# #             ScalarSolution(sol.basis, vals[ii, 0])
# #             for ii in range(vals.shape[0])
# #         ]

# #     def split_into_immutable_functions(self, sol: VectorSolution):
# #         vals = sol.get_values()
# #         if vals.shape[1] != 1:
# #             raise ValueError("sol must be a vector-valued funciton")
# #         return [
# #             ScalarFunction(sol.basis, vals[ii, 0])
# #             for ii in range(vals.shape[0])
# #         ]

# #     def __call__(self, sol: VectorSolution):
# #         # split into sols with jacobian and without to easily compute
# #         # jacobian entries
# #         nop_rows = len(self._ops)
# #         sols = self.split_into_scalar_solutions(sol)
# #         funs = self.split_into_immutable_functions(sol)
# #         result = MatrixOperator(basis, nop_rows, nop_cols)
# #         values = self._bkd.zeros(result.matrix_jacobian_shape()[:-1])
# #         jac = self._bkd.zeros(result.matrix_jacobian_shape())
# #         # loop over element of output vector
# #         for ii in range(nop_rows):
# #             # compute derivative of ith output vector operator with respect to
# #             # each input vector element
# #             for jj in range(sol._nrows):
# #                 quantities = [
# #                     funs[jj] if jj != kk else sols[kk]
# #                     for kk in range(sol._nrows)
# #                 ]
# #                 result_ii = op(quantities)
# #                 values[ii, 0] = result_ii.get_values()
# #                 # jac (npts, npts)
# #                 jac[ii, jj, ...] = result_ii.get_jacobian()


# # nabla f(u) = [D_1f_1,    0  ], d/du (nabla f(u)) = [D_1f_1'(u),     0     ]
# #            = [  0   , D_2f_2]                      [   0      , D_2 f'(u) ]
# # where f'(u) = d/du f(u)
# def nabla(fun: MatrixOperator):
#     """Gradient of a scalar valued function"""
#     funvalues = fun.get_matrix_values()[0, 0]
#     fun_jac = fun.get_matrix_jacobian()
#     # todo need to create 3d array
#     grad_vals = fun._bkd.stack(
#         [
#             fun.basis._deriv_mats[dd] @ funvalues
#             for dd in range(fun.nphys_vars())
#         ],
#         axis=0,
#     )[:, None, :]
#     grad_jacs = fun._bkd.stack(
#         [
#             (fun.basis._deriv_mats[dd] @ fun_jac[0, 0])[None, :]
#             for dd in range(fun.nphys_vars())
#         ],
#         axis=0,
#     )
#     return MatrixOperator(
#         fun.basis, fun.nphys_vars(), 1, fun._nsols, grad_vals, grad_jacs
#     )


# # div f = [D_1 f_1(u) + D_2f_2(u)],  (div f)' = [D_1f'_1(u) + D_2f'_2(u)]
# def div(fun: MatrixOperator):
#     """Divergence of a vector valued function."""
#     if fun._ncols != 1:
#         raise ValueError("Fun must be a vector valued function")
#     fun_values = fun.get_values()[:, 0, ...]
#     fun_jacs = fun.get_matrix_jacobian()[:, 0, ...]
#     dmats = fun._bkd.stack(fun.basis._deriv_mats, axis=0)
#     # dmats: (nrows, n, n)
#     # fun_values : (nrows, n)
#     div_vals = fun._bkd.sum(
#         fun._bkd.einsum("ijk,ik->ij", dmats, fun_values), axis=0
#     )
#     # dmats: (nrows, n, n)
#     # fun_jacs : (nrows, n, n)
#     div_jac = fun._bkd.sum(
#         fun._bkd.einsum("ijk,ikm->ijm", dmats, fun_jacs), axis=0
#     )
#     return MatrixOperator(
#         fun.basis, 1, 1, div_vals[None, None, :], div_jac[None, None, ...]
#     )


# def sqmagnitude(fun: MatrixOperator):
#     """Squared magnitude of a vector valued function."""
#     if fun._ncols != 1:
#         raise ValueError("Fun must be a vector valued function")
#     fun_values = fun.get_values()[:, 0, ...]
#     fun_jacs = fun.get_matrix_jacobian()[:, 0, ...]
#     mag_vals = fun._bkd.sum(fun_values**2, axis=0)
#     mag_jacs = fun._bkd.sum(2 * fun_jacs * fun_values[..., None], axis=0)
#     return ScalarOperator(fun.basis, mag_vals, mag_jacs)


# # div (nabla f)  = [D_1, D_2][D_1f_1,    0  ] = [D_1D_1f_1,    0     ]
# #                            [  0   , D_2f_2] = [  0      , D_2D_2f_2]
# # d/du (nabla f(u)) = [D_1D_1f_1'(u),     0        ]
# #                     [   0      ,    D_2D_2 f'(u) ]
# def laplace(fun: MatrixOperator):
#     """Laplacian of a scalar valued function"""
#     return div(nabla(fun))


# def fdotgradf(fun: MatrixOperator):
#     r"""(f \cdot nabla f)f of a vector-valued function f"""
#     pass
