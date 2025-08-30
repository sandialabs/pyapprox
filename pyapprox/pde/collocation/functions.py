import textwrap
from abc import ABC, abstractmethod
from typing import Union, List

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

from pyapprox.util.backends.template import Array
from pyapprox.util.visualization import get_meshgrid_samples
from pyapprox.pde.collocation.basis import OrthogonalCoordinateCollocationBasis
from pyapprox.surrogates.affine.kle import (
    MeshKLE,
    PeriodicReiszGaussianRandomField,
)
from pyapprox.pde.collocation.sparsejac import (
    SparseJacobian,
    ZeroJac,
    DiagJac,
)

# Note from typing documentation
# When a type hint contains names that have not been defined yet, that
# definition may be expressed as a string literal, to be resolved later.
# The string literal should contain a valid Python expression and it
# should evaluate without errors once the module has been fully loaded.


class JacDepType(ABC):
    # def __init__(self, shape: tuple)

    def __repr__(self):
        return "{0}".format(self.__class__.__name__)

    @abstractmethod
    def new_type(self, other: "JacDepType"):
        raise NotImplementedError


class DenseJacDep(JacDepType):
    def new_type(self, other: JacDepType):
        return DenseJacDep()


class DiagJacDep(JacDepType):
    def new_type(self, other: JacDepType):
        if isinstance(other, DenseJacDep):
            return DenseJacDep
        return DiagJacDep()


class ZeroJacDep(JacDepType):
    def new_type(self, other: JacDepType):
        if isinstance(other, ZeroJacDep):
            return ZeroJacDep()
        return other.jac_type()


class ScalarOperator:
    def __init__(
        self,
        basis: OrthogonalCoordinateCollocationBasis,
        ninput_funs: int,
        values: Array = None,
        jac: SparseJacobian = None,
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

        jac : SparseJacobian
            The jacobian of each function at the mesh points
        """
        self._ninput_funs = ninput_funs
        if not isinstance(basis, OrthogonalCoordinateCollocationBasis):
            raise ValueError(
                "basis must be an instance of "
                "OrthogonalCoordinateCollocationBasis"
            )
        self._bkd = basis._bkd
        self._basis = basis
        if values is not None:
            self.set_values(values)
        if jac is not None:
            self.set_jacobian(jac)

    def _jacobian_shape(self, basis, ninput_funs) -> tuple:
        return (
            basis.mesh().nmesh_pts(),
            basis.mesh().nmesh_pts() * ninput_funs,
        )

    def ninput_funs(self) -> int:
        return self._ninput_funs

    def basis(self) -> OrthogonalCoordinateCollocationBasis:
        return self._basis

    def nphys_vars(self) -> int:
        return self.basis().nphys_vars()

    def nmesh_pts(self) -> int:
        return self.basis().mesh().nmesh_pts()

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
            raise RuntimeError("must first call set_values()")
        return self._values

    def get_jacobian(self) -> Array:
        if not hasattr(self, "_jac"):
            raise RuntimeError("must call set_jacobian()")
        return self._jac.get_jacobian()

    def set_jacobian(self, jac: SparseJacobian):
        if not isinstance(jac, SparseJacobian):
            raise ValueError("jac must be an instance of SparseJacobian")
        self._jac = jac

    def get_flattened_values(self):
        return self.get_values()

    def get_flattened_jacobian(self):
        return self.get_jacobian()

    def sparse_jacobian(self) -> SparseJacobian:
        return self._jac

    def _add_functions(
        self, other: Union["ScalarOperator", float]
    ) -> "ScalarOperator":
        if isinstance(other, float):
            return ScalarOperator(
                self.basis(),
                self.ninput_funs(),
                self.get_values() + other,
                self.sparse_jacobian().copy(),
            )
        return ScalarOperator(
            self.basis(),
            self.ninput_funs(),
            self.get_values() + other.get_values(),
            self.sparse_jacobian() + other.sparse_jacobian(),
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
            fun1.basis(),
            self.ninput_funs(),
            fun1.get_values() - fun2.get_values(),
            fun1.sparse_jacobian() - fun2.sparse_jacobian(),
        )

    def __sub__(
        self, other: Union["ScalarOperator", float]
    ) -> "ScalarOperator":
        if self._is_float(other):
            # second condition is for scalar tensors
            return ScalarOperator(
                self.basis(),
                self.ninput_funs(),
                self.get_values() - other,
                self.sparse_jacobian().copy(),
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
                self.basis(),
                self.ninput_funs(),
                other - self.get_values(),
                -self.sparse_jacobian(),
            )
        if not isinstance(other, ScalarOperator):
            raise ValueError(
                f"cannot subtract a ScalarOperator from {type(other)}"
            )
        return self._subtract_functions(other, self)

    # def _init_function(self, values):
    #     # initialize operator with zero jacobian
    #     return ScalarOperator(
    #         self.basis(),
    #         self.ninput_funs(),
    #         values,
    #         ZeroJac(
    #             self._bkd,
    #             self._jacobian_shape()
    #         )
    #     )

    def _is_float(self, other):
        return isinstance(other, float) or (
            isinstance(other, self._bkd.array_type())
            and other.ndim == 0
            and other.dtype == self._bkd.double_type()
        )

    def _multiply_functions(
        self, other: Union["ScalarOperator", float]
    ) -> "ScalarOperator":
        if not isinstance(other, ScalarOperator) and not self._is_float(other):
            raise ValueError(
                f"cannot multiply ScalarOperator by {type(other)}"
            )

        if self._is_float(other):
            return ScalarOperator(
                self.basis(),
                self.ninput_funs(),
                self.get_values() * other,
                self.sparse_jacobian() * other,
            )

        values = self.get_values() * other.get_values()
        # use product rule
        jac = (
            other.get_values() * self.sparse_jacobian()
            + other.sparse_jacobian() * self.get_values()
        )
        return ScalarOperator(
            other.basis(),
            self.ninput_funs(),
            values,
            jac,
        )

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
            self.basis(),
            self.ninput_funs(),
            self.get_values() ** other,
            float(other)
            * self.sparse_jacobian()
            * self.get_values() ** (other - 1),
        )

    def _divide_functions(self, fun1, fun2) -> "ScalarOperator":
        if isinstance(fun2, float):
            if fun2 == 0:
                raise ValueError("Cannot divide by zero")
            return ScalarOperator(
                fun1.basis(),
                self.ninput_funs(),
                fun1.get_values() / fun2,
                fun1.sparse_jacobian() / fun2,
            )

        if isinstance(fun1, float):
            if self._bkd.any(fun2.get_values() == 0.0):
                raise ValueError("Cannot divide by zero")
            return ScalarOperator(
                fun2.basis(),
                self.ninput_funs(),
                fun1 / fun2.get_values(),
                -fun1 * fun2.sparse_jacobian() / (fun2.get_values() ** 2),
            )

        values = fun1.get_values() / fun2.get_values()
        # use quotient rule
        jac = (
            fun2.get_values() * fun1.sparse_jacobian()
            - fun2.sparse_jacobian() * fun1.get_values()
        ) / fun2.get_values() ** 2
        return ScalarOperator(
            fun1.basis(),
            self.ninput_funs(),
            values,
            jac,
        )

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
        jac = self.sparse_jacobian().rdot(self.basis()._deriv_mats[physvar_id])
        return ScalarOperator(
            self.basis(),
            self.ninput_funs(),
            self.basis()._deriv_mats[physvar_id] @ self.get_values(),
            jac,
        )

    def __call__(self, eval_samples: Array) -> Array:
        values = self.get_values()[:, None]
        return self.basis().interpolate(values, eval_samples)[:, 0]

    def integrate(self):
        return (
            self.get_values() @ self.basis().quadrature_rule_at_mesh_pts()[1]
        )

    def _plot_1d(self, ax, nplot_pts_1d, **kwargs):
        plot_samples = self._bkd.linspace(
            *self.basis().mesh().trans()._ranges, nplot_pts_1d
        )[None, :]
        return ax.plot(
            self._bkd.to_numpy(plot_samples[0]),
            self._bkd.to_numpy(self(plot_samples)),
            **kwargs,
        )

    def _get_2d_plot_samples(self, npts_1d):
        orth_range = self.basis().mesh().trans()._orthog_ranges
        orth_X, orth_Y, orth_pts_2d = get_meshgrid_samples(
            self._bkd.tile(orth_range, (2,)), npts_1d, bkd=self._bkd
        )
        pts = self.basis().mesh().trans().map_from_orthogonal(orth_pts_2d)
        return pts, orth_X

    def _plot_2d(self, ax, npts_1d, **kwargs):
        pts, orth_X = self._get_2d_plot_samples(npts_1d)
        X = self._bkd.reshape(pts[0], orth_X.shape)
        Y = self._bkd.reshape(pts[1], orth_X.shape)
        Z = self._bkd.reshape(self(pts), X.shape)
        zbounds = kwargs.pop("zbounds", None)
        if zbounds is not None:
            zmin, zmax = zbounds
            if Z.min() < zmin:
                raise ValueError(
                    "Z min {0} outside zbounds {1}".format(Z.min(), zbounds[0])
                )
            if Z.max() > zmax:
                raise ValueError(
                    "Z max {0} outside zbounds {1}".format(Z.max(), zbounds[1])
                )
            Z = (Z - zmin) / (zmax - zmin)
        X = self._bkd.to_numpy(X)
        Y = self._bkd.to_numpy(Y)
        Z = self._bkd.to_numpy(Z)
        if ax.name != "3d":
            kwargs_copy = kwargs.copy()
            if Z.max() - Z.min() < 1e-12:
                kwargs_copy["levels"] = 1
            return ax.contourf(X, Y, Z, **kwargs_copy)
        return ax.plot_surface(X, Y, Z, **kwargs)

    def _set_plot_3d_limits(self, ax, orth_range):
        orth_ranges = self._bkd.reshape(
            self._bkd.tile(orth_range, (3,)), (3, 2)
        )
        ranges = self.basis().mesh().trans().map_from_orthogonal(orth_ranges)
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
        pts = self.basis().mesh().trans().map_from_orthogonal(orth_pts)
        vals = self._bkd.reshape(self(pts), X_shape)
        X = self._bkd.reshape(pts[0], X_shape)
        Y = self._bkd.reshape(pts[1], X_shape)
        Z = self._bkd.reshape(pts[2], X_shape)
        X = self._bkd.to_numpy(X)
        Y = self._bkd.to_numpy(Y)
        Z = self._bkd.to_numpy(Z)
        vals = self._bkd.to_numpy(vals)
        return X, Y, Z, vals

    def _plot_3d_internal(self, ax, npts_1d, fig):
        orth_range = self.basis().mesh().trans()._orthog_ranges
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
        orth_range = self.basis().mesh().trans()._orthog_ranges
        orth_X, orth_Y, orth_pts_2d = get_meshgrid_samples(
            self._bkd.tile(orth_range, (2,)), npts_1d, bkd=self._bkd
        )
        edges_np = [
            self._prepare_edge_3d(
                ax, orth_pts_2d, orth_X.shape, ii, orth_range[0]
            )
            for ii in range(3)
        ]
        edges_np += [
            self._prepare_edge_3d(
                ax, orth_pts_2d, orth_X.shape, ii, orth_range[1]
            )
            for ii in range(3)
        ]
        vmin = np.min(np.hstack([edge[-1] for edge in edges_np]))
        vmax = np.max(np.hstack([edge[-1] for edge in edges_np]))
        ims = []
        for edge in edges_np:
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

    def get_plot_axis(self, figsize=(8, 6), surface=False):
        if self.nphys_vars() < 3 and not surface:
            fig = plt.figure(figsize=figsize)
            return fig, fig.gca()
        fig = plt.figure(figsize=figsize)
        return fig, fig.add_subplot(111, projection="3d")

    def __repr__(self):
        return "{0}(\n{1}\n{2}\n)".format(
            self.__class__.__name__,
            textwrap.indent(
                f"ninput_funs={self.ninput_funs()}", prefix="    "
            ),
            textwrap.indent("basis=" + str(self.basis()), prefix="    "),
        )

    def values_shape(self) -> tuple:
        return (self.basis().mesh().nmesh_pts(),)


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
        zero_jac = ZeroJac(
            self._bkd, self._jacobian_shape(self.basis(), self.ninput_funs())
        )
        super().set_jacobian(zero_jac)

    def set_jacobian(self, jac: Array):
        raise NotImplementedError(
            "do not call set_jacobian because it is called by set_values"
        )


class ConstantScalarFunction(ScalarFunction):
    def __init__(
        self,
        basis: OrthogonalCoordinateCollocationBasis,
        const: float,
        ninput_funs: int = 1,
    ):
        values = basis._bkd.full((basis.mesh().nmesh_pts(),), const)
        super().__init__(basis, values, ninput_funs)


class ZeroScalarFunction(ConstantScalarFunction):
    def __init__(
        self,
        basis: OrthogonalCoordinateCollocationBasis,
        ninput_funs: int = 1,
    ):
        super().__init__(basis, 0.0, ninput_funs)


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
        # ident_jac = self.basis()._bkd.eye(self.basis().mesh().nmesh_pts())
        ident_jac = DiagJac(
            self._bkd,
            self._jacobian_shape(self.basis(), self.ninput_funs()),
            self._bkd.ones((self.basis().mesh().nmesh_pts(), 1)),
        )
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
        ninput_funs: int = 1,
    ):
        super().__init__(
            basis, self._get_values(basis, fun), ninput_funs=ninput_funs
        )

    def _get_values(
        self, basis: OrthogonalCoordinateCollocationBasis, fun: callable
    ):
        self._fun = fun
        return self._fun(basis.mesh().mesh_pts())


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
        vals = self._eval(self.basis().mesh().mesh_pts())
        if vals.ndim > 1:
            vals = vals.T
        self.set_values(vals)
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
            textwrap.indent("basis=" + str(self.basis()), prefix="    "),
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
        sparse_jac_array = basis._bkd.zeros(
            (basis.mesh().nmesh_pts(), ninput_funs)
        )
        sparse_jac_array[:, input_id] = 1.0
        jac = DiagJac(
            basis._bkd,
            self._jacobian_shape(basis, ninput_funs),
            sparse_jac_array,
        )
        super().__init__(basis, ninput_funs, values, jac)


class MatrixOperator:
    def __init__(
        self,
        basis: OrthogonalCoordinateCollocationBasis,
        ninput_funs: int,
        nrows: int,
        ncols: int,
    ):
        self._basis = basis
        self._bkd = basis._bkd
        self._ninput_funs = ninput_funs
        self._nrows = nrows
        self._ncols = ncols

    def basis(self) -> OrthogonalCoordinateCollocationBasis:
        return self._basis

    def ninput_funs(self) -> int:
        return self._ninput_funs

    def _check_components(self, components):
        # todo check all components have the same ninput_funs()
        if len(components) != self.nrows():
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
        return "{0}(nrows={1}, ncols={2}, ninput_funs={3})".format(
            self.__class__.__name__,
            self.nrows(),
            self.ncols(),
            self.ninput_funs(),
        )

    def sqnorm(self) -> ScalarOperator:
        """Squared Frobenius norm of a matrix valued function."""
        ops = []
        for ii in range(self.nrows()):
            for jj in range(self.ncols()):
                ops.append(self._components[ii][jj] ** 2)
        return sum(ops)

    def _is_float(self, other):
        return isinstance(other, float) or (
            isinstance(other, self._bkd.array_type())
            and other.ndim == 0
            and other.dtype == self._bkd.double_type()
        )

    def _multiply_functions(
        self, other: Union["MatrixOperator", ScalarOperator, float]
    ) -> "MatrixOperator":
        if not hasattr(self, "_components"):
            raise RuntimeError("must call set_commponents()")

        if not isinstance(
            other, (MatrixOperator, ScalarOperator)
        ) and not self._is_float(other):
            raise ValueError(
                f"cannot multiply MatrixOperator by {type(other)}"
            )

        if isinstance(other, MatrixOperator):
            op = MatrixOperator(
                self.basis(), self.ninput_funs(), self.nrows(), self.ncols()
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
            self.basis(), self.ninput_funs(), self.nrows(), self.ncols()
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

    def __matmul__(self, other: "MatrixOperator") -> "MatrixOperator":
        if not hasattr(self, "_components"):
            raise RuntimeError("must call set_commponents()")

        if not isinstance(other, MatrixOperator):
            raise ValueError(
                f"cannot dot product a MatrixOperator with {type(other)}"
            )

        if self.ncols() != other.nrows():
            raise ValueError("matrix shapes are inconsistent")

        op = MatrixOperator(
            self.basis(), self.ninput_funs(), self.nrows(), other.ncols()
        )
        components = [
            [None for kk in range(other.ncols())] for ii in range(self.nrows())
        ]
        for ii in range(self.nrows()):
            for kk in range(other.ncols()):
                components[ii][kk] = sum(
                    [
                        self._components[ii][jj] * other._components[jj][kk]
                        for jj in range(self.ncols())
                    ]
                )
        op.set_components(components)
        return op

    @property
    def T(self):
        """
        Returns the transpose of the matrix opeator.
        """
        op = MatrixOperator(
            self.basis(), self.ninput_funs(), self.ncols(), self.nrows()
        )
        components_transpose = [
            [self._components[ii][jj] for ii in range(self.nrows())]
            for jj in range(self.ncols())
        ]
        op.set_components(components_transpose)
        return op

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
                self.basis(), self.ninput_funs(), self.nrows(), self.ncols()
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
            self.basis(), self.ninput_funs(), self.nrows(), self.ncols()
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
                self.basis(), self.ninput_funs(), self.nrows(), self.ncols()
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
            self.basis(), self.ninput_funs(), self.nrows(), self.ncols()
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
                self.basis(), self.ninput_funs(), self.nrows(), self.ncols()
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
            self.basis(), self.ninput_funs(), self.nrows(), self.ncols()
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
        return ScalarOperator(self.basis(), self.ninput_funs(), values)

    def _set_matrix_components(self, components: List[ScalarOperator]):
        MatrixOperator.set_components(self, components)

    def set_flattened_values(self, values: Array):
        values_shape = (
            self.nrows() * self.ncols() * self.basis().mesh().nmesh_pts(),
        )
        if values.shape != values_shape:
            raise ValueError(
                f"values.shape {values.shape} must be {values_shape}"
            )
        reshaped_values = self._bkd.reshape(
            values,
            (self.nrows(), self.ncols(), self.basis().mesh().nmesh_pts()),
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
                row.append(self._components[ii][jj](eval_samples))
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
            self.basis().mesh().nmesh_pts(),
        )

    def plot_vector_field(self, ax, npts_1d: int = 31):
        if self.nrows() != 2 or self.ncols() != 1:
            raise ValueError("Can only plot 2D vector field")
        # pts = self.basis().mesh().mesh_pts()
        # xvel, yvel = self.get_values()
        pts, orth_X = self._components[0][0]._get_2d_plot_samples(npts_1d)
        X = self._bkd.reshape(pts[0], orth_X.shape)
        Y = self._bkd.reshape(pts[1], orth_X.shape)
        xvel = self._components[0][0](pts)
        yvel = self._components[1][0](pts)
        return ax.quiver(
            pts[0], pts[1], xvel, yvel, scale_units="xy", angles="xy"
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
        return "{0}(nrows={1}, ninput_funs={2})".format(
            self.__class__.__name__, self.nrows(), self.ninput_funs()
        )

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
        return (
            self.nrows(),
            self.basis().mesh().nmesh_pts(),
        )


class VectorOperatorOperation(ABC):
    """Shortcut to create complex operator operation on ScalarOperators"""

    @abstractmethod
    def __call__(self, fun: VectorOperator) -> VectorOperator:
        raise NotImplementedError

    def __repr__(self):
        return "{0}".format(self.__class__.__name__)


class VectorSolution(VectorOperator):
    def create_scalar_operator_from_values(
        self, values: Array, row: int, col: int
    ) -> VectorSolutionComponent:
        return VectorSolutionComponent(
            self.basis(), self.ninput_funs(), row, values
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
    def create_scalar_operator_from_values(
        self, values: Array, row: int, col: int
    ) -> "ScalarFunction":
        return ScalarFunction(
            self.basis(),
            values,
            ninput_funs=self.ninput_funs(),
        )

    def set_components(self, components: List[ScalarFunction]):
        for comp in components:
            if not isinstance(comp, ScalarFunction):
                raise ValueError("component must be an instance of Function")
        super().set_components([comp for comp in components])

    def set_values(self, values: Array):
        super().set_values(values)


class ConstantVectorFunction(VectorFunction):
    def __init__(
        self,
        basis: OrthogonalCoordinateCollocationBasis,
        ninput_funs: int,
        nrows: int,
        consts: List[float],
    ):
        if len(consts) != nrows:
            raise ValueError(
                "One constant must be provided for each component"
            )
        values = basis._bkd.stack(
            [
                basis._bkd.full((basis.mesh().nmesh_pts(),), const)
                for const in consts
            ],
            axis=0,
        )
        super().__init__(basis, ninput_funs, nrows)
        self.set_values(values)


class TransientVectorSolution(TransientOperatorMixin, VectorSolution):
    pass


class TransientVectorFunction(TransientOperatorMixin, VectorFunction):
    def create_scalar_operator_from_values(
        self, values: Array, row: int, col: int
    ) -> TransientScalarFunction:
        return TransientScalarFunction(
            self.basis(), values, ninput_funs=self.ninput_funs()
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
        values = self._fun(basis.mesh().mesh_pts())
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
        return ScalarFunction(
            self.basis(), values, ninput_funs=self.ninput_funs()
        )


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
            self.basis(),
            lambda xx, time: self._fun(xx, time)[:, row],
            ninput_funs=self.ninput_funs(),
        )
        fun.set_time(self._time)
        return fun


def nabla(op: ScalarOperator) -> VectorOperator:
    """
    Gradient of a scalar-valued function.
    Returns VectorOperator with shape (nphys_vars, 1)
    """
    vec_op = VectorOperator(op.basis(), op.ninput_funs(), op.nphys_vars())
    ops = [op.deriv(dd) for dd in range(op.nphys_vars())]
    vec_op.set_components(ops)
    return vec_op


def vector_nabla(vec_op: VectorOperator) -> MatrixOperator:
    """
    Gradient of a vector-valued function.
    Returns VectorOperator with shape (nrows, nphys_vars).
    This is opposite convention of scalar nabla. We change format here
    because we want each row to correspond to a unique equation
    """
    mat_op = MatrixOperator(
        vec_op.basis(),
        vec_op.ninput_funs(),
        vec_op.nrows(),
        vec_op.nphys_vars(),
    )
    comps = [
        [comp.deriv(dd) for dd in range(comp.nphys_vars())]
        for comp in vec_op.get_components()
    ]
    mat_op.set_components(comps)
    return mat_op


def div(mat_op: MatrixOperator):
    """
    Divergence of a Matrix valued function.

    The divergence operator is applied to each column of the matrix
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

    op = VectorOperator(mat_op.basis(), mat_op.ninput_funs(), len(ops))
    op.set_components(ops)
    return op


class ScalarKLEFunction(ScalarFunction):
    def __init__(
        self,
        basis: OrthogonalCoordinateCollocationBasis,
        lenscale: float,
        nterms: int,
        sigma: float = 1,
        use_log: bool = False,
        matern_nu: float = np.inf,
        mean_field: ScalarFunction = None,
        ninput_funs: int = 1,
        use_quadrature: bool = True,
    ):
        super().__init__(basis, ninput_funs=ninput_funs)
        self._setup_kle(
            lenscale,
            sigma,
            use_log,
            matern_nu,
            nterms,
            mean_field,
            use_quadrature,
        )

    def _setup_kle(
        self,
        lenscale: float,
        sigma: float,
        use_log: bool,
        matern_nu: float,
        nterms: int,
        mean_field: ScalarFunction,
        use_quadrature: bool,
    ):
        if use_quadrature:
            quad_weights = self.basis().quadrature_rule_at_mesh_pts()[1][:, 0]
        else:
            quad_weights = None
        self._kle = MeshKLE(
            self.basis().mesh().mesh_pts(),
            lenscale,
            sigma,
            0.0 if mean_field is None else mean_field.get_values(),
            use_log,
            matern_nu,
            quad_weights,
            nterms,
            backend=self.basis()._bkd,
        )
        # initialize to mean
        self.set_param(
            self._bkd.zeros(
                self._kle.nvars(),
            )
        )

    def kle(self) -> MeshKLE:
        return self._kle

    def set_param(self, param):
        self.set_values(self._kle(param[:, None])[:, 0])

    def eigenfunctions(self):
        return [
            ScalarFunction(self._basis, self._kle.eigenvectors()[:, ii])
            for ii in range(self._kle.nvars())
        ]


class ScalarPeriodicReiszGaussianRandomField(ScalarFunction):
    def __init__(
        self,
        basis: OrthogonalCoordinateCollocationBasis,
        neigs: int,
        sigma: float,
        tau: float,
        gamma: float,
        use_log: bool = False,
        ninput_funs: int = 1,
    ):
        super().__init__(basis, ninput_funs=ninput_funs)
        self._setup_kle(sigma, tau, gamma, use_log, neigs)

    def _setup_kle(
        self,
        sigma: float,
        tau: float,
        gamma: float,
        use_log: bool,
        neigs: int,
    ):
        self._kle = PeriodicReiszGaussianRandomField(
            sigma,
            tau,
            gamma,
            neigs,
            self.basis().mesh()._trans._ranges,
            backend=self.basis()._bkd,
        )
        self._kle.set_domain_samples(self.basis().mesh().mesh_pts())
        # initialize to mean
        self.set_param(
            self._bkd.zeros(
                self._kle.nterms(),
            )
        )

    def kle(self) -> PeriodicReiszGaussianRandomField:
        return self._kle

    def set_param(self, param):
        self.set_values(self._kle.values(param[:, None])[:, 0])


def remove_3d_axis_panels(ax):
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    return ax


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


def animate_transient_2d_scalar_solution(
    basis: OrthogonalCoordinateCollocationBasis,
    sol: Array,
    times: Array,
    plot_surface: bool = False,
):
    if basis.nphys_vars() != 2:
        raise ValueError("This function only creates 2D animations")
    bkd = basis._bkd
    gs = GridSpec(10, 1, hspace=1)  # 10 rows, 1 column
    fig = plt.figure(figsize=(8, 6))
    if plot_surface:
        ax0 = fig.add_subplot(gs[:9, 0], projection="3d")
    else:
        ax0 = fig.add_subplot(gs[:9, 0])
    ax1 = fig.add_subplot(gs[-1, 0])
    axs = [ax0, ax1]

    # get the maximum values of the plot so
    # that colorscale is consistent between frames
    npts1d = 51
    solfun = ScalarFunction(basis)
    plot_samples = solfun._get_2d_plot_samples(npts1d)[0]
    sol_plot_values = []
    for ii in range(sol.shape[1]):
        solfun.set_values(sol[:, ii])
        sol_plot_values.append(solfun(plot_samples))
    sol_plot_values = bkd.stack(sol_plot_values, axis=1)
    zmin, zmax = sol_plot_values.min(), sol_plot_values.max()
    state_bounds = bkd.array([zmin, zmax])
    levels = bkd.linspace(*state_bounds, 51)

    def animate(ii):
        [ax.clear() for ax in axs]
        solfun = ScalarSolution(basis)
        solfun.set_values(sol[:, ii])
        if not plot_surface:
            solfun.plot(axs[0], npts1d, levels=levels)
        else:
            solfun.plot(axs[0], npts1d)
            axs[0].set_zlim((zmin, zmax))
            axs[0] = remove_3d_axis_panels(axs[0])
            # rotate frame as animation evolves
            # axs[0].view_init(0, 2*ii)

        # imshow reverts to the same color as ii == 0
        # at ii = times.shape[0]-1 because all values
        # in the plot are the same. To avoid this set colormap
        # at final time to be the correct value
        if ii == times.shape[0] - 1:
            cmap = truncate_colormap(plt.cm.binary, minval=1.0)
        else:
            cmap = truncate_colormap(plt.cm.binary, minval=0.1)
        timebar = bkd.zeros((times.shape[0] - 1))
        timebar[:ii] = 1.0

        # TODO last timestep plot of timebar is all one value
        # so it plots all the first color and not all the last color
        axs[1].imshow(
            timebar[None, :],
            extent=[times[0], times[-1], 0, 1],
            aspect="auto",
            cmap=cmap,
        )
        axs[1].set_xlabel("Time")
        axs[1].set_yticks([])

    ani = animation.FuncAnimation(
        fig,
        animate,
        interval=125,
        repeat_delay=1000,
        frames=sol.shape[-1],
    )
    return ani


def animate_transient_2d_vector_solution(
    basis: OrthogonalCoordinateCollocationBasis,
    sol: Array,
    times: Array,
    ncomponents: int,
    components_as_contour_plots: Union[list, Array] = None,
    components_as_surface_plots: Union[list, Array] = None,
    npts1d: int = 51,
    contour_plot_kwargs: dict = {},
    surface_plot_kwargs: dict = {},
):
    bkd = basis._bkd
    if times.shape[0] != sol.shape[1]:
        raise ValueError("times and sol are inconsistent")

    if components_as_contour_plots is None:
        components_as_contour_plots = bkd.arange(ncomponents())
    if components_as_surface_plots is None:
        components_as_surface_plots = bkd.arange(ncomponents())
    components_as_contour_plots = bkd.asarray(
        components_as_contour_plots, dtype=int
    )
    components_as_surface_plots = bkd.asarray(
        components_as_surface_plots, dtype=int
    )
    if bkd.max(components_as_contour_plots) >= ncomponents:
        raise ValueError(
            "entry in components_as_contour_plots exceeds ncomponents"
        )
    if bkd.max(components_as_surface_plots) >= ncomponents:
        raise ValueError(
            "entry in components_as_contour_plots exceeds ncomponents"
        )

    ncomponent_plots = (
        components_as_contour_plots.shape[0]
        + components_as_surface_plots.shape[0]
    )
    fig = plt.figure(figsize=(ncomponent_plots * 8, 6))
    gs = GridSpec(
        10, ncomponent_plots, hspace=1
    )  # 10 rows, ncomponent columns
    # plot all surface plots then all contour plots
    axs = [
        fig.add_subplot(gs[:9, ii], projection="3d")
        for ii in range(components_as_surface_plots.shape[0])
    ] + [
        fig.add_subplot(gs[:9, ii + components_as_surface_plots.shape[0]])
        for ii in range(components_as_contour_plots.shape[0])
    ]
    # add axis for time bar
    axs += [fig.add_subplot(gs[-1, :])]

    # get the maximum values of the plot so
    # that colorscale is consistent between frames
    solfun = VectorSolution(basis, ncomponents, ncomponents)
    solfun.set_flattened_values(sol[:, 0])  # needed so next line will work
    plot_samples = solfun.get_components()[0]._get_2d_plot_samples(npts1d)[0]
    sol_plot_values = []
    for ii in range(sol.shape[1]):
        solfun.set_flattened_values(sol[:, ii])
        sol_plot_values.append(solfun(plot_samples))
    sol_plot_values = bkd.stack(sol_plot_values, axis=-1)
    zmin = bkd.min(bkd.min(sol_plot_values, axis=-1), axis=-1)
    zmax = bkd.max(bkd.max(sol_plot_values, axis=-1), axis=-1)
    state_bounds = bkd.stack([zmin, zmax], axis=1)
    levels = [bkd.linspace(*bounds, 51) for bounds in state_bounds]

    def animate(ii):
        [ax.clear() for ax in axs]
        solfun.set_flattened_values(sol[:, ii])
        components = solfun.get_components()
        for kk, component_id in enumerate(components_as_surface_plots):
            components[component_id].plot(
                axs[kk], npts1d, **surface_plot_kwargs
            )
            axs[kk].set_zlim(state_bounds[component_id])
            axs[kk] = remove_3d_axis_panels(axs[kk])
        for jj, component_id in enumerate(components_as_contour_plots):
            kk = jj + components_as_surface_plots.shape[0]
            components[component_id].plot(
                axs[kk],
                npts1d,
                levels=levels[component_id],
                **contour_plot_kwargs,
            )
        # imshow reverts to the same color as ii == 0
        # at ii = times.shape[0]-1 because all values
        # in the plot are the same. To avoid this set colormap
        # at final time to be the correct value
        if ii == times.shape[0] - 1:
            cmap = truncate_colormap(plt.cm.binary, minval=1.0)
        else:
            cmap = truncate_colormap(plt.cm.binary, minval=0.1)
        timebar = bkd.zeros(times.shape[0] - 1)
        timebar[:ii] = 1.0
        # TODO last timestep plot of timebar is all one value
        # so it plots all the first color and not all the last color
        axs[-1].imshow(
            timebar[None, :],
            extent=[times[0], times[-1], 0, 1],
            aspect="auto",
            cmap=cmap,
        )
        axs[-1].set_xlabel("Time")
        axs[-1].set_yticks([])

    ani = animation.FuncAnimation(
        fig,
        animate,
        interval=125,
        repeat_delay=1000,
        frames=sol.shape[-1],
        init_func=lambda: None,
    )
    return ani


def get_water_cmap():
    return truncate_colormap(plt.cm.Blues, minval=0.1)


def plot_vector_function(
    vec: MatrixOperator,
    components_as_contour_plots: Union[list, Array] = None,
    components_as_surface_plots: Union[list, Array] = None,
    npts1d: int = 51,
    contour_plot_kwargs: dict = {},
    surface_plot_kwargs: dict = {},
):
    bkd = vec._bkd
    if components_as_contour_plots is None:
        components_as_contour_plots = bkd.arange(vec.nrows())
    if components_as_surface_plots is None:
        components_as_surface_plots = bkd.arange(vec.nrows())
    components_as_contour_plots = bkd.asarray(
        components_as_contour_plots, dtype=int
    )
    components_as_surface_plots = bkd.asarray(
        components_as_surface_plots, dtype=int
    )
    if bkd.max(components_as_contour_plots) >= vec.nrows():
        raise ValueError(
            "entry in components_as_contour_plots exceeds ncomponents"
        )
    if bkd.max(components_as_surface_plots) >= vec.nrows():
        raise ValueError(
            "entry in components_as_contour_plots exceeds ncomponents"
        )

    ncomponent_plots = (
        components_as_contour_plots.shape[0]
        + components_as_surface_plots.shape[0]
    )

    gs = GridSpec(1, ncomponent_plots, hspace=1)
    fig = plt.figure(figsize=(ncomponent_plots * 8, 6))
    axs = [
        fig.add_subplot(gs[0, ii], projection="3d")
        for ii in range(components_as_surface_plots.shape[0])
    ] + [
        fig.add_subplot(gs[0, ii + components_as_surface_plots.shape[0]])
        for ii in range(components_as_contour_plots.shape[0])
    ]

    tempvec = VectorFunction(vec.basis(), vec.ninput_funs(), vec.nrows())
    tempvec.set_flattened_values(vec.get_flattened_values())
    plot_samples = tempvec.get_components()[0]._get_2d_plot_samples(npts1d)[0]
    tempvals = tempvec(plot_samples)
    zmin = bkd.min(tempvals, axis=-1)
    zmax = bkd.max(tempvals, axis=-1)

    # todo create function similar to below that can also be used in
    # animate vector solution. Also use above to create function that
    # creates a grid for plotting vector functions

    state_bounds = bkd.stack([zmin, zmax], axis=1)
    levels = [bkd.linspace(*bounds, 51) for bounds in state_bounds]
    components = vec.get_components()
    for kk, component_id in enumerate(components_as_surface_plots):
        components[component_id].plot(axs[kk], npts1d, **surface_plot_kwargs)
        axs[kk].set_zlim(state_bounds[component_id])
        axs[kk] = remove_3d_axis_panels(axs[kk])
    for jj, component_id in enumerate(components_as_contour_plots):
        kk = jj + components_as_surface_plots.shape[0]
        im = components[component_id].plot(
            axs[kk],
            npts1d,
            levels=levels[component_id],
            **contour_plot_kwargs,
        )
        plt.colorbar(im, ax=axs[kk])
