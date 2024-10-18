from abc import ABC, abstractmethod

from pyapprox.util.linearalgebra.linalgbase import Array
from pyapprox.pde.collocation.mesh import OrthogonalCoordinateMeshBoundary
from pyapprox.pde.collocation.functions import MatrixFunction
from pyapprox.pde.collocation.basis import OrthogonalCoordinateCollocationBasis


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

        stride = idx[1] - idx[0]
        if axis == 0:
            return vec[idx[0] : idx[-1] + 1 : stride]
        return vec[:, idx[0] : idx[-1] + 1 : stride]

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
        bndry_vals = self._bkd.flatten(self(self._mesh_bndry._bndry_mesh_pts))
        res[idx] = self._bndry_slice(sol, idx, 0) - bndry_vals
        return res

    def apply_to_jacobian(self, sol: Array, jac: Array):
        idx = self._mesh_bndry._bndry_idx
        jac[idx,] = 0.0
        jac[idx, idx] = 1.0
        return jac


class BoundaryFromFunctionMixin:
    def _set_function(self, fun: MatrixFunction):
        if not isinstance(fun, MatrixFunction):
            raise ValueError("fun must be an instance of MatrixFunction")
        self._fun = fun

    def __call__(self, bndry_mesh_pts):
        return self._fun(bndry_mesh_pts)


class DirichletBoundaryFromFunction(
    BoundaryFromFunctionMixin, DirichletBoundary
):
    def __init__(
        self, mesh_bndry: OrthogonalCoordinateMeshBoundary, fun: MatrixFunction
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
            self._normal_vals[:, dd : dd + 1] * flux_jac[dd]
            for dd in range(self._mesh_bndry.nphys_vars())
        ]
        return flux_normal_jac

    def apply_to_residual(
        self, sol_array: Array, res_array: Array, jac: Array
    ):
        idx = self._mesh_bndry._bndry_idx
        bndry_vals = self._bkd.flatten(self(self._mesh_bndry._bndry_mesh_pts))
        # todo flux_normal vals gets called here and in apply_to_jacobian
        # remove this computational redundancy
        jac[idx] = sum(self._flux_normal_jacobian(sol_array))
        res_array[idx] = (
            self._alpha * self._bndry_slice(sol_array, idx, 0)
            + self._beta * (self._bndry_slice(jac, idx, 0) @ sol_array)
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


class RobinBoundaryFromFunction(BoundaryFromFunctionMixin, RobinBoundary):
    def __init__(
        self,
        mesh_bndry: OrthogonalCoordinateMeshBoundary,
        fun: MatrixFunction,
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
                normal_vals[:, dd : dd + 1]
                * (self._bndry_slice(self._basis._deriv_mats[dd], idx, 0))
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
        res_array[idx1] = sol_array[idx1] - sol_array[idx2]
        # match flux
        res_array[idx2] = self._gradient_dot_normal(
            self._mesh_bndry, self._normal_vals, sol_array
        ) + self._gradient_dot_normal(
            self._partner_mesh_bndry, self._partner_normal_vals, sol_array
        )
        return res_array

    def apply_to_jacobian(self, sol_array: Array, jac: Array):
        idx1 = self._mesh_bndry._bndry_idx
        idx2 = self._partner_mesh_bndry._bndry_idx
        jac[idx1, :] = 0
        jac[idx1, idx1] = 1
        jac[idx1, idx2] = -1
        jac[idx2] = self._gradient_dot_normal_jacobian(
            self._mesh_bndry, self._normal_vals
        ) + self._gradient_dot_normal_jacobian(
            self._partner_mesh_bndry, self._partner_normal_vals
        )
        return jac

    def __call__(self, samples):
        raise NotImplementedError("Periodic Boundary does not need __call__")
