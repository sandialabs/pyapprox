from abc import ABC, abstractmethod
from typing import Union

from pyapprox.util.linearalgebra.linalgbase import Array
from pyapprox.pde.collocation.mesh import OrthogonalCoordinateMeshBoundary
from pyapprox.pde.collocation.functions import (
    ScalarOperator,
    VectorOperator,
    MatrixOperator,
    TransientOperatorMixin,
)
from pyapprox.pde.collocation.basis import OrthogonalCoordinateCollocationBasis


class BoundaryOperator(ABC):
    def __init__(
            self,
            mesh_bndry: OrthogonalCoordinateMeshBoundary,
            index_shift: int = 0,
    ):
        if not isinstance(mesh_bndry, OrthogonalCoordinateMeshBoundary):
            raise ValueError(
                "mesh_bndry must be an instance of "
                "OrthogonalCoordinateMeshBoundary"
            )
        self._mesh_bndry = mesh_bndry
        self._bkd = self._mesh_bndry._bkd
        # self._mesh_bndry._bndry_idx indexes mesh points on the boundary
        # of a single mesh
        # self._residual_bndry_idx indexes point
        # VectorFunction.get_flattened_values() that are on the boundary
        # index shift should be component_id * mesh.nmesh_pts()
        self._residual_bndry_idx = self._mesh_bndry._bndry_idx + index_shift

    @abstractmethod
    def apply_to_residual(self, sol: Array, res_array: Array):
        raise NotImplementedError

    @abstractmethod
    def apply_to_jacobian(self, sol: Array, jac: Array):
        raise NotImplementedError

    def _bndry_slice(self, vec, idx, axis):
        # todo move to mesh_bndry
        if self._mesh_bndry.nphys_vars() == 3:
            return vec[idx]
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


class DirichletBoundary(BoundaryOperator):
    def apply_to_residual(self, sol: Array, res: Array):
        idx = self._residual_bndry_idx
        bndry_vals = self._bkd.flatten(self(self._mesh_bndry._bndry_mesh_pts))
        res[idx] = self._bndry_slice(sol, idx, 0) - bndry_vals
        return res

    def apply_to_jacobian(self, sol: Array, jac: Array):
        idx = self._residual_bndry_idx
        jac[idx,] = 0.0
        jac[idx, idx] = 1.0
        return jac


class BoundaryFromOperatorMixin:
    def _set_function(self, fun: Union[ScalarOperator, VectorOperator]):
        if not isinstance(fun, (ScalarOperator, VectorOperator)):
            raise ValueError(
                "fun must be an instance of ScalarOperator or VectorOperator"
            )
        self._fun = fun

    def __call__(self, bndry_mesh_pts: Array) -> Array:
        # todo consider avoiding interpolation and just
        # accessing correct values in self._fun.get_flattened_values()
        # by passing in index to array returned by get_values()
        # need to think about whether this affects hybrid discontinous Galerkin
        # perhaps support both options
        return self._bkd.flatten(self._fun(bndry_mesh_pts))


class ConstantDirichletBoundary(
    BoundaryFromOperatorMixin, DirichletBoundary
):
    def __init__(
        self,
        mesh_bndry: OrthogonalCoordinateMeshBoundary,
        const: float,
        index_shift: int = 0,
    ):
        self._const = const
        super().__init__(mesh_bndry, index_shift)

    def __call__(self, bndry_mesh_pts: Array) -> Array:
        return self._bkd.full((bndry_mesh_pts.shape[1],), self._const)


class DirichletBoundaryFromOperator(
    BoundaryFromOperatorMixin, DirichletBoundary
):
    def __init__(
        self,
        mesh_bndry: OrthogonalCoordinateMeshBoundary,
        fun: MatrixOperator,
        index_shift: int = 0,
    ):
        super().__init__(mesh_bndry, index_shift)
        self._set_function(fun)

    def set_time(self, time):
        if not isinstance(self._fun, TransientOperatorMixin):
            raise ValueError(
                "set_time can only be used with transient functions"
            )


class RobinBoundary(BoundaryOperator):
    def __init__(
        self,
        mesh_bndry: OrthogonalCoordinateMeshBoundary,
        alpha: float,
        beta: float,
        index_shift: int = 0,
        component_id: int = 0,
    ):
        super().__init__(mesh_bndry, index_shift)
        self._component_id = component_id
        self._alpha = alpha
        self._beta = beta
        self._normal_vals = self._mesh_bndry.normals(
            self._mesh_bndry._bndry_mesh_pts
        )
        self._flux = None
        self._flux_jac = None

    def set_flux_functions(self, flux: callable, flux_jac: callable):
        self._flux = flux
        self._flux_jac = flux_jac

    def _normal_flux(self, sol_array: Array):
        # self._flux only returns one component of flux
        # so  use self._mesh_bndry._bndry_idx instead of
        # self._residual_bndry_idx
        idx = self._mesh_bndry._bndry_idx
        flux = self._flux(sol_array)
        normal_flux = sum(
            [
                self._normal_vals[:, dd] * flux[dd, idx]
                for dd in range(self._mesh_bndry.nphys_vars())
            ]
        )
        return normal_flux

    def _normal_flux_jacobian(self, sol_array: Array):
        # pass in sol in case flux depends on sol
        # todo. (determine if sol is ever needed an remove if not)
        # todo only compute once for linear transient problems
        # todo: flux_jac called here evaluates flux jac on all grid points
        #       find way to only evalyate on boundary

        # self._flux only returns one component of flux
        # so  use self._mesh_bndry._bndry_idx instead of
        # self._residual_bndry_idx
        idx = self._mesh_bndry._bndry_idx
        flux_jac = self._flux_jac(sol_array)
        # flux_jac = [self._bndry_slice(f, idx, 0) for f in flux_jac]
        flux_jac = flux_jac[:, idx]
        normal_flux_jac = sum(
            [
                self._normal_vals[:, dd : dd + 1] * flux_jac[dd]
                for dd in range(self._mesh_bndry.nphys_vars())
            ]
        )
        # def autofun(sarray):
        #     return self._normal_flux(sarray)
        # jac_auto = self._bkd.jacobian(autofun, sol_array)
        # import torch
        # torch.set_printoptions(linewidth=1000, threshold=10000)
        # print(jac_auto.shape, "J")
        # print(normal_flux_jac.shape)
        # assert self._bkd.allclose(normal_flux_jac, jac_auto, atol=1e-15)
        return normal_flux_jac

    def apply_to_residual(self, sol_array: Array, res_array: Array):
        idx = self._residual_bndry_idx
        bndry_vals = self._bkd.flatten(self(self._mesh_bndry._bndry_mesh_pts))
        res_array[idx] = (
            self._alpha * self._bndry_slice(sol_array, idx, 0)
            + self._beta * self._normal_flux(sol_array)
            - bndry_vals
        )
        return res_array

    def apply_to_jacobian(self, sol: Array, jac: Array):
        # ignoring normals
        # res = D1 * u + D2 * u + alpha * I
        # so jac(res) = D1 + D2 + alpha * I
        idx = self._residual_bndry_idx
        # D1 + D2
        jac[idx] = self._beta * self._normal_flux_jacobian(sol)
        # alpha * I
        jac[idx, idx] += self._alpha
        return jac

    def __repr__(self):
        return "{0}(bndry={1}, component_id={2})".format(
            self.__class__.__name__, self._mesh_bndry, self._component_id
        )


class RobinBoundaryFromOperator(BoundaryFromOperatorMixin, RobinBoundary):
    # beta * flux(u(x)) @ n + alpha * u(x) = g(x)
    def __init__(
        self,
        mesh_bndry: OrthogonalCoordinateMeshBoundary,
        fun: MatrixOperator,
        alpha: float,
        beta: float,
        index_shift: int,
        component_id: int,
    ):
        super().__init__(
            mesh_bndry, alpha, beta, index_shift, component_id
        )
        self._set_function(fun)


class ConstantRobinBoundary(RobinBoundary):
    # beta * flux(u(x)) @ n + alpha * u(x) = 0
    def __init__(
        self,
        mesh_bndry: OrthogonalCoordinateMeshBoundary,
        const: float,
        alpha: float,
        beta: float,
        index_shift: int,
        component_id: int,
    ):
        self._const = const
        super().__init__(
            mesh_bndry, alpha, beta, index_shift, component_id
        )

    def __call__(self, bndry_mesh_pts: Array) -> Array:
        return self._bkd.full((bndry_mesh_pts.shape[1],), self._const)


class PeriodicBoundary(BoundaryOperator):
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

    def apply_to_residual(self, sol_array: Array, res_array: Array):
        idx1 = self._residual_bndry_idx
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
        idx1 = self._residual_bndry_idx
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
