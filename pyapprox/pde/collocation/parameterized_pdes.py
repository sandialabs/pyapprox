from typing import Tuple
import math

from pyapprox.util.linearalgebra.linalgbase import Array, LinAlgMixin
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.pde.collocation.adjoint_models import (
    TransientAdjointModel,
    TransientAdjointFunctional,
)
from pyapprox.pde.collocation.timeintegration import (
    TimeIntegratorNewtonResidual,
    TransientNewtonResidual,
)
# warning shallowwater wrapper needs updating
from pyapprox.pde.collocation.solvers import (
    TransientPDE, TransientPhysicsNewtonResidual
)
from pyapprox.pde.collocation.newton import NewtonSolver
from pyapprox.pde.collocation.functions import (
    ScalarFunction,
    ConstantScalarFunction,
    VectorFunction,
    ScalarKLEFunction,
    ZeroScalarFunction,
    VectorFunctionFromCallable,
    ScalarFunctionFromCallable,
)
from pyapprox.pde.collocation.boundaryconditions import (
    DirichletBoundaryFromOperator,
    RobinBoundaryFromOperator,
    ConstantRobinBoundary,
    ConstantDirichletBoundary,
)
from pyapprox.pde.collocation.physics import (
    Physics,
    ShallowWaveEquation,
    AdvectionDiffusionReactionEquation,
)
from pyapprox.pde.collocation.mesh import ChebyshevCollocationMesh2D
from pyapprox.pde.collocation.basis import ChebyshevCollocationBasis2D
from pyapprox.pde.collocation.mesh_transforms import (
    ScaleAndTranslationTransform2D,
)


class TransientAdvectionDiffusionReactionModel(TransientAdjointModel):
    def __init__(
        self,
        init_time: float,
        final_time: float,
        deltat: float,
        time_residual_cls: TimeIntegratorNewtonResidual,
        newton_solver: NewtonSolver = None,
        functional: TransientAdjointFunctional = None,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        self._bkd = backend
        self.setup_basis()
        self._nominal_val = 0.
        self.setup_diffusion()
        self.setup_velocity()
        self.setup_forcing()
        self._physics = AdvectionDiffusionReactionEquation(
            self._forcing,
            self._diffusion,
            reaction_op=None,
            velocity_field=self._vel_field,
        )
        self._set_boundaries()
        # make following default for all collocation based adjoint models
        time_residual = time_residual_cls(
            TransientPhysicsNewtonResidual(self._physics)
        )
        time_residual._apply_constraints_to_residual = (
            time_residual.native_residual._apply_constraints_to_residual
        )
        time_residual._apply_constraints_to_jacobian = (
            time_residual.native_residual._apply_constraints_to_jacobian
        )

        super().__init__(
            init_time,
            final_time,
            deltat,
            time_residual,
            functional,
            newton_solver,
            backend=backend,
        )

    def setup_basis(self):
        Lx, Ly = 1, 1
        bounds = self._bkd.array([0, Lx, 0, Ly])
        transform = ScaleAndTranslationTransform2D(
            [-1, 1, -1, 1], bounds, self._bkd
        )
        mesh = ChebyshevCollocationMesh2D([20, 20], transform)
        self._basis = ChebyshevCollocationBasis2D(mesh)

    def setup_diffusion(self):
        self._diffusion = ScalarKLEFunction(
            self._basis,
            0.1,
            3,
            sigma=0.1,
            mean_field=ConstantScalarFunction(self._basis, -1., 1),
            ninput_funs=self._basis.mesh.nphys_vars() + 1,
            use_log=True,
        )

    def setup_velocity(self):
        self._vel_field = VectorFunctionFromCallable(
            self._basis,
            1,
            self._basis.nphys_vars(),
            lambda x: self._bkd.stack(
                (
                    self._bkd.cos(4*(x[0]-2) + 2*(4*(x[1]-2))),
                    2 * self._bkd.sin(2*(2*x[0]-2) - 2*(2*(2*x[1]-2)))
                ),
                axis=1
            )
        )

    def setup_forcing(self):
        # self._forcing = ConstantScalarFunction(self._basis, 0., 1)
        self._forcing = ScalarFunctionFromCallable(
            self._basis,
            lambda x: self._bkd.prod(x**10*(1-x)**10, axis=0)*1e7,
            ninput_funs=1
        )

    def get_initial_condition(self):
        return ConstantScalarFunction(
            self._basis, self._nominal_val, ninput_funs=1
        ).get_values()
        # return ScalarFunctionFromCallable(
        #     self._basis,
        #     lambda x: self._bkd.prod(x**5*(1-x)**5, axis=0)*1e7,
        #     ninput_funs=1
        # ).get_values()

    def _set_boundaries(self):
        # make flux depend on solution's value relative to a nominal_val
        # beta * flux(u(x)) @ n = u(x) - nominal_val
        # beta * flux(u(x)) @ n - u(x) = nominal_val
        # beta * flux(u(x)) @ n + alpha * u(x) = nominal_val, alpha = -1

        # Conservative rules are written as
        # du/dt = -div(F) + g for flux F.
        bndrys = []
        alpha, beta = -1.0, 1.0
        for (
            bndry_name,
            mesh_bndry,
        ) in self._basis.mesh.get_boundaries().items():
            bndrys.append(
                ConstantRobinBoundary(
                    mesh_bndry, self._nominal_val, alpha, beta, 0, 0
                )
            )
            # bndrys.append(
            #     ConstantDirichletBoundary(mesh_bndry, 0.)
            # )
        self._physics.set_boundaries(bndrys)

    def nvars(self) -> int:
        # TODO MAKE THIS REQQUIRED FOR ALL PARAMETERIZED MODELS
        if not hasattr(self, "_functional"):
            return self._diffusion._kle.nvars()
        return (
            self._functional.nunique_functional_params()
            + self._diffusion._kle.nvars()
        )

    def forward_solve(self, sample: Array) -> Tuple[Array, Array]:
        sols, times = super().forward_solve(sample)
        return (
            self._sols.reshape(
                (self._basis.mesh.nmesh_pts(), self._times.shape[0])
            ),
            self._times
        )


class ShallowWaterWaveModel(TransientAdjointModel):
    def __init__(
        self,
        init_time: float,
        final_time: float,
        deltat: float,
        time_residual_cls: TimeIntegratorNewtonResidual,
        bed: ScalarKLEFunction,
        init_surface: ScalarFunction,
        functional: TransientAdjointFunctional = None,
        newton_solver: NewtonSolver = None,
        forcing: VectorFunction = None,
    ):
        self._basis = bed.basis
        self._bed = bed
        self._forcing = forcing
        self._physics = ShallowWaveEquation(self._bed, self._forcing)

        self._init_surface = init_surface
        bndrys = self.setup_reflective_boundaries()
        # bndrys = self.setup_dirichlet_boundaries()
        self._physics.set_boundaries(bndrys)
        # make following default for all collocation based adjoint models
        time_residual = time_residual_cls(
            TransientPhysicsNewtonResidual(self._physics)
        )
        time_residual._apply_constraints_to_residual = (
            time_residual.native_residual._apply_constraints_to_residual
        )
        time_residual._apply_constraints_to_jacobian = (
            time_residual.native_residual._apply_constraints_to_jacobian
        )

        super().__init__(
            init_time,
            final_time,
            deltat,
            time_residual,
            functional,
            newton_solver,
            backend=self._basis._bkd,
        )

    def nvars(self) -> int:
        # TODO MAKE THIS REQQUIRED FOR ALL PARAMETERIZED MODELS
        return self._bed._kle.nvars()

    def physics(self) -> Physics:
        return self._physics

    def get_initial_condition(self):
        init_cond = VectorFunction(
            self._basis,
            self._physics.ncomponents(),
            self._physics.ncomponents(),
        )
        if self._basis._bkd.any(
            self._init_surface.get_values() <= self._bed.get_values()
        ):
            print(self._init_surface.get_values())
            print(self._bed.get_values())
            raise ValueError(
                "bed and initial surface given cause negative depths"
            )
        init_depth = ScalarFunction(
            self._basis,
            (self._init_surface - self._bed).get_values(),
            ninput_funs=self._bed.ninput_funs(),
        )
        init_cond.set_components(
            [init_depth]
            + [self._get_zerofun() for ii in range(self._basis.nphys_vars())]
        )
        return init_cond.get_flattened_values()

    def _get_zerofun(self):
        return ZeroScalarFunction(
            self._basis,
            ninput_funs=self._physics.ncomponents(),
        )

    def setup_reflective_boundaries(self):
        # bndrys_funs: list[BoundaryOperator],
        bndry_funs = []
        # loop over all boundaries
        for (
            bndry_name,
            mesh_bndry,
        ) in self._basis.mesh.get_boundaries().items():
            # loop over momentum solution components
            for component_id in range(self._physics.ncomponents()):
                if (component_id == 1 and bndry_name in ["left", "right"]) or (
                    component_id == 2 and bndry_name in ["bottom", "top"]
                ):
                    # set vertical velocity on horizontal boundaries to zero
                    bndry_funs.append(
                        DirichletBoundaryFromOperator(
                            mesh_bndry,
                            self._get_zerofun(),
                            component_id * self._basis.mesh.nmesh_pts(),
                        )
                    )
                elif False:
                    bndry_funs.append(
                        RobinBoundaryFromOperator(
                            mesh_bndry,
                            self._get_zerofun(),
                            0,
                            1.0,
                            component_id * self._basis.mesh.nmesh_pts(),
                            component_id,
                        )
                    )
        return bndry_funs

    def set_param(self, param: Array):
        self._physics._bed.set_param(param)


class LotkaVolterraResidual(TransientNewtonResidual):
    def set_time(self, time: float):
        self._time = time

    def nstates(self) -> int:
        return 3

    def set_param(self, param: Array):
        if param.shape[0] != self.nvars():
            raise ValueError("param has the wrong shape")
        self._param = param
        self._rcoefs = param[: self.nstates()]
        self._acoefs = self._bkd.reshape(
            param[self.nstates() :], (self.nstates(), self.nstates())
        )

    def __call__(self, sol: Array) -> Array:
        return self._rcoefs * sol * (1.0 - self._acoefs @ sol)

    def jacobian(self, sol: Array) -> Array:
        return (
            self._bkd.diag(self._rcoefs)
            - self._rcoefs * self._bkd.diag(self._acoefs @ sol)
            - (self._rcoefs * sol) * self._acoefs.T
        ).T

    def _param_jacobian(self, sol: Array) -> Array:
        jac_r = self._bkd.diag(sol) - sol * self._bkd.diag(self._acoefs @ sol)
        jac_a_rows = -(self._rcoefs * sol)[:, None] * sol[None, :]
        jac_a = self._bkd.zeros((3, 9))
        for ii in range(3):
            jac_a[ii, 3 * ii : 3 * (ii + 1)] = jac_a_rows[ii]
        jac = self._bkd.hstack((jac_r, jac_a))
        return jac

    def nvars(self) -> int:
        return (self.nstates() + 1) * self.nstates()

    def _initial_param_jacobian(self) -> Array:
        return self._bkd.zeros((self.nstates(), self.nvars()))


class LotkaVolterraModel(TransientAdjointModel):
    def __init__(
        self,
        init_time: float,
        final_time: float,
        deltat: float,
        time_residual_cls: TimeIntegratorNewtonResidual,
        functional: TransientAdjointFunctional = None,
        newton_solver: NewtonSolver = None,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        self._residual = LotkaVolterraResidual(backend)
        super().__init__(
            init_time,
            final_time,
            deltat,
            time_residual_cls(self._residual),
            functional,
            newton_solver,
            backend=backend,
        )

    def nvars(self) -> int:
        if not hasattr(self, "_functional"):
            return self._residual.nvars()
        return (
            self._functional.nunique_functional_params()
            + self._residual.nvars()
        )

    def get_initial_condition(self) -> Array:
        return self._bkd.array([0.3, 0.4, 0.3])
