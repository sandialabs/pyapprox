from pyapprox.util.linearalgebra.linalgbase import Array
from pyapprox.pde.collocation.adjoint_models import (
    TransientAdjointModel,
    TransientAdjointFunctional,
)
from pyapprox.pde.collocation.timeintegration import (
    TimeIntegratorNewtonResidual,
    TransientNewtonResidual,
)
from pyapprox.pde.collocation.solvers import TransientPDE
from pyapprox.pde.collocation.newton import NewtonSolver
from pyapprox.pde.collocation.functions import (
    ScalarFunction,
    VectorFunction,
    ScalarKLEFunction,
    ZeroScalarFunction,
)
from pyapprox.pde.collocation.boundaryconditions import (
    DirichletBoundaryFromOperator,
    RobinBoundaryFromOperator,
)
from pyapprox.pde.collocation.physics import ShallowWaveEquation


class ShallowWaterWaveModel(TransientAdjointModel):
    def __init__(
        self,
        init_time: float,
        final_time: float,
        deltat: float,
        time_residual_cls: TimeIntegratorNewtonResidual,
        functional: TransientAdjointFunctional,
        bed: ScalarKLEFunction,
        init_surface: ScalarFunction,
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

        self._solver = TransientPDE(self._physics, newton_solver)
        self._solver.setup_time_integrator(
            time_residual_cls,
            init_time,
            final_time,
            deltat,
        )

    # TODO make adjoint model and this class consistent
    def _fwd_solve(self):
        init_sol = self.get_initial_condition()
        self._sols, self._times = self._solver.solve(init_sol)

    def nvars(self) -> int:
        # TODO MAKE THIS REQQUIRED FOR ALL PARAMETERIZED MODELS
        return self._bed._kle.nvars()

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
        return init_cond

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
        jac_r = self._bkd.diag(sol) - sol*self._bkd.diag(self._acoefs @ sol)
        jac_a_rows = -(self._rcoefs * sol)[:, None] * sol[None, :]
        jac_a = self._bkd.zeros((3, 9))
        for ii in range(3):
            jac_a[ii, 3*ii:3*(ii+1)] = jac_a_rows[ii]
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
        functional: TransientAdjointFunctional,
        newton_solver: NewtonSolver = None,
    ):
        self._residual = LotkaVolterraResidual(functional._bkd)
        super().__init__(
            init_time,
            final_time,
            deltat,
            time_residual_cls(self._residual),
            functional,
            newton_solver,
            backend=functional._bkd,
        )

    def nvars(self) -> int:
        return self._residual.nvars()

    def get_initial_condition(self) -> Array:
        return self._bkd.array([0.3, 0.4, 0.3])
