from pyapprox.util.linearalgebra.linalgbase import Array
from pyapprox.pde.collocation.adjoint_models import (
    TransientAdjointModel,
    TransientAdjointFunctional,
)
from pyapprox.pde.collocation.timeintegration import (
    TimeIntegratorNewtonResidual,
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
        init_depth = ScalarFunction(
            self._basis,
            (self._init_surface - self._bed).get_values(),
            ninput_funs=self._bed.ninput_funs(),
        )
        print(init_depth.get_values().min(), init_depth.get_values().max(), "A")
        init_cond.set_components(
            [init_depth]
            + [self._get_zerofun() for ii in range(self._basis.nphys_vars())]
        )
        return init_cond

    # def setup_dirichlet_boundaries(self):
    #     bndry_funs = []
    #     for bndry_name, mesh_bndry in self._basis.mesh.get_boundaries().items():
    #         for ii in range(1, self._basis.nphys_vars()+1):
    #             bndry_funs.append(
    #                 DirichletBoundaryFromOperator(
    #                     mesh_bndry,
    #                     self._zerofun,
    #                     ii * self._basis.mesh.nmesh_pts(),
    #                 )
    #             )
    #     return bndry_funs

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
                elif False:#component_id == 0:
                    bndry_funs.append(
                        RobinBoundaryFromOperator(
                            mesh_bndry,
                            self._get_zerofun(),
                            0.0,
                            1.0,
                            component_id * self._basis.mesh.nmesh_pts(),
                            component_id,
                        )
                    )
        return bndry_funs

    def set_param(self, param: Array):
        self._physics._bed.set_param(param)
