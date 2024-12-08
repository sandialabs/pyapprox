from typing import Tuple

from scipy.special import beta as beta_fn

from pyapprox.util.linearalgebra.linalgbase import Array, LinAlgMixin
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.pde.collocation.adjoint_models import (
    TransientAdjointFunctional,
    TransientAdjointModel,
)
from pyapprox.pde.collocation.timeintegration import (
    TimeIntegratorNewtonResidual,
    TransientNewtonResidual,
)
from pyapprox.pde.collocation.solvers import TransientAdjointCollocationModel
from pyapprox.pde.collocation.newton import NewtonSolver
from pyapprox.pde.collocation.functions import (
    ScalarFunction,
    ConstantScalarFunction,
    VectorFunction,
    ScalarKLEFunction,
    ZeroScalarFunction,
    VectorFunctionFromCallable,
    ScalarFunctionFromCallable,
    TransientScalarFunctionFromCallable,
    TransientVectorFunctionFromCallable,
    ConstantVectorFunction,
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
    FitzHughNagumo
)
from pyapprox.pde.collocation.mesh import ChebyshevCollocationMesh2D
from pyapprox.pde.collocation.basis import ChebyshevCollocationBasis2D
from pyapprox.pde.collocation.mesh_transforms import (
    ScaleAndTranslationTransform2D,
)


class TransientAdvectionDiffusionReactionModel(
    TransientAdjointCollocationModel
):
    def setup_physics(self):
        self._nominal_val = 0.0
        self.setup_diffusion()
        self.setup_velocity()
        self.setup_forcing()
        self._physics = AdvectionDiffusionReactionEquation(
            self._forcing,
            self._diffusion,
            reaction_op=None,
            velocity_field=self._vel_field,
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
            mean_field=ConstantScalarFunction(self._basis, -2.0, 1),
            ninput_funs=self._basis.mesh().nphys_vars() + 1,
            use_log=True,
        )

    def setup_velocity(self):
        self._vel_field = VectorFunctionFromCallable(
            self._basis,
            1,
            self._basis.nphys_vars(),
            # rotates in a circle
            # lambda x: 10 * self._bkd.stack((-(2*x[1]-1), 2*x[0]-1, axis=1))
            lambda x: self._bkd.stack(
                (
                    self._bkd.cos(4 * (x[0] - 2) + 2 * (4 * (x[1] - 2))),
                    2
                    * self._bkd.sin(
                        2 * (2 * x[0] - 2) - 2 * (2 * (2 * x[1] - 2))
                    ),
                ),
                axis=1,
            ),
        )

    def setup_forcing(self):
        # self._forcing = ConstantScalarFunction(self._basis, 0., 1)
        a0, b0, a1, b1 = 10, 15, 15, 5
        const = 1.0 / beta_fn(a0, b0) / beta_fn(a1, b1)
        # self._forcing = ScalarFunctionFromCallable(
        self._forcing = TransientScalarFunctionFromCallable(
            self._basis,
            lambda x, time: (time < self._final_time / 2)
            * x[0] ** a0
            * (1 - x[0]) ** b0
            * x[1] ** a1
            * (1 - x[1]) ** b1
            * const,
            ninput_funs=1,
        )

    def get_initial_condition(self):
        return ConstantScalarFunction(
            self._basis, self._nominal_val, ninput_funs=1
        ).get_flattened_values()
        # return ScalarFunctionFromCallable(
        #     self._basis,
        #     lambda x: self._bkd.prod(x**5*(1-x)**5, axis=0)*1e7,
        #     ninput_funs=1
        # ).get_flattened_values()

    def setup_boundaries(self):
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
        ) in (
            self._basis.mesh().get_boundaries().items()
        ):
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
                (self._basis.mesh().nmesh_pts(), self._times.shape[0])
            ),
            self._times,
        )


class ShallowWaterWaveModel(TransientAdjointCollocationModel):
    def setup_physics(self):
        self.setup_bed()
        self._physics = ShallowWaveEquation(self._bed)

    def setup_basis(self):
        Lx, Ly = 100, 200
        self._bounds = self._bkd.array([0, Lx, 0, Ly])
        transform = ScaleAndTranslationTransform2D(
            [-1, 1, -1, 1], self._bounds, self._bkd
        )
        mesh = ChebyshevCollocationMesh2D([30, 30], transform)
        self._basis = ChebyshevCollocationBasis2D(mesh)

    def _bed_callable(self, xx: Array) -> Array:
        # wave propagates faster in deeper water. Make bed increase in
        # elevation close to bottom and top boundaries so that it 0.1
        # at boundaries and 1.1 at midpoint off y domain
        xn = 1 / self._bounds[1::2, None] * xx
        return -1 + (-0.1 + xn[1] * (xn[1] - 1)) * (1 - 0.9 * xn[0])

    def setup_bed(self):
        self._bed = ScalarFunctionFromCallable(
            self._basis,
            self._bed_callable,
            ninput_funs=self._basis.nphys_vars() + 1,
        )

    def _beta_surface_callable(self, beta_shapes: Array, xx: Array) -> Array:
        a0, b0, a1, b1 = beta_shapes
        # The higher the shape values the higher basis orders need to be
        xn = 1 / self._bounds[1::2, None] * xx
        const0 = 1.0 / beta_fn(a0, b0)
        const1 = 1.0 / beta_fn(a1, b1)
        return (
            (
                xn[0] ** (a0 - 1)
                * (1 - xn[0]) ** (b0 - 1)
                * xn[1] ** (a1 - 1)
                * (1 - xn[1]) ** (b1 - 1)
            )
            * const0
            * const1
            / 20
        )

    def _init_surface_callable(self, xx: Array) -> Array:
        # beta_shapes0 = self._bkd.array([25, 20, 20, 20])
        # beta_shapes1 = self._bkd.array([5, 20, 20, 20])
        beta_shapes0 = self._param[:4]
        beta_shapes1 = self._param[4:]
        return self._beta_surface_callable(
            beta_shapes0, xx
        ) + self._beta_surface_callable(beta_shapes1, xx)

    def setup_init_surface(self):
        self._init_surface = ScalarFunctionFromCallable(
            self._basis,
            self._init_surface_callable,
            self._basis.nphys_vars() + 1,
        )

    def nvars(self) -> int:
        return 8

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

    def setup_boundaries(self):
        # bndrys_funs: list[BoundaryOperator],
        bndry_funs = []
        # loop over all boundaries
        for (
            bndry_name,
            mesh_bndry,
        ) in (
            self._basis.mesh().get_boundaries().items()
        ):
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
                            component_id * self._basis.mesh().nmesh_pts(),
                        )
                    )
                # elif component_id != 0 and bndry_name == "left":
                #     bndry_funs.append(
                #         RobinBoundaryFromOperator(
                #             mesh_bndry,
                #             self._get_zerofun(),
                #             -10,
                #             1.0,
                #             component_id * self._basis.mesh().nmesh_pts(),
                #             component_id,
                #         )
                #     )
        self._physics.set_boundaries(bndry_funs)

    def set_param(self, param: Array):
        self._param = param
        self.setup_init_surface()


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


class FitzHughNagumoModel(TransientAdjointCollocationModel):
    def setup_physics(self):
        self.setup_forcing()
        self._physics = FitzHughNagumo(self._basis, self._forcing)

    def setup_boundaries(self):
        bndrys = []
        alpha, beta = 0, 1.0
        # loop over solution components
        for (
            bndry_name,
            mesh_bndry,
        ) in (
            self._basis.mesh().get_boundaries().items()
        ):
            for component_id in range(self._physics.ncomponents()):
                if False:  # (component_id == 0 and bndry_name in ["left"]):
                    bndrys.append(
                        DirichletBoundaryFromOperator(
                            mesh_bndry,
                            TransientScalarFunctionFromCallable(
                                self._basis,
                                lambda x, time: (
                                    time < self._final_time/4
                                )*(x[0] * 0 + 1),
                                ninput_funs=self._physics.ncomponents(),
                            ),
                            component_id * self._basis.mesh().nmesh_pts(),
                        )
                    )
                else:
                    bndrys.append(
                        ConstantRobinBoundary(
                            mesh_bndry, 0, alpha, beta, 0, 0
                        )
                    )
        self._physics.set_boundaries(bndrys)

    def setup_basis(self):
        Lx, Ly = 2.5, 2.5
        self._bounds = self._bkd.array([0, Lx, 0, Ly])
        transform = ScaleAndTranslationTransform2D(
            [-1, 1, -1, 1], self._bounds, self._bkd
        )
        mesh = ChebyshevCollocationMesh2D([20, 20], transform)
        self._basis = ChebyshevCollocationBasis2D(mesh)

    def _beta_function(self, shapes, x):
        a0, b0, a1, b1 = shapes
        const = 1.0 / beta_fn(a0, b0) / beta_fn(a1, b1)
        Lx, Ly = self._bounds[1::2]
        return (
            (x[0]/Lx) ** (a0-1)
            * (1 - x[0]/Lx) ** (b0-1)
            * (x[1]/Ly) ** (a1-1)
            * (1 - x[1]/Ly) ** (b1-1)
            * const
        )

    def setup_forcing(self):
        shapes0 = 5, 10, 5, 10
        shapes1 = 10, 5, 10, 6
        Lx, Ly = self._bounds[1::2]
        self._forcing = TransientVectorFunctionFromCallable(
            self._basis,
            2,
            2,
            lambda x, time: self._bkd.stack(
                (
                    (time < self._final_time / 4)
                    * (
                        self._beta_function(shapes0, x)
                        + self._beta_function(shapes1, x)
                    ),
                    x[0] * 0
                ),
                axis=1
            )
        )

    def nvars(self) -> int:
        return 4

    def get_initial_condition(self):
        init_cond = ConstantVectorFunction(
            self._basis, 2, 2, [1., 0.],
        )
        # import numpy as np
        # np.random.seed(1)
        # init_cond = VectorFunctionFromCallable(
        #     self._basis, 2, 2, lambda x: self._bkd.stack(
        #         (
        #             self._bkd.array(np.random.uniform(-1, 1, x.shape[1])),
        #             0*x[0]
        #         ),
        #         axis=1
        #     ),
        # )
        return init_cond.get_flattened_values()

    def set_param(self, param: Array):
        self._param = param
        self._physics.set_coefficients(param)
