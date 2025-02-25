from typing import Tuple
import math

from scipy.special import beta as beta_fn

from pyapprox.util.linearalgebra.linalgbase import Array, LinAlgMixin
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.pde.collocation.adjoint_models import (
    TransientAdjointFunctional,
    AdjointFunctional,
)
from pyapprox.pde.collocation.timeintegration import (
    TimeIntegratorNewtonResidual,
)
from pyapprox.pde.collocation.solvers import (
    SteadyAdjointCollocationModel,
    TransientAdjointCollocationModel,
)
from pyapprox.pde.collocation.newton import (
    NewtonSolver,
    ParameterizedNewtonResidualMixin,
)
from pyapprox.pde.collocation.functions import (
    ScalarFunction,
    ConstantScalarFunction,
    VectorFunction,
    ScalarSolution,
    ScalarKLEFunction,
    ZeroScalarFunction,
    VectorFunctionFromCallable,
    ScalarFunctionFromCallable,
    TransientScalarFunctionFromCallable,
    TransientVectorFunctionFromCallable,
    ConstantVectorFunction,
    nabla,
    ScalarPeriodicReiszGaussianRandomField,
)
from pyapprox.pde.collocation.boundaryconditions import (
    DirichletBoundaryFromOperator,
    RobinBoundaryFromOperator,
    ConstantRobinBoundary,
    ConstantDirichletBoundary,
    PeriodicBoundary,
)
from pyapprox.pde.collocation.physics import (
    Physics,
    AdvectionDiffusionReactionPhysics,
    TransientPhysicsNewtonResidualMixin,
    SteadyPhysicsNewtonResidualMixin,
    TransientShallowWavePhysics,
    FitzHughNagumoPhysics,
    ShallowShelfVelocityPhysics,
    TransientBurgersPhysics1D,
)
from pyapprox.pde.collocation.mesh import (
    ChebyshevCollocationMesh1D,
    ChebyshevCollocationMesh2D,
)
from pyapprox.pde.collocation.basis import (
    ChebyshevCollocationBasis1D,
    ChebyshevCollocationBasis2D,
    OrthogonalCoordinateCollocationBasis,
)
from pyapprox.pde.collocation.mesh_transforms import (
    ScaleAndTranslationTransform1D,
    ScaleAndTranslationTransform2D,
)


class TransientSolutionTimeSnapshotFunctional(TransientAdjointFunctional):
    """Return all the states at one time step"""

    def __init__(
        self, model: TransientAdjointCollocationModel, timestep_idx: int
    ):
        self._model = model
        self._timestep_idx = timestep_idx

    def nqoi(self) -> int:
        return self._model._basis.mesh().nmesh_pts()

    def nparams(self) -> int:
        self._model.nvars()

    def nstates(self) -> int:
        return self._model._basis.mesh().nmesh_pts()

    def nunique_functional_params(self) -> int:
        return 0

    def _value(self, sol: Array) -> Array:
        return sol[:, self._timestep_idx]


class SteadySolutionFunctional(AdjointFunctional):
    """Return all the states."""

    def __init__(self, model: SteadyAdjointCollocationModel):
        self._model = model

    def nqoi(self) -> int:
        return self._model._basis.mesh().nmesh_pts()

    def nparams(self) -> int:
        self._model.nvars()

    def nstates(self) -> int:
        return self._model._basis.mesh().nmesh_pts()

    def nunique_functional_params(self) -> int:
        return 0

    def _value(self, sol: Array) -> Array:
        return sol


class ParameterizedDiffusionPhysics(
    AdvectionDiffusionReactionPhysics, ParameterizedNewtonResidualMixin
):
    def __init__(
        self,
        basis: OrthogonalCoordinateCollocationBasis,
        kle_nvars: int,
        kle_sigma: float,
        kle_lenscale: float,
        kle_mean_field: float,
    ):
        self._nvars = kle_nvars
        diffusion = self._setup_diffusion(
            basis, kle_sigma, kle_lenscale, kle_mean_field
        )
        super().__init__(
            ConstantScalarFunction(basis, 1.0), diffusion, None, None
        )

    def _setup_diffusion(
        self,
        basis: OrthogonalCoordinateCollocationBasis,
        kle_sigma: float,
        kle_lenscale: float,
        kle_mean_field: float,
    ):
        return ScalarKLEFunction(
            basis,
            kle_lenscale,
            self.nvars(),
            sigma=kle_sigma,
            mean_field=ConstantScalarFunction(basis, kle_mean_field, 1),
            ninput_funs=1,
            use_log=True,
        )

    def set_param(self, param: Array):
        self._param = param
        self._diffusion.set_param(param)

    def nvars(self) -> int:
        return self._nvars


class SteadyParameterizedDiffusionPhysics(
    ParameterizedDiffusionPhysics,
    SteadyPhysicsNewtonResidualMixin,
):
    pass


class ParameterizedDiffusionFixedAdvectionPhysics(
    AdvectionDiffusionReactionPhysics, ParameterizedNewtonResidualMixin
):
    def __init__(
        self,
        forcing: ScalarFunction,
        velocity_field: VectorFunction,
        nvars: int = 3,
        sigma: float = 0.1,
        lenscale: float = 0.1,
        mean_field: float = -2.0,
        use_quadrature: bool = True,
    ):
        self._nvars = nvars
        self._sigma = sigma
        self._mean_field = mean_field
        self._lenscale = lenscale
        self._use_quadrature = use_quadrature
        diffusion = self._setup_diffusion(forcing.basis())
        super().__init__(forcing, diffusion, None, velocity_field)

    def _setup_diffusion(self, basis: OrthogonalCoordinateCollocationBasis):
        return ScalarKLEFunction(
            basis,
            self._lenscale,
            self.nvars(),
            sigma=self._sigma,
            mean_field=ConstantScalarFunction(basis, self._mean_field, 1),
            ninput_funs=1,
            use_log=True,
            use_quadrature=self._use_quadrature,
        )

    def set_param(self, param: Array):
        self._param = param
        self._diffusion.set_param(param)

    def nvars(self) -> int:
        return self._nvars


class SteadyParameterizedDiffusionFixedAdvectionPhysics(
    ParameterizedDiffusionFixedAdvectionPhysics,
    SteadyPhysicsNewtonResidualMixin,
):
    pass


class PyApproxPaperAdvectionDiffusionKLEInversionModel(
    SteadyAdjointCollocationModel
):
    def __init__(
        self,
        nmesh_pts_1d: Tuple = [21, 21],
        nvars: int = 3,
        sigma: float = 0.1,
        lenscale: float = 0.1,
        mean_field: float = 0.0,
        source_amp: float = 100.0,
        source_loc: Tuple = [0.25, 0.75],
        source_scale: float = 0.1,
        newton_solver: NewtonSolver = None,
        functional: AdjointFunctional = None,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        self._nmesh_pts_1d = backend.asarray(nmesh_pts_1d, dtype=int)
        self._nvars = nvars
        self._sigma = sigma
        self._mean_field = mean_field
        self._lenscale = lenscale
        self._source_amp = source_amp
        self._source_loc = backend.asarray(source_loc)
        self._source_scale = source_scale
        super().__init__(newton_solver, functional, backend)

        # original paper defiend velocity field as [1, 0] everywhere
        # however the code used to produce that paper defined flux
        # with the wrong sign and the flux did not account for the velocity
        # to make field look similar to paper we set velocity to 0.01 here
        # To recover old paper, in collocation.physics.py
        # residual = div(self._diffusion * nabla(sol))
        #     - div(sol *self._velocity_field) + self._forcing
        # and flux=self._diffusion * nabla(sol)

    def jacobian_implemented(self) -> bool:
        return True

    def apply_hessian_implemented(self) -> bool:
        return self._bkd.hvp_implemented()

    def nvars(self) -> int:
        return self._physics.nvars()

    def setup_physics(self):
        self._nominal_val = 0.0
        self.setup_velocity()
        self.setup_forcing()
        self._physics = SteadyParameterizedDiffusionFixedAdvectionPhysics(
            self._forcing,
            self._vel_field,
            self._nvars,
            self._sigma,
            self._lenscale,
            self._mean_field,
            use_quadrature=False,
        )

    def setup_basis(self):
        Lx, Ly = 1, 1
        bounds = self._bkd.array([0, Lx, 0, Ly])
        transform = ScaleAndTranslationTransform2D(
            [-1, 1, -1, 1], bounds, self._bkd
        )
        mesh = ChebyshevCollocationMesh2D(self._nmesh_pts_1d, transform)
        self._basis = ChebyshevCollocationBasis2D(mesh)

    def setup_velocity(self):
        self._vel_field = VectorFunctionFromCallable(
            self._basis,
            1,
            self._basis.nphys_vars(),
            lambda x: self._bkd.stack(
                (
                    0.01 * self._bkd.ones(x.shape[1]),
                    self._bkd.zeros(x.shape[1]),
                ),
                axis=1,
            ),
        )

    def _gaussian_forcing(self, xx: Array) -> Array:
        return self._source_amp * self._bkd.exp(
            -self._bkd.sum(
                (xx - self._source_loc[:, None]) ** 2 / self._source_scale**2,
                axis=0,
            )
        )

    def setup_forcing(self):
        self._forcing = ScalarFunctionFromCallable(
            self._basis,
            self._gaussian_forcing,
            ninput_funs=1,
        )

    def get_initial_iterate(self) -> ScalarFunction:
        return ConstantScalarFunction(
            self._basis, self._nominal_val, ninput_funs=1
        ).get_flattened_values()

    def setup_boundaries(self):
        # make flux depend on solution's value relative to a nominal_val
        # beta * flux(u(x)) @ n = u(x) - nominal_val
        # beta * flux(u(x)) @ n - u(x) = nominal_val
        # beta * flux(u(x)) @ n + alpha * u(x) = nominal_val, alpha = -1

        # Conservative rules are written as
        # du/dt = -div(F) + g for flux F.
        bndrys = []
        alpha, beta = 0.1, 1.0
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
        self._physics.set_boundaries(bndrys)

    def forward_solve(self, sample: Array) -> Tuple[Array, Array]:
        self._adjoint_solver.set_initial_iterate(self.get_initial_iterate())
        super().forward_solve(sample)
        return self._sols


class TransientParameterizedDiffusionFixedAdvectionPhysics(
    ParameterizedDiffusionFixedAdvectionPhysics,
    TransientPhysicsNewtonResidualMixin,
):
    pass


class TransientDiffusionAdvectionModel(TransientAdjointCollocationModel):
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
        super().__init__(
            init_time,
            final_time,
            deltat,
            time_residual_cls,
            newton_solver,
            functional,
            backend,
        )

    def jacobian_implemented(self) -> bool:
        return True

    def nvars(self) -> int:
        return self._physics.nvars()

    def setup_physics(self):
        self._nominal_val = 0.0
        self.setup_velocity()
        self.setup_forcing()
        self._physics = TransientParameterizedDiffusionFixedAdvectionPhysics(
            self._forcing,
            velocity_field=self._vel_field,
        )

    def setup_basis(self):
        Lx, Ly = 1, 1
        bounds = self._bkd.array([0, Lx, 0, Ly])
        transform = ScaleAndTranslationTransform2D(
            [-1, 1, -1, 1], bounds, self._bkd
        )
        mesh = ChebyshevCollocationMesh2D([12, 12], transform)
        self._basis = ChebyshevCollocationBasis2D(mesh)

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

    def forward_solve(self, sample: Array) -> Tuple[Array, Array]:
        sols, times = super().forward_solve(sample)
        return (
            self._sols.reshape(
                (self._basis.mesh().nmesh_pts(), self._times.shape[0])
            ),
            self._times,
        )


class ShallowWaterWaveModel(TransientAdjointCollocationModel):
    # TODO CONVERT TO NEW paramerterizedresidual API and TEST
    def setup_physics(self):
        self.setup_bed()
        self._physics = TransientShallowWavePhysics(self._bed)

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


class ParameterizedFitzHughNagumoPhysics(FitzHughNagumoPhysics):
    def set_param(self, param: Array):
        self._param = param
        self.set_coefficients(param)

    def nvars(self) -> int:
        return 4


class TransientParameterizedFitzHughNagumoPhysics(
    ParameterizedFitzHughNagumoPhysics,
    TransientPhysicsNewtonResidualMixin,
):
    pass


class FitzHughNagumoModel(TransientAdjointCollocationModel):
    def setup_physics(self):
        self.setup_forcing()
        self._physics = TransientParameterizedFitzHughNagumoPhysics(
            self._basis, self._forcing
        )

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
                                lambda x, time: (time < self._final_time / 4)
                                * (x[0] * 0 + 1),
                                ninput_funs=self._physics.ncomponents(),
                            ),
                            component_id * self._basis.mesh().nmesh_pts(),
                        )
                    )
                else:
                    bndrys.append(
                        ConstantRobinBoundary(mesh_bndry, 0, alpha, beta, 0, 0)
                    )
        self._physics.set_boundaries(bndrys)

    def setup_basis(self):
        Lx, Ly = 2.5, 2.5
        self._bounds = self._bkd.array([0, Lx, 0, Ly])
        transform = ScaleAndTranslationTransform2D(
            [-1, 1, -1, 1], self._bounds, self._bkd
        )
        mesh = ChebyshevCollocationMesh2D([15, 15], transform)
        self._basis = ChebyshevCollocationBasis2D(mesh)

    def _beta_function(self, shapes, x):
        a0, b0, a1, b1 = shapes
        const = 1.0 / beta_fn(a0, b0) / beta_fn(a1, b1)
        Lx, Ly = self._bounds[1::2]
        return (
            (x[0] / Lx) ** (a0 - 1)
            * (1 - x[0] / Lx) ** (b0 - 1)
            * (x[1] / Ly) ** (a1 - 1)
            * (1 - x[1] / Ly) ** (b1 - 1)
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
                    x[0] * 0,
                ),
                axis=1,
            ),
        )

    def nvars(self) -> int:
        return self.physics.nvars()

    def get_initial_condition(self):
        init_cond = ConstantVectorFunction(
            self._basis,
            2,
            2,
            [1.0, 0.0],
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


class ParameterizedShallowShelfVelocityPhysics(
    ShallowShelfVelocityPhysics, ParameterizedNewtonResidualMixin
):
    def __init__(
        self,
        depth: ScalarFunction,
        bed: ScalarFunction,
        A: float,
        rho: float,
        friction_lenscale: float,
    ):
        friction = self._setup_friction(depth.basis(), friction_lenscale)
        super().__init__(depth, bed, friction, A, rho)

    def _setup_friction(
        self, basis: OrthogonalCoordinateCollocationBasis, lenscale: float
    ):
        friction = ScalarKLEFunction(
            basis,
            lenscale,
            self.nvars(),
            sigma=1,
            mean_field=ConstantScalarFunction(basis, math.log(1000), 1),
            ninput_funs=basis.mesh().nphys_vars(),
            use_log=True,
        )
        return friction

    def nvars(self) -> int:
        return 10

    def set_param(self, param: Array):
        self._param = param
        self._friction.set_param(param)

    def _param_jacobian(self, sol: Array) -> Array:
        # use chain rule for exp(g(p)) = exp(g(p))g'(p)
        friction_vals = self._friction.get_values()[:, None]
        eig_vecs = self._friction._kle._eig_vecs
        jac = self._bkd.vstack(
            (friction_vals * eig_vecs, friction_vals * eig_vecs)
        )
        # must set jac to zero on boundaries since
        # they do not depend on parameter. Because of nature of this residual
        # this can be done by setting bndry elements of solution to zero
        # However, in general must do the following after jac is formed
        # for bndry in self._bndrys:
        #     jac[bndry._residual_bndry_idx] = 0.
        sol_copy = self._bkd.copy(sol)
        for bndry in self._bndrys:
            sol_copy[bndry._residual_bndry_idx] = 0.0
        jac = -sol_copy[:, None] * jac
        return jac

    def param_param_hvp(
        self, fwd_sol: Array, adj_sol: Array, vvec: Array
    ) -> Array:
        friction = self._friction.get_values()
        eigvecs = self._friction._kle._eig_vecs
        # stack kle quantities for each component
        eigvecs_stack = self._bkd.vstack((eigvecs, eigvecs))
        friction_stack = self._bkd.hstack((friction, friction))
        gvec = eigvecs_stack @ vvec
        # must set jac to zero on boundaries since
        # they do not depend on parameter. Because of nature of this residual
        # this can be done by setting bndry elements of solution to zero
        fwd_sol_copy = self._bkd.copy(fwd_sol)
        for bndry in self._bndrys:
            fwd_sol_copy[bndry._residual_bndry_idx] = 0.0
        hvp = (
            -(gvec * adj_sol * fwd_sol_copy * friction_stack)[None, :]
            @ eigvecs_stack
        )
        return hvp

    def _param_state_hvp(
        self, fwd_sol: Array, adj_sol: Array, wvec: Array
    ) -> Array:
        friction = self._friction.get_values()
        eigvecs = self._friction._kle._eig_vecs
        eigvecs_stack = self._bkd.vstack((eigvecs, eigvecs))
        friction_stack = self._bkd.hstack((friction, friction))
        # must set jac to zero on boundaries since
        # they do not depend on parameter. Because of nature of this residual
        # this can be done by setting bndry elements of friction to zero
        for bndry in self._bndrys:
            friction_stack[bndry._residual_bndry_idx] = 0.0
        return -((friction_stack * adj_sol)[:, None] * eigvecs_stack).T @ wvec

    def _state_param_hvp(
        self, fwd_sol: Array, adj_sol: Array, vvec: Array
    ) -> Array:
        friction = self._friction.get_values()
        eigvecs = self._friction._kle._eig_vecs
        eigvecs_stack = self._bkd.vstack((eigvecs, eigvecs))
        friction_stack = self._bkd.hstack((friction, friction))
        # must set jac to zero on boundaries since
        # they do not depend on parameter. Because of nature of this residual
        # this can be done by setting bndry elements of friction to zero
        for bndry in self._bndrys:
            friction_stack[bndry._residual_bndry_idx] = 0.0
        gvec = eigvecs_stack @ vvec
        return -friction_stack * gvec * adj_sol

    def state_state_hvp(
        self, fwd_sol: Array, adj_sol: Array, wvec: Array
    ) -> Array:
        # state_state_hvp is very challening to derive analytically for these
        # equations so we must rely on auto differentiation. However, we can
        # speed it up by apply jvp to the analytical expression
        # of the jacobian. Rather than hvp to the residual

        # The actual improvement is marginal for moderate problem sizes with
        # torch. However applying torch.hvp directly to model.__call__ is many
        # times slower than only using auto diff to compute state_state_hvp,
        # state_param_jvp etc.
        self(fwd_sol)
        return self._bkd.jvp(
            lambda x: (adj_sol[None, :] @ self.jacobian(x))[0], fwd_sol, wvec
        )


class SteadyParameterizedShallowShelfVelocityPhysics(
    ParameterizedShallowShelfVelocityPhysics,
    SteadyPhysicsNewtonResidualMixin,
):
    pass


class SteadyShallowShelfModel2D(SteadyAdjointCollocationModel):
    def __init__(
        self,
        newton_solver: NewtonSolver = None,
        functional: TransientAdjointFunctional = None,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        super().__init__(newton_solver, functional, backend)

    def jacobian_implemented(self) -> bool:
        return True

    def setup_physics(self):
        # one part of hessian (state_state_hvp) still relies on auto diff
        # so use (default) self._apply_hessian_implemented = False

        self.setup_bed()
        self.setup_depth()
        self.setup_forcing()

        self._rho = 910
        self._A = 1e-16

        Lx, Ly = self._bounds[1::2]
        lenscale = (min(Lx, Ly) / 10,)

        self._physics = SteadyParameterizedShallowShelfVelocityPhysics(
            self._depth,
            self._bed,
            self._A,
            self._rho,
            lenscale,
        )

    def setup_basis(self):
        # distances in meters
        Lx, Ly = 100 * 1000, 100 * 1000
        self._bounds = self._bkd.asarray([0, Lx, 0, Ly])
        transform = ScaleAndTranslationTransform2D(
            [-1, 1, -1, 1], self._bounds, self._bkd
        )
        # Note: Multiple picard iterations become more useful as the
        # number of mesh points increases.
        mesh = ChebyshevCollocationMesh2D([15, 15], transform)
        self._basis = ChebyshevCollocationBasis2D(mesh)

    def _bed_callable(self, xx: Array) -> Array:
        self._bed_mean = 10.0
        yy = (self.basis().mesh().trans().map_to_orthogonal(xx) + 1) / 2
        # Convention is 0 represents see level. So make bed positive
        # return self._bed_mean + self._bkd.cos(
        #     2*yy[0]*math.pi
        # )*self._bkd.sin(2*yy[1]*math.pi)

        return self._bed_mean - yy[1] * (1 - yy[1])

    def setup_bed(self):
        self._bed = ScalarFunctionFromCallable(
            self._basis,
            self._bed_callable,
            ninput_funs=self._basis.nphys_vars(),
        )

    def _beta_function(self, shapes, x):
        a0, b0, a1, b1 = shapes
        yy = (self.basis().mesh().trans().map_to_orthogonal(x) + 1) / 2
        const = 1.0 / beta_fn(a0, b0) / beta_fn(a1, b1)
        Lx, Ly = self._bounds[1::2]
        return 1 + (
            (yy[0]) ** (a0 - 1)
            * (1 - yy[0]) ** (b0 - 1)
            * (yy[1]) ** (a1 - 1)
            * (1 - yy[1]) ** (b1 - 1)
            * const
        )

    def _surface_callable(self, xx: Array) -> Array:
        # depth ar right boundary (fixed margin) must be very small, e.g. 1
        left_height, right_height = 1000, self._bed_mean + 2
        xprofile_coefs = self._bkd.array(
            [
                left_height,
                0.0,
                3 * (right_height - left_height) - 1,
                2 * (left_height - right_height) + 1,
            ]
        )
        yy = (self.basis().mesh().trans().map_to_orthogonal(xx) + 1) / 2
        scale = 0.9  # ratio of height on left boundary and top and
        # bottom edges relative to middle of boundary
        xprofile = (
            yy[0][:, None] ** self._bkd.arange(4)[None, :] @ xprofile_coefs
        )
        yprofile_coefs = self._bkd.array(
            [scale, 4 * (1 - scale), 4 * (scale - 1)]
        )
        yprofile = (
            yy[1][:, None] ** self._bkd.arange(3)[None, :] @ yprofile_coefs
        )
        # zz = self._bkd.linspace(0, 1, 101)
        # import matplotlib.pyplot as plt
        # plt.plot(zz,  zz[:, None] ** self._bkd.arange(3)[None, :]
        #          @ yprofile_coefs)
        # plt.show()
        vals = xprofile * yprofile
        return vals

    def setup_depth(self):
        # from functools import partial
        self._surface = ScalarFunctionFromCallable(
            self._basis,
            self._surface_callable,
            # partial(self._beta_function, [2, 2, 2, 2]),
            ninput_funs=self._basis.nphys_vars(),
        )
        self._depth = self._surface - self._bed
        Lx, Ly = self._bounds[1::2]
        if self._depth(self._bkd.array([Lx, Ly / 2])[:, None]) > 2.5:
            print(self._depth(self._bkd.array([Lx, Ly / 2])[:, None]))
            raise RuntimeError("depth at right boundary must be small")
        if self._bkd.min(self._depth.get_values()) <= 0:
            raise RuntimeError(
                "Depth was set to be negative {0}".format(
                    self._bkd.min(self._depth.get_values())
                )
            )

    def setup_forcing(self):
        self._velocity_forcing = None

    def _picard_iteration(self):
        iterate = self._bkd.ones(
            (self.physics().ncomponents() * self.basis().mesh().nmesh_pts())
        )
        self._adjoint_solver.set_initial_iterate(iterate)
        self.physics()._fix_strain_rate(
            ConstantScalarFunction(
                self.basis(), 1e3, ninput_funs=self.physics().ncomponents()
            )
        )
        # do not call iterate = self._adjoint_solver.forward_solve()
        # as it updates self._adjoint_solver._fwd_sol_param so
        # that new forward solution is not computed
        iterate = self._adjoint_solver._newton_solver.solve(iterate)
        npicard_iterations = 1
        for it in range(npicard_iterations):
            fixed_strain_rate = (
                self.physics()._get_strain_rate_from_solution_array(iterate)
            )
            self.physics()._fix_strain_rate(fixed_strain_rate)
            self._adjoint_solver.set_initial_iterate(self._bkd.copy(iterate))
            iterate = self._adjoint_solver._newton_solver.solve(iterate)
            # res_array = self._adjoint_solver._newton_solver._residual(
            #     iterate
            # )
            # res_norm = self._bkd.norm(res_array)
            # print("Picard Iter", it, "rnorm", res_norm)
        self.physics()._unfix_strain_rate()
        return iterate

    def _initial_iterate(self) -> Array:
        init_iterate = self._picard_iteration()
        # init_sol = self.physics().solution_from_array(init_iterate)
        # import matplotlib.pyplot as plt
        # from pyapprox.pde.collocation.functions import plot_vector_function
        # plot_vector_function(init_sol)
        # # plt.show()
        return init_iterate

    def set_param(self, param: Array):
        self._adjoint_solver.set_param(param)
        self._adjoint_solver.set_initial_iterate(self._initial_iterate())

    def setup_boundaries(self):
        bndry_funs = []
        # loop over all boundaries
        for (
            bndry_name,
            mesh_bndry,
        ) in (
            self._basis.mesh().get_boundaries().items()
        ):
            for component_id in range(self._physics.ncomponents()):
                if (
                    component_id == 0
                    and bndry_name == "left"
                    or (component_id == 1 and bndry_name in ["bottom", "top"])
                    or bndry_name == "right"
                ):
                    # set velocity at boundary to zero
                    bndry_funs.append(
                        DirichletBoundaryFromOperator(
                            mesh_bndry,
                            self._get_zerofun(),
                            component_id * self._basis.mesh().nmesh_pts(),
                        )
                    )
                elif bndry_name == "right":
                    bndry_funs.append(
                        RobinBoundaryFromOperator(
                            mesh_bndry,
                            (self._physics._rho * self._physics._g * 0.5)
                            * self._depth**2,
                            0.0,
                            1.0,
                            component_id * self._basis.mesh().nmesh_pts(),
                            component_id,
                        )
                    )
                else:
                    # Since we are setting vertical velocity v = sol[1] = 0
                    # at top and bottom boundary, we must set v = 0 or F(v)= 0
                    # to be zero on left and right boundary otherwise
                    # there will be an inconsistency in the boundary condition
                    # at the corners of the domain
                    bndry_funs.append(
                        RobinBoundaryFromOperator(
                            mesh_bndry,
                            self._get_zerofun(),
                            0.0,
                            1.0,
                            component_id * self._basis.mesh().nmesh_pts(),
                            component_id,
                        )
                    )
        self._physics.set_boundaries(bndry_funs)

    def _get_zerofun(self):
        return ZeroScalarFunction(
            self._basis,
            ninput_funs=self._physics.ncomponents(),
        )


def compute_shallow_shelf_surface():
    import sympy as sp

    # used to find cubic polynomial that satisfies four constraints
    a, b, c, d, e, f, x = sp.symbols(("a", "b", "c", "d", "e", "f", "x"))
    poly = a + b * x + c * x**2 + d * x**3
    result = sp.solve(
        [
            poly.subs(x, 0) - e,
            poly.subs(x, 1) - f,
            poly.diff(x, 1).subs(x, 0) - 0,
            poly.diff(x, 1).subs(x, 1) - 1,
        ],
        [a, b, c, d],
    )
    print(result)


class TransientViscousBurgers1DModel(TransientAdjointCollocationModel):
    def __init__(
        self,
        init_time: float,
        final_time: float,
        deltat: float,
        nmesh_pts_1d: int,
        viscosity: float,
        nkle_eigs: int,
        kle_sigma: float,
        kle_tau: float,
        kle_gamma: float,
        time_residual_cls: TimeIntegratorNewtonResidual,
        newton_solver: NewtonSolver = None,
        functional: TransientAdjointFunctional = None,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        self._nmesh_pts_1d = nmesh_pts_1d
        self._visc = viscosity
        self._nkle_eigs = nkle_eigs
        self._kle_sigma = kle_sigma
        self._kle_tau = kle_tau
        self._kle_gamma = kle_gamma
        super().__init__(
            init_time,
            final_time,
            deltat,
            time_residual_cls,
            newton_solver,
            functional,
            backend,
        )

    def jacobian_implemented(self) -> bool:
        return True

    def nvars(self) -> int:
        return self._init_cond.kle().nterms()

    def setup_physics(self):
        self.setup_initial_condition()
        self._physics = TransientBurgersPhysics1D(
            viscosity=ConstantScalarFunction(self._basis, self._visc),
            forcing=None,
        )

    def setup_basis(self):
        Lx = 1
        bounds = self._bkd.array([0, Lx])
        transform = ScaleAndTranslationTransform1D([-1, 1], bounds, self._bkd)
        mesh = ChebyshevCollocationMesh1D([self._nmesh_pts_1d], transform)
        self._basis = ChebyshevCollocationBasis1D(mesh)

    def setup_initial_condition(self):
        self._init_cond = ScalarPeriodicReiszGaussianRandomField(
            self._basis,
            self._nkle_eigs,
            self._kle_sigma,
            self._kle_tau,
            self._kle_gamma,
            False,
            1,
        )

    def set_param(self, param: Array):
        self._init_cond.set_param(param)
        super().set_param(param)

    def get_initial_condition(self):
        return self._init_cond.get_flattened_values()

    def setup_boundaries(self):
        bndrys = []
        boundaries = self._basis.mesh().get_boundaries()
        mesh_bndry = boundaries[list(boundaries.keys())[0]]
        bndry_pair_names_dict = mesh_bndry.names_of_boundary_pairs()
        for (
            bndry_name,
            mesh_bndry,
        ) in (
            self._basis.mesh().get_boundaries().items()
        ):
            if bndry_name not in bndry_pair_names_dict:
                continue
            partner_mesh_bndry = boundaries[bndry_pair_names_dict[bndry_name]]
            bndrys.append(
                PeriodicBoundary(mesh_bndry, partner_mesh_bndry, self._basis)
            )
        self._physics.set_boundaries(bndrys)

    def forward_solve(self, sample: Array) -> Tuple[Array, Array]:
        sols, times = super().forward_solve(sample)
        return (
            self._sols.reshape(
                (self._basis.mesh().nmesh_pts(), self._times.shape[0])
            ),
            self._times,
        )


class SteadyDarcy2DKLEModel(SteadyAdjointCollocationModel):
    def __init__(
        self,
        kle_nvars: int,
        kle_sigma: float,
        kle_lenscale: float,
        kle_mean_field: float,
        newton_solver: NewtonSolver = None,
        functional: AdjointFunctional = None,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        self._kle_nvars = kle_nvars
        self._kle_sigma = kle_sigma
        self._kle_lenscale = kle_lenscale
        self._kle_mean_field = kle_mean_field
        super().__init__(newton_solver, functional, backend)
        # PDE is linear so initial guess does not matter
        self._adjoint_solver.set_initial_iterate(
            self._bkd.zeros(self.basis().mesh().nmesh_pts())
        )

    def setup_physics(self):
        self._physics = SteadyParameterizedDiffusionPhysics(
            self.basis(),
            self._kle_nvars,
            self._kle_sigma,
            self._kle_lenscale,
            self._kle_mean_field,
        )

    def setup_basis(self):
        Lx, Ly = 1, 1
        bounds = self._bkd.array([0, Lx, 0, Ly])
        transform = ScaleAndTranslationTransform2D(
            [-1, 1, -1, 1], bounds, self._bkd
        )
        mesh = ChebyshevCollocationMesh2D([20, 20], transform)
        self._basis = ChebyshevCollocationBasis2D(mesh)

    def setup_boundaries(self):
        bndrys = []
        for (
            bndry_name,
            mesh_bndry,
        ) in (
            self._basis.mesh().get_boundaries().items()
        ):
            bndrys.append(ConstantDirichletBoundary(mesh_bndry, 0.0))
        self._physics.set_boundaries(bndrys)

    def velocity_field(self, sol: ScalarSolution):
        return nabla(-self.physics()._diffusion * sol)
