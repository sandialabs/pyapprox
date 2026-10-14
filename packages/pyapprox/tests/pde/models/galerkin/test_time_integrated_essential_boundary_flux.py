"""The time-integrated essential-boundary flux: discrete conservation.

1D heat equation on [0, 1], P1 on 10 cells,

    u_t = (kappa(x) u')' + f(x, t),   kappa = exp(W p) (log-KLE),
    f = (1 + x)(1 + sin(5 t)),

with u(0, t) = 0 and a moving right end u(1, t) = 0.5 sin(4 t), from
u(x, 0) = 0.3 sin(pi x), over [0, 0.1] with dt = 0.002 (stable for
forward Euler).

Summed over every row, each step's non-essential rows vanish at the
solution, and the interpolant of 1 annihilates the diffusion term. So
with unit weights on both ends the flux functional must equal, for
every scheme with a step-residual rule,

    Q_total = integral u_h(T) - integral u_h(0)
              - sum_j w_j * 1.5 (1 + sin(5 t_j)),

the change of total content minus the source integrated by the
scheme's own rule; integral (1 + x) dx = 1.5, and the content integrals
use the trapezoid rule on the nodes, exact for P1. None of these terms
is computed by the assembly.
"""

import pickle
from typing import Any, Tuple

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np

from pyapprox.ode.config import TimeIntegrationConfig
from pyapprox.ode.mass_matrix import IdentityMassMatrix
from pyapprox.ode.time_quadrature import (
    OneStepMassTelescope,
    StepResidualDeclaration,
    right_rectangle_quadrature,
)
from pyapprox.pde.boundary import BoundarySignal
from pyapprox.pde.constitutive.coefficient_functions import (
    NodalFieldDiffusion,
    TimeDependent,
)
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.field_maps.transformed import (
    TransformedFieldMap,
    _ExpTransform,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.galerkin.time_integration.galerkin_model import GalerkinModel
from pyapprox.pde.models.galerkin import (
    GalerkinParameterizedResidual,
    GalerkinSteadyForwardModel,
    GalerkinTransientForwardModel,
    TimeIntegratedEssentialBoundaryFluxFunctional,
)
from pyapprox.pde.parameterizations.galerkin_advection_diffusion import (
    AdvectionDiffusionParameterization,
)
from pyapprox.util.backends.numpy import NumpyBkd
from tests._helpers.adjoint_checks import NumpyArray

_PARAM = np.array([[0.4], [-0.3], [0.2]])
_ONE_STEP_METHODS = [
    "backward_euler",
    "forward_euler",
    "crank_nicolson",
    "implicit_midpoint",
]


def _forcing(coords: Any, time: float) -> Any:
    return (1.0 + coords[0]) * (1.0 + np.sin(5.0 * time))


def _right_value(coords: Any, time: float) -> Any:
    return np.full(coords.shape[1], 0.5 * np.sin(4.0 * time))


def _right_rate(coords: Any, time: float) -> Any:
    return np.full(coords.shape[1], 2.0 * np.cos(4.0 * time))


def _log_kle_map(bkd: NumpyBkd, basis: LagrangeBasis[NumpyArray]) -> Any:
    coords = bkd.to_numpy(basis.dof_coordinates())[0]
    modes = np.stack(
        [0.4 * np.sin((k + 1) * np.pi * coords) / (k + 1) for k in range(3)],
        axis=1,
    )
    kle = MeshKLEFieldMap(
        bkd, bkd.asarray(np.zeros(coords.shape[0])), bkd.asarray(modes)
    )
    exp = _ExpTransform(bkd)
    return TransformedFieldMap(kle, exp, exp, bkd, transform_deriv2=exp)


class _Problem:
    """The heat problem, its parameterization and both end DOFs."""

    def __init__(self, bkd: NumpyBkd) -> None:
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        self.basis = LagrangeBasis(mesh, degree=1)
        self.physics = AdvectionDiffusionReaction(
            basis=self.basis,
            diffusivity=NodalFieldDiffusion(self.basis),
            bkd=bkd,
            forcing=TimeDependent(_forcing),
        )
        right = BoundarySignal(
            TimeDependent(_right_value), [TimeDependent(_right_rate)]
        )
        self.system: GalerkinSystem[NumpyArray] = compose_galerkin_system(
            self.physics,
            [
                DirichletBC(self.basis, "left", 0.0, bkd),
                DirichletBC(self.basis, "right", right, bkd),
            ],
        )
        self.param_obj = AdvectionDiffusionParameterization(
            self.physics, diffusivity_map=_log_kle_map(bkd, self.basis), bkd=bkd
        )
        self.ends = bkd.concatenate(
            [self.basis.get_dofs("left"), self.basis.get_dofs("right")]
        )
        x = bkd.to_numpy(self.basis.dof_coordinates())[0]
        self.init_state = bkd.asarray(0.3 * np.sin(np.pi * x))
        self.bkd = bkd

    def functional(
        self, weights: Any = None
    ) -> TimeIntegratedEssentialBoundaryFluxFunctional[NumpyArray]:
        bkd = self.bkd
        return TimeIntegratedEssentialBoundaryFluxFunctional(
            GalerkinParameterizedResidual(
                self.system.spatial_operator(), self.param_obj
            ),
            self.system,
            self.ends,
            bkd.ones((1, 2)) if weights is None else weights,
        )

    def model(
        self,
        method: str,
        functional: TimeIntegratedEssentialBoundaryFluxFunctional[NumpyArray],
    ) -> GalerkinTransientForwardModel[NumpyArray]:
        return GalerkinTransientForwardModel(
            self.system,
            self.param_obj,
            self.init_state,
            self.config(method),
            self.bkd,
            functional=functional,
        )

    def config(
        self, method: str, lumped_mass: bool = False
    ) -> TimeIntegrationConfig[Any]:
        return TimeIntegrationConfig(
            method=method,
            init_time=0.0,
            final_time=0.1,
            deltat=0.002,
            newton_tol=1e-13,
            newton_maxiter=20,
            lumped_mass=lumped_mass,
            verbosity=0,
        )

    def content(self, column: NumpyArray) -> float:
        """integral of u_h over [0, 1] (trapezoid on the nodes, exact)."""
        x = self.bkd.to_numpy(self.basis.dof_coordinates())[0]
        order = np.argsort(x)
        return float(np.trapezoid(self.bkd.to_numpy(column)[order], x[order]))


def _solve(problem: _Problem, method: str) -> Tuple[Any, NumpyArray, NumpyArray, Any]:
    functional = problem.functional()
    model = problem.model(method, functional)
    param = problem.bkd.asarray(_PARAM)
    traj, times = model.forward_solve(param)
    quadrature = model.last_integrator().time_residual().trajectory_quadrature(times)
    return functional, traj, param, quadrature


def _balance(problem: _Problem, traj: NumpyArray, quadrature: Any) -> float:
    """Content change minus the source integrated by the scheme's rule.

    Lumping keeps the mass row sums, so the content is integral u_h
    whether or not the scheme lumps.
    """
    bkd = problem.bkd
    rule = quadrature.step_residual_rule()
    assert rule is not None
    source = sum(
        bkd.to_float(rule.weights()[jj])
        * 1.5
        * (1.0 + np.sin(5.0 * bkd.to_float(rule.sample_times()[jj])))
        for jj in range(rule.nsamples())
    )
    return problem.content(traj[:, -1]) - problem.content(traj[:, 0]) - source


class TestDiscreteConservation:
    @pytest.mark.parametrize("method", _ONE_STEP_METHODS)
    def test_total_flux_balances_content_and_source(
        self, numpy_bkd: NumpyBkd, method: str
    ) -> None:
        bkd = numpy_bkd
        problem = _Problem(bkd)
        functional, traj, param, quadrature = _solve(problem, method)
        bkd.assert_allclose(
            functional(traj, param),
            bkd.asarray([[_balance(problem, traj, quadrature)]]),
            rtol=1e-10,
        )

    def test_lumped_scheme_balances_with_its_own_mass(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """Forward Euler with a lumped mass: the functional takes the
        lumped mass from the scheme, so the balance still holds. The
        parameterized model has no lumping option, so the plain model
        solves with the parameters applied directly."""
        bkd = numpy_bkd
        problem = _Problem(bkd)
        param = bkd.asarray(_PARAM)
        problem.param_obj.apply(param[:, 0])
        model = GalerkinModel(problem.system, bkd)
        traj, times = model.solve_transient(
            problem.init_state, problem.config("forward_euler", lumped_mass=True)
        )
        quadrature = (
            model.last_integrator().time_residual().trajectory_quadrature(times)
        )
        functional = problem.functional()
        functional.set_time_quadrature(quadrature)
        bkd.assert_allclose(
            functional(traj, param),
            bkd.asarray([[_balance(problem, traj, quadrature)]]),
            rtol=1e-10,
        )

    def test_integrated_steady_flux_breaks_conservation(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """Wrong-model control: integrating the steady flux -F_E over
        time drops M u_t on the end rows, and misses the balance by far
        more than the solver tolerance."""
        bkd = numpy_bkd
        problem = _Problem(bkd)
        functional, traj, param, quadrature = _solve(problem, "backward_euler")
        rule = quadrature.step_residual_rule()
        assert rule is not None
        residual = GalerkinParameterizedResidual(
            problem.system.spatial_operator(), problem.param_obj
        )
        stages = rule.sample(traj)
        steady_integrated = 0.0
        for jj in range(rule.nsamples()):
            rows = residual.residual(
                stages[:, jj],
                bkd.to_float(rule.sample_times()[jj]),
                param[:, 0],
            )
            steady_integrated -= bkd.to_float(
                rule.weights()[jj] * bkd.sum(rows[problem.ends])
            )
        balanced = bkd.to_float(functional(traj, param)[0, 0])
        assert abs(steady_integrated - balanced) > 1e-3


class TestSchemeRequirements:
    def test_heun_is_refused_at_model_construction(self, numpy_bkd: NumpyBkd) -> None:
        problem = _Problem(numpy_bkd)
        with pytest.raises(ValueError, match="no step-residual rule"):
            problem.model("heun", problem.functional())

    def test_steady_model_rejects_it(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        problem = _Problem(bkd)
        with pytest.raises(TypeError, match="transient"):
            GalerkinSteadyForwardModel(
                problem.system.steady_snapshot(0.0),
                problem.param_obj,
                problem.init_state,
                bkd,
                functional=problem.functional(),  # type: ignore[arg-type]
            )

    def test_value_needs_an_injected_rule(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        problem = _Problem(bkd)
        traj = bkd.zeros((problem.system.nstates(), 3))
        with pytest.raises(RuntimeError, match="time quadrature not set"):
            problem.functional()(traj, bkd.asarray(_PARAM))

    def test_refuses_a_scheme_mass_without_its_unconstrained_rows(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """The functional reads the essential rows of the scheme's own
        mass; a scheme whose mass cannot give them is refused rather than
        paired with some other mass."""
        bkd = numpy_bkd
        problem = _Problem(bkd)
        quadrature = right_rectangle_quadrature(
            bkd.asarray([0.0, 0.1, 0.2]),
            bkd,
            StepResidualDeclaration(
                OneStepMassTelescope(),
                IdentityMassMatrix(problem.system.nstates(), bkd),
            ),
        )
        with pytest.raises(TypeError, match="does not keep the mass"):
            problem.functional().set_time_quadrature(quadrature)


class TestValidation:
    def test_rejects_bad_arguments(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        problem = _Problem(bkd)
        residual = GalerkinParameterizedResidual(
            problem.system.spatial_operator(), problem.param_obj
        )
        with pytest.raises(ValueError, match="not essential"):
            TimeIntegratedEssentialBoundaryFluxFunctional(
                residual,
                problem.system,
                bkd.asarray([3], dtype=bkd.int64_dtype()),
                bkd.ones((1, 1)),
            )
        with pytest.raises(ValueError, match="weights must have shape"):
            TimeIntegratedEssentialBoundaryFluxFunctional(
                residual, problem.system, problem.ends, bkd.ones((1, 3))
            )
        other = _Problem(bkd)
        with pytest.raises(ValueError, match="share one spatial operator"):
            TimeIntegratedEssentialBoundaryFluxFunctional(
                residual, other.system, problem.ends, bkd.ones((1, 2))
            )

    def test_rejects_a_trajectory_of_the_wrong_length(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        bkd = numpy_bkd
        problem = _Problem(bkd)
        functional, traj, param, _ = _solve(problem, "backward_euler")
        with pytest.raises(ValueError, match="times but the injected rule"):
            functional(traj[:, :-1], param)


class TestPickle:
    def test_round_trip_keeps_the_value(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        problem = _Problem(bkd)
        functional, traj, param, _ = _solve(problem, "crank_nicolson")
        restored = pickle.loads(pickle.dumps(functional))
        bkd.assert_allclose(restored(traj, param), functional(traj, param))
