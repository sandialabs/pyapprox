"""Worked transient gradient + HVP example through the public function API.

The capstone of the galerkin adjoint stack: a 2D transient
advection-diffusion problem with a bilaplacian (Whittle-Matern)
log-diffusivity KLE, solved through ``galerkin_transient_system`` and
``TransientImplicitMap`` and driven entirely through ``ImplicitMapFunction``
(``__call__`` -> ``derivatives().jacobian`` -> ``derivatives().hvp``)
with a mass-weighted subdomain-average QoI, c^T y(T). Component tiers
below this (assemblies, engine terms, steppers, BC wrappers) each have
their own FD suites; this file validates the composed public surface:

- gradient and HVP vs DerivativeChecker FD for each parameterized
  coefficient (diffusivity/forcing/velocity/composite), BE and CN;
- the HVP symmetry identity <Hu, v> = <Hv, u> (FD-noise immune);
- the scalar QoI's adjoint row vs the final state's tangent-column
  Jacobian (adjoint and forward propagation must agree to 1e-10);
- a Tikhonov parameter term (nonzero direct dQ/dp and d2Q/dp2 —
  otherwise those functional pathways are untested) evaluated at p = 0
  where the parameterized forcing field VANISHES at the test point
  (the configuration where wrongly-Zeroed derivative paths surface);
- a time-integrated quadratic QoI, the method's own quadrature of
  y^T M y, plus the Tikhonov term;
- Robin boundary conditions, and the quasilinear kappa(u)*a(x) physics
  (genuine mixed state-parameter curvature, invisible to gradient-only
  tests);
- a reaction control modulated in time.

FD calibration: central differences. One-sided differences of the
gradient floor near 1e-6 relative for the HVP, at the bound itself, and
where that floor falls moves between platforms. Central differences of
every case floor at most at 4.4e-10 (HVP) and give Jacobian and HVP
error ratios at most 1.3e-8 and 5.2e-9, so the checks assert the V shape
(second order) with bounds 1e-6 on the ratios and 1e-8 on the HVP
minimum. The V's order is fitted against each curve's predicted
rounding (``predicted_rounding``), over the two decades above where the
error nears it, because the floor arrives anywhere from h = 1e-3 to 1e-5
by case and platform and a fit reaching it misreads the order (0.8 to
2.4 seen). Locally every case fits 2.000 to 2.004, and a relative defect
of 1e-5 in the Jacobian or the HVP fails. The symmetry identity carries
the tight validation.
"""

from typing import Any, Callable, List, Optional, Protocol, Tuple, runtime_checkable

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np

from pyapprox.implicitfunction.functional_terms import ParameterTermProtocol
from pyapprox.implicitfunction.implicit_map import (
    AdjointRows,
    ArrayParameters,
    ImplicitMapFunction,
    TangentColumns,
)
from pyapprox.implicitfunction.observation_terms import (
    StateEntries,
    WeightedEntries,
)
from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.implicitfunction.tikhonov_term import TikhonovTerm
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
    predicted_rounding,
)
from pyapprox.ode.observation_terms import EndpointTerm
from pyapprox.ode.step_plan import FixedStep
from pyapprox.ode.stepping_methods import RungeKutta
from pyapprox.ode.tableau import backward_euler, crank_nicolson
from pyapprox.ode.trajectory_terms import TrajectoryFunctional
from pyapprox.ode.transient_implicit_map import (
    FixedTrajectoryFunctional,
    ParameterFanOut,
    SummedFunctional,
    TrajectoryFunctionalSourceProtocol,
    TransientImplicitMap,
)
from pyapprox.pde.boundary.protocols import BoundaryConditionRole
from pyapprox.pde.constitutive.coefficient_functions import (
    CallableReaction,
    NodalFieldDiffusion,
    NodalFieldForcing,
    NodalFieldVelocity,
    TimeIndependent,
    TimeModulatedNodalFieldLinearReaction,
)
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.field_maps.modulation import PiecewiseLinearModulation
from pyapprox.pde.field_maps.transformed import TransformedFieldMap
from pyapprox.pde.galerkin.basis import LagrangeBasis, VectorLagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import (
    DirichletBC,
    RobinBC,
)
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh2D
from pyapprox.pde.galerkin.physics import (
    AdvectionDiffusionReaction,
    QuasilinearDiffusion,
)
from pyapprox.pde.galerkin.protocols.basis import GalerkinBasisProtocol
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.galerkin.time_integration.skfem_discretization import (
    skfem_discretization,
)
from pyapprox.pde.models.galerkin.transient_observations import (
    time_integrated_weighted_l2,
)
from pyapprox.pde.models.galerkin.transient_system import (
    GalerkinTransientSystem,
    galerkin_transient_system,
)
from pyapprox.pde.parameterizations.field_term import _FieldParameterizationTerm
from pyapprox.pde.parameterizations.galerkin_advection_diffusion import (
    FROM_FIELD,
    AdvectionDiffusionParameterization,
)
from pyapprox.pde.parameterizations.galerkin_quasilinear_diffusion import (
    create_quasilinear_diffusivity_parameterization,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.scipy_stack import ScipySparseLU
from pyapprox.util.linalg.symmetric_weight import ScipySparseWeight
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.adjoint_checks import NumpyArray
from tests._helpers.spde_fields import (
    spde_lognormal_field_map,
)

_N_KLE_MODES = 4
_N_FORCING_MODES = 2

_Function = ImplicitMapFunction[Any, NumpyArray, NumpyArray]
_Source = TrajectoryFunctionalSourceProtocol[NumpyArray, NumpyArray]


def _kappa(u: NumpyArray) -> NumpyArray:
    return 1.0 + u**2


def _kappa_deriv(u: NumpyArray) -> NumpyArray:
    return 2.0 * u


def _kappa_second_deriv(u: NumpyArray) -> NumpyArray:
    return np.full_like(u, 2.0)


def _method(name: str) -> RungeKutta[NumpyArray, NumpyArray]:
    tableau = {"backward_euler": backward_euler, "crank_nicolson": crank_nicolson}
    return RungeKutta(tableau[name](), Newton(atol=1e-12, rtol=0.0, maxiters=20))


def _steps() -> FixedStep:
    return FixedStep(0.02, (0.0, 0.1))


def _build_physics(
    bkd: NumpyBkd, cubic_reaction: bool = False, robin: bool = False
) -> Tuple[
    AdvectionDiffusionReaction[NumpyArray],
    GalerkinSystem[NumpyArray],
    LagrangeBasis[NumpyArray],
    VectorLagrangeBasis[NumpyArray],
]:
    """2D unit-square ADR: nodal diffusivity/forcing, constant (1, 0)
    velocity, homogeneous Dirichlet. The optional CUBIC reaction is the
    only state-nonlinear term, and unlike the component tiers'
    quadratic variant its second derivative R'' = 6u is
    state-dependent — per-step staleness in the second-adjoint RHS
    would surface here and nowhere else. With ``robin`` the left/right
    Dirichlet conditions become Robin conditions (varying-coefficient
    alpha(y) on the left, constant on the right): the adjoint then must
    handle a mixed essential/natural set where the Robin boundary term
    enters the transposed operator rather than the row machinery."""
    mesh = StructuredMesh2D(nx=6, ny=6, bounds=[(0.0, 1.0), (0.0, 1.0)], bkd=bkd)
    basis = LagrangeBasis(mesh, degree=1)
    vel_basis = VectorLagrangeBasis(mesh, degree=1)
    vel_dofs = np.zeros(vel_basis.ndofs())
    vel_dofs[0::2] = 1.0  # constant velocity (1, 0)
    reaction = (
        CallableReaction(
            lambda x, u: u**3,
            lambda x, u: 3.0 * u**2,
            lambda x, u: 6.0 * u,
        )
        if cubic_reaction
        else None
    )
    boundary_conditions: List[BoundaryConditionRole[NumpyArray]]
    if robin:
        boundary_conditions = [
            RobinBC(basis, "left", TimeIndependent(lambda x: 0.5 + x[1]), 0.0, bkd),
            RobinBC(basis, "right", 0.3, 0.0, bkd),
            DirichletBC(basis, "bottom", 0.0, bkd),
            DirichletBC(basis, "top", 0.0, bkd),
        ]
    else:
        boundary_conditions = [
            DirichletBC(basis, name, 0.0, bkd)
            for name in ("left", "right", "bottom", "top")
        ]
    physics = AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=NodalFieldDiffusion(basis, dofs=0.1 * np.ones(basis.ndofs())),
        bkd=bkd,
        velocity=NodalFieldVelocity(vel_basis, vel_dofs),
        reaction=reaction,
        forcing=NodalFieldForcing(basis, dofs=np.ones(basis.ndofs())),
    )
    system = compose_galerkin_system(physics, boundary_conditions)
    return physics, system, basis, vel_basis


def _diffusivity_map(
    bkd: NumpyBkd, basis: LagrangeBasis[NumpyArray]
) -> TransformedFieldMap[NumpyArray]:
    """Bilaplacian-prior lognormal KLE (correlation length ~0.5)."""
    nnodes = basis.ndofs()
    return spde_lognormal_field_map(
        basis,
        bkd.asarray(np.log(0.1) * np.ones(nnodes)),
        bkd,
        n_modes=_N_KLE_MODES,
        gamma=1.0,
        delta=4.0,
        sigma=0.3,
    )


def _forcing_map(
    bkd: NumpyBkd, basis: LagrangeBasis[NumpyArray]
) -> MeshKLEFieldMap[NumpyArray]:
    """Linear KLE with zero mean: the forcing field VANISHES at p = 0."""
    coords = bkd.to_numpy(basis.dof_coordinates())
    modes = np.stack(
        [
            np.sin((k + 1) * np.pi * coords[0]) * np.sin(np.pi * coords[1])
            for k in range(_N_FORCING_MODES)
        ],
        axis=1,
    )
    return MeshKLEFieldMap(
        bkd,
        bkd.asarray(np.zeros(coords.shape[1])),
        bkd.asarray(modes),
    )


def _velocity_map(
    bkd: NumpyBkd, vel_basis: VectorLagrangeBasis[NumpyArray]
) -> MeshKLEFieldMap[NumpyArray]:
    """Linear KLE over the interleaved velocity DOFs around (1, 0)."""
    nvel = vel_basis.ndofs()
    mean = np.zeros(nvel)
    mean[0::2] = 1.0
    rng = np.random.default_rng(41)
    modes = 0.3 * rng.normal(0.0, 1.0, (nvel, _N_FORCING_MODES))
    return MeshKLEFieldMap(bkd, bkd.asarray(mean), bkd.asarray(modes))


def _build_parameterization(
    bkd: NumpyBkd, case: str, robin: bool = False
) -> Tuple[
    AdvectionDiffusionReaction[NumpyArray],
    GalerkinSystem[NumpyArray],
    AdvectionDiffusionParameterization[NumpyArray],
]:
    physics, system, basis, vel_basis = _build_physics(
        bkd, cubic_reaction=case.startswith("cubic"), robin=robin
    )
    param_obj = AdvectionDiffusionParameterization(
        physics,
        bkd=bkd,
        diffusivity_map=(
            _diffusivity_map(bkd, basis)
            if case in ("diffusivity", "composite", "cubic-exp")
            else None
        ),
        forcing_map=(
            _forcing_map(bkd, basis)
            if case in ("forcing", "composite", "cubic-linear")
            else None
        ),
        velocity_map=_velocity_map(bkd, vel_basis) if case == "velocity" else None,
    )
    return physics, system, param_obj


@runtime_checkable
class _PhysicsProtocol(Protocol):
    """What the weight and initial-state helpers read of a physics."""

    def basis(self) -> GalerkinBasisProtocol[NumpyArray]: ...

    def mass_matrix(self) -> NumpyArray: ...


def _subdomain_average_weights(bkd: NumpyBkd, physics: _PhysicsProtocol) -> NumpyArray:
    """Mass-weighted subdomain average: c = M @ indicator(x > 0.5)."""
    coords = bkd.to_numpy(physics.basis().dof_coordinates())
    indicator = (coords[0] > 0.5).astype(float)
    mass = physics.mass_matrix()
    cvec = np.asarray(mass @ indicator).flatten()
    return bkd.asarray(cvec.reshape(-1, 1))


def _gaussian_bump_ic(bkd: NumpyBkd, physics: _PhysicsProtocol) -> NumpyArray:
    coords = bkd.to_numpy(physics.basis().dof_coordinates())
    vals = np.exp(-50.0 * ((coords[0] - 0.4) ** 2 + (coords[1] - 0.5) ** 2))
    return bkd.asarray(vals[:, None])


def _endpoint_average(
    bkd: NumpyBkd,
    physics: _PhysicsProtocol,
    parameter_terms: Tuple[ParameterTermProtocol[NumpyArray, NumpyArray], ...] = (),
) -> _Source:
    """c^T y(T) with c the subdomain-average weights, plus
    ``parameter_terms``."""
    weights = _subdomain_average_weights(bkd, physics)
    nstates = weights.shape[0]
    average = WeightedEntries(
        bkd,
        nstates,
        bkd.asarray(np.arange(nstates)[None, :], dtype=bkd.index_dtype()),
        weights.T,
    )
    return FixedTrajectoryFunctional(
        TrajectoryFunctional.from_terms(
            bkd,
            1,
            parameter_terms=parameter_terms,
            state_terms=(EndpointTerm(average),),
        )
    )


def _transient(
    system: GalerkinSystem[NumpyArray],
    param_obj: Any,
) -> GalerkinTransientSystem[NumpyArray, Any]:
    return galerkin_transient_system(
        skfem_discretization(system), ScipySparseLU(), ParameterState(param_obj)
    )


def _function(
    bkd: NumpyBkd,
    transient: GalerkinTransientSystem[NumpyArray, Any],
    method: str,
    init_state: NumpyArray,
    functional: _Source,
    *readers: Callable[[NumpyArray], None],
    strategy: Optional[Any] = None,
) -> _Function:
    implicit = TransientImplicitMap(
        transient,
        ParameterFanOut(transient, readers),
        _method(method),
        RecomputeAlways(transient.solved.solver),
        _steps(),
        init_state,
        functional,
    )
    if strategy is None:
        return ImplicitMapFunction(implicit, ArrayParameters(bkd))
    return ImplicitMapFunction(implicit, ArrayParameters(bkd), strategy)


def _make_model(
    bkd: NumpyBkd, case: str, method: str, robin: bool = False
) -> _Function:
    physics, system, param_obj = _build_parameterization(bkd, case, robin=robin)
    return _function(
        bkd,
        _transient(system, param_obj),
        method,
        _gaussian_bump_ic(bkd, physics),
        _endpoint_average(bkd, physics),
    )


def _sample_and_direction(nparams: int) -> Tuple[NumpyArray, NumpyArray]:
    rng = np.random.default_rng(7)
    sample = 0.4 * rng.normal(0.0, 1.0, (nparams, 1))
    direction = rng.normal(0.0, 1.0, (nparams, 1))
    return sample, direction


def _check_gradient_and_hvp(
    bkd: NumpyBkd,
    model: _Function,
    expect_zero_hessian: bool = False,
) -> None:
    """DerivativeChecker on the model itself + HVP symmetry identity.

    ``expect_zero_hessian`` covers the linear-in-parameters cases
    (linear map onto a linearly-entering coefficient, linear QoI):
    Q(p) is affine, so the machinery must return an EXACTLY zero HVP
    (relative FD ratios are meaningless around zero).
    """
    sample_np, direction_np = _sample_and_direction(model.nvars())
    sample = bkd.asarray(sample_np)
    direction = bkd.asarray(direction_np)
    hvp_fn = model.derivatives().hvp
    assert hvp_fn is not None

    if expect_zero_hessian:
        checker = DerivativeChecker(model)
        errors = checker.check_derivatives(
            sample, direction=direction, relative=True, central=True
        )
        jac_ratio = float(bkd.to_numpy(checker.error_ratio(errors[0])))
        assert jac_ratio <= 1e-6
        hvp_val = hvp_fn(sample, direction)
        bkd.assert_allclose(hvp_val, bkd.zeros(hvp_val.shape), atol=1e-14, rtol=0.0)
        return

    checker = DerivativeChecker(model)
    errors = checker.check_derivatives(
        sample, direction=direction, relative=True, central=True
    )
    jac_ratio = float(bkd.to_numpy(checker.error_ratio(errors[0])))
    assert jac_ratio <= 1e-6
    hvp_min = float(bkd.to_numpy(bkd.min(errors[1])))
    assert hvp_min <= 1e-8
    hvp_ratio = float(bkd.to_numpy(checker.error_ratio(errors[1])))
    assert hvp_ratio <= 1e-6
    # The V of a second-order difference, judged against the rounding
    # predicted for each curve: the order is fitted over the two decades
    # above the steps where the error is within 100 times that rounding,
    # so the fit stays off the floor wherever a case and platform put it
    # (the floor arrives between h = 1e-3 and 1e-5 here, and a fit that
    # reaches it reads 2.3 to 2.4). A curve that is rounding at every
    # step (the time-integrated case's gradient, a quantity quadratic
    # along the direction, so its difference is exact) passes as
    # truncation free, which a wrong derivative cannot be. Steps are the
    # checker's defaults, largest first.
    steps_list = [10.0**-k for k in range(1, 14)]
    steps = bkd.asarray(steps_list)
    jacobian_fn = model.derivatives().jacobian
    assert jacobian_fn is not None
    gradient = jacobian_fn(sample)
    sizes = (
        (model(sample), bkd.dot(gradient, direction)),
        (gradient, hvp_fn(sample, direction)),
    )
    eps = float(np.finfo(bkd.to_numpy(sample).dtype).eps)
    for error, (value, action) in zip(errors, sizes):
        rounding = predicted_rounding(
            eps,
            bkd.to_float(bkd.norm(value)),
            bkd.to_float(bkd.norm(action)),
            bkd.to_float(bkd.norm(sample)),
            steps_list,
        )
        curve = checker.curve(error, steps, central=True, rounding=rounding)
        report = curve.v_shape()
        assert report.passed, report

    other = bkd.asarray(np.random.default_rng(11).normal(0.0, 1.0, (model.nvars(), 1)))
    h_dir = bkd.flatten(hvp_fn(sample, direction))
    h_other = bkd.flatten(hvp_fn(sample, other))
    bkd.assert_allclose(
        bkd.sum(h_dir * bkd.flatten(other)),
        bkd.sum(h_other * bkd.flatten(direction)),
        rtol=1e-12,
    )


class TestTransientAdjointWorkedExample:
    @pytest.mark.parametrize(
        "case",
        [
            # One-nonlinearity-at-a-time diagonal: each case activates
            # exactly one second-order pathway (plus the composite).
            "diffusivity",  # param_param via exp-map curvature
            "forcing",  # fully linear: Hessian exactly zero
            "velocity",  # linear map through the advection term
            "composite",  # diffusivity+forcing block interactions
            "cubic-linear",  # cubic reaction + linear map: state_state
            "cubic-exp",  # cubic reaction + exp map: cross terms
        ],
    )
    @pytest.mark.parametrize("method", ["backward_euler", "crank_nicolson"])
    def test_gradient_and_hvp_match_fd(
        self, numpy_bkd: NumpyBkd, method: str, case: str
    ) -> None:
        bkd = numpy_bkd
        model = _make_model(bkd, case, method)
        assert model.derivatives().hvp is not None
        # The forcing case is linear end to end (linear map, linearly
        # entering coefficient, linear QoI): Q(p) is affine and the
        # exact Hessian is zero. The cubic-linear case shares the
        # linear forcing map but its Hessian is NONZERO: the state
        # enters Q through the cubic reaction.
        _check_gradient_and_hvp(bkd, model, expect_zero_hessian=(case == "forcing"))

    def test_scalar_adjoint_matches_forward_sensitivity_row(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """Adjoint gradient row == c^T W_T with W_T the final state's
        tangent-column Jacobian (backward and forward propagation of the
        same linearization must agree to solver precision)."""
        bkd = numpy_bkd
        # ONE physics + parameterization shared by both functions:
        # separate KLE constructions are distinct parameterizations
        # (eigensolver sign/ordering indeterminacy), which would make the
        # comparison meaningless.
        physics, system, param_obj = _build_parameterization(bkd, "diffusivity")
        transient = _transient(system, param_obj)
        ic = _gaussian_bump_ic(bkd, physics)
        scalar = _function(
            bkd,
            transient,
            "backward_euler",
            ic,
            _endpoint_average(bkd, physics),
            strategy=AdjointRows(),
        )
        nstates = physics.nstates()
        final_state = FixedTrajectoryFunctional(
            TrajectoryFunctional.from_terms(
                bkd,
                nstates,
                state_terms=(
                    EndpointTerm(StateEntries(bkd, nstates, list(range(nstates)))),
                ),
            )
        )
        vector = _function(
            bkd,
            transient,
            "backward_euler",
            ic,
            final_state,
            strategy=TangentColumns(),
        )
        sample_np, _ = _sample_and_direction(scalar.nvars())
        sample = bkd.asarray(sample_np)
        scalar_jac = scalar.derivatives().jacobian
        vector_jac = vector.derivatives().jacobian
        assert scalar_jac is not None
        assert vector_jac is not None
        grad = bkd.to_numpy(scalar_jac(sample))
        w_final = bkd.to_numpy(vector_jac(sample))
        weights = bkd.to_numpy(_subdomain_average_weights(bkd, physics))[:, 0]
        bkd.assert_allclose(
            bkd.asarray(grad.flatten()),
            bkd.asarray(weights @ w_final),
            rtol=1e-10,
        )

    def test_tikhonov_functional_at_zero_forcing_field(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """Tikhonov-augmented QoI at p = 0, where the parameterized
        forcing field vanishes identically at the test point: FD
        perturbs p, the field leaves zero, and any wrongly-Zeroed or
        value/structure-conflating derivative path disagrees
        immediately. The direct dQ/dp and d2Q/dp2 terms are nonzero."""
        bkd = numpy_bkd
        physics, system, param_obj = _build_parameterization(bkd, "forcing")
        tikhonov = TikhonovTerm(bkd, 0.7)
        model = _function(
            bkd,
            _transient(system, param_obj),
            "backward_euler",
            _gaussian_bump_ic(bkd, physics),
            _endpoint_average(bkd, physics, (tikhonov,)),
            tikhonov.set_param,
        )
        sample = bkd.zeros((model.nvars(), 1))
        direction = bkd.asarray(
            np.random.default_rng(13).normal(0.0, 1.0, (model.nvars(), 1))
        )
        checker = DerivativeChecker(model)
        errors = checker.check_derivatives(
            sample, direction=direction, relative=True, central=True
        )
        jac_ratio = float(bkd.to_numpy(checker.error_ratio(errors[0])))
        assert jac_ratio <= 1e-6
        # Q is quadratic in p here, so central differences of the
        # gradient are exact up to rounding: no V, only a small floor.
        hvp_min = float(bkd.to_numpy(bkd.min(errors[1])))
        assert hvp_min <= 1e-8

    @pytest.mark.parametrize("method", ["backward_euler", "crank_nicolson"])
    def test_time_integrated_qoi_gradient_and_hvp(
        self, numpy_bkd: NumpyBkd, method: str
    ) -> None:
        """A time-integrated QoI: the integral of y^T M y by the method
        the trajectory is solved with (its nodes are the method's own),
        plus a Tikhonov term. dQ/dy is nonzero at every stored step
        (per-step adjoint RHS accumulation, which endpoint QoIs never
        exercise), the quadratic QoI supplies state_state curvature, and
        the Tikhonov term supplies the direct parameter pathways."""
        bkd = numpy_bkd
        physics, system, param_obj = _build_parameterization(bkd, "forcing")
        transient = _transient(system, param_obj)
        integral = time_integrated_weighted_l2(
            transient,
            _method(method),
            RecomputeAlways(transient.solved.solver),
            ScipySparseWeight(physics.mass_matrix(), bkd),
        )
        tikhonov = TikhonovTerm(bkd, 0.7)
        regularization = FixedTrajectoryFunctional(
            TrajectoryFunctional.from_terms(bkd, 1, parameter_terms=(tikhonov,))
        )
        model = _function(
            bkd,
            transient,
            method,
            _gaussian_bump_ic(bkd, physics),
            SummedFunctional([integral, regularization]),
            tikhonov.set_param,
        )
        _check_gradient_and_hvp(bkd, model)

    @pytest.mark.parametrize("case", ["forcing", "diffusivity"])
    @pytest.mark.parametrize("method", ["backward_euler", "crank_nicolson"])
    def test_robin_bc_gradient_and_hvp(
        self, numpy_bkd: NumpyBkd, method: str, case: str
    ) -> None:
        """Gradient + HVP with ROBIN boundary conditions, including a
        varying-coefficient alpha(y): the Robin boundary term enters
        the transposed operator (no row replacement), so the adjoint
        must reproduce it automatically alongside the remaining
        essential rows. First adjoint coverage of galerkin Robin BCs —
        needed before an inflow can become the Danckwerts condition."""
        bkd = numpy_bkd
        model = _make_model(bkd, case, method, robin=True)
        assert model.derivatives().hvp is not None
        # Linear forcing map + linear QoI: exactly zero Hessian, as in
        # the Dirichlet variant of this configuration.
        _check_gradient_and_hvp(bkd, model, expect_zero_hessian=(case == "forcing"))

    def test_quasilinear_gradient_and_hvp(self, numpy_bkd: NumpyBkd) -> None:
        """Quasilinear kappa(u)*a(x) diffusion through the model: the
        genuinely state-dependent mixed curvature is exercised end to
        end (a plausibly-wrong Hessian is invisible to gradient-only
        tests)."""
        bkd = numpy_bkd
        mesh = StructuredMesh2D(nx=5, ny=5, bounds=[(0.0, 1.0), (0.0, 1.0)], bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        physics = QuasilinearDiffusion(
            basis=basis,
            diffusivity=NodalFieldDiffusion(basis, dofs=0.1 * np.ones(basis.ndofs())),
            bkd=bkd,
            kappa=_kappa,
            kappa_deriv=_kappa_deriv,
            kappa_second_deriv=_kappa_second_deriv,
        )
        system = compose_galerkin_system(
            physics,
            [
                DirichletBC(basis, name, 0.0, bkd)
                for name in ("left", "right", "bottom", "top")
            ],
        )
        param_obj = create_quasilinear_diffusivity_parameterization(
            physics, _diffusivity_map(bkd, basis), bkd
        )
        model = _function(
            bkd,
            _transient(system, param_obj),
            "backward_euler",
            _gaussian_bump_ic(bkd, physics),
            _endpoint_average(bkd, physics),
        )
        assert model.derivatives().hvp is not None
        _check_gradient_and_hvp(bkd, model)


class TestTimeModulatedControl:
    """The adjoint gradient of a control that varies in time.

    The whole separable chain end to end: a coefficient holding its own
    modes and profiles, a facade that derives the parameter map from
    those same modes, and a per-column b(t) scaling reaching the
    adjoint. Verified against finite differences rather than by
    inspection, because every layer here can be wrong in a way that
    still produces plausible numbers.
    """

    _KNOTS = [0.0, 0.06, 0.12]

    def _build(
        self, method: str
    ) -> Tuple[
        NumpyBkd,
        _Function,
        TimeModulatedNodalFieldLinearReaction,
        AdvectionDiffusionParameterization[NumpyArray],
    ]:
        bkd = NumpyBkd()
        mesh = StructuredMesh2D(nx=5, ny=5, bounds=[(0.0, 1.0), (0.0, 1.0)], bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        ndofs = basis.ndofs()
        coords = bkd.to_numpy(basis.dof_coordinates())
        modes = np.stack(
            [
                np.exp(-15.0 * ((coords[0] - c) ** 2 + (coords[1] - 0.5) ** 2))
                for c in (0.3, 0.5, 0.7)
            ],
            axis=1,
        )
        # Knots offset from the step grid (dt = 0.02) so b(t) varies
        # WITHIN a step. Crank-Nicolson evaluates at t_prev and t_curr;
        # if those coincided, a stepper binding one time per step would
        # be indistinguishable from a correct one.
        reaction = TimeModulatedNodalFieldLinearReaction(
            basis,
            modes,
            PiecewiseLinearModulation(bkd, self._KNOTS),
            np.array([0.5, 0.3, 0.4]),
        )
        physics = AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=NodalFieldDiffusion(basis, dofs=0.1 * np.ones(ndofs)),
            bkd=bkd,
            reaction=reaction,
            forcing=NodalFieldForcing(basis, dofs=np.ones(ndofs)),
        )
        system = compose_galerkin_system(
            physics,
            [
                DirichletBC(basis, name, 0.0, bkd)
                for name in ("left", "right", "bottom", "top")
            ],
        )
        param = AdvectionDiffusionParameterization(
            physics, reaction_map=FROM_FIELD, bkd=bkd
        )
        constrained = set(int(d) for d in bkd.to_numpy(system.constraint_set().dofs()))
        free = [i for i in range(ndofs) if i not in constrained]
        nstates = physics.nstates()
        # The QoI is the value at one free DOF of the final state.
        functional = FixedTrajectoryFunctional(
            TrajectoryFunctional.from_terms(
                bkd,
                1,
                state_terms=(
                    EndpointTerm(StateEntries(bkd, nstates, [free[len(free) // 2]])),
                ),
            )
        )
        model = _function(
            bkd,
            _transient(system, param),
            method,
            bkd.zeros((nstates, 1)),
            functional,
        )
        return bkd, model, reaction, param

    @pytest.mark.parametrize("method", ["backward_euler", "crank_nicolson"])
    def test_gradient_matches_fd(self, method: str) -> None:
        bkd, model, _, param = self._build(method)
        assert model.nvars() == param.nparams() == 3

        sample, direction = _sample_and_direction(model.nvars())
        checker = DerivativeChecker(model)
        errors = checker.check_derivatives(
            bkd.asarray(sample),
            direction=bkd.asarray(direction),
            relative=True,
            central=True,
        )
        ratio = float(bkd.to_numpy(checker.error_ratio(errors[0])))
        assert ratio <= 1e-6

    def test_the_control_actually_moves_the_qoi(self) -> None:
        """Without this every other assertion still passes if the
        control is silently dropped: a gradient of zero agrees with a
        finite difference of zero."""
        bkd, model, _, _ = self._build("crank_nicolson")
        quiet = float(bkd.to_numpy(model(bkd.asarray(np.zeros((3, 1)))))[0, 0])
        acting = float(
            bkd.to_numpy(model(bkd.asarray(np.array([[0.5], [0.3], [0.4]]))))[0, 0]
        )
        assert abs(acting - quiet) > 1e-8

    def test_the_modes_have_one_holder(self) -> None:
        """The field's modes and the map's jacobian must be the
        SAME array. Two copies could disagree with nothing detecting
        it --- the forward solve using one and the gradient the other,
        each self-consistent, with finite differences agreeing because
        they perturb the same wrong forward field."""
        bkd, _, reaction, param = self._build("backward_euler")
        term = param._inner
        # One coefficient is parameterized, so the inner term is not a
        # composite.
        assert isinstance(term, _FieldParameterizationTerm)
        derived = term._field_map.jacobian(bkd.asarray(np.zeros(3)))
        np.testing.assert_array_equal(
            np.asarray(bkd.to_numpy(derived)),
            np.asarray(reaction.spatial_modes()),
        )

    def test_a_supplied_map_is_refused(self) -> None:
        """Passing a map for a coefficient that owns its own modes is a
        contradiction, so it raises rather than being silently ignored:
        the caller would otherwise believe their map was used."""
        bkd, _, _, param = self._build("backward_euler")
        physics = param.physics()
        nstates = physics.nstates()
        unrelated_map = MeshKLEFieldMap(
            bkd,
            bkd.asarray(np.zeros(nstates)),
            bkd.asarray(np.ones((nstates, 3))),
        )
        with pytest.raises(TypeError, match="FROM_FIELD"):
            AdvectionDiffusionParameterization(
                physics, reaction_map=unrelated_map, bkd=bkd
            )
