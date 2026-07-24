"""Parity validation of the field-parameterization derivative engine.

The engine (_FieldParameterizationTerm with FromLinearity slots) wired
to the galerkin ADR diffusivity assemblies must reproduce the
hand-rolled AffineDiffusivityFieldParameterization — the most heavily
validated parameterization in the repo (steady 14-check suites,
transient BE/CN/FE/Heun DerivativeChecker tests) — entrywise at random
states, and pass the same component-wise checker suite when swapped
into the steady HVP adapter.
"""

import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Tuple

import numpy as np
from pyapprox.optimization.implicitfunction.functionals.weighted_sum import (
    WeightedSumFunctional,
)
from pyapprox.optimization.implicitfunction.operator.check_derivatives import (
    ImplicitFunctionDerivativeChecker,
)
from pyapprox.optimization.implicitfunction.operator.operator_with_hvp import (
    AdjointOperatorWithJacobianAndHVP,
)
from pyapprox.pde.constitutive.coefficient_functions import (
    CallableReaction,
    NodalFieldDiffusion,
    NodalFieldForcing,
    NodalFieldLinearReaction,
    NodalFieldVelocity,
)
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.field_maps.transformed import (
    TransformedFieldMap,
    _ExpTransform,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis, VectorLagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.models.galerkin.steady import (
    GalerkinStateEquationWithHVPAdapter,
)
from pyapprox.pde.parameterizations.derivatives import ParamDerivatives
from pyapprox.pde.parameterizations.field_term import (
    FromLinearity,
    Zero,
    _FieldParameterizationTerm,
)
from pyapprox.pde.parameterizations.galerkin_diffusivity import (
    AffineDiffusivityFieldParameterization,
)
from pyapprox.util.backends.numpy import NumpyBkd

from tests._helpers.adjoint_checks import NumpyArray

_NPARAMS = 3


def _build_physics_and_map(
    bkd: NumpyBkd, nonlinear_reaction: bool
) -> Tuple[
    AdvectionDiffusionReaction[NumpyArray],
    TransformedFieldMap[NumpyArray],
]:
    mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
    basis = LagrangeBasis(mesh, degree=1)
    reaction = (
        CallableReaction(
            lambda x, u: u**2,
            lambda x, u: 2.0 * u,
            lambda x, u: np.full_like(u, 2.0),
        )
        if nonlinear_reaction
        else None
    )
    physics = AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=NodalFieldDiffusion(basis),
        bkd=bkd,
        reaction=reaction,
        forcing=lambda x: np.ones(x.shape[1]),
        boundary_conditions=[
            DirichletBC(basis, "left", 0.0, bkd),
            DirichletBC(basis, "right", 0.0, bkd),
        ],
    )
    coords = bkd.to_numpy(basis.dof_coordinates())[0]
    modes = np.stack(
        [
            0.4 * np.sin((k + 1) * np.pi * coords) / (k + 1)
            for k in range(_NPARAMS)
        ],
        axis=1,
    )
    kle = MeshKLEFieldMap(
        bkd, bkd.asarray(np.zeros(coords.shape[0])), bkd.asarray(modes)
    )
    exp = _ExpTransform(bkd)
    field_map = TransformedFieldMap(kle, exp, exp, bkd, transform_deriv2=exp)
    return physics, field_map


def _build_engine_term(
    bkd: NumpyBkd,
    physics: AdvectionDiffusionReaction[NumpyArray],
    field_map: TransformedFieldMap[NumpyArray],
) -> _FieldParameterizationTerm[NumpyArray]:
    diffusion = physics.diffusion_function()
    assert isinstance(diffusion, NodalFieldDiffusion)
    return _FieldParameterizationTerm.linear_field_state(
        setter=lambda field: diffusion.set_dofs(bkd.to_numpy(field)),
        field_jacobian=lambda state, time: (
            physics.residual_diffusivity_jacobian(state)
        ),
        field_state_jacobian=lambda delta, state, time: (
            physics.residual_diffusivity_state_jacobian(delta, state)
        ),
        field_map=field_map,
        bkd=bkd,
        nstates=physics.nstates(),
        nfield_dofs=physics.nstates(),
        require_positive=True,
    )


class _TermParameterization:
    """Minimal ParameterizationProtocol adapter over one engine term."""

    def __init__(
        self,
        term: _FieldParameterizationTerm[NumpyArray],
        physics: AdvectionDiffusionReaction[NumpyArray],
    ) -> None:
        self._term = term
        self._physics = physics

    def nparams(self) -> int:
        return self._term.nparams()

    def physics(self) -> AdvectionDiffusionReaction[NumpyArray]:
        return self._physics

    def apply(self, params_1d: NumpyArray) -> None:
        self._term.apply(params_1d)

    def param_derivatives(self) -> ParamDerivatives[NumpyArray]:
        return self._term.param_derivatives()


class TestFieldTermDiffusivityParity:
    @pytest.mark.parametrize("nonlinear_reaction", [False, True])
    def test_entrywise_parity_with_oracle(
        self, numpy_bkd: NumpyBkd, nonlinear_reaction: bool
    ) -> None:
        """Engine bundle == hand-rolled oracle at random arguments."""
        bkd = numpy_bkd
        physics, field_map = _build_physics_and_map(bkd, nonlinear_reaction)
        oracle = AffineDiffusivityFieldParameterization(
            physics, field_map, bkd
        )
        term = _build_engine_term(bkd, physics, field_map)

        nstates = physics.nstates()
        rng = np.random.default_rng(7)
        state = bkd.asarray(rng.normal(0.0, 0.5, nstates))
        adj = bkd.asarray(rng.normal(0.0, 1.0, nstates))
        wvec = bkd.asarray(rng.normal(0.0, 1.0, nstates))
        params = bkd.asarray(np.array([0.4, -0.3, 0.2]))
        vvec = bkd.asarray(np.array([0.5, 0.7, -0.6]))

        o_derivs = oracle.param_derivatives()
        assert o_derivs.param_jacobian is not None
        assert o_derivs.param_param_hvp is not None
        assert o_derivs.state_param_hvp is not None
        assert o_derivs.param_state_hvp is not None

        oracle.apply(params)
        bkd.assert_allclose(
            term.param_jacobian(state, 0.0, params),
            o_derivs.param_jacobian(state, 0.0, params),
            rtol=1e-12,
        )
        bkd.assert_allclose(
            term.param_param_hvp(state, 0.0, params, adj, vvec),
            o_derivs.param_param_hvp(state, 0.0, params, adj, vvec),
            rtol=1e-12,
        )
        bkd.assert_allclose(
            term.state_param_hvp(state, 0.0, params, adj, vvec),
            o_derivs.state_param_hvp(state, 0.0, params, adj, vvec),
            rtol=1e-12,
        )
        bkd.assert_allclose(
            term.param_state_hvp(state, 0.0, params, adj, wvec),
            o_derivs.param_state_hvp(state, 0.0, params, adj, wvec),
            rtol=1e-12,
        )
        bkd.assert_allclose(
            term.initial_param_jacobian(params),
            o_derivs.initial_param_jacobian(params),
            rtol=1e-12,
        )

    def test_mixed_assembly_linearity_identity(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """A(delta) w == S(w) delta — both contractions of the constant
        mixed tensor agree (validates the new mixed assembly against
        the sensitivity assembly)."""
        bkd = numpy_bkd
        physics, _ = _build_physics_and_map(bkd, False)
        nstates = physics.nstates()
        rng = np.random.default_rng(11)
        delta = bkd.asarray(rng.normal(0.0, 1.0, nstates))
        wvec = bkd.asarray(rng.normal(0.0, 1.0, nstates))
        lhs = physics.residual_diffusivity_state_jacobian(
            delta, wvec
        ) @ bkd.to_numpy(wvec)
        rhs = physics.residual_diffusivity_jacobian(wvec) @ bkd.to_numpy(
            delta
        )
        bkd.assert_allclose(
            bkd.asarray(lhs), bkd.asarray(rhs), rtol=1e-12
        )

    @pytest.mark.parametrize("nonlinear_reaction", [False, True])
    def test_engine_term_passes_component_checker(
        self, numpy_bkd: NumpyBkd, nonlinear_reaction: bool
    ) -> None:
        """The engine-wired term passes the same 14-check suite the
        oracle passes (steady ADR log-KLE)."""
        bkd = numpy_bkd
        physics, field_map = _build_physics_and_map(bkd, nonlinear_reaction)
        term = _build_engine_term(bkd, physics, field_map)
        param_obj = _TermParameterization(term, physics)
        state_eq = GalerkinStateEquationWithHVPAdapter(
            physics, param_obj, bkd
        )

        nstates = physics.nstates()
        constrained = set(
            int(d) for d in bkd.to_numpy(physics.constraint_set().dofs())
        )
        state_idx = next(
            ii for ii in range(nstates) if ii not in constrained
        )
        weights = bkd.zeros((nstates, 1))
        weights = bkd.copy(weights)
        weights[state_idx] = 1.0
        functional = WeightedSumFunctional(weights, _NPARAMS, bkd)

        adjoint_op = AdjointOperatorWithJacobianAndHVP(state_eq, functional)
        checker = ImplicitFunctionDerivativeChecker(adjoint_op)
        param = bkd.asarray(np.array([[0.4], [-0.3], [0.2]]))
        init_state = bkd.zeros((nstates, 1))
        tols = bkd.copy(checker.get_derivative_tolerances(1e-6))
        tols[4] = 5e-6
        tols[5] = 5e-6
        tols[8] = 5e-6
        checker.check_derivatives(init_state, param, tols)

    def test_forcing_term_passes_component_checker(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """A state-independent (forcing) term through the engine passes
        the 14-check suite — validating the Zero() slot paths (the
        mixed and field-curvature contractions are certified zero and
        the checker FDs confirm it)."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        nodal_forcing = NodalFieldForcing(basis)
        physics = AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=NodalFieldDiffusion(basis),
            bkd=bkd,
            forcing=nodal_forcing,
            boundary_conditions=[
                DirichletBC(basis, "left", 0.0, bkd),
                DirichletBC(basis, "right", 0.0, bkd),
            ],
        )
        coords = bkd.to_numpy(basis.dof_coordinates())[0]
        modes = np.stack(
            [
                0.6 * np.sin((k + 1) * np.pi * coords) / (k + 1)
                for k in range(_NPARAMS)
            ],
            axis=1,
        )
        kle = MeshKLEFieldMap(
            bkd,
            bkd.asarray(np.ones(coords.shape[0])),
            bkd.asarray(modes),
        )
        exp = _ExpTransform(bkd)
        forcing_map = TransformedFieldMap(
            kle, exp, exp, bkd, transform_deriv2=exp
        )
        term = _FieldParameterizationTerm.state_independent(
            setter=lambda field: nodal_forcing.set_dofs(
                bkd.to_numpy(field)
            ),
            field_jacobian=lambda state, time: (
                physics.residual_forcing_jacobian()
            ),
            field_map=forcing_map,
            bkd=bkd,
            nstates=physics.nstates(),
            nfield_dofs=physics.nstates(),
        )
        param_obj = _TermParameterization(term, physics)
        state_eq = GalerkinStateEquationWithHVPAdapter(
            physics, param_obj, bkd
        )
        nstates = physics.nstates()
        constrained = set(
            int(d) for d in bkd.to_numpy(physics.constraint_set().dofs())
        )
        state_idx = next(
            ii for ii in range(nstates) if ii not in constrained
        )
        weights = bkd.zeros((nstates, 1))
        weights = bkd.copy(weights)
        weights[state_idx] = 1.0
        functional = WeightedSumFunctional(weights, _NPARAMS, bkd)

        adjoint_op = AdjointOperatorWithJacobianAndHVP(state_eq, functional)
        checker = ImplicitFunctionDerivativeChecker(adjoint_op)
        param = bkd.asarray(np.array([[0.4], [-0.3], [0.2]]))
        init_state = bkd.zeros((nstates, 1))
        tols = bkd.copy(checker.get_derivative_tolerances(1e-6))
        tols[4] = 5e-6
        tols[5] = 5e-6
        tols[8] = 5e-6
        checker.check_derivatives(init_state, param, tols)

    def test_reaction_term_passes_component_checker(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """A linear_field_state reaction term (r(x)*u through an exp
        map) through the engine passes the 14-check suite — the second
        FromLinearity instance, with a non-self-adjoint... rather, a
        mass-structured mixed tensor distinct from diffusivity's."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        nodal_reaction = NodalFieldLinearReaction(basis)
        physics = AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=NodalFieldDiffusion(basis),
            bkd=bkd,
            reaction=nodal_reaction,
            forcing=lambda x: np.ones(x.shape[1]),
            boundary_conditions=[
                DirichletBC(basis, "left", 0.0, bkd),
                DirichletBC(basis, "right", 0.0, bkd),
            ],
        )
        coords = bkd.to_numpy(basis.dof_coordinates())[0]
        modes = np.stack(
            [
                0.4 * np.sin((k + 1) * np.pi * coords) / (k + 1)
                for k in range(_NPARAMS)
            ],
            axis=1,
        )
        kle = MeshKLEFieldMap(
            bkd,
            bkd.asarray(-1.0 * np.ones(coords.shape[0])),
            bkd.asarray(modes),
        )
        exp = _ExpTransform(bkd)
        # r = -exp(...) (damping) keeps the steady operator coercive.
        reaction_map = TransformedFieldMap(
            kle, exp, exp, bkd, transform_deriv2=exp
        )
        term = _FieldParameterizationTerm.linear_field_state(
            setter=lambda field: nodal_reaction.set_dofs(
                -bkd.to_numpy(field)
            ),
            field_jacobian=lambda state, time: (
                -physics.residual_reaction_jacobian(state)
            ),
            field_state_jacobian=lambda delta, state, time: (
                -physics.residual_reaction_state_jacobian(delta, state)
            ),
            field_map=reaction_map,
            bkd=bkd,
            nstates=physics.nstates(),
            nfield_dofs=physics.nstates(),
        )
        param_obj = _TermParameterization(term, physics)
        state_eq = GalerkinStateEquationWithHVPAdapter(
            physics, param_obj, bkd
        )
        nstates = physics.nstates()
        constrained = set(
            int(d) for d in bkd.to_numpy(physics.constraint_set().dofs())
        )
        state_idx = next(
            ii for ii in range(nstates) if ii not in constrained
        )
        weights = bkd.zeros((nstates, 1))
        weights = bkd.copy(weights)
        weights[state_idx] = 1.0
        functional = WeightedSumFunctional(weights, _NPARAMS, bkd)

        adjoint_op = AdjointOperatorWithJacobianAndHVP(state_eq, functional)
        checker = ImplicitFunctionDerivativeChecker(adjoint_op)
        param = bkd.asarray(np.array([[0.4], [-0.3], [0.2]]))
        init_state = bkd.zeros((nstates, 1))
        tols = bkd.copy(checker.get_derivative_tolerances(1e-6))
        # Noise-limited assembled/mixed checks (established calibration;
        # a genuine bug plateaus orders of magnitude higher).
        tols[4] = 5e-6
        tols[5] = 5e-6
        tols[8] = 5e-6
        tols[13] = 5e-6
        checker.check_derivatives(init_state, param, tols)

    def test_velocity_term_passes_component_checker(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """A linear_field_state velocity term through the engine passes
        the 14-check suite — the rectangular-S instance with the
        NON-symmetric advection mixed tensor (the case FromLinearity's
        required A assembly exists for)."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        vel_basis = VectorLagrangeBasis(mesh, degree=1)
        nodal_velocity = NodalFieldVelocity(
            vel_basis, np.zeros(vel_basis.ndofs())
        )
        physics = AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=NodalFieldDiffusion(basis),
            bkd=bkd,
            velocity=nodal_velocity,
            forcing=lambda x: np.ones(x.shape[1]),
            boundary_conditions=[
                DirichletBC(basis, "left", 0.0, bkd),
                DirichletBC(basis, "right", 0.0, bkd),
            ],
        )
        nvel = vel_basis.ndofs()
        coords = np.linspace(0.0, 1.0, nvel)
        modes = np.stack(
            [
                0.4 * np.sin((k + 1) * np.pi * coords) / (k + 1)
                for k in range(_NPARAMS)
            ],
            axis=1,
        )
        kle = MeshKLEFieldMap(
            bkd, bkd.asarray(np.zeros(nvel)), bkd.asarray(modes)
        )
        exp = _ExpTransform(bkd)
        velocity_map = TransformedFieldMap(
            kle, exp, exp, bkd, transform_deriv2=exp
        )
        term = _FieldParameterizationTerm.linear_field_state(
            setter=lambda field: nodal_velocity.set_dofs(
                bkd.to_numpy(field)
            ),
            field_jacobian=lambda state, time: (
                physics.residual_velocity_jacobian(state)
            ),
            field_state_jacobian=lambda delta, state, time: (
                physics.residual_velocity_state_jacobian(delta, state)
            ),
            field_map=velocity_map,
            bkd=bkd,
            nstates=physics.nstates(),
            nfield_dofs=nvel,
        )
        param_obj = _TermParameterization(term, physics)
        state_eq = GalerkinStateEquationWithHVPAdapter(
            physics, param_obj, bkd
        )
        nstates = physics.nstates()
        constrained = set(
            int(d) for d in bkd.to_numpy(physics.constraint_set().dofs())
        )
        state_idx = next(
            ii for ii in range(nstates) if ii not in constrained
        )
        weights = bkd.zeros((nstates, 1))
        weights = bkd.copy(weights)
        weights[state_idx] = 1.0
        functional = WeightedSumFunctional(weights, _NPARAMS, bkd)

        adjoint_op = AdjointOperatorWithJacobianAndHVP(state_eq, functional)
        checker = ImplicitFunctionDerivativeChecker(adjoint_op)
        param = bkd.asarray(np.array([[0.4], [-0.3], [0.2]]))
        init_state = bkd.zeros((nstates, 1))
        tols = bkd.copy(checker.get_derivative_tolerances(1e-6))
        # Noise-limited assembled/mixed checks (established calibration).
        tols[4] = 5e-6
        tols[5] = 5e-6
        tols[8] = 5e-6
        tols[13] = 5e-6
        checker.check_derivatives(init_state, param, tols)

    def test_slot_validation(self, numpy_bkd: NumpyBkd) -> None:
        """FromLinearity on the state-shaped slot requires the mixed
        assembly; field_field rejects FromLinearity."""
        bkd = numpy_bkd
        physics, field_map = _build_physics_and_map(bkd, False)
        diffusion = physics.diffusion_function()
        assert isinstance(diffusion, NodalFieldDiffusion)
        with pytest.raises(TypeError, match="field_state_jacobian"):
            _FieldParameterizationTerm(
                setter=lambda f: diffusion.set_dofs(bkd.to_numpy(f)),
                field_jacobian=lambda s, t: (
                    physics.residual_diffusivity_jacobian(s)
                ),
                field_state_hvp=FromLinearity(),
                state_field_hvp=FromLinearity(),
                field_field_hvp=Zero(),
                field_map=field_map,
                bkd=bkd,
                nstates=physics.nstates(),
                nfield_dofs=physics.nstates(),
            )
        with pytest.raises(TypeError, match="FromLinearity"):
            _FieldParameterizationTerm(
                setter=lambda f: diffusion.set_dofs(bkd.to_numpy(f)),
                field_jacobian=lambda s, t: (
                    physics.residual_diffusivity_jacobian(s)
                ),
                field_state_hvp=FromLinearity(),
                state_field_hvp=Zero(),
                field_field_hvp=FromLinearity(),  # type: ignore[arg-type]
                field_map=field_map,
                bkd=bkd,
                nstates=physics.nstates(),
                nfield_dofs=physics.nstates(),
            )

    def test_apply_positivity_and_length(self, numpy_bkd: NumpyBkd) -> None:
        """require_positive raise and wrong-length field raise."""
        bkd = numpy_bkd
        physics, field_map = _build_physics_and_map(bkd, False)
        term = _build_engine_term(bkd, physics, field_map)
        # exp map is always positive: apply succeeds
        term.apply(bkd.asarray(np.array([0.4, -0.3, 0.2])))

        diffusion = physics.diffusion_function()
        assert isinstance(diffusion, NodalFieldDiffusion)
        nstates = physics.nstates()
        coords = np.linspace(0.0, 1.0, nstates)
        # Linear map that goes negative -> positivity raise
        negative_map = MeshKLEFieldMap(
            bkd,
            bkd.asarray(-1.0 * np.ones(nstates)),
            bkd.asarray(coords[:, None]),
        )
        bad_term = _FieldParameterizationTerm.linear_field_state(
            setter=lambda f: diffusion.set_dofs(bkd.to_numpy(f)),
            field_jacobian=lambda s, t: (
                physics.residual_diffusivity_jacobian(s)
            ),
            field_state_jacobian=lambda d, s, t: (
                physics.residual_diffusivity_state_jacobian(d, s)
            ),
            field_map=negative_map,
            bkd=bkd,
            nstates=nstates,
            nfield_dofs=nstates,
            require_positive=True,
        )
        with pytest.raises(ValueError, match="positive"):
            bad_term.apply(bkd.asarray(np.array([0.1])))

        # Wrong-length field -> raise
        short_map = MeshKLEFieldMap(
            bkd,
            bkd.asarray(np.ones(nstates - 2)),
            bkd.asarray(np.ones((nstates - 2, 1))),
        )
        wrong_term = _FieldParameterizationTerm.linear_field_state(
            setter=lambda f: diffusion.set_dofs(bkd.to_numpy(f)),
            field_jacobian=lambda s, t: (
                physics.residual_diffusivity_jacobian(s)
            ),
            field_state_jacobian=lambda d, s, t: (
                physics.residual_diffusivity_state_jacobian(d, s)
            ),
            field_map=short_map,
            bkd=bkd,
            nstates=nstates,
            nfield_dofs=nstates,
        )
        with pytest.raises(ValueError, match="DOFs"):
            wrong_term.apply(bkd.asarray(np.array([0.1])))
