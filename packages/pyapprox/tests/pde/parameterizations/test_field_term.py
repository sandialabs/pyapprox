"""Validation of the field-parameterization derivative engine.

The engine (_FieldParameterizationTerm with FromLinearity slots) wired
to each galerkin ADR field assembly (diffusivity, forcing, reaction,
velocity) must pass the component-wise 14-check FD suite through the
steady HVP adapter, and the mixed assembly must satisfy the linearity
identity A(\\delta) w == S(w) \\delta.
"""

import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Tuple

import numpy as np
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
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
from pyapprox.pde.parameterizations.field_term import (
    FromLinearity,
    MixedHVPAdapter,
    StateJacobianAdapter,
    ToNumpySetter,
    Zero,
    _FieldParameterizationTerm,
)
from pyapprox.util.backends.numpy import NumpyBkd

from tests._helpers.adjoint_checks import NumpyArray

_NPARAMS = 3


class _QuadraticFieldToy:
    """Analytic residual R_j(g) = c_j g_j^2 / 2: state-independent and
    quadratic in the field, so the field-field curvature slot is a
    genuine callable (no galerkin physics has one yet)."""

    def __init__(self, cvec: np.ndarray) -> None:
        self._c = np.asarray(cvec)
        self._g = np.zeros_like(self._c)

    def set_field(self, gvec: np.ndarray) -> None:
        self._g = np.asarray(gvec)

    def field_jacobian(self, state: NumpyArray) -> np.ndarray:
        """S = dR/dg = diag(c * g)."""
        return np.diag(self._c * self._g)

    def field_field_hvp(
        self, state: NumpyArray, adj: NumpyArray, delta: NumpyArray
    ) -> np.ndarray:
        """[lambda^T d^2R/dg^2](delta) = lambda * c * delta."""
        return np.asarray(adj) * self._c * np.asarray(delta)


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
) -> _FieldParameterizationTerm[
    NumpyArray, AdvectionDiffusionReaction[NumpyArray]
]:
    diffusion = physics.diffusion_function()
    assert isinstance(diffusion, NodalFieldDiffusion)
    return _FieldParameterizationTerm.linear_field_state(
        setter=lambda field: diffusion.set_dofs(bkd.to_numpy(field)),
        physics=physics,
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


class TestFieldParameterizationTerm:
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
        param_obj = term
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
            physics=physics,
            field_jacobian=lambda state, time: (
                physics.residual_forcing_jacobian()
            ),
            field_map=forcing_map,
            bkd=bkd,
            nstates=physics.nstates(),
            nfield_dofs=physics.nstates(),
        )
        param_obj = term
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
            physics=physics,
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
        param_obj = term
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
            physics=physics,
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
        param_obj = term
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

    def test_field_field_hvp_slot_matches_fd(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """The field-field curvature slot's contribution to
        param_param_hvp is FD-validated on an analytic quadratic-field
        toy (the slot the galerkin physics never populate; the
        collocation hyperelastic-lame migration is its physics
        exemplar)."""
        bkd = numpy_bkd
        nfield = 5
        rng = np.random.default_rng(37)
        cvec = rng.normal(1.0, 0.3, nfield)
        toy = _QuadraticFieldToy(cvec)
        modes = rng.normal(0.0, 1.0, (nfield, _NPARAMS))
        field_map = MeshKLEFieldMap(
            bkd, bkd.asarray(np.zeros(nfield)), bkd.asarray(modes)
        )
        term = _FieldParameterizationTerm(
            setter=ToNumpySetter(toy.set_field, bkd),
            physics=toy,
            field_jacobian=StateJacobianAdapter(toy.field_jacobian),
            field_state_hvp=Zero(),
            state_field_hvp=Zero(),
            field_field_hvp=MixedHVPAdapter(toy.field_field_hvp),
            field_map=field_map,
            bkd=bkd,
            nstates=nfield,
            nfield_dofs=nfield,
        )
        state = bkd.zeros((nfield,))
        adj = bkd.asarray(rng.normal(0.0, 1.0, nfield))
        adj_np = bkd.to_numpy(adj)
        params = bkd.asarray(np.array([0.4, -0.3, 0.2]))

        # psi(p) = lambda^T R(G(p)) has gradient G'^T S^T lambda
        # (via the engine's param_jacobian) and analytic Hessian
        # H = W^T diag(lambda * c) W (linear map: G' = W, map hvp = 0,
        # so ALL curvature flows through the field-field slot).
        hess = modes.T @ np.diag(adj_np * cvec) @ modes

        def grad_of_params(samples: NumpyArray) -> NumpyArray:
            results = []
            for ii in range(samples.shape[1]):
                term.apply(samples[:, ii])
                results.append(
                    bkd.to_numpy(
                        term.param_jacobian(state, 0.0, samples[:, ii])
                    ).T
                    @ adj_np
                )
            term.apply(params)
            return bkd.asarray(np.stack(results, axis=1))

        def hess_of_params(sample: NumpyArray) -> NumpyArray:
            return bkd.asarray(hess)

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=_NPARAMS,
            nvars=_NPARAMS,
            fun=grad_of_params,
            jacobian=hess_of_params,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(
            bkd.asarray(bkd.to_numpy(params).reshape(-1, 1)), relative=True
        )[0]
        ratio = float(bkd.to_numpy(checker.error_ratio(errors)))
        assert ratio <= 1e-6

        # The engine's param_param_hvp must equal H @ v exactly.
        term.apply(params)
        vvec = bkd.asarray(np.array([0.5, 0.7, -0.6]))
        bkd.assert_allclose(
            term.param_param_hvp(state, 0.0, params, adj, vvec),
            bkd.asarray(hess @ bkd.to_numpy(vvec)),
            rtol=1e-13,
        )

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
                physics=physics,
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
                physics=physics,
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
            physics=physics,
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
            physics=physics,
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


class _DenseLinearFieldToy:
    """Dense-assembly toy physics :math:`R(u) = \\mathrm{diag}(g) A u`.

    Linear in the field and the state, so the engine's
    linear_field_state tier applies with :math:`S(u) =
    \\mathrm{diag}(A u)` and mixed assembly :math:`A(\\delta, u) =
    \\mathrm{diag}(\\delta) A`. Assemblies are DENSE backend arrays,
    exercising the engine's backend-space matmul branch (the galerkin
    physics only ever return scipy sparse).
    """

    def __init__(self, amat, bkd):
        self._amat = amat
        self._bkd = bkd
        self._g = bkd.zeros((amat.shape[0],))

    def nstates(self):
        return self._amat.shape[0]

    def set_field(self, gvec):
        self._g = gvec

    def field_jacobian(self, state, time):
        return self._bkd.diag(self._amat @ state)

    def field_state_jacobian(self, delta, state, time):
        return self._bkd.diag(delta) @ self._amat


class _LinearFieldMapToy:
    """G(p) = Phi p: linear map with exactly zero curvature."""

    def __init__(self, phi, bkd):
        self._phi = phi
        self._bkd = bkd

    def nvars(self):
        return self._phi.shape[1]

    def __call__(self, params_1d):
        return self._phi @ params_1d

    def jacobian(self, params_1d):
        return self._phi

    def hvp(self, params_1d, adj_state, vvec):
        return self._bkd.zeros((self.nvars(),))


class TestDenseBackendAssemblies:
    """Dense backend-array assemblies stay in backend space.

    The galerkin suites cover the sparse branch; these tests drive the
    dense branch on both backends and assert output type and dtype
    match the inputs — a silent numpy round trip produces correct
    numbers with hidden device/dtype conversions, which the value
    checks alone cannot catch.
    """

    def _build(self, bkd):
        npts = 6
        amat = bkd.asarray(np.random.normal(0.0, 1.0, (npts, npts)))
        phi = bkd.asarray(np.random.normal(0.0, 1.0, (npts, _NPARAMS)))
        physics = _DenseLinearFieldToy(amat, bkd)
        field_map = _LinearFieldMapToy(phi, bkd)
        term = _FieldParameterizationTerm.linear_field_state(
            setter=physics.set_field,
            physics=physics,
            field_jacobian=physics.field_jacobian,
            field_state_jacobian=physics.field_state_jacobian,
            field_map=field_map,
            bkd=bkd,
            nstates=npts,
            nfield_dofs=npts,
        )
        state = bkd.asarray(np.random.normal(0.0, 1.0, (npts,)))
        adj = bkd.asarray(np.random.normal(0.0, 1.0, (npts,)))
        params = bkd.asarray(np.random.normal(0.0, 1.0, (_NPARAMS,)))
        return term, amat, phi, state, adj, params

    def test_bundle_is_second_order(self, bkd):
        term = self._build(bkd)[0]
        derivs = term.param_derivatives()
        assert derivs.param_jacobian is not None
        assert derivs.param_param_hvp is not None
        assert derivs.state_param_hvp is not None
        assert derivs.param_state_hvp is not None

    def test_param_jacobian_dense_backend(self, bkd):
        term, amat, phi, state, _, params = self._build(bkd)
        out = term.param_jacobian(state, 0.0, params)
        assert isinstance(out, type(state))
        assert out.dtype == state.dtype
        expected = bkd.diag(amat @ state) @ phi
        bkd.assert_allclose(out, expected, rtol=1e-12)

    def test_hvps_dense_backend(self, bkd):
        term, amat, phi, state, adj, params = self._build(bkd)
        vvec = bkd.asarray(np.random.normal(0.0, 1.0, (_NPARAMS,)))
        wvec = bkd.asarray(np.random.normal(0.0, 1.0, (state.shape[0],)))

        # Linear term through a linear map: exactly zero curvature.
        out_pp = term.param_param_hvp(state, 0.0, params, adj, vvec)
        assert isinstance(out_pp, type(state))
        assert out_pp.dtype == state.dtype
        bkd.assert_allclose(out_pp, bkd.zeros((_NPARAMS,)), atol=1e-15)

        delta = phi @ vvec
        out_sp = term.state_param_hvp(state, 0.0, params, adj, vvec)
        assert isinstance(out_sp, type(state))
        assert out_sp.dtype == state.dtype
        expected_sp = (bkd.diag(delta) @ amat).T @ adj
        bkd.assert_allclose(out_sp, expected_sp, rtol=1e-12)

        out_ps = term.param_state_hvp(state, 0.0, params, adj, wvec)
        assert isinstance(out_ps, type(state))
        assert out_ps.dtype == state.dtype
        expected_ps = phi.T @ (bkd.diag(amat @ wvec) @ adj)
        bkd.assert_allclose(out_ps, expected_ps, rtol=1e-12)

    def test_param_jacobian_preserves_autograd(self, torch_bkd):
        """A numpy detour raises on a grad-requiring tensor; the dense
        branch must keep the computation graph intact."""
        import torch

        term = self._build(torch_bkd)[0]
        params = torch_bkd.asarray(np.random.normal(0.0, 1.0, (_NPARAMS,)))
        state = torch.randn(6, dtype=torch.float64, requires_grad=True)
        out = term.param_jacobian(state, 0.0, params)
        assert out.grad_fn is not None
