"""The raw parameterized residual and the essential-boundary flux.

Setup throughout: -(kappa u')' (+ optional u^2 reaction) = 1 on [0, 1],
u(0) = u(1) = 0, P1 elements on 10 cells, kappa = exp(W p) with three
smooth modes (p = 0 gives kappa = 1).

Checks that do not depend on the derivation:
- global balance: with unit weights on every essential row the flux is
  minus the total applied load, for any kappa(p) (constants are in the
  diffusion operator's kernel);
- exact flux: at p = 0, P1 is nodally exact, so the left flux equals
  -kappa u'(0) = -1/2 to roundoff.
"""

import pickle
from typing import Any, Optional, Tuple

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
    FunctionWithJVPFromCallable,
)
from pyapprox.optimization.implicitfunction.operator.adjoint_operator import (
    AdjointOperator,
)
from pyapprox.optimization.implicitfunction.operator.check_derivatives import (
    ImplicitFunctionDerivativeChecker,
)
from pyapprox.pde.constitutive.coefficient_functions import (
    CallableReaction,
    NodalFieldDiffusion,
    TimeIndependent,
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
from pyapprox.pde.models.galerkin.essential_boundary_flux import (
    EssentialBoundaryFluxFunctional,
)
from pyapprox.pde.models.galerkin.parameterized_residual import (
    GalerkinParameterizedResidual,
    ParameterizedResidualDerivatives,
)
from pyapprox.pde.models.galerkin.steady import (
    GalerkinParameterizedStateEquation,
)
from pyapprox.pde.parameterizations.galerkin_advection_diffusion import (
    AdvectionDiffusionParameterization,
)
from pyapprox.util.backends.numpy import NumpyBkd
from tests._helpers.adjoint_checks import NoHVPQuadraticFieldMap, NumpyArray

_NPARAMS = 3
_NX = 10


def _unit_forcing(x: Any) -> Any:
    return np.ones(x.shape[1])


def _square(x: Any, u: Any) -> Any:
    return u**2


def _square_deriv(x: Any, u: Any) -> Any:
    return 2.0 * u


def _square_deriv2(x: Any, u: Any) -> Any:
    return np.full_like(u, 2.0)


def _log_kle_map(
    bkd: NumpyBkd, basis: LagrangeBasis[NumpyArray]
) -> TransformedFieldMap[NumpyArray]:
    """kappa = exp(W p): three smooth modes on the DOF nodes."""
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
    return TransformedFieldMap(kle, exp, exp, bkd, transform_deriv2=exp)


def _problem(
    bkd: NumpyBkd, reaction: bool = False, hvp_map: bool = True
) -> Tuple[
    GalerkinSystem[NumpyArray],
    AdvectionDiffusionParameterization[NumpyArray],
    LagrangeBasis[NumpyArray],
]:
    mesh = StructuredMesh1D(nx=_NX, bounds=(0.0, 1.0), bkd=bkd)
    basis = LagrangeBasis(mesh, degree=1)
    physics = AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=NodalFieldDiffusion(basis),
        bkd=bkd,
        reaction=(
            CallableReaction(_square, _square_deriv, _square_deriv2)
            if reaction
            else None
        ),
        forcing=TimeIndependent(_unit_forcing),
    )
    system = compose_galerkin_system(
        physics,
        [
            DirichletBC(basis, "left", 0.0, bkd),
            DirichletBC(basis, "right", 0.0, bkd),
        ],
    )
    if hvp_map:
        field_map: Any = _log_kle_map(bkd, basis)
    else:
        coords = bkd.to_numpy(basis.dof_coordinates())[0]
        modes = np.stack(
            [np.ones_like(coords), coords, coords**2], axis=1
        )
        field_map = NoHVPQuadraticFieldMap(
            bkd, bkd.asarray(np.ones(coords.shape[0])), bkd.asarray(modes)
        )
    param_obj = AdvectionDiffusionParameterization(
        physics, diffusivity_map=field_map, bkd=bkd
    )
    return system, param_obj, basis


def _boundary_dof(
    bkd: NumpyBkd, basis: LagrangeBasis[NumpyArray], name: str
) -> int:
    dofs = bkd.to_numpy(basis.get_dofs(name))
    return int(dofs[0])


def _flux(
    bkd: NumpyBkd,
    system: GalerkinSystem[NumpyArray],
    param_obj: AdvectionDiffusionParameterization[NumpyArray],
    dofs: list,
    weights: Optional[np.ndarray] = None,
) -> EssentialBoundaryFluxFunctional[NumpyArray]:
    residual = GalerkinParameterizedResidual(
        system.spatial_operator(), param_obj
    )
    if weights is None:
        weights = np.ones((1, len(dofs)))
    return EssentialBoundaryFluxFunctional(
        residual,
        system.steady(),
        bkd.asarray(np.array(dofs), dtype=bkd.int64_dtype()),
        bkd.asarray(weights),
    )


def _solve(
    bkd: NumpyBkd,
    system: GalerkinSystem[NumpyArray],
    param_obj: AdvectionDiffusionParameterization[NumpyArray],
    param_np: np.ndarray,
) -> NumpyArray:
    state_eq = GalerkinParameterizedStateEquation(
        system.steady(), param_obj, bkd
    )
    return state_eq.solve(
        bkd.zeros((system.nstates(), 1)), bkd.asarray(param_np)
    )


_PARAM = np.array([[0.4], [-0.3], [0.2]])


def _jac(state: Any, time: float, params: Any) -> Any:
    return state


def _hvp(state: Any, time: float, params: Any, adj: Any, vec: Any) -> Any:
    return vec


class TestParameterizedResidualDerivatives:
    def test_named_constructors(self) -> None:
        assert ParameterizedResidualDerivatives.none().param_jacobian is None
        first = ParameterizedResidualDerivatives.first_order(_jac)
        assert first.param_jacobian is _jac
        assert first.state_state_hvp is None
        second = ParameterizedResidualDerivatives.second_order(
            _jac, _hvp, _hvp, _hvp, _hvp
        )
        assert second.param_param_hvp is _hvp

    def test_rejects_a_partial_hvp_set(self) -> None:
        with pytest.raises(TypeError, match="together"):
            ParameterizedResidualDerivatives(
                param_jacobian=_jac, state_state_hvp=_hvp
            )

    def test_hvps_require_the_jacobian(self) -> None:
        with pytest.raises(TypeError, match="require"):
            ParameterizedResidualDerivatives(
                state_state_hvp=_hvp,
                state_param_hvp=_hvp,
                param_state_hvp=_hvp,
                param_param_hvp=_hvp,
            )

    def test_rejects_a_non_callable_field(self) -> None:
        with pytest.raises(TypeError, match="param_jacobian"):
            ParameterizedResidualDerivatives(
                param_jacobian=1.0,  # type: ignore[arg-type]
            )


class TestParameterizedResidual:
    def test_matches_operator_after_apply(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        system, param_obj, _ = _problem(bkd)
        residual = GalerkinParameterizedResidual(
            system.spatial_operator(), param_obj
        )
        state = bkd.asarray(
            np.random.default_rng(1).normal(size=system.nstates())
        )
        params = bkd.asarray(_PARAM[:, 0])
        value = residual.residual(state, 0.0, params)
        param_obj.apply(params)
        bkd.assert_allclose(
            value,
            system.spatial_operator().spatial_residual(state, 0.0),
            rtol=1e-14,
        )

    def test_applies_params_itself(self, numpy_bkd: NumpyBkd) -> None:
        """A later call with other parameters is not stale."""
        bkd = numpy_bkd
        system, param_obj, _ = _problem(bkd)
        residual = GalerkinParameterizedResidual(
            system.spatial_operator(), param_obj
        )
        state = bkd.asarray(
            np.random.default_rng(2).normal(size=system.nstates())
        )
        first = residual.residual(state, 0.0, bkd.asarray(_PARAM[:, 0]))
        residual.residual(state, 0.0, bkd.zeros((_NPARAMS,)))
        again = residual.residual(state, 0.0, bkd.asarray(_PARAM[:, 0]))
        bkd.assert_allclose(again, first, rtol=1e-14)

    def test_rejects_unowned_parameterization(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        bkd = numpy_bkd
        system, _, _ = _problem(bkd)
        _, other_param, _ = _problem(bkd)
        with pytest.raises(ValueError, match="does not hold"):
            GalerkinParameterizedResidual(
                system.spatial_operator(), other_param
            )

    def test_capability_follows_the_bundles(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        bkd = numpy_bkd
        system, param_obj, _ = _problem(bkd)
        derivs = GalerkinParameterizedResidual(
            system.spatial_operator(), param_obj
        ).derivatives()
        assert derivs.param_param_hvp is not None
        system, param_obj, _ = _problem(bkd, hvp_map=False)
        derivs = GalerkinParameterizedResidual(
            system.spatial_operator(), param_obj
        ).derivatives()
        assert derivs.param_jacobian is not None
        assert derivs.param_param_hvp is None

    def test_param_jacobian_matches_fd(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        system, param_obj, _ = _problem(bkd, reaction=True)
        residual = GalerkinParameterizedResidual(
            system.spatial_operator(), param_obj
        )
        param_jacobian = residual.derivatives().param_jacobian
        assert param_jacobian is not None
        state = bkd.asarray(
            np.random.default_rng(3).normal(size=system.nstates())
        )

        def values(samples: NumpyArray) -> NumpyArray:
            return bkd.stack(
                [
                    residual.residual(state, 0.0, samples[:, ii])
                    for ii in range(samples.shape[1])
                ],
                axis=1,
            )

        checker = DerivativeChecker(
            FunctionWithJacobianFromCallable(
                nqoi=system.nstates(),
                nvars=_NPARAMS,
                fun=values,
                jacobian=lambda p: param_jacobian(state, 0.0, p[:, 0]),
                bkd=bkd,
            )
        )
        # The error falls at first order to a roundoff floor near 1e-8
        # against a 1e-2 maximum, which puts the 1e-6 error ratio on a
        # knife edge; the shape of the curve is the robust test.
        fd_eps = bkd.flip(bkd.logspace(-13, -1, 13))
        errors = checker.check_derivatives(
            bkd.asarray(_PARAM), fd_eps=fd_eps, relative=True
        )[0]
        assert checker.check_v_shape(errors, fd_eps).passed


class TestEssentialBoundaryFluxFunctional:
    def test_value_is_minus_weighted_raw_rows(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        bkd = numpy_bkd
        system, param_obj, basis = _problem(bkd)
        left = _boundary_dof(bkd, basis, "left")
        right = _boundary_dof(bkd, basis, "right")
        weights = np.array([[2.0, -0.5]])
        functional = _flux(bkd, system, param_obj, [left, right], weights)
        state = bkd.asarray(
            np.random.default_rng(4).normal(size=(system.nstates(), 1))
        )
        param = bkd.asarray(_PARAM)
        param_obj.apply(param[:, 0])
        raw = system.spatial_operator().spatial_residual(state[:, 0], 0.0)
        expected = -(2.0 * raw[left] - 0.5 * raw[right])
        bkd.assert_allclose(
            functional(state, param),
            bkd.reshape(bkd.asarray([expected]), (1, 1)),
            rtol=1e-14,
        )

    def test_global_balance(self, numpy_bkd: NumpyBkd) -> None:
        """Unit weights on all essential rows: the total flux is minus
        the applied load (here 1), whatever the diffusivity."""
        bkd = numpy_bkd
        system, param_obj, basis = _problem(bkd)
        dofs = [
            _boundary_dof(bkd, basis, "left"),
            _boundary_dof(bkd, basis, "right"),
        ]
        functional = _flux(bkd, system, param_obj, dofs)
        for param_np in (np.zeros((_NPARAMS, 1)), _PARAM):
            state = _solve(bkd, system, param_obj, param_np)
            bkd.assert_allclose(
                functional(state, bkd.asarray(param_np)),
                bkd.asarray([[-1.0]]),
                rtol=1e-10,
            )

    def test_exact_flux_at_unit_diffusivity(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """kappa = 1, f = 1: u = x(1 - x)/2, so the flux out of x = 0 is
        -kappa u'(0) = -1/2; P1 is nodally exact in 1D."""
        bkd = numpy_bkd
        system, param_obj, basis = _problem(bkd)
        functional = _flux(
            bkd, system, param_obj, [_boundary_dof(bkd, basis, "left")]
        )
        param = np.zeros((_NPARAMS, 1))
        state = _solve(bkd, system, param_obj, param)
        bkd.assert_allclose(
            functional(state, bkd.asarray(param)),
            bkd.asarray([[-0.5]]),
            rtol=1e-12,
        )

    def test_rejects_bad_arguments(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        system, param_obj, basis = _problem(bkd)
        left = _boundary_dof(bkd, basis, "left")
        with pytest.raises(ValueError, match="not essential"):
            _flux(bkd, system, param_obj, [left + 1])
        with pytest.raises(ValueError, match="unique"):
            _flux(bkd, system, param_obj, [left, left])
        with pytest.raises(ValueError, match="shape"):
            _flux(bkd, system, param_obj, [left], np.ones((1, 2)))
        other_system, _, _ = _problem(bkd)
        residual = GalerkinParameterizedResidual(
            system.spatial_operator(), param_obj
        )
        with pytest.raises(ValueError, match="share"):
            EssentialBoundaryFluxFunctional(
                residual,
                other_system.steady(),
                bkd.asarray(np.array([left]), dtype=bkd.int64_dtype()),
                bkd.ones((1, 1)),
            )

    def test_bundle_follows_residual_and_nqoi(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        bkd = numpy_bkd
        system, param_obj, basis = _problem(bkd)
        dofs = [
            _boundary_dof(bkd, basis, "left"),
            _boundary_dof(bkd, basis, "right"),
        ]
        scalar = _flux(bkd, system, param_obj, dofs).derivatives()
        assert scalar.state_state_hvp is not None
        vector = _flux(
            bkd, system, param_obj, dofs, np.eye(2)
        ).derivatives()
        assert vector.state_jacobian is not None
        assert vector.state_state_hvp is None
        system, param_obj, basis = _problem(bkd, hvp_map=False)
        first = _flux(bkd, system, param_obj, dofs).derivatives()
        assert first.param_jacobian is not None
        assert first.state_state_hvp is None

    def test_adjoint_gradient_and_hvp_match_fd(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """The flux as a QoI of the steady solve: every component and the
        assembled gradient and HVP, by finite differences."""
        bkd = numpy_bkd
        system, param_obj, basis = _problem(bkd, reaction=True)
        functional = _flux(
            bkd, system, param_obj, [_boundary_dof(bkd, basis, "left")]
        )
        state_eq = GalerkinParameterizedStateEquation(
            system.steady(), param_obj, bkd
        )
        adjoint_op = AdjointOperator(state_eq, functional)
        checker = ImplicitFunctionDerivativeChecker(adjoint_op)
        tols = bkd.copy(checker.get_derivative_tolerances(1e-6))
        # One-sided FD of PDE-scale quantities floors near 1e-6 for the
        # assembled gradient and two state-equation contractions, as in
        # the steady adjoint HVP tests, and for the flux functional's four
        # contractions (9-12), which read the same residual rows: their
        # errors fall at first order from 1e-2 to floors giving ratios of
        # about 2e-6. The symmetry check below is immune to that noise.
        tols[4] = 5e-6
        tols[5] = 5e-6
        tols[8] = 5e-6
        tols[9:13] = 5e-6
        checker.check_derivatives(
            bkd.zeros((system.nstates(), 1)), bkd.asarray(_PARAM), tols
        )
        hvp = adjoint_op.derivatives().hvp
        assert hvp is not None
        init = bkd.zeros((system.nstates(), 1))
        vvec = bkd.asarray(np.array([[0.5], [0.7], [-0.6]]))
        uvec = bkd.asarray(np.array([[-0.2], [0.9], [0.3]]))
        param = bkd.asarray(_PARAM)
        bkd.assert_allclose(
            bkd.sum(hvp(init, param, vvec) * uvec),
            bkd.sum(hvp(init, param, uvec) * vvec),
            rtol=1e-12,
        )

    def test_vector_flux_jacobian_matches_fd(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        bkd = numpy_bkd
        system, param_obj, basis = _problem(bkd, reaction=True)
        dofs = [
            _boundary_dof(bkd, basis, "left"),
            _boundary_dof(bkd, basis, "right"),
        ]
        functional = _flux(bkd, system, param_obj, dofs, np.eye(2))
        derivs = functional.derivatives()
        state_jacobian = derivs.state_jacobian
        assert state_jacobian is not None
        param = bkd.asarray(_PARAM)

        def values(samples: NumpyArray) -> NumpyArray:
            return bkd.hstack(
                [
                    functional(samples[:, ii : ii + 1], param)
                    for ii in range(samples.shape[1])
                ]
            )

        checker = DerivativeChecker(
            FunctionWithJacobianFromCallable(
                nqoi=2,
                nvars=system.nstates(),
                fun=values,
                jacobian=lambda u: state_jacobian(u, param),
                bkd=bkd,
            )
        )
        state = bkd.asarray(
            np.random.default_rng(5).normal(size=(system.nstates(), 1))
        )
        errors = checker.check_derivatives(state, relative=True)[0]
        assert bkd.to_float(checker.error_ratio(errors)) <= 1e-6

    def test_state_state_hvp_matches_fd(self, numpy_bkd: NumpyBkd) -> None:
        """Nonzero only through the u^2 reaction."""
        bkd = numpy_bkd
        system, param_obj, basis = _problem(bkd, reaction=True)
        functional = _flux(
            bkd, system, param_obj, [_boundary_dof(bkd, basis, "left")]
        )
        derivs = functional.derivatives()
        state_jacobian = derivs.state_jacobian
        state_state_hvp = derivs.state_state_hvp
        assert state_jacobian is not None and state_state_hvp is not None
        param = bkd.asarray(_PARAM)
        checker = DerivativeChecker(
            FunctionWithJVPFromCallable(
                nqoi=system.nstates(),
                nvars=system.nstates(),
                fun=lambda u: state_jacobian(u, param).T,
                jvp=lambda u, w: state_state_hvp(u, param, w),
                bkd=bkd,
            )
        )
        state = bkd.asarray(
            np.random.default_rng(6).normal(size=(system.nstates(), 1))
        )
        errors = checker.check_derivatives(state, relative=True)[0]
        assert bkd.to_float(checker.error_ratio(errors)) <= 1e-6

    def test_pickles(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        system, param_obj, basis = _problem(bkd, reaction=True)
        functional = _flux(
            bkd, system, param_obj, [_boundary_dof(bkd, basis, "left")]
        )
        clone = pickle.loads(pickle.dumps(functional))
        state = bkd.asarray(
            np.random.default_rng(7).normal(size=(system.nstates(), 1))
        )
        param = bkd.asarray(_PARAM)
        bkd.assert_allclose(
            clone(state, param), functional(state, param), rtol=1e-14
        )
