"""The raw parameterized residual and the essential-boundary flux.

Setup throughout: -(kappa u')' (+ optional u^2 reaction) = 1 on [0, 1],
u(0) = u(1) = 0, P1 elements on 10 cells, kappa = exp(W p) with three
smooth modes (p = 0 gives kappa = 1).

The flux is ``essential_boundary_flux`` on the steady residual
(``galerkin_steady_residual``), solved by Newton (atol 1e-13) with sparse
LU; its derivatives through the solve come from
``SteadyImplicitFunction``, checked by central finite differences.

Checks that do not depend on the derivation:
- global balance: with unit weights on every essential row the flux is
  minus the total applied load, for any kappa(p) (constants are in the
  diffusion operator's kernel);
- exact flux: at p = 0, P1 is nodally exact, so the left flux equals
  -kappa u'(0) = -1/2 to roundoff.
"""

import pickle
from typing import Any, List, Optional, Tuple

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np

from pyapprox.implicitfunction.observed_residual import ObservedResidual
from pyapprox.implicitfunction.pinned_residual import PinnedResidual
from pyapprox.implicitfunction.steady_function import SteadyImplicitFunction
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
    FunctionWithJVPFromCallable,
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
from pyapprox.pde.galerkin.time_integration import skfem_discretization
from pyapprox.pde.models.galerkin.essential_boundary_flux import (
    essential_boundary_flux,
)
from pyapprox.pde.models.galerkin.parameterized_residual import (
    GalerkinParameterizedResidual,
    ParameterizedResidualDerivatives,
)
from pyapprox.pde.models.galerkin.steady_residual import galerkin_steady_residual
from pyapprox.pde.parameterizations.galerkin_advection_diffusion import (
    AdvectionDiffusionParameterization,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.util.backends.numpy import NumpyBkd
from tests._helpers.adjoint_checks import NoHVPQuadraticFieldMap, NumpyArray
from tests._helpers.galerkin_steady_checks import flux_steady_function

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


def _steady(
    system: GalerkinSystem[NumpyArray],
    param_obj: AdvectionDiffusionParameterization[NumpyArray],
) -> Tuple[PinnedResidual[NumpyArray, Any], ParameterState[NumpyArray]]:
    """The steady residual and the parameter state it reads."""
    parameters = ParameterState(param_obj)
    residual = galerkin_steady_residual(skfem_discretization(system), parameters)
    return residual, parameters


def _flux(
    bkd: NumpyBkd,
    residual: PinnedResidual[NumpyArray, Any],
    dofs: List[int],
    weights: Optional[np.ndarray] = None,
) -> ObservedResidual[NumpyArray, NumpyArray]:
    if weights is None:
        weights = np.ones((1, len(dofs)))
    return essential_boundary_flux(
        bkd,
        residual,
        bkd.asarray(np.array(dofs), dtype=bkd.int64_dtype()),
        bkd.asarray(weights),
    )


def _flux_function(
    bkd: NumpyBkd,
    system: GalerkinSystem[NumpyArray],
    param_obj: AdvectionDiffusionParameterization[NumpyArray],
    dofs: List[int],
    weights: Optional[np.ndarray] = None,
) -> SteadyImplicitFunction[NumpyArray, Any]:
    """The flux of the solved state as a function of the parameters."""
    if weights is None:
        weights = np.ones((1, len(dofs)))
    return flux_steady_function(
        system, param_obj, dofs, bkd.asarray(weights), bkd
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


def _both_ends(bkd: NumpyBkd, basis: LagrangeBasis[NumpyArray]) -> List[int]:
    return [
        _boundary_dof(bkd, basis, "left"),
        _boundary_dof(bkd, basis, "right"),
    ]


def _random_state(nstates: int, seed: int) -> NumpyArray:
    return np.random.default_rng(seed).normal(size=(nstates, 1))


class TestEssentialBoundaryFlux:
    def test_value_is_minus_weighted_raw_rows(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        bkd = numpy_bkd
        system, param_obj, basis = _problem(bkd)
        left, right = _both_ends(bkd, basis)
        residual, parameters = _steady(system, param_obj)
        flux = _flux(bkd, residual, [left, right], np.array([[2.0, -0.5]]))
        state = bkd.asarray(_random_state(system.nstates(), 4))
        parameters.set_param(bkd.asarray(_PARAM[:, 0]))
        raw = system.spatial_operator().spatial_residual(state[:, 0], 0.0)
        expected = -(2.0 * raw[left] - 0.5 * raw[right])
        bkd.assert_allclose(
            flux.value(state),
            bkd.reshape(bkd.asarray([expected]), (1, 1)),
            rtol=1e-14,
        )

    def test_global_balance(self, numpy_bkd: NumpyBkd) -> None:
        """Unit weights on all essential rows: the total flux is minus
        the applied load (here 1), whatever the diffusivity."""
        bkd = numpy_bkd
        system, param_obj, basis = _problem(bkd)
        function = _flux_function(bkd, system, param_obj, _both_ends(bkd, basis))
        for param_np in (np.zeros((_NPARAMS, 1)), _PARAM):
            bkd.assert_allclose(
                function(bkd.asarray(param_np)),
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
        function = _flux_function(
            bkd, system, param_obj, [_boundary_dof(bkd, basis, "left")]
        )
        bkd.assert_allclose(
            function(bkd.zeros((_NPARAMS, 1))),
            bkd.asarray([[-0.5]]),
            rtol=1e-12,
        )

    def test_rejects_bad_arguments(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        system, param_obj, basis = _problem(bkd)
        left = _boundary_dof(bkd, basis, "left")
        residual, _ = _steady(system, param_obj)
        with pytest.raises(ValueError, match="not essential"):
            _flux(bkd, residual, [left + 1])
        with pytest.raises(ValueError, match="unique"):
            _flux(bkd, residual, [left, left])
        with pytest.raises(ValueError, match="at least one"):
            _flux(bkd, residual, [], np.ones((1, 0)))
        with pytest.raises(ValueError, match="not in"):
            _flux(bkd, residual, [system.nstates()])
        with pytest.raises(ValueError, match="shape"):
            _flux(bkd, residual, [left], np.ones((1, 2)))
        with pytest.raises(TypeError, match="PinnedResidual"):
            essential_boundary_flux(
                bkd,
                residual.unpinned(),  # type: ignore[arg-type]
                bkd.asarray(np.array([left]), dtype=bkd.int64_dtype()),
                bkd.ones((1, 1)),
            )

    def test_bundle_follows_the_residual(self, numpy_bkd: NumpyBkd) -> None:
        """Second order for any number of QoIs when the residual's bundle
        is; first order without the field map's HVP."""
        bkd = numpy_bkd
        system, param_obj, basis = _problem(bkd)
        dofs = _both_ends(bkd, basis)
        residual, _ = _steady(system, param_obj)
        assert _flux(bkd, residual, dofs).derivatives().state_state is not None
        vector = _flux(bkd, residual, dofs, np.eye(2)).derivatives()
        assert vector.state_state is not None
        system, param_obj, basis = _problem(bkd, hvp_map=False)
        residual, _ = _steady(system, param_obj)
        first = _flux(bkd, residual, dofs).derivatives()
        assert first.wrt_param is not None
        assert first.state_state is None

    def test_adjoint_gradient_and_hvp_match_fd(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """The flux as a QoI of the steady solve: the adjoint gradient and
        HVP by central finite differences, and the HVP symmetric."""
        bkd = numpy_bkd
        system, param_obj, basis = _problem(bkd, reaction=True)
        function = _flux_function(
            bkd, system, param_obj, [_boundary_dof(bkd, basis, "left")]
        )
        param = bkd.asarray(_PARAM)
        errors = DerivativeChecker(function).check_derivatives(
            param, relative=True, central=True
        )
        assert len(errors) == 2
        assert max(bkd.to_float(bkd.min(e)) for e in errors) < 1e-6
        hvp = function.derivatives().hvp
        assert hvp is not None
        vvec = bkd.asarray(np.array([[0.5], [0.7], [-0.6]]))
        uvec = bkd.asarray(np.array([[-0.2], [0.9], [0.3]]))
        bkd.assert_allclose(
            bkd.sum(hvp(param, vvec) * uvec),
            bkd.sum(hvp(param, uvec) * vvec),
            rtol=1e-12,
        )

    def test_vector_flux_jacobian_matches_fd(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """Two QoIs: the state derivative of the term, and the parameter
        Jacobian through the solve."""
        bkd = numpy_bkd
        system, param_obj, basis = _problem(bkd, reaction=True)
        dofs = _both_ends(bkd, basis)
        residual, parameters = _steady(system, param_obj)
        parameters.set_param(bkd.asarray(_PARAM[:, 0]))
        flux = _flux(bkd, residual, dofs, np.eye(2))
        wrt_state = flux.derivatives().wrt_state
        assert wrt_state is not None
        nstates = system.nstates()
        eye = bkd.eye(nstates)

        def values(samples: NumpyArray) -> NumpyArray:
            return bkd.hstack(
                [flux.value(samples[:, ii : ii + 1]) for ii in range(samples.shape[1])]
            )

        def jacobian(state: NumpyArray) -> NumpyArray:
            linearization = wrt_state(state)
            return bkd.hstack(
                [linearization.apply(eye[:, jj : jj + 1]) for jj in range(nstates)]
            )

        checker = DerivativeChecker(
            FunctionWithJacobianFromCallable(
                nqoi=2, nvars=nstates, fun=values, jacobian=jacobian, bkd=bkd
            )
        )
        state = bkd.asarray(_random_state(nstates, 5))
        errors = checker.check_derivatives(state, relative=True)[0]
        assert bkd.to_float(checker.error_ratio(errors)) <= 1e-6

        function = _flux_function(bkd, system, param_obj, dofs, np.eye(2))
        solved_errors = DerivativeChecker(function).check_derivatives(
            bkd.asarray(_PARAM), relative=True, central=True
        )
        assert bkd.to_float(bkd.min(solved_errors[0])) < 1e-6

    def test_state_state_hvp_matches_fd(self, numpy_bkd: NumpyBkd) -> None:
        """Nonzero only through the u^2 reaction."""
        bkd = numpy_bkd
        system, param_obj, basis = _problem(bkd, reaction=True)
        residual, parameters = _steady(system, param_obj)
        parameters.set_param(bkd.asarray(_PARAM[:, 0]))
        flux = _flux(bkd, residual, [_boundary_dof(bkd, basis, "left")])
        derivs = flux.derivatives()
        wrt_state, state_state = derivs.wrt_state, derivs.state_state
        assert wrt_state is not None and state_state is not None
        weights = bkd.ones((1, 1))

        def gradient(samples: NumpyArray) -> NumpyArray:
            return bkd.hstack(
                [
                    wrt_state(samples[:, ii : ii + 1]).apply_transpose(weights)
                    for ii in range(samples.shape[1])
                ]
            )

        def curvature(state: NumpyArray, w: NumpyArray) -> NumpyArray:
            block = state_state(state, weights, w)
            assert block is not None
            return block

        nstates = system.nstates()
        checker = DerivativeChecker(
            FunctionWithJVPFromCallable(
                nqoi=nstates, nvars=nstates, fun=gradient, jvp=curvature, bkd=bkd
            )
        )
        state = bkd.asarray(_random_state(nstates, 6))
        errors = checker.check_derivatives(state, relative=True)[0]
        assert bkd.to_float(checker.error_ratio(errors)) <= 1e-6

    def test_pickles(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        system, param_obj, basis = _problem(bkd, reaction=True)
        residual, parameters = _steady(system, param_obj)
        parameters.set_param(bkd.asarray(_PARAM[:, 0]))
        flux = _flux(bkd, residual, [_boundary_dof(bkd, basis, "left")])
        clone = pickle.loads(pickle.dumps(flux))
        state = bkd.asarray(_random_state(system.nstates(), 7))
        bkd.assert_allclose(clone.value(state), flux.value(state), rtol=1e-14)
