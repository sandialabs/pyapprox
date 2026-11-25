r"""The initial value as a parameterization target.

``u0(p) = u + Phi p + r p_0 p_1`` written by an
:class:`InitialValueParameterization` placed second in a composite after
a one-parameter coefficient part, so its parameters are a slice and the
coefficient part's columns of the initial-state Jacobian are zero. The
base initial state read through :class:`ParameterizedInitialState`
follows the parameters, and its Jacobian and curvature pass
finite-difference checks; the order follows the function's bundle;
mismatches are refused; :class:`Owners` holds what any member holds.
"""

from typing import Any, Generic, List, Tuple

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.interface.functions.fromcallable.hessian import (
    FunctionWithJacobianAndHVPFromCallable,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.ode.residual_derivatives import backend_spaces
from pyapprox.ode.residual_form import BackendColumnForm
from pyapprox.pde.ownership import Owners, owns
from pyapprox.pde.parameterizations.binding import require_owned_targets
from pyapprox.pde.parameterizations.composite import CompositeParameterization
from pyapprox.pde.parameterizations.derivatives import (
    DenseParamJacobian,
    ParamDerivatives,
)
from pyapprox.pde.parameterizations.initial_value import (
    InitialValueData,
    InitialValueParameterization,
    ParameterizedInitialState,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.util.backends.protocols import Array, Backend

_NSTATES = 3
_PARAM = [0.25, 0.7, -0.4]  # the coefficient's, then u0's two


class _U0(Generic[Array]):
    r""":math:`u_0(p) = u + \Phi p + r\, p_0 p_1`."""

    def __init__(self, bkd: Backend[Array], order: int = 2) -> None:
        self._bkd = bkd
        self.u = bkd.asarray([[0.5], [-1.0], [2.0]])
        self.phi = bkd.asarray([[1.0, 0.2], [-0.3, 0.8], [0.4, 0.1]])
        self.r = bkd.asarray([[0.6], [1.1], [-0.7]])
        self._derivs: Derivatives[Array] = (
            Derivatives.none()
            if order == 0
            else Derivatives.first_order(self.jacobian)
            if order == 1
            else Derivatives.second_order_weighted(self.jacobian, self.whvp)
        )

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return 2

    def nqoi(self) -> int:
        return _NSTATES

    def __call__(self, samples: Array) -> Array:
        return self.u + self.phi @ samples + self.r * (samples[0:1] * samples[1:2])

    def jacobian(self, sample: Array) -> Array:
        bkd = self._bkd
        return self.phi + self.r @ bkd.stack([sample[1, 0], sample[0, 0]])[None, :]

    def whvp(self, sample: Array, vec: Array, weights: Array) -> Array:
        c = self._bkd.sum(weights * self.r)
        return c * self._bkd.stack([vec[1], vec[0]])

    def derivatives(self) -> Derivatives[Array]:
        return self._derivs


class _Coefficient(Generic[Array]):
    """A one-parameter coefficient part: it reads no initial state, so its
    initial-state derivatives are zero."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self.written: List[Array] = []

    def nparams(self) -> int:
        return 1

    def targets(self) -> Tuple[object, ...]:
        return (self,)

    def owned_coefficients(self) -> Tuple[str, ...]:
        return ("diffusion",)

    def apply(self, params_1d: Array) -> None:
        self.written.append(params_1d)

    def param_derivatives(self) -> ParamDerivatives[Array]:
        return ParamDerivatives.second_order(
            self._param_jacobian,
            self._initial_param_jacobian,
            self._to_params,
            self._to_state,
            self._to_params,
            self._initial_param_hvp,
        )

    def _param_jacobian(
        self, state: Array, time: float, params_1d: Array
    ) -> DenseParamJacobian[Array]:
        return DenseParamJacobian(self._bkd.zeros((state.shape[0], 1)))

    def _initial_param_jacobian(self, params_1d: Array) -> Array:
        return self._bkd.zeros((_NSTATES, 1))

    def _initial_param_hvp(self, params_1d: Array, weight: Array, vvec: Array) -> Array:
        return self._bkd.zeros((1,))

    def _to_params(self, *args: Any) -> Array:
        return self._bkd.zeros((1,))

    def _to_state(self, state: Array, *args: Any) -> Array:
        return self._bkd.zeros((state.shape[0],))


class _Problem(Generic[Array]):
    """The composite over the coefficient and the initial value, and the
    base initial state reading it."""

    def __init__(self, bkd: Backend[Array], order: int = 2) -> None:
        self.bkd = bkd
        self.u0 = _U0(bkd, order)
        self.data = InitialValueData(bkd.zeros((_NSTATES, 1)))
        self.coefficient = _Coefficient(bkd)
        self.parameters = ParameterState(
            CompositeParameterization(
                [self.coefficient, InitialValueParameterization(self.data, self.u0)],
                bkd,
            )
        )
        self.states, self.pspace = backend_spaces(bkd, _NSTATES, len(_PARAM))
        self.initial = ParameterizedInitialState(
            self.data,
            self.parameters,
            self.states,
            self.pspace,
            BackendColumnForm(bkd),
            bkd,
        )
        self.parameters.set_param(bkd.asarray(_PARAM))

    def set(self, sample: Array) -> None:
        self.parameters.set_param(sample[:, 0])

    def jacobian_matrix(self) -> Array:
        field = self.initial.derivatives().param_derivative
        assert field is not None
        op = field()
        eye = self.bkd.eye(len(_PARAM))
        return self.bkd.hstack([op.apply(eye[:, k : k + 1]) for k in range(3)])


def _sample(bkd: Backend[Array]) -> Array:
    return bkd.asarray(np.reshape(_PARAM, (-1, 1)))


class TestValue:
    def test_is_written_by_the_parameterization(self, bkd: Backend[Array]) -> None:
        problem = _Problem(bkd)
        expected = problem.u0(_sample(bkd)[1:])
        bkd.assert_allclose(problem.initial.value(), expected, rtol=1e-14)
        bkd.assert_allclose(problem.coefficient.written[-1], bkd.asarray(_PARAM[:1]))

    def test_follows_the_parameters(self, bkd: Backend[Array]) -> None:
        problem = _Problem(bkd)
        sample = bkd.asarray([[0.0], [1.5], [0.2]])
        problem.set(sample)
        bkd.assert_allclose(problem.initial.value(), problem.u0(sample[1:]), rtol=1e-14)

    def test_value_is_a_new_vector(self, bkd: Backend[Array]) -> None:
        problem = _Problem(bkd)
        assert problem.initial.value() is not problem.data.value()


class TestDerivatives:
    def test_jacobian_is_zero_on_the_coefficient(self, bkd: Backend[Array]) -> None:
        problem = _Problem(bkd)
        expected = bkd.hstack(
            [bkd.zeros((_NSTATES, 1)), problem.u0.jacobian(_sample(bkd)[1:])]
        )
        bkd.assert_allclose(problem.jacobian_matrix(), expected, rtol=1e-14)

    def test_jacobian_matches_finite_differences(self, bkd: Backend[Array]) -> None:
        problem = _Problem(bkd)

        def values(samples: Array) -> Array:
            columns: List[Array] = []
            for k in range(samples.shape[1]):
                problem.set(samples[:, k : k + 1])
                columns.append(problem.initial.value()[:, 0])
            return bkd.stack(columns, axis=1)

        def jacobian(sample: Array) -> Array:
            problem.set(sample)
            return problem.jacobian_matrix()

        checker = DerivativeChecker(
            FunctionWithJacobianFromCallable(
                nqoi=_NSTATES, nvars=3, fun=values, jacobian=jacobian, bkd=bkd
            )
        )
        errors = checker.check_derivatives(_sample(bkd), relative=True, central=True)
        assert bkd.to_float(checker.error_ratio(errors[0])) <= 1e-6

    def test_curvature_matches_finite_differences(self, bkd: Backend[Array]) -> None:
        problem = _Problem(bkd)
        weight = bkd.asarray([[0.4], [-1.3], [0.8]])

        def values(samples: Array) -> Array:
            columns: List[Array] = []
            for k in range(samples.shape[1]):
                problem.set(samples[:, k : k + 1])
                columns.append(bkd.sum(weight * problem.initial.value())[None])
            return bkd.stack(columns, axis=1)

        def gradient(sample: Array) -> Array:
            problem.set(sample)
            field = problem.initial.derivatives().param_derivative
            assert field is not None
            return field().apply_transpose(weight).T

        def hvp(sample: Array, vec: Array) -> Array:
            problem.set(sample)
            field = problem.initial.derivatives().curvature
            assert field is not None
            return field(weight).apply(vec)

        checker = DerivativeChecker(
            FunctionWithJacobianAndHVPFromCallable(
                nvars=3, fun=values, jacobian=gradient, hvp=hvp, bkd=bkd
            )
        )
        errors = checker.check_derivatives(_sample(bkd), relative=True, central=True)
        for error in errors:
            assert bkd.to_float(checker.error_ratio(error)) <= 1e-6

    @pytest.mark.parametrize("order", [0, 1, 2])
    def test_order_follows_the_function(self, bkd: Backend[Array], order: int) -> None:
        target = InitialValueData(bkd.zeros((_NSTATES, 1)))
        part = InitialValueParameterization(target, _U0(bkd, order))
        states, pspace = backend_spaces(bkd, _NSTATES, 2)
        derivs = ParameterizedInitialState(
            target,
            ParameterState(part),
            states,
            pspace,
            BackendColumnForm(bkd),
            bkd,
        ).derivatives()
        found = (
            0
            if derivs.param_derivative is None
            else 1
            if derivs.curvature is None
            else 2
        )
        assert found == order

    def test_without_parameters_has_none(self, bkd: Backend[Array]) -> None:
        states, _ = backend_spaces(bkd, _NSTATES, 2)
        initial = ParameterizedInitialState(
            InitialValueData(bkd.ones((_NSTATES, 1))),
            None,
            states,
            None,
            BackendColumnForm(bkd),
            bkd,
        )
        assert initial.derivatives().param_derivative is None
        bkd.assert_allclose(initial.value(), bkd.ones((_NSTATES, 1)))


class TestRefusals:
    def test_data_must_be_a_column(self, bkd: Backend[Array]) -> None:
        with pytest.raises(ValueError, match="shape"):
            InitialValueData(bkd.zeros((_NSTATES,)))
        data = InitialValueData(bkd.zeros((_NSTATES, 1)))
        with pytest.raises(ValueError, match="entries"):
            data.set_value(bkd.zeros((_NSTATES + 1, 1)))

    def test_function_must_match_the_target(self, bkd: Backend[Array]) -> None:
        with pytest.raises(ValueError, match="outputs"):
            InitialValueParameterization(
                InitialValueData(bkd.zeros((_NSTATES + 1, 1))), _U0(bkd)
            )

    def test_parameters_need_a_parameter_space(self, bkd: Backend[Array]) -> None:
        target = InitialValueData(bkd.zeros((_NSTATES, 1)))
        states, _ = backend_spaces(bkd, _NSTATES, 2)
        with pytest.raises(ValueError, match="param_space"):
            ParameterizedInitialState(
                target,
                ParameterState(InitialValueParameterization(target, _U0(bkd))),
                states,
                None,
                BackendColumnForm(bkd),
                bkd,
            )


class TestOwners:
    def test_hold_what_any_member_holds(self, bkd: Backend[Array]) -> None:
        system, target = object(), InitialValueData(bkd.zeros((_NSTATES, 1)))
        owners = Owners(system, target)
        assert owns(owners, system) and owns(owners, target)
        assert not owns(owners, object())

    def test_accept_a_problem_whose_initial_value_is_parameterized(
        self, bkd: Backend[Array]
    ) -> None:
        problem = _Problem(bkd)
        parameterization = problem.parameters.parameterization()
        require_owned_targets(
            parameterization, Owners(problem.coefficient, problem.data)
        )
        with pytest.raises(ValueError, match="does not hold"):
            require_owned_targets(parameterization, problem.coefficient)
