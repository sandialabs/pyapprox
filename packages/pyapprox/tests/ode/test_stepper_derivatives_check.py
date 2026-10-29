r"""The stepper checker: it fails exactly when it should.

The subject is backward Euler on Lotka-Volterra with a mass, at a
converged step from (10, 5), t = 0.2, dt = 0.05, with a random step
adjoint. Its own derivatives pass (the tableau tests show that for every
method); here one operator is replaced by a faulty one, and the report
must name exactly the checks that differentiate it. Backward Euler's
off-diagonal block is the constant :math:`-M` and Lotka-Volterra's
parameter Jacobian does not depend on the parameters, so:

- ``off_diag`` scaled by 1.01: only ``off_diag[1]`` (the checks reading
  its transpose differentiate a constant, still zero);
- ``param`` scaled by 1.01: ``param`` and ``param_state[0]``, which
  differentiates :math:`\mathrm{param}^\top \mu` in the stage
  (``param_param`` differentiates it in the parameters: still zero);
- the ``(0, 0)`` declaration dropped: ``state_state[0,0]``, compared with
  zero although :math:`\mathrm{diag}^\top \mu` varies with the stage;
- ``diag`` scaled by 1.01 with its handle unchanged: ``diag``,
  ``state_state[0,0]`` and ``state_param[0]`` (both differentiate
  :math:`\mathrm{diag}^\top \mu`), and both handle checks;
- ``off_diag``'s transpose off by 1.01, its action right: only the dot
  test of ``off_diag[1]``; its finite differences pass.

A parameter derivative off by a relative 1e-3 or 1e-5 is caught and one
off by 1e-8 is not, so the detection threshold is the 1e-6 tolerance.
The checker refuses what it cannot check, leaves the parameters at the
point checked, and runs unchanged on the mock stack, whose vectors
cannot be looked inside.
"""

from dataclasses import replace
from typing import Callable, Generic, Optional

import pytest

from pyapprox.ode.runge_kutta import RungeKuttaStepper
from pyapprox.ode.stepper_derivatives import (
    StepOperator,
    StepperDerivatives,
    StepWindow,
)
from pyapprox.ode.stepper_derivatives_check import StepperBundleChecker
from pyapprox.ode.tableau import ButcherTableau, backward_euler, forward_euler
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import DenseFormat
from pyapprox.util.linalg.operators import (
    BlockInnerProductSpace,
    BlockVector,
    FormatProtocol,
    MatrixFree,
    OperatorProperties,
    ScaledSum,
    SolverHandleProtocol,
    SpaceDescriptor,
)
from tests._helpers.ode.runge_kutta_cases import (
    StepCase,
    backend_checker,
    backend_sample,
    lotka_volterra_step,
    mock_sample,
    mock_stepper,
)

_Stepper = RungeKuttaStepper[Array, Array, DenseFormat[Array]]
_Edit = Callable[
    [StepperDerivatives[Array], _Stepper[Array]], StepperDerivatives[Array]
]


class _Planted(Generic[Array]):
    """The stepper with one planted fault: its bundle edited, or its
    ``diag`` (not the operator its handle inverts) scaled."""

    def __init__(
        self,
        stepper: _Stepper[Array],
        edit: Optional[_Edit[Array]] = None,
        diag_scale: float = 1.0,
    ) -> None:
        self._stepper, self._edit, self._diag_scale = stepper, edit, diag_scale

    def depth(self) -> int:
        return self._stepper.depth()

    def unknown_space(self) -> BlockInnerProductSpace[Array, Array]:
        return self._stepper.unknown_space()

    def residual_space(self) -> BlockInnerProductSpace[Array, Array]:
        return self._stepper.residual_space()

    def state_space(self) -> BlockInnerProductSpace[Array, Array]:
        return self._stepper.state_space()

    def param_space(self) -> Optional[BlockInnerProductSpace[Array, Array]]:
        return self._stepper.param_space()

    def readout(self) -> StepOperator[Array]:
        return self._stepper.readout()

    def step_residual(self, window: StepWindow[Array]) -> BlockVector[Array]:
        return self._stepper.step_residual(window)

    def diag(self, window: StepWindow[Array]) -> StepOperator[Array]:
        return _scaled(
            self._diag_scale,
            self._stepper.diag(window),
            self.unknown_space(),
            self.residual_space(),
        )

    def diag_handle(
        self, window: StepWindow[Array]
    ) -> SolverHandleProtocol[BlockVector[Array]]:
        return self._stepper.diag_handle(window)

    def derivatives(self) -> StepperDerivatives[Array]:
        derivs = self._stepper.derivatives()
        return derivs if self._edit is None else self._edit(derivs, self._stepper)

    def parameters_version(self) -> int:
        return self._stepper.parameters_version()


def _scaled(
    scale: float,
    op: StepOperator[Array],
    domain: BlockInnerProductSpace[Array, Array],
    range_: BlockInnerProductSpace[Array, Array],
) -> ScaledSum[BlockVector[Array], FormatProtocol]:
    return ScaledSum([(scale, op)], range_, domain)


class _WrongTranspose(Generic[Array]):
    """An operator whose action is right and whose transpose is off by a
    factor of 1.01."""

    def __init__(
        self, op: StepOperator[Array], domain: BlockInnerProductSpace[Array, Array]
    ) -> None:
        self._op, self._domain = op, domain

    def domain(self) -> SpaceDescriptor:
        return self._op.domain()

    def range(self) -> SpaceDescriptor:
        return self._op.range()

    def properties(self) -> OperatorProperties:
        return self._op.properties()

    def format(self) -> MatrixFree:
        return MatrixFree()

    def apply(
        self, x: BlockVector[Array], out: Optional[BlockVector[Array]] = None
    ) -> BlockVector[Array]:
        return self._op.apply(x)

    def apply_transpose(
        self, y: BlockVector[Array], out: Optional[BlockVector[Array]] = None
    ) -> BlockVector[Array]:
        return self._domain.scale(1.01, self._op.apply_transpose(y))


def _scale_off_diag(scale: float) -> _Edit[Array]:
    def edit(
        derivs: StepperDerivatives[Array], stepper: _Stepper[Array]
    ) -> StepperDerivatives[Array]:
        field = derivs.off_diag
        assert field is not None

        def off_diag(window: StepWindow[Array], lag: int) -> StepOperator[Array]:
            return _scaled(
                scale,
                field(window, lag),
                stepper.state_space(),
                stepper.residual_space(),
            )

        return replace(derivs, off_diag=off_diag)

    return edit


def _scale_param(scale: float) -> _Edit[Array]:
    def edit(
        derivs: StepperDerivatives[Array], stepper: _Stepper[Array]
    ) -> StepperDerivatives[Array]:
        field, params = derivs.param, stepper.param_space()
        assert field is not None and params is not None

        def param(window: StepWindow[Array]) -> StepOperator[Array]:
            return _scaled(scale, field(window), params, stepper.residual_space())

        return replace(derivs, param=param)

    return edit


def _drop_declarations(
    derivs: StepperDerivatives[Array], stepper: _Stepper[Array]
) -> StepperDerivatives[Array]:
    return replace(derivs, state_pairs=frozenset())


def _wrong_off_diag_transpose(
    derivs: StepperDerivatives[Array], stepper: _Stepper[Array]
) -> StepperDerivatives[Array]:
    field = derivs.off_diag
    assert field is not None

    def off_diag(window: StepWindow[Array], lag: int) -> StepOperator[Array]:
        return _WrongTranspose(field(window, lag), stepper.state_space())

    return replace(derivs, off_diag=off_diag)


def _planted_report(
    bkd: Backend[Array], edit: Optional[_Edit[Array]] = None, diag_scale: float = 1.0
) -> list[str]:
    case = lotka_volterra_step(bkd, backward_euler())
    planted = _Planted(case.stepper, edit, diag_scale)
    checker = StepperBundleChecker(
        planted, backend_sample(bkd), set_param=case.model.set_param
    )
    return checker.check(case.window, case.adj, case.param).failures()


class TestPlantedFaults:
    def test_a_sound_bundle_passes(self, bkd: Backend[Array]) -> None:
        assert _planted_report(bkd) == []

    def test_scaled_off_diagonal(self, bkd: Backend[Array]) -> None:
        assert _planted_report(bkd, _scale_off_diag(1.01)) == ["off_diag[1]"]

    def test_scaled_parameter_derivative(self, bkd: Backend[Array]) -> None:
        assert _planted_report(bkd, _scale_param(1.01)) == ["param", "param_state[0]"]

    def test_a_dropped_declaration(self, bkd: Backend[Array]) -> None:
        assert _planted_report(bkd, _drop_declarations) == ["state_state[0,0]"]

    def test_scaled_diagonal_block(self, bkd: Backend[Array]) -> None:
        assert _planted_report(bkd, diag_scale=1.01) == [
            "diag",
            "state_state[0,0]",
            "state_param[0]",
            "diag_handle",
            "diag_handle (transpose)",
        ]

    def test_a_wrong_transpose_reaches_only_the_dot_test(
        self, bkd: Backend[Array]
    ) -> None:
        case = lotka_volterra_step(bkd, backward_euler())
        planted = _Planted(case.stepper, _wrong_off_diag_transpose)
        checker = StepperBundleChecker(
            planted, backend_sample(bkd), set_param=case.model.set_param
        )
        report = checker.check(case.window, case.adj, case.param)
        assert report.failures() == ["off_diag[1]"]
        assert report.fd_errors["off_diag[1]"] < 1e-6
        assert report.transpose_errors["off_diag[1]"] > 1e-3

    @pytest.mark.parametrize(
        "size, caught", [(1e-3, True), (1e-5, True), (1e-8, False)]
    )
    def test_detection_threshold_is_the_tolerance(
        self, bkd: Backend[Array], size: float, caught: bool
    ) -> None:
        failures = _planted_report(bkd, _scale_param(1.0 + size))
        assert ("param" in failures) == caught


class TestCheckerContract:
    def test_parameter_derivatives_need_set_param(self, bkd: Backend[Array]) -> None:
        case = lotka_volterra_step(bkd, backward_euler())
        checker = StepperBundleChecker(case.stepper, backend_sample(bkd))
        with pytest.raises(ValueError, match="set_param"):
            checker.check(case.window, case.adj, case.param)

    def test_second_derivatives_need_the_adjoint(self, bkd: Backend[Array]) -> None:
        case = lotka_volterra_step(bkd, backward_euler())
        with pytest.raises(ValueError, match="pass adj"):
            backend_checker(case, bkd).check(case.window, param=case.param)

    def test_a_non_stepper_is_refused(self, bkd: Backend[Array]) -> None:
        with pytest.raises(TypeError, match="DifferentiableStepperProtocol"):
            StepperBundleChecker(object(), backend_sample(bkd))  # type: ignore[arg-type]

    def test_the_parameters_are_left_at_the_point(self, bkd: Backend[Array]) -> None:
        case: StepCase[Array] = lotka_volterra_step(bkd, backward_euler())
        before = case.model(case.window.past_state(1)[:, 0], 0.2)
        backend_checker(case, bkd).check(case.window, case.adj, case.param)
        after = case.model(case.window.past_state(1)[:, 0], 0.2)
        bkd.assert_allclose(after, before, rtol=0.0)


class TestOnAnOpaqueStack:
    @pytest.mark.parametrize("make", [forward_euler, backward_euler])
    def test_the_checker_uses_only_the_spaces(
        self, make: Callable[[], ButcherTableau]
    ) -> None:
        stepper = mock_stepper(make())
        state = mock_sample(stepper.system().space.descriptor(), 3)
        window = StepWindow(0.0, 0.1, (state,), stepper.step(state, 0.0, 0.1))
        report = StepperBundleChecker(stepper, mock_sample).check(window)
        assert set(report.fd_errors) == {"diag", "off_diag[1]"}
        assert report.failures() == []
