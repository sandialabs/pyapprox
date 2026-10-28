"""The finite-difference check of an ODE residual's derivatives bundle.

On the Lotka-Volterra test model (state (10, 5), adjoint (0.4, -1.1),
parameters (1, 0.1, 0.075, 1.5), t = 0.3) every term of the second-order
bundle passes, forward-difference errors below 1e-6 and dot tests below
1e-12. Each fault is then planted in one field of an otherwise correct
bundle and must be reported under that field's name only: a Jacobian or
parameter Jacobian scaled by 1.01, a mixed term scaled by 1.01 (seen
both ways round), a nonzero state-state term replaced by a stated zero,
a matrix-free Jacobian with a wrong transpose (invisible to finite
differences, caught by the dot test), and a non-symmetric operator
declared symmetric. A caller's y0(p) that depends on p fails the stated
zero initial state; a constant one passes.
"""

from dataclasses import replace
from typing import Callable, Generic, Optional

import pytest

from pyapprox.ode.residual_derivatives import (
    STATED_ZERO,
    JacobianFn,
    ODEResidualDerivatives,
    backend_spaces,
    dense_jacobian,
    dense_operator,
    matrix_free_jacobian,
    matrix_free_second_derivative,
    second_derivative_or_zero,
)
from pyapprox.ode.residual_derivatives_check import ODEResidualBundleChecker
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import DenseFormat
from pyapprox.util.linalg.operators import (
    FormatProtocol,
    JacobianSnapshotProtocol,
    LinearizationPoint,
)
from tests._helpers.ode_models import LotkaVolterra

_Bundle = ODEResidualDerivatives[Array, FormatProtocol]
_Edit = Callable[[_Bundle[Array], LotkaVolterra[Array]], _Bundle[Array]]
_TIME = 0.3


class _Edited(Generic[Array]):
    """The Lotka-Volterra model with one field of its bundle replaced."""

    def __init__(self, model: LotkaVolterra[Array], edit: _Edit[Array]) -> None:
        self._model, self._edit = model, edit

    def bkd(self) -> Backend[Array]:
        return self._model.bkd()

    def __call__(self, state: Array, time: float) -> Array:
        return self._model(state, time)

    def set_param(self, param: Array) -> None:
        self._model.set_param(param)

    def parameters_version(self) -> int:
        return self._model.parameters_version()

    def derivatives(self) -> _Bundle[Array]:
        return self._edit(self._model.derivatives(), self._model)


class _Unversioned(Generic[Array]):
    """The model with ``set_param`` but no parameter version."""

    def __init__(self, model: LotkaVolterra[Array]) -> None:
        self._model = model

    def bkd(self) -> Backend[Array]:
        return self._model.bkd()

    def __call__(self, state: Array, time: float) -> Array:
        return self._model(state, time)

    def set_param(self, param: Array) -> None:
        self._model.set_param(param)

    def derivatives(self) -> _Bundle[Array]:
        return self._model.derivatives()


class _RecordingJacobian(Generic[Array]):
    """A Jacobian field recording the parameter version of each label."""

    def __init__(self, field: JacobianFn[Array, FormatProtocol]) -> None:
        self._field = field
        self.versions: list[int] = []

    def __call__(
        self, state: Array, time: float, point: LinearizationPoint
    ) -> JacobianSnapshotProtocol[Array, FormatProtocol]:
        self.versions.append(point.parameters_version)
        return self._field(state, time, point)


def _point(bkd: Backend[Array]) -> tuple[Array, Array, Array]:
    return (
        bkd.asarray([10.0, 5.0]),
        bkd.asarray([0.4, -1.1]),
        bkd.asarray([1.0, 0.1, 0.075, 1.5]),
    )


def _failures(
    residual: _Edited[Array] | LotkaVolterra[Array], bkd: Backend[Array]
) -> list[str]:
    state, adj, param = _point(bkd)
    report = ODEResidualBundleChecker(residual).check(state, _TIME, adj, param)
    return report.failures()


def _scaled_jacobian(
    bundle: _Bundle[Array], model: LotkaVolterra[Array]
) -> _Bundle[Array]:
    bkd = model.bkd()
    states, _ = backend_spaces(bkd, 2, 4)

    def wrong(state: Array, time: float) -> Array:
        return 1.01 * model.jacobian(state, time)

    return replace(bundle, jacobian=dense_jacobian(wrong, states, bkd))


def _scaled_param(
    bundle: _Bundle[Array], model: LotkaVolterra[Array]
) -> _Bundle[Array]:
    bkd = model.bkd()
    states, params = backend_spaces(bkd, 2, 4)

    def wrong(state: Array, time: float) -> Array:
        return 1.01 * model.param_jacobian(state, time)

    return replace(bundle, param=dense_operator(wrong, params, states, bkd))


def _scaled_state_param(
    bundle: _Bundle[Array], model: LotkaVolterra[Array]
) -> _Bundle[Array]:
    bkd = model.bkd()
    states, params = backend_spaces(bkd, 2, 4)
    right = bundle.state_param
    assert right is not None

    def wrong(state: Array, time: float, adj: Array) -> Array:
        return 1.01 * right(state, time, adj).apply(bkd.eye(4))

    return replace(bundle, state_param=dense_operator(wrong, params, states, bkd))


def _zero_state_state(
    bundle: _Bundle[Array], model: LotkaVolterra[Array]
) -> _Bundle[Array]:
    states, _ = backend_spaces(model.bkd(), 2, 4)
    zero = second_derivative_or_zero(STATED_ZERO, states, states)
    return replace(bundle, state_state=zero)


def _wrong_transpose(
    bundle: _Bundle[Array], model: LotkaVolterra[Array]
) -> _Bundle[Array]:
    states, _ = backend_spaces(model.bkd(), 2, 4)

    def wrong_transpose(state: Array, time: float, w: Array) -> Array:
        return 1.01 * model.apply_jacobian_transpose(state, time, w)

    jacobian = matrix_free_jacobian(model.apply_jacobian, wrong_transpose, states)
    return replace(bundle, jacobian=jacobian)


def _false_symmetry(
    bundle: _Bundle[Array], model: LotkaVolterra[Array]
) -> _Bundle[Array]:
    """The true state-state term plus a skew part, whose transpose is
    given correctly but which is declared symmetric. The skew part
    changes the action, so it also fails its finite differences."""
    bkd = model.bkd()
    states, _ = backend_spaces(bkd, 2, 4)
    right = bundle.state_state
    assert right is not None
    skew = bkd.asarray([[0.0, 1.0], [-1.0, 0.0]])

    def apply(state: Array, time: float, adj: Array, vec: Array) -> Array:
        symmetric = right(state, time, adj).apply(vec[:, None])[:, 0]
        return symmetric + bkd.dot(skew, vec)

    def apply_transpose(state: Array, time: float, adj: Array, vec: Array) -> Array:
        symmetric = right(state, time, adj).apply(vec[:, None])[:, 0]
        return symmetric - bkd.dot(skew, vec)

    field = matrix_free_second_derivative(
        apply, apply_transpose, states, states, bkd, symmetric=True
    )
    return replace(bundle, state_state=field)


def _checker(
    bkd: Backend[Array],
    initial_state: Optional[Callable[[Array], Array]] = None,
    central: bool = False,
) -> ODEResidualBundleChecker[Array]:
    """The checker on the unmodified Lotka-Volterra model."""
    model: LotkaVolterra[Array] = LotkaVolterra(bkd)
    return ODEResidualBundleChecker(model, initial_state, central=central)


class TestCorrectBundles:
    def test_every_term_passes(self, bkd: Backend[Array]) -> None:
        state, adj, param = _point(bkd)
        report = _checker(bkd).check(state, _TIME, adj, param)
        assert set(report.fd_errors) == {
            "jacobian",
            "param",
            "state_state",
            "state_param",
            "param_state",
            "param_param",
        }
        assert set(report.transpose_errors) == {
            "jacobian",
            "param",
            "initial_state",
            "state_state",
            "state_state (symmetric)",
            "state_param",
            "param_param",
            "initial_state_curvature",
        }
        assert report.failures() == []

    def test_a_state_only_bundle_needs_no_parameters(self, bkd: Backend[Array]) -> None:
        model = LotkaVolterra(bkd)
        states, _ = backend_spaces(bkd, 2, 4)
        only: ODEResidualDerivatives[Array, DenseFormat[Array]] = (
            ODEResidualDerivatives.state_only(
                dense_jacobian(model.jacobian, states, bkd)
            )
        )
        residual = _Edited(model, lambda bundle, _: only)
        state, _, _ = _point(bkd)
        report = ODEResidualBundleChecker(residual).check(state, _TIME)
        assert set(report.fd_errors) == {"jacobian"}
        assert report.failures() == []

    def test_central_differences(self, bkd: Backend[Array]) -> None:
        state, adj, param = _point(bkd)
        report = _checker(bkd, central=True).check(state, _TIME, adj, param)
        assert report.failures(fd_tol=1e-9) == []

    def test_jacobian_labels_carry_the_residuals_version(
        self, bkd: Backend[Array]
    ) -> None:
        """The checker labels with the residual's parameter version, which
        its parameter moves bump, rather than keeping a count of its own."""
        model = LotkaVolterra(bkd)
        jacobian = model.derivatives().jacobian
        assert jacobian is not None
        recording = _RecordingJacobian(jacobian)
        residual = _Edited(model, lambda bundle, _: replace(bundle, jacobian=recording))
        state, adj, param = _point(bkd)
        ODEResidualBundleChecker(residual).check(state, _TIME, adj, param)
        # The final restore of the parameters bumps once more after the
        # last label was made.
        assert recording.versions[0] >= 1
        assert max(recording.versions) == model.parameters_version() - 1
        assert len(set(recording.versions)) > 1

    def test_the_parameters_are_left_at_the_point(self, bkd: Backend[Array]) -> None:
        model = LotkaVolterra(bkd)
        state, adj, param = _point(bkd)
        ODEResidualBundleChecker(model).check(state, _TIME, adj, param)
        reference = LotkaVolterra(bkd)
        reference.set_param(param)
        bkd.assert_allclose(model(state, _TIME), reference(state, _TIME))


class TestPlantedFaults:
    @pytest.mark.parametrize(
        "edit, expected",
        [
            (_scaled_jacobian, ["jacobian", "state_state", "state_param"]),
            (_scaled_param, ["param", "param_state"]),
            (_scaled_state_param, ["state_param", "param_state"]),
            (_zero_state_state, ["state_state"]),
            (_wrong_transpose, ["state_state", "state_param", "jacobian"]),
            (_false_symmetry, ["state_state", "state_state (symmetric)"]),
        ],
        ids=[
            "jacobian",
            "param",
            "mixed",
            "false-zero",
            "transpose",
            "symmetry",
        ],
    )
    def test_reported_under_its_field_and_the_terms_checked_against_it(
        self, bkd: Backend[Array], edit: _Edit[Array], expected: list[str]
    ) -> None:
        assert _failures(_Edited(LotkaVolterra(bkd), edit), bkd) == expected

    @pytest.mark.parametrize(
        "size, caught", [(1e-3, True), (1e-5, True), (1e-8, False)]
    )
    def test_detection_threshold_is_the_tolerance(
        self, bkd: Backend[Array], size: float, caught: bool
    ) -> None:
        """A Jacobian off by a relative ``size`` is caught down to the
        default tolerance of 1e-6 and not below it, where the finite
        differences' own error (about 1e-8 here) hides it."""
        states, _ = backend_spaces(bkd, 2, 4)

        def edit(bundle: _Bundle[Array], model: LotkaVolterra[Array]) -> _Bundle[Array]:
            def off(state: Array, time: float) -> Array:
                return (1.0 + size) * model.jacobian(state, time)

            return replace(bundle, jacobian=dense_jacobian(off, states, bkd))

        state, adj, param = _point(bkd)
        report = ODEResidualBundleChecker(_Edited(LotkaVolterra(bkd), edit)).check(
            state, _TIME, adj, param
        )
        assert ("jacobian" in report.failures()) == caught

    def test_the_wrong_transpose_passes_its_finite_differences(
        self, bkd: Backend[Array]
    ) -> None:
        """Only the dot test reaches the Jacobian's transpose."""
        state, adj, param = _point(bkd)
        residual = _Edited(LotkaVolterra(bkd), _wrong_transpose)
        report = ODEResidualBundleChecker(residual).check(state, _TIME, adj, param)
        assert report.fd_errors["jacobian"] <= 1e-6
        assert report.transpose_errors["jacobian"] > 1e-3


class TestInitialState:
    def test_a_stated_zero_passes_with_a_constant_initial_state(
        self, bkd: Backend[Array]
    ) -> None:
        state, adj, param = _point(bkd)

        def constant(p: Array) -> Array:
            return bkd.asarray([1.0, 2.0])

        report = _checker(bkd, constant).check(state, _TIME, adj, param)
        assert {"initial_state", "initial_state_curvature"} <= set(report.fd_errors)
        assert report.failures() == []

    def test_a_stated_zero_fails_when_the_initial_state_moves(
        self, bkd: Backend[Array]
    ) -> None:
        """Only ``initial_state`` fails: the curvature differentiates the
        bundle's own ``initial_state``, a zero consistent with it."""
        state, adj, param = _point(bkd)

        def moving(p: Array) -> Array:
            return bkd.stack([p[0], p[1] * p[2]])

        report = _checker(bkd, moving).check(state, _TIME, adj, param)
        assert report.failures() == ["initial_state"]


class TestRefusals:
    def test_a_non_residual_is_refused(self) -> None:
        with pytest.raises(TypeError, match="DifferentiableODEResidualProtocol"):
            ODEResidualBundleChecker(object())  # type: ignore[arg-type]

    def test_parameter_derivatives_need_param(self, bkd: Backend[Array]) -> None:
        state, adj, _ = _point(bkd)
        with pytest.raises(ValueError, match="pass param"):
            _checker(bkd).check(state, _TIME, adj)

    def test_second_derivatives_need_adj(self, bkd: Backend[Array]) -> None:
        state, _, param = _point(bkd)
        with pytest.raises(ValueError, match="pass adj"):
            _checker(bkd).check(state, _TIME, param=param)

    def test_a_residual_without_a_parameter_version_is_refused(
        self, bkd: Backend[Array]
    ) -> None:
        state, adj, param = _point(bkd)
        residual = _Unversioned(LotkaVolterra(bkd))
        with pytest.raises(TypeError, match="parameters_version"):
            ODEResidualBundleChecker(residual).check(state, _TIME, adj, param)

    def test_a_field_lost_after_set_param_is_refused(self, bkd: Backend[Array]) -> None:
        """Which fields are present must not depend on the parameters."""
        fetches: list[None] = []

        def shrinking(
            bundle: _Bundle[Array], model: LotkaVolterra[Array]
        ) -> _Bundle[Array]:
            fetches.append(None)
            return bundle if len(fetches) == 1 else ODEResidualDerivatives.none()

        state, adj, param = _point(bkd)
        residual = _Edited(LotkaVolterra(bkd), shrinking)
        with pytest.raises(ValueError, match="must not depend on the parameter"):
            ODEResidualBundleChecker(residual).check(state, _TIME, adj, param)
