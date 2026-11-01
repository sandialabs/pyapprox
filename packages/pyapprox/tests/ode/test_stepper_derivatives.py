"""The step window, the stepper derivatives bundle and the residual form.

The window returns its past states by lag, and its times and size from
its span exactly, and refuses lags it does not hold and a span without
one size per past state. The named constructors fill exactly their tier, and the bundle
refuses a partial or inconsistent declaration: parameters without the
state coupling, part of the second order, second order without
parameters, lag declarations without second order, a lag pair without
its swap, a negative lag, a value where a function belongs. The backend
residual form turns a single column into the 1-D state a residual takes
and back, refusing other shapes; the identity form passes vectors
through.
"""


import pytest

from pyapprox.ode.residual_form import (
    BackendColumnForm,
    IdentityForm,
    ResidualFormProtocol,
)
from pyapprox.ode.step_span import StepSpan
from pyapprox.ode.stepper_derivatives import (
    StepOperator,
    StepperDerivatives,
    StepWindow,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.operators import BlockVector


def _window(bkd: Backend[Array]) -> StepWindow[Array]:
    return StepWindow(
        span=StepSpan(0.25, (0.25, 0.5)),
        past_states=(bkd.asarray([[1.0], [2.0]]), bkd.asarray([[3.0], [4.0]])),
        unknown=BlockVector((bkd.asarray([[5.0], [6.0]]),)),
    )


# Fields of the bundle: never called, so they only need the signatures.


def _off_diag(window: StepWindow[Array], lag: int) -> StepOperator[Array]:
    raise AssertionError("not called")


def _param(window: StepWindow[Array]) -> StepOperator[Array]:
    raise AssertionError("not called")


def _state_state(
    window: StepWindow[Array], adj: BlockVector[Array], a: int, b: int
) -> StepOperator[Array]:
    raise AssertionError("not called")


def _state_param(
    window: StepWindow[Array], adj: BlockVector[Array], a: int
) -> StepOperator[Array]:
    raise AssertionError("not called")


def _param_param(
    window: StepWindow[Array], adj: BlockVector[Array]
) -> StepOperator[Array]:
    raise AssertionError("not called")


_PAIRS = frozenset({(0, 0), (0, 1), (1, 0)})


class TestStepWindow:
    def test_past_states_by_lag(self, bkd: Backend[Array]) -> None:
        window = _window(bkd)
        bkd.assert_allclose(window.past_state(1), bkd.asarray([[1.0], [2.0]]))
        bkd.assert_allclose(window.past_state(2), bkd.asarray([[3.0], [4.0]]))

    @pytest.mark.parametrize("lag", [0, 3])
    def test_refuses_a_lag_it_does_not_hold(
        self, bkd: Backend[Array], lag: int
    ) -> None:
        with pytest.raises(ValueError, match="between 1 and 2"):
            _window(bkd).past_state(lag)

    def test_times_by_lag(self, bkd: Backend[Array]) -> None:
        window = _window(bkd)
        assert window.time() == 0.5
        assert window.deltat() == 0.5
        assert [window.lag_time(lag) for lag in range(3)] == [1.0, 0.5, 0.25]

    @pytest.mark.parametrize("sizes", [(0.5,), (0.25, 0.25, 0.5)])
    def test_refuses_a_span_that_does_not_fit(
        self, bkd: Backend[Array], sizes: tuple[float, ...]
    ) -> None:
        window = _window(bkd)
        with pytest.raises(ValueError, match="needs a span with 2 sizes"):
            StepWindow(StepSpan(0.0, sizes), window.past_states, window.unknown)


class TestNamedConstructors:
    def test_each_tier_fills_exactly_its_fields(self, bkd: Backend[Array]) -> None:
        none: StepperDerivatives[Array] = StepperDerivatives.none()
        assert none.off_diag is None and none.param is None
        state: StepperDerivatives[Array] = StepperDerivatives.state_only(_off_diag)
        assert state.off_diag is not None and state.param is None
        first: StepperDerivatives[Array] = StepperDerivatives.first_order(
            _off_diag, _param
        )
        assert first.param is not None and first.state_state is None
        second: StepperDerivatives[Array] = StepperDerivatives.second_order(
            _off_diag,
            _param,
            _state_state,
            _state_param,
            _param_param,
            _PAIRS,
            frozenset({0}),
        )
        assert second.state_state is not None and second.param_param is not None
        assert second.state_pairs == _PAIRS and second.param_lags == {0}


class TestInvariants:
    def test_param_requires_the_state_coupling(self) -> None:
        with pytest.raises(ValueError, match="param requires off_diag"):
            StepperDerivatives(param=_param)

    def test_second_order_comes_whole(self) -> None:
        with pytest.raises(ValueError, match="come together"):
            StepperDerivatives(
                off_diag=_off_diag, param=_param, state_state=_state_state
            )

    def test_second_order_requires_parameters(self) -> None:
        with pytest.raises(ValueError, match="second order requires param"):
            StepperDerivatives(
                off_diag=_off_diag,
                state_state=_state_state,
                state_param=_state_param,
                param_param=_param_param,
            )

    def test_lag_declarations_come_with_second_order(self) -> None:
        with pytest.raises(ValueError, match="come with second order"):
            StepperDerivatives(off_diag=_off_diag, state_pairs=frozenset({(0, 0)}))

    def test_a_pair_needs_its_swap(self) -> None:
        with pytest.raises(ValueError, match=r"has \(0, 1\) but not \(1, 0\)"):
            StepperDerivatives.second_order(
                _off_diag,
                _param,
                _state_state,
                _state_param,
                _param_param,
                frozenset({(0, 0), (0, 1)}),
                frozenset(),
            )

    def test_lags_are_nonnegative(self) -> None:
        with pytest.raises(ValueError, match="nonnegative"):
            StepperDerivatives.second_order(
                _off_diag,
                _param,
                _state_state,
                _state_param,
                _param_param,
                frozenset(),
                frozenset({-1}),
            )

    def test_a_value_is_refused_where_a_function_belongs(self) -> None:
        with pytest.raises(TypeError, match="must be callable"):
            StepperDerivatives(off_diag=3.0)  # type: ignore[arg-type]


class TestResidualForm:
    def test_backend_form_reshapes_both_ways(self, bkd: Backend[Array]) -> None:
        form: BackendColumnForm[Array] = BackendColumnForm(bkd)
        assert isinstance(form, ResidualFormProtocol)
        column = bkd.asarray([[1.0], [2.0], [3.0]])
        state = form.to_residual(column)
        assert state.shape == (3,)
        bkd.assert_allclose(form.from_residual(state), column, rtol=0.0)

    def test_backend_form_refuses_other_shapes(self, bkd: Backend[Array]) -> None:
        form: BackendColumnForm[Array] = BackendColumnForm(bkd)
        with pytest.raises(ValueError, match="single column"):
            form.to_residual(bkd.ones((3, 2)))
        with pytest.raises(ValueError, match="must be 1-D"):
            form.from_residual(bkd.ones((3, 1)))

    def test_identity_form_passes_vectors_through(self, bkd: Backend[Array]) -> None:
        form: IdentityForm[Array] = IdentityForm()
        column = bkd.asarray([[1.0], [2.0]])
        assert form.to_residual(column) is column
        assert form.from_residual(column) is column
