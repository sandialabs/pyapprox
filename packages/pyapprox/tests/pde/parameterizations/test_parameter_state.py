"""ParameterState: one source of the parameters and their version.

- Each ``set_param`` moves the version by one and records the parameters.
- A parameterization whose ``apply`` fails partway has still moved the
  version, so a consumer comparing versions sees the coefficients may have
  changed; the parameters recorded are the ones it was applying.
- A 1-D parameter vector is required; nothing moves when it is refused.
"""

from typing import Generic, List, Tuple

import pytest

from pyapprox.pde.parameterizations.derivatives import ParamDerivatives
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.util.backends.protocols import Array, Backend


class _Recording(Generic[Array]):
    """Records what it is applied with; fails when told to."""

    def __init__(self, fail: bool = False) -> None:
        self.applied: List[Array] = []
        self._fail = fail

    def nparams(self) -> int:
        return 2

    def targets(self) -> Tuple[object, ...]:
        return ()

    def owned_coefficients(self) -> Tuple[str, ...]:
        return ("kappa",)

    def apply(self, params_1d: Array) -> None:
        self.applied.append(params_1d)
        if self._fail:
            raise RuntimeError("apply failed partway")

    def param_derivatives(self) -> ParamDerivatives[Array]:
        return ParamDerivatives()


class TestParameterState:
    def test_each_set_param_moves_the_version(self, bkd: Backend[Array]) -> None:
        parameterization: _Recording[Array] = _Recording()
        state = ParameterState(parameterization)
        assert state.parameters_version() == 0
        first, second = bkd.asarray([1.0, 2.0]), bkd.asarray([3.0, 4.0])
        state.set_param(first)
        state.set_param(second)
        assert state.parameters_version() == 2
        bkd.assert_allclose(state.current(), second)
        assert len(parameterization.applied) == 2

    def test_a_failed_apply_has_moved_the_version(self, bkd: Backend[Array]) -> None:
        state = ParameterState(_Recording[Array](fail=True))
        param = bkd.asarray([1.0, 2.0])
        with pytest.raises(RuntimeError, match="partway"):
            state.set_param(param)
        assert state.parameters_version() == 1
        bkd.assert_allclose(state.current(), param)

    def test_a_2d_parameter_vector_is_refused(self, bkd: Backend[Array]) -> None:
        state = ParameterState(_Recording[Array]())
        with pytest.raises(ValueError, match="1D"):
            state.set_param(bkd.asarray([[1.0], [2.0]]))
        assert state.parameters_version() == 0
        with pytest.raises(RuntimeError, match="set_param"):
            state.current()
