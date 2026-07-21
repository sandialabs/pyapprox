"""Tests for the closed stepper table and typed factory handle."""

import pytest
from pyapprox.ode.explicit_steppers.forward_euler import ForwardEulerHVP
from pyapprox.ode.implicit_steppers.backward_euler import BackwardEulerHVP
from pyapprox.ode.protocols.time_stepping import TimeSteppingResidualProtocol
from pyapprox.ode.stepper_table import (
    EXPLICIT_METHOD_NAMES,
    STEPPER_TABLE,
    create_stepper,
    resolve_stepper_factory,
)


class _ToyResidual:
    """Minimal ODEResidualProtocol implementation: dy/dt = -y."""

    def __init__(self, bkd) -> None:
        self._bkd = bkd

    def bkd(self):
        return self._bkd

    def __call__(self, state):
        return -state


class TestStepperTable:
    def test_all_builtins_present(self):
        assert sorted(STEPPER_TABLE) == [
            "backward_euler",
            "crank_nicolson",
            "forward_euler",
            "heun",
            "implicit_midpoint",
        ]

    def test_table_is_frozen(self):
        with pytest.raises(TypeError):
            STEPPER_TABLE["rk4"] = ForwardEulerHVP  # type: ignore[index]

    def test_explicit_names_subset_of_table(self):
        assert EXPLICIT_METHOD_NAMES <= set(STEPPER_TABLE)

    def test_resolve_builtin_name(self):
        assert resolve_stepper_factory("backward_euler") is BackwardEulerHVP

    def test_resolve_unknown_name_lists_valid(self):
        with pytest.raises(ValueError, match="implicit_midpoint"):
            resolve_stepper_factory("rk4")

    def test_resolve_rejects_non_callable(self):
        with pytest.raises(TypeError, match="StepperFactory"):
            resolve_stepper_factory(42)  # type: ignore[arg-type]

    def test_resolve_passes_factory_through(self):
        def factory(residual):
            return BackwardEulerHVP(residual)

        assert resolve_stepper_factory(factory) is factory

    def test_create_stepper_from_name(self, numpy_bkd):
        residual = _ToyResidual(numpy_bkd)
        stepper = create_stepper("backward_euler", residual)
        assert isinstance(stepper, BackwardEulerHVP)
        assert isinstance(stepper, TimeSteppingResidualProtocol)
        assert stepper.native_residual is residual

    def test_create_stepper_from_factory(self, numpy_bkd):
        residual = _ToyResidual(numpy_bkd)
        stepper = create_stepper(
            lambda res: ForwardEulerHVP(res), residual
        )
        assert isinstance(stepper, ForwardEulerHVP)

    def test_create_stepper_rejects_nonconforming_product(self, numpy_bkd):
        with pytest.raises(TypeError, match="TimeSteppingResidualProtocol"):
            create_stepper(lambda res: object(), _ToyResidual(numpy_bkd))
