"""Tests for the closed stepper table and typed factory handle."""

from typing import Any

import pytest
from numpy.typing import NDArray

from pyapprox.ode.explicit_steppers.forward_euler import ForwardEulerHVP
from pyapprox.ode.implicit_steppers.backward_euler import BackwardEulerHVP
from pyapprox.ode.protocols.ode_residual import ODEResidualProtocol
from pyapprox.ode.protocols.time_stepping import TimeSteppingResidualProtocol
from pyapprox.ode.stepper_table import (
    EXPLICIT_METHOD_NAMES,
    STEPPER_TABLE,
    create_stepper,
    resolve_stepper_factory,
)
from pyapprox.util.backends.numpy import NumpyBkd
from tests._helpers.ode_models import LotkaVolterra

# NumPy backend arrays: the dtype varies.
_Np = NDArray[Any]


class TestStepperTable:
    def test_all_builtins_present(self) -> None:
        assert sorted(STEPPER_TABLE) == [
            "backward_euler",
            "crank_nicolson",
            "forward_euler",
            "heun",
            "implicit_midpoint",
        ]

    def test_table_is_frozen(self) -> None:
        with pytest.raises(TypeError):
            STEPPER_TABLE["rk4"] = ForwardEulerHVP  # type: ignore[index]

    def test_explicit_names_subset_of_table(self) -> None:
        assert EXPLICIT_METHOD_NAMES <= set(STEPPER_TABLE)

    def test_resolve_builtin_name(self) -> None:
        assert resolve_stepper_factory("backward_euler") is BackwardEulerHVP

    def test_resolve_unknown_name_lists_valid(self) -> None:
        with pytest.raises(ValueError, match="implicit_midpoint"):
            resolve_stepper_factory("rk4")

    def test_resolve_rejects_non_callable(self) -> None:
        with pytest.raises(TypeError, match="StepperFactory"):
            resolve_stepper_factory(42)  # type: ignore[arg-type]

    def test_resolve_passes_factory_through(self) -> None:
        def factory(
            residual: ODEResidualProtocol[_Np],
        ) -> TimeSteppingResidualProtocol[_Np]:
            return BackwardEulerHVP(residual)

        assert resolve_stepper_factory(factory) is factory

    def test_create_stepper_from_name(self, numpy_bkd: NumpyBkd) -> None:
        residual = LotkaVolterra(numpy_bkd)
        stepper = create_stepper("backward_euler", residual)
        assert isinstance(stepper, BackwardEulerHVP)
        assert isinstance(stepper, TimeSteppingResidualProtocol)
        assert stepper.native_residual is residual

    def test_create_stepper_from_factory(self, numpy_bkd: NumpyBkd) -> None:
        def factory(
            residual: ODEResidualProtocol[_Np],
        ) -> TimeSteppingResidualProtocol[_Np]:
            return ForwardEulerHVP(residual)

        stepper = create_stepper(factory, LotkaVolterra(numpy_bkd))
        assert isinstance(stepper, ForwardEulerHVP)

    def test_create_stepper_rejects_nonconforming_product(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        with pytest.raises(TypeError, match="TimeSteppingResidualProtocol"):
            create_stepper(
                lambda res: object(),  # type: ignore[arg-type, return-value]
                LotkaVolterra(numpy_bkd),
            )
