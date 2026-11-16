"""Collocation nodal coefficients: values, declarations and refusals."""

import pickle
from typing import Generic

import numpy as np
import pytest

from pyapprox.pde.collocation.physics.coefficients import (
    CheckedValues,
    NodalCoefficientProtocol,
    NodalField,
    NodalSupplier,
    NodalTimeFunction,
    UndeclaredTimeDependenceWarning,
    as_nodal_coefficient,
)
from pyapprox.pde.constitutive.coefficient_functions import (
    TimeDependent,
    TimeIndependent,
)
from pyapprox.util.backends.protocols import Array, Backend

_NPTS = 4


def _nodes(bkd: Backend[Array]) -> Array:
    return bkd.asarray(np.linspace(-1.0, 1.0, 2 * _NPTS).reshape(2, _NPTS))


def _sum_of_coords(coords: Array) -> Array:
    return coords[0] + coords[1]


def _growing(coords: Array, time: float) -> Array:
    return (1.0 + time) * coords[0]


class TestNodalField:
    def test_values_fixed_in_time(self, bkd: Backend[Array]) -> None:
        values = bkd.asarray(np.arange(_NPTS, dtype=float))
        field = NodalField(values)
        assert isinstance(field, NodalCoefficientProtocol)
        bkd.assert_allclose(field.values(0.0), values)
        bkd.assert_allclose(field.values(7.0), values)
        assert not field.is_time_dependent()

    def test_set_values_bumps_the_version(self, bkd: Backend[Array]) -> None:
        field = NodalField(bkd.zeros((_NPTS,)))
        assert field.version() == 0
        new = bkd.ones((_NPTS,))
        field.set_values(new)
        assert field.version() == 1
        bkd.assert_allclose(field.values(0.0), new)

    def test_shapes_refused(self, numpy_bkd: Backend[Array]) -> None:
        with pytest.raises(ValueError, match="1-D"):
            NodalField(numpy_bkd.zeros((_NPTS, 1)))
        field = NodalField(numpy_bkd.zeros((_NPTS,)))
        with pytest.raises(ValueError, match=f"\\({_NPTS},\\)"):
            field.set_values(numpy_bkd.zeros((_NPTS + 1,)))

    def test_pickles(self, numpy_bkd: Backend[Array]) -> None:
        field = NodalField(numpy_bkd.ones((_NPTS,)))
        field.set_values(numpy_bkd.full((_NPTS,), 2.0))
        clone = pickle.loads(pickle.dumps(field))
        numpy_bkd.assert_allclose(clone.values(0.0), field.values(0.0))
        assert clone.version() == 1


class TestNodalSupplier:
    def test_declared_time_dependent(self, bkd: Backend[Array]) -> None:
        nodes = _nodes(bkd)
        supplier: TimeDependent[Array] = TimeDependent(_growing)
        coefficient = NodalSupplier(supplier, nodes)
        assert coefficient.is_time_dependent()
        bkd.assert_allclose(coefficient.values(1.0), 2.0 * nodes[0])

    def test_declared_time_independent(self, bkd: Backend[Array]) -> None:
        nodes = _nodes(bkd)
        supplier: TimeIndependent[Array] = TimeIndependent(_sum_of_coords)
        coefficient = NodalSupplier(supplier, nodes)
        assert not coefficient.is_time_dependent()
        bkd.assert_allclose(coefficient.values(5.0), nodes[0] + nodes[1])

    def test_undeclared_supplier_refused(self, numpy_bkd: Backend[Array]) -> None:
        with pytest.raises(TypeError, match="declare its time dependence"):
            NodalSupplier(_growing, _nodes(numpy_bkd))  # type: ignore[arg-type]


class TestAsNodalCoefficient:
    def test_constant(self, bkd: Backend[Array]) -> None:
        coefficient = as_nodal_coefficient(2.5, _NPTS, bkd)
        assert isinstance(coefficient, NodalField)
        bkd.assert_allclose(coefficient.values(0.0), bkd.full((_NPTS,), 2.5))

    def test_nodal_coefficient_kept(self, bkd: Backend[Array]) -> None:
        field = NodalField(bkd.ones((_NPTS,)))
        assert as_nodal_coefficient(field, _NPTS, bkd) is field

    def test_bare_time_function_warns(self, bkd: Backend[Array]) -> None:
        values = bkd.ones((_NPTS,))
        with pytest.warns(UndeclaredTimeDependenceWarning):
            coefficient = as_nodal_coefficient(lambda t: t * values, _NPTS, bkd)
        assert isinstance(coefficient, NodalTimeFunction)
        assert coefficient.is_time_dependent()
        bkd.assert_allclose(coefficient.values(3.0), 3.0 * values)

    def test_raw_array_refused(self, numpy_bkd: Backend[Array]) -> None:
        with pytest.raises(TypeError, match="NodalField"):
            as_nodal_coefficient(
                numpy_bkd.ones((_NPTS,)),  # type: ignore[arg-type]
                _NPTS,
                numpy_bkd,
            )

    def test_declared_supplier_refused(self, numpy_bkd: Backend[Array]) -> None:
        """A declared f(coords, t) must not be mistaken for an f(t)."""
        with pytest.raises(TypeError, match="NodalSupplier"):
            as_nodal_coefficient(
                TimeDependent(_growing),  # type: ignore[arg-type]
                _NPTS,
                numpy_bkd,
            )


class _CountingCheck(Generic[Array]):
    """A check that counts its calls and refuses values below a bound."""

    def __init__(self, bkd: Backend[Array], lower: float = 0.0) -> None:
        self._bkd = bkd
        self._lower = lower
        self.ncalls = 0

    def __call__(self, values: Array, time: float) -> None:
        self.ncalls += 1
        if self._bkd.to_float(self._bkd.min(values)) < self._lower:
            raise ValueError("below the bound")


class TestCheckedValues:
    def test_field_checked_once_per_change(self, bkd: Backend[Array]) -> None:
        """A NodalField is checked on the first read and after each
        set_values, not on every read."""
        field = NodalField(bkd.ones((_NPTS,)))
        check = _CountingCheck(bkd)
        checked = CheckedValues(field, check)
        for time in (0.0, 0.5, 1.0):
            bkd.assert_allclose(checked.values(time), bkd.ones((_NPTS,)))
        assert check.ncalls == 1
        field.set_values(bkd.full((_NPTS,), 2.0))
        checked.values(0.0)
        checked.values(0.0)
        assert check.ncalls == 2

    def test_failed_check_is_repeated(self, bkd: Backend[Array]) -> None:
        """Values that fail are not recorded as checked: every read of
        them raises."""
        field = NodalField(bkd.full((_NPTS,), -1.0))
        checked = CheckedValues(field, _CountingCheck(bkd))
        for _ in range(2):
            with pytest.raises(ValueError, match="below the bound"):
                checked.values(0.0)

    def test_declared_suppliers(self, bkd: Backend[Array]) -> None:
        """A time-independent supplier is checked once; a time-dependent
        one on every read, since its values change without a version
        change."""
        fixed: TimeIndependent[Array] = TimeIndependent(_sum_of_coords)
        fixed_check = _CountingCheck(bkd, lower=-10.0)
        fixed_values = CheckedValues(NodalSupplier(fixed, _nodes(bkd)), fixed_check)
        growing: TimeDependent[Array] = TimeDependent(_growing)
        growing_check = _CountingCheck(bkd, lower=-10.0)
        growing_values = CheckedValues(
            NodalSupplier(growing, _nodes(bkd)), growing_check
        )
        for time in (0.0, 0.5, 1.0):
            fixed_values.values(time)
            growing_values.values(time)
        assert fixed_check.ncalls == 1
        assert growing_check.ncalls == 3
