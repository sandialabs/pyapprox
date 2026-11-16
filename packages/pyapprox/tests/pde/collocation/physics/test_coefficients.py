"""Collocation nodal coefficients: values, declarations and refusals."""

import pickle

import numpy as np
import pytest

from pyapprox.pde.collocation.physics.coefficients import (
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
