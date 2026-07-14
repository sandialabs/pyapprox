"""Tests for the bundle-driven numpy boundary adapter (replaces the
NumpyFunction*Wrapper ladder tests)."""

import pickle

import numpy as np
import pytest

from tests._helpers.optimizer_fixtures import (
    QuadraticNoDerivatives,
    QuadraticWithJacobian,
    QuadraticWithJacobianAndHVP,
    SumConstraintWithJacobianAndWHVP,
)

from pyapprox.interface.functions.numpy.adapter import (
    NumpyDerivativesAdapter,
)


def _make(bkd, producer_cls):
    producer = producer_cls(bkd, [1.0, -0.5])
    return producer, NumpyDerivativesAdapter(
        producer, producer.derivatives()
    )


class TestNumpyDerivativesAdapter:
    def test_value_conversion(self, bkd):
        _, adapter = _make(bkd, QuadraticNoDerivatives)
        values = adapter(np.array([[1.0, 0.0], [-0.5, 0.0]]))
        assert isinstance(values, np.ndarray)
        bkd.assert_allclose(
            bkd.asarray(values), bkd.asarray([[0.0, 1.25]])
        )

    def test_absent_capabilities_are_none(self, bkd):
        _, adapter = _make(bkd, QuadraticNoDerivatives)
        assert adapter.jacobian() is None
        assert adapter.hvp() is None
        assert adapter.whvp() is None

    def test_jacobian_conversion(self, bkd):
        _, adapter = _make(bkd, QuadraticWithJacobian)
        np_jac = adapter.jacobian()
        assert np_jac is not None
        jac = np_jac(np.array([[2.0], [0.0]]))
        assert isinstance(jac, np.ndarray)
        bkd.assert_allclose(bkd.asarray(jac), bkd.asarray([[2.0, 1.0]]))

    def test_hvp_coerces_int8_probe_to_double(self, bkd):
        producer, adapter = _make(bkd, QuadraticWithJacobianAndHVP)
        np_hvp = adapter.hvp()
        assert np_hvp is not None
        result = np_hvp(
            np.array([[1.0], [1.0]]), np.array([[1], [2]], dtype=np.int8)
        )
        bkd.assert_allclose(
            bkd.asarray(result), bkd.asarray([[2.0], [4.0]])
        )
        assert producer.foreign_vec_dtypes == []

    def test_whvp_resolved_with_owning_nqoi(self, bkd):
        # producer exposes only hvp (nqoi==1): adapter lifts it to whvp
        _, adapter = _make(bkd, QuadraticWithJacobianAndHVP)
        np_whvp = adapter.whvp()
        assert np_whvp is not None
        result = np_whvp(
            np.array([[1.0], [1.0]]),
            np.array([[1.0], [2.0]]),
            np.array([[3]], dtype=np.int8),  # weights probe coerced too
        )
        bkd.assert_allclose(
            bkd.asarray(result), bkd.asarray([[6.0], [12.0]])
        )

    def test_constraint_whvp_passthrough(self, bkd):
        constraint = SumConstraintWithJacobianAndWHVP(
            bkd, 2, 1.0, float("inf")
        )
        adapter = NumpyDerivativesAdapter(
            constraint, constraint.derivatives()
        )
        np_whvp = adapter.whvp()
        assert np_whvp is not None
        result = np_whvp(
            np.array([[1.0], [1.0]]),
            np.array([[1.0], [2.0]]),
            np.array([[3.0]]),
        )
        bkd.assert_allclose(bkd.asarray(result), bkd.zeros((2, 1)))
        assert constraint.foreign_dtypes == []

    def test_rejects_non_function(self, numpy_bkd):
        with pytest.raises(TypeError, match="FunctionProtocol"):
            NumpyDerivativesAdapter(
                object(),
                QuadraticNoDerivatives(numpy_bkd, [0.0]).derivatives(),
            )

    def test_rejects_non_bundle(self, numpy_bkd):
        producer = QuadraticNoDerivatives(numpy_bkd, [0.0])
        with pytest.raises(TypeError, match="Derivatives bundle"):
            NumpyDerivativesAdapter(producer, {"jacobian": None})

    def test_adapter_pickles(self, numpy_bkd):
        # multiprocessing requirement: numpy-facing callables are bound
        # methods, never closures
        _, adapter = _make(numpy_bkd, QuadraticWithJacobianAndHVP)
        restored = pickle.loads(pickle.dumps(adapter))
        np_jac = restored.jacobian()
        assert np_jac is not None
        numpy_bkd.assert_allclose(
            numpy_bkd.asarray(np_jac(np.array([[2.0], [0.0]]))),
            numpy_bkd.asarray([[2.0, 1.0]]),
        )
        np_hvp = restored.hvp()
        assert np_hvp is not None
        numpy_bkd.assert_allclose(
            numpy_bkd.asarray(
                np_hvp(np.array([[1.0], [1.0]]), np.array([[1.0], [2.0]]))
            ),
            numpy_bkd.asarray([[2.0], [4.0]]),
        )

    def test_sample_ndim_one(self, bkd):
        producer, _ = _make(bkd, QuadraticNoDerivatives)
        adapter = NumpyDerivativesAdapter(
            producer, producer.derivatives(), sample_ndim=1
        )
        values = adapter(np.array([1.0, -0.5]))
        bkd.assert_allclose(bkd.asarray(values), bkd.asarray([[0.0]]))
