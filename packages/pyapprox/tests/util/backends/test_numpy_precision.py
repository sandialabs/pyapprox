"""``NumpyBkd`` configured by ``Precision``: creation methods take the
real working dtype, Python data is coerced by kind, an existing array is
adopted by ``asarray`` and normalized by ``array``, and a device other
than the CPU is refused."""

from typing import Any, Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.precision import Precision

_REAL: dict[Precision, type[np.floating[Any]]] = {
    Precision.SINGLE: np.float32,
    Precision.DOUBLE: np.float64,
}
_COMPLEX = {Precision.SINGLE: np.complex64, Precision.DOUBLE: np.complex128}

_CREATION: dict[str, Callable[[NumpyBkd], NDArray[Any]]] = {
    "zeros": lambda bkd: bkd.zeros((2, 3)),
    "ones": lambda bkd: bkd.ones((2, 3)),
    "empty": lambda bkd: bkd.empty((2, 3)),
    "full-float": lambda bkd: bkd.full((2, 3), 0.5),
    "eye": lambda bkd: bkd.eye(3),
    "linspace": lambda bkd: bkd.linspace(0.0, 1.0, 5),
    "logspace": lambda bkd: bkd.logspace(0.0, 1.0, 5),
    "arange-float": lambda bkd: bkd.arange(0.0, 1.0, 0.25),
    "asarray-float-list": lambda bkd: bkd.asarray([1.0, 2.0]),
    "array-float-list": lambda bkd: bkd.array([1.0, 2.0]),
}


@pytest.mark.parametrize("precision", list(Precision))
@pytest.mark.parametrize("method", list(_CREATION))
def test_creation_takes_the_real_working_dtype(
    precision: Precision, method: str
) -> None:
    bkd = NumpyBkd(precision=precision)
    assert _CREATION[method](bkd).dtype == _REAL[precision]


@pytest.mark.parametrize("precision", list(Precision))
def test_dtype_accessors(precision: Precision) -> None:
    bkd = NumpyBkd(precision=precision)
    assert bkd.default_dtype() == _REAL[precision]
    assert bkd.double_dtype() == _REAL[precision]
    assert bkd.complex_dtype() == _COMPLEX[precision]
    assert bkd.int64_dtype() == np.int64


@pytest.mark.parametrize("precision", list(Precision))
def test_role_accessors(precision: Precision) -> None:
    bkd = NumpyBkd(precision=precision)
    real, complex_ = _REAL[precision], _COMPLEX[precision]
    assert bkd.precision() is precision
    assert bkd.device() == "cpu"
    assert bkd.real_dtype() == real
    assert bkd.index_dtype() == np.int64
    assert bkd.eps() == float(np.finfo(real).eps)
    assert bkd.real_dtype_of(complex_) == real
    assert bkd.complex_dtype_of(real) == complex_
    # Aliases agree with the role accessors.
    assert bkd.double_dtype() == bkd.default_dtype() == bkd.real_dtype()
    assert bkd.int64_dtype() == bkd.index_dtype()


def test_single_precision_result_type_follows_numpy() -> None:
    bkd = NumpyBkd(precision=Precision.SINGLE)
    floats = bkd.zeros((2,))
    assert bkd.result_type(floats, bkd.asarray([1 + 2j])) == np.complex64
    # NumPy promotes int64 with float32 to float64, whatever the precision.
    assert bkd.result_type(bkd.arange(2), floats) == np.float64


def test_default_is_double() -> None:
    assert NumpyBkd().default_dtype() == np.float64


@pytest.mark.parametrize("precision", list(Precision))
def test_explicit_dtype_wins(precision: Precision) -> None:
    bkd = NumpyBkd(precision=precision)
    dtype = np.dtype(np.float32 if precision is Precision.DOUBLE else np.float64)
    assert bkd.zeros((2,), dtype=dtype).dtype == dtype
    assert bkd.eye(2, dtype=dtype).dtype == dtype
    assert bkd.asarray([1.0], dtype=dtype).dtype == dtype
    assert bkd.array([1.0], dtype=dtype).dtype == dtype
    assert bkd.full((2,), 1.0, dtype=dtype).dtype == dtype
    assert bkd.arange(0.0, 1.0, 0.5, dtype=dtype).dtype == dtype


@pytest.mark.parametrize("precision", list(Precision))
def test_python_data_keeps_its_kind(precision: Precision) -> None:
    bkd = NumpyBkd(precision=precision)
    assert bkd.asarray([1, 2, 3]).dtype == np.int64
    assert bkd.array([1, 2, 3]).dtype == np.int64
    assert bkd.asarray([True, False]).dtype == np.bool_
    assert bkd.asarray([1 + 2j]).dtype == _COMPLEX[precision]
    assert bkd.array([1 + 2j]).dtype == _COMPLEX[precision]
    assert bkd.full((2,), 3).dtype == np.int64
    assert bkd.arange(4).dtype == np.int64
    assert bkd.arange(0, 4, 2).dtype == np.int64


def test_asarray_adopts_an_existing_array() -> None:
    bkd = NumpyBkd(precision=Precision.SINGLE)
    foreign = np.asarray([1.0, 2.0], dtype=np.float64)
    assert bkd.asarray(foreign) is foreign


@pytest.mark.parametrize("precision", list(Precision))
def test_array_normalizes_precision_and_keeps_kind(precision: Precision) -> None:
    bkd = NumpyBkd(precision=precision)
    for real in (np.float32, np.float64):
        source = np.asarray([1.0, 2.0], dtype=real)
        result = bkd.array(source)
        assert result.dtype == _REAL[precision]
        assert not np.shares_memory(result, source)
    for complex_ in (np.complex64, np.complex128):
        source_c = np.asarray([1 + 2j], dtype=complex_)
        assert bkd.array(source_c).dtype == _COMPLEX[precision]
    integers = np.asarray([1, 2], dtype=np.int32)
    assert bkd.array(integers).dtype == np.int32


def test_array_copies_at_the_working_dtype() -> None:
    bkd = NumpyBkd()
    source = np.asarray([1.0, 2.0])
    result = bkd.array(source)
    assert result.dtype == np.float64
    assert not np.shares_memory(result, source)


@pytest.mark.parametrize("device", [None, "cpu"])
def test_cpu_is_accepted(device: Any) -> None:
    NumpyBkd(device=device)


def test_other_devices_are_refused() -> None:
    with pytest.raises(ValueError, match="CPU only"):
        NumpyBkd(device="cuda")


def test_precision_must_be_a_precision() -> None:
    with pytest.raises(TypeError, match="Precision"):
        NumpyBkd(precision="single")  # type: ignore[arg-type]
