"""``TorchBkd`` configured by ``Precision``: creation methods take the
real working dtype, ``array`` normalizes an existing tensor's or NumPy
array's precision and keeps its kind, MPS refuses double precision, and
construction leaves torch's global default dtype as it found it.

Torch is optional, so the whole module is skipped without it.
"""

from typing import Any, Callable

import numpy as np
import pytest

# Skip the module without torch; the imports below are then ordinary ones.
pytest.importorskip("torch")

import torch

from pyapprox.util.backends.precision import Precision
from pyapprox.util.backends.torch import TorchBkd

_REAL = {Precision.SINGLE: torch.float32, Precision.DOUBLE: torch.float64}
_COMPLEX = {Precision.SINGLE: torch.complex64, Precision.DOUBLE: torch.complex128}

_CREATION: dict[str, Callable[[TorchBkd], torch.Tensor]] = {
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
    bkd = TorchBkd(precision=precision)
    assert _CREATION[method](bkd).dtype == _REAL[precision]


@pytest.mark.parametrize("precision", list(Precision))
def test_dtype_accessors(precision: Precision) -> None:
    bkd = TorchBkd(precision=precision)
    assert bkd.default_dtype() == _REAL[precision]
    assert bkd.double_dtype() == _REAL[precision]
    assert bkd.complex_dtype() == _COMPLEX[precision]
    assert bkd.int64_dtype() == torch.int64


@pytest.mark.parametrize("precision", list(Precision))
def test_role_accessors(precision: Precision) -> None:
    bkd = TorchBkd(precision=precision)
    real, complex_ = _REAL[precision], _COMPLEX[precision]
    assert bkd.precision() is precision
    assert bkd.device() == "cpu"
    assert bkd.real_dtype() == real
    assert bkd.index_dtype() == torch.int64
    assert bkd.eps() == float(torch.finfo(real).eps)
    assert bkd.real_dtype_of(complex_) == real
    assert bkd.complex_dtype_of(real) == complex_
    # Aliases agree with the role accessors.
    assert bkd.double_dtype() == bkd.default_dtype() == bkd.real_dtype()
    assert bkd.int64_dtype() == bkd.index_dtype()


def test_single_precision_result_type_follows_numpy() -> None:
    bkd = TorchBkd(precision=Precision.SINGLE)
    floats = bkd.zeros((2,))
    assert bkd.result_type(floats, bkd.array([1 + 2j])) == torch.complex64
    # NumPy's rule, not torch's: int64 with float32 is float64.
    assert bkd.result_type(bkd.arange(2), floats) == torch.float64


def test_result_type_refuses_a_dtype_numpy_lacks() -> None:
    bkd = TorchBkd()
    with pytest.raises(ValueError, match="no NumPy equivalent"):
        bkd.result_type(torch.zeros(2, dtype=torch.bfloat16))


def test_default_is_double() -> None:
    assert TorchBkd().default_dtype() == torch.float64


@pytest.mark.parametrize("precision", list(Precision))
def test_integer_arange_stays_integer(precision: Precision) -> None:
    bkd = TorchBkd(precision=precision)
    assert bkd.arange(4).dtype == torch.int64
    assert bkd.arange(0, 4, 2).dtype == torch.int64


@pytest.mark.parametrize("precision", list(Precision))
def test_array_normalizes_precision_and_keeps_kind(precision: Precision) -> None:
    bkd = TorchBkd(precision=precision)
    sources: list[Any] = []
    for real in (torch.float32, torch.float64):
        sources.append(torch.tensor([1.0, 2.0], dtype=real))
    for real_np in (np.float32, np.float64):
        sources.append(np.asarray([1.0, 2.0], dtype=real_np))
    for source in sources:
        assert bkd.array(source).dtype == _REAL[precision]
    complex_sources: list[Any] = [
        torch.tensor([1 + 2j], dtype=torch.complex64),
        torch.tensor([1 + 2j], dtype=torch.complex128),
        np.asarray([1 + 2j], dtype=np.complex64),
        np.asarray([1 + 2j], dtype=np.complex128),
    ]
    for source in complex_sources:
        assert bkd.array(source).dtype == _COMPLEX[precision]
    assert bkd.array(torch.tensor([1, 2], dtype=torch.int32)).dtype == torch.int32
    assert bkd.array([True, False]).dtype == torch.bool


def test_array_copies() -> None:
    bkd = TorchBkd()
    source = torch.tensor([1.0, 2.0], dtype=torch.float64)
    result = bkd.array(source)
    assert result.data_ptr() != source.data_ptr()


@pytest.mark.parametrize("precision", list(Precision))
def test_construction_keeps_the_global_default_dtype(precision: Precision) -> None:
    # The torch fixture runs with float64 as the global default.
    before = torch.get_default_dtype()
    TorchBkd(precision=precision)
    assert torch.get_default_dtype() == before


def test_mps_refuses_double_precision() -> None:
    # Checked before any tensor is placed, so no MPS device is needed.
    with pytest.raises(ValueError, match="MPS has no float64"):
        TorchBkd(device="mps", precision=Precision.DOUBLE)


def test_precision_must_be_a_precision() -> None:
    with pytest.raises(TypeError, match="Precision"):
        TorchBkd(precision="single")  # type: ignore[arg-type]
