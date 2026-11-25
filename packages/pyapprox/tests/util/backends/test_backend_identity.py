"""Backend identity and the bridges between backends: ``==``, ``hash``,
``repr``, ``is_compatible``, ``replace``, ``from_other``, and
``validate_backends`` comparing configurations.

Variants are derived with ``replace`` from the fixtures, so every
configuration of ``config_bkd`` (CPU and available devices) is covered.
"""

import pytest

from pyapprox.util.backends.autodiff import AutodiffBackend
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.precision import Precision
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.validation import validate_backends


def _other_precision(bkd: Backend[Array]) -> Precision:
    single = bkd.precision() is Precision.SINGLE
    return Precision.DOUBLE if single else Precision.SINGLE


class TestIdentity:
    def test_equal_configurations_are_equal(self, config_bkd: Backend[Array]) -> None:
        twin = config_bkd.replace()
        assert twin is not config_bkd
        assert twin == config_bkd
        assert hash(twin) == hash(config_bkd)
        assert len({config_bkd, twin}) == 1
        assert config_bkd.is_compatible(twin)

    def test_precision_distinguishes(self, bkd: Backend[Array]) -> None:
        # bkd is double precision on the CPU, so single always exists.
        other = bkd.replace(precision=Precision.SINGLE)
        assert other != bkd
        assert not bkd.is_compatible(other)
        assert other.precision() is Precision.SINGLE
        assert other.device() == bkd.device()

    def test_libraries_differ(
        self, numpy_bkd: NumpyBkd, torch_bkd: Backend[Array]
    ) -> None:
        assert not numpy_bkd.is_compatible(torch_bkd)
        assert not torch_bkd.is_compatible(numpy_bkd)
        assert numpy_bkd != torch_bkd

    def test_not_compatible_with_a_non_backend(self, bkd: Backend[Array]) -> None:
        assert not bkd.is_compatible(object())
        assert bkd != "NumpyBkd"  # type: ignore[comparison-overlap]

    def test_repr_names_the_configuration(self, numpy_bkd: NumpyBkd) -> None:
        assert repr(numpy_bkd) == "NumpyBkd(precision=double, device=cpu)"
        single = numpy_bkd.replace(precision=Precision.SINGLE)
        assert repr(single) == "NumpyBkd(precision=single, device=cpu)"

    def test_torch_compile_is_compatible_not_equal(
        self, torch_bkd: Backend[Array]
    ) -> None:
        from pyapprox.util.backends.torch import TorchBkd

        assert isinstance(torch_bkd, TorchBkd)
        compiled = torch_bkd.replace(compile=True)
        assert compiled.is_compatible(torch_bkd)
        assert compiled != torch_bkd
        assert "compile=True" in repr(compiled)
        assert compiled.replace(compile=False) == torch_bkd

    def test_replace_keeps_what_it_is_not_given(
        self, config_bkd: Backend[Array]
    ) -> None:
        assert config_bkd.replace().device() == config_bkd.device()
        assert config_bkd.replace().precision() is config_bkd.precision()


class TestFromOther:
    def test_compatible_source_returns_the_array(
        self, config_bkd: Backend[Array]
    ) -> None:
        x = config_bkd.linspace(0.0, 1.0, 3)
        assert config_bkd.from_other(x, config_bkd.replace()) is x

    def test_numpy_torch_round_trip(
        self, numpy_bkd: NumpyBkd, torch_float32_bkd: Backend[Array]
    ) -> None:
        x = numpy_bkd.asarray([[0.5, -1.25], [2.0, 3.0]])
        moved = torch_float32_bkd.from_other(x, numpy_bkd)
        # Precision follows the destination; values survive float32.
        assert moved.dtype == torch_float32_bkd.real_dtype()
        back = numpy_bkd.from_other(moved, torch_float32_bkd)
        assert back.dtype == numpy_bkd.real_dtype()
        numpy_bkd.assert_allclose(back, x)

    def test_kind_is_kept(
        self, numpy_bkd: NumpyBkd, torch_float32_bkd: Backend[Array]
    ) -> None:
        indices = numpy_bkd.arange(3)
        assert torch_float32_bkd.from_other(indices, numpy_bkd).dtype == (
            torch_float32_bkd.index_dtype()
        )
        values = numpy_bkd.asarray([1 + 2j])
        assert torch_float32_bkd.from_other(values, numpy_bkd).dtype == (
            torch_float32_bkd.complex_dtype()
        )

    def test_numpy_precision_change(self, numpy_bkd: NumpyBkd) -> None:
        single = numpy_bkd.replace(precision=Precision.SINGLE)
        x = numpy_bkd.asarray([1.0, 2.0])
        assert single.from_other(x, numpy_bkd).dtype == single.real_dtype()

    def test_torch_to_torch_keeps_the_graph(
        self,
        autodiff_bkd: AutodiffBackend[Array],
    ) -> None:
        single = autodiff_bkd.replace(precision=Precision.SINGLE)

        def through_single(x: Array) -> Array:
            narrowed = single.from_other(x, autodiff_bkd)
            return autodiff_bkd.from_other(narrowed * 2.0, single)

        x = autodiff_bkd.asarray([1.0, 2.0, 3.0])
        jacobian = autodiff_bkd.jacobian(through_single, x)
        autodiff_bkd.assert_allclose(jacobian, 2.0 * autodiff_bkd.eye(3))


class TestValidateBackends:
    def test_compatible_backends_pass(self, config_bkd: Backend[Array]) -> None:
        validate_backends([config_bkd, config_bkd.replace()])

    def test_mixed_precision_raises_naming_both(self, bkd: Backend[Array]) -> None:
        other = bkd.replace(precision=_other_precision(bkd))
        with pytest.raises(ValueError) as info:
            validate_backends([bkd, other])
        assert repr(bkd) in str(info.value)
        assert repr(other) in str(info.value)

    def test_mixed_libraries_raise(
        self, numpy_bkd: NumpyBkd, torch_bkd: Backend[Array]
    ) -> None:
        with pytest.raises(ValueError, match="Inconsistent backends"):
            validate_backends([numpy_bkd, torch_bkd])

    def test_a_class_is_refused(self, numpy_bkd: NumpyBkd) -> None:
        with pytest.raises(TypeError, match="backend instance"):
            validate_backends([numpy_bkd, NumpyBkd])

    def test_empty_is_refused(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            validate_backends([])
