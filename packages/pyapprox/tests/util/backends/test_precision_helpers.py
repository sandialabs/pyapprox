"""The precision checks in ``tests/_helpers/precision.py``: ``assert_close``
scales its tolerance by the configured eps and keeps the strict dtype
check; ``no_dtype_leak`` fails a torch computation that widens a float
past the configured precision and passes one that does not."""

import pytest

from pyapprox.util.backends.precision import Precision
from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.precision import assert_close, no_dtype_leak


class TestAssertClose:
    def test_passes_within_the_configured_eps(self, config_bkd: Backend[Array]) -> None:
        desired = config_bkd.asarray([1.0, 2.0])
        actual = desired * (1.0 + 2.0 * config_bkd.eps())
        assert_close(config_bkd, actual, desired)

    def test_fails_beyond_it(self, config_bkd: Backend[Array]) -> None:
        desired = config_bkd.asarray([1.0, 2.0])
        actual = desired * (1.0 + 100.0 * config_bkd.eps())
        with pytest.raises(AssertionError):
            assert_close(config_bkd, actual, desired)

    def test_factor_loosens_it(self, config_bkd: Backend[Array]) -> None:
        desired = config_bkd.asarray([1.0, 2.0])
        actual = desired * (1.0 + 100.0 * config_bkd.eps())
        assert_close(config_bkd, actual, desired, factor=1000.0)

    def test_single_precision_error_fails_under_double(
        self, bkd: Backend[Array]
    ) -> None:
        # bkd is double precision: float32 rounding is far beyond 10 eps.
        desired = bkd.asarray([1.0, 2.0])
        actual = desired * (1.0 + 1e-7)
        with pytest.raises(AssertionError):
            assert_close(bkd, actual, desired)

    def test_dtype_stays_strict(self, config_bkd: Backend[Array]) -> None:
        desired = config_bkd.asarray([1.0, 2.0])
        wider = config_bkd.asarray([1.0, 2.0], dtype=config_bkd.complex_dtype())
        with pytest.raises(AssertionError):
            assert_close(config_bkd, wider, desired)

    def test_integers_compare_exactly(self, config_bkd: Backend[Array]) -> None:
        desired = config_bkd.arange(3)
        assert_close(config_bkd, config_bkd.arange(3), desired)
        with pytest.raises(AssertionError):
            assert_close(config_bkd, config_bkd.arange(1, 4), desired)


class TestNoDtypeLeak:
    def test_passes_a_computation_in_the_working_dtype(
        self, config_bkd: Backend[Array]
    ) -> None:
        x = config_bkd.linspace(0.0, 1.0, 4)
        with no_dtype_leak(config_bkd):
            result = config_bkd.sum(config_bkd.sin(x) * 2.0)
        assert result.dtype == config_bkd.real_dtype()

    def test_fails_a_float64_step_under_single_precision(
        self, torch_bkd: Backend[Array], torch_float32_bkd: Backend[Array]
    ) -> None:
        single = torch_float32_bkd
        assert single.precision() is Precision.SINGLE
        x = single.linspace(0.0, 1.0, 4)
        wide = torch_bkd.ones((4,))  # float64, the double backend's
        with pytest.raises(AssertionError, match="wider than torch.float32"):
            with no_dtype_leak(single):
                # Widened inside, cast back at the end: the result alone
                # would pass any dtype check.
                single.asarray(x * wide, dtype=single.real_dtype())

    def test_double_precision_allows_float64(self, torch_bkd: Backend[Array]) -> None:
        x = torch_bkd.linspace(0.0, 1.0, 4)
        with no_dtype_leak(torch_bkd):
            torch_bkd.sum(x * x)
