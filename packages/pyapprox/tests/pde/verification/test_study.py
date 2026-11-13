r"""Convergence studies and checks, against fake stacks with known errors.

A fake discretizer returns approximations whose errors follow a stated
law, so the checks are verified without PDE code:

- :math:`e = 3 h^2` gives a fitted rate of 2 to 1e-12, on halving mesh
  sizes and on irregular ones (0.3, 0.17, 0.11, 0.05);
- errors (0.1, 0.1, 0.025, 0.00625) at h = 0.5 ... 0.0625 fail
  ``Converges(2)`` over all levels and pass with the finest two;
- ``Recovers`` passes errors at 1e-14 and fails one at 1e-9;
- ``Stalls`` passes a constant error, fails :math:`h^2` and a zero error;
- ``NodalError`` reads the stack's nodal error, not its L2 error.

Also the study's and fit's refusals, and that the stack seam and the
checks import neither sympy nor an array backend.
"""

import subprocess
import sys
from typing import Callable, List, Optional, Sequence

import pytest

from pyapprox.pde.verification.protocols import (
    ApproximationProtocol,
    DiscretizerProtocol,
)
from pyapprox.pde.verification.study import (
    CheckResult,
    Converges,
    L2Error,
    NodalError,
    Recovers,
    Stalls,
    Study,
    fitted_rate,
    run_study,
    verify,
)
from pyapprox.util.backends.numpy import NumpyBkd

_HALVING = (0.5, 0.25, 0.125, 0.0625, 0.03125)


class _Approximation:
    def __init__(self, h: float, error: float) -> None:
        self._h, self._error = h, error

    def h(self) -> float:
        return self._h

    def l2_error(self) -> float:
        return self._error

    def nodal_error(self) -> float:
        return 2.0 * self._error


class _Fake:
    """Approximations with errors from a law, or given outright."""

    def __init__(
        self,
        hs: Sequence[float],
        law: Optional[Callable[[float], float]] = None,
        errors: Optional[Sequence[float]] = None,
    ) -> None:
        self._hs = hs
        self._errors = [law(h) for h in hs] if law is not None else list(errors or [])

    def approximations(self, problem: None) -> List[_Approximation]:
        return [_Approximation(h, e) for h, e in zip(self._hs, self._errors)]


def _close(bkd: NumpyBkd, actual: float, expected: float, rtol: float) -> None:
    bkd.assert_allclose(bkd.asarray([actual]), bkd.asarray([expected]), rtol=rtol)


class TestFittedRate:
    @pytest.mark.parametrize("hs", [_HALVING, (0.3, 0.17, 0.11, 0.05)])
    def test_power_law(self, numpy_bkd: NumpyBkd, hs: Sequence[float]) -> None:
        study = run_study(None, _Fake(hs, lambda h: 3.0 * h**2), L2Error())
        _close(numpy_bkd, fitted_rate(study), 2.0, 1e-12)

    def test_refusals(self) -> None:
        with pytest.raises(ValueError, match="at least two levels"):
            fitted_rate(Study((0.5,), (0.1,)))
        with pytest.raises(ValueError, match="at least two levels"):
            fitted_rate(Study((0.5, 0.25), (0.1, 0.01)), last=1)
        with pytest.raises(ValueError, match="nonzero errors"):
            fitted_rate(Study((0.5, 0.25), (0.1, 0.0)))
        with pytest.raises(ValueError, match="distinct mesh sizes"):
            fitted_rate(Study((0.5, 0.5), (0.1, 0.01)))


class TestStudy:
    @pytest.mark.parametrize(
        "h, errors, match",
        [
            ((), (), "at least one"),
            ((0.5, 0.25), (0.1,), "one error per mesh size"),
            ((0.0,), (0.1,), "positive"),
            ((0.5,), (-0.1,), "non-negative"),
            ((0.5,), (float("nan"),), "finite"),
        ],
    )
    def test_refusals(
        self, h: tuple[float, ...], errors: tuple[float, ...], match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            Study(h, errors)

    def test_nodal_error_reads_the_stack(self, numpy_bkd: NumpyBkd) -> None:
        fake = _Fake(_HALVING[:2], lambda h: h)
        l2 = run_study(None, fake, L2Error())
        nodal = run_study(None, fake, NodalError())
        numpy_bkd.assert_allclose(
            numpy_bkd.asarray(nodal.errors), 2.0 * numpy_bkd.asarray(l2.errors)
        )
        assert nodal.h == _HALVING[:2]


class TestChecks:
    _PRE_ASYMPTOTIC = _Fake(_HALVING[:4], errors=(0.1, 0.1, 0.025, 0.00625))

    def test_converges(self, numpy_bkd: NumpyBkd) -> None:
        result = verify(
            None, _Fake(_HALVING, lambda h: 3.0 * h**2), L2Error(), Converges(2.0)
        )
        assert isinstance(result, CheckResult) and result.passed
        _close(numpy_bkd, result.observed, 2.0, 1e-12)
        failed = verify(
            None, _Fake(_HALVING, lambda h: 3.0 * h), L2Error(), Converges(2.0)
        )
        assert not failed.passed and "fitted rate 1.000" in failed.message

    def test_converges_past_a_pre_asymptotic_start(self, numpy_bkd: NumpyBkd) -> None:
        fake = self._PRE_ASYMPTOTIC
        assert not verify(None, fake, L2Error(), Converges(2.0)).passed
        result = verify(None, fake, L2Error(), Converges(2.0, last=2))
        assert result.passed
        _close(numpy_bkd, result.observed, 2.0, 1e-12)

    def test_recovers(self) -> None:
        exact = _Fake(_HALVING[:2], errors=(1e-14, 3e-15))
        assert verify(None, exact, L2Error(), Recovers(1e-12)).passed
        inexact = _Fake(_HALVING[:2], errors=(1e-14, 1e-9))
        result = verify(None, inexact, L2Error(), Recovers(1e-12))
        assert not result.passed and result.observed == 1e-9

    def test_stalls(self) -> None:
        constant = _Fake(_HALVING, lambda h: 0.3)
        assert verify(None, constant, L2Error(), Stalls(0.5)).passed
        converging = _Fake(_HALVING, lambda h: h**2)
        assert not verify(None, converging, L2Error(), Stalls(0.5)).passed
        zero = _Fake(_HALVING[:2], errors=(0.1, 0.0))
        assert not verify(None, zero, L2Error(), Stalls(0.5)).passed

    def test_reprs(self) -> None:
        assert repr(Converges(2.0, last=3)) == "Converges(rate=2.0, atol=0.1, last=3)"
        assert repr(Recovers(1e-12)) == "Recovers(atol=1e-12)"
        assert repr(Stalls(0.5)) == "Stalls(below=0.5)"


class TestRefusals:
    def test_wrong_types(self) -> None:
        fake = _Fake(_HALVING[:2], lambda h: h)
        assert isinstance(fake, DiscretizerProtocol)
        with pytest.raises(TypeError, match="DiscretizerProtocol"):
            run_study(None, 3.0, L2Error())  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="ErrorMeasureProtocol"):
            run_study(None, fake, 3.0)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="CheckProtocol"):
            verify(None, fake, L2Error(), 3.0)  # type: ignore[arg-type]

    def test_an_approximation_of_the_wrong_type(self) -> None:
        class _Broken:
            def approximations(self, problem: None) -> List[float]:
                return [0.5]

        assert not isinstance(0.5, ApproximationProtocol)
        with pytest.raises(TypeError, match="ApproximationProtocol"):
            run_study(None, _Broken(), L2Error())  # type: ignore[arg-type]


def test_seam_and_checks_import_no_sympy_or_backend() -> None:
    code = (
        "import sys\n"
        "import pyapprox.pde.verification\n"
        "import pyapprox.pde.verification.domains\n"
        "import pyapprox.pde.verification.protocols\n"
        "import pyapprox.pde.verification.study\n"
        "heavy = [m for m in ('sympy', 'pyapprox.util.backends.numpy', 'torch')\n"
        "         if m in sys.modules]\n"
        "print(heavy)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"
