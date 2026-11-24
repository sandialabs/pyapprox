r"""The time axis, against a fake transient stack with a known error law.

The fake's error at time :math:`t` of a level is
:math:`(A h^p + B \Delta t^r)\, w(t)`, beside a truth norm :math:`2 - t`
on :math:`[0, 1]`:

- :func:`refine_time` at :math:`h = 10^{-4}` (p = 2, r = 1) fits a rate of
  1 along ``DT`` to 1e-3;
- :func:`refine_space` at :math:`\Delta t = 10^{-4}` (p = 2, r = 2) fits 2
  along ``H`` to 1e-3;
- :func:`refine_jointly` with :math:`q = p / r = 2` runs
  :math:`\Delta t = 0.1 (h / 0.5)^2` and fits 2 along ``H`` to 1e-10; with
  :math:`q = 1` the rate in :math:`h` falls toward :math:`q r = 1` and
  ``Converges(2)`` fails;
- :func:`fit_joint` recovers :math:`A = 3`, :math:`B = 0.5` to 1e-10 from
  a joint plan concatenated with a time-only one, and refuses the joint
  plan alone;
- with :math:`w(t) = \cos(\pi t / 2)`, an error that changes sign at the
  final time, ``Recovers(atol=1e-12)`` passes :func:`final_time` and
  fails :func:`max_over_steps`;
- the maximum's norm is the largest truth norm (2, at :math:`t = 0`), the
  final time's is its own (1).

Also the grid's, the plans', the combinators' and the fit's refusals, a
strict JSON round trip of a fit, and that the module imports neither sympy
nor an array backend.
"""

import dataclasses
import json
import math
import subprocess
import sys
from typing import Callable, Dict, List, Optional, Sequence

import pytest

from pyapprox.pde.verification.protocols import DT, H, Sizes, SpaceLevel
from pyapprox.pde.verification.study import (
    Converges,
    Kind,
    Measure,
    Measurement,
    Recovers,
    fitted_rate,
    run_study,
)
from pyapprox.pde.verification.transient import (
    JointFit,
    SeriesProtocol,
    SpaceSizesProtocol,
    TimeGrid,
    TransientLevel,
    final_time,
    fit_joint,
    max_over_steps,
    refine_jointly,
    refine_space,
    refine_time,
)
from pyapprox.util.backends.numpy import NumpyBkd

_HALVING = (0.5, 0.25, 0.125, 0.0625, 0.03125)


class _Approximation:
    def __init__(self, error: float, norm: float) -> None:
        self._error, self._norm = error, norm

    def error(self) -> float:
        return self._error

    def norm(self) -> float:
        return self._norm


def _l2() -> Measure[_Approximation]:
    return Measure("l2", lambda a: Measurement(a.error(), a.norm()))


class _Series:
    def __init__(
        self, scale: float, grid: TimeGrid, weight: Callable[[float], float]
    ) -> None:
        self._scale, self._times, self._weight = scale, grid.times(), weight

    def ntimes(self) -> int:
        return len(self._times)

    def times(self) -> Sequence[float]:
        return self._times

    def at(self, n: int) -> _Approximation:
        t = self._times[n]
        return _Approximation(abs(self._scale * self._weight(t)), 2.0 - t)


class _TransientFake:
    """Space levels of sizes ``hs``, each solved on any grid, with the
    error :math:`(A h^p + B \\Delta t^r) w(t)`."""

    def __init__(
        self,
        hs: Sequence[float],
        a: float,
        p: float,
        b: float,
        r: float,
        weight: Callable[[float], float] = lambda t: 1.0,
    ) -> None:
        self._hs, self._a, self._p, self._b, self._r = list(hs), a, p, b, r
        self._weight = weight

    def spaces(self) -> List[SpaceLevel]:
        return [SpaceLevel(index) for index in range(len(self._hs))]

    def requests(self, problem: None) -> List[TransientLevel[SpaceLevel]]:
        return refine_space(self.spaces(), TimeGrid(0.0, 1.0, 0.1))

    def space_sizes(self, problem: None, space: SpaceLevel) -> Sizes:
        return Sizes({H: self._hs[space.index]})

    def sizes(self, problem: None, request: TransientLevel[SpaceLevel]) -> Sizes:
        return Sizes({H: self._hs[request.space.index], DT: request.grid.dt})

    def solve(self, problem: None, request: TransientLevel[SpaceLevel]) -> _Series:
        h, dt = self._hs[request.space.index], request.grid.dt
        scale = self._a * h**self._p + self._b * dt**self._r
        return _Series(scale, request.grid, self._weight)


def _close(
    bkd: NumpyBkd, actual: Optional[float], expected: float, rtol: float
) -> None:
    assert actual is not None
    bkd.assert_allclose(bkd.asarray([actual]), bkd.asarray([expected]), rtol=rtol)


class TestTimeGrid:
    def test_steps_and_times(self) -> None:
        grid = TimeGrid(0.5, 1.5, 0.1)
        times = grid.times()
        assert grid.nsteps() == 10
        assert len(times) == 11
        assert times[0] == 0.5 and times[-1] == 1.5

    @pytest.mark.parametrize(
        "t0, tfinal, dt, match",
        [
            (1.0, 1.0, 0.1, "after t0"),
            (1.0, 0.5, 0.1, "after t0"),
            (0.0, 1.0, 0.0, "positive and finite"),
            (0.0, 1.0, -0.1, "positive and finite"),
            (0.0, math.inf, 0.1, "finite"),
            (0.0, 1.0, 0.3, "does not divide"),
            (0.0, 1.0, 2.0, "does not divide"),
        ],
    )
    def test_refusals(self, t0: float, tfinal: float, dt: float, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            TimeGrid(t0, tfinal, dt)


class TestPlans:
    def test_refine_time_isolates_the_temporal_order(self, numpy_bkd: NumpyBkd) -> None:
        fake = _TransientFake((1e-4,), 1.0, 2.0, 1.0, 1.0)
        requests = refine_time(SpaceLevel(0), TimeGrid(0.0, 1.0, 0.1), 5)
        assert [r.grid.nsteps() for r in requests] == [10, 20, 40, 80, 160]
        study = run_study(None, fake, [max_over_steps(_l2())], requests)
        _close(numpy_bkd, fitted_rate(study, axis=DT), 1.0, 1e-3)
        assert Converges(1.0, axis=DT, atol=0.01).verify(study).passed

    def test_refine_space_isolates_the_spatial_order(self, numpy_bkd: NumpyBkd) -> None:
        fake = _TransientFake(_HALVING, 1.0, 2.0, 1.0, 2.0)
        requests = refine_space(fake.spaces(), TimeGrid(0.0, 1.0, 1e-4))
        study = run_study(None, fake, [final_time(_l2())], requests)
        _close(numpy_bkd, fitted_rate(study), 2.0, 1e-3)

    def test_joint_with_q_equal_p_over_r_keeps_the_spatial_rate(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        fake = _TransientFake(_HALVING, 3.0, 2.0, 0.5, 1.0)
        assert isinstance(fake, SpaceSizesProtocol)
        requests = refine_jointly(None, fake, fake.spaces(), 2.0, 0.1, 0.0, 1.0)
        for request, h in zip(requests, _HALVING):
            _close(numpy_bkd, request.grid.dt, 0.1 * (h / 0.5) ** 2, 1e-12)
        study = run_study(None, fake, [max_over_steps(_l2())], requests)
        _close(numpy_bkd, fitted_rate(study), 2.0, 1e-10)

    def test_joint_with_q_too_small_loses_the_spatial_rate(self) -> None:
        fake = _TransientFake(_HALVING, 1.0, 2.0, 1.0, 1.0)
        requests = refine_jointly(None, fake, fake.spaces(), 1.0, 0.1, 0.0, 1.0)
        study = run_study(None, fake, [max_over_steps(_l2())], requests)
        assert not Converges(2.0).verify(study).passed
        assert fitted_rate(study, last=2) < 1.5

    def test_joint_rounds_the_step_down_to_divide_the_interval(self) -> None:
        fake = _TransientFake(_HALVING, 1.0, 2.0, 1.0, 1.0)
        requests = refine_jointly(None, fake, fake.spaces(), 1.0, 0.3, 0.0, 1.0)
        for request, h in zip(requests, _HALVING):
            assert request.grid.dt <= 0.3 * (h / 0.5) * (1.0 + 1e-12)
        assert requests[0].grid.nsteps() == 4

    def test_refusals(self) -> None:
        fake = _TransientFake(_HALVING, 1.0, 2.0, 1.0, 1.0)
        grid = TimeGrid(0.0, 1.0, 0.1)
        with pytest.raises(ValueError, match="at least one space level"):
            refine_space([], grid)
        with pytest.raises(TypeError, match="TimeGrid"):
            refine_space(fake.spaces(), 0.1)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="at least two levels"):
            refine_time(SpaceLevel(0), grid, 1)
        with pytest.raises(ValueError, match="at least 2"):
            refine_time(SpaceLevel(0), grid, 3, ratio=1)
        with pytest.raises(TypeError, match="SpaceSizesProtocol"):
            refine_jointly(None, 3.0, fake.spaces(), 2.0, 0.1, 0.0, 1.0)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="at least one space level"):
            refine_jointly(None, fake, [], 2.0, 0.1, 0.0, 1.0)
        for q, dt0 in ((0.0, 0.1), (-1.0, 0.1), (2.0, 0.0), (math.nan, 0.1)):
            with pytest.raises(ValueError, match="positive and finite"):
                refine_jointly(None, fake, fake.spaces(), q, dt0, 0.0, 1.0)
        with pytest.raises(ValueError, match="shrinking sizes"):
            refine_jointly(None, fake, fake.spaces()[::-1], 2.0, 0.1, 0.0, 1.0)
        with pytest.raises(ValueError, match="after t0"):
            refine_jointly(None, fake, fake.spaces(), 2.0, 0.1, 1.0, 0.0)


class TestCombinators:
    def test_an_error_changing_sign_at_the_final_time(self) -> None:
        """The final time sees the zero crossing; the maximum does not."""
        fake = _TransientFake(
            _HALVING[:3], 1.0, 2.0, 1.0, 1.0, lambda t: math.cos(math.pi * t / 2.0)
        )
        study = run_study(None, fake, [max_over_steps(_l2()), final_time(_l2())])
        assert study.labels() == ("max_t l2", "l2(T)")
        assert Recovers(atol=1e-12, column="l2(T)").verify(study).passed
        assert not Recovers(atol=1e-12, column="max_t l2").verify(study).passed

    def test_norms(self) -> None:
        fake = _TransientFake(_HALVING[:2], 1.0, 2.0, 1.0, 1.0)
        study = run_study(
            None, fake, [max_over_steps(_l2(), "max"), final_time(_l2(), "end")]
        )
        assert study.norms("max") == (2.0, 2.0)
        assert study.norms("end") == (1.0, 1.0)

    def test_final_time_of_a_value(self) -> None:
        value = Measure[_Approximation](
            "q", lambda a: Measurement(-a.error()), Kind.VALUE
        )
        fake = _TransientFake(_HALVING[:2], 1.0, 2.0, 1.0, 1.0)
        study = run_study(None, fake, [final_time(value)])
        assert study.find("q(T)").kind is Kind.VALUE
        assert all(v < 0.0 for v in study.values("q(T)"))

    def test_refusals(self) -> None:
        value = Measure[_Approximation](
            "q", lambda a: Measurement(a.error()), Kind.VALUE
        )
        with pytest.raises(ValueError, match="holds values"):
            max_over_steps(value)
        with pytest.raises(TypeError, match="MeasureProtocol"):
            max_over_steps(3.0)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="MeasureProtocol"):
            final_time(3.0)  # type: ignore[arg-type]
        grid = TimeGrid(0.0, 1.0, 0.5)
        series = _Series(1.0, grid, lambda t: 1.0)
        mixed = Measure[_Approximation](
            "mixed",
            lambda a: Measurement(a.error(), a.norm() if a.norm() == 2.0 else None),
        )
        with pytest.raises(ValueError, match="some times only"):
            max_over_steps(mixed).measure(series)
        bare = Measure[_Approximation]("bare", lambda a: a.error())  # type: ignore[arg-type, return-value]
        with pytest.raises(TypeError, match="must return a Measurement"):
            max_over_steps(bare).measure(series)
        with pytest.raises(TypeError, match="must return a Measurement"):
            final_time(bare).measure(series)

        class _Empty(_Series):
            def ntimes(self) -> int:
                return 0

        empty = _Empty(1.0, grid, lambda t: 1.0)
        assert isinstance(empty, SeriesProtocol)
        for combinator in (max_over_steps, final_time):
            with pytest.raises(ValueError, match="at least one time"):
                combinator(_l2()).measure(empty)


class TestFitJoint:
    def test_recovers_both_constants(self, numpy_bkd: NumpyBkd) -> None:
        fake = _TransientFake(_HALVING, 3.0, 2.0, 0.5, 1.0)
        joint = refine_jointly(None, fake, fake.spaces(), 2.0, 0.1, 0.0, 1.0)
        time = refine_time(SpaceLevel(0), TimeGrid(0.0, 1.0, 0.1), 3)
        study = run_study(None, fake, [max_over_steps(_l2())], joint + time)
        fit = fit_joint(study, 2.0, 1.0)
        _close(numpy_bkd, fit.space, 3.0, 1e-10)
        _close(numpy_bkd, fit.time, 0.5, 1e-10)
        assert fit.misfit < 1e-12

    def test_a_missing_term_fits_as_zero(self, numpy_bkd: NumpyBkd) -> None:
        fake = _TransientFake(_HALVING, 3.0, 2.0, 0.0, 1.0)
        joint = refine_jointly(None, fake, fake.spaces(), 2.0, 0.1, 0.0, 1.0)
        time = refine_time(SpaceLevel(2), TimeGrid(0.0, 1.0, 0.1), 3)
        study = run_study(None, fake, [max_over_steps(_l2())], joint + time)
        fit = fit_joint(study, 2.0, 1.0)
        _close(numpy_bkd, fit.space, 3.0, 1e-10)
        assert abs(fit.time) < 1e-10

    def test_strict_json(self) -> None:
        fit = JointFit(3.0, 0.5, 1e-14)
        record: Dict[str, object] = dataclasses.asdict(fit)
        assert JointFit(**json.loads(json.dumps(record, allow_nan=False))) == fit

    def test_refusals(self) -> None:
        fake = _TransientFake(_HALVING, 3.0, 2.0, 0.5, 1.0)
        joint = refine_jointly(None, fake, fake.spaces(), 2.0, 0.1, 0.0, 1.0)
        study = run_study(None, fake, [max_over_steps(_l2())], joint)
        with pytest.raises(ValueError, match="proportional"):
            fit_joint(study, 2.0, 1.0)
        for p, r in ((0.0, 1.0), (2.0, -1.0), (math.inf, 1.0)):
            with pytest.raises(ValueError, match="positive and finite"):
                fit_joint(study, p, r)
        with pytest.raises(ValueError, match="at least two levels"):
            fit_joint(study, 2.0, 1.0, last=1)
        zero = _TransientFake(_HALVING[:2], 0.0, 2.0, 0.0, 1.0)
        zeros = run_study(None, zero, [max_over_steps(_l2())])
        with pytest.raises(ValueError, match="nonzero errors"):
            fit_joint(zeros, 2.0, 1.0)


def test_transient_imports_no_sympy_or_backend() -> None:
    code = (
        "import sys\n"
        "import pyapprox.pde.verification.transient\n"
        "heavy = [m for m in ('sympy', 'pyapprox.util.backends.numpy', 'torch')\n"
        "         if m in sys.modules]\n"
        "print(heavy)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"
