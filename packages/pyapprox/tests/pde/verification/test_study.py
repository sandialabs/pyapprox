r"""Convergence studies and checks, against fake stacks with known errors.

A fake discretizer returns approximations whose errors follow a stated
law of its sizes, so the checks are verified without PDE code:

- :math:`e = 3 h^2` gives a fitted rate of 2 to 1e-12, on halving mesh
  sizes and on irregular ones (0.3, 0.17, 0.11, 0.05);
- :math:`e = 5 e^{-1.5 p}` gives an exponential decay of 1.5 per degree
  to 1e-12;
- errors (0.1, 0.1, 0.025, 0.00625) at h = 0.5 ... 0.0625 fail
  ``Converges(2)`` over all levels and pass with the finest two;
- :math:`e = h^3` fails ``Converges(2)`` from above, and passes when the
  case names superconvergence up to 3;
- ``Recovers`` passes errors at 1e-14 and fails one at 1e-9;
- ``Stalls`` passes a constant error, fails :math:`h^2` and a zero error;
- one solve per level feeds two measures and two checks (a counting
  fake);
- a study from plain records is judged as one from a run, along an axis a
  test declares and along one part's axis;
- every check result is a strict JSON record.

Also the study's, the sizes' and the fit's refusals, and that the stack
seam and the checks import neither sympy nor an array backend.
"""

import dataclasses
import json
import math
import subprocess
import sys
from typing import Callable, Dict, List, Optional, Sequence

import pytest

from pyapprox.pde.verification.protocols import (
    DEGREE,
    DT,
    Axis,
    DiscretizerProtocol,
    H,
    Sizes,
    SpaceLevel,
    part,
)
from pyapprox.pde.verification.study import (
    CheckResult,
    Column,
    Converges,
    DecaysExponentially,
    Kind,
    Measure,
    Measurement,
    MeasureProtocol,
    ObservedOrder,
    Recovers,
    Stalls,
    Study,
    field_label,
    fitted_rate,
    run_study,
    verify,
)
from pyapprox.util.backends.numpy import NumpyBkd

_HALVING = (0.5, 0.25, 0.125, 0.0625, 0.03125)


class _Approximation:
    def __init__(self, error: float, norm: float = 1.0) -> None:
        self._error, self._norm = error, norm

    def error(self) -> float:
        return self._error

    def norm(self) -> float:
        return self._norm


class _L2:
    """The fake's error, beside its norm, in column ``"l2"``."""

    def label(self) -> str:
        return "l2"

    def kind(self) -> Kind:
        return Kind.ERROR

    def measure(self, approximation: _Approximation) -> Measurement:
        return Measurement(approximation.error(), approximation.norm())


class _Nodal(_L2):
    """Twice the fake's error, without a norm, in column ``"nodal"``."""

    def label(self) -> str:
        return "nodal"

    def measure(self, approximation: _Approximation) -> Measurement:
        return Measurement(2.0 * approximation.error())


class _Iterations:
    """A value column: an iteration count that grows with the level."""

    def label(self) -> str:
        return "iterations"

    def kind(self) -> Kind:
        return Kind.VALUE

    def measure(self, approximation: _Approximation) -> Measurement:
        return Measurement(-math.log2(approximation.error()))


class _OtherApproximation:
    """Another stack's approximation: what ``_L2`` must not be paired
    with."""

    def residual(self) -> float:
        return 0.0


class _OtherStack:
    def requests(self, problem: None) -> List[SpaceLevel]:
        return [SpaceLevel(0)]

    def sizes(self, problem: None, request: SpaceLevel) -> Sizes:
        return Sizes({H: 0.5})

    def solve(self, problem: None, request: SpaceLevel) -> _OtherApproximation:
        return _OtherApproximation()


class _Fake:
    """Levels of given sizes along one axis, with errors from a law of
    the size or given outright; counts its solves."""

    def __init__(
        self,
        values: Sequence[float],
        law: Optional[Callable[[float], float]] = None,
        errors: Optional[Sequence[float]] = None,
        axis: Axis = H,
    ) -> None:
        self._values, self._axis = list(values), axis
        self._errors = (
            [law(v) for v in values] if law is not None else list(errors or [])
        )
        self._nsolves = 0

    def requests(self, problem: None) -> List[SpaceLevel]:
        return [SpaceLevel(index) for index in range(len(self._values))]

    def sizes(self, problem: None, request: SpaceLevel) -> Sizes:
        return Sizes({self._axis: self._values[request.index]})

    def solve(self, problem: None, request: SpaceLevel) -> _Approximation:
        self._nsolves += 1
        return _Approximation(self._errors[request.index])

    def nsolves(self) -> int:
        return self._nsolves


def _close(
    bkd: NumpyBkd, actual: Optional[float], expected: float, rtol: float
) -> None:
    assert actual is not None
    bkd.assert_allclose(bkd.asarray([actual]), bkd.asarray([expected]), rtol=rtol)


def _study(hs: Sequence[float], errors: Sequence[float]) -> Study:
    return Study.from_records([(Sizes({H: h}), {"l2": e}) for h, e in zip(hs, errors)])


class TestFittedRate:
    @pytest.mark.parametrize("hs", [_HALVING, (0.3, 0.17, 0.11, 0.05)])
    def test_power_law(self, numpy_bkd: NumpyBkd, hs: Sequence[float]) -> None:
        study = run_study(None, _Fake(hs, lambda h: 3.0 * h**2), [_L2()])
        _close(numpy_bkd, fitted_rate(study), 2.0, 1e-12)

    def test_refusals(self) -> None:
        with pytest.raises(ValueError, match="at least two levels"):
            fitted_rate(_study((0.5,), (0.1,)))
        with pytest.raises(ValueError, match="at least two levels"):
            fitted_rate(_study((0.5, 0.25), (0.1, 0.01)), last=1)
        with pytest.raises(ValueError, match="nonzero errors"):
            fitted_rate(_study((0.5, 0.25), (0.1, 0.0)))
        with pytest.raises(ValueError, match="distinct sizes"):
            fitted_rate(_study((0.5, 0.5), (0.1, 0.01)))
        with pytest.raises(ValueError, match="no size 'dt'; it reports"):
            fitted_rate(_study((0.5, 0.25), (0.1, 0.01)), axis=DT)


class TestSizes:
    def test_refusals(self) -> None:
        with pytest.raises(ValueError, match="at least one axis"):
            Sizes({})
        with pytest.raises(ValueError, match="positive and finite; h is 0.0"):
            Sizes({H: 0.0})
        with pytest.raises(ValueError, match="positive and finite; p is inf"):
            Sizes({DEGREE: math.inf})
        with pytest.raises(TypeError, match="keyed by Axis"):
            Sizes({"h": 0.5})  # type: ignore[dict-item]
        with pytest.raises(ValueError, match="non-negative"):
            SpaceLevel(-1)

    def test_axes_are_open(self) -> None:
        """An axis a test declares, and one part's axis, read like the
        core's: no core change for a new way of refining."""
        aspect = Axis("aspect")
        sizes = Sizes({aspect: 4.0, part("left", H): 0.25, H: 0.5})
        assert sizes.of(aspect) == 4.0
        assert sizes.of(part("left", H)) == 0.25
        assert part("left", H) != part("right", H) and part("left", H) != H


class TestStudy:
    @pytest.mark.parametrize(
        "sizes, columns, match",
        [
            ((), (Column("l2", ()),), "at least one level"),
            ((Sizes({H: 0.5}),), (), "one column"),
            ((Sizes({H: 0.5}),), (Column("l2", (0.1, 0.2)),), "one value per level"),
            ((Sizes({H: 0.5}),), (Column("l2", (-0.1,)),), "non-negative"),
            ((Sizes({H: 0.5}),), (Column("l2", (math.nan,)),), "finite"),
            (
                (Sizes({H: 0.5}),),
                (Column("l2", (0.1,), Kind.ERROR, (1.0, 2.0)),),
                "one norm per level",
            ),
            (
                (Sizes({H: 0.5}),),
                (Column("l2", (0.1,), Kind.ERROR, (-1.0,)),),
                "norms must be finite and non-negative",
            ),
            (
                (Sizes({H: 0.5}),),
                (Column("q", (0.1,), Kind.VALUE, (1.0,)),),
                "only an error column has norms",
            ),
            (
                (Sizes({H: 0.5}),),
                (Column("l2", (0.1,)), Column("l2", (0.2,))),
                "unique",
            ),
        ],
    )
    def test_refusals(
        self, sizes: tuple[Sizes, ...], columns: tuple[Column, ...], match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            Study(sizes, columns)

    def test_record_refusals(self) -> None:
        with pytest.raises(ValueError, match="no records"):
            Study.from_records([])
        with pytest.raises(ValueError, match="the labels"):
            Study.from_records(
                [(Sizes({H: 0.5}), {"l2": 0.1}), (Sizes({H: 0.25}), {"h1": 0.1})]
            )

    def test_columns(self) -> None:
        study = Study.from_records(
            [(Sizes({H: h}), {"l2": h, "nodal": 2 * h}) for h in _HALVING[:2]]
        )
        assert study.labels() == ("l2", "nodal")
        assert study.column("nodal") == (1.0, 0.5)
        assert study.axis(H) == _HALVING[:2]
        with pytest.raises(ValueError, match="name one of the columns"):
            study.column()
        with pytest.raises(ValueError, match="no column 'h1'"):
            study.column("h1")

    def test_one_solve_per_level_feeds_every_measure_and_check(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        fake = _Fake(_HALVING, lambda h: 3.0 * h**2)
        study = run_study(None, fake, [_L2(), _Nodal()])
        assert fake.nsolves() == len(_HALVING)
        numpy_bkd.assert_allclose(
            numpy_bkd.asarray(study.column("nodal")),
            2.0 * numpy_bkd.asarray(study.column("l2")),
        )
        assert Converges(2.0, column="l2").verify(study).passed
        assert Converges(2.0, column="nodal").verify(study).passed
        assert fake.nsolves() == len(_HALVING)

    def test_column_kinds(self) -> None:
        """An error column carries its norms when the measure gives them;
        a value column (negative values allowed) records an iteration
        count beside the errors, and the checks refuse to fit it."""
        study = run_study(
            None, _Fake(_HALVING[:3], lambda h: h**2), [_L2(), _Nodal(), _Iterations()]
        )
        assert study.norms("l2") == (1.0, 1.0, 1.0)
        assert study.norms("nodal") is None
        assert study.column("iterations") == (2.0, 4.0, 6.0)
        with pytest.raises(ValueError, match="holds values, not errors"):
            study.errors("iterations")
        with pytest.raises(ValueError, match="holds values, not errors"):
            Converges(2.0, column="iterations").verify(study)
        values = Study.from_records(
            [(Sizes({H: h}), {"q": -h}) for h in _HALVING[:2]], {"q": Kind.VALUE}
        )
        assert values.column("q") == (-0.5, -0.25)
        with pytest.raises(ValueError, match="unknown columns"):
            Study.from_records([(Sizes({H: 0.5}), {"q": 1.0})], {"r": Kind.VALUE})
        with pytest.raises(ValueError, match="norms at some levels only"):
            Study.from_records(
                [
                    (Sizes({H: 0.5}), {"l2": Measurement(0.1, 1.0)}),
                    (Sizes({H: 0.25}), {"l2": 0.01}),
                ]
            )

    def test_measures_from_functions(self) -> None:
        """A ``Measure`` of a label and a reader is a measure: an error
        and a value column of mixed measures record together, labels by
        ``field_label``; an empty label or fields, or a reader that is
        not callable, are refused."""
        error: Measure[_Approximation] = Measure(
            field_label("l2", ("u", "v")),
            lambda a: Measurement(a.error(), a.norm()),
        )
        count: Measure[_Approximation] = Measure(
            "iterations", lambda a: Measurement(-math.log2(a.error())), Kind.VALUE
        )
        assert isinstance(error, MeasureProtocol)
        study = run_study(
            None, _Fake(_HALVING[:3], lambda h: h**2), [error, count, _L2()]
        )
        assert study.labels() == ("l2(u,v)", "iterations", "l2")
        assert study.column("iterations") == (2.0, 4.0, 6.0)
        assert field_label("max", None) == "max"
        with pytest.raises(ValueError, match="at least one field"):
            field_label("l2", ())
        with pytest.raises(ValueError, match="non-empty label"):
            Measure("", lambda a: Measurement(0.0))
        with pytest.raises(TypeError, match="must be callable"):
            Measure("l2", 3.0)  # type: ignore[arg-type]

    def test_given_levels(self) -> None:
        """A study solves the levels it is given, not the family's."""
        fake = _Fake(_HALVING, lambda h: h)
        study = run_study(None, fake, [_L2()], [SpaceLevel(4), SpaceLevel(2)])
        assert study.axis(H) == (_HALVING[4], _HALVING[2])
        assert fake.nsolves() == 2

    def test_from_records_judged_like_a_run(self, numpy_bkd: NumpyBkd) -> None:
        """Plain numbers, as read from a code run elsewhere, along an
        axis the test declares and along one subdomain's own size."""
        cells = Axis("cells")
        records = [
            (Sizes({cells: n, part("left", H): 1.0 / n}), {"l2": 3.0 / n**2})
            for n in (8, 16, 32)
        ]
        study = Study.from_records(records)
        result = Converges(-2.0, axis=cells).verify(study)
        assert result.passed
        _close(numpy_bkd, result.observed, -2.0, 1e-12)
        assert Converges(2.0, axis=part("left", H)).verify(study).passed


class TestChecks:
    _PRE_ASYMPTOTIC = (0.1, 0.1, 0.025, 0.00625)

    def test_converges(self, numpy_bkd: NumpyBkd) -> None:
        result = verify(
            None, _Fake(_HALVING, lambda h: 3.0 * h**2), _L2(), Converges(2.0)
        )
        assert isinstance(result, CheckResult) and result.passed
        _close(numpy_bkd, result.observed, 2.0, 1e-12)
        failed = verify(None, _Fake(_HALVING, lambda h: 3.0 * h), _L2(), Converges(2.0))
        assert not failed.passed and "fitted rate 1.000" in failed.message

    def test_converges_past_a_pre_asymptotic_start(self, numpy_bkd: NumpyBkd) -> None:
        fake = _Fake(_HALVING[:4], errors=self._PRE_ASYMPTOTIC)
        assert not verify(None, fake, _L2(), Converges(2.0)).passed
        result = verify(None, fake, _L2(), Converges(2.0, last=2))
        assert result.passed
        _close(numpy_bkd, result.observed, 2.0, 1e-12)

    def test_converges_too_fast_fails_unless_named(self) -> None:
        """A rate above the expected one is a finding, not a pass."""
        fake = _Fake(_HALVING, lambda h: h**3)
        result = verify(None, fake, _L2(), Converges(2.0))
        assert not result.passed and "between 1.900 and 2.100" in result.message
        allowed = Converges(2.0, superconvergence=3.0)
        assert verify(None, fake, _L2(), allowed).passed
        with pytest.raises(ValueError, match="below the rate"):
            Converges(2.0, superconvergence=1.0)

    def test_decays_exponentially(self, numpy_bkd: NumpyBkd) -> None:
        """e = 5 exp(-1.5 p) over degrees 2, 4, 6, 8: decay 1.5 per
        degree; requiring 2 fails; a zero error fails; one level or a
        study without degrees is refused."""
        fake = _Fake((2, 4, 6, 8), lambda p: 5.0 * math.exp(-1.5 * p), axis=DEGREE)
        result = verify(None, fake, _L2(), DecaysExponentially(1.0))
        assert result.passed
        _close(numpy_bkd, result.observed, 1.5, 1e-12)
        assert not verify(None, fake, _L2(), DecaysExponentially(2.0)).passed
        zero = _Fake((2, 4, 6), errors=(1.0, 0.1, 0.0), axis=DEGREE)
        assert not verify(None, zero, _L2(), DecaysExponentially(1.0)).passed
        with pytest.raises(ValueError, match="at least two levels"):
            verify(None, fake, _L2(), DecaysExponentially(1.0, last=1))
        with pytest.raises(ValueError, match="no size 'p'"):
            verify(None, _Fake(_HALVING, lambda h: h), _L2(), DecaysExponentially(1.0))

    def test_recovers(self, numpy_bkd: NumpyBkd) -> None:
        """Absolutely: errors at 1e-14 pass 1e-12 and one at 1e-9 fails,
        observed as the error over its tolerance. Relatively: errors
        1e-10 and 1e-4 of norms 1e2 and 1e8 pass rtol 1e-11 though no
        absolute tolerance would; a zero field (norm 0, error 1e-15)
        passes only through atol; a column without norms refuses rtol."""
        exact = _Fake(_HALVING[:2], errors=(1e-14, 3e-15))
        assert verify(None, exact, _L2(), Recovers(1e-12)).passed
        inexact = _Fake(_HALVING[:2], errors=(1e-14, 1e-9))
        result = verify(None, inexact, _L2(), Recovers(1e-12))
        assert not result.passed
        _close(numpy_bkd, result.observed, 1e3, 1e-12)
        scaled = Study.from_records(
            [
                (Sizes({H: 0.5}), {"l2": Measurement(1e-10, 1e2)}),
                (Sizes({H: 0.25}), {"l2": Measurement(1e-4, 1e8)}),
            ]
        )
        assert Recovers(rtol=1e-11).verify(scaled).passed
        assert not Recovers(atol=1e-9).verify(scaled).passed
        zero = Study.from_records([(Sizes({H: 0.5}), {"l2": Measurement(1e-15, 0.0)})])
        assert not Recovers(rtol=1e-11).verify(zero).passed
        assert Recovers(atol=1e-14, rtol=1e-11).verify(zero).passed
        with pytest.raises(ValueError, match="needs the norms"):
            Recovers(rtol=1e-11).verify(_study((0.5,), (0.0,)))
        with pytest.raises(ValueError, match="positive atol or rtol"):
            Recovers()
        with pytest.raises(ValueError, match="rtol must be non-negative"):
            Recovers(rtol=-1.0)

    def test_stalls(self) -> None:
        constant = _Fake(_HALVING, lambda h: 0.3)
        assert verify(None, constant, _L2(), Stalls(0.5)).passed
        converging = _Fake(_HALVING, lambda h: h**2)
        assert not verify(None, converging, _L2(), Stalls(0.5)).passed
        zero = _Fake(_HALVING[:2], errors=(0.1, 0.0))
        assert not verify(None, zero, _L2(), Stalls(0.5)).passed

    def test_reprs(self) -> None:
        assert repr(Converges(2.0, last=3)) == (
            "Converges(rate=2.0, atol=0.1, last=3, axis=Axis(name='h'), "
            "column=None, superconvergence=2.0)"
        )
        assert repr(Recovers(1e-12)) == "Recovers(atol=1e-12, rtol=0.0, column=None)"
        assert repr(Stalls(0.5)) == (
            "Stalls(below=0.5, axis=Axis(name='h'), column=None)"
        )
        assert repr(DecaysExponentially(1.0)) == (
            "DecaysExponentially(rate=1.0, last=None, axis=Axis(name='p'), column=None)"
        )


def test_check_results_are_strict_json() -> None:
    """Every check's verdicts, passing and failing, including one with no
    observed number, survive a strict JSON round trip unchanged."""
    halving = _study(_HALVING, [h**2 for h in _HALVING])
    degrees = Study.from_records(
        [(Sizes({DEGREE: p}), {"l2": math.exp(-p)}) for p in (2, 4, 6)]
    )
    zero = _study(_HALVING[:2], (0.1, 0.0))
    oscillating = Study.from_records(
        [
            (Sizes({H: h}), {"Q": q})
            for h, q in zip(_HALVING[:4], (1.0, 0.5, 0.75, 0.6))
        ],
        {"Q": Kind.VALUE},
    )
    results: List[CheckResult] = [
        Converges(2.0).verify(halving),
        Converges(3.0).verify(halving),
        DecaysExponentially(0.5).verify(degrees),
        Recovers(1e-12).verify(halving),
        Stalls(0.5).verify(zero),
        ObservedOrder(2.0).verify(oscillating),
    ]
    assert results[-1].observed is None
    for result in results:
        record: Dict[str, object] = dataclasses.asdict(result)
        text = json.dumps(record, allow_nan=False)
        assert CheckResult(**json.loads(text)) == result


class TestRefusals:
    def test_wrong_types(self) -> None:
        fake = _Fake(_HALVING[:2], lambda h: h)
        assert isinstance(fake, DiscretizerProtocol)
        with pytest.raises(TypeError, match="DiscretizerProtocol"):
            run_study(None, 3.0, [_L2()])  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="MeasureProtocol"):
            run_study(None, fake, [3.0])  # type: ignore[list-item]
        with pytest.raises(TypeError, match="CheckProtocol"):
            verify(None, fake, _L2(), 3.0)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="at least one measure"):
            run_study(None, fake, [])

    def test_shared_labels(self) -> None:
        with pytest.raises(ValueError, match="labels must be unique"):
            run_study(None, _Fake(_HALVING[:2], lambda h: h), [_L2(), _L2()])

    def test_a_measurement_or_sizes_of_the_wrong_type(self) -> None:
        class _BareFloat(_L2):
            def measure(self, approximation: _Approximation) -> float:  # type: ignore[override]
                return 0.5

        class _BrokenSizes(_Fake):
            def sizes(self, problem: None, request: SpaceLevel) -> float:  # type: ignore[override]
                return 0.5

        with pytest.raises(TypeError, match="must return a Measurement"):
            run_study(None, _Fake(_HALVING[:2], lambda h: h), [_BareFloat()])  # type: ignore[list-item]
        with pytest.raises(TypeError, match="sizes must be Sizes"):
            run_study(None, _BrokenSizes(_HALVING[:2], lambda h: h), [_L2()])  # type: ignore[arg-type]

    def test_a_measure_of_another_stack_is_a_type_error(self) -> None:
        """``_L2`` reads ``_Approximation``; ``_OtherStack`` returns
        ``_OtherApproximation``. mypy refuses the pairing (the ignore
        below is flagged as unused under ``--strict`` if it does not), and
        at run time the measure finds no ``error``."""
        assert isinstance(_L2(), MeasureProtocol)
        with pytest.raises(AttributeError, match="error"):
            run_study(None, _OtherStack(), [_L2()])  # type: ignore[misc]


def test_seam_and_checks_import_no_sympy_or_backend() -> None:
    code = (
        "import sys\n"
        "import pyapprox.pde.verification\n"
        "import pyapprox.pde.verification.domains\n"
        "import pyapprox.pde.verification.protocols\n"
        "import pyapprox.pde.verification.quantities\n"
        "import pyapprox.pde.verification.study\n"
        "heavy = [m for m in ('sympy', 'pyapprox.util.backends.numpy', 'torch')\n"
        "         if m in sys.modules]\n"
        "print(heavy)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"
