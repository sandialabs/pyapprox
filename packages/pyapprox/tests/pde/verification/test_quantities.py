r"""Quantities of interest: exact values, truths and quantity studies.

Hand-derived exact values, with :math:`u^* = x^2 y` and :math:`q =
-\nabla u` on the unit square:

- :math:`u^*(1/2, 3/10) = 3/40`; :math:`\int u^* = 1/6`;
- outward fluxes :math:`\int q\, n`: right :math:`-1`, left :math:`0`,
  bottom :math:`1/3`, top :math:`-1/3`. Their sum equals
  :math:`\int_\Omega \nabla\cdot q = \int_\Omega -2y = -1`, the
  divergence theorem, a check that needs no derivation;
- 1D, :math:`u^* = x^3`, :math:`q = -u_x`: right :math:`-3`, left
  :math:`0`.

Without a closed form: :math:`\int_0^1\int_0^2 \sin(\sin x)\, y \,dy\,dx
= 2 \times 0.4306061031206906` (the published value of
:math:`\int_0^1 \sin(\sin x)\,dx`), by quadrature.

Quantity columns against a fake stack with :math:`Q_h = Q + 3h^2`,
recorded beside a field error from the same solves: the exact truth
appends errors with a fitted rate of 2; a reference refuses levels its
uncertainty does not resolve.

Richardson on known sequences: :math:`Q + 3h^2` gives order 2 and limit
:math:`Q` to roundoff at ratios 2 and 3, and :math:`Q + 3\,dt^2` order
2 along the time step at a fixed mesh size; with a :math:`5h^3` term added
the extrapolated limit is closer to :math:`Q` than the finest value and
the GCI band covers the finest value's true error. Oscillatory,
diverging and stopped sequences fail with their reason; a pre-asymptotic
first level passes once dropped.
"""

from typing import Callable, Dict, List, Mapping, Sequence, Tuple

import pytest

from pyapprox.pde.verification.conditions import ConditionProtocol, Dirichlet
from pyapprox.pde.verification.domains import Box, Disk, DomainProtocol
from pyapprox.pde.verification.exact import exact_value
from pyapprox.pde.verification.problem import ManufacturedProblem
from pyapprox.pde.verification.protocols import DT, H, Sizes, SpaceLevel
from pyapprox.pde.verification.quantities import (
    BoundaryFlux,
    Integral,
    PointValue,
    QuantityProtocol,
)
from pyapprox.pde.verification.study import (
    Converges,
    ExactValue,
    Kind,
    Measure,
    Measurement,
    ObservedOrder,
    ReferenceValue,
    Study,
    extrapolate,
    fitted_rate,
    quantity_value,
    run_study,
)
from pyapprox.pde.verification.terms import Divergence, TimeDerivative
from pyapprox.util.backends.numpy import NumpyBkd

_SQUARE = Box([(0.0, 1.0), (0.0, 1.0)])
_HALVING = (0.5, 0.25, 0.125, 0.0625)
_SIN_SIN = 0.4306061031206906


def _close(bkd: NumpyBkd, actual: float, expected: float, atol: float) -> None:
    bkd.assert_allclose(
        bkd.asarray([actual]), bkd.asarray([expected]), rtol=0.0, atol=atol
    )


def _dirichlet(domain: DomainProtocol) -> Dict[str, ConditionProtocol]:
    return {part: Dirichlet() for part in domain.boundaries()}


def _diffusion(solution: str, domain: Box) -> ManufacturedProblem:
    """:math:`\\nabla\\cdot(-\\nabla u) = f` with ``solution`` on ``domain``."""
    flux = [f"-u_{name}" for name in domain.coordinates()]
    return ManufacturedProblem(
        {"u": solution}, [Divergence("u", flux)], domain, _dirichlet(domain)
    )


class TestExactValues:
    def test_point_value_and_integral(self, numpy_bkd: NumpyBkd) -> None:
        """A point value and a domain integral of x^2 y match their hand
        values."""
        problem = _diffusion("x**2*y", _SQUARE)
        point = exact_value(problem, PointValue("u", (0.5, 0.3)))
        _close(numpy_bkd, point, 3.0 / 40.0, 1e-15)
        _close(numpy_bkd, exact_value(problem, Integral("u")), 1.0 / 6.0, 1e-15)

    def test_boundary_fluxes_and_the_divergence_theorem(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """Each side's outward flux matches its hand value (the normal's
        sign included), and the fluxes sum to the integral of div q."""
        problem = _diffusion("x**2*y", _SQUARE)
        expected = {"left": 0.0, "right": -1.0, "bottom": 1.0 / 3.0, "top": -1.0 / 3.0}
        fluxes = {
            part: exact_value(problem, BoundaryFlux("u", part)) for part in expected
        }
        for part, value in expected.items():
            _close(numpy_bkd, fluxes[part], value, 1e-15)
        _close(numpy_bkd, sum(fluxes.values()), -1.0, 1e-15)

    def test_one_dimensional_flux_is_a_point_value(self, numpy_bkd: NumpyBkd) -> None:
        """In 1D a side is a point: the flux of x^3 is -3 at the right end
        and 0 at the left."""
        problem = _diffusion("x**3", Box([(0.0, 1.0)]))
        right = exact_value(problem, BoundaryFlux("u", "right"))
        _close(numpy_bkd, right, -3.0, 1e-15)
        _close(numpy_bkd, exact_value(problem, BoundaryFlux("u", "left")), 0.0, 1e-15)

    def test_quadrature_when_there_is_no_closed_form(self, numpy_bkd: NumpyBkd) -> None:
        """An integrand sympy cannot integrate (sin(sin x) y on a
        rectangle) falls back to quadrature and matches the published
        value."""
        rectangle = Box([(0.0, 1.0), (0.0, 2.0)])
        problem = _diffusion("sin(sin(x))*y", rectangle)
        value = exact_value(problem, Integral("u"))
        _close(numpy_bkd, value, 2.0 * _SIN_SIN, 1e-14)


class TestExactValueRefusals:
    def test_wrong_names(self) -> None:
        """A field, row or side the problem lacks, and a point of the
        wrong length, are refused."""
        problem = _diffusion("x**2*y", _SQUARE)
        with pytest.raises(ValueError, match="no field 'w'"):
            exact_value(problem, Integral("w"))
        with pytest.raises(ValueError, match="no side 'circle'"):
            exact_value(problem, BoundaryFlux("u", "circle"))
        with pytest.raises(ValueError, match="no row 'w'"):
            exact_value(problem, BoundaryFlux("w", "left"))
        with pytest.raises(ValueError, match="one value per coordinate"):
            exact_value(problem, PointValue("u", (0.5,)))
        with pytest.raises(ValueError, match="at least one coordinate"):
            PointValue("u", ())

    def test_disk_integral_is_not_built(self) -> None:
        """An integral over a disk is refused (polar coordinates are not
        built); a point value on the disk still works."""
        disk = Disk()
        problem = ManufacturedProblem(
            {"u": "x**2 + y"},
            [Divergence("u", ["-u_x", "-u_y"])],
            disk,
            _dirichlet(disk),
        )
        with pytest.raises(ValueError, match="needs a Box domain"):
            exact_value(problem, Integral("u"))
        assert exact_value(problem, PointValue("u", (0.5, 0.25))) == 0.5

    def test_transient_problem_is_refused(self) -> None:
        """A transient problem's quantity needs a time, which is not
        built."""
        line = Box([(0.0, 1.0)])
        problem = ManufacturedProblem(
            {"u": "x*(1 - x)*exp(-t)"},
            [TimeDerivative("u"), Divergence("u", ["-u_x"])],
            line,
            _dirichlet(line),
        )
        with pytest.raises(ValueError, match="transient"):
            exact_value(problem, Integral("u"))

    def test_quadrature_beyond_three_dimensions_is_refused(self) -> None:
        """Without a closed form, a four-dimensional integral is refused
        rather than left to a quadrature that would take minutes."""
        box = Box([(0.0, 1.0)] * 4, coordinates=("x", "y", "z", "w"))
        problem = _diffusion("sin(sin(x))*y*z*w", box)
        with pytest.raises(ValueError, match="at most 3 dimensions"):
            exact_value(problem, Integral("u"))

    def test_not_a_quantity(self) -> None:
        """Something that is not a quantity is a type error."""
        problem = _diffusion("x**2*y", _SQUARE)
        with pytest.raises(TypeError, match="QuantityProtocol"):
            exact_value(problem, 3.0)  # type: ignore[arg-type]


class TestBoxSides:
    def test_sides(self) -> None:
        """Each side names the coordinate across it and the sign of its
        outward normal, for default and named coordinates."""
        assert _SQUARE.side("left") == ("x", -1.0)
        assert _SQUARE.side("top") == ("y", 1.0)
        phase = Box([(0.0, 1.0), (-1.0, 1.0)], coordinates=("x", "v"))
        assert phase.side("vmin") == ("v", -1.0)
        with pytest.raises(ValueError, match="no side"):
            phase.side("left")


class _Approximation:
    def __init__(self, h: float, value: float) -> None:
        self._h, self._value = h, value

    def error(self) -> float:
        return 0.1 * self._h**2

    def quantity(self, quantity: QuantityProtocol) -> float:
        return self._value


class _Fake:
    """Values :math:`Q + 3h^2`, whatever the quantity; counts its
    solves."""

    def __init__(self, hs: Sequence[float], exact: float) -> None:
        self._hs, self._exact, self._nsolves = list(hs), exact, 0

    def requests(self, problem: None) -> List[SpaceLevel]:
        return [SpaceLevel(i) for i in range(len(self._hs))]

    def sizes(self, problem: None, request: SpaceLevel) -> Sizes:
        return Sizes({H: self._hs[request.index]})

    def solve(self, problem: None, request: SpaceLevel) -> _Approximation:
        self._nsolves += 1
        h = self._hs[request.index]
        return _Approximation(h, self._exact + 3.0 * h**2)

    def nsolves(self) -> int:
        return self._nsolves


_L2: Measure[_Approximation] = Measure("l2", lambda a: Measurement(a.error()))


def _values(hs: Sequence[float], law: Callable[[float], float]) -> Study:
    return Study.from_records(
        [(Sizes({H: h}), {"Q": law(h)}) for h in hs], {"Q": Kind.VALUE}
    )


class TestQuantityColumns:
    def test_one_solve_feeds_a_quantity_and_an_error(self, numpy_bkd: NumpyBkd) -> None:
        """A quantity is a value column beside the field errors, from the
        same solves; the exact truth appends its errors, rate 2, with
        the exact value's magnitude as the norm."""
        fake = _Fake(_HALVING, 0.7)
        study = run_study(None, fake, [_L2, quantity_value(Integral("u"))])
        assert fake.nsolves() == len(_HALVING)
        assert study.labels() == ("l2", "Integral('u')")
        errors = ExactValue(0.7).errors(study, "Integral('u')")
        assert errors.labels()[-1] == "Integral('u') error"
        assert errors.norms("Integral('u') error") == (0.7,) * len(_HALVING)
        rate = fitted_rate(errors, "Integral('u') error")
        _close(numpy_bkd, rate, 2.0, 1e-10)
        assert Converges(2.0, column="Integral('u') error").verify(errors).passed
        assert Converges(2.0, column="l2").verify(errors).passed

    def test_reference_truth(self, numpy_bkd: NumpyBkd) -> None:
        """A reference resolves errors above its uncertainty and refuses
        the levels it cannot resolve."""
        study = run_study(
            None, _Fake(_HALVING, 0.7), [quantity_value(Integral("u"), "Q")]
        )
        resolved = ReferenceValue(0.7 + 1e-6, uncertainty=2e-6).errors(
            study, label="error"
        )
        _close(numpy_bkd, fitted_rate(resolved, "error"), 2.0, 1e-3)
        with pytest.raises(ValueError, match="within the reference's uncertainty"):
            ReferenceValue(0.7, uncertainty=0.02).errors(study)

    def test_truth_refusals(self) -> None:
        """Truths need finite values and a positive, finite uncertainty,
        and read value columns only; a quantity must be one."""
        with pytest.raises(ValueError, match="finite"):
            ExactValue(float("nan"))
        with pytest.raises(ValueError, match="finite"):
            ReferenceValue(float("inf"), 1e-3)
        with pytest.raises(ValueError, match="positive"):
            ReferenceValue(0.5, 0.0)
        errors = Study.from_records([(Sizes({H: 0.5}), {"l2": 0.1})])
        with pytest.raises(ValueError, match="holds errors, not values"):
            ExactValue(0.5).errors(errors)
        with pytest.raises(TypeError, match="QuantityProtocol"):
            quantity_value(3.0)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="holds values, not errors"):
            Converges(2.0).verify(_values(_HALVING, lambda h: h))

    def test_reprs(self) -> None:
        """Quantities and truths print what they are."""
        assert repr(PointValue("u", (0.5, 1))) == "PointValue('u', [0.5, 1.0])"
        assert repr(Integral("u")) == "Integral('u')"
        assert repr(BoundaryFlux("u", "left")) == "BoundaryFlux('u', 'left')"
        assert repr(ExactValue(0.5)) == "ExactValue(0.5)"
        assert repr(ReferenceValue(0.5, 0.1)) == "ReferenceValue(0.5, uncertainty=0.1)"


class TestRichardson:
    @pytest.mark.parametrize("ratio", [2.0, 3.0])
    def test_exact_order_and_limit(self, numpy_bkd: NumpyBkd, ratio: float) -> None:
        """Q + 3h^2 gives order 2 and limit Q to roundoff, at a refinement
        ratio of 2 and of 3."""
        hs = [0.5 / ratio**k for k in range(4)]
        estimate = extrapolate(_values(hs, lambda h: 0.7 + 3.0 * h**2))
        _close(numpy_bkd, estimate.order, 2.0, 1e-8)
        _close(numpy_bkd, estimate.limit, 0.7, 1e-13)

    def test_along_another_axis(self, numpy_bkd: NumpyBkd) -> None:
        """Q + 3 dt^2 along the time step, with a mesh size that does not
        change, gives order 2 in dt; along h it is refused."""
        records: List[Tuple[Sizes, Mapping[str, float]]] = [
            (Sizes({H: 0.1, DT: dt}), {"Q": 0.7 + 3.0 * dt**2}) for dt in _HALVING
        ]
        study = Study.from_records(records, {"Q": Kind.VALUE})
        _close(numpy_bkd, extrapolate(study, axis=DT).order, 2.0, 1e-8)
        assert ObservedOrder(2.0, axis=DT).verify(study).passed
        with pytest.raises(ValueError, match="constant refinement ratio"):
            extrapolate(study)

    def test_limit_improves_and_band_covers(self, numpy_bkd: NumpyBkd) -> None:
        """With a higher-order term the limit beats the finest value and
        the GCI band covers the finest value's true error."""
        study = _values(_HALVING, lambda h: 0.7 + 3.0 * h**2 + 5.0 * h**3)
        estimate = extrapolate(study)
        finest_error = abs(study.values()[-1] - 0.7)
        # 1.4e-3 against 1.3e-2 at h = 1/16, where 5h^3 is still a tenth
        # of 3h^2.
        assert abs(estimate.limit - 0.7) < 0.2 * finest_error
        assert estimate.gci >= finest_error
        # 2.26: the cubic term lifts the observed order above 2.
        _close(numpy_bkd, estimate.order, 2.0, 0.3)

    def test_observed_order_check(self, numpy_bkd: NumpyBkd) -> None:
        """ObservedOrder on a quantity's column, beside an error column,
        passes order 2 and fails an expected order of 3, reporting the
        observed order and the limit."""
        study = run_study(
            None, _Fake(_HALVING, 0.7), [_L2, quantity_value(Integral("u"), "Q")]
        )
        result = ObservedOrder(2.0, column="Q").verify(study)
        assert result.passed and "limit 0.7" in result.message
        assert result.observed is not None
        _close(numpy_bkd, result.observed, 2.0, 1e-8)
        assert not ObservedOrder(3.0, column="Q").verify(study).passed

    @pytest.mark.parametrize(
        "values, reason",
        [
            ((1.1, 0.95, 1.025, 0.9875), "oscillatory"),
            ((1.0, 1.1, 1.3, 1.7), "not converging"),
            ((1.0, 0.9, 0.9, 0.9), "stop changing"),
        ],
    )
    def test_not_asymptotic_fails_with_its_reason(
        self, values: tuple[float, ...], reason: str
    ) -> None:
        """Oscillatory, diverging and stopped sequences fail with their
        reason, not an order; extrapolate refuses them."""
        study = _values(_HALVING, dict(zip(_HALVING, values)).__getitem__)
        result = ObservedOrder(2.0).verify(study)
        assert not result.passed and reason in result.message
        assert result.observed is None
        with pytest.raises(ValueError, match=reason):
            extrapolate(study)

    def test_pre_asymptotic_start(self) -> None:
        """A first level off the asymptotic range fails over all levels
        and passes once dropped with last=3."""
        values = (0.6, 0.7 + 3 * 0.25**2, 0.7 + 3 * 0.125**2, 0.7 + 3 * 0.0625**2)
        study = _values(_HALVING, dict(zip(_HALVING, values)).__getitem__)
        assert not ObservedOrder(2.0).verify(study).passed
        assert ObservedOrder(2.0, last=3).verify(study).passed

    def test_refusals(self) -> None:
        """Too few levels, a ratio that is not constant, an error column,
        and last below 3 are refused."""
        with pytest.raises(ValueError, match="three levels"):
            extrapolate(_values((0.5, 0.25), lambda h: h))
        with pytest.raises(ValueError, match="constant refinement ratio"):
            extrapolate(_values((0.5, 0.25, 0.1), lambda h: 1.0 + h))
        with pytest.raises(ValueError, match="holds errors, not values"):
            extrapolate(
                Study.from_records([(Sizes({H: h}), {"l2": h}) for h in _HALVING])
            )
        with pytest.raises(ValueError, match="three levels"):
            ObservedOrder(2.0, last=2)
        assert repr(ObservedOrder(2.0)) == (
            "ObservedOrder(order=2.0, atol=0.1, last=None, axis=Axis(name='h'), "
            "column=None)"
        )
