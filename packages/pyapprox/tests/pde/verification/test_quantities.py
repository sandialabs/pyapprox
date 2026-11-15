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

Truths and studies against a fake stack with :math:`Q_h = Q + 3h^2`:
the exact truth gives a fitted rate of 2; a reference refuses levels its
uncertainty does not resolve.
"""

from typing import Dict, List, Sequence

import pytest

from pyapprox.pde.verification.conditions import ConditionProtocol, Dirichlet
from pyapprox.pde.verification.domains import Box, Disk, DomainProtocol
from pyapprox.pde.verification.exact import exact_value
from pyapprox.pde.verification.problem import ManufacturedProblem
from pyapprox.pde.verification.protocols import QuantityDiscretizerProtocol
from pyapprox.pde.verification.quantities import (
    BoundaryFlux,
    Integral,
    PointValue,
    QuantityProtocol,
)
from pyapprox.pde.verification.study import (
    Converges,
    ExactValue,
    QuantityStudy,
    ReferenceValue,
    fitted_rate,
    run_quantity_study,
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

    def h(self) -> float:
        return self._h

    def quantity(self, quantity: QuantityProtocol) -> float:
        return self._value


class _Fake:
    """Values :math:`Q + 3h^2`, whatever the quantity."""

    def __init__(self, hs: Sequence[float], exact: float) -> None:
        self._hs, self._exact = hs, exact

    def approximations(self, problem: None) -> List[_Approximation]:
        return [_Approximation(h, self._exact + 3.0 * h**2) for h in self._hs]


class TestQuantityStudies:
    def test_exact_truth_gives_the_rate(self, numpy_bkd: NumpyBkd) -> None:
        """Values Q + 3h^2 against the exact Q have a fitted rate of 2."""
        fake = _Fake(_HALVING, 0.7)
        assert isinstance(fake, QuantityDiscretizerProtocol)
        study = run_quantity_study(None, fake, Integral("u"))
        errors = ExactValue(0.7).errors(study)
        _close(numpy_bkd, fitted_rate(errors), 2.0, 1e-10)
        assert Converges(2.0).verify(errors).passed

    def test_reference_truth(self, numpy_bkd: NumpyBkd) -> None:
        """A reference resolves errors above its uncertainty and refuses
        the levels it cannot resolve."""
        study = run_quantity_study(None, _Fake(_HALVING, 0.7), Integral("u"))
        resolved = ReferenceValue(0.7 + 1e-6, uncertainty=2e-6).errors(study)
        _close(numpy_bkd, fitted_rate(resolved), 2.0, 1e-3)
        with pytest.raises(ValueError, match="within the reference's uncertainty"):
            ReferenceValue(0.7, uncertainty=0.02).errors(study)

    @pytest.mark.parametrize(
        "h, values, match",
        [
            ((), (), "at least one"),
            ((0.5, 0.25), (0.1,), "one value per mesh size"),
            ((0.0,), (0.1,), "positive"),
            ((0.5,), (float("inf"),), "finite"),
        ],
    )
    def test_study_refusals(
        self, h: tuple[float, ...], values: tuple[float, ...], match: str
    ) -> None:
        """A quantity study needs matching, positive sizes and finite
        values."""
        with pytest.raises(ValueError, match=match):
            QuantityStudy(h, values)

    def test_truth_refusals(self) -> None:
        """Truths need finite values and a positive, finite uncertainty."""
        with pytest.raises(ValueError, match="finite"):
            ExactValue(float("nan"))
        with pytest.raises(ValueError, match="finite"):
            ReferenceValue(float("inf"), 1e-3)
        with pytest.raises(ValueError, match="positive"):
            ReferenceValue(0.5, 0.0)

    def test_wrong_types(self) -> None:
        """A discretizer, quantity or approximation of the wrong type is
        refused."""
        fake = _Fake(_HALVING, 0.7)
        with pytest.raises(TypeError, match="QuantityDiscretizerProtocol"):
            run_quantity_study(None, 3.0, Integral("u"))  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="QuantityProtocol"):
            run_quantity_study(None, fake, 3.0)  # type: ignore[arg-type]

        class _Broken:
            def approximations(self, problem: None) -> List[float]:
                return [0.5]

        with pytest.raises(TypeError, match="QuantityApproximationProtocol"):
            run_quantity_study(None, _Broken(), Integral("u"))  # type: ignore[arg-type]

    def test_reprs(self) -> None:
        """Quantities and truths print what they are."""
        assert repr(PointValue("u", (0.5, 1))) == "PointValue('u', [0.5, 1.0])"
        assert repr(Integral("u")) == "Integral('u')"
        assert repr(BoundaryFlux("u", "left")) == "BoundaryFlux('u', 'left')"
        assert repr(ExactValue(0.5)) == "ExactValue(0.5)"
        assert repr(ReferenceValue(0.5, 0.1)) == "ReferenceValue(0.5, uncertainty=0.1)"
