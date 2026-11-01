r"""The multistep formulas: their order on any steps, and validation.

BDF2, Adams-Bashforth 2 and Adams-Moulton 3 give each step's formula from
its span's sizes. Checked by the definition of order, so independently
of how the coefficients are derived:

- on random uneven sizes, every formula, full and startup, is exact,
  :math:`\sum_l \alpha_l\, p(t_{n+1-l}) = \Delta t \sum_l \beta_l\,
  p'(t_{n+1-l})` at the span's times, for the monomials :math:`p = t^q`
  up to its stated order (the startup formula one order lower), and fails
  at the next degree, so the stated order is exact;
- at equal steps they satisfy the classical order conditions to the same
  orders and equal the classical tables to 1e-14; equal sizes give one
  formula bit for bit wherever the step lies (BDF4 from 0 and 1000).

A fixed table (:class:`MultistepMethod`) picks its formula by the states
read and refuses steps that are not exactly equal, those recovered from
the times 0.1, 0.2, 0.3 included. Refusals of both.
"""

from typing import Callable, List

import numpy as np
import pytest

from pyapprox.ode.multistep_methods import (
    AdamsBashforth,
    AdamsMoulton,
    BackwardDifferentiation,
    MultistepCoefficients,
    MultistepFormulaProtocol,
    MultistepMethod,
    adams_bashforth2,
    adams_moulton3,
    bdf2,
)
from pyapprox.ode.step_span import StepSpan
from tests._helpers.ode.multistep_cases import METHOD_IDS, METHODS

_EQUAL = StepSpan(0.1, (0.1, 0.1))
"""Two equal steps of 0.1 from 0.1."""
_ONE = StepSpan.first(0.2, 0.1)
"""One step of 0.1 from 0.2."""


def _exactness_defect(formula: MultistepCoefficients, span: StepSpan, q: int) -> float:
    r""":math:`\sum_l \alpha_l\, p(t_l) - \Delta t \sum_l \beta_l\, p'(t_l)`
    for :math:`p = t^q` at the span's lag times, relative to the size of
    its terms."""
    times = [span.lag_time(lag) for lag in range(span.nlags())]
    values = [t**q for t in times]
    slopes = [q * t ** (q - 1) if q > 0 else 0.0 for t in times]
    left = sum(a * v for a, v in zip(formula.alpha, values))
    right = span.deltat() * sum(b * s for b, s in zip(formula.beta, slopes))
    scale = sum(abs(a * v) for a, v in zip(formula.alpha, values)) + 1.0
    return abs(left - right) / scale


def _uneven_span(seed: int, nstates: int) -> StepSpan:
    """A step reading ``nstates`` states from 0.5, with step sizes varying
    by up to a factor of 3."""
    rng = np.random.default_rng(seed)
    sizes = rng.uniform(0.1, 0.3, size=nstates)
    return StepSpan(0.5, tuple(float(size) for size in sizes))


class TestOrderOnAnySteps:
    @pytest.mark.parametrize("make, order, explicit", METHODS, ids=METHOD_IDS)
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_exact_to_the_stated_order_and_no_further(
        self,
        make: Callable[[], MultistepFormulaProtocol],
        order: int,
        explicit: bool,
        seed: int,
    ) -> None:
        method = make()
        # The full formula reads the depth's states; the startup one fewer
        # and is one order lower.
        for nstates, stated in ((method.depth(), order), (1, order - 1)):
            span = _uneven_span(seed, nstates)
            formula = method.coefficients_at(span)
            assert len(formula.alpha) == nstates + 1
            for q in range(stated + 1):
                assert _exactness_defect(formula, span, q) < 1e-13, (nstates, q)
            assert _exactness_defect(formula, span, stated + 1) > 1e-6

    @pytest.mark.parametrize("make, order, explicit", METHODS, ids=METHOD_IDS)
    def test_kind_and_curvature(
        self,
        make: Callable[[], MultistepFormulaProtocol],
        order: int,
        explicit: bool,
    ) -> None:
        method = make()
        assert isinstance(method, MultistepFormulaProtocol)
        assert method.is_explicit() == explicit
        assert method.depth() == 2
        formula = method.coefficients_at(_uneven_span(5, 2))
        assert formula.is_explicit() == explicit
        nonzero = {lag for lag, beta in enumerate(formula.beta) if beta != 0.0}
        assert nonzero <= method.curved_lags()


def _uniform_defect(formula: MultistepCoefficients, q: int) -> float:
    """The classical order-q condition's defect at equal steps."""
    lags = [float(-lag) for lag in range(len(formula.alpha))]
    left = sum(a * lag**q for lag, a in zip(lags, formula.alpha))
    if q == 0:
        return left
    right = q * sum(b * lag ** (q - 1) for lag, b in zip(lags, formula.beta))
    return left - right


def _uniform_order(formula: MultistepCoefficients) -> int:
    q = 0
    while abs(_uniform_defect(formula, q)) < 1e-13:
        q += 1
    return q - 1


_CLASSICAL: List[
    tuple[Callable[[], MultistepFormulaProtocol], tuple[float, ...], tuple[float, ...]]
] = [
    (bdf2, (1.0, -4.0 / 3.0, 1.0 / 3.0), (2.0 / 3.0, 0.0, 0.0)),
    (adams_bashforth2, (1.0, -1.0, 0.0), (0.0, 1.5, -0.5)),
    (adams_moulton3, (1.0, -1.0, 0.0), (5.0 / 12.0, 8.0 / 12.0, -1.0 / 12.0)),
]


class TestEqualSteps:
    @pytest.mark.parametrize("make, order, explicit", METHODS, ids=METHOD_IDS)
    def test_classical_order_conditions(
        self,
        make: Callable[[], MultistepFormulaProtocol],
        order: int,
        explicit: bool,
    ) -> None:
        method = make()
        assert _uniform_order(method.coefficients_at(_EQUAL)) == order
        assert _uniform_order(method.coefficients_at(_ONE)) == order - 1

    @pytest.mark.parametrize(
        "make, alpha, beta", _CLASSICAL, ids=[m().name for m, *_ in _CLASSICAL]
    )
    def test_the_classical_tables(
        self,
        make: Callable[[], MultistepFormulaProtocol],
        alpha: tuple[float, ...],
        beta: tuple[float, ...],
    ) -> None:
        formula = make().coefficients_at(_EQUAL)
        np.testing.assert_allclose(formula.alpha, alpha, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(formula.beta, beta, rtol=0.0, atol=1e-14)

    def test_equal_sizes_give_one_formula_anywhere(self) -> None:
        """BDF4 on four equal steps of 0.001: the same formula, bit for bit,
        from t = 0 and from t = 1000 (where 0.001 n would not subtract
        back to 0.001), so every such step labels one stage operator."""
        bdf4 = BackwardDifferentiation(4, "BDF4")
        near = bdf4.coefficients_at(StepSpan(0.0, (0.001,) * 4))
        far = bdf4.coefficients_at(StepSpan(1000.0, (0.001,) * 4))
        assert near == far
        np.testing.assert_allclose(
            near.alpha, (1.0, -48 / 25, 36 / 25, -16 / 25, 3 / 25), atol=1e-14
        )


def _fixed_bdf2() -> MultistepMethod:
    return MultistepMethod(
        coefficients=MultistepCoefficients(
            alpha=(1.0, -4.0 / 3.0, 1.0 / 3.0), beta=(2.0 / 3.0, 0.0, 0.0)
        ),
        startup=(MultistepCoefficients(alpha=(1.0, -1.0), beta=(1.0, 0.0)),),
        order=2,
        name="fixed BDF2",
    )


class TestFixedTable:
    def test_picks_the_formula_for_the_states_read(self) -> None:
        method = _fixed_bdf2()
        assert isinstance(method, MultistepFormulaProtocol)
        assert method.at(1) is method.startup[0]
        assert method.at(2) is method.coefficients
        assert method.coefficients_at(_EQUAL) is method.coefficients
        assert method.curved_lags() == frozenset({0})
        with pytest.raises(ValueError, match="between 1 and 2"):
            method.at(3)

    def test_refuses_steps_not_exactly_equal(self) -> None:
        method = _fixed_bdf2()
        with pytest.raises(ValueError, match="fixed coefficients"):
            method.coefficients_at(StepSpan(0.05, (0.15, 0.1)))
        # Sizes recovered from the times 0.1, 0.2, 0.3 are 0.1 and
        # 0.09999999999999998: not equal steps, so refused.
        with pytest.raises(ValueError, match="fixed coefficients"):
            method.coefficients_at(StepSpan.from_times((0.1, 0.2, 0.3)))
        # One state spans one step, of any size.
        method.coefficients_at(StepSpan.first(0.2, 0.5))


class TestRefusals:
    def test_coefficients(self) -> None:
        with pytest.raises(ValueError, match="one entry per lag"):
            MultistepCoefficients(alpha=(1.0, -1.0), beta=(1.0,))
        with pytest.raises(ValueError, match="at least one stored state"):
            MultistepCoefficients(alpha=(1.0,), beta=(1.0,))
        with pytest.raises(ValueError, match="alpha_0 must be nonzero"):
            MultistepCoefficients(alpha=(0.0, 1.0), beta=(1.0, 0.0))

    def test_fixed_table(self) -> None:
        table = _fixed_bdf2().coefficients
        with pytest.raises(ValueError, match="needs 1 startup"):
            MultistepMethod(table, (), 2, "no startup")
        explicit = MultistepCoefficients(alpha=(1.0, -1.0), beta=(0.0, 1.0))
        with pytest.raises(ValueError, match="both explicit or both implicit"):
            MultistepMethod(table, (explicit,), 2, "mixed")

    @pytest.mark.parametrize(
        "family", [BackwardDifferentiation, AdamsBashforth, AdamsMoulton]
    )
    def test_variable_step(
        self,
        family: Callable[[int, str], MultistepFormulaProtocol],
    ) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            family(0, "none")
        method = family(2, "two")
        with pytest.raises(ValueError, match="between 1 and 2"):
            method.coefficients_at(StepSpan(0.1, (0.1, 0.1, 0.1)))
