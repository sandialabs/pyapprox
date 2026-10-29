r"""The multistep coefficients: order conditions and validation.

Every formula of BDF2, Adams-Bashforth 2 and Adams-Moulton 3, startup
included, satisfies the order conditions
:math:`\sum_l \alpha_l (-l)^q = q \sum_l \beta_l (-l)^{q-1}` for
:math:`q = 0, \dots, p` (with :math:`0^0 = 1`) to its stated order
(the startup formula one order lower), and fails them at the next
order, so the stated order is exact.
"""

from typing import Callable

import pytest

from pyapprox.ode.multistep_methods import MultistepCoefficients, MultistepMethod
from tests._helpers.ode.multistep_cases import METHOD_IDS, METHODS


def _defect(formula: MultistepCoefficients, q: int) -> float:
    """The order-:math:`q` condition's defect."""
    lags = [float(-lag) for lag in range(len(formula.alpha))]
    left = sum(a * lag**q for lag, a in zip(lags, formula.alpha))
    if q == 0:
        return left
    right = q * sum(b * lag ** (q - 1) for lag, b in zip(lags, formula.beta))
    return left - right


def _order(formula: MultistepCoefficients) -> int:
    """The highest order whose conditions all hold."""
    q = 0
    while abs(_defect(formula, q)) < 1e-14:
        q += 1
    return q - 1


class TestCoefficients:
    @pytest.mark.parametrize("make, order, explicit", METHODS, ids=METHOD_IDS)
    def test_orders(
        self, make: Callable[[], MultistepMethod], order: int, explicit: bool
    ) -> None:
        method = make()
        assert _order(method.coefficients) == order
        for formula in method.startup:
            assert _order(formula) == order - 1
        assert method.is_explicit() == explicit
        assert method.depth() == 2

    def test_at_picks_the_formula_for_the_states_read(self) -> None:
        method = METHODS[0][0]()
        assert method.at(1) is method.startup[0]
        assert method.at(2) is method.coefficients
        with pytest.raises(ValueError, match="between 1 and 2"):
            method.at(3)

    def test_refusals(self) -> None:
        with pytest.raises(ValueError, match="one entry per lag"):
            MultistepCoefficients(alpha=(1.0, -1.0), beta=(1.0,))
        with pytest.raises(ValueError, match="at least one stored state"):
            MultistepCoefficients(alpha=(1.0,), beta=(1.0,))
        with pytest.raises(ValueError, match="alpha_0 must be nonzero"):
            MultistepCoefficients(alpha=(0.0, 1.0), beta=(1.0, 0.0))
        bdf2 = METHODS[0][0]()
        with pytest.raises(ValueError, match="needs 1 startup"):
            MultistepMethod(bdf2.coefficients, (), 2, "no startup")
        explicit = MultistepCoefficients(alpha=(1.0, -1.0), beta=(0.0, 1.0))
        with pytest.raises(ValueError, match="both explicit or both implicit"):
            MultistepMethod(bdf2.coefficients, (explicit,), 2, "mixed")
