"""A residual with rows pinned to data, on every linear-algebra stack.

The bodies are the stack-generic suites of
``tests._helpers.implicitfunction_suites.pinned_residual``, run on the
``case`` fixture (CSR, dense NumPy, dense Torch and the opaque mock).
"""

from tests._helpers.implicitfunction_suites.pinned_residual import (
    LevelsAndZerosSuite,
    RefusalsSuite,
    TheDerivativesSuite,
    TheValueSuite,
)


class TestTheValue(TheValueSuite):
    pass


class TestTheDerivatives(TheDerivativesSuite):
    pass


class TestLevelsAndZeros(LevelsAndZerosSuite):
    pass


class TestRefusals(RefusalsSuite):
    pass
