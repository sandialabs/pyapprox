"""A linear observation of a residual on a steady solution, on array
stacks.

The bodies are the suites of
``tests._helpers.implicitfunction_suites.observed_residual``, run on the
``array_case`` fixture (CSR, dense NumPy, dense Torch).
"""

from tests._helpers.implicitfunction_suites.observed_residual import (
    RefusalsSuite,
    TheBundleSuite,
    TheDerivativesSuite,
    TheValueSuite,
)


class TestTheValue(TheValueSuite):
    pass


class TestTheDerivatives(TheDerivativesSuite):
    pass


class TestTheBundle(TheBundleSuite):
    pass


class TestRefusals(RefusalsSuite):
    pass
