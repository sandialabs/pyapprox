"""A steady functional as a function of the parameters, on array stacks.

The bodies are the suites of
``tests._helpers.implicitfunction_suites.steady_function``, run on the
``array_case`` fixture (CSR, dense NumPy, dense Torch):
``SteadyImplicitFunction`` takes states as backend arrays.
"""

from tests._helpers.implicitfunction_suites.steady_function import (
    DerivativesSuite,
    RefusalsSuite,
    SetupsSuite,
    ValuesSuite,
)


class TestValues(ValuesSuite):
    pass


class TestDerivatives(DerivativesSuite):
    pass


class TestSetups(SetupsSuite):
    pass


class TestRefusals(RefusalsSuite):
    pass
