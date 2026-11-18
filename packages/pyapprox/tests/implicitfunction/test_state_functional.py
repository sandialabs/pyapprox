"""A functional of one state, on every linear-algebra stack.

The bodies are the stack-generic suites of
``tests._helpers.implicitfunction_suites.state_functional``, run on the
``case`` fixture (CSR, dense NumPy, dense Torch and the opaque mock).
"""

from tests._helpers.implicitfunction_suites.state_functional import (
    TheStackSuite,
    TheValueSuite,
)


class TestTheValue(TheValueSuite):
    pass


class TestTheStack(TheStackSuite):
    pass
