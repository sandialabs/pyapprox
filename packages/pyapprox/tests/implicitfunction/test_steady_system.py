"""The steady problem solved by Newton, on every linear-algebra stack.

The bodies are the stack-generic suites of
``tests._helpers.implicitfunction_suites.steady_system``, run on the
``case`` fixture (CSR, dense NumPy, dense Torch and the opaque mock).
"""

from tests._helpers.implicitfunction_suites.steady_system import (
    AffineSolveSuite,
    NonlinearSolveSuite,
    RefusalsSuite,
    SharedStrategySuite,
)


class TestANonlinearSolve(NonlinearSolveSuite):
    pass


class TestAnAffineSolve(AffineSolveSuite):
    pass


class TestASharedStrategy(SharedStrategySuite):
    pass


class TestRefusals(RefusalsSuite):
    pass
