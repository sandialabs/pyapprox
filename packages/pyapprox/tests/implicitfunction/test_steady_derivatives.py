"""The adjoint gradient and HVP of a steady functional, on every stack.

The bodies are the stack-generic suites of
``tests._helpers.implicitfunction_suites.steady_derivatives``, run on the
``case`` fixture (CSR, dense NumPy, dense Torch and the opaque mock).
"""

from tests._helpers.implicitfunction_suites.steady_derivatives import (
    LevelsSuite,
    RefusalsSuite,
    SetupsSuite,
    TheGradientSuite,
    TheHessianSuite,
    TheTangentSuite,
)


class TestTheGradient(TheGradientSuite):
    pass


class TestTheHessian(TheHessianSuite):
    pass


class TestTheTangent(TheTangentSuite):
    pass


class TestSetups(SetupsSuite):
    pass


class TestLevels(LevelsSuite):
    pass


class TestRefusals(RefusalsSuite):
    pass
