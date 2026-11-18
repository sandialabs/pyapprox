"""The strategy checker on steady requests, on every linear-algebra stack.

The bodies are the stack-generic suite ``TheCheckerSuite`` of
``tests._helpers.implicitfunction_suites.setups``, run on the ``case``
fixture (CSR, dense NumPy, dense Torch and the opaque mock).
"""

from tests._helpers.implicitfunction_suites.setups import TheCheckerSuite


class TestTheChecker(TheCheckerSuite):
    pass
