"""Setup strategies on steady requests, on every linear-algebra stack.

The bodies are the stack-generic suites of
``tests._helpers.implicitfunction_suites.setups``, run on the ``case``
fixture (CSR, dense NumPy, dense Torch and the opaque mock).
"""

from tests._helpers.implicitfunction_suites.setups import (
    ExtensibilitySuite,
    KeysSuite,
    StrategiesSuite,
    TheBuilderSuite,
)


class TestKeys(KeysSuite):
    pass


class TestTheBuilder(TheBuilderSuite):
    pass


class TestStrategies(StrategiesSuite):
    pass


class TestExtensibility(ExtensibilitySuite):
    pass
