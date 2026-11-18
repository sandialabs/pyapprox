"""The observation, misfit and Tikhonov terms on a steady solution, on
array stacks.

The bodies are the suites of
``tests._helpers.implicitfunction_suites.steady_observation_terms``, run
on the ``array_case`` fixture (CSR, dense NumPy, dense Torch).
"""

from tests._helpers.implicitfunction_suites.steady_observation_terms import (
    RefusalsSuite,
    TheMisfitAndTikhonovTermsSuite,
    TheObservedValueSuite,
)


class TestTheMisfitAndTikhonovTerms(TheMisfitAndTikhonovTermsSuite):
    pass


class TestTheObservedValue(TheObservedValueSuite):
    pass


class TestRefusals(RefusalsSuite):
    pass
