"""The ``case`` fixture: one parametrization per linear-algebra stack."""

import pytest

from tests._helpers.linalg.stacks import STACK_NAMES, StackCase, stack_case


@pytest.fixture(params=STACK_NAMES)
def case(request: pytest.FixtureRequest) -> StackCase:
    return stack_case(request)
