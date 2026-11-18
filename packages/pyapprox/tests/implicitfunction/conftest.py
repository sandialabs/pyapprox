"""The ``case`` fixture, one parametrization per linear-algebra stack, and
``array_case``, the stacks whose vectors are backend arrays."""

import pytest

from tests._helpers.linalg.stacks import (
    ARRAY_STACK_NAMES,
    STACK_NAMES,
    StackCase,
    stack_case,
)


@pytest.fixture(params=STACK_NAMES)
def case(request: pytest.FixtureRequest) -> StackCase:
    return stack_case(request)


@pytest.fixture(params=ARRAY_STACK_NAMES)
def array_case(request: pytest.FixtureRequest) -> StackCase:
    return stack_case(request)
