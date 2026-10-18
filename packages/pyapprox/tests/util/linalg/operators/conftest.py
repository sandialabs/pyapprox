"""The ``case`` fixture: one parametrization per linear-algebra stack."""

import pytest

from tests._helpers.operator_stacks import (
    STACK_NAMES,
    StackCase,
    csr_case,
    dense_case,
    mock_case,
)


@pytest.fixture(params=STACK_NAMES)
def case(request: pytest.FixtureRequest) -> StackCase:
    if request.param == "csr":
        return csr_case()
    if request.param == "mock":
        return mock_case()
    fixture = "numpy_bkd" if request.param == "dense-numpy" else "torch_bkd"
    return dense_case(request.getfixturevalue(fixture))
