"""ZeroOperator on every stack, and the ``zero`` operator property."""

import pytest

from pyapprox.util.linalg.operators import (
    OperatorProperties,
)
from tests._helpers.linalg.operator_suites.zero import ZeroOperatorSuite


class TestZeroOperator(ZeroOperatorSuite):
    pass


class TestZeroProperty:
    def test_excludes_identity(self) -> None:
        assert OperatorProperties(zero=True).zero
        with pytest.raises(ValueError, match="excludes identity"):
            OperatorProperties(
                symmetric=True, spd=True, diagonal=True, identity=True, zero=True
            )
