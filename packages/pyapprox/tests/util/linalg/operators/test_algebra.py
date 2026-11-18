"""Conformance tests for the operator combinators on every stack."""

import pytest

from pyapprox.util.linalg.operators import (
    OperatorProperties,
    product_properties,
    scaled_sum_properties,
)
from tests._helpers.linalg.operator_suites.algebra import (
    AsMatrixFreeSuite,
    ProductSuite,
    ScaledSumSuite,
    TransposeCapabilitySuite,
    TransposedSuite,
)

_NONE = OperatorProperties()
_SYM = OperatorProperties(symmetric=True)
_SPD = OperatorProperties(symmetric=True, spd=True)
_DIAG = OperatorProperties(symmetric=True, diagonal=True)
_DIAG_SPD = OperatorProperties(symmetric=True, spd=True, diagonal=True)
_IDENT = OperatorProperties(symmetric=True, spd=True, diagonal=True, identity=True)
_FWD = OperatorProperties(transposable=False)


class TestScaledSum(ScaledSumSuite):
    pass


class TestTransposed(TransposedSuite):
    pass


class TestTransposeCapability(TransposeCapabilitySuite):
    pass


class TestProduct(ProductSuite):
    pass


class TestAsMatrixFree(AsMatrixFreeSuite):
    pass


class TestPropertyPropagation:
    @pytest.mark.parametrize(
        "terms, expected",
        [
            ([(1.0, _SYM), (-2.0, _SYM)], _SYM),
            ([(1.0, _SPD), (2.0, _SPD)], _SPD),
            ([(1.0, _SPD), (-2.0, _SPD)], _SYM),
            ([(1.0, _SPD), (1.0, _NONE)], _NONE),
            ([(1.0, _DIAG_SPD), (3.0, _IDENT)], _DIAG_SPD),
            ([(1.0, _IDENT)], _DIAG_SPD),
            ([(1.0, _DIAG), (1.0, _SPD)], _SYM),
            ([(1.0, _SYM), (2.0, _FWD)], _FWD),
            ([(1.0, _IDENT), (1.0, _FWD)], _FWD),
        ],
    )
    def test_scaled_sum(
        self,
        terms: list[tuple[float, OperatorProperties]],
        expected: OperatorProperties,
    ) -> None:
        """A sum is never declared the identity, even of one identity."""
        assert scaled_sum_properties(terms) == expected

    @pytest.mark.parametrize(
        "factors, expected",
        [
            ([_SPD, _SPD], _NONE),
            ([_SYM, _DIAG], _NONE),
            ([_DIAG, _DIAG_SPD], _DIAG),
            ([_DIAG_SPD, _IDENT], _DIAG_SPD),
            ([_IDENT, _IDENT], _IDENT),
            ([_SPD, _FWD], _FWD),
            ([_DIAG, _FWD], _FWD),
        ],
    )
    def test_product(
        self, factors: list[OperatorProperties], expected: OperatorProperties
    ) -> None:
        """A product of symmetric matrices is not symmetric in general."""
        assert product_properties(factors) == expected
