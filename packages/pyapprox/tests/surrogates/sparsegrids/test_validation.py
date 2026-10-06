"""Tests for validate_piecewise_growth_compatibility.

The check builds each factory's basis at the growth rule's node counts, so
the basis itself decides which counts it accepts.
"""

from typing import Callable

import pytest

from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.indices import (
    ClenshawCurtisGrowthRule,
    CubicNestedGrowthRule,
    LinearGrowthRule,
)
from pyapprox.surrogates.affine.protocols import IndexGrowthRuleProtocol
from pyapprox.surrogates.affine.univariate.piecewisepoly import (
    PiecewiseCubic,
    PiecewiseLinear,
    PiecewisePolynomialProtocol,
    PiecewiseQuadratic,
)
from pyapprox.surrogates.sparsegrids.basis_factory import (
    GaussLagrangeFactory,
    PiecewiseFactory,
)
from pyapprox.surrogates.sparsegrids.validation import (
    validate_piecewise_growth_compatibility,
)
from pyapprox.util.backends.protocols import Array, Backend

_BasisClass = Callable[[Array, Backend[Array]], PiecewisePolynomialProtocol[Array]]


class TestPiecewiseGrowthCompatibility:
    @pytest.mark.parametrize(
        "basis_class, rule",
        [
            (PiecewiseLinear, LinearGrowthRule(scale=1, shift=1)),
            (PiecewiseQuadratic, ClenshawCurtisGrowthRule()),
            (PiecewiseCubic, CubicNestedGrowthRule()),
        ],
    )
    def test_compatible_rules_pass(
        self,
        bkd: Backend[Array],
        basis_class: _BasisClass[Array],
        rule: IndexGrowthRuleProtocol,
    ) -> None:
        factory = PiecewiseFactory(UniformMarginal(-1.0, 1.0, bkd), bkd, basis_class)
        validate_piecewise_growth_compatibility([factory], rule)

    @pytest.mark.parametrize(
        "basis_class, rule",
        [
            # n(l) = l + 1 gives an even count at level 1
            (PiecewiseQuadratic, LinearGrowthRule(scale=1, shift=1)),
            # 3, 5, 9, ... are not 3k + 1
            (PiecewiseCubic, ClenshawCurtisGrowthRule()),
        ],
    )
    def test_incompatible_rules_raise(
        self,
        bkd: Backend[Array],
        basis_class: _BasisClass[Array],
        rule: IndexGrowthRuleProtocol,
    ) -> None:
        factory = PiecewiseFactory(UniformMarginal(-1.0, 1.0, bkd), bkd, basis_class)
        with pytest.raises(ValueError, match="dimension 0 rejects"):
            validate_piecewise_growth_compatibility([factory], rule)

    def test_reports_the_failing_dimension(self, bkd: Backend[Array]) -> None:
        marginal = UniformMarginal(-1.0, 1.0, bkd)
        factories = [
            GaussLagrangeFactory(marginal, bkd),
            PiecewiseFactory(marginal, bkd, PiecewiseQuadratic),
        ]
        with pytest.raises(ValueError, match="dimension 1 rejects"):
            validate_piecewise_growth_compatibility(
                factories, LinearGrowthRule(scale=1, shift=1)
            )
