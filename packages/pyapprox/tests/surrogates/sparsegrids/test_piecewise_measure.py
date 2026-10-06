"""Piecewise sparse grids integrate under the input measure.

The 1D weights of a piecewise basis are probability weights under its
marginal, so a sparse grid's quadrature gives expectations. For
``f = 2x + 3y + 1`` both bases reproduce ``f``; the quadratic one also
integrates ``f^2`` exactly under the measure, so its variance is exact.
Lebesgue weights would give a mean of 4 on [-1, 1]^2 and ignore a
Gaussian density.
"""

from typing import Callable, List

import pytest
from scipy import stats

from pyapprox.probability import GaussianMarginal, UniformMarginal
from pyapprox.probability.protocols.distribution import MarginalProtocol
from pyapprox.surrogates.affine.indices import ClenshawCurtisGrowthRule
from pyapprox.surrogates.affine.univariate.piecewisepoly import (
    PiecewiseLinear,
    PiecewisePolynomialProtocol,
    PiecewiseQuadratic,
)
from pyapprox.surrogates.sparsegrids.basis_factory import (
    BasisFactoryProtocol,
    PiecewiseFactory,
    get_bounds_from_marginal,
)
from pyapprox.surrogates.sparsegrids.combination_surrogate import (
    CombinationSurrogate,
)
from pyapprox.surrogates.sparsegrids.isotropic_fitter import (
    IsotropicSparseGridFitter,
)
from pyapprox.surrogates.sparsegrids.statistics.moments import QuadratureMoments
from pyapprox.surrogates.sparsegrids.subspace_factory import (
    TensorProductSubspaceFactory,
)
from pyapprox.util.backends.protocols import Array, Backend

_BasisClass = Callable[[Array, Backend[Array]], PiecewisePolynomialProtocol[Array]]


def _fit(
    bkd: Backend[Array],
    marginals: List[MarginalProtocol[Array]],
    basis_class: _BasisClass[Array],
) -> CombinationSurrogate[Array]:
    """Fit ``f = 2x + 3y + 1`` on a level-4 isotropic sparse grid."""
    factories: List[BasisFactoryProtocol[Array]] = [
        PiecewiseFactory(marginal, bkd, basis_class) for marginal in marginals
    ]
    subspaces = TensorProductSubspaceFactory(bkd, factories, ClenshawCurtisGrowthRule())
    fitter = IsotropicSparseGridFitter(bkd, subspaces, 4)
    samples = fitter.get_samples()
    assert not isinstance(samples, dict)
    values = bkd.reshape(2 * samples[0] + 3 * samples[1] + 1, (1, -1))
    surrogate = fitter.fit(values).surrogate
    assert isinstance(surrogate, CombinationSurrogate)
    return surrogate


def _truncated_variance(marginal: MarginalProtocol[Array]) -> float:
    """Variance of N(0, 1) on the factory's truncated interval."""
    lower, upper = get_bounds_from_marginal(marginal, 1e-6)
    return float(stats.truncnorm(lower, upper).var())


class TestPiecewiseSparseGridMoments:
    @pytest.mark.parametrize("basis_class", [PiecewiseLinear, PiecewiseQuadratic])
    def test_uniform_mean(
        self, bkd: Backend[Array], basis_class: _BasisClass[Array]
    ) -> None:
        marginals: List[MarginalProtocol[Array]] = [
            UniformMarginal(-1.0, 1.0, bkd) for _ in range(2)
        ]
        moments = QuadratureMoments(_fit(bkd, marginals, basis_class))
        bkd.assert_allclose(moments.mean(), bkd.asarray([1.0]), rtol=1e-12)

    def test_uniform_variance(self, bkd: Backend[Array]) -> None:
        marginals: List[MarginalProtocol[Array]] = [
            UniformMarginal(-1.0, 1.0, bkd) for _ in range(2)
        ]
        moments = QuadratureMoments(_fit(bkd, marginals, PiecewiseQuadratic))
        # 4 Var[x] + 9 Var[y] with Var = 1/3 on [-1, 1].
        bkd.assert_allclose(moments.variance(), bkd.asarray([13.0 / 3.0]), rtol=1e-12)

    def test_gaussian_mean_and_variance(self, bkd: Backend[Array]) -> None:
        marginals: List[MarginalProtocol[Array]] = [
            GaussianMarginal(0.0, 1.0, bkd) for _ in range(2)
        ]
        moments = QuadratureMoments(_fit(bkd, marginals, PiecewiseQuadratic))
        bkd.assert_allclose(moments.mean(), bkd.asarray([1.0]), atol=1e-10)
        bkd.assert_allclose(
            moments.variance(),
            bkd.asarray([13.0 * _truncated_variance(marginals[0])]),
            rtol=1e-9,
        )

    def test_mixed_marginals(self, bkd: Backend[Array]) -> None:
        marginals: List[MarginalProtocol[Array]] = [
            UniformMarginal(-1.0, 1.0, bkd),
            GaussianMarginal(0.0, 1.0, bkd),
        ]
        moments = QuadratureMoments(_fit(bkd, marginals, PiecewiseQuadratic))
        bkd.assert_allclose(moments.mean(), bkd.asarray([1.0]), atol=1e-10)
        bkd.assert_allclose(
            moments.variance(),
            bkd.asarray([4.0 / 3.0 + 9.0 * _truncated_variance(marginals[1])]),
            rtol=1e-9,
        )
