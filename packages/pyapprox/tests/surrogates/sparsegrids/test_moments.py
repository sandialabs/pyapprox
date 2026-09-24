"""Tests for sparse grid moments classes.

Each class is tested against its own definition, computed independently
in the test:

- ``QuadratureMoments``: Q_K[f^2] - (Q_K f)^2, where Q_K is the signed
  Smolyak rule.
- ``PCEMoments``: the exact moments of I_K f, checked against analytic
  values on grids that resolve the target and against the orthonormal
  PCE coefficients otherwise.

The variance definitions are not tested against each other. They agree
whenever the grid resolves the target and disagree otherwise, so any
such test would assert an accident of the chosen case. The means are
compared, since the mean is definition-independent.
"""

from typing import List, Tuple

import pytest
from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.expansions import pce_statistics
from pyapprox.surrogates.affine.indices import (
    ClenshawCurtisGrowthRule,
    LinearGrowthRule,
)
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.sparsegrids import create_basis_factories
from pyapprox.surrogates.sparsegrids.combination_surrogate import (
    CombinationSurrogate,
)
from pyapprox.surrogates.sparsegrids.converters.pce import (
    SparseGridToPCEConverter,
)
from pyapprox.surrogates.sparsegrids.isotropic_fitter import (
    IsotropicSparseGridFitter,
)
from pyapprox.surrogates.sparsegrids.smolyak import (
    compute_smolyak_coefficients,
)
from pyapprox.surrogates.sparsegrids.statistics.moments import (
    PCEMoments,
    QuadratureMoments,
)
from pyapprox.surrogates.sparsegrids.statistics.subspace_moments import (
    subspace_mean,
    subspace_raw_moment,
)
from pyapprox.surrogates.sparsegrids.subspace_factory import (
    TensorProductSubspaceFactory,
)

# f = x^2 + y^2 on [0,1]^2 with Gauss rules over this index set is the
# case the plan records: the rule's variance is 7.375/45 while the true
# variance is 8/45. The nonzero coefficients are -1, +1, +1.
_REFERENCE_KEYS: List[Tuple[int, int]] = [
    (0, 0),
    (1, 0),
    (2, 0),
    (0, 1),
    (0, 2),
]


def _make_factory(bkd, nvars, basis_type, lower=0.0, upper=1.0):
    marginals = [UniformMarginal(lower, upper, bkd) for _ in range(nvars)]
    factories = create_basis_factories(marginals, bkd, basis_type)
    growth = (
        ClenshawCurtisGrowthRule()
        if basis_type == "clenshaw_curtis"
        else LinearGrowthRule(scale=1, shift=1)
    )
    return TensorProductSubspaceFactory(bkd, factories, growth), marginals


def _surrogate_from_keys(bkd, keys, target_fn, nvars=2):
    """Build a combination surrogate over an explicit index set."""
    tp_factory, marginals = _make_factory(bkd, nvars, "gauss")
    subspaces = []
    for key in keys:
        idx = bkd.asarray(list(key), dtype=bkd.int64_dtype())
        subspace = tp_factory(idx)
        subspace.set_values(target_fn(subspace.get_samples()))
        subspaces.append(subspace)
    indices = bkd.asarray(
        [[k[d] for k in keys] for d in range(nvars)],
        dtype=bkd.int64_dtype(),
    )
    coefs = compute_smolyak_coefficients(indices, bkd)
    surrogate = CombinationSurrogate(
        bkd, nvars, subspaces, coefs, 1, indices=indices
    )
    return surrogate, marginals


def _isotropic(bkd, nvars, level, basis_type, target_fn, lower=0.0, upper=1.0):
    tp_factory, marginals = _make_factory(
        bkd, nvars, basis_type, lower, upper
    )
    fitter = IsotropicSparseGridFitter(bkd, tp_factory, level)
    result = fitter.fit(target_fn(fitter.get_samples()))
    return result.surrogate, marginals


def _sum_of_squares(bkd):
    """f(x, y) = x^2 + y^2."""

    def fun(samples):
        return bkd.reshape(samples[0] ** 2 + samples[1] ** 2, (1, -1))

    return fun


def _combine(bkd, surrogate, statistic):
    """sum_k c_k statistic(subspace_k), computed in the test."""
    coefs = surrogate.coefficients()
    total = bkd.zeros((surrogate.nqoi(),))
    for j, subspace in enumerate(surrogate.subspaces()):
        total = total + coefs[j] * statistic(subspace)
    return total


class TestQuadratureMoments:
    """mean = sum c_k Q_k f; variance = Q_K[f^2] - (Q_K f)^2."""

    @pytest.mark.parametrize("basis_type", ["gauss", "leja"])
    @pytest.mark.parametrize("level", [1, 2, 3])
    def test_matches_its_definition(
        self, bkd, basis_type: str, level: int
    ) -> None:
        """Against the signed combination of raw moments."""
        surrogate, _ = _isotropic(
            bkd, 2, level, basis_type, _sum_of_squares(bkd)
        )
        moments = QuadratureMoments(surrogate)
        expected_mean = _combine(bkd, surrogate, subspace_mean)
        expected_second = _combine(
            bkd, surrogate, lambda s: subspace_raw_moment(s, 2)
        )
        bkd.assert_allclose(moments.mean(), expected_mean, rtol=1e-12)
        bkd.assert_allclose(
            moments.second_moment(), expected_second, rtol=1e-12
        )
        bkd.assert_allclose(
            moments.variance(),
            expected_second - expected_mean**2,
            rtol=1e-12,
        )

    def test_reference_case(self, bkd) -> None:
        """The signed rule gives 7.375/45 where the truth is 8/45."""
        surrogate, _ = _surrogate_from_keys(
            bkd, _REFERENCE_KEYS, _sum_of_squares(bkd)
        )
        moments = QuadratureMoments(surrogate)
        bkd.assert_allclose(
            moments.variance(), bkd.asarray([7.375 / 45.0]), rtol=1e-10
        )
        bkd.assert_allclose(
            moments.mean(), bkd.asarray([2.0 / 3.0]), rtol=1e-10
        )

    def test_variance_of_sum_on_resolved_grid(self, bkd) -> None:
        """Var[x + y] = 2/3 on [-1,1]^2."""

        def fun(samples):
            return bkd.reshape(samples[0] + samples[1], (1, -1))

        surrogate, _ = _isotropic(bkd, 2, 2, "gauss", fun, -1.0, 1.0)
        bkd.assert_allclose(
            QuadratureMoments(surrogate).variance(),
            bkd.asarray([2.0 / 3.0]),
            rtol=1e-10,
        )

    def test_variance_of_product_on_resolved_grid(self, bkd) -> None:
        """Var[xy] = 1/9 on [-1,1]^2."""

        def fun(samples):
            return bkd.reshape(samples[0] * samples[1], (1, -1))

        surrogate, _ = _isotropic(bkd, 2, 2, "gauss", fun, -1.0, 1.0)
        bkd.assert_allclose(
            QuadratureMoments(surrogate).variance(),
            bkd.asarray([1.0 / 9.0]),
            rtol=1e-10,
        )

    def test_memoizes(self, bkd) -> None:
        surrogate, _ = _surrogate_from_keys(
            bkd, _REFERENCE_KEYS, _sum_of_squares(bkd)
        )
        moments = QuadratureMoments(surrogate)
        assert moments.mean() is moments.mean()
        assert moments.second_moment() is moments.second_moment()

    def test_rejects_non_surrogate(self, bkd) -> None:
        with pytest.raises(TypeError, match="CombinationSurrogate"):
            QuadratureMoments("not a surrogate")


class TestPCEMoments:
    """Exact moments of I_K f."""

    @pytest.mark.parametrize("basis_type", ["gauss", "leja"])
    @pytest.mark.parametrize("level", [2, 3])
    def test_matches_its_definition(
        self, bkd, basis_type: str, level: int
    ) -> None:
        """Against the orthonormal PCE coefficients of the same grid."""
        surrogate, marginals = _isotropic(
            bkd, 2, level, basis_type, _sum_of_squares(bkd)
        )
        pce = SparseGridToPCEConverter(
            bkd, create_bases_1d(marginals, bkd)
        ).convert(surrogate)
        moments = PCEMoments(surrogate, create_bases_1d(marginals, bkd))
        bkd.assert_allclose(
            moments.mean(), pce_statistics.mean(pce), rtol=1e-10
        )
        bkd.assert_allclose(
            moments.variance(), pce_statistics.variance(pce), rtol=1e-10
        )

    def test_reference_case_is_the_true_variance(self, bkd) -> None:
        """The grid resolves x^2 + y^2, so the exact value is 8/45."""
        surrogate, marginals = _surrogate_from_keys(
            bkd, _REFERENCE_KEYS, _sum_of_squares(bkd)
        )
        moments = PCEMoments(surrogate, create_bases_1d(marginals, bkd))
        bkd.assert_allclose(
            moments.variance(), bkd.asarray([8.0 / 45.0]), rtol=1e-10
        )
        bkd.assert_allclose(
            moments.mean(), bkd.asarray([2.0 / 3.0]), rtol=1e-10
        )

    def test_variance_of_product_on_resolved_grid(self, bkd) -> None:
        """Var[xy] = 1/9 on [-1,1]^2."""

        def fun(samples):
            return bkd.reshape(samples[0] * samples[1], (1, -1))

        surrogate, marginals = _isotropic(
            bkd, 2, 2, "gauss", fun, -1.0, 1.0
        )
        moments = PCEMoments(surrogate, create_bases_1d(marginals, bkd))
        bkd.assert_allclose(
            moments.variance(), bkd.asarray([1.0 / 9.0]), rtol=1e-10
        )

    def test_converts_once(self, bkd) -> None:
        surrogate, marginals = _surrogate_from_keys(
            bkd, _REFERENCE_KEYS, _sum_of_squares(bkd)
        )
        moments = PCEMoments(surrogate, create_bases_1d(marginals, bkd))
        assert moments.pce() is moments.pce()

    def test_rejects_non_surrogate(self, bkd) -> None:
        with pytest.raises(TypeError, match="CombinationSurrogate"):
            PCEMoments("not a surrogate", [])


class TestMeansAgree:
    """The mean is definition-independent, unlike the variance."""

    @pytest.mark.parametrize(
        "basis_type", ["gauss", "leja", "clenshaw_curtis"]
    )
    def test_both_means_agree(self, bkd, basis_type: str) -> None:
        surrogate, marginals = _isotropic(
            bkd, 2, 2, basis_type, _sum_of_squares(bkd)
        )
        quadrature = QuadratureMoments(surrogate).mean()
        exact = PCEMoments(
            surrogate, create_bases_1d(marginals, bkd)
        ).mean()
        bkd.assert_allclose(quadrature, exact, rtol=1e-12)


class TestAutograd:
    """Moments stay differentiable with respect to subspace values."""

    def test_mean_gradient_is_the_quadrature_weights(self, torch_bkd) -> None:
        """The mean is linear in the values, so d(mean)/dv = w."""
        import torch

        tp_factory, _ = _make_factory(torch_bkd, 2, "gauss")
        idx = torch_bkd.asarray([1, 1], dtype=torch_bkd.int64_dtype())
        subspace = tp_factory(idx)
        values = torch.ones(
            (1, subspace.nsamples()), dtype=torch.double, requires_grad=True
        )
        subspace.set_values(values)
        indices = torch_bkd.asarray(
            [[1], [1]], dtype=torch_bkd.int64_dtype()
        )
        surrogate = CombinationSurrogate(
            torch_bkd,
            2,
            [subspace],
            torch_bkd.asarray([1.0]),
            1,
            indices=indices,
        )
        QuadratureMoments(surrogate).mean().sum().backward()
        assert values.grad is not None
        torch_bkd.assert_allclose(
            values.grad[0], subspace.get_quadrature_weights(), rtol=1e-12
        )
