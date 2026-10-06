"""Tests for PCEVarianceIndicator.

The indicator reports |Delta m| + |Delta sigma| for the variance
PCEMoments defines, so the checks are against that class applied to the
selected set and to the selected set plus the candidate. Building those
two surrogates and converting each is exactly the work the indicator
avoids, which makes it a genuine cross-check rather than a restatement.
"""

from typing import List, Tuple

import pytest

from pyapprox.interface.functions.fromcallable.function import (
    FunctionFromCallable,
)
from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.indices import (
    LinearGrowthRule,
    MaxLevelCriteria,
)
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.sparsegrids import create_basis_factories
from pyapprox.surrogates.sparsegrids.adaptive_fitter import (
    SingleFidelityAdaptiveSparseGridFitter,
)
from pyapprox.surrogates.sparsegrids.candidate_info import (
    Candidate,
    SmolyakSelection,
)
from pyapprox.surrogates.sparsegrids.combination_surrogate import (
    CombinationSurrogate,
)
from pyapprox.surrogates.sparsegrids.error_indicators import (
    ErrorIndicatorProtocol,
)
from pyapprox.surrogates.sparsegrids.pce_variance_indicator import (
    PCEVarianceIndicator,
)
from pyapprox.surrogates.sparsegrids.sample_tracker import SampleTracker
from pyapprox.surrogates.sparsegrids.smolyak import (
    IncrementalSmolyakCoefficients,
    compute_smolyak_coefficients,
)
from pyapprox.surrogates.sparsegrids.statistics.moments import PCEMoments
from pyapprox.surrogates.sparsegrids.subspace_factory import (
    TensorProductSubspaceFactory,
)


def _level_set(nvars: int, level: int) -> List[Tuple[int, ...]]:
    keys: List[Tuple[int, ...]] = []
    for total in range(level + 1):
        keys.extend(_compositions(nvars, total))
    return keys


def _compositions(nvars: int, total: int) -> List[Tuple[int, ...]]:
    if nvars == 1:
        return [(total,)]
    out: List[Tuple[int, ...]] = []
    for first in range(total + 1):
        for rest in _compositions(nvars - 1, total - first):
            out.append((first,) + rest)
    return out


class _Grid:
    """Supplies the selection snapshot, which is all this needs."""

    def __init__(self, selection) -> None:
        self._selection = selection

    def selection(self):
        return self._selection


def _build(
    bkd,
    nvars,
    selected_level,
    candidate_key,
    target_fn,
    basis_type="leja",
    cost=1.0,
):
    """Build a candidate, a grid view, the two surrogates, and marginals."""
    marginals = [UniformMarginal(0.0, 1.0, bkd) for _ in range(nvars)]
    factories = create_basis_factories(marginals, bkd, basis_type)
    tp_factory = TensorProductSubspaceFactory(
        bkd, factories, LinearGrowthRule(scale=1, shift=1)
    )
    selected_keys = _level_set(nvars, selected_level)

    smolyak = IncrementalSmolyakCoefficients(nvars)
    for key in selected_keys:
        smolyak.add(key)

    tracker = SampleTracker(bkd, tp_factory)
    subspaces = {}
    positions = {}
    for key in selected_keys + [candidate_key]:
        idx = bkd.asarray(list(key), dtype=bkd.int64_dtype())
        subspace = tp_factory(idx)
        subspaces[key] = subspace
        positions[key] = tracker.register(idx, subspace)
    tracker.append_new_values(target_fn(tracker.collect_unique_samples()))
    tracker.distribute_values_to_subspaces()

    selection = SmolyakSelection(
        terms=tuple(
            (coef, subspaces[key])
            for key, coef in smolyak.nonzero_items()
        )
    )
    candidate = Candidate(
        index=bkd.asarray(list(candidate_key), dtype=bkd.int64_dtype()),
        subspace=subspaces[candidate_key],
        box=[
            (sign, subspaces[key])
            for key, sign in smolyak.delta(candidate_key)
        ],
        new_sample_local_indices=tracker.get_unique_local_indices(
            positions[candidate_key]
        ),
        config_idx=None,
        cost=cost,
    )

    def surrogate(keys):
        indices = bkd.asarray(
            [[k[d] for k in keys] for d in range(nvars)],
            dtype=bkd.int64_dtype(),
        )
        return CombinationSurrogate(
            bkd,
            nvars,
            [subspaces[k] for k in keys],
            compute_smolyak_coefficients(indices, bkd),
            1,
            indices=indices,
        )

    return (
        candidate,
        _Grid(selection),
        surrogate(selected_keys),
        surrogate(selected_keys + [candidate_key]),
        marginals,
    )


# The target used throughout, written once. ``_cubic`` evaluates it and
# ``_cubic_moments`` integrates it, so the numbers a test compares
# against cannot drift away from the function it fitted.
def _cubic_expression(x, y):
    return x**3 + x * y + 2 * y**2


def _cubic(bkd):
    def fun(samples):
        return bkd.reshape(
            _cubic_expression(samples[0], samples[1]), (1, -1)
        )

    return fun


def _cubic_moments() -> Tuple[float, float]:
    """Integrate the target over U[0,1]^2 symbolically.

    Returns
    -------
    Tuple[float, float]
        Exact mean and variance, as floats.
    """
    import sympy as sp

    x, y = sp.symbols("x y")
    expr = _cubic_expression(x, y)
    mean = sp.integrate(sp.integrate(expr, (x, 0, 1)), (y, 0, 1))
    second = sp.integrate(sp.integrate(expr**2, (x, 0, 1)), (y, 0, 1))
    return float(mean), float(sp.simplify(second - mean**2))


def _expected_error(bkd, selected, sel_plus, marginals):
    """|Delta m| + |Delta sigma| from the two surrogates directly."""
    before = PCEMoments(selected, create_bases_1d(marginals, bkd))
    after = PCEMoments(sel_plus, create_bases_1d(marginals, bkd))
    delta_mean = after.mean() - before.mean()
    delta_variance = after.variance() - before.variance()
    sigma_before = bkd.sqrt(before.variance())
    sigma_after = bkd.sqrt(after.variance())
    qstar = bkd.to_int(bkd.argmax(bkd.abs(delta_variance)))
    return bkd.to_float(
        bkd.abs(delta_mean[qstar])
        + bkd.abs(sigma_after[qstar] - sigma_before[qstar])
    )


class TestMatchesPCEMoments:
    """The reported change is the one PCEMoments would give."""

    @pytest.mark.parametrize(
        "selected_level,candidate_key",
        [
            (1, (2, 0)),
            (1, (0, 2)),
            (1, (1, 1)),
            (2, (2, 1)),
        ],
    )
    def test_error_matches_recombined_surrogates(
        self, bkd, selected_level: int, candidate_key: Tuple[int, ...]
    ) -> None:
        candidate, grid, selected, sel_plus, marginals = _build(
            bkd, 2, selected_level, candidate_key, _cubic(bkd)
        )
        indicator = PCEVarianceIndicator(
            bkd, create_bases_1d(marginals, bkd)
        )
        expected = _expected_error(bkd, selected, sel_plus, marginals)
        bkd.assert_allclose(
            bkd.asarray([indicator(candidate, grid)]),
            bkd.asarray([expected]),
            atol=1e-12,
        )

    def test_snapshot_variance_matches_pce_moments(self, bkd) -> None:
        """The selected set's variance, before any candidate."""
        _, grid, selected, _, marginals = _build(
            bkd, 2, 2, (2, 1), _cubic(bkd)
        )
        indicator = PCEVarianceIndicator(
            bkd, create_bases_1d(marginals, bkd)
        )
        coefs = indicator._selected_coefficients(grid.selection())
        expected = PCEMoments(selected, create_bases_1d(marginals, bkd))
        bkd.assert_allclose(
            indicator._variance(coefs), expected.variance(), rtol=1e-10
        )
        bkd.assert_allclose(
            indicator._mean(coefs), expected.mean(), rtol=1e-10
        )

    def test_variance_delta_matches_recombination(self, bkd) -> None:
        """Delta V equals forming the combined set and subtracting.

        The delta form exists to avoid that subtraction, so agreeing
        with it is what says the algebra is right.
        """
        candidate, grid, selected, sel_plus, marginals = _build(
            bkd, 2, 2, (2, 1), _cubic(bkd)
        )
        indicator = PCEVarianceIndicator(
            bkd, create_bases_1d(marginals, bkd)
        )
        coefs = indicator._selected_coefficients(grid.selection())
        delta = indicator._box_sum(candidate.box)

        before = PCEMoments(selected, create_bases_1d(marginals, bkd))
        after = PCEMoments(sel_plus, create_bases_1d(marginals, bkd))
        bkd.assert_allclose(
            indicator._variance_delta(coefs, delta),
            after.variance() - before.variance(),
            atol=1e-12,
        )
        bkd.assert_allclose(
            indicator._mean(delta), after.mean() - before.mean(), atol=1e-12
        )


class TestAnalyticValues:
    """Pin the reported quantities against values known in closed form."""

    def test_variance_of_exactly_reproduced_target(self, bkd) -> None:
        """A target the grid reproduces has the target's own moments.

        The expected values are integrated symbolically from the same
        expression the target evaluates, rather than written in as
        literals: a literal would have to be re-derived by hand if the
        target ever changed, and nothing would catch it if that were
        skipped.
        """
        expected_mean, expected_variance = _cubic_moments()

        _, grid, _, _, marginals = _build(
            bkd, 2, 3, (3, 1), _cubic(bkd)
        )
        indicator = PCEVarianceIndicator(
            bkd, create_bases_1d(marginals, bkd)
        )
        coefs = indicator._selected_coefficients(grid.selection())
        bkd.assert_allclose(
            indicator._mean(coefs),
            bkd.asarray([expected_mean]),
            rtol=1e-10,
        )
        bkd.assert_allclose(
            indicator._variance(coefs),
            bkd.asarray([expected_variance]),
            rtol=1e-10,
        )

    def test_variance_is_never_negative(self, bkd) -> None:
        """A sum of squares, unlike the signed quadrature rule."""
        _, grid, _, _, marginals = _build(
            bkd, 2, 2, (2, 1), _cubic(bkd)
        )
        indicator = PCEVarianceIndicator(
            bkd, create_bases_1d(marginals, bkd)
        )
        coefs = indicator._selected_coefficients(grid.selection())
        assert bkd.to_float(indicator._variance(coefs)[0]) >= 0.0

    def test_constant_target_reports_a_small_error(self, bkd) -> None:
        """No variance to resolve, so the error is at the noise floor."""

        def constant(samples):
            return bkd.full((1, samples.shape[1]), 3.0)

        candidate, grid, _, _, marginals = _build(
            bkd, 2, 1, (2, 0), constant
        )
        indicator = PCEVarianceIndicator(
            bkd, create_bases_1d(marginals, bkd)
        )
        assert indicator(candidate, grid) < 1e-10


class TestCaching:
    """Conversion is the expensive part and must happen once per subspace."""

    def test_each_subspace_converted_once(self, bkd) -> None:
        candidate, grid, _, _, marginals = _build(
            bkd, 2, 2, (2, 1), _cubic(bkd)
        )
        indicator = PCEVarianceIndicator(
            bkd, create_bases_1d(marginals, bkd)
        )
        calls = {"n": 0}
        inner = indicator._coefficients

        def counting(subspace):
            calls["n"] += 1
            return inner(subspace)

        indicator._cache._statistic = counting

        for _ in range(5):
            indicator(candidate, grid)

        nsubspaces = indicator._cache.nentries()
        assert calls["n"] == nsubspaces
        assert nsubspaces > 0

    def test_snapshot_combined_once_per_round(self, bkd) -> None:
        """Re-scoring against one selection reuses the combination."""
        candidate, grid, _, _, marginals = _build(
            bkd, 2, 2, (2, 1), _cubic(bkd)
        )
        indicator = PCEVarianceIndicator(
            bkd, create_bases_1d(marginals, bkd)
        )
        first = indicator._selected_coefficients(grid.selection())
        second = indicator._selected_coefficients(grid.selection())
        assert first is second


class TestBasisRestriction:
    """A piecewise basis cannot be projected; say so rather than guess."""

    def test_piecewise_basis_is_rejected(self, bkd) -> None:
        from pyapprox.surrogates.affine.indices import (
            ClenshawCurtisGrowthRule,
        )

        marginals = [UniformMarginal(0.0, 1.0, bkd) for _ in range(2)]
        tp_factory = TensorProductSubspaceFactory(
            bkd,
            create_basis_factories(marginals, bkd, "piecewise_linear"),
            ClenshawCurtisGrowthRule(),
        )
        subspace = tp_factory(bkd.asarray([2, 2], dtype=bkd.int64_dtype()))
        samples = subspace.get_samples()
        subspace.set_values(
            bkd.reshape(bkd.sum(samples**2, axis=0), (1, -1))
        )
        indicator = PCEVarianceIndicator(
            bkd, create_bases_1d(marginals, bkd)
        )
        with pytest.raises((TypeError, ValueError)):
            indicator._coefficients(subspace)


class TestProtocolAndIntegration:
    """Usable where the protocol is required, and drives refinement."""

    def test_satisfies_protocol(self, bkd) -> None:
        marginals = [UniformMarginal(0.0, 1.0, bkd) for _ in range(2)]
        assert isinstance(
            PCEVarianceIndicator(bkd, create_bases_1d(marginals, bkd)),
            ErrorIndicatorProtocol,
        )

    def test_end_to_end_convergence(self, bkd) -> None:
        """Refinement driven by this indicator converges."""
        marginals = [UniformMarginal(0.0, 1.0, bkd) for _ in range(2)]
        factories = create_basis_factories(marginals, bkd, "leja")
        tp_factory = TensorProductSubspaceFactory(
            bkd, factories, LinearGrowthRule(scale=1, shift=1)
        )
        admis = MaxLevelCriteria(max_level=5, pnorm=1.0, bkd=bkd)
        fitter = SingleFidelityAdaptiveSparseGridFitter(
            bkd,
            tp_factory,
            admis,
            error_indicator=PCEVarianceIndicator(
                bkd, create_bases_1d(marginals, bkd)
            ),
        )

        def fun(samples):
            return bkd.reshape(
                samples[0] ** 2 + samples[0] * samples[1], (1, -1)
            )

        result = fitter.refine_to_tolerance(
            FunctionFromCallable(1, 2, fun, bkd), tol=1e-12, max_steps=40
        )
        samples = bkd.asarray([[0.1, 0.35, 0.7], [0.2, 0.55, 0.9]])
        bkd.assert_allclose(
            result.surrogate(samples), fun(samples), rtol=1e-8
        )
