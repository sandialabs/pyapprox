"""Quadrature-scored refinement at one Leja point per level, end to end.

With one new point per level, the level-1 Leja rule on U[0,1] is
{0.5, 1} with weights [1, 0], so every level-1 candidate adds a
weightless point. SummedSubspaceVarianceIndicator then raises on the
first round. Wrapped in ZeroWeightFallback, those candidates are scored
by L2SurplusIndicator instead and refinement proceeds in every
dimension.

Reference figures, 4D Genz oscillatory, decay none, 150 samples, NumPy:
scoring the weightless candidates zero (refinement stalls) took 89.5 s
for an RMS error of 2.53e-03; two points per level, which never forms
the weightless rule, took 0.69 s for 5.68e-06. Only the error is
asserted, as a bound well inside that gap; wall time varies too much
across machines to assert.
"""

from typing import List

import numpy as np
import pytest

from pyapprox.interface.functions.protocols.function import FunctionProtocol
from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.indices import (
    LinearGrowthRule,
    MaxLevelCriteria,
    max_level_for_univariate_npoints,
)
from pyapprox.surrogates.sparsegrids.adaptive_fitter import (
    SingleFidelityAdaptiveSparseGridFitter,
)
from pyapprox.surrogates.sparsegrids.basis_factory import LejaLagrangeFactory
from pyapprox.surrogates.sparsegrids.error_indicators import (
    ErrorIndicatorProtocol,
    L2SurplusIndicator,
    QuadratureRefinementStalled,
    QuadratureRefinementStalledError,
    SummedSubspaceVarianceIndicator,
    ZeroWeightFallback,
)
from pyapprox.surrogates.sparsegrids.subspace_factory import (
    TensorProductSubspaceFactory,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox_benchmarks.quadrature.genz import GenzOscillatoryBenchmark

_NVARS = 4
_BUDGET = 150
_NTEST = 1000


def _make_fitter(
    bkd: Backend[Array], indicator: ErrorIndicatorProtocol[Array]
) -> SingleFidelityAdaptiveSparseGridFitter[Array]:
    """A fitter on U[0,1]^4 adding one Leja point per level."""
    # One factory shared by every dimension, so they share one Leja
    # sequence rather than each optimizing the same points.
    leja = LejaLagrangeFactory(UniformMarginal(0.0, 1.0, bkd), bkd)
    growth = LinearGrowthRule(scale=1, shift=1)
    factory = TensorProductSubspaceFactory(bkd, [leja] * _NVARS, growth)
    admissibility = MaxLevelCriteria(
        max_level_for_univariate_npoints(growth, _BUDGET), 1.0, bkd
    )
    return SingleFidelityAdaptiveSparseGridFitter(
        bkd, factory, admissibility, error_indicator=indicator
    )


def _target(bkd: Backend[Array]) -> FunctionProtocol[Array]:
    return GenzOscillatoryBenchmark(bkd, _NVARS, "none").problem().function()


def _fallback(bkd: Backend[Array]) -> ZeroWeightFallback[Array]:
    return ZeroWeightFallback(
        SummedSubspaceVarianceIndicator(bkd), L2SurplusIndicator(bkd)
    )


def _refine(bkd: Backend[Array], indicator: ErrorIndicatorProtocol[Array]) -> float:
    """Refine to the budget at one Leja point per level; return RMS error."""
    target = _target(bkd)
    fitter = _make_fitter(bkd, indicator)
    while True:
        samples = fitter.step_samples()
        if samples is None:
            break
        fitter.step_values(target(samples))
        if fitter.get_samples("all").shape[1] >= _BUDGET:
            break

    test_samples = bkd.asarray(np.random.uniform(0.0, 1.0, (_NVARS, _NTEST)))
    diff = fitter.result().surrogate(test_samples) - target(test_samples)
    return bkd.to_float(bkd.sqrt(bkd.sum(diff * diff) / _NTEST))


class TestOnePointPerLevel:
    def test_unwrapped_indicator_raises(self, bkd: Backend[Array]) -> None:
        """The stall stops the run in the first round instead of finishing."""
        with pytest.raises(QuadratureRefinementStalledError):
            _refine(bkd, SummedSubspaceVarianceIndicator(bkd))

    def test_fallback_refines_every_dimension(self, bkd: Backend[Array]) -> None:
        with pytest.warns(QuadratureRefinementStalled) as record:
            rms_error = _refine(bkd, _fallback(bkd))
        assert len(record) == 1
        # Ten times below the stalled run's 2.53e-03.
        assert rms_error < 2.5e-4, rms_error


def _batches(
    bkd: Backend[Array],
    fitter: SingleFidelityAdaptiveSparseGridFitter[Array],
    budget: int,
) -> List[Array]:
    """Refine to ``budget`` samples; on a stall, wrap and rescore.

    Returns every batch requested. A fitter wrapped from the start never
    reaches the ``except`` branch.
    """
    target = _target(bkd)
    batches: List[Array] = []
    while True:
        samples = fitter.step_samples()
        if samples is None:
            break
        batches.append(samples)
        try:
            fitter.step_values(target(samples))
        except QuadratureRefinementStalledError:
            fitter.set_error_indicator(_fallback(bkd))
            fitter.rescore()
        if fitter.get_samples("all").shape[1] >= budget:
            break
    return batches


class TestWrapAfterStall:
    """Wrapping after the raise refines exactly as wrapping from the start.

    The stall stops the bare run while it scores the first round, after
    it has absorbed that round's values. Swapping in the wrapper and
    rescoring must then cost no model evaluation and request the same
    batches, in the same order, as a run wrapped from the start.
    """

    _BUDGET = 60

    def test_same_batches_and_selection_order(self, bkd: Backend[Array]) -> None:
        wrapped = _make_fitter(bkd, _fallback(bkd))
        stopped = _make_fitter(bkd, SummedSubspaceVarianceIndicator(bkd))
        with pytest.warns(QuadratureRefinementStalled):
            desired = _batches(bkd, wrapped, self._BUDGET)
        with pytest.warns(QuadratureRefinementStalled):
            actual = _batches(bkd, stopped, self._BUDGET)

        assert len(actual) == len(desired)
        for actual_batch, desired_batch in zip(actual, desired):
            bkd.assert_allclose(actual_batch, desired_batch, rtol=1e-12)
        actual_selected = stopped.get_selected_indices()
        desired_selected = wrapped.get_selected_indices()
        assert actual_selected.shape == desired_selected.shape
        assert bkd.all_bool(bkd.equal(actual_selected, desired_selected))
