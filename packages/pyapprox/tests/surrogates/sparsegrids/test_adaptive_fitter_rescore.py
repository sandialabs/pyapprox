"""Tests for stopping and continuing an adaptive sparse grid fitter.

When an error indicator raises while ``step_values`` scores candidates,
the fitter keeps every value it absorbed, discards the partly built
queue, and refuses ``step_samples`` until ``rescore``. The caller may
swap the indicator with ``set_error_indicator`` first. Continuing must
cost no model evaluation and must refine exactly as a run that never
stopped.

The stopping indicator wraps L2SurplusIndicator and raises on a chosen
call, so up to the stop it scores exactly as the plain indicator and the
two runs can be compared sample for sample.

Tests run on both NumPy and PyTorch backends.
"""

from typing import Dict, Generic, List, Optional

import pytest

from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.indices import (
    LinearGrowthRule,
    MaxLevelCriteria,
)
from pyapprox.surrogates.sparsegrids.adaptive_fitter import (
    MultiFidelityAdaptiveSparseGridFitter,
    SingleFidelityAdaptiveSparseGridFitter,
)
from pyapprox.surrogates.sparsegrids.basis_factory import GaussLagrangeFactory
from pyapprox.surrogates.sparsegrids.candidate_info import (
    AdaptiveGridViewProtocol,
    Candidate,
    ConfigIdx,
)
from pyapprox.surrogates.sparsegrids.error_indicators import (
    ErrorIndicatorProtocol,
    L2SurplusIndicator,
)
from pyapprox.surrogates.sparsegrids.subspace_factory import (
    TensorProductSubspaceFactory,
)
from pyapprox.util.backends.protocols import Array, Backend

_NVARS = 3


class _ScoringStopped(Exception):
    """Raised by the stopping indicator."""


class _StopsOnCall(Generic[Array]):
    """Scores as ``inner`` until armed, then raises on the nth call."""

    def __init__(self, inner: ErrorIndicatorProtocol[Array]) -> None:
        self._inner = inner
        self._countdown: Optional[int] = None

    def arm(self, nth_call: int) -> None:
        """Raise on the ``nth_call``-th call from now, once."""
        self._countdown = nth_call

    def __call__(
        self, candidate: Candidate[Array], grid: AdaptiveGridViewProtocol[Array]
    ) -> float:
        if self._countdown is not None:
            self._countdown -= 1
            if self._countdown == 0:
                self._countdown = None
                raise _ScoringStopped("stopped while scoring")
        return self._inner(candidate, grid)


def _target(bkd: Backend[Array], samples: Array) -> Array:
    """Smooth, anisotropic, with an interaction term."""
    x0, x1, x2 = samples[0], samples[1], samples[2]
    vals = bkd.exp(0.7 * x0 + 0.3 * x1 + 0.1 * x2) + 0.2 * x0 * x1
    return bkd.reshape(vals, (1, -1))


def _make_fitter(
    bkd: Backend[Array], indicator: ErrorIndicatorProtocol[Array]
) -> SingleFidelityAdaptiveSparseGridFitter[Array]:
    marginal = UniformMarginal(-1.0, 1.0, bkd)
    factories = [GaussLagrangeFactory(marginal, bkd) for _ in range(_NVARS)]
    growth = LinearGrowthRule(scale=1, shift=1)
    tp_factory = TensorProductSubspaceFactory(bkd, factories, growth)
    admis = MaxLevelCriteria(max_level=5, pnorm=1.0, bkd=bkd)
    return SingleFidelityAdaptiveSparseGridFitter(
        bkd, tp_factory, admis, error_indicator=indicator
    )


def _advance(
    bkd: Backend[Array], fitter: SingleFidelityAdaptiveSparseGridFitter[Array]
) -> Array:
    """Run one step_samples/step_values round; return its samples."""
    samples = fitter.step_samples()
    assert samples is not None
    fitter.step_values(_target(bkd, samples))
    return samples


def _stopped_and_reference_runs(
    bkd: Backend[Array],
) -> tuple[
    SingleFidelityAdaptiveSparseGridFitter[Array],
    _StopsOnCall[Array],
    SingleFidelityAdaptiveSparseGridFitter[Array],
    Array,
]:
    """A run stopped on the third candidate of round two, and a plain one.

    Both have absorbed the same two batches. The plain run has scored
    round two; the stopped run raised during it. Also returns the second
    batch, which the stopped run absorbed before raising.
    """
    stopping = _StopsOnCall(L2SurplusIndicator(bkd))
    stopped = _make_fitter(bkd, stopping)
    reference = _make_fitter(bkd, L2SurplusIndicator(bkd))

    _advance(bkd, stopped)
    _advance(bkd, reference)

    batch = stopped.step_samples()
    assert batch is not None
    # Round two scores at least three candidates in 3D: the two left
    # from round one and the child of the promoted one.
    stopping.arm(3)
    with pytest.raises(_ScoringStopped):
        stopped.step_values(_target(bkd, batch))
    _advance(bkd, reference)
    return stopped, stopping, reference, batch


def _assert_same_values(
    bkd: Backend[Array],
    actual: SingleFidelityAdaptiveSparseGridFitter[Array],
    desired: SingleFidelityAdaptiveSparseGridFitter[Array],
) -> None:
    bkd.assert_allclose(actual.get_samples(), desired.get_samples(), rtol=1e-12)
    actual_vals = actual.get_values()
    desired_vals = desired.get_values()
    assert actual_vals is not None and desired_vals is not None
    bkd.assert_allclose(actual_vals, desired_vals, rtol=1e-12)


def _assert_same_indices(bkd: Backend[Array], actual: Array, desired: Array) -> None:
    assert actual.shape == desired.shape
    assert bkd.all_bool(bkd.equal(actual, desired))


class TestStopKeepsState:
    """An indicator that raises leaves the fitter consistent and blocked."""

    def test_values_survive_the_raise(self, bkd: Backend[Array]) -> None:
        """Every value of the interrupted batch is kept."""
        stopped, _, reference, _ = _stopped_and_reference_runs(bkd)
        _assert_same_values(bkd, stopped, reference)
        assert stopped.nselected() == reference.nselected()
        assert stopped.ncandidates() == reference.ncandidates()

    def test_step_samples_refuses_until_rescore(self, bkd: Backend[Array]) -> None:
        """The partial queue is never used to pick the next batch."""
        stopped, _, _, _ = _stopped_and_reference_runs(bkd)
        with pytest.raises(RuntimeError, match="rescore"):
            stopped.step_samples()
        # Refusing changes nothing, so it refuses again.
        with pytest.raises(RuntimeError, match="rescore"):
            stopped.step_samples()

    def test_rescore_that_raises_stays_blocked(self, bkd: Backend[Array]) -> None:
        """A failed rescore leaves the fitter as a failed step_values does."""
        stopped, stopping, reference, _ = _stopped_and_reference_runs(bkd)
        stopping.arm(2)
        with pytest.raises(_ScoringStopped):
            stopped.rescore()
        with pytest.raises(RuntimeError, match="rescore"):
            stopped.step_samples()
        _assert_same_values(bkd, stopped, reference)


class TestContinueAfterStop:
    """Continuing refines exactly as a run that never stopped."""

    def test_next_batches_match_uninterrupted_run(self, bkd: Backend[Array]) -> None:
        """Swap in the plain indicator, rescore, and follow the reference."""
        stopped, _, reference, _ = _stopped_and_reference_runs(bkd)
        stopped.set_error_indicator(L2SurplusIndicator(bkd))
        stopped.rescore()
        bkd.assert_allclose(
            bkd.asarray([stopped.current_error()]),
            bkd.asarray([reference.current_error()]),
            rtol=1e-12,
        )
        for _ in range(4):
            bkd.assert_allclose(
                _advance(bkd, stopped), _advance(bkd, reference), rtol=1e-12
            )
        _assert_same_indices(
            bkd, stopped.get_selected_indices(), reference.get_selected_indices()
        )
        _assert_same_values(bkd, stopped, reference)

    def test_rescore_with_same_indicator(self, bkd: Backend[Array]) -> None:
        """An indicator whose failure was transient needs no swap."""
        stopped, _, reference, _ = _stopped_and_reference_runs(bkd)
        stopped.rescore()
        for _ in range(3):
            bkd.assert_allclose(
                _advance(bkd, stopped), _advance(bkd, reference), rtol=1e-12
            )

    def test_rescore_without_stop_changes_nothing(self, bkd: Backend[Array]) -> None:
        """Rescoring a scored fitter with its own indicator is idempotent."""
        fitter = _make_fitter(bkd, L2SurplusIndicator(bkd))
        reference = _make_fitter(bkd, L2SurplusIndicator(bkd))
        for _ in range(2):
            _advance(bkd, fitter)
            _advance(bkd, reference)
        fitter.rescore()
        for _ in range(3):
            bkd.assert_allclose(
                _advance(bkd, fitter), _advance(bkd, reference), rtol=1e-12
            )


class TestSetErrorIndicator:
    """Changing the indicator partway through a run."""

    def test_before_first_step_needs_no_rescore(self, bkd: Backend[Array]) -> None:
        """With nothing scored yet, a swap is the same as passing it in."""
        fitter = _make_fitter(bkd, _StopsOnCall(L2SurplusIndicator(bkd)))
        fitter.set_error_indicator(L2SurplusIndicator(bkd))
        reference = _make_fitter(bkd, L2SurplusIndicator(bkd))
        for _ in range(3):
            bkd.assert_allclose(
                _advance(bkd, fitter), _advance(bkd, reference), rtol=1e-12
            )

    def test_after_scoring_requires_rescore(self, bkd: Backend[Array]) -> None:
        """The queue was scored by the old indicator, so it is stale."""
        fitter = _make_fitter(bkd, L2SurplusIndicator(bkd))
        _advance(bkd, fitter)
        fitter.set_error_indicator(L2SurplusIndicator(bkd))
        with pytest.raises(RuntimeError, match="rescore"):
            fitter.step_samples()
        fitter.rescore()
        assert fitter.step_samples() is not None

    def test_between_samples_and_values(self, bkd: Backend[Array]) -> None:
        """step_values scores with the new indicator; no rescore needed."""
        stopping = _StopsOnCall(L2SurplusIndicator(bkd))
        fitter = _make_fitter(bkd, stopping)
        reference = _make_fitter(bkd, L2SurplusIndicator(bkd))
        _advance(bkd, fitter)
        _advance(bkd, reference)
        batch = fitter.step_samples()
        assert batch is not None
        stopping.arm(1)
        fitter.set_error_indicator(L2SurplusIndicator(bkd))
        fitter.step_values(_target(bkd, batch))
        _advance(bkd, reference)
        for _ in range(2):
            bkd.assert_allclose(
                _advance(bkd, fitter), _advance(bkd, reference), rtol=1e-12
            )

    def test_rejects_non_indicator(self, bkd: Backend[Array]) -> None:
        fitter = _make_fitter(bkd, L2SurplusIndicator(bkd))
        with pytest.raises(TypeError, match="ErrorIndicatorProtocol"):
            fitter.set_error_indicator(3.0)  # type: ignore[arg-type]


class TestRescoreErrors:
    """rescore refuses when there is nothing consistent to score."""

    def test_before_any_values(self, bkd: Backend[Array]) -> None:
        fitter = _make_fitter(bkd, L2SurplusIndicator(bkd))
        with pytest.raises(RuntimeError, match="step_values"):
            fitter.rescore()
        fitter.step_samples()
        with pytest.raises(RuntimeError, match="step_values"):
            fitter.rescore()

    def test_while_batch_awaits_values(self, bkd: Backend[Array]) -> None:
        """Rescoring now would score new candidates without their values."""
        fitter = _make_fitter(bkd, L2SurplusIndicator(bkd))
        _advance(bkd, fitter)
        assert fitter.step_samples() is not None
        with pytest.raises(RuntimeError, match="step_values"):
            fitter.rescore()


class TestMultiFidelityFitter:
    """The core fitter carries the same contract."""

    def test_stop_and_rescore(self, bkd: Backend[Array]) -> None:
        stopping = _StopsOnCall(L2SurplusIndicator(bkd))
        sf = _make_fitter(bkd, stopping)
        reference = _make_fitter(bkd, L2SurplusIndicator(bkd))
        # The single-fidelity wrapper holds a nconfig_vars=0 core fitter;
        # drive the core directly through its Dict interface.
        core: MultiFidelityAdaptiveSparseGridFitter[Array] = sf._fitter

        def advance_core() -> Dict[ConfigIdx, Array]:
            samples = core.step_samples()
            assert samples is not None
            core.step_values({k: _target(bkd, s) for k, s in samples.items()})
            return samples

        advance_core()
        _advance(bkd, reference)
        stopping.arm(2)
        with pytest.raises(_ScoringStopped):
            advance_core()
        _advance(bkd, reference)
        with pytest.raises(RuntimeError, match="rescore"):
            core.step_samples()
        core.set_error_indicator(L2SurplusIndicator(bkd))
        core.rescore()
        batches: List[Array] = [advance_core()[()] for _ in range(2)]
        for batch in batches:
            bkd.assert_allclose(batch, _advance(bkd, reference), rtol=1e-12)
