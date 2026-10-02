"""Sources of covariance blocks: run a joint evaluator, or reuse its outputs.

A source owns where the outputs come from; the accumulator owns how they
are combined. Accumulators hold state, so a source takes a function that
makes a fresh one rather than an instance, and computes its blocks once.
"""

from typing import Callable, Generic, Optional, Protocol, runtime_checkable

from pyapprox.interface.functions.joint import (
    JointEvaluatorProtocol,
    JointOutputs,
)
from pyapprox.probability.moments.accumulators import WeightedAccumulator
from pyapprox.probability.moments.protocols import (
    CovarianceBlocksProtocol,
    MomentAccumulatorProtocol,
)
from pyapprox.probability.moments.rules import WeightedRuleProtocol
from pyapprox.util.backends.protocols import Array, Backend

#: Makes a fresh accumulator for one computation of the blocks.
AccumulatorFactory = Callable[[], MomentAccumulatorProtocol[Array]]


@runtime_checkable
class MomentSourceProtocol(Protocol, Generic[Array]):
    """Anything that produces covariance blocks."""

    def blocks(self) -> CovarianceBlocksProtocol[Array]:
        """The covariance blocks."""
        ...


class QuadratureMoments(Generic[Array]):
    """Blocks from evaluating a joint evaluator on a rule's points.

    Parameters
    ----------
    rule : WeightedRuleProtocol[Array]
        Points and weights; weights of shape (npoints,) are reshaped to
        (1, npoints) here.
    evaluator : JointEvaluatorProtocol[Array]
        Produces targets and observations at the points.
    make_accumulator : Callable[[], MomentAccumulatorProtocol[Array]], optional
        Makes the accumulator. Default ``WeightedAccumulator``.
    batch_size : int, optional
        Points evaluated per call to the evaluator. Default all at once.
    """

    def __init__(
        self,
        rule: WeightedRuleProtocol[Array],
        evaluator: JointEvaluatorProtocol[Array],
        make_accumulator: Optional[AccumulatorFactory[Array]] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        if not isinstance(rule, WeightedRuleProtocol):
            raise TypeError(
                f"rule must satisfy WeightedRuleProtocol, got {type(rule).__name__}"
            )
        if not isinstance(evaluator, JointEvaluatorProtocol):
            raise TypeError(
                "evaluator must satisfy JointEvaluatorProtocol, got "
                f"{type(evaluator).__name__}"
            )
        if rule.nvars() != evaluator.nvars():
            raise ValueError(
                f"rule has {rule.nvars()} variables but the evaluator takes "
                f"{evaluator.nvars()}"
            )
        if batch_size is not None and batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self._rule = rule
        self._evaluator = evaluator
        self._make_accumulator = make_accumulator
        self._batch_size = batch_size
        self._blocks: Optional[CovarianceBlocksProtocol[Array]] = None

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._rule.bkd()

    def blocks(self) -> CovarianceBlocksProtocol[Array]:
        """Evaluate on the first call; return the same blocks afterwards."""
        if self._blocks is None:
            self._blocks = self._compute()
        return self._blocks

    def _compute(self) -> CovarianceBlocksProtocol[Array]:
        bkd = self.bkd()
        points, weights = self._rule()
        npoints = points.shape[1]
        if weights.ndim != 1 or weights.shape[0] != npoints:
            raise ValueError(
                f"rule weights must have shape ({npoints},), got {tuple(weights.shape)}"
            )
        weights = bkd.reshape(weights, (1, npoints))
        accumulator = _new_accumulator(self._make_accumulator, bkd)
        step = npoints if self._batch_size is None else self._batch_size
        for start in range(0, npoints, step):
            cols = slice(start, min(start + step, npoints))
            accumulator.update(
                weights[:, cols], self._evaluator.evaluate(points[:, cols])
            )
        return accumulator.finalize()


class CachedMoments(Generic[Array]):
    """Blocks from outputs already computed, without re-running the model.

    Parameters
    ----------
    outputs : JointOutputs[Array]
        Stored targets and observations.
    weights : Array
        Their weights. Shape: (1, nsamples)
    bkd : Backend[Array]
        Computational backend.
    make_accumulator : Callable[[], MomentAccumulatorProtocol[Array]], optional
        Makes the accumulator. Default ``WeightedAccumulator``.
    """

    def __init__(
        self,
        outputs: JointOutputs[Array],
        weights: Array,
        bkd: Backend[Array],
        make_accumulator: Optional[AccumulatorFactory[Array]] = None,
    ) -> None:
        self._outputs = outputs
        self._weights = weights
        self._bkd = bkd
        self._make_accumulator = make_accumulator
        self._blocks: Optional[CovarianceBlocksProtocol[Array]] = None

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def blocks(self) -> CovarianceBlocksProtocol[Array]:
        """Accumulate on the first call; return the same blocks afterwards."""
        if self._blocks is None:
            accumulator = _new_accumulator(self._make_accumulator, self._bkd)
            accumulator.update(self._weights, self._outputs)
            self._blocks = accumulator.finalize()
        return self._blocks


def _new_accumulator(
    make_accumulator: Optional[AccumulatorFactory[Array]], bkd: Backend[Array]
) -> MomentAccumulatorProtocol[Array]:
    if make_accumulator is None:
        return WeightedAccumulator(bkd)
    accumulator = make_accumulator()
    if not isinstance(accumulator, MomentAccumulatorProtocol):
        raise TypeError(
            "make_accumulator must return a MomentAccumulatorProtocol, got "
            f"{type(accumulator).__name__}"
        )
    return accumulator
