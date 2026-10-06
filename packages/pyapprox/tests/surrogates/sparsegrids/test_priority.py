"""Tests for the priority seam.

Indicators report error; priority decides queue order from that error
and the candidate's cost. These are separate so either can be replaced
alone, and so cost policy is not repeated in every indicator.
"""

import pytest

from pyapprox.interface.functions.fromcallable.function import (
    FunctionFromCallable,
)
from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.indices import (
    LinearGrowthRule,
    MaxLevelCriteria,
)
from pyapprox.surrogates.sparsegrids import create_basis_factories
from pyapprox.surrogates.sparsegrids.adaptive_fitter import (
    SingleFidelityAdaptiveSparseGridFitter,
)
from pyapprox.surrogates.sparsegrids.candidate_info import Candidate
from pyapprox.surrogates.sparsegrids.priority import (
    CostWeightedPriority,
    PriorityProtocol,
)
from pyapprox.surrogates.sparsegrids.subspace_factory import (
    TensorProductSubspaceFactory,
)


def _candidate(cost: float) -> Candidate:
    """A candidate carrying only the field priority reads."""
    return Candidate(
        index=None,
        subspace=None,
        box=(),
        new_sample_local_indices=(),
        config_idx=None,
        cost=cost,
    )


class TestCostWeightedPriority:
    """The default policy: error per unit cost."""

    def test_satisfies_protocol(self) -> None:
        assert isinstance(CostWeightedPriority(), PriorityProtocol)

    @pytest.mark.parametrize(
        "error,cost,expected",
        [(1.0, 2.0, 0.5), (6.0, 3.0, 2.0), (0.0, 4.0, 0.0)],
    )
    def test_divides_by_cost(
        self, error: float, cost: float, expected: float
    ) -> None:
        assert CostWeightedPriority()(error, _candidate(cost)) == expected

    @pytest.mark.parametrize("cost", [0.0, -1.0])
    def test_falls_back_to_error_without_positive_cost(
        self, cost: float
    ) -> None:
        """A grid with no cost model still orders by error alone."""
        assert CostWeightedPriority()(2.5, _candidate(cost)) == 2.5

    def test_higher_cost_lowers_priority(self) -> None:
        """Equal errors, unequal costs: the cheaper one wins."""
        priority = CostWeightedPriority()
        cheap = priority(1.0, _candidate(1.0))
        dear = priority(1.0, _candidate(10.0))
        assert cheap > dear

    def test_higher_error_raises_priority(self) -> None:
        """Equal costs, unequal errors: the larger error wins."""
        priority = CostWeightedPriority()
        small = priority(1.0, _candidate(2.0))
        large = priority(5.0, _candidate(2.0))
        assert large > small


class _RecordingPriority:
    """Custom policy that records what it was asked to rank."""

    def __init__(self) -> None:
        self.calls: list = []

    def __call__(self, error: float, candidate: Candidate) -> float:
        self.calls.append((error, candidate.cost))
        # Invert the usual ordering, so a fitter honoring this policy
        # visibly departs from the cost-weighted default.
        return -error


class TestFitterHonorsInjectedPriority:
    """The fitter forms priority through whatever it was given."""

    def _fitter(self, bkd, priority):
        marginals = [UniformMarginal(0.0, 1.0, bkd) for _ in range(2)]
        factories = create_basis_factories(marginals, bkd, "gauss")
        tp_factory = TensorProductSubspaceFactory(
            bkd, factories, LinearGrowthRule(scale=1, shift=1)
        )
        admis = MaxLevelCriteria(max_level=4, pnorm=1.0, bkd=bkd)
        return SingleFidelityAdaptiveSparseGridFitter(
            bkd, tp_factory, admis, priority=priority
        )

    def _target(self, bkd):
        def fun(samples):
            x, y = samples[0, :], samples[1, :]
            return bkd.reshape(x**4 + y**3, (1, -1))

        return FunctionFromCallable(1, 2, fun, bkd)

    def test_custom_priority_is_called(self, bkd) -> None:
        """Every scored candidate passes through the injected policy."""
        recording = _RecordingPriority()
        fitter = self._fitter(bkd, recording)
        target = self._target(bkd)
        for _ in range(3):
            samples = fitter.step_samples()
            if samples is None:
                break
            fitter.step_values(target(samples))
        assert len(recording.calls) > 0
        # Costs reach the policy, not just errors.
        assert all(cost > 0 for _, cost in recording.calls)

    def test_default_is_cost_weighted(self, bkd) -> None:
        """Omitting priority gives CostWeightedPriority."""
        fitter = self._fitter(bkd, None)
        assert isinstance(
            fitter._fitter._priority, CostWeightedPriority
        )

    def test_rejects_non_protocol(self, bkd) -> None:
        with pytest.raises(TypeError, match="PriorityProtocol"):
            self._fitter(bkd, "not a priority")
