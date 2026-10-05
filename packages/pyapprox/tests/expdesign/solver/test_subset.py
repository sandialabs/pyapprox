"""Tests for the subset searches and TopK rounding.

Two kinds of objective:

- a linear-Gaussian A-optimal design over six observations, whose subset
  values ``BinaryDesignSubsetObjective`` gives exactly; the reference is
  the minimum over all subsets, enumerated directly;
- tables of subset values built so greedy or single swaps get stuck.
"""

import itertools
from typing import Dict, FrozenSet, Generic, Sequence, Tuple

import numpy as np
import pytest
from pyapprox.expdesign.design_space import (
    BinaryDesignSubsetObjective,
    GroupedDesign,
    ParameterizedObjective,
    ReevaluatingIncremental,
)
from pyapprox.expdesign.gaussian import AOptimal, BlendedObservation, DesignObjective
from pyapprox.expdesign.protocols import RoundingProtocol
from pyapprox.expdesign.solver import (
    ExchangeSubsetSolver,
    ExhaustiveSubsetSolver,
    GreedySubsetSolver,
    TopK,
)
from pyapprox.inverse.joint_gaussian import JointGaussian
from pyapprox.probability.covariance import DenseCholeskyCovarianceOperator
from pyapprox.probability.moments import DenseBlocks
from pyapprox.util.backends.protocols import Array, Backend

_BY_SENSOR = [[0, 3], [1, 4], [2, 5]]


def _a_optimal(bkd: Backend[Array]) -> DesignObjective[Array]:
    """Three parameters observed six times with correlated noise."""
    rng = np.random.default_rng(32)
    root = rng.normal(size=(3, 3))
    prior_cov = root @ root.T / 3 + np.eye(3)
    noise_root = rng.normal(size=(6, 6))
    noise_cov = 0.1 * (noise_root @ noise_root.T / 6 + np.eye(6))
    blocks = DenseBlocks.from_linear_model(
        bkd.asarray(rng.normal(size=(6, 3))),
        bkd.zeros((3, 1)),
        bkd.asarray(prior_cov),
        [bkd.eye(3)],
        bkd,
    )
    noise = DenseCholeskyCovarianceOperator(bkd.asarray(noise_cov), bkd)
    joint = JointGaussian(blocks, noise)
    return DesignObjective(joint, BlendedObservation.from_noise(noise), AOptimal(), 0)


class _Table(Generic[Array]):
    """Subset values from a table, with a default for unlisted subsets."""

    def __init__(
        self,
        ncandidates: int,
        values: Dict[FrozenSet[int], float],
        default: float,
        bkd: Backend[Array],
    ) -> None:
        self._ncandidates = ncandidates
        self._values = values
        self._default = default
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def ncandidates(self) -> int:
        return self._ncandidates

    def value(self, subset: Sequence[int]) -> float:
        return self._values.get(frozenset(subset), self._default)


def _greedy_trap(bkd: Backend[Array]) -> _Table[Array]:
    """Greedy picks 0 first, but the best pair is {1, 2}."""
    values = {frozenset(s): v for s, v in [((0,), 1.0), ((1,), 2.0), ((2,), 2.0)]}
    values.update({frozenset((0, ii)): 1.5 for ii in (1, 2, 3)})
    values[frozenset((1, 2))] = 0.5
    return _Table(4, values, 3.0, bkd)


def _swap_trap(bkd: Backend[Array]) -> _Table[Array]:
    """{0, 1} is best under single swaps, but {2, 3} is better."""
    values = {frozenset((0, 1)): 1.0, frozenset((2, 3)): 0.5}
    return _Table(4, values, 2.0, bkd)


def _assert_value(
    bkd: Backend[Array], actual: float, expected: float, rtol: float = 1e-12
) -> None:
    bkd.assert_allclose(bkd.asarray([actual]), bkd.asarray([expected]), rtol=rtol)


def _best(
    subset_objective: BinaryDesignSubsetObjective[Array] | _Table[Array], k: int
) -> Tuple[Tuple[int, ...], float]:
    """The minimum over all k-subsets, enumerated directly."""
    scored = [
        (subset_objective.value(subset), subset)
        for subset in itertools.combinations(range(subset_objective.ncandidates()), k)
    ]
    value, subset = min(scored)
    return subset, value


class TestExhaustive:
    @pytest.mark.parametrize("k", [1, 2, 3, 6])
    def test_matches_enumeration(self, bkd: Backend[Array], k: int) -> None:
        subset_objective = BinaryDesignSubsetObjective(_a_optimal(bkd))
        result = ExhaustiveSubsetSolver(subset_objective).solve(k)
        subset, value = _best(subset_objective, k)
        assert result.subset == subset
        _assert_value(bkd, result.value, value)
        assert result.nevaluations == len(list(itertools.combinations(range(6), k)))

    def test_over_groups(self, bkd: Backend[Array]) -> None:
        subset_objective = BinaryDesignSubsetObjective(
            ParameterizedObjective(_a_optimal(bkd), GroupedDesign(_BY_SENSOR, 6, bkd))
        )
        result = ExhaustiveSubsetSolver(subset_objective).solve(2)
        subset, value = _best(subset_objective, 2)
        assert result.subset == subset
        _assert_value(bkd, result.value, value)

    @pytest.mark.parametrize("k", [0, 7])
    def test_rejects_bad_k(self, bkd: Backend[Array], k: int) -> None:
        solver = ExhaustiveSubsetSolver(BinaryDesignSubsetObjective(_a_optimal(bkd)))
        with pytest.raises(ValueError):
            solver.solve(k)


class TestGreedy:
    def test_one_is_exhaustive(self, bkd: Backend[Array]) -> None:
        subset_objective = BinaryDesignSubsetObjective(_a_optimal(bkd))
        result = GreedySubsetSolver(ReevaluatingIncremental(subset_objective)).solve(1)
        assert result.subset == _best(subset_objective, 1)[0]
        assert result.nevaluations == 6

    @pytest.mark.parametrize("k", [2, 3])
    def test_batch_of_k_is_exhaustive(self, bkd: Backend[Array], k: int) -> None:
        subset_objective = BinaryDesignSubsetObjective(_a_optimal(bkd))
        greedy = GreedySubsetSolver(
            ReevaluatingIncremental(subset_objective), batch_size=k
        )
        result = greedy.solve(k)
        subset, value = _best(subset_objective, k)
        assert result.subset == subset
        _assert_value(bkd, result.value, value)

    def test_value_is_of_the_chosen_subset(self, bkd: Backend[Array]) -> None:
        subset_objective = BinaryDesignSubsetObjective(_a_optimal(bkd))
        result = GreedySubsetSolver(ReevaluatingIncremental(subset_objective)).solve(4)
        assert len(result.subset) == 4
        _assert_value(bkd, result.value, subset_objective.value(result.subset))
        assert result.nevaluations == 6 + 5 + 4 + 3

    def test_last_batch_is_capped(self, bkd: Backend[Array]) -> None:
        """k = 3 in batches of 2 adds two, then one."""
        subset_objective = BinaryDesignSubsetObjective(_a_optimal(bkd))
        greedy = GreedySubsetSolver(
            ReevaluatingIncremental(subset_objective), batch_size=2
        )
        result = greedy.solve(3)
        assert len(result.subset) == 3
        assert result.nevaluations == 15 + 4

    def test_batches_escape_the_trap(self, bkd: Backend[Array]) -> None:
        incremental = ReevaluatingIncremental(_greedy_trap(bkd))
        single = GreedySubsetSolver(incremental).solve(2)
        assert single.subset == (0, 1)
        _assert_value(bkd, single.value, 1.5)
        paired = GreedySubsetSolver(incremental, batch_size=2).solve(2)
        assert paired.subset == (1, 2)
        _assert_value(bkd, paired.value, 0.5)

    def test_rejects_bad_inputs(self, bkd: Backend[Array]) -> None:
        subset_objective = BinaryDesignSubsetObjective(_a_optimal(bkd))
        with pytest.raises(ValueError):
            GreedySubsetSolver(ReevaluatingIncremental(subset_objective), batch_size=0)
        with pytest.raises(TypeError, match="ReevaluatingIncremental"):
            GreedySubsetSolver(subset_objective)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            GreedySubsetSolver(ReevaluatingIncremental(subset_objective)).solve(7)


class TestExchange:
    @pytest.mark.parametrize("k", [2, 3])
    def test_never_worsens_greedy_and_is_swap_optimal(
        self, bkd: Backend[Array], k: int
    ) -> None:
        subset_objective = BinaryDesignSubsetObjective(_a_optimal(bkd))
        incremental = ReevaluatingIncremental(subset_objective)
        greedy = GreedySubsetSolver(incremental).solve(k)
        result = ExchangeSubsetSolver(incremental).solve(greedy.subset)
        assert result.value <= greedy.value
        _assert_value(bkd, result.value, subset_objective.value(result.subset))
        unchosen = [ii for ii in range(6) if ii not in result.subset]
        for removed in result.subset:
            for added in unchosen:
                swapped = [ii for ii in result.subset if ii != removed] + [added]
                assert subset_objective.value(swapped) >= result.value

    def test_escapes_the_greedy_trap(self, bkd: Backend[Array]) -> None:
        incremental = ReevaluatingIncremental(_greedy_trap(bkd))
        result = ExchangeSubsetSolver(incremental).solve([0, 1])
        assert result.subset == (1, 2)
        _assert_value(bkd, result.value, 0.5)

    def test_swaps_of_k_reach_the_optimum(self, bkd: Backend[Array]) -> None:
        incremental = ReevaluatingIncremental(_swap_trap(bkd))
        single = ExchangeSubsetSolver(incremental).solve([0, 1])
        assert single.subset == (0, 1)
        _assert_value(bkd, single.value, 1.0)
        paired = ExchangeSubsetSolver(incremental, swap_size=2).solve([0, 1])
        assert paired.subset == (2, 3)
        _assert_value(bkd, paired.value, 0.5)

    @pytest.mark.parametrize("start", [(0, 1, 2), (3, 4, 5), (0, 2, 4)])
    def test_swaps_of_k_are_exhaustive(
        self, bkd: Backend[Array], start: Tuple[int, ...]
    ) -> None:
        subset_objective = BinaryDesignSubsetObjective(_a_optimal(bkd))
        exchange = ExchangeSubsetSolver(
            ReevaluatingIncremental(subset_objective), swap_size=3
        )
        assert exchange.solve(start).subset == _best(subset_objective, 3)[0]

    def test_full_set_is_returned(self, bkd: Backend[Array]) -> None:
        incremental = ReevaluatingIncremental(_swap_trap(bkd))
        result = ExchangeSubsetSolver(incremental).solve([3, 2, 1, 0])
        assert result.subset == (0, 1, 2, 3) and result.nevaluations == 1

    @pytest.mark.parametrize("start", [[], [0, 0], [0, 4]])
    def test_rejects_bad_starts(self, bkd: Backend[Array], start: list[int]) -> None:
        exchange = ExchangeSubsetSolver(ReevaluatingIncremental(_swap_trap(bkd)))
        with pytest.raises(ValueError):
            exchange.solve(start)

    def test_rejects_bad_swap_size(self, bkd: Backend[Array]) -> None:
        with pytest.raises(ValueError):
            ExchangeSubsetSolver(ReevaluatingIncremental(_swap_trap(bkd)), swap_size=0)


class TestTopK:
    def test_keeps_largest_weights(self, bkd: Backend[Array]) -> None:
        rounding = TopK(bkd)
        assert isinstance(rounding, RoundingProtocol)
        weights = bkd.asarray([[0.1], [0.9], [0.4], [0.7]])
        assert rounding.round(weights, 2) == (1, 3)
        assert rounding.round(weights, 4) == (0, 1, 2, 3)

    def test_ties_go_to_lower_index(self, bkd: Backend[Array]) -> None:
        weights = bkd.asarray([[0.5], [0.2], [0.5], [0.5]])
        assert TopK(bkd).round(weights, 2) == (0, 2)

    @pytest.mark.parametrize("k", [0, 5])
    def test_rejects_bad_k(self, bkd: Backend[Array], k: int) -> None:
        with pytest.raises(ValueError):
            TopK(bkd).round(bkd.ones((4, 1)), k)

    def test_rejects_bad_shape(self, bkd: Backend[Array]) -> None:
        with pytest.raises(ValueError):
            TopK(bkd).round(bkd.ones((4, 2)), 1)
