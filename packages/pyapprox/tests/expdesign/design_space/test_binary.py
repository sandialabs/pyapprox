"""Tests for BinaryDesignSubsetObjective and ReevaluatingIncremental.

The oracle is the linear-Gaussian posterior given only the chosen
observations, with the noise covariance restricted to them:
``Gamma_post = S - S G_s^T (G_s S G_s^T + N_ss)^{-1} G_s S``, and
``EIG = 1/2 log det(G_s S G_s^T + N_ss) - 1/2 log det N_ss``.
"""

import itertools
from typing import Generic

import numpy as np
import pytest
from numpy.typing import NDArray

from pyapprox.expdesign.design_space import (
    BinaryDesignSubsetObjective,
    GroupedDesign,
    ParameterizedObjective,
    ReevaluatingIncremental,
)
from pyapprox.expdesign.gaussian import (
    AOptimal,
    BlendedObservation,
    DesignObjective,
    ExpectedInformationGain,
)
from pyapprox.expdesign.protocols import (
    IncrementalSubsetObjectiveProtocol,
    SubsetObjectiveProtocol,
)
from pyapprox.expdesign.protocols.gaussian_criterion import (
    GaussianDesignCriterionProtocol,
)
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.inverse.joint_gaussian import JointGaussian
from pyapprox.probability.covariance import DenseCholeskyCovarianceOperator
from pyapprox.probability.moments import DenseBlocks
from pyapprox.util.backends.protocols import Array, Backend

_BY_SENSOR = [[0, 3], [1, 4], [2, 5]]


class _NegativeLogWeights(Generic[Array]):
    """``-sum(log w)``, the precision-weighting term, infinite at ``w = 0``."""

    def __init__(self, nvars: int, bkd: Backend[Array]) -> None:
        self._nvars = nvars
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._nvars

    def nqoi(self) -> int:
        return 1

    def __call__(self, design_weights: Array) -> Array:
        if self._bkd.to_float(self._bkd.min(design_weights)) == 0.0:
            return self._bkd.full((1, 1), float("inf"))
        return self._bkd.reshape(
            -self._bkd.sum(self._bkd.log(design_weights), axis=0), (1, -1)
        )

    def derivatives(self) -> Derivatives[Array]:
        return Derivatives.none()


class _Problem:
    """Three parameters observed six times with correlated noise."""

    def __init__(self) -> None:
        rng = np.random.default_rng(31)
        root = rng.normal(size=(3, 3))
        self.prior_cov = root @ root.T / 3 + np.eye(3)
        noise_root = rng.normal(size=(6, 6))
        self.noise_cov = 0.1 * (noise_root @ noise_root.T / 6 + np.eye(6))
        self.obs_matrix = rng.normal(size=(6, 3))

    def objective(
        self, bkd: Backend[Array], criterion: GaussianDesignCriterionProtocol[Array]
    ) -> DesignObjective[Array]:
        blocks = DenseBlocks.from_linear_model(
            bkd.asarray(self.obs_matrix),
            bkd.zeros((3, 1)),
            bkd.asarray(self.prior_cov),
            [bkd.eye(3)],
            bkd,
        )
        noise = DenseCholeskyCovarianceOperator(bkd.asarray(self.noise_cov), bkd)
        joint = JointGaussian(blocks, noise)
        return DesignObjective(
            joint, BlendedObservation.from_noise(noise), criterion, 0
        )

    def _parts(self, rows: list[int]) -> tuple[NDArray[np.float64], ...]:
        g = self.obs_matrix[rows]
        n = self.noise_cov[np.ix_(rows, rows)]
        return g, n, g @ self.prior_cov @ g.T + n

    def a_optimal(self, rows: list[int]) -> float:
        if not rows:
            return float(np.trace(self.prior_cov))
        g, _, data_cov = self._parts(rows)
        cross = self.prior_cov @ g.T
        post = self.prior_cov - cross @ np.linalg.solve(data_cov, cross.T)
        return float(np.trace(post))

    def negative_eig(self, rows: list[int]) -> float:
        if not rows:
            return 0.0
        _, n, data_cov = self._parts(rows)
        return -0.5 * float(np.linalg.slogdet(data_cov)[1] - np.linalg.slogdet(n)[1])


def _subsets(ncandidates: int) -> list[tuple[int, ...]]:
    return [
        subset
        for size in range(ncandidates + 1)
        for subset in itertools.combinations(range(ncandidates), size)
    ]


class TestBinaryDesignSubsetObjective:
    @pytest.mark.parametrize(
        "criterion, oracle",
        [(AOptimal(), "a_optimal"), (ExpectedInformationGain(), "negative_eig")],
    )
    def test_matches_posterior_of_chosen_observations(
        self,
        bkd: Backend[Array],
        criterion: GaussianDesignCriterionProtocol[Array],
        oracle: str,
    ) -> None:
        problem = _Problem()
        subset_objective = BinaryDesignSubsetObjective(
            problem.objective(bkd, criterion)
        )
        assert isinstance(subset_objective, SubsetObjectiveProtocol)
        assert subset_objective.ncandidates() == 6
        subsets = _subsets(6)
        bkd.assert_allclose(
            bkd.asarray([subset_objective.value(subset) for subset in subsets]),
            bkd.asarray([getattr(problem, oracle)(list(subset)) for subset in subsets]),
            rtol=1e-10,
            atol=1e-12,
        )

    def test_groups_choose_all_their_observations(self, bkd: Backend[Array]) -> None:
        problem = _Problem()
        design = GroupedDesign(_BY_SENSOR, 6, bkd)
        subset_objective = BinaryDesignSubsetObjective(
            ParameterizedObjective(problem.objective(bkd, AOptimal()), design)
        )
        assert subset_objective.ncandidates() == 3
        subsets = _subsets(3)
        rows = [
            sorted(ii for jj in subset for ii in _BY_SENSOR[jj]) for subset in subsets
        ]
        bkd.assert_allclose(
            bkd.asarray([subset_objective.value(subset) for subset in subsets]),
            bkd.asarray([problem.a_optimal(chosen) for chosen in rows]),
            rtol=1e-10,
        )

    def test_design_is_zero_one(self, bkd: Backend[Array]) -> None:
        subset_objective = BinaryDesignSubsetObjective(
            _Problem().objective(bkd, AOptimal())
        )
        bkd.assert_allclose(
            subset_objective.design([4, 1]),
            bkd.asarray([[0.0], [1.0], [0.0], [0.0], [1.0], [0.0]]),
        )

    @pytest.mark.parametrize("subset", [[6], [-1], [2, 2]])
    def test_rejects_bad_subsets(self, bkd: Backend[Array], subset: list[int]) -> None:
        subset_objective = BinaryDesignSubsetObjective(
            _Problem().objective(bkd, AOptimal())
        )
        with pytest.raises(ValueError):
            subset_objective.value(subset)

    def test_rejects_objective_undefined_at_zero(self, bkd: Backend[Array]) -> None:
        """A precision-weighted objective's ``log w`` term is infinite at 0."""
        subset_objective = BinaryDesignSubsetObjective(_NegativeLogWeights(3, bkd))
        with pytest.raises(ValueError, match="defined at zero weights"):
            subset_objective.value([0, 2])
        assert subset_objective.value([0, 1, 2]) == 0.0

    def test_rejects_non_objective(self, bkd: Backend[Array]) -> None:
        with pytest.raises(TypeError):
            BinaryDesignSubsetObjective(AOptimal())  # type: ignore[arg-type]


class TestReevaluatingIncremental:
    def _incremental(
        self, bkd: Backend[Array]
    ) -> tuple[BinaryDesignSubsetObjective[Array], ReevaluatingIncremental[Array]]:
        subset_objective = BinaryDesignSubsetObjective(
            _Problem().objective(bkd, ExpectedInformationGain())
        )
        return subset_objective, ReevaluatingIncremental(subset_objective)

    def test_satisfies_protocol(self, bkd: Backend[Array]) -> None:
        _, incremental = self._incremental(bkd)
        assert isinstance(incremental, IncrementalSubsetObjectiveProtocol)
        assert incremental.ncandidates() == 6
        assert incremental.initial_state() == ()

    def test_values_after_rescore_extended_sets(self, bkd: Backend[Array]) -> None:
        subset_objective, incremental = self._incremental(bkd)
        state = incremental.add(incremental.initial_state(), [4])
        additions = [[0], [2, 5], [1, 3]]
        expected = [subset_objective.value([4, *addition]) for addition in additions]
        bkd.assert_allclose(
            incremental.values_after(state, additions),
            bkd.asarray(expected),
            rtol=1e-12,
        )

    def test_add_leaves_state_unchanged(self, bkd: Backend[Array]) -> None:
        _, incremental = self._incremental(bkd)
        state = incremental.add(incremental.initial_state(), [3])
        grown = incremental.add(state, [5, 0])
        assert state == (3,)
        assert grown == (0, 3, 5)

    @pytest.mark.parametrize("addition", [[3], [1, 1], [6]])
    def test_rejects_bad_additions(
        self, bkd: Backend[Array], addition: list[int]
    ) -> None:
        _, incremental = self._incremental(bkd)
        state = incremental.add(incremental.initial_state(), [3])
        with pytest.raises(ValueError):
            incremental.add(state, addition)
        with pytest.raises(ValueError):
            incremental.values_after(state, [addition])

    def test_rejects_non_subset_objective(self, bkd: Backend[Array]) -> None:
        with pytest.raises(TypeError):
            ReevaluatingIncremental(
                _Problem().objective(bkd, AOptimal())  # type: ignore[arg-type]
            )
