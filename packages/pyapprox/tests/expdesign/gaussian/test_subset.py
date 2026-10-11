"""Tests for AOptimalSubsetObjective against the relaxed composition."""

from typing import Generic, List, Optional

import numpy as np
import pytest

from pyapprox.expdesign.design_space import (
    BinaryDesignSubsetObjective,
    GroupedDesign,
    ParameterizedObjective,
)
from pyapprox.expdesign.gaussian import (
    AOptimal,
    AOptimalSubsetObjective,
    BlendedObservation,
    DesignObjective,
)
from pyapprox.expdesign.protocols.subset import SubsetObjectiveProtocol
from pyapprox.inverse.joint_gaussian import JointGaussian
from pyapprox.probability.covariance import DenseCholeskyCovarianceOperator
from pyapprox.probability.moments import DenseBlocks
from pyapprox.util.backends.protocols import Array, Backend


class TestAOptimalSubsetObjective(Generic[Array]):
    """xi ~ N(0, P) in R^4, data H xi (6 x 4) with correlated noise, and
    targets B xi (2 x 4) and xi."""

    _nx, _nobs = 4, 6
    _groups = [[0, 3], [1], [2, 5], [4]]
    _joint: JointGaussian[Array]
    _noise: DenseCholeskyCovarianceOperator[Array]
    _map: Array

    def _setup(self, bkd: Backend[Array]) -> None:
        rng = np.random.default_rng(3)
        root = rng.normal(size=(self._nx, self._nx))
        prior_cov = root @ root.T / self._nx + np.eye(self._nx)
        hmat = rng.normal(size=(self._nobs, self._nx))
        bmat = rng.normal(size=(2, self._nx))
        noise_root = rng.normal(size=(self._nobs, self._nobs))
        noise_cov = 0.1 * (noise_root @ noise_root.T / self._nobs + np.eye(self._nobs))
        blocks = DenseBlocks.from_linear_model(
            bkd.asarray(hmat),
            bkd.zeros((self._nx, 1)),
            bkd.asarray(prior_cov),
            [bkd.asarray(bmat), bkd.eye(self._nx)],
            bkd,
        )
        self._noise = DenseCholeskyCovarianceOperator(bkd.asarray(noise_cov), bkd)
        self._joint = JointGaussian(blocks, self._noise)
        self._map = bkd.asarray(rng.normal(size=(3, self._nx)))

    def _relaxed(
        self,
        bkd: Backend[Array],
        index: int,
        groups: List[List[int]],
        target_map: Optional[Array],
    ) -> BinaryDesignSubsetObjective[Array]:
        objective = DesignObjective(
            self._joint,
            BlendedObservation.from_noise(self._noise),
            AOptimal(target_map),
            index,
        )
        grouped = GroupedDesign(groups, self._nobs, bkd)
        return BinaryDesignSubsetObjective(ParameterizedObjective(objective, grouped))

    def _subsets(self, ncandidates: int) -> List[List[int]]:
        return [
            [jj for jj in range(ncandidates) if mask >> jj & 1]
            for mask in range(1, 2**ncandidates)
        ]

    @pytest.mark.parametrize("use_map", [False, True])
    @pytest.mark.parametrize("grouped", [True, False])
    def test_matches_relaxed_objective_at_binary_designs(
        self, bkd: Backend[Array], use_map: bool, grouped: bool
    ) -> None:
        self._setup(bkd)
        groups = self._groups if grouped else [[ii] for ii in range(self._nobs)]
        index = 1
        target_map = self._map if use_map else None
        fast = AOptimalSubsetObjective(self._joint, index, groups, target_map)
        relaxed = self._relaxed(bkd, index, groups, target_map)
        assert isinstance(fast, SubsetObjectiveProtocol)
        assert fast.ncandidates() == len(groups)
        subsets = self._subsets(len(groups))
        fast_values = bkd.asarray([fast.value(s) for s in subsets])
        relaxed_values = bkd.asarray([relaxed.value(s) for s in subsets])
        bkd.assert_allclose(fast_values, relaxed_values, rtol=1e-10)

    def test_matches_conditioning_on_selected_rows(self, bkd: Backend[Array]) -> None:
        self._setup(bkd)
        fast = AOptimalSubsetObjective(self._joint, 0, self._groups)
        subset = [0, 2]
        rows = fast.observations(subset)
        assert rows == [0, 3, 2, 5]
        _, cov = self._joint.select(rows).condition(bkd.zeros((len(rows), 1)), 0)
        bkd.assert_allclose(
            bkd.asarray([fast.value(subset)]),
            bkd.asarray([bkd.to_float(bkd.trace(cov))]),
        )

    def test_empty_subset_is_prior_value(self, bkd: Backend[Array]) -> None:
        self._setup(bkd)
        fast = AOptimalSubsetObjective(self._joint, 0, self._groups)
        prior = self._joint.blocks().target_covariance(0)
        bkd.assert_allclose(
            bkd.asarray([fast.value([])]), bkd.asarray([bkd.to_float(bkd.trace(prior))])
        )

    def test_invalid_arguments_raise(self, bkd: Backend[Array]) -> None:
        self._setup(bkd)
        with pytest.raises(ValueError, match="index must lie"):
            AOptimalSubsetObjective(self._joint, 2, self._groups)
        with pytest.raises(ValueError, match="repeats observations"):
            AOptimalSubsetObjective(self._joint, 0, [[0, 1], [1]])
        with pytest.raises(ValueError, match="outside the 6 observations"):
            AOptimalSubsetObjective(self._joint, 0, [[6]])
        with pytest.raises(ValueError, match="target_map must have shape"):
            AOptimalSubsetObjective(self._joint, 0, self._groups, bkd.eye(3))
        fast = AOptimalSubsetObjective(self._joint, 0, self._groups)
        with pytest.raises(ValueError, match="repeats design variables"):
            fast.value([1, 1])
        with pytest.raises(ValueError, match="outside the 4 design variables"):
            fast.value([4])
