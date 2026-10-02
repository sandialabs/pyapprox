"""Tests for joint evaluation of observations and targets."""

import numpy as np
import pytest
from pyapprox.interface.functions.fromcallable.function import (
    FunctionFromCallable,
)
from pyapprox.interface.functions.joint import (
    InputTarget,
    JointEvaluatorProtocol,
    JointOutputs,
    SeparateFunctions,
    SplitFunction,
)
from pyapprox.util.backends.protocols import Array, Backend


class _CountingFunction:
    """Wraps a callable and counts how often it is evaluated."""

    def __init__(self, nqoi: int, nvars: int, bkd: Backend[Array]) -> None:
        self.ncalls = 0
        self._mat = bkd.asarray(np.random.default_rng(1).normal(size=(nqoi, nvars)))
        self._bkd = bkd
        self._nqoi = nqoi
        self._nvars = nvars

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._nvars

    def nqoi(self) -> int:
        return self._nqoi

    def __call__(self, samples: Array, /) -> Array:
        self.ncalls += 1
        return self._bkd.sin(self._bkd.dot(self._mat, samples))


class TestJointEvaluation:
    """One function with 5 outputs over 3 inputs, split three ways."""

    _nvars, _nqoi = 3, 5
    _obs_rows = [0, 2, 3]
    _target_rows = [[1], [4, 0]]

    def _samples(self, bkd: Backend[Array]) -> Array:
        return bkd.asarray(np.random.default_rng(2).normal(size=(self._nvars, 7)))

    def _row_function(
        self, shared: _CountingFunction, rows: list[int]
    ) -> FunctionFromCallable[Array]:
        return FunctionFromCallable(
            len(rows), self._nvars, lambda x: shared(x)[rows], shared.bkd()
        )

    def test_split_matches_separate(self, bkd: Backend[Array]) -> None:
        shared = _CountingFunction(self._nqoi, self._nvars, bkd)
        split = SplitFunction(shared, self._obs_rows, self._target_rows)
        separate = SeparateFunctions(
            self._row_function(shared, self._obs_rows),
            [self._row_function(shared, rows) for rows in self._target_rows],
        )
        samples = self._samples(bkd)
        a, b = split.evaluate(samples), separate.evaluate(samples)
        bkd.assert_allclose(a.observations, b.observations, rtol=1e-12)
        assert len(a.targets) == len(b.targets) == 2
        for ta, tb in zip(a.targets, b.targets):
            bkd.assert_allclose(ta, tb, rtol=1e-12)
        assert split.target_sizes() == separate.target_sizes() == (1, 2)
        assert split.nobs() == separate.nobs() == 3

    def test_split_runs_function_once(self, bkd: Backend[Array]) -> None:
        shared = _CountingFunction(self._nqoi, self._nvars, bkd)
        split = SplitFunction(shared, self._obs_rows, self._target_rows)
        split.evaluate(self._samples(bkd))
        assert shared.ncalls == 1

    def test_satisfies_protocol(self, bkd: Backend[Array]) -> None:
        shared = _CountingFunction(self._nqoi, self._nvars, bkd)
        split = SplitFunction(shared, self._obs_rows, self._target_rows)
        separate = SeparateFunctions(shared, [InputTarget(self._nvars, bkd)])
        assert isinstance(split, JointEvaluatorProtocol)
        assert isinstance(separate, JointEvaluatorProtocol)

    def test_input_target(self, bkd: Backend[Array]) -> None:
        samples = self._samples(bkd)
        bkd.assert_allclose(InputTarget(self._nvars, bkd)(samples), samples)
        selected = InputTarget(self._nvars, bkd, rows=[2, 0])
        assert selected.nqoi() == 2
        bkd.assert_allclose(selected(samples), samples[[2, 0]])

    def test_separate_rejects_mismatched_inputs(self, bkd: Backend[Array]) -> None:
        shared = _CountingFunction(self._nqoi, self._nvars, bkd)
        with pytest.raises(ValueError):
            SeparateFunctions(shared, [InputTarget(self._nvars + 1, bkd)])

    def test_separate_rejects_non_function(self, bkd: Backend[Array]) -> None:
        shared = _CountingFunction(self._nqoi, self._nvars, bkd)
        with pytest.raises(TypeError):
            SeparateFunctions(shared, [lambda x: x])

    @pytest.mark.parametrize("obs_rows,target_rows", [([], [[1]]), ([0], [[5]])])
    def test_split_rejects_bad_rows(
        self,
        bkd: Backend[Array],
        obs_rows: list[int],
        target_rows: list[list[int]],
    ) -> None:
        shared = _CountingFunction(self._nqoi, self._nvars, bkd)
        with pytest.raises(ValueError):
            SplitFunction(shared, obs_rows, target_rows)

    def test_evaluate_rejects_bad_samples(self, bkd: Backend[Array]) -> None:
        shared = _CountingFunction(self._nqoi, self._nvars, bkd)
        split = SplitFunction(shared, self._obs_rows, self._target_rows)
        with pytest.raises(ValueError):
            split.evaluate(bkd.ones((self._nvars + 1, 4)))
        with pytest.raises(ValueError):
            split.evaluate(bkd.ones((self._nvars,)))

    def test_outputs_reject_mismatched_samples(self, bkd: Backend[Array]) -> None:
        with pytest.raises(ValueError):
            JointOutputs(targets=(bkd.ones((2, 3)),), observations=bkd.ones((4, 5)))
        with pytest.raises(ValueError):
            JointOutputs(targets=(), observations=bkd.ones((4,)))
        assert JointOutputs(targets=(), observations=bkd.ones((4, 5))).nsamples() == 5
