"""Joint evaluators: separate functions, or one function split by rows.

Every function a joint evaluator holds takes the same inputs, so that
observations and targets are evaluated at the same samples. A function of
only some of the inputs must first be lifted to take all of them.
"""

from typing import Generic, Sequence

from pyapprox.interface.functions.joint.outputs import JointOutputs
from pyapprox.interface.functions.protocols import FunctionProtocol
from pyapprox.util.backends.protocols import Array, Backend


def _check_function(name: str, function: object) -> None:
    if not isinstance(function, FunctionProtocol):
        raise TypeError(
            f"{name} must satisfy FunctionProtocol, got {type(function).__name__}"
        )


def _check_samples(samples: Array, nvars: int) -> None:
    if samples.ndim != 2 or samples.shape[0] != nvars:
        raise ValueError(
            f"samples must have shape ({nvars}, nsamples), got {tuple(samples.shape)}"
        )


class SeparateFunctions(Generic[Array]):
    """Observations and each target from their own functions.

    All functions take the same inputs and are evaluated at the same
    samples, once per call.

    Parameters
    ----------
    obs_function : FunctionProtocol[Array]
        Maps inputs to noise-free observations.
    target_functions : Sequence[FunctionProtocol[Array]]
        One function per target block, in order. Each must take the same
        number of inputs as ``obs_function``.
    """

    def __init__(
        self,
        obs_function: FunctionProtocol[Array],
        target_functions: Sequence[FunctionProtocol[Array]],
    ) -> None:
        _check_function("obs_function", obs_function)
        for ii, function in enumerate(target_functions):
            _check_function(f"target_functions[{ii}]", function)
            if function.nvars() != obs_function.nvars():
                raise ValueError(
                    f"target_functions[{ii}] takes {function.nvars()} inputs "
                    f"but obs_function takes {obs_function.nvars()}"
                )
        self._obs_function = obs_function
        self._target_functions = tuple(target_functions)

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._obs_function.bkd()

    def nvars(self) -> int:
        """Number of input variables."""
        return self._obs_function.nvars()

    def nobs(self) -> int:
        """Number of observations."""
        return self._obs_function.nqoi()

    def target_sizes(self) -> tuple[int, ...]:
        """Number of rows in each target block."""
        return tuple(function.nqoi() for function in self._target_functions)

    def evaluate(self, samples: Array) -> JointOutputs[Array]:
        """Evaluate every function at the samples. Shape: (nvars, nsamples)"""
        _check_samples(samples, self.nvars())
        return JointOutputs(
            targets=tuple(function(samples) for function in self._target_functions),
            observations=self._obs_function(samples),
        )


class SplitFunction(Generic[Array]):
    """Observations and targets as row subsets of one function's output.

    The function runs once per call, which makes this the choice when
    observations and targets come from the same expensive solve.

    Parameters
    ----------
    function : FunctionProtocol[Array]
        The shared function.
    obs_rows : Sequence[int]
        Output rows that are observations, in order.
    target_rows : Sequence[Sequence[int]]
        Output rows of each target block, in order. Rows may repeat
        across blocks and may overlap ``obs_rows``.
    """

    def __init__(
        self,
        function: FunctionProtocol[Array],
        obs_rows: Sequence[int],
        target_rows: Sequence[Sequence[int]],
    ) -> None:
        _check_function("function", function)
        nqoi = function.nqoi()
        blocks = [("obs_rows", obs_rows)] + [
            (f"target_rows[{ii}]", rows) for ii, rows in enumerate(target_rows)
        ]
        for name, rows in blocks:
            if len(rows) == 0:
                raise ValueError(f"{name} is empty")
            bad = [row for row in rows if not 0 <= row < nqoi]
            if bad:
                raise ValueError(
                    f"{name} has rows {bad} outside the function's {nqoi} outputs"
                )
        self._function = function
        self._obs_rows = list(obs_rows)
        self._target_rows = tuple(list(rows) for rows in target_rows)

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._function.bkd()

    def nvars(self) -> int:
        """Number of input variables."""
        return self._function.nvars()

    def nobs(self) -> int:
        """Number of observations."""
        return len(self._obs_rows)

    def target_sizes(self) -> tuple[int, ...]:
        """Number of rows in each target block."""
        return tuple(len(rows) for rows in self._target_rows)

    def evaluate(self, samples: Array) -> JointOutputs[Array]:
        """Run the function once and split its rows. Shape: (nvars, nsamples)"""
        _check_samples(samples, self.nvars())
        values = self._function(samples)
        return JointOutputs(
            targets=tuple(values[rows] for rows in self._target_rows),
            observations=values[self._obs_rows],
        )
