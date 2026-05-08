"""Sampling schedule implementations for adaptive GP."""


class ConstantSamplingSchedule:
    """Add a constant number of samples each step until a maximum is reached.

    Parameters
    ----------
    increment : int
        Number of samples to add at each step.
    max_nsamples : int
        Maximum total number of samples.
    """

    def __init__(self, increment: int, max_nsamples: int) -> None:
        if increment <= 0:
            raise ValueError("increment must be positive")
        if max_nsamples <= 0:
            raise ValueError("max_nsamples must be positive")
        self._increment = increment
        self._max_nsamples = max_nsamples
        self._nsamples_so_far = 0

    def nnew_samples(self) -> int:
        """Return number of new samples and advance state."""
        if self._nsamples_so_far >= self._max_nsamples:
            raise StopIteration("Schedule exhausted")
        n = min(
            self._increment,
            self._max_nsamples - self._nsamples_so_far,
        )
        self._nsamples_so_far += n
        return n

    def is_exhausted(self) -> bool:
        """Return True if max_nsamples has been reached."""
        return self._nsamples_so_far >= self._max_nsamples


class ListSamplingSchedule:
    """Add samples according to a predefined list of increments.

    Parameters
    ----------
    increments : list[int]
        List of sample counts for each step, e.g. [10, 20, 30].
    """

    def __init__(self, increments: list[int]) -> None:
        if len(increments) == 0:
            raise ValueError("increments must be non-empty")
        for inc in increments:
            if inc <= 0:
                raise ValueError("All increments must be positive")
        self._increments = list(increments)
        self._index = 0

    def nnew_samples(self) -> int:
        """Return the next increment and advance state."""
        if self._index >= len(self._increments):
            raise StopIteration("Schedule exhausted")
        n = self._increments[self._index]
        self._index += 1
        return n

    def is_exhausted(self) -> bool:
        """Return True if all increments have been used."""
        return self._index >= len(self._increments)
