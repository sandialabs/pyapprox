"""A target made of the inputs themselves."""

from typing import Generic, Optional, Sequence

from pyapprox.util.backends.protocols import Array, Backend


class InputTarget(Generic[Array]):
    """Return chosen input variables unchanged.

    Maps samples of shape (nvars, nsamples) to ``samples[rows]``. Use it
    as a target function of a joint evaluator when the quantity to learn
    is some of the input variables themselves, rather than a model
    output.

    For example, with six inputs stacking a parameter ``m`` (rows 0-2)
    and nuisance variables (rows 3-5), ``InputTarget(6, bkd, rows=[0, 1,
    2])`` makes ``m`` the target, so a design is chosen to learn ``m``.

    To target a function of the inputs instead, such as ``exp(m)``, pass
    that function as the target function directly.

    Parameters
    ----------
    nvars : int
        Number of input variables.
    bkd : Backend[Array]
        Computational backend.
    rows : Sequence[int], optional
        Input rows to return, in order. Default all of them.
    """

    def __init__(
        self,
        nvars: int,
        bkd: Backend[Array],
        rows: Optional[Sequence[int]] = None,
    ) -> None:
        selected = list(range(nvars)) if rows is None else list(rows)
        if len(selected) == 0:
            raise ValueError("rows is empty")
        bad = [row for row in selected if not 0 <= row < nvars]
        if bad:
            raise ValueError(f"rows {bad} are outside the {nvars} inputs")
        self._nvars = nvars
        self._bkd = bkd
        self._rows = selected

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Number of input variables."""
        return self._nvars

    def nqoi(self) -> int:
        """Number of selected rows."""
        return len(self._rows)

    def __call__(self, samples: Array, /) -> Array:
        """Selected rows of the samples. Shape: (nqoi, nsamples)"""
        if samples.ndim != 2 or samples.shape[0] != self._nvars:
            raise ValueError(
                f"samples must have shape ({self._nvars}, nsamples), got "
                f"{tuple(samples.shape)}"
            )
        return samples[self._rows]
