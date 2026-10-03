"""Statistics of a cheaper model that are known exactly.

A polynomial chaos surrogate's mean and variance under the input
distribution follow from its coefficients, so an estimator need not
estimate them from samples. Such a known value is a fact about a model,
stated without any estimator or statistic in hand: ``KnownMean(model=3,
values=...)``.

Which known statistics are consistent with what is being estimated is
the statistic's to say, not the estimator's. A statistic checks the
known values it is handed against the kinds it accepts and the coupling
it requires -- see ``check_known`` on the statistic classes -- so the
estimator names no statistic and no kind, and a new statistic defines
its own known kinds without the estimator changing.
"""

from dataclasses import dataclass
from typing import Generic, Protocol, final, runtime_checkable

from pyapprox.util.backends.protocols import Array, Array_co

__all__ = ["KnownMean", "KnownStatistic", "KnownVariance"]


@runtime_checkable
class KnownStatistic(Protocol[Array_co]):
    """A statistic of one model, known exactly rather than estimated."""

    @property
    def model(self) -> int:
        """Which model, by its index in the ensemble."""
        ...

    @property
    def values(self) -> Array_co:
        """The known values, one per slot the statistic assigns them."""
        ...


@final
@dataclass(frozen=True)
class KnownMean(Generic[Array]):
    """A model's mean, known exactly.

    Attributes
    ----------
    model : int
        Which model, by its index in the ensemble.
    values : Array
        Shape ``(nqoi,)``.
    """

    model: int
    values: Array

    def __post_init__(self) -> None:
        _check(self.model, self.values, "KnownMean")


@final
@dataclass(frozen=True)
class KnownVariance(Generic[Array]):
    """A model's variance, known exactly.

    Attributes
    ----------
    model : int
        Which model, by its index in the ensemble.
    values : Array
        One value per variance slot of the statistic it is given to:
        shape ``(nqoi,)`` for the diagonal, or the covariance entries the
        statistic tracks.
    """

    model: int
    values: Array

    def __post_init__(self) -> None:
        _check(self.model, self.values, "KnownVariance")


def _check(model: int, values: Array, kind: str) -> None:
    """What a known statistic can check without its statistic.

    How many values it must hold depends on the statistic it is given
    to, so that is the statistic's ``check_known``; that they are one
    vector, for a real model, is checked here, where they are written.
    """
    if isinstance(model, bool) or not isinstance(model, int) or model < 0:
        raise ValueError(f"{kind} model must be a non-negative int, got {model!r}")
    if values.ndim != 1:
        raise ValueError(
            f"{kind} values must be one-dimensional, got shape "
            f"{tuple(values.shape)}"
        )
