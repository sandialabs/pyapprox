"""Inducing point initialization strategies for Deep GPs."""

from typing import Generic, Protocol, Tuple, runtime_checkable

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.sampling.sobol import SobolSampler


@runtime_checkable
class InducingInitializer(Protocol, Generic[Array]):
    """Strategy for placing inducing points at construction time.

    Implementations are consumed once during DGP construction and then
    discarded — they do not persist on the layer or influence training.
    """

    def initialize(
        self,
        num_inducing: int,
        nvars: int,
        bkd: Backend[Array],
        rng: np.random.RandomState,
    ) -> Array:
        """Generate initial inducing point locations.

        Parameters
        ----------
        num_inducing : int
            Number of inducing points M.
        nvars : int
            Input dimensionality for the layer.
        bkd : Backend[Array]
            Backend for array creation.
        rng : np.random.RandomState
            Random state for reproducibility.

        Returns
        -------
        Array
            Inducing locations, shape (nvars, num_inducing).
        """
        ...


class RandomUniformInitializer(Generic[Array]):
    """Uniform random inducing points in [low, high]."""

    def __init__(
        self, bounds: Tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        self._low = bounds[0]
        self._high = bounds[1]

    def initialize(
        self,
        num_inducing: int,
        nvars: int,
        bkd: Backend[Array],
        rng: np.random.RandomState,
    ) -> Array:
        return bkd.array(
            rng.uniform(self._low, self._high, (nvars, num_inducing))
        )


class SobolInitializer(Generic[Array]):
    """Sobol quasi-random inducing points in [low, high].

    Uses pyapprox's SobolSampler for low-discrepancy space-filling.
    """

    def __init__(
        self, bounds: Tuple[float, float] = (-1.0, 1.0),
        scramble: bool = True,
    ) -> None:
        self._low = bounds[0]
        self._high = bounds[1]
        self._scramble = scramble

    def initialize(
        self,
        num_inducing: int,
        nvars: int,
        bkd: Backend[Array],
        rng: np.random.RandomState,
    ) -> Array:
        seed = int(rng.randint(0, 2**31 - 1))
        sampler = SobolSampler(
            nvars, bkd, scramble=self._scramble, seed=seed,
        )
        pts, _ = sampler.sample(num_inducing)
        return pts * (self._high - self._low) + self._low


class CustomInitializer(Generic[Array]):
    """User-supplied inducing point locations."""

    def __init__(self, locations: Array) -> None:
        self._locations = locations

    def initialize(
        self,
        num_inducing: int,
        nvars: int,
        bkd: Backend[Array],
        rng: np.random.RandomState,
    ) -> Array:
        if self._locations.shape != (nvars, num_inducing):
            raise ValueError(
                f"Fixed locations shape {self._locations.shape} does not "
                f"match expected ({nvars}, {num_inducing})"
            )
        return self._locations
