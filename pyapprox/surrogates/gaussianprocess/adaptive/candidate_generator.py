"""Candidate generator for adaptive GP sampling."""

from typing import Generic

import numpy as np

from pyapprox.expdesign.quadrature import SobolSampler
from pyapprox.util.backends.protocols import Array, Backend


class HybridSobolRandomCandidateGenerator(Generic[Array]):
    """Generate candidates as half Sobol, half uniform random in scaled space.

    Parameters
    ----------
    nvars : int
        Number of input variables.
    bkd : Backend[Array]
        Backend for numerical computations.
    scaled_bounds : Array | None
        Bounds of shape (nvars, 2) in scaled space. Default: [0, 1]^d.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        nvars: int,
        bkd: Backend[Array],
        scaled_bounds: Array | None = None,
        seed: int = 42,
    ) -> None:
        self._nvars = nvars
        self._bkd = bkd
        self._seed = seed
        if scaled_bounds is not None:
            if scaled_bounds.shape != (nvars, 2):
                raise ValueError(
                    f"scaled_bounds must have shape ({nvars}, 2), "
                    f"got {scaled_bounds.shape}"
                )
            self._lower = scaled_bounds[:, 0]
            self._upper = scaled_bounds[:, 1]
        else:
            self._lower = bkd.zeros((nvars,))
            self._upper = bkd.ones((nvars,))

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def generate(self, ncandidates: int) -> Array:
        """Generate candidate samples in scaled space.

        First half uses Sobol sequences, second half uses uniform random.

        Parameters
        ----------
        ncandidates : int
            Total number of candidates to generate.

        Returns
        -------
        candidates : Array
            Candidates of shape (nvars, ncandidates) in scaled space.
        """
        bkd = self._bkd
        n_sobol = ncandidates // 2
        n_random = ncandidates - n_sobol

        parts: list[Array] = []

        if n_sobol > 0:
            sobol = SobolSampler(
                self._nvars,
                bkd,
                scramble=True,
                seed=self._seed,
            )
            sobol_pts, _ = sobol.sample(n_sobol)
            sobol_scaled = (
                self._lower[:, None] + sobol_pts * (self._upper - self._lower)[:, None]
            )
            parts.append(sobol_scaled)

        if n_random > 0:
            rng = np.random.RandomState(self._seed + 1)
            uniform_01 = rng.random((self._nvars, n_random))
            uniform_arr = bkd.asarray(uniform_01)
            random_scaled = (
                self._lower[:, None]
                + uniform_arr * (self._upper - self._lower)[:, None]
            )
            parts.append(random_scaled)

        return bkd.hstack(parts)
