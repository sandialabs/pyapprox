"""Marginal parameter sweeps.

Linear sweeps through physical space for arbitrary independent
marginals.  Computes physical-space bounds from each marginal's
quantiles, then delegates to `BoundedParameterSweeper` so the sweep
is a straight line in user (physical) coordinates.
"""

from typing import Generic, List, Protocol, Tuple, runtime_checkable

from scipy.stats import norm

from pyapprox.interface.functions.sweeps.bounded import (
    BoundedParameterSweeper,
)
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class InvCDFProtocol(Protocol[Array]):
    """Minimal protocol: anything with an invcdf method."""

    def invcdf(self, quantiles: Array) -> Array: ...


class MarginalParameterSweeper(Generic[Array]):
    """Parameter sweeper for arbitrary independent marginals.

    Converts marginal distributions to physical-space bounds via
    quantiles, then sweeps along random orthogonal directions in that
    bounded physical space.  The result is a straight line in physical
    coordinates.

    Parameters
    ----------
    marginals : list of MarginalProtocol
        One marginal per input variable.
    sweep_radius : float
        Number of standard deviations for the quantile bounds.
        Physical bounds are ``[F^{-1}(Phi(-r)), F^{-1}(Phi(r))]``
        where ``r`` is ``sweep_radius``.
    nsamples_per_sweep : int
        Number of evenly spaced samples per sweep.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        marginals: List[InvCDFProtocol[Array]],
        sweep_radius: float,
        nsamples_per_sweep: int,
        bkd: Backend[Array],
    ) -> None:
        self._bkd = bkd
        self._marginals = marginals
        self._nvars = len(marginals)
        self._sweep_radius = sweep_radius
        self._nsamples_per_sweep = nsamples_per_sweep

        prob_lo = float(norm.cdf(-sweep_radius))
        prob_hi = float(norm.cdf(sweep_radius))

        bounds_list = []
        for m in marginals:
            lo = bkd.to_float(
                m.invcdf(bkd.asarray([[prob_lo]]))[0, 0]
            )
            hi = bkd.to_float(
                m.invcdf(bkd.asarray([[prob_hi]]))[0, 0]
            )
            bounds_list.append([lo, hi])

        bounds = bkd.asarray(bounds_list)
        self._bounded = BoundedParameterSweeper(
            bounds, nsamples_per_sweep, bkd,
        )

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._nvars

    def nsamples_per_sweep(self) -> int:
        return self._nsamples_per_sweep

    def set_nsamples_per_sweep(self, nsamples_per_sweep: int) -> None:
        self._nsamples_per_sweep = nsamples_per_sweep
        self._bounded.set_nsamples_per_sweep(nsamples_per_sweep)

    def random_rotation_matrices(self, nsweeps: int) -> Array:
        return self._bounded.random_rotation_matrices(nsweeps)

    def set_sweep_rotation_matrices(self, mat: Array) -> None:
        self._bounded.set_sweep_rotation_matrices(mat)

    def sweep_bounds(self, rotation_vec: Array) -> Tuple[float, float]:
        return self._bounded.sweep_bounds(rotation_vec)

    def canonical_sweep_samples(self, rotation_vec: Array) -> Array:
        return self._bounded.canonical_sweep_samples(rotation_vec)

    def rvs(self, nsweeps: int) -> Array:
        """Generate sweep samples in physical space.

        Parameters
        ----------
        nsweeps : int
            Number of sweep directions.

        Returns
        -------
        Array
            Shape ``(nvars, nsamples_per_sweep * nsweeps)``.
        """
        return self._bounded.rvs(nsweeps)

    def canonical_active_samples(self) -> Array:
        return self._bounded.canonical_active_samples()

    def samples(self) -> Array:
        return self._bounded.samples()

    def __repr__(self) -> str:
        return (
            f"MarginalParameterSweeper(nvars={self._nvars}, "
            f"sweep_radius={self._sweep_radius}, "
            f"nsamples_per_sweep={self._nsamples_per_sweep})"
        )
