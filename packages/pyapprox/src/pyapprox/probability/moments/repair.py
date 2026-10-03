r"""Repairs for indefinite covariance estimates.

A rule with negative weights, such as a sparse grid, or known target
moments substituted into estimated blocks, can make the stacked covariance
indefinite. ``NoRepair`` refuses such blocks; ``EigenClip`` raises every
eigenvalue to a floor. With a zero floor, ``EigenClip`` returns the nearest
positive-semidefinite matrix in the Frobenius norm, so no separate
nearest-PSD repair is needed.
"""

from typing import Generic

from pyapprox.probability.moments.blocks import DenseBlocks
from pyapprox.util.backends.protocols import Array


class NoRepair(Generic[Array]):
    """Accept blocks unchanged, or raise if they are indefinite.

    Parameters
    ----------
    tol : float
        Smallest eigenvalue allowed, relative to the largest:
        ``lambda_min >= -tol * lambda_max``. Default 1e-10.
    """

    def __init__(self, tol: float = 1e-10) -> None:
        if tol < 0.0:
            raise ValueError(f"tol must be non-negative, got {tol}")
        self._tol = tol

    def repair(self, blocks: DenseBlocks[Array]) -> DenseBlocks[Array]:
        """Return ``blocks`` if positive semidefinite within ``tol``."""
        bkd = blocks.bkd()
        eigvals = bkd.eigvalsh(blocks.covariance())
        lmin = bkd.to_float(bkd.min(eigvals))
        lmax = bkd.to_float(bkd.max(bkd.abs(eigvals)))
        if lmin < -self._tol * lmax:
            raise ValueError(
                f"covariance is indefinite: smallest eigenvalue {lmin:.3e} "
                f"against largest {lmax:.3e}; repair it, for example with "
                "EigenClip"
            )
        return blocks


class EigenClip(Generic[Array]):
    """Raise eigenvalues of the stacked covariance to a floor.

    Parameters
    ----------
    rel_floor : float
        Floor relative to the largest eigenvalue. Default 0, which gives
        the nearest positive-semidefinite matrix in the Frobenius norm.
        A positive floor makes the result positive definite.
    """

    def __init__(self, rel_floor: float = 0.0) -> None:
        if rel_floor < 0.0:
            raise ValueError(f"rel_floor must be non-negative, got {rel_floor}")
        self._rel_floor = rel_floor

    def repair(self, blocks: DenseBlocks[Array]) -> DenseBlocks[Array]:
        """Blocks with every eigenvalue below the floor raised to it."""
        bkd = blocks.bkd()
        cov = blocks.covariance()
        eigvals, eigvecs = bkd.eigh(0.5 * (cov + cov.T))
        floor = self._rel_floor * bkd.max(bkd.abs(eigvals))
        clipped = bkd.maximum(eigvals, floor)
        repaired = bkd.dot(eigvecs * clipped, eigvecs.T)
        return DenseBlocks(
            blocks.mean(),
            repaired,
            blocks.target_sizes(),
            blocks.nobs(),
            bkd,
            blocks.nsamples(),
            blocks.exact_targets(),
        )
