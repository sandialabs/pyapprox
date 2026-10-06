"""Finite-difference directions that stay inside box bounds.

``DerivativeChecker`` uses forward differences, ``f(x + h v)``, along a unit
direction ``v``. At a point on the boundary of a box, a random ``v`` can
leave the box. An inward direction keeps every step feasible: components
of ``v`` at a lower bound are made non-negative and those at an upper
bound non-positive. The largest feasible step is then the distance from
the interior components to their own bounds.
"""

from typing import Optional

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend


def inward_direction(
    point: np.ndarray,
    bkd: Backend[Array],
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
    seed: int = 0,
) -> Array:
    """A random unit direction pointing into the box at ``point``.

    Parameters
    ----------
    point : np.ndarray
        The point. Shape: (nvars, 1)
    bkd : Backend[Array]
        Computational backend for the result.
    lower, upper : np.ndarray, optional
        Bounds of each variable, shape (nvars, 1). A variable equal to its
        lower bound gets a non-negative component, one equal to its upper
        bound a non-positive one. None means unbounded on that side.
    seed : int
        Seed for the random direction.

    Returns
    -------
    Array
        Unit direction. Shape: (nvars, 1)
    """
    direction = np.random.default_rng(seed).standard_normal(point.shape)
    if lower is not None:
        at_lower = point <= lower
        direction[at_lower] = np.abs(direction[at_lower])
    if upper is not None:
        at_upper = point >= upper
        direction[at_upper] = -np.abs(direction[at_upper])
    return bkd.asarray(direction / np.linalg.norm(direction))
